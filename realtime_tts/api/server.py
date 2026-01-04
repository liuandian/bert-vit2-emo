"""
FastAPI服务器

提供HTTP REST API和WebSocket流式服务
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import uuid
import json
import base64
from datetime import datetime

from realtime_tts.core import StreamingInferenceEngine
from realtime_tts.api.schemas import (
    TTSRequest,
    TTSResponse,
    StreamConfig,
    HealthResponse
)
from emotion_utils import EMOTION_LABELS


# 全局变量
engine: StreamingInferenceEngine = None
output_dir = Path("realtime_tts/outputs/audio")
output_dir.mkdir(parents=True, exist_ok=True)


def create_app(
    model_path: str,
    config_path: str,
    device: str = "cuda"
) -> FastAPI:
    """
    创建FastAPI应用

    Args:
        model_path: 模型路径
        config_path: 配置路径
        device: 设备

    Returns:
        FastAPI应用实例
    """
    app = FastAPI(
        title="实时TTS API",
        description="基于Bert-VITS2的实时流式TTS服务",
        version="1.0.0"
    )

    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 初始化引擎
    global engine

    @app.on_event("startup")
    async def startup_event():
        global engine
        print(f"[Server] 初始化TTS引擎...")
        engine = StreamingInferenceEngine(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        print(f"[Server] TTS引擎初始化完成")

    @app.get("/", response_model=dict)
    async def root():
        """根路径"""
        return {
            "message": "实时TTS API服务",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "generate": "/api/tts/generate",
                "stream": "/api/tts/stream (WebSocket)",
            }
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查"""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # 获取可用说话人
        speakers = list(engine.hps.data.spk2id.keys())

        return HealthResponse(
            status="ok",
            device=engine.device,
            model_loaded=True,
            available_speakers=speakers[:10],  # 只返回前10个
            available_emotions=EMOTION_LABELS
        )

    @app.post("/api/tts/generate", response_model=TTSResponse)
    async def generate_tts(request: TTSRequest):
        """
        生成完整音频

        Args:
            request: TTS请求

        Returns:
            TTS响应，包含音频文件路径
        """
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # 验证说话人
        if request.speaker_id not in engine.hps.data.spk2id:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker '{request.speaker_id}' not found"
            )

        try:
            # 生成音频
            audio = engine.generate_complete(
                text=request.text,
                speaker_id=request.speaker_id,
                emotion=request.emotion,
                emotion_intensity=request.emotion_intensity,
                language=request.language,
                enable_prosody_planning=request.enable_prosody_planning,
                sdp_ratio=request.sdp_ratio,
                noise_scale=request.noise_scale,
                noise_scale_w=request.noise_scale_w,
                length_scale=request.length_scale,
            )

            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            output_path = output_dir / filename

            # 保存音频
            wavfile.write(
                str(output_path),
                engine.hps.data.sampling_rate,
                (audio * 32767).astype(np.int16)
            )

            # 获取统计
            stats = engine.get_stats()

            return TTSResponse(
                success=True,
                message="Audio generated successfully",
                audio_path=str(output_path),
                duration=len(audio) / engine.hps.data.sampling_rate,
                stats=stats
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tts/audio/{filename}")
    async def get_audio(filename: str):
        """获取生成的音频文件"""
        file_path = output_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(
            path=str(file_path),
            media_type="audio/wav",
            filename=filename
        )

    @app.websocket("/api/tts/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket流式TTS

        客户端发送JSON配置，服务器流式返回音频chunks
        """
        await websocket.accept()

        if engine is None:
            await websocket.send_json({
                "error": "Engine not initialized"
            })
            await websocket.close()
            return

        try:
            # 接收配置
            config_data = await websocket.receive_json()
            config = StreamConfig(**config_data)

            # 发送开始信号
            await websocket.send_json({
                "type": "start",
                "message": "Starting generation"
            })

            # 流式生成
            chunk_index = 0
            for chunk_audio in engine.stream_generate(
                text=config.text,
                speaker_id=config.speaker_id,
                emotion=config.emotion,
                emotion_intensity=config.emotion_intensity,
                language=config.language,
                enable_prosody_planning=config.enable_prosody_planning
            ):
                # 将音频转为bytes
                audio_int16 = (chunk_audio * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Base64编码
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                # 发送chunk
                await websocket.send_json({
                    "type": "chunk",
                    "index": chunk_index,
                    "audio_data": audio_b64,
                    "sample_rate": engine.hps.data.sampling_rate,
                    "samples": len(chunk_audio),
                })

                chunk_index += 1

            # 发送完成信号
            stats = engine.get_stats()
            await websocket.send_json({
                "type": "end",
                "total_chunks": chunk_index,
                "stats": stats
            })

        except WebSocketDisconnect:
            print("[WebSocket] Client disconnected")
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            await websocket.close()

    return app


def start_server(
    model_path: str,
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cuda"
):
    """
    启动FastAPI服务器

    Args:
        model_path: 模型路径
        config_path: 配置路径
        host: 主机地址
        port: 端口
        device: 设备
    """
    app = create_app(model_path, config_path, device)

    print(f"\n{'=' * 60}")
    print(f"实时TTS API服务器")
    print(f"{'=' * 60}")
    print(f"服务地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"WebSocket: ws://{host}:{port}/api/tts/stream")
    print(f"{'=' * 60}\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="启动实时TTS API服务器")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, required=True, help="配置路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    start_server(
        model_path=args.model,
        config_path=args.config,
        host=args.host,
        port=args.port,
        device=args.device
    )
