"""
Gradio Web界面

提供可视化的实时TTS生成界面
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import gradio as gr
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from datetime import datetime

from realtime_tts.core import StreamingInferenceEngine
from emotion_utils import EMOTION_LABELS


# 全局引擎
engine: StreamingInferenceEngine = None


def create_interface(
    model_path: str,
    config_path: str,
    device: str = "cuda"
) -> gr.Blocks:
    """
    创建Gradio界面

    Args:
        model_path: 模型路径
        config_path: 配置路径
        device: 设备

    Returns:
        Gradio Blocks界面
    """
    global engine

    # 初始化引擎
    print(f"[WebUI] 初始化TTS引擎...")
    engine = StreamingInferenceEngine(
        model_path=model_path,
        config_path=config_path,
        device=device
    )
    print(f"[WebUI] TTS引擎初始化完成")

    # 获取可用说话人和情感
    speakers = list(engine.hps.data.spk2id.keys())
    emotions = EMOTION_LABELS

    def generate_audio(
        text,
        speaker,
        emotion,
        emotion_intensity,
        language,
        length_scale,
        noise_scale,
        enable_streaming:True
    ):
        """生成音频"""
        if not text:
            return None, "请输入文本"

        if speaker not in engine.hps.data.spk2id:
            return None, f"说话人 '{speaker}' 不存在"

        try:
            # 生成音频
            audio = engine.generate_complete(
                text=text,
                speaker_id=speaker,
                emotion=emotion,
                emotion_intensity=emotion_intensity,
                language=language,
                length_scale=length_scale,
                noise_scale=noise_scale,
                enable_prosody_planning=enable_streaming
            )

            # 获取统计
            stats = engine.get_stats()

            # 构建统计信息
            stats_text = f"""
## 生成统计

- **首包时间 (TTFB)**: {stats.get('ttfb', 0)*1000:.1f} ms
- **总生成时间**: {stats.get('total_time', 0):.2f} s
- **总音频时长**: {stats.get('total_audio_duration', 0):.2f} s
- **实时率 (RTF)**: {stats.get('rtf', 0):.3f}
- **Chunks数量**: {len(stats.get('chunk_times', []))}
- **平均Chunk时间**: {stats.get('avg_chunk_time', 0)*1000:.1f} ms
"""

            # 返回音频和统计
            return (
                (engine.hps.data.sampling_rate, audio),
                stats_text
            )

        except Exception as e:
            return None, f"生成失败: {str(e)}"

    # 创建界面
    with gr.Blocks(title="实时TTS系统") as interface:
        gr.Markdown("""
        # 实时TTS系统

        基于Bert-VITS2的流式文本转语音系统
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # 输入区域
                gr.Markdown("### 输入设置")

                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="请输入要合成的文本...",
                    lines=5
                )

                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        label="说话人",
                        choices=speakers,
                        value=speakers[0] if speakers else None
                    )

                    emotion_dropdown = gr.Dropdown(
                        label="情感",
                        choices=emotions,
                        value="neutral"
                    )

                emotion_intensity_slider = gr.Slider(
                    label="情感强度",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1
                )

                with gr.Accordion("高级设置", open=False):
                    language_radio = gr.Radio(
                        label="语言",
                        choices=["ZH", "JP", "EN"],
                        value="ZH"
                    )

                    length_scale_slider = gr.Slider(
                        label="语速 (越大越慢)",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )

                    noise_scale_slider = gr.Slider(
                        label="噪声缩放",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.6,
                        step=0.1
                    )


                generate_btn = gr.Button("生成语音", variant="primary", size="lg")

            with gr.Column(scale=1):
                # 输出区域
                gr.Markdown("### 生成结果")

                audio_output = gr.Audio(
                    label="生成的音频",
                    type="numpy"
                )

                stats_output = gr.Markdown(
                    label="统计信息",
                    value="点击生成按钮开始..."
                )

        # 示例
        gr.Markdown("### 示例文本")
        gr.Examples(
            examples=[
                ["天气开始下雨。他们马上也去厂房。"],
                ["你好，欢迎使用实时语音合成！"],
                ["春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"],
            ],
            inputs=[text_input]
        )

        # 绑定生成函数
        generate_btn.click(
            fn=generate_audio,
            inputs=[
                text_input,
                speaker_dropdown,
                emotion_dropdown,
                emotion_intensity_slider,
                language_radio,
                length_scale_slider,
                noise_scale_slider
            ],
            outputs=[audio_output, stats_output]
        )


    return interface


def launch_interface(
    model_path: str,
    config_path: str,
    device: str = "cuda",
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False
):
    """
    启动Gradio界面

    Args:
        model_path: 模型路径
        config_path: 配置路径
        device: 设备
        server_name: 服务器地址
        server_port: 端口
        share: 是否创建公共链接
    """
    interface = create_interface(model_path, config_path, device)

    print(f"\n{'=' * 60}")
    print(f"实时TTS Web界面")
    print(f"{'=' * 60}")
    print(f"本地访问: http://localhost:{server_port}")
    print(f"网络访问: http://{server_name}:{server_port}")
    print(f"{'=' * 60}\n")

    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="启动实时TTS Web界面")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, required=True, help="配置路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")

    args = parser.parse_args()

    launch_interface(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        server_port=args.port,
        share=args.share
    )
