"""
流式生成演示

演示流式生成的使用方法
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from realtime_tts.core import StreamingInferenceEngine
from scipy.io import wavfile
import numpy as np


def main():
    # 配置
    MODEL_PATH = "data/casia/models/G_13000.pth"  # 修改为你的模型路径（相对于项目根目录）
    CONFIG_PATH = "data/casia/configs/config.json"  # 修改为你的配置路径
    SPEAKER_ID = "liuchanhg"  # 修改为你的说话人ID（必须在config.json的spk2id中存在）
    DEVICE = "cuda"

    # 创建引擎
    print("正在加载模型...")
    engine = StreamingInferenceEngine(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        device=DEVICE
    )
    print("模型加载完成！\n")

    # 测试文本
    text = """
    春眠不觉晓，处处闻啼鸟。
    夜来风雨声，花落知多少。
    这是唐代诗人孟浩然的经典诗作，描绘了春天清晨的美好景象。
    """

    print(f"输入文本:\n{text}\n")
    print("=" * 60)

    # 流式生成
    audio_chunks = []
    print("开始流式生成...\n")

    for i, chunk_audio in enumerate(engine.stream_generate(
        text=text,
        speaker_id=SPEAKER_ID,
        emotion="neutral",
        language="ZH",
        enable_prosody_planning=True
    )):
        chunk_duration = len(chunk_audio) / engine.hps.data.sampling_rate
        print(f"收到 Chunk {i+1}: {len(chunk_audio)} samples ({chunk_duration:.2f}s)")

        audio_chunks.append(chunk_audio)

        # 这里可以实时播放或流式传输
        # play_audio_chunk(chunk_audio, engine.hps.data.sampling_rate)

    # 保存完整音频
    final_audio = np.concatenate(audio_chunks)
    output_path = "realtime_tts/outputs/audio/streaming_demo.wav"

    wavfile.write(
        output_path,
        engine.hps.data.sampling_rate,
        (final_audio * 32767).astype(np.int16)
    )

    print(f"\n音频已保存: {output_path}")
    print(f"总时长: {len(final_audio)/engine.hps.data.sampling_rate:.2f}秒")
    print(f"总chunks: {len(audio_chunks)}\n")

    # 打印性能统计
    engine.print_stats()


if __name__ == "__main__":
    main()
