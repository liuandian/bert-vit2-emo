"""
基础使用示例

演示如何使用实时TTS系统生成音频
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
    DEVICE = "cuda"  # 或 "cuda"

    # ⚠️ 注意：
    # 1. 如果你的模型是自己训练的，请确保：
    #    - MODEL_PATH 指向你的模型文件
    #    - CONFIG_PATH 指向训练时使用的配置文件（包含你的说话人）
    #    - SPEAKER_ID 在配置文件的 spk2id 中存在
    # 2. 如果使用原神角色模型，SPEAKER_ID 可以是：
    #    派蒙_ZH, 纳西妲_ZH, 钟离_ZH 等（共850个）

    # 创建引擎
    print("正在加载模型...")
    engine = StreamingInferenceEngine(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        device=DEVICE
    )
    print("模型加载完成！\n")

    # 生成音频
    text = "今天天气真不错，阳光明媚，我们一起去公园散步吧。今天天气不好，不好看，我们不要一起去公园散步吧。"
    print(f"输入文本: {text}\n")

    print("正在生成音频...")
    audio = engine.generate_complete(
        text=text,
        speaker_id=SPEAKER_ID,
        emotion="fear",
        emotion_intensity=0.8,
        language="ZH"
    )

    # 保存音频
    output_path = "realtime_tts/outputs/audio/basic_example.wav"
    wavfile.write(
        output_path,
        engine.hps.data.sampling_rate,
        (audio * 32767).astype(np.int16)
    )

    print(f"音频已保存: {output_path}")
    print(f"音频时长: {len(audio)/engine.hps.data.sampling_rate:.2f}秒\n")

    # 打印统计
    engine.print_stats()


if __name__ == "__main__":
    main()
