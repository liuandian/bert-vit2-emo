"""
配置检查工具

检查模型文件、配置文件和说话人是否匹配
"""

import os
import json
import sys

def check_config():
    """检查配置"""
    print("\n" + "=" * 60)
    print("实时TTS配置检查")
    print("=" * 60 + "\n")

    # 检查模型文件
    model_path = "data/casia/models/G_0.pth"
    print(f"1️⃣ 检查模型文件: {model_path}")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"   ✅ 模型文件存在 ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ 模型文件不存在")
        print(f"   提示: 检查路径是否正确")
        return

    # 检查配置文件
    config_path = "configs/config.json"
    print(f"\n2️⃣ 检查配置文件: {config_path}")

    if os.path.exists(config_path):
        print(f"   ✅ 配置文件存在")

        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 检查说话人
        spk2id = config['data']['spk2id']
        n_speakers = config['data']['n_speakers']

        print(f"\n3️⃣ 说话人配置:")
        print(f"   总说话人数: {n_speakers}")
        print(f"   spk2id中的说话人数: {len(spk2id)}")

        # 显示前10个说话人
        print(f"\n   前10个说话人:")
        for i, (name, idx) in enumerate(list(spk2id.items())[:10]):
            print(f"      {idx}. {name}")

        if len(spk2id) > 10:
            print(f"   ... 还有 {len(spk2id) - 10} 个")

    else:
        print(f"   ❌ 配置文件不存在")
        return

    # 检查是否有自定义说话人
    print(f"\n4️⃣ 检查特定说话人:")

    custom_speakers = ["liuchanhg"]  # 添加你想检查的说话人

    for spk in custom_speakers:
        if spk in spk2id:
            print(f"   ✅ '{spk}' 存在于配置中 (ID: {spk2id[spk]})")
        else:
            print(f"   ❌ '{spk}' 不存在于配置中")
            print(f"   提示: 可能需要使用正确的配置文件，或使用配置中存在的说话人")

    # 检查版本
    print(f"\n5️⃣ 配置版本:")
    version = config.get('version', 'unknown')
    print(f"   版本: {version}")

    # 检查采样率
    sampling_rate = config['data']['sampling_rate']
    print(f"   采样率: {sampling_rate} Hz")

    # 推荐配置
    print(f"\n" + "=" * 60)
    print("📝 推荐配置:")
    print("=" * 60)

    print(f"\n在 basic_usage.py 中使用:")
    print(f"   MODEL_PATH = \"{model_path}\"")
    print(f"   CONFIG_PATH = \"{config_path}\"")

    # 推荐一个存在的说话人
    first_speaker = list(spk2id.keys())[0]
    print(f"   SPEAKER_ID = \"{first_speaker}\"  # 或其他配置中存在的说话人")

    print(f"\n可用的说话人（部分）:")
    for name in list(spk2id.keys())[:5]:
        print(f"   - {name}")

    print(f"\n💡 提示:")
    print(f"   - 如果你的模型是自己训练的，确保使用训练时的配置文件")
    print(f"   - 说话人ID必须在配置文件的 spk2id 中存在")
    print(f"   - 路径是相对于项目根目录: /Users/liuan/work/TTS-test/Bert-VITS2/")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    check_config()
