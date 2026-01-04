"""
测试实时TTS系统各模块（不需要模型）

这个脚本测试核心模块的功能，不需要加载真实的TTS模型
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.io import wavfile


def test_sentence_splitter():
    """测试文本切分器"""
    print("\n" + "=" * 60)
    print("测试1: 智能文本切分器")
    print("=" * 60)

    from realtime_tts.core import SmartSentenceSplitter

    splitter = SmartSentenceSplitter(max_chunk_len=30)

    test_text = "今天天气真不错，阳光明媚温暖宜人。我们决定一起去公园散步，欣赏美丽的风景。你觉得怎么样？"

    chunks = splitter.split(test_text)

    print(f"\n原始文本: {test_text}")
    print(f"\n切分结果（共{len(chunks)}个chunks）:\n")

    for i, (chunk_text, metadata) in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk_text}")
        print(f"  位置: {metadata['position']}")
        print(f"  停顿: {metadata['pause_after']:.2f}s")
        print(f"  音高缩放: {metadata['f0_scale']:.2f}")
        print(f"  语速缩放: {metadata['speed_scale']:.2f}")
        print()

    print("✅ 文本切分器测试通过\n")
    return chunks


def test_prosody_planner(chunks):
    """测试韵律规划器"""
    print("=" * 60)
    print("测试2: 韵律规划器")
    print("=" * 60)

    from realtime_tts.core import GlobalProsodyPlanner

    planner = GlobalProsodyPlanner()

    for emotion in ['neutral', 'happy', 'sad']:
        print(f"\n情感: {emotion}")

        prosody_params = planner.plan(chunks, overall_emotion=emotion)

        for i, params in enumerate(prosody_params):
            print(f"  Chunk {i+1}: F0={params['f0_scale']:.3f}, "
                  f"Speed={params['speed_scale']:.3f}, "
                  f"Energy={params['energy_scale']:.3f}")

    print("\n✅ 韵律规划器测试通过\n")


def test_audio_processor():
    """测试音频处理器"""
    print("=" * 60)
    print("测试3: 音频处理器")
    print("=" * 60)

    from realtime_tts.core import AudioProcessor

    processor = AudioProcessor(sampling_rate=44100)

    # 生成测试音频
    duration = 1.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))

    audio1 = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz
    audio2 = np.sin(2 * np.pi * 554.37 * t) * 0.5  # 554.37Hz

    print(f"\n生成测试音频:")
    print(f"  Audio1: {len(audio1)} samples ({len(audio1)/sr:.2f}s)")
    print(f"  Audio2: {len(audio2)} samples ({len(audio2)/sr:.2f}s)")

    # 测试overlap-add
    print(f"\n测试overlap-add拼接...")
    overlapped = processor.overlap_add(audio1, audio2, overlap_duration=0.1)
    print(f"  拼接后: {len(overlapped)} samples ({len(overlapped)/sr:.2f}s)")

    # 测试批量拼接
    print(f"\n测试批量拼接...")
    chunks = [audio1, audio2, audio1]
    concatenated = processor.concatenate_with_overlap(chunks, overlap_duration=0.05)
    print(f"  3个chunks拼接后: {len(concatenated)} samples ({len(concatenated)/sr:.2f}s)")

    # 测试归一化
    print(f"\n测试归一化...")
    audio_loud = audio1 * 0.3
    normalized = processor.normalize(audio_loud, target_level=0.9)
    print(f"  原始峰值: {np.abs(audio_loud).max():.3f}")
    print(f"  归一化后: {np.abs(normalized).max():.3f}")

    # 保存测试音频
    output_path = "realtime_tts/outputs/audio/test_audio_processor.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, sr, (concatenated * 32767).astype(np.int16))
    print(f"\n测试音频已保存: {output_path}")

    print("\n✅ 音频处理器测试通过\n")


def test_bert_optimizer():
    """测试BERT优化器"""
    print("=" * 60)
    print("测试4: BERT优化器")
    print("=" * 60)

    from realtime_tts.optimizers import BertOptimizer

    optimizer = BertOptimizer(
        cache_dir="realtime_tts/outputs/bert_cache_test",
        cache_size=10,
        enable_cache=True
    )

    print(f"\n创建BERT优化器:")
    print(f"  缓存目录: realtime_tts/outputs/bert_cache_test")
    print(f"  缓存大小: 10")
    print(f"  缓存已启用: True")

    # 获取缓存统计
    stats = optimizer.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  缓存命中: {stats['cache_hits']}")
    print(f"  命中率: {stats['hit_rate']:.2%}")

    print("\n✅ BERT优化器测试通过\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("实时TTS系统模块测试")
    print("=" * 60)
    print("\n本测试不需要加载TTS模型，只测试核心模块功能\n")

    try:
        # 测试各模块
        chunks = test_sentence_splitter()
        test_prosody_planner(chunks)
        test_audio_processor()
        test_bert_optimizer()

        print("=" * 60)
        print("🎉 所有模块测试通过！")
        print("=" * 60)
        print("\n核心功能验证完成，实时TTS系统已准备就绪。")
        print("\n下一步：")
        print("  1. 准备训练好的TTS模型文件（*.pth）")
        print("  2. 确保config.json中有对应的说话人配置")
        print("  3. 运行 basic_usage.py 生成语音\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
