"""
性能测试

测试TTFB、RTF等关键性能指标
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from realtime_tts.core import StreamingInferenceEngine
from scipy.io import wavfile


def test_performance(
    engine: StreamingInferenceEngine,
    test_texts: list,
    speaker_id: str,
    num_runs: int = 3,
    output_dir: str = "realtime_tts/tests/benchmark_results"
):
    """
    性能测试

    Args:
        engine: 推理引擎
        test_texts: 测试文本列表
        speaker_id: 说话人ID
        num_runs: 每个文本运行次数
        output_dir: 输出目录

    Returns:
        测试结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\n{'=' * 80}")
    print(f"性能测试")
    print(f"{'=' * 80}")
    print(f"测试文本数: {len(test_texts)}")
    print(f"每个文本运行: {num_runs} 次")
    print(f"说话人: {speaker_id}")
    print(f"{'=' * 80}\n")

    for text_idx, text in enumerate(test_texts):
        print(f"\n[测试 {text_idx+1}/{len(test_texts)}] 文本: {text[:50]}...")

        text_results = []

        for run_idx in range(num_runs):
            print(f"  运行 {run_idx+1}/{num_runs}...", end="", flush=True)

            # 生成音频
            audio = engine.generate_complete(
                text=text,
                speaker_id=speaker_id,
                emotion="neutral",
                enable_prosody_planning=True
            )

            # 获取统计
            stats = engine.get_stats()

            text_results.append({
                'run_index': run_idx,
                'ttfb_ms': stats.get('ttfb', 0) * 1000,
                'total_time_s': stats.get('total_time', 0),
                'audio_duration_s': stats.get('total_audio_duration', 0),
                'rtf': stats.get('rtf', 0),
                'num_chunks': len(stats.get('chunk_times', [])),
                'avg_chunk_time_ms': stats.get('avg_chunk_time', 0) * 1000,
            })

            print(f" TTFB={stats.get('ttfb', 0)*1000:.1f}ms, RTF={stats.get('rtf', 0):.3f}")

        # 计算平均值
        avg_result = {
            'text': text,
            'text_length': len(text),
            'avg_ttfb_ms': np.mean([r['ttfb_ms'] for r in text_results]),
            'avg_total_time_s': np.mean([r['total_time_s'] for r in text_results]),
            'avg_audio_duration_s': np.mean([r['audio_duration_s'] for r in text_results]),
            'avg_rtf': np.mean([r['rtf'] for r in text_results]),
            'avg_num_chunks': np.mean([r['num_chunks'] for r in text_results]),
            'avg_chunk_time_ms': np.mean([r['avg_chunk_time_ms'] for r in text_results]),
            'runs': text_results
        }

        results.append(avg_result)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"performance_{timestamp}.json"

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print(f"测试完成！结果已保存到: {result_file}")
    print(f"{'=' * 80}\n")

    # 打印汇总
    print_performance_summary(results)

    return results


def print_performance_summary(results: list):
    """打印性能汇总"""
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)

    # 总体统计
    all_ttfb = [r['avg_ttfb_ms'] for r in results]
    all_rtf = [r['avg_rtf'] for r in results]

    print(f"\n总体性能指标:")
    print(f"  平均TTFB: {np.mean(all_ttfb):.1f} ms (最小: {np.min(all_ttfb):.1f}, 最大: {np.max(all_ttfb):.1f})")
    print(f"  平均RTF: {np.mean(all_rtf):.3f} (最小: {np.min(all_rtf):.3f}, 最大: {np.max(all_rtf):.3f})")

    # 详细结果
    print(f"\n详细结果:")
    print(f"{'文本长度':<10} {'TTFB(ms)':<12} {'RTF':<10} {'Chunks':<10} {'文本'}")
    print("-" * 80)

    for r in results:
        text_preview = r['text'][:40] + "..." if len(r['text']) > 40 else r['text']
        print(f"{r['text_length']:<10} {r['avg_ttfb_ms']:<12.1f} {r['avg_rtf']:<10.3f} "
              f"{r['avg_num_chunks']:<10.0f} {text_preview}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时TTS性能测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, required=True, help="配置路径")
    parser.add_argument("--speaker", type=str, required=True, help="说话人ID")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--runs", type=int, default=3, help="每个文本运行次数")

    args = parser.parse_args()

    # 测试文本（不同长度）
    test_texts = [
        "你好。",  # 短
        "今天天气真不错，我们一起去公园散步吧。",  # 中
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。这是唐代诗人孟浩然的作品，描绘了春天清晨的美好景象。",  # 长
        "人工智能技术的快速发展，正在深刻改变着我们的生活方式。从智能手机到自动驾驶，从语音助手到图像识别，AI技术无处不在，为我们带来了前所未有的便利。",  # 很长
    ]

    # 创建引擎
    print(f"正在加载模型...")
    engine = StreamingInferenceEngine(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )

    # 运行测试
    results = test_performance(
        engine=engine,
        test_texts=test_texts,
        speaker_id=args.speaker,
        num_runs=args.runs
    )
