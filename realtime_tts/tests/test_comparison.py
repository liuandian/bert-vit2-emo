"""
对比测试

对比流式vs非流式的延迟和音质
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
from infer import infer, get_net_g
import utils
from scipy.io import wavfile


def test_comparison(
    model_path: str,
    config_path: str,
    test_text: str,
    speaker_id: str,
    device: str = "cuda",
    output_dir: str = "realtime_tts/tests/benchmark_results"
):
    """
    对比测试：流式 vs 非流式

    Args:
        model_path: 模型路径
        config_path: 配置路径
        test_text: 测试文本
        speaker_id: 说话人ID
        device: 设备
        output_dir: 输出目录

    Returns:
        对比结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 80}")
    print(f"对比测试: 流式 vs 非流式")
    print(f"{'=' * 80}")
    print(f"测试文本: {test_text}")
    print(f"说话人: {speaker_id}")
    print(f"{'=' * 80}\n")

    # === 测试流式生成 ===
    print("[1] 测试流式生成...")

    streaming_engine = StreamingInferenceEngine(
        model_path=model_path,
        config_path=config_path,
        device=device
    )

    streaming_audio = streaming_engine.generate_complete(
        text=test_text,
        speaker_id=speaker_id,
        emotion="neutral",
        enable_prosody_planning=True
    )

    streaming_stats = streaming_engine.get_stats()

    # 保存音频
    streaming_audio_path = output_dir / f"streaming_{timestamp}.wav"
    wavfile.write(
        str(streaming_audio_path),
        streaming_engine.hps.data.sampling_rate,
        (streaming_audio * 32767).astype(np.int16)
    )

    print(f"  TTFB: {streaming_stats.get('ttfb', 0)*1000:.1f} ms")
    print(f"  总时间: {streaming_stats.get('total_time', 0):.2f} s")
    print(f"  RTF: {streaming_stats.get('rtf', 0):.3f}")
    print(f"  音频保存: {streaming_audio_path}")

    # === 测试非流式生成 ===
    print("\n[2] 测试非流式生成...")

    # 加载模型（非流式）
    hps = utils.get_hparams_from_file(config_path)
    net_g = get_net_g(
        model_path=model_path,
        version=getattr(hps, 'version', "2.3"),
        device=device,
        hps=hps
    )
    net_g.eval()

    # 生成音频
    start_time = time.time()

    non_streaming_audio = infer(
        text=test_text,
        emotion="neutral",
        sid=speaker_id,
        language="ZH",
        hps=hps,
        net_g=net_g,
        device=device
    )

    non_streaming_time = time.time() - start_time

    # 保存音频
    non_streaming_audio_path = output_dir / f"non_streaming_{timestamp}.wav"
    wavfile.write(
        str(non_streaming_audio_path),
        hps.data.sampling_rate,
        (non_streaming_audio * 32767).astype(np.int16)
    )

    non_streaming_duration = len(non_streaming_audio) / hps.data.sampling_rate
    non_streaming_rtf = non_streaming_time / non_streaming_duration

    print(f"  总时间: {non_streaming_time:.2f} s")
    print(f"  RTF: {non_streaming_rtf:.3f}")
    print(f"  音频保存: {non_streaming_audio_path}")

    # === 对比结果 ===
    results = {
        'test_text': test_text,
        'speaker_id': speaker_id,
        'timestamp': timestamp,
        'streaming': {
            'ttfb_ms': streaming_stats.get('ttfb', 0) * 1000,
            'total_time_s': streaming_stats.get('total_time', 0),
            'audio_duration_s': streaming_stats.get('total_audio_duration', 0),
            'rtf': streaming_stats.get('rtf', 0),
            'num_chunks': len(streaming_stats.get('chunk_times', [])),
            'audio_path': str(streaming_audio_path),
        },
        'non_streaming': {
            'total_time_s': non_streaming_time,
            'audio_duration_s': non_streaming_duration,
            'rtf': non_streaming_rtf,
            'audio_path': str(non_streaming_audio_path),
        }
    }

    # 保存结果
    result_file = output_dir / f"comparison_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print(f"对比结果已保存到: {result_file}")
    print(f"{'=' * 80}\n")

    # 打印对比
    print_comparison_summary(results)

    return results


def print_comparison_summary(results: dict):
    """打印对比汇总"""
    print("\n" + "=" * 80)
    print("对比汇总")
    print("=" * 80)

    streaming = results['streaming']
    non_streaming = results['non_streaming']

    print(f"\n流式生成:")
    print(f"  首包时间 (TTFB): {streaming['ttfb_ms']:.1f} ms")
    print(f"  总生成时间: {streaming['total_time_s']:.2f} s")
    print(f"  音频时长: {streaming['audio_duration_s']:.2f} s")
    print(f"  实时率 (RTF): {streaming['rtf']:.3f}")
    print(f"  Chunks数: {streaming['num_chunks']}")

    print(f"\n非流式生成:")
    print(f"  总生成时间: {non_streaming['total_time_s']:.2f} s")
    print(f"  音频时长: {non_streaming['audio_duration_s']:.2f} s")
    print(f"  实时率 (RTF): {non_streaming['rtf']:.3f}")

    print(f"\n性能对比:")

    # 时间对比
    time_diff = streaming['total_time_s'] - non_streaming['total_time_s']
    time_diff_percent = (time_diff / non_streaming['total_time_s']) * 100

    if time_diff > 0:
        print(f"  总时间: 流式更慢 {abs(time_diff):.2f}s ({abs(time_diff_percent):.1f}%)")
    else:
        print(f"  总时间: 流式更快 {abs(time_diff):.2f}s ({abs(time_diff_percent):.1f}%)")

    # RTF对比
    rtf_diff = streaming['rtf'] - non_streaming['rtf']
    print(f"  RTF差异: {rtf_diff:+.3f}")

    # 首包时间优势
    print(f"  首包时间优势: 流式有 {streaming['ttfb_ms']:.1f}ms TTFB（非流式需等待完整生成）")

    print(f"\n音频文件:")
    print(f"  流式: {streaming['audio_path']}")
    print(f"  非流式: {non_streaming['audio_path']}")
    print(f"\n  可使用音频播放器对比音质差异")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流式vs非流式对比测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, required=True, help="配置路径")
    parser.add_argument("--speaker", type=str, required=True, help="说话人ID")
    parser.add_argument("--text", type=str,
                       default="今天天气真不错，阳光明媚。我们一起去公园散步，欣赏美丽的风景。",
                       help="测试文本")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    # 运行对比测试
    results = test_comparison(
        model_path=args.model,
        config_path=args.config,
        test_text=args.text,
        speaker_id=args.speaker,
        device=args.device
    )
