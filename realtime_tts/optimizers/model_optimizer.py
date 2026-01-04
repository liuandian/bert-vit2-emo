"""
模型优化工具

提供模型量化、性能分析等优化功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import time
import numpy as np


def quantize_model(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    quantize_layers: Optional[List[type]] = None
) -> nn.Module:
    """
    量化模型以加速推理并减少内存占用

    Args:
        model: PyTorch模型
        dtype: 量化类型（torch.qint8, torch.float16）
        quantize_layers: 要量化的层类型列表

    Returns:
        量化后的模型
    """
    model.eval()

    if quantize_layers is None:
        # 默认量化Linear和Conv层
        quantize_layers = [torch.nn.Linear, torch.nn.Conv1d]

    if dtype == torch.qint8:
        # INT8动态量化
        print("[ModelOptimizer] 应用INT8动态量化...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec=set(quantize_layers),
            dtype=dtype
        )

    elif dtype == torch.float16:
        # FP16半精度
        print("[ModelOptimizer] 转换为FP16半精度...")
        quantized_model = model.half()

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return quantized_model


def profile_model(
    model: nn.Module,
    dummy_inputs: Tuple,
    device: str = "cuda",
    warmup_runs: int = 5,
    benchmark_runs: int = 20
) -> Dict:
    """
    性能分析：测量模型推理时间和内存占用

    Args:
        model: 模型
        dummy_inputs: 示例输入
        device: 设备
        warmup_runs: 预热运行次数
        benchmark_runs: 基准测试运行次数

    Returns:
        性能统计字典
    """
    model.eval()
    model = model.to(device)

    # 将输入移到设备
    dummy_inputs = tuple(
        inp.to(device) if isinstance(inp, torch.Tensor) else inp
        for inp in dummy_inputs
    )

    print(f"[ProfileModel] 预热 {warmup_runs} 次...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*dummy_inputs)

    # GPU同步
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"[ProfileModel] 基准测试 {benchmark_runs} 次...")

    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.time()

            if device == "cuda":
                torch.cuda.synchronize()

            _ = model(*dummy_inputs)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.time()
            times.append(end - start)

    # 测量内存占用
    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    else:
        mem_allocated = 0
        mem_reserved = 0

    # 统计
    times = np.array(times)

    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),
        'mem_allocated_mb': mem_allocated,
        'mem_reserved_mb': mem_reserved,
        'device': device,
        'runs': benchmark_runs,
    }


def print_profile_results(stats: Dict):
    """
    打印性能分析结果

    Args:
        stats: profile_model返回的统计字典
    """
    print("\n" + "=" * 60)
    print("模型性能分析结果")
    print("=" * 60)
    print(f"设备: {stats['device']}")
    print(f"运行次数: {stats['runs']}")
    print(f"\n推理时间:")
    print(f"  平均: {stats['mean_time']*1000:.2f} ms")
    print(f"  标准差: {stats['std_time']*1000:.2f} ms")
    print(f"  中位数: {stats['median_time']*1000:.2f} ms")
    print(f"  最小: {stats['min_time']*1000:.2f} ms")
    print(f"  最大: {stats['max_time']*1000:.2f} ms")

    if stats['device'] == "cuda":
        print(f"\nGPU内存:")
        print(f"  已分配: {stats['mem_allocated_mb']:.1f} MB")
        print(f"  已保留: {stats['mem_reserved_mb']:.1f} MB")

    print("=" * 60 + "\n")


def compare_models(
    original_model: nn.Module,
    optimized_model: nn.Module,
    dummy_inputs: Tuple,
    device: str = "cuda"
) -> Dict:
    """
    对比原始模型和优化后模型的性能

    Args:
        original_model: 原始模型
        optimized_model: 优化后模型
        dummy_inputs: 示例输入
        device: 设备

    Returns:
        对比结果字典
    """
    print("[CompareModels] 分析原始模型...")
    original_stats = profile_model(original_model, dummy_inputs, device)

    print("[CompareModels] 分析优化后模型...")
    optimized_stats = profile_model(optimized_model, dummy_inputs, device)

    # 计算提升
    speedup = original_stats['mean_time'] / optimized_stats['mean_time']
    memory_reduction = (
        (original_stats['mem_allocated_mb'] - optimized_stats['mem_allocated_mb'])
        / original_stats['mem_allocated_mb'] * 100
        if original_stats['mem_allocated_mb'] > 0
        else 0
    )

    return {
        'original': original_stats,
        'optimized': optimized_stats,
        'speedup': speedup,
        'memory_reduction_percent': memory_reduction,
    }


def print_comparison_results(results: Dict):
    """
    打印对比结果

    Args:
        results: compare_models返回的结果
    """
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)

    orig = results['original']
    opt = results['optimized']

    print(f"\n原始模型:")
    print(f"  平均推理时间: {orig['mean_time']*1000:.2f} ms")
    if orig['device'] == "cuda":
        print(f"  GPU内存: {orig['mem_allocated_mb']:.1f} MB")

    print(f"\n优化后模型:")
    print(f"  平均推理时间: {opt['mean_time']*1000:.2f} ms")
    if opt['device'] == "cuda":
        print(f"  GPU内存: {opt['mem_allocated_mb']:.1f} MB")

    print(f"\n性能提升:")
    print(f"  加速比: {results['speedup']:.2f}x")

    if orig['device'] == "cuda" and results['memory_reduction_percent'] != 0:
        print(f"  内存减少: {results['memory_reduction_percent']:.1f}%")

    print("=" * 60 + "\n")


def count_parameters(model: nn.Module) -> Dict:
    """
    统计模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'total_params_M': total_params / 1e6,
    }


def print_model_info(model: nn.Module):
    """
    打印模型信息

    Args:
        model: PyTorch模型
    """
    params = count_parameters(model)

    print("\n" + "=" * 60)
    print("模型信息")
    print("=" * 60)
    print(f"总参数量: {params['total_params']:,} ({params['total_params_M']:.2f}M)")
    print(f"可训练参数: {params['trainable_params']:,}")
    print(f"固定参数: {params['non_trainable_params']:,}")
    print("=" * 60 + "\n")


# 使用示例
if __name__ == "__main__":
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 创建模型和输入
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel().to(device)
    dummy_input = (torch.randn(1, 100),)

    print("=" * 60)
    print("模型优化工具测试")
    print("=" * 60)

    # 打印模型信息
    print_model_info(model)

    # 性能分析
    print("\n1. 原始模型性能分析")
    stats = profile_model(model, dummy_input, device)
    print_profile_results(stats)

    # 量化
    print("\n2. 模型量化")
    if device == "cuda":
        quantized_model = quantize_model(model.cpu(), dtype=torch.qint8)
        quantized_model = quantized_model.to(device)
    else:
        quantized_model = quantize_model(model, dtype=torch.qint8)

    # 对比
    print("\n3. 模型对比")
    comparison = compare_models(model, quantized_model, dummy_input, device)
    print_comparison_results(comparison)
