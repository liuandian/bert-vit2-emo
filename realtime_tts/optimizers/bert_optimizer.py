"""
BERT特征提取优化

提供BERT特征缓存、批量处理等优化功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import hashlib
import pickle
from pathlib import Path

from text import get_bert


class BertOptimizer:
    """BERT特征提取优化器"""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
        cache_size: int = 1000,
        enable_cache: bool = True
    ):
        """
        初始化BERT优化器

        Args:
            device: 运行设备
            cache_dir: 缓存目录（如果为None，只使用内存缓存）
            cache_size: 内存缓存大小（LRU）
            enable_cache: 是否启用缓存
        """
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_size = cache_size
        self.enable_cache = enable_cache

        # 内存缓存（LRU）
        self._memory_cache = OrderedDict()

        # 统计信息
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0

        # 创建缓存目录
        if self.cache_dir and self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_bert(
        self,
        text: str,
        word2ph: List[int],
        language: str,
        style_text: Optional[str] = None,
        style_weight: float = 0.7
    ) -> torch.Tensor:
        """
        提取BERT特征（带缓存）

        Args:
            text: 文本
            word2ph: word2ph映射
            language: 语言
            style_text: 风格参考文本
            style_weight: 风格权重

        Returns:
            bert特征 [1024, seq_len]
        """
        self._total_requests += 1

        # 生成缓存key
        cache_key = self._make_cache_key(
            text, language, style_text, style_weight
        )

        # 检查缓存
        if self.enable_cache:
            bert_features = self._get_from_cache(cache_key)
            if bert_features is not None:
                self._cache_hits += 1
                return bert_features

        # 缓存未命中，调用原始get_bert
        self._cache_misses += 1

        bert_features = get_bert(
            text,
            word2ph,
            language,
            self.device,
            style_text,
            style_weight
        )

        # 保存到缓存
        if self.enable_cache:
            self._save_to_cache(cache_key, bert_features)

        return bert_features

    def extract_batch(
        self,
        texts: List[str],
        word2phs: List[List[int]],
        languages: List[str],
        style_texts: Optional[List[str]] = None,
        style_weights: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        """
        批量提取BERT特征

        注意：当前实现是串行的，因为get_bert不支持批处理
        这个方法主要用于未来的批处理优化

        Args:
            texts: 文本列表
            word2phs: word2ph列表
            languages: 语言列表
            style_texts: 风格参考文本列表
            style_weights: 风格权重列表

        Returns:
            bert特征列表
        """
        if style_texts is None:
            style_texts = [None] * len(texts)
        if style_weights is None:
            style_weights = [0.7] * len(texts)

        results = []
        for text, word2ph, lang, style_text, style_weight in zip(
            texts, word2phs, languages, style_texts, style_weights
        ):
            bert = self.extract_bert(
                text, word2ph, lang, style_text, style_weight
            )
            results.append(bert)

        return results

    def _make_cache_key(
        self,
        text: str,
        language: str,
        style_text: Optional[str],
        style_weight: float
    ) -> str:
        """
        生成缓存key

        Args:
            text: 文本
            language: 语言
            style_text: 风格文本
            style_weight: 风格权重

        Returns:
            cache key
        """
        # 组合所有参数
        key_str = f"{text}|{language}|{style_text}|{style_weight:.2f}"

        # 使用MD5生成短key
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """
        从缓存获取特征

        Args:
            cache_key: 缓存key

        Returns:
            bert特征或None
        """
        # 先查内存缓存
        if cache_key in self._memory_cache:
            # LRU：移到末尾
            self._memory_cache.move_to_end(cache_key)
            return self._memory_cache[cache_key]

        # 查文件缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        bert_features = pickle.load(f)

                    # 加载到内存缓存
                    self._add_to_memory_cache(cache_key, bert_features)

                    return bert_features
                except Exception as e:
                    print(f"[BertCache] 加载缓存失败: {e}")

        return None

    def _save_to_cache(self, cache_key: str, bert_features: torch.Tensor):
        """
        保存特征到缓存

        Args:
            cache_key: 缓存key
            bert_features: bert特征
        """
        # 保存到内存缓存
        self._add_to_memory_cache(cache_key, bert_features)

        # 保存到文件缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(bert_features, f)
            except Exception as e:
                print(f"[BertCache] 保存缓存失败: {e}")

    def _add_to_memory_cache(self, cache_key: str, bert_features: torch.Tensor):
        """
        添加到内存缓存（LRU）

        Args:
            cache_key: 缓存key
            bert_features: bert特征
        """
        # 如果已存在，移到末尾
        if cache_key in self._memory_cache:
            self._memory_cache.move_to_end(cache_key)
        else:
            self._memory_cache[cache_key] = bert_features

            # 如果超过缓存大小，删除最旧的
            if len(self._memory_cache) > self.cache_size:
                self._memory_cache.popitem(last=False)

    def clear_cache(self, clear_disk: bool = False):
        """
        清空缓存

        Args:
            clear_disk: 是否也清空磁盘缓存
        """
        self._memory_cache.clear()

        if clear_disk and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    print(f"[BertCache] 删除缓存文件失败: {e}")

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        hit_rate = (
            self._cache_hits / self._total_requests
            if self._total_requests > 0
            else 0.0
        )

        disk_cache_size = 0
        if self.cache_dir and self.cache_dir.exists():
            disk_cache_size = len(list(self.cache_dir.glob("*.pkl")))

        return {
            'total_requests': self._total_requests,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self._memory_cache),
            'memory_cache_max': self.cache_size,
            'disk_cache_size': disk_cache_size,
            'cache_enabled': self.enable_cache,
        }

    def print_cache_stats(self):
        """打印缓存统计信息"""
        stats = self.get_cache_stats()

        print("\n" + "=" * 60)
        print("BERT缓存统计")
        print("=" * 60)
        print(f"总请求数: {stats['total_requests']}")
        print(f"缓存命中: {stats['cache_hits']}")
        print(f"缓存未命中: {stats['cache_misses']}")
        print(f"命中率: {stats['hit_rate']:.2%}")
        print(f"内存缓存: {stats['memory_cache_size']}/{stats['memory_cache_max']}")

        if self.cache_dir:
            print(f"磁盘缓存: {stats['disk_cache_size']} 个文件")

        print("=" * 60 + "\n")


# 使用示例
if __name__ == "__main__":
    from text.cleaner import clean_text

    # 创建优化器
    optimizer = BertOptimizer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir="./bert_cache",
        cache_size=100,
        enable_cache=True
    )

    # 测试文本
    test_texts = [
        "今天天气真好",
        "我们一起去公园",
        "今天天气真好",  # 重复，应该命中缓存
    ]

    print("测试BERT缓存功能\n")

    for i, text in enumerate(test_texts):
        print(f"[{i+1}] 提取BERT特征: {text}")

        # 清理文本
        norm_text, phones, tones, word2ph = clean_text(text, "ZH")

        # 提取BERT特征
        bert = optimizer.extract_bert(
            norm_text,
            word2ph,
            "ZH"
        )

        print(f"    BERT shape: {bert.shape}")

    # 打印统计
    optimizer.print_cache_stats()

    # 清空缓存
    print("清空缓存...")
    optimizer.clear_cache(clear_disk=True)

    optimizer.print_cache_stats()
