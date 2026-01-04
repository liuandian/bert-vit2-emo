"""
韵律规划器

为文本chunks规划全局韵律参数，确保chunk拼接时的韵律连续性
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class GlobalProsodyPlanner:
    """全局韵律规划器"""

    def __init__(self):
        """初始化韵律规划器"""
        # 音高范围（相对缩放）
        self.f0_range = (0.8, 1.2)

        # 语速范围（相对倍数）
        self.speed_range = (0.8, 1.3)

        # 能量范围（相对倍数）
        self.energy_range = (0.7, 1.2)

        # 情感到韵律参数的映射
        self.emotion_params = self._init_emotion_params()

    def _init_emotion_params(self) -> Dict:
        """初始化情感参数映射"""
        return {
            'neutral': {
                'f0_scale': 1.0,
                'speed_scale': 1.0,
                'energy_scale': 1.0,
                'variation': 0.05  # 变化幅度
            },
            'happy': {
                'f0_scale': 1.15,
                'speed_scale': 1.1,
                'energy_scale': 1.2,
                'variation': 0.1
            },
            'sad': {
                'f0_scale': 0.85,
                'speed_scale': 0.9,
                'energy_scale': 0.8,
                'variation': 0.08
            },
            'angry': {
                'f0_scale': 1.2,
                'speed_scale': 1.15,
                'energy_scale': 1.3,
                'variation': 0.12
            },
            'fear': {
                'f0_scale': 1.1,
                'speed_scale': 1.2,
                'energy_scale': 0.9,
                'variation': 0.15
            },
            'fearful': {  # 别名
                'f0_scale': 1.1,
                'speed_scale': 1.2,
                'energy_scale': 0.9,
                'variation': 0.15
            },
            'surprise': {
                'f0_scale': 1.25,
                'speed_scale': 1.0,
                'energy_scale': 1.15,
                'variation': 0.13
            },
            'surprised': {  # 别名
                'f0_scale': 1.25,
                'speed_scale': 1.0,
                'energy_scale': 1.15,
                'variation': 0.13
            },
        }

    def plan(
        self,
        chunks: List[Tuple[str, Dict]],
        overall_emotion: str = "neutral",
        smooth_factor: float = 0.3
    ) -> List[Dict]:
        """
        为所有chunks规划韵律参数

        Args:
            chunks: [(chunk_text, metadata), ...] from SentenceSplitter
            overall_emotion: 整体情感
            smooth_factor: 平滑因子（0-1），越大越平滑

        Returns:
            prosody_params: 每个chunk的韵律参数列表
        """
        if not chunks:
            return []

        num_chunks = len(chunks)
        prosody_params = []

        # 获取情感基础参数
        emotion_base = self.emotion_params.get(
            overall_emotion,
            self.emotion_params['neutral']
        )

        for i, (text, metadata) in enumerate(chunks):
            # 从metadata继承部分参数
            params = {
                'text': text,
                'chunk_index': i,
                'total_chunks': num_chunks,
            }

            # 位置信息
            position = metadata.get('position', 'middle')
            params['position'] = position

            # ===== 音高调整 =====
            # 基于位置的音高调整
            position_f0_map = {
                'start': 1.08,
                'middle': 1.0,
                'end': 0.92,
                'single': 1.0,
            }
            position_f0 = position_f0_map.get(position, 1.0)

            # 应用情感调整
            params['f0_scale'] = position_f0 * emotion_base['f0_scale']

            # ===== 语速调整 =====
            # 从metadata继承
            metadata_speed = metadata.get('speed_scale', 1.0)

            # 应用情感调整
            params['speed_scale'] = metadata_speed * emotion_base['speed_scale']

            # ===== 能量调整 =====
            params['energy_scale'] = emotion_base['energy_scale']

            # ===== 其他参数 =====
            params['pause_after'] = metadata.get('pause_after', 0.3)
            params['chunk_length'] = metadata.get('chunk_length', len(text))

            # 平滑处理：与前一个chunk的参数不要相差太大
            if i > 0 and smooth_factor > 0:
                params = self._smooth_transition(
                    prosody_params[-1],
                    params,
                    smooth_factor
                )

            prosody_params.append(params)

        # 全局平滑：使用正弦曲线调整整体音高走势
        prosody_params = self._apply_global_contour(prosody_params, overall_emotion)

        return prosody_params

    def _smooth_transition(
        self,
        prev_params: Dict,
        current_params: Dict,
        smooth_factor: float = 0.3
    ) -> Dict:
        """
        平滑过渡：防止参数突变

        Args:
            prev_params: 前一个chunk的参数
            current_params: 当前chunk的参数
            smooth_factor: 平滑因子（0-1），0=不平滑，1=完全平滑

        Returns:
            平滑后的参数
        """
        smoothed = current_params.copy()

        # 对关键参数进行平滑
        for key in ['f0_scale', 'speed_scale', 'energy_scale']:
            if key in prev_params and key in current_params:
                prev_val = prev_params[key]
                curr_val = current_params[key]

                # 如果变化过大，进行平滑
                diff = abs(curr_val - prev_val)
                max_allowed_diff = 0.15  # 允许的最大变化

                if diff > max_allowed_diff:
                    # 线性插值平滑
                    smoothed[key] = (
                        prev_val * smooth_factor +
                        curr_val * (1 - smooth_factor)
                    )

        return smoothed

    def _apply_global_contour(
        self,
        prosody_params: List[Dict],
        overall_emotion: str
    ) -> List[Dict]:
        """
        应用全局音高轮廓（让音高有自然的起伏）

        Args:
            prosody_params: 韵律参数列表
            overall_emotion: 整体情感

        Returns:
            应用轮廓后的参数列表
        """
        num_chunks = len(prosody_params)
        if num_chunks <= 1:
            return prosody_params

        # 获取情感变化幅度
        emotion_base = self.emotion_params.get(
            overall_emotion,
            self.emotion_params['neutral']
        )
        variation = emotion_base['variation']

        # 生成正弦轮廓（模拟自然的音高起伏）
        t = np.linspace(0, np.pi, num_chunks)
        contour = np.sin(t) * variation

        # 应用轮廓
        result = []
        for i, params in enumerate(prosody_params):
            params_copy = params.copy()

            # 叠加轮廓到f0_scale
            params_copy['f0_scale'] *= (1 + contour[i])

            # 确保在合理范围内
            params_copy['f0_scale'] = np.clip(
                params_copy['f0_scale'],
                self.f0_range[0],
                self.f0_range[1]
            )

            result.append(params_copy)

        return result

    def adjust_for_emotion_intensity(
        self,
        params: Dict,
        emotion_intensity: float = 1.0
    ) -> Dict:
        """
        根据情感强度调整参数

        Args:
            params: 原始参数
            emotion_intensity: 情感强度（0-1）

        Returns:
            调整后的参数
        """
        adjusted = params.copy()

        # 情感强度影响参数偏离neutral的程度
        for key in ['f0_scale', 'speed_scale', 'energy_scale']:
            if key in adjusted:
                # 计算与neutral(1.0)的偏差
                deviation = adjusted[key] - 1.0

                # 按强度缩放偏差
                scaled_deviation = deviation * emotion_intensity

                # 应用缩放后的偏差
                adjusted[key] = 1.0 + scaled_deviation

        return adjusted

    def get_prosody_summary(self, prosody_params: List[Dict]) -> Dict:
        """
        获取韵律参数的统计摘要

        Args:
            prosody_params: 韵律参数列表

        Returns:
            统计摘要字典
        """
        if not prosody_params:
            return {}

        f0_scales = [p['f0_scale'] for p in prosody_params]
        speed_scales = [p['speed_scale'] for p in prosody_params]
        energy_scales = [p['energy_scale'] for p in prosody_params]

        return {
            'num_chunks': len(prosody_params),
            'f0_scale': {
                'mean': float(np.mean(f0_scales)),
                'std': float(np.std(f0_scales)),
                'min': float(np.min(f0_scales)),
                'max': float(np.max(f0_scales)),
            },
            'speed_scale': {
                'mean': float(np.mean(speed_scales)),
                'std': float(np.std(speed_scales)),
                'min': float(np.min(speed_scales)),
                'max': float(np.max(speed_scales)),
            },
            'energy_scale': {
                'mean': float(np.mean(energy_scales)),
                'std': float(np.std(energy_scales)),
                'min': float(np.min(energy_scales)),
                'max': float(np.max(energy_scales)),
            },
            'total_pause_duration': sum(p['pause_after'] for p in prosody_params),
        }


# 使用示例
if __name__ == "__main__":
    # 模拟来自SentenceSplitter的chunks
    chunks = [
        ("今天天气真不错，", {
            'position': 'start',
            'pause_after': 0.3,
            'speed_scale': 1.0,
            'chunk_length': 8
        }),
        ("阳光明媚温暖宜人。", {
            'position': 'middle',
            'pause_after': 0.5,
            'speed_scale': 1.0,
            'chunk_length': 9
        }),
        ("我们决定一起去公园散步，", {
            'position': 'middle',
            'pause_after': 0.3,
            'speed_scale': 1.0,
            'chunk_length': 12
        }),
        ("你觉得怎么样？", {
            'position': 'end',
            'pause_after': 0.5,
            'speed_scale': 0.95,
            'chunk_length': 7
        }),
    ]

    # 创建规划器
    planner = GlobalProsodyPlanner()

    # 测试不同情感
    emotions = ['neutral', 'happy', 'sad']

    for emotion in emotions:
        print(f"\n{'=' * 60}")
        print(f"情感: {emotion}")
        print('=' * 60)

        # 规划韵律
        prosody_params = planner.plan(chunks, overall_emotion=emotion)

        # 打印结果
        for i, params in enumerate(prosody_params):
            print(f"\nChunk {i+1}: {params['text']}")
            print(f"  位置: {params['position']}")
            print(f"  音高缩放: {params['f0_scale']:.3f}")
            print(f"  语速缩放: {params['speed_scale']:.3f}")
            print(f"  能量缩放: {params['energy_scale']:.3f}")
            print(f"  停顿时长: {params['pause_after']:.2f}s")

        # 打印统计摘要
        summary = planner.get_prosody_summary(prosody_params)
        print(f"\n统计摘要:")
        print(f"  总chunks数: {summary['num_chunks']}")
        print(f"  音高范围: {summary['f0_scale']['min']:.3f} - {summary['f0_scale']['max']:.3f}")
        print(f"  语速范围: {summary['speed_scale']['min']:.3f} - {summary['speed_scale']['max']:.3f}")
        print(f"  总停顿时长: {summary['total_pause_duration']:.2f}s")
