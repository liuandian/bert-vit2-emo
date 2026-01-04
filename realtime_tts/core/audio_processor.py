"""
音频处理工具

提供音频拼接、平滑、F0调整等功能
"""

import numpy as np
from typing import Optional, Tuple
import warnings


class AudioProcessor:
    """音频处理器"""

    def __init__(self, sampling_rate: int = 44100):
        """
        初始化音频处理器

        Args:
            sampling_rate: 采样率（Hz）
        """
        self.sampling_rate = sampling_rate

    def overlap_add(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        overlap_duration: float = 0.05,
        window_type: str = 'cosine'
    ) -> np.ndarray:
        """
        重叠相加，平滑拼接两段音频

        Args:
            audio1: 前一段音频
            audio2: 后一段音频
            overlap_duration: 重叠时长（秒）
            window_type: 窗口类型（'cosine', 'linear', 'hann'）

        Returns:
            拼接后的audio2（开头已平滑）
        """
        overlap_samples = int(self.sampling_rate * overlap_duration)

        # 如果音频太短，无法overlap，直接返回
        if len(audio1) < overlap_samples or len(audio2) < overlap_samples:
            warnings.warn(
                f"Audio too short for overlap (need {overlap_samples} samples). "
                f"Got audio1={len(audio1)}, audio2={len(audio2)}. Returning audio2 as-is."
            )
            return audio2

        # 提取重叠区域
        tail = audio1[-overlap_samples:]
        head = audio2[:overlap_samples]

        # 生成窗口函数
        fade_out, fade_in = self._get_fade_windows(overlap_samples, window_type)

        # 混合重叠区域
        mixed_overlap = tail * fade_out + head * fade_in

        # 替换audio2的开头
        result = audio2.copy()
        result[:overlap_samples] = mixed_overlap

        return result

    def concatenate_with_overlap(
        self,
        audio_chunks: list,
        overlap_duration: float = 0.05,
        window_type: str = 'cosine'
    ) -> np.ndarray:
        """
        将多段音频使用overlap-add拼接

        Args:
            audio_chunks: 音频chunk列表
            overlap_duration: 重叠时长（秒）
            window_type: 窗口类型

        Returns:
            拼接后的完整音频
        """
        if not audio_chunks:
            return np.array([], dtype=np.float32)

        if len(audio_chunks) == 1:
            return audio_chunks[0]

        # 逐个拼接
        result = audio_chunks[0]
        overlap_samples = int(self.sampling_rate * overlap_duration)

        for i in range(1, len(audio_chunks)):
            current_chunk = audio_chunks[i]

            # 应用overlap-add
            if len(result) >= overlap_samples and len(current_chunk) >= overlap_samples:
                # 提取重叠区域
                tail = result[-overlap_samples:]
                head = current_chunk[:overlap_samples]

                # 窗口混合
                fade_out, fade_in = self._get_fade_windows(overlap_samples, window_type)
                mixed = tail * fade_out + head * fade_in

                # 拼接
                result = np.concatenate([
                    result[:-overlap_samples],
                    mixed,
                    current_chunk[overlap_samples:]
                ])
            else:
                # 直接拼接
                result = np.concatenate([result, current_chunk])

        return result

    def _get_fade_windows(
        self,
        length: int,
        window_type: str = 'cosine'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成淡入淡出窗口

        Args:
            length: 窗口长度
            window_type: 窗口类型

        Returns:
            (fade_out, fade_in) 窗口数组
        """
        if window_type == 'linear':
            fade_out = np.linspace(1, 0, length)
            fade_in = np.linspace(0, 1, length)

        elif window_type == 'cosine':
            # 余弦窗，更平滑
            fade_out = np.cos(np.linspace(0, np.pi / 2, length)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, length)) ** 2

        elif window_type == 'hann':
            # Hann窗
            hann_window = np.hanning(length * 2)
            fade_out = hann_window[:length]
            fade_in = hann_window[length:]

        else:
            raise ValueError(f"Unknown window type: {window_type}")

        return fade_out, fade_in

    def add_silence(
        self,
        audio: np.ndarray,
        duration: float,
        position: str = 'end'
    ) -> np.ndarray:
        """
        在音频前后添加静音

        Args:
            audio: 输入音频
            duration: 静音时长（秒）
            position: 位置（'start', 'end', 'both'）

        Returns:
            添加静音后的音频
        """
        silence_samples = int(self.sampling_rate * duration)
        silence = np.zeros(silence_samples, dtype=audio.dtype)

        if position == 'start':
            return np.concatenate([silence, audio])
        elif position == 'end':
            return np.concatenate([audio, silence])
        elif position == 'both':
            return np.concatenate([silence, audio, silence])
        else:
            raise ValueError(f"Unknown position: {position}")

    def normalize(
        self,
        audio: np.ndarray,
        target_level: float = 0.95,
        method: str = 'peak'
    ) -> np.ndarray:
        """
        音频归一化

        Args:
            audio: 输入音频
            target_level: 目标电平（0-1）
            method: 归一化方法（'peak', 'rms'）

        Returns:
            归一化后的音频
        """
        if method == 'peak':
            # 峰值归一化
            peak = np.abs(audio).max()
            if peak > 0:
                return audio * (target_level / peak)
            return audio

        elif method == 'rms':
            # RMS归一化
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                return audio * (target_level / rms)
            return audio

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def apply_fade(
        self,
        audio: np.ndarray,
        fade_in_duration: float = 0.01,
        fade_out_duration: float = 0.01,
        window_type: str = 'cosine'
    ) -> np.ndarray:
        """
        对音频应用淡入淡出

        Args:
            audio: 输入音频
            fade_in_duration: 淡入时长（秒）
            fade_out_duration: 淡出时长（秒）
            window_type: 窗口类型

        Returns:
            应用淡入淡出后的音频
        """
        result = audio.copy()

        # 淡入
        if fade_in_duration > 0:
            fade_in_samples = int(self.sampling_rate * fade_in_duration)
            fade_in_samples = min(fade_in_samples, len(audio))

            if window_type == 'linear':
                fade_in_window = np.linspace(0, 1, fade_in_samples)
            elif window_type == 'cosine':
                fade_in_window = np.sin(np.linspace(0, np.pi / 2, fade_in_samples)) ** 2
            else:
                fade_in_window = np.hanning(fade_in_samples * 2)[:fade_in_samples]

            result[:fade_in_samples] *= fade_in_window

        # 淡出
        if fade_out_duration > 0:
            fade_out_samples = int(self.sampling_rate * fade_out_duration)
            fade_out_samples = min(fade_out_samples, len(audio))

            if window_type == 'linear':
                fade_out_window = np.linspace(1, 0, fade_out_samples)
            elif window_type == 'cosine':
                fade_out_window = np.cos(np.linspace(0, np.pi / 2, fade_out_samples)) ** 2
            else:
                fade_out_window = np.hanning(fade_out_samples * 2)[fade_out_samples:]

            result[-fade_out_samples:] *= fade_out_window

        return result

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        重采样音频

        Args:
            audio: 输入音频
            orig_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            重采样后的音频
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # 简单的线性插值重采样
            duration = len(audio) / orig_sr
            new_length = int(duration * target_sr)
            old_indices = np.linspace(0, len(audio) - 1, len(audio))
            new_indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(new_indices, old_indices, audio)

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        裁剪音频开头和结尾的静音

        Args:
            audio: 输入音频
            threshold: 能量阈值（相对于最大值）
            frame_length: 帧长
            hop_length: 帧移

        Returns:
            裁剪后的音频
        """
        # 计算每帧的能量
        energy = np.array([
            np.sum(audio[i:i+frame_length] ** 2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])

        # 归一化
        if energy.max() > 0:
            energy = energy / energy.max()

        # 找到非静音帧
        non_silent = np.where(energy > threshold)[0]

        if len(non_silent) == 0:
            return audio  # 全是静音，返回原音频

        # 计算起止位置
        start_frame = non_silent[0]
        end_frame = non_silent[-1]

        start_sample = start_frame * hop_length
        end_sample = min((end_frame + 1) * hop_length + frame_length, len(audio))

        return audio[start_sample:end_sample]


# 使用示例
if __name__ == "__main__":
    # 创建处理器
    processor = AudioProcessor(sampling_rate=44100)

    # 创建测试音频
    duration1 = 1.0
    duration2 = 1.0
    sr = 44100

    # 生成正弦波音频
    t1 = np.linspace(0, duration1, int(sr * duration1))
    audio1 = np.sin(2 * np.pi * 440 * t1) * 0.5  # 440Hz, A4音

    t2 = np.linspace(0, duration2, int(sr * duration2))
    audio2 = np.sin(2 * np.pi * 554.37 * t2) * 0.5  # 554.37Hz, C#5音

    print("测试音频处理功能\n")

    # 测试overlap-add
    print("1. 测试overlap-add拼接...")
    overlapped = processor.overlap_add(audio1, audio2, overlap_duration=0.1)
    print(f"   Audio1长度: {len(audio1)} samples ({len(audio1)/sr:.2f}s)")
    print(f"   Audio2长度: {len(audio2)} samples ({len(audio2)/sr:.2f}s)")
    print(f"   拼接后长度: {len(overlapped)} samples ({len(overlapped)/sr:.2f}s)")

    # 测试批量拼接
    print("\n2. 测试批量拼接...")
    chunks = [audio1, audio2, audio1]
    concatenated = processor.concatenate_with_overlap(chunks, overlap_duration=0.05)
    print(f"   拼接 {len(chunks)} 个chunks")
    print(f"   总长度: {len(concatenated)} samples ({len(concatenated)/sr:.2f}s)")

    # 测试添加静音
    print("\n3. 测试添加静音...")
    with_silence = processor.add_silence(audio1, duration=0.5, position='both')
    print(f"   原始长度: {len(audio1)} samples")
    print(f"   添加静音后: {len(with_silence)} samples")

    # 测试归一化
    print("\n4. 测试归一化...")
    audio_loud = audio1 * 0.3  # 降低音量
    normalized = processor.normalize(audio_loud, target_level=0.9)
    print(f"   原始峰值: {np.abs(audio_loud).max():.3f}")
    print(f"   归一化后峰值: {np.abs(normalized).max():.3f}")

    # 测试淡入淡出
    print("\n5. 测试淡入淡出...")
    faded = processor.apply_fade(audio1, fade_in_duration=0.1, fade_out_duration=0.1)
    print(f"   应用淡入淡出完成")

    print("\n所有测试完成！")
