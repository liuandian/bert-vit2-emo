"""
情感控制工具模块（基于BERT embedding）
支持6种离散情感 + 强度控制（通过embedding加权混合）
"""

import torch
import numpy as np
from typing import Union, Optional
from pathlib import Path

# 6种支持的情感类型
EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

# 情感到代码本索引的映射 (0-9共10个代码本，预留4个)
EMOTION_TO_INDEX = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "surprise": 5,
    # 索引 6-9 预留给未来扩展
}

# 反向映射
INDEX_TO_EMOTION = {v: k for k, v in EMOTION_TO_INDEX.items()}

# 情感预设embedding目录
EMOTION_PRESET_DIR = "emotional/presets"


def get_emotion_index(emotion: str) -> int:
    """
    获取情感对应的代码本索引

    Args:
        emotion: 情感名称 (angry/fear/happy/neutral/sad/surprise)

    Returns:
        代码本索引 (0-9)

    Raises:
        ValueError: 如果情感类型不支持
    """
    emotion = emotion.lower().strip()
    if emotion not in EMOTION_TO_INDEX:
        raise ValueError(
            f"Unsupported emotion '{emotion}'. "
            f"Supported: {', '.join(EMOTION_LABELS)}"
        )
    return EMOTION_TO_INDEX[emotion]


def load_emotion_embedding(emotion: str, preset_dir: str = EMOTION_PRESET_DIR) -> torch.Tensor:
    """
    从预设库加载情感embedding

    Args:
        emotion: 情感名称
        preset_dir: 预设embedding目录

    Returns:
        torch.Tensor (1024,)

    Raises:
        FileNotFoundError: 如果找不到对应的embedding文件
    """
    emotion = emotion.lower().strip()
    preset_path = Path(preset_dir)

    # 优先尝试.emo.pt格式
    pt_path = preset_path / f"{emotion}.emo.pt"
    if pt_path.exists():
        return torch.load(pt_path, map_location='cpu')

    # 回退到.npy格式
    npy_path = preset_path / f"{emotion}.npy"
    if npy_path.exists():
        return torch.from_numpy(np.load(npy_path)).float()

    raise FileNotFoundError(
        f"找不到情感'{emotion}'的embedding文件\n"
        f"请先运行: python emo_gen_bert.py\n"
        f"查找路径: {pt_path} 或 {npy_path}"
    )


def mix_emotion_embeddings(
    neutral_emb: torch.Tensor,
    target_emb: torch.Tensor,
    intensity: float
) -> torch.Tensor:
    """
    混合neutral和目标情感的embedding

    公式: final_emb = (1 - intensity) * neutral_emb + intensity * target_emb

    Args:
        neutral_emb: neutral情感embedding (1024,)
        target_emb: 目标情感embedding (1024,)
        intensity: 强度 (0.0-1.0)

    Returns:
        混合后的embedding (1024,)
    """
    intensity = max(0.0, min(1.0, intensity))  # Clamp to [0, 1]
    return (1.0 - intensity) * neutral_emb + intensity * target_emb


def get_emotion_embedding(
    emotion: Union[str, int],
    intensity: float = 1.0,
    preset_dir: str = EMOTION_PRESET_DIR,
    device: str = "cpu",
) -> torch.Tensor:
    """
    获取情感embedding（支持强度控制）

    Args:
        emotion: 情感类别名称或索引
        intensity: 情感强度 (0.0-1.0)
            - 0.0: 完全neutral
            - 1.0: 完全目标情感
        preset_dir: 预设embedding目录
        device: 设备

    Returns:
        情感embedding tensor (1024,)
    """
    # 处理情感索引
    if isinstance(emotion, int):
        if emotion < 0 or emotion >= 10:
            raise ValueError(f"Emotion index must be 0-9, got {emotion}")
        emotion = INDEX_TO_EMOTION.get(emotion, "neutral")
    elif isinstance(emotion, str):
        emotion = emotion.lower().strip()
        if emotion not in EMOTION_LABELS:
            raise ValueError(
                f"Unsupported emotion '{emotion}'. "
                f"Supported: {', '.join(EMOTION_LABELS)}"
            )
    else:
        raise TypeError(f"emotion must be str or int, got {type(emotion)}")

    # 加载neutral embedding
    neutral_emb = load_emotion_embedding("neutral", preset_dir)

    # 如果目标就是neutral或强度为0，直接返回
    if emotion == "neutral" or intensity <= 0.0:
        return neutral_emb.to(device)

    # 加载目标情感embedding
    target_emb = load_emotion_embedding(emotion, preset_dir)

    # 强度混合
    final_emb = mix_emotion_embeddings(neutral_emb, target_emb, intensity)

    return final_emb.to(device)


def prepare_emotion_for_model(
    emotion: Union[str, int, None] = None,
    intensity: float = 1.0,
    preset_dir: str = EMOTION_PRESET_DIR,
    device: str = "cpu",
) -> torch.Tensor:
    """
    为模型准备情感embedding（便捷函数）

    Args:
        emotion: 情感类别/索引，None表示neutral
        intensity: 强度 (0.0-1.0)
        preset_dir: 预设embedding目录
        device: 设备

    Returns:
        情感embedding tensor (1024,)，可直接输入模型
    """
    if emotion is None:
        emotion = "neutral"

    return get_emotion_embedding(emotion, intensity, preset_dir, device)


def emotion_to_string(emotion: Union[str, int, torch.Tensor, np.ndarray]) -> str:
    """
    将情感转换为可读字符串

    Args:
        emotion: 情感

    Returns:
        情感名称
    """
    if isinstance(emotion, str):
        return emotion.lower()

    if isinstance(emotion, (torch.Tensor, np.ndarray)):
        # 如果是embedding，无法直接判断
        return "custom_embedding"

    if isinstance(emotion, int):
        return INDEX_TO_EMOTION.get(emotion, f"unknown_{emotion}")

    return "unknown"


def validate_emotion_params(emotion: str, intensity: float) -> tuple:
    """
    验证并规范化情感参数

    Args:
        emotion: 情感名称
        intensity: 强度

    Returns:
        (规范化的emotion, 规范化的intensity)

    Raises:
        ValueError: 参数无效
    """
    # 验证情感
    emotion = emotion.lower().strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(
            f"Invalid emotion '{emotion}'. "
            f"Valid emotions: {', '.join(EMOTION_LABELS)}"
        )

    # 验证并限制强度
    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        raise ValueError(f"Intensity must be a number, got {intensity}")

    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"Intensity must be between 0.0 and 1.0, got {intensity}")

    return emotion, intensity


def check_emotion_presets_exist(preset_dir: str = EMOTION_PRESET_DIR) -> dict:
    """
    检查情感预设embedding是否存在

    Args:
        preset_dir: 预设目录

    Returns:
        dict: {emotion: exists}
    """
    preset_path = Path(preset_dir)
    status = {}

    for emotion in EMOTION_LABELS:
        pt_exists = (preset_path / f"{emotion}.emo.pt").exists()
        npy_exists = (preset_path / f"{emotion}.npy").exists()
        status[emotion] = pt_exists or npy_exists

    return status


if __name__ == "__main__":
    # 测试
    print("=== Emotion Utils Test (BERT-based) ===")
    print(f"Supported emotions: {EMOTION_LABELS}")
    print(f"Emotion mapping: {EMOTION_TO_INDEX}")

    # 检查预设是否存在
    print("\n=== Checking emotion presets ===")
    status = check_emotion_presets_exist()
    for emotion, exists in status.items():
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {emotion}")

    if not all(status.values()):
        print("\n⚠ 部分预设不存在，请运行: python emo_gen_bert.py")
    else:
        print("\n=== Testing emotion embedding loading ===")
        try:
            # 测试加载
            neutral_emb = load_emotion_embedding("neutral")
            print(f"✓ Loaded neutral: shape={neutral_emb.shape}, dtype={neutral_emb.dtype}")

            happy_emb = load_emotion_embedding("happy")
            print(f"✓ Loaded happy: shape={happy_emb.shape}, dtype={happy_emb.dtype}")

            # 测试混合
            print("\n=== Testing emotion mixing ===")
            mixed_05 = mix_emotion_embeddings(neutral_emb, happy_emb, 0.5)
            print(f"✓ Mixed (0.5): shape={mixed_05.shape}")

            mixed_08 = mix_emotion_embeddings(neutral_emb, happy_emb, 0.8)
            print(f"✓ Mixed (0.8): shape={mixed_08.shape}")

            # 测试高级接口
            print("\n=== Testing high-level API ===")
            emo_emb = get_emotion_embedding("sad", intensity=0.7)
            print(f"✓ get_emotion_embedding('sad', 0.7): shape={emo_emb.shape}")

        except FileNotFoundError as e:
            print(f"✗ Error: {e}")

    # 测试验证
    print("\n=== Testing validation ===")
    try:
        validate_emotion_params("happy", 0.5)
        print("✓ Valid params passed")
    except ValueError as e:
        print(f"✗ {e}")

    try:
        validate_emotion_params("invalid", 0.5)
    except ValueError as e:
        print(f"✓ Invalid emotion caught: {e}")

    print("\n=== Test completed ===")
