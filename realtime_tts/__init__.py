"""
实时TTS系统

基于Bert-VITS2的流式文本转语音系统
"""

__version__ = "1.0.0"

from .core import (
    StreamingInferenceEngine,
    SmartSentenceSplitter,
    AudioProcessor,
    GlobalProsodyPlanner,
)

__all__ = [
    "StreamingInferenceEngine",
    "SmartSentenceSplitter",
    "AudioProcessor",
    "GlobalProsodyPlanner",
]
