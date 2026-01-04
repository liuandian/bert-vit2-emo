"""
实时TTS核心模块

包含:
- sentence_splitter: 智能文本切分器
- audio_processor: 音频处理工具
- prosody_planner: 韵律规划器
- streaming_engine: 流式推理引擎
"""

from .sentence_splitter import SmartSentenceSplitter
from .audio_processor import AudioProcessor
from .prosody_planner import GlobalProsodyPlanner
from .streaming_engine import StreamingInferenceEngine

__all__ = [
    "SmartSentenceSplitter",
    "AudioProcessor",
    "GlobalProsodyPlanner",
    "StreamingInferenceEngine",
]
