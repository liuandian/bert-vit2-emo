"""
API数据模型定义
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str = Field(..., description="输入文本", min_length=1)
    speaker_id: str = Field(..., description="说话人ID")
    emotion: str = Field(default="neutral", description="情感类型")
    emotion_intensity: float = Field(default=1.0, ge=0.0, le=1.0, description="情感强度")
    language: str = Field(default="ZH", description="语言（ZH/JP/EN）")
    sdp_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="SDP比例")
    noise_scale: float = Field(default=0.6, ge=0.0, le=2.0, description="噪声缩放")
    noise_scale_w: float = Field(default=0.8, ge=0.0, le=2.0, description="时长噪声缩放")
    length_scale: float = Field(default=1.0, ge=0.1, le=2.0, description="长度缩放")
    enable_prosody_planning: bool = Field(default=True, description="是否启用韵律规划")


class TTSResponse(BaseModel):
    """TTS响应模型"""
    success: bool
    message: str
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    stats: Optional[dict] = None


class StreamConfig(BaseModel):
    """流式生成配置"""
    text: str = Field(..., description="输入文本")
    speaker_id: str = Field(..., description="说话人ID")
    emotion: str = Field(default="neutral", description="情感")
    emotion_intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = Field(default="ZH")
    enable_prosody_planning: bool = Field(default=True)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    device: str
    model_loaded: bool
    available_speakers: List[str]
    available_emotions: List[str]
