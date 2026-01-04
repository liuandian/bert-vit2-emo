"""
流式推理引擎

实现实时TTS的核心功能：
1. 流式生成：边生成边输出
2. 上下文窗口：保持韵律连续性
3. 音频拼接：overlap-add平滑拼接
"""

import sys
import os

# 添加父目录到路径，以便导入Bert-VITS2模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from typing import Generator, Optional, Dict, List, Union
import time

# 导入Bert-VITS2模块
from infer import get_text, get_net_g
from emotion_utils import prepare_emotion_for_model
import utils

# 导入本地模块
from .sentence_splitter import SmartSentenceSplitter
from .audio_processor import AudioProcessor
from .prosody_planner import GlobalProsodyPlanner


class StreamingInferenceEngine:
    """流式推理引擎"""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        context_window_size: int = 5,
        overlap_duration: float = 0.05,
        max_chunk_len: int = 40
    ):
        """
        初始化流式推理引擎

        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径
            device: 运行设备（cuda/cpu）
            context_window_size: 上下文窗口大小（字符数）
            overlap_duration: 音频重叠时长（秒）
            max_chunk_len: 最大chunk长度
        """
        self.device = device
        self.context_window_size = context_window_size
        self.overlap_duration = overlap_duration

        print(f"[StreamEngine] 正在加载配置: {config_path}")
        self.hps = utils.get_hparams_from_file(config_path)

        print(f"[StreamEngine] 正在加载模型: {model_path}")
        self.net_g = get_net_g(
            model_path=model_path,
            version=getattr(self.hps, 'version', "2.3"),
            device=device,
            hps=self.hps
        )
        self.net_g.eval()
        print(f"[StreamEngine] 模型加载完成，设备: {device}")

        # 初始化工具
        self.splitter = SmartSentenceSplitter(max_chunk_len=max_chunk_len)
        self.audio_processor = AudioProcessor(
            sampling_rate=self.hps.data.sampling_rate
        )
        self.prosody_planner = GlobalProsodyPlanner()

        # 上下文缓存
        self.context_text = ""
        self.previous_audio_tail = None

        # 性能统计
        self.stats = {
            'ttfb': None,  # Time to First Byte
            'chunk_times': [],  # 每个chunk的生成时间
            'total_time': 0,
            'total_audio_duration': 0,
        }

    def infer_chunk(
            self,
            text: str,
            speaker_id: str,
            emotion: str = "neutral",
            emotion_intensity: float = 1.0,
            prosody_params: Optional[Dict] = None,
            language: str = "ZH",
            sdp_ratio: float = 0.2,
            noise_scale: float = 0.6,
            noise_scale_w: float = 0.8,
            length_scale: float = 1.0,
            style_text: Optional[str] = None,
            style_weight: float = 0.7,
        ) -> np.ndarray:
            
            # 1. 
            # 如果没有上下文，就直接用 text
            # full_text = self.context_text + text if self.context_text else text

            bert, ja_bert, en_bert, emo_bert, phones, tones, lang_ids = get_text(
                text,
                language,
                emotion,
                self.hps,
                self.device,
                style_text=style_text,
                style_weight=style_weight
            )
            
            # 记录完整音素的长度，用于后续校验
            full_phones_len = phones.size(0)

            # 准备情感embedding (保持不变)
            if emotion is not None:
                try:
                    emo = prepare_emotion_for_model(
                        emotion=emotion,
                        intensity=emotion_intensity,
                        device=self.device
                    )
                except FileNotFoundError:
                    emo = prepare_emotion_for_model(emotion="neutral", intensity=1.0, device=self.device)
            else:
                emo = None

            # 应用韵律参数 (保持不变)
            if prosody_params:
                speed_scale = prosody_params.get('speed_scale', 1.0)
                length_scale = length_scale / speed_scale

            # 模型推理
            with torch.no_grad():
                x_tst = phones.to(self.device).unsqueeze(0)
                tones_t = tones.to(self.device).unsqueeze(0)
                lang_ids_t = lang_ids.to(self.device).unsqueeze(0)
                bert_t = bert.to(self.device).unsqueeze(0)
                ja_bert_t = ja_bert.to(self.device).unsqueeze(0)
                en_bert_t = en_bert.to(self.device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
                speakers = torch.LongTensor([self.hps.data.spk2id[speaker_id]]).to(self.device)
                if emo is not None:
                    emo = emo.to(self.device).unsqueeze(0)

                
                outputs = self.net_g.infer(
                    x_tst, x_tst_lengths, speakers, tones_t, lang_ids_t,
                    bert_t, ja_bert_t, en_bert_t,
                    emo_bert=emo,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )
                
                # 提取音频 [1, 1, samples] -> [samples]
                audio = outputs[0][0, 0].data.cpu().float().numpy()
                
            #     # 4. 【关键】提取音素时长 durations
            #     # outputs[2] 是 durations: [1, phones_len] -> [phones_len]
            #     # 必须 flatten，否则无法计算 sum
            #     durations = outputs[2][0].data.cpu().numpy().flatten()
                
            #     # 清理GPU
            #     del x_tst, tones_t, lang_ids_t, bert_t, ja_bert_t, en_bert_t, x_tst_lengths, speakers, emo
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()

            # # 5. 【核心逻辑】精准裁剪
            # if self.context_text:
            #     # A. 计算上下文产生了多少个音素
            #     # 注意：必须用同样的参数调用 get_text，确保生成的音素序列一致
            #     res_ctx = get_text(self.context_text, language, emotion, self.hps, self.device)
            #     context_phones = res_ctx[4] # phones 是第5个返回结果
            #     num_context_phones = context_phones.size(0)

            #     # B. 校验长度（防止索引越界）
            #     if num_context_phones < len(durations):
            #         # C. 计算上下文占用的帧数
            #         # sum(上下文音素的时长) = 上下文的总帧数
            #         context_frames = np.sum(durations[:num_context_phones])
                    
            #         # D. 转换为采样点 (hop_length 通常是 512)
            #         hop_length = self.hps.data.filter_length // 4
            #         context_samples = int(context_frames * hop_length)
                    
            #         # E. 【终极杀招】增加安全偏移量 (Safety Margin)
            #         # 为了防止上一句标点符号的“尾气”留下来，我们多切掉 50ms (约2200点)
            #         # 只有当 context 确实存在时才切
            #         safety_margin = int(0.05 * self.hps.data.sampling_rate) # 50ms
            #         cut_point = context_samples + safety_margin
                    
            #         # F. 边界保护：不能把整段音频都切没了
            #         # 至少保留 100 个点，或者保留一半（如果生成很短）
            #         max_cut = len(audio) - 100
            #         cut_point = min(cut_point, max_cut)
                    
            #         # 执行切割
            #         audio = audio[cut_point:]
            #     else:
            #         print(f"[Warning] 上下文音素长度({num_context_phones}) >= 总时长({len(durations)})，跳过裁剪")

            # # 6. 【更新上下文】保持滑动窗口
            # # 简单策略：保留最后一句，或者保留固定长度
            # # 这里建议只保留 full_text 的后半部分，防止 context 无限增长
            # max_context_window = 50 # 保留最近50个字
            # self.context_text = (self.context_text + text)[-max_context_window:]

            return audio

    def stream_generate(
        self,
        text: str,
        speaker_id: str,
        emotion: str = "neutral",
        emotion_intensity: float = 1.0,
        language: str = "ZH",
        enable_prosody_planning: bool = True,
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """
        流式生成音频

        Args:
            text: 完整输入文本
            speaker_id: 说话人ID
            emotion: 情感
            emotion_intensity: 情感强度
            language: 语言
            enable_prosody_planning: 是否启用韵律规划
            **kwargs: 其他推理参数

        Yields:
            audio_chunk: numpy array，每个chunk的音频
        """
        # 重置状态
        self.context_text = ""
        self.previous_audio_tail = None
        self.stats = {
            'ttfb': None,
            'chunk_times': [],
            'total_time': 0,
            'total_audio_duration': 0,
        }

        start_time = time.time()
        # 切分文本
        chunks = self.splitter.split(text)
        print(f"[StreamEngine] 文本切分为 {len(chunks)} 个chunks")

        # 规划韵律（如果启用）
        # if enable_prosody_planning:
        #     prosody_params_list = self.prosody_planner.plan(
        #         chunks,
        #         overall_emotion=emotion
        #     )
        # else:
        #     prosody_params_list = [None] * len(chunks)
        prosody_params_list = [None] * len(chunks)
        
        # 逐个生成chunks
        for i, ((chunk_text, chunk_meta), prosody_params) in enumerate(zip(chunks, prosody_params_list)):
            chunk_start_time = time.time()

            print(f"[StreamEngine] 生成 chunk {i+1}/{len(chunks)}: {chunk_text}")

            # 推理当前chunk
            chunk_audio = self.infer_chunk(
                text=chunk_text,
                speaker_id=speaker_id,
                emotion=emotion,
                emotion_intensity=emotion_intensity,
                prosody_params=prosody_params,
                language=language,
                **kwargs
            )
            print("生成chunk成功")
            # 如果不是第一个chunk，进行overlap-add拼接
            if self.previous_audio_tail is not None:
                chunk_audio = self.audio_processor.overlap_add(
                    self.previous_audio_tail,
                    chunk_audio,
                    overlap_duration=self.overlap_duration
                )

            # 记录第一个chunk的时间（TTFB）
            if i == 0:
                self.stats['ttfb'] = time.time() - start_time

            # 记录chunk生成时间
            chunk_time = time.time() - chunk_start_time
            self.stats['chunk_times'].append(chunk_time)

            # 保存尾部用于下次拼接
            overlap_samples = int(self.hps.data.sampling_rate * self.overlap_duration)

            if len(chunk_audio) > overlap_samples:
                self.previous_audio_tail = chunk_audio[-overlap_samples:]

                # 输出音频（去掉会被下次覆盖的重叠部分）
                if i < len(chunks) - 1:
                    output_audio = chunk_audio[:-overlap_samples]
                else:
                    # 最后一个chunk全部输出
                    output_audio = chunk_audio
            else:
                # chunk太短，直接输出
                output_audio = chunk_audio
                self.previous_audio_tail = None

            # 更新统计
            audio_duration = len(output_audio) / self.hps.data.sampling_rate
            self.stats['total_audio_duration'] += audio_duration

            # 输出chunk
            yield output_audio

        # 记录总时间
        self.stats['total_time'] = time.time() - start_time

    def generate_complete(
        self,
        text: str,
        speaker_id: str,
        emotion: str = "neutral",
        emotion_intensity: float = 1.0,
        language: str = "ZH",
        enable_prosody_planning: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        生成完整音频（内部使用流式，但返回完整结果）

        Args:
            text: 输入文本
            speaker_id: 说话人ID
            emotion: 情感
            emotion_intensity: 情感强度
            language: 语言
            enable_prosody_planning: 是否启用韵律规划
            **kwargs: 其他参数

        Returns:
            完整音频 numpy array
        """
        audio_chunks = []

        for chunk_audio in self.stream_generate(
            text=text,
            speaker_id=speaker_id,
            emotion=emotion,
            emotion_intensity=emotion_intensity,
            language=language,
            enable_prosody_planning=enable_prosody_planning,
            **kwargs
        ):
            audio_chunks.append(chunk_audio)

        # 拼接所有chunks
        if audio_chunks:
            return np.concatenate(audio_chunks)
        else:
            return np.array([], dtype=np.float32)

    def get_stats(self) -> Dict:
        """
        获取性能统计信息

        Returns:
            统计信息字典
        """
        stats = self.stats.copy()

        if stats['total_audio_duration'] > 0 and stats['total_time'] > 0:
            # 计算实时率（Real-Time Factor）
            stats['rtf'] = stats['total_time'] / stats['total_audio_duration']
        else:
            stats['rtf'] = None

        # 计算平均chunk生成时间
        if stats['chunk_times']:
            stats['avg_chunk_time'] = np.mean(stats['chunk_times'])
            stats['max_chunk_time'] = np.max(stats['chunk_times'])
            stats['min_chunk_time'] = np.min(stats['chunk_times'])
        else:
            stats['avg_chunk_time'] = None
            stats['max_chunk_time'] = None
            stats['min_chunk_time'] = None

        return stats

    def print_stats(self):
        """打印性能统计信息"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("性能统计")
        print("=" * 60)

        if stats['ttfb'] is not None:
            print(f"首包时间 (TTFB): {stats['ttfb']*1000:.1f} ms")

        print(f"总生成时间: {stats['total_time']:.2f} s")
        print(f"总音频时长: {stats['total_audio_duration']:.2f} s")

        if stats['rtf'] is not None:
            print(f"实时率 (RTF): {stats['rtf']:.3f}")

        if stats['avg_chunk_time'] is not None:
            print(f"\nChunk统计:")
            print(f"  平均生成时间: {stats['avg_chunk_time']*1000:.1f} ms")
            print(f"  最快: {stats['min_chunk_time']*1000:.1f} ms")
            print(f"  最慢: {stats['max_chunk_time']*1000:.1f} ms")
            print(f"  总chunks数: {len(stats['chunk_times'])}")

        print("=" * 60 + "\n")


# 使用示例
if __name__ == "__main__":
    import argparse
    from scipy.io import wavfile

    parser = argparse.ArgumentParser(description="流式TTS推理引擎测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--text", type=str, default="今天天气真好，我们一起去公园散步吧。", help="输入文本")
    parser.add_argument("--speaker", type=str, required=True, help="说话人ID")
    parser.add_argument("--emotion", type=str, default="neutral", help="情感")
    parser.add_argument("--output", type=str, default="output_streaming.wav", help="输出文件")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    # 创建引擎
    engine = StreamingInferenceEngine(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )

    print(f"\n输入文本: {args.text}")
    print(f"说话人: {args.speaker}")
    print(f"情感: {args.emotion}\n")

    # 流式生成
    audio_chunks = []
    for i, chunk_audio in enumerate(engine.stream_generate(
        text=args.text,
        speaker_id=args.speaker,
        emotion=args.emotion,
        emotion_intensity=0.8
    )):
        print(f"[主程序] 收到 chunk {i+1}: {len(chunk_audio)} samples "
              f"({len(chunk_audio)/engine.hps.data.sampling_rate:.2f}s)")
        audio_chunks.append(chunk_audio)

    # 拼接并保存
    final_audio = np.concatenate(audio_chunks)

    # 保存音频
    wavfile.write(
        args.output,
        engine.hps.data.sampling_rate,
        (final_audio * 32767).astype(np.int16)
    )

    print(f"\n[完成] 音频已保存: {args.output}")
    print(f"总时长: {len(final_audio)/engine.hps.data.sampling_rate:.2f}s")

    # 打印统计
    engine.print_stats()
