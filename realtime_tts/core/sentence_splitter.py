"""
智能句子切分器

实现基于标点的三级切分策略：
1. 主要标点（句号、问号、感叹号）
2. 次要标点（逗号、分号）
3. 第三级标点（顿号）

并为每个chunk生成韵律元数据
"""

import re
from typing import List, Tuple, Dict


class SmartSentenceSplitter:
    """智能句子切分器"""

    def __init__(
        self,
        max_chunk_len: int = 40,
        min_chunk_len: int = 5,
        enable_merge: bool = True
    ):
        """
        初始化切分器

        Args:
            max_chunk_len: 最大chunk长度（字符数）
            min_chunk_len: 最小chunk长度
            enable_merge: 是否合并短句
        """
        self.max_chunk_len = max_chunk_len
        self.min_chunk_len = min_chunk_len
        self.enable_merge = enable_merge

        # 优先级分层的切分符号
        self.split_patterns = {
            'primary': r'[。！？…!?]',      # 句子结束符 - 最高优先级
            'secondary': r'[，；,;]',       # 从句分隔符 - 中优先级
            'tertiary': r'[、]',           # 短语分隔符 - 低优先级
        }

        # 停顿时长映射（秒）
        self.pause_map = {
            '。': 0.0, '！': 0.0, '？': 0.0, '…': 0.6,
            '.': 0.0, '!': 0.0, '?': 0.0,
            '，': 0.0, '；': 0.4, ',': 0.3, ';': 0.4,
            '、': 0.2,
        }

    def split(self, text: str) -> List[Tuple[str, Dict]]:
        """
        切分文本，返回chunk列表及元数据

        Args:
            text: 输入文本

        Returns:
            List of (chunk_text, metadata)
            metadata包含:
                - position: 位置（start/middle/end）
                - pause_after: 停顿时长（秒）
                - ending_punct: 结束标点
                - is_sub_chunk: 是否为子chunk（长句切分的结果）
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        chunks = []

        # 第一步：按主要标点切分
        sentences = self._split_by_pattern(text, 'primary')

        for i, sent in enumerate(sentences):
            # 如果单句过长，继续细分
            if len(sent) > self.max_chunk_len:
                sub_chunks = self._split_long_sentence(sent)
                for j, sub in enumerate(sub_chunks):
                    metadata = self._create_metadata(
                        chunk_text=sub,
                        position=self._get_position(i, len(sentences)),
                        is_sub_chunk=True,
                        sub_position=self._get_position(j, len(sub_chunks))
                    )
                    chunks.append((sub, metadata))
            else:
                # 短句：如果启用合并，可以和前一句合并
                if self.enable_merge and chunks and len(sent) < self.min_chunk_len:
                    # 合并到前一个chunk
                    prev_text, prev_meta = chunks[-1]
                    if len(prev_text) + len(sent) <= self.max_chunk_len:
                        merged_text = prev_text + sent
                        merged_meta = self._update_metadata_for_merge(
                            prev_meta, sent
                        )
                        chunks[-1] = (merged_text, merged_meta)
                        continue

                metadata = self._create_metadata(
                    chunk_text=sent,
                    position=self._get_position(i, len(sentences)),
                    is_sub_chunk=False
                )
                chunks.append((sent, metadata))

        # 更新所有chunks的position（因为合并可能改变了总数）
        if chunks:
            chunks = self._update_positions(chunks)

        return chunks

    def _split_by_pattern(self, text: str, pattern_type: str) -> List[str]:
        """
        按指定模式切分文本

        Args:
            text: 输入文本
            pattern_type: 模式类型（primary/secondary/tertiary）

        Returns:
            切分后的文本列表
        """
        pattern = self.split_patterns[pattern_type]

        # 使用正则表达式切分，保留分隔符
        parts = re.split(f'({pattern})', text)

        # 重组：把标点符号附加回前面的文本
        sentences = []
        i = 0
        while i < len(parts):
            if not parts[i].strip():
                i += 1
                continue

            # 如果当前是文本，检查下一个是否是标点
            if i + 1 < len(parts) and re.match(pattern, parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                sentences.append(parts[i])
                i += 1

        return [s for s in sentences if s.strip()]

    def _split_long_sentence(self, sent: str) -> List[str]:
        """
        切分过长的句子

        Args:
            sent: 长句子

        Returns:
            切分后的子句列表
        """
        # 优先在次要标点处切分
        parts = self._split_by_pattern(sent, 'secondary')

        # 检查是否都在max_chunk_len内
        if all(len(p) <= self.max_chunk_len for p in parts):
            return parts

        # 仍有过长的，在第三级标点切分
        result = []
        for part in parts:
            if len(part) > self.max_chunk_len:
                sub_parts = self._split_by_pattern(part, 'tertiary')

                # 如果还是过长，硬切分
                final_parts = []
                for sub in sub_parts:
                    if len(sub) > self.max_chunk_len:
                        # 强制切分（每max_chunk_len个字符）
                        final_parts.extend(self._force_split(sub))
                    else:
                        final_parts.append(sub)

                result.extend(final_parts)
            else:
                result.append(part)

        return result

    def _force_split(self, text: str) -> List[str]:
        """
        强制切分过长文本

        Args:
            text: 输入文本

        Returns:
            切分后的列表
        """
        chunks = []
        for i in range(0, len(text), self.max_chunk_len):
            chunks.append(text[i:i + self.max_chunk_len])
        return chunks

    def _get_position(self, idx: int, total: int) -> str:
        """
        判断位置：start/middle/end

        Args:
            idx: 当前索引
            total: 总数

        Returns:
            位置标识
        """
        if total == 1:
            return 'single'
        if idx == 0:
            return 'start'
        elif idx == total - 1:
            return 'end'
        else:
            return 'middle'

    def _create_metadata(
        self,
        chunk_text: str,
        position: str,
        is_sub_chunk: bool,
        sub_position: str = None
    ) -> Dict:
        """
        创建元数据用于韵律控制

        Args:
            chunk_text: chunk文本
            position: 位置（start/middle/end/single）
            is_sub_chunk: 是否为子chunk
            sub_position: 子chunk位置

        Returns:
            元数据字典
        """
        # 提取结束标点
        ending_punct = None
        for char in reversed(chunk_text):
            if char in self.pause_map:
                ending_punct = char
                break

        # 停顿时长
        pause_after = self.pause_map.get(ending_punct, 0.3)

        # 根据位置调整停顿
        if position == 'end' or position == 'single':
            pause_after = max(pause_after, 0.5)  # 结尾至少0.5秒

        # 音高调整（开头稍高，结尾降低）
        f0_scale_map = {
            'start': 1.05,
            'middle': 1.0,
            'end': 0.95,
            'single': 1.0,
        }
        f0_scale = f0_scale_map.get(position, 1.0)

        # 语速调整（根据chunk长度）
        chunk_len = len(chunk_text)
        if chunk_len < 10:
            speed_scale = 0.95  # 短句稍慢，更清晰
        elif chunk_len > 30:
            speed_scale = 1.05  # 长句稍快，避免拖沓
        else:
            speed_scale = 1.0

        return {
            'position': position,
            'pause_after': pause_after,
            'ending_punct': ending_punct,
            'is_sub_chunk': is_sub_chunk,
            'sub_position': sub_position,
            'f0_scale': f0_scale,
            'speed_scale': speed_scale,
            'energy_scale': 1.0,
            'chunk_length': chunk_len,
        }

    def _update_metadata_for_merge(
        self,
        prev_meta: Dict,
        added_text: str
    ) -> Dict:
        """
        更新合并后的元数据

        Args:
            prev_meta: 前一个chunk的元数据
            added_text: 新增的文本

        Returns:
            更新后的元数据
        """
        # 更新chunk长度
        new_meta = prev_meta.copy()
        new_meta['chunk_length'] += len(added_text)

        # 更新结束标点和停顿
        ending_punct = None
        for char in reversed(added_text):
            if char in self.pause_map:
                ending_punct = char
                break

        if ending_punct:
            new_meta['ending_punct'] = ending_punct
            new_meta['pause_after'] = self.pause_map[ending_punct]

        return new_meta

    def _update_positions(
        self,
        chunks: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """
        更新所有chunks的position字段

        Args:
            chunks: chunk列表

        Returns:
            更新后的chunk列表
        """
        total = len(chunks)
        if total == 0:
            return chunks

        updated = []
        for i, (text, meta) in enumerate(chunks):
            meta = meta.copy()
            meta['position'] = self._get_position(i, total)

            # 重新计算f0_scale
            f0_scale_map = {
                'start': 1.05,
                'middle': 1.0,
                'end': 0.95,
                'single': 1.0,
            }
            meta['f0_scale'] = f0_scale_map.get(meta['position'], 1.0)

            updated.append((text, meta))

        return updated

    def split_simple(self, text: str) -> List[str]:
        """
        简单切分（只返回文本，不返回元数据）

        Args:
            text: 输入文本

        Returns:
            切分后的文本列表
        """
        chunks_with_meta = self.split(text)
        return [chunk_text for chunk_text, _ in chunks_with_meta]


# 使用示例
if __name__ == "__main__":
    splitter = SmartSentenceSplitter(max_chunk_len=30)

    test_text = """
    今天天气真不错，阳光明媚温暖宜人。
    我们决定一起去公园散步，欣赏美丽的风景。
    你觉得怎么样？要不要一起来？
    """.strip()

    print("原始文本:")
    print(test_text)
    print("\n" + "=" * 60 + "\n")

    chunks = splitter.split(test_text)

    print(f"切分结果（共 {len(chunks)} 个chunks）:\n")
    for i, (chunk, meta) in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
        print(f"  位置: {meta['position']}")
        print(f"  停顿: {meta['pause_after']:.2f}s")
        print(f"  音高缩放: {meta['f0_scale']:.2f}")
        print(f"  语速缩放: {meta['speed_scale']:.2f}")
        print(f"  长度: {meta['chunk_length']} 字符")
        if meta['is_sub_chunk']:
            print(f"  子chunk位置: {meta['sub_position']}")
        print()
