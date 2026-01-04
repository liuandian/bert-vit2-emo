"""
基于BERT的情感embedding生成工具
使用中文BERT对情感描述文本生成1024维embedding
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path

# 使用与主系统相同的中文BERT模型
DEFAULT_BERT_MODEL = "./bert/chinese-roberta-wwm-ext-large"

# 6种情感的描述文本（用于生成embedding）
EMOTION_TEXTS = {
    "neutral": "这是平静的、中性的、正常的语气。",
    "happy": "我非常开心！太高兴了！这真是太棒了！我感到无比快乐和兴奋！",
    "sad": "我很难过，感到悲伤和失落。心情低落，十分沮丧。",
    "angry": "我非常生气！太气愤了！这让我愤怒！我感到十分恼火！",
    "fear": "我很害怕，感到恐惧和紧张。我担心会发生不好的事情。",
    "surprise": "哇！真是令人惊讶！太意外了！我完全没想到会是这样！"
}


class EmotionBERTGenerator:
    """情感BERT embedding生成器"""

    def __init__(self, model_name=DEFAULT_BERT_MODEL, device="cpu"):
        """
        初始化BERT模型

        Args:
            model_name: BERT模型名称或路径
            device: 设备 (cpu/cuda)
        """
        self.device = device
        self.model_name = model_name

        print(f"[EmoBERT] 加载BERT模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"[EmoBERT] 模型已加载到 {device}")

    def get_embedding(self, text):
        """
        从文本生成1024维BERT embedding

        Args:
            text: 输入文本

        Returns:
            numpy array (1024,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token的hidden state作为句子表示
            # outputs.last_hidden_state: [batch, seq_len, hidden_size]
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # 转换为numpy并去除batch维度
        embedding = cls_embedding.cpu().numpy().squeeze(0)

        # 确保是1024维（如果BERT模型是768维，需要投影）
        if embedding.shape[0] == 768:
            # 简单的线性投影（零填充）
            embedding_1024 = np.zeros(1024, dtype=np.float32)
            embedding_1024[:768] = embedding
            embedding = embedding_1024
            print(f"[EmoBERT] 警告: BERT输出是768维，已扩展到1024维（零填充）")

        return embedding

    def generate_emotion_embeddings(self, emotion_texts=None):
        """
        生成所有情感的embeddings

        Args:
            emotion_texts: 自定义情感文本字典，默认使用EMOTION_TEXTS

        Returns:
            dict: {emotion_name: embedding_array}
        """
        if emotion_texts is None:
            emotion_texts = EMOTION_TEXTS

        embeddings = {}

        print("\n[EmoBERT] 生成情感embeddings...")
        for emotion, text in emotion_texts.items():
            print(f"  处理: {emotion:10s} - {text[:30]}...")
            embedding = self.get_embedding(text)
            embeddings[emotion] = embedding
            print(f"    ✓ 生成embedding: shape={embedding.shape}, mean={embedding.mean():.4f}")

        return embeddings

    def save_embeddings(self, embeddings, save_dir="emotional/presets"):
        """
        保存情感embeddings到文件

        Args:
            embeddings: {emotion_name: embedding_array}
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[EmoBERT] 保存embeddings到: {save_path}")

        for emotion, embedding in embeddings.items():
            file_path = save_path / f"{emotion}.npy"
            np.save(file_path, embedding)
            print(f"  ✓ 保存: {file_path}")

        # 同时保存为.pt格式（与BERT特征格式一致）
        for emotion, embedding in embeddings.items():
            file_path = save_path / f"{emotion}.emo.pt"
            torch.save(torch.from_numpy(embedding), file_path)
            print(f"  ✓ 保存: {file_path}")

        print(f"\n[EmoBERT] 全部保存完成！")

    @staticmethod
    def load_embedding(emotion, save_dir="emotional/presets"):
        """
        加载情感embedding

        Args:
            emotion: 情感名称
            save_dir: 保存目录

        Returns:
            torch.Tensor (1024,)
        """
        # 优先尝试.emo.pt格式
        pt_path = Path(save_dir) / f"{emotion}.emo.pt"
        if pt_path.exists():
            return torch.load(pt_path)

        # 回退到.npy格式
        npy_path = Path(save_dir) / f"{emotion}.npy"
        if npy_path.exists():
            return torch.from_numpy(np.load(npy_path))

        raise FileNotFoundError(
            f"找不到情感'{emotion}'的embedding文件: {pt_path} 或 {npy_path}"
        )


def main():
    """生成并保存默认的6种情感embeddings"""
    import argparse

    parser = argparse.ArgumentParser(description="生成情感BERT embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BERT_MODEL,
        help="BERT模型名称或路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cpu/cuda)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="emotional/presets",
        help="输出目录"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("情感BERT Embedding生成器")
    print("=" * 60)
    print(f"BERT模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"输出目录: {args.output}")
    print("=" * 60)

    # 创建生成器
    generator = EmotionBERTGenerator(model_name=args.model, device=args.device)

    # 生成embeddings
    embeddings = generator.generate_emotion_embeddings()

    # 保存
    generator.save_embeddings(embeddings, save_dir=args.output)

    # 验证
    print("\n" + "=" * 60)
    print("验证生成的embeddings")
    print("=" * 60)

    for emotion in EMOTION_TEXTS.keys():
        try:
            emb = EmotionBERTGenerator.load_embedding(emotion, args.output)
            print(f"✓ {emotion:10s}: shape={emb.shape}, dtype={emb.dtype}")
        except Exception as e:
            print(f"✗ {emotion:10s}: {e}")

    print("\n✓ 完成！")


if __name__ == "__main__":
    main()
