#!/usr/bin/env python3
"""
一键下载 Bert-VITS2 所需的 Hugging Face 模型

依赖: pip install huggingface-hub
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(repo_id: str, local_dir: str):
    """
    下载 Hugging Face 模型到指定目录

    Args:
        repo_id: Hugging Face 仓库 ID (例如: hfl/chinese-roberta-wwm-ext-large)
        local_dir: 本地存储目录
    """
    print(f"\n{'='*60}")
    print(f"开始下载: {repo_id}")
    print(f"目标目录: {local_dir}")
    print(f"{'='*60}\n")

    try:
        # 创建目录（如果不存在）
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=True,  # 支持断点续传
        )

        print(f"\n✓ {repo_id} 下载完成！")

    except Exception as e:
        print(f"\n✗ 下载 {repo_id} 时出错: {e}")
        raise

def main():
    """主函数：下载所有必需的模型"""

    # 获取脚本所在目录（项目根目录）
    project_root = Path(__file__).parent.absolute()

    # 定义要下载的模型列表
    models = [
        {
            "repo_id": "hfl/chinese-roberta-wwm-ext-large",
            "local_dir": project_root / "bert" / "chinese-roberta-wwm-ext-large"
        },
        {
            "repo_id": "microsoft/wavlm-base-plus",
            "local_dir": project_root / "slm" / "wavlm-base-plus"
        }
    ]

    print("\n" + "="*60)
    print("Bert-VITS2 模型下载工具")
    print("="*60)
    print(f"\n项目根目录: {project_root}")
    print(f"将下载以下模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['repo_id']}")
    print()

    # 依次下载每个模型
    success_count = 0
    failed_models = []

    for model in models:
        try:
            download_model(
                repo_id=model["repo_id"],
                local_dir=str(model["local_dir"])
            )
            success_count += 1
        except Exception as e:
            failed_models.append(model["repo_id"])
            print(f"跳过失败的模型，继续下载下一个...\n")

    # 输出总结
    print("\n" + "="*60)
    print("下载完成总结")
    print("="*60)
    print(f"成功: {success_count}/{len(models)}")

    if failed_models:
        print(f"\n失败的模型:")
        for model in failed_models:
            print(f"  - {model}")
        print("\n提示: 请检查网络连接或尝试使用镜像源")
    else:
        print("\n所有模型下载成功！✓")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()
