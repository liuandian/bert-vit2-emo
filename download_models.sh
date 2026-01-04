#!/bin/bash
# Bert-VITS2 模型一键下载脚本

set -e  # 遇到错误时退出

echo "=========================================="
echo "  Bert-VITS2 模型下载工具"
echo "=========================================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python 3，请先安装 Python 3"
    exit 1
fi

# 检查并安装 huggingface-hub
echo "检查依赖..."
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "安装 huggingface-hub..."
    pip install huggingface-hub
fi

# 运行 Python 下载脚本
echo "开始下载模型..."
python3 download_models.py

echo ""
echo "=========================================="
echo "  完成！"
echo "=========================================="
