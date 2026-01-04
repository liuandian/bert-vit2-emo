## 快速开始

### 环境配置

#### 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n bert-vits2 python=3.11 -y
conda activate bert-vits2
```

#### 安装 PyTorch

根据您的 CUDA 版本安装 PyTorch：

```bash
pip install torch torchvision torchaudio
```

#### 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 安装额外依赖

```bash
# 安装 huggingface-hub 用于下载模型
pip install huggingface-hub 
```

---

### 下载预训练模型


#### 使用命令行脚本

```bash
# 一键下载
./download_models.sh

# 或者使用 Python 脚本
python download_models.py
```

#### 方法：手动下载

从 Hugging Face 下载以下模型：

- [中文 RoBERTa](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) → 放到 `bert/chinese-roberta-wwm-ext-large/`
- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus) → 放到 `slm/wavlm-base-plus/`
- [日文 DeBERTa](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) → 放到 `bert/deberta-v2-large-japanese-char-wwm/`（可选）
- [英文 DeBERTa](https://huggingface.co/microsoft/deberta-v3-large) → 放到 `bert/deberta-v3-large/`（可选）

### 生成情绪bert后的embedding

```bash
# 使用 Python 脚本
python emo_gen_bert.py
```

## 一键复现

### 数据以及训练好的模型准备

将casia文件夹复制到根目录的data文件夹中
```
data/
└── casia/
    ├── esd.list          # 标注文件
    ├── models            # 底模以及训练后的模型
    |   ├── D_0.pth
    |   ├── G_0.pth
    |   ├── DUR_0.pth
    |   ├── WD_0.pth
    |   └── ...
    └── raw/              # 原始音频文件
        ├── audio1.wav
        ├── audio2.wav
        └── ...
```

#### 启动流式推理测试
在终端直接启动web

```bash
python -m realtime_tts.webui.gradio_app \
    --model data/casia/models/G_53000.pth \
    --config data/casia/configs/config.json \
    --device cuda \
    --port 7864
```


## 训练

### 数据准备与预处理

#### 启动 Web 预处理界面

```bash
python webui_preprocess.py
```

浏览器会自动打开 `http://127.0.0.1:7860`

#### 准备数据集

将您的数据按照以下结构组织：

```
data/
└── {你的数据集名称}/
    ├── esd.list          # 标注文件
    └── raw/              # 原始音频文件
        ├── audio1.wav
        ├── audio2.wav
        └── ...
```

**标注文件格式 (esd.list):**

```
audio1.wav|说话人名|语言ID|文本内容|情绪
audio2.wav|说话人名|语言ID|文本内容|情绪
```

**示例：**

```
vo_ABDLQ001_1_paimon_02.wav|派蒙|ZH|没什么没什么，只是平时他总是站在这里，有点奇怪而已。|happy
noa_501_0001.wav|NOA|JP|そうだね、油断しないのはとても大事なことだと思う|fear
Albedo_vo_ABDLQ002_4_albedo_01.wav|Albedo|EN|Who are you? Why did you alarm them?|neutral
```

**语言 ID:**
- `ZH`: 中文
- `JP`: 日语
- `EN`: 英语

**情绪标签:** `neutral`, `happy`, `sad`, `angry`, `fear` 等

#### Web UI 预处理步骤

在 Web 界面中依次执行：

1. **生成配置文件** - 设置 Batch Size（24GB 显存推荐 12）
2. **预处理音频文件** - 重采样到 44.1kHz
3. **预处理标签文件** - 分割训练集和验证集
4. **生成 BERT 特征** - 提取文本特征（耗时较长）

---

### 训练模型

#### 下载预训练底模

从 [ModelScope](https://www.modelscope.cn/models/Showmelater/bert-Vits2.3/tree/master/Bert-VITS2_2.3%E5%BA%95%E6%A8%A1) 下载以下文件到 `data/{你的数据集名称}/models/`:

- `D_0.pth`
- `DUR_0.pth`
- `WD_0.pth`
- `G_0.pth`

#### 修改配置文件

编辑根目录下的 `config.yml`:

```yaml
dataset_path: "data/{你的数据集名称}"
```

#### 开始训练

```bash
# 单卡训练
torchrun --nproc_per_node=1 train_ms.py

```

**训练监控：**

```bash
tensorboard --logdir=data/{你的数据集名称}/models
```

---

### 推理与部署

#### 修改配置

编辑 `config.yml` 中的 webui 部分：

```yaml
webui:
  device: "cuda"  # 或 "cpu"
  model: "data/{你的数据集名称}/models/G_10000.pth"  # 你的模型路径
  config_path: "data/{你的数据集名称}/configs/config.json"
  port: 7863
```

#### 启动推理 WebUI

```bash
python webui.py
```

访问 `http://127.0.0.1:7863` 进行语音合成。

---





### 技术参考
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
- [emotional-vits](https://github.com/innnky/emotional-vits)
- [fish-speech](https://github.com/fishaudio/fish-speech)
- [Bert-VITS2-UI](https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI)

