# 说话人编码器模块

本模块包含两种说话人编码器实现：
1. SpeakerEncoder - 基于LSTM的编码器
2. XVectorEncoder - 基于TDNN的X-Vector编码器(推荐使用)

X-Vector是一种用于说话人识别的深度神经网络嵌入方法，具有较强的说话人特征提取能力。

## 使用X-Vector进行训练

### 数据准备
训练X-Vector模型需要准备大量的说话人数据，每个说话人至少需要几十条语音样本。
数据应按以下结构组织：
```
data_dir/
├── speaker1/
│   ├── utt1.wav
│   ├── utt2.wav
│   └── ...
├── speaker2/
│   ├── utt1.wav
│   ├── utt2.wav
│   └── ...
└── ...
```

### 训练命令
```bash
python -m model.speaker_encoder.train_xvector \
  --data_dir /path/to/speaker_data \
  --batch_size 32 \
  --epochs 100 \
  --embedding_dim 512 \
  --save_dir /path/to/save/model \
  --augment
```

### 主要参数
- `--data_dir`: 数据目录，包含多个说话人子目录
- `--embedding_dim`: 输出嵌入向量维度，默认512
- `--batch_size`: 批次大小，根据显存调整
- `--epochs`: 训练轮数
- `--augment`: 是否使用数据增强
- `--save_dir`: 模型保存目录

## 说话人验证

使用训练好的X-Vector模型进行说话人验证：

```bash
python -m model.tools.speaker_verify \
  --audio1 /path/to/audio1.wav \
  --audio2 /path/to/audio2.wav \
  --model_path /path/to/xvector_best.pt \
  --threshold 0.75
```

### 主要参数
- `--audio1`: 第一个音频文件路径
- `--audio2`: 第二个音频文件路径
- `--model_path`: X-Vector模型路径
- `--threshold`: 判定为同一说话人的相似度阈值（0.0-1.0）
- `--save_embeddings`: 是否保存嵌入向量（可选）

## 集成到语音克隆系统

在初始化`VoiceCloneSystem`时，可以指定使用X-Vector作为说话人编码器：

```python
from model import VoiceCloneSystem

# 使用X-Vector说话人编码器
system = VoiceCloneSystem(
    model_dir="models",
    encoder_type="xvector"  # 使用X-Vector
)

# 克隆语音
system.clone_voice(
    text="这是一段用于测试的文本。",
    reference_audio_path="speaker.wav",
    output_path="output.wav"
)
```

## 嵌入向量提取

可以单独使用X-Vector模型提取说话人嵌入向量：

```python
from model.speaker_encoder import XVectorEncoder

# 初始化X-Vector模型
model = XVectorEncoder(mel_n_channels=80, embedding_dim=512)
model.load("path/to/xvector_model.pt")

# 从文件提取嵌入向量
embed = model.embed_from_file("speaker.wav")
```

## 模型架构

X-Vector模型由以下几个部分组成：
1. 帧级特征提取层 - 使用TDNN (时延神经网络) 层
2. 统计池化层 - 计算均值和标准差
3. 段级特征提取层 - 全连接层
4. 嵌入层 - 生成最终的固定维度嵌入向量 