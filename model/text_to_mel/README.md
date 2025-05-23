# 文本到梅尔频谱转换模块

本模块包含两种文本到梅尔频谱转换模型实现：
1. 传统的Tacotron2风格模型（TextEncoder + MelDecoder）
2. **基于Transformer的TTS模型**（推荐使用）

## 基于Transformer的TTS模型

基于Transformer的TTS模型是一种现代的序列到序列模型，相比于基于RNN的Tacotron2，它具有以下优势：
- 并行计算能力更强，训练速度更快
- 能够处理更长的序列
- 更好地捕捉长距离依赖关系
- 注意力机制更加灵活

### 模型架构

模型主要包含以下组件：
- 文本编码器：基于Transformer Encoder，将文本序列编码为隐藏表示
- 说话人嵌入：将说话人特征融合到文本编码中
- 梅尔解码器：基于Transformer Decoder，将编码后的表示解码为梅尔频谱
- 位置编码：为序列提供位置信息
- 停止标志预测：预测生成何时应该停止

### 使用方法

#### 1. 推理

```python
from model import TransformerTTS

# 初始化模型
model = TransformerTTS(
    vocab_size=256,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    speaker_dim=512,
    mel_dim=80
)

# 加载预训练权重
model.load_state_dict(torch.load("model/data/checkpoints/transformer_tts.pt"))
model.eval()

# 准备输入
text_ids = torch.tensor([[ord(c) % 256 for c in "你好，世界！"]], dtype=torch.long)
speaker_embedding = torch.randn(1, 512)  # 从说话人编码器获取

# 推理
with torch.no_grad():
    mel_outputs = model.inference(text_ids, speaker_embedding)
```

#### 2. 与VoiceCloneSystem集成

```python
from model import VoiceCloneSystem

# 初始化语音克隆系统，指定使用Transformer TTS
system = VoiceCloneSystem(
    model_dir="model/data/checkpoints",
    encoder_type="xvector",
    tts_type="transformer",
    vocoder_type="hifigan"
)

# 克隆语音
system.clone_voice("你好，世界！", "model/data/wavs/reference.wav", "model/data/outputs/output.wav")
```

### 训练模型

使用提供的训练脚本训练模型：

```bash
python -m model.text_to_mel.train_transformer_tts \
  --train_metadata model/data/metadata/train.txt \
  --val_metadata model/data/metadata/val.txt \
  --mel_dir model/data/mels \
  --speaker_embed_dir model/data/embeddings \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 0.0001 \
  --checkpoint_dir model/data/checkpoints \
  --log_dir model/data/logs
```

### 数据格式

训练数据元数据文件格式为：
```
文本|梅尔频谱路径|说话人ID|说话人嵌入路径(可选)
```

例如：
```
你好，世界！|speaker1_001.npy|1|speaker1.npy
早上好！|speaker2_001.npy|2|speaker2.npy
```

### 模型参数说明

- `vocab_size`: 词汇表大小，默认256（基本ASCII字符集）
- `d_model`: 模型维度，默认512
- `nhead`: 注意力头数，默认8
- `num_encoder_layers`: 编码器层数，默认6
- `num_decoder_layers`: 解码器层数，默认6
- `dim_feedforward`: 前馈网络维度，默认2048
- `dropout`: Dropout率，默认0.1
- `speaker_dim`: 说话人嵌入维度，默认512
- `mel_dim`: 梅尔频谱维度，默认80
- `max_seq_len`: 最大序列长度，默认1000

### 示例脚本

可以使用`model/examples/transformer_tts_demo.py`脚本来测试模型：

```bash
python -m model.examples.transformer_tts_demo \
  --reference_audio model/data/wavs/reference.wav \
  --text "你好，世界！" \
  --output_dir model/data/outputs \
  --visualize
``` 