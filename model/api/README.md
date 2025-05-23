# 语音克隆API

本API提供了基于深度学习的语音克隆功能，能够根据参考音频和目标文本生成具有相同说话人特征的语音。

## 系统架构

语音克隆系统由三个核心组件组成：

1. **说话人编码器（X-Vector）**：从参考音频中提取说话人特征embedding
2. **文本到梅尔频谱转换（Transformer）**：将文本和说话人特征转换为梅尔频谱
3. **声码器（HIFI-GAN）**：将梅尔频谱转换为高质量音频波形

## 启动API服务器

```bash
python -m model.run_api_server --port 7860
```

参数说明：
- `--host`：服务器主机地址，默认为0.0.0.0
- `--port`：服务器端口，默认为7860
- `--model-dir`：模型目录，默认为model/data/checkpoints
- `--device`：设备，可选cpu或cuda，默认自动选择
- `--debug`：是否开启调试模式
- `--config`：配置文件路径

## API接口

### 1. 健康检查

**请求**：
```
GET /api/health
```

**响应**：
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. 语音克隆（一站式）

**请求**：
```
POST /api/clone
Content-Type: application/json

{
  "wav_path": "/path/to/reference.wav",
  "content": "参考音频的文本内容",
  "target_text": "要合成的目标文本"
}
```

**参数说明**：
- `wav_path`：参考音频文件路径
- `content`：参考音频的文本内容
- `target_text`：要合成的目标文本

**响应**：
```json
{
  "status": "success",
  "message": "语音克隆完成",
  "output_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "output_path": "/path/to/output.wav",
  "embedding_path": "/path/to/embedding.npy",
  "download_url": "/api/audio/a1b2c3d4-e5f6-7890-abcd-1234567890ab"
}
```

### 3. 提取说话人嵌入

**请求**：
```
POST /api/extract_embedding
Content-Type: multipart/form-data

audio: <audio_file>
```

**响应**：
```json
{
  "status": "success",
  "embedding_id": "a1b2c3d4.npy",
  "embedding_path": "/path/to/embedding.npy"
}
```

### 4. 合成语音

**请求**：
```
POST /api/synthesize
Content-Type: application/json

{
  "text": "要合成的文本",
  "embedding_id": "a1b2c3d4.npy"
}
```

或者：

```json
{
  "text": "要合成的文本",
  "reference_audio_id": "a1b2c3d4.wav"
}
```

**响应**：
```json
{
  "status": "success",
  "output_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "output_path": "/path/to/output.wav"
}
```

### 5. 上传参考音频

**请求**：
```
POST /api/upload_reference
Content-Type: multipart/form-data

audio: <audio_file>
```

**响应**：
```json
{
  "status": "success",
  "reference_audio_id": "a1b2c3d4.wav",
  "reference_path": "/path/to/reference.wav"
}
```

### 6. 获取音频文件

**请求**：
```
GET /api/audio/{output_id}
```

**响应**：
音频文件内容（audio/wav）

### 7. 训练X-Vector说话人编码器

**请求**：
```
POST /api/train/xvector
Content-Type: application/json

{
  "data_dir": "/path/to/speaker_data",
  "epochs": 100,
  "batch_size": 32,
  "embedding_dim": 512,
  "augment": true,
  "save_dir": "model/data/checkpoints"
}
```

**参数说明**：
- `data_dir`：训练数据目录，包含多个说话人子目录
- `epochs`：训练轮数（可选，默认100）
- `batch_size`：批次大小（可选，默认32）
- `embedding_dim`：嵌入向量维度（可选，默认512）
- `augment`：是否使用数据增强（可选，默认false）
- `save_dir`：模型保存目录（可选，默认model/data/checkpoints）

**响应**：
```json
{
  "status": "success",
  "message": "已启动X-Vector模型训练",
  "save_dir": "model/data/checkpoints",
  "expected_model_path": "model/data/checkpoints/xvector_best.pt",
  "final_model_path": "model/data/checkpoints/xvector_encoder.pt"
}
```

**注意**：训练过程在后台运行，可能需要较长时间。训练完成后，模型将自动加载。

### 8. 训练Transformer TTS模型

**请求**：
```
POST /api/train/transformer_tts
Content-Type: application/json

{
  "train_metadata": "/path/to/train_metadata.txt",
  "val_metadata": "/path/to/val_metadata.txt",
  "mel_dir": "/path/to/mel_spectrograms",
  "speaker_embed_dir": "/path/to/speaker_embeddings",
  "epochs": 100,
  "batch_size": 32,
  "d_model": 512,
  "speaker_dim": 512,
  "checkpoint_dir": "model/data/checkpoints"
}
```

**参数说明**：
- `train_metadata`：训练元数据文件路径
- `val_metadata`：验证元数据文件路径
- `mel_dir`：梅尔频谱目录（可选）
- `speaker_embed_dir`：说话人嵌入目录（可选）
- `epochs`：训练轮数（可选，默认100）
- `batch_size`：批次大小（可选，默认32）
- `d_model`：模型维度（可选，默认512）
- `speaker_dim`：说话人嵌入维度（可选，默认512）
- `checkpoint_dir`：检查点保存目录（可选，默认model/data/checkpoints）

**响应**：
```json
{
  "status": "success",
  "message": "已启动Transformer TTS模型训练",
  "checkpoint_dir": "model/data/checkpoints",
  "expected_model_path": "model/data/checkpoints/transformer_tts_best.pt",
  "final_model_path": "model/data/checkpoints/transformer_tts.pt"
}
```

**注意**：训练过程在后台运行，可能需要较长时间。训练完成后，模型将自动加载。

## 测试API

使用提供的测试脚本测试API：

```bash
python -m model.api.test_api \
  --url http://localhost:7860 \
  --wav /path/to/reference.wav \
  --content "参考音频的文本内容" \
  --text "要合成的目标文本"
```

参数说明：
- `--url`：API服务器URL，默认为http://localhost:7860
- `--wav`：参考音频文件路径
- `--content`：参考音频的文本内容，默认为"这是参考音频的内容"
- `--text`：要合成的目标文本，默认为"这是要合成的目标文本"
- `--config`：配置文件路径

## 工作流程

1. 客户端发送包含wav文件路径、content内容和target_text的请求到/api/clone接口
2. 服务器加载wav文件，使用X-Vector模型提取说话人特征embedding
3. 服务器将embedding和target_text输入到Transformer模型，生成梅尔频谱
4. 服务器使用HIFI-GAN将梅尔频谱转换为音频波形
5. 服务器保存生成的音频文件，并返回下载链接
6. 客户端可以通过下载链接获取生成的音频文件 

## 训练流程

### X-Vector训练数据准备

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

### Transformer TTS训练数据准备

训练Transformer TTS模型需要准备元数据文件，包含文本、梅尔频谱路径和说话人ID/嵌入路径。
元数据文件格式为：
```
文本内容|梅尔频谱路径|说话人ID|说话人嵌入路径
```

例如：
```
这是一段测试文本|/path/to/mel/file1.npy|speaker1|/path/to/embed/file1.npy
这是另一段文本|/path/to/mel/file2.npy|speaker2|/path/to/embed/file2.npy
``` 