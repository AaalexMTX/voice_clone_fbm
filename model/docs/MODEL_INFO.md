# 语音克隆系统预训练模型

本项目使用了以下预训练模型来实现语音克隆功能：

## 1. X-Vector模型（说话人编码器）

**用途**：提取说话人的声音特征，将声音特征编码为固定维度的向量表示。

**模型文件**：
- 模型：`model/data/checkpoints/speaker_encoder/xvector.ckpt`
- 配置：`model/data/checkpoints/speaker_encoder/xvector_config.yaml`

**来源**：[speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)

**说明**：这是在VoxCeleb数据集上训练的X-Vector模型，能够从语音中提取高质量的说话人特征。

## 2. Transformer TTS模型（文本到梅尔频谱转换）

**用途**：将文本转换为梅尔频谱图，同时融合说话人的声音特征。

**模型文件**：
- 模型：`model/data/checkpoints/transformer_tts/coqui_XTTS-v2_model.pth`
- 配置：`model/data/checkpoints/transformer_tts/coqui_XTTS-v2_config.yaml`

**来源**：[coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

**说明**：这是Coqui的XTTS-v2模型，支持高质量的语音合成和跨语言语音克隆。

## 3. HiFi-GAN模型（声码器）

**用途**：将梅尔频谱图转换为波形，生成最终的音频。

**模型文件**：
- 模型：`model/vocoder/models/hifigan_vocoder.pt`
- 配置：`model/vocoder/models/hifigan_config.json`

**来源**：预装在项目中

**说明**：这是一个通用的HiFi-GAN声码器，能够从梅尔频谱图生成高质量的语音波形。

## 使用方法

1. 从参考音频中提取说话人特征（使用X-Vector模型）
2. 将文本和说话人特征输入Transformer TTS模型，生成梅尔频谱图
3. 使用HiFi-GAN声码器将梅尔频谱图转换为音频波形

## 模型下载脚本

本项目提供了以下脚本用于下载和管理预训练模型：

- `download_models.py`：下载X-Vector和备选TTS模型
- `download_tts_model.py`：专门用于下载Transformer TTS模型
- `download_hifigan.py`：下载HiFi-GAN声码器模型（如需替换现有声码器） 