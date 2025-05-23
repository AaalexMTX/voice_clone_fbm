# 语音克隆系统 (Voice Clone System)

基于预训练模型的语音克隆系统，使用参考音频的声音特征为任意文本生成具有相同声音特征的合成语音。

## 项目结构

```
├── model/                  # 核心模型目录
│   ├── api/                # API服务
│   │   ├── server.py       # 服务器实现
│   │   └── voice_clone_service.py  # 服务层实现
│   ├── core/               # 核心功能
│   │   └── voice_clone.py  # 语音克隆核心实现
│   ├── speaker_encoder/    # 说话人编码器
│   │   └── xvector.py      # X-Vector模型
│   ├── text_to_mel/        # 文本到梅尔频谱转换
│   │   └── transformer_tts.py  # Transformer TTS模型
│   ├── vocoder/            # 声码器
│   │   └── hifigan.py      # HiFi-GAN声码器
│   ├── data/               # 数据和模型
│   │   └── checkpoints/    # 预训练模型
│   │       ├── speaker_encoder/  # X-Vector模型
│   │       ├── transformer_tts/  # Transformer TTS模型
│   │       └── vocoder/          # HiFi-GAN模型
│   ├── run_api_server.py   # API服务器启动脚本
│   └── requirements.txt    # 依赖列表
├── backend/                # 后端实现
│   └── voice_clone_client.py  # API客户端示例
├── outputs/                # 输出目录
├── MODEL_INFO.md           # 模型信息
├── download_models.py      # 模型下载脚本
└── README.md               # 项目文档
```

## 功能特点

- **语音克隆**：使用参考音频生成具有相同声音特征的合成语音
- **预训练模型**：使用高质量预训练模型，无需额外训练
- **REST API**：提供易于集成的HTTP接口
- **说话人特征缓存**：优化性能，避免重复计算
- **异步处理**：支持长文本和批量处理
- **多格式支持**：支持多种音频格式

## 技术架构

本项目使用三阶段语音克隆流程：

1. **说话人编码（X-Vector模型）**：从参考音频中提取说话人特征
2. **文本到梅尔频谱（Transformer TTS模型）**：将文本和说话人特征转换为梅尔频谱图
3. **波形生成（HiFi-GAN声码器）**：将梅尔频谱图转换为高质量音频波形

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/voice_clone_fbm.git
cd voice_clone_fbm
```

2. 安装依赖：

```bash
pip install -r model/requirements.txt
```

3. 下载预训练模型：

```bash
python download_models.py
```

## 使用方法

### 启动API服务器

```bash
python model/run_api_server.py
```

服务默认在 http://localhost:5000 运行

### 使用客户端示例

```bash
python backend/voice_clone_client.py --reference-audio samples/reference.wav --text "这是一段测试语音"
```

命令行参数：

- `--reference-audio`: 参考音频文件路径
- `--text`: 要合成的文本
- `--output`: 输出文件路径（可选）
- `--format`: 输出格式，默认wav（可选）
- `--async`: 使用异步模式（可选）
- `--no-cache`: 禁用特征缓存（可选）

### API接口

#### 健康检查

```
GET /api/health
```

#### 提取说话人特征

```
POST /api/extract_embedding
{
  "reference_audio": "path/to/audio.wav",
  "force_recompute": false
}
```

#### 克隆语音

```
POST /api/clone_voice
{
  "reference_audio": "path/to/audio.wav",
  "text": "要合成的文本内容",
  "output_format": "wav",
  "use_cache": true
}
```

#### 异步克隆语音

```
POST /api/async_clone_voice
{
  "reference_audio": "path/to/audio.wav",
  "text": "要合成的文本内容",
  "output_format": "wav",
  "use_cache": true
}
```

#### 获取任务状态

```
GET /api/task/{task_id}
```

#### 列出任务

```
GET /api/tasks?limit=10&status=completed
```

#### 获取音频文件

```
GET /api/audio/{filename}
```

#### 上传参考音频

```
POST /api/upload_reference
Content-Type: multipart/form-data
file: (音频文件)
```

## 配置

可以通过创建 `config.yaml` 文件并使用 `--config` 命令行参数指定配置文件。

配置选项：

```yaml
server:
  host: "0.0.0.0"
  port: 5000
  
model:
  output_dir: "outputs"
  temp_dir: "temp"
  model_dir: "model/data/checkpoints"
  cache_dir: "model/data/cache"
  max_text_length: 500
  allowed_audio_formats:
    - wav
    - mp3
    - flac
    - ogg
```

## 模型信息

本项目使用以下预训练模型：

- **X-Vector**：用于从音频中提取说话人特征
- **Transformer TTS**：将文本和说话人特征转换为梅尔频谱图
- **HiFi-GAN**：将梅尔频谱图转换为高质量音频

详细信息请查看 [MODEL_INFO.md](MODEL_INFO.md)

## 性能优化

- 说话人特征提取是计算密集型任务，使用缓存可显著提高性能
- 对于长文本，建议使用异步API并分段处理
- GPU加速可大幅提高处理速度

## 待办事项

- [ ] 添加多语言支持
- [ ] 提供情感控制参数
- [ ] 改进异常处理
- [ ] 添加用户界面

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎贡献！请在Pull Request之前先提Issues讨论。