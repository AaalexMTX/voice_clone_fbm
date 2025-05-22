# 语音克隆系统 (Voice Clone FBM)

这是一个基于Transformer模型的语音克隆系统，能够从参考音频中提取说话人特征，并将这些特征应用到新的文本上，生成具有相同声音特征的合成语音。

## 项目结构

```
.
├── model/             # 核心模型和功能
│   ├── api/           # API服务相关模块
│   ├── config/        # 配置相关模块
│   ├── core/          # 核心功能模块
│   ├── decoder/       # 解码器模块
│   ├── encoder/       # 编码器模块
│   └── utils/         # 工具函数
├── backend/           # 后端服务（可选）
├── frontend/          # 前端界面（可选）
├── visualization/     # 可视化工具（可选）
├── tests/             # 测试代码
├── utils/             # 通用工具
└── data/              # 数据处理脚本和示例
```

## 特点

- 基于Transformer的端到端语音克隆
- 支持零样本语音克隆（无需重新训练）
- REST API支持
- 可扩展的模块化设计
- 支持多种配置选项

## 快速开始

### 安装依赖

```bash
pip install -r model/requirements.txt
```

### 初始化工作空间

```bash
python -m model init
```

### 启动服务器

```bash
python -m model server
```

### 运行API测试

```bash
python -m model test
```

### 运行推理

```bash
python -m model inference --text "这是一段测试语音" --reference_audio model/tests/sample.wav
```

## 详细文档

请查看 [model/README.md](model/README.md) 了解更多详细信息。

## 许可证

[MIT License](LICENSE)