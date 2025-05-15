# 语音克隆系统

基于Transformer的语音克隆系统，实现个性化语音合成。

## 项目结构

```
├── model/          # 语音克隆模型
│   ├── encoder/    # 声音编码器
│   └── decoder/    # 声音解码器
├── backend/        # Golang后端服务
│   ├── api/        # API接口
│   ├── service/    # 业务逻辑
│   └── model/      # 数据模型
└── frontend/       # Web前端
    ├── src/        # 源代码
    └── public/     # 静态资源
```

## 技术栈

- 模型：PyTorch + Transformer
- 后端：Golang + Gin
- 前端：React + TypeScript

## 功能特性

- 音频录制和上传
- 声音特征提取
- 语音合成
- 实时预览