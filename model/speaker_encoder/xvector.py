import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .audio_processing import preprocess_wav
from .mel_features import extract_mel_features

class TDNN(nn.Module):
    """
    时延神经网络(TDNN)层实现，适用于提取帧级特征
    """
    def __init__(self, input_dim, output_dim, context_size, dilation=1, stride=1):
        """
        初始化TDNN层
        
        参数:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            context_size: 上下文窗口大小
            dilation: 空洞卷积扩张系数
            stride: 步长
        """
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        
        self.kernel = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=context_size,
            stride=stride,
            padding=0,
            dilation=dilation
        )
        self.nonlinearity = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, seq_len, input_dim]
            
        返回:
            [batch_size, new_seq_len, output_dim]
        """
        # 将输入转置为 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 计算有效帧长
        x = self.kernel(x)
        x = self.nonlinearity(x)
        x = self.bn(x)
        
        # 转置回 [batch_size, seq_len, output_dim]
        x = x.transpose(1, 2)
        return x

class StatsPooling(nn.Module):
    """
    统计池化层，计算均值和标准差
    """
    def __init__(self):
        super(StatsPooling, self).__init__()
        
    def forward(self, x):
        """
        计算输入张量沿时间维度的均值和标准差
        
        参数:
            x: 输入特征 [batch_size, seq_len, dim]
            
        返回:
            [batch_size, dim*2]
        """
        # 均值池化
        mean = torch.mean(x, dim=1)
        
        # 标准差池化
        std = torch.std(x, dim=1)
        
        # 拼接均值和标准差
        pooled = torch.cat((mean, std), dim=1)
        
        return pooled

class XVectorEncoder(nn.Module):
    """
    X-Vector说话人编码器: 基于TDNN架构提取说话人嵌入
    参考论文: "X-vectors: Robust DNN Embeddings for Speaker Recognition"
    """
    def __init__(self, mel_n_channels=80, embedding_dim=512):
        """
        初始化X-Vector模型
        
        参数:
            mel_n_channels: 梅尔频谱通道数
            embedding_dim: 输出嵌入维度
        """
        super(XVectorEncoder, self).__init__()
        
        # TDNN帧级特征提取层
        self.frame_layers = nn.Sequential(
            TDNN(mel_n_channels, 512, 5, 1),     # 帧1: 使用5帧上下文
            TDNN(512, 512, 3, 2),               # 帧2: 使用3帧上下文，步长为2
            TDNN(512, 512, 3, 3),               # 帧3: 使用3帧上下文，步长为3
            TDNN(512, 512, 1, 1),               # 帧4: 使用当前帧
            TDNN(512, 1500, 1, 1)               # 帧5: 使用当前帧
        )
        
        # 统计池化层
        self.stats_pooling = StatsPooling()
        
        # 段级特征提取层
        self.segment_layer1 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # 嵌入层
        self.embedding_layer = nn.Linear(512, embedding_dim)
        
    def forward(self, mels):
        """
        前向传播
        
        参数:
            mels: 梅尔频谱特征 [batch_size, seq_len, n_mels]
            
        返回:
            [batch_size, embedding_dim]的嵌入向量
        """
        # 帧级特征提取
        x = self.frame_layers(mels)
        
        # 统计池化
        x = self.stats_pooling(x)
        
        # 段级特征提取
        x = self.segment_layer1(x)
        
        # 嵌入层
        x = self.embedding_layer(x)
        
        # 归一化嵌入
        embeds = F.normalize(x, p=2, dim=1)
        
        return embeds
    
    def extract_embedding(self, mels):
        """
        提取嵌入向量(用于推理阶段)
        
        参数:
            mels: 梅尔频谱特征 [batch_size, seq_len, n_mels]
            
        返回:
            归一化的嵌入向量
        """
        self.eval()
        with torch.no_grad():
            embeds = self.forward(mels)
        return embeds
    
    def embed_utterance(self, wav, sr=16000, return_partials=False):
        """
        从完整的语音中提取嵌入向量
        
        参数:
            wav: 预处理过的语音波形
            sr: 采样率
            return_partials: 是否返回部分嵌入向量
            
        返回:
            说话人嵌入向量
        """
        # 提取梅尔特征
        mel = extract_mel_features(wav, sr=sr)
        
        # 将数据转换为张量
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # [1, seq_length, n_mels]
        
        # 确保在评估模式
        self.eval()
        with torch.no_grad():
            embed = self.extract_embedding(mel_tensor)[0].cpu().numpy()
            
        return embed
    
    def embed_from_file(self, file_path):
        """
        直接从音频文件提取说话人嵌入向量
        
        参数:
            file_path: 音频文件路径
            
        返回:
            说话人嵌入向量
        """
        # 预处理音频
        wav = preprocess_wav(file_path)
        
        # 提取嵌入向量
        return self.embed_utterance(wav)
    
    def save(self, path):
        """保存模型"""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """加载模型"""
        weights = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(weights)
        self.eval() 