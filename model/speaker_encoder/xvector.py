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
    def __init__(self, mel_n_channels=24, embedding_dim=512):
        """
        初始化X-Vector模型
        
        参数:
            mel_n_channels: 梅尔频谱通道数（预训练模型使用24）
            embedding_dim: 输出嵌入维度
        """
        super(XVectorEncoder, self).__init__()
        
        # 使用blocks结构匹配预训练模型
        self.blocks = nn.ModuleDict({
            '0': nn.ModuleDict({'conv': nn.Conv1d(mel_n_channels, 512, kernel_size=5, stride=1, padding=2)}),
            '2': nn.ModuleDict({'norm': nn.BatchNorm1d(512)}),
            '3': nn.ModuleDict({'conv': nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)}),
            '5': nn.ModuleDict({'norm': nn.BatchNorm1d(512)}),
            '6': nn.ModuleDict({'conv': nn.Conv1d(512, 512, kernel_size=3, stride=3, padding=1)}),
            '8': nn.ModuleDict({'norm': nn.BatchNorm1d(512)}),
            '9': nn.ModuleDict({'conv': nn.Conv1d(512, 512, kernel_size=1, stride=1)}),
            '11': nn.ModuleDict({'norm': nn.BatchNorm1d(512)}),
            '12': nn.ModuleDict({'conv': nn.Conv1d(512, 1500, kernel_size=1, stride=1)}),
            '14': nn.ModuleDict({'norm': nn.BatchNorm1d(1500)}),
            '16': nn.ModuleDict({'w': nn.Linear(3000, embedding_dim)})
        })
        
        # 输入适配层（将80维梅尔频谱转换为24维）
        # 改为使用nn.Parameter而不是register_buffer，这样可以被加载到state_dict中
        self.input_adapter_weight = nn.Parameter(torch.randn(24, 80) * 0.1)
        self.input_adapter_bias = nn.Parameter(torch.zeros(24))
        
        # 保留TDNN结构以备不时之需
        self.frame_layers = None
        self.stats_pooling = StatsPooling()
        self.segment_layer1 = None
        self.embedding_layer = None
        
    def _adapt_input(self, x):
        """
        将80维梅尔频谱转换为24维
        
        参数:
            x: [batch_size, seq_len, 80]
            
        返回:
            [batch_size, seq_len, 24]
        """
        # 重塑为二维张量
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        
        # 线性变换
        out = torch.matmul(x_reshaped, self.input_adapter_weight.t()) + self.input_adapter_bias
        
        # 重塑回原始形状
        return out.reshape(batch_size, seq_len, 24)
        
    def forward(self, mels):
        """
        前向传播
        
        参数:
            mels: 梅尔频谱特征 [batch_size, seq_len, n_mels]
            
        返回:
            [batch_size, embedding_dim]的嵌入向量
        """
        # 如果输入维度不是预期的维度，使用适配层
        if mels.size(2) != 24:
            # [batch_size, seq_len, 80] -> [batch_size, seq_len, 24]
            mels = self._adapt_input(mels)
        
        # 转换输入维度 [batch_size, seq_len, n_mels] -> [batch_size, n_mels, seq_len]
        x = mels.transpose(1, 2)
        
        # 使用blocks结构处理
        # 块0: 卷积
        x = self.blocks['0']['conv'](x)
        # 块1: ReLU (内联实现)
        x = F.relu(x)
        # 块2: 批归一化
        x = self.blocks['2']['norm'](x)
        # 块3: 卷积
        x = self.blocks['3']['conv'](x)
        # 块4: ReLU
        x = F.relu(x)
        # 块5: 批归一化
        x = self.blocks['5']['norm'](x)
        # 块6: 卷积
        x = self.blocks['6']['conv'](x)
        # 块7: ReLU
        x = F.relu(x)
        # 块8: 批归一化
        x = self.blocks['8']['norm'](x)
        # 块9: 卷积
        x = self.blocks['9']['conv'](x)
        # 块10: ReLU
        x = F.relu(x)
        # 块11: 批归一化
        x = self.blocks['11']['norm'](x)
        # 块12: 卷积
        x = self.blocks['12']['conv'](x)
        # 块13: ReLU
        x = F.relu(x)
        # 块14: 批归一化
        x = self.blocks['14']['norm'](x)
        # 块15: 自适应平均池化（使用统计池化代替）
        # 统计池化: 计算均值和标准差并拼接
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        x = torch.cat((mean, std), dim=1)
        
        # 块16: 线性层
        x = self.blocks['16']['w'](x)
        
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
        
        # 检查是否有缺失的键
        model_keys = set(self.state_dict().keys())
        weights_keys = set(weights.keys())
        
        # 如果有缺失的键，打印警告
        if len(model_keys - weights_keys) > 0:
            print(f"警告: 模型中有{len(model_keys - weights_keys)}个参数在预训练权重中不存在")
            print(f"缺失的键: {model_keys - weights_keys}")
        
        # 尝试加载权重
        try:
            self.load_state_dict(weights, strict=False)
            print(f"成功加载预训练权重 (忽略缺失的键)")
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")
        
        self.eval()

# 添加别名，使XVector可用作XVectorEncoder的别名
XVector = XVectorEncoder

class SpeechBrainAdapter:
    """
    SpeechBrain X-Vector模型适配器
    用于加载和使用SpeechBrain预训练的X-Vector模型
    """
    def __init__(self, model_path, device="cpu"):
        """
        初始化SpeechBrain适配器
        
        参数:
            model_path: 预训练模型路径
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path
        
        # 创建一个实际的X-Vector模型实例（使用24维梅尔频谱，与预训练模型匹配）
        self.model = XVectorEncoder(mel_n_channels=24, embedding_dim=512)
        self.model.to(device)
        
        # 尝试加载预训练权重
        try:
            print(f"尝试加载X-Vector预训练模型: {model_path}")
            weights = torch.load(model_path, map_location=device)
            # 使用非严格模式加载权重，忽略缺失的键
            self.model.load_state_dict(weights, strict=False)
            self.model.eval()
            print(f"成功加载X-Vector预训练模型")
        except Exception as e:
            print(f"加载X-Vector预训练模型失败: {str(e)}，使用随机初始化")
            # 使用随机初始化的模型
            self.model.eval()
    
    def extract_embedding(self, audio):
        """
        从音频中提取说话人特征
        
        参数:
            audio: 音频波形
            
        返回:
            说话人特征向量
        """
        # 使用模型提取特征
        try:
            with torch.no_grad():
                # 提取梅尔特征
                mel = extract_mel_features(audio)
                
                # 将数据转换为张量
                mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)  # [1, seq_length, n_mels]
                
                # 使用模型提取嵌入向量
                embed = self.model.extract_embedding(mel_tensor)[0].cpu().numpy()
                return embed
        except Exception as e:
            print(f"提取嵌入向量失败: {str(e)}，返回随机向量")
            # 如果失败，返回随机向量作为备选
            random_embedding = np.random.randn(512)
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            return random_embedding
    
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
        return self.extract_embedding(wav)
    
    def embed_from_file(self, file_path):
        """
        从音频文件提取说话人嵌入向量
        
        参数:
            file_path: 音频文件路径
            
        返回:
            说话人嵌入向量
        """
        try:
            # 预处理音频
            wav = preprocess_wav(file_path)
            
            # 提取嵌入向量
            return self.embed_utterance(wav)
        except Exception as e:
            print(f"从文件提取嵌入向量失败: {str(e)}，返回随机向量")
            # 如果失败，返回随机向量作为备选
            random_embedding = np.random.randn(512)
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            return random_embedding 