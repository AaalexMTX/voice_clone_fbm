import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .audio_processing import preprocess_wav
from .mel_features import extract_mel_features

class SpeakerEncoder(nn.Module):
    """
    说话人编码模型：从语音中提取说话人身份特征
    基于简化的GE2E损失函数和LSTM架构
    """
    def __init__(self, mel_n_channels=80, model_hidden_size=256, model_embedding_size=256, model_num_layers=3):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=mel_n_channels,
            hidden_size=model_hidden_size,
            num_layers=model_num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        
    def forward(self, mels):
        """
        输入梅尔频谱，输出固定维度的嵌入向量
        
        参数:
            mels: 形状为[batch_size, seq_length, n_mels]的梅尔频谱
            
        返回:
            形状为[batch_size, embedding_size]的说话人嵌入向量
        """
        # 通过LSTM提取序列特征
        lstm_out, _ = self.lstm(mels)
        
        # 取最后一帧输出
        last_frame = lstm_out[:, -1]
        
        # 投影到嵌入空间
        embeds = self.linear(last_frame)
        
        # 归一化嵌入向量
        embeds = self.relu(embeds)
        embeds = F.normalize(embeds, p=2, dim=1)
        
        return embeds
    
    @staticmethod
    def compute_partial_slices(n_samples, partial_utterance_n_frames=160, 
                              overlap=0.5):
        """
        计算音频分割的起始和结束索引
        
        参数:
            n_samples: 样本总数
            partial_utterance_n_frames: 每个部分的帧数
            overlap: 重叠比例
            
        返回:
            切片列表: (start, end)元组的列表
        """
        assert 0 <= overlap < 1
        assert partial_utterance_n_frames > 0
        
        step = int(partial_utterance_n_frames * (1 - overlap))
        slices = []
        for start_idx in range(0, n_samples - partial_utterance_n_frames + 1, step):
            end_idx = start_idx + partial_utterance_n_frames
            slices.append((start_idx, end_idx))
            
        return slices
        
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
            embed = self.forward(mel_tensor)[0].cpu().numpy()
            
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
        self.load_state_dict(torch.load(path))
        self.eval() 