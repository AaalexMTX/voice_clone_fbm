"""
声码器基类，定义了将梅尔频谱转换为波形的接口
"""

import enum
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List, Tuple

class VocoderType(enum.Enum):
    """声码器类型枚举"""
    HIFIGAN = "hifigan"
    GRIFFINLIM = "griffinlim"
    MELGAN = "melgan"
    WAVEGLOW = "waveglow"
    
    @classmethod
    def from_string(cls, name: str) -> "VocoderType":
        """从字符串获取声码器类型"""
        name = name.lower()
        for t in cls:
            if t.value == name:
                return t
        raise ValueError(f"未知的声码器类型: {name}")

class Vocoder(nn.Module):
    """
    声码器基类，定义了将梅尔频谱转换为波形的接口
    所有声码器实现都应继承此类
    """
    def __init__(self):
        super(Vocoder, self).__init__()
        self.vocoder_type = None
        self.n_mels = 80  # 默认值
        
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        将梅尔频谱转换为音频波形
        
        参数:
            mel_spectrogram: [batch_size, n_mels, time] 或 [batch_size, time, n_mels]的梅尔频谱
            
        返回:
            [batch_size, time']的音频波形
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def generate_audio(self, mel_spectrogram: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        从梅尔频谱生成音频
        
        参数:
            mel_spectrogram: 梅尔频谱，可以是张量或NumPy数组
            
        返回:
            音频波形的NumPy数组
        """
        # 转换为张量（如果不是）
        if not isinstance(mel_spectrogram, torch.Tensor):
            mel_spectrogram = torch.FloatTensor(mel_spectrogram)
            
        # 添加批次维度（如果没有）
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
            
        # 确保处于评估模式
        self.eval()
        
        # 使用无梯度运行
        with torch.no_grad():
            # 前向传播
            waveform = self.forward(mel_spectrogram)
            
            # 转换为NumPy数组
            waveform = waveform.squeeze().cpu().numpy()
            
        return waveform
    
    def get_vocoder_type(self) -> VocoderType:
        """获取声码器类型"""
        if self.vocoder_type is None:
            raise NotImplementedError("子类必须设置vocoder_type属性")
        return self.vocoder_type
    
    @classmethod
    def from_pretrained(cls, model_path: str, config_path: Optional[str] = None):
        """
        从预训练模型加载声码器
        
        参数:
            model_path: 预训练模型路径
            config_path: 配置文件路径
        
        返回:
            加载了预训练权重的声码器
        """
        raise NotImplementedError("子类必须实现from_pretrained方法") 