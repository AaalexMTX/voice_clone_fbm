#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
声码器基础类
将梅尔频谱转换为音频波形
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto
from typing import Optional, Union, Dict, Any

class VocoderType(Enum):
    """声码器类型枚举"""
    SIMPLE = auto()       # 简单声码器
    HIFIGAN = auto()      # HiFi-GAN
    GRIFFINLIM = auto()   # Griffin-Lim
    MELGAN = auto()       # MelGAN
    UNIVERSAL = auto()    # Universal HiFi-GAN
    
    @staticmethod
    def from_string(name: str) -> 'VocoderType':
        """从字符串转换为枚举值"""
        name = name.lower()
        if name == "simple":
            return VocoderType.SIMPLE
        elif name == "hifigan":
            return VocoderType.HIFIGAN
        elif name == "universal_hifigan" or name == "universal":
            return VocoderType.UNIVERSAL
        elif name == "griffinlim":
            return VocoderType.GRIFFINLIM
        elif name == "melgan":
            return VocoderType.MELGAN
        else:
            raise ValueError(f"未知的声码器类型: {name}")
    
    def __str__(self) -> str:
        """转换为字符串"""
        if self == VocoderType.SIMPLE:
            return "simple"
        elif self == VocoderType.HIFIGAN:
            return "hifigan"
        elif self == VocoderType.UNIVERSAL:
            return "universal_hifigan"
        elif self == VocoderType.GRIFFINLIM:
            return "griffinlim"
        elif self == VocoderType.MELGAN:
            return "melgan"
        else:
            return "unknown"

class ConvLayer(nn.Module):
    """卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class TransposeConvLayer(nn.Module):
    """转置卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TransposeConvLayer, self).__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Vocoder(nn.Module):
    """
    基础声码器模型，将梅尔频谱转换为音频波形
    """
    def __init__(self, n_mels=80, channels=[512, 256, 128, 64, 32]):
        super(Vocoder, self).__init__()
        
        # 保存配置
        self.n_mels = n_mels
        self.channels = channels
        self.vocoder_type = VocoderType.SIMPLE
        
        # 上采样层
        self.pre_conv = ConvLayer(n_mels, channels[0], kernel_size=7, stride=1)
        
        # 上采样堆栈
        self.up_stack = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.up_stack.append(
                TransposeConvLayer(
                    channels[i], 
                    channels[i+1], 
                    kernel_size=16 if i < 2 else 8 if i == 2 else 4,
                    stride=8 if i < 2 else 4 if i == 2 else 2
                )
            )
        
        # 输出层
        self.output_layer = nn.Conv1d(channels[-1], 1, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
        
    def forward(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频波形
        
        参数:
            mel_spectrogram: [batch_size, time, n_mels]的梅尔频谱
            
        返回:
            [batch_size, 1, time*factor]的音频波形
        """
        # 转换输入维度 [batch, time, n_mels] -> [batch, n_mels, time]
        if mel_spectrogram.dim() == 3 and mel_spectrogram.size(2) == self.n_mels:
            x = mel_spectrogram.transpose(1, 2)
        else:
            x = mel_spectrogram
        
        # 初始卷积
        x = self.pre_conv(x)
        
        # 上采样堆栈
        for up_layer in self.up_stack:
            x = up_layer(x)
            
        # 输出层
        x = self.output_layer(x)
        x = self.tanh(x)
        
        return x
    
    @classmethod
    def from_pretrained(cls, model_path, config_path=None):
        """
        从预训练模型加载声码器
        
        参数:
            model_path: 预训练模型路径
            config_path: 模型配置路径（可选）
            
        返回:
            加载了预训练权重的Vocoder实例
        """
        # 加载配置（如果提供）
        if config_path and os.path.exists(config_path):
            try:
                config = torch.load(config_path)
                n_mels = config.get("in_channels", 80)
            except Exception as e:
                print(f"加载配置失败: {str(e)}，使用默认配置")
                n_mels = 80
        else:
            n_mels = 80
        
        # 创建模型实例
        model = cls(n_mels=n_mels)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            try:
                # 尝试不使用weights_only参数
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            except TypeError:
                # 如果出现TypeError，可能是因为PyTorch版本需要weights_only参数
                state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                
            model.load_state_dict(state_dict)
            print(f"成功加载预训练声码器: {model_path}")
        else:
            print(f"找不到预训练模型: {model_path}，使用随机初始化的权重")
        
        return model 