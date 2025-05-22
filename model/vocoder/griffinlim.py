#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Griffin-Lim声码器实现
基于TorchAudio的Griffin-Lim算法
"""

import os
import torch
import torch.nn as nn
import torchaudio
import json
import logging
from typing import Optional, Dict, Any

from .base import Vocoder, VocoderType

logger = logging.getLogger(__name__)

class GriffinLim(nn.Module):
    """
    基于Griffin-Lim算法的声码器
    将梅尔频谱转换为音频波形
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        sample_rate: int = 22050,
        min_frequency: float = 0.0,
        max_frequency: float = 8000.0,
        power: float = 1.0,
        normalize: bool = True,
        n_iter: int = 60,
        momentum: float = 0.99
    ):
        """
        初始化Griffin-Lim声码器
        
        参数:
            n_fft: FFT窗口大小
            hop_length: 帧移
            win_length: 窗口长度
            n_mels: 梅尔滤波器组数量
            sample_rate: 采样率
            min_frequency: 最小频率
            max_frequency: 最大频率
            power: 功率因子
            normalize: 是否归一化梅尔滤波器
            n_iter: Griffin-Lim迭代次数
            momentum: Griffin-Lim动量
        """
        super(GriffinLim, self).__init__()
        
        # 保存配置
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.power = power
        self.normalize = normalize
        self.n_iter = n_iter
        self.momentum = momentum
        self.vocoder_type = VocoderType.GRIFFINLIM
        
        # 创建梅尔到频谱变换
        self.mel_basis = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=min_frequency,
            f_max=max_frequency,
            norm="slaney" if normalize else None
        )
        
        # 创建Griffin-Lim变换
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
            n_iter=n_iter,
            momentum=momentum
        )
    
    def forward(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频波形
        
        参数:
            mel_spectrogram: [batch_size, time, n_mels]的梅尔频谱
            
        返回:
            [batch_size, 1, time*factor]的音频波形
        """
        # 保存原始批次大小
        batch_size = mel_spectrogram.size(0)
        
        # 转换输入维度 [batch, time, n_mels] -> [batch, n_mels, time]
        if mel_spectrogram.dim() == 3 and mel_spectrogram.size(2) == self.n_mels:
            mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # 将指数梅尔频谱转换为线性频谱
        # 注意：Griffin-Lim通常在线性功率谱上操作，不是对数梅尔频谱
        linear_spectrogram = self.mel_basis(mel_spectrogram)
        
        # 初始化输出张量
        waveforms = []
        
        # 对每个样本单独处理
        for i in range(batch_size):
            # 应用Griffin-Lim算法重建波形
            waveform = self.griffin_lim(linear_spectrogram[i:i+1])
            waveforms.append(waveform)
        
        # 将波形拼接为批次
        waveforms = torch.stack(waveforms, dim=0)
        
        # 添加通道维度，变为 [batch, 1, time]
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        
        return waveforms
    
    @classmethod
    def from_pretrained(cls, model_path, config_path=None):
        """
        从预训练模型加载声码器
        
        参数:
            model_path: 预训练模型路径
            config_path: 模型配置路径（可选）
            
        返回:
            加载了预训练权重的GriffinLim实例
        """
        # 加载配置（如果提供）
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 使用配置创建实例
                instance = cls(
                    n_fft=config.get("n_fft", 1024),
                    hop_length=config.get("hop_length", 256),
                    win_length=config.get("win_length", 1024),
                    n_mels=config.get("n_mels", 80),
                    sample_rate=config.get("sampling_rate", 22050),
                    min_frequency=config.get("min_frequency", 0.0),
                    max_frequency=config.get("max_frequency", 8000.0),
                    power=config.get("power", 1.0),
                    normalize=config.get("normalize", True),
                    n_iter=config.get("n_iter", 60),
                    momentum=config.get("momentum", 0.99)
                )
            except Exception as e:
                logger.error(f"加载配置失败: {str(e)}，使用默认配置")
                instance = cls()
        else:
            # 使用默认配置创建实例
            instance = cls()
        
        # 模型权重通常不需要加载，因为Griffin-Lim是基于算法的，不是基于参数的
        # 但如果提供了模型权重，可以尝试加载梅尔基础矩阵
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                
                # 尝试加载状态字典
                # 注意：这只是为了兼容性，Griffin-Lim通常不需要预训练权重
                if isinstance(state_dict, dict) and len(state_dict) > 0:
                    instance.load_state_dict(state_dict, strict=False)
                
                logger.info(f"已加载Griffin-Lim预训练权重: {model_path}")
            except Exception as e:
                logger.warning(f"加载预训练权重失败: {str(e)}，使用初始化参数")
        
        return instance 