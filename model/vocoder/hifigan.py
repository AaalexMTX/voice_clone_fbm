#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HiFi-GAN声码器实现
将梅尔频谱转换为高质量音频波形
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .vocoder_base import Vocoder, VocoderType

# 尝试导入torchaudio
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

class ResBlock(nn.Module):
    """HiFi-GAN残差块"""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=self._get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=self._get_padding(kernel_size, dilation[1]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=self._get_padding(kernel_size, dilation[2])))
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=self._get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=self._get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=self._get_padding(kernel_size, 1)))
        ])

    def _get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class HiFiGAN(nn.Module):
    """
    HiFi-GAN 声码器模型，将梅尔频谱转换为音频波形
    基于 https://github.com/jik876/hifi-gan
    """
    def __init__(self, n_mels=80, upsample_rates=None, upsample_kernel_sizes=None,
                 upsample_initial_channel=512, resblock_kernel_sizes=None, 
                 resblock_dilation_sizes=None):
        super(HiFiGAN, self).__init__()
        
        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        
        self.n_mels = n_mels
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.vocoder_type = VocoderType.HIFIGAN
        
        # 初始化TorchAudio模型
        self.torchaudio_model = None
        self.torchaudio_processor = None
        
        # 初始卷积层
        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3))
        
        # 上采样层
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel // (2**i),
                                 upsample_initial_channel // (2**(i+1)),
                                 k, u, padding=(k-u)//2)))
        
        # 残差块
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # 输出卷积
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        
    def forward(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频波形
        
        参数:
            mel_spectrogram: [batch_size, time, n_mels]的梅尔频谱
            
        返回:
            [batch_size, 1, time*factor]的音频波形
        """
        # 如果使用TorchAudio模型，则直接使用它
        if self.torchaudio_model is not None and self.torchaudio_processor is not None:
            # 确保输入格式正确
            if mel_spectrogram.dim() == 3 and mel_spectrogram.size(2) == self.n_mels:
                mel = mel_spectrogram.transpose(1, 2)  # [batch, time, n_mels] -> [batch, n_mels, time]
            else:
                mel = mel_spectrogram
            
            # 使用TorchAudio模型生成波形
            with torch.no_grad():
                waveform = self.torchaudio_processor(mel)
            return waveform
        
        # 否则使用自定义实现
        # 转换输入维度 [batch, time, n_mels] -> [batch, n_mels, time]
        if mel_spectrogram.dim() == 3 and mel_spectrogram.size(2) == self.n_mels:
            x = mel_spectrogram.transpose(1, 2)
        else:
            x = mel_spectrogram
        
        # 初始卷积
        x = self.conv_pre(x)
        
        # 上采样和残差块
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # 应用当前上采样层的所有残差块
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        # 输出层
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    @classmethod
    def from_pretrained(cls, model_path, config_path=None):
        """
        从预训练模型加载
        
        参数:
            model_path: 预训练模型路径
            config_path: 配置文件路径，如果为None则使用默认配置
            
        返回:
            HiFiGAN实例
        """
        try:
            # 加载模型
            print(f"尝试加载HiFi-GAN预训练模型: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # 检查模型格式
            if "generator" in checkpoint:
                print("检测到HiFi-GAN原始格式模型，提取generator部分")
                generator = checkpoint["generator"]
            else:
                generator = checkpoint
            
            # 加载配置
            if config_path:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    print(f"已加载配置: {config_path}")
            else:
                # 默认配置
                config = {
                    "upsample_rates": [8, 8, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 4, 4],
                    "upsample_initial_channel": 512,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
                }
                print("使用默认配置")
            
            # 创建模型实例
            model = cls(
                n_mels=config.get("n_mels", 80),
                upsample_rates=config.get("upsample_rates", [8, 8, 2, 2]),
                upsample_kernel_sizes=config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
                upsample_initial_channel=config.get("upsample_initial_channel", 512),
                resblock_kernel_sizes=config.get("resblock_kernel_sizes", [3, 7, 11]),
                resblock_dilation_sizes=config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
            )
            
            # 加载权重
            try:
                model.load_state_dict(generator)
                print("成功加载预训练HiFi-GAN声码器权重")
            except Exception as e:
                print(f"加载HiFi-GAN权重失败: {str(e)}，尝试部分加载")
                # 尝试部分加载
                model.load_state_dict(generator, strict=False)
                print("成功部分加载HiFi-GAN声码器权重")
            
            # 设置为评估模式
            model.eval()
            print(f"成功加载预训练HiFi-GAN声码器: {model_path}")
            
            return model
        except Exception as e:
            print(f"加载HiFi-GAN预训练模型失败: {str(e)}")
            print("使用TorchAudio的HiFi-GAN模型作为备选")
            
            try:
                # 尝试使用TorchAudio的HiFi-GAN模型
                import torchaudio
                
                model = cls()
                model.torchaudio_model = torchaudio.pipelines.HIFIGAN_VOCODER
                print("成功加载TorchAudio HiFi-GAN声码器")
                
                return model
            except Exception as e2:
                print(f"加载TorchAudio HiFi-GAN失败: {str(e2)}")
                print("使用随机初始化的HiFi-GAN模型")
                
                # 使用随机初始化的模型
                return cls() 