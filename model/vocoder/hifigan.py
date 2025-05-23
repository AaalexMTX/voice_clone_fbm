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
    def from_pretrained(cls, model_path: str = None, config_path: Optional[str] = None):
        """
        从预训练模型加载HiFi-GAN声码器
        
        参数:
            model_path: 预训练模型路径，如果为None则尝试使用TorchAudio
            config_path: 模型配置路径
            
        返回:
            加载了预训练权重的HiFiGAN实例
        """
        # 尝试使用TorchAudio的预训练模型
        if model_path is None or not os.path.exists(model_path):
            if TORCHAUDIO_AVAILABLE:
                try:
                    print("使用TorchAudio官方声码器模型")
                    model = cls()
                    
                    # 加载TorchAudio的Tacotron2模型
                    tacotron2_bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
                    
                    # 提取声码器部分
                    model.torchaudio_model = tacotron2_bundle.vocoder
                    model.torchaudio_processor = tacotron2_bundle.vocoder
                    model.n_mels = 80  # Tacotron2使用80个梅尔频带
                    
                    print("成功加载TorchAudio声码器")
                    return model
                except Exception as e:
                    print(f"加载TorchAudio声码器失败: {str(e)}，尝试使用Griffin-Lim")
                    try:
                        # 尝试使用Griffin-Lim
                        tacotron2_bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH
                        model = cls()
                        model.torchaudio_model = tacotron2_bundle.vocoder
                        model.torchaudio_processor = tacotron2_bundle.vocoder
                        model.n_mels = 80
                        print("成功加载TorchAudio Griffin-Lim声码器")
                        return model
                    except Exception as e:
                        print(f"加载TorchAudio Griffin-Lim声码器失败: {str(e)}，使用未训练的模型")
            
            if model_path is None:
                print("未指定模型路径，使用未训练的HiFi-GAN模型")
                return cls()
            else:
                raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
        
        if config_path is None or not os.path.exists(config_path):
            # 使用默认配置
            print(f"配置文件不存在: {config_path}，使用默认配置")
            model = cls()
        else:
            # 从配置文件加载
            with open(config_path) as f:
                config = json.load(f)
            
            model = cls(
                n_mels=config.get("num_mels", 80),
                upsample_rates=config.get("upsample_rates", [8, 8, 2, 2]),
                upsample_kernel_sizes=config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
                upsample_initial_channel=config.get("upsample_initial_channel", 512),
                resblock_kernel_sizes=config.get("resblock_kernel_sizes", [3, 7, 11]),
                resblock_dilation_sizes=config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
            )
        
        # 加载预训练权重
        try:
            # 尝试使用weights_only=False参数
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            print(f"加载模型失败: {str(e)}，尝试其他方法")
            try:
                # 尝试使用默认参数
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            except Exception as e:
                print(f"加载模型再次失败: {str(e)}，尝试使用TorchAudio模型")
                if TORCHAUDIO_AVAILABLE:
                    try:
                        tacotron2_bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
                        model.torchaudio_model = tacotron2_bundle.vocoder
                        model.torchaudio_processor = tacotron2_bundle.vocoder
                        model.n_mels = 80
                        print("成功加载TorchAudio声码器")
                        return model
                    except Exception:
                        try:
                            tacotron2_bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH
                            model.torchaudio_model = tacotron2_bundle.vocoder
                            model.torchaudio_processor = tacotron2_bundle.vocoder
                            model.n_mels = 80
                            print("成功加载TorchAudio Griffin-Lim声码器")
                            return model
                        except Exception as e:
                            print(f"加载TorchAudio声码器失败: {str(e)}，使用未训练的模型")
                            return cls()
                else:
                    raise
        
        # 检查模型结构，处理不同格式的模型文件
        if 'generator' in checkpoint:
            print("检测到HiFi-GAN原始格式模型，提取generator部分")
            checkpoint = checkpoint['generator']
        
        try:
            model.load_state_dict(checkpoint)
            print(f"成功加载预训练HiFi-GAN声码器: {model_path}")
        except Exception as e:
            print(f"模型状态加载失败: {str(e)}，可能是模型格式不兼容")
            # 尝试使用Griffin-Lim作为备选
            if TORCHAUDIO_AVAILABLE:
                try:
                    tacotron2_bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH
                    model.torchaudio_model = tacotron2_bundle.vocoder
                    model.torchaudio_processor = tacotron2_bundle.vocoder
                    model.n_mels = 80
                    print("无法加载HiFi-GAN模型，改用TorchAudio Griffin-Lim声码器")
                    return model
                except Exception:
                    print("所有声码器加载尝试均失败，使用未训练的模型")
        
        return model 