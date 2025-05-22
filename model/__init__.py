#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆系统主模块

这个模块包含了基于深度学习的语音克隆系统的核心功能：
1. 说话人编码器 - 从音频中提取说话人特征
2. 文本到梅尔频谱转换 - 将文本和说话人特征转换为梅尔频谱
3. 声码器 - 将梅尔频谱转换为高质量音频波形
"""

# 导入主要功能模块
from .speaker_encoder import SpeakerEncoder, preprocess_wav, extract_mel_features
from .text_to_mel import TextEncoder, MelDecoder 
from .vocoder import HiFiGAN, Vocoder, VocoderType

# 导入系统接口
from .core.voice_clone import VoiceCloneSystem

# 配置管理
from .config import load_config, get_config, save_config

# 不再需要导入模型模块

__version__ = "0.1.0"
__all__ = [
    # 说话人编码器
    'SpeakerEncoder', 'preprocess_wav', 'extract_mel_features',
    
    # 文本到梅尔频谱转换
    'TextEncoder', 'MelDecoder',
    
    # 声码器
    'HiFiGAN', 'Vocoder', 'VocoderType',
    
    # 系统接口
    'VoiceCloneSystem',
    'load_config',
    'get_config',
    'save_config'
] 