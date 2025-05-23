#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
梅尔频谱特征提取
"""

import numpy as np
import librosa

def extract_mel_features(wav, sr=16000, n_fft=512, hop_length=160, win_length=400, n_mels=24,
                         fmin=0, fmax=8000, normalize=True):
    """
    从音频波形中提取梅尔频谱特征
    
    参数:
        wav: 音频波形
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度
        n_mels: 梅尔滤波器组数量（默认为24，与预训练X-Vector模型匹配）
        fmin: 最低频率
        fmax: 最高频率
        normalize: 是否规范化特征
        
    返回:
        梅尔频谱特征 [seq_len, n_mels]
    """
    # 确保音频是单声道
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    
    # 提取梅尔频谱
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # 转换为对数刻度
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 规范化
    if normalize:
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / (log_mel_spectrogram.std() + 1e-5)
    
    # 转置为 [seq_len, n_mels]
    log_mel_spectrogram = log_mel_spectrogram.T
    
    return log_mel_spectrogram 