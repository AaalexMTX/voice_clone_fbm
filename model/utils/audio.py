#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
import soundfile as sf
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_audio(audio_path, sr=22050, normalize=True):
    """
    预处理音频文件
    
    参数:
        audio_path: 音频文件路径
        sr: 采样率
        normalize: 是否规范化音频
    
    返回:
        预处理后的音频波形
    """
    logger.info(f"预处理音频: {audio_path}")
    
    try:
        # 加载音频
        wav, source_sr = librosa.load(audio_path, sr=None)
        
        # 重采样（如果需要）
        if source_sr != sr:
            wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sr)
        
        # 规范化
        if normalize:
            wav = wav / np.max(np.abs(wav))
        
        return wav
    except Exception as e:
        logger.error(f"音频预处理失败: {str(e)}")
        raise

def save_audio(wav, path, sr=22050):
    """
    保存音频波形到文件
    
    参数:
        wav: 音频波形
        path: 输出文件路径
        sr: 采样率
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # 保存音频
        sf.write(path, wav, sr)
        logger.info(f"音频已保存到: {path}")
    except Exception as e:
        logger.error(f"保存音频失败: {str(e)}")
        raise

def compute_mel_spectrogram(wav, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    计算梅尔频谱
    
    参数:
        wav: 音频波形
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: 梅尔滤波器数量
    
    返回:
        梅尔频谱
    """
    # 确保音频是单声道
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    
    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # 转换为对数刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def mel_spectrogram_to_audio(mel_spec_db, sr=22050, n_fft=1024, hop_length=256):
    """
    将梅尔频谱转换回音频波形（近似转换，质量有限）
    
    参数:
        mel_spec_db: 梅尔频谱（分贝刻度）
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
    
    返回:
        重建的音频波形
    """
    # 转换回功率谱
    mel_spec = librosa.db_to_power(mel_spec_db)
    
    # 使用逆梅尔变换（近似）
    wav = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return wav 