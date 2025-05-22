import numpy as np
import librosa

def extract_mel_features(wav, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """
    从音频波形中提取梅尔频谱特征
    
    参数:
        wav: 输入音频波形, numpy数组, 归一化到[-1, 1]
        sr: 采样率, 默认16kHz
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: 梅尔滤波器组数量
        
    返回:
        梅尔频谱特征，shape为[时间帧数, n_mels]
    """
    # 计算短时傅里叶变换
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    
    # 计算功率谱
    magnitude = np.abs(stft)
    power_spec = magnitude ** 2
    
    # 转换为梅尔频谱
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_basis, power_spec)
    
    # 转换为对数尺度
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 归一化
    normalized_mel = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    
    return normalized_mel.T  # 转置为[时间, 特征]形式 