import os
import numpy as np
import librosa
import soundfile as sf

TARGET_SAMPLE_RATE = 16000

def load_wav(path, sr=TARGET_SAMPLE_RATE):
    """
    读取音频文件，并转换为指定采样率
    """
    wav, original_sr = librosa.load(path, sr=None)
    if original_sr != sr:
        wav = librosa.resample(wav, orig_sr=original_sr, target_sr=sr)
    return wav

def normalize_volume(wav, target_dBFS=-30, increase_only=False, decrease_only=False):
    """
    音量归一化为目标 dBFS（默认 -30dBFS）
    """
    rms = np.sqrt(np.mean(np.square(wav)))
    current_dBFS = 20 * np.log10(rms + 1e-6)
    dBFS_change = target_dBFS - current_dBFS

    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav

    return wav * (10 ** (dBFS_change / 20))

def trim_long_silences(wav, top_db=30):
    """
    移除音频中的静音部分（使用 librosa）
    """
    intervals = librosa.effects.split(wav, top_db=top_db)
    trimmed_wav = np.concatenate([wav[start:end] for start, end in intervals])
    return trimmed_wav

def preprocess_wav(fpath_or_wav, target_sr=16000):
    """
    加载和预处理音频，统一采样率和归一化。
    参数:
        fpath_or_wav: str 或 ndarray，可以是路径或原始波形
        target_sr: 目标采样率，默认16kHz
    返回:
        预处理后的波形，float32格式，[-1, 1]
    """
    # 判断是路径还是波形数据
    if isinstance(fpath_or_wav, str):
        # 路径字符串
        wav, sr = librosa.load(fpath_or_wav, sr=target_sr)
    else:
        # NumPy 数组说明已经加载了音频，就直接用它
        wav = fpath_or_wav
        sr = target_sr

    # 去除音频开头和结尾的静音段
    wav, _ = librosa.effects.trim(wav)

    # 将整个音频归一化到 [-1, 1] 范围
    if np.abs(wav).max() > 0:
        wav = wav / np.abs(wav).max()
    
    return wav.astype(np.float32)

def save_wav(wav, path, sr=TARGET_SAMPLE_RATE):
    """
    保存音频为 wav 文件
    """
    sf.write(path, wav, sr) 