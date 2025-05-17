import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，这样在没有GUI的环境下也能工作
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(audio_path, sr=22050, n_fft=1024, hop_length=256, n_mels=80, save_path=None):
    """
    生成并可视化音频的Mel频谱图
    
    参数:
        audio_path (str): 音频文件路径
        sr (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移
        n_mels (int): Mel滤波器组的数量
        save_path (str, optional): 保存图像的路径，如果为None则显示图像
    """
    # 读取音频
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 提取Mel频谱
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                            hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 创建图像
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')  # 使用深色背景
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, 
                           x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel Spectrogram', color='white')
    # 设置标签颜色为白色
    plt.xlabel('Time (s)', color='white')
    plt.ylabel('Mel Frequency', color='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#000033')
        plt.close()
    else:
        plt.show()
    plt.style.use('default')  # 恢复默认样式

def plot_waveform(audio_path, sr=22050, save_path=None):
    """
    生成并可视化音频波形图
    
    参数:
        audio_path (str): 音频文件路径
        sr (int): 采样率
        save_path (str, optional): 保存图像的路径，如果为None则显示图像
    """
    # 读取音频
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 创建图像
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(y)) / sr, y, color='blue', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_spectrogram(audio_path, sr=22050, n_fft=2048, hop_length=512, save_path=None):
    """
    生成并可视化音频的频谱图
    
    参数:
        audio_path (str): 音频文件路径
        sr (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移
        save_path (str, optional): 保存图像的路径，如果为None则显示图像
    """
    # 读取音频
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 计算频谱图
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 创建图像
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')  # 使用深色背景
    img = librosa.display.specshow(D_db, sr=sr, hop_length=hop_length,
                           x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram', color='white')
    # 设置标签颜色为白色
    plt.xlabel('Time (s)', color='white')
    plt.ylabel('Frequency (Hz)', color='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#000033')
        plt.close()
    else:
        plt.show()
    plt.style.use('default')  # 恢复默认样式 