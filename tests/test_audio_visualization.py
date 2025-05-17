import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils import plot_mel_spectrogram, plot_waveform, plot_spectrogram


def test_audio_visualization():
    """
    测试音频可视化函数
    ~/VOICE_CLONE_FBM python tests/test_audio_visualization.py
    """
    # 音频文件路径
    audio_path = "./data/raw/dingzhen_8.wav"
    
    # 确保音频文件存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}\n请将音频文件放在 {audio_path} 位置")
    
    # 确保输出目录存在
    os.makedirs("./data/visualization", exist_ok=True)
    
    # 测试所有可视化函数
    try:
        # 生成并保存Mel频谱图
        plot_mel_spectrogram(
            audio_path=audio_path,
            save_path="./data/visualization/dingzhen_8_mel.png",
            sr=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        print("✓ Mel频谱图生成成功")
        
        # 生成并保存波形图
        plot_waveform(
            audio_path=audio_path,
            save_path="./data/visualization/dingzhen_8_wave.png",
            sr=22050
        )
        print("✓ 波形图生成成功")
        
        # 生成并保存频谱图
        plot_spectrogram(
            audio_path=audio_path,
            save_path="./data/visualization/dingzhen_8_spec.png",
            sr=22050,
            n_fft=2048,
            hop_length=512
        )
        print("✓ 频谱图生成成功")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise e

if __name__ == "__main__":
    test_audio_visualization() 