import os
import sys
import glob
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils import plot_mel_spectrogram, plot_waveform, plot_spectrogram

def visualize_audio(audio_path, output_dir='visualization'):
    """
    为音频文件生成频谱图、Mel频谱图和波形图
    
    参数:
        audio_path (str): 音频文件路径
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不带扩展名）
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 设置输出路径
    mel_path = os.path.join(output_dir, f"{filename}_mel.png")
    wave_path = os.path.join(output_dir, f"{filename}_wave.png")
    spec_path = os.path.join(output_dir, f"{filename}_spec.png")
    
    print(f"处理音频文件: {audio_path}")
    
    try:
        # 生成并保存Mel频谱图
        plot_mel_spectrogram(
            audio_path=audio_path,
            save_path=mel_path,
            sr=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        print(f"✓ Mel频谱图生成成功: {mel_path}")
        
        # 生成并保存波形图
        plot_waveform(
            audio_path=audio_path,
            save_path=wave_path,
            sr=22050
        )
        print(f"✓ 波形图生成成功: {wave_path}")
        
        # 生成并保存频谱图
        plot_spectrogram(
            audio_path=audio_path,
            save_path=spec_path,
            sr=22050,
            n_fft=2048,
            hop_length=512
        )
        print(f"✓ 频谱图生成成功: {spec_path}")
        
        return True
    except Exception as e:
        print(f"处理音频文件 {audio_path} 时出错: {str(e)}")
        return False

def process_directory(audio_dir, output_dir='visualization'):
    """
    处理指定目录下的所有音频文件
    
    参数:
        audio_dir (str): 音频文件目录
        output_dir (str): 输出目录
    """
    # 支持的音频格式
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    
    # 统计成功和失败的数量
    success_count = 0
    fail_count = 0
    
    # 查找所有音频文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    
    if not audio_files:
        print(f"警告: 在 {audio_dir} 中没有找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 处理每个音频文件
    for audio_path in audio_files:
        if visualize_audio(audio_path, output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='为音频文件生成可视化图像')
    parser.add_argument('--audio_path', type=str, help='单个音频文件路径')
    parser.add_argument('--audio_dir', type=str, help='包含多个音频文件的目录')
    parser.add_argument('--output_dir', type=str, default='visualization', help='输出目录')
    
    args = parser.parse_args()
    
    if args.audio_path:
        # 处理单个文件
        visualize_audio(args.audio_path, args.output_dir)
    elif args.audio_dir:
        # 处理整个目录
        process_directory(args.audio_dir, args.output_dir)
    else:
        # 默认处理visualization/temp_audio目录
        temp_audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization', 'temp_audio')
        if os.path.exists(temp_audio_dir):
            print(f"处理默认目录: {temp_audio_dir}")
            process_directory(temp_audio_dir, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization'))
        else:
            print("错误: 未指定音频文件或目录，且默认目录不存在")
            parser.print_help() 