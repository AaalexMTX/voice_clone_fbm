#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于Transformer的TTS模型演示脚本

演示如何使用基于Transformer的TTS模型进行语音克隆。
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

# 确保能导入模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from model import VoiceCloneSystem
from model.speaker_encoder.audio_processing import preprocess_wav

def plot_mel(mel, title=None, save_path=None):
    """
    绘制梅尔频谱图
    
    参数:
        mel: 梅尔频谱
        title: 标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.T, aspect='auto', origin='lower')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于Transformer的TTS模型演示")
    
    parser.add_argument("--reference_audio", required=True, help="参考音频文件路径")
    parser.add_argument("--text", required=True, help="要合成的文本")
    parser.add_argument("--output_dir", default="model/data/outputs", help="输出目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--model_dir", default="model/data/checkpoints", help="模型目录")
    parser.add_argument("--vocoder_type", default="hifigan", choices=["hifigan", "griffinlim"], help="声码器类型")
    parser.add_argument("--encoder_type", default="xvector", choices=["xvector", "speaker_encoder"], help="说话人编码器类型")
    parser.add_argument("--tts_type", default="transformer", choices=["transformer", "default"], help="TTS模型类型")
    parser.add_argument("--visualize", action="store_true", help="是否可视化梅尔频谱")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化语音克隆系统
    system = VoiceCloneSystem(
        model_dir=args.model_dir,
        device=args.device,
        vocoder_type=args.vocoder_type,
        encoder_type=args.encoder_type,
        tts_type=args.tts_type
    )
    
    # 输出文件路径
    output_path = os.path.join(args.output_dir, "synthesized_speech.wav")
    
    # 克隆语音
    system.clone_voice(args.text, args.reference_audio, output_path)
    
    print(f"合成的语音已保存到: {output_path}")
    
    # 可视化（可选）
    if args.visualize:
        # 预处理参考音频
        ref_wav = preprocess_wav(args.reference_audio)
        
        # 提取说话人嵌入
        speaker_embedding = system.extract_speaker_embedding(args.reference_audio)
        
        # 将文本转换为ID序列
        text_sequence = torch.tensor([[ord(c) % 256 for c in args.text]], dtype=torch.long).to(args.device)
        
        # 生成梅尔频谱
        with torch.no_grad():
            if args.tts_type == "transformer":
                mel_outputs = system.tts_model.inference(text_sequence, torch.tensor(speaker_embedding).unsqueeze(0).to(args.device))
            else:
                mel_outputs = system.tts_model(text_sequence, torch.tensor(speaker_embedding).unsqueeze(0).to(args.device))
        
        # 可视化梅尔频谱
        mel_output = mel_outputs[0].cpu().numpy()
        plot_mel(
            mel_output,
            title="Generated Mel Spectrogram",
            save_path=os.path.join(args.output_dir, "mel_spectrogram.png")
        )
        
        print(f"梅尔频谱图已保存到: {os.path.join(args.output_dir, 'mel_spectrogram.png')}")

if __name__ == "__main__":
    main() 