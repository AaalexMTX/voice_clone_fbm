#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试模型输出质量
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

def plot_mel(mel, title=None, save_path=None):
    """绘制梅尔频谱图"""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.colorbar()
    plt.tight_layout()
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"已保存梅尔频谱图到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def test_speaker_encoder():
    """测试说话人编码器"""
    print("\n=== 测试说话人编码器 ===")
    
    from model.speaker_encoder.xvector import XVectorEncoder, SpeechBrainAdapter
    from model.speaker_encoder.audio_processing import preprocess_wav
    
    # 测试音频文件
    audio_file = "data/root/source/dingzhen_8.wav"
    
    if not os.path.exists(audio_file):
        print(f"✗ 测试音频文件不存在: {audio_file}")
        return False
    
    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)
    
    # 预处理音频
    wav = preprocess_wav(audio_file)
    
    # 测试SpeechBrainAdapter
    try:
        # 使用预训练模型路径
        model_path = "model/data/checkpoints/speaker_encoder/xvector.ckpt"
        
        # 创建适配器
        adapter = SpeechBrainAdapter(model_path=model_path)
        
        # 提取嵌入向量
        embedding = adapter.embed_utterance(wav)
        
        print(f"✓ 成功提取说话人特征，维度: {embedding.shape}")
        print(f"- 嵌入向量范数: {np.linalg.norm(embedding)}")
        print(f"- 嵌入向量均值: {np.mean(embedding)}")
        print(f"- 嵌入向量标准差: {np.std(embedding)}")
        
        # 保存嵌入向量
        np.save("test_outputs/speaker_embedding.npy", embedding)
        print(f"已保存说话人嵌入向量到: test_outputs/speaker_embedding.npy")
        
        return embedding
    except Exception as e:
        print(f"✗ 测试说话人编码器失败: {str(e)}")
        return None

def test_tts_model(speaker_embedding=None):
    """测试TTS模型"""
    print("\n=== 测试TTS模型 ===")
    
    from model.text_to_mel.transformer_tts import XTTSAdapter
    
    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)
    
    # 如果没有提供说话人嵌入向量，尝试加载
    if speaker_embedding is None:
        try:
            speaker_embedding = np.load("test_outputs/speaker_embedding.npy")
        except Exception as e:
            print(f"✗ 加载说话人嵌入向量失败: {str(e)}，使用随机向量")
            speaker_embedding = np.random.randn(512)
            speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)
    
    # 测试文本
    text = "这是一个测试文本，用于检查TTS模型的输出质量。"
    
    try:
        # 模型和配置路径
        model_path = "model/data/checkpoints/transformer_tts/coqui_XTTS-v2_model.pth"
        config_path = "model/data/checkpoints/transformer_tts/coqui_XTTS-v2_config.yaml"
        
        # 创建适配器
        adapter = XTTSAdapter(model_path=model_path, config_path=config_path)
        
        # 生成梅尔频谱
        mel = adapter.generate_mel(text, speaker_embedding)
        
        print(f"✓ 成功生成梅尔频谱，形状: {mel.shape}")
        print(f"- 梅尔频谱均值: {np.mean(mel)}")
        print(f"- 梅尔频谱标准差: {np.std(mel)}")
        
        # 保存梅尔频谱
        np.save("test_outputs/mel_spectrogram.npy", mel)
        print(f"已保存梅尔频谱到: test_outputs/mel_spectrogram.npy")
        
        # 绘制梅尔频谱图
        plot_mel(mel, title="生成的梅尔频谱", save_path="test_outputs/mel_spectrogram.png")
        
        return mel
    except Exception as e:
        print(f"✗ 测试TTS模型失败: {str(e)}")
        return None

def test_vocoder(mel_spectrogram=None):
    """测试声码器"""
    print("\n=== 测试声码器 ===")
    
    from model.vocoder.hifigan import HiFiGAN
    
    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)
    
    # 如果没有提供梅尔频谱，尝试加载
    if mel_spectrogram is None:
        try:
            mel_spectrogram = np.load("test_outputs/mel_spectrogram.npy")
        except Exception as e:
            print(f"✗ 加载梅尔频谱失败: {str(e)}，使用随机梅尔频谱")
            mel_spectrogram = np.random.randn(80, 100) * 0.1
    
    try:
        # 模型和配置路径
        model_path = "model/vocoder/models/hifigan_vocoder.pt"
        config_path = "model/vocoder/models/hifigan_config.json"
        
        # 加载模型
        vocoder = HiFiGAN.from_pretrained(model_path=model_path, config_path=config_path)
        
        # 将梅尔频谱转换为张量
        mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0)  # [1, mel_dim, time]
        
        # 生成音频
        with torch.no_grad():
            waveform = vocoder(mel_tensor)
        
        # 转换为numpy数组
        audio = waveform.cpu().numpy().squeeze()
        
        print(f"✓ 成功生成音频，长度: {len(audio)}")
        print(f"- 音频均值: {np.mean(audio)}")
        print(f"- 音频标准差: {np.std(audio)}")
        
        # 保存音频
        sf.write("test_outputs/generated_audio.wav", audio, 22050)
        print(f"已保存音频到: test_outputs/generated_audio.wav")
        
        return audio
    except Exception as e:
        print(f"✗ 测试声码器失败: {str(e)}")
        return None

def test_complete_pipeline():
    """测试完整的语音克隆流程"""
    print("\n=== 测试完整的语音克隆流程 ===")
    
    from model.core.voice_clone import VoiceCloneSystem
    
    # 测试音频文件
    audio_file = "data/root/source/dingzhen_8.wav"
    
    if not os.path.exists(audio_file):
        print(f"✗ 测试音频文件不存在: {audio_file}")
        return False
    
    # 测试文本
    text = "这是一个完整的语音克隆测试，用于检查系统的输出质量。"
    
    try:
        # 创建语音克隆系统
        system = VoiceCloneSystem()
        
        # 执行语音克隆
        output_path, audio = system.clone_voice(audio_file, text)
        
        print(f"✓ 成功克隆语音，输出: {output_path}")
        
        return True
    except Exception as e:
        print(f"✗ 测试完整流程失败: {str(e)}")
        return False

def test_pretrained_models():
    """测试是否能够加载预训练模型"""
    print("\n=== 检查预训练模型文件 ===")
    
    # 检查X-Vector模型
    xvector_path = "model/data/checkpoints/speaker_encoder/xvector.ckpt"
    if os.path.exists(xvector_path):
        print(f"✓ X-Vector模型文件存在: {xvector_path}")
        print(f"- 文件大小: {os.path.getsize(xvector_path) / (1024*1024):.2f} MB")
    else:
        print(f"✗ X-Vector模型文件不存在: {xvector_path}")
    
    # 检查XTTS模型
    xtts_path = "model/data/checkpoints/transformer_tts/coqui_XTTS-v2_model.pth"
    if os.path.exists(xtts_path):
        print(f"✓ XTTS模型文件存在: {xtts_path}")
        print(f"- 文件大小: {os.path.getsize(xtts_path) / (1024*1024):.2f} MB")
    else:
        print(f"✗ XTTS模型文件不存在: {xtts_path}")
    
    # 检查HiFi-GAN模型
    hifigan_path = "model/vocoder/models/hifigan_vocoder.pt"
    if os.path.exists(hifigan_path):
        print(f"✓ HiFi-GAN模型文件存在: {hifigan_path}")
        print(f"- 文件大小: {os.path.getsize(hifigan_path) / (1024*1024):.2f} MB")
    else:
        print(f"✗ HiFi-GAN模型文件不存在: {hifigan_path}")
    
    # 尝试加载每个模型
    print("\n=== 尝试加载预训练模型 ===")
    
    # 加载X-Vector模型
    try:
        import torch
        print("尝试加载X-Vector模型...")
        weights = torch.load(xvector_path, map_location="cpu")
        print(f"✓ 成功加载X-Vector模型权重")
        print(f"- 权重键数量: {len(weights)}")
        print(f"- 权重键名: {list(weights.keys())[:5]}...")
    except Exception as e:
        print(f"✗ 加载X-Vector模型失败: {str(e)}")
    
    # 加载XTTS模型
    try:
        print("\n尝试加载XTTS模型...")
        weights = torch.load(xtts_path, map_location="cpu", weights_only=True)
        print(f"✓ 成功加载XTTS模型权重")
        if isinstance(weights, dict):
            print(f"- 权重键数量: {len(weights)}")
            print(f"- 权重键名: {list(weights.keys())[:5]}...")
        else:
            print(f"- 权重类型: {type(weights)}")
    except Exception as e:
        print(f"✗ 加载XTTS模型失败: {str(e)}")
    
    # 加载HiFi-GAN模型
    try:
        print("\n尝试加载HiFi-GAN模型...")
        weights = torch.load(hifigan_path, map_location="cpu")
        print(f"✓ 成功加载HiFi-GAN模型权重")
        if isinstance(weights, dict):
            print(f"- 权重键数量: {len(weights)}")
            print(f"- 权重键名: {list(weights.keys())[:5]}...")
        else:
            print(f"- 权重类型: {type(weights)}")
    except Exception as e:
        print(f"✗ 加载HiFi-GAN模型失败: {str(e)}")

if __name__ == "__main__":
    print("开始测试模型输出...")
    
    # 测试预训练模型
    test_pretrained_models()
    
    # 测试说话人编码器
    embedding = test_speaker_encoder()
    
    # 测试TTS模型
    mel = test_tts_model(embedding)
    
    # 测试声码器
    audio = test_vocoder(mel)
    
    # 测试完整流程
    test_complete_pipeline()
    
    print("\n测试完成！") 