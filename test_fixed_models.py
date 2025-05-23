#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的模型加载
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

def test_xvector_loading():
    """测试X-Vector模型加载"""
    print("\n=== 测试X-Vector模型加载 ===")
    
    from model.speaker_encoder.xvector import XVectorEncoder
    
    # 创建模型实例
    model = XVectorEncoder(mel_n_channels=80, embedding_dim=512)
    
    # 加载预训练权重
    model_path = "model/data/checkpoints/speaker_encoder/xvector.ckpt"
    print(f"尝试加载模型: {model_path}")
    
    try:
        weights = torch.load(model_path, map_location="cpu")
        model.load_state_dict(weights)
        print("✓ 成功加载X-Vector模型权重")
        
        # 打印模型结构
        print(f"模型结构:")
        print(f"- 块数量: {len(model.blocks)}")
        
        # 测试模型前向传播
        dummy_input = torch.randn(1, 100, 80)  # [batch_size, seq_len, n_mels]
        with torch.no_grad():
            output = model(dummy_input)
        print(f"- 输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 加载X-Vector模型失败: {str(e)}")
        return False

def test_xtts_loading():
    """测试XTTS模型加载"""
    print("\n=== 测试XTTS模型加载 ===")
    
    from model.text_to_mel.transformer_tts import XTTSAdapter
    
    # 模型和配置路径
    model_path = "model/data/checkpoints/transformer_tts/coqui_XTTS-v2_model.pth"
    config_path = "model/data/checkpoints/transformer_tts/coqui_XTTS-v2_config.yaml"
    
    print(f"尝试加载模型: {model_path}")
    print(f"配置文件: {config_path}")
    
    try:
        # 创建适配器
        adapter = XTTSAdapter(model_path=model_path, config_path=config_path, device="cpu")
        
        # 检查是否成功加载
        if hasattr(adapter, 'use_coqui_model') and adapter.use_coqui_model:
            print("✓ 成功加载Coqui TTS模型")
            return True
        elif adapter.model is not None:
            print("✓ 成功加载TransformerTTS模型")
            
            # 测试模型前向传播
            dummy_text = "这是一个测试文本"
            dummy_speaker = torch.randn(512)
            
            try:
                mel = adapter.generate_mel(dummy_text, dummy_speaker)
                print(f"- 生成的梅尔频谱形状: {mel.shape}")
                return True
            except Exception as e:
                print(f"✗ 模型前向传播失败: {str(e)}")
                return False
        else:
            print("✗ 模型加载失败，没有可用的模型")
            return False
    except Exception as e:
        print(f"✗ 加载XTTS模型失败: {str(e)}")
        return False

def test_hifigan_loading():
    """测试HiFi-GAN模型加载"""
    print("\n=== 测试HiFi-GAN模型加载 ===")
    
    from model.vocoder.hifigan import HiFiGAN
    
    # 模型和配置路径
    model_path = "model/vocoder/models/hifigan_vocoder.pt"
    config_path = "model/vocoder/models/hifigan_config.json"
    
    print(f"尝试加载模型: {model_path}")
    print(f"配置文件: {config_path}")
    
    try:
        # 加载模型
        model = HiFiGAN.from_pretrained(model_path=model_path, config_path=config_path)
        
        # 检查是否成功加载
        if model is not None:
            print("✓ 成功加载HiFi-GAN模型")
            
            # 测试模型前向传播
            dummy_mel = torch.randn(1, 80, 100)  # [batch_size, n_mels, time]
            
            try:
                with torch.no_grad():
                    output = model(dummy_mel)
                print(f"- 生成的音频形状: {output.shape}")
                return True
            except Exception as e:
                print(f"✗ 模型前向传播失败: {str(e)}")
                return False
        else:
            print("✗ 模型加载失败，返回了None")
            return False
    except Exception as e:
        print(f"✗ 加载HiFi-GAN模型失败: {str(e)}")
        return False

def test_voice_clone_system():
    """测试完整的语音克隆系统"""
    print("\n=== 测试完整的语音克隆系统 ===")
    
    from model.core.voice_clone import VoiceCloneSystem
    
    try:
        # 创建语音克隆系统
        system = VoiceCloneSystem()
        
        # 测试说话人特征提取
        print("\n测试说话人特征提取...")
        reference_audio = "data/root/source/dingzhen_8.wav"
        
        if os.path.exists(reference_audio):
            embedding = system.extract_speaker_embedding(reference_audio)
            print(f"✓ 成功提取说话人特征，维度: {embedding.shape}")
            
            # 测试梅尔频谱生成
            print("\n测试梅尔频谱生成...")
            text = "这是一个测试文本，用于验证模型是否正常工作。"
            mel = system.generate_mel_spectrogram(text, embedding)
            print(f"✓ 成功生成梅尔频谱，形状: {mel.shape}")
            
            # 测试音频生成
            print("\n测试音频生成...")
            audio = system.generate_audio(mel)
            print(f"✓ 成功生成音频，长度: {len(audio)}")
            
            # 测试完整的语音克隆流程
            print("\n测试完整的语音克隆流程...")
            output_path, audio = system.clone_voice(reference_audio, "这是一个完整的语音克隆测试。")
            print(f"✓ 成功克隆语音，输出: {output_path}")
            
            # 播放生成的音频（可选）
            try:
                import IPython.display as ipd
                print("\n尝试播放生成的音频...")
                ipd.Audio(audio, rate=16000)
                print("✓ 成功播放音频")
            except Exception as e:
                print(f"✗ 无法播放音频: {str(e)}")
            
            return True
        else:
            print(f"✗ 参考音频不存在: {reference_audio}")
            return False
    except Exception as e:
        print(f"✗ 测试语音克隆系统失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试修复后的模型...")
    
    # 测试各个模型
    xvector_success = test_xvector_loading()
    xtts_success = test_xtts_loading()
    hifigan_success = test_hifigan_loading()
    
    # 测试完整系统
    system_success = test_voice_clone_system()
    
    # 打印总结
    print("\n=== 测试结果总结 ===")
    print(f"X-Vector模型: {'✓ 成功' if xvector_success else '✗ 失败'}")
    print(f"XTTS模型: {'✓ 成功' if xtts_success else '✗ 失败'}")
    print(f"HiFi-GAN模型: {'✓ 成功' if hifigan_success else '✗ 失败'}")
    print(f"完整系统: {'✓ 成功' if system_success else '✗ 失败'}")
    
    if xvector_success and xtts_success and hifigan_success and system_success:
        print("\n所有测试都通过了！系统应该能够正常工作。")
    else:
        print("\n有些测试失败了。请检查上面的错误信息。") 