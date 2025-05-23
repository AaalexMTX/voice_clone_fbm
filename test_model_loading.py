#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试模型加载和使用
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

from model.core.voice_clone import VoiceCloneSystem

def test_model_loading():
    """测试模型加载"""
    print("开始测试模型加载...")
    
    # 创建语音克隆系统
    system = VoiceCloneSystem()
    
    # 测试说话人特征提取
    print("\n测试说话人特征提取...")
    reference_audio = "data/root/source/dingzhen_8.wav"
    if os.path.exists(reference_audio):
        embedding = system.extract_speaker_embedding(reference_audio)
        print(f"说话人特征维度: {embedding.shape}")
        print(f"说话人特征前5个值: {embedding[:5]}")
    else:
        print(f"参考音频不存在: {reference_audio}")
    
    # 测试梅尔频谱生成
    print("\n测试梅尔频谱生成...")
    text = "这是一个测试文本，用于验证模型是否正常工作。"
    mel = system.generate_mel_spectrogram(text, embedding)
    print(f"梅尔频谱形状: {mel.shape}")
    
    # 测试音频生成
    print("\n测试音频生成...")
    audio = system.generate_audio(mel)
    print(f"音频长度: {len(audio)}")
    print(f"音频前5个值: {audio[:5]}")
    
    # 测试完整的语音克隆流程
    print("\n测试完整的语音克隆流程...")
    output_path, audio = system.clone_voice(reference_audio, "这是一个完整的语音克隆测试。")
    print(f"输出音频路径: {output_path}")
    print(f"音频长度: {len(audio)}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_model_loading() 