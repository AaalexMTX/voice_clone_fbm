#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试X-Vector模型能否正确加载预训练权重
"""

import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

from model.speaker_encoder.xvector import XVectorEncoder, SpeechBrainAdapter
from model.speaker_encoder.audio_processing import preprocess_wav
from model.speaker_encoder.mel_features import extract_mel_features

def test_direct_loading():
    """测试直接加载预训练权重"""
    print("\n=== 测试直接加载预训练权重 ===")
    
    # 创建模型
    model = XVectorEncoder(mel_n_channels=24, embedding_dim=512)
    
    # 加载预训练权重
    model_path = "model/data/checkpoints/speaker_encoder/xvector.ckpt"
    
    try:
        weights = torch.load(model_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)
        print("✓ 成功加载预训练权重")
        
        # 打印模型参数
        print("\n模型参数:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}, 均值: {param.mean().item():.4f}, 标准差: {param.std().item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 加载预训练权重失败: {str(e)}")
        return False

def test_adapter_loading():
    """测试通过适配器加载预训练权重"""
    print("\n=== 测试通过适配器加载预训练权重 ===")
    
    # 创建适配器
    model_path = "model/data/checkpoints/speaker_encoder/xvector.ckpt"
    
    try:
        adapter = SpeechBrainAdapter(model_path=model_path)
        print("✓ 成功创建适配器并加载预训练权重")
        
        # 测试提取嵌入向量
        audio_file = "data/root/source/dingzhen_8.wav"
        
        if Path(audio_file).exists():
            # 预处理音频
            wav = preprocess_wav(audio_file)
            
            # 提取嵌入向量
            embedding = adapter.embed_utterance(wav)
            
            print(f"✓ 成功提取嵌入向量，维度: {embedding.shape}")
            print(f"  嵌入向量范数: {np.linalg.norm(embedding)}")
            print(f"  嵌入向量均值: {np.mean(embedding)}")
            print(f"  嵌入向量标准差: {np.std(embedding)}")
            
            # 保存嵌入向量
            np.save("test_outputs/fixed_speaker_embedding.npy", embedding)
            print(f"已保存嵌入向量到: test_outputs/fixed_speaker_embedding.npy")
        else:
            print(f"✗ 测试音频文件不存在: {audio_file}")
        
        return True
    except Exception as e:
        print(f"✗ 通过适配器加载预训练权重失败: {str(e)}")
        return False

def compare_embeddings():
    """比较修复前后的嵌入向量"""
    print("\n=== 比较修复前后的嵌入向量 ===")
    
    # 创建输出目录
    Path("test_outputs").mkdir(exist_ok=True)
    
    # 尝试加载修复前的嵌入向量
    try:
        old_embedding = np.load("test_outputs/speaker_embedding.npy")
        print(f"✓ 成功加载修复前的嵌入向量，维度: {old_embedding.shape}")
    except Exception as e:
        print(f"✗ 加载修复前的嵌入向量失败: {str(e)}")
        old_embedding = None
    
    # 尝试加载修复后的嵌入向量
    try:
        new_embedding = np.load("test_outputs/fixed_speaker_embedding.npy")
        print(f"✓ 成功加载修复后的嵌入向量，维度: {new_embedding.shape}")
    except Exception as e:
        print(f"✗ 加载修复后的嵌入向量失败: {str(e)}")
        new_embedding = None
    
    # 如果两个嵌入向量都存在，比较它们
    if old_embedding is not None and new_embedding is not None:
        # 计算余弦相似度
        cos_sim = np.dot(old_embedding, new_embedding) / (np.linalg.norm(old_embedding) * np.linalg.norm(new_embedding))
        print(f"余弦相似度: {cos_sim:.4f}")
        
        # 计算欧氏距离
        euclidean_dist = np.linalg.norm(old_embedding - new_embedding)
        print(f"欧氏距离: {euclidean_dist:.4f}")
        
        # 绘制嵌入向量的对比图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(old_embedding, label='修复前')
        plt.plot(new_embedding, label='修复后')
        plt.title('嵌入向量对比')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(np.abs(old_embedding - new_embedding), 'r-', label='差异')
        plt.title('嵌入向量差异')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("test_outputs/embedding_comparison.png")
        print(f"已保存嵌入向量对比图到: test_outputs/embedding_comparison.png")
        
        return True
    else:
        print("无法比较嵌入向量，因为至少有一个嵌入向量不存在")
        return False

if __name__ == "__main__":
    print("开始测试X-Vector模型加载...")
    
    # 测试直接加载预训练权重
    test_direct_loading()
    
    # 测试通过适配器加载预训练权重
    test_adapter_loading()
    
    # 比较修复前后的嵌入向量
    compare_embeddings()
    
    print("\n测试完成！") 