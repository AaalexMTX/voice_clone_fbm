#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查X-Vector预训练权重的结构
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

def inspect_weights(model_path):
    """检查模型权重的结构"""
    print(f"加载模型权重: {model_path}")
    weights = torch.load(model_path, map_location="cpu")
    
    print(f"权重类型: {type(weights)}")
    
    if isinstance(weights, dict):
        print(f"权重键数量: {len(weights)}")
        print("\n所有权重键:")
        for i, key in enumerate(weights.keys()):
            shape = weights[key].shape if hasattr(weights[key], 'shape') else 'N/A'
            print(f"{i+1}. {key}: {shape}")
            
        # 分析权重结构
        print("\n权重结构分析:")
        
        # 查找卷积层
        conv_layers = [k for k in weights.keys() if 'conv' in k]
        print(f"卷积层数量: {len(conv_layers)}")
        for k in conv_layers:
            print(f"  {k}: {weights[k].shape}")
        
        # 查找批归一化层
        bn_layers = [k for k in weights.keys() if 'norm' in k or 'bn' in k]
        print(f"批归一化层数量: {len(bn_layers)}")
        for k in bn_layers:
            shape = weights[k].shape if hasattr(weights[k], 'shape') else 'N/A'
            print(f"  {k}: {shape}")
        
        # 查找线性层
        linear_layers = [k for k in weights.keys() if 'linear' in k or 'w' in k]
        print(f"线性层数量: {len(linear_layers)}")
        for k in linear_layers:
            print(f"  {k}: {weights[k].shape}")
    else:
        print("权重不是字典类型，无法进一步分析")

def create_matching_model():
    """创建匹配预训练权重结构的模型"""
    from model.speaker_encoder.xvector import XVectorEncoder
    
    print("\n创建模型实例...")
    model = XVectorEncoder(mel_n_channels=24, embedding_dim=512)
    
    print("模型结构:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    return model

def try_load_weights(model, model_path):
    """尝试将权重加载到模型中"""
    print(f"\n尝试加载权重到模型: {model_path}")
    weights = torch.load(model_path, map_location="cpu")
    
    try:
        model.load_state_dict(weights)
        print("✓ 成功加载权重！")
        return True
    except Exception as e:
        print(f"✗ 加载权重失败: {str(e)}")
        
        # 分析不匹配的键
        model_keys = set(model.state_dict().keys())
        weights_keys = set(weights.keys())
        
        print("\n缺失的键 (在模型中但不在权重中):")
        for key in model_keys - weights_keys:
            print(f"  {key}: {model.state_dict()[key].shape}")
        
        print("\n多余的键 (在权重中但不在模型中):")
        for key in weights_keys - model_keys:
            shape = weights[key].shape if hasattr(weights[key], 'shape') else 'N/A'
            print(f"  {key}: {shape}")
        
        return False

if __name__ == "__main__":
    model_path = "model/data/checkpoints/speaker_encoder/xvector.ckpt"
    
    # 检查权重结构
    inspect_weights(model_path)
    
    # 创建模型
    model = create_matching_model()
    
    # 尝试加载权重
    try_load_weights(model, model_path) 