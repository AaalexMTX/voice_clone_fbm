#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用X-Vector进行说话人验证的命令行工具

该脚本用于比较两个音频文件，判断它们是否来自同一个说话人。
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine

# 导入项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.speaker_encoder import XVectorEncoder, preprocess_wav

def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度
    
    参数:
        a, b: 需要计算相似度的两个向量
        
    返回:
        相似度分数 (0~1)，1表示完全相同，0表示完全不同
    """
    return 1 - cosine(a, b)

def verify_speaker(args):
    """
    执行说话人验证
    
    参数:
        args: 命令行参数
    """
    # 加载模型
    model = XVectorEncoder(mel_n_channels=80, embedding_dim=args.embedding_dim)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件 {args.model_path} 不存在")
        return
    
    # 加载模型权重
    try:
        model.load(args.model_path)
        print(f"已加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型出错: {str(e)}")
        return
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio1):
        print(f"错误：音频文件 {args.audio1} 不存在")
        return
    
    if not os.path.exists(args.audio2):
        print(f"错误：音频文件 {args.audio2} 不存在")
        return
    
    # 提取音频嵌入向量
    try:
        print(f"正在处理第一个音频文件: {args.audio1}")
        embedding1 = model.embed_from_file(args.audio1)
        
        print(f"正在处理第二个音频文件: {args.audio2}")
        embedding2 = model.embed_from_file(args.audio2)
    except Exception as e:
        print(f"提取嵌入向量出错: {str(e)}")
        return
    
    # 计算相似度
    similarity = cosine_similarity(embedding1, embedding2)
    
    # 判断是否为同一说话人
    is_same_speaker = similarity >= args.threshold
    
    # 输出结果
    print("\n----- 说话人验证结果 -----")
    print(f"音频1: {args.audio1}")
    print(f"音频2: {args.audio2}")
    print(f"相似度分数: {similarity:.4f}")
    print(f"阈值: {args.threshold:.4f}")
    print(f"判定结果: {'同一说话人' if is_same_speaker else '不同说话人'}")
    
    # 保存嵌入向量（如果需要）
    if args.save_embeddings:
        save_dir = Path(args.save_embeddings)
        save_dir.mkdir(exist_ok=True)
        
        np.save(save_dir / "embedding1.npy", embedding1)
        np.save(save_dir / "embedding2.npy", embedding2)
        print(f"已保存嵌入向量到目录: {save_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用X-Vector进行说话人验证")
    
    parser.add_argument("--audio1", type=str, required=True, help="第一个音频文件路径")
    parser.add_argument("--audio2", type=str, required=True, help="第二个音频文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="X-Vector模型路径")
    parser.add_argument("--threshold", type=float, default=0.75, help="判定同一说话人的相似度阈值")
    parser.add_argument("--embedding_dim", type=int, default=512, help="嵌入向量维度")
    parser.add_argument("--save_embeddings", type=str, help="保存嵌入向量的目录（可选）")
    
    args = parser.parse_args()
    
    # 执行说话人验证
    verify_speaker(args)

if __name__ == "__main__":
    main() 