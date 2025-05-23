#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
X-Vector模型使用示例

演示如何使用X-Vector模型进行说话人嵌入提取和验证
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# 确保能导入模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.speaker_encoder import XVectorEncoder, preprocess_wav
from model import VoiceCloneSystem

def extract_embeddings(model, audio_files):
    """
    提取多个音频文件的嵌入向量
    
    参数:
        model: XVectorEncoder模型
        audio_files: 音频文件路径列表
    
    返回:
        嵌入向量列表和文件名列表
    """
    embeddings = []
    filenames = []
    
    for audio_file in audio_files:
        print(f"处理文件: {audio_file}")
        try:
            # 提取嵌入向量
            embedding = model.embed_from_file(audio_file)
            embeddings.append(embedding)
            filenames.append(Path(audio_file).stem)
        except Exception as e:
            print(f"处理文件 {audio_file} 出错: {str(e)}")
    
    return embeddings, filenames

def visualize_embeddings(embeddings, labels):
    """
    使用t-SNE可视化嵌入向量
    
    参数:
        embeddings: 嵌入向量列表 
        labels: 对应的标签
    """
    # 将嵌入向量转换为numpy数组
    X = np.array(embeddings)
    
    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], label=label)
        plt.text(X_tsne[i, 0], X_tsne[i, 1], label)
    
    plt.title('t-SNE 说话人嵌入向量可视化')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('speaker_embeddings.png')
    plt.show()

def verify_speakers(embeddings, filenames):
    """
    计算所有嵌入向量之间的相似度
    
    参数:
        embeddings: 嵌入向量列表
        filenames: 文件名列表
    """
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 计算余弦相似度
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = similarity
    
    # 可视化相似度矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('说话人嵌入向量相似度矩阵')
    plt.xticks(range(n), filenames, rotation=45)
    plt.yticks(range(n), filenames)
    plt.tight_layout()
    plt.savefig('similarity_matrix.png')
    plt.show()
    
    # 打印相似度
    print("\n说话人相似度矩阵:")
    for i in range(n):
        for j in range(i+1, n):
            print(f"{filenames[i]} 与 {filenames[j]} 的相似度: {similarity_matrix[i, j]:.4f}")

def demo_voice_clone(model_path, reference_audio, text, output_path):
    """
    演示使用X-Vector进行语音克隆
    
    参数:
        model_path: X-Vector模型路径
        reference_audio: 参考音频路径
        text: 要合成的文本
        output_path: 输出音频路径
    """
    # 初始化语音克隆系统，使用X-Vector
    system = VoiceCloneSystem(
        model_dir=str(Path(model_path).parent),
        encoder_type="xvector"
    )
    
    # 克隆语音
    system.clone_voice(text, reference_audio, output_path)
    print(f"已生成语音: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="X-Vector模型使用示例")
    
    parser.add_argument("--mode", choices=["extract", "verify", "clone"], 
                        required=True, help="运行模式: 提取嵌入向量、验证说话人、克隆语音")
    parser.add_argument("--model", type=str, required=True, 
                        help="X-Vector模型路径")
    parser.add_argument("--data_dir", type=str, 
                        help="包含多个音频文件的目录（用于extract和verify模式）")
    parser.add_argument("--reference", type=str, 
                        help="参考音频路径（用于clone模式）")
    parser.add_argument("--text", type=str, 
                        help="要合成的文本（用于clone模式）")
    parser.add_argument("--output", type=str, default="output.wav", 
                        help="输出音频路径（用于clone模式）")
    
    args = parser.parse_args()
    
    # 加载X-Vector模型
    model = XVectorEncoder(mel_n_channels=80, embedding_dim=512)
    
    try:
        if torch.__version__ >= "2.6.0":
            model.load(args.model)
        else:
            state_dict = torch.load(args.model, map_location='cpu')
            model.load_state_dict(state_dict)
        print(f"已加载模型: {args.model}")
    except Exception as e:
        print(f"加载模型出错: {str(e)}")
        print("使用未训练的模型")
    
    model.eval()
    
    if args.mode == "extract" or args.mode == "verify":
        if not args.data_dir:
            print("错误: 必须指定--data_dir")
            return
        
        # 获取所有音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(list(Path(args.data_dir).glob(f"**/*{ext}")))
        
        if not audio_files:
            print(f"错误: 在 {args.data_dir} 中未找到音频文件")
            return
        
        # 提取嵌入向量
        embeddings, filenames = extract_embeddings(model, audio_files)
        
        if args.mode == "extract":
            # 可视化嵌入向量
            visualize_embeddings(embeddings, filenames)
        else:
            # 验证说话人
            verify_speakers(embeddings, filenames)
    
    elif args.mode == "clone":
        if not args.reference or not args.text:
            print("错误: 克隆模式必须指定--reference和--text")
            return
        
        # 执行语音克隆
        demo_voice_clone(args.model, args.reference, args.text, args.output)

if __name__ == "__main__":
    main() 