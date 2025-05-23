#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
X-Vector说话人编码器训练脚本

这个脚本用于训练X-Vector说话人编码器模型，从音频数据中学习说话人特征。
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
import random
import logging
import soundfile as sf
from pathlib import Path

from .xvector import XVectorEncoder
from .audio_processing import preprocess_wav
from .mel_features import extract_mel_features

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakerDataset(Dataset):
    """
    说话人数据集，用于训练X-Vector模型
    """
    def __init__(self, data_list, min_duration=3.0, max_duration=8.0, sample_rate=16000, augment=False):
        """
        初始化数据集
        
        参数:
            data_list: 包含(音频路径, 说话人ID)的列表
            min_duration: 最小音频长度（秒）
            max_duration: 最大音频长度（秒）
            sample_rate: 采样率
            augment: 是否进行数据增强
        """
        self.data_list = data_list
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.augment = augment
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        audio_path, speaker_id = self.data_list[idx]
        
        # 加载和预处理音频
        wav = preprocess_wav(audio_path)
        
        # 提取随机段
        frames = len(wav)
        min_frames = int(self.min_duration * self.sample_rate)
        max_frames = min(int(self.max_duration * self.sample_rate), frames)
        
        if max_frames > min_frames:
            # 随机选择一段
            start = random.randint(0, frames - min_frames)
            end = min(start + max_frames, frames)
            wav = wav[start:end]
        
        # 提取梅尔频谱特征
        mel_spec = extract_mel_features(wav, sr=self.sample_rate)
        
        # 数据增强（可选）
        if self.augment:
            # 添加随机噪声
            noise_level = random.uniform(0, 0.005)
            mel_spec += np.random.normal(0, noise_level, mel_spec.shape)
        
        return {
            'mel_spec': torch.FloatTensor(mel_spec),
            'speaker_id': torch.LongTensor([speaker_id])[0]
        }

def prepare_data(data_dir, train_ratio=0.9, ext="wav"):
    """
    准备训练和验证数据
    
    参数:
        data_dir: 包含说话人子目录的数据目录
        train_ratio: 训练集比例
        ext: 音频文件扩展名
    
    返回:
        train_list, val_list: 训练和验证数据列表
    """
    # 获取说话人ID映射
    speaker_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    speaker_ids = {speaker_dir.name: i for i, speaker_dir in enumerate(speaker_dirs)}
    
    all_data = []
    for speaker_dir in speaker_dirs:
        speaker_name = speaker_dir.name
        audio_files = list(speaker_dir.glob(f"*.{ext}"))
        
        for audio_file in audio_files:
            all_data.append((str(audio_file), speaker_ids[speaker_name]))
    
    # 随机打乱
    random.shuffle(all_data)
    
    # 分割训练和验证集
    train_size = int(len(all_data) * train_ratio)
    train_list = all_data[:train_size]
    val_list = all_data[train_size:]
    
    return train_list, val_list, len(speaker_ids)
    
def train_xvector(args):
    """
    训练X-Vector模型
    
    参数:
        args: 训练参数
    """
    # 准备数据
    logger.info(f"准备数据集: {args.data_dir}")
    train_list, val_list, num_speakers = prepare_data(
        args.data_dir, 
        train_ratio=args.train_ratio,
        ext=args.ext
    )
    
    logger.info(f"找到 {num_speakers} 个说话人，{len(train_list)} 个训练样本，{len(val_list)} 个验证样本")
    
    # 创建数据加载器
    train_dataset = SpeakerDataset(
        train_list, 
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        sample_rate=args.sample_rate,
        augment=args.augment
    )
    
    val_dataset = SpeakerDataset(
        val_list, 
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        sample_rate=args.sample_rate,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 初始化模型
    logger.info("初始化X-Vector模型")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = XVectorEncoder(mel_n_channels=args.mel_channels, embedding_dim=args.embedding_dim)
    
    # 添加分类头
    classification_head = nn.Linear(args.embedding_dim, num_speakers)
    
    # 将模型移动到设备
    model = model.to(device)
    classification_head = classification_head.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': classification_head.parameters()}
    ], lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    
    # 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
        
        # 训练阶段
        model.train()
        classification_head.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            # 获取批次数据
            mel_specs = batch['mel_spec'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            
            # 前向传播
            embeddings = model(mel_specs)
            logits = classification_head(embeddings)
            
            # 计算损失
            loss = criterion(logits, speaker_ids)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += speaker_ids.size(0)
            train_correct += (predicted == speaker_ids).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        classification_head.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                # 获取批次数据
                mel_specs = batch['mel_spec'].to(device)
                speaker_ids = batch['speaker_id'].to(device)
                
                # 前向传播
                embeddings = model(mel_specs)
                logits = classification_head(embeddings)
                
                # 计算损失
                loss = criterion(logits, speaker_ids)
                
                # 统计
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += speaker_ids.size(0)
                val_correct += (predicted == speaker_ids).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 记录和保存模型
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
                    f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.save_dir, "xvector_best.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存最佳模型到 {model_path}, 验证准确率: {val_acc:.2f}%")
        
        # 保存最新模型
        if (epoch + 1) % args.save_freq == 0:
            model_path = os.path.join(args.save_dir, f"xvector_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
    
    # 保存最终模型
    model_path = os.path.join(args.save_dir, "xvector_final.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"训练完成，最终模型保存到 {model_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="X-Vector说话人编码器训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录，包含说话人子目录")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--ext", type=str, default="wav", help="音频文件扩展名")
    parser.add_argument("--min_duration", type=float, default=3.0, help="最小音频长度（秒）")
    parser.add_argument("--max_duration", type=float, default=8.0, help="最大音频长度（秒）")
    parser.add_argument("--sample_rate", type=int, default=16000, help="采样率")
    parser.add_argument("--mel_channels", type=int, default=80, help="梅尔频谱通道数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--lr_step", type=int, default=20, help="学习率衰减步长")
    parser.add_argument("--no_cuda", action="store_true", help="不使用CUDA")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    parser.add_argument("--augment", action="store_true", help="使用数据增强")
    parser.add_argument("--embedding_dim", type=int, default=512, help="嵌入向量维度")
    
    # 保存参数
    parser.add_argument("--save_dir", type=str, default="models", help="模型保存目录")
    parser.add_argument("--save_freq", type=int, default=10, help="模型保存频率（轮）")
    
    args = parser.parse_args()
    
    # 训练模型
    train_xvector(args)

if __name__ == "__main__":
    main() 