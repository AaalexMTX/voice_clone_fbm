#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练基于Transformer的文本到梅尔频谱转换模型
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from .transformer_tts import TransformerTTS, TransformerTTSLoss, create_stop_targets

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TTSDataset(Dataset):
    """
    TTS数据集
    """
    def __init__(self, metadata_file, text_processor=None, mel_dir=None, speaker_embed_dir=None):
        """
        初始化数据集
        
        参数:
            metadata_file: 元数据文件路径，包含文本和对应的梅尔频谱路径
            text_processor: 文本处理器，将文本转换为ID序列
            mel_dir: 梅尔频谱目录
            speaker_embed_dir: 说话人嵌入目录
        """
        self.metadata = []
        self.text_processor = text_processor
        self.mel_dir = Path(mel_dir) if mel_dir else None
        self.speaker_embed_dir = Path(speaker_embed_dir) if speaker_embed_dir else None
        
        # 加载元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    self.metadata.append({
                        'text': parts[0],
                        'mel_path': parts[1],
                        'speaker_id': parts[2],
                        'speaker_embed_path': parts[3] if len(parts) > 3 else None
                    })
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # 处理文本
        if self.text_processor:
            text_ids = self.text_processor(item['text'])
        else:
            # 简单的字符到ID映射
            text_ids = [ord(c) % 256 for c in item['text']]
        
        # 加载梅尔频谱
        mel_path = self.mel_dir / item['mel_path'] if self.mel_dir else item['mel_path']
        mel = np.load(mel_path)
        
        # 加载说话人嵌入
        if item['speaker_embed_path'] and self.speaker_embed_dir:
            speaker_embed_path = self.speaker_embed_dir / item['speaker_embed_path']
            speaker_embedding = np.load(speaker_embed_path)
        else:
            # 如果没有嵌入，使用one-hot向量
            speaker_id = int(item['speaker_id'])
            speaker_embedding = np.zeros(512)  # 假设嵌入维度为512
            speaker_embedding[speaker_id % 512] = 1.0
        
        # 创建停止标志
        stop_target = np.zeros(mel.shape[0])
        stop_target[-1] = 1.0
        
        return {
            'text': item['text'],
            'text_ids': np.array(text_ids, dtype=np.int64),
            'mel': mel.astype(np.float32),
            'speaker_embedding': speaker_embedding.astype(np.float32),
            'speaker_id': int(item['speaker_id']),
            'stop_target': stop_target.astype(np.float32)
        }


def collate_fn(batch):
    """
    批次整理函数
    
    参数:
        batch: 批次数据
        
    返回:
        整理后的批次数据
    """
    # 获取批次中的最大长度
    max_text_len = max(len(item['text_ids']) for item in batch)
    max_mel_len = max(item['mel'].shape[0] for item in batch)
    
    # 初始化批次张量
    batch_size = len(batch)
    text_ids = np.zeros((batch_size, max_text_len), dtype=np.int64)
    mels = np.zeros((batch_size, max_mel_len, batch[0]['mel'].shape[1]), dtype=np.float32)
    speaker_embeddings = np.stack([item['speaker_embedding'] for item in batch])
    stop_targets = np.zeros((batch_size, max_mel_len), dtype=np.float32)
    
    # 填充批次张量
    for i, item in enumerate(batch):
        text_len = len(item['text_ids'])
        mel_len = item['mel'].shape[0]
        
        text_ids[i, :text_len] = item['text_ids']
        mels[i, :mel_len] = item['mel']
        stop_targets[i, :mel_len] = item['stop_target']
    
    return {
        'text_ids': torch.LongTensor(text_ids),
        'mel_targets': torch.FloatTensor(mels),
        'speaker_embedding': torch.FloatTensor(speaker_embeddings),
        'stop_targets': torch.FloatTensor(stop_targets)
    }


def save_checkpoint(model, optimizer, step, checkpoint_dir, model_name="transformer_tts"):
    """
    保存检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        step: 步数
        checkpoint_dir: 检查点目录
        model_name: 模型名称
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_step{step}.pt")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, checkpoint_path)
    logger.info(f"保存检查点到: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    返回:
        步数
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['step']


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


def train(args):
    """
    训练模型
    
    参数:
        args: 命令行参数
    """
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    train_dataset = TTSDataset(
        args.train_metadata,
        mel_dir=args.mel_dir,
        speaker_embed_dir=args.speaker_embed_dir
    )
    
    val_dataset = TTSDataset(
        args.val_metadata,
        mel_dir=args.mel_dir,
        speaker_embed_dir=args.speaker_embed_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = TransformerTTS(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        speaker_dim=args.speaker_dim,
        mel_dim=args.mel_dim,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    # 创建损失函数和优化器
    criterion = TransformerTTSLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_gamma
    )
    
    # 加载检查点
    global_step = 0
    if args.checkpoint:
        global_step = load_checkpoint(model, optimizer, args.checkpoint)
        logger.info(f"从步骤 {global_step} 恢复训练")
    
    # 训练循环
    model.train()
    start_time = time.time()
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # 准备数据
            text_ids = batch['text_ids'].to(device)
            mel_targets = batch['mel_targets'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            stop_targets = batch['stop_targets'].to(device)
            
            # 前向传播
            mel_outputs, stop_preds = model(
                text_ids,
                speaker_embedding,
                mel_targets,
                teacher_forcing_ratio=args.teacher_forcing_ratio
            )
            
            # 计算损失
            total_loss, mel_loss, stop_loss = criterion(
                mel_outputs,
                mel_targets,
                stop_preds,
                stop_targets
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            scheduler.step()
            
            # 更新全局步数
            global_step += 1
            
            # 记录日志
            if global_step % args.log_step == 0:
                elapsed_time = time.time() - start_time
                logger.info(f"Step {global_step}, Loss: {total_loss.item():.4f}, "
                           f"Mel Loss: {mel_loss.item():.4f}, Stop Loss: {stop_loss.item():.4f}, "
                           f"Time: {elapsed_time:.2f}s")
                start_time = time.time()
            
            # 保存检查点
            if global_step % args.save_step == 0:
                save_checkpoint(model, optimizer, global_step, args.checkpoint_dir)
            
            # 验证
            if global_step % args.eval_step == 0:
                validate(model, val_loader, criterion, device, global_step, args)
    
    # 保存最终模型
    save_checkpoint(model, optimizer, global_step, args.checkpoint_dir, "transformer_tts_final")
    logger.info("训练完成")


def validate(model, val_loader, criterion, device, step, args):
    """
    验证模型
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        step: 步数
        args: 命令行参数
    """
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # 准备数据
            text_ids = batch['text_ids'].to(device)
            mel_targets = batch['mel_targets'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            stop_targets = batch['stop_targets'].to(device)
            
            # 前向传播
            mel_outputs, stop_preds = model(
                text_ids,
                speaker_embedding,
                mel_targets,
                teacher_forcing_ratio=1.0  # 验证时使用教师强制
            )
            
            # 计算损失
            total_loss, _, _ = criterion(
                mel_outputs,
                mel_targets,
                stop_preds,
                stop_targets
            )
            
            total_val_loss += total_loss.item()
            
            # 只保存第一个批次的样本
            if i == 0:
                # 生成梅尔频谱图
                mel_output = mel_outputs[0].cpu().numpy()
                mel_target = mel_targets[0].cpu().numpy()
                
                # 保存梅尔频谱图
                plot_mel(
                    mel_output,
                    title=f"Predicted Mel (Step {step})",
                    save_path=os.path.join(args.log_dir, f"mel_pred_step{step}.png")
                )
                
                plot_mel(
                    mel_target,
                    title=f"Ground Truth Mel (Step {step})",
                    save_path=os.path.join(args.log_dir, f"mel_true_step{step}.png")
                )
    
    # 计算平均损失
    avg_val_loss = total_val_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    # 恢复训练模式
    model.train()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练基于Transformer的TTS模型")
    
    # 数据参数
    parser.add_argument("--train_metadata", required=True, help="训练元数据文件")
    parser.add_argument("--val_metadata", required=True, help="验证元数据文件")
    parser.add_argument("--mel_dir", help="梅尔频谱目录")
    parser.add_argument("--speaker_embed_dir", help="说话人嵌入目录")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=256, help="词汇表大小")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="编码器层数")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="解码器层数")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--speaker_dim", type=int, default=512, help="说话人嵌入维度")
    parser.add_argument("--mel_dim", type=int, default=80, help="梅尔频谱维度")
    parser.add_argument("--max_seq_len", type=int, default=1000, help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")
    parser.add_argument("--lr_decay_step", type=int, default=50000, help="学习率衰减步长")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5, help="学习率衰减系数")
    parser.add_argument("--grad_clip_thresh", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0, help="教师强制比例")
    
    # 其他参数
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    parser.add_argument("--checkpoint", help="检查点路径")
    parser.add_argument("--checkpoint_dir", default="model/data/checkpoints", help="检查点目录")
    parser.add_argument("--log_dir", default="model/data/logs", help="日志目录")
    parser.add_argument("--log_step", type=int, default=100, help="日志记录步长")
    parser.add_argument("--save_step", type=int, default=1000, help="保存检查点步长")
    parser.add_argument("--eval_step", type=int, default=1000, help="验证步长")
    
    args = parser.parse_args()
    
    # 训练模型
    train(args)


if __name__ == "__main__":
    main() 