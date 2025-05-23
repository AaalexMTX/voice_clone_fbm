#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于Transformer的文本到梅尔频谱转换模型

该模型接收文本和说话人嵌入向量作为输入，输出梅尔频谱。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供位置信息
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，而不是模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        将位置编码添加到输入张量
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            添加了位置编码的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTTS(nn.Module):
    """
    基于Transformer的文本到梅尔频谱转换模型
    """
    def __init__(self, 
                 vocab_size=256,      # 词汇表大小
                 d_model=512,         # 模型维度
                 nhead=8,             # 注意力头数
                 num_encoder_layers=6, # 编码器层数
                 num_decoder_layers=6, # 解码器层数
                 dim_feedforward=2048, # 前馈网络维度
                 dropout=0.1,         # Dropout率
                 speaker_dim=512,     # 说话人嵌入维度
                 mel_dim=80,          # 梅尔频谱维度
                 max_seq_len=1000):   # 最大序列长度
        super(TransformerTTS, self).__init__()
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # 说话人嵌入投影
        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # 输出投影
        self.mel_proj = nn.Linear(d_model, mel_dim)
        
        # 预测停止标志
        self.stop_proj = nn.Linear(d_model, 1)
        
        # 梅尔频谱预测器
        self.mel_prenet = nn.Sequential(
            nn.Linear(mel_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 长度调节器（可选，用于非自回归生成）
        self.length_regulator = None  # 可以在这里实现长度调节
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """
        创建填充掩码
        
        参数:
            seq: 输入序列
            pad_idx: 填充索引
            
        返回:
            填充掩码
        """
        return (seq == pad_idx).to(seq.device)
    
    def create_look_ahead_mask(self, size):
        """
        创建前瞻掩码（用于解码器自注意力）
        
        参数:
            size: 序列长度
            
        返回:
            前瞻掩码
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device=next(self.parameters()).device)
    
    def encode_text(self, text_ids, speaker_embedding):
        """
        编码文本和说话人信息
        
        参数:
            text_ids: [batch_size, text_len]
            speaker_embedding: [batch_size, speaker_dim]
            
        返回:
            编码器输出
        """
        # 文本嵌入
        text_mask = self.create_padding_mask(text_ids)
        text_embedded = self.text_embedding(text_ids)  # [batch_size, text_len, d_model]
        
        # 位置编码
        text_embedded = self.pos_encoder(text_embedded)
        
        # 说话人嵌入投影
        speaker_proj = self.speaker_proj(speaker_embedding)  # [batch_size, d_model]
        speaker_proj = speaker_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 融合说话人信息
        text_embedded = text_embedded + speaker_proj
        
        # Transformer编码
        memory = self.transformer_encoder(text_embedded, src_key_padding_mask=text_mask)
        
        return memory, text_mask
    
    def decode_step(self, memory, encoder_padding_mask, decoder_input, decoder_mask=None):
        """
        解码单步
        
        参数:
            memory: 编码器输出
            encoder_padding_mask: 编码器填充掩码
            decoder_input: 解码器输入
            decoder_mask: 解码器掩码
            
        返回:
            解码器输出
        """
        # 预处理解码器输入
        decoder_input = self.mel_prenet(decoder_input)
        decoder_input = self.pos_encoder(decoder_input)
        
        # Transformer解码
        decoder_output = self.transformer_decoder(
            decoder_input, 
            memory,
            tgt_mask=decoder_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        # 预测梅尔频谱
        mel_output = self.mel_proj(decoder_output)
        
        # 预测停止标志
        stop_pred = self.stop_proj(decoder_output).squeeze(-1)
        
        return mel_output, stop_pred
    
    def forward(self, text_ids, speaker_embedding, mel_targets=None, teacher_forcing_ratio=1.0):
        """
        前向传播
        
        参数:
            text_ids: [batch_size, text_len]
            speaker_embedding: [batch_size, speaker_dim]
            mel_targets: [batch_size, mel_len, mel_dim]（训练时提供）
            teacher_forcing_ratio: 教师强制比例
            
        返回:
            mel_outputs: [batch_size, mel_len, mel_dim]
            stop_preds: [batch_size, mel_len]
            alignments: 注意力权重（可选）
        """
        batch_size = text_ids.size(0)
        device = text_ids.device
        
        # 编码文本和说话人信息
        memory, src_padding_mask = self.encode_text(text_ids, speaker_embedding)
        
        # 训练模式
        if mel_targets is not None:
            mel_len = mel_targets.size(1)
            
            # 创建解码器掩码
            tgt_mask = self.create_look_ahead_mask(mel_len)
            
            # 教师强制：使用目标梅尔频谱作为输入
            if random.random() < teacher_forcing_ratio:
                # 右移目标梅尔频谱作为解码器输入
                decoder_input = torch.zeros_like(mel_targets)
                decoder_input[:, 1:] = mel_targets[:, :-1]
                
                # 解码
                mel_outputs, stop_preds = self.decode_step(
                    memory, 
                    src_padding_mask,
                    decoder_input,
                    tgt_mask
                )
                
                return mel_outputs, stop_preds
            
        # 推理模式或不使用教师强制
        max_len = 1000 if mel_targets is None else mel_targets.size(1) * 2
        
        # 初始化解码器输入和输出
        decoder_input = torch.zeros(batch_size, 1, self.mel_proj.out_features, device=device)
        mel_outputs = []
        stop_preds = []
        
        # 自回归解码
        for i in range(max_len):
            # 解码单步
            mel_output, stop_pred = self.decode_step(
                memory,
                src_padding_mask,
                decoder_input,
                None if i == 0 else self.create_look_ahead_mask(i+1)
            )
            
            # 保存输出
            mel_outputs.append(mel_output[:, -1:])
            stop_preds.append(stop_pred[:, -1:])
            
            # 更新解码器输入
            decoder_input = torch.cat([decoder_input, mel_output[:, -1:]], dim=1)
            
            # 检查停止条件
            if mel_targets is None and torch.sigmoid(stop_pred[:, -1]).item() > 0.5:
                break
        
        # 连接所有输出
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_preds = torch.cat(stop_preds, dim=1)
        
        return mel_outputs, stop_preds
    
    def inference(self, text_ids, speaker_embedding):
        """
        推理模式
        
        参数:
            text_ids: [batch_size, text_len]
            speaker_embedding: [batch_size, speaker_dim]
            
        返回:
            mel_outputs: [batch_size, mel_len, mel_dim]
        """
        with torch.no_grad():
            mel_outputs, _ = self.forward(text_ids, speaker_embedding, None)
        return mel_outputs


# 损失函数
class TransformerTTSLoss(nn.Module):
    """
    Transformer TTS 损失函数
    """
    def __init__(self):
        super(TransformerTTSLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, mel_outputs, mel_targets, stop_preds, stop_targets):
        """
        计算损失
        
        参数:
            mel_outputs: [batch_size, mel_len, mel_dim]
            mel_targets: [batch_size, mel_len, mel_dim]
            stop_preds: [batch_size, mel_len]
            stop_targets: [batch_size, mel_len]
            
        返回:
            总损失
        """
        # 梅尔频谱损失
        mel_loss = self.mse_loss(mel_outputs, mel_targets)
        
        # 停止标志损失
        stop_loss = self.bce_loss(stop_preds, stop_targets.float())
        
        # 总损失
        total_loss = mel_loss + stop_loss
        
        return total_loss, mel_loss, stop_loss


# 用于训练的函数
import random
import time

def train_step(model, optimizer, criterion, batch, device):
    """
    单步训练
    
    参数:
        model: 模型
        optimizer: 优化器
        criterion: 损失函数
        batch: 批次数据
        device: 设备
        
    返回:
        损失
    """
    # 准备数据
    text_ids = batch["text_ids"].to(device)
    speaker_embedding = batch["speaker_embedding"].to(device)
    mel_targets = batch["mel_targets"].to(device)
    stop_targets = batch["stop_targets"].to(device)
    
    # 前向传播
    mel_outputs, stop_preds = model(text_ids, speaker_embedding, mel_targets)
    
    # 计算损失
    total_loss, mel_loss, stop_loss = criterion(mel_outputs, mel_targets, stop_preds, stop_targets)
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    optimizer.step()
    
    return total_loss.item(), mel_loss.item(), stop_loss.item()


def create_stop_targets(mel_targets, stop_threshold=1e-5):
    """
    创建停止标志目标
    
    参数:
        mel_targets: [batch_size, mel_len, mel_dim]
        stop_threshold: 停止阈值
        
    返回:
        stop_targets: [batch_size, mel_len]
    """
    # 最后一帧为1，其他为0
    batch_size, mel_len = mel_targets.size(0), mel_targets.size(1)
    stop_targets = torch.zeros(batch_size, mel_len, device=mel_targets.device)
    stop_targets[:, -1] = 1.0
    return stop_targets 