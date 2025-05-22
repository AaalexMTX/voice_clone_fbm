#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceCloneModel(nn.Module):
    """
    基于Transformer的语音克隆模型
    
    将文本和说话人嵌入转换为梅尔频谱
    """
    
    def __init__(self, 
                 vocab_size=100,      # 词汇表大小
                 embedding_dim=512,    # 嵌入维度
                 hidden_dim=512,       # 隐藏层维度
                 n_heads=8,            # 注意力头数
                 n_layers=6,           # Transformer层数
                 speaker_dim=256,      # 说话人嵌入维度
                 mel_dim=80,           # 梅尔频谱维度
                 max_len=1000,         # 最大序列长度
                 dropout=0.1):         # Dropout率
        super().__init__()
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # 说话人嵌入投影
        self.speaker_proj = nn.Linear(speaker_dim, embedding_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 梅尔频谱预测
        self.mel_proj = nn.Linear(embedding_dim, mel_dim)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, text_ids, speaker_embedding, mel_targets=None):
        """
        前向传播
        
        参数:
            text_ids: 文本ID序列 [batch_size, text_len]
            speaker_embedding: 说话人嵌入 [batch_size, speaker_dim]
            mel_targets: 目标梅尔频谱 (训练时使用) [batch_size, mel_len, mel_dim]
        
        返回:
            mel_outputs: 预测的梅尔频谱 [batch_size, mel_len, mel_dim]
        """
        batch_size = text_ids.size(0)
        
        # 嵌入文本
        text_embedded = self.text_embedding(text_ids)  # [batch_size, text_len, embedding_dim]
        text_embedded = self.pos_encoding(text_embedded)
        
        # 处理说话人嵌入
        speaker_proj = self.speaker_proj(speaker_embedding)  # [batch_size, embedding_dim]
        speaker_proj = speaker_proj.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # 融合说话人嵌入（简单地添加到每个位置）
        text_embedded = text_embedded + speaker_proj
        
        # Transformer编码
        memory = self.transformer_encoder(text_embedded)
        
        # 如果提供了目标梅尔频谱（训练模式）
        if mel_targets is not None:
            # 创建解码器输入（右移一个位置，开始用零填充）
            decoder_input = torch.zeros_like(mel_targets)
            decoder_input[:, 1:] = mel_targets[:, :-1]
            
            # 应用位置编码
            decoder_input = self.pos_encoding(decoder_input)
            
            # Transformer解码
            decoder_output = self.transformer_decoder(decoder_input, memory)
            
            # 预测梅尔频谱
            mel_outputs = self.mel_proj(decoder_output)
        else:
            # 推理模式 - 自回归生成
            # 简化实现，实际系统应使用更高效的推理算法
            # 这里只是生成固定长度的输出作为示例
            
            # 初始化解码器输入
            decoder_input = torch.zeros(batch_size, 1, self.mel_proj.in_features, device=text_ids.device)
            
            # 生成梅尔频谱，这里固定生成100帧
            mel_outputs = []
            for i in range(100):
                # Transformer解码
                decoder_output = self.transformer_decoder(decoder_input, memory)
                
                # 预测下一帧
                next_frame = self.mel_proj(decoder_output[:, -1:])
                mel_outputs.append(next_frame)
                
                # 更新解码器输入（注意：需要先将next_frame投影回embedding空间）
                next_frame_proj = torch.zeros(batch_size, 1, self.mel_proj.in_features, device=text_ids.device)
                decoder_input = torch.cat([decoder_input, next_frame_proj], dim=1)
            
            # 连接所有帧
            mel_outputs = torch.cat(mel_outputs, dim=1)
        
        return mel_outputs


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    给序列添加位置信息
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不作为参数）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        添加位置编码
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        返回:
            位置编码后的张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x 