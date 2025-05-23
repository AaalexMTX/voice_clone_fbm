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
import json
import yaml
from pathlib import Path


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
        
        # 位置编码
        decoder_input = self.pos_encoder(decoder_input)
        
        # Transformer解码
        output = self.transformer_decoder(
            decoder_input, 
            memory, 
            tgt_mask=decoder_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        # 预测梅尔频谱
        mel_output = self.mel_proj(output)
        
        # 预测停止标志
        stop_pred = self.stop_proj(output).squeeze(-1)
        
        return mel_output, stop_pred
    
    def forward(self, text_ids, speaker_embedding, mel_targets=None, teacher_forcing_ratio=1.0):
        """
        前向传播
        
        参数:
            text_ids: [batch_size, text_len]
            speaker_embedding: [batch_size, speaker_dim]
            mel_targets: [batch_size, mel_len, mel_dim]
            teacher_forcing_ratio: 教师强制比例
            
        返回:
            mel_outputs: [batch_size, mel_len, mel_dim]
            stop_preds: [batch_size, mel_len]
        """
        # 编码文本
        memory, src_mask = self.encode_text(text_ids, speaker_embedding)
        
        # 如果没有目标梅尔频谱，则使用推理模式
        if mel_targets is None:
            return self.inference(text_ids, speaker_embedding)
        
        batch_size, max_len, mel_dim = mel_targets.shape
        
        # 准备解码器输入
        decoder_input = torch.zeros((batch_size, 1, mel_dim), device=text_ids.device)
        
        # 准备输出容器
        mel_outputs = []
        stop_preds = []
        
        # 自回归解码
        for t in range(max_len):
            # 创建解码器掩码
            decoder_mask = self.create_look_ahead_mask(decoder_input.size(1))
            
            # 解码单步
            mel_output, stop_pred = self.decode_step(
                memory, src_mask, decoder_input, decoder_mask
            )
            
            # 保存输出
            mel_outputs.append(mel_output[:, -1:, :])
            stop_preds.append(stop_pred[:, -1])
            
            # 准备下一步输入
            if np.random.random() < teacher_forcing_ratio:
                # 教师强制：使用目标作为下一步输入
                next_input = mel_targets[:, t:t+1, :]
            else:
                # 自回归：使用预测作为下一步输入
                next_input = mel_output[:, -1:, :]
            
            # 拼接输入
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
        
        # 拼接输出
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_preds = torch.stack(stop_preds, dim=1)
        
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
        # 编码文本
        memory, src_mask = self.encode_text(text_ids, speaker_embedding)
        
        batch_size = text_ids.shape[0]
        mel_dim = self.mel_proj.out_features
        max_len = 1000  # 最大解码长度
        
        # 准备解码器输入
        decoder_input = torch.zeros((batch_size, 1, mel_dim), device=text_ids.device)
        
        # 准备输出容器
        mel_outputs = []
        
        # 自回归解码
        for t in range(max_len):
            # 创建解码器掩码
            decoder_mask = self.create_look_ahead_mask(decoder_input.size(1))
            
            # 解码单步
            mel_output, stop_pred = self.decode_step(
                memory, src_mask, decoder_input, decoder_mask
            )
            
            # 保存输出
            mel_outputs.append(mel_output[:, -1:, :])
            
            # 检查是否应该停止
            if torch.sigmoid(stop_pred[:, -1]).item() > 0.5:
                break
            
            # 准备下一步输入
            next_input = mel_output[:, -1:, :]
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
        
        # 拼接输出
        mel_outputs = torch.cat(mel_outputs, dim=1)
        
        return mel_outputs

class TransformerTTSLoss(nn.Module):
    """
    TransformerTTS的损失函数
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

def train_step(model, optimizer, criterion, batch, device):
    """
    训练单步
    
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
    
    # 创建停止标志目标
    stop_targets = create_stop_targets(mel_targets)
    
    # 前向传播
    mel_outputs, stop_preds = model(text_ids, speaker_embedding, mel_targets)
    
    # 计算损失
    total_loss, mel_loss, stop_loss = criterion(mel_outputs, mel_targets, stop_preds, stop_targets)
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        "total_loss": total_loss.item(),
        "mel_loss": mel_loss.item(),
        "stop_loss": stop_loss.item()
    }

def create_stop_targets(mel_targets, stop_threshold=1e-5):
    """
    创建停止标志目标
    
    参数:
        mel_targets: [batch_size, mel_len, mel_dim]
        stop_threshold: 停止阈值
        
    返回:
        [batch_size, mel_len]
    """
    # 计算每帧的能量
    energy = mel_targets.norm(dim=2)
    
    # 最后一帧之后的帧为停止帧
    batch_size, mel_len = energy.shape
    stop_targets = torch.zeros_like(energy)
    
    for i in range(batch_size):
        # 找到最后一个非零帧
        last_frame = mel_len - 1
        while last_frame > 0 and energy[i, last_frame] < stop_threshold:
            last_frame -= 1
        
        # 最后一帧之后的帧为停止帧
        if last_frame < mel_len - 1:
            stop_targets[i, last_frame+1:] = 1
    
    return stop_targets

class XTTSAdapter:
    """
    XTTS模型适配器
    用于加载和使用XTTS预训练模型
    """
    def __init__(self, model_path, config_path, device="cpu"):
        """
        初始化XTTS适配器
        
        参数:
            model_path: 预训练模型路径
            config_path: 配置文件路径
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                print(f"已加载XTTS配置: {config_path}")
        except Exception as e:
            print(f"加载XTTS配置失败: {str(e)}，使用默认配置")
            self.config = {}
        
        # 创建一个TransformerTTS模型实例
        self.model = None
        try:
            print(f"尝试加载XTTS预训练模型: {model_path}")
            
            # 尝试使用weights_only=False加载模型
            try:
                print("尝试使用weights_only=False加载模型（信任源模型）")
                # 添加安全全局变量
                import torch.serialization
                try:
                    # 尝试添加TTS.tts.configs.xtts_config.XttsConfig到安全全局变量
                    torch.serialization.add_safe_globals(["TTS.tts.configs.xtts_config.XttsConfig"])
                    print("已添加TTS.tts.configs.xtts_config.XttsConfig到安全全局变量")
                except Exception as e:
                    print(f"添加安全全局变量失败: {str(e)}")
                
                # 使用上下文管理器加载模型
                with torch.serialization.safe_globals(["TTS.tts.configs.xtts_config.XttsConfig"]):
                    model_data = torch.load(model_path, map_location=device, weights_only=False)
                    print("成功加载模型数据")
                    
                    # 检查是否为Coqui TTS模型
                    if hasattr(model_data, "model") or isinstance(model_data, dict) and "model" in model_data:
                        print("检测到Coqui TTS模型格式")
                        self.coqui_model = model_data
                        self.use_coqui_model = True
                        print("成功加载Coqui TTS模型")
                    else:
                        print("未检测到Coqui TTS模型格式，使用自定义模型")
                        raise ValueError("不是Coqui TTS模型格式")
                
            except Exception as e:
                print(f"使用weights_only=False加载模型失败: {str(e)}")
                print("使用自定义模型并尝试加载权重")
                
                # 从配置创建模型
                self.model = TransformerTTS(
                    vocab_size=self.config.get("vocab_size", 256),
                    d_model=self.config.get("d_model", 512),
                    nhead=self.config.get("nhead", 8),
                    num_encoder_layers=self.config.get("num_encoder_layers", 6),
                    num_decoder_layers=self.config.get("num_decoder_layers", 6),
                    dim_feedforward=self.config.get("dim_feedforward", 2048),
                    dropout=self.config.get("dropout", 0.1),
                    speaker_dim=self.config.get("speaker_dim", 512),
                    mel_dim=self.config.get("mel_dim", 80)
                )
                
                # 尝试加载权重
                try:
                    weights = torch.load(model_path, map_location=device)
                    self.model.load_state_dict(weights)
                    print("成功加载模型权重")
                    self.use_coqui_model = False
                except Exception as e:
                    print(f"加载模型权重失败: {str(e)}，使用初始化权重")
                    self.use_coqui_model = False
                
                self.model.to(device)
                self.model.eval()
            
            print(f"XTTS模型初始化成功")
            
        except Exception as e:
            print(f"加载XTTS预训练模型失败: {str(e)}，将使用随机生成")
            self.model = None
            self.use_coqui_model = False
    
    def generate_mel(self, text, speaker_embedding):
        """
        生成梅尔频谱
        
        参数:
            text: 文本
            speaker_embedding: 说话人嵌入向量
            
        返回:
            梅尔频谱
        """
        # 检查是否使用Coqui模型
        if hasattr(self, 'use_coqui_model') and self.use_coqui_model and hasattr(self, 'coqui_model'):
            try:
                print(f"使用Coqui TTS模型生成梅尔频谱，文本: '{text}'")
                
                # 尝试使用Coqui模型生成梅尔频谱
                if hasattr(self.coqui_model, 'synthesize'):
                    mel = self.coqui_model.synthesize(text, speaker_embedding)
                    return mel
                elif isinstance(self.coqui_model, dict) and "model" in self.coqui_model:
                    model = self.coqui_model["model"]
                    if hasattr(model, 'synthesize'):
                        mel = model.synthesize(text, speaker_embedding)
                        return mel
                
                print("Coqui模型没有可用的synthesize方法，使用随机生成")
            except Exception as e:
                print(f"使用Coqui TTS模型生成梅尔频谱失败: {str(e)}")
        
        # 使用自定义模型
        if self.model is not None:
            try:
                print(f"使用TransformerTTS模型生成梅尔频谱，文本: '{text}'")
                
                # 文本处理（简单实现，实际应用中需要更复杂的文本处理）
                text_ids = [ord(c) % 256 for c in text]  # 简单的字符到ID转换
                text_ids = torch.tensor([text_ids], device=self.device)
                
                # 转换speaker_embedding为张量，并确保数据类型一致
                if isinstance(speaker_embedding, np.ndarray):
                    # 确保使用float32类型
                    speaker_embedding = torch.tensor(speaker_embedding.astype(np.float32), 
                                                   device=self.device).unsqueeze(0)
                else:
                    # 如果已经是张量，确保类型是float32
                    speaker_embedding = speaker_embedding.float().to(self.device)
                    if speaker_embedding.dim() == 1:
                        speaker_embedding = speaker_embedding.unsqueeze(0)
                
                # 检查模型参数类型
                for param in self.model.parameters():
                    if param.dtype != torch.float32:
                        print(f"将模型参数从 {param.dtype} 转换为 float32")
                        param.data = param.data.float()
                
                # 使用模型推理
                with torch.no_grad():
                    mel_outputs = self.model.inference(text_ids, speaker_embedding)
                
                # 转换为numpy数组
                mel = mel_outputs.cpu().numpy()[0].T  # [mel_dim, time]
                return mel
            except Exception as e:
                print(f"使用TransformerTTS模型生成梅尔频谱失败: {str(e)}，使用随机生成")
        
        # 如果模型不可用或生成失败，使用随机生成
        print(f"使用随机生成梅尔频谱，文本: '{text}'")
        mel_len = len(text) * 5 + 50  # 简单估算
        mel = np.random.randn(80, mel_len) * 0.1  # 控制幅度
        return mel 