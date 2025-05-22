import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供位置信息
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
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
        return x

class TextEncoder(nn.Module):
    """
    文本编码器，将文本转换为特征表示
    """
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        
        # 从配置中读取参数
        self.vocab_size = getattr(config.model, "vocab_size", 256)
        self.embed_dim = getattr(config.model, "text_embed_dim", 512)
        self.hidden_dim = getattr(config.model, "encoder_hidden_dim", 512)
        self.num_layers = getattr(config.model, "encoder_layers", 4)
        self.num_heads = getattr(config.model, "encoder_heads", 8)
        self.dropout = getattr(config.model, "encoder_dropout", 0.1)
        
        # 嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
    def forward(self, text_indices):
        """
        前向传播
        
        参数:
            text_indices: [batch_size, seq_len]的文本索引
            
        返回:
            [batch_size, seq_len, embed_dim]的文本特征表示
        """
        # 生成mask以忽略padding（如果有的话）
        padding_mask = (text_indices == 0)
        
        # 嵌入层
        embedded = self.embedding(text_indices)
        
        # 位置编码
        embedded = self.pos_encoder(embedded)
        
        # Transformer编码器
        output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        return output
    
    def load_pretrained(self, pretrained_path: Optional[str] = None):
        """
        加载预训练模型权重
        
        参数:
            pretrained_path: 预训练模型路径
        """
        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict)
                print(f"成功加载预训练文本编码器: {pretrained_path}")
            except Exception as e:
                print(f"加载预训练文本编码器失败: {str(e)}") 