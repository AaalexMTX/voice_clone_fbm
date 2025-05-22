import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 注意力加权
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out(out)
        
        return out

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class MelDecoder(nn.Module):
    """
    梅尔解码器，将文本特征和说话人嵌入转换为梅尔频谱
    """
    def __init__(self, config):
        super(MelDecoder, self).__init__()
        
        # 从配置中读取参数
        self.d_model = getattr(config.model, "decoder_dim", 512)
        self.text_dim = getattr(config.model, "text_embed_dim", 512)
        self.speaker_dim = getattr(config.model, "speaker_embedding_dim", 256)
        self.n_mels = getattr(config.model, "n_mels", 80)
        self.num_layers = getattr(config.model, "decoder_layers", 4)
        self.num_heads = getattr(config.model, "decoder_heads", 8)
        self.d_ff = getattr(config.model, "decoder_ff_dim", 2048)
        self.dropout = getattr(config.model, "decoder_dropout", 0.1)
        
        # 文本和说话人嵌入融合层
        self.text_proj = nn.Linear(self.text_dim, self.d_model)
        self.speaker_proj = nn.Linear(self.speaker_dim, self.d_model)
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.out = nn.Linear(self.d_model, self.n_mels)
        
    def forward(self, text_encoding, speaker_embedding):
        """
        前向传播
        
        参数:
            text_encoding: [batch_size, seq_len, text_dim]的文本特征
            speaker_embedding: [batch_size, speaker_dim]的说话人嵌入
            
        返回:
            [batch_size, seq_len, n_mels]的梅尔频谱
        """
        batch_size, seq_len = text_encoding.size(0), text_encoding.size(1)
        
        # 投影文本特征
        text_proj = self.text_proj(text_encoding)  # [batch, seq_len, d_model]
        
        # 扩展说话人嵌入
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, speaker_dim]
        speaker_proj = self.speaker_proj(speaker_expanded)  # [batch, seq_len, d_model]
        
        # 融合文本和说话人特征
        x = text_proj + speaker_proj
        
        # 解码器层
        for layer in self.decoder_layers:
            x = layer(x, text_encoding)
        
        # 输出层
        mel_output = self.out(x)  # [batch, seq_len, n_mels]
        
        return mel_output
    
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
                print(f"成功加载预训练梅尔解码器: {pretrained_path}")
            except Exception as e:
                print(f"加载预训练梅尔解码器失败: {str(e)}") 