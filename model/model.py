import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        x = self.input_projection(x)
        
        # Transformer处理
        x = x.transpose(0, 1)  # [sequence_length, batch_size, hidden_dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch_size, sequence_length, hidden_dim]
        
        # 全局特征
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, sequence_length]
        x = self.global_pool(x)  # [batch_size, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_dim]
        
        return x

class SpeechDecoder(nn.Module):
    def __init__(self, output_dim=80, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, tgt, memory):
        # tgt: [batch_size, target_length, hidden_dim]
        # memory: [batch_size, hidden_dim]
        
        # 扩展memory维度以匹配目标序列长度
        memory = memory.unsqueeze(1).repeat(1, tgt.size(1), 1)
        
        # Transformer处理
        tgt = tgt.transpose(0, 1)  # [target_length, batch_size, hidden_dim]
        memory = memory.transpose(0, 1)  # [target_length, batch_size, hidden_dim]
        
        output = self.transformer(tgt, memory)
        output = output.transpose(0, 1)  # [batch_size, target_length, hidden_dim]
        
        # 输出投影
        output = self.output_projection(output)
        
        return output

class VoiceCloneModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512):
        super().__init__()
        self.encoder = SpeechEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = SpeechDecoder(output_dim=input_dim, hidden_dim=hidden_dim)
        
    def forward(self, source, target):
        # 编码源语音
        source_embedding = self.encoder(source)
        
        # 解码目标语音
        output = self.decoder(target, source_embedding)
        
        return output

    def clone_voice(self, source, target_length):
        # 用于推理时的语音克隆
        with torch.no_grad():
            # 获取源语音的编码
            source_embedding = self.encoder(source)
            
            # 生成初始目标序列
            device = source.device
            batch_size = source.size(0)
            tgt = torch.zeros(batch_size, target_length, self.decoder.hidden_dim).to(device)
            
            # 自回归生成
            for i in range(target_length):
                output = self.decoder(tgt[:,:i+1], source_embedding)
                if i < target_length - 1:
                    tgt[:, i+1] = output[:, -1]
            
            return output