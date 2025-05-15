import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class TransposeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TransposeConvLayer, self).__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SimpleVocoder(nn.Module):
    """
    简单的声码器模型，将梅尔频谱转换为音频波形
    """
    def __init__(self, n_mels=80, channels=512):
        super(SimpleVocoder, self).__init__()
        
        # 上采样层
        self.pre_conv = ConvLayer(n_mels, channels, kernel_size=7, stride=1)
        
        # 上采样堆栈
        self.up_stack = nn.ModuleList([
            TransposeConvLayer(channels, channels // 2, kernel_size=16, stride=8),
            TransposeConvLayer(channels // 2, channels // 4, kernel_size=16, stride=8),
            TransposeConvLayer(channels // 4, channels // 8, kernel_size=8, stride=4),
            TransposeConvLayer(channels // 8, channels // 16, kernel_size=4, stride=2),
        ])
        
        # 输出层
        self.output_layer = nn.Conv1d(channels // 16, 1, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
        
    def forward(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频波形
        
        参数:
            mel_spectrogram: [batch_size, time, n_mels]的梅尔频谱
            
        返回:
            [batch_size, 1, time*factor]的音频波形
        """
        # 转换输入维度 [batch, time, n_mels] -> [batch, n_mels, time]
        x = mel_spectrogram.transpose(1, 2)
        
        # 初始卷积
        x = self.pre_conv(x)
        
        # 上采样堆栈
        for up_layer in self.up_stack:
            x = up_layer(x)
            
        # 输出层
        x = self.output_layer(x)
        x = self.tanh(x)
        
        return x 