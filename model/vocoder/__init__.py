"""
声码器模块，用于将梅尔频谱转换为波形
"""

from .hifigan import HiFiGAN
from .vocoder_base import Vocoder, VocoderType

__all__ = ['HiFiGAN', 'Vocoder', 'VocoderType'] 