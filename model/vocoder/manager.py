#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
声码器管理器
用于加载、管理和使用不同类型的声码器模型
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from .base import Vocoder, VocoderType
from .griffinlim import GriffinLim
from .hifigan import HiFiGAN

logger = logging.getLogger(__name__)

class VocoderManager:
    """
    声码器管理器，用于加载和管理不同类型的声码器模型
    """
    
    def __init__(self, model_dir: str = "model/vocoder/models"):
        """
        初始化声码器管理器
        
        参数:
            model_dir: 模型目录路径
        """
        self.vocoder = None
        self.vocoder_type = None
        self.config = None
        self.model_dir = Path(model_dir)
        
    def load_vocoder(self, vocoder_type: str = "auto", model_path: Optional[str] = None,
                    config_path: Optional[str] = None) -> None:
        """
        加载声码器模型
        
        参数:
            vocoder_type: 声码器类型，可选值: "auto", "simple", "hifigan", "universal_hifigan", "griffinlim", "melgan"
            model_path: 预训练模型路径
            config_path: 模型配置路径
        """
        if model_path is None:
            if vocoder_type == "universal_hifigan" or vocoder_type == "auto":
                # 尝试加载Universal HiFi-GAN模型
                universal_path = self.model_dir / "universal_hifigan.pt"
                universal_config = self.model_dir / "universal_hifigan_config.json"
                
                if universal_path.exists():
                    model_path = str(universal_path)
                    config_path = str(universal_config) if universal_config.exists() else None
                    vocoder_type = "universal_hifigan"
            
            if (vocoder_type == "hifigan" or vocoder_type == "auto") and model_path is None:
                # 尝试加载HiFi-GAN模型
                hifigan_path = self.model_dir / "hifigan_vocoder.pt"
                hifigan_config = self.model_dir / "hifigan_config.json"
                
                if hifigan_path.exists():
                    model_path = str(hifigan_path)
                    config_path = str(hifigan_config) if hifigan_config.exists() else None
                    vocoder_type = "hifigan"
            
            if (vocoder_type == "griffinlim" or vocoder_type == "auto") and model_path is None:
                # 尝试加载Griffin-Lim模型
                griffinlim_path = self.model_dir / "torchaudio_vocoder.pt"
                griffinlim_config = self.model_dir / "torchaudio_vocoder_config.json"
                
                if griffinlim_path.exists():
                    model_path = str(griffinlim_path)
                    config_path = str(griffinlim_config) if griffinlim_config.exists() else None
                    vocoder_type = "griffinlim"
            
            if (vocoder_type == "auto" or vocoder_type == "melgan") and model_path is None:
                # 尝试加载MelGAN模型
                melgan_path = self.model_dir / "melgan_vocoder.pt"
                melgan_config = self.model_dir / "melgan_config.pt"
                
                if melgan_path.exists():
                    model_path = str(melgan_path)
                    config_path = str(melgan_config) if melgan_config.exists() else None
                    vocoder_type = "melgan"
            
            if (vocoder_type == "auto" or vocoder_type == "simple") and model_path is None:
                # 使用简单声码器
                simple_path = self.model_dir / "vocoder.pt"
                simple_config = self.model_dir / "vocoder_config.pt"
                
                if simple_path.exists():
                    model_path = str(simple_path)
                    config_path = str(simple_config) if simple_config.exists() else None
                    vocoder_type = "simple"
                else:
                    # 使用未训练的简单声码器
                    logger.warning("未找到预训练声码器模型，使用未训练的简单声码器")
                    self.vocoder = Vocoder()
                    self.vocoder_type = VocoderType.SIMPLE
                    self.config = {"type": "simple", "pretrained": False}
                    return
        
        if model_path is None:
            logger.warning("未找到任何声码器模型，使用未训练的简单声码器")
            self.vocoder = Vocoder()
            self.vocoder_type = VocoderType.SIMPLE
            self.config = {"type": "simple", "pretrained": False}
            return
        
        try:
            # 根据类型加载声码器
            if vocoder_type == "hifigan" or vocoder_type == "universal_hifigan":
                logger.info(f"加载HiFi-GAN声码器: {model_path}")
                # 使用HiFiGAN类
                self.vocoder = HiFiGAN.from_pretrained(model_path, config_path)
                self.vocoder_type = VocoderType.from_string(vocoder_type)
                self.config = {
                    "type": vocoder_type,
                    "model_path": model_path,
                    "config_path": config_path,
                    "pretrained": True
                }
            elif vocoder_type == "griffinlim":
                logger.info(f"加载Griffin-Lim声码器: {model_path}")
                # 使用GriffinLim类
                self.vocoder = GriffinLim.from_pretrained(model_path, config_path)
                self.vocoder_type = VocoderType.GRIFFINLIM
                self.config = {
                    "type": "griffinlim",
                    "model_path": model_path,
                    "config_path": config_path,
                    "pretrained": True
                }
            elif vocoder_type == "melgan":
                logger.info(f"加载MelGAN声码器: {model_path}")
                self.vocoder = Vocoder.from_pretrained(model_path)
                self.vocoder_type = VocoderType.MELGAN
                self.config = {
                    "type": "melgan",
                    "model_path": model_path,
                    "config_path": config_path,
                    "pretrained": True
                }
            else:
                # 默认使用简单声码器
                logger.info(f"加载简单声码器: {model_path}")
                self.vocoder = Vocoder.from_pretrained(model_path)
                self.vocoder_type = VocoderType.SIMPLE
                self.config = {
                    "type": "simple",
                    "model_path": model_path,
                    "config_path": config_path,
                    "pretrained": True
                }
        except Exception as e:
            logger.error(f"加载声码器模型失败: {str(e)}")
            logger.warning("使用未训练的简单声码器")
            self.vocoder = Vocoder()
            self.vocoder_type = VocoderType.SIMPLE
            self.config = {"type": "simple", "pretrained": False}
    
    def generate_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        生成音频波形
        
        参数:
            mel_spectrogram: [batch_size, time, n_mels]的梅尔频谱
            
        返回:
            [batch_size, 1, time*factor]的音频波形
        """
        if self.vocoder is None:
            # 如果还没有加载声码器，使用默认声码器
            self.load_vocoder()
            
        # 确保模型处于评估模式
        self.vocoder.eval()
        
        # 前向传播生成波形
        with torch.no_grad():
            waveform = self.vocoder(mel_spectrogram)
            
        return waveform
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取声码器信息
        
        返回:
            包含声码器信息的字典
        """
        return {
            "type": str(self.vocoder_type) if self.vocoder_type else "none",
            "config": self.config,
            "is_loaded": self.vocoder is not None
        }
    
    @staticmethod
    def list_available_vocoders(model_dir: str = "model/vocoder/models") -> List[Dict[str, Any]]:
        """
        列出可用的声码器模型
        
        参数:
            model_dir: 模型目录路径
            
        返回:
            可用声码器列表
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            logger.warning(f"模型目录不存在: {model_dir}")
            return []
        
        vocoders = []
        
        # 检查HiFi-GAN
        if (model_dir / "hifigan_vocoder.pt").exists():
            vocoders.append({
                "type": "hifigan",
                "model_path": str(model_dir / "hifigan_vocoder.pt"),
                "config_path": str(model_dir / "hifigan_config.json") if (model_dir / "hifigan_config.json").exists() else None
            })
        
        # 检查Universal HiFi-GAN
        if (model_dir / "universal_hifigan.pt").exists():
            vocoders.append({
                "type": "universal_hifigan",
                "model_path": str(model_dir / "universal_hifigan.pt"),
                "config_path": str(model_dir / "universal_hifigan_config.json") if (model_dir / "universal_hifigan_config.json").exists() else None
            })
        
        # 检查Griffin-Lim
        if (model_dir / "torchaudio_vocoder.pt").exists():
            vocoders.append({
                "type": "griffinlim",
                "model_path": str(model_dir / "torchaudio_vocoder.pt"),
                "config_path": str(model_dir / "torchaudio_vocoder_config.json") if (model_dir / "torchaudio_vocoder_config.json").exists() else None
            })
        
        # 检查MelGAN
        if (model_dir / "melgan_vocoder.pt").exists():
            vocoders.append({
                "type": "melgan",
                "model_path": str(model_dir / "melgan_vocoder.pt"),
                "config_path": str(model_dir / "melgan_config.pt") if (model_dir / "melgan_config.pt").exists() else None
            })
        
        # 检查简单声码器
        if (model_dir / "vocoder.pt").exists():
            vocoders.append({
                "type": "simple",
                "model_path": str(model_dir / "vocoder.pt"),
                "config_path": str(model_dir / "vocoder_config.pt") if (model_dir / "vocoder_config.pt").exists() else None
            })
        
        return vocoders 