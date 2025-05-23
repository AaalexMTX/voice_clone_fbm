#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import logging
from pathlib import Path

from ..speaker_encoder.speaker_encoder import SpeakerEncoder
from ..speaker_encoder.xvector import XVectorEncoder
from ..speaker_encoder.audio_processing import preprocess_wav
from ..text_to_mel.transformer_tts import TransformerTTS
from ..vocoder.hifigan import HiFiGAN
from ..vocoder.griffinlim import GriffinLim
from .model import VoiceCloneModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceCloneSystem:
    """
    语音克隆系统
    
    这个类是语音克隆系统的主要接口，集成了：
    1. 说话人编码器（从目标音频中提取说话人嵌入）
    2. 文本到梅尔频谱生成器（将文本和说话人嵌入转换为梅尔频谱）
    3. 声码器（将梅尔频谱转换为音频波形）
    
    使用流程：
    1. 从目标音频提取说话人嵌入
    2. 将文本和说话人嵌入转换为梅尔频谱
    3. 将梅尔频谱转换为音频波形
    """
    
    def __init__(self, model_dir="model/data/checkpoints", device=None, vocoder_type="hifigan", encoder_type="xvector", tts_type="transformer"):
        """
        初始化语音克隆系统
        
        参数:
            model_dir: 模型目录
            device: 设备 (None为自动选择)
            vocoder_type: 声码器类型，当前默认: "hifigan"
            encoder_type: 说话人编码器类型: "xvector"或"speaker_encoder"
            tts_type: 文本到梅尔频谱模型类型: "transformer"或"default"
        """
        # 设置设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 设置编码器类型和TTS类型
        self.encoder_type = encoder_type
        self.tts_type = tts_type
        
        # 模型路径
        self.model_dir = Path(model_dir)
        self.speaker_encoder_path = self.model_dir / "speaker_encoder.pt"
        self.xvector_encoder_path = self.model_dir / "xvector_encoder.pt"
        self.tts_model_path = self.model_dir / "tts_model.pt"
        self.transformer_tts_path = self.model_dir / "transformer_tts.pt"
        # 修改声码器路径，指向model/vocoder/models目录
        self.vocoder_path = Path("model/vocoder/models/hifigan_vocoder.pt")
        self.vocoder_config = Path("model/vocoder/models/hifigan_config.json")
        
        # 初始化模型
        self._init_models(vocoder_type)
    
    def _init_models(self, vocoder_type="hifigan"):
        """初始化所有模型"""
        # 初始化说话人编码器
        if self.encoder_type == "xvector":
            # 使用X-Vector作为说话人编码器
            self.speaker_encoder = XVectorEncoder(mel_n_channels=80, embedding_dim=512).to(self.device)
            encoder_path = self.xvector_encoder_path
        else:
            # 使用原始SpeakerEncoder
            self.speaker_encoder = SpeakerEncoder().to(self.device)
            encoder_path = self.speaker_encoder_path
        
        if os.path.exists(encoder_path):
            logger.info(f"加载说话人编码器: {encoder_path}")
            try:
                # 对于PyTorch 2.6+版本，处理weights_only参数
                if torch.__version__ >= "2.6.0":
                    self.speaker_encoder.load(str(encoder_path))
                else:
                    state_dict = torch.load(encoder_path, map_location=self.device)
                    self.speaker_encoder.load_state_dict(state_dict)
                logger.info(f"成功加载说话人编码器: {encoder_path}")
            except Exception as e:
                logger.error(f"加载说话人编码器时出错: {str(e)}")
                logger.warning("使用未训练的模型")
        else:
            logger.warning(f"找不到说话人编码器模型: {encoder_path}，使用未训练的模型")
        self.speaker_encoder.eval()
        
        # 初始化TTS模型
        if self.tts_type == "transformer":
            # 使用基于Transformer的TTS模型
            self.tts_model = TransformerTTS(
                vocab_size=256,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                speaker_dim=512,
                mel_dim=80
            ).to(self.device)
            tts_path = self.transformer_tts_path
        else:
            # 使用默认的TTS模型
            self.tts_model = VoiceCloneModel().to(self.device)
            tts_path = self.tts_model_path
            
        if os.path.exists(tts_path):
            logger.info(f"加载TTS模型: {tts_path}")
            try:
                self.tts_model.load_state_dict(torch.load(tts_path, map_location=self.device))
                logger.info(f"成功加载TTS模型: {tts_path}")
            except Exception as e:
                logger.error(f"加载TTS模型时出错: {str(e)}")
                logger.warning("使用未训练的模型")
        else:
            logger.warning(f"找不到TTS模型: {tts_path}，使用未训练的模型")
        self.tts_model.eval()
        
        # 初始化声码器
        if vocoder_type == "griffinlim":
            # 使用Griffin-Lim声码器
            griffin_path = self.model_dir / "torchaudio_vocoder.pt"
            griffin_config = self.model_dir / "torchaudio_vocoder_config.json"
            
            if os.path.exists(griffin_path):
                self.vocoder = GriffinLim.from_pretrained(
                    str(griffin_path),
                    str(griffin_config) if os.path.exists(griffin_config) else None
                ).to(self.device)
                logger.info(f"加载Griffin-Lim声码器: {griffin_path}")
            else:
                logger.warning(f"找不到Griffin-Lim声码器: {griffin_path}，使用未训练的模型")
                self.vocoder = GriffinLim().to(self.device)
        else:
            # 使用HIFI-GAN声码器
            if os.path.exists(self.vocoder_path):
                self.vocoder = HiFiGAN.from_pretrained(
                    str(self.vocoder_path), 
                    str(self.vocoder_config) if os.path.exists(self.vocoder_config) else None
                ).to(self.device)
                logger.info(f"加载HIFI-GAN声码器: {self.vocoder_path}")
            else:
                logger.warning(f"找不到HIFI-GAN声码器: {self.vocoder_path}，使用未训练的模型")
                self.vocoder = HiFiGAN().to(self.device)
        
        self.vocoder.eval()
    
    def extract_speaker_embedding(self, audio_path):
        """
        从音频文件中提取说话人嵌入
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            说话人嵌入向量
        """
        logger.info(f"从音频提取说话人嵌入: {audio_path}")
        
        try:
            # 预处理音频
            wav = preprocess_wav(audio_path)
            
            # 提取嵌入
            with torch.no_grad():
                embedding = self.speaker_encoder.embed_utterance(wav)
            
            logger.info(f"成功提取说话人嵌入，维度: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"提取说话人嵌入时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def synthesize(self, text, speaker_embedding, output_path):
        """
        使用说话人嵌入合成语音
        
        参数:
            text: 要合成的文本
            speaker_embedding: 说话人嵌入向量
            output_path: 输出音频文件路径
        """
        logger.info(f"合成语音: {text}")
        
        try:
            # 将说话人嵌入转换为tensor
            if isinstance(speaker_embedding, np.ndarray):
                speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0).to(self.device)
            
            # 使用TTS模型生成梅尔频谱
            with torch.no_grad():
                # 将文本转换为序列（简化实现，实际中需要更复杂的文本处理）
                # TODO: 实现文本到ID的转换
                logger.info("生成文本序列")
                text_sequence = torch.tensor([[ord(c) % 256 for c in text]], dtype=torch.long).to(self.device)
                
                # 生成梅尔频谱
                logger.info("使用TTS模型生成梅尔频谱")
                if self.tts_type == "transformer":
                    # 使用Transformer TTS模型
                    mel_outputs = self.tts_model.inference(text_sequence, speaker_embedding)
                else:
                    # 使用默认TTS模型
                    mel_outputs = self.tts_model(text_sequence, speaker_embedding)
                
                # 使用声码器生成波形
                logger.info("使用声码器生成波形")
                waveform = self.vocoder(mel_outputs)
                waveform = waveform.squeeze().cpu().numpy()
            
            # 保存音频
            logger.info(f"保存波形到: {output_path}")
            self._save_wav(waveform, output_path)
            
            logger.info(f"语音已保存到: {output_path}")
        except Exception as e:
            logger.error(f"语音合成过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def clone_voice(self, text, reference_audio_path, output_path):
        """
        一步完成语音克隆
        
        参数:
            text: 要合成的文本
            reference_audio_path: 参考音频文件路径
            output_path: 输出音频文件路径
        """
        # 提取说话人嵌入
        speaker_embedding = self.extract_speaker_embedding(reference_audio_path)
        
        # 合成语音
        self.synthesize(text, speaker_embedding, output_path)
    
    def _save_wav(self, waveform, path, sample_rate=22050):
        """保存波形到WAV文件"""
        # 这里使用scipy或其他库保存音频文件
        import soundfile as sf
        sf.write(path, waveform, sample_rate)
    
    def train(self, dataset_path, output_dir="trained_models", epochs=100):
        """
        训练模型（占位，实际实现需要更多代码）
        
        参数:
            dataset_path: 数据集路径
            output_dir: 输出目录
            epochs: 训练轮数
        """
        logger.info("训练功能尚未实现")
        # TODO: 实现训练功能 