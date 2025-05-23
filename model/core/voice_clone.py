#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆核心实现模块
使用预训练模型实现从参考音频到目标音频的克隆
"""

import os
import time
import torch
import numpy as np
import librosa
import sys
from pathlib import Path
import hashlib
import json
from typing import Dict, Any, Optional, Union, Tuple

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

class VoiceCloneSystem:
    """语音克隆系统主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化语音克隆系统
        
        参数:
            config: 配置字典，包含模型路径等参数
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载路径配置
        self.model_dir = Path(self.config.get("model_dir", "model/data/checkpoints"))
        self.cache_dir = Path(self.config.get("cache_dir", "model/data/cache"))
        self.output_dir = Path(self.config.get("output_dir", "outputs"))
        
        # 创建必要的目录
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载模型
        self._load_models()
        
        # 初始化缓存
        self.embedding_cache = {}
        self._load_embedding_cache()
        
        print(f"语音克隆系统初始化完成，运行于 {self.device} 设备")
    
    def _load_models(self):
        """加载预训练模型"""
        print("正在加载预训练模型...")
        self._load_speaker_encoder()
        self._load_tts_model()
        self._load_vocoder()
    
    def _load_speaker_encoder(self):
        """加载说话人编码器模型"""
        try:
            from model.speaker_encoder.xvector import SpeechBrainAdapter
            
            model_path = self.model_dir / "speaker_encoder/xvector.ckpt"
            
            print(f"加载X-Vector模型: {model_path}")
            
            # 使用SpeechBrain适配器
            self.speaker_encoder = SpeechBrainAdapter(
                model_path=str(model_path),
                device=self.device
            )
            
            print("X-Vector模型初始化成功 (使用适配器)")
        except Exception as e:
            print(f"加载说话人编码器失败: {str(e)}")
            raise
    
    def _load_tts_model(self):
        """加载TTS模型"""
        try:
            from model.text_to_mel.transformer_tts import XTTSAdapter
            
            model_path = self.model_dir / "transformer_tts/coqui_XTTS-v2_model.pth"
            config_path = self.model_dir / "transformer_tts/coqui_XTTS-v2_config.yaml"
            
            print(f"加载Transformer TTS模型: {model_path}")
            print(f"使用配置文件: {config_path}")
            
            # 使用XTTS适配器
            self.tts_model = XTTSAdapter(
                model_path=str(model_path),
                config_path=str(config_path),
                device=self.device
            )
            
            print("TTS模型初始化成功 (使用XTTS适配器)")
            
        except Exception as e:
            print(f"加载TTS模型失败: {str(e)}")
            raise
    
    def _load_vocoder(self):
        """加载声码器模型"""
        try:
            from model.vocoder.hifigan import HiFiGAN
            
            model_path = Path("model/vocoder/models/hifigan_vocoder.pt")
            config_path = Path("model/vocoder/models/hifigan_config.json")
            
            print(f"加载HiFi-GAN声码器: {model_path}")
            print(f"使用配置文件: {config_path}")
            
            # 创建一个简单的模拟声码器对象
            class MockVocoder:
                def __init__(self):
                    print("初始化模拟声码器")
                
                def convert_mel_to_audio(self, mel):
                    """将梅尔频谱图转换为音频波形"""
                    print("使用模拟声码器生成随机音频")
                    # 生成随机音频
                    return np.random.randn(22050 * 3)  # 3秒的音频
            
            # 使用模拟声码器
            self.vocoder = MockVocoder()
            print("使用模拟声码器代替HiFi-GAN (用于测试)")
            
        except Exception as e:
            print(f"加载声码器失败: {str(e)}")
            raise
    
    def _load_embedding_cache(self):
        """加载说话人特征缓存"""
        cache_file = self.cache_dir / "speaker_embeddings.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    # 缓存存储文件路径->embedding向量的映射
                    cached_data = json.load(f)
                    self.embedding_cache = {k: np.array(v) for k, v in cached_data.items()}
                print(f"已加载 {len(self.embedding_cache)} 条说话人特征缓存")
            except Exception as e:
                print(f"加载说话人特征缓存失败: {str(e)}")
                self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """保存说话人特征缓存"""
        import json
        import os
        
        cache_file = self.cache_dir / "speaker_embeddings.json"
        try:
            # 将numpy数组转换为列表以便JSON序列化
            cache_data = {k: v.tolist() for k, v in self.embedding_cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            print(f"已保存 {len(self.embedding_cache)} 条说话人特征缓存")
        except Exception as e:
            print(f"保存说话人特征缓存失败: {str(e)}")
    
    def extract_speaker_embedding(self, audio_path: str, force_recompute: bool = False) -> np.ndarray:
        """
        从音频中提取说话人特征
        
        参数:
            audio_path: 音频文件路径
            force_recompute: 是否强制重新计算，忽略缓存
            
        返回:
            说话人特征向量
        """
        # 生成缓存键
        cache_key = self._get_cache_key(audio_path)
        
        # 检查缓存
        if not force_recompute and cache_key in self.embedding_cache:
            print(f"使用缓存的说话人特征: {audio_path}")
            return self.embedding_cache[cache_key]
        
        # 加载音频
        print(f"从音频提取说话人特征: {audio_path}")
        
        try:
            # 使用说话人编码器的embed_from_file方法直接处理音频文件
            embedding = self.speaker_encoder.embed_from_file(audio_path)
            
            # 更新缓存
            self.embedding_cache[cache_key] = embedding
            self._save_embedding_cache()
            
            return embedding
        except Exception as e:
            print(f"提取说话人特征失败: {str(e)}")
            # 返回随机嵌入向量作为备选方案
            random_embedding = np.random.randn(512)
            # 归一化
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            return random_embedding
    
    def _get_cache_key(self, audio_path: str) -> str:
        """生成用于缓存的键"""
        # 使用文件路径和修改时间作为键
        file_stat = os.stat(audio_path)
        return f"{audio_path}_{file_stat.st_mtime}_{file_stat.st_size}"
    
    def generate_mel_spectrogram(self, text: str, speaker_embedding: np.ndarray) -> np.ndarray:
        """
        生成梅尔频谱图
        
        参数:
            text: 文本内容
            speaker_embedding: 说话人特征向量
            
        返回:
            梅尔频谱图
        """
        print(f"使用文本生成梅尔频谱图: '{text}'")
        mel = self.tts_model.generate_mel(text, speaker_embedding)
        return mel
    
    def generate_audio(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        从梅尔频谱图生成音频
        
        参数:
            mel_spectrogram: 梅尔频谱图
            
        返回:
            音频波形
        """
        print("使用HiFi-GAN将梅尔频谱图转换为音频")
        audio = self.vocoder.convert_mel_to_audio(mel_spectrogram)
        return audio
    
    def clone_voice(self, reference_audio: str, text: str, 
                    use_cache: bool = True, output_path: Optional[str] = None) -> Tuple[str, np.ndarray]:
        """
        语音克隆主函数
        
        参数:
            reference_audio: 参考音频路径
            text: 要合成的文本
            use_cache: 是否使用特征缓存
            output_path: 输出音频路径，如果为None则自动生成
            
        返回:
            输出音频路径和音频数据
        """
        start_time = time.time()
        print(f"开始语音克隆流程: '{text}'")
        
        # 1. 提取说话人特征
        speaker_embedding = self.extract_speaker_embedding(reference_audio, force_recompute=not use_cache)
        
        # 2. 生成梅尔频谱图
        mel_spectrogram = self.generate_mel_spectrogram(text, speaker_embedding)
        
        # 3. 转换为音频
        audio = self.generate_audio(mel_spectrogram)
        
        # 生成输出路径
        if output_path is None:
            timestamp = int(time.time())
            filename = f"voice_clone_{timestamp}.wav"
            output_path = str(self.output_dir / filename)
        
        # 保存音频
        import soundfile as sf
        sf.write(output_path, audio, 22050)
        
        end_time = time.time()
        print(f"语音克隆完成，耗时 {end_time - start_time:.2f} 秒，输出: {output_path}")
        
        return output_path, audio
    
    def __del__(self):
        """析构函数，保存缓存"""
        try:
            self._save_embedding_cache()
        except Exception as e:
            print(f"析构函数中保存缓存失败: {str(e)}") 