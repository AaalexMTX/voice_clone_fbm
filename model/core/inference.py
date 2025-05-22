#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

from .voice_clone import VoiceCloneSystem
from ..config import load_config, get_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(text, reference_audio=None, embedding_file=None, output_file="output.wav", save_embedding=False):
    """
    运行语音克隆推理
    
    参数:
        text: 要合成的文本
        reference_audio: 参考音频文件路径
        embedding_file: 预先提取的说话人嵌入文件
        output_file: 输出音频文件路径
        save_embedding: 是否保存提取的说话人嵌入
    """
    # 加载配置
    config = get_config()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 初始化语音克隆系统
    device = config["model"]["device"]
    model_dir = config["model"]["model_dir"]
    
    logger.info(f"初始化语音克隆系统 - 设备: {device}, 模型目录: {model_dir}")
    voice_clone_system = VoiceCloneSystem(model_dir=model_dir, device=device)
    
    # 检查参数
    if embedding_file is None and reference_audio is None:
        raise ValueError("必须提供embedding_file或reference_audio")
    
    try:
        if embedding_file:
            # 使用预先提取的说话人嵌入
            logger.info(f"加载说话人嵌入: {embedding_file}")
            speaker_embedding = voice_clone_system.load_speaker_embedding(embedding_file)
            voice_clone_system.synthesize(text, speaker_embedding, output_file)
        else:
            # 从参考音频提取说话人嵌入并合成
            logger.info(f"使用参考音频: {reference_audio}")
            
            # 提取说话人嵌入
            speaker_embedding = voice_clone_system.extract_speaker_embedding(reference_audio)
            logger.info(f"成功提取说话人嵌入，形状: {speaker_embedding.shape}")
            
            # 如果需要，保存嵌入
            if save_embedding:
                embedding_output = os.path.splitext(output_file)[0] + "_embedding.npy"
                np.save(embedding_output, speaker_embedding)
                logger.info(f"说话人嵌入已保存到: {embedding_output}")
            
            # 合成语音
            logger.info(f"开始合成语音到: {output_file}")
            voice_clone_system.synthesize(text, speaker_embedding, output_file)
        
        logger.info(f"语音已生成到: {output_file}")
    except Exception as e:
        logger.error(f"推理过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="语音克隆推理工具")
    parser.add_argument("--text", type=str, required=True, help="要合成的文本")
    parser.add_argument("--reference_audio", type=str, help="参考音频文件路径")
    parser.add_argument("--embedding_file", type=str, help="预先提取的说话人嵌入文件路径 (.npy)")
    parser.add_argument("--output_file", type=str, default="output.wav", help="输出音频文件路径")
    parser.add_argument("--save_embedding", action="store_true", help="是否保存提取的说话人嵌入")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    load_config(args.config)
    
    # 检查参数
    if args.embedding_file is None and args.reference_audio is None:
        parser.error("必须提供--embedding_file或--reference_audio")
    
    # 运行推理
    run_inference(
        text=args.text,
        reference_audio=args.reference_audio,
        embedding_file=args.embedding_file,
        output_file=args.output_file,
        save_embedding=args.save_embedding
    ) 