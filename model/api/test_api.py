#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import requests
import json
import os
from pathlib import Path
from ..config import load_config, get_config

def test_health(base_url):
    """测试健康检查接口"""
    url = f"{base_url}/api/health"
    response = requests.get(url)
    print(f"健康检查结果: {response.json()}")
    return response.json()

def extract_embedding(base_url, audio_file):
    """测试提取说话人嵌入接口"""
    url = f"{base_url}/api/extract_embedding"
    
    with open(audio_file, 'rb') as f:
        files = {'audio': (os.path.basename(audio_file), f, 'audio/wav')}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(f"提取嵌入向量成功: {response.json()}")
        return response.json()
    else:
        print(f"提取嵌入向量失败: {response.text}")
        return None

def upload_reference(base_url, audio_file):
    """测试上传参考音频接口"""
    url = f"{base_url}/api/upload_reference"
    
    with open(audio_file, 'rb') as f:
        files = {'audio': (os.path.basename(audio_file), f, 'audio/wav')}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(f"上传参考音频成功: {response.json()}")
        return response.json()
    else:
        print(f"上传参考音频失败: {response.text}")
        return None

def synthesize_from_embedding(base_url, text, embedding_id):
    """测试使用嵌入向量合成语音接口"""
    url = f"{base_url}/api/synthesize"
    
    data = {
        'text': text,
        'embedding_id': embedding_id
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"使用嵌入向量合成语音成功: {response.json()}")
        return response.json()
    else:
        print(f"使用嵌入向量合成语音失败: {response.text}")
        return None

def synthesize_from_reference(base_url, text, reference_audio_id):
    """测试使用参考音频合成语音接口"""
    url = f"{base_url}/api/synthesize"
    
    data = {
        'text': text,
        'reference_audio_id': reference_audio_id
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"使用参考音频合成语音成功: {response.json()}")
        return response.json()
    else:
        print(f"使用参考音频合成语音失败: {response.text}")
        return None

def one_step_tts(base_url, text, audio_file):
    """测试一站式TTS接口"""
    url = f"{base_url}/api/tts"
    
    with open(audio_file, 'rb') as f:
        files = {'audio': (os.path.basename(audio_file), f, 'audio/wav')}
        data = {'text': text}
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        print(f"一站式TTS成功: {response.json()}")
        return response.json()
    else:
        print(f"一站式TTS失败: {response.text}")
        return None

def download_audio(base_url, output_id, save_path):
    """下载生成的音频文件"""
    url = f"{base_url}/api/audio/{output_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"音频已下载到: {save_path}")
        return True
    else:
        print(f"下载音频失败: {response.text}")
        return False

def test_full_pipeline(base_url, audio_file, text, output_dir="./output"):
    """测试完整的语音克隆流程"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=========== 开始测试完整流程 ===========")
    
    # 1. 检查健康状态
    print("\n1. 检查服务健康状态")
    health = test_health(base_url)
    if not health.get('model_loaded', False):
        print("错误: 模型未加载，无法继续测试")
        return False
    
    # 2. 提取说话人嵌入
    print("\n2. 提取说话人嵌入")
    embedding_result = extract_embedding(base_url, audio_file)
    if not embedding_result:
        print("错误: 提取说话人嵌入失败，无法继续测试")
        return False
    
    embedding_id = embedding_result.get('embedding_id')
    
    # 3. 使用嵌入向量合成语音
    print("\n3. 使用嵌入向量合成语音")
    synthesis_result = synthesize_from_embedding(base_url, text, embedding_id)
    if not synthesis_result:
        print("错误: 使用嵌入向量合成语音失败")
        return False
    
    output_id = synthesis_result.get('output_id')
    
    # 4. 下载合成的音频
    print("\n4. 下载合成的音频")
    output_path = os.path.join(output_dir, f"synthesis_from_embedding.wav")
    if not download_audio(base_url, output_id, output_path):
        print("错误: 下载音频失败")
        return False
    
    # 5. 上传参考音频
    print("\n5. 上传参考音频")
    reference_result = upload_reference(base_url, audio_file)
    if not reference_result:
        print("错误: 上传参考音频失败")
        return False
    
    reference_audio_id = reference_result.get('reference_audio_id')
    
    # 6. 使用参考音频合成语音
    print("\n6. 使用参考音频合成语音")
    synthesis_result2 = synthesize_from_reference(base_url, text, reference_audio_id)
    if not synthesis_result2:
        print("错误: 使用参考音频合成语音失败")
        return False
    
    output_id2 = synthesis_result2.get('output_id')
    
    # 7. 下载合成的音频
    print("\n7. 下载合成的音频")
    output_path2 = os.path.join(output_dir, f"synthesis_from_reference.wav")
    if not download_audio(base_url, output_id2, output_path2):
        print("错误: 下载音频失败")
        return False
    
    # 8. 测试一站式TTS
    print("\n8. 测试一站式TTS")
    tts_result = one_step_tts(base_url, text, audio_file)
    if not tts_result:
        print("错误: 一站式TTS失败")
        return False
    
    output_id3 = tts_result.get('output_id')
    
    # 9. 下载合成的音频
    print("\n9. 下载一站式TTS音频")
    output_path3 = os.path.join(output_dir, f"one_step_tts.wav")
    if not download_audio(base_url, output_id3, output_path3):
        print("错误: 下载音频失败")
        return False
    
    print("\n=========== 测试完成 ===========")
    print(f"所有合成的音频文件都保存在 {output_dir} 目录下")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆API测试工具")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--url", type=str, help="API服务器URL")
    parser.add_argument("--audio", type=str, help="参考音频文件路径")
    parser.add_argument("--text", type=str, help="要合成的文本")
    parser.add_argument("--output", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    base_url = args.url or config["test"]["url"]
    audio_file = args.audio or config["test"]["audio"]
    text = args.text or config["test"]["text"]
    output_dir = args.output or config["test"]["output_dir"]
    
    # 检查音频文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误: 找不到音频文件 {audio_file}")
        return
    
    # 测试完整流程
    test_full_pipeline(base_url, audio_file, text, output_dir) 