#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试前后端交互API的客户端
"""

import os
import sys
import argparse
import json
import requests
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_health_check(base_url):
    """测试健康检查接口"""
    url = f"{base_url}/health"
    
    print(f"\n[测试] 健康检查: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"[成功] 状态码: {response.status_code}")
            print(f"响应: {response.json()}")
            return True
        else:
            print(f"[失败] 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"[错误] 请求失败: {str(e)}")
        return False

def test_upload_reference(base_url, audio_path):
    """测试上传参考音频接口"""
    url = f"{base_url}/upload_reference"
    
    print(f"\n[测试] 上传参考音频: {url}")
    print(f"音频文件: {audio_path}")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_path), f, 'audio/wav')}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print(f"[成功] 状态码: {response.status_code}")
            result = response.json()
            print(f"响应: {result}")
            return result.get('reference_audio_id')
        else:
            print(f"[失败] 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return None
    except Exception as e:
        print(f"[错误] 请求失败: {str(e)}")
        return None

def test_extract_embedding(base_url, audio_path):
    """测试提取说话人嵌入接口"""
    url = f"{base_url}/extract_embedding"
    
    print(f"\n[测试] 提取说话人嵌入: {url}")
    print(f"音频文件: {audio_path}")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_path), f, 'audio/wav')}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print(f"[成功] 状态码: {response.status_code}")
            result = response.json()
            print(f"响应: {result}")
            return result.get('embedding_id')
        else:
            print(f"[失败] 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return None
    except Exception as e:
        print(f"[错误] 请求失败: {str(e)}")
        return None

def test_synthesize(base_url, text, embedding_id):
    """测试合成语音接口"""
    url = f"{base_url}/synthesize"
    
    print(f"\n[测试] 合成语音: {url}")
    print(f"文本: {text}")
    print(f"嵌入ID: {embedding_id}")
    
    try:
        data = {
            'text': text,
            'embedding_id': embedding_id
        }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"[成功] 状态码: {response.status_code}")
            result = response.json()
            print(f"响应: {result}")
            return result.get('output_id')
        else:
            print(f"[失败] 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return None
    except Exception as e:
        print(f"[错误] 请求失败: {str(e)}")
        return None

def test_tts(base_url, text, audio_path):
    """测试一站式TTS接口"""
    url = f"{base_url}/tts"
    
    print(f"\n[测试] 一站式TTS: {url}")
    print(f"文本: {text}")
    print(f"音频文件: {audio_path}")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_path), f, 'audio/wav')}
            data = {'text': text}
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            print(f"[成功] 状态码: {response.status_code}")
            result = response.json()
            print(f"响应: {result}")
            return result.get('output_id')
        else:
            print(f"[失败] 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return None
    except Exception as e:
        print(f"[错误] 请求失败: {str(e)}")
        return None

def test_get_audio(base_url, output_id, save_path):
    """测试获取音频接口"""
    url = f"{base_url}/audio/{output_id}"
    
    print(f"\n[测试] 获取音频: {url}")
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            print(f"[成功] 状态码: {response.status_code}")
            
            # 保存音频文件
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"音频已保存到: {save_path}")
            return True
        else:
            print(f"[失败] 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"[错误] 请求失败: {str(e)}")
        return False

def test_full_pipeline(base_url, audio_path, text, output_dir):
    """测试完整流程"""
    print("\n===== 测试完整语音克隆流程 =====")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 健康检查
    if not test_health_check(base_url):
        print("健康检查失败，终止测试")
        return False
    
    # 2. 提取说话人嵌入
    embedding_id = test_extract_embedding(base_url, audio_path)
    if not embedding_id:
        print("提取说话人嵌入失败，终止测试")
        return False
    
    # 3. 合成语音
    output_id = test_synthesize(base_url, text, embedding_id)
    if not output_id:
        print("合成语音失败，终止测试")
        return False
    
    # 4. 下载音频
    output_path = os.path.join(output_dir, f"test_synthesized_{output_id}.wav")
    if not test_get_audio(base_url, output_id, output_path):
        print("下载音频失败，终止测试")
        return False
    
    # 5. 一站式TTS
    output_id = test_tts(base_url, f"这是一站式TTS测试，使用与原文不同的文本: {text}", audio_path)
    if not output_id:
        print("一站式TTS失败，终止测试")
        return False
    
    # 6. 下载一站式TTS音频
    output_path = os.path.join(output_dir, f"test_one_step_{output_id}.wav")
    if not test_get_audio(base_url, output_id, output_path):
        print("下载一站式TTS音频失败，终止测试")
        return False
    
    print("\n===== 完整测试流程成功! =====")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试前后端交互API")
    parser.add_argument("--url", type=str, default="http://localhost:7860/api", help="API基础URL")
    parser.add_argument("--audio", type=str, required=True, help="测试音频文件路径")
    parser.add_argument("--text", type=str, default="这是一个测试文本，用于测试语音克隆API。", help="测试文本")
    parser.add_argument("--output", type=str, default="model/outputs", help="输出目录")
    
    args = parser.parse_args()
    
    # 执行完整测试流程
    test_full_pipeline(args.url, args.audio, args.text, args.output)

if __name__ == "__main__":
    main() 