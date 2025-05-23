#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆接口测试脚本
模拟前端发送HTTP请求到后端，测试语音克隆功能
"""

import requests
import json
import argparse
import time
from pathlib import Path

# API服务器地址
API_URL = "http://127.0.0.1:5000"

def test_clone_voice(audio_path, text, output_format="wav"):
    """
    测试语音克隆接口
    
    参数:
        audio_path: 参考音频文件路径
        text: 要合成的文本内容
        output_format: 输出音频格式
    """
    url = f"{API_URL}/api/clone_voice"
    
    # 准备请求数据
    data = {
        "reference_audio": audio_path,  # 这里直接传递音频文件路径
        "text": text,
        "output_format": output_format,
        "use_cache": True
    }
    
    print(f"发送语音克隆请求...")
    print(f"参考音频: {audio_path}")
    print(f"文本内容: {text}")
    print(f"输出格式: {output_format}")
    
    # 发送请求
    response = requests.post(url, json=data)
    
    print(f"响应状态码: {response.status_code}")
    
    # 解析响应
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("success"):
            print(f"语音克隆成功!")
            print(f"输出文件: {result.get('output_file')}")
            return result
        else:
            print(f"语音克隆失败: {result.get('error')}")
            return None
    else:
        print(f"请求失败: {response.text}")
        return None

def test_upload_reference_audio(audio_path):
    """
    测试上传参考音频文件
    
    参数:
        audio_path: 音频文件路径
    """
    url = f"{API_URL}/api/upload_reference"
    
    # 准备文件
    files = {
        'file': open(audio_path, 'rb')
    }
    
    print(f"上传参考音频: {audio_path}")
    
    # 发送请求
    response = requests.post(url, files=files)
    
    print(f"响应状态码: {response.status_code}")
    
    # 解析响应
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("success"):
            print(f"文件上传成功!")
            print(f"服务器路径: {result.get('file_path')}")
            return result.get('file_path')
        else:
            print(f"文件上传失败: {result.get('error')}")
            return None
    else:
        print(f"请求失败: {response.text}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆接口测试脚本")
    parser.add_argument("--audio", type=str, required=True, help="参考音频文件路径")
    parser.add_argument("--text", type=str, default="这是一个语音克隆测试。", help="要合成的文本内容")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3", "flac", "ogg"], help="输出音频格式")
    
    args = parser.parse_args()
    
    # 先上传参考音频
    server_path = test_upload_reference_audio(args.audio)
    
    if server_path:
        # 使用上传后的服务器路径进行语音克隆
        test_clone_voice(server_path, args.text, args.format)
    else:
        print("由于文件上传失败，无法继续语音克隆测试")

if __name__ == "__main__":
    main() 