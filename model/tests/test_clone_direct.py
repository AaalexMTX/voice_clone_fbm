#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆接口直接测试脚本
直接使用multipart/form-data格式发送请求，模拟前端表单提交
"""

import requests
import json
import argparse
import os
import time
from pathlib import Path

# API服务器地址
API_URL = "http://127.0.0.1:5000"

def test_clone_voice_direct(audio_path, text, output_path=None):
    """
    直接测试语音克隆接口
    
    参数:
        audio_path: 参考音频文件路径
        text: 要合成的文本内容
        output_path: 输出音频文件路径
    """
    url = f"{API_URL}/api/clone_voice"
    
    # 如果没有指定输出路径，则使用当前目录
    if output_path is None:
        output_dir = Path("outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"clone_output_{int(time.time())}.wav"
    
    # 准备请求数据
    files = {
        'reference_audio': open(audio_path, 'rb'),
    }
    
    data = {
        'text': text,
        'output_format': 'wav',
    }
    
    print(f"发送语音克隆请求...")
    print(f"参考音频: {audio_path}")
    print(f"文本内容: {text}")
    print(f"输出路径: {output_path}")
    
    # 发送请求
    response = requests.post(url, files=files, data=data)
    
    print(f"响应状态码: {response.status_code}")
    
    # 处理响应
    if response.status_code == 200:
        try:
            # 尝试解析为JSON
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            if result.get("success"):
                print(f"语音克隆成功!")
                print(f"输出文件: {result.get('output_file')}")
                return result
            else:
                print(f"语音克隆失败: {result.get('error')}")
                return None
        except json.JSONDecodeError:
            # 如果不是JSON，则可能是二进制音频数据
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"音频已保存到: {output_path}")
            return output_path
    else:
        print(f"请求失败: {response.text}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆接口直接测试脚本")
    parser.add_argument("--audio", type=str, required=True, help="参考音频文件路径")
    parser.add_argument("--text", type=str, default="这是一个语音克隆测试。", help="要合成的文本内容")
    parser.add_argument("--output", type=str, default=None, help="输出音频文件路径")
    
    args = parser.parse_args()
    
    # 直接测试语音克隆
    test_clone_voice_direct(args.audio, args.text, args.output)

if __name__ == "__main__":
    main() 