#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试语音克隆API接口
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

# API端点
API_URL = "http://localhost:5000"
API_PREFIX = "/api"

def test_health():
    """测试健康检查接口"""
    print("\n=== 测试健康检查接口 ===")
    
    try:
        response = requests.get(f"{API_URL}{API_PREFIX}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 请求失败: {str(e)}")
        return False

def test_clone_voice():
    """测试语音克隆接口"""
    print("\n=== 测试语音克隆接口 ===")
    
    # 测试音频文件
    audio_file = "data/root/source/dingzhen_8.wav"
    
    if not os.path.exists(audio_file):
        print(f"✗ 测试音频文件不存在: {audio_file}")
        return False
    
    # 测试文本
    text = "这是一个简单的语音克隆测试。"
    
    try:
        # 准备数据
        data = {
            'reference_audio': audio_file,
            'text': text
        }
        
        # 发送请求
        print(f"发送请求: 音频={audio_file}, 文本='{text}'")
        start_time = time.time()
        response = requests.post(f"{API_URL}{API_PREFIX}/clone_voice", json=data)
        elapsed_time = time.time() - start_time
        
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {elapsed_time:.2f}秒")
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            print(f"响应内容: {result}")
            
            if result.get('success'):
                output_path = result.get('output_path')
                print(f"✓ 成功生成语音: {output_path}")
                return True
            else:
                print(f"✗ 请求失败: {result.get('error', '未知错误')}")
                return False
        else:
            print(f"✗ 请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {str(e)}")
        return False

def test_cached_clone():
    """测试使用缓存的语音克隆接口"""
    print("\n=== 测试使用缓存的语音克隆接口 ===")
    
    # 测试音频文件
    audio_file = "data/root/source/dingzhen_8.wav"
    
    if not os.path.exists(audio_file):
        print(f"✗ 测试音频文件不存在: {audio_file}")
        return False
    
    # 测试文本
    text = "这是一个使用缓存的语音克隆测试。"
    
    try:
        # 准备数据
        data = {
            'reference_audio': audio_file,
            'text': text,
            'use_cache': True
        }
        
        # 发送请求
        print(f"发送请求: 音频路径={audio_file}, 文本='{text}'")
        start_time = time.time()
        response = requests.post(f"{API_URL}{API_PREFIX}/clone_voice", json=data)
        elapsed_time = time.time() - start_time
        
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {elapsed_time:.2f}秒")
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            print(f"响应内容: {result}")
            
            if result.get('success'):
                output_path = result.get('output_path')
                print(f"✓ 成功生成语音: {output_path}")
                return True
            else:
                print(f"✗ 请求失败: {result.get('error', '未知错误')}")
                return False
        else:
            print(f"✗ 请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试API接口...")
    
    # 测试健康检查接口
    health_success = test_health()
    
    # 测试语音克隆接口
    clone_success = test_clone_voice()
    
    # 测试使用缓存的语音克隆接口
    cached_success = test_cached_clone()
    
    # 打印总结
    print("\n=== 测试结果总结 ===")
    print(f"健康检查接口: {'✓ 成功' if health_success else '✗ 失败'}")
    print(f"语音克隆接口: {'✓ 成功' if clone_success else '✗ 失败'}")
    print(f"使用缓存的语音克隆接口: {'✓ 成功' if cached_success else '✗ 失败'}")
    
    if health_success and clone_success and cached_success:
        print("\n所有API测试都通过了！系统应该能够正常工作。")
    else:
        print("\n有些API测试失败了。请检查上面的错误信息。") 