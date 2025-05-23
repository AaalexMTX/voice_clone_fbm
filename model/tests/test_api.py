#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆API测试脚本
用于测试API服务器的各个接口
"""

import requests
import json
import os
import time
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

# API服务器地址
API_URL = "http://127.0.0.1:5000"

def test_health():
    """测试健康检查接口"""
    url = f"{API_URL}/api/health"
    response = requests.get(url)
    print(f"健康检查: {response.status_code}")
    print(response.json())
    return response.json()

def test_clone_voice(reference_audio, text, output_path=None):
    """测试语音克隆接口"""
    url = f"{API_URL}/api/clone_voice"
    
    # 如果没有指定输出路径，则使用项目根目录下的output文件夹
    if output_path is None:
        output_dir = project_root / "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"test_output_{int(time.time())}.wav"
    
    # 准备请求数据
    files = {
        'reference_audio': open(reference_audio, 'rb'),
    }
    
    data = {
        'text': text,
    }
    
    print(f"开始语音克隆请求...")
    print(f"参考音频: {reference_audio}")
    print(f"文本内容: {text}")
    print(f"输出路径: {output_path}")
    
    # 发送请求
    response = requests.post(url, files=files, data=data)
    
    print(f"语音克隆: {response.status_code}")
    
    # 如果请求成功，保存音频文件
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"音频已保存到: {output_path}")
    else:
        print(f"请求失败: {response.text}")
    
    return output_path if response.status_code == 200 else None

def test_async_clone_voice(reference_audio, text):
    """测试异步语音克隆接口"""
    url = f"{API_URL}/api/async_clone_voice"
    
    # 准备请求数据
    files = {
        'reference_audio': open(reference_audio, 'rb'),
    }
    
    data = {
        'text': text,
    }
    
    print(f"开始异步语音克隆请求...")
    print(f"参考音频: {reference_audio}")
    print(f"文本内容: {text}")
    
    # 发送请求
    response = requests.post(url, files=files, data=data)
    
    print(f"异步语音克隆提交: {response.status_code}")
    
    # 如果请求成功，获取任务ID
    if response.status_code == 202:
        result = response.json()
        task_id = result.get('task_id')
        print(f"任务ID: {task_id}")
        return task_id
    else:
        print(f"请求失败: {response.text}")
        return None

def test_get_task_status(task_id):
    """测试获取任务状态接口"""
    url = f"{API_URL}/api/task/{task_id}"
    
    print(f"获取任务状态: {task_id}")
    
    # 发送请求
    response = requests.get(url)
    
    print(f"任务状态: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    return response.json()

def test_get_task_result(task_id, output_path=None):
    """测试获取任务结果接口"""
    url = f"{API_URL}/api/audio/{task_id}"
    
    # 如果没有指定输出路径，则使用项目根目录下的output文件夹
    if output_path is None:
        output_dir = project_root / "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"async_output_{task_id}.wav"
    
    print(f"获取任务结果: {task_id}")
    print(f"输出路径: {output_path}")
    
    # 发送请求
    response = requests.get(url)
    
    print(f"任务结果: {response.status_code}")
    
    # 如果请求成功，保存音频文件
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"音频已保存到: {output_path}")
    else:
        print(f"请求失败: {response.text}")
    
    return output_path if response.status_code == 200 else None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆API测试脚本")
    parser.add_argument("--test", choices=["all", "health", "clone", "async"], 
                        default="all", help="要测试的接口")
    parser.add_argument("--audio", type=str, default=str(project_root / "data/samples/reference.wav"), 
                        help="参考音频文件路径")
    parser.add_argument("--text", type=str, default="这是一个语音克隆测试。", 
                        help="要合成的文本内容")
    parser.add_argument("--output", type=str, default=None, 
                        help="输出音频文件路径")
    parser.add_argument("--task-id", type=str, default=None, 
                        help="异步任务ID，用于获取任务状态和结果")
    
    args = parser.parse_args()
    
    # 测试健康检查接口
    if args.test in ["all", "health"]:
        test_health()
        print()
    
    # 测试同步语音克隆接口
    if args.test in ["all", "clone"]:
        if not os.path.exists(args.audio):
            print(f"错误: 参考音频文件不存在: {args.audio}")
        else:
            test_clone_voice(args.audio, args.text, args.output)
        print()
    
    # 测试异步语音克隆接口
    if args.test in ["all", "async"]:
        if args.task_id:
            # 如果提供了任务ID，则获取任务状态和结果
            status = test_get_task_status(args.task_id)
            if status.get("status") == "completed":
                test_get_task_result(args.task_id, args.output)
        else:
            # 否则提交新的异步任务
            if not os.path.exists(args.audio):
                print(f"错误: 参考音频文件不存在: {args.audio}")
            else:
                task_id = test_async_clone_voice(args.audio, args.text)
                if task_id:
                    print("等待5秒后检查任务状态...")
                    time.sleep(5)
                    status = test_get_task_status(task_id)
                    if status.get("status") == "completed":
                        test_get_task_result(task_id, args.output)
        print()

if __name__ == "__main__":
    main() 