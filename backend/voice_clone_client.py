#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆API客户端示例
展示如何从后端调用语音克隆API接口
"""

import os
import sys
import time
import json
import base64
import requests
import argparse
from pathlib import Path

class VoiceCloneClient:
    """语音克隆API客户端"""
    
    def __init__(self, api_url="http://localhost:5000"):
        """
        初始化客户端
        
        参数:
            api_url: API服务器地址
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """健康检查"""
        url = f"{self.api_url}/api/health"
        response = self.session.get(url)
        return response.json()
    
    def extract_embedding(self, audio_path, force_recompute=False):
        """
        提取说话人特征
        
        参数:
            audio_path: 音频文件路径
            force_recompute: 是否强制重新计算
            
        返回:
            响应JSON
        """
        url = f"{self.api_url}/api/extract_embedding"
        data = {
            "reference_audio": audio_path,
            "force_recompute": force_recompute
        }
        response = self.session.post(url, json=data)
        return response.json()
    
    def clone_voice(self, reference_audio, text, output_format="wav", use_cache=True):
        """
        克隆语音
        
        参数:
            reference_audio: 参考音频路径
            text: 合成文本
            output_format: 输出格式
            use_cache: 是否使用缓存
            
        返回:
            响应JSON
        """
        url = f"{self.api_url}/api/clone_voice"
        data = {
            "reference_audio": reference_audio,
            "text": text,
            "output_format": output_format,
            "use_cache": use_cache
        }
        response = self.session.post(url, json=data)
        return response.json()
    
    def async_clone_voice(self, reference_audio, text, output_format="wav", use_cache=True):
        """
        异步克隆语音
        
        参数:
            reference_audio: 参考音频路径
            text: 合成文本
            output_format: 输出格式
            use_cache: 是否使用缓存
            
        返回:
            响应JSON
        """
        url = f"{self.api_url}/api/async_clone_voice"
        data = {
            "reference_audio": reference_audio,
            "text": text,
            "output_format": output_format,
            "use_cache": use_cache
        }
        response = self.session.post(url, json=data)
        return response.json()
    
    def get_task_status(self, task_id):
        """
        获取任务状态
        
        参数:
            task_id: 任务ID
            
        返回:
            响应JSON
        """
        url = f"{self.api_url}/api/task/{task_id}"
        response = self.session.get(url)
        return response.json()
    
    def list_tasks(self, limit=10, status=None):
        """
        列出任务
        
        参数:
            limit: 最大返回数量
            status: 过滤状态
            
        返回:
            响应JSON
        """
        url = f"{self.api_url}/api/tasks"
        params = {"limit": limit}
        if status:
            params["status"] = status
        response = self.session.get(url, params=params)
        return response.json()
    
    def download_audio(self, filename, save_path=None):
        """
        下载音频
        
        参数:
            filename: 文件名
            save_path: 保存路径
            
        返回:
            保存路径
        """
        url = f"{self.api_url}/api/audio/{filename}"
        response = self.session.get(url)
        
        if response.status_code != 200:
            raise Exception(f"下载失败: {response.text}")
        
        if save_path is None:
            save_path = filename
        
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        return save_path
    
    def upload_reference(self, audio_path):
        """
        上传参考音频
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            响应JSON
        """
        url = f"{self.api_url}/api/upload_reference"
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            response = self.session.post(url, files=files)
        return response.json()
    
    def wait_for_task(self, task_id, polling_interval=1.0, timeout=300):
        """
        等待任务完成
        
        参数:
            task_id: 任务ID
            polling_interval: 轮询间隔
            timeout: 超时时间
            
        返回:
            任务状态
        """
        start_time = time.time()
        while True:
            status = self.get_task_status(task_id)
            
            if status.get("status") in ["completed", "failed"]:
                return status
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"等待任务超时: {task_id}")
            
            time.sleep(polling_interval)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆API客户端")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000", help="API服务器地址")
    parser.add_argument("--reference-audio", type=str, required=True, help="参考音频文件路径")
    parser.add_argument("--text", type=str, required=True, help="合成文本")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--format", type=str, default="wav", help="输出格式")
    parser.add_argument("--async", action="store_true", help="是否使用异步模式")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = VoiceCloneClient(api_url=args.api_url)
    
    # 检查服务健康状态
    health = client.health_check()
    print(f"服务健康状态: {health}")
    
    # 如果需要上传文件
    if not args.reference_audio.startswith(("http://", "https://", "/")):
        print(f"上传参考音频: {args.reference_audio}")
        upload_result = client.upload_reference(args.reference_audio)
        if upload_result.get("success"):
            reference_audio = upload_result["file_path"]
            print(f"上传成功，路径: {reference_audio}")
        else:
            print(f"上传失败: {upload_result.get('error')}")
            sys.exit(1)
    else:
        reference_audio = args.reference_audio
    
    # 克隆语音
    print(f"开始语音克隆: '{args.text}'")
    if getattr(args, "async", False):
        # 异步模式
        result = client.async_clone_voice(
            reference_audio=reference_audio,
            text=args.text,
            output_format=args.format,
            use_cache=not args.no_cache
        )
        
        if result.get("success"):
            task_id = result["task_id"]
            print(f"任务已提交，ID: {task_id}")
            
            # 等待任务完成
            print("等待任务完成...")
            task_result = client.wait_for_task(task_id)
            
            if task_result.get("status") == "completed":
                output_filename = task_result["output_filename"]
                print(f"语音克隆完成: {output_filename}")
                
                # 下载音频
                output_path = args.output or f"output_{int(time.time())}.{args.format}"
                client.download_audio(output_filename, output_path)
                print(f"已下载音频: {output_path}")
            else:
                print(f"任务失败: {task_result.get('error')}")
        else:
            print(f"提交任务失败: {result.get('error')}")
    else:
        # 同步模式
        result = client.clone_voice(
            reference_audio=reference_audio,
            text=args.text,
            output_format=args.format,
            use_cache=not args.no_cache
        )
        
        if result.get("success"):
            output_filename = result["output_filename"]
            print(f"语音克隆完成: {output_filename}")
            
            # 下载音频
            output_path = args.output or f"output_{int(time.time())}.{args.format}"
            client.download_audio(output_filename, output_path)
            print(f"已下载音频: {output_path}")
        else:
            print(f"语音克隆失败: {result.get('error')}")

if __name__ == "__main__":
    main() 