#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试语音克隆API
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
import time

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.config import load_config, get_config

def test_health(base_url):
    """测试健康检查接口"""
    url = f"{base_url}/api/health"
    response = requests.get(url)
    print(f"健康检查: {response.status_code}")
    print(response.json())
    return response.status_code == 200

def test_clone_api(base_url, wav_path, content, target_text):
    """测试语音克隆API"""
    url = f"{base_url}/api/clone"
    
    # 准备请求数据
    data = {
        "wav_path": wav_path,
        "content": content,
        "target_text": target_text
    }
    
    # 发送请求
    print(f"发送请求到 {url}")
    print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    print(f"响应状态码: {response.status_code}")
    print(f"处理时间: {end_time - start_time:.2f}秒")
    
    try:
        result = response.json()
        print(f"响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if response.status_code == 200 and result.get('status') == 'success':
            output_path = result.get('output_path')
            embedding_path = result.get('embedding_path')
            
            print(f"语音克隆成功!")
            print(f"输出音频路径: {output_path}")
            print(f"嵌入向量路径: {embedding_path}")
            
            # 检查文件是否存在
            if os.path.exists(output_path):
                print(f"输出音频文件存在，大小: {os.path.getsize(output_path)} 字节")
            else:
                print(f"警告: 输出音频文件不存在!")
                
            if os.path.exists(embedding_path):
                print(f"嵌入向量文件存在，大小: {os.path.getsize(embedding_path)} 字节")
            else:
                print(f"警告: 嵌入向量文件不存在!")
                
            return True
        else:
            print(f"语音克隆失败: {result.get('error', '未知错误')}")
            return False
    except Exception as e:
        print(f"解析响应时出错: {str(e)}")
        print(f"原始响应: {response.text}")
        return False

def test_download_audio(base_url, output_id):
    """测试下载音频接口"""
    url = f"{base_url}/api/audio/{output_id}"
    
    print(f"尝试下载音频: {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        # 保存音频文件
        output_path = f"downloaded_{output_id}.wav"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"音频下载成功，保存到: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path)} 字节")
        return True
    else:
        print(f"音频下载失败: {response.status_code}")
        try:
            print(response.json())
        except:
            print(response.text)
        return False

def test_train_xvector(base_url, data_dir, epochs=100, batch_size=32, embedding_dim=512, augment=False):
    """测试训练X-Vector模型API"""
    url = f"{base_url}/api/train/xvector"
    
    # 准备请求数据
    data = {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "augment": augment
    }
    
    # 发送请求
    print(f"发送请求到 {url}")
    print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(url, json=data)
        
        print(f"响应状态码: {response.status_code}")
        
        try:
            result = response.json()
            print(f"响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            if response.status_code == 200 and result.get('status') == 'success':
                print(f"X-Vector模型训练启动成功!")
                print(f"预期模型路径: {result.get('expected_model_path')}")
                print(f"最终模型路径: {result.get('final_model_path')}")
                return True
            else:
                print(f"X-Vector模型训练启动失败: {result.get('error', '未知错误')}")
                return False
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"原始响应: {response.text}")
            return False
    except Exception as e:
        print(f"发送请求时出错: {str(e)}")
        return False

def test_train_transformer_tts(base_url, train_metadata, val_metadata, mel_dir=None, speaker_embed_dir=None, 
                              epochs=100, batch_size=32, d_model=512, speaker_dim=512):
    """测试训练Transformer TTS模型API"""
    url = f"{base_url}/api/train/transformer_tts"
    
    # 准备请求数据
    data = {
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "speaker_dim": speaker_dim
    }
    
    # 添加可选参数
    if mel_dir:
        data["mel_dir"] = mel_dir
    if speaker_embed_dir:
        data["speaker_embed_dir"] = speaker_embed_dir
    
    # 发送请求
    print(f"发送请求到 {url}")
    print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(url, json=data)
        
        print(f"响应状态码: {response.status_code}")
        
        try:
            result = response.json()
            print(f"响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            if response.status_code == 200 and result.get('status') == 'success':
                print(f"Transformer TTS模型训练启动成功!")
                print(f"预期模型路径: {result.get('expected_model_path')}")
                print(f"最终模型路径: {result.get('final_model_path')}")
                return True
            else:
                print(f"Transformer TTS模型训练启动失败: {result.get('error', '未知错误')}")
                return False
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"原始响应: {response.text}")
            return False
    except Exception as e:
        print(f"发送请求时出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试语音克隆API")
    parser.add_argument("--url", type=str, default="http://localhost:7860", help="API服务器URL")
    parser.add_argument("--wav", type=str, help="WAV文件路径")
    parser.add_argument("--content", type=str, default="这是参考音频的内容", help="参考音频的文本内容")
    parser.add_argument("--text", type=str, default="这是要合成的目标文本", help="要合成的目标文本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 添加训练相关参数
    parser.add_argument("--mode", type=str, choices=["clone", "train_xvector", "train_tts"], default="clone",
                       help="测试模式: clone=测试语音克隆, train_xvector=测试X-Vector训练, train_tts=测试TTS训练")
    parser.add_argument("--data_dir", type=str, help="X-Vector训练数据目录")
    parser.add_argument("--train_metadata", type=str, help="TTS训练元数据文件")
    parser.add_argument("--val_metadata", type=str, help="TTS验证元数据文件")
    parser.add_argument("--mel_dir", type=str, help="梅尔频谱目录")
    parser.add_argument("--speaker_embed_dir", type=str, help="说话人嵌入目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--embedding_dim", type=int, default=512, help="嵌入向量维度")
    parser.add_argument("--augment", action="store_true", help="是否使用数据增强")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        load_config(args.config)
    
    # 测试健康检查
    if not test_health(args.url):
        print("健康检查失败，退出测试")
        return
    
    # 根据模式选择测试功能
    if args.mode == "clone":
        if not args.wav:
            print("错误: 克隆模式需要指定--wav参数")
            return
            
        # 测试语音克隆API
        result = test_clone_api(args.url, args.wav, args.content, args.text)
        
        if result:
            print("语音克隆API测试成功!")
        else:
            print("语音克隆API测试失败!")
            
    elif args.mode == "train_xvector":
        if not args.data_dir:
            print("错误: X-Vector训练模式需要指定--data_dir参数")
            return
            
        # 测试X-Vector训练API
        result = test_train_xvector(
            args.url, 
            args.data_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            augment=args.augment
        )
        
        if result:
            print("X-Vector训练API测试成功!")
        else:
            print("X-Vector训练API测试失败!")
            
    elif args.mode == "train_tts":
        if not args.train_metadata or not args.val_metadata:
            print("错误: TTS训练模式需要指定--train_metadata和--val_metadata参数")
            return
            
        # 测试Transformer TTS训练API
        result = test_train_transformer_tts(
            args.url,
            args.train_metadata,
            args.val_metadata,
            mel_dir=args.mel_dir,
            speaker_embed_dir=args.speaker_embed_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.embedding_dim,
            speaker_dim=args.embedding_dim
        )
        
        if result:
            print("Transformer TTS训练API测试成功!")
        else:
            print("Transformer TTS训练API测试失败!")

if __name__ == "__main__":
    main() 