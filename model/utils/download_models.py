#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载语音克隆系统预训练模型
"""

import os
import sys
from huggingface_hub import hf_hub_download
import gdown
from tqdm import tqdm
import requests
import shutil

def download_file(url, dest_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

def download_xvector_model():
    """下载X-Vector模型"""
    print("正在下载X-Vector模型...")
    
    # 创建输出目录
    output_dir = "model/data/checkpoints/speaker_encoder"
    os.makedirs(output_dir, exist_ok=True)
    
    # 从HuggingFace下载预训练的X-Vector模型
    try:
        model_path = hf_hub_download(
            repo_id="speechbrain/spkrec-xvect-voxceleb", 
            filename="embedding_model.ckpt",
            local_dir=output_dir
        )
        config_path = hf_hub_download(
            repo_id="speechbrain/spkrec-xvect-voxceleb", 
            filename="hyperparams.yaml",
            local_dir=output_dir
        )
        
        # 重命名为更通用的名称
        target_model_path = os.path.join(output_dir, "xvector.ckpt")
        target_config_path = os.path.join(output_dir, "xvector_config.yaml")
        
        shutil.copy(model_path, target_model_path)
        shutil.copy(config_path, target_config_path)
        
        print(f"X-Vector模型已下载到: {target_model_path}")
        print(f"X-Vector配置已下载到: {target_config_path}")
        return True
    except Exception as e:
        print(f"下载X-Vector模型时出错: {str(e)}")
        return False

def download_transformer_tts_model():
    """下载Transformer TTS模型"""
    print("正在下载Transformer TTS模型...")
    
    # 创建输出目录
    output_dir = "model/data/checkpoints/transformer_tts"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用gdown从Google Drive下载预训练的Transformer TTS模型
    try:
        # 这里使用一个流行的开源Transformer TTS模型的Google Drive链接
        # 实际链接应替换为实际可用的模型链接
        file_id = "1qoocyCSRxZXgFM5FVQkgZ9xsLj9JKwvS"  # 示例ID，实际应替换为真实ID
        url = f"https://drive.google.com/uc?id={file_id}"
        
        output_path = os.path.join(output_dir, "transformer_tts.pth")
        gdown.download(url, output_path, quiet=False)
        
        print(f"Transformer TTS模型已下载到: {output_path}")
        
        # 也可以尝试从HuggingFace下载替代模型
        try:
            model_path = hf_hub_download(
                repo_id="espnet/kan-bayashi_ljspeech_vits", 
                filename="train.loss.ave_5best.pth",
                local_dir=output_dir
            )
            config_path = hf_hub_download(
                repo_id="espnet/kan-bayashi_ljspeech_vits", 
                filename="config.yaml",
                local_dir=output_dir
            )
            
            print(f"备选TTS模型已下载到: {model_path}")
            print(f"备选TTS配置已下载到: {config_path}")
        except Exception as e:
            print(f"下载备选TTS模型时出错: {str(e)}")
        
        return True
    except Exception as e:
        print(f"下载Transformer TTS模型时出错: {str(e)}")
        
        # 尝试从GitHub下载替代模型
        try:
            github_url = "https://github.com/mozilla/TTS/releases/download/v0.6.1/tts_models--en--ljspeech--tacotron2-DDC.zip"
            output_zip = os.path.join(output_dir, "tts_model.zip")
            download_file(github_url, output_zip)
            
            print(f"已下载备选TTS模型到: {output_zip}")
            # 这里可以添加解压代码
            return True
        except Exception as e2:
            print(f"下载备选TTS模型时出错: {str(e2)}")
            return False

if __name__ == "__main__":
    # 安装必要的依赖
    try:
        import gdown
        import huggingface_hub
    except ImportError:
        print("安装必要的依赖...")
        os.system(f"{sys.executable} -m pip install gdown huggingface_hub tqdm requests")
        import gdown
        import huggingface_hub
    
    # 下载模型
    print("开始下载预训练模型...")
    
    x_vector_success = download_xvector_model()
    transformer_success = download_transformer_tts_model()
    
    if x_vector_success and transformer_success:
        print("所有模型下载完成！")
    else:
        if not x_vector_success:
            print("X-Vector模型下载失败")
        if not transformer_success:
            print("Transformer TTS模型下载失败") 