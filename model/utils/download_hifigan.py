#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载HiFi-GAN预训练模型的脚本
"""

import os
import sys
import requests
import zipfile
import argparse
from tqdm import tqdm
import gdown

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

def download_hifigan_models(model_type="LJ_V1", output_dir="model/vocoder/models"):
    """
    下载HiFi-GAN预训练模型
    
    参数:
        model_type: 模型类型，可选值: LJ_V1, LJ_V2, LJ_V3, VCTK_V1, VCTK_V2, VCTK_V3, UNIVERSAL_V1
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # Google Drive文件夹ID
    folder_id = "1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y"
    
    # 模型文件名映射
    model_files = {
        "LJ_V1": "g_02500000",
        "LJ_V2": "g_02500000",
        "LJ_V3": "g_02500000",
        "VCTK_V1": "g_02500000",
        "VCTK_V2": "g_02500000", 
        "VCTK_V3": "g_02500000",
        "UNIVERSAL_V1": "g_02500000",
        "LJ_FT_T2_V1": "g_02500000",
        "LJ_FT_T2_V2": "g_02500000",
        "LJ_FT_T2_V3": "g_02500000"
    }
    
    # 检查模型类型是否有效
    if model_type not in model_files:
        print(f"错误: 无效的模型类型 '{model_type}'")
        print(f"有效的模型类型: {', '.join(model_files.keys())}")
        return False
    
    # 下载模型文件
    print(f"下载 {model_type} 模型...")
    
    # 使用gdown下载整个模型文件夹
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=output_dir, quiet=False)
    
    # 移动文件到正确的位置
    source_dir = os.path.join(output_dir, model_type)
    if os.path.exists(source_dir):
        model_file = os.path.join(source_dir, model_files[model_type])
        if os.path.exists(model_file):
            # 复制模型文件到输出目录
            target_file = os.path.join(output_dir, "hifigan_vocoder.pt")
            with open(model_file, 'rb') as src, open(target_file, 'wb') as dst:
                dst.write(src.read())
            print(f"模型已保存到: {target_file}")
            
            # 复制config.json文件
            config_file = os.path.join(source_dir, "config.json")
            if os.path.exists(config_file):
                target_config = os.path.join(output_dir, "hifigan_config.json")
                with open(config_file, 'rb') as src, open(target_config, 'wb') as dst:
                    dst.write(src.read())
                print(f"配置文件已保存到: {target_config}")
            
            return True
    
    print("下载失败或文件不存在")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载HiFi-GAN预训练模型")
    parser.add_argument("--model_type", type=str, default="UNIVERSAL_V1", 
                        help="模型类型: LJ_V1, LJ_V2, LJ_V3, VCTK_V1, VCTK_V2, VCTK_V3, UNIVERSAL_V1, LJ_FT_T2_V1, LJ_FT_T2_V2, LJ_FT_T2_V3")
    parser.add_argument("--output_dir", type=str, default="model/vocoder/models", 
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 安装gdown（如果需要）
    try:
        import gdown
    except ImportError:
        print("安装gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    # 下载模型
    download_hifigan_models(args.model_type, args.output_dir) 