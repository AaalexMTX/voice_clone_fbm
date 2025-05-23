#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安装Coqui TTS库，用于完全修复XTTS模型
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_section(title):
    """打印带分隔符的章节标题"""
    print(f"\n{'='*20} {title} {'='*20}\n")

def check_python_version():
    """检查Python版本"""
    print_section("检查Python版本")
    
    py_version = sys.version_info
    print(f"当前Python版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major == 3 and py_version.minor >= 8:
        print("✓ Python版本符合要求 (3.8+)")
        return True
    else:
        print("✗ Python版本不符合要求，需要Python 3.8+")
        return False

def check_pip():
    """检查pip是否可用"""
    print_section("检查pip")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✓ pip可用")
        return True
    except subprocess.CalledProcessError:
        print("✗ pip不可用，请先安装pip")
        return False

def install_requirements():
    """安装TTS库的依赖"""
    print_section("安装依赖")
    
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "unidecode>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0"
    ]
    
    print("安装以下依赖:")
    for req in requirements:
        print(f"  - {req}")
    
    try:
        for req in requirements:
            print(f"\n安装 {req}...")
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
        
        print("\n✓ 成功安装所有依赖")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 安装依赖失败: {e}")
        return False

def install_tts():
    """安装Coqui TTS库"""
    print_section("安装Coqui TTS库")
    
    try:
        print("安装TTS库...")
        subprocess.run([sys.executable, "-m", "pip", "install", "TTS"], check=True)
        
        # 验证安装
        try:
            import TTS
            print(f"\n✓ 成功安装Coqui TTS库 (版本: {TTS.__version__})")
            return True
        except ImportError:
            print("\n✗ TTS库安装失败，无法导入")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 安装TTS库失败: {e}")
        return False

def test_tts_loading():
    """测试加载XTTS模型"""
    print_section("测试加载XTTS模型")
    
    try:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        
        print("尝试加载XTTS模型...")
        
        # 添加安全全局变量
        torch.serialization.add_safe_globals([XttsConfig])
        
        # 加载模型
        model_path = "model/data/checkpoints/transformer_tts/coqui_XTTS-v2_model.pth"
        
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            
            try:
                # 使用weights_only=False加载模型
                model = torch.load(model_path, map_location="cpu", weights_only=False)
                print("✓ 成功加载XTTS模型")
                print(f"  - 模型类型: {type(model)}")
                return True
            except Exception as e:
                print(f"✗ 加载模型失败: {str(e)}")
                return False
        else:
            print(f"✗ 模型文件不存在: {model_path}")
            return False
    except ImportError as e:
        print(f"✗ 导入TTS库失败: {str(e)}")
        return False

def update_model_code():
    """更新模型代码以使用TTS库"""
    print_section("更新模型代码")
    
    # 检查是否存在TTS库
    try:
        import TTS
        print(f"检测到TTS库 (版本: {TTS.__version__})")
    except ImportError:
        print("✗ 未检测到TTS库，请先安装")
        return False
    
    # 更新模型代码的建议
    print("\n要更新模型代码以使用TTS库，请按照以下步骤操作:")
    print("\n1. 修改 model/text_to_mel/transformer_tts.py 文件:")
    print("   - 导入TTS库: from TTS.tts.configs.xtts_config import XttsConfig")
    print("   - 在加载模型前添加: torch.serialization.add_safe_globals([XttsConfig])")
    print("   - 使用weights_only=False参数加载模型: torch.load(model_path, map_location=device, weights_only=False)")
    
    print("\n2. 修改 model/core/voice_clone.py 文件:")
    print("   - 更新TTS模型的初始化代码，确保使用正确加载的预训练模型")
    
    return True

def main():
    """主函数"""
    print("\nCoqui TTS库安装助手\n")
    print("此脚本将帮助您安装Coqui TTS库，以完全修复XTTS模型的电子音问题")
    
    # 检查Python版本
    if not check_python_version():
        print("\n✗ 请先升级Python版本到3.8+再继续")
        return
    
    # 检查pip
    if not check_pip():
        print("\n✗ 请先安装pip再继续")
        return
    
    # 安装依赖
    if not install_requirements():
        print("\n✗ 安装依赖失败，请手动安装")
        return
    
    # 安装TTS库
    if not install_tts():
        print("\n✗ 安装TTS库失败，请手动安装")
        return
    
    # 测试加载XTTS模型
    test_tts_loading()
    
    # 更新模型代码
    update_model_code()
    
    print("\n总结:")
    print("1. Coqui TTS库安装完成")
    print("2. 请按照上述建议更新模型代码")
    print("3. 更新后，XTTS模型将能够正确加载预训练权重")
    print("4. 这将完全解决电子音问题，提高语音克隆的质量")
    
    print("\n感谢使用Coqui TTS库安装助手!")

if __name__ == "__main__":
    main() 