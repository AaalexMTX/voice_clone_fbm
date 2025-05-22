#!/usr/bin/env python3
"""
语音克隆系统安装脚本
"""

import os
import sys
from setuptools import setup, find_packages

# 获取requirements.txt内容
def get_requirements():
    # 读取model目录下的requirements.txt文件
    req_file = os.path.join('model', 'requirements.txt')
    requirements = []
    if os.path.exists(req_file):
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                requirements.append(line)
    return requirements

setup(
    name="voice-clone",
    version="0.1.0",
    description="基于Transformer的语音克隆系统",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements(),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'voice-clone=model.__main__:main',
        ],
    },
) 