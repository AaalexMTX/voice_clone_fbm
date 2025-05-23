#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启动语音克隆API服务器
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from model.api.server import setup_dirs, init_model, app
from model.config import load_config, get_config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动语音克隆API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--model-dir", type=str, default="model/data/checkpoints", help="模型目录")
    parser.add_argument("--device", type=str, default=None, help="设备 (cpu或cuda)")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        load_config(args.config)
    
    config = get_config()
    
    # 设置目录
    setup_dirs()
    
    # 初始化模型
    print(f"正在加载模型，模型目录: {args.model_dir}")
    init_model(args.model_dir, args.device)
    
    # 启动服务器
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    debug = args.debug or config["server"]["debug"]
    
    print(f"启动服务器，地址: {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main() 