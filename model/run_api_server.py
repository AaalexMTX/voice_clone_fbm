#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆API服务器启动脚本
用于启动语音克隆HTTP服务
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return {}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--config", type=str, default="model/config/config.yaml", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    # 设置服务器配置
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "0.0.0.0")
    port = args.port or server_config.get("port", 5000)
    
    # 设置模型配置
    model_config = config.get("model", {})
    output_dir = model_config.get("output_dir", "outputs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 导入服务器模块
    try:
        from model.api.server import init_server, start_server
        
        # 设置配置
        server_config = {
            "output_dir": output_dir,
            "temp_dir": model_config.get("temp_dir", "temp"),
            "model_dir": model_config.get("model_dir", "model/data/checkpoints"),
            "cache_dir": model_config.get("cache_dir", "model/data/cache"),
            "max_text_length": model_config.get("max_text_length", 500),
            "allowed_audio_formats": model_config.get("allowed_audio_formats", ["wav", "mp3", "flac", "ogg"])
        }
        
        # 初始化服务器
        app = init_server(server_config)
        
        # 启动服务器
        logger.info(f"启动语音克隆API服务器 - http://{host}:{port}/")
        app.run(host=host, port=port, debug=args.debug)
    except Exception as e:
        logger.error(f"启动服务器失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 