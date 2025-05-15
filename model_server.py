#!/usr/bin/env python3
"""
语音克隆模型服务入口
提供REST API供Go后端调用
"""

import argparse
import os
from model.service import start_model_service

def main():
    parser = argparse.ArgumentParser(description="语音克隆模型服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--model-dir", default="saved_models", help="模型保存目录")
    parser.add_argument("--upload-dir", default="uploads", help="上传文件目录")
    parser.add_argument("--result-dir", default="results", help="结果文件目录")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["UPLOAD_DIR"] = args.upload_dir
    os.environ["RESULT_DIR"] = args.result_dir
    
    # 启动服务
    print(f"启动模型服务 - 地址: {args.host}:{args.port}")
    start_model_service(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 