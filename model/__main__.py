#!/usr/bin/env python3
"""
语音克隆系统主入口模块
支持直接使用 python -m model 命令运行
"""

import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入子模块
from .api import app, init_model, setup_dirs, start_model_service
from .api.test_api import main as test_main
from .config import load_config, get_config, save_config
from .core.inference import run_inference

def server_main():
    """启动服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="语音克隆模型服务")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--host", help="服务主机地址")
    parser.add_argument("--port", type=int, help="服务端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--model-dir", help="模型保存目录")
    parser.add_argument("--type", choices=["default"], default="default", help="服务类型: 默认使用统一服务")
    
    args = parser.parse_args(sys.argv[2:])
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    debug = args.debug or config["server"]["debug"]
    model_dir = args.model_dir or config["model"]["model_dir"]
    device = config["model"]["device"]
    
    # 启动统一服务
    # 设置目录
    setup_dirs()
    
    # 初始化模型
    init_model(model_dir, device)
    
    # 启动服务器
    logger.info(f"启动服务器于 {host}:{port}")
    app.run(
        host=host, 
        port=port, 
        debug=debug
    )

def init_workspace():
    """初始化工作空间"""
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化语音克隆系统工作空间")
    parser.add_argument("--config", type=str, default="model/config/config.yaml", help="配置文件路径")
    
    args = parser.parse_args(sys.argv[2:])
    
    # 保存配置
    save_config(args.config)
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建必要的目录
    dirs = [
        config["model"]["model_dir"],
        config["data"]["upload_dir"],
        config["data"]["output_dir"],
        config["test"]["output_dir"],
    ]
    
    # 确保model/tests目录存在（用于示例音频）
    if not config["test"]["audio"].startswith("model/tests"):
        dirs.append("model/tests")
    
    for dir_path in dirs:
        if dir_path:
            Path(dir_path).mkdir(exist_ok=True, parents=True)
            logger.info(f"创建目录: {dir_path}")
    
    logger.info("工作空间初始化完成")

def inference_main():
    """推理入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="语音克隆推理工具")
    parser.add_argument("--text", type=str, required=True, help="要合成的文本")
    parser.add_argument("--reference_audio", type=str, help="参考音频文件路径")
    parser.add_argument("--embedding_file", type=str, help="预先提取的说话人嵌入文件路径 (.npy)")
    parser.add_argument("--output_file", type=str, default="output.wav", help="输出音频文件路径")
    parser.add_argument("--save_embedding", action="store_true", help="是否保存提取的说话人嵌入")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args(sys.argv[2:])
    
    # 加载配置
    load_config(args.config)
    
    # 检查参数
    if args.embedding_file is None and args.reference_audio is None:
        parser.error("必须提供--embedding_file或--reference_audio")
    
    # 运行推理
    run_inference(
        text=args.text,
        reference_audio=args.reference_audio,
        embedding_file=args.embedding_file,
        output_file=args.output_file,
        save_embedding=args.save_embedding
    )

def download_vocoders(args):
    """下载声码器模型"""
    import os
    import requests
    import shutil
    from pathlib import Path
    from tqdm import tqdm
    
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 预定义的模型URL
    model_urls = {
        "hifigan": {
            "model": "https://github.com/jik876/hifi-gan/releases/download/v1/g_02500000.pt",
            "config": "https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json"
        },
        "universal": {
            "model": "https://github.com/jik876/hifi-gan/releases/download/universal_v1/g_02500000.pt",
            "config": "https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json"
        }
    }
    
    def download_file(url, dest_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
    
    # 下载模型
    if args.hifigan:
        print("下载HiFi-GAN声码器...")
        model_path = output_dir / "hifigan_vocoder.pt"
        config_path = output_dir / "hifigan_config.json"
        
        download_file(model_urls["hifigan"]["model"], model_path)
        download_file(model_urls["hifigan"]["config"], config_path)
        
        print(f"HiFi-GAN声码器已下载到: {model_path}")
    
    if args.universal:
        print("下载Universal HiFi-GAN声码器...")
        model_path = output_dir / "universal_hifigan.pt"
        config_path = output_dir / "universal_hifigan_config.json"
        
        download_file(model_urls["universal"]["model"], model_path)
        download_file(model_urls["universal"]["config"], config_path)
        
        print(f"Universal HiFi-GAN声码器已下载到: {model_path}")
    
    if not (args.hifigan or args.universal):
        print("请指定要下载的声码器类型 (--hifigan, --universal)")

def test_vocoder(args):
    """测试声码器模型"""
    import torch
    import numpy as np
    import soundfile as sf
    from .vocoder.manager import VocoderManager
    from .config import get_config
    
    config = get_config()
    
    if args.mel_file is None:
        print("错误: 必须提供梅尔频谱文件 (--mel-file)")
        return
    
    try:
        # 加载梅尔频谱
        mel_spectrogram = np.load(args.mel_file)
        mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0)
        
        # 初始化声码器管理器
        manager = VocoderManager(config["model"]["model_dir"])
        manager.load_vocoder(args.type)
        
        # 生成波形
        waveform = manager.generate_waveform(mel_tensor)
        waveform = waveform.squeeze().cpu().numpy()
        
        # 保存波形
        sf.write(args.output_file, waveform, config.get("sample_rate", 22050))
        
        print(f"测试完成，音频已保存到: {args.output_file}")
    except Exception as e:
        print(f"测试失败: {str(e)}")

def vocoder_main():
    """声码器操作入口"""
    from .vocoder.manager import VocoderManager
    from .config import get_config
    
    config = get_config()
    vocoder_command = sys.argv[2] if len(sys.argv) > 2 else None
    
    if vocoder_command == "list":
        # 解析参数
        parser = argparse.ArgumentParser(description="列出已下载的声码器")
        parser.add_argument("--model-dir", type=str, default=config["model"]["model_dir"], help="模型目录路径")
        args = parser.parse_args(sys.argv[3:])
        
        # 列出声码器
        vocoders = VocoderManager.list_available_vocoders(args.model_dir)
        
        # 打印结果
        if vocoders:
            print(f"找到 {len(vocoders)} 个声码器模型:")
            for i, vocoder in enumerate(vocoders, 1):
                print(f"{i}. 类型: {vocoder['type']}")
                print(f"   模型路径: {vocoder['model_path']}")
                print(f"   配置路径: {vocoder['config_path'] or '无'}")
                print()
        else:
            print("未找到任何声码器模型")
    
    elif vocoder_command == "download":
        # 解析参数
        parser = argparse.ArgumentParser(description="下载声码器模型")
        parser.add_argument("--hifigan", action="store_true", help="下载HiFi-GAN声码器")
        parser.add_argument("--universal", action="store_true", help="下载Universal HiFi-GAN声码器")
        parser.add_argument("--output-dir", type=str, default=config["model"]["model_dir"], help="下载输出目录")
        args = parser.parse_args(sys.argv[3:])
        
        # 下载声码器
        download_vocoders(args)
    
    elif vocoder_command == "test":
        # 解析参数
        parser = argparse.ArgumentParser(description="测试声码器")
        parser.add_argument("--type", type=str, default="auto", help="声码器类型")
        parser.add_argument("--mel-file", type=str, help="梅尔频谱文件路径")
        parser.add_argument("--output-file", type=str, default="test_output.wav", help="输出音频文件")
        args = parser.parse_args(sys.argv[3:])
        
        # 测试声码器
        test_vocoder(args)
    
    else:
        print("用法: python -m model vocoder {list|download|test}")
        print("命令:")
        print("  list       列出已下载的声码器")
        print("  download   下载声码器模型")
        print("  test       测试声码器")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆系统")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # server命令 - 启动服务器
    server_parser = subparsers.add_parser("server", help="启动服务器")
    
    # test命令 - 运行API测试
    test_parser = subparsers.add_parser("test", help="运行API测试")
    
    # init命令 - 初始化工作空间
    init_parser = subparsers.add_parser("init", help="初始化工作空间")
    
    # inference命令 - 运行推理
    inference_parser = subparsers.add_parser("inference", help="运行推理")
    
    # vocoder命令 - 声码器操作
    vocoder_parser = subparsers.add_parser("vocoder", help="声码器管理")
    
    # 解析命令
    if len(sys.argv) < 2:
        parser.print_help()
        return
    
    # 执行对应命令
    if sys.argv[1] == "server":
        server_main()
    elif sys.argv[1] == "test":
        test_main()
    elif sys.argv[1] == "init":
        init_workspace()
    elif sys.argv[1] == "inference":
        inference_main()
    elif sys.argv[1] == "vocoder":
        vocoder_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 