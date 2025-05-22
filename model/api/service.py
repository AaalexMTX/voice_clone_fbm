import os
import torch
import numpy as np
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from ..core.model import VoiceCloneModel
from ..speaker_encoder import preprocess_wav, SpeakerEncoder
from ..speaker_encoder.audio_processing import save_wav
from ..vocoder import Vocoder
from ..config import load_config, get_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 模型实例
voice_clone_model = None
speaker_encoder = None
vocoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 任务状态储存
tasks = {}

# 获取配置
config = get_config()

# 模型路径配置
MODEL_DIR = os.environ.get("MODEL_DIR", config["model"]["model_dir"])
VOICE_CLONE_MODEL_PATH = os.path.join(MODEL_DIR, "voice_clone_model.pt")
SPEAKER_ENCODER_PATH = os.path.join(MODEL_DIR, "speaker_encoder.pt")
VOCODER_PATH = os.path.join(MODEL_DIR, "vocoder.pt")

# 数据路径配置
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", config["data"]["upload_dir"])
RESULT_DIR = os.environ.get("RESULT_DIR", config["data"]["output_dir"])

def load_models():
    """加载所有需要的模型"""
    global voice_clone_model, speaker_encoder, vocoder
    
    # 创建模型实例
    voice_clone_model = VoiceCloneModel().to(device)
    speaker_encoder = SpeakerEncoder().to(device)
    vocoder = Vocoder().to(device)
    
    # 加载预训练参数（如果存在）
    if os.path.exists(VOICE_CLONE_MODEL_PATH):
        voice_clone_model.load_state_dict(torch.load(VOICE_CLONE_MODEL_PATH, map_location=device))
        logger.info(f"加载语音克隆模型: {VOICE_CLONE_MODEL_PATH}")
    
    if os.path.exists(SPEAKER_ENCODER_PATH):
        speaker_encoder.load_state_dict(torch.load(SPEAKER_ENCODER_PATH, map_location=device))
        logger.info(f"加载说话人编码器: {SPEAKER_ENCODER_PATH}")
    
    if os.path.exists(VOCODER_PATH):
        vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=device))
        logger.info(f"加载声码器: {VOCODER_PATH}")
    
    # 设置为评估模式
    voice_clone_model.eval()
    speaker_encoder.eval()
    vocoder.eval()

@app.route('/api/clone', methods=['POST'])
def clone_voice():
    """处理语音克隆请求"""
    try:
        # 解析JSON请求
        data = request.json
        task_id = data.get('task_id')
        audio_id = data.get('audio_id')
        text = data.get('text', '')  # 可选文本输入
        
        # 检查必要参数
        if not task_id or not audio_id:
            return jsonify({"error": "缺少必要参数"}), 400
        
        # 更新任务状态
        tasks[task_id] = {
            "status": "processing",
            "created_at": time.time(),
            "audio_id": audio_id
        }
        
        # 启动异步处理任务
        # 注意：在实际生产环境中，这应该在后台线程或队列中处理
        process_voice_clone_task(task_id, audio_id, text)
        
        return jsonify({"message": "任务已提交", "task_id": task_id})
        
    except Exception as e:
        logger.error(f"处理克隆请求时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_voice_clone_task(task_id, audio_id, text=''):
    """处理语音克隆任务"""
    try:
        # 构建输入音频路径
        audio_path = os.path.join(UPLOAD_DIR, f"{audio_id}.wav")
        
        # 确保结果目录存在
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        # 输出音频路径
        output_path = os.path.join(RESULT_DIR, f"{task_id}.wav")
        
        # 处理音频
        wav = preprocess_wav(audio_path)
        
        # 提取说话人特征
        with torch.no_grad():
            speaker_embedding = speaker_encoder.embed_utterance(wav)
            speaker_embedding_tensor = torch.tensor(speaker_embedding).unsqueeze(0).to(device)
            
            # 根据文本和说话人特征生成语音
            # 这里简化处理，实际应考虑文本处理
            # TODO: 实现文本到语音的处理逻辑
            
            # 临时：使用预设生成长度
            target_length = min(len(wav) // 256, 500)  # 限制生成长度
            
            # 生成梅尔频谱
            generated_mel = voice_clone_model.clone_voice(
                speaker_embedding_tensor,
                target_length
            )
            
            # 使用声码器将梅尔频谱转换为波形
            audio_output = vocoder(generated_mel).squeeze().cpu().numpy()
            
            # 保存结果
            save_wav(audio_output, output_path)
            
            # 更新任务状态
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result_path"] = output_path
            
    except Exception as e:
        logger.error(f"任务处理失败: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task = tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "created_at": task["created_at"]
    }
    
    # 如果任务已完成，添加结果路径
    if task["status"] == "completed" and "result_path" in task:
        response["result_path"] = task["result_path"]
    
    # 如果任务失败，添加错误信息
    if task["status"] == "failed" and "error" in task:
        response["error"] = task["error"]
    
    return jsonify(response)

@app.route('/test', methods=['POST'])
def test_endpoint():
    data = request.get_json()
    return jsonify({
        'status': 'success',
        'received_text': data.get('text', ''),
        'model_status': 'running'
    })

def start_model_service(host='0.0.0.0', port=5000, debug=False):
    """启动传统模型服务"""
    load_models()
    app.run(host=host, port=port, debug=debug)

def start_transformer_server(host='0.0.0.0', port=5001, model_dir='models', device=None):
    """启动基于Transformer的语音克隆服务器"""
    logger.info("正在启动基于Transformer的语音克隆服务器...")
    
    # 确定脚本路径
    from ..api import server
    
    # 确定设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载配置
    cfg = get_config()
    
    # 设置目录
    server.setup_dirs()
    
    # 初始化模型
    server.init_model(model_dir, device)
    
    # 启动服务器
    logger.info(f"启动服务器于 {host}:{port}")
    server.app.run(
        host=host, 
        port=port, 
        debug=cfg["server"]["debug"]
    )

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description="语音克隆服务")
    parser.add_argument("--type", type=str, choices=["legacy", "transformer"], default="transformer", 
                       help="服务类型: legacy (旧版) 或 transformer (新版)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器监听地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器监听端口")
    parser.add_argument("--model_dir", type=str, default="models", help="模型目录")
    parser.add_argument("--device", type=str, default=None, help="设备: cuda 或 cpu")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    load_config(args.config)
    
    if args.type == "legacy":
        # 启动传统服务
        start_model_service(host=args.host, port=args.port)
    else:
        # 启动新版Transformer服务
        start_transformer_server(
            host=args.host, 
            port=args.port, 
            model_dir=args.model_dir,
            device=args.device
        )
        
        # 让主线程等待，不要立即退出
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭服务器...") 