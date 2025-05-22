#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import argparse
import numpy as np
from pathlib import Path
import torch
import logging
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import soundfile as sf

from ..core.voice_clone import VoiceCloneSystem
from ..config import load_config, get_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局配置和目录变量
config = get_config()
UPLOAD_FOLDER = None
OUTPUT_FOLDER = None
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# 初始化语音克隆系统
voice_clone_system = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': voice_clone_system is not None
    })

@app.route('/api/extract_embedding', methods=['POST'])
def extract_embedding():
    """从音频提取说话人嵌入向量"""
    if 'audio' not in request.files:
        return jsonify({'error': '没有上传音频文件'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': '没有选择音频文件'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = str(UPLOAD_FOLDER / filename)
        file.save(filepath)
        
        try:
            # 提取说话人嵌入
            embedding = voice_clone_system.extract_speaker_embedding(filepath)
            
            # 保存嵌入文件
            embedding_filename = filename.rsplit('.', 1)[0] + '.npy'
            embedding_path = str(OUTPUT_FOLDER / embedding_filename)
            np.save(embedding_path, embedding)
            
            return jsonify({
                'status': 'success',
                'embedding_id': embedding_filename,
                'embedding_path': embedding_path
            })
        except Exception as e:
            logger.error(f"提取嵌入向量时出错: {str(e)}")
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500
        finally:
            # 清理上传文件
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/api/synthesize', methods=['POST'])
def synthesize_speech():
    """合成语音"""
    data = request.json
    
    if not data:
        return jsonify({'error': '没有提供请求数据'}), 400
    
    if 'text' not in data:
        return jsonify({'error': '缺少文本参数'}), 400
    
    text = data.get('text')
    embedding_id = data.get('embedding_id')
    reference_audio_id = data.get('reference_audio_id')
    
    if not embedding_id and not reference_audio_id:
        return jsonify({'error': '必须提供embedding_id或reference_audio_id'}), 400
    
    try:
        output_id = str(uuid.uuid4())
        output_path = str(OUTPUT_FOLDER / f"{output_id}.wav")
        
        if embedding_id:
            # 使用预先提取的嵌入向量
            embedding_path = str(OUTPUT_FOLDER / embedding_id)
            if not os.path.exists(embedding_path):
                return jsonify({'error': '找不到指定的嵌入向量文件'}), 404
            
            speaker_embedding = voice_clone_system.load_speaker_embedding(embedding_path)
            voice_clone_system.synthesize(text, speaker_embedding, output_path)
        else:
            # 使用参考音频
            reference_path = str(UPLOAD_FOLDER / reference_audio_id)
            if not os.path.exists(reference_path):
                return jsonify({'error': '找不到指定的参考音频文件'}), 404
            
            voice_clone_system.clone_voice(text, reference_path, output_path)
        
        return jsonify({
            'status': 'success',
            'output_id': output_id,
            'output_path': output_path
        })
    except Exception as e:
        logger.error(f"合成语音时出错: {str(e)}")
        return jsonify({'error': f'合成语音时出错: {str(e)}'}), 500

@app.route('/api/upload_reference', methods=['POST'])
def upload_reference():
    """上传参考音频"""
    if 'audio' not in request.files:
        return jsonify({'error': '没有上传音频文件'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': '没有选择音频文件'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = str(UPLOAD_FOLDER / filename)
        file.save(filepath)
        
        return jsonify({
            'status': 'success',
            'reference_audio_id': filename,
            'reference_path': filepath
        })
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/api/audio/<output_id>', methods=['GET'])
def get_audio(output_id):
    """获取生成的音频文件"""
    filepath = OUTPUT_FOLDER / f"{output_id}.wav"
    
    if not filepath.exists():
        return jsonify({'error': '找不到指定的音频文件'}), 404
    
    return send_file(str(filepath), mimetype='audio/wav')

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """一站式语音克隆API：上传参考音频并直接合成"""
    if 'audio' not in request.files:
        return jsonify({'error': '没有上传音频文件'}), 400
    
    if 'text' not in request.form:
        return jsonify({'error': '没有提供文本'}), 400
    
    file = request.files['audio']
    text = request.form['text']
    
    if file.filename == '':
        return jsonify({'error': '没有选择音频文件'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 保存上传的音频
            reference_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            reference_path = str(UPLOAD_FOLDER / reference_filename)
            file.save(reference_path)
            
            # 生成输出路径
            output_id = str(uuid.uuid4())
            output_path = str(OUTPUT_FOLDER / f"{output_id}.wav")
            
            # 执行语音克隆
            voice_clone_system.clone_voice(text, reference_path, output_path)
            
            # 清理参考音频文件
            if os.path.exists(reference_path):
                os.remove(reference_path)
            
            return jsonify({
                'status': 'success',
                'output_id': output_id,
                'output_url': f"/api/audio/{output_id}"
            })
        except Exception as e:
            logger.error(f"处理TTS请求时出错: {str(e)}")
            return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500
    
    return jsonify({'error': '不支持的文件类型'}), 400

def init_model(model_dir, device):
    """初始化模型"""
    global voice_clone_system
    logger.info(f"正在加载语音克隆模型，模型目录: {model_dir}，设备: {device}")
    voice_clone_system = VoiceCloneSystem(model_dir=model_dir, device=device)
    logger.info("模型加载完成")

def setup_dirs():
    """设置上传和输出目录"""
    global UPLOAD_FOLDER, OUTPUT_FOLDER
    
    # 设置上传和输出目录
    temp_dir = config["data"]["temp_dir"] or tempfile.gettempdir()
    UPLOAD_FOLDER = Path(config["data"]["upload_dir"]) if config["data"]["upload_dir"] else Path(temp_dir) / "voice_clone_uploads"
    OUTPUT_FOLDER = Path(config["data"]["output_dir"]) if config["data"]["output_dir"] else Path(temp_dir) / "voice_clone_outputs"
    
    # 确保目录存在
    UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音克隆API服务器")
    parser.add_argument("--host", type=str, default=config["server"]["host"], help="服务器主机地址")
    parser.add_argument("--port", type=int, default=config["server"]["port"], help="服务器端口")
    parser.add_argument("--model-dir", type=str, default="model/vocoder/models", help="模型目录")
    parser.add_argument("--device", type=str, default=None, help="设备 (cpu或cuda)")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    
    args = parser.parse_args()
    
    # 设置目录
    setup_dirs()
    
    # 初始化模型
    init_model(args.model_dir, args.device)
    
    # 启动服务器
    logger.info(f"启动服务器，地址: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 