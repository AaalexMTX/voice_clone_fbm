#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆API服务器
提供REST API接口，使用Flask实现
"""

import os
import json
import base64
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

from typing import Dict, Any, Optional, Union

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from model.api.voice_clone_service import VoiceCloneService

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 全局服务实例
voice_clone_service = None

# 配置
CONFIG = {
    "output_dir": "outputs",
    "temp_dir": "temp",
    "model_dir": "model/data/checkpoints",
    "cache_dir": "model/data/cache",
    "max_text_length": 500,
    "allowed_audio_formats": ["wav", "mp3", "flac", "ogg"]
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "service": "voice_clone_api",
        "version": "1.0.0"
    })

@app.route('/api/extract_embedding', methods=['POST'])
def extract_embedding():
    """从参考音频中提取说话人特征"""
    try:
        # 获取请求数据
        data = request.json or {}
        
        # 验证参数
        if "reference_audio" not in data:
            return jsonify({"success": False, "error": "缺少参数: reference_audio"}), 400
        
        # 处理强制重新计算参数
        force_recompute = data.get("force_recompute", False)
        
        # 提取特征
        result = voice_clone_service.extract_speaker_embedding(
            data["reference_audio"],
            force_recompute=force_recompute
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"提取特征时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/clone_voice', methods=['POST'])
def clone_voice():
    """同步语音克隆接口"""
    try:
        # 获取请求数据
        data = request.json or {}
        
        # 验证必要参数
        if "reference_audio" not in data:
            return jsonify({"success": False, "error": "缺少参数: reference_audio"}), 400
        if "text" not in data:
            return jsonify({"success": False, "error": "缺少参数: text"}), 400
        
        # 检查文本长度
        if len(data["text"]) > CONFIG["max_text_length"]:
            return jsonify({
                "success": False, 
                "error": f"文本长度超过限制: {len(data['text'])}/{CONFIG['max_text_length']}"
            }), 400
        
        # 获取可选参数
        output_format = data.get("output_format", "wav")
        use_cache = data.get("use_cache", True)
        
        # 检查输出格式
        if output_format not in CONFIG["allowed_audio_formats"]:
            return jsonify({
                "success": False, 
                "error": f"不支持的输出格式: {output_format}，支持的格式: {CONFIG['allowed_audio_formats']}"
            }), 400
        
        # 执行语音克隆
        result = voice_clone_service.clone_voice(
            reference_audio=data["reference_audio"],
            text=data["text"],
            output_format=output_format,
            use_cache=use_cache
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"语音克隆时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/async_clone_voice', methods=['POST'])
def async_clone_voice():
    """异步语音克隆接口"""
    try:
        # 获取请求数据
        data = request.json or {}
        
        # 验证必要参数
        if "reference_audio" not in data:
            return jsonify({"success": False, "error": "缺少参数: reference_audio"}), 400
        if "text" not in data:
            return jsonify({"success": False, "error": "缺少参数: text"}), 400
        
        # 检查文本长度
        if len(data["text"]) > CONFIG["max_text_length"]:
            return jsonify({
                "success": False, 
                "error": f"文本长度超过限制: {len(data['text'])}/{CONFIG['max_text_length']}"
            }), 400
        
        # 获取可选参数
        output_format = data.get("output_format", "wav")
        use_cache = data.get("use_cache", True)
        
        # 检查输出格式
        if output_format not in CONFIG["allowed_audio_formats"]:
            return jsonify({
                "success": False, 
                "error": f"不支持的输出格式: {output_format}，支持的格式: {CONFIG['allowed_audio_formats']}"
            }), 400
        
        # 提交异步任务
        result = voice_clone_service.async_clone_voice(
            reference_audio=data["reference_audio"],
            text=data["text"],
            output_format=output_format,
            use_cache=use_cache
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"提交异步任务时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    try:
        result = voice_clone_service.get_task_status(task_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取任务状态时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """列出任务"""
    try:
        # 获取查询参数
        limit = request.args.get('limit', default=10, type=int)
        status = request.args.get('status', default=None, type=str)
        
        result = voice_clone_service.list_tasks(limit=limit, status=status)
        return jsonify(result)
    except Exception as e:
        logger.error(f"列出任务时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """获取生成的音频文件"""
    try:
        # 构建文件路径
        file_path = os.path.join(CONFIG["output_dir"], filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"文件不存在: {filename}"}), 404
        
        # 检查文件是否为音频文件
        ext = os.path.splitext(filename)[1].lower()[1:]
        if ext not in CONFIG["allowed_audio_formats"]:
            return jsonify({"success": False, "error": f"不支持的音频格式: {ext}"}), 400
        
        # 设置MIME类型
        mime_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg"
        }
        
        # 返回文件
        return send_file(file_path, mimetype=mime_types.get(ext, "audio/wav"))
    except Exception as e:
        logger.error(f"获取音频文件时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/upload_reference', methods=['POST'])
def upload_reference():
    """上传参考音频文件"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有上传文件"}), 400
        
        file = request.files['file']
        
        # 检查文件名
        if file.filename == '':
            return jsonify({"success": False, "error": "文件名为空"}), 400
        
        # 检查文件扩展名
        ext = os.path.splitext(file.filename)[1].lower()[1:]
        if ext not in CONFIG["allowed_audio_formats"]:
            return jsonify({
                "success": False, 
                "error": f"不支持的音频格式: {ext}，支持的格式: {CONFIG['allowed_audio_formats']}"
            }), 400
        
        # 保存文件
        temp_dir = Path(CONFIG["temp_dir"])
        os.makedirs(temp_dir, exist_ok=True)
        
        import uuid
        filename = f"upload_{uuid.uuid4()}.{ext}"
        file_path = temp_dir / filename
        
        file.save(file_path)
        
        return jsonify({
            "success": True,
            "file_path": str(file_path),
            "filename": filename,
            "message": "文件上传成功"
        })
    except Exception as e:
        logger.error(f"上传文件时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

def init_server(config=None):
    """初始化服务器"""
    global voice_clone_service, CONFIG
    
    # 更新配置
    if config:
        CONFIG.update(config)
    
    # 创建必要的目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["temp_dir"], exist_ok=True)
    
    # 初始化服务
    voice_clone_service = VoiceCloneService(CONFIG)
    
    logger.info("语音克隆API服务器初始化完成")
    return app

def start_server(host="0.0.0.0", port=5000, debug=False):
    """启动服务器"""
    # 确保服务已初始化
    global voice_clone_service
    if voice_clone_service is None:
        init_server()
    
    # 启动服务器
    logger.info(f"启动语音克隆API服务器 - http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_server(debug=True) 