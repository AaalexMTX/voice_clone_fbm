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
            
            # 加载嵌入向量
            speaker_embedding = np.load(embedding_path)
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

@app.route('/api/clone', methods=['POST'])
def voice_clone_api():
    """
    一站式语音克隆API：接收wav文件路径+content内容，生成目标音频
    流程：
    1. wav+content生成说话人特征embedding
    2. embedding+预期需合成的text -> Transformer -> 生成Mel频谱
    3. Mel频谱 -> HIFI-GAN生成目标音频
    """
    # 解析请求数据
    data = request.json
    if not data:
        return jsonify({'error': '没有提供请求数据'}), 400
    
    # 检查必要参数
    if 'wav_path' not in data:
        return jsonify({'error': '缺少wav_path参数'}), 400
    if 'content' not in data:
        return jsonify({'error': '缺少content参数'}), 400
    if 'target_text' not in data:
        return jsonify({'error': '缺少target_text参数'}), 400
    
    wav_path = data.get('wav_path')
    content = data.get('content')
    target_text = data.get('target_text')
    
    # 可选参数
    tts_type = data.get('tts_type', 'transformer')  # 默认使用transformer
    vocoder_type = data.get('vocoder_type', 'hifigan')  # 默认使用hifigan
    
    try:
        # 检查音频文件是否存在
        if not os.path.exists(wav_path):
            return jsonify({'error': f'找不到音频文件: {wav_path}'}), 404
        
        # 生成唯一的输出ID和路径
        output_id = str(uuid.uuid4())
        output_path = str(OUTPUT_FOLDER / f"{output_id}.wav")
        
        logger.info(f"处理语音克隆请求: wav_path={wav_path}, content={content}, target_text={target_text}")
        
        # 步骤1: 提取说话人嵌入
        logger.info("步骤1: 提取说话人嵌入")
        speaker_embedding = voice_clone_system.extract_speaker_embedding(wav_path)
        
        # 保存嵌入向量（可选）
        embedding_path = str(OUTPUT_FOLDER / f"{output_id}_embedding.npy")
        np.save(embedding_path, speaker_embedding)
        
        # 步骤2和3: 合成语音（内部会使用Transformer生成Mel频谱，然后用HIFI-GAN生成音频）
        logger.info("步骤2和3: 使用Transformer生成Mel频谱并用HIFI-GAN生成音频")
        voice_clone_system.synthesize(target_text, speaker_embedding, output_path)
        
        return jsonify({
            'status': 'success',
            'message': '语音克隆完成',
            'output_id': output_id,
            'output_path': output_path,
            'embedding_path': embedding_path,
            'download_url': f"/api/audio/{output_id}"
        })
    except Exception as e:
        logger.error(f"语音克隆过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'语音克隆过程中出错: {str(e)}'}), 500

@app.route('/api/train/xvector', methods=['POST'])
def train_xvector_api():
    """
    训练X-Vector说话人编码器API
    
    请求参数:
    - data_dir: 训练数据目录，包含多个说话人子目录
    - epochs: 训练轮数（可选，默认100）
    - batch_size: 批次大小（可选，默认32）
    - embedding_dim: 嵌入向量维度（可选，默认512）
    - augment: 是否使用数据增强（可选，默认False）
    - save_dir: 模型保存目录（可选，默认model/data/checkpoints）
    """
    # 解析请求数据
    data = request.json
    if not data:
        return jsonify({'error': '没有提供请求数据'}), 400
    
    # 检查必要参数
    if 'data_dir' not in data:
        return jsonify({'error': '缺少data_dir参数'}), 400
    
    data_dir = data.get('data_dir')
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 32)
    embedding_dim = data.get('embedding_dim', 512)
    augment = data.get('augment', False)
    save_dir = data.get('save_dir', 'model/data/checkpoints')
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 检查训练模块是否可用
        import importlib
        try:
            speaker_encoder_module = importlib.import_module("model.speaker_encoder.train_xvector")
            train_xvector = speaker_encoder_module.train_xvector
            logger.info("成功导入X-Vector训练模块")
        except ImportError as e:
            logger.error(f"导入X-Vector训练模块失败: {str(e)}")
            return jsonify({'error': f'导入训练模块失败: {str(e)}'}), 500
        
        # 创建参数对象
        class Args:
            pass
        
        args = Args()
        args.data_dir = data_dir
        args.epochs = epochs
        args.batch_size = batch_size
        args.embedding_dim = embedding_dim
        args.augment = augment
        args.save_dir = save_dir
        args.train_ratio = 0.9
        args.ext = "wav"
        args.min_duration = 3.0
        args.max_duration = 8.0
        args.sample_rate = 16000
        args.mel_channels = 80
        args.lr = 0.001
        args.lr_step = 20
        args.no_cuda = False
        args.num_workers = 4
        args.save_freq = 10
        
        # 启动训练（在后台线程中运行）
        import threading
        
        def train_thread():
            try:
                logger.info(f"开始训练X-Vector模型: data_dir={data_dir}, epochs={epochs}")
                train_xvector(args)
                logger.info("X-Vector模型训练完成")
                
                # 复制模型到指定位置
                src_path = os.path.join(save_dir, "xvector_best.pt")
                dst_path = os.path.join("model/data/checkpoints", "xvector_encoder.pt")
                
                if os.path.exists(src_path):
                    import shutil
                    logger.info(f"复制模型: {src_path} -> {dst_path}")
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy(src_path, dst_path)
                    logger.info(f"已将最佳模型复制到: {dst_path}")
                else:
                    logger.warning(f"找不到训练好的模型: {src_path}")
                
                # 重新加载模型
                try:
                    global voice_clone_system
                    logger.info("重新初始化语音克隆系统...")
                    voice_clone_system = VoiceCloneSystem(
                        model_dir="model/data/checkpoints", 
                        device=None, 
                        tts_type="transformer"
                    )
                    logger.info("已重新加载语音克隆系统")
                except Exception as e:
                    logger.error(f"重新加载语音克隆系统时出错: {str(e)}")
            except Exception as e:
                logger.error(f"训练X-Vector模型时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 启动训练线程
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '已启动X-Vector模型训练',
            'save_dir': save_dir,
            'expected_model_path': os.path.join(save_dir, "xvector_best.pt"),
            'final_model_path': os.path.join("model/data/checkpoints", "xvector_encoder.pt")
        })
    except Exception as e:
        logger.error(f"启动X-Vector模型训练时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'启动X-Vector模型训练时出错: {str(e)}'}), 500

@app.route('/api/train/transformer_tts', methods=['POST'])
def train_transformer_tts_api():
    """
    训练Transformer TTS模型API
    
    请求参数:
    - train_metadata: 训练元数据文件路径
    - val_metadata: 验证元数据文件路径
    - mel_dir: 梅尔频谱目录（可选）
    - speaker_embed_dir: 说话人嵌入目录（可选）
    - epochs: 训练轮数（可选，默认100）
    - batch_size: 批次大小（可选，默认32）
    - d_model: 模型维度（可选，默认512）
    - speaker_dim: 说话人嵌入维度（可选，默认512）
    - checkpoint_dir: 检查点保存目录（可选，默认model/data/checkpoints）
    """
    # 解析请求数据
    data = request.json
    if not data:
        return jsonify({'error': '没有提供请求数据'}), 400
    
    # 检查必要参数
    if 'train_metadata' not in data:
        return jsonify({'error': '缺少train_metadata参数'}), 400
    if 'val_metadata' not in data:
        return jsonify({'error': '缺少val_metadata参数'}), 400
    
    train_metadata = data.get('train_metadata')
    val_metadata = data.get('val_metadata')
    mel_dir = data.get('mel_dir')
    speaker_embed_dir = data.get('speaker_embed_dir')
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 32)
    d_model = data.get('d_model', 512)
    speaker_dim = data.get('speaker_dim', 512)
    checkpoint_dir = data.get('checkpoint_dir', 'model/data/checkpoints')
    
    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # 检查训练模块是否可用
        import importlib
        try:
            text_to_mel_module = importlib.import_module("model.text_to_mel.train_transformer_tts")
            train = text_to_mel_module.train
            logger.info("成功导入Transformer TTS训练模块")
        except ImportError as e:
            logger.error(f"导入Transformer TTS训练模块失败: {str(e)}")
            return jsonify({'error': f'导入训练模块失败: {str(e)}'}), 500
        
        # 创建参数对象
        class Args:
            pass
        
        args = Args()
        args.train_metadata = train_metadata
        args.val_metadata = val_metadata
        args.mel_dir = mel_dir
        args.speaker_embed_dir = speaker_embed_dir
        args.epochs = epochs
        args.batch_size = batch_size
        args.d_model = d_model
        args.speaker_dim = speaker_dim
        args.checkpoint_dir = checkpoint_dir
        args.log_dir = os.path.join(checkpoint_dir, "logs")
        args.vocab_size = 256
        args.nhead = 8
        args.num_encoder_layers = 6
        args.num_decoder_layers = 6
        args.dim_feedforward = 2048
        args.dropout = 0.1
        args.mel_dim = 80
        args.max_seq_len = 1000
        args.learning_rate = 0.0001
        args.lr_decay_step = 50000
        args.lr_decay_gamma = 0.5
        args.grad_clip_thresh = 1.0
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.num_workers = 4
        args.save_step = 5000
        args.eval_step = 1000
        args.log_step = 100
        
        # 启动训练（在后台线程中运行）
        import threading
        
        def train_thread():
            try:
                logger.info(f"开始训练Transformer TTS模型: train_metadata={train_metadata}, epochs={epochs}")
                train(args)
                logger.info("Transformer TTS模型训练完成")
                
                # 复制模型到指定位置
                src_path = os.path.join(checkpoint_dir, "transformer_tts_best.pt")
                dst_path = os.path.join("model/data/checkpoints", "transformer_tts.pt")
                if os.path.exists(src_path):
                    import shutil
                    logger.info(f"复制模型: {src_path} -> {dst_path}")
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy(src_path, dst_path)
                    logger.info(f"已将最佳模型复制到: {dst_path}")
                else:
                    logger.warning(f"找不到训练好的模型: {src_path}")
                
                # 重新加载模型
                try:
                    global voice_clone_system
                    logger.info("重新初始化语音克隆系统...")
                    voice_clone_system = VoiceCloneSystem(
                        model_dir="model/data/checkpoints", 
                        device=None, 
                        tts_type="transformer"
                    )
                    logger.info("已重新加载语音克隆系统")
                except Exception as e:
                    logger.error(f"重新加载语音克隆系统时出错: {str(e)}")
            except Exception as e:
                logger.error(f"训练Transformer TTS模型时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 启动训练线程
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '已启动Transformer TTS模型训练',
            'checkpoint_dir': checkpoint_dir,
            'expected_model_path': os.path.join(checkpoint_dir, "transformer_tts_best.pt"),
            'final_model_path': os.path.join("model/data/checkpoints", "transformer_tts.pt")
        })
    except Exception as e:
        logger.error(f"启动Transformer TTS模型训练时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'启动Transformer TTS模型训练时出错: {str(e)}'}), 500

def init_model(model_dir, device):
    """初始化模型"""
    global voice_clone_system
    logger.info(f"正在加载语音克隆模型，模型目录: {model_dir}，设备: {device}")
    voice_clone_system = VoiceCloneSystem(model_dir=model_dir, device=device, tts_type="transformer")
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
    parser.add_argument("--model-dir", type=str, default="model/data/checkpoints", help="模型目录")
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