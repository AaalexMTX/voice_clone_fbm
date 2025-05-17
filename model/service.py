import os
import torch
import numpy as np
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from .model import VoiceCloneModel
from .encoder import preprocess_wav, save_wav, SpeakerEncoder
from .decoder import SimpleVocoder

app = Flask(__name__)
CORS(app)

# 模型实例
voice_clone_model = None
speaker_encoder = None
vocoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 任务状态储存
tasks = {}

# 模型路径配置
MODEL_DIR = os.environ.get("MODEL_DIR", "saved_models")
VOICE_CLONE_MODEL_PATH = os.path.join(MODEL_DIR, "voice_clone_model.pt")
SPEAKER_ENCODER_PATH = os.path.join(MODEL_DIR, "speaker_encoder.pt")
VOCODER_PATH = os.path.join(MODEL_DIR, "vocoder.pt")

# 数据路径配置
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
RESULT_DIR = os.environ.get("RESULT_DIR", "results")

def load_models():
    """加载所有需要的模型"""
    global voice_clone_model, speaker_encoder, vocoder
    
    # 创建模型实例
    voice_clone_model = VoiceCloneModel().to(device)
    speaker_encoder = SpeakerEncoder().to(device)
    vocoder = SimpleVocoder().to(device)
    
    # 加载预训练参数（如果存在）
    if os.path.exists(VOICE_CLONE_MODEL_PATH):
        voice_clone_model.load_state_dict(torch.load(VOICE_CLONE_MODEL_PATH, map_location=device))
        print(f"加载语音克隆模型: {VOICE_CLONE_MODEL_PATH}")
    
    if os.path.exists(SPEAKER_ENCODER_PATH):
        speaker_encoder.load_state_dict(torch.load(SPEAKER_ENCODER_PATH, map_location=device))
        print(f"加载说话人编码器: {SPEAKER_ENCODER_PATH}")
    
    if os.path.exists(VOCODER_PATH):
        vocoder.load_state_dict(torch.load(VOCODER_PATH, map_location=device))
        print(f"加载声码器: {VOCODER_PATH}")
    
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
        print(f"任务处理失败: {str(e)}")
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
    """启动模型服务"""
    load_models()
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_model_service(debug=True) 