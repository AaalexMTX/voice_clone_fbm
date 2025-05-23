#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音克隆API服务实现
提供REST API接口，用于语音克隆服务
"""

import os
import time
import json
import uuid
import tempfile
import base64
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 添加到Python路径

from model.core.voice_clone import VoiceCloneSystem

class VoiceCloneService:
    """语音克隆服务类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化语音克隆服务
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        # 设置输出目录
        self.output_dir = Path(self.config.get("output_dir", "outputs"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置临时目录
        self.temp_dir = Path(self.config.get("temp_dir", "temp"))
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 初始化语音克隆系统
        self.voice_clone_system = VoiceCloneSystem(config)
        
        # 跟踪任务
        self.tasks = {}
        
        print("语音克隆服务初始化完成")
    
    def extract_speaker_embedding(self, audio_path: str, force_recompute: bool = False) -> Dict[str, Any]:
        """
        从音频中提取说话人特征
        
        参数:
            audio_path: 音频文件路径
            force_recompute: 是否强制重新计算
            
        返回:
            包含说话人特征的字典
        """
        try:
            # 验证文件存在
            if not os.path.exists(audio_path):
                return {"success": False, "error": f"音频文件不存在: {audio_path}"}
            
            # 提取特征
            embedding = self.voice_clone_system.extract_speaker_embedding(audio_path, force_recompute)
            
            # 返回结果
            return {
                "success": True,
                "embedding_dim": embedding.shape[0],
                "message": "特征提取成功"
            }
        except Exception as e:
            return {"success": False, "error": f"特征提取失败: {str(e)}"}
    
    def _process_audio_input(self, audio_input: Union[str, Dict[str, Any]]) -> str:
        """
        处理音频输入，支持文件路径、Base64编码和临时上传
        
        参数:
            audio_input: 音频输入(文件路径或包含Base64数据的字典)
            
        返回:
            处理后的音频文件路径
        """
        # 如果是字符串，则视为文件路径
        if isinstance(audio_input, str):
            if os.path.exists(audio_input):
                return audio_input
            else:
                raise ValueError(f"音频文件不存在: {audio_input}")
        
        # 如果是字典，则处理Base64数据
        elif isinstance(audio_input, dict) and "audio_data" in audio_input:
            # 生成临时文件路径
            temp_file = self.temp_dir / f"temp_audio_{uuid.uuid4()}.wav"
            
            # 解码并保存Base64数据
            audio_data = base64.b64decode(audio_input["audio_data"])
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            return str(temp_file)
        else:
            raise ValueError("无效的音频输入格式")
    
    def clone_voice(self, 
                   reference_audio: Union[str, Dict[str, Any]], 
                   text: str,
                   output_format: str = "wav",
                   use_cache: bool = True) -> Dict[str, Any]:
        """
        克隆语音
        
        参数:
            reference_audio: 参考音频(文件路径或Base64编码)
            text: 目标文本
            output_format: 输出格式，默认为wav
            use_cache: 是否使用缓存的说话人特征
            
        返回:
            包含克隆结果的字典
        """
        try:
            start_time = time.time()
            
            # 处理参考音频
            audio_path = self._process_audio_input(reference_audio)
            
            # 生成输出文件名
            task_id = str(uuid.uuid4())
            timestamp = int(time.time())
            output_filename = f"voice_clone_{timestamp}.{output_format}"
            output_path = str(self.output_dir / output_filename)
            
            # 执行语音克隆
            output_path, audio = self.voice_clone_system.clone_voice(
                reference_audio=audio_path,
                text=text,
                use_cache=use_cache,
                output_path=output_path
            )
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 返回结果
            result = {
                "success": True,
                "task_id": task_id,
                "output_path": output_path,
                "output_filename": os.path.basename(output_path),
                "text": text,
                "process_time": f"{process_time:.2f}秒",
                "message": "语音克隆成功"
            }
            
            # 如果临时文件，则清理
            if audio_path.startswith(str(self.temp_dir)):
                os.remove(audio_path)
            
            return result
        except Exception as e:
            import traceback
            error_msg = f"语音克隆失败: {str(e)}\n{traceback.format_exc()}"
            return {"success": False, "error": error_msg}
    
    def async_clone_voice(self, 
                        reference_audio: Union[str, Dict[str, Any]], 
                        text: str,
                        output_format: str = "wav",
                        use_cache: bool = True) -> Dict[str, Any]:
        """
        异步克隆语音
        
        参数:
            reference_audio: 参考音频(文件路径或Base64编码)
            text: 目标文本
            output_format: 输出格式，默认为wav
            use_cache: 是否使用缓存的说话人特征
            
        返回:
            包含任务ID的字典
        """
        try:
            # 生成任务ID
            task_id = str(uuid.uuid4())
            
            # 处理参考音频
            audio_path = self._process_audio_input(reference_audio)
            
            # 初始化任务状态
            self.tasks[task_id] = {
                "status": "pending",
                "created_at": time.time(),
                "text": text,
                "reference_audio": audio_path,
                "output_format": output_format,
                "result": None
            }
            
            # 启动异步任务
            import threading
            thread = threading.Thread(
                target=self._process_async_task,
                args=(task_id, audio_path, text, output_format, use_cache)
            )
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "task_id": task_id,
                "status": "pending",
                "message": "语音克隆任务已提交"
            }
        except Exception as e:
            return {"success": False, "error": f"提交任务失败: {str(e)}"}
    
    def _process_async_task(self, 
                           task_id: str, 
                           audio_path: str, 
                           text: str, 
                           output_format: str,
                           use_cache: bool):
        """处理异步任务"""
        try:
            # 更新任务状态
            self.tasks[task_id]["status"] = "processing"
            
            # 生成输出文件名
            timestamp = int(time.time())
            output_filename = f"voice_clone_{timestamp}.{output_format}"
            output_path = str(self.output_dir / output_filename)
            
            # 执行语音克隆
            output_path, _ = self.voice_clone_system.clone_voice(
                reference_audio=audio_path,
                text=text,
                use_cache=use_cache,
                output_path=output_path
            )
            
            # 更新任务结果
            self.tasks[task_id].update({
                "status": "completed",
                "completed_at": time.time(),
                "result": {
                    "output_path": output_path,
                    "output_filename": os.path.basename(output_path)
                }
            })
            
            # 如果是临时文件，则清理
            if audio_path.startswith(str(self.temp_dir)):
                os.remove(audio_path)
                
        except Exception as e:
            # 更新任务状态为失败
            self.tasks[task_id].update({
                "status": "failed",
                "completed_at": time.time(),
                "error": str(e)
            })
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        参数:
            task_id: 任务ID
            
        返回:
            任务状态字典
        """
        if task_id not in self.tasks:
            return {"success": False, "error": "任务不存在"}
        
        task = self.tasks[task_id]
        result = {
            "success": True,
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"],
            "text": task["text"]
        }
        
        # 如果任务已完成，添加结果
        if task["status"] == "completed":
            result.update({
                "completed_at": task["completed_at"],
                "output_path": task["result"]["output_path"],
                "output_filename": task["result"]["output_filename"],
                "process_time": f"{task['completed_at'] - task['created_at']:.2f}秒"
            })
        # 如果任务失败，添加错误信息
        elif task["status"] == "failed":
            result.update({
                "completed_at": task["completed_at"],
                "error": task["error"]
            })
        
        return result
    
    def list_tasks(self, limit: int = 10, status: Optional[str] = None) -> Dict[str, Any]:
        """
        列出任务
        
        参数:
            limit: 最大返回数量
            status: 过滤状态
            
        返回:
            任务列表
        """
        tasks_list = []
        
        for task_id, task in sorted(
            self.tasks.items(), 
            key=lambda x: x[1]["created_at"],
            reverse=True
        ):
            # 如果指定了状态过滤
            if status and task["status"] != status:
                continue
                
            # 添加到列表
            tasks_list.append({
                "task_id": task_id,
                "status": task["status"],
                "created_at": task["created_at"],
                "text": task["text"]
            })
            
            # 检查数量限制
            if len(tasks_list) >= limit:
                break
        
        return {
            "success": True,
            "count": len(tasks_list),
            "tasks": tasks_list
        }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            # 计算任务年龄（小时）
            task_age_hours = (current_time - task["created_at"]) / 3600
            
            # 如果任务超过最大年龄
            if task_age_hours > max_age_hours:
                tasks_to_remove.append(task_id)
        
        # 删除旧任务
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        return {"success": True, "removed_count": len(tasks_to_remove)} 