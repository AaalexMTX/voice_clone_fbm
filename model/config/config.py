#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 获取模块目录
MODULE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认配置
DEFAULT_CONFIG = {
    # 服务器配置
    "server": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False
    },
    
    # 模型配置
    "model": {
        "model_dir": "model/vocoder/models",
        "device": "auto",  # auto表示自动选择(cuda或cpu)
        "embedding_dim": 256,
        "hidden_dim": 512,
        "n_mels": 80
    },
    
    # 数据目录配置
    "data": {
        "upload_dir": "model/uploads",
        "output_dir": "model/outputs",
        "temp_dir": None  # None表示使用系统临时目录
    },
    
    # API测试配置
    "test": {
        "url": "http://localhost:5000",
        "audio": "model/tests/sample.wav",
        "text": "这是一段测试语音，用于测试语音克隆系统。",
        "output_dir": "model/outputs/test"
    }
}

# 全局配置对象
config = DEFAULT_CONFIG.copy()

class AttributeDict(dict):
    """
    允许使用点号访问的字典
    """
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        # 递归转换嵌套字典
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttributeDict(value)
    
    def __getattr__(self, name):
        # 当属性不存在时返回None而不是抛出异常
        return None

class Config:
    """
    配置类，用于加载和管理模型配置
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        # 默认配置路径
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       "model", "config", "config.yaml")
        
        # 加载配置
        self.config_dict = self._load_config(config_path)
        
        # 将配置转换为可访问属性
        for key, value in self.config_dict.items():
            setattr(self, key, AttributeDict(value) if isinstance(value, dict) else value)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        参数:
            config_path: 配置文件路径
            
        返回:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"已加载配置文件: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"加载配置文件失败: {str(e)}，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        返回默认配置
        
        返回:
            默认配置字典
        """
        return {
            "model": {
                # 通用配置
                "n_mels": 80,
                "sample_rate": 22050,
                
                # 文本编码器配置
                "vocab_size": 256,
                "text_embed_dim": 512,
                "encoder_hidden_dim": 1024,
                "encoder_layers": 4,
                "encoder_heads": 8,
                "encoder_dropout": 0.1,
                
                # 说话人编码器配置
                "audio_n_mels": 80,
                "speaker_hidden_dim": 256,
                "speaker_embedding_dim": 256,
                "speaker_encoder_layers": 3,
                
                # 梅尔解码器配置
                "decoder_dim": 512,
                "decoder_layers": 4,
                "decoder_heads": 8,
                "decoder_ff_dim": 2048,
                "decoder_dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.0001,
                "max_epochs": 1000,
                "warmup_steps": 4000,
                "save_interval": 10,
                "eval_interval": 5
            },
            "paths": {
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs",
                "data_dir": "data"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        参数:
            key: 配置键，可以是点分隔的路径
            default: 默认值
            
        返回:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self, config_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       "model", "config", "config.yaml")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f, default_flow_style=False)
                logger.info(f"已保存配置文件: {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        参数:
            updates: 更新的配置字典
        """
        def _update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _update_dict(d[k], v)
                else:
                    d[k] = v
        
        _update_dict(self.config_dict, updates)
        
        # 更新属性
        for key, value in self.config_dict.items():
            setattr(self, key, AttributeDict(value) if isinstance(value, dict) else value)

def load_config(config_path=None):
    """
    从文件加载配置
    
    参数:
        config_path: 配置文件路径，支持.json和.yaml格式
    
    返回:
        配置字典
    """
    global config
    
    # 如果未指定配置文件，则尝试加载默认位置的配置
    if config_path is None:
        # 尝试按顺序加载不同位置的配置文件
        possible_paths = [
            # 模块内部配置
            str(MODULE_DIR / "config" / "config.yaml"),
            str(MODULE_DIR / "config" / "config.yml"),
            str(MODULE_DIR / "config" / "config.json"),
            # 原来的位置
            str(MODULE_DIR / "config.yaml"),
            str(MODULE_DIR / "config.yml"),
            str(MODULE_DIR / "config.json"),
            # 项目根目录配置
            "config.yaml",
            "config.yml",
            "config.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # 如果找到配置文件，则加载
    if config_path and os.path.exists(config_path):
        try:
            ext = os.path.splitext(config_path)[1].lower()
            with open(config_path, 'r', encoding='utf-8') as f:
                if ext in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                elif ext == '.json':
                    user_config = json.load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {ext}")
                    return config
                
            # 递归更新配置
            _update_config(config, user_config)
            logger.info(f"已加载配置文件: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
    else:
        logger.info("未找到配置文件，使用默认配置")
    
    # 处理特殊配置
    _process_special_config()
    
    return config

def _update_config(base_config, update_config):
    """递归更新配置"""
    for key, value in update_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            _update_config(base_config[key], value)
        else:
            base_config[key] = value

def _process_special_config():
    """处理特殊配置项"""
    # 处理device自动选择
    if config["model"]["device"] == "auto":
        import torch
        config["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 处理临时目录
    if config["data"]["temp_dir"] is None:
        import tempfile
        config["data"]["temp_dir"] = tempfile.gettempdir()
    
    # 转换相对路径为绝对路径（相对于模块目录）
    for section in ["model", "data", "test"]:
        for key in config[section]:
            if isinstance(config[section][key], str) and not os.path.isabs(config[section][key]):
                # 排除URL和设备名
                if not (key == "url" or key == "device"):
                    if config[section][key].startswith("model/"):
                        # 如果已经以model/开头，从工作目录开始计算
                        config[section][key] = str(MODULE_DIR.parent / config[section][key])
                    else:
                        # 否则，默认路径在模块目录下
                        config[section][key] = str(MODULE_DIR / config[section][key])
    
    # 确保目录存在
    for key in ["upload_dir", "output_dir"]:
        os.makedirs(config["data"][key], exist_ok=True)

def get_config():
    """获取当前配置"""
    return config

def save_config(config_path="config.yaml"):
    """保存当前配置到文件"""
    try:
        # 处理路径
        if not os.path.isabs(config_path):
            # 检查路径是否已包含config目录
            if os.path.dirname(config_path).endswith('config'):
                # 已包含config目录，直接使用相对于MODULE_DIR的路径
                config_path = str(MODULE_DIR.parent / config_path)
            else:
                # 不包含config目录，添加到MODULE_DIR/config下
                config_path = str(MODULE_DIR / "config" / config_path)
            
        ext = os.path.splitext(config_path)[1].lower()
        with open(config_path, 'w', encoding='utf-8') as f:
            if ext in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            elif ext == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
        
        logger.info(f"配置已保存到: {config_path}")
        return True
    except Exception as e:
        logger.error(f"保存配置失败: {str(e)}")
        return False

# 加载配置
load_config() 