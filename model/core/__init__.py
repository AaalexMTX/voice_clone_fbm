#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
核心功能模块，包含语音克隆的核心算法
"""

from .voice_clone import VoiceCloneSystem
from .model import VoiceCloneModel

__all__ = ['VoiceCloneSystem', 'VoiceCloneModel'] 