"""
系统状态管理器
提供实时状态更新，支持SSE推送
"""
import time
import json
import threading
from collections import deque
from datetime import datetime


class StatusManager:
    """系统状态管理器，单例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.status_queue = deque(maxlen=100)
        self.current_status = {
            'phase': 'idle',
            'message': '就绪，等待提问',
            'progress': 0,
            'timestamp': datetime.now().isoformat(),
            'bge_loaded': False,
            'llm_loaded': False,
            'web_enabled': False
        }
        self.listeners = []

    def update_status(self, phase: str, message: str, progress: int = 0, **kwargs):
        """
        更新当前状态

        Args:
            phase: 阶段 (idle/loading_bge/searching/loading_llm/generating/done)
            message: 状态消息
            progress: 进度百分比 0-100
        """
        self.current_status.update({
            'phase': phase,
            'message': message,
            'progress': progress,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })

        # 添加到历史队列
        self.status_queue.append(dict(self.current_status))

        # 通知监听器
        self._notify_listeners()

        print(f"📡 状态更新: [{phase}] {message} ({progress}%)")

    def get_current_status(self) -> dict:
        """获取当前状态"""
        return dict(self.current_status)

    def get_status_history(self, limit: int = 20) -> list:
        """获取状态历史"""
        return list(self.status_queue)[-limit:]

    def add_listener(self, callback):
        """添加状态变化监听器"""
        self.listeners.append(callback)

    def remove_listener(self, callback):
        """移除监听器"""
        if callback in self.listeners:
            self.listeners.remove(callback)

    def _notify_listeners(self):
        """通知所有监听器"""
        status = self.get_current_status()
        for callback in self.listeners:
            try:
                callback(status)
            except:
                pass


# 全局单例
status_manager = StatusManager()