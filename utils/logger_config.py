# utils/logger_config.py

import logging
import os
from typing import Optional
from enum import Enum
from datetime import datetime
import sys


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FileLogHandler:
    """文件日志处理器"""

    _instance = None
    _file_handler = None
    _log_file = None

    @classmethod
    def set_log_file(cls, problem_id: str, model_name: str = "unknown"):
        """设置日志文件路径"""
        # 支持按轮次分开保存（通过 ROUND 环境变量）
        round_num = os.environ.get("ROUND", "")
        if round_num:
            log_dir = f"./log/{model_name}-round{round_num}"
        else:
            log_dir = f"./log/{model_name}"
        os.makedirs(log_dir, exist_ok=True)

        # 生成日志文件名（使用problem_id）
        log_filename = f"{problem_id}.log"
        cls._log_file = os.path.join(log_dir, log_filename)

        # 创建文件handler
        if cls._file_handler:
            cls._file_handler.close()

        cls._file_handler = open(cls._log_file, 'w', encoding='utf-8')
        return cls._log_file

    @classmethod
    def write(cls, message: str):
        """写入日志到文件"""
        if cls._file_handler:
            # 移除ANSI颜色代码
            clean_message = cls._remove_ansi_codes(message)
            cls._file_handler.write(clean_message + '\n')
            cls._file_handler.flush()

    @classmethod
    def _remove_ansi_codes(cls, text: str) -> str:
        """移除ANSI颜色代码"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    @classmethod
    def close(cls):
        """关闭文件handler"""
        if cls._file_handler:
            cls._file_handler.close()
            cls._file_handler = None


class AgentLogger:
    """智能体专用日志器"""

    # 全局控制开关
    ENABLED_AGENTS = {
        "OBSERVER": True,
        "PROBE": True,
        "EXECUTOR": True,
        "COMPRESSOR": False,  # 关闭压缩器日志
        "PLATFORM": True,  # 平台关键信息
        "EVALUATOR": True,  # 评估器日志
    }

    # 颜色代码
    COLORS = {
        "OBSERVER": "\033[94m",  # 蓝色
        "PROBE": "\033[92m",  # 绿色
        "EXECUTOR": "\033[93m",  # 黄色
        "PLATFORM": "\033[95m",  # 紫色
        "ERROR": "\033[91m",  # 红色
        "RESET": "\033[0m"  # 重置
    }

    def __init__(self, agent_name: str):
        """
        初始化智能体日志器

        Args:
            agent_name: 智能体名称
        """
        self.agent_name = agent_name.upper()
        self.enabled = self.ENABLED_AGENTS.get(self.agent_name, False)
        self.color = self.COLORS.get(self.agent_name, self.COLORS["RESET"])

    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """输出日志"""
        if not self.enabled:
            return

        # 构建标签
        tag = f"[{self.agent_name} AGENT]" if self.agent_name != "PLATFORM" else "[PLATFORM]"
        if self.agent_name == "EVALUATOR":
            tag = "[EVALUATOR]"

        # 添加颜色
        if level == LogLevel.ERROR:
            color = self.COLORS["ERROR"]
        else:
            color = self.color

        formatted_message = f"{color}{tag}{self.COLORS['RESET']} {message}"

        # 输出到控制台
        print(formatted_message)

        # 同时写入文件
        FileLogHandler.write(f"{tag} {message}")

    def info(self, message: str):
        """信息日志"""
        self.log(message, LogLevel.INFO)

    def error(self, message: str):
        """错误日志"""
        self.log(message, LogLevel.ERROR)

    def success(self, message: str):
        """成功日志（绿色）"""
        if self.enabled:
            formatted = f"\033[92m[{self.agent_name}] ✅ {message}\033[0m"
            print(formatted)
            FileLogHandler.write(f"[{self.agent_name}] ✅ {message}")

    def warning(self, message: str):
        """警告日志"""
        if self.enabled:
            formatted = f"\033[93m[{self.agent_name}] ⚠️ {message}\033[0m"
            print(formatted)
            FileLogHandler.write(f"[{self.agent_name}] ⚠️ {message}")
    
    def file_only(self, message: str, level: LogLevel = LogLevel.INFO):
        """只写入文件，不打印到控制台"""
        if not self.enabled:
            return

        # 构建标签
        tag = f"[{self.agent_name} AGENT]" if self.agent_name != "PLATFORM" else "[PLATFORM]"
        if self.agent_name == "EVALUATOR":
            tag = "[EVALUATOR]"
        
        # 只写入文件，不打印
        FileLogHandler.write(f"{tag} {message}")
    
    def debug_file_only(self, message: str):
        """调试信息只写入文件"""
        self.file_only(message, LogLevel.DEBUG)
    
    def console_only(self, message: str, level: LogLevel = LogLevel.INFO):
        """只打印到控制台，不写入文件"""
        if not self.enabled:
            return
        
        # 构建标签
        tag = f"[{self.agent_name} AGENT]" if self.agent_name != "PLATFORM" else "[PLATFORM]"
        if self.agent_name == "EVALUATOR":
            tag = "[EVALUATOR]"
        
        # 添加颜色
        if level == LogLevel.ERROR:
            color = self.COLORS["ERROR"]
        else:
            color = self.color
        
        formatted_message = f"{color}{tag}{self.COLORS['RESET']} {message}"
        
        # 只打印到控制台，不写入文件
        print(formatted_message)


def setup_logging():
    """配置全局日志（静默大部分第三方库日志）"""
    # 设置根日志器为ERROR级别，减少噪音
    logging.getLogger().setLevel(logging.ERROR)

    # 静默特定的日志器
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("aworld").setLevel(logging.ERROR)

    # 只在错误时显示
    logging.basicConfig(
        level=logging.ERROR,
        format='%(message)s'
    )