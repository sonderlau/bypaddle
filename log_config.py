# log_config.py

import logging
from contextvars import ContextVar
from typing import Optional
import sys

# 全局的用户ID上下文变量
current_user_id: ContextVar[Optional[str]] = ContextVar('current_user_id', default=None)

class UserIDFilter(logging.Filter):
    """为所有日志添加用户ID的过滤器"""
    def filter(self, record):
        try:
            user_id = current_user_id.get()
            record.user_id = user_id if user_id else 'system'
        except Exception:
            record.user_id = 'system'
        return True

def setup_logging():
    """配置全局日志系统"""
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - [用户:%(user_id)s] - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加用户ID过滤器到根日志记录器
    root_logger.addFilter(UserIDFilter())
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    return root_logger

def set_current_user_id(user_id: Optional[str]):
    """设置当前上下文的用户ID"""
    current_user_id.set(user_id)

def get_current_user_id() -> Optional[str]:
    """获取当前上下文的用户ID"""
    return current_user_id.get()