import logging
from typing import Optional

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    配置日志系统，防止重复日志
    
    Args:
        name: 日志记录器的名称，默认使用 __name__
        
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 清除所有已存在的处理器
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # 只添加一个控制台处理器
        ]
    )
    
    # 创建并配置特定的日志记录器
    logger = logging.getLogger(name if name else __name__)
    logger.propagate = False  # 防止日志向上传播
    
    # 如果该日志记录器还没有处理器，添加一个
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger 