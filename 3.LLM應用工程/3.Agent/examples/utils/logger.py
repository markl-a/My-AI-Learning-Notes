"""
日誌配置

統一的日誌設置。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "agent",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    設置並返回一個配置好的 logger

    Args:
        name: logger 名稱
        level: 日誌級別
        log_file: 日誌文件路徑（可選）
        format_string: 自定義格式字符串（可選）

    Returns:
        配置好的 logger 實例
    """
    # 默認格式
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )

    # 創建 formatter
    formatter = logging.Formatter(format_string)

    # 創建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重複添加 handler
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler（如果指定了文件路徑）
    if log_file:
        # 確保目錄存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "agent") -> logging.Logger:
    """
    獲取 logger 實例

    Args:
        name: logger 名稱

    Returns:
        logger 實例
    """
    return logging.getLogger(name)


# 預設 logger
default_logger = setup_logger()
