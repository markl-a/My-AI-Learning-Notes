"""工具函數"""
import os
import hashlib
import time
import yaml
from typing import Any, Dict, Optional
from functools import wraps
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加載配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def get_env_or_config(key: str, config: Dict, default: Any = None) -> Any:
    """優先從環境變數獲取，否則從配置獲取"""
    # 嘗試從環境變數獲取
    env_value = os.getenv(key.upper())
    if env_value is not None:
        return env_value

    # 從配置獲取（支持嵌套鍵，如 "llm.model"）
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default

    return value if value is not None else default


def compute_file_hash(file_path: str) -> str:
    """計算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute hash for {file_path}: {e}")
        return ""


def timing_decorator(func):
    """計時裝飾器"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result

    # 判斷是否為異步函數
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def setup_logging(config: Dict[str, Any]):
    """配置日誌"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', 'text')

    # 創建日誌目錄
    log_file = log_config.get('file', './logs/app.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 配置格式
    if log_format == 'json':
        from logging import Formatter
        formatter = Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}'
        )
    else:
        from logging import Formatter
        formatter = Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # 配置處理器
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 配置根日誌器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """將文本分塊

    Args:
        text: 原始文本
        chunk_size: 塊大小
        overlap: 重疊大小

    Returns:
        分塊後的文本列表
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # 嘗試在句子邊界處分割
        if end < text_length:
            # 尋找最近的句號、問號或換行
            for delimiter in ['\n\n', '。', '！', '？', '\n', '. ', '! ', '? ']:
                pos = text.rfind(delimiter, start, end)
                if pos != -1:
                    end = pos + len(delimiter)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 下一個塊的起始位置（考慮重疊）
        start = end - overlap if end < text_length else text_length

    return chunks


def truncate_text(text: str, max_length: int = 100) -> str:
    """截斷文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def ensure_dir(directory: str):
    """確保目錄存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)


class SimpleCache:
    """簡單的內存緩存"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """獲取緩存"""
        if key not in self.cache:
            return None

        # 檢查是否過期
        if time.time() - self.timestamps[key] > self.ttl:
            self.delete(key)
            return None

        return self.cache[key]

    def set(self, key: str, value: Any):
        """設置緩存"""
        # 如果緩存已滿，刪除最舊的項
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self.delete(oldest_key)

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def delete(self, key: str):
        """刪除緩存"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

    def clear(self):
        """清空緩存"""
        self.cache.clear()
        self.timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }
