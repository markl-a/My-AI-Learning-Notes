"""
速率限制中間件

提供基於 IP 和 API Key 的速率限制功能，防止 API 濫用。
支持滑動窗口算法和令牌桶算法。
"""

import time
from collections import defaultdict
from typing import Callable, Optional
from functools import wraps
import asyncio
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    速率限制器

    使用滑動窗口算法實現請求限制。

    Attributes:
        requests_per_minute: 每分鐘允許的請求數
        requests_per_hour: 每小時允許的請求數
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit

        # 存儲請求記錄 {client_id: [(timestamp, count), ...]}
        self._minute_requests: dict[str, list[float]] = defaultdict(list)
        self._hour_requests: dict[str, list[float]] = defaultdict(list)
        self._burst_requests: dict[str, list[float]] = defaultdict(list)

        # 清理任務
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """啟動清理任務"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """停止清理任務"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """定期清理過期的請求記錄"""
        while True:
            await asyncio.sleep(60)  # 每分鐘清理一次
            self._cleanup_old_requests()

    def _cleanup_old_requests(self):
        """清理過期請求記錄"""
        current_time = time.time()

        # 清理分鐘級記錄
        for client_id in list(self._minute_requests.keys()):
            self._minute_requests[client_id] = [
                t for t in self._minute_requests[client_id]
                if current_time - t < 60
            ]
            if not self._minute_requests[client_id]:
                del self._minute_requests[client_id]

        # 清理小時級記錄
        for client_id in list(self._hour_requests.keys()):
            self._hour_requests[client_id] = [
                t for t in self._hour_requests[client_id]
                if current_time - t < 3600
            ]
            if not self._hour_requests[client_id]:
                del self._hour_requests[client_id]

    def _get_client_id(self, request: Request) -> str:
        """
        獲取客戶端標識符

        優先使用 API Key，否則使用 IP 地址
        """
        # 嘗試從 Header 獲取 API Key
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            # 移除 "Bearer " 前綴（如果有）
            if api_key.startswith("Bearer "):
                api_key = api_key[7:]
            return f"key:{api_key[:16]}"  # 只使用前 16 字符

        # 獲取客戶端 IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    async def check_rate_limit(self, request: Request) -> bool:
        """
        檢查請求是否超過速率限制

        Returns:
            True 如果請求被允許，否則拋出 HTTPException
        """
        client_id = self._get_client_id(request)
        current_time = time.time()

        # 檢查突發限制（每秒）
        burst_requests = [
            t for t in self._burst_requests[client_id]
            if current_time - t < 1
        ]
        if len(burst_requests) >= self.burst_limit:
            logger.warning(f"Rate limit exceeded (burst) for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests per second",
                    "retry_after": 1
                },
                headers={"Retry-After": "1"}
            )

        # 檢查分鐘限制
        minute_requests = [
            t for t in self._minute_requests[client_id]
            if current_time - t < 60
        ]
        if len(minute_requests) >= self.requests_per_minute:
            retry_after = int(60 - (current_time - minute_requests[0]))
            logger.warning(f"Rate limit exceeded (minute) for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests per minute. Limit: {self.requests_per_minute}",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )

        # 檢查小時限制
        hour_requests = [
            t for t in self._hour_requests[client_id]
            if current_time - t < 3600
        ]
        if len(hour_requests) >= self.requests_per_hour:
            retry_after = int(3600 - (current_time - hour_requests[0]))
            logger.warning(f"Rate limit exceeded (hour) for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests per hour. Limit: {self.requests_per_hour}",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )

        # 記錄請求
        self._burst_requests[client_id].append(current_time)
        self._minute_requests[client_id].append(current_time)
        self._hour_requests[client_id].append(current_time)

        return True

    def get_remaining_requests(self, request: Request) -> dict:
        """獲取剩餘請求配額"""
        client_id = self._get_client_id(request)
        current_time = time.time()

        minute_requests = len([
            t for t in self._minute_requests[client_id]
            if current_time - t < 60
        ])
        hour_requests = len([
            t for t in self._hour_requests[client_id]
            if current_time - t < 3600
        ])

        return {
            "remaining_per_minute": max(0, self.requests_per_minute - minute_requests),
            "remaining_per_hour": max(0, self.requests_per_hour - hour_requests),
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour
        }


# 全局速率限制器實例
rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_limit=10
)


def rate_limit(
    requests_per_minute: Optional[int] = None,
    requests_per_hour: Optional[int] = None
):
    """
    速率限制裝飾器

    用於對特定端點應用自定義速率限制

    Usage:
        @app.get("/api/chat")
        @rate_limit(requests_per_minute=10)
        async def chat():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 獲取 request 對象
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request:
                # 使用自定義限制或默認限制
                custom_limiter = RateLimiter(
                    requests_per_minute=requests_per_minute or rate_limiter.requests_per_minute,
                    requests_per_hour=requests_per_hour or rate_limiter.requests_per_hour
                )
                await custom_limiter.check_rate_limit(request)

            return await func(*args, **kwargs)
        return wrapper
    return decorator


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI 速率限制中間件

    Usage:
        app.middleware("http")(rate_limit_middleware)
    """
    # 跳過健康檢查和靜態文件
    skip_paths = ["/health", "/api/health", "/docs", "/redoc", "/openapi.json"]
    if any(request.url.path.startswith(path) for path in skip_paths):
        return await call_next(request)

    try:
        await rate_limiter.check_rate_limit(request)
        response = await call_next(request)

        # 添加速率限制相關的響應頭
        remaining = rate_limiter.get_remaining_requests(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(remaining["limit_per_minute"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["remaining_per_minute"])
        response.headers["X-RateLimit-Limit-Hour"] = str(remaining["limit_per_hour"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["remaining_per_hour"])

        return response

    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=e.detail,
            headers=dict(e.headers) if e.headers else None
        )
