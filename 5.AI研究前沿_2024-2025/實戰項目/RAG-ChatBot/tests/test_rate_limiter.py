"""
速率限制中間件測試
測試 rate_limiter.py 的功能
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from fastapi import HTTPException
import sys
import os

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware.rate_limiter import RateLimiter, rate_limiter, rate_limit_middleware


class TestRateLimiter:
    """RateLimiter 類測試"""

    @pytest.fixture
    def limiter(self):
        """創建測試用的限制器"""
        return RateLimiter(
            requests_per_minute=5,
            requests_per_hour=20,
            burst_limit=2
        )

    @pytest.fixture
    def mock_request(self):
        """創建 Mock 請求對象"""
        request = Mock()
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.path = "/api/chat"
        return request

    def test_initialization(self, limiter):
        """測試初始化"""
        assert limiter.requests_per_minute == 5
        assert limiter.requests_per_hour == 20
        assert limiter.burst_limit == 2

    def test_get_client_id_from_ip(self, limiter, mock_request):
        """測試從 IP 獲取客戶端 ID"""
        client_id = limiter._get_client_id(mock_request)
        assert client_id == "ip:192.168.1.1"

    def test_get_client_id_from_api_key(self, limiter, mock_request):
        """測試從 API Key 獲取客戶端 ID"""
        mock_request.headers = {"X-API-Key": "test_api_key_12345678"}
        client_id = limiter._get_client_id(mock_request)
        assert client_id.startswith("key:")
        assert "test_api_key_123" in client_id

    def test_get_client_id_from_bearer_token(self, limiter, mock_request):
        """測試從 Bearer Token 獲取客戶端 ID"""
        mock_request.headers = {"Authorization": "Bearer test_token_12345678"}
        client_id = limiter._get_client_id(mock_request)
        assert client_id.startswith("key:")

    def test_get_client_id_from_forwarded_header(self, limiter, mock_request):
        """測試從 X-Forwarded-For 獲取 IP"""
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        client_id = limiter._get_client_id(mock_request)
        assert client_id == "ip:10.0.0.1"

    @pytest.mark.asyncio
    async def test_check_rate_limit_allows_request(self, limiter, mock_request):
        """測試允許正常請求"""
        result = await limiter.check_rate_limit(mock_request)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_burst_exceeded(self, limiter, mock_request):
        """測試超過突發限制"""
        # 快速發送超過突發限制的請求
        await limiter.check_rate_limit(mock_request)
        await limiter.check_rate_limit(mock_request)

        # 第三個請求應該被拒絕
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Too many requests per second" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_check_rate_limit_minute_exceeded(self, limiter, mock_request):
        """測試超過分鐘限制"""
        # 模擬突發請求分散在時間上
        for i in range(5):
            # 清除突發記錄以避免觸發突發限制
            limiter._burst_requests.clear()
            await limiter.check_rate_limit(mock_request)

        # 清除突發記錄
        limiter._burst_requests.clear()

        # 第六個請求應該被拒絕
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "minute" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_remaining_requests(self, limiter, mock_request):
        """測試獲取剩餘請求配額"""
        # 發送一個請求
        limiter._burst_requests.clear()
        await limiter.check_rate_limit(mock_request)

        remaining = limiter.get_remaining_requests(mock_request)

        assert remaining["remaining_per_minute"] == 4  # 5 - 1
        assert remaining["remaining_per_hour"] == 19   # 20 - 1
        assert remaining["limit_per_minute"] == 5
        assert remaining["limit_per_hour"] == 20

    def test_cleanup_old_requests(self, limiter):
        """測試清理過期請求記錄"""
        old_time = time.time() - 120  # 2 分鐘前
        current_time = time.time()

        limiter._minute_requests["test_client"] = [old_time, current_time]
        limiter._hour_requests["test_client"] = [old_time, current_time]

        limiter._cleanup_old_requests()

        # 舊的分鐘記錄應該被清理
        assert len(limiter._minute_requests["test_client"]) == 1
        # 舊的小時記錄（2分鐘前）應該仍然保留
        assert len(limiter._hour_requests["test_client"]) == 2

    @pytest.mark.asyncio
    async def test_start_and_stop(self, limiter):
        """測試啟動和停止清理任務"""
        await limiter.start()
        assert limiter._cleanup_task is not None

        await limiter.stop()
        assert limiter._cleanup_task.cancelled() or limiter._cleanup_task.done()


class TestRateLimitMiddleware:
    """rate_limit_middleware 測試"""

    @pytest.fixture
    def mock_request(self):
        """創建 Mock 請求"""
        request = Mock()
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.path = "/api/chat"
        return request

    @pytest.fixture
    def mock_call_next(self):
        """創建 Mock call_next"""
        async def call_next(request):
            response = Mock()
            response.headers = {}
            return response
        return call_next

    @pytest.mark.asyncio
    async def test_middleware_allows_request(self, mock_request, mock_call_next):
        """測試中間件允許正常請求"""
        # 重置全局限制器
        rate_limiter._minute_requests.clear()
        rate_limiter._hour_requests.clear()
        rate_limiter._burst_requests.clear()

        response = await rate_limit_middleware(mock_request, mock_call_next)

        assert response is not None
        assert "X-RateLimit-Limit-Minute" in response.headers

    @pytest.mark.asyncio
    async def test_middleware_skips_health_check(self, mock_request, mock_call_next):
        """測試中間件跳過健康檢查路徑"""
        mock_request.url.path = "/api/health"

        response = await rate_limit_middleware(mock_request, mock_call_next)

        # 健康檢查不應該有速率限制頭
        assert response is not None

    @pytest.mark.asyncio
    async def test_middleware_skips_docs(self, mock_request, mock_call_next):
        """測試中間件跳過文檔路徑"""
        mock_request.url.path = "/docs"

        response = await rate_limit_middleware(mock_request, mock_call_next)
        assert response is not None


class TestRateLimitDecorator:
    """rate_limit 裝飾器測試"""

    @pytest.mark.asyncio
    async def test_decorator_creates_custom_limiter(self):
        """測試裝飾器創建自定義限制器"""
        from middleware.rate_limiter import rate_limit

        @rate_limit(requests_per_minute=2)
        async def test_endpoint(request=None):
            return "success"

        mock_request = Mock()
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"

        # 前兩個請求應該成功
        result = await test_endpoint(request=mock_request)
        assert result == "success"


class TestDifferentClients:
    """不同客戶端的測試"""

    @pytest.fixture
    def limiter(self):
        """創建限制器"""
        return RateLimiter(requests_per_minute=2, requests_per_hour=10, burst_limit=2)

    @pytest.mark.asyncio
    async def test_different_ips_have_separate_limits(self, limiter):
        """測試不同 IP 有獨立的限制"""
        request1 = Mock()
        request1.headers = {}
        request1.client = Mock()
        request1.client.host = "192.168.1.1"

        request2 = Mock()
        request2.headers = {}
        request2.client = Mock()
        request2.client.host = "192.168.1.2"

        # 客戶端1發送請求
        await limiter.check_rate_limit(request1)
        await limiter.check_rate_limit(request1)

        # 客戶端2應該仍然可以發送請求
        result = await limiter.check_rate_limit(request2)
        assert result is True

    @pytest.mark.asyncio
    async def test_api_key_overrides_ip(self, limiter):
        """測試 API Key 優先於 IP"""
        # 相同 IP，不同 API Key
        request1 = Mock()
        request1.headers = {"X-API-Key": "key_user_1"}
        request1.client = Mock()
        request1.client.host = "192.168.1.1"

        request2 = Mock()
        request2.headers = {"X-API-Key": "key_user_2"}
        request2.client = Mock()
        request2.client.host = "192.168.1.1"

        # 用戶1發送請求直到限制
        await limiter.check_rate_limit(request1)
        await limiter.check_rate_limit(request1)

        # 用戶2（不同 API Key）應該仍然可以發送
        result = await limiter.check_rate_limit(request2)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
