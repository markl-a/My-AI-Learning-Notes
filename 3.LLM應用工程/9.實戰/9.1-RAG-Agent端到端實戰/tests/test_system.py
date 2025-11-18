"""系統集成測試"""
import pytest
import sys
import os

# 添加 src 到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import chunk_text, SimpleCache
from src.models import QueryRequest, Source


class TestUtils:
    """測試工具函數"""

    def test_chunk_text(self):
        """測試文本分塊"""
        text = "這是一個測試文本。" * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_simple_cache(self):
        """測試簡單緩存"""
        cache = SimpleCache(max_size=10, ttl=60)

        # 測試設置和獲取
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # 測試不存在的鍵
        assert cache.get("nonexistent") is None

        # 測試刪除
        cache.delete("key1")
        assert cache.get("key1") is None

        # 測試清空
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key2") is None


class TestModels:
    """測試數據模型"""

    def test_query_request(self):
        """測試查詢請求模型"""
        request = QueryRequest(
            question="測試問題",
            use_agent=True,
            top_k=5
        )

        assert request.question == "測試問題"
        assert request.use_agent is True
        assert request.top_k == 5

    def test_source_model(self):
        """測試來源模型"""
        source = Source(
            content="測試內容",
            document="test.pdf",
            score=0.95
        )

        assert source.content == "測試內容"
        assert source.document == "test.pdf"
        assert source.score == 0.95


# 集成測試（需要實際的 API 密鑰才能運行）
@pytest.mark.skip(reason="Requires API keys and running system")
class TestSystemIntegration:
    """系統集成測試"""

    def test_document_processing(self):
        """測試文檔處理"""
        # TODO: 實現完整的集成測試
        pass

    def test_rag_query(self):
        """測試 RAG 查詢"""
        # TODO: 實現完整的集成測試
        pass

    def test_agent_query(self):
        """測試 Agent 查詢"""
        # TODO: 實現完整的集成測試
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
