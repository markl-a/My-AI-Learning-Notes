"""
數據模型單元測試

測試 Pydantic 模型的驗證和功能。
"""

import pytest
from datetime import datetime
import sys
import os

# 添加路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_PATH = os.path.join(
    PROJECT_ROOT,
    '3.LLM應用工程', '9.實戰', '9.1-RAG-Agent端到端實戰', 'src'
)
sys.path.insert(0, MODELS_PATH)

from models import (
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
    Source,
    DocumentUploadRequest,
    DocumentUploadResponse,
    SystemStats,
    HealthResponse,
    ErrorResponse
)


class TestDocumentMetadata:
    """DocumentMetadata 模型測試"""

    def test_basic_creation(self):
        """測試基本創建"""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024
        )

        assert metadata.filename == "test.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.file_size == 1024
        assert metadata.chunk_count == 0
        assert metadata.language == "zh"

    def test_with_all_fields(self):
        """測試所有欄位"""
        upload_date = datetime.now()
        metadata = DocumentMetadata(
            filename="document.txt",
            file_type="txt",
            file_size=2048,
            upload_date=upload_date,
            chunk_count=10,
            language="en"
        )

        assert metadata.upload_date == upload_date
        assert metadata.chunk_count == 10
        assert metadata.language == "en"


class TestQueryRequest:
    """QueryRequest 模型測試"""

    def test_minimal_creation(self):
        """測試最小創建"""
        request = QueryRequest(question="什麼是機器學習？")

        assert request.question == "什麼是機器學習？"
        assert request.use_agent is True
        assert request.top_k == 5
        assert request.session_id is None
        assert request.filters is None

    def test_full_creation(self):
        """測試完整創建"""
        request = QueryRequest(
            question="解釋深度學習",
            use_agent=False,
            top_k=10,
            session_id="session_123",
            filters={"category": "dl"}
        )

        assert request.use_agent is False
        assert request.top_k == 10
        assert request.session_id == "session_123"
        assert request.filters == {"category": "dl"}

    def test_question_min_length(self):
        """測試問題最小長度驗證"""
        # 空字串應該失敗
        with pytest.raises(ValueError):
            QueryRequest(question="")

    def test_top_k_bounds(self):
        """測試 top_k 邊界驗證"""
        # 有效值
        request = QueryRequest(question="test", top_k=1)
        assert request.top_k == 1

        request = QueryRequest(question="test", top_k=20)
        assert request.top_k == 20

        # 無效值
        with pytest.raises(ValueError):
            QueryRequest(question="test", top_k=0)

        with pytest.raises(ValueError):
            QueryRequest(question="test", top_k=21)


class TestSource:
    """Source 模型測試"""

    def test_basic_creation(self):
        """測試基本創建"""
        source = Source(
            content="這是一段內容",
            document="doc.pdf",
            score=0.95
        )

        assert source.content == "這是一段內容"
        assert source.document == "doc.pdf"
        assert source.score == 0.95
        assert source.page is None

    def test_with_page(self):
        """測試帶頁碼的創建"""
        source = Source(
            content="內容",
            document="doc.pdf",
            page=5,
            score=0.8
        )

        assert source.page == 5

    def test_score_bounds(self):
        """測試分數邊界"""
        # 有效值
        source = Source(content="test", document="doc", score=0.0)
        assert source.score == 0.0

        source = Source(content="test", document="doc", score=1.0)
        assert source.score == 1.0

        # 無效值
        with pytest.raises(ValueError):
            Source(content="test", document="doc", score=-0.1)

        with pytest.raises(ValueError):
            Source(content="test", document="doc", score=1.1)


class TestQueryResponse:
    """QueryResponse 模型測試"""

    def test_minimal_creation(self):
        """測試最小創建"""
        response = QueryResponse(answer="這是回答")

        assert response.answer == "這是回答"
        assert response.sources == []
        assert response.tools_used == []
        assert response.confidence == 0.0
        assert response.suggestions == []
        assert response.processing_time == 0.0

    def test_full_creation(self):
        """測試完整創建"""
        sources = [
            Source(content="內容1", document="doc1.pdf", score=0.9),
            Source(content="內容2", document="doc2.pdf", score=0.85)
        ]

        response = QueryResponse(
            answer="完整回答",
            sources=sources,
            tools_used=["rag_search", "calculator"],
            confidence=0.92,
            suggestions=["追問1", "追問2"],
            processing_time=1.5
        )

        assert len(response.sources) == 2
        assert len(response.tools_used) == 2
        assert response.confidence == 0.92
        assert len(response.suggestions) == 2
        assert response.processing_time == 1.5

    def test_confidence_bounds(self):
        """測試置信度邊界"""
        response = QueryResponse(answer="test", confidence=0.0)
        assert response.confidence == 0.0

        response = QueryResponse(answer="test", confidence=1.0)
        assert response.confidence == 1.0

        with pytest.raises(ValueError):
            QueryResponse(answer="test", confidence=-0.1)

        with pytest.raises(ValueError):
            QueryResponse(answer="test", confidence=1.1)


class TestDocumentUploadRequest:
    """DocumentUploadRequest 模型測試"""

    def test_basic_creation(self):
        """測試基本創建"""
        request = DocumentUploadRequest(
            filename="test.pdf",
            content="文檔內容"
        )

        assert request.filename == "test.pdf"
        assert request.content == "文檔內容"
        assert request.metadata is None

    def test_with_metadata(self):
        """測試帶元數據的創建"""
        request = DocumentUploadRequest(
            filename="test.pdf",
            content="內容",
            metadata={"author": "test", "date": "2024-01-01"}
        )

        assert request.metadata["author"] == "test"


class TestDocumentUploadResponse:
    """DocumentUploadResponse 模型測試"""

    def test_success_response(self):
        """測試成功響應"""
        response = DocumentUploadResponse(
            success=True,
            document_id="doc_123",
            message="上傳成功",
            chunks_created=10
        )

        assert response.success is True
        assert response.document_id == "doc_123"
        assert response.chunks_created == 10

    def test_failure_response(self):
        """測試失敗響應"""
        response = DocumentUploadResponse(
            success=False,
            document_id="",
            message="上傳失敗：文件格式不支援",
            chunks_created=0
        )

        assert response.success is False


class TestSystemStats:
    """SystemStats 模型測試"""

    def test_default_values(self):
        """測試預設值"""
        stats = SystemStats()

        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_queries == 0
        assert stats.avg_response_time == 0.0
        assert stats.cache_hit_rate == 0.0
        assert stats.uptime_seconds == 0.0

    def test_with_values(self):
        """測試帶值創建"""
        stats = SystemStats(
            total_documents=100,
            total_chunks=1000,
            total_queries=500,
            avg_response_time=0.5,
            cache_hit_rate=0.75,
            uptime_seconds=86400.0
        )

        assert stats.total_documents == 100
        assert stats.cache_hit_rate == 0.75


class TestHealthResponse:
    """HealthResponse 模型測試"""

    def test_default_values(self):
        """測試預設值"""
        health = HealthResponse()

        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert isinstance(health.timestamp, datetime)
        assert health.components == {}

    def test_with_components(self):
        """測試帶組件狀態"""
        health = HealthResponse(
            status="degraded",
            components={
                "database": "healthy",
                "vector_store": "degraded",
                "llm_api": "healthy"
            }
        )

        assert health.status == "degraded"
        assert len(health.components) == 3


class TestErrorResponse:
    """ErrorResponse 模型測試"""

    def test_basic_error(self):
        """測試基本錯誤"""
        error = ErrorResponse(error="發生錯誤")

        assert error.error == "發生錯誤"
        assert error.detail is None
        assert isinstance(error.timestamp, datetime)

    def test_with_detail(self):
        """測試帶詳情的錯誤"""
        error = ErrorResponse(
            error="驗證失敗",
            detail="欄位 'question' 不能為空"
        )

        assert error.detail == "欄位 'question' 不能為空"


class TestModelSerialization:
    """模型序列化測試"""

    def test_query_request_json(self):
        """測試 QueryRequest JSON 序列化"""
        request = QueryRequest(
            question="測試問題",
            use_agent=True,
            top_k=5
        )

        json_data = request.model_dump()

        assert json_data["question"] == "測試問題"
        assert json_data["use_agent"] is True
        assert json_data["top_k"] == 5

    def test_query_response_json(self):
        """測試 QueryResponse JSON 序列化"""
        response = QueryResponse(
            answer="測試回答",
            sources=[
                Source(content="內容", document="doc.pdf", score=0.9)
            ],
            confidence=0.85
        )

        json_data = response.model_dump()

        assert json_data["answer"] == "測試回答"
        assert len(json_data["sources"]) == 1
        assert json_data["sources"][0]["score"] == 0.9
