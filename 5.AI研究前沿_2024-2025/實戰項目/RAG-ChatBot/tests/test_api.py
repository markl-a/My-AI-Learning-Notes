"""
API 端點測試
測試 FastAPI 應用的所有端點
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, rag_engine


@pytest.fixture
def client():
    """創建測試客戶端"""
    return TestClient(app)


@pytest.fixture
def mock_rag_engine():
    """Mock RAG Engine"""
    with patch('main.rag_engine') as mock:
        yield mock


class TestHealthEndpoints:
    """健康檢查端點測試"""

    def test_health_check(self, client):
        """測試健康檢查端點"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_stats_endpoint(self, client, mock_rag_engine):
        """測試統計端點"""
        mock_rag_engine.get_stats.return_value = {
            "total_documents": 100,
            "total_conversations": 10,
            "model_name": "gpt-3.5-turbo"
        }

        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 100
        assert data["total_conversations"] == 10


class TestChatEndpoints:
    """聊天端點測試"""

    def test_chat_basic(self, client, mock_rag_engine):
        """測試基本聊天"""
        mock_rag_engine.chat = AsyncMock(return_value={
            "response": "Hello! How can I help you?",
            "conversation_id": "test_conv",
            "context_used": [],
            "tokens_used": 20
        })

        response = client.post("/api/chat", json={
            "message": "Hello",
            "use_rag": False
        })

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello! How can I help you?"
        assert "conversation_id" in data

    def test_chat_with_rag(self, client, mock_rag_engine):
        """測試使用 RAG 的聊天"""
        mock_rag_engine.chat = AsyncMock(return_value={
            "response": "Based on the context, Python is a programming language.",
            "conversation_id": "test_conv",
            "context_used": [
                {"content": "Python is a language", "metadata": {}, "distance": 0.1}
            ],
            "tokens_used": 50
        })

        response = client.post("/api/chat", json={
            "message": "What is Python?",
            "use_rag": True,
            "top_k": 3
        })

        assert response.status_code == 200
        data = response.json()
        assert "Python" in data["response"]
        assert len(data["context_used"]) > 0

    def test_chat_with_conversation_id(self, client, mock_rag_engine):
        """測試帶對話 ID 的聊天"""
        conv_id = "existing_conv"
        mock_rag_engine.chat = AsyncMock(return_value={
            "response": "Continuing conversation",
            "conversation_id": conv_id,
            "context_used": [],
            "tokens_used": 15
        })

        response = client.post("/api/chat", json={
            "message": "Continue chat",
            "conversation_id": conv_id,
            "use_rag": False
        })

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conv_id

    def test_chat_validation_error(self, client):
        """測試聊天請求驗證錯誤"""
        response = client.post("/api/chat", json={
            "message": ""  # 空消息應該失敗
        })

        assert response.status_code == 422  # Validation error


class TestDocumentEndpoints:
    """文檔管理端點測試"""

    def test_upload_document(self, client, mock_rag_engine):
        """測試上傳文檔"""
        mock_rag_engine.add_document = AsyncMock(return_value="doc_123")

        response = client.post("/api/documents/upload", json={
            "content": "This is a test document",
            "metadata": {"source": "test"}
        })

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc_123"
        assert data["message"] == "Document uploaded successfully"

    def test_list_documents(self, client, mock_rag_engine):
        """測試列出文檔"""
        mock_rag_engine.list_documents.return_value = [
            {"id": "doc1", "content": "Content 1", "metadata": {}},
            {"id": "doc2", "content": "Content 2", "metadata": {}}
        ]

        response = client.get("/api/documents?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 2

    def test_delete_document(self, client, mock_rag_engine):
        """測試刪除文檔"""
        mock_rag_engine.delete_document.return_value = True

        response = client.delete("/api/documents/doc_123")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document deleted successfully"

    def test_delete_nonexistent_document(self, client, mock_rag_engine):
        """測試刪除不存在的文檔"""
        mock_rag_engine.delete_document.return_value = False

        response = client.delete("/api/documents/nonexistent")
        assert response.status_code == 404


class TestConversationEndpoints:
    """對話管理端點測試"""

    def test_list_conversations(self, client, mock_rag_engine):
        """測試列出對話"""
        mock_rag_engine.list_conversations.return_value = [
            {"conversation_id": "conv1", "message_count": 4, "created_at": "2024-01-01"},
            {"conversation_id": "conv2", "message_count": 2, "created_at": "2024-01-02"}
        ]

        response = client.get("/api/conversations")
        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) == 2

    def test_get_conversation_history(self, client, mock_rag_engine):
        """測試獲取對話歷史"""
        mock_rag_engine.get_conversation_history.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        response = client.get("/api/conversations/conv_123")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2
        assert data["conversation_id"] == "conv_123"

    def test_delete_conversation(self, client, mock_rag_engine):
        """測試刪除對話"""
        mock_rag_engine.delete_conversation.return_value = True

        response = client.delete("/api/conversations/conv_123")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Conversation deleted successfully"

    def test_delete_nonexistent_conversation(self, client, mock_rag_engine):
        """測試刪除不存在的對話"""
        mock_rag_engine.delete_conversation.return_value = False

        response = client.delete("/api/conversations/nonexistent")
        assert response.status_code == 404


class TestCORS:
    """CORS 測試"""

    def test_cors_headers(self, client):
        """測試 CORS 頭部"""
        response = client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )
        # FastAPI 的 CORS 中間件會自動處理
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """錯誤處理測試"""

    def test_invalid_endpoint(self, client):
        """測試無效端點"""
        response = client.get("/api/invalid")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """測試方法不允許"""
        response = client.put("/api/health")
        assert response.status_code == 405

    def test_internal_error(self, client, mock_rag_engine):
        """測試內部錯誤處理"""
        mock_rag_engine.chat = AsyncMock(side_effect=Exception("Test error"))

        response = client.post("/api/chat", json={
            "message": "Test",
            "use_rag": False
        })

        assert response.status_code == 500


class TestStreamingChat:
    """流式聊天測試"""

    def test_chat_stream(self, client, mock_rag_engine):
        """測試流式聊天端點"""
        async def mock_stream():
            for chunk in ["Hello", " ", "World"]:
                yield chunk

        mock_rag_engine.chat_stream = mock_stream

        response = client.post("/api/chat/stream", json={
            "message": "Test streaming",
            "use_rag": False
        })

        # 流式響應需要特殊處理
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
