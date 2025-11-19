"""
RAG Engine 單元測試
測試 RAG 引擎的核心功能
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import RAGEngine


class TestRAGEngine:
    """RAG Engine 測試類"""

    @pytest.fixture
    async def rag_engine(self):
        """創建 RAG Engine 實例"""
        with patch('rag_engine.SentenceTransformer'), \
             patch('rag_engine.chromadb.Client'), \
             patch('rag_engine.AsyncOpenAI'):
            engine = RAGEngine(
                model_name="gpt-3.5-turbo",
                embedding_model="all-MiniLM-L6-v2",
                collection_name="test_collection"
            )
            yield engine

    @pytest.mark.asyncio
    async def test_initialization(self, rag_engine):
        """測試初始化"""
        assert rag_engine.model_name == "gpt-3.5-turbo"
        assert rag_engine.collection_name == "test_collection"
        assert rag_engine.conversations == {}

    @pytest.mark.asyncio
    async def test_add_document(self, rag_engine):
        """測試添加文檔"""
        # Mock encoder
        rag_engine.encoder.encode = Mock(return_value=[0.1, 0.2, 0.3])
        rag_engine.collection.add = Mock()

        doc_id = await rag_engine.add_document(
            content="This is a test document",
            metadata={"source": "test"}
        )

        assert doc_id is not None
        assert len(doc_id) == 36  # UUID length
        rag_engine.collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_documents(self, rag_engine):
        """測試文檔檢索"""
        # Mock encoder and collection
        rag_engine.encoder.encode = Mock(return_value=[0.1, 0.2, 0.3])
        rag_engine.collection.query = Mock(return_value={
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]],
            'distances': [[0.1, 0.2]]
        })

        results = await rag_engine.retrieve(query="test query", top_k=2)

        assert len(results) == 2
        assert results[0]['content'] == 'doc1'
        assert results[0]['metadata']['source'] == 'test1'
        assert results[0]['distance'] == 0.1

    @pytest.mark.asyncio
    async def test_chat_without_rag(self, rag_engine):
        """測試不使用 RAG 的聊天"""
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )

        rag_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await rag_engine.chat(
            message="Hello",
            conversation_id="test_conv",
            use_rag=False
        )

        assert result['response'] == "Test response"
        assert result['conversation_id'] == "test_conv"
        assert result['tokens_used'] == 30

    @pytest.mark.asyncio
    async def test_chat_with_rag(self, rag_engine):
        """測試使用 RAG 的聊天"""
        # Mock retrieve
        rag_engine.retrieve = AsyncMock(return_value=[
            {'content': 'Context doc 1', 'metadata': {}, 'distance': 0.1},
            {'content': 'Context doc 2', 'metadata': {}, 'distance': 0.2}
        ])

        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="RAG response"))]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80
        )

        rag_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await rag_engine.chat(
            message="What is RAG?",
            conversation_id="test_conv",
            use_rag=True,
            top_k=2
        )

        assert result['response'] == "RAG response"
        assert len(result['context_used']) == 2
        assert result['tokens_used'] == 80

    @pytest.mark.asyncio
    async def test_conversation_history(self, rag_engine):
        """測試對話歷史管理"""
        conv_id = "test_conv"

        # Mock OpenAI responses
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20
        )
        rag_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # 第一輪對話
        await rag_engine.chat(
            message="First message",
            conversation_id=conv_id,
            use_rag=False
        )

        # 第二輪對話
        await rag_engine.chat(
            message="Second message",
            conversation_id=conv_id,
            use_rag=False
        )

        # 檢查歷史
        history = rag_engine.get_conversation_history(conv_id)
        assert len(history) == 4  # 2 user + 2 assistant messages
        assert history[0]['role'] == 'user'
        assert history[1]['role'] == 'assistant'

    @pytest.mark.asyncio
    async def test_delete_conversation(self, rag_engine):
        """測試刪除對話"""
        conv_id = "test_conv"
        rag_engine.conversations[conv_id] = [
            {"role": "user", "content": "test"}
        ]

        result = rag_engine.delete_conversation(conv_id)
        assert result is True
        assert conv_id not in rag_engine.conversations

        # 測試刪除不存在的對話
        result = rag_engine.delete_conversation("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_documents(self, rag_engine):
        """測試列出文檔"""
        rag_engine.collection.get = Mock(return_value={
            'ids': ['id1', 'id2'],
            'documents': ['doc1', 'doc2'],
            'metadatas': [{'source': 'test1'}, {'source': 'test2'}]
        })

        docs = rag_engine.list_documents(limit=10)
        assert len(docs) == 2
        assert docs[0]['id'] == 'id1'
        assert docs[0]['content'] == 'doc1'

    @pytest.mark.asyncio
    async def test_delete_document(self, rag_engine):
        """測試刪除文檔"""
        rag_engine.collection.delete = Mock()

        result = rag_engine.delete_document("test_id")
        assert result is True
        rag_engine.collection.delete.assert_called_once_with(ids=["test_id"])

    @pytest.mark.asyncio
    async def test_get_stats(self, rag_engine):
        """測試獲取統計信息"""
        rag_engine.collection.count = Mock(return_value=100)
        rag_engine.conversations = {
            'conv1': [{"role": "user", "content": "test"}],
            'conv2': [{"role": "user", "content": "test"}]
        }

        stats = rag_engine.get_stats()
        assert stats['total_documents'] == 100
        assert stats['total_conversations'] == 2
        assert stats['model_name'] == "gpt-3.5-turbo"


class TestRAGEngineIntegration:
    """RAG Engine 集成測試"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_rag_workflow(self):
        """測試完整的 RAG 工作流"""
        # 這需要真實的 API key 和服務
        # 在 CI/CD 中可以使用環境變量控制是否執行
        if not os.getenv('RUN_INTEGRATION_TESTS'):
            pytest.skip("Integration tests disabled")

        engine = RAGEngine(
            model_name="gpt-3.5-turbo",
            embedding_model="all-MiniLM-L6-v2"
        )

        # 添加文檔
        doc_id = await engine.add_document(
            content="Python is a high-level programming language.",
            metadata={"topic": "programming"}
        )
        assert doc_id is not None

        # 檢索
        results = await engine.retrieve("What is Python?", top_k=1)
        assert len(results) > 0

        # 聊天
        response = await engine.chat(
            message="Tell me about Python",
            use_rag=True
        )
        assert 'response' in response
        assert len(response['response']) > 0


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v", "--tb=short"])
