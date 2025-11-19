"""
Pytest 配置和共享 Fixtures
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_env():
    """設置測試環境變量"""
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["TESTING"] = "true"
    yield
    # 清理
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI 客戶端"""
    with patch('openai.AsyncOpenAI') as mock:
        client = Mock()

        # Mock chat completion
        async def mock_create(*args, **kwargs):
            response = Mock()
            response.choices = [
                Mock(message=Mock(content="Test response"))
            ]
            response.usage = Mock(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
            return response

        client.chat.completions.create = mock_create
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock Sentence Transformer"""
    with patch('sentence_transformers.SentenceTransformer') as mock:
        encoder = Mock()
        encoder.encode = Mock(return_value=[0.1, 0.2, 0.3])
        mock.return_value = encoder
        yield mock


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB"""
    with patch('chromadb.Client') as mock:
        client = Mock()
        collection = Mock()

        # Mock collection methods
        collection.add = Mock()
        collection.query = Mock(return_value={
            'documents': [['test doc']],
            'metadatas': [[{'source': 'test'}]],
            'distances': [[0.1]]
        })
        collection.get = Mock(return_value={
            'ids': ['id1'],
            'documents': ['doc1'],
            'metadatas': [{'source': 'test'}]
        })
        collection.delete = Mock()
        collection.count = Mock(return_value=10)

        client.get_or_create_collection = Mock(return_value=collection)
        mock.return_value = client
        yield mock


@pytest.fixture
def sample_documents():
    """示例文檔數據"""
    return [
        {
            "content": "Python is a high-level programming language.",
            "metadata": {"topic": "programming", "language": "en"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"topic": "AI", "language": "en"}
        },
        {
            "content": "RAG combines retrieval and generation for better responses.",
            "metadata": {"topic": "NLP", "language": "en"}
        }
    ]


@pytest.fixture
def sample_conversations():
    """示例對話數據"""
    return {
        "conv1": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ],
        "conv2": [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
        ]
    }


@pytest.fixture
def sample_chat_requests():
    """示例聊天請求"""
    return [
        {
            "message": "Hello",
            "use_rag": False,
            "top_k": 3
        },
        {
            "message": "What is Python?",
            "use_rag": True,
            "top_k": 5,
            "conversation_id": "test_conv"
        }
    ]


@pytest.fixture
def sample_upload_requests():
    """示例文檔上傳請求"""
    return [
        {
            "content": "Test document content",
            "metadata": {"source": "test", "type": "text"}
        },
        {
            "content": "Another test document",
            "metadata": {"source": "api", "category": "example"}
        }
    ]


def pytest_configure(config):
    """Pytest 配置鉤子"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """修改測試收集"""
    skip_integration = pytest.mark.skip(reason="Integration tests disabled by default")

    for item in items:
        if "integration" in item.keywords:
            if not config.getoption("--run-integration"):
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """添加命令行選項"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="OpenAI API key for integration tests"
    )
