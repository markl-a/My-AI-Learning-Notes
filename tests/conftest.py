"""
Pytest 配置和共享 fixtures

提供測試所需的共享配置和 fixtures。
"""

import pytest
import sys
import os
import tempfile
import shutil

# 確保專案根目錄在 Python 路徑中
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root():
    """返回專案根目錄路徑"""
    return PROJECT_ROOT


@pytest.fixture
def temp_dir():
    """創建臨時目錄，測試後自動清理"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file():
    """創建臨時文件，測試後自動清理"""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_text():
    """返回範例文本"""
    return """
    機器學習是人工智能的一個分支，它使計算機能夠從數據中學習並做出決策或預測，
    而無需明確編程。深度學習是機器學習的一個子領域，使用多層神經網絡來學習
    數據的層次表示。
    """


@pytest.fixture
def sample_documents():
    """返回範例文檔列表"""
    return [
        {
            "id": "doc1",
            "content": "機器學習基礎介紹",
            "metadata": {"category": "ml", "language": "zh"}
        },
        {
            "id": "doc2",
            "content": "深度學習與神經網絡",
            "metadata": {"category": "dl", "language": "zh"}
        },
        {
            "id": "doc3",
            "content": "Introduction to AI",
            "metadata": {"category": "ai", "language": "en"}
        }
    ]


@pytest.fixture
def mock_llm_response():
    """模擬 LLM 回應"""
    return {
        "content": "這是一個模擬的 LLM 回應。",
        "model": "gpt-4",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


# 標記定義
def pytest_configure(config):
    """添加自定義標記"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require external API access"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
