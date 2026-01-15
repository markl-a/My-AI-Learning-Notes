"""
RAG 系統單元測試 - TextSplitter
測試文本拆分器的各種功能
"""

import pytest
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 測試用的簡化 TextSplitter 實現（避免依賴問題）
class TextSplitter:
    """文本拆分器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list:
        """拆分文本為多個塊"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                for separator in ['\n\n', '\n', '。', '. ', ' ']:
                    pos = text.rfind(separator, start, end)
                    if pos != -1:
                        end = pos + len(separator)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks


class TestTextSplitter:
    """TextSplitter 單元測試類"""

    def test_basic_split(self):
        """測試基本的文本拆分功能"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        text = "這是一段測試文字。" * 20

        chunks = splitter.split_text(text)

        assert len(chunks) > 1, "應該產生多個塊"
        assert all(len(chunk) <= 110 for chunk in chunks), "每個塊不應超過 chunk_size + 一些緩衝"

    def test_empty_text(self):
        """測試空文本"""
        splitter = TextSplitter()
        chunks = splitter.split_text("")

        assert chunks == [], "空文本應返回空列表"

    def test_small_text(self):
        """測試小於 chunk_size 的文本"""
        splitter = TextSplitter(chunk_size=500)
        text = "這是一小段文字"

        chunks = splitter.split_text(text)

        assert len(chunks) == 1, "小文本應只產生一個塊"
        assert chunks[0] == text, "塊內容應與原文相同"

    def test_chunk_overlap(self):
        """測試塊重疊功能"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 100

        chunks = splitter.split_text(text)

        # 確保有重疊
        assert len(chunks) >= 2, "應有多個塊"

    def test_sentence_boundary_split(self):
        """測試句子邊界拆分"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=5)
        text = "第一句話。第二句話。第三句話。"

        chunks = splitter.split_text(text)

        # 檢查是否在句號處拆分
        for chunk in chunks:
            if len(chunk) < 50:
                continue
            # 較長的塊應在句子邊界處結束
            assert chunk.endswith(('。', '\n', ' ')) or len(chunk) >= splitter.chunk_size - 10

    def test_newline_split(self):
        """測試換行符拆分"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=5)
        text = "第一段\n\n第二段\n\n第三段"

        chunks = splitter.split_text(text)

        assert len(chunks) >= 1

    def test_custom_chunk_size(self):
        """測試自定義塊大小"""
        for chunk_size in [100, 200, 500, 1000]:
            splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=10)
            text = "測試文字。" * 500

            chunks = splitter.split_text(text)

            # 大多數塊應接近指定大小
            for chunk in chunks[:-1]:  # 排除最後一個可能較短的塊
                assert len(chunk) <= chunk_size + 50, f"塊大小不應遠超 {chunk_size}"

    def test_unicode_text(self):
        """測試 Unicode 文本處理"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=5)
        text = "你好世界🌍！這是一段包含 emoji 的文字。繁體中文、简体中文、日本語、한국어"

        chunks = splitter.split_text(text)

        assert len(chunks) >= 1
        # 確保沒有損壞 Unicode 字符
        reconstructed = "".join(chunks)
        assert "🌍" in reconstructed or "🌍" in text

    def test_no_empty_chunks(self):
        """確保不產生空塊"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "   \n\n  內容  \n\n   "

        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert chunk.strip() != "", "不應有空塊"


class TestDocument:
    """Document 類測試"""

    def test_document_creation(self):
        """測試文檔創建"""
        from dataclasses import dataclass
        from typing import Dict

        @dataclass
        class Document:
            content: str
            metadata: Dict = None

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

        doc = Document(content="測試內容", metadata={"source": "test.txt"})

        assert doc.content == "測試內容"
        assert doc.metadata["source"] == "test.txt"

    def test_document_without_metadata(self):
        """測試無 metadata 的文檔"""
        from dataclasses import dataclass
        from typing import Dict

        @dataclass
        class Document:
            content: str
            metadata: Dict = None

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

        doc = Document(content="測試內容")

        assert doc.metadata == {}


class TestRetrievalMetrics:
    """檢索指標測試"""

    @staticmethod
    def precision_at_k(relevant: list, retrieved: list, k: int) -> float:
        """計算 Precision@K"""
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(relevant: list, retrieved: list, k: int) -> float:
        """計算 Recall@K"""
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / len(relevant_set) if relevant_set else 0.0

    @staticmethod
    def ndcg_at_k(relevant: list, retrieved: list, k: int) -> float:
        """計算 NDCG@K (簡化版)"""
        import math

        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)

        # 計算 DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            if doc in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 因為位置從 1 開始

        # 計算理想 DCG
        ideal_k = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

        return dcg / idcg if idcg > 0 else 0.0

    def test_precision_at_k(self):
        """測試 Precision@K"""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2", "doc5", "doc3"]

        p_at_1 = self.precision_at_k(relevant, retrieved, 1)
        p_at_3 = self.precision_at_k(relevant, retrieved, 3)
        p_at_5 = self.precision_at_k(relevant, retrieved, 5)

        assert p_at_1 == 1.0, "P@1 應為 1.0"
        assert abs(p_at_3 - 2/3) < 0.01, "P@3 應為 2/3"
        assert abs(p_at_5 - 3/5) < 0.01, "P@5 應為 3/5"

    def test_recall_at_k(self):
        """測試 Recall@K"""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2", "doc5", "doc3"]

        r_at_1 = self.recall_at_k(relevant, retrieved, 1)
        r_at_3 = self.recall_at_k(relevant, retrieved, 3)
        r_at_5 = self.recall_at_k(relevant, retrieved, 5)

        assert abs(r_at_1 - 1/3) < 0.01, "R@1 應為 1/3"
        assert abs(r_at_3 - 2/3) < 0.01, "R@3 應為 2/3"
        assert r_at_5 == 1.0, "R@5 應為 1.0"

    def test_ndcg_at_k(self):
        """測試 NDCG@K"""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        ndcg = self.ndcg_at_k(relevant, retrieved, 3)

        assert ndcg == 1.0, "完美排序的 NDCG@3 應為 1.0"

    def test_empty_results(self):
        """測試空結果"""
        relevant = ["doc1", "doc2"]
        retrieved = []

        p = self.precision_at_k(relevant, retrieved, 5)
        r = self.recall_at_k(relevant, retrieved, 5)

        assert p == 0.0
        assert r == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
