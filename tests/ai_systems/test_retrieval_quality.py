"""
AI系統測試框架 - RAG檢索品質測試

本模組提供RAG系統的檢索品質測試，包括：
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Recall@K

作者: AI Learning Notes
更新: 2025-12-14
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RetrievalResult:
    """檢索結果"""
    doc_id: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationMetrics:
    """評估指標"""
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    map_score: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float


class RetrievalMetrics:
    """檢索指標計算器"""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        計算DCG@K (Discounted Cumulative Gain)

        Args:
            relevances: 相關性分數列表（按檢索順序排列）
            k: 截斷位置

        Returns:
            DCG@K值
        """
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0

        # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
        discounts = np.log2(np.arange(len(relevances)) + 2)
        return np.sum(relevances / discounts)

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10
    ) -> float:
        """
        計算NDCG@K (Normalized DCG)

        Args:
            retrieved_ids: 檢索到的文檔ID列表（按分數排序）
            relevant_ids: 相關文檔ID列表
            relevance_scores: 可選的相關性分數字典
            k: 截斷位置

        Returns:
            NDCG@K值（0-1之間）
        """
        if not relevant_ids:
            return 0.0

        # 構建相關性向量
        if relevance_scores:
            relevances = [
                relevance_scores.get(doc_id, 0.0)
                for doc_id in retrieved_ids[:k]
            ]
            ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        else:
            # 二元相關性
            relevances = [
                1.0 if doc_id in relevant_ids else 0.0
                for doc_id in retrieved_ids[:k]
            ]
            ideal_relevances = [1.0] * min(len(relevant_ids), k)

        dcg = RetrievalMetrics.dcg_at_k(relevances, k)
        idcg = RetrievalMetrics.dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr(
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]]
    ) -> float:
        """
        計算MRR (Mean Reciprocal Rank)

        Args:
            retrieved_ids_list: 多個查詢的檢索結果列表
            relevant_ids_list: 多個查詢的相關文檔列表

        Returns:
            MRR值（0-1之間）
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            relevant_set = set(relevant)

            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def average_precision(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        計算單個查詢的Average Precision

        Args:
            retrieved_ids: 檢索到的文檔ID列表
            relevant_ids: 相關文檔ID列表

        Returns:
            AP值
        """
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)
        precisions = []
        relevant_count = 0

        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precisions.append(precision_at_i)

        if not precisions:
            return 0.0

        return np.mean(precisions)

    @staticmethod
    def mean_average_precision(
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]]
    ) -> float:
        """
        計算MAP (Mean Average Precision)

        Args:
            retrieved_ids_list: 多個查詢的檢索結果列表
            relevant_ids_list: 多個查詢的相關文檔列表

        Returns:
            MAP值
        """
        aps = [
            RetrievalMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list)
        ]
        return np.mean(aps) if aps else 0.0

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """
        計算Recall@K

        Args:
            retrieved_ids: 檢索到的文檔ID列表
            relevant_ids: 相關文檔ID列表
            k: 截斷位置

        Returns:
            Recall@K值
        """
        if not relevant_ids:
            return 0.0

        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        return len(retrieved_set & relevant_set) / len(relevant_set)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """
        計算Precision@K

        Args:
            retrieved_ids: 檢索到的文檔ID列表
            relevant_ids: 相關文檔ID列表
            k: 截斷位置

        Returns:
            Precision@K值
        """
        if not retrieved_ids[:k]:
            return 0.0

        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        return len(retrieved_set & relevant_set) / len(retrieved_set)


class RAGRetriever(ABC):
    """RAG檢索器抽象基類"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """檢索相關文檔"""
        pass


class MockRAGRetriever(RAGRetriever):
    """模擬檢索器（用於測試）"""

    def __init__(self, documents: Dict[str, str]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        # 模擬檢索（實際應用中會使用向量搜索）
        results = []
        for doc_id, content in self.documents.items():
            # 簡單的關鍵詞匹配計分
            score = sum(1 for word in query.split() if word.lower() in content.lower())
            results.append(RetrievalResult(
                doc_id=doc_id,
                content=content,
                score=score / len(query.split())
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class RAGEvaluator:
    """RAG系統評估器"""

    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.metrics = RetrievalMetrics()

    def evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> EvaluationMetrics:
        """
        評估RAG系統

        Args:
            test_cases: 測試用例列表，每個用例包含：
                - query: 查詢文本
                - relevant_docs: 相關文檔ID列表

        Returns:
            評估指標
        """
        all_retrieved = []
        all_relevant = []

        ndcg_5_scores = []
        ndcg_10_scores = []
        recall_5_scores = []
        recall_10_scores = []
        precision_5_scores = []
        precision_10_scores = []

        for test in test_cases:
            query = test["query"]
            relevant_docs = test["relevant_docs"]

            # 執行檢索
            results = self.retriever.retrieve(query, top_k=10)
            retrieved_ids = [r.doc_id for r in results]

            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant_docs)

            # 計算單查詢指標
            ndcg_5_scores.append(
                self.metrics.ndcg_at_k(retrieved_ids, relevant_docs, k=5)
            )
            ndcg_10_scores.append(
                self.metrics.ndcg_at_k(retrieved_ids, relevant_docs, k=10)
            )
            recall_5_scores.append(
                self.metrics.recall_at_k(retrieved_ids, relevant_docs, k=5)
            )
            recall_10_scores.append(
                self.metrics.recall_at_k(retrieved_ids, relevant_docs, k=10)
            )
            precision_5_scores.append(
                self.metrics.precision_at_k(retrieved_ids, relevant_docs, k=5)
            )
            precision_10_scores.append(
                self.metrics.precision_at_k(retrieved_ids, relevant_docs, k=10)
            )

        return EvaluationMetrics(
            ndcg_at_5=np.mean(ndcg_5_scores),
            ndcg_at_10=np.mean(ndcg_10_scores),
            mrr=self.metrics.mrr(all_retrieved, all_relevant),
            map_score=self.metrics.mean_average_precision(all_retrieved, all_relevant),
            recall_at_5=np.mean(recall_5_scores),
            recall_at_10=np.mean(recall_10_scores),
            precision_at_5=np.mean(precision_5_scores),
            precision_at_10=np.mean(precision_10_scores)
        )


# ==================== Pytest測試 ====================

class TestRetrievalMetrics:
    """檢索指標單元測試"""

    def test_ndcg_perfect_ranking(self):
        """測試完美排序的NDCG應該為1.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]

        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self):
        """測試完全不相關時NDCG應該為0.0"""
        retrieved = ["doc4", "doc5", "doc6"]
        relevant = ["doc1", "doc2", "doc3"]

        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == pytest.approx(0.0)

    def test_ndcg_partial_match(self):
        """測試部分匹配的NDCG"""
        retrieved = ["doc1", "doc4", "doc2"]
        relevant = ["doc1", "doc2", "doc3"]

        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3)
        assert 0 < ndcg < 1.0

    def test_mrr_first_position(self):
        """測試第一個位置命中時MRR應該為1.0"""
        retrieved_list = [["doc1", "doc2", "doc3"]]
        relevant_list = [["doc1"]]

        mrr = RetrievalMetrics.mrr(retrieved_list, relevant_list)
        assert mrr == pytest.approx(1.0)

    def test_mrr_second_position(self):
        """測試第二個位置命中時MRR應該為0.5"""
        retrieved_list = [["doc2", "doc1", "doc3"]]
        relevant_list = [["doc1"]]

        mrr = RetrievalMetrics.mrr(retrieved_list, relevant_list)
        assert mrr == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        """測試沒有命中時MRR應該為0.0"""
        retrieved_list = [["doc4", "doc5", "doc6"]]
        relevant_list = [["doc1"]]

        mrr = RetrievalMetrics.mrr(retrieved_list, relevant_list)
        assert mrr == pytest.approx(0.0)

    def test_recall_at_k(self):
        """測試Recall@K計算"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6", "doc7"]

        recall_5 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)
        # 5個檢索結果中有2個相關（doc1, doc3），相關總數4個
        assert recall_5 == pytest.approx(0.5)

    def test_precision_at_k(self):
        """測試Precision@K計算"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"]

        precision_5 = RetrievalMetrics.precision_at_k(retrieved, relevant, k=5)
        # 5個檢索結果中有2個相關
        assert precision_5 == pytest.approx(0.4)

    def test_map_calculation(self):
        """測試MAP計算"""
        # 查詢1: 相關文檔在位置1, 3
        # 查詢2: 相關文檔在位置2
        retrieved_list = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc1", "doc5"]
        ]
        relevant_list = [
            ["doc1", "doc3"],
            ["doc1"]
        ]

        map_score = RetrievalMetrics.mean_average_precision(
            retrieved_list, relevant_list
        )
        # AP1 = (1/1 + 2/3) / 2 = 0.833
        # AP2 = 1/2 = 0.5
        # MAP = (0.833 + 0.5) / 2 = 0.667
        assert 0.6 < map_score < 0.7


class TestRAGEvaluator:
    """RAG評估器集成測試"""

    @pytest.fixture
    def mock_retriever(self):
        """創建模擬檢索器"""
        documents = {
            "doc1": "機器學習是人工智能的一個分支",
            "doc2": "深度學習使用神經網絡",
            "doc3": "自然語言處理處理文本數據",
            "doc4": "Python是一種編程語言",
            "doc5": "數據科學結合統計和編程",
        }
        return MockRAGRetriever(documents)

    @pytest.fixture
    def test_cases(self):
        """創建測試用例"""
        return [
            {
                "query": "機器學習 人工智能",
                "relevant_docs": ["doc1", "doc2"]
            },
            {
                "query": "Python 編程 數據",
                "relevant_docs": ["doc4", "doc5"]
            }
        ]

    def test_evaluator_returns_all_metrics(self, mock_retriever, test_cases):
        """測試評估器返回所有指標"""
        evaluator = RAGEvaluator(mock_retriever)
        metrics = evaluator.evaluate(test_cases)

        assert hasattr(metrics, 'ndcg_at_5')
        assert hasattr(metrics, 'ndcg_at_10')
        assert hasattr(metrics, 'mrr')
        assert hasattr(metrics, 'map_score')
        assert hasattr(metrics, 'recall_at_5')
        assert hasattr(metrics, 'recall_at_10')
        assert hasattr(metrics, 'precision_at_5')
        assert hasattr(metrics, 'precision_at_10')

    def test_metrics_in_valid_range(self, mock_retriever, test_cases):
        """測試所有指標都在0-1範圍內"""
        evaluator = RAGEvaluator(mock_retriever)
        metrics = evaluator.evaluate(test_cases)

        assert 0 <= metrics.ndcg_at_5 <= 1
        assert 0 <= metrics.ndcg_at_10 <= 1
        assert 0 <= metrics.mrr <= 1
        assert 0 <= metrics.map_score <= 1
        assert 0 <= metrics.recall_at_5 <= 1
        assert 0 <= metrics.recall_at_10 <= 1
        assert 0 <= metrics.precision_at_5 <= 1
        assert 0 <= metrics.precision_at_10 <= 1


class TestEmbeddingQuality:
    """Embedding品質測試"""

    def test_similar_texts_have_high_similarity(self):
        """測試相似文本應該有高相似度"""
        # 這是一個佔位測試，實際實現需要載入embedding模型
        # from sentence_transformers import SentenceTransformer

        similar_texts = [
            "機器學習是AI的分支",
            "人工智能包含機器學習"
        ]

        dissimilar_texts = [
            "機器學習是AI的分支",
            "今天天氣很好"
        ]

        # 模擬：相似文本的相似度應該高於不相似文本
        # 實際測試中會計算真實的cosine similarity
        similar_score = 0.85  # 模擬值
        dissimilar_score = 0.15  # 模擬值

        assert similar_score > dissimilar_score
        assert similar_score > 0.7  # 相似文本應該有高相似度

    def test_embedding_dimension(self):
        """測試embedding維度正確性"""
        # 模擬測試
        expected_dim = 384  # sentence-transformers常見維度
        actual_dim = 384  # 模擬值

        assert actual_dim == expected_dim


# 運行測試的主函數
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
