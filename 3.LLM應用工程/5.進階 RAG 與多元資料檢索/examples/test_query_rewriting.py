"""
測試查詢改寫和 HyDE 功能

這個測試檔案驗證：
1. Query Rewriter 的各種功能
2. HyDE 檢索器的功能
3. 整合系統的端到端測試

運行方式：
    python test_query_rewriting.py
    或
    pytest test_query_rewriting.py -v
"""

import os
import sys
from typing import List
import pytest
from dotenv import load_dotenv

# 確保可以導入主模組
sys.path.insert(0, os.path.dirname(__file__))

from query_rewriting_hyde import (
    QueryRewriter,
    HyDERetriever,
    AdvancedQueryRAG,
    QueryResult
)

# Load environment variables
load_dotenv()

# 測試用的簡單文檔集
TEST_DOCUMENTS = [
    "Python 是一種高級編程語言，以其簡潔和可讀性著稱。",
    "機器學習是人工智能的一個子領域，專注於讓計算機從數據中學習。",
    "深度學習使用多層神經網絡來處理複雜的模式識別任務。",
    "自然語言處理（NLP）讓計算機能夠理解和生成人類語言。"
]


class TestQueryRewriter:
    """測試 QueryRewriter 類別"""

    @pytest.fixture
    def rewriter(self):
        """創建 QueryRewriter 實例"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("需要 OPENAI_API_KEY 環境變數")
        return QueryRewriter()

    def test_rewrite_query(self, rewriter):
        """測試基本的查詢改寫功能"""
        original = "python 好用嗎"
        rewritten = rewriter.rewrite_query(original)

        assert isinstance(rewritten, str)
        assert len(rewritten) > 0
        assert rewritten != original  # 改寫後應該不同
        print(f"\n原始: {original}")
        print(f"改寫: {rewritten}")

    def test_generate_multi_queries(self, rewriter):
        """測試多查詢生成功能"""
        query = "什麼是機器學習？"
        multi_queries = rewriter.generate_multi_queries(query, n=3)

        assert isinstance(multi_queries, list)
        assert len(multi_queries) <= 3
        assert all(isinstance(q, str) for q in multi_queries)
        assert all(len(q) > 0 for q in multi_queries)

        print(f"\n原始查詢: {query}")
        print("生成的變體:")
        for i, mq in enumerate(multi_queries, 1):
            print(f"  {i}. {mq}")

    def test_expand_query(self, rewriter):
        """測試查詢擴展功能"""
        query = "深度學習"
        expanded = rewriter.expand_query(query, expansion_type="semantic")

        assert isinstance(expanded, str)
        assert len(expanded) >= len(query)
        print(f"\n原始: {query}")
        print(f"擴展: {expanded}")


class TestAdvancedQueryRAG:
    """測試完整的進階 RAG 系統"""

    @pytest.fixture
    def rag_system(self):
        """創建並初始化 RAG 系統"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("需要 OPENAI_API_KEY 環境變數")

        system = AdvancedQueryRAG(persist_directory="./test_chroma_db")
        system.ingest_documents(TEST_DOCUMENTS)
        return system

    def test_standard_retrieval(self, rag_system):
        """測試標準檢索"""
        result = rag_system.query(
            "Python 編程",
            method="standard",
            top_k=2,
            verbose=False
        )

        assert isinstance(result, QueryResult)
        assert result.retrieved_docs is not None
        assert len(result.retrieved_docs) > 0
        assert len(result.retrieved_docs) <= 2
        print(f"\n標準檢索找到 {len(result.retrieved_docs)} 個文檔")

    def test_rewrite_retrieval(self, rag_system):
        """測試查詢改寫檢索"""
        result = rag_system.query(
            "python 好嗎",
            method="rewrite",
            top_k=2,
            verbose=False
        )

        assert result.rewritten_query is not None
        assert isinstance(result.rewritten_query, str)
        assert len(result.rewritten_query) > 0
        print(f"\n原始: python 好嗎")
        print(f"改寫: {result.rewritten_query}")

    def test_multi_query_retrieval(self, rag_system):
        """測試多查詢檢索"""
        result = rag_system.query(
            "機器學習是什麼",
            method="multi_query",
            top_k=3,
            verbose=False
        )

        assert result.multi_queries is not None
        assert isinstance(result.multi_queries, list)
        assert len(result.multi_queries) > 0
        print(f"\n生成了 {len(result.multi_queries)} 個查詢變體")

    def test_hyde_retrieval(self, rag_system):
        """測試 HyDE 檢索"""
        result = rag_system.query(
            "解釋深度學習的原理",
            method="hyde",
            top_k=2,
            verbose=False
        )

        assert result.hypothetical_doc is not None
        assert isinstance(result.hypothetical_doc, str)
        assert len(result.hypothetical_doc) > 50  # 假設文檔應該有一定長度
        print(f"\n假設文檔長度: {len(result.hypothetical_doc)} 字符")
        print(f"假設文檔預覽: {result.hypothetical_doc[:100]}...")

    def test_hybrid_retrieval(self, rag_system):
        """測試混合檢索"""
        result = rag_system.query(
            "NLP 技術",
            method="hybrid",
            top_k=3,
            verbose=False
        )

        assert result.rewritten_query is not None
        assert result.multi_queries is not None
        assert result.retrieved_docs is not None
        print(f"\n混合檢索整合了多種技術")
        print(f"檢索到 {len(result.retrieved_docs)} 個文檔")

    def test_query_and_answer(self, rag_system):
        """測試完整的查詢和回答流程"""
        response = rag_system.query_and_answer(
            "Python 是什麼",
            method="hybrid",
            top_k=2,
            verbose=False
        )

        assert "answer" in response
        assert "query" in response
        assert "retrieved_docs" in response
        assert isinstance(response["answer"], str)
        assert len(response["answer"]) > 0

        print(f"\n問題: {response['query']}")
        print(f"答案: {response['answer']}")
        print(f"使用了 {len(response['retrieved_docs'])} 個文檔")


def run_manual_tests():
    """
    手動運行測試（不使用 pytest）

    適合快速驗證和演示
    """
    print("=" * 80)
    print("手動測試：查詢改寫與 HyDE")
    print("=" * 80)
    print()

    # 檢查 API 金鑰
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 錯誤: 需要設置 OPENAI_API_KEY 環境變數")
        print("提示: 複製 .env.example 到 .env 並填入你的 API 金鑰")
        return

    try:
        # 測試 Query Rewriter
        print("1. 測試 Query Rewriter")
        print("-" * 80)
        rewriter = QueryRewriter()

        test_query = "transformer 模型怎麼用"
        print(f"原始查詢: {test_query}")

        rewritten = rewriter.rewrite_query(test_query)
        print(f"改寫查詢: {rewritten}")

        multi_queries = rewriter.generate_multi_queries(test_query, n=3)
        print(f"多查詢變體:")
        for i, mq in enumerate(multi_queries, 1):
            print(f"  {i}. {mq}")

        print("\n✓ Query Rewriter 測試通過\n")

        # 測試完整 RAG 系統
        print("2. 測試完整 RAG 系統")
        print("-" * 80)

        rag = AdvancedQueryRAG(persist_directory="./test_chroma_manual")
        print("✓ 系統初始化完成")

        rag.ingest_documents(TEST_DOCUMENTS)
        print("✓ 文檔攝取完成")

        # 測試不同方法
        methods_to_test = ["standard", "rewrite", "hyde"]

        for method in methods_to_test:
            print(f"\n測試方法: {method}")
            result = rag.query(
                "什麼是機器學習",
                method=method,
                top_k=2,
                verbose=False
            )
            print(f"  ✓ 檢索到 {len(result.retrieved_docs)} 個文檔")

        print("\n✓ RAG 系統測試通過\n")

        # 測試端到端
        print("3. 測試端到端查詢回答")
        print("-" * 80)

        response = rag.query_and_answer(
            "深度學習和機器學習有什麼關係",
            method="hybrid",
            top_k=2,
            verbose=False
        )

        print(f"問題: {response['query']}")
        print(f"答案: {response['answer']}")
        print(f"\n✓ 端到端測試通過\n")

        print("=" * 80)
        print("✅ 所有測試通過！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果直接運行此文件，執行手動測試
    run_manual_tests()
