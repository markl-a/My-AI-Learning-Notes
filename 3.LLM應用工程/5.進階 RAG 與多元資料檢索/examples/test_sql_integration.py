"""
測試 SQL + RAG 整合系統

運行方式：
    python test_sql_integration.py
    或
    pytest test_sql_integration.py -v
"""

import os
import sys
import sqlite3
import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from sql_integration import (
    SQLRAGIntegration,
    QueryType,
    HybridQueryResult
)

load_dotenv()

TEST_DB_PATH = "./test_company_db.sqlite"


class TestSQLRAGIntegration:
    """測試 SQL + RAG 整合系統"""

    @pytest.fixture
    def sql_rag_system(self):
        """創建測試系統"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("需要 OPENAI_API_KEY 環境變數")

        # 清理舊測試資料庫
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

        system = SQLRAGIntegration(
            db_path=TEST_DB_PATH,
            vector_store_path="./test_chroma_sql"
        )

        # 創建示例資料庫
        system.create_sample_database()
        system.init_sql_database()

        # 準備文檔
        documents = [
            "公司提供年假、病假和事假。年假工作滿一年有7天。",
            "員工福利包括三節獎金、健康檢查和訓練課程。",
            "產品保固政策：電子產品一年保固，家具六個月保固。"
        ]
        system.ingest_documents(documents)

        yield system

        # 清理
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_database_creation(self, sql_rag_system):
        """測試資料庫創建"""
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()

        # 檢查表格是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "employees" in tables
        assert "products" in tables
        assert "sales" in tables

        # 檢查數據
        cursor.execute("SELECT COUNT(*) FROM employees")
        employee_count = cursor.fetchone()[0]
        assert employee_count > 0

        conn.close()
        print(f"\n✓ 資料庫創建成功，employees 表有 {employee_count} 筆記錄")

    def test_query_classification(self, sql_rag_system):
        """測試查詢分類"""
        # SQL 查詢
        query_type, _ = sql_rag_system.classify_query("技術部有多少員工？")
        print(f"\n查詢: '技術部有多少員工？' -> {query_type.value}")
        assert query_type in [QueryType.SQL, QueryType.HYBRID]

        # 向量查詢
        query_type, _ = sql_rag_system.classify_query("公司的休假政策是什麼？")
        print(f"查詢: '公司的休假政策是什麼？' -> {query_type.value}")
        assert query_type in [QueryType.VECTOR, QueryType.HYBRID]

    def test_nl_to_sql(self, sql_rag_system):
        """測試自然語言轉 SQL"""
        query = "列出所有員工的姓名和部門"
        sql_query = sql_rag_system.natural_language_to_sql(query)

        assert isinstance(sql_query, str)
        assert "SELECT" in sql_query.upper()
        assert "employees" in sql_query.lower()

        print(f"\n自然語言: {query}")
        print(f"生成的 SQL: {sql_query}")

    def test_sql_execution(self, sql_rag_system):
        """測試 SQL 執行"""
        sql_query = "SELECT name, department FROM employees LIMIT 3"
        results = sql_rag_system.execute_sql_safely(sql_query)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "name" in results[0]
        assert "department" in results[0]

        print(f"\n執行 SQL: {sql_query}")
        print(f"結果: {len(results)} 筆記錄")
        print(f"第一筆: {results[0]}")

    def test_sql_safety(self, sql_rag_system):
        """測試 SQL 安全性"""
        # 測試危險操作被阻止
        dangerous_queries = [
            "DROP TABLE employees",
            "DELETE FROM employees",
            "UPDATE employees SET salary = 0",
            "INSERT INTO employees VALUES (99, 'hacker', 'IT', 'hacker', 99999, '2024-01-01')"
        ]

        for query in dangerous_queries:
            with pytest.raises(ValueError):
                sql_rag_system.execute_sql_safely(query)

        print("\n✓ 危險 SQL 操作已被成功阻止")

    def test_vector_search(self, sql_rag_system):
        """測試向量檢索"""
        docs = sql_rag_system.vector_search("休假政策", top_k=2)

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert hasattr(docs[0], 'page_content')

        print(f"\n檢索 '休假政策'，找到 {len(docs)} 個文檔")
        print(f"第一個文檔預覽: {docs[0].page_content[:100]}...")

    def test_hybrid_query_sql(self, sql_rag_system):
        """測試 SQL 類型的混合查詢"""
        result = sql_rag_system.hybrid_query(
            "技術部有哪些員工？",
            verbose=False
        )

        assert isinstance(result, HybridQueryResult)
        assert result.query_type in [QueryType.SQL, QueryType.HYBRID]
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

        print(f"\n查詢: {result.query}")
        print(f"類型: {result.query_type.value}")
        print(f"答案: {result.final_answer[:150]}...")

    def test_hybrid_query_vector(self, sql_rag_system):
        """測試向量類型的混合查詢"""
        result = sql_rag_system.hybrid_query(
            "公司有哪些福利？",
            verbose=False
        )

        assert isinstance(result, HybridQueryResult)
        assert result.query_type in [QueryType.VECTOR, QueryType.HYBRID]
        assert result.final_answer is not None

        print(f"\n查詢: {result.query}")
        print(f"類型: {result.query_type.value}")
        print(f"答案: {result.final_answer[:150]}...")

    def test_hybrid_query_combined(self, sql_rag_system):
        """測試真正的混合查詢（需要 SQL + 向量）"""
        result = sql_rag_system.hybrid_query(
            "業務部員工的薪資情況和相關福利政策",
            verbose=False
        )

        assert isinstance(result, HybridQueryResult)
        assert result.final_answer is not None

        print(f"\n混合查詢: {result.query}")
        print(f"類型: {result.query_type.value}")
        print(f"答案長度: {len(result.final_answer)} 字符")


def run_manual_tests():
    """手動運行測試"""
    print("=" * 80)
    print("SQL + RAG 整合系統測試")
    print("=" * 80)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 錯誤: 需要設置 OPENAI_API_KEY 環境變數")
        return

    try:
        # 清理舊資料庫
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

        # 初始化系統
        print("1. 初始化系統")
        print("-" * 80)
        system = SQLRAGIntegration(
            db_path=TEST_DB_PATH,
            vector_store_path="./test_chroma_manual_sql"
        )
        print("✓ 系統初始化完成\n")

        # 創建資料庫
        print("2. 創建示例資料庫")
        print("-" * 80)
        system.create_sample_database()
        system.init_sql_database()
        print()

        # 攝取文檔
        print("3. 攝取文檔")
        print("-" * 80)
        documents = [
            "公司提供完整的休假制度，包括年假、病假和事假。",
            "員工福利包括三節獎金、健康檢查、訓練課程和團體保險。",
            "所有產品都有保固，電子產品一年，家具六個月。"
        ]
        system.ingest_documents(documents)
        print()

        # 測試查詢
        print("4. 測試不同類型的查詢")
        print("-" * 80)

        test_cases = [
            ("技術部的員工名單", "SQL 查詢"),
            ("公司的福利政策", "向量檢索"),
            ("業務部員工和相關制度", "混合查詢")
        ]

        for query, description in test_cases:
            print(f"\n測試: {description}")
            print(f"查詢: {query}")
            print("-" * 40)

            result = system.hybrid_query(query, verbose=False)

            print(f"查詢類型: {result.query_type.value}")
            if result.sql_query:
                print(f"SQL: {result.sql_query}")
            if result.sql_results:
                print(f"SQL 結果: {len(result.sql_results)} 筆")
            if result.vector_docs:
                print(f"向量文檔: {len(result.vector_docs)} 個")
            print(f"答案: {result.final_answer[:200]}...")
            print()

        print("=" * 80)
        print("✅ 所有測試完成！")
        print("=" * 80)

        # 清理
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_tests()
