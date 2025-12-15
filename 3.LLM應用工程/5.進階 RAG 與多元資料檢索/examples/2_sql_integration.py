"""
SQL 資料庫整合 RAG 系統

這個模組展示如何將 RAG 與結構化資料庫（SQL）整合：
1. 自然語言轉 SQL 查詢
2. SQL 查詢結果與向量檢索結合
3. 混合資料源（結構化 + 非結構化）
4. 智能路由：決定使用 SQL 還是向量檢索

使用場景：
- 企業內部知識庫（文檔 + 資料庫）
- 數據分析問答系統
- 客戶服務系統（FAQ + 訂單資料）
"""

import os
import sqlite3
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from dotenv import load_dotenv

load_dotenv()


class QueryType(Enum):
    """查詢類型"""
    SQL = "sql"  # 需要查詢資料庫
    VECTOR = "vector"  # 需要向量檢索
    HYBRID = "hybrid"  # 需要兩者結合
    UNKNOWN = "unknown"  # 無法確定


@dataclass
class HybridQueryResult:
    """混合查詢結果"""
    query: str
    query_type: QueryType
    sql_query: Optional[str] = None
    sql_results: Optional[List[Dict]] = None
    vector_docs: Optional[List[Document]] = None
    final_answer: Optional[str] = None
    reasoning: Optional[str] = None


class SQLRAGIntegration:
    """
    SQL + RAG 整合系統

    核心功能：
    1. 智能路由：分析查詢並決定使用 SQL、向量檢索或兩者
    2. NL2SQL：將自然語言轉換為 SQL 查詢
    3. 結果融合：整合結構化和非結構化數據的結果
    4. 安全執行：防止 SQL 注入和危險操作
    """

    def __init__(
        self,
        db_path: str,
        vector_store_path: str = "./chroma_db_sql",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        初始化 SQL + RAG 系統

        Args:
            db_path: SQLite 資料庫路徑
            vector_store_path: 向量數據庫路徑
            llm_model: LLM 模型名稱
        """
        self.db_path = db_path
        self.vector_store_path = vector_store_path

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # 初始化 SQL 數據庫
        self.sql_db = None
        self.init_sql_database()

        # 初始化向量數據庫
        self.vector_store = None

    def init_sql_database(self):
        """初始化 SQL 資料庫連接"""
        try:
            self.sql_db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            print(f"✓ SQL 資料庫連接成功: {self.db_path}")
        except Exception as e:
            print(f"✗ SQL 資料庫連接失敗: {e}")

    def create_sample_database(self):
        """
        創建示例資料庫

        包含以下表格：
        - employees: 員工資料
        - products: 產品資料
        - sales: 銷售記錄
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 創建員工表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            position TEXT NOT NULL,
            salary REAL,
            hire_date TEXT
        )
        """)

        # 創建產品表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        )
        """)

        # 創建銷售表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            employee_id INTEGER,
            quantity INTEGER,
            total_amount REAL,
            sale_date TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
        """)

        # 插入示例數據
        employees_data = [
            (1, "張三", "業務部", "業務經理", 80000, "2020-01-15"),
            (2, "李四", "技術部", "軟體工程師", 90000, "2019-06-01"),
            (3, "王五", "業務部", "業務員", 50000, "2021-03-10"),
            (4, "趙六", "人資部", "人資專員", 55000, "2020-08-20"),
            (5, "錢七", "技術部", "資深工程師", 120000, "2018-02-14")
        ]

        products_data = [
            (1, "筆記型電腦", "電子產品", 35000, 50),
            (2, "滑鼠", "電子產品", 500, 200),
            (3, "鍵盤", "電子產品", 1500, 150),
            (4, "顯示器", "電子產品", 8000, 80),
            (5, "辦公椅", "家具", 3000, 30)
        ]

        sales_data = [
            (1, 1, 1, 2, 70000, "2024-01-15"),
            (2, 2, 3, 5, 2500, "2024-01-16"),
            (3, 3, 1, 3, 4500, "2024-01-20"),
            (4, 4, 3, 1, 8000, "2024-02-05"),
            (5, 1, 1, 1, 35000, "2024-02-10")
        ]

        cursor.executemany("INSERT OR REPLACE INTO employees VALUES (?,?,?,?,?,?)", employees_data)
        cursor.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?)", products_data)
        cursor.executemany("INSERT OR REPLACE INTO sales VALUES (?,?,?,?,?,?)", sales_data)

        conn.commit()
        conn.close()

        print("✓ 示例資料庫創建完成")
        print(f"  - employees: {len(employees_data)} 筆")
        print(f"  - products: {len(products_data)} 筆")
        print(f"  - sales: {len(sales_data)} 筆")

    def ingest_documents(self, documents: List[str]):
        """
        攝取文檔到向量數據庫

        Args:
            documents: 文檔列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = [Document(page_content=doc) for doc in documents]
        splits = text_splitter.split_documents(docs)

        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )

        print(f"✓ 向量數據庫創建完成: {len(splits)} 個文檔塊")

    def classify_query(self, query: str) -> Tuple[QueryType, str]:
        """
        分類查詢類型

        分析使用者查詢，決定應該：
        - 查詢 SQL 資料庫
        - 使用向量檢索
        - 或兩者結合

        Args:
            query: 使用者查詢

        Returns:
            (查詢類型, 推理過程)
        """
        # 獲取資料庫 schema
        schema_info = self.get_database_schema()

        classify_prompt = ChatPromptTemplate.from_template("""
你是一個智能查詢路由系統。你需要分析使用者的查詢，並決定應該使用哪種方式來回答。

可用的資料源：
1. SQL 資料庫：包含結構化數據
   {schema_info}

2. 向量數據庫：包含非結構化文檔（公司政策、產品說明、技術文檔等）

使用者查詢：{query}

請分析這個查詢並決定：
- "sql": 如果查詢需要精確的數據查詢（如統計、篩選、聚合）
- "vector": 如果查詢需要語義理解或文檔檢索（如政策說明、操作指南）
- "hybrid": 如果需要結合兩者（如「部門政策和員工數據」）
- "unknown": 如果無法確定

請以 JSON 格式回答：
{{
    "query_type": "sql/vector/hybrid/unknown",
    "reasoning": "你的推理過程",
    "confidence": 0.0-1.0
}}
""")

        messages = classify_prompt.format_messages(
            query=query,
            schema_info=schema_info
        )

        response = self.llm.invoke(messages)

        try:
            result = json.loads(response.content)
            query_type = QueryType(result["query_type"])
            reasoning = result["reasoning"]
            return query_type, reasoning
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"解析失敗: {e}")
            return QueryType.UNKNOWN, "無法分類查詢"

    def get_database_schema(self) -> str:
        """獲取資料庫 schema 資訊"""
        if not self.sql_db:
            return "無資料庫連接"

        try:
            tables = self.sql_db.get_usable_table_names()
            schema_info = []

            for table in tables:
                # 獲取表格 schema
                table_info = self.sql_db.get_table_info([table])
                schema_info.append(f"表格: {table}\n{table_info}")

            return "\n\n".join(schema_info)
        except Exception as e:
            return f"獲取 schema 失敗: {e}"

    def natural_language_to_sql(self, query: str) -> str:
        """
        將自然語言轉換為 SQL 查詢

        Args:
            query: 自然語言查詢

        Returns:
            SQL 查詢語句
        """
        schema_info = self.get_database_schema()

        nl2sql_prompt = ChatPromptTemplate.from_template("""
你是一個專業的 SQL 專家。根據使用者的自然語言查詢，生成對應的 SQL 語句。

資料庫 Schema：
{schema_info}

使用者查詢：{query}

要求：
1. 只生成 SELECT 查詢（不允許 INSERT, UPDATE, DELETE）
2. 使用正確的表格和欄位名稱
3. 適當使用 JOIN、WHERE、GROUP BY 等子句
4. 查詢應該高效且正確
5. 只輸出 SQL 語句，不要解釋

SQL 查詢：
""")

        messages = nl2sql_prompt.format_messages(
            query=query,
            schema_info=schema_info
        )

        response = self.llm.invoke(messages)
        sql_query = response.content.strip()

        # 移除可能的 markdown 標記
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        return sql_query

    def execute_sql_safely(self, sql_query: str) -> List[Dict]:
        """
        安全執行 SQL 查詢

        Args:
            sql_query: SQL 查詢語句

        Returns:
            查詢結果列表
        """
        # 安全檢查：只允許 SELECT 查詢
        sql_upper = sql_query.upper().strip()
        if not sql_upper.startswith("SELECT"):
            raise ValueError("只允許 SELECT 查詢")

        # 禁止的關鍵字
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        if any(keyword in sql_upper for keyword in forbidden_keywords):
            raise ValueError(f"查詢包含禁止的關鍵字")

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使結果可以按列名訪問
            cursor = conn.cursor()

            cursor.execute(sql_query)
            rows = cursor.fetchall()

            # 轉換為字典列表
            results = [dict(row) for row in rows]

            conn.close()
            return results

        except sqlite3.Error as e:
            raise Exception(f"SQL 執行錯誤: {e}")

    def vector_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        向量檢索

        Args:
            query: 查詢字串
            top_k: 返回的文檔數量

        Returns:
            檢索到的文檔列表
        """
        if not self.vector_store:
            raise ValueError("向量數據庫未初始化")

        return self.vector_store.similarity_search(query, k=top_k)

    def hybrid_query(
        self,
        query: str,
        top_k: int = 3,
        verbose: bool = True
    ) -> HybridQueryResult:
        """
        執行混合查詢

        自動分析查詢類型，並使用適當的方法回答

        Args:
            query: 使用者查詢
            top_k: 向量檢索返回的文檔數量
            verbose: 是否輸出詳細信息

        Returns:
            HybridQueryResult 對象
        """
        result = HybridQueryResult(query=query, query_type=QueryType.UNKNOWN)

        # 1. 分類查詢
        query_type, reasoning = self.classify_query(query)
        result.query_type = query_type
        result.reasoning = reasoning

        if verbose:
            print(f"查詢類型: {query_type.value}")
            print(f"推理: {reasoning}\n")

        # 2. 根據類型執行查詢
        try:
            if query_type == QueryType.SQL:
                # 純 SQL 查詢
                sql_query = self.natural_language_to_sql(query)
                result.sql_query = sql_query

                if verbose:
                    print(f"生成的 SQL:\n{sql_query}\n")

                sql_results = self.execute_sql_safely(sql_query)
                result.sql_results = sql_results

                if verbose:
                    print(f"SQL 結果: {len(sql_results)} 筆記錄\n")

                # 生成自然語言答案
                result.final_answer = self.generate_answer_from_sql(
                    query, sql_query, sql_results
                )

            elif query_type == QueryType.VECTOR:
                # 純向量檢索
                docs = self.vector_search(query, top_k=top_k)
                result.vector_docs = docs

                if verbose:
                    print(f"檢索到 {len(docs)} 個文檔\n")

                # 生成自然語言答案
                result.final_answer = self.generate_answer_from_docs(query, docs)

            elif query_type == QueryType.HYBRID:
                # 混合查詢
                # SQL 部分
                sql_query = self.natural_language_to_sql(query)
                result.sql_query = sql_query
                sql_results = self.execute_sql_safely(sql_query)
                result.sql_results = sql_results

                # 向量部分
                docs = self.vector_search(query, top_k=top_k)
                result.vector_docs = docs

                if verbose:
                    print(f"SQL 查詢:\n{sql_query}")
                    print(f"SQL 結果: {len(sql_results)} 筆")
                    print(f"向量文檔: {len(docs)} 個\n")

                # 整合答案
                result.final_answer = self.generate_hybrid_answer(
                    query, sql_results, docs
                )

            else:
                result.final_answer = "抱歉，我無法理解這個查詢。"

        except Exception as e:
            result.final_answer = f"查詢執行失敗: {str(e)}"
            if verbose:
                print(f"錯誤: {e}")

        return result

    def generate_answer_from_sql(
        self,
        query: str,
        sql_query: str,
        results: List[Dict]
    ) -> str:
        """從 SQL 結果生成自然語言答案"""
        answer_prompt = ChatPromptTemplate.from_template("""
使用者查詢：{query}

執行的 SQL 查詢：
{sql_query}

查詢結果：
{results}

請基於查詢結果，用自然、易懂的語言回答使用者的問題。
如果結果為空，請明確說明。

答案：
""")

        messages = answer_prompt.format_messages(
            query=query,
            sql_query=sql_query,
            results=json.dumps(results, ensure_ascii=False, indent=2)
        )

        response = self.llm.invoke(messages)
        return response.content.strip()

    def generate_answer_from_docs(self, query: str, docs: List[Document]) -> str:
        """從向量檢索文檔生成答案"""
        context = "\n\n".join([
            f"[文檔 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        answer_prompt = ChatPromptTemplate.from_template("""
基於以下文檔回答問題。

文檔：
{context}

問題：{query}

請提供準確、詳細的回答。如果文檔中沒有相關信息，請明確說明。

答案：
""")

        messages = answer_prompt.format_messages(query=query, context=context)
        response = self.llm.invoke(messages)
        return response.content.strip()

    def generate_hybrid_answer(
        self,
        query: str,
        sql_results: List[Dict],
        docs: List[Document]
    ) -> str:
        """整合 SQL 和向量檢索結果生成答案"""
        context = "\n\n".join([
            f"[文檔 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        answer_prompt = ChatPromptTemplate.from_template("""
使用者查詢：{query}

資料庫查詢結果：
{sql_results}

相關文檔：
{context}

請整合以上資訊，提供完整、準確的回答。

答案：
""")

        messages = answer_prompt.format_messages(
            query=query,
            sql_results=json.dumps(sql_results, ensure_ascii=False, indent=2),
            context=context
        )

        response = self.llm.invoke(messages)
        return response.content.strip()


def main():
    """示例程式"""
    print("=" * 80)
    print("SQL + RAG 整合系統示範")
    print("=" * 80)
    print()

    # 初始化系統
    db_path = "./company_db.sqlite"
    system = SQLRAGIntegration(db_path=db_path)

    # 創建示例資料庫
    print("創建示例資料庫...")
    system.create_sample_database()
    print()

    # 重新初始化 SQL 連接
    system.init_sql_database()

    # 準備向量數據庫的文檔
    documents = [
        """
        公司休假政策：
        - 年假：工作滿一年享有7天年假，之後每年增加1天，最多15天
        - 病假：每年12天有薪病假
        - 事假：每年可請5天事假（無薪）
        - 特休：婚假7天、喪假3-5天（視親等）
        申請流程：提前3天向主管提出申請，經核准後生效。
        """,
        """
        員工福利制度：
        - 三節獎金：端午、中秋、春節各發放5000元獎金
        - 績效獎金：根據年度績效評核，最高可達3個月薪資
        - 健康檢查：每年提供一次免費健康檢查
        - 員工訓練：每人每年至少30小時的訓練時數，公司全額補助
        - 團體保險：提供團體醫療保險和意外險
        """,
        """
        產品保固政策：
        - 電子產品：提供一年保固，可延長至三年（需額外付費）
        - 家具產品：提供六個月保固
        - 保固範圍：包含製造瑕疵和正常使用下的故障
        - 不保固項目：人為損壞、自然耗損、未經授權的維修
        退換貨政策：購買後7天內可無條件退換貨（需保持商品完整）
        """
    ]

    print("攝取文檔到向量數據庫...")
    system.ingest_documents(documents)
    print()

    # 測試不同類型的查詢
    test_queries = [
        "技術部有哪些員工？他們的薪資是多少？",  # SQL 查詢
        "公司的休假政策是什麼？",  # 向量檢索
        "業務部門的員工福利和薪資情況",  # 混合查詢
        "筆記型電腦的庫存數量和保固政策"  # 混合查詢
    ]

    for i, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"測試 {i}: {query}")
        print("=" * 80)

        result = system.hybrid_query(query, verbose=True)

        print(f"最終答案:\n{result.final_answer}\n")

    print("=" * 80)
    print("示範完成！")
    print("=" * 80)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤: 請設置 OPENAI_API_KEY 環境變數")
        print("提示: 複製 .env.example 到 .env 並填入你的 API 金鑰")
    else:
        main()
