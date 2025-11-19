"""
HyDE (Hypothetical Document Embeddings) 檢索
通過生成假設性文檔來改進檢索效果
特別適合處理複雜查詢和語義搜索
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import faiss


@dataclass
class Document:
    """文檔"""
    content: str
    metadata: Dict = None
    embedding: np.ndarray = None


class HyDERetriever:
    """HyDE 檢索器"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_generate_fn=None
    ):
        """
        初始化 HyDE 檢索器

        Args:
            embedding_model: 嵌入模型
            llm_generate_fn: LLM 生成函數（用於生成假設文檔）
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.llm_generate = llm_generate_fn
        self.documents: List[Document] = []
        self.index = None

        print(f"HyDE 檢索器初始化完成")
        print(f"嵌入模型: {embedding_model}")

    def add_documents(self, documents: List[str]):
        """
        添加文檔到檢索庫

        Args:
            documents: 文檔內容列表
        """
        print(f"正在添加 {len(documents)} 個文檔...")

        for doc_text in documents:
            # 生成嵌入
            embedding = self.encoder.encode(doc_text)

            # 創建文檔對象
            doc = Document(
                content=doc_text,
                embedding=embedding
            )
            self.documents.append(doc)

        # 構建 FAISS 索引
        self._build_index()

        print(f"文檔添加完成，總計 {len(self.documents)} 個文檔")

    def _build_index(self):
        """構建向量索引"""
        if not self.documents:
            return

        # 提取所有嵌入
        embeddings = np.array([doc.embedding for doc in self.documents])

        # 創建 FAISS 索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

    def generate_hypothetical_document(self, query: str) -> str:
        """
        生成假設性文檔

        Args:
            query: 用戶查詢

        Returns:
            假設性文檔
        """
        if self.llm_generate is None:
            # 如果沒有 LLM，使用簡單的模板
            return f"關於{query}的詳細說明：這是一個關於{query}的文檔，包含相關信息和解釋。"

        # 使用 LLM 生成假設性文檔
        prompt = f"""請為以下查詢生成一個假設性的答案文檔：

查詢: {query}

要求：
1. 假設這是一個包含答案的文檔
2. 內容應該全面且相關
3. 使用自然的文檔風格
4. 長度約100-200字

假設性文檔:"""

        hypothetical_doc = self.llm_generate(prompt)
        return hypothetical_doc

    def retrieve_standard(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        標準檢索（直接使用查詢）

        Args:
            query: 查詢
            top_k: 返回Top-K文檔

        Returns:
            [(文檔, 分數), ...]
        """
        # 編碼查詢
        query_embedding = self.encoder.encode(query)

        # 搜索
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )

        # 返回結果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.documents[idx], float(dist)))

        return results

    def retrieve_hyde(
        self,
        query: str,
        top_k: int = 5,
        num_hypothetical: int = 1
    ) -> List[Tuple[Document, float]]:
        """
        HyDE 檢索（使用假設性文檔）

        Args:
            query: 查詢
            top_k: 返回Top-K文檔
            num_hypothetical: 生成假設性文檔數量

        Returns:
            [(文檔, 分數), ...]
        """
        print(f"\n使用 HyDE 檢索: {query}")

        # 1. 生成假設性文檔
        hypothetical_docs = []
        for i in range(num_hypothetical):
            hypo_doc = self.generate_hypothetical_document(query)
            hypothetical_docs.append(hypo_doc)
            print(f"\n假設性文檔 {i+1}:")
            print(hypo_doc[:200] + "...")

        # 2. 編碼假設性文檔
        hypo_embeddings = self.encoder.encode(hypothetical_docs)

        # 如果有多個假設性文檔，取平均嵌入
        if num_hypothetical > 1:
            query_embedding = np.mean(hypo_embeddings, axis=0)
        else:
            query_embedding = hypo_embeddings[0]

        # 3. 搜索
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )

        # 4. 返回結果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.documents[idx], float(dist)))

        return results

    def compare_methods(
        self,
        query: str,
        top_k: int = 3
    ):
        """
        比較標準檢索和 HyDE 檢索

        Args:
            query: 查詢
            top_k: Top-K
        """
        print(f"\n{'='*60}")
        print(f"查詢: {query}")
        print(f"{'='*60}")

        # 標準檢索
        print("\n【標準檢索結果】")
        standard_results = self.retrieve_standard(query, top_k)
        for i, (doc, score) in enumerate(standard_results, 1):
            print(f"\n{i}. 分數: {score:.4f}")
            print(f"   內容: {doc.content[:150]}...")

        # HyDE 檢索
        print("\n【HyDE 檢索結果】")
        hyde_results = self.retrieve_hyde(query, top_k)
        for i, (doc, score) in enumerate(hyde_results, 1):
            print(f"\n{i}. 分數: {score:.4f}")
            print(f"   內容: {doc.content[:150]}...")


def simple_llm_generate(prompt: str) -> str:
    """簡單的 LLM 生成函數（模擬）"""
    # 在實際使用中，這裡應該調用真實的 LLM
    # 例如：OpenAI API, Anthropic API, 或本地模型

    # 這裡使用簡單的規則生成
    if "什麼是" in prompt or "介紹" in prompt:
        return """這是一個詳細的技術文檔，介紹了相關概念的定義、
原理、應用場景和最佳實踐。文檔包含了多個實例和代碼示例，
幫助讀者理解核心思想。此外，還討論了常見問題和解決方案。"""
    elif "如何" in prompt or "怎麼" in prompt:
        return """這是一份實用指南，提供了詳細的步驟說明。
首先介紹了基礎準備工作，然後逐步講解具體操作流程，
包括必要的配置、常用命令和注意事項。最後提供了
完整的示例和故障排除建議。"""
    else:
        return """這是一個全面的參考文檔，涵蓋了主題的各個方面。
文檔結構清晰，包含背景介紹、技術細節、實踐經驗和
未來展望。適合不同層次的讀者參考學習。"""


def example_basic_hyde():
    """示例 1: 基本 HyDE 檢索"""
    print("=== 示例 1: 基本 HyDE 檢索 ===\n")

    # 創建檢索器
    retriever = HyDERetriever(llm_generate_fn=simple_llm_generate)

    # 添加文檔
    documents = [
        "Transformer 是一種基於自注意力機制的神經網絡架構，由 Google 在 2017 年提出。它完全摒棄了循環神經網絡，使用自注意力機制來處理序列數據。",
        "BERT（Bidirectional Encoder Representations from Transformers）是一個預訓練語言模型，使用雙向 Transformer 編碼器。它在多個 NLP 任務上取得了突破性成果。",
        "GPT（Generative Pre-trained Transformer）是一系列基於 Transformer 的語言模型，採用單向（從左到右）的語言建模方式，擅長文本生成任務。",
        "注意力機制允許模型在處理序列時動態地關注不同位置的信息。自注意力是一種特殊的注意力機制，計算序列內部元素之間的關聯。",
        "預訓練-微調範式是現代 NLP 的主流方法。首先在大規模無標註數據上進行預訓練，然後在特定任務上進行微調。",
        "詞嵌入（Word Embedding）是將詞語映射到連續向量空間的技術。Word2Vec 和 GloVe 是早期的代表性方法，而現代方法使用上下文相關的嵌入。"
    ]

    retriever.add_documents(documents)

    # 測試查詢
    query = "什麼模型適合生成任務？"

    retriever.compare_methods(query, top_k=3)


def example_complex_query():
    """示例 2: 複雜查詢"""
    print("\n\n=== 示例 2: 複雜查詢的HyDE優勢 ===\n")

    retriever = HyDERetriever(llm_generate_fn=simple_llm_generate)

    # 技術文檔
    documents = [
        "FastAPI 是一個現代、快速的 Python Web 框架，用於構建 API。它基於標準的 Python 類型提示，提供自動的請求驗證和文檔生成。",
        "Docker 是一個容器化平台，允許開發者打包應用及其依賴項到可移植的容器中。容器提供了一致的運行環境，簡化了部署流程。",
        "Kubernetes（K8s）是一個開源的容器編排系統，用於自動化容器的部署、擴展和管理。它提供了服務發現、負載均衡等功能。",
        "RESTful API 是一種基於 HTTP 協議的 API 設計風格，使用標準的 HTTP 方法（GET、POST、PUT、DELETE）來操作資源。",
        "GraphQL 是一種 API 查詢語言，允許客戶端精確指定需要的數據。相比 REST API，它可以減少過度獲取或獲取不足的問題。"
    ]

    retriever.add_documents(documents)

    # 複雜查詢（不直接包含文檔中的關鍵詞）
    query = "我想要構建一個能自動擴展的微服務，應該用什麼技術？"

    retriever.compare_methods(query, top_k=3)


def example_multiple_hypothetical():
    """示例 3: 多個假設性文檔"""
    print("\n\n=== 示例 3: 使用多個假設性文檔 ===\n")

    retriever = HyDERetriever(llm_generate_fn=simple_llm_generate)

    # 添加文檔
    documents = [
        "機器學習是人工智能的一個分支，通過算法從數據中學習模式。主要分為監督學習、無監督學習和強化學習三大類。",
        "深度學習是機器學習的子領域，使用多層神經網絡學習數據的層次化表示。CNN、RNN、Transformer 是常見的深度學習架構。",
        "數據預處理是機器學習流程中的重要步驟，包括數據清洗、標準化、特徵工程等。高質量的數據是模型性能的基礎。",
        "模型評估使用各種指標來衡量性能，如準確率、召回率、F1分數等。交叉驗證是常用的評估方法，可以避免過擬合。",
        "過擬合是模型在訓練數據上表現好但在新數據上表現差的現象。正則化、Dropout、數據增強是常用的防止過擬合的技術。"
    ]

    retriever.add_documents(documents)

    query = "如何提高模型的泛化能力？"

    # 生成多個假設性文檔
    results = retriever.retrieve_hyde(query, top_k=3, num_hypothetical=3)

    print("\n【最終檢索結果】")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. 分數: {score:.4f}")
        print(f"   內容: {doc.content}")


def example_with_real_llm():
    """示例 4: 使用真實 LLM（需要 API key）"""
    print("\n\n=== 示例 4: 使用真實 LLM ===\n")

    try:
        from openai import OpenAI
        import os

        if not os.getenv("OPENAI_API_KEY"):
            print("跳過此示例：需要 OPENAI_API_KEY")
            return

        client = OpenAI()

        def llm_generate(prompt: str) -> str:
            """使用 GPT 生成假設性文檔"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content

        retriever = HyDERetriever(llm_generate_fn=llm_generate)

        documents = [
            "Python 是一種高級編程語言，以其簡潔的語法和強大的庫生態系統而聞名。",
            "JavaScript 是網頁開發的核心語言，可以實現動態的用戶界面交互。",
            "Rust 是一種系統編程語言，強調安全性和性能，無需垃圾回收器。"
        ]

        retriever.add_documents(documents)

        query = "我需要一個既安全又高效的語言來開發系統軟件"

        retriever.compare_methods(query, top_k=2)

    except ImportError:
        print("需要安裝 OpenAI: pip install openai")


if __name__ == "__main__":
    print("HyDE (Hypothetical Document Embeddings) 檢索示例")
    print("=" * 60)
    print()

    # 運行示例
    example_basic_hyde()
    example_complex_query()
    example_multiple_hypothetical()
    # example_with_real_llm()  # 需要 API key

    print("\n\nHyDE 原理:")
    print("1. 用戶提出查詢")
    print("2. LLM 生成假設性答案文檔")
    print("3. 使用假設性文檔的嵌入進行檢索")
    print("4. 返回最相關的真實文檔")

    print("\nHyDE 優勢:")
    print("✓ 改善語義匹配（用文檔檢索文檔）")
    print("✓ 處理複雜查詢更有效")
    print("✓ 減少查詢-文檔的語義鴻溝")
    print("✓ 不需要重新訓練檢索模型")

    print("\n適用場景:")
    print("1. 問答系統（查詢通常簡短，文檔詳細）")
    print("2. 技術文檔搜索")
    print("3. 學術論文檢索")
    print("4. 複雜的信息檢索任務")

    print("\n實現建議:")
    print("1. 選擇合適的 LLM 生成假設性文檔")
    print("2. 調整生成的詳細程度（通過 prompt）")
    print("3. 可以生成多個假設性文檔並取平均")
    print("4. 結合標準檢索和 HyDE（混合策略）")
