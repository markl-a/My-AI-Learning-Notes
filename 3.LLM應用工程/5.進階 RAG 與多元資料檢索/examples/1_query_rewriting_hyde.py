"""
進階查詢技術：Query Rewriting 與 HyDE (Hypothetical Document Embeddings)

這個模組展示如何使用查詢改寫和 HyDE 技術來改善 RAG 檢索效果。

主要技術：
1. Query Rewriting - 改寫查詢以提升檢索準確度
2. Multi-Query Generation - 生成多個查詢變體以提升召回率
3. HyDE - 生成假設性文檔來改善語義匹配
4. Query Expansion - 使用 AI 擴展查詢內容
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class QueryResult:
    """查詢結果資料結構"""
    original_query: str
    rewritten_query: Optional[str] = None
    multi_queries: Optional[List[str]] = None
    hypothetical_doc: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    relevance_scores: Optional[List[float]] = None


class QueryRewriter:
    """
    查詢改寫器

    使用 LLM 來改寫使用者查詢，使其更適合向量檢索：
    - 補充缺失的上下文
    - 使用更精確的術語
    - 分解複雜查詢
    - 修正模糊的表達
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        """
        初始化查詢改寫器

        Args:
            model_name: 使用的 LLM 模型名稱
            temperature: 生成的隨機性（0 = 確定性，1 = 高隨機性）
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def rewrite_query(
        self,
        original_query: str,
        context: str = "",
        domain: str = "general"
    ) -> str:
        """
        改寫單一查詢

        Args:
            original_query: 原始查詢
            context: 對話上下文（可選）
            domain: 領域（用於術語優化）

        Returns:
            改寫後的查詢
        """
        rewrite_prompt = ChatPromptTemplate.from_template("""
你是一個專業的查詢優化助手。你的任務是改寫使用者的查詢，使其更適合向量檢索系統。

原始查詢：{query}
對話上下文：{context}
領域：{domain}

請根據以下原則改寫查詢：
1. 補充必要的背景信息和上下文
2. 使用該領域的專業術語
3. 如果查詢模糊，增加具體性
4. 保持查詢的核心意圖
5. 使用完整的句子或短語，而非關鍵詞堆砌

只輸出改寫後的查詢，不要解釋。

改寫後的查詢：""")

        messages = rewrite_prompt.format_messages(
            query=original_query,
            context=context or "無",
            domain=domain
        )

        response = self.llm.invoke(messages)
        return response.content.strip()

    def generate_multi_queries(
        self,
        query: str,
        n: int = 3,
        perspective: str = "diverse"
    ) -> List[str]:
        """
        生成多個查詢變體

        這個方法從不同角度生成查詢變體，以提升檢索的召回率。

        Args:
            query: 原始查詢
            n: 要生成的查詢數量
            perspective: 視角類型（"diverse", "specific", "related"）

        Returns:
            查詢變體列表
        """
        perspective_instructions = {
            "diverse": "從完全不同的角度和措辭來表達相同的資訊需求",
            "specific": "生成更具體、更細節的查詢變體",
            "related": "生成相關但探索不同子主題的查詢"
        }

        multi_query_prompt = ChatPromptTemplate.from_template("""
你是一個檢索優化專家。給定一個查詢，生成 {n} 個不同的查詢變體。

原始查詢：{query}

指示：{instruction}

要求：
1. 每個變體都應該能獨立檢索到相關文檔
2. 保持原始查詢的核心意圖
3. 使用不同的措辭和表達方式
4. 每行輸出一個查詢，格式為「1. 查詢內容」

生成 {n} 個查詢變體：""")

        messages = multi_query_prompt.format_messages(
            query=query,
            n=n,
            instruction=perspective_instructions.get(perspective, perspective_instructions["diverse"])
        )

        response = self.llm.invoke(messages)

        # 解析輸出
        queries = []
        for line in response.content.strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # 移除編號（如 "1. " 或 "1) "）
                query_text = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if query_text:
                    queries.append(query_text)

        return queries[:n]

    def expand_query(self, query: str, expansion_type: str = "semantic") -> str:
        """
        擴展查詢內容

        Args:
            query: 原始查詢
            expansion_type: 擴展類型（"semantic", "keywords", "context"）

        Returns:
            擴展後的查詢
        """
        expansion_prompts = {
            "semantic": "添加語義相關的概念和同義詞",
            "keywords": "添加相關的關鍵詞和術語",
            "context": "添加相關的背景信息和上下文"
        }

        expand_prompt = ChatPromptTemplate.from_template("""
原始查詢：{query}

請{expansion_type}來擴展這個查詢，使其包含更多相關信息，但不改變核心意圖。

擴展後的查詢：""")

        messages = expand_prompt.format_messages(
            query=query,
            expansion_type=expansion_prompts.get(expansion_type, expansion_type)
        )

        response = self.llm.invoke(messages)
        return response.content.strip()


class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) 檢索器

    核心思想：
    1. 使用 LLM 生成一個"假設性"的答案文檔
    2. 對這個假設文檔進行向量化
    3. 使用假設文檔的向量來檢索實際文檔

    優勢：
    - 改善查詢與文檔之間的語義匹配
    - 特別適合複雜或抽象的查詢
    - 可以找到措辭不同但語義相似的文檔
    """

    def __init__(
        self,
        vector_store: Chroma,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        初始化 HyDE 檢索器

        Args:
            vector_store: 向量存儲
            model_name: LLM 模型名稱
            temperature: 生成的創造性（較高值產生更多樣化的假設文檔）
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_hypothetical_document(
        self,
        query: str,
        doc_style: str = "informative"
    ) -> str:
        """
        生成假設性文檔

        Args:
            query: 使用者查詢
            doc_style: 文檔風格（"informative", "technical", "conversational"）

        Returns:
            假設性文檔內容
        """
        style_instructions = {
            "informative": "撰寫一篇資訊豐富的說明文章",
            "technical": "撰寫一篇技術性的詳細文檔",
            "conversational": "以對話的方式詳細回答"
        }

        hyde_prompt = ChatPromptTemplate.from_template("""
假設你是一位專家，需要針對以下問題{style}。

問題：{query}

要求：
1. 提供詳細、準確的資訊
2. 使用適當的專業術語
3. 結構清晰、邏輯連貫
4. 不要說「我不知道」，基於常識和專業知識生成合理的內容
5. 長度約 200-300 字

回答：""")

        messages = hyde_prompt.format_messages(
            query=query,
            style=style_instructions.get(doc_style, style_instructions["informative"])
        )

        response = self.llm.invoke(messages)
        return response.content.strip()

    def retrieve_with_hyde(
        self,
        query: str,
        top_k: int = 5,
        use_multi_hyde: bool = False
    ) -> List[Document]:
        """
        使用 HyDE 方法檢索文檔

        Args:
            query: 使用者查詢
            top_k: 返回的文檔數量
            use_multi_hyde: 是否使用多個假設文檔（提升穩健性）

        Returns:
            檢索到的文檔列表
        """
        if use_multi_hyde:
            # 生成多個假設文檔並合併結果
            all_docs = []
            seen_content = set()

            for style in ["informative", "technical"]:
                hypo_doc = self.generate_hypothetical_document(query, doc_style=style)
                docs = self.vector_store.similarity_search(hypo_doc, k=top_k)

                for doc in docs:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)

            return all_docs[:top_k]
        else:
            # 單一假設文檔
            hypothetical_doc = self.generate_hypothetical_document(query)
            return self.vector_store.similarity_search(hypothetical_doc, k=top_k)

    def retrieve_with_query_and_hyde(
        self,
        query: str,
        top_k: int = 5,
        weight_query: float = 0.5
    ) -> List[Document]:
        """
        結合原始查詢和 HyDE 的混合檢索

        Args:
            query: 使用者查詢
            top_k: 返回的文檔數量
            weight_query: 原始查詢的權重（0-1）

        Returns:
            檢索到的文檔列表
        """
        # 使用原始查詢檢索
        query_docs = self.vector_store.similarity_search(query, k=top_k * 2)

        # 使用 HyDE 檢索
        hyde_docs = self.retrieve_with_hyde(query, top_k=top_k * 2)

        # 合併去重
        all_docs = []
        seen_content = set()

        # 先加入查詢結果（根據權重）
        for doc in query_docs[:int(top_k * weight_query)]:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

        # 再加入 HyDE 結果
        for doc in hyde_docs:
            if doc.page_content not in seen_content and len(all_docs) < top_k:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

        return all_docs[:top_k]


class AdvancedQueryRAG:
    """
    整合進階查詢技術的完整 RAG 系統

    整合了：
    - Query Rewriting
    - Multi-Query Generation
    - HyDE
    - Query Expansion
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
        persist_directory: str = "./chroma_db"
    ):
        """初始化進階 RAG 系統"""
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.persist_directory = persist_directory
        self.vector_store = None
        self.query_rewriter = QueryRewriter(model_name=llm_model)
        self.hyde_retriever = None

    def ingest_documents(self, documents: List[str], chunk_size: int = 1000):
        """
        攝取文檔到向量數據庫

        Args:
            documents: 文檔列表
            chunk_size: 分塊大小
        """
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # 創建 Document 對象
        docs = [Document(page_content=doc) for doc in documents]
        splits = text_splitter.split_documents(docs)

        # 創建向量存儲
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # 初始化 HyDE 檢索器
        self.hyde_retriever = HyDERetriever(
            vector_store=self.vector_store,
            model_name=self.llm.model_name
        )

        print(f"✓ 已攝取 {len(documents)} 個文檔，分割為 {len(splits)} 個塊")

    def query(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 3,
        verbose: bool = True
    ) -> QueryResult:
        """
        執行查詢

        Args:
            query: 使用者查詢
            method: 檢索方法（"standard", "rewrite", "multi_query", "hyde", "hybrid"）
            top_k: 返回的文檔數量
            verbose: 是否輸出詳細信息

        Returns:
            QueryResult 對象
        """
        if self.vector_store is None:
            raise ValueError("請先使用 ingest_documents() 攝取文檔")

        result = QueryResult(original_query=query)

        if method == "standard":
            # 標準向量檢索
            docs = self.vector_store.similarity_search(query, k=top_k)
            result.retrieved_docs = docs

        elif method == "rewrite":
            # 查詢改寫
            rewritten = self.query_rewriter.rewrite_query(query)
            result.rewritten_query = rewritten
            docs = self.vector_store.similarity_search(rewritten, k=top_k)
            result.retrieved_docs = docs

            if verbose:
                print(f"原始查詢: {query}")
                print(f"改寫查詢: {rewritten}\n")

        elif method == "multi_query":
            # 多查詢檢索
            multi_queries = self.query_rewriter.generate_multi_queries(query, n=3)
            result.multi_queries = multi_queries

            all_docs = []
            seen_content = set()

            # 對每個查詢變體進行檢索
            for mq in [query] + multi_queries:
                docs = self.vector_store.similarity_search(mq, k=top_k)
                for doc in docs:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)

            result.retrieved_docs = all_docs[:top_k]

            if verbose:
                print(f"原始查詢: {query}")
                print("查詢變體:")
                for i, mq in enumerate(multi_queries, 1):
                    print(f"  {i}. {mq}")
                print()

        elif method == "hyde":
            # HyDE 檢索
            hypothetical_doc = self.hyde_retriever.generate_hypothetical_document(query)
            result.hypothetical_doc = hypothetical_doc
            docs = self.hyde_retriever.retrieve_with_hyde(query, top_k=top_k)
            result.retrieved_docs = docs

            if verbose:
                print(f"原始查詢: {query}")
                print(f"假設文檔:\n{hypothetical_doc[:200]}...\n")

        elif method == "hybrid":
            # 混合方法：結合查詢改寫、多查詢和 HyDE
            rewritten = self.query_rewriter.rewrite_query(query)
            multi_queries = self.query_rewriter.generate_multi_queries(query, n=2)

            result.rewritten_query = rewritten
            result.multi_queries = multi_queries

            all_docs = []
            seen_content = set()

            # 1. 改寫查詢檢索
            docs = self.vector_store.similarity_search(rewritten, k=top_k)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)

            # 2. 多查詢檢索
            for mq in multi_queries:
                docs = self.vector_store.similarity_search(mq, k=2)
                for doc in docs:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)

            # 3. HyDE 檢索
            docs = self.hyde_retriever.retrieve_with_hyde(query, top_k=2)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)

            result.retrieved_docs = all_docs[:top_k]

            if verbose:
                print(f"原始查詢: {query}")
                print(f"改寫查詢: {rewritten}")
                print(f"查詢變體: {', '.join(multi_queries)}\n")

        else:
            raise ValueError(f"未知的檢索方法: {method}")

        return result

    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        基於檢索到的文檔生成答案

        Args:
            query: 使用者查詢
            retrieved_docs: 檢索到的文檔

        Returns:
            生成的答案
        """
        context = "\n\n".join([
            f"[文檔 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        answer_prompt = ChatPromptTemplate.from_template("""
基於以下檢索到的文檔回答問題。

檢索到的文檔：
{context}

問題：{question}

要求：
1. 只根據提供的文檔回答
2. 如果文檔中沒有相關信息，明確說明「根據提供的文檔，無法回答這個問題」
3. 引用具體的文檔編號
4. 回答要準確、簡潔

答案：""")

        messages = answer_prompt.format_messages(
            context=context,
            question=query
        )

        response = self.llm.invoke(messages)
        return response.content.strip()

    def query_and_answer(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        執行完整的查詢和回答流程

        Returns:
            包含查詢結果和答案的字典
        """
        # 檢索
        result = self.query(query, method=method, top_k=top_k, verbose=verbose)

        # 生成答案
        answer = self.generate_answer(query, result.retrieved_docs)

        return {
            "query": query,
            "method": method,
            "answer": answer,
            "retrieved_docs": [doc.page_content for doc in result.retrieved_docs],
            "rewritten_query": result.rewritten_query,
            "multi_queries": result.multi_queries,
            "hypothetical_doc": result.hypothetical_doc
        }


def main():
    """示例程式"""
    print("=" * 80)
    print("進階查詢技術示範：Query Rewriting 與 HyDE")
    print("=" * 80)
    print()

    # 示例文檔
    sample_documents = [
        """
        Transformer 架構是一種深度學習模型架構，由 Vaswani 等人在 2017 年的論文
        「Attention is All You Need」中提出。它完全基於注意力機制，摒棄了循環神經網絡
        （RNN）和卷積神經網絡（CNN）。Transformer 由編碼器和解碼器組成，每個都包含
        多層的自注意力層和前饋神經網絡。這種架構在機器翻譯、文本生成等任務上取得了
        突破性的成果。
        """,
        """
        注意力機制（Attention Mechanism）是 Transformer 的核心組件。自注意力機制允許
        模型在處理每個詞時考慮輸入序列中的所有其他詞。計算注意力時，模型會為每個詞
        生成三個向量：查詢（Query）、鍵（Key）和值（Value）。通過計算查詢和鍵的相似度，
        模型可以決定應該關注哪些詞。多頭注意力機制則允許模型同時從多個不同的角度
        學習注意力模式。
        """,
        """
        BERT（Bidirectional Encoder Representations from Transformers）是 Google 在
        2018 年提出的預訓練語言模型。它使用 Transformer 的編碼器架構，通過在大量
        文本上進行雙向訓練來學習語言表示。BERT 的訓練包括兩個任務：掩碼語言模型
        （Masked Language Model）和下一句預測（Next Sentence Prediction）。BERT
        在多個 NLP 任務上都取得了顯著的性能提升，成為了預訓練模型的重要里程碑。
        """,
        """
        GPT（Generative Pre-trained Transformer）系列模型由 OpenAI 開發，是基於
        Transformer 解碼器的自回歸語言模型。與 BERT 不同，GPT 採用單向（從左到右）
        的訓練方式，專注於生成任務。GPT-3 擁有 1750 億個參數，展現了驚人的少樣本
        學習能力。後續的 GPT-3.5 和 GPT-4 進一步提升了性能，在各種 NLP 任務中
        表現出色。
        """,
        """
        檢索增強生成（Retrieval-Augmented Generation, RAG）是一種結合檢索和生成
        的技術，用於提升語言模型的回答質量。RAG 系統首先從知識庫中檢索相關文檔，
        然後將這些文檔作為上下文提供給語言模型來生成答案。這種方法可以有效減少
        模型的幻覺問題，提供更準確和最新的信息，特別適合需要專業知識的領域。
        """,
        """
        向量嵌入（Vector Embeddings）是將文本轉換為高維向量的技術，使得語義相似
        的文本在向量空間中距離更近。常用的嵌入模型包括 Word2Vec、GloVe 和現代的
        Transformer 基礎模型如 sentence-transformers。在 RAG 系統中，向量嵌入用於
        將文檔和查詢轉換為向量，然後通過計算向量相似度來檢索最相關的文檔。
        向量數據庫如 Chroma、Pinecone、Weaviate 等專門用於高效存儲和檢索這些向量。
        """
    ]

    # 初始化系統
    print("初始化 RAG 系統...")
    rag_system = AdvancedQueryRAG(
        llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        persist_directory="./chroma_db_demo"
    )

    # 攝取文檔
    print("攝取文檔...\n")
    rag_system.ingest_documents(sample_documents)
    print()

    # 測試不同的查詢方法
    test_query = "transformer 怎麼工作"

    methods = ["standard", "rewrite", "multi_query", "hyde", "hybrid"]

    for method in methods:
        print("=" * 80)
        print(f"方法: {method.upper()}")
        print("=" * 80)

        try:
            result = rag_system.query_and_answer(
                query=test_query,
                method=method,
                top_k=2,
                verbose=True
            )

            print(f"答案:\n{result['answer']}\n")
            print(f"檢索到 {len(result['retrieved_docs'])} 個文檔")
            print()
        except Exception as e:
            print(f"錯誤: {e}\n")

    print("=" * 80)
    print("示範完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 檢查環境變數
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤: 請設置 OPENAI_API_KEY 環境變數")
        print("提示: 複製 .env.example 到 .env 並填入你的 API 金鑰")
    else:
        main()
