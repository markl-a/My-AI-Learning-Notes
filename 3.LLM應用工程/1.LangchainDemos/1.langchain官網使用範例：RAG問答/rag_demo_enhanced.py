"""
增強版 RAG 問答系統
功能：
1. 支援多種文件來源（網頁、PDF、文字檔）
2. 對話歷史記錄
3. 來源追蹤
4. 串流回應
5. 錯誤處理
"""

import os
import sys
from pathlib import Path

# 添加父目錄到路徑以導入 utils
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_environment,
    get_llm,
    get_embeddings,
    create_vector_store,
    format_docs,
    print_source_documents,
    setup_langsmith
)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class EnhancedRAG:
    """增強版 RAG 系統"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        初始化 RAG 系統

        Args:
            model_name: 使用的模型名稱
            temperature: 溫度參數
        """
        load_environment()
        setup_langsmith()

        self.llm = get_llm(model=model_name, temperature=temperature)
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.retriever = None
        self.chat_history = []

        print("✓ RAG 系統初始化完成")

    def load_from_web(self, urls: list, parse_only_classes: list = None):
        """
        從網頁載入文件

        Args:
            urls: 網頁 URL 列表
            parse_only_classes: 只解析特定 CSS 類別的內容
        """
        print(f"\n正在載入 {len(urls)} 個網頁...")

        bs_kwargs = {}
        if parse_only_classes:
            bs_kwargs["parse_only"] = bs4.SoupStrainer(
                class_=tuple(parse_only_classes)
            )

        loader = WebBaseLoader(web_paths=tuple(urls), bs_kwargs=bs_kwargs)
        documents = loader.load()

        print(f"✓ 已載入 {len(documents)} 個文件")
        return self._split_and_store(documents)

    def _split_and_store(self, documents, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        分割文件並儲存到向量資料庫

        Args:
            documents: 文件列表
            chunk_size: 區塊大小
            chunk_overlap: 區塊重疊

        Returns:
            分割後的文件數量
        """
        print("\n正在分割文件...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)
        print(f"✓ 已分割成 {len(splits)} 個區塊")

        print("\n正在建立向量資料庫...")
        self.vectorstore = create_vector_store(splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        print("✓ 向量資料庫建立完成")

        return len(splits)

    def simple_query(self, question: str, show_sources: bool = True) -> str:
        """
        簡單查詢（無對話歷史）

        Args:
            question: 問題
            show_sources: 是否顯示來源文件

        Returns:
            答案
        """
        if not self.retriever:
            raise ValueError("請先載入文件！")

        print(f"\n問題: {question}")
        print("正在查詢...")

        # 檢索相關文件
        docs = self.retriever.invoke(question)

        if show_sources:
            print_source_documents(docs)

        # 建立簡單的 RAG 鏈
        template = """你是一個專業的問答助手。請根據以下提供的上下文來回答問題。
如果你不確定答案，請誠實地說不知道，不要編造答案。
請用繁體中文回答，並保持答案簡潔明確。

上下文:
{context}

問題: {question}

答案:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        print(f"\n答案: {answer}")

        return answer

    def conversational_query(self, question: str) -> dict:
        """
        對話式查詢（包含歷史記錄）

        Args:
            question: 問題

        Returns:
            包含答案和來源的字典
        """
        if not self.retriever:
            raise ValueError("請先載入文件！")

        print(f"\n問題: {question}")
        print("正在查詢...")

        # 建立考慮歷史的檢索器
        contextualize_q_system_prompt = """根據對話歷史和最新的使用者問題，
        如果問題參考了對話歷史中的內容，請將其改寫為一個獨立的問題。
        不要回答問題，只需要在需要時改寫它，否則按原樣返回。"""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 建立問答鏈
        qa_system_prompt = """你是一個專業的問答助手。
        請使用以下檢索到的上下文來回答問題。
        如果你不確定答案，請誠實地說不知道，不要編造答案。
        請用繁體中文回答，最多使用三句話，並保持答案簡潔明確。

        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 執行查詢
        result = rag_chain.invoke({
            "input": question,
            "chat_history": self.chat_history
        })

        # 更新對話歷史
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=result["answer"])
        ])

        print(f"\n答案: {result['answer']}")
        print(f"\n使用了 {len(result.get('context', []))} 個來源文件")

        return result

    def stream_query(self, question: str):
        """
        串流查詢（即時顯示回應）

        Args:
            question: 問題
        """
        if not self.retriever:
            raise ValueError("請先載入文件！")

        print(f"\n問題: {question}")
        print("\n答案（串流）: ", end="", flush=True)

        template = """你是一個專業的問答助手。請根據以下提供的上下文來回答問題。
如果你不確定答案，請誠實地說不知道，不要編造答案。
請用繁體中文回答，並保持答案簡潔明確。

上下文:
{context}

問題: {question}

答案:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        full_response = ""
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n")
        return full_response

    def clear_history(self):
        """清除對話歷史"""
        self.chat_history = []
        print("✓ 對話歷史已清除")


def demo_simple_rag():
    """示範簡單的 RAG 查詢"""
    print("=" * 80)
    print("示範 1: 簡單 RAG 查詢")
    print("=" * 80)

    rag = EnhancedRAG(temperature=0)

    # 載入文件
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    rag.load_from_web(
        urls,
        parse_only_classes=["post-content", "post-title", "post-header"]
    )

    # 簡單查詢
    rag.simple_query("What is Task Decomposition? 請用繁體中文回答。")


def demo_conversational_rag():
    """示範對話式 RAG"""
    print("\n" + "=" * 80)
    print("示範 2: 對話式 RAG（包含歷史記錄）")
    print("=" * 80)

    rag = EnhancedRAG(temperature=0)

    # 載入文件
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    rag.load_from_web(
        urls,
        parse_only_classes=["post-content", "post-title", "post-header"]
    )

    # 第一個問題
    rag.conversational_query("什麼是 Task Decomposition？請用繁體中文回答。")

    # 後續問題（參考前面的對話）
    rag.conversational_query("有哪些常見的方法可以做到這件事？")

    # 另一個相關問題
    rag.conversational_query("這些方法的優缺點是什麼？")


def demo_streaming_rag():
    """示範串流 RAG"""
    print("\n" + "=" * 80)
    print("示範 3: 串流 RAG（即時回應）")
    print("=" * 80)

    rag = EnhancedRAG(temperature=0.7)

    # 載入文件
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    rag.load_from_web(
        urls,
        parse_only_classes=["post-content", "post-title", "post-header"]
    )

    # 串流查詢
    rag.stream_query("請總結一下這篇文章的主要內容，用繁體中文回答。")


if __name__ == "__main__":
    try:
        # 執行所有示範
        demo_simple_rag()
        demo_conversational_rag()
        demo_streaming_rag()

        print("\n" + "=" * 80)
        print("✓ 所有示範執行完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
