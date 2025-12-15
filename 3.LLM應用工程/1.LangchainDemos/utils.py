"""
LangChain Demos 工具模組
提供常用的輔助函數和配置
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


def load_environment():
    """載入環境變數"""
    load_dotenv()

    # 檢查必要的 API 金鑰
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "未找到 OPENAI_API_KEY！\n"
            "請在 .env 檔案中設定，或參考 .env.example 範例檔案"
        )

    print("✓ 環境變數載入成功")


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.7, **kwargs):
    """
    取得 LLM 實例

    Args:
        model: 模型名稱，預設為 gpt-4o-mini
        temperature: 溫度參數，預設為 0.7
        **kwargs: 其他參數

    Returns:
        ChatOpenAI 實例
    """
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def get_embeddings(model: str = "text-embedding-3-small"):
    """
    取得 Embeddings 實例

    Args:
        model: 嵌入模型名稱

    Returns:
        OpenAIEmbeddings 實例
    """
    return OpenAIEmbeddings(model=model)


def create_vector_store(
    documents,
    embeddings=None,
    persist_directory: Optional[str] = None
):
    """
    建立向量資料庫

    Args:
        documents: 文件列表
        embeddings: 嵌入模型，若為 None 則使用預設
        persist_directory: 持久化目錄

    Returns:
        Chroma 向量資料庫實例
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if persist_directory:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"✓ 向量資料庫已建立並保存至 {persist_directory}")
    else:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )
        print("✓ 向量資料庫已建立（記憶體模式）")

    return vectorstore


def format_docs(docs):
    """
    格式化文件列表為字串

    Args:
        docs: 文件列表

    Returns:
        格式化後的字串
    """
    return "\n\n".join(doc.page_content for doc in docs)


def print_source_documents(docs, max_content_length: int = 200):
    """
    列印來源文件資訊

    Args:
        docs: 文件列表
        max_content_length: 最大顯示內容長度
    """
    print(f"\n找到 {len(docs)} 個相關文件：")
    print("=" * 80)

    for i, doc in enumerate(docs, 1):
        print(f"\n文件 {i}:")
        print(f"來源: {doc.metadata.get('source', 'Unknown')}")
        content = doc.page_content[:max_content_length]
        if len(doc.page_content) > max_content_length:
            content += "..."
        print(f"內容: {content}")
        print("-" * 80)


def setup_langsmith():
    """
    設定 LangSmith 追蹤（若有提供 API 金鑰）
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if not os.getenv("LANGCHAIN_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        if not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "langchain-demos"
        print("✓ LangSmith 追蹤已啟用")
    else:
        print("ℹ LangSmith 未設定（選用功能）")


if __name__ == "__main__":
    # 測試工具模組
    print("測試工具模組...")
    load_environment()
    setup_langsmith()

    print("\n測試 LLM...")
    llm = get_llm()
    response = llm.invoke("你好！請用一句話介紹自己。")
    print(f"LLM 回應: {response.content}")

    print("\n✓ 工具模組測試完成")
