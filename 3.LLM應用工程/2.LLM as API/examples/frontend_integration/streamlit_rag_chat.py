"""
Streamlit RAG 聊天機器人
支援文檔上傳、向量搜索和基於知識庫的問答
"""

import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
from typing import List, Dict
import hashlib

# 載入環境變數
load_dotenv()

# 嘗試導入 LangChain（如果可用）
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import (
        TextLoader,
        PDFMinerLoader,
        UnstructuredMarkdownLoader
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("⚠️ LangChain 未安裝。某些功能可能無法使用。")

# 頁面配置
st.set_page_config(
    page_title="RAG 智能問答系統",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .doc-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .source-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_openai_client():
    """初始化 OpenAI 客戶端"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("請設定 OPENAI_API_KEY 環境變數")
        return None
    return OpenAI(api_key=api_key)


@st.cache_resource
def init_embeddings():
    """初始化 Embeddings"""
    if not LANGCHAIN_AVAILABLE:
        return None
    return OpenAIEmbeddings()


def process_uploaded_file(uploaded_file) -> List[str]:
    """處理上傳的文件"""
    # 保存臨時文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 根據文件類型選擇 loader
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        elif file_extension == 'pdf':
            loader = PDFMinerLoader(tmp_file_path)
        elif file_extension == 'md':
            loader = UnstructuredMarkdownLoader(tmp_file_path)
        else:
            st.error(f"不支援的文件格式: {file_extension}")
            return []

        # 載入文檔
        documents = loader.load()

        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        splits = text_splitter.split_documents(documents)

        return splits

    except Exception as e:
        st.error(f"處理文件時發生錯誤: {e}")
        return []

    finally:
        # 清理臨時文件
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def create_vector_store(documents, embeddings):
    """創建向量存儲"""
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"創建向量存儲時發生錯誤: {e}")
        return None


def retrieve_relevant_docs(vectorstore, query: str, k: int = 3):
    """檢索相關文檔"""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"檢索文檔時發生錯誤: {e}")
        return []


def generate_rag_response(client, query: str, context_docs: List, model: str, temperature: float):
    """生成 RAG 回應"""
    # 構建上下文
    context = "\n\n".join([
        f"文檔片段 {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])

    # 構建提示
    prompt = f"""基於以下文檔內容回答問題。如果文檔中沒有相關資訊，請誠實地說明。

文檔內容：
{context}

問題：{query}

請提供詳細且準確的回答："""

    messages = [
        {"role": "system", "content": "你是一個專業的文檔問答助理。基於提供的文檔內容回答問題，如果文檔中沒有相關資訊，請明確指出。"},
        {"role": "user", "content": prompt}
    ]

    try:
        # 串流回應
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        full_response = ""
        message_placeholder = st.empty()

        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"生成回應時發生錯誤: {e}")
        return None


def main():
    """主應用程式"""

    # 標題
    st.markdown('<h1 class="main-header">📚 RAG 智能問答系統</h1>', unsafe_allow_html=True)
    st.markdown("上傳文檔，基於文檔內容進行智能問答")

    # 檢查依賴
    if not LANGCHAIN_AVAILABLE:
        st.error("""
        ❌ 缺少必要的依賴套件。請安裝：
        ```bash
        pip install langchain openai faiss-cpu pypdf pdfminer.six unstructured markdown
        ```
        """)
        return

    # 初始化客戶端
    client = init_openai_client()
    if not client:
        return

    embeddings = init_embeddings()
    if not embeddings:
        return

    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 設定")

        # 模型選擇
        model = st.selectbox(
            "選擇模型",
            ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo']
        )

        # 參數設定
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="較低的值會讓回答更準確"
        )

        k_docs = st.slider(
            "檢索文檔數量",
            min_value=1,
            max_value=5,
            value=3,
            help="每次查詢檢索的相關文檔數量"
        )

        st.divider()

        # 文檔上傳
        st.subheader("📄 上傳文檔")

        uploaded_files = st.file_uploader(
            "支援 TXT, PDF, MD 格式",
            type=['txt', 'pdf', 'md'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🔄 處理文檔", use_container_width=True):
                with st.spinner("正在處理文檔..."):
                    all_documents = []

                    for uploaded_file in uploaded_files:
                        st.info(f"處理: {uploaded_file.name}")
                        docs = process_uploaded_file(uploaded_file)
                        all_documents.extend(docs)

                    if all_documents:
                        st.success(f"✅ 成功處理 {len(all_documents)} 個文檔片段")

                        # 創建向量存儲
                        vectorstore = create_vector_store(all_documents, embeddings)

                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.documents = all_documents
                            st.session_state.doc_names = [f.name for f in uploaded_files]
                            st.success("✅ 向量存儲創建成功！")
                    else:
                        st.error("❌ 文檔處理失敗")

        st.divider()

        # 文檔統計
        if 'vectorstore' in st.session_state:
            st.subheader("📊 文檔統計")
            st.metric("已載入文檔", len(st.session_state.doc_names))
            st.metric("文檔片段", len(st.session_state.documents))

            with st.expander("查看文檔列表"):
                for i, doc_name in enumerate(st.session_state.doc_names, 1):
                    st.text(f"{i}. {doc_name}")

        # 清除按鈕
        if st.button("🗑️ 清除所有數據", use_container_width=True):
            for key in ['vectorstore', 'documents', 'doc_names', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # 主要內容區域
    if 'vectorstore' not in st.session_state:
        # 未上傳文檔時的提示
        st.info("""
        ### 🚀 開始使用

        1. 在左側側邊欄上傳一個或多個文檔
        2. 點擊「處理文檔」按鈕
        3. 開始提問！

        ### 📝 支援的文件格式

        - **TXT**: 純文本文件
        - **PDF**: PDF 文檔
        - **MD**: Markdown 文件

        ### 💡 使用技巧

        - 上傳相關領域的文檔以獲得更準確的答案
        - 問題越具體，答案越精確
        - 可以同時上傳多個文檔進行交叉查詢
        """)

        # 示例問題
        st.subheader("💡 示例場景")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **技術文檔問答**
            - 上傳 API 文檔
            - 詢問如何使用特定功能
            - 獲取程式碼範例
            """)

        with col2:
            st.markdown("""
            **學習資料整理**
            - 上傳課程筆記
            - 快速查找特定概念
            - 生成摘要和重點
            """)

    else:
        # 初始化對話歷史
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 顯示對話歷史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 顯示來源文檔
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 查看來源文檔"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>來源 {i}:</strong><br>
                                {source.page_content[:300]}...
                            </div>
                            """, unsafe_allow_html=True)

        # 使用者輸入
        if prompt := st.chat_input("詢問關於文檔的問題..."):
            # 添加使用者訊息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 檢索相關文檔
            with st.chat_message("assistant"):
                with st.spinner("🔍 搜索相關文檔..."):
                    relevant_docs = retrieve_relevant_docs(
                        st.session_state.vectorstore,
                        prompt,
                        k=k_docs
                    )

                if relevant_docs:
                    st.info(f"找到 {len(relevant_docs)} 個相關文檔片段")

                    # 生成回應
                    response = generate_rag_response(
                        client,
                        prompt,
                        relevant_docs,
                        model,
                        temperature
                    )

                    if response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": relevant_docs
                        })
                else:
                    error_msg = "未找到相關文檔。請嘗試不同的問題或上傳更多文檔。"
                    st.warning(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
