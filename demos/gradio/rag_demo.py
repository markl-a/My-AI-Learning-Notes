"""
RAG 系統 Gradio 演示

這個演示展示了一個完整的 RAG（檢索增強生成）系統，
包括文檔上傳、向量化、檢索和生成功能。

運行方式：
    python demos/gradio/rag_demo.py

訪問：http://localhost:7860
"""

import gradio as gr
import os
from typing import List, Tuple
from pathlib import Path

# 檢查是否安裝了必要的包
try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError as e:
    print(f"❌ 缺少依賴：{e}")
    print("請安裝: pip install langchain langchain-community langchain-openai chromadb pypdf")
    exit(1)

# ==================== 配置 ====================

VECTOR_DB_PATH = "./demos/gradio/chroma_db"
DEFAULT_MODEL = "gpt-4o-mini"  # 或使用 "llama3.2" (Ollama)


# ==================== RAG 系統類 ====================

class RAGSystem:
    """RAG 系統封裝類"""

    def __init__(self, use_openai: bool = True):
        """初始化 RAG 系統

        Args:
            use_openai: 是否使用 OpenAI（False 則使用 Ollama）
        """
        self.use_openai = use_openai
        self.vectorstore = None
        self.qa_chain = None

        # 初始化 Embeddings
        if use_openai and os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings()
        else:
            # 使用免費的 HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

        # 初始化 LLM
        if use_openai and os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.7)
        else:
            # 使用 Ollama 本地模型
            self.llm = Ollama(model="llama3.2", base_url="http://localhost:11434")

    def load_documents(self, file_paths: List[str]) -> Tuple[str, int]:
        """加載文檔並創建向量數據庫

        Args:
            file_paths: 文檔路徑列表

        Returns:
            (狀態消息, 文檔數量)
        """
        try:
            all_documents = []

            for file_path in file_paths:
                # 根據文件類型選擇加載器
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')

                documents = loader.load()
                all_documents.extend(documents)

            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            splits = text_splitter.split_documents(all_documents)

            # 創建向量數據庫
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=VECTOR_DB_PATH
            )

            # 創建 QA chain
            self._create_qa_chain()

            return f"✅ 成功加載 {len(all_documents)} 個文檔，分割為 {len(splits)} 個片段", len(splits)

        except Exception as e:
            return f"❌ 錯誤：{str(e)}", 0

    def _create_qa_chain(self):
        """創建 QA Chain"""
        # 自定義提示詞模板
        template = """使用以下上下文來回答問題。如果你不知道答案，就說你不知道，不要試圖編造答案。

上下文：
{context}

問題：{question}

請用中文詳細回答："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        """查詢 RAG 系統

        Args:
            question: 用戶問題

        Returns:
            (答案, 相關文檔列表)
        """
        if not self.qa_chain:
            return "❌ 請先上傳文檔", []

        try:
            result = self.qa_chain.invoke({"query": question})

            answer = result["result"]
            sources = [doc.page_content[:200] + "..." for doc in result["source_documents"]]

            return answer, sources

        except Exception as e:
            return f"❌ 錯誤：{str(e)}", []


# ==================== Gradio 界面 ====================

def create_demo():
    """創建 Gradio 演示界面"""

    # 初始化 RAG 系統
    rag_system = RAGSystem(use_openai=bool(os.getenv("OPENAI_API_KEY")))

    def upload_files(files):
        """處理文件上傳"""
        if not files:
            return "❌ 請上傳文件", ""

        file_paths = [file.name for file in files]
        status, count = rag_system.load_documents(file_paths)
        return status, f"已索引 {count} 個文檔片段"

    def ask_question(question, history):
        """處理問題查詢"""
        if not question.strip():
            return history, ""

        answer, sources = rag_system.query(question)

        # 添加到歷史
        history.append((question, answer))

        # 格式化來源
        sources_text = "\n\n**相關文檔片段：**\n" + "\n---\n".join(
            [f"{i+1}. {source}" for i, source in enumerate(sources)]
        )

        return history, sources_text

    # 創建 Gradio 界面
    with gr.Blocks(title="RAG 系統演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📚 RAG（檢索增強生成）系統演示

        這個演示展示了如何使用 LangChain 構建一個完整的 RAG 系統。

        ## 使用步驟：
        1. 上傳文檔（支援 .txt, .pdf）
        2. 等待文檔處理完成
        3. 在聊天框中提問
        4. 查看答案和相關文檔片段
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ 上傳文檔")
                file_upload = gr.File(
                    label="選擇文檔",
                    file_count="multiple",
                    file_types=[".txt", ".pdf"]
                )
                upload_btn = gr.Button("📤 上傳並處理", variant="primary")
                upload_status = gr.Textbox(label="狀態", interactive=False)
                doc_count = gr.Textbox(label="文檔統計", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 2️⃣ 提問")
                chatbot = gr.Chatbot(label="對話歷史", height=400)
                question_input = gr.Textbox(
                    label="您的問題",
                    placeholder="請輸入您的問題...",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 提問", variant="primary")
                    clear_btn = gr.Button("🗑️ 清除歷史")

                sources_output = gr.Markdown(label="相關文檔")

        # 設置事件處理
        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[upload_status, doc_count]
        )

        submit_btn.click(
            fn=ask_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, sources_output]
        ).then(
            lambda: "",  # 清空輸入框
            outputs=[question_input]
        )

        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, sources_output]
        )

        # 添加示例
        gr.Examples(
            examples=[
                ["這篇文檔的主要內容是什麼？"],
                ["請總結關鍵要點"],
                ["文檔中提到了哪些重要概念？"],
            ],
            inputs=question_input,
        )

        gr.Markdown("""
        ---
        ### 💡 提示
        - 支援中英文文檔
        - 可以上傳多個文檔
        - 問題可以是開放式的
        - 系統會自動檢索相關內容

        ### ⚙️ 技術棧
        - **框架**: LangChain
        - **嵌入模型**: OpenAI Embeddings / HuggingFace
        - **向量數據庫**: ChromaDB
        - **LLM**: OpenAI GPT / Ollama
        """)

    return demo


# ==================== 主函數 ====================

if __name__ == "__main__":
    # 檢查環境變量
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  未檢測到 OPENAI_API_KEY，將使用本地模型")
        print("💡 請確保 Ollama 正在運行：ollama serve")
        print("💡 並且已下載模型：ollama pull llama3.2")

    # 創建並啟動演示
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 設為 True 可生成公開鏈接
        show_error=True
    )
