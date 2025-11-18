"""
完整的 RAG 問答系統
整合文檔處理、向量檢索和 LLM 生成
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import json


class Document:
    """文檔類"""

    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content='{self.content[:50]}...', metadata={self.metadata})"


class TextSplitter:
    """文本拆分器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """拆分文本為多個塊"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # 尋找合適的斷點
                for separator in ['\n\n', '\n', '。', '. ', ' ']:
                    pos = text.rfind(separator, start, end)
                    if pos != -1:
                        end = pos + len(separator)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """拆分文檔列表"""
        split_docs = []

        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk_id'] = i
                metadata['total_chunks'] = len(chunks)
                split_docs.append(Document(content=chunk, metadata=metadata))

        return split_docs


class VectorStore:
    """向量存儲"""

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]):
        """添加文檔"""
        # 提取文本內容
        texts = [doc.content for doc in documents]

        # 生成嵌入
        new_embeddings = self.model.encode(texts, show_progress_bar=True)

        # 更新存儲
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)

        print(f"已添加 {len(documents)} 個文檔塊")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """相似度搜索"""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # 生成查詢嵌入
        query_embedding = self.model.encode([query])[0]

        # 計算相似度
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # 獲取 top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        return [self.documents[idx] for idx in top_k_indices]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """計算餘弦相似度"""
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        dot_product = np.dot(vec2, vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)

        return dot_product / (norm1 * norm2)


class SimpleLLM:
    """
    簡單的 LLM 模擬器
    在沒有 API 的情況下提供基於模板的回答
    """

    def __init__(self):
        self.name = "SimpleLLM"

    def generate(self, prompt: str) -> str:
        """
        生成回答（簡化版本）
        實際應用中應該調用真實的 LLM API
        """
        # 這是一個簡化的實現，實際應該調用 OpenAI/Anthropic 等 API
        response = f"""基於提供的上下文，這是一個示例回答。

在實際應用中，這裡會調用真實的 LLM API（如 OpenAI GPT-4、Anthropic Claude 等）來生成更準確和詳細的回答。

提示詞長度: {len(prompt)} 字符

要使用真實的 LLM，請取消註釋相關的 API 調用代碼。"""

        return response


class OpenAILLM:
    """OpenAI LLM 包裝器"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        # 嘗試導入 openai
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.available = True
        except ImportError:
            print("警告: openai 包未安裝，請運行: pip install openai")
            self.available = False
        except Exception as e:
            print(f"警告: OpenAI 初始化失敗: {e}")
            self.available = False

    def generate(self, prompt: str) -> str:
        """生成回答"""
        if not self.available:
            return SimpleLLM().generate(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API 調用失敗: {e}")
            return SimpleLLM().generate(prompt)


class RAGSystem:
    """RAG 系統"""

    def __init__(
        self,
        llm: Optional[object] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """初始化 RAG 系統"""
        self.llm = llm or SimpleLLM()
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(embedding_model)

    def add_documents(self, documents: List[Document]):
        """添加文檔到知識庫"""
        # 拆分文檔
        print("正在拆分文檔...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"文檔已拆分為 {len(split_docs)} 個塊")

        # 添加到向量存儲
        print("正在生成嵌入向量並添加到向量庫...")
        self.vector_store.add_documents(split_docs)

    def add_text(self, text: str, metadata: Dict = None):
        """添加文本到知識庫"""
        doc = Document(content=text, metadata=metadata)
        self.add_documents([doc])

    def query(self, question: str, k: int = 3, return_sources: bool = True) -> Dict:
        """
        查詢系統

        Args:
            question: 用戶問題
            k: 檢索的文檔數量
            return_sources: 是否返回來源文檔

        Returns:
            包含答案和來源的字典
        """
        # 檢索相關文檔
        print(f"\n正在檢索相關文檔 (top-{k})...")
        relevant_docs = self.vector_store.similarity_search(question, k=k)

        if not relevant_docs:
            return {
                "answer": "抱歉，我在知識庫中沒有找到相關信息。",
                "sources": []
            }

        # 構建上下文
        context = "\n\n".join([
            f"文檔 {i+1}:\n{doc.content}"
            for i, doc in enumerate(relevant_docs)
        ])

        # 構建提示詞
        prompt = self._build_prompt(question, context)

        # 生成答案
        print("正在生成答案...")
        answer = self.llm.generate(prompt)

        result = {
            "answer": answer,
            "question": question
        }

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]

        return result

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        """構建提示詞"""
        prompt = f"""你是一個有幫助的 AI 助手。請基於以下上下文回答用戶的問題。

上下文:
{context}

問題: {question}

請基於上下文提供準確、詳細的回答。如果上下文中沒有足夠的信息來回答問題，請誠實地說明。

回答:"""

        return prompt


def demo_basic_rag():
    """基礎 RAG 演示"""
    print("=" * 60)
    print("基礎 RAG 系統演示")
    print("=" * 60)

    # 創建 RAG 系統
    print("\n1. 初始化 RAG 系統...")
    rag = RAGSystem()

    # 準備知識庫文檔
    documents = [
        Document(
            content="""機器學習是人工智慧的一個分支，它使計算機系統能夠從數據中學習和改進，
而無需被明確編程。機器學習算法通過訓練數據來學習模式和規律，然後將這些學到的
知識應用於新的、未見過的數據。機器學習主要分為三大類：監督學習、非監督學習和
強化學習。監督學習使用標記的訓練數據，非監督學習處理未標記的數據，而強化學習
通過與環境互動來學習最優策略。""",
            metadata={"source": "ML入門", "topic": "機器學習基礎"}
        ),
        Document(
            content="""深度學習是機器學習的一個子領域，它使用人工神經網絡來模擬人腦的學習
過程。深度學習模型由多層神經元組成，每層都能學習數據的不同抽象級別的特徵。
深度學習在圖像識別、語音識別、自然語言處理等領域取得了突破性進展。常見的深度
學習架構包括卷積神經網絡（CNN）用於圖像處理，循環神經網絡（RNN）和長短期記憶
網絡（LSTM）用於序列數據，以及 Transformer 用於自然語言處理。""",
            metadata={"source": "DL指南", "topic": "深度學習"}
        ),
        Document(
            content="""自然語言處理（NLP）是人工智慧的一個重要分支，專注於使計算機能夠
理解、解釋和生成人類語言。NLP 的應用包括機器翻譯、情感分析、文本摘要、問答系統
和聊天機器人。現代 NLP 系統大量使用深度學習技術，特別是 Transformer 架構。
BERT、GPT 等預訓練語言模型的出現，極大地推動了 NLP 領域的發展。這些模型首先
在大規模文本語料上進行預訓練，然後在特定任務上進行微調。""",
            metadata={"source": "NLP教程", "topic": "自然語言處理"}
        ),
        Document(
            content="""TensorFlow 是由 Google 開發的開源機器學習框架。它提供了全面的工具、
庫和社區資源，使研究人員能夠推動 ML 的最新技術，開發人員能夠輕鬆構建和部署
ML 應用。TensorFlow 支持多種編程語言，包括 Python、JavaScript 和 C++。它可以
在 CPU、GPU 和 TPU 上運行，並支持分布式訓練。TensorFlow 2.0 引入了 Keras 作為
其高級 API，使模型構建變得更加簡單和直觀。""",
            metadata={"source": "框架文檔", "topic": "TensorFlow"}
        ),
        Document(
            content="""PyTorch 是由 Facebook 開發的開源深度學習框架。它以其動態計算圖和
簡潔的 Pythonic API 而聞名，深受研究人員喜愛。PyTorch 提供了強大的 GPU 加速
功能，並支持自動微分，使得實現和訓練神經網絡變得簡單。PyTorch 還包括 torchvision
用於計算機視覺任務，torchaudio 用於音頻處理，以及 torchtext 用於自然語言處理。
PyTorch 的生態系統還包括 PyTorch Lightning，它簡化了訓練循環的編寫。""",
            metadata={"source": "框架文檔", "topic": "PyTorch"}
        )
    ]

    # 添加文檔
    print("\n2. 添加文檔到知識庫...")
    rag.add_documents(documents)

    # 測試查詢
    questions = [
        "什麼是深度學習？",
        "有哪些機器學習框架？",
        "NLP 有哪些應用？",
        "監督學習和非監督學習的區別是什麼？"
    ]

    print("\n3. 開始問答:")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n問題 {i}: {question}")
        print("-" * 60)

        result = rag.query(question, k=2)

        print(f"\n回答:\n{result['answer']}")

        print(f"\n參考來源:")
        for j, source in enumerate(result['sources'], 1):
            print(f"\n  來源 {j}:")
            print(f"  內容: {source['content'][:100]}...")
            print(f"  元數據: {source['metadata']}")

        print("\n" + "=" * 60)


def demo_custom_knowledge_base():
    """自定義知識庫演示"""
    print("\n\n" + "=" * 60)
    print("自定義知識庫 RAG 演示")
    print("=" * 60)

    # 創建 RAG 系統
    rag = RAGSystem(chunk_size=300, chunk_overlap=30)

    # 添加自定義知識
    knowledge_items = [
        ("Python 是一種高級編程語言，以其簡潔的語法和強大的功能而聞名。它廣泛應用於"
         "Web 開發、數據科學、機器學習、自動化等領域。Python 擁有豐富的第三方庫生態系統。",
         {"category": "編程語言"}),

        ("Git 是一個分布式版本控制系統，用於跟蹤代碼變更歷史。它允許多個開發者協作開發，"
         "並能輕鬆地合併代碼、解決衝突。常用命令包括 commit、push、pull、merge 等。",
         {"category": "開發工具"}),

        ("Docker 是一個開源的容器化平台，它允許開發者將應用及其依賴打包到容器中。容器提供了"
         "隔離的運行環境，確保應用在不同環境中的一致性。Docker 簡化了應用的部署和擴展。",
         {"category": "容器技術"}),

        ("API（應用程式接口）是不同軟件系統之間交互的接口。RESTful API 是一種設計風格，"
         "使用 HTTP 方法（GET、POST、PUT、DELETE）來操作資源。API 使得不同系統可以互相通信。",
         {"category": "Web開發"})
    ]

    print("\n正在添加知識項目...")
    for text, metadata in knowledge_items:
        rag.add_text(text, metadata)

    # 互動式問答
    print("\n" + "=" * 60)
    print("知識庫已準備就緒！")
    print("=" * 60)

    test_questions = [
        "什麼是 Docker？",
        "Python 有什麼特點？",
        "如何使用 Git？"
    ]

    for question in test_questions:
        print(f"\n問題: {question}")
        print("-" * 60)

        result = rag.query(question, k=2, return_sources=True)

        print(f"\n{result['answer']}")

        if result['sources']:
            print("\n引用的知識:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  [{i}] {source['content'][:80]}...")


def main():
    """主函數"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "完整 RAG 系統範例" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    demo_basic_rag()
    demo_custom_knowledge_base()

    print("\n\n" + "=" * 60)
    print("RAG 系統演示完成！")
    print("=" * 60)
    print("\n系統特點:")
    print("1. ✓ 文檔自動拆分和向量化")
    print("2. ✓ 語義檢索相關內容")
    print("3. ✓ 基於上下文生成回答")
    print("4. ✓ 保留來源引用")
    print("\n進階擴展:")
    print("- 集成 OpenAI/Anthropic API 獲得更好的回答質量")
    print("- 添加重排序（Reranking）提高檢索精度")
    print("- 實現混合檢索（向量 + 關鍵字）")
    print("- 添加對話記憶實現多輪對話")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
