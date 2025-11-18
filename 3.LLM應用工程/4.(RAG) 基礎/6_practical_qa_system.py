"""
實戰項目：多文檔問答系統
支持多種文件格式、對話記憶、來源引用等功能
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer


class Document:
    """文檔類"""

    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        source = self.metadata.get('source', 'unknown')
        return f"Document(source={source}, length={len(self.content)})"


class TextSplitter:
    """智能文本拆分器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """拆分文本"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
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
                metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'original_length': len(doc.content)
                })
                split_docs.append(Document(content=chunk, metadata=metadata))

        return split_docs


class DocumentLoader:
    """多格式文檔載入器"""

    @staticmethod
    def load_text(file_path: str) -> Document:
        """載入文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={'source': file_path, 'type': 'text', 'loaded_at': datetime.now().isoformat()}
        )

    @staticmethod
    def load_json(file_path: str) -> List[Document]:
        """載入 JSON 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content = item.get('content', str(item))
                    metadata = {k: v for k, v in item.items() if k != 'content'}
                    metadata.update({
                        'source': file_path,
                        'type': 'json',
                        'index': i
                    })
                    documents.append(Document(content=content, metadata=metadata))
                else:
                    documents.append(Document(
                        content=str(item),
                        metadata={'source': file_path, 'type': 'json', 'index': i}
                    ))
        elif isinstance(data, dict):
            content = data.get('content', json.dumps(data, ensure_ascii=False))
            metadata = {k: v for k, v in data.items() if k != 'content'}
            metadata.update({'source': file_path, 'type': 'json'})
            documents.append(Document(content=content, metadata=metadata))

        return documents

    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """載入 Markdown 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={'source': file_path, 'type': 'markdown'}
        )

    @staticmethod
    def load_directory(directory: str, extensions: List[str] = None) -> List[Document]:
        """載入目錄中的所有文件"""
        if extensions is None:
            extensions = ['.txt', '.md', '.json']

        documents = []
        directory_path = Path(directory)

        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    if file_path.suffix == '.json':
                        documents.extend(DocumentLoader.load_json(str(file_path)))
                    elif file_path.suffix == '.md':
                        documents.append(DocumentLoader.load_markdown(str(file_path)))
                    else:
                        documents.append(DocumentLoader.load_text(str(file_path)))
                    print(f"已載入: {file_path}")
                except Exception as e:
                    print(f"載入失敗 {file_path}: {e}")

        return documents


class VectorStore:
    """向量存儲"""

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]):
        """添加文檔"""
        if not documents:
            return

        texts = [doc.content for doc in documents]
        new_embeddings = self.model.encode(texts, show_progress_bar=True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_metadata: Dict = None
    ) -> List[Tuple[Document, float]]:
        """相似度搜索，支持元數據過濾"""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # 應用元數據過濾
        if filter_metadata:
            filtered_indices = []
            for i, doc in enumerate(self.documents):
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if match:
                    filtered_indices.append(i)

            if not filtered_indices:
                return []

            search_embeddings = self.embeddings[filtered_indices]
            search_documents = [self.documents[i] for i in filtered_indices]
        else:
            search_embeddings = self.embeddings
            search_documents = self.documents

        # 生成查詢嵌入
        query_embedding = self.model.encode([query])[0]

        # 計算相似度
        similarities = self._cosine_similarity(query_embedding, search_embeddings)

        # 獲取 top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        return [
            (search_documents[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """計算餘弦相似度"""
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        dot_product = np.dot(vec2, vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)

        return dot_product / (norm1 * norm2)

    def save(self, path: str):
        """保存向量庫"""
        import pickle
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"向量庫已保存到: {path}")

    def load(self, path: str):
        """載入向量庫"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        print(f"向量庫已載入: {len(self.documents)} 個文檔")


class ConversationMemory:
    """對話記憶"""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: List[Dict] = []

    def add_exchange(self, question: str, answer: str, sources: List[str] = None):
        """添加問答交換"""
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources or [],
            'timestamp': datetime.now().isoformat()
        })

        # 保持最大歷史記錄
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self) -> str:
        """獲取對話上下文"""
        if not self.history:
            return ""

        context_parts = []
        for exchange in self.history:
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"A: {exchange['answer']}")

        return "\n".join(context_parts)

    def clear(self):
        """清空記憶"""
        self.history = []


class MultiDocQASystem:
    """多文檔問答系統"""

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        enable_memory: bool = True
    ):
        """初始化問答系統"""
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(embedding_model)
        self.memory = ConversationMemory() if enable_memory else None
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_queries': 0
        }

    def load_documents(self, sources: List[str]):
        """
        載入文檔

        Args:
            sources: 文件路徑或目錄路徑列表
        """
        all_documents = []

        for source in sources:
            source_path = Path(source)

            if source_path.is_file():
                # 載入單個文件
                if source_path.suffix == '.json':
                    all_documents.extend(DocumentLoader.load_json(str(source_path)))
                elif source_path.suffix == '.md':
                    all_documents.append(DocumentLoader.load_markdown(str(source_path)))
                else:
                    all_documents.append(DocumentLoader.load_text(str(source_path)))
            elif source_path.is_dir():
                # 載入目錄
                all_documents.extend(DocumentLoader.load_directory(str(source_path)))

        self.stats['total_documents'] = len(all_documents)

        # 拆分文檔
        print(f"\n正在拆分 {len(all_documents)} 個文檔...")
        split_documents = self.text_splitter.split_documents(all_documents)
        self.stats['total_chunks'] = len(split_documents)

        # 添加到向量庫
        print(f"正在生成嵌入向量...")
        self.vector_store.add_documents(split_documents)

        print(f"\n載入完成:")
        print(f"  - 文檔數: {self.stats['total_documents']}")
        print(f"  - 文檔塊數: {self.stats['total_chunks']}")

    def query(
        self,
        question: str,
        k: int = 3,
        filter_metadata: Dict = None,
        use_memory: bool = True
    ) -> Dict:
        """
        查詢系統

        Args:
            question: 用戶問題
            k: 檢索的文檔數量
            filter_metadata: 元數據過濾條件
            use_memory: 是否使用對話記憶

        Returns:
            包含答案和來源的字典
        """
        self.stats['total_queries'] += 1

        # 檢索相關文檔
        results = self.vector_store.similarity_search(
            question,
            k=k,
            filter_metadata=filter_metadata
        )

        if not results:
            return {
                'question': question,
                'answer': '抱歉，我在知識庫中沒有找到相關信息。',
                'sources': [],
                'confidence': 0.0
            }

        # 構建上下文
        context_parts = []
        sources_info = []

        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[文檔 {i}]\n{doc.content}")
            sources_info.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'similarity_score': score
            })

        context = "\n\n".join(context_parts)

        # 添加對話記憶
        if use_memory and self.memory:
            memory_context = self.memory.get_context()
            if memory_context:
                context = f"對話歷史:\n{memory_context}\n\n當前檢索到的相關信息:\n{context}"

        # 構建提示詞
        prompt = self._build_prompt(question, context)

        # 生成答案（這裡使用模擬答案，實際應該調用 LLM API）
        answer = self._generate_answer(prompt, results)

        # 計算置信度
        confidence = sum(score for _, score in results) / len(results)

        # 保存到記憶
        if use_memory and self.memory:
            source_refs = [s['metadata'].get('source', 'unknown') for s in sources_info]
            self.memory.add_exchange(question, answer, source_refs)

        return {
            'question': question,
            'answer': answer,
            'sources': sources_info,
            'confidence': confidence,
            'num_sources': len(results)
        }

    def _build_prompt(self, question: str, context: str) -> str:
        """構建提示詞"""
        return f"""基於以下上下文回答問題。如果上下文中沒有足夠的信息，請誠實說明。

上下文:
{context}

問題: {question}

請提供準確、詳細的回答，並在適當的地方引用文檔編號。

回答:"""

    def _generate_answer(self, prompt: str, results: List[Tuple[Document, float]]) -> str:
        """
        生成答案（簡化版本）
        實際應用中應該調用真實的 LLM API
        """
        # 提取關鍵信息
        top_doc, top_score = results[0]

        answer = f"""基於檢索到的 {len(results)} 個相關文檔，我找到了以下信息：

{top_doc.content[:200]}...

（這是一個示例答案。在實際應用中，這裡會調用 OpenAI GPT-4 或 Anthropic Claude 等 LLM API 來生成更準確和自然的回答。）

相關文檔來源: {top_doc.metadata.get('source', 'unknown')}
相似度分數: {top_score:.4f}"""

        return answer

    def get_stats(self) -> Dict:
        """獲取統計信息"""
        return self.stats.copy()

    def clear_memory(self):
        """清空對話記憶"""
        if self.memory:
            self.memory.clear()
            print("對話記憶已清空")


def demo_qa_system():
    """問答系統演示"""
    print("=" * 60)
    print("多文檔問答系統演示")
    print("=" * 60)

    # 創建測試數據目錄
    test_dir = Path("3.LLM應用工程/4.(RAG) 基礎/test_data/knowledge_base")
    test_dir.mkdir(parents=True, exist_ok=True)

    # 創建測試文檔
    docs_data = {
        "ml_basics.txt": """機器學習基礎

機器學習是人工智慧的一個分支，它使計算機系統能夠從數據中學習和改進。主要分為三大類：

1. 監督學習：使用標記的訓練數據，學習輸入到輸出的映射。常見算法包括線性回歸、邏輯回歸、決策樹、隨機森林和支持向量機。

2. 非監督學習：處理未標記的數據，尋找數據中的隱藏模式。包括聚類（K-means、層次聚類）和降維（PCA、t-SNE）。

3. 強化學習：通過與環境互動學習最優策略，使用獎勵機制指導學習過程。""",

        "dl_frameworks.txt": """深度學習框架

主流的深度學習框架包括：

TensorFlow：
- 由 Google 開發
- 提供完整的機器學習生態系統
- 支持多種部署平台
- Keras 作為高級 API

PyTorch：
- 由 Facebook 開發
- 動態計算圖
- Pythonic API
- 研究人員的首選

其他框架：
- JAX：自動微分和XLA編譯
- MXNet：支持多語言
- Caffe：專注於計算機視覺""",

        "nlp_guide.txt": """自然語言處理指南

NLP 是使計算機理解和生成人類語言的技術。主要任務包括：

1. 文本分類：情感分析、主題分類
2. 命名實體識別：識別人名、地名、組織名
3. 機器翻譯：將文本從一種語言翻譯成另一種語言
4. 文本摘要：生成文本的簡短摘要
5. 問答系統：基於給定上下文回答問題

預訓練模型：
- BERT：雙向編碼表示
- GPT：生成式預訓練
- T5：統一的文本到文本框架"""
    }

    # 寫入測試文檔
    for filename, content in docs_data.items():
        with open(test_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)

    # 創建問答系統
    print("\n1. 初始化問答系統...")
    qa_system = MultiDocQASystem(
        chunk_size=300,
        chunk_overlap=50,
        enable_memory=True
    )

    # 載入文檔
    print("\n2. 載入知識庫...")
    qa_system.load_documents([str(test_dir)])

    # 測試查詢
    questions = [
        "什麼是監督學習？",
        "有哪些深度學習框架？",
        "NLP 有哪些主要任務？",
        "PyTorch 的特點是什麼？"
    ]

    print("\n3. 開始問答:")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n[問題 {i}] {question}")
        print("-" * 60)

        result = qa_system.query(question, k=2)

        print(f"\n回答:\n{result['answer']}")
        print(f"\n置信度: {result['confidence']:.4f}")
        print(f"使用了 {result['num_sources']} 個來源")

        print(f"\n來源詳情:")
        for j, source in enumerate(result['sources'], 1):
            print(f"\n  來源 {j}:")
            print(f"  - 文件: {source['metadata'].get('source', 'unknown')}")
            print(f"  - 相似度: {source['similarity_score']:.4f}")
            print(f"  - 內容: {source['content'][:100]}...")

        print("\n" + "=" * 60)

    # 顯示統計
    print("\n4. 系統統計:")
    stats = qa_system.get_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")


def main():
    """主函數"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "實戰：多文檔問答系統" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")

    demo_qa_system()

    print("\n\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n系統功能:")
    print("✓ 多格式文檔載入（TXT, JSON, MD）")
    print("✓ 智能文本拆分")
    print("✓ 向量化存儲和檢索")
    print("✓ 元數據過濾")
    print("✓ 對話記憶")
    print("✓ 來源引用和置信度評分")
    print("\n可擴展功能:")
    print("- 集成真實的 LLM API")
    print("- 添加更多文件格式支持（PDF, DOCX）")
    print("- 實現重排序提高準確度")
    print("- 添加用戶界面（Gradio/Streamlit）")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
