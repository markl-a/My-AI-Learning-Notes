"""
進階 RAG 技術範例
包括：重排序(Reranking)、混合檢索(Hybrid Search)、查詢擴展等
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import Counter
import re


class Document:
    """文檔類"""

    def __init__(self, content: str, metadata: Dict = None, score: float = 0.0):
        self.content = content
        self.metadata = metadata or {}
        self.score = score

    def __repr__(self):
        return f"Document(score={self.score:.4f}, content='{self.content[:50]}...')"


class BM25Retriever:
    """
    BM25 檢索器
    基於詞頻的稀疏檢索方法
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 檢索器

        Args:
            k1: 詞頻飽和參數
            b: 長度正規化參數
        """
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.idf: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """分詞"""
        # 簡單的分詞實現
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def add_documents(self, documents: List[str]):
        """添加文檔"""
        self.documents = documents

        # 分詞
        self.doc_tokens = [self.tokenize(doc) for doc in documents]

        # 計算文檔長度
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        # 計算 IDF
        self._compute_idf()

    def _compute_idf(self):
        """計算 IDF 值"""
        N = len(self.documents)
        df = Counter()

        # 計算文檔頻率
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        # 計算 IDF
        for token, freq in df.items():
            self.idf[token] = np.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """計算查詢和文檔的 BM25 分數"""
        query_tokens = self.tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        score = 0.0
        token_freq = Counter(doc_tokens)

        for token in query_tokens:
            if token not in self.idf:
                continue

            tf = token_freq[token]
            idf = self.idf[token]

            # BM25 公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """搜索相關文檔"""
        scores = []

        for idx in range(len(self.documents)):
            score = self.score(query, idx)
            scores.append((idx, score))

        # 排序並返回 top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class HybridRetriever:
    """
    混合檢索器
    結合密集檢索（向量）和稀疏檢索（BM25）
    """

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ):
        """
        初始化混合檢索器

        Args:
            embedding_model: 嵌入模型名稱
            dense_weight: 密集檢索權重
            sparse_weight: 稀疏檢索權重
        """
        self.dense_model = SentenceTransformer(embedding_model)
        self.bm25 = BM25Retriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None

    def add_documents(self, documents: List[Document]):
        """添加文檔"""
        self.documents = documents

        # 提取文本
        texts = [doc.content for doc in documents]

        # 密集檢索：生成嵌入
        print("生成向量嵌入...")
        self.embeddings = self.dense_model.encode(texts, show_progress_bar=True)

        # 稀疏檢索：添加到 BM25
        print("構建 BM25 索引...")
        self.bm25.add_documents(texts)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """混合搜索"""
        # 密集檢索
        query_embedding = self.dense_model.encode([query])[0]
        dense_scores = self._cosine_similarity(query_embedding, self.embeddings)

        # 稀疏檢索
        sparse_results = self.bm25.search(query, k=len(self.documents))
        sparse_scores = np.zeros(len(self.documents))
        for idx, score in sparse_results:
            sparse_scores[idx] = score

        # 正規化分數
        dense_scores = self._normalize(dense_scores)
        sparse_scores = self._normalize(sparse_scores)

        # 合併分數
        final_scores = (
            self.dense_weight * dense_scores +
            self.sparse_weight * sparse_scores
        )

        # 獲取 top-k
        top_k_indices = np.argsort(final_scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            doc.score = float(final_scores[idx])
            results.append(doc)

        return results

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """計算餘弦相似度"""
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        dot_product = np.dot(vec2, vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """正規化分數到 [0, 1]"""
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score == 0:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)


class Reranker:
    """
    重排序器
    使用 Cross-Encoder 對檢索結果進行重排序
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """初始化重排序器"""
        try:
            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception as e:
            print(f"警告: Cross-Encoder 載入失敗: {e}")
            print("將使用簡單的相似度重排序")
            self.available = False
            self.fallback_model = SentenceTransformer('all-MiniLM-L6-v2')

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        重排序文檔

        Args:
            query: 查詢文本
            documents: 候選文檔列表
            top_k: 返回的文檔數量

        Returns:
            重排序後的文檔列表
        """
        if not documents:
            return []

        if self.available:
            # 使用 Cross-Encoder
            pairs = [(query, doc.content) for doc in documents]
            scores = self.model.predict(pairs)

            # 更新分數
            for doc, score in zip(documents, scores):
                doc.score = float(score)
        else:
            # 回退到簡單相似度
            query_emb = self.fallback_model.encode([query])[0]
            doc_embs = self.fallback_model.encode([doc.content for doc in documents])

            similarities = self._cosine_similarity(query_emb, doc_embs)

            for doc, score in zip(documents, similarities):
                doc.score = float(score)

        # 排序並返回 top-k
        documents.sort(key=lambda x: x.score, reverse=True)
        return documents[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """計算餘弦相似度"""
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        dot_product = np.dot(vec2, vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)

        return dot_product / (norm1 * norm2)


class QueryExpander:
    """
    查詢擴展器
    通過同義詞、相關詞擴展查詢
    """

    def __init__(self):
        """初始化查詢擴展器"""
        # 簡單的同義詞字典（實際應用中可使用 WordNet 等）
        self.synonyms = {
            "機器學習": ["ML", "機器學習算法", "機器學習技術"],
            "深度學習": ["DL", "深度神經網絡", "深度學習模型"],
            "神經網絡": ["神經網路", "NN", "人工神經網絡"],
            "自然語言處理": ["NLP", "語言處理", "文本處理"],
            "程式語言": ["編程語言", "程式設計語言", "programming language"]
        }

    def expand_query(self, query: str, max_expansions: int = 2) -> str:
        """
        擴展查詢

        Args:
            query: 原始查詢
            max_expansions: 最大擴展數量

        Returns:
            擴展後的查詢
        """
        expanded_terms = []

        for term, synonyms in self.synonyms.items():
            if term in query:
                # 添加同義詞
                expanded_terms.extend(synonyms[:max_expansions])

        if expanded_terms:
            return query + " " + " ".join(expanded_terms)

        return query


def demo_bm25_retrieval():
    """BM25 檢索演示"""
    print("=" * 60)
    print("BM25 稀疏檢索演示")
    print("=" * 60)

    # 準備文檔
    documents = [
        "Python is a high-level programming language used for data science",
        "Machine learning enables computers to learn from data",
        "Deep learning is a subset of machine learning",
        "Natural language processing helps computers understand human language",
        "TensorFlow is a popular deep learning framework",
        "PyTorch provides flexible deep learning tools",
        "Data preprocessing is a crucial step in machine learning pipelines"
    ]

    print(f"\n知識庫包含 {len(documents)} 個文檔")

    # 創建 BM25 檢索器
    print("\n構建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.add_documents(documents)

    # 測試查詢
    queries = [
        "deep learning framework",
        "machine learning data",
        "programming language"
    ]

    print("\n" + "=" * 60)
    for query in queries:
        print(f"\n查詢: {query}")
        print("-" * 60)

        results = bm25.search(query, k=3)

        print(f"\nTop 3 結果:")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"\n  {rank}. BM25 分數: {score:.4f}")
            print(f"     文檔: {documents[idx]}")


def demo_hybrid_retrieval():
    """混合檢索演示"""
    print("\n\n" + "=" * 60)
    print("混合檢索演示（向量 + BM25）")
    print("=" * 60)

    # 準備文檔
    docs = [
        Document(
            content="Python 是一種高級程式語言，廣泛應用於數據科學和機器學習",
            metadata={"category": "編程"}
        ),
        Document(
            content="機器學習使計算機能夠從數據中學習而無需明確編程",
            metadata={"category": "AI"}
        ),
        Document(
            content="深度學習使用多層神經網絡來學習數據的複雜表示",
            metadata={"category": "AI"}
        ),
        Document(
            content="TensorFlow 是一個流行的深度學習框架，由 Google 開發",
            metadata={"category": "框架"}
        ),
        Document(
            content="PyTorch 提供了靈活的深度學習工具和動態計算圖",
            metadata={"category": "框架"}
        ),
        Document(
            content="數據預處理包括清洗、轉換和特徵工程等步驟",
            metadata={"category": "數據"}
        )
    ]

    # 創建混合檢索器
    print("\n初始化混合檢索器...")
    retriever = HybridRetriever(dense_weight=0.6, sparse_weight=0.4)
    retriever.add_documents(docs)

    # 測試查詢
    queries = [
        "深度學習框架有哪些？",
        "如何處理數據？",
        "Python 機器學習"
    ]

    print("\n" + "=" * 60)
    for query in queries:
        print(f"\n查詢: {query}")
        print("-" * 60)

        results = retriever.search(query, k=3)

        print(f"\nTop 3 混合檢索結果:")
        for rank, doc in enumerate(results, 1):
            print(f"\n  {rank}. 混合分數: {doc.score:.4f}")
            print(f"     內容: {doc.content}")
            print(f"     元數據: {doc.metadata}")


def demo_reranking():
    """重排序演示"""
    print("\n\n" + "=" * 60)
    print("重排序演示")
    print("=" * 60)

    # 準備文檔（故意放一些不太相關的）
    docs = [
        Document("深度學習使用多層神經網絡進行特徵學習"),
        Document("Python 是一種流行的編程語言"),
        Document("TensorFlow 和 PyTorch 是主流的深度學習框架"),
        Document("數據預處理是機器學習的重要步驟"),
        Document("卷積神經網絡擅長處理圖像數據"),
        Document("循環神經網絡適合處理序列數據")
    ]

    query = "深度學習框架有哪些？"

    print(f"\n查詢: {query}")
    print("\n原始文檔順序:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.content}")

    # 創建重排序器
    print("\n\n執行重排序...")
    reranker = Reranker()
    reranked_docs = reranker.rerank(query, docs, top_k=3)

    print("\n重排序後的 Top 3:")
    print("-" * 60)
    for rank, doc in enumerate(reranked_docs, 1):
        print(f"\n  {rank}. 重排序分數: {doc.score:.4f}")
        print(f"     內容: {doc.content}")


def demo_query_expansion():
    """查詢擴展演示"""
    print("\n\n" + "=" * 60)
    print("查詢擴展演示")
    print("=" * 60)

    # 創建查詢擴展器
    expander = QueryExpander()

    # 測試查詢
    queries = [
        "什麼是機器學習？",
        "深度學習的應用",
        "自然語言處理技術",
        "推薦一個程式語言"
    ]

    print("\n查詢擴展結果:")
    print("-" * 60)

    for query in queries:
        expanded = expander.expand_query(query)
        print(f"\n原始查詢: {query}")
        print(f"擴展查詢: {expanded}")


def demo_advanced_rag_pipeline():
    """進階 RAG 管道演示"""
    print("\n\n" + "=" * 60)
    print("進階 RAG 管道演示（混合檢索 + 重排序）")
    print("=" * 60)

    # 準備文檔
    docs = [
        Document("Python 是數據科學和機器學習的首選語言", {"topic": "Python"}),
        Document("機器學習算法可以分為監督學習和非監督學習", {"topic": "ML"}),
        Document("深度學習在計算機視覺領域取得了突破性進展", {"topic": "DL"}),
        Document("TensorFlow 提供了完整的機器學習生態系統", {"topic": "框架"}),
        Document("PyTorch 的動態計算圖使模型開發更加靈活", {"topic": "框架"}),
        Document("Keras 是 TensorFlow 的高級 API", {"topic": "框架"}),
        Document("卷積神經網絡（CNN）專門用於處理圖像數據", {"topic": "DL"}),
        Document("循環神經網絡（RNN）適合處理時間序列數據", {"topic": "DL"}),
        Document("Transformer 架構革新了自然語言處理", {"topic": "NLP"}),
        Document("BERT 是基於 Transformer 的預訓練模型", {"topic": "NLP"})
    ]

    # 步驟 1: 查詢擴展
    query = "深度學習框架"
    print(f"\n原始查詢: {query}")

    expander = QueryExpander()
    expanded_query = expander.expand_query(query)
    print(f"擴展查詢: {expanded_query}")

    # 步驟 2: 混合檢索
    print("\n執行混合檢索...")
    retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
    retriever.add_documents(docs)
    candidates = retriever.search(expanded_query, k=5)

    print("\n混合檢索 Top 5:")
    for i, doc in enumerate(candidates, 1):
        print(f"  {i}. 分數: {doc.score:.4f} | {doc.content}")

    # 步驟 3: 重排序
    print("\n執行重排序...")
    reranker = Reranker()
    final_results = reranker.rerank(query, candidates, top_k=3)

    print("\n最終 Top 3 結果:")
    print("=" * 60)
    for rank, doc in enumerate(final_results, 1):
        print(f"\n  {rank}. 最終分數: {doc.score:.4f}")
        print(f"     內容: {doc.content}")
        print(f"     元數據: {doc.metadata}")


def main():
    """主函數"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "進階 RAG 技術範例" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    demo_bm25_retrieval()
    demo_hybrid_retrieval()
    demo_reranking()
    demo_query_expansion()
    demo_advanced_rag_pipeline()

    print("\n\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n進階技術總結:")
    print("1. BM25: 基於詞頻的稀疏檢索，適合關鍵字匹配")
    print("2. 混合檢索: 結合向量和 BM25，兼顧語義和關鍵字")
    print("3. 重排序: 使用 Cross-Encoder 提高排序精度")
    print("4. 查詢擴展: 添加同義詞增加召回率")
    print("5. 管道組合: 多種技術組合使用效果最佳")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
