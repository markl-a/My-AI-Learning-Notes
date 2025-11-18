"""
向量資料庫範例
展示如何使用不同的向量資料庫進行存儲和檢索
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
import pickle
from pathlib import Path


class SimpleVectorDB:
    """
    簡單的向量資料庫實現
    使用 numpy 進行相似度計算
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """初始化向量資料庫"""
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None
        self.metadata: List[Dict] = []

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        添加文檔到資料庫

        Args:
            documents: 文檔列表
            metadatas: 元數據列表
        """
        # 生成嵌入
        new_embeddings = self.model.encode(documents)

        # 更新存儲
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)

        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in documents])

        print(f"已添加 {len(documents)} 個文檔，當前總數: {len(self.documents)}")

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似文檔

        Args:
            query: 查詢文本
            k: 返回的結果數量

        Returns:
            (文檔, 相似度分數, 元數據) 的列表
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # 生成查詢嵌入
        query_embedding = self.model.encode([query])[0]

        # 計算餘弦相似度
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # 獲取 top-k 結果
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            results.append((
                self.documents[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))

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

    def save(self, path: str):
        """保存資料庫到磁盤"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"資料庫已保存到: {path}")

    def load(self, path: str):
        """從磁盤載入資料庫"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        print(f"資料庫已從 {path} 載入，包含 {len(self.documents)} 個文檔")


class FAISSVectorDB:
    """
    使用 FAISS 的向量資料庫
    FAISS 是 Facebook 開發的高效相似度搜索庫
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """初始化向量資料庫"""
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.index = None
        self.dimension = None

        # 嘗試導入 faiss
        try:
            import faiss
            self.faiss = faiss
            self.faiss_available = True
        except ImportError:
            print("警告: FAISS 未安裝，將使用簡單實現")
            self.faiss_available = False
            self.simple_embeddings = None

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文檔到資料庫"""
        # 生成嵌入
        embeddings = self.model.encode(documents).astype('float32')

        # 初始化索引
        if self.index is None:
            self.dimension = embeddings.shape[1]
            if self.faiss_available:
                # 使用 FAISS 的 IndexFlatL2（精確搜索）
                self.index = self.faiss.IndexFlatL2(self.dimension)
            else:
                # 使用簡單的 numpy 數組
                self.simple_embeddings = embeddings
        else:
            if not self.faiss_available:
                self.simple_embeddings = np.vstack([self.simple_embeddings, embeddings])

        # 添加到索引
        if self.faiss_available:
            self.index.add(embeddings)

        # 更新文檔和元數據
        self.documents.extend(documents)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in documents])

        print(f"已添加 {len(documents)} 個文檔，當前總數: {len(self.documents)}")

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, Dict]]:
        """搜索相似文檔"""
        if len(self.documents) == 0:
            return []

        # 生成查詢嵌入
        query_embedding = self.model.encode([query]).astype('float32')

        if self.faiss_available:
            # 使用 FAISS 搜索
            distances, indices = self.index.search(query_embedding, k)
            distances = distances[0]
            indices = indices[0]
        else:
            # 使用簡單實現
            similarities = self._cosine_similarity(
                query_embedding[0],
                self.simple_embeddings
            )
            indices = np.argsort(similarities)[::-1][:k]
            # 轉換為距離（1 - 相似度）
            distances = 1 - similarities[indices]

        results = []
        for idx, distance in zip(indices, distances):
            if idx < len(self.documents):  # 確保索引有效
                results.append((
                    self.documents[idx],
                    float(distance),
                    self.metadata[idx]
                ))

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


def demo_simple_vector_db():
    """簡單向量資料庫演示"""
    print("=" * 60)
    print("簡單向量資料庫演示")
    print("=" * 60)

    # 創建資料庫
    print("\n1. 創建向量資料庫...")
    db = SimpleVectorDB()

    # 準備文檔
    documents = [
        "Python 是一種高級程式語言，廣泛用於數據科學",
        "機器學習使計算機能夠從數據中學習",
        "深度學習是機器學習的一個子集",
        "自然語言處理幫助計算機理解人類語言",
        "TensorFlow 是一個流行的深度學習框架",
        "PyTorch 提供了靈活的深度學習工具",
        "數據預處理是機器學習管道中的關鍵步驟",
        "神經網絡由多層互連的節點組成"
    ]

    metadatas = [
        {"category": "編程語言", "difficulty": "初級"},
        {"category": "機器學習", "difficulty": "中級"},
        {"category": "深度學習", "difficulty": "中級"},
        {"category": "NLP", "difficulty": "中級"},
        {"category": "框架", "difficulty": "中級"},
        {"category": "框架", "difficulty": "中級"},
        {"category": "數據處理", "difficulty": "初級"},
        {"category": "深度學習", "difficulty": "高級"}
    ]

    # 添加文檔
    print("\n2. 添加文檔到資料庫...")
    db.add_documents(documents, metadatas)

    # 執行搜索
    queries = [
        "什麼是深度學習？",
        "推薦一個機器學習框架",
        "如何處理自然語言？"
    ]

    print("\n3. 執行搜索:")
    print("-" * 60)

    for query in queries:
        print(f"\n查詢: {query}")
        results = db.search(query, k=3)

        print(f"\nTop 3 結果:")
        for i, (doc, score, metadata) in enumerate(results, 1):
            print(f"\n  {i}. 相似度分數: {score:.4f}")
            print(f"     文檔: {doc}")
            print(f"     元數據: {metadata}")

    # 保存和載入
    print("\n4. 保存資料庫...")
    save_path = "3.LLM應用工程/4.(RAG) 基礎/test_data/simple_db.pkl"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    db.save(save_path)

    print("\n5. 載入資料庫...")
    db2 = SimpleVectorDB()
    db2.load(save_path)

    # 驗證載入的資料庫
    print("\n6. 驗證載入的資料庫...")
    test_query = "Python 機器學習"
    results = db2.search(test_query, k=2)
    print(f"\n測試查詢: {test_query}")
    for i, (doc, score, metadata) in enumerate(results, 1):
        print(f"\n  {i}. {doc}")
        print(f"     分數: {score:.4f}")


def demo_faiss_vector_db():
    """FAISS 向量資料庫演示"""
    print("\n\n" + "=" * 60)
    print("FAISS 向量資料庫演示")
    print("=" * 60)

    # 創建資料庫
    print("\n1. 創建 FAISS 向量資料庫...")
    db = FAISSVectorDB()

    # 準備更大的知識庫
    documents = [
        "Python 支持多種編程範式，包括面向對象和函數式編程",
        "Java 是一種靜態類型的面向對象編程語言",
        "JavaScript 是 Web 開發中最流行的語言",
        "機器學習算法可以分為監督學習、非監督學習和強化學習",
        "深度學習使用多層神經網絡進行特徵學習",
        "卷積神經網絡（CNN）擅長處理圖像數據",
        "循環神經網絡（RNN）適合處理序列數據",
        "Transformer 架構革新了自然語言處理領域",
        "BERT 是一個預訓練的語言表示模型",
        "GPT 系列模型在文本生成任務中表現出色",
        "數據清洗是數據科學項目的重要步驟",
        "特徵工程可以顯著提升模型性能",
        "交叉驗證用於評估模型的泛化能力",
        "過擬合是機器學習中的常見問題",
        "正則化技術可以幫助防止過擬合"
    ]

    metadatas = [
        {"topic": "編程語言", "language": "Python"},
        {"topic": "編程語言", "language": "Java"},
        {"topic": "編程語言", "language": "JavaScript"},
        {"topic": "機器學習", "subtopic": "基礎"},
        {"topic": "深度學習", "subtopic": "基礎"},
        {"topic": "深度學習", "subtopic": "CNN"},
        {"topic": "深度學習", "subtopic": "RNN"},
        {"topic": "深度學習", "subtopic": "Transformer"},
        {"topic": "NLP", "subtopic": "BERT"},
        {"topic": "NLP", "subtopic": "GPT"},
        {"topic": "數據科學", "subtopic": "數據處理"},
        {"topic": "機器學習", "subtopic": "特徵工程"},
        {"topic": "機器學習", "subtopic": "模型評估"},
        {"topic": "機器學習", "subtopic": "過擬合"},
        {"topic": "機器學習", "subtopic": "正則化"}
    ]

    # 添加文檔
    print("\n2. 添加文檔...")
    db.add_documents(documents, metadatas)

    # 測試查詢
    test_queries = [
        "什麼是 Transformer？",
        "如何防止過擬合？",
        "推薦一種編程語言",
        "神經網絡處理圖像"
    ]

    print("\n3. 測試查詢:")
    print("-" * 60)

    for query in test_queries:
        print(f"\n查詢: {query}")
        results = db.search(query, k=3)

        print(f"\nTop 3 結果:")
        for i, (doc, distance, metadata) in enumerate(results, 1):
            print(f"\n  {i}. 距離: {distance:.4f}")
            print(f"     文檔: {doc}")
            print(f"     元數據: {metadata}")


def demo_performance_comparison():
    """性能比較演示"""
    print("\n\n" + "=" * 60)
    print("向量資料庫性能比較")
    print("=" * 60)

    import time

    # 準備測試數據
    documents = [
        f"這是測試文檔 {i}，內容關於機器學習和人工智慧的各種主題。"
        for i in range(100)
    ]

    # 測試簡單資料庫
    print("\n1. 測試簡單向量資料庫...")
    start_time = time.time()
    simple_db = SimpleVectorDB()
    simple_db.add_documents(documents)
    add_time_simple = time.time() - start_time

    start_time = time.time()
    for _ in range(10):
        simple_db.search("機器學習", k=5)
    search_time_simple = (time.time() - start_time) / 10

    print(f"   添加 100 個文檔耗時: {add_time_simple:.4f} 秒")
    print(f"   平均搜索耗時: {search_time_simple:.4f} 秒")

    # 測試 FAISS 資料庫
    print("\n2. 測試 FAISS 向量資料庫...")
    start_time = time.time()
    faiss_db = FAISSVectorDB()
    faiss_db.add_documents(documents)
    add_time_faiss = time.time() - start_time

    start_time = time.time()
    for _ in range(10):
        faiss_db.search("機器學習", k=5)
    search_time_faiss = (time.time() - start_time) / 10

    print(f"   添加 100 個文檔耗時: {add_time_faiss:.4f} 秒")
    print(f"   平均搜索耗時: {search_time_faiss:.4f} 秒")

    # 性能總結
    print("\n3. 性能總結:")
    print(f"   簡單資料庫 - 添加: {add_time_simple:.4f}s, 搜索: {search_time_simple:.4f}s")
    print(f"   FAISS 資料庫 - 添加: {add_time_faiss:.4f}s, 搜索: {search_time_faiss:.4f}s")


def main():
    """主函數"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 17 + "向量資料庫範例" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")

    demo_simple_vector_db()
    demo_faiss_vector_db()
    demo_performance_comparison()

    print("\n\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n重點回顧:")
    print("1. 向量資料庫使用嵌入向量進行高效的語義搜索")
    print("2. FAISS 等專門的庫提供了更高效的索引和搜索")
    print("3. 元數據可以幫助過濾和組織搜索結果")
    print("4. 選擇合適的向量資料庫取決於數據規模和性能需求")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
