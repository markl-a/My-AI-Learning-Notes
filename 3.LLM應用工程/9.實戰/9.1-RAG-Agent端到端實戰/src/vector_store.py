"""向量存儲管理"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存儲管理器"""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_model: Optional[Embeddings] = None
    ):
        """初始化向量存儲

        Args:
            persist_directory: 持久化目錄
            collection_name: 集合名稱
            embedding_model: Embedding 模型
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 確保目錄存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客戶端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 獲取或創建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings"}
            )
            logger.info(f"Created new collection: {collection_name}")

        # 初始化 Embedding 模型
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加文檔到向量存儲

        Args:
            texts: 文本列表
            metadatas: 元數據列表
            ids: ID 列表

        Returns:
            文檔 ID 列表
        """
        if not texts:
            return []

        try:
            # 生成 ID（如果未提供）
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in texts]

            # 生成 embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_model.embed_documents(texts)

            # 準備元數據
            if metadatas is None:
                metadatas = [{}] * len(texts)

            # 添加到集合
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully added {len(texts)} documents")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """相似度搜索

        Args:
            query: 查詢文本
            top_k: 返回結果數量
            filter_dict: 過濾條件

        Returns:
            (文本, 元數據, 分數) 的列表
        """
        try:
            # 生成查詢 embedding
            query_embedding = self.embedding_model.embed_query(query)

            # 執行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict
            )

            # 格式化結果
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0.0

                    # 將距離轉換為相似度分數 (0-1)
                    # ChromaDB 使用歐氏距離，越小越相似
                    score = 1.0 / (1.0 + distance)

                    documents.append((doc, metadata, score))

            return documents

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def delete_documents(self, ids: List[str]) -> bool:
        """刪除文檔

        Args:
            ids: 文檔 ID 列表

        Returns:
            是否成功
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_document_count(self) -> int:
        """獲取文檔數量"""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self):
        """清空集合"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.get_document_count(),
            "persist_directory": self.persist_directory
        }


class HybridSearcher:
    """混合搜索器（向量 + 關鍵詞）"""

    def __init__(self, vector_store: VectorStoreManager):
        """初始化混合搜索器

        Args:
            vector_store: 向量存儲管理器
        """
        self.vector_store = vector_store
        self.documents = []  # 用於 BM25 搜索
        self.doc_ids = []

    def add_documents_for_keyword_search(self, texts: List[str], ids: List[str]):
        """添加文檔用於關鍵詞搜索

        Args:
            texts: 文本列表
            ids: ID 列表
        """
        self.documents.extend(texts)
        self.doc_ids.extend(ids)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """混合搜索

        Args:
            query: 查詢文本
            top_k: 返回結果數量
            vector_weight: 向量搜索權重 (0-1)

        Returns:
            (文本, 元數據, 分數) 的列表
        """
        # 向量搜索
        vector_results = self.vector_store.similarity_search(query, top_k=top_k*2)

        # 關鍵詞搜索 (BM25)
        keyword_results = self._bm25_search(query, top_k=top_k*2)

        # 合併結果
        merged_results = self._merge_results(
            vector_results,
            keyword_results,
            vector_weight
        )

        return merged_results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, Dict, float]]:
        """BM25 關鍵詞搜索"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba

            # 分詞
            tokenized_docs = [list(jieba.cut(doc)) for doc in self.documents]
            tokenized_query = list(jieba.cut(query))

            # BM25 搜索
            bm25 = BM25Okapi(tokenized_docs)
            scores = bm25.get_scores(tokenized_query)

            # 獲取 top_k 結果
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                results.append((
                    self.documents[idx],
                    {"doc_id": self.doc_ids[idx]},
                    scores[idx]
                ))

            return results

        except ImportError:
            logger.warning("BM25 not available, install rank-bm25")
            return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _merge_results(
        self,
        vector_results: List[Tuple[str, Dict, float]],
        keyword_results: List[Tuple[str, Dict, float]],
        vector_weight: float
    ) -> List[Tuple[str, Dict, float]]:
        """合併和重排序結果

        使用倒數排名融合 (Reciprocal Rank Fusion)
        """
        # 創建文檔到分數的映射
        doc_scores = {}

        # 處理向量搜索結果
        for rank, (doc, metadata, score) in enumerate(vector_results, 1):
            doc_key = doc[:100]  # 使用前100個字符作為鍵
            rrf_score = vector_weight / (rank + 60)  # RRF 公式
            doc_scores[doc_key] = {
                "doc": doc,
                "metadata": metadata,
                "score": rrf_score
            }

        # 處理關鍵詞搜索結果
        keyword_weight = 1 - vector_weight
        for rank, (doc, metadata, score) in enumerate(keyword_results, 1):
            doc_key = doc[:100]
            rrf_score = keyword_weight / (rank + 60)

            if doc_key in doc_scores:
                doc_scores[doc_key]["score"] += rrf_score
            else:
                doc_scores[doc_key] = {
                    "doc": doc,
                    "metadata": metadata,
                    "score": rrf_score
                }

        # 排序並返回
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [(d["doc"], d["metadata"], d["score"]) for d in sorted_docs]
