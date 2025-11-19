"""
RAG Engine - 檢索增強生成核心引擎
整合向量檢索、LLM生成、對話管理
"""

from typing import List, Dict, Optional, AsyncIterator
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
import os


class RAGEngine:
    """RAG 引擎"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        chroma_persist_dir: str = "./chroma_db"
    ):
        """
        初始化 RAG 引擎

        Args:
            embedding_model: 嵌入模型
            llm_model: LLM 模型
            chroma_persist_dir: ChromaDB 持久化目錄
        """
        # 嵌入模型
        self.encoder = SentenceTransformer(embedding_model)

        # 向量數據庫
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_persist_dir
        ))

        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # LLM 配置
        self.llm_model = llm_model
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 對話歷史（簡單版本，生產環境應使用數據庫）
        self.conversations: Dict[str, List[Dict]] = {}

        # 文檔元數據
        self.documents: Dict[str, Dict] = {}

        print(f"RAG 引擎初始化完成")
        print(f"嵌入模型: {embedding_model}")
        print(f"LLM 模型: {llm_model}")

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加文檔

        Args:
            content: 文檔內容
            metadata: 元數據

        Returns:
            文檔 ID
        """
        doc_id = str(uuid.uuid4())

        # 生成嵌入
        embedding = self.encoder.encode(content).tolist()

        # 添加到向量數據庫
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )

        # 保存元數據
        self.documents[doc_id] = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

        return doc_id

    async def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        檢索相關文檔

        Args:
            query: 查詢
            top_k: 返回數量

        Returns:
            相關文檔列表
        """
        # 生成查詢嵌入
        query_embedding = self.encoder.encode(query).tolist()

        # 檢索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 格式化結果
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_id, doc, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['distances'][0]
            )):
                documents.append({
                    "id": doc_id,
                    "content": doc,
                    "score": 1 - distance,  # 轉換為相似度分數
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return documents

    async def generate_response(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        生成回覆

        Args:
            query: 用戶查詢
            context_docs: 上下文文檔
            conversation_history: 對話歷史

        Returns:
            生成的回覆
        """
        # 構建提示
        messages = []

        # 系統提示
        system_prompt = "你是一個helpful的AI助手。"

        if context_docs:
            context_text = "\n\n".join([
                f"文檔 {i+1}:\n{doc['content']}"
                for i, doc in enumerate(context_docs)
            ])
            system_prompt += f"\n\n請基於以下文檔回答問題：\n\n{context_text}"

        messages.append({"role": "system", "content": system_prompt})

        # 添加對話歷史
        if conversation_history:
            messages.extend(conversation_history[-5:])  # 最近5輪對話

        # 添加當前查詢
        messages.append({"role": "user", "content": query})

        # 調用 LLM
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
        top_k: int = 3
    ) -> Dict:
        """
        聊天

        Args:
            message: 用戶消息
            conversation_id: 對話 ID
            use_rag: 是否使用 RAG
            top_k: 檢索文檔數量

        Returns:
            回覆字典
        """
        # 創建或獲取對話
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # 檢索相關文檔
        context_docs = None
        if use_rag:
            context_docs = await self.retrieve(message, top_k=top_k)

        # 生成回覆
        response = await self.generate_response(
            query=message,
            context_docs=context_docs,
            conversation_history=self.conversations[conversation_id]
        )

        # 更新對話歷史
        self.conversations[conversation_id].append(
            {"role": "user", "content": message}
        )
        self.conversations[conversation_id].append(
            {"role": "assistant", "content": response}
        )

        # 返回結果
        return {
            "response": response,
            "conversation_id": conversation_id,
            "sources": context_docs,
            "metadata": {
                "use_rag": use_rag,
                "retrieved_docs": len(context_docs) if context_docs else 0
            }
        }

    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
        top_k: int = 3
    ) -> AsyncIterator[str]:
        """流式聊天（簡化版本）"""
        result = await self.chat(message, conversation_id, use_rag, top_k)

        # 模擬流式輸出
        response = result["response"]
        words = response.split()

        for i, word in enumerate(words):
            yield word + " "
            if i % 5 == 0:  # 每5個字暫停一下
                import asyncio
                await asyncio.sleep(0.1)

    async def list_documents(self) -> List[Dict]:
        """列出所有文檔"""
        return list(self.documents.values())

    async def delete_document(self, doc_id: str):
        """刪除文檔"""
        if doc_id in self.documents:
            self.collection.delete(ids=[doc_id])
            del self.documents[doc_id]

    async def get_conversation(self, conversation_id: str) -> List[Dict]:
        """獲取對話歷史"""
        return self.conversations.get(conversation_id, [])

    async def delete_conversation(self, conversation_id: str):
        """刪除對話"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    async def get_stats(self) -> Dict:
        """獲取統計信息"""
        return {
            "total_documents": len(self.documents),
            "total_conversations": len(self.conversations),
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": self.llm_model
        }
