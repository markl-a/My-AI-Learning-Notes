"""
文檔處理器
處理多種格式的文檔：PDF、DOCX、TXT、Markdown、HTML
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

# 文檔處理庫
try:
    import PyPDF2
    from docx import Document
    import markdown
    from bs4 import BeautifulSoup
except ImportError:
    logging.warning("Some document processing libraries not installed")

# 向量數據庫
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文檔處理器類"""

    def __init__(
        self,
        storage_path: str = "./documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        初始化文檔處理器

        Args:
            storage_path: 文檔存儲路徑
            embedding_model: 嵌入模型名稱
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # 初始化嵌入模型
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        # 初始化向量數據庫
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document embeddings"}
        )

        # 文檔元數據存儲
        self.metadata_file = os.path.join(storage_path, "metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """加載文檔元數據"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """保存文檔元數據"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def get_supported_formats(self) -> List[str]:
        """獲取支持的文檔格式"""
        return ['pdf', 'docx', 'txt', 'md', 'markdown', 'html']

    def get_document_count(self) -> int:
        """獲取文檔總數"""
        return len(self.metadata)

    async def process_document(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        處理上傳的文檔

        Args:
            content: 文檔內容（字節）
            filename: 文件名
            content_type: MIME 類型

        Returns:
            處理結果字典
        """
        # 生成文檔 ID
        doc_id = str(uuid.uuid4())

        # 確定文件類型
        file_ext = filename.split('.')[-1].lower()

        # 提取文本內容
        text_content, metadata = await self._extract_text(content, file_ext, filename)

        # 保存原始文件
        file_path = os.path.join(self.storage_path, f"{doc_id}.{file_ext}")
        with open(file_path, 'wb') as f:
            f.write(content)

        # 計算文檔哈希
        doc_hash = hashlib.sha256(content).hexdigest()

        # 分塊處理長文檔
        chunks = self._split_into_chunks(text_content, chunk_size=500)

        # 生成嵌入並存儲
        embeddings = []
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            embedding = self.encoder.encode(chunk).tolist()
            embeddings.append(embedding)

        # 存儲到向量數據庫
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{
                "document_id": doc_id,
                "chunk_index": i,
                "filename": filename,
                "file_type": file_ext
            } for i in range(len(chunks))]
        )

        # 保存元數據
        self.metadata[doc_id] = {
            "filename": filename,
            "file_type": file_ext,
            "file_path": file_path,
            "content_type": content_type,
            "hash": doc_hash,
            "pages": metadata.get('pages', 1),
            "word_count": len(text_content.split()),
            "chunk_count": len(chunks),
            "created_at": datetime.utcnow().isoformat(),
            "text_content": text_content[:1000]  # 保存前1000字符預覽
        }
        self._save_metadata()

        logger.info(f"Document processed: {doc_id} ({filename})")

        return {
            "document_id": doc_id,
            "filename": filename,
            "pages": metadata.get('pages', 1),
            "word_count": len(text_content.split()),
            "chunks": len(chunks)
        }

    async def _extract_text(
        self,
        content: bytes,
        file_ext: str,
        filename: str
    ) -> tuple[str, Dict]:
        """
        從不同格式提取文本

        Args:
            content: 文件內容
            file_ext: 文件擴展名
            filename: 文件名

        Returns:
            (text_content, metadata)
        """
        metadata = {}

        try:
            if file_ext == 'pdf':
                return self._extract_pdf(content)
            elif file_ext == 'docx':
                return self._extract_docx(content)
            elif file_ext in ['txt', 'md', 'markdown']:
                text = content.decode('utf-8')
                return text, metadata
            elif file_ext == 'html':
                return self._extract_html(content)
            else:
                # 嘗試作為純文本處理
                text = content.decode('utf-8', errors='ignore')
                return text, metadata

        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise

    def _extract_pdf(self, content: bytes) -> tuple[str, Dict]:
        """提取 PDF 文本"""
        import io
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        metadata = {"pages": len(pdf_reader.pages)}
        return text, metadata

    def _extract_docx(self, content: bytes) -> tuple[str, Dict]:
        """提取 DOCX 文本"""
        import io
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)

        text = "\n".join([para.text for para in doc.paragraphs])
        metadata = {"pages": len(doc.sections)}
        return text, metadata

    def _extract_html(self, content: bytes) -> tuple[str, Dict]:
        """提取 HTML 文本"""
        html = content.decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')

        # 移除腳本和樣式
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        # 清理空白
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)

        return text, {}

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        將文本分塊

        Args:
            text: 文本內容
            chunk_size: 每塊的單詞數
            overlap: 重疊的單詞數

        Returns:
            文本塊列表
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        file_type: Optional[str] = None
    ) -> List[Dict]:
        """列出文檔"""
        documents = list(self.metadata.values())

        # 按文件類型篩選
        if file_type:
            documents = [d for d in documents if d['file_type'] == file_type]

        # 按創建時間排序
        documents.sort(key=lambda x: x['created_at'], reverse=True)

        # 分頁
        return documents[offset:offset + limit]

    def get_document(self, document_id: str) -> Optional[Dict]:
        """獲取文檔詳情"""
        return self.metadata.get(document_id)

    def delete_document(self, document_id: str) -> bool:
        """刪除文檔"""
        if document_id not in self.metadata:
            return False

        # 刪除向量數據庫中的數據
        try:
            # 獲取所有相關的 chunk IDs
            result = self.collection.get(
                where={"document_id": document_id}
            )
            if result['ids']:
                self.collection.delete(ids=result['ids'])
        except Exception as e:
            logger.error(f"Error deleting from vector DB: {str(e)}")

        # 刪除文件
        file_path = self.metadata[document_id].get('file_path')
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # 刪除元數據
        del self.metadata[document_id]
        self._save_metadata()

        logger.info(f"Document deleted: {document_id}")
        return True

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None
    ) -> List[Dict]:
        """
        語義搜索文檔

        Args:
            query: 查詢文本
            limit: 返回結果數量
            file_type: 文件類型篩選

        Returns:
            搜索結果列表
        """
        # 生成查詢嵌入
        query_embedding = self.encoder.encode(query).tolist()

        # 構建查詢條件
        where_clause = {}
        if file_type:
            where_clause["file_type"] = file_type

        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause if where_clause else None
        )

        # 組織結果
        search_results = []
        if results['documents']:
            for i, doc_id in enumerate(results['ids'][0]):
                document_id = results['metadatas'][0][i]['document_id']
                search_results.append({
                    "document_id": document_id,
                    "chunk": results['documents'][0][i],
                    "metadata": self.metadata.get(document_id, {}),
                    "distance": results['distances'][0][i]
                })

        return search_results

    def get_document_text(self, document_id: str) -> Optional[str]:
        """獲取文檔完整文本"""
        if document_id not in self.metadata:
            return None

        file_path = self.metadata[document_id]['file_path']
        if not os.path.exists(file_path):
            return None

        # 讀取文件
        with open(file_path, 'rb') as f:
            content = f.read()

        # 提取文本
        file_ext = self.metadata[document_id]['file_type']
        text, _ = self._extract_text(content, file_ext, file_path)

        return text
