"""數據模型定義"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentMetadata(BaseModel):
    """文檔元數據"""
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime = Field(default_factory=datetime.now)
    chunk_count: int = 0
    language: Optional[str] = "zh"


class QueryRequest(BaseModel):
    """查詢請求"""
    question: str = Field(..., min_length=1, description="用戶問題")
    use_agent: bool = Field(True, description="是否使用 Agent")
    top_k: int = Field(5, ge=1, le=20, description="返回的文檔數量")
    session_id: Optional[str] = Field(None, description="會話 ID")
    filters: Optional[Dict[str, Any]] = Field(None, description="過濾條件")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "什麼是機器學習？",
                "use_agent": True,
                "top_k": 5
            }
        }


class Source(BaseModel):
    """來源文檔"""
    content: str = Field(..., description="文檔內容片段")
    document: str = Field(..., description="文檔名稱")
    page: Optional[int] = Field(None, description="頁碼")
    score: float = Field(..., ge=0, le=1, description="相關性分數")
    metadata: Optional[Dict[str, Any]] = Field(None, description="其他元數據")


class QueryResponse(BaseModel):
    """查詢響應"""
    answer: str = Field(..., description="回答內容")
    sources: List[Source] = Field(default_factory=list, description="來源列表")
    tools_used: List[str] = Field(default_factory=list, description="使用的工具")
    confidence: float = Field(0.0, ge=0, le=1, description="回答置信度")
    suggestions: List[str] = Field(default_factory=list, description="追問建議")
    processing_time: float = Field(0.0, description="處理時間（秒）")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "機器學習是一種人工智能技術...",
                "sources": [
                    {
                        "content": "機器學習定義...",
                        "document": "ml_basics.pdf",
                        "score": 0.95
                    }
                ],
                "tools_used": ["rag_search"],
                "confidence": 0.92,
                "suggestions": ["深度學習和機器學習的區別是什麼？"]
            }
        }


class DocumentUploadRequest(BaseModel):
    """文檔上傳請求"""
    filename: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """文檔上傳響應"""
    success: bool
    document_id: str
    message: str
    chunks_created: int


class SystemStats(BaseModel):
    """系統統計"""
    total_documents: int = 0
    total_chunks: int = 0
    total_queries: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    uptime_seconds: float = 0.0


class HealthResponse(BaseModel):
    """健康檢查響應"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """錯誤響應"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
