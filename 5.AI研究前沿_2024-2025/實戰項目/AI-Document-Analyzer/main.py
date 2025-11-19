"""
AI 文檔分析系統 - 主應用
支持多種文檔格式的智能分析、摘要、問答
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime

from document_processor import DocumentProcessor
from analyzer import DocumentAnalyzer

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 創建 FastAPI 應用
app = FastAPI(
    title="AI 文檔分析系統",
    description="智能文檔處理、分析和問答系統",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應該限制具體域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心組件
document_processor = DocumentProcessor()
document_analyzer = DocumentAnalyzer()

# ==================== Pydantic 模型 ====================

class AnalysisRequest(BaseModel):
    """文檔分析請求"""
    document_id: str
    analysis_type: str  # summary, entities, keywords, topics
    options: Optional[Dict[str, Any]] = {}


class QuestionRequest(BaseModel):
    """文檔問答請求"""
    document_id: str
    question: str
    context_window: Optional[int] = 1000


class CompareRequest(BaseModel):
    """文檔比較請求"""
    document_ids: List[str]
    comparison_aspects: Optional[List[str]] = ["content", "structure", "topics"]


class BatchAnalysisRequest(BaseModel):
    """批量分析請求"""
    document_ids: List[str]
    analysis_types: List[str]


# ==================== 健康檢查 ====================

@app.get("/api/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Document Analyzer"
    }


@app.get("/api/stats")
async def get_stats():
    """獲取系統統計信息"""
    return {
        "total_documents": document_processor.get_document_count(),
        "supported_formats": document_processor.get_supported_formats(),
        "analysis_types": document_analyzer.get_supported_analyses(),
        "uptime": "healthy"
    }


# ==================== 文檔上傳與管理 ====================

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上傳文檔並進行初步處理

    支持格式：PDF, DOCX, TXT, MD, HTML
    """
    try:
        logger.info(f"Uploading document: {file.filename}")

        # 讀取文件內容
        content = await file.read()

        # 處理文檔
        result = await document_processor.process_document(
            content=content,
            filename=file.filename,
            content_type=file.content_type
        )

        # 在後台執行初步分析
        if background_tasks:
            background_tasks.add_task(
                document_analyzer.analyze_background,
                result['document_id']
            )

        return {
            "document_id": result['document_id'],
            "filename": file.filename,
            "pages": result.get('pages', 1),
            "word_count": result.get('word_count', 0),
            "message": "Document uploaded successfully"
        }

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    file_type: Optional[str] = None
):
    """列出所有文檔"""
    try:
        documents = document_processor.list_documents(
            limit=limit,
            offset=offset,
            file_type=file_type
        )

        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """獲取文檔詳細信息"""
    try:
        document = document_processor.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """刪除文檔"""
    try:
        success = document_processor.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文檔分析 ====================

@app.post("/api/analyze")
async def analyze_document(request: AnalysisRequest):
    """
    分析文檔

    分析類型：
    - summary: 生成摘要
    - entities: 提取實體（人名、地名、組織等）
    - keywords: 提取關鍵詞
    - topics: 主題建模
    - sentiment: 情感分析
    - structure: 文檔結構分析
    """
    try:
        logger.info(f"Analyzing document {request.document_id}: {request.analysis_type}")

        result = await document_analyzer.analyze(
            document_id=request.document_id,
            analysis_type=request.analysis_type,
            options=request.options
        )

        return {
            "document_id": request.document_id,
            "analysis_type": request.analysis_type,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/batch")
async def batch_analyze(request: BatchAnalysisRequest):
    """批量分析多個文檔"""
    try:
        results = await document_analyzer.batch_analyze(
            document_ids=request.document_ids,
            analysis_types=request.analysis_types
        )

        return {
            "results": results,
            "total_documents": len(request.document_ids),
            "analysis_types": request.analysis_types
        }

    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文檔問答 ====================

@app.post("/api/qa")
async def question_answering(request: QuestionRequest):
    """
    文檔問答
    基於上傳的文檔回答問題
    """
    try:
        logger.info(f"QA for document {request.document_id}: {request.question}")

        answer = await document_analyzer.answer_question(
            document_id=request.document_id,
            question=request.question,
            context_window=request.context_window
        )

        return {
            "document_id": request.document_id,
            "question": request.question,
            "answer": answer['answer'],
            "confidence": answer.get('confidence', 0.0),
            "sources": answer.get('sources', [])
        }

    except Exception as e:
        logger.error(f"Error in QA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文檔比較 ====================

@app.post("/api/compare")
async def compare_documents(request: CompareRequest):
    """
    比較多個文檔
    分析相似度、差異、共同主題等
    """
    try:
        logger.info(f"Comparing documents: {request.document_ids}")

        if len(request.document_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 documents required for comparison"
            )

        comparison = await document_analyzer.compare_documents(
            document_ids=request.document_ids,
            aspects=request.comparison_aspects
        )

        return {
            "document_ids": request.document_ids,
            "comparison": comparison,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 搜索與檢索 ====================

@app.get("/api/search")
async def search_documents(
    query: str,
    limit: int = 10,
    file_type: Optional[str] = None
):
    """
    搜索文檔
    基於語義搜索找到相關文檔
    """
    try:
        results = await document_processor.search_documents(
            query=query,
            limit=limit,
            file_type=file_type
        )

        return {
            "query": query,
            "results": results,
            "total": len(results)
        }

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 導出功能 ====================

@app.get("/api/export/{document_id}")
async def export_analysis(
    document_id: str,
    format: str = "json"  # json, markdown, pdf
):
    """
    導出分析結果
    支持多種格式：JSON, Markdown, PDF
    """
    try:
        export_data = await document_analyzer.export_analysis(
            document_id=document_id,
            format=format
        )

        return export_data

    except Exception as e:
        logger.error(f"Error exporting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 啟動應用 ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
