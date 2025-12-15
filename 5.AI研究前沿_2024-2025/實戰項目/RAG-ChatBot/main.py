"""
RAG ChatBot - FastAPI 後端
完整的檢索增強生成聊天機器人 API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from rag_engine import RAGEngine
from middleware.rate_limiter import rate_limiter, rate_limit_middleware


# 生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 啟動時
    await rate_limiter.start()
    yield
    # 關閉時
    await rate_limiter.stop()


# 創建 FastAPI 應用
app = FastAPI(
    title="RAG ChatBot API",
    description="檢索增強生成聊天機器人",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置 - 從環境變量讀取允許的來源
import os
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# 速率限制中間件
app.middleware("http")(rate_limit_middleware)

# 初始化 RAG 引擎
rag_engine = RAGEngine()


# === 數據模型 ===

class ChatRequest(BaseModel):
    """聊天請求"""
    message: str
    conversation_id: Optional[str] = None
    use_rag: bool = True
    top_k: int = 3


class ChatResponse(BaseModel):
    """聊天回應"""
    response: str
    conversation_id: str
    sources: Optional[List[dict]] = None
    metadata: Optional[dict] = None


class DocumentRequest(BaseModel):
    """文檔添加請求"""
    content: str
    metadata: Optional[dict] = None


# === API 路由 ===

@app.get("/")
async def root():
    """根路徑 - 返回 Web 界面"""
    return HTMLResponse(content=open("frontend/index.html", encoding="utf-8").read())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    - **message**: 用戶消息
    - **conversation_id**: 對話 ID（可選）
    - **use_rag**: 是否使用 RAG
    - **top_k**: 檢索文檔數量
    """
    try:
        result = await rag_engine.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            use_rag=request.use_rag,
            top_k=request.top_k
        )

        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            sources=result.get("sources"),
            metadata=result.get("metadata")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    async def generate():
        try:
            async for chunk in rag_engine.chat_stream(
                message=request.message,
                conversation_id=request.conversation_id,
                use_rag=request.use_rag,
                top_k=request.top_k
            ):
                yield f"data: {chunk}\n\n"

        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/documents")
async def add_document(request: DocumentRequest):
    """添加文檔到知識庫"""
    try:
        doc_id = await rag_engine.add_document(
            content=request.content,
            metadata=request.metadata
        )

        return {
            "status": "success",
            "document_id": doc_id,
            "message": "文檔添加成功"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上傳文檔文件"""
    try:
        # 讀取文件內容
        content = await file.read()
        text_content = content.decode('utf-8')

        doc_id = await rag_engine.add_document(
            content=text_content,
            metadata={"filename": file.filename}
        )

        return {
            "status": "success",
            "document_id": doc_id,
            "filename": file.filename,
            "message": "文件上傳成功"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """列出所有文檔"""
    try:
        documents = await rag_engine.list_documents()
        return {
            "status": "success",
            "count": len(documents),
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """刪除文檔"""
    try:
        await rag_engine.delete_document(doc_id)
        return {
            "status": "success",
            "message": f"文檔 {doc_id} 已刪除"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """獲取對話歷史"""
    try:
        history = await rag_engine.get_conversation(conversation_id)
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """刪除對話"""
    try:
        await rag_engine.delete_conversation(conversation_id)
        return {
            "status": "success",
            "message": f"對話 {conversation_id} 已刪除"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "documents_count": len(rag_engine.documents)
    }


@app.get("/api/stats")
async def get_stats():
    """獲取統計信息"""
    try:
        stats = await rag_engine.get_stats()
        return {
            "status": "success",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === 啟動配置 ===

if __name__ == "__main__":
    # 開發模式
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

    # 生產模式（取消註釋）
    # uvicorn.run(
    #     "main:app",
    #     host="0.0.0.0",
    #     port=8000,
    #     workers=4,
    #     log_level="warning"
    # )
