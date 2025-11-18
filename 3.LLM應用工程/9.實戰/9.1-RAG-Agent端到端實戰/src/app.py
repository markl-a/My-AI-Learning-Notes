"""FastAPI 應用主入口"""
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from langchain_openai import ChatOpenAI

from src.models import (
    QueryRequest, QueryResponse, HealthResponse,
    SystemStats, ErrorResponse, DocumentUploadResponse
)
from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor
from src.rag_agent_system import RAGAgentSystem
from src.utils import load_config, setup_logging, ensure_dir, SimpleCache

logger = logging.getLogger(__name__)

# Prometheus 指標
REQUEST_COUNT = Counter('rag_requests_total', 'Total number of requests', ['endpoint'])
REQUEST_LATENCY = Histogram('rag_request_duration_seconds', 'Request latency', ['endpoint'])
ERROR_COUNT = Counter('rag_errors_total', 'Total number of errors', ['type'])

# 全局變量
config = None
rag_system = None
vector_store = None
doc_processor = None
cache = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 啟動時初始化
    global config, rag_system, vector_store, doc_processor, cache

    # 加載配置
    config = load_config("config/config.yaml")
    setup_logging(config)

    logger.info("Initializing RAG Agent System...")

    # 初始化緩存
    cache_config = config.get('cache', {})
    if cache_config.get('enabled', True):
        cache = SimpleCache(
            max_size=cache_config.get('max_size', 1000),
            ttl=cache_config.get('ttl', 3600)
        )

    # 初始化向量存儲
    vector_store = VectorStoreManager(
        persist_directory=config.get('vector_store', {}).get('persist_directory', './data/chroma_db'),
        collection_name=config.get('vector_store', {}).get('collection_name', 'documents')
    )

    # 初始化文檔處理器
    doc_processor = DocumentProcessor(
        vector_store=vector_store,
        chunk_size=config.get('rag', {}).get('chunk_size', 1000),
        chunk_overlap=config.get('rag', {}).get('chunk_overlap', 200)
    )

    # 初始化 LLM
    llm_config = config.get('llm', {})
    api_key = os.getenv('OPENAI_API_KEY') or llm_config.get('api_key')

    if not api_key:
        logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")

    llm = ChatOpenAI(
        model=llm_config.get('model', 'gpt-4'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 2000),
        api_key=api_key
    )

    # 初始化 RAG Agent 系統
    rag_system = RAGAgentSystem(
        llm=llm,
        vector_store=vector_store,
        config=config
    )

    logger.info("RAG Agent System initialized successfully")

    yield

    # 關閉時清理
    logger.info("Shutting down RAG Agent System...")


# 創建 FastAPI 應用
app = FastAPI(
    title="RAG Agent System API",
    description="智能文檔問答系統 - RAG + Agent + 部署",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
api_config = config.get('api', {}) if config else {}
cors_origins = api_config.get('cors_origins', ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 中間件：請求日誌和指標
@app.middleware("http")
async def log_requests(request, call_next):
    """請求日誌中間件"""
    start_time_req = time.time()

    # 處理請求
    response = await call_next(request)

    # 記錄指標
    duration = time.time() - start_time_req
    endpoint = request.url.path

    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    logger.info(
        f"{request.method} {endpoint} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.2f}s"
    )

    return response


# API 端點

@app.get("/", tags=["General"])
async def root():
    """根端點"""
    return {
        "message": "RAG Agent System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """健康檢查"""
    components = {
        "vector_store": "healthy",
        "rag_system": "healthy" if rag_system else "unavailable",
        "cache": "healthy" if cache else "disabled"
    }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components=components
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """查詢問答

    處理用戶查詢，使用 RAG 或 Agent 模式返回答案。
    """
    try:
        # 檢查緩存
        if cache:
            cache_key = f"query:{request.question}:{request.use_agent}:{request.top_k}"
            cached_response = cache.get(cache_key)

            if cached_response:
                logger.info(f"Cache hit for query: {request.question[:50]}...")
                return cached_response

        # 處理查詢
        logger.info(f"Processing query: {request.question[:50]}...")

        response = rag_system.query(request)

        # 緩存結果
        if cache and response:
            cache.set(cache_key, response)

        return response

    except Exception as e:
        ERROR_COUNT.labels(type="query_error").inc()
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"]
)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """上傳文檔

    上傳文檔並建立索引。支持 PDF、TXT、MD、DOCX、PPTX 格式。
    """
    try:
        # 檢查文件大小
        max_size = config.get('document_processing', {}).get('max_file_size_mb', 50) * 1024 * 1024

        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {max_size / 1024 / 1024}MB"
            )

        # 保存文件
        ensure_dir("./data/uploads")
        file_path = f"./data/uploads/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(content)

        # 處理文檔
        logger.info(f"Processing uploaded file: {file.filename}")
        result = doc_processor.process_file(file_path)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail="Failed to process document")

        return DocumentUploadResponse(
            success=True,
            document_id=result.get("file_hash", ""),
            message="Document uploaded and indexed successfully",
            chunks_created=result.get("chunks", 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(type="upload_error").inc()
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=SystemStats, tags=["System"])
async def get_stats():
    """獲取系統統計"""
    try:
        system_stats = rag_system.get_stats()

        return SystemStats(
            total_documents=system_stats.get('vector_store', {}).get('document_count', 0),
            total_chunks=system_stats.get('vector_store', {}).get('document_count', 0),
            total_queries=system_stats.get('total_queries', 0),
            avg_response_time=system_stats.get('avg_response_time', 0.0),
            cache_hit_rate=0.0,  # TODO: 實現緩存命中率統計
            uptime_seconds=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus 指標端點"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/cache/clear", tags=["System"])
async def clear_cache():
    """清空緩存"""
    if cache:
        cache.clear()
        return {"message": "Cache cleared successfully"}
    return {"message": "Cache is disabled"}


# 錯誤處理

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 異常處理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用異常處理"""
    ERROR_COUNT.labels(type="unhandled_error").inc()
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


# 主函數

def main():
    """主函數"""
    import uvicorn

    # 加載配置
    config = load_config("config/config.yaml")
    api_config = config.get('api', {})

    # 啟動服務
    uvicorn.run(
        "src.app:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', False),
        workers=api_config.get('workers', 1)
    )


if __name__ == "__main__":
    main()
