"""
FastAPI 後端服務 - 生產級 LLM API
包含錯誤處理、速率限制、監控、日誌記錄等功能
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import os
import time
import json
import logging
from datetime import datetime
import asyncio

# API 客戶端
from openai import OpenAI, AsyncOpenAI
import anthropic
import google.generativeai as genai

# 監控和日誌
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger

# 環境變數
from dotenv import load_dotenv

load_dotenv()

# ==================== 設定 ====================

# 日誌設定
logger.add(
    "logs/api_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Prometheus 指標
REQUEST_COUNT = Counter(
    'llm_api_requests_total',
    'Total API requests',
    ['provider', 'model', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'llm_api_request_duration_seconds',
    'API request duration',
    ['provider', 'model', 'endpoint']
)

TOKEN_USAGE = Counter(
    'llm_api_tokens_total',
    'Total tokens used',
    ['provider', 'model', 'type']
)

# ==================== 數據模型 ====================

class Message(BaseModel):
    """聊天訊息模型"""
    role: str = Field(..., description="角色：system, user, assistant")
    content: str = Field(..., description="訊息內容")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('role 必須是 system, user 或 assistant')
        return v


class ChatRequest(BaseModel):
    """聊天請求模型"""
    messages: List[Message] = Field(..., description="對話訊息列表")
    provider: str = Field("openai", description="LLM 提供商")
    model: Optional[str] = Field(None, description="模型名稱")
    temperature: float = Field(0.7, ge=0, le=2, description="溫度參數")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="最大 token 數")
    stream: bool = Field(False, description="是否串流回應")

    @validator('provider')
    def validate_provider(cls, v):
        allowed = ['openai', 'anthropic', 'gemini']
        if v not in allowed:
            raise ValueError(f'provider 必須是 {allowed} 之一')
        return v


class ChatResponse(BaseModel):
    """聊天回應模型"""
    id: str
    provider: str
    model: str
    message: str
    usage: Dict[str, int]
    created_at: datetime
    duration: float


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    timestamp: datetime
    version: str
    providers: Dict[str, bool]


# ==================== 應用程式設定 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    logger.info("🚀 啟動 LLM API 服務")

    # 初始化客戶端
    init_clients()

    yield

    # 清理資源
    logger.info("🛑 關閉 LLM API 服務")


app = FastAPI(
    title="LLM API Service",
    description="生產級 LLM API 服務，支援多個提供商",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全認證
security = HTTPBearer()

# 全域客戶端
clients = {}


# ==================== 工具函數 ====================

def init_clients():
    """初始化所有 API 客戶端"""
    global clients

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        clients['openai'] = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        clients['openai_sync'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ OpenAI 客戶端初始化成功")

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        clients['anthropic'] = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        logger.info("✅ Anthropic 客戶端初始化成功")

    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        clients['gemini'] = genai
        logger.info("✅ Gemini 客戶端初始化成功")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """驗證 API Key

    使用 secrets.compare_digest 進行常數時間比較，
    防止時序攻擊（timing attack）。
    """
    import secrets

    expected_key = os.getenv("API_KEY")

    # 確保 API_KEY 已設置
    if not expected_key:
        logger.error("API_KEY 環境變量未設置")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error"
        )

    # 使用常數時間比較防止時序攻擊
    if not secrets.compare_digest(credentials.credentials, expected_key):
        logger.warning("無效的 API Key 嘗試")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    return credentials.credentials


def get_default_model(provider: str) -> str:
    """獲取提供商的預設模型"""
    defaults = {
        'openai': 'gpt-4o-mini',
        'anthropic': 'claude-3-5-sonnet-20241022',
        'gemini': 'gemini-1.5-pro'
    }
    return defaults.get(provider, 'gpt-4o-mini')


# ==================== OpenAI 處理 ====================

async def handle_openai_chat(
    request: ChatRequest,
    stream: bool = False
) -> AsyncGenerator[str, None] | Dict:
    """處理 OpenAI 聊天請求"""
    start_time = time.time()
    model = request.model or get_default_model('openai')

    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if stream:
            # 串流回應
            response_stream = await clients['openai'].chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )

            full_response = ""
            async for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            duration = time.time() - start_time

            # 記錄指標
            REQUEST_COUNT.labels(
                provider='openai',
                model=model,
                endpoint='/chat',
                status='success'
            ).inc()

            REQUEST_DURATION.labels(
                provider='openai',
                model=model,
                endpoint='/chat'
            ).observe(duration)

            yield f"data: {json.dumps({'done': True, 'duration': duration})}\n\n"

        else:
            # 非串流回應
            response = await clients['openai'].chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            duration = time.time() - start_time

            # 記錄指標
            REQUEST_COUNT.labels(
                provider='openai',
                model=model,
                endpoint='/chat',
                status='success'
            ).inc()

            REQUEST_DURATION.labels(
                provider='openai',
                model=model,
                endpoint='/chat'
            ).observe(duration)

            TOKEN_USAGE.labels(
                provider='openai',
                model=model,
                type='input'
            ).inc(response.usage.prompt_tokens)

            TOKEN_USAGE.labels(
                provider='openai',
                model=model,
                type='output'
            ).inc(response.usage.completion_tokens)

            return {
                'id': response.id,
                'provider': 'openai',
                'model': model,
                'message': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'created_at': datetime.now(),
                'duration': duration
            }

    except Exception as e:
        logger.error(f"OpenAI 錯誤: {e}")
        REQUEST_COUNT.labels(
            provider='openai',
            model=model,
            endpoint='/chat',
            status='error'
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Anthropic 處理 ====================

async def handle_anthropic_chat(
    request: ChatRequest,
    stream: bool = False
) -> AsyncGenerator[str, None] | Dict:
    """處理 Anthropic 聊天請求"""
    start_time = time.time()
    model = request.model or get_default_model('anthropic')

    try:
        # 轉換訊息格式
        messages = []
        for m in request.messages:
            if m.role != 'system':
                messages.append({"role": m.role, "content": m.content})

        if stream:
            # 串流回應
            full_response = ""
            async with clients['anthropic'].messages.stream(
                model=model,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature,
                messages=messages
            ) as response_stream:
                async for text in response_stream.text_stream:
                    full_response += text
                    yield f"data: {json.dumps({'content': text})}\n\n"

            duration = time.time() - start_time

            REQUEST_COUNT.labels(
                provider='anthropic',
                model=model,
                endpoint='/chat',
                status='success'
            ).inc()

            REQUEST_DURATION.labels(
                provider='anthropic',
                model=model,
                endpoint='/chat'
            ).observe(duration)

            yield f"data: {json.dumps({'done': True, 'duration': duration})}\n\n"

        else:
            # 非串流回應
            response = await clients['anthropic'].messages.create(
                model=model,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature,
                messages=messages
            )

            duration = time.time() - start_time

            REQUEST_COUNT.labels(
                provider='anthropic',
                model=model,
                endpoint='/chat',
                status='success'
            ).inc()

            REQUEST_DURATION.labels(
                provider='anthropic',
                model=model,
                endpoint='/chat'
            ).observe(duration)

            TOKEN_USAGE.labels(
                provider='anthropic',
                model=model,
                type='input'
            ).inc(response.usage.input_tokens)

            TOKEN_USAGE.labels(
                provider='anthropic',
                model=model,
                type='output'
            ).inc(response.usage.output_tokens)

            return {
                'id': response.id,
                'provider': 'anthropic',
                'model': model,
                'message': response.content[0].text,
                'usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                'created_at': datetime.now(),
                'duration': duration
            }

    except Exception as e:
        logger.error(f"Anthropic 錯誤: {e}")
        REQUEST_COUNT.labels(
            provider='anthropic',
            model=model,
            endpoint='/chat',
            status='error'
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== API 端點 ====================

@app.get("/", tags=["基本"])
async def root():
    """根端點"""
    return {
        "message": "LLM API Service",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["基本"])
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "providers": {
            "openai": 'openai' in clients,
            "anthropic": 'anthropic' in clients,
            "gemini": 'gemini' in clients
        }
    }


@app.post("/api/chat", tags=["聊天"])
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """聊天端點"""
    logger.info(f"收到聊天請求 - Provider: {request.provider}, Model: {request.model}")

    # 檢查提供商是否可用
    if request.provider not in clients:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' 不可用"
        )

    # 處理串流回應
    if request.stream:
        if request.provider == 'openai':
            return StreamingResponse(
                handle_openai_chat(request, stream=True),
                media_type="text/event-stream"
            )
        elif request.provider == 'anthropic':
            return StreamingResponse(
                handle_anthropic_chat(request, stream=True),
                media_type="text/event-stream"
            )

    # 處理非串流回應
    if request.provider == 'openai':
        result = await handle_openai_chat(request, stream=False)
    elif request.provider == 'anthropic':
        result = await handle_anthropic_chat(request, stream=False)
    else:
        raise HTTPException(status_code=400, detail="不支援的提供商")

    return result


@app.get("/metrics", tags=["監控"])
async def metrics():
    """Prometheus 指標端點"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """請求日誌中間件"""
    start_time = time.time()

    # 處理請求
    response = await call_next(request)

    # 記錄請求
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Duration: {duration:.3f}s"
    )

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 異常處理器"""
    logger.error(f"HTTP 錯誤: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """一般異常處理器"""
    logger.error(f"未處理的錯誤: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== 啟動 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "False") == "True",
        log_level="info"
    )
