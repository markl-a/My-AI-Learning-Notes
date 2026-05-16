"""
生產級 LLM FastAPI 應用

這個範例展示如何構建一個生產就緒的 LLM API 服務，包含：
- 健康檢查端點
- 結構化日誌
- 錯誤處理
- API 金鑰驗證
- 請求/回應模型
- 監控指標

運行方式：
1. Docker: docker-compose up
2. 本地: uvicorn app:app --reload

API 文檔: http://localhost:8000/docs

作者：AI Learning Notes
日期：2024-11
"""

import os
import logging
import time
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# ===== 配置日誌 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== FastAPI 應用初始化 =====
app = FastAPI(
    title="LLM API Service",
    description="生產級 LLM API 服務，支持 OpenAI 和 Anthropic Claude",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== CORS 配置 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應該限制具體域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API 金鑰配置 =====
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 從環境變數獲取 API 金鑰
VALID_API_KEYS = os.getenv("API_KEYS", "test-key-123").split(",")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ===== 請求/回應模型 =====
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: user 或 assistant")
    content: str = Field(..., description="消息內容")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="對話歷史")
    model: str = Field(default="gpt-4o-mini", description="使用的模型")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="溫度參數")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="最大生成 token 數")

    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v > 4000:
            logger.warning(f"max_tokens {v} exceeds limit, setting to 4000")
            return 4000
        return v


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 回應內容")
    model: str = Field(..., description="使用的模型")
    usage: dict = Field(default={}, description="Token 使用情況")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    checks: dict


# ===== 中間件：請求日誌 =====
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # 記錄請求
    logger.info(f"Request: {request.method} {request.url.path}")

    # 處理請求
    response = await call_next(request)

    # 記錄響應時間
    process_time = time.time() - start_time
    logger.info(f"Completed in {process_time:.2f}s with status {response.status_code}")

    # 添加自定義 header
    response.headers["X-Process-Time"] = str(process_time)

    return response


# ===== 依賴項：API 金鑰驗證 =====
async def verify_api_key(api_key: str = Security(api_key_header)):
    """驗證 API 金鑰"""
    if api_key is None:
        logger.warning("API key missing")
        raise HTTPException(
            status_code=401,
            detail="API Key is missing. Please provide X-API-Key header."
        )

    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key: {api_key[:10]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )

    return api_key


# ===== LLM 調用函數 =====
async def call_openai(messages: List[dict], model: str, temperature: float, max_tokens: int) -> dict:
    """調用 OpenAI API"""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


async def call_claude(messages: List[dict], model: str, temperature: float, max_tokens: int) -> dict:
    """調用 Claude API"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Anthropic API key not configured"
        )

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # 分離 system message
        system_msg = ""
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append(msg)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=claude_messages
        )

        return {
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")


# ===== API 端點 =====

@app.get("/", tags=["Root"])
async def root():
    """根端點"""
    return {
        "message": "LLM API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    健康檢查端點

    用於 Docker 健康檢查、Kubernetes liveness/readiness probes 等
    """
    checks = {
        "api": "healthy",
        "openai_configured": OPENAI_API_KEY is not None,
        "claude_configured": ANTHROPIC_API_KEY is not None
    }

    # 檢查是否至少有一個 LLM API 可用
    if not any([checks["openai_configured"], checks["claude_configured"]]):
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
                "message": "No LLM API configured"
            }
        )

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        checks=checks
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    聊天端點

    發送消息到 LLM 並獲取回應。

    **認證：** 需要在 header 中提供 X-API-Key

    **支持的模型：**
    - OpenAI: gpt-4o-mini, gpt-4
    - Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229

    **範例請求：**
    ```json
    {
        "messages": [
            {"role": "user", "content": "你好！"}
        ],
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    ```
    """
    logger.info(f"Chat request with model: {request.model}")

    # 轉換消息格式
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # 根據模型選擇調用的 API
    try:
        if request.model.startswith("gpt"):
            result = await call_openai(messages, request.model, request.temperature, request.max_tokens)
        elif request.model.startswith("claude"):
            result = await call_claude(messages, request.model, request.temperature, request.max_tokens)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}"
            )

        return ChatResponse(
            response=result["response"],
            model=request.model,
            usage=result.get("usage", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models", tags=["Info"])
async def list_models(api_key: str = Depends(verify_api_key)):
    """列出可用的模型"""
    models = []

    if OPENAI_API_KEY:
        models.extend([
            {"provider": "OpenAI", "model": "gpt-4o-mini", "description": "Fast and cost-effective"},
            {"provider": "OpenAI", "model": "gpt-4", "description": "Most capable model"},
        ])

    if ANTHROPIC_API_KEY:
        models.extend([
            {"provider": "Anthropic", "model": "claude-3-5-sonnet-20241022", "description": "Balanced performance"},
            {"provider": "Anthropic", "model": "claude-3-opus-20240229", "description": "Most intelligent"},
        ])

    return {"models": models, "total": len(models)}


# ===== 啟動事件 =====
@app.on_event("startup")
async def startup_event():
    """應用啟動時執行"""
    logger.info("="*50)
    logger.info("LLM API Service Starting...")
    logger.info(f"OpenAI API: {'Configured' if OPENAI_API_KEY else 'Not Configured'}")
    logger.info(f"Claude API: {'Configured' if ANTHROPIC_API_KEY else 'Not Configured'}")
    logger.info("="*50)


@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時執行"""
    logger.info("LLM API Service Shutting down...")


# ===== 錯誤處理 =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局異常處理"""
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
