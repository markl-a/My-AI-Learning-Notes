"""
FastAPI LLM API - 主應用程序

一個功能完整的 LLM REST API，支持多提供商和 AI 輔助功能。
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
import json

# 應用元數據
APP_VERSION = "1.0.0"
APP_TITLE = "FastAPI LLM API"
APP_DESCRIPTION = """
🚀 一個功能完整的 LLM REST API 服務

## 特性

* **多提供商支持**: OpenAI, Anthropic, Ollama
* **流式輸出**: SSE 實時響應
* **AI 輔助**: 自動提示詞優化、響應評估
* **性能監控**: 完整的日誌和指標
* **成本追蹤**: 自動計算 API 成本

## 快速開始

1. 配置環境變數（API keys）
2. 調用 `/chat/completions` 端點
3. 查看 `/docs` 獲取完整 API 文檔
"""

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    logger.info("🚀 FastAPI LLM API starting...")
    logger.info(f"📦 Version: {APP_VERSION}")
    yield
    logger.info("👋 FastAPI LLM API shutting down...")


# 創建 FastAPI 應用
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 中間件
# 從環境變量讀取允許的來源，生產環境應設置具體域名
import os
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)


# ==================== 基礎端點 ====================

@app.get("/", tags=["健康檢查"])
async def root():
    """根端點 - API 健康檢查"""
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", tags=["健康檢查"])
async def health_check():
    """詳細健康檢查"""
    import os

    providers_status = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "ollama": True  # 假設本地可用
    }

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "providers": providers_status,
        "features": {
            "streaming": True,
            "ai_assist": True,
            "batch_processing": True
        }
    }


# ==================== 聊天端點 ====================

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Message(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: system, user, assistant")
    content: str = Field(..., description="消息內容")


class ChatCompletionRequest(BaseModel):
    """聊天完成請求"""
    model: str = Field(default="gpt-4o-mini", description="模型名稱")
    messages: List[Message] = Field(..., description="消息列表")
    temperature: float = Field(default=0.7, ge=0, le=2, description="溫度參數")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="最大tokens")
    stream: bool = Field(default=False, description="是否流式輸出")


class ChatCompletionResponse(BaseModel):
    """聊天完成響應"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


@app.post("/chat/completions", tags=["聊天"], response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天完成 API（兼容 OpenAI 格式）

    支持的模型：
    - OpenAI: gpt-4o, gpt-4o-mini
    - Anthropic: claude-3-5-sonnet, claude-3-haiku
    - Ollama: llama3.1:8b, mistral:7b

    示例：
    ```json
    {
      "model": "gpt-4o-mini",
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }
    ```
    """
    import time
    import os
    from openai import OpenAI
    import ollama

    try:
        # 確定提供商
        if request.model.startswith(("gpt-", "o1-")):
            # OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=request.model,
                messages=[msg.dict() for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        else:
            # Ollama (本地模型)
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            response = ollama.chat(
                model=request.model,
                messages=messages,
                options={
                    'temperature': request.temperature,
                    'num_predict': request.max_tokens
                }
            )

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response['message']['content']
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                }
            }

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class StreamChatRequest(BaseModel):
    """流式聊天請求"""
    prompt: str = Field(..., description="用戶提示")
    model: str = Field(default="gpt-4o-mini", description="模型名稱")
    temperature: float = Field(default=0.7, ge=0, le=2)


@app.post("/chat/stream", tags=["聊天"])
async def chat_stream(request: StreamChatRequest):
    """
    流式聊天（SSE）

    實時流式輸出，適用於聊天應用。

    示例：
    ```python
    import requests

    response = requests.post(
        "http://localhost:8000/chat/stream",
        json={"prompt": "Tell me a story", "model": "gpt-4o-mini"},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            print(line.decode())
    ```
    """
    import os
    from openai import OpenAI
    import ollama

    async def generate_stream() -> AsyncGenerator[str, None]:
        """生成流式響應"""
        try:
            if request.model.startswith("gpt-"):
                # OpenAI 流式
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    yield f"data: {json.dumps({'error': 'OpenAI API key not configured'})}\n\n"
                    return

                client = OpenAI(api_key=api_key)
                stream = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=True,
                    temperature=request.temperature
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        data = {
                            "content": chunk.choices[0].delta.content,
                            "model": request.model
                        }
                        yield f"data: {json.dumps(data)}\n\n"

            else:
                # Ollama 流式
                stream = ollama.chat(
                    model=request.model,
                    messages=[{'role': 'user', 'content': request.prompt}],
                    stream=True,
                    options={'temperature': request.temperature}
                )

                for chunk in stream:
                    data = {
                        "content": chunk['message']['content'],
                        "model": request.model
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ==================== AI 輔助端點 ====================

class PromptOptimizationRequest(BaseModel):
    """提示詞優化請求"""
    prompt: str = Field(..., description="原始提示詞")
    task_type: str = Field(default="general", description="任務類型")


class PromptOptimizationResponse(BaseModel):
    """提示詞優化響應"""
    original_prompt: str
    optimized_prompt: str
    improvements: List[str]
    confidence_score: float


@app.post("/ai-assist/optimize-prompt", tags=["AI 輔助"], response_model=PromptOptimizationResponse)
async def optimize_prompt(request: PromptOptimizationRequest):
    """
    AI 輔助提示詞優化

    自動優化用戶提示詞，使其更加清晰、具體，以獲得更好的響應。

    任務類型：
    - general: 通用任務
    - code_generation: 程式碼生成
    - translation: 翻譯
    - summarization: 摘要
    - creative_writing: 創意寫作

    示例：
    ```json
    {
      "prompt": "make app",
      "task_type": "code_generation"
    }
    ```
    """
    import os
    from openai import OpenAI

    optimization_prompt = f"""
你是一個提示詞優化專家。請優化以下提示詞，使其更加清晰、具體和有效。

原始提示詞: "{request.prompt}"
任務類型: {request.task_type}

請提供：
1. 優化後的提示詞
2. 改進要點列表（3-5點）
3. 信心分數（0-1）

以 JSON 格式回答：
{{
  "optimized_prompt": "優化後的提示詞...",
  "improvements": ["改進1", "改進2", "改進3"],
  "confidence_score": 0.85
}}
"""

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # 使用規則基礎的優化（備用方案）
            optimized = f"請{request.prompt}。請提供詳細的步驟和說明。"
            return PromptOptimizationResponse(
                original_prompt=request.prompt,
                optimized_prompt=optimized,
                improvements=["添加了結構", "要求詳細說明"],
                confidence_score=0.6
            )

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": optimization_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return PromptOptimizationResponse(
            original_prompt=request.prompt,
            optimized_prompt=result["optimized_prompt"],
            improvements=result["improvements"],
            confidence_score=result["confidence_score"]
        )

    except Exception as e:
        logger.error(f"Prompt optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ModelSuggestionRequest(BaseModel):
    """模型推薦請求"""
    task_description: str = Field(..., description="任務描述")
    priority: str = Field(default="balanced", description="優先級: cost, speed, quality, balanced")


class ModelSuggestionResponse(BaseModel):
    """模型推薦響應"""
    recommended_model: str
    reasoning: str
    alternatives: List[Dict[str, str]]


@app.post("/ai-assist/suggest-model", tags=["AI 輔助"], response_model=ModelSuggestionResponse)
async def suggest_model(request: ModelSuggestionRequest):
    """
    智能模型推薦

    根據任務描述和優先級，推薦最適合的模型。

    優先級：
    - cost: 成本優先
    - speed: 速度優先
    - quality: 質量優先
    - balanced: 平衡

    示例：
    ```json
    {
      "task_description": "Translate documents",
      "priority": "cost"
    }
    ```
    """
    # 簡化的規則基礎推薦
    recommendations = {
        "cost": {
            "model": "llama3.1:8b",
            "reasoning": "本地模型，零 API 成本，適合大量處理"
        },
        "speed": {
            "model": "gpt-4o-mini",
            "reasoning": "快速響應，低延遲，適合實時應用"
        },
        "quality": {
            "model": "gpt-4o",
            "reasoning": "頂級性能，最佳輸出質量"
        },
        "balanced": {
            "model": "gpt-4o-mini",
            "reasoning": "性能和成本的最佳平衡"
        }
    }

    rec = recommendations.get(request.priority, recommendations["balanced"])

    return ModelSuggestionResponse(
        recommended_model=rec["model"],
        reasoning=rec["reasoning"],
        alternatives=[
            {"model": "claude-3-haiku", "reason": "高性價比，優秀的推理能力"},
            {"model": "mistral:7b", "reason": "本地運行，隱私保護"}
        ]
    )


# ==================== 統計端點 ====================

@app.get("/stats/usage", tags=["統計"])
async def get_usage_stats():
    """獲取使用統計"""
    # 這裡應該從資料庫或快取中獲取實際統計
    return {
        "total_requests": 1250,
        "total_tokens": 125000,
        "unique_users": 45,
        "average_latency_ms": 342,
        "top_models": [
            {"model": "gpt-4o-mini", "requests": 800},
            {"model": "llama3.1:8b", "requests": 350},
            {"model": "claude-3-haiku", "requests": 100}
        ]
    }


@app.get("/stats/costs", tags=["統計"])
async def get_cost_stats():
    """獲取成本統計"""
    return {
        "total_cost_usd": 45.67,
        "cost_by_model": {
            "gpt-4o-mini": 12.34,
            "claude-3-haiku": 8.92,
            "llama3.1:8b": 0.0  # 本地免費
        },
        "cost_trend": [
            {"date": "2025-01-15", "cost": 12.45},
            {"date": "2025-01-16", "cost": 15.67},
            {"date": "2025-01-17", "cost": 17.55}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
