"""
FastAPI 基礎 LLM API 範例

這是一個最簡化的 FastAPI LLM API 範例，適合快速開始學習。

運行方式：
pip install -r requirements.txt
uvicorn main:app --reload

API 文檔: http://localhost:8000/docs

作者：AI Learning Notes
日期：2024-11
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 創建 FastAPI 應用
app = FastAPI(
    title="Simple LLM API",
    description="最簡單的 LLM API 範例",
    version="1.0.0"
)

# ===== 數據模型 =====
class Message(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-3.5-turbo"


class ChatResponse(BaseModel):
    response: str
    model: str


# ===== API 端點 =====

@app.get("/")
async def root():
    """根端點"""
    return {
        "message": "Welcome to Simple LLM API",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """健康檢查"""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    簡單的聊天端點

    **範例：**
    ```json
    {
        "message": "你好！",
        "model": "gpt-3.5-turbo"
    }
    ```
    """
    # 檢查 API 金鑰
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured. Please set it in .env file"
        )

    try:
        # 調用 OpenAI API
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "user", "content": request.message}
            ]
        )

        return ChatResponse(
            response=response.choices[0].message.content,
            model=request.model
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


# ===== 額外的實用端點 =====

@app.post("/summarize")
async def summarize(text: str):
    """文本摘要端點"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是一個專業的摘要助手。請用簡潔的語言總結用戶提供的文本。"
                },
                {
                    "role": "user",
                    "content": f"請總結以下文本：\n\n{text}"
                }
            ]
        )

        return {"summary": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate(text: str, target_language: str = "英文"):
    """翻譯端點"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"你是一個專業的翻譯助手。請將用戶提供的文本翻譯成{target_language}。"
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        return {
            "original": text,
            "translated": response.choices[0].message.content,
            "target_language": target_language
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
