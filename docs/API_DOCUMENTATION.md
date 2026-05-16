# API 文檔索引

> **最後更新**: 2025-01
> **版本**: 1.0.0

本文檔提供 My-AI-Learning-Notes 專案中所有 API 服務的統一入口和說明。

---

## 📋 目錄

- [概述](#概述)
- [可用 API 服務](#可用-api-服務)
- [認證方式](#認證方式)
- [通用響應格式](#通用響應格式)
- [錯誤碼](#錯誤碼)
- [速率限制](#速率限制)
- [API 詳細文檔](#api-詳細文檔)

---

## 概述

本專案包含多個 FastAPI 驅動的 REST API 服務，提供 LLM、RAG、文檔分析和程式碼審查等功能。

### 技術棧

- **框架**: FastAPI 0.115+
- **伺服器**: Uvicorn (ASGI)
- **認證**: HTTPBearer + API Key
- **文檔**: OpenAPI 3.0 (Swagger UI + ReDoc)

---

## 可用 API 服務

| 服務名稱 | 端口 | 說明 | Swagger 文檔 |
|---------|------|------|-------------|
| **RAG ChatBot** | 8000 | RAG 增強的聊天機器人 | `/docs` |
| **FastAPI LLM API** | 8000 | 多提供商 LLM 服務 | `/docs` |
| **AI Document Analyzer** | 8001 | 文檔分析服務 | `/docs` |
| **AI Code Review** | 8002 | 程式碼審查服務 | `/docs` |

---

## 認證方式

### API Key 認證

大多數端點需要 API Key 認證：

```bash
# 使用 Authorization Header
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/api/chat

# 或使用 X-API-Key Header
curl -H "X-API-Key: YOUR_API_KEY" \
     http://localhost:8000/api/chat
```

### 無需認證的端點

以下端點不需要認證：
- `GET /` - 根路徑
- `GET /health` - 健康檢查
- `GET /docs` - Swagger 文檔
- `GET /redoc` - ReDoc 文檔

---

## 通用響應格式

### 成功響應

```json
{
  "status": "success",
  "data": {
    // 響應資料
  },
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "duration": 0.123
  }
}
```

### 錯誤響應

```json
{
  "error": "Error message",
  "status_code": 400,
  "timestamp": "2025-01-15T10:30:00Z",
  "details": {
    // 可選的詳細錯誤資訊
  }
}
```

---

## 錯誤碼

| 狀態碼 | 說明 | 常見原因 |
|-------|------|---------|
| `200` | 成功 | 請求成功處理 |
| `400` | 錯誤請求 | 參數錯誤、格式不正確 |
| `401` | 未授權 | API Key 無效或缺失 |
| `403` | 禁止訪問 | 無權限訪問該資源 |
| `404` | 未找到 | 資源不存在 |
| `429` | 請求過多 | 超過速率限制 |
| `500` | 伺服器錯誤 | 內部錯誤 |
| `503` | 服務不可用 | LLM 提供商不可用 |

---

## 速率限制

所有 API 端點都有速率限制：

| 類型 | 限制 | 重置時間 |
|------|------|---------|
| 突發 | 10 請求/秒 | 1 秒 |
| 分鐘 | 60 請求/分鐘 | 60 秒 |
| 小時 | 1000 請求/小時 | 3600 秒 |

### 響應頭

```
X-RateLimit-Limit-Minute: 60
X-RateLimit-Remaining-Minute: 55
X-RateLimit-Limit-Hour: 1000
X-RateLimit-Remaining-Hour: 990
Retry-After: 30  (當被限制時)
```

---

## API 詳細文檔

### 1. RAG ChatBot API

**位置**: `5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot/`

#### 端點列表

| 方法 | 端點 | 說明 |
|------|------|------|
| `POST` | `/api/chat` | 聊天接口 |
| `POST` | `/api/chat/stream` | 流式聊天 |
| `POST` | `/api/documents` | 添加文檔 |
| `POST` | `/api/documents/upload` | 上傳文檔 |
| `GET` | `/api/documents` | 列出文檔 |
| `DELETE` | `/api/documents/{id}` | 刪除文檔 |
| `GET` | `/api/conversations/{id}` | 獲取對話 |
| `DELETE` | `/api/conversations/{id}` | 刪除對話 |
| `GET` | `/api/health` | 健康檢查 |
| `GET` | `/api/stats` | 統計資訊 |

#### 聊天請求示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "什麼是 RAG?",
    "use_rag": true,
    "top_k": 3
  }'
```

#### 響應示例

```json
{
  "response": "RAG (Retrieval-Augmented Generation) 是一種...",
  "conversation_id": "conv_123",
  "sources": [
    {"content": "...", "metadata": {"source": "doc1.pdf"}}
  ],
  "metadata": {
    "tokens_used": 150,
    "model": "gpt-4o-mini"
  }
}
```

---

### 2. FastAPI LLM API

**位置**: `3.LLM應用工程/1.LLM 部署/projects/fastapi-llm-api/`

#### 端點列表

| 方法 | 端點 | 說明 |
|------|------|------|
| `POST` | `/chat/completions` | 聊天完成 (OpenAI 兼容) |
| `POST` | `/chat/stream` | 流式聊天 |
| `POST` | `/ai-assist/optimize-prompt` | 提示詞優化 |
| `POST` | `/ai-assist/suggest-model` | 模型推薦 |
| `GET` | `/stats/usage` | 使用統計 |
| `GET` | `/stats/costs` | 成本統計 |

#### 聊天完成請求

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
  }'
```

---

### 3. 多提供商 LLM 服務

**位置**: `3.LLM應用工程/2.LLM as API/examples/frontend_integration/fastapi_backend/`

#### 支持的提供商

| 提供商 | 模型 | 環境變量 |
|-------|------|---------|
| OpenAI | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-3-5-sonnet | `ANTHROPIC_API_KEY` |
| Google | gemini-1.5-pro | `GOOGLE_API_KEY` |

#### 請求示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": "openai",
    "model": "gpt-4o-mini"
  }'
```

---

## 環境配置

### 必需的環境變量

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# 應用配置
API_KEY=your-api-key-for-auth
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# 可選配置
LOG_LEVEL=INFO
API_PORT=8000
```

---

## 客戶端 SDK 示例

### Python

```python
import requests

class RAGChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def chat(self, message: str, use_rag: bool = True) -> dict:
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json={"message": message, "use_rag": use_rag}
        )
        response.raise_for_status()
        return response.json()

# 使用
client = RAGChatClient()
result = client.chat("什麼是 LangChain?")
print(result["response"])
```

### JavaScript/TypeScript

```typescript
import axios from 'axios';

class RAGChatClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async chat(message: string, useRag: boolean = true) {
    const response = await axios.post(`${this.baseUrl}/api/chat`, {
      message,
      use_rag: useRag
    });
    return response.data;
  }
}

// 使用
const client = new RAGChatClient();
const result = await client.chat('什麼是 LangChain?');
console.log(result.response);
```

---

## 開發和測試

### 本地運行

```bash
# 安裝依賴
pip install -r requirements.txt

# 運行服務
uvicorn main:app --reload --port 8000

# 訪問文檔
open http://localhost:8000/docs
```

### 運行測試

```bash
# 運行所有測試
pytest tests/

# 運行 API 測試
pytest tests/test_api.py -v

# 生成覆蓋率報告
pytest --cov=. --cov-report=html
```

---

## 相關文檔

- [QUICKSTART.md](../QUICKSTART.md) - 快速入門指南
- [DEPLOYMENT.md](../DEPLOYMENT.md) - 部署指南
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - 故障排除指南
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 貢獻指南

---

## 更新日誌

### v1.0.0 (2025-01)
- 初始版本
- 統一 API 文檔入口
- 添加認證和速率限制說明
- 添加客戶端 SDK 示例
