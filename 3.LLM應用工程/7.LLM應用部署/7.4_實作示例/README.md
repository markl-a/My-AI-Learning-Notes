# 7.4 實作示例：端到端完整案例

## 概述

本章節提供兩個完整的端到端實作案例，從原型開發到生產部署，展示 LLM 應用的完整生命週期。

## 📁 目錄結構

```
7.4_實作示例/
├── README.md                          # 本文件
├── gradio_chatbot/                    # 示例 1：Gradio 聊天機器人
│   ├── app.py                        # 主應用
│   ├── requirements.txt              # 依賴
│   ├── .env.example                  # 環境變數範例
│   ├── README.md                     # 部署指南
│   └── screenshots/                  # 截圖
└── production_api/                    # 示例 2：生產級 API
    ├── src/
    │   ├── main.py                   # FastAPI 應用
    │   ├── models.py                 # 資料模型
    │   ├── auth.py                   # 認證邏輯
    │   └── utils.py                  # 工具函數
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── tests/                         # 測試文件
    ├── .env.example
    ├── requirements.txt
    └── README.md
```

---

## 🤖 示例 1：智能客服聊天機器人 (Gradio)

### 功能特色

- ✅ 多模型支持（OpenAI GPT, Claude）
- ✅ 流式回應顯示
- ✅ 對話歷史管理
- ✅ AI 輔助功能（摘要、翻譯、情感分析）
- ✅ 系統提示詞自定義
- ✅ 一鍵部署到 Hugging Face Spaces

### 技術棧

- **框架**: Gradio 4.0
- **LLM API**: OpenAI, Anthropic
- **部署**: Hugging Face Spaces（免費）

### 快速開始

```bash
# 1. 克隆或複製程式碼到本地
cd gradio_chatbot

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 配置環境變數
cp .env.example .env
# 編輯 .env，填入 API 金鑰

# 4. 運行應用
python app.py

# 5. 訪問 http://localhost:7860
```

### 部署到 Hugging Face Spaces

```bash
# 1. 建立 Space
# 訪問 https://huggingface.co/spaces
# 點擊 "Create new Space"，選擇 Gradio SDK

# 2. 克隆 Space 倉庫
git clone https://huggingface.co/spaces/你的用戶名/chatbot
cd chatbot

# 3. 複製文件
cp ../gradio_chatbot/* ./

# 4. 提交並推送
git add .
git commit -m "Add chatbot"
git push

# 5. 在 Space 設置中添加 Secrets
# OPENAI_API_KEY
# ANTHROPIC_API_KEY
```

### 核心功能展示

#### 1. 多輪對話

```python
# 自動管理對話歷史
def chat(message, history):
    # history 格式: [(user_msg, bot_msg), ...]
    response = get_llm_response(message, history)
    return response
```

#### 2. AI 輔助功能

```python
# 文字摘要
def summarize(text):
    prompt = f"請總結以下文字：\n\n{text}"
    return call_llm(prompt)

# 翻譯
def translate(text, target_lang):
    prompt = f"將以下文字翻譯成{target_lang}：\n\n{text}"
    return call_llm(prompt)

# 情感分析
def analyze_sentiment(text):
    prompt = f"分析以下文字的情感（正面/負面/中性）：\n\n{text}"
    return call_llm(prompt)
```

#### 3. 流式回應

```python
def chat_stream(message, history):
    """流式生成回應"""
    full_response = ""
    for chunk in llm_stream(message, history):
        full_response += chunk
        yield full_response
```

### 進階功能

- **對話歷史導出**：下載為 JSON 或 CSV
- **自定義角色**：預設多種 AI 角色（程式設計師、翻譯官等）
- **多語言支持**：UI 支持中英文切換
- **主題切換**：亮色/暗色主題

---

## 🏭 示例 2：生產級 LLM API 服務

### 功能特色

- ✅ RESTful API 設計
- ✅ JWT 認證 + API Key 雙重認證
- ✅ 速率限制（防濫用）
- ✅ 請求批處理（提高吞吐量）
- ✅ 完整的監控和日誌
- ✅ Docker 容器化部署
- ✅ 健康檢查和自動重啟
- ✅ API 文檔自動生成

### 技術棧

- **框架**: FastAPI
- **認證**: JWT + API Key
- **速率限制**: slowapi
- **容器化**: Docker + Docker Compose
- **監控**: Prometheus + Grafana
- **日誌**: Python logging + ELK

### 架構圖

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Nginx     │─────▶│  FastAPI    │─────▶│ OpenAI/     │
│ (反向代理)  │      │   服務      │      │ Claude API  │
└─────────────┘      └─────────────┘      └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  Prometheus │
                     │  (監控)     │
                     └─────────────┘
```

### 快速開始

#### 1. 本地開發

```bash
cd production_api

# 安裝依賴
pip install -r requirements.txt

# 配置環境變數
cp .env.example .env

# 運行開發伺服器
uvicorn src.main:app --reload

# 訪問 API 文檔
open http://localhost:8000/docs
```

#### 2. Docker 部署

```bash
# 構建並啟動
cd production_api/docker
docker-compose up --build -d

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down
```

### API 端點

#### 認證

```bash
# 獲取 JWT Token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "secret"
  }'

# 回應
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

#### 聊天

```bash
# 使用 JWT Token
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1..." \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "model": "gpt-3.5-turbo"
  }'

# 使用 API Key
curl -X POST http://localhost:8000/api/v1/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好！"}
    ]
  }'
```

#### 批量處理

```bash
# 批量請求
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1..." \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"messages": [{"role": "user", "content": "問題1"}]},
      {"messages": [{"role": "user", "content": "問題2"}]},
      {"messages": [{"role": "user", "content": "問題3"}]}
    ]
  }'
```

#### 監控指標

```bash
# Prometheus 指標
curl http://localhost:8000/metrics

# 輸出示例
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/chat"} 1234

# HELP api_request_duration_seconds Request duration
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{le="0.1"} 500
api_request_duration_seconds_bucket{le="0.5"} 800
```

### 核心功能實現

#### 1. JWT 認證

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

#### 2. 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # 每分鐘 10 次
async def chat(request: Request, chat_request: ChatRequest):
    # 處理請求
    pass
```

#### 3. 請求批處理

```python
import asyncio

async def process_batch(requests: List[ChatRequest]):
    """並行處理多個請求"""
    tasks = [process_single_request(req) for req in requests]
    results = await asyncio.gather(*tasks)
    return results
```

#### 4. 監控指標

```python
from prometheus_client import Counter, Histogram

request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds'
)

@app.post("/chat")
async def chat(request: ChatRequest):
    request_count.labels(method='POST', endpoint='/chat').inc()

    with request_duration.time():
        # 處理請求
        result = await process_request(request)

    return result
```

### 部署到生產環境

#### AWS EC2 部署

```bash
# 1. 啟動 EC2 實例（建議 t3.medium 或更高）
# 2. SSH 連接到實例
ssh -i key.pem ubuntu@your-ec2-ip

# 3. 安裝 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. 克隆程式碼
git clone your-repo-url
cd production_api/docker

# 5. 配置環境變數
nano .env

# 6. 啟動服務
docker-compose up -d

# 7. 配置 Nginx 反向代理（可選）
sudo apt install nginx
sudo nano /etc/nginx/sites-available/llm-api
```

Nginx 配置：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### 配置 HTTPS

```bash
# 使用 Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 📊 性能優化

### 1. 請求快取

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_response(prompt_hash: str, prompt: str):
    return call_llm(prompt)

def chat(message: str):
    key = hashlib.md5(message.encode()).hexdigest()
    return get_cached_response(key, message)
```

### 2. 連接池

```python
from openai import OpenAI

# 使用連接池
client = OpenAI(
    api_key=API_KEY,
    max_retries=3,
    timeout=30
)
```

### 3. 異步處理

```python
import asyncio

async def parallel_requests(requests):
    """並行處理多個請求"""
    tasks = [process_request(req) for req in requests]
    return await asyncio.gather(*tasks)
```

---

## 🧪 測試

### 單元測試

```python
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post(
        "/api/v1/chat",
        headers={"X-API-Key": "test-key"},
        json={
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

### 壓力測試

```bash
# 使用 Apache Bench
ab -n 1000 -c 10 -H "X-API-Key: test-key" \
  -p request.json -T application/json \
  http://localhost:8000/api/v1/chat

# 使用 wrk
wrk -t12 -c400 -d30s \
  -H "X-API-Key: test-key" \
  --script post.lua \
  http://localhost:8000/api/v1/chat
```

---

## 📈 監控和告警

### Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana 儀表板

關鍵指標：
- **請求速率**：QPS (Queries Per Second)
- **延遲**：P50, P95, P99
- **錯誤率**：4xx, 5xx 錯誤比例
- **資源使用**：CPU, 記憶體, GPU

---

## 🎯 總結

### 示例 1 vs 示例 2

| 維度 | Gradio 聊天機器人 | 生產級 API |
|------|-------------------|-----------|
| **複雜度** | ⭐⭐ 簡單 | ⭐⭐⭐⭐ 複雜 |
| **部署時間** | 10 分鐘 | 2-4 小時 |
| **適用場景** | 原型展示、內部工具 | 生產服務、對外 API |
| **可擴展性** | 有限 | 高 |
| **維護成本** | 低 | 中高 |

### 學習路徑

1. **初學者**：從示例 1 開始，理解 LLM 基礎應用
2. **進階**：完成示例 2，掌握生產級部署
3. **專家**：優化性能，實現自動擴展和多區域部署

---

## 📚 參考資源

- [Gradio 文檔](https://www.gradio.app/docs/)
- [FastAPI 最佳實踐](https://fastapi.tiangolo.com/tutorial/)
- [Docker 部署指南](https://docs.docker.com/get-started/)
- [Prometheus 監控](https://prometheus.io/docs/)
- [AWS 部署教程](https://aws.amazon.com/getting-started/)

**下一步**：部署你的第一個 LLM 應用！
