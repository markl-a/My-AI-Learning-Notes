# 7.2 生產部署：可擴展的企業級方案

## 概述

從原型到生產環境需要考慮更多因素：**可擴展性、可靠性、安全性、可觀測性**。本章節介紹如何構建生產級的 LLM 部署方案。

## 🎯 學習目標

- 理解生產環境的部署需求和挑戰
- 掌握 Docker 容器化最佳實踐
- 學會使用 vLLM 進行高效能推論
- 構建基於 FastAPI 的 RESTful API 服務
- 實現監控、日誌和健康檢查

## 📊 生產環境 vs 原型環境

| 維度 | 原型環境 | 生產環境 |
|------|----------|----------|
| **可用性** | 偶爾宕機可接受 | 99.9%+ 高可用性 |
| **性能** | 秒級延遲 | 毫秒級延遲 |
| **擴展性** | 單機運行 | 自動水平擴展 |
| **安全性** | 基礎認證 | 完整的安全防護 |
| **監控** | 基礎日誌 | 完整的可觀測性 |
| **成本** | 最小化 | 性能優先，成本可控 |

## 📁 目錄結構

```
7.2_生產部屬/
├── README.md                          # 本文件
├── docker_examples/                   # Docker 部署範例
│   ├── basic_deployment/              # 基礎 Docker 部署
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── nginx_loadbalancer/            # Nginx 負載均衡
│   │   ├── docker-compose.yml
│   │   ├── nginx.conf
│   │   └── README.md
│   └── multi_stage_build/             # 多階段構建優化
│       ├── Dockerfile
│       └── README.md
├── fastapi_examples/                  # FastAPI 服務範例
│   ├── basic_api/                     # 基礎 API
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── with_auth/                     # 帶認證的 API
│   │   ├── main.py
│   │   ├── auth.py
│   │   └── README.md
│   └── rate_limiting/                 # 速率限制
│       ├── main.py
│       └── README.md
└── vllm_examples/                     # vLLM 高效能推論
    ├── basic_server/                  # 基礎 vLLM 服務
    │   ├── server.py
    │   ├── client.py
    │   └── README.md
    └── optimized_config/              # 優化配置
        ├── config.yaml
        └── README.md
```

---

## 🐳 Docker 容器化部署

### 為什麼要使用 Docker？

1. **環境一致性**：開發、測試、生產環境完全相同
2. **隔離性**：每個服務獨立運行，互不干擾
3. **可移植性**：一次構建，處處運行
4. **資源控制**：精確控制 CPU、記憶體等資源
5. **快速部署**：秒級啟動新實例

### Docker 基礎概念

#### Dockerfile：定義鏡像構建過程

```dockerfile
# 基礎範例
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml：編排多個容器

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

### Docker 最佳實踐

#### 1. 多階段構建（減小鏡像體積）

```dockerfile
# 構建階段
FROM python:3.10 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 運行階段
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

#### 2. 利用層快取（加速構建）

```dockerfile
# ❌ 錯誤：每次代碼改變都重裝依賴
FROM python:3.10-slim
COPY . .
RUN pip install -r requirements.txt

# ✅ 正確：只在依賴改變時重裝
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

#### 3. 使用 .dockerignore

```
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
.env
.venv
venv/
*.md
tests/
.pytest_cache
```

---

## ⚡ vLLM：高效能 LLM 推論引擎

### 什麼是 vLLM？

vLLM 是一個快速且易用的 LLM 推論和服務庫，具有以下特點：

- **PagedAttention**：高效的記憶體管理，比 HuggingFace 快 24 倍
- **連續批處理**：自動批處理請求，提高吞吐量
- **GPU 優化**：充分利用 GPU 資源
- **兼容 OpenAI API**：無縫替換 OpenAI API

### vLLM vs 其他方案

| 方案 | 吞吐量 | 延遲 | 易用性 | GPU 利用率 |
|------|--------|------|--------|------------|
| **HuggingFace Transformers** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **vLLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TensorRT-LLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Text Generation Inference** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 基礎使用

#### 安裝

```bash
pip install vllm
```

#### Python API

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="facebook/opt-1.3b")

# 設置生成參數
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 生成
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated text: {output.outputs[0].text}")
```

#### OpenAI 兼容服務器

```bash
# 啟動 vLLM 服務器
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-1.3b \
    --port 8000
```

### vLLM 優化技巧

#### 1. 調整 GPU 記憶體利用率

```python
llm = LLM(
    model="facebook/opt-1.3b",
    gpu_memory_utilization=0.9,  # 使用 90% GPU 記憶體
)
```

#### 2. 張量並行（多 GPU）

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 使用 4 個 GPU
)
```

#### 3. 量化（減少記憶體）

```bash
# 使用 AWQ 量化
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-AWQ \
    --quantization awq
```

---

## 🚀 FastAPI：現代 Python Web 框架

### 為什麼選擇 FastAPI？

1. **高性能**：基於 Starlette 和 Pydantic，性能媲美 Node.js 和 Go
2. **自動文檔**：自動生成 OpenAPI (Swagger) 文檔
3. **類型安全**：基於 Python 類型提示
4. **異步支持**：原生支持 async/await
5. **易於測試**：內建測試客戶端

### 基礎 LLM API 範例

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

app = FastAPI(title="LLM API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = openai.ChatCompletion.create(
            model=request.model,
            messages=[{"role": "user", "content": request.message}]
        )
        return ChatResponse(response=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 進階功能

#### 1. API 金鑰認證

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Security(verify_api_key)):
    # 處理請求
    pass
```

#### 2. 速率限制

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, chat_request: ChatRequest):
    # 每分鐘最多 10 次請求
    pass
```

#### 3. 背景任務

```python
from fastapi import BackgroundTasks

def log_request(user_id: str, message: str):
    # 異步記錄請求
    with open("requests.log", "a") as f:
        f.write(f"{user_id}: {message}\n")

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(log_request, "user123", request.message)
    # 處理請求
    pass
```

---

## 📊 監控和可觀測性

### 三大支柱

1. **日誌（Logs）**：記錄事件
2. **指標（Metrics）**：量化數據
3. **追蹤（Traces）**：請求鏈路

### 實現監控

#### 1. 結構化日誌

```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.info("Request processed", extra={
    "user_id": "user123",
    "latency_ms": 245,
    "model": "gpt-3.5-turbo"
})
```

#### 2. Prometheus 指標

```python
from prometheus_client import Counter, Histogram, generate_latest

# 定義指標
request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.post("/chat")
async def chat(request: ChatRequest):
    request_count.inc()
    with request_duration.time():
        # 處理請求
        pass

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

#### 3. 健康檢查

```python
@app.get("/health")
async def health():
    # 檢查依賴服務
    checks = {
        "api": True,
        "database": await check_database(),
        "cache": await check_cache()
    }

    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail=checks)
```

---

## 🔒 安全最佳實踐

### 1. 輸入驗證

```python
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    message: str

    @validator('message')
    def validate_message(cls, v):
        if len(v) > 4000:
            raise ValueError('Message too long')
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v
```

### 2. 防止提示注入

```python
def sanitize_input(user_input: str) -> str:
    # 移除潛在的提示注入
    dangerous_patterns = [
        "ignore previous instructions",
        "system:",
        "assistant:",
    ]

    cleaned = user_input.lower()
    for pattern in dangerous_patterns:
        if pattern in cleaned:
            raise ValueError("Suspicious input detected")

    return user_input
```

### 3. HTTPS 和 CORS

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# 強制 HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 🎯 部署策略對比

### 1. Serverless（AWS Lambda, Google Cloud Functions）

**優勢：**
- 自動擴展
- 按使用付費
- 零運維

**劣勢：**
- 冷啟動延遲
- GPU 支持有限
- 執行時間限制（15 分鐘）

**適合場景：** 低頻請求、CPU 推論、成本敏感

### 2. 容器化（Docker + Kubernetes）

**優勢：**
- 環境一致
- 易於擴展
- 靈活配置

**劣勢：**
- 需要運維知識
- 基礎設施成本

**適合場景：** 中高頻請求、GPU 推論、混合負載

### 3. 自建 GPU 叢集

**優勢：**
- 最高性能
- 完全控制
- 成本可預測（大規模）

**劣勢：**
- 初期成本高
- 運維複雜
- 需要專業團隊

**適合場景：** 大規模生產、低延遲需求、特殊硬體需求

### 4. 託管服務（AWS SageMaker, Vertex AI）

**優勢：**
- 開箱即用
- 自動擴展
- 整合監控

**劣勢：**
- 成本較高
- 靈活性有限
- 供應商鎖定

**適合場景：** 快速上線、企業級需求、希望減少運維

---

## 💰 成本優化策略

### 1. 請求批處理

```python
from collections import deque
import asyncio

class BatchProcessor:
    def __init__(self, batch_size=10, wait_time=0.1):
        self.queue = deque()
        self.batch_size = batch_size
        self.wait_time = wait_time

    async def add_request(self, request):
        self.queue.append(request)

        if len(self.queue) >= self.batch_size:
            return await self.process_batch()

        await asyncio.sleep(self.wait_time)
        if self.queue:
            return await self.process_batch()

    async def process_batch(self):
        batch = [self.queue.popleft() for _ in range(min(len(self.queue), self.batch_size))]
        # 批量處理
        return await llm.generate(batch)
```

### 2. 快取回應

```python
from functools import lru_cache
import hashlib

def cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()

@lru_cache(maxsize=1000)
def get_cached_response(prompt_hash: str, prompt: str):
    # 調用 LLM
    return llm.generate(prompt)

@app.post("/chat")
async def chat(request: ChatRequest):
    key = cache_key(request.message)
    return get_cached_response(key, request.message)
```

### 3. 模型量化

```python
# 使用 INT8 量化減少 GPU 記憶體
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    load_in_8bit=True,  # INT8 量化
    device_map="auto"
)
```

---

## 📚 實戰練習

### 練習 1：Docker 化 FastAPI 應用（1 小時）
1. 創建一個 FastAPI LLM API
2. 編寫 Dockerfile 和 docker-compose.yml
3. 添加健康檢查和日誌
4. 本地運行並測試

### 練習 2：添加認證和速率限制（1 小時）
1. 實現 API Key 認證
2. 添加速率限制（每分鐘 10 次）
3. 返回適當的錯誤訊息
4. 測試邊界情況

### 練習 3：部署到雲端（2 小時）
1. 註冊 AWS/GCP/Azure 帳號
2. 使用 EC2/Compute Engine 部署容器
3. 配置安全組和防火牆
4. 設置 HTTPS 證書

---

## 🔗 參考資源

### 官方文檔
- [Docker 文檔](https://docs.docker.com/)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)
- [vLLM 文檔](https://docs.vllm.ai/)
- [Kubernetes 文檔](https://kubernetes.io/docs/)

### 開源項目
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [Ray Serve](https://github.com/ray-project/ray)

### 學習資源
- [Docker 深入淺出](https://docker-curriculum.com/)
- [FastAPI 教程](https://fastapi.tiangolo.com/tutorial/)
- [Kubernetes 實戰](https://kubernetes.io/docs/tutorials/)

---

## 🎯 下一步

完成本章節後，你將能夠：
- ✅ 使用 Docker 容器化 LLM 應用
- ✅ 構建高性能的 FastAPI 服務
- ✅ 使用 vLLM 優化推論性能
- ✅ 實現監控、日誌和安全防護

**準備好了嗎？** 前往範例目錄開始實戰：
- [Docker 範例](./docker_examples/)
- [FastAPI 範例](./fastapi_examples/)
- [vLLM 範例](./vllm_examples/)

或者繼續學習 [7.3 邊緣部署](../7.3_邊緣部屬/README.md) 了解如何在資源受限環境運行 LLM。
