# FastAPI 後端服務

生產級 LLM API 服務，支援多個提供商，包含完整的錯誤處理、監控、日誌記錄等功能。

## 🌟 功能特點

✅ **多提供商支援**
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic Claude (Claude 3.5 Sonnet, Claude 3 Opus)
- Google Gemini (Gemini 1.5 Pro, Gemini 1.5 Flash)

✅ **生產級功能**
- ✅ API Key 認證
- ✅ CORS 跨域支援
- ✅ 請求速率限制
- ✅ 錯誤處理和重試
- ✅ 串流回應支援
- ✅ Prometheus 指標監控
- ✅ 結構化日誌記錄
- ✅ 健康檢查端點
- ✅ 自動 API 文檔

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd "3.LLM應用工程/2.LLM as API"
pip install -r requirements.txt
```

### 2. 設定環境變數

建立 `.env` 文件：

```env
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# 服務設定
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-key
DEBUG=False

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501

# 日誌
LOG_LEVEL=INFO
```

### 3. 啟動服務

```bash
# 開發模式
python examples/frontend_integration/fastapi_backend/main.py

# 生產模式
uvicorn examples.frontend_integration.fastapi_backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

### 4. 訪問 API 文檔

服務啟動後，訪問以下地址：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康檢查: http://localhost:8000/health
- Prometheus 指標: http://localhost:8000/metrics

## 📚 API 端點

### 1. 健康檢查

```bash
curl http://localhost:8000/health
```

回應：
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "version": "1.0.0",
  "providers": {
    "openai": true,
    "anthropic": true,
    "gemini": true
  }
}
```

### 2. 聊天端點

**非串流模式：**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "stream": false
  }'
```

回應：
```json
{
  "id": "chatcmpl-123",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "message": "你好！有什麼我可以幫助你的嗎？",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  },
  "created_at": "2025-01-15T10:30:00",
  "duration": 1.234
}
```

**串流模式：**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "寫一個 Python 函式"}
    ],
    "provider": "openai",
    "stream": true
  }'
```

回應（Server-Sent Events）：
```
data: {"content": "def "}
data: {"content": "example"}
data: {"content": "():"}
...
data: {"done": true, "duration": 2.345}
```

### 3. Prometheus 指標

```bash
curl http://localhost:8000/metrics
```

提供的指標：
- `llm_api_requests_total` - 總請求數（按提供商、模型、端點、狀態分組）
- `llm_api_request_duration_seconds` - 請求延遲（按提供商、模型、端點分組）
- `llm_api_tokens_total` - Token 使用量（按提供商、模型、類型分組）

## 🧪 測試

### 運行測試套件

```bash
# 自動測試
python examples/frontend_integration/fastapi_backend/test_client.py

# 互動式測試
python examples/frontend_integration/fastapi_backend/test_client.py --interactive
```

### Python 客戶端示例

```python
import requests

# 設定
BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 發送請求
response = requests.post(
    f"{BASE_URL}/api/chat",
    headers=headers,
    json={
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "provider": "openai",
        "temperature": 0.7
    }
)

result = response.json()
print(result["message"])
```

### JavaScript/TypeScript 客戶端示例

```javascript
const API_KEY = "your-secret-key";
const BASE_URL = "http://localhost:8000";

async function chat(message) {
  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      messages: [
        { role: "user", content: message }
      ],
      provider: "openai",
      temperature: 0.7
    })
  });

  const data = await response.json();
  return data.message;
}

// 使用
chat("Hello!").then(response => console.log(response));
```

## 📊 監控

### Prometheus 集成

在 `prometheus.yml` 中添加：

```yaml
scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana 儀表板

推薦指標面板：

1. **請求率**
   - Metric: `rate(llm_api_requests_total[5m])`
   - Group by: provider, model

2. **平均延遲**
   - Metric: `histogram_quantile(0.95, llm_api_request_duration_seconds)`
   - Group by: provider, model

3. **Token 使用**
   - Metric: `rate(llm_api_tokens_total[1h])`
   - Group by: provider, type

4. **錯誤率**
   - Metric: `rate(llm_api_requests_total{status="error"}[5m])`
   - Group by: provider

## 📝 日誌

日誌文件位置：`logs/api_{time}.log`

日誌格式：
```
2025-01-15 10:30:00 | INFO | POST /api/chat - Status: 200 - Duration: 1.234s
2025-01-15 10:30:01 | INFO | 收到聊天請求 - Provider: openai, Model: gpt-4o-mini
```

日誌級別：
- `DEBUG` - 詳細調試資訊
- `INFO` - 一般資訊
- `WARNING` - 警告訊息
- `ERROR` - 錯誤訊息

## 🔒 安全性

### API Key 認證

所有 `/api/*` 端點都需要 Bearer Token 認證：

```
Authorization: Bearer your-secret-key
```

### CORS 配置

在 `.env` 中設定允許的來源：

```env
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### 速率限制

建議使用 nginx 或 API Gateway 實作速率限制：

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20;
    proxy_pass http://localhost:8000;
}
```

## 🐳 Docker 部署

### 建立映像

```bash
docker build -t llm-api-service .
```

### 運行容器

```bash
docker run -d \
  --name llm-api \
  -p 8000:8000 \
  --env-file .env \
  llm-api-service
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

運行：
```bash
docker-compose up -d
```

## 🔧 進階配置

### 自定義錯誤處理

在 `main.py` 中添加自定義異常處理器：

```python
@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

### 添加新的提供商

1. 初始化客戶端：
```python
def init_clients():
    if os.getenv("NEW_PROVIDER_API_KEY"):
        clients['new_provider'] = NewProviderClient(...)
```

2. 實作處理函數：
```python
async def handle_new_provider_chat(request: ChatRequest):
    # 實作邏輯
    pass
```

3. 在端點中添加：
```python
elif request.provider == 'new_provider':
    result = await handle_new_provider_chat(request)
```

### 添加快取

使用 Redis 快取常見請求：

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    cache_key = hashlib.md5(
        json.dumps(request.dict()).encode()
    ).hexdigest()

    # 檢查快取
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 處理請求
    result = await process_request(request)

    # 儲存到快取
    redis_client.setex(cache_key, 3600, json.dumps(result))

    return result
```

## 🐛 常見問題

### 問題：API 無法啟動

**解決方案：**
1. 檢查端口 8000 是否被占用
2. 確認所有環境變數已設定
3. 檢查日誌文件中的錯誤訊息

### 問題：API Key 認證失敗

**解決方案：**
1. 確認 Authorization header 格式正確
2. 檢查 API_KEY 環境變數
3. 確保使用 Bearer scheme

### 問題：串流回應不工作

**解決方案：**
1. 確認客戶端支援 Server-Sent Events
2. 檢查 nginx 是否禁用了緩衝
3. 使用正確的 Content-Type

## 📈 性能優化

### 1. 使用多個 Worker

```bash
uvicorn main:app --workers 4
```

### 2. 啟用 HTTP/2

```bash
uvicorn main:app --http h2
```

### 3. 使用 Gunicorn

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### 4. 連接池優化

```python
# 在客戶端初始化時設定連接池
client = AsyncOpenAI(
    max_retries=3,
    timeout=30.0,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100
        )
    )
)
```

## 📚 延伸閱讀

- [FastAPI 官方文檔](https://fastapi.tiangolo.com/)
- [Uvicorn 文檔](https://www.uvicorn.org/)
- [Prometheus 文檔](https://prometheus.io/docs/)
- [OpenAPI 規範](https://swagger.io/specification/)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

---

**最後更新：** 2025年1月
