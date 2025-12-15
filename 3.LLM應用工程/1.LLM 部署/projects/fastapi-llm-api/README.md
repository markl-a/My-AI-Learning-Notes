# FastAPI LLM REST API

一個功能完整的 LLM REST API 服務，基於 FastAPI 構建，支持多種 LLM 提供商和 AI 輔助功能。

## ✨ 特性

### 核心功能
- 🚀 **高性能**: 基於 FastAPI 的異步 API
- 🔌 **多提供商**: 支持 OpenAI、Anthropic、Ollama
- 💬 **流式輸出**: 實時 SSE (Server-Sent Events) 響應
- 🎯 **智能路由**: 根據任務類型自動選擇最佳模型
- 📊 **請求追蹤**: 完整的日誌和性能監控

### AI 輔助功能
- 🤖 **提示詞優化**: 自動優化用戶提示詞以獲得更好的響應
- 📈 **響應評分**: AI 評估響應質量並提供改進建議
- ⚡ **性能分析**: 實時性能監控和優化建議
- 💰 **成本追蹤**: 自動計算和追蹤 API 調用成本
- 🔄 **自動重試**: 智能錯誤處理和重試機制

### 安全功能
- 🔐 **API Key 認證**: JWT token 認證
- 🛡️ **速率限制**: 防止濫用
- 📝 **輸入驗證**: Pydantic 數據驗證
- 🔒 **CORS 配置**: 靈活的跨域設置

## 📋 前置需求

```bash
Python 3.9+
pip install -r requirements.txt
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd fastapi-llm-api
pip install -r requirements.txt
```

### 2. 配置環境變數

```bash
cp .env.example .env
# 編輯 .env 文件，填入你的 API keys
```

### 3. 啟動服務

```bash
# 開發模式（自動重載）
python run.py

# 或使用 uvicorn 直接啟動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 訪問 API 文檔

打開瀏覽器訪問：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 API 端點

### 基礎端點

#### `GET /`
健康檢查

#### `GET /health`
詳細健康狀態

### 聊天端點

#### `POST /chat/completions`
標準聊天完成 API（兼容 OpenAI 格式）

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

#### `POST /chat/stream`
流式聊天響應（SSE）

```json
{
  "prompt": "Tell me a story",
  "model": "gpt-4o-mini"
}
```

### AI 輔助端點

#### `POST /ai-assist/optimize-prompt`
優化提示詞

```json
{
  "prompt": "make app",
  "task_type": "code_generation"
}
```

響應：
```json
{
  "original_prompt": "make app",
  "optimized_prompt": "Create a web application with the following requirements...",
  "improvements": ["Added context", "Specified requirements"],
  "confidence_score": 0.85
}
```

#### `POST /ai-assist/evaluate-response`
評估響應質量

```json
{
  "prompt": "What is AI?",
  "response": "AI is artificial intelligence...",
  "criteria": ["accuracy", "clarity", "completeness"]
}
```

#### `POST /ai-assist/suggest-model`
智能模型推薦

```json
{
  "task_description": "Translate English to Chinese",
  "priority": "quality",
  "constraints": {
    "max_cost_per_1k": 0.01,
    "max_latency_ms": 5000
  }
}
```

### 工具端點

#### `POST /tools/batch-inference`
批量推理

```json
{
  "prompts": [
    "Question 1",
    "Question 2",
    "Question 3"
  ],
  "model": "gpt-4o-mini",
  "parallel": true
}
```

#### `POST /tools/compare-models`
模型比較

```json
{
  "prompt": "Explain machine learning",
  "models": ["gpt-4o-mini", "claude-3-haiku", "llama3.1:8b"]
}
```

### 統計端點

#### `GET /stats/usage`
使用統計

#### `GET /stats/costs`
成本統計

## 🔧 配置

### 環境變數

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# 服務配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 安全
API_SECRET_KEY=your-secret-key
ENABLE_AUTH=true

# 速率限制
RATE_LIMIT_PER_MINUTE=60

# AI 輔助
ENABLE_AI_ASSIST=true
AUTO_OPTIMIZE_PROMPTS=false
```

### 模型配置

編輯 `app/config/models.yaml` 來配置可用的模型。

## 📖 使用示例

### Python 客戶端

```python
import requests

# 基礎聊天
response = requests.post(
    "http://localhost:8000/chat/completions",
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json())

# 使用 AI 輔助優化提示詞
response = requests.post(
    "http://localhost:8000/ai-assist/optimize-prompt",
    json={
        "prompt": "write code for sorting",
        "task_type": "code_generation"
    }
)

optimized = response.json()
print(f"優化後: {optimized['optimized_prompt']}")

# 使用優化後的提示詞
response = requests.post(
    "http://localhost:8000/chat/completions",
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": optimized['optimized_prompt']}
        ]
    }
)

print(response.json())
```

### JavaScript/TypeScript

```typescript
// 流式響應
const response = await fetch('http://localhost:8000/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'Tell me a story',
    model: 'gpt-4o-mini'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log(data.content);
    }
  }
}
```

### cURL

```bash
# 基礎請求
curl -X POST "http://localhost:8000/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# AI 輔助
curl -X POST "http://localhost:8000/ai-assist/suggest-model" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Generate product descriptions",
    "priority": "cost"
  }'
```

## 🧪 測試

```bash
# 運行所有測試
pytest

# 運行特定測試
pytest tests/test_chat.py

# 生成覆蓋率報告
pytest --cov=app tests/
```

## 📊 性能

- **吞吐量**: ~1000 請求/秒（取決於後端 LLM）
- **延遲**:
  - P50: ~200ms (不含 LLM 處理時間)
  - P95: ~500ms
  - P99: ~1000ms
- **並發**: 支持數千並發連接

## 🐳 Docker 部署

```bash
# 構建鏡像
docker build -t fastapi-llm-api .

# 運行容器
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  fastapi-llm-api

# 使用 docker-compose
docker-compose up -d
```

## 📈 監控

服務提供了 Prometheus 指標端點：

```bash
curl http://localhost:8000/metrics
```

可以使用 Grafana 進行可視化監控。

## 🔐 安全最佳實踐

1. **永遠不要**在代碼中硬編碼 API keys
2. 使用環境變數或密鑰管理服務
3. 啟用 HTTPS（在生產環境）
4. 實施速率限制
5. 定期輪換 API keys
6. 監控異常使用模式

## 🤝 貢獻

歡迎貢獻！請遵循以下步驟：

1. Fork 本項目
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📝 許可證

MIT License

## 🙏 致謝

- FastAPI - 現代、快速的 Web 框架
- OpenAI, Anthropic - 強大的 LLM 提供商
- Ollama - 本地 LLM 運行時

## 📞 支持

- 📧 Email: support@example.com
- 💬 Discord: [加入我們的社群](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/markl-a/fastapi-llm-api/issues)
