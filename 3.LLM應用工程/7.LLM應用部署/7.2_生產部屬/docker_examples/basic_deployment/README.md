# Docker 基礎部署範例

這是一個生產就緒的 LLM API Docker 部署範例，展示了容器化 FastAPI 應用的最佳實踐。

## 📁 文件說明

```
basic_deployment/
├── Dockerfile              # 多階段構建 Dockerfile
├── docker-compose.yml      # Docker Compose 配置
├── app.py                  # FastAPI 應用主程式
├── requirements.txt        # Python 依賴
├── .env.example           # 環境變數範例
└── README.md              # 本文件
```

## 🚀 快速開始

### 1. 配置環境變數

```bash
# 複製環境變數範例
cp .env.example .env

# 編輯 .env 文件，填入你的 API 金鑰
nano .env
```

### 2. 構建並啟動服務

```bash
# 構建鏡像並啟動容器
docker-compose up --build

# 或者在後台運行
docker-compose up -d
```

### 3. 測試 API

訪問 API 文檔：
```
http://localhost:8000/docs
```

使用 curl 測試：
```bash
# 健康檢查
curl http://localhost:8000/health

# 聊天請求（需要 API Key）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-123" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### 4. 查看日誌

```bash
# 查看實時日誌
docker-compose logs -f

# 查看特定服務的日誌
docker-compose logs -f llm-api
```

### 5. 停止服務

```bash
# 停止但保留容器
docker-compose stop

# 停止並刪除容器
docker-compose down

# 停止並刪除容器及鏡像
docker-compose down --rmi all
```

## 🔧 配置說明

### 環境變數

| 變數名 | 說明 | 預設值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 金鑰 | - |
| `ANTHROPIC_API_KEY` | Anthropic API 金鑰 | - |
| `API_KEYS` | 服務 API 金鑰（逗號分隔）| test-key-123 |
| `LOG_LEVEL` | 日誌級別 | INFO |
| `MAX_WORKERS` | 工作進程數 | 4 |

### Docker Compose 配置

```yaml
services:
  llm-api:
    build: .
    ports:
      - "8000:8000"  # 映射端口
    environment:     # 環境變數
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:         # 掛載卷
      - ./logs:/app/logs
    restart: unless-stopped  # 自動重啟
```

## 📊 API 端點

### GET /

根端點，返回 API 資訊

### GET /health

健康檢查端點

**回應：**
```json
{
  "status": "healthy",
  "timestamp": "2024-11-18T10:00:00",
  "checks": {
    "api": "healthy",
    "openai_configured": true,
    "claude_configured": true
  }
}
```

### POST /chat

聊天端點

**請求：**
```json
{
  "messages": [
    {"role": "user", "content": "你好！"}
  ],
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**回應：**
```json
{
  "response": "你好！很高興見到你！",
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  },
  "timestamp": "2024-11-18T10:00:00"
}
```

### GET /models

列出可用模型

**回應：**
```json
{
  "models": [
    {
      "provider": "OpenAI",
      "model": "gpt-3.5-turbo",
      "description": "Fast and cost-effective"
    }
  ],
  "total": 1
}
```

## 🔐 認證

所有需要認證的端點都需要在 header 中提供 API 金鑰：

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/chat
```

## 🐛 故障排除

### 問題 1：容器無法啟動

**檢查：**
```bash
# 查看容器日誌
docker-compose logs llm-api

# 檢查容器狀態
docker-compose ps
```

### 問題 2：API 金鑰錯誤

**解決方案：**
1. 檢查 `.env` 文件是否存在
2. 確認 API 金鑰格式正確
3. 重啟容器：`docker-compose restart`

### 問題 3：端口被佔用

**解決方案：**
```bash
# 修改 docker-compose.yml 中的端口映射
ports:
  - "8080:8000"  # 改用 8080 端口
```

## 📈 生產環境優化

### 1. 使用 Gunicorn + Uvicorn

修改 Dockerfile 的 CMD：
```dockerfile
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 2. 添加 Nginx 反向代理

```bash
# 啟用 nginx 服務
docker-compose up -d nginx
```

### 3. 配置資源限制

在 `docker-compose.yml` 中添加：
```yaml
services:
  llm-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 4. 添加監控

安裝 Prometheus 和 Grafana：
```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 🎯 下一步

- [ ] 添加速率限制
- [ ] 實現請求快取
- [ ] 添加 Prometheus 指標
- [ ] 配置 HTTPS
- [ ] 實現負載均衡

## 📚 參考資源

- [Docker 最佳實踐](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI 部署](https://fastapi.tiangolo.com/deployment/)
- [Docker Compose 文檔](https://docs.docker.com/compose/)
