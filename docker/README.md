# Docker 使用指南

## 📋 目錄

1. [快速開始](#快速開始)
2. [可用服務](#可用服務)
3. [常用命令](#常用命令)
4. [服務配置](#服務配置)
5. [故障排除](#故障排除)

---

## 🚀 快速開始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- （可選）NVIDIA Docker 支援（用於 GPU）

### 啟動所有服務

```bash
# 從項目根目錄
cd /path/to/My-AI-Learning-Notes

# 啟動所有服務
docker-compose up -d

# 查看運行狀態
docker-compose ps

# 查看日誌
docker-compose logs -f
```

### 啟動特定服務

```bash
# 只啟動 ChromaDB
docker-compose up -d chromadb

# 啟動 LLM 相關服務
docker-compose up -d ollama chromadb

# 啟動 RAG 所需服務
docker-compose up -d chromadb postgres redis
```

---

## 🛠️ 可用服務

### 向量資料庫

| 服務 | 端口 | 用途 | 訪問地址 |
|------|------|------|----------|
| ChromaDB | 8000 | 輕量級向量資料庫 | http://localhost:8000 |
| Qdrant | 6333, 6334 | 高性能向量資料庫 | http://localhost:6333 |

**ChromaDB 範例：**
```python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
```

**Qdrant 範例：**
```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
```

---

### 本地 LLM

| 服務 | 端口 | 用途 | 訪問地址 |
|------|------|------|----------|
| Ollama | 11434 | 本地運行 LLM | http://localhost:11434 |
| Ollama WebUI | 3000 | Ollama 網頁界面 | http://localhost:3000 |

**Ollama 使用：**
```bash
# 拉取模型
docker exec -it ai-learning-ollama ollama pull llama3.2

# 運行模型
docker exec -it ai-learning-ollama ollama run llama3.2

# 列出模型
docker exec -it ai-learning-ollama ollama list
```

**Python 範例：**
```python
from langchain_community.llms import Ollama
llm = Ollama(base_url="http://localhost:11434", model="llama3.2")
response = llm.invoke("Hello, how are you?")
```

---

### 資料庫

| 服務 | 端口 | 用途 | 憑證 |
|------|------|------|------|
| PostgreSQL | 5432 | 關係型資料庫 + pgvector | user: ai_user, pass: ai_password |
| MongoDB | 27017 | NoSQL 資料庫 | user: admin, pass: admin_password |
| Redis | 6379 | 快取 & 消息隊列 | pass: redis_password |

**PostgreSQL 連接：**
```python
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="ai_learning",
    user="ai_user",
    password="ai_password_change_in_production"
)
```

**Redis 連接：**
```python
import redis
r = redis.Redis(
    host='localhost',
    port=6379,
    password='redis_password_change_in_production'
)
```

---

### 搜尋引擎

| 服務 | 端口 | 用途 | 訪問地址 |
|------|------|------|----------|
| Elasticsearch | 9200, 9300 | 全文搜尋引擎 | http://localhost:9200 |

**Elasticsearch 範例：**
```python
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
```

---

### 監控與追蹤

| 服務 | 端口 | 用途 | 訪問地址 | 憑證 |
|------|------|------|----------|------|
| MLflow | 5000 | 實驗追蹤 | http://localhost:5000 | - |
| Prometheus | 9090 | 指標收集 | http://localhost:9090 | - |
| Grafana | 3001 | 資料可視化 | http://localhost:3001 | user: admin, pass: admin |

**MLflow 範例：**
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.95)
```

---

### 開發環境

| 服務 | 端口 | 用途 | 訪問地址 | Token |
|------|------|------|----------|-------|
| Jupyter Lab | 8888 | 開發環境 | http://localhost:8888 | your-token |

---

## 📝 常用命令

### 服務管理

```bash
# 啟動服務
docker-compose up -d [service_name]

# 停止服務
docker-compose stop [service_name]

# 重啟服務
docker-compose restart [service_name]

# 刪除服務（保留資料）
docker-compose down

# 刪除服務和資料
docker-compose down -v

# 查看服務狀態
docker-compose ps

# 查看服務資源使用
docker stats
```

### 日誌查看

```bash
# 查看所有服務日誌
docker-compose logs -f

# 查看特定服務日誌
docker-compose logs -f chromadb

# 查看最近 100 行日誌
docker-compose logs --tail=100 ollama
```

### 進入容器

```bash
# 進入容器 shell
docker-compose exec chromadb sh
docker-compose exec postgres psql -U ai_user -d ai_learning

# 執行命令
docker-compose exec ollama ollama list
docker-compose exec redis redis-cli
```

### 資料備份

```bash
# 備份 PostgreSQL
docker-compose exec -T postgres pg_dump -U ai_user ai_learning > backup.sql

# 恢復 PostgreSQL
cat backup.sql | docker-compose exec -T postgres psql -U ai_user ai_learning

# 備份 MongoDB
docker-compose exec mongodb mongodump --out /data/backup

# 備份所有資料卷
docker run --rm -v my-ai-learning-notes_chromadb_data:/data -v $(pwd):/backup alpine tar czf /backup/chromadb_backup.tar.gz /data
```

---

## ⚙️ 服務配置

### 修改端口

編輯 `docker-compose.yml`：

```yaml
services:
  chromadb:
    ports:
      - "8001:8000"  # 改為 8001
```

### 修改資源限制

```yaml
services:
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          memory: 512M
```

### 添加環境變量

```yaml
services:
  your-service:
    environment:
      - CUSTOM_VAR=value
    env_file:
      - .env
```

### 啟用 GPU 支援

確保已安裝 NVIDIA Docker：

```bash
# 檢查 GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 在 docker-compose.yml 中啟用 GPU（已預配置）
# 取消註釋 Ollama 和 Jupyter 的 GPU 配置
```

---

## 🔧 故障排除

### 問題 1: 端口已被佔用

**錯誤：** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**解決：**
```bash
# 查看佔用端口的進程
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# 修改 docker-compose.yml 中的端口
ports:
  - "8001:8000"
```

### 問題 2: 容器啟動失敗

**解決：**
```bash
# 查看詳細日誌
docker-compose logs [service_name]

# 檢查容器狀態
docker-compose ps

# 重新構建
docker-compose build --no-cache [service_name]
docker-compose up -d [service_name]
```

### 問題 3: 資料丟失

**解決：**
```bash
# 檢查資料卷
docker volume ls

# 檢查資料卷詳情
docker volume inspect my-ai-learning-notes_chromadb_data

# 定期備份（見上方備份命令）
```

### 問題 4: 內存不足

**解決：**
```bash
# 查看資源使用
docker stats

# 增加 Docker 內存限制（Docker Desktop）
# Settings > Resources > Memory

# 限制特定服務內存
# 見上方"修改資源限制"部分
```

### 問題 5: Ollama 模型下載慢

**解決：**
```bash
# 使用鏡像（中國大陸）
docker exec -it ai-learning-ollama sh
export OLLAMA_MODELS=/root/.ollama/models
# 手動下載模型文件後放入該目錄

# 或使用代理
docker-compose.yml 中添加：
environment:
  - HTTP_PROXY=http://your-proxy:port
  - HTTPS_PROXY=http://your-proxy:port
```

### 問題 6: PostgreSQL 連接被拒絕

**解決：**
```bash
# 檢查服務是否運行
docker-compose ps postgres

# 查看日誌
docker-compose logs postgres

# 等待服務完全啟動
docker-compose exec postgres pg_isready -U ai_user

# 檢查網絡
docker network inspect my-ai-learning-notes_ai-network
```

---

## 🔐 安全建議

### 生產環境部署

1. **修改所有預設密碼**
   ```bash
   # 在 docker-compose.yml 中搜索 "change_in_production"
   # 替換為強密碼
   ```

2. **使用 Docker Secrets**
   ```yaml
   services:
     postgres:
       secrets:
         - postgres_password
   secrets:
     postgres_password:
       file: ./secrets/postgres_password.txt
   ```

3. **限制網絡訪問**
   ```yaml
   services:
     postgres:
       ports:
         - "127.0.0.1:5432:5432"  # 只允許本地訪問
   ```

4. **啟用 TLS/SSL**
   ```yaml
   # 為服務配置證書
   volumes:
     - ./certs:/certs:ro
   ```

---

## 📚 更多資源

- [Docker 官方文檔](https://docs.docker.com/)
- [Docker Compose 文檔](https://docs.docker.com/compose/)
- [Ollama 文檔](https://github.com/ollama/ollama/blob/main/docs/docker.md)
- [ChromaDB 文檔](https://docs.trychroma.com/)
- [Qdrant 文檔](https://qdrant.tech/documentation/)

---

**提示：** 記得定期備份資料和更新服務鏡像！

```bash
# 更新所有服務
docker-compose pull
docker-compose up -d
```
