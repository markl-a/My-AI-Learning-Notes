# 故障排除指南

> **最後更新**: 2025-01
> **適用版本**: 1.0.0+

本指南幫助您診斷和解決使用 My-AI-Learning-Notes 專案時可能遇到的常見問題。

---

## 📋 目錄

- [快速診斷](#快速診斷)
- [安裝問題](#安裝問題)
- [API 相關問題](#api-相關問題)
- [LLM 服務問題](#llm-服務問題)
- [RAG 系統問題](#rag-系統問題)
- [Docker 問題](#docker-問題)
- [性能問題](#性能問題)
- [常見錯誤碼](#常見錯誤碼)

---

## 快速診斷

### 系統狀態檢查

```bash
# 檢查 Python 版本
python --version  # 需要 >= 3.9

# 檢查依賴安裝
pip list | grep -E "(fastapi|langchain|openai)"

# 檢查環境變量
echo $OPENAI_API_KEY

# 檢查服務健康狀態
curl http://localhost:8000/api/health
```

### 日誌查看

```bash
# 查看應用日誌
tail -f logs/api_*.log

# 查看 Docker 日誌
docker-compose logs -f

# 查看特定服務日誌
docker-compose logs -f rag-chatbot
```

---

## 安裝問題

### ❌ 問題：`pip install` 失敗

**症狀**:
```
ERROR: Could not find a version that satisfies the requirement...
```

**解決方案**:

1. **更新 pip**:
   ```bash
   pip install --upgrade pip
   ```

2. **使用國內鏡像** (中國大陸):
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **安裝系統依賴** (Linux):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   ```

4. **使用 conda** (如果 pip 持續失敗):
   ```bash
   conda create -n ai-learning python=3.11
   conda activate ai-learning
   pip install -r requirements.txt
   ```

---

### ❌ 問題：`torch` 安裝失敗

**症狀**:
```
ERROR: Could not build wheels for torch
```

**解決方案**:

1. **使用官方安裝命令**:
   ```bash
   # CPU 版本
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # CUDA 版本 (NVIDIA GPU)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **檢查系統兼容性**:
   - Windows: 需要 Visual C++ Build Tools
   - macOS: 需要 Xcode Command Line Tools
   - Linux: 需要 gcc 和 g++

---

### ❌ 問題：ChromaDB 安裝失敗

**症狀**:
```
ERROR: Failed building wheel for chroma-hnswlib
```

**解決方案**:

1. **安裝編譯工具**:
   ```bash
   # macOS
   xcode-select --install

   # Ubuntu/Debian
   sudo apt-get install build-essential

   # Windows
   # 安裝 Visual Studio Build Tools
   ```

2. **使用預編譯版本**:
   ```bash
   pip install chromadb --prefer-binary
   ```

---

## API 相關問題

### ❌ 問題：401 Unauthorized

**症狀**:
```json
{"error": "Invalid API Key", "status_code": 401}
```

**解決方案**:

1. **檢查 API Key 設置**:
   ```bash
   # 確保環境變量已設置
   export API_KEY="your-api-key"

   # 或在 .env 文件中
   echo "API_KEY=your-api-key" >> .env
   ```

2. **檢查請求格式**:
   ```bash
   # 正確格式
   curl -H "Authorization: Bearer YOUR_API_KEY" ...

   # 或
   curl -H "X-API-Key: YOUR_API_KEY" ...
   ```

---

### ❌ 問題：429 Too Many Requests

**症狀**:
```json
{"error": "Rate limit exceeded", "retry_after": 30}
```

**解決方案**:

1. **等待重試**:
   ```python
   import time

   response = requests.post(url, json=data)
   if response.status_code == 429:
       retry_after = int(response.headers.get('Retry-After', 30))
       time.sleep(retry_after)
       response = requests.post(url, json=data)
   ```

2. **實現指數退避**:
   ```python
   from tenacity import retry, wait_exponential, stop_after_attempt

   @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
   def make_request():
       response = requests.post(url, json=data)
       response.raise_for_status()
       return response
   ```

3. **調整速率限制** (如果是自己的服務):
   ```python
   # 在 rate_limiter.py 中調整
   rate_limiter = RateLimiter(
       requests_per_minute=120,  # 增加限制
       requests_per_hour=2000
   )
   ```

---

### ❌ 問題：CORS 錯誤

**症狀**:
```
Access to fetch at 'http://localhost:8000/api/chat' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**解決方案**:

1. **設置允許的來源**:
   ```bash
   # 在 .env 中設置
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
   ```

2. **重啟服務**:
   ```bash
   # 重啟 FastAPI 服務
   uvicorn main:app --reload
   ```

---

## LLM 服務問題

### ❌ 問題：OpenAI API 錯誤

**症狀**:
```
openai.error.AuthenticationError: Incorrect API key provided
```

**解決方案**:

1. **檢查 API Key**:
   ```bash
   # 確保 Key 格式正確 (sk-...)
   echo $OPENAI_API_KEY
   ```

2. **檢查配額**:
   - 訪問 https://platform.openai.com/usage
   - 確保有足夠的配額

3. **測試連接**:
   ```python
   from openai import OpenAI
   client = OpenAI()

   try:
       response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "Hello"}],
           max_tokens=10
       )
       print("連接成功!")
   except Exception as e:
       print(f"錯誤: {e}")
   ```

---

### ❌ 問題：Ollama 連接失敗

**症狀**:
```
Connection refused: http://localhost:11434
```

**解決方案**:

1. **檢查 Ollama 是否運行**:
   ```bash
   # 檢查狀態
   ollama list

   # 如果未運行，啟動服務
   ollama serve
   ```

2. **拉取模型**:
   ```bash
   ollama pull llama3.2
   ```

3. **檢查端口**:
   ```bash
   # 確保端口未被佔用
   lsof -i :11434
   ```

---

## RAG 系統問題

### ❌ 問題：向量檢索無結果

**症狀**:
- 查詢返回空結果
- 相關文檔未被檢索到

**解決方案**:

1. **檢查文檔是否已索引**:
   ```bash
   curl http://localhost:8000/api/documents
   ```

2. **檢查嵌入模型**:
   ```python
   # 確保嵌入模型正常工作
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   embedding = model.encode("測試文字")
   print(f"嵌入維度: {len(embedding)}")
   ```

3. **調整檢索參數**:
   ```python
   # 增加 top_k
   result = await rag_engine.chat(
       message="問題",
       top_k=10  # 增加檢索數量
   )
   ```

4. **重建索引**:
   ```bash
   # 刪除 ChromaDB 資料
   rm -rf ./chroma_db

   # 重新索引文檔
   python scripts/index_documents.py
   ```

---

### ❌ 問題：ChromaDB 持久化失敗

**症狀**:
```
sqlite3.OperationalError: database is locked
```

**解決方案**:

1. **確保只有一個進程訪問**:
   ```bash
   # 查找佔用進程
   lsof ./chroma_db/chroma.sqlite3

   # 終止進程
   kill -9 <PID>
   ```

2. **使用持久化配置**:
   ```python
   import chromadb

   client = chromadb.PersistentClient(
       path="./chroma_db",
       settings=chromadb.Settings(
           anonymized_telemetry=False,
           allow_reset=True
       )
   )
   ```

---

## Docker 問題

### ❌ 問題：容器無法啟動

**症狀**:
```
Error response from daemon: Conflict. The container name is already in use
```

**解決方案**:

```bash
# 停止並刪除現有容器
docker-compose down

# 清理所有停止的容器
docker container prune

# 重新啟動
docker-compose up -d
```

---

### ❌ 問題：內存不足

**症狀**:
```
docker: Error response from daemon: OCI runtime create failed: container_linux.go:380: starting container process caused: process_linux.go:545: container init caused: Running hook #0:: error running hook: exit status 1, stdout: , stderr: Auto-detected mode: legacy
```

**解決方案**:

1. **增加 Docker 內存限制**:
   - Docker Desktop → Settings → Resources → Memory
   - 建議至少 8GB

2. **限制容器資源**:
   ```yaml
   # docker-compose.yml
   services:
     rag-chatbot:
       deploy:
         resources:
           limits:
             memory: 4G
   ```

---

## 性能問題

### ❌ 問題：響應緩慢

**診斷**:
```bash
# 檢查響應時間
time curl http://localhost:8000/api/chat -d '{"message":"test"}'

# 檢查系統資源
htop
nvidia-smi  # 如果使用 GPU
```

**解決方案**:

1. **啟用快取**:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=100)
   def get_embedding(text: str):
       return model.encode(text)
   ```

2. **使用更快的模型**:
   ```python
   # 使用更小的嵌入模型
   model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
   ```

3. **啟用 GPU 加速**:
   ```python
   # 確保使用 GPU
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = model.to(device)
   ```

---

## 常見錯誤碼

| 錯誤碼 | 說明 | 解決方案 |
|-------|------|---------|
| `ECONNREFUSED` | 服務未啟動 | 啟動相應服務 |
| `TIMEOUT` | 請求超時 | 檢查網絡、增加超時時間 |
| `MEMORY_ERROR` | 內存不足 | 減少批量大小、增加內存 |
| `RATE_LIMIT` | 速率限制 | 等待或調整限制 |
| `AUTH_FAILED` | 認證失敗 | 檢查 API Key |
| `MODEL_NOT_FOUND` | 模型未找到 | 下載/拉取模型 |

---

## 獲取幫助

如果以上方法無法解決您的問題：

1. **搜索現有 Issues**: [GitHub Issues](https://github.com/markl-a/My-AI-Learning-Notes/issues)

2. **建立新 Issue**:
   - 提供詳細的錯誤資訊
   - 包含復現步驟
   - 附上相關日誌

3. **參與討論**: [GitHub Discussions](https://github.com/markl-a/My-AI-Learning-Notes/discussions)

---

## 相關文檔

- [QUICKSTART.md](../QUICKSTART.md) - 快速入門
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) - API 文檔
- [DEPLOYMENT.md](../DEPLOYMENT.md) - 部署指南
