# FastAPI 基礎 LLM API

最簡單的 FastAPI LLM API 範例，適合快速開始學習。

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置 API 金鑰

```bash
# 創建 .env 文件
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. 運行服務

```bash
# 開發模式（自動重載）
uvicorn main:app --reload

# 生產模式
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 訪問 API 文檔

打開瀏覽器訪問：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端點

### POST /chat

簡單聊天

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好！",
    "model": "gpt-3.5-turbo"
  }'
```

### POST /summarize

文本摘要

```bash
curl -X POST "http://localhost:8000/summarize?text=這是一段很長的文本..."
```

### POST /translate

翻譯

```bash
curl -X POST "http://localhost:8000/translate?text=Hello&target_language=中文"
```

## 學習要點

1. **FastAPI 基礎**：路由、請求/回應模型
2. **Pydantic 驗證**：自動數據驗證
3. **OpenAI 整合**：調用 LLM API
4. **錯誤處理**：HTTPException
5. **自動文檔**：Swagger UI

## 進階學習

完成這個基礎範例後，可以學習：
- [帶認證的 API](../with_auth/)
- [速率限制](../rate_limiting/)
- [Docker 部署](../../docker_examples/)
