# RAG ChatBot - 檢索增強生成聊天機器人

完整的端到端 RAG 應用，包含 FastAPI 後端、向量檢索、LLM 生成和 Docker 部署。

## 🌟 特性

- ✅ **FastAPI 後端**: 高性能異步 API
- ✅ **向量檢索**: ChromaDB + Sentence Transformers
- ✅ **LLM 集成**: OpenAI GPT-3.5/GPT-4
- ✅ **對話管理**: 多輪對話支持
- ✅ **文檔管理**: 上傳、檢索、刪除
- ✅ **流式輸出**: 實時響應
- ✅ **Docker 部署**: 一鍵部署
- ✅ **RESTful API**: 完整的 API 文檔

## 📋 系統架構

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Frontend  │─────▶│  FastAPI     │─────▶│  RAG Engine  │
│  (React/Vue)│      │  Backend     │      │              │
└─────────────┘      └──────────────┘      └──────┬───────┘
                                                   │
                     ┌─────────────────────────────┼─────────────┐
                     │                             │             │
              ┌──────▼────────┐          ┌────────▼────┐  ┌─────▼─────┐
              │  ChromaDB     │          │  OpenAI     │  │ Embedding │
              │  (Vector DB)  │          │  API        │  │  Model    │
              └───────────────┘          └─────────────┘  └───────────┘
```

## 🚀 快速開始

### 1. 本地運行

#### 前置要求
- Python 3.11+
- OpenAI API Key

#### 安裝步驟

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設置環境變數
export OPENAI_API_KEY="your-api-key"

# 3. 啟動服務
python main.py
```

服務將在 `http://localhost:8000` 啟動。

訪問 API 文檔: `http://localhost:8000/docs`

### 2. Docker 部署

```bash
# 1. 創建 .env 文件
echo "OPENAI_API_KEY=your-api-key" > .env

# 2. 構建並啟動
docker-compose up -d

# 3. 查看日誌
docker-compose logs -f

# 4. 停止服務
docker-compose down
```

### 3. 生產環境部署

```bash
# 使用 Nginx 反向代理
docker-compose -f docker-compose.prod.yml up -d
```

## 📖 API 使用示例

### 1. 聊天

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "什麼是 Transformer？",
    "use_rag": true,
    "top_k": 3
  }'
```

### 2. 添加文檔

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Transformer 是 Google 在 2017 年提出的神經網絡架構...",
    "metadata": {"source": "research_paper"}
  }'
```

### 3. 上傳文件

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@document.txt"
```

### 4. 列出文檔

```bash
curl http://localhost:8000/api/documents
```

### 5. 健康檢查

```bash
curl http://localhost:8000/api/health
```

## 🔧 配置

### 環境變數

| 變數名 | 說明 | 預設值 |
|-------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | 必須設置 |
| `EMBEDDING_MODEL` | 嵌入模型 | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | LLM 模型 | `gpt-3.5-turbo` |
| `CHROMA_PERSIST_DIR` | ChromaDB 目錄 | `./chroma_db` |

### 修改模型

編輯 `rag_engine.py`:

```python
def __init__(
    self,
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # 更好的嵌入
    llm_model="gpt-4",  # 使用 GPT-4
    chroma_persist_dir="./chroma_db"
):
```

## 📊 API 端點列表

| 方法 | 端點 | 說明 |
|------|------|------|
| `POST` | `/api/chat` | 聊天 |
| `POST` | `/api/chat/stream` | 流式聊天 |
| `POST` | `/api/documents` | 添加文檔 |
| `POST` | `/api/documents/upload` | 上傳文件 |
| `GET` | `/api/documents` | 列出文檔 |
| `DELETE` | `/api/documents/{id}` | 刪除文檔 |
| `GET` | `/api/conversations/{id}` | 獲取對話 |
| `DELETE` | `/api/conversations/{id}` | 刪除對話 |
| `GET` | `/api/health` | 健康檢查 |
| `GET` | `/api/stats` | 統計信息 |

## 🧪 測試

```bash
# 運行測試
pytest tests/

# 測試覆蓋率
pytest --cov=. tests/
```

## 📈 性能優化

### 1. 使用更快的嵌入模型
```python
embedding_model = "all-MiniLM-L6-v2"  # 快速
# vs
embedding_model = "all-mpnet-base-v2"  # 更準確
```

### 2. 批量處理
```python
# 批量添加文檔
for doc in documents:
    await rag_engine.add_document(doc)
```

### 3. 緩存
- 使用 Redis 緩存頻繁查詢
- 緩存嵌入結果

### 4. 水平擴展
```bash
# 運行多個實例
docker-compose up --scale rag-api=4
```

## 🔒 安全建議

1. **API Key 管理**
   - 使用環境變數
   - 定期輪換 API Key
   - 不要提交到版本控制

2. **訪問控制**
   - 添加 API 認證
   - 限制請求頻率
   - 使用 HTTPS

3. **數據隱私**
   - 加密敏感數據
   - 定期清理對話歷史
   - 遵守數據保護法規

## 🐛 故障排除

### 問題 1: ChromaDB 錯誤
```bash
# 清除數據庫
rm -rf ./chroma_db
# 重新啟動
python main.py
```

### 問題 2: OpenAI API 錯誤
```bash
# 檢查 API Key
echo $OPENAI_API_KEY
# 檢查配額
curl https://api.openai.com/v1/usage
```

### 問題 3: 內存不足
```bash
# 限制模型大小
# 使用較小的嵌入模型
# 減少批次大小
```

## 📚 擴展功能

### 1. 添加更多向量數據庫
- Pinecone
- Weaviate
- Milvus

### 2. 集成更多 LLM
- Anthropic Claude
- Google PaLM
- 本地模型（Llama, Mistral）

### 3. 高級 RAG 技術
- HyDE (Hypothetical Document Embeddings)
- Self-RAG (Self-Reflective RAG)
- GraphRAG

### 4. 前端集成
- React
- Vue.js
- Streamlit

## 📄 授權

MIT License

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📞 聯繫

- Issues: [GitHub Issues](https://github.com/yourusername/rag-chatbot/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/rag-chatbot/discussions)

## 🙏 致謝

- FastAPI
- OpenAI
- ChromaDB
- Sentence Transformers
