# 9.1 RAG + Agent + 部署的端到端實戰：智能文檔問答系統

## 項目概述

這是一個完整的生產級智能文檔問答系統，結合了：
- **RAG (檢索增強生成)**：高效的文檔檢索和向量搜索
- **Agent 系統**：多工具協作的智能代理
- **生產部署**：完整的 API 服務、Docker 容器化和監控

## 系統架構

```
用戶查詢
    ↓
API Gateway (FastAPI)
    ↓
Agent 決策層
    ↓
┌─────────────┬──────────────┬────────────┐
│   RAG 檢索   │  計算工具     │  網路搜索   │
└─────────────┴──────────────┴────────────┘
    ↓
LLM 生成回答
    ↓
返回結果 + 來源引用
```

## 核心功能

### 1. RAG 文檔檢索
- 支持多種文檔格式（PDF、TXT、Markdown、Word）
- 智能文本分塊（Chunk）策略
- 向量化存儲（ChromaDB）
- 混合檢索（向量 + 關鍵詞）

### 2. Agent 系統
- **工具集成**：
  - 文檔檢索工具
  - 數學計算工具
  - 網路搜索工具
  - 程式碼執行工具
- **智能決策**：根據問題類型自動選擇合適的工具
- **多輪對話**：維護對話上下文

### 3. AI 輔助功能
- 自動問題改寫（提高檢索效果）
- 答案品質評估
- 來源可信度分析
- 自動生成追問建議

### 4. 生產級特性
- RESTful API（FastAPI）
- 請求限流和緩存
- 完整的錯誤處理
- 結構化日誌
- 性能監控指標
- Docker 容器化部署

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置環境變數

```bash
# 複製配置模板
cp config/config.yaml.example config/config.yaml

# 編輯配置，填入你的 API Key
# - OPENAI_API_KEY 或其他 LLM API Key
# - SERPER_API_KEY（可選，用於網路搜索）
```

### 3. 準備文檔

將你的文檔放入 `docs/` 目錄：

```bash
# 支持的格式
docs/
  ├── document1.pdf
  ├── document2.txt
  ├── document3.md
  └── document4.docx
```

### 4. 初始化向量數據庫

```bash
python src/document_processor.py --init
```

### 5. 啟動服務

```bash
# 開發模式
python src/app.py

# 或使用 uvicorn
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 6. 測試 API

```bash
# 健康檢查
curl http://localhost:8000/health

# 提問
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什麼是機器學習？",
    "use_agent": true
  }'
```

## Docker 部署

### 構建鏡像

```bash
docker build -t rag-agent-system .
```

### 運行容器

```bash
docker run -d \
  --name rag-agent \
  -p 8000:8000 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/config:/app/config \
  -e OPENAI_API_KEY=your_key \
  rag-agent-system
```

### 使用 Docker Compose

```bash
docker-compose up -d
```

## API 文檔

啟動服務後訪問：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要端點

#### 1. 查詢問答

```
POST /api/v1/query
```

請求體：
```json
{
  "question": "你的問題",
  "use_agent": true,
  "top_k": 3,
  "session_id": "optional_session_id"
}
```

響應：
```json
{
  "answer": "回答內容",
  "sources": [
    {
      "content": "來源文本",
      "document": "文檔名稱",
      "score": 0.95
    }
  ],
  "tools_used": ["rag_search", "calculator"],
  "confidence": 0.92,
  "suggestions": ["相關問題1", "相關問題2"]
}
```

#### 2. 上傳文檔

```
POST /api/v1/documents/upload
```

#### 3. 查看統計

```
GET /api/v1/stats
```

## 項目結構

```
9.1-RAG-Agent端到端實戰/
├── README.md                 # 本文件
├── requirements.txt          # Python 依賴
├── Dockerfile               # Docker 配置
├── docker-compose.yml       # Docker Compose 配置
├── .env.example            # 環境變數模板
├── config/
│   ├── config.yaml         # 主配置文件
│   └── logging.yaml        # 日誌配置
├── src/
│   ├── __init__.py
│   ├── app.py              # FastAPI 應用主入口
│   ├── rag_agent_system.py # RAG + Agent 核心系統
│   ├── document_processor.py # 文檔處理和索引
│   ├── vector_store.py     # 向量存儲管理
│   ├── agent_tools.py      # Agent 工具集
│   ├── llm_client.py       # LLM 客戶端封裝
│   ├── models.py           # 數據模型
│   └── utils.py            # 工具函數
├── docs/                   # 存放要索引的文檔
│   └── sample_doc.md
├── tests/
│   ├── __init__.py
│   ├── test_system.py      # 系統集成測試
│   ├── test_rag.py         # RAG 功能測試
│   └── test_agent.py       # Agent 功能測試
└── data/                   # 運行時數據（向量庫等）
    └── .gitkeep
```

## 技術棧

- **LLM**: OpenAI GPT-4 / Claude / 本地模型
- **向量數據庫**: ChromaDB
- **Embedding**: OpenAI text-embedding-3-small
- **Web 框架**: FastAPI
- **Agent 框架**: LangChain / LlamaIndex
- **文檔處理**: PyPDF2, python-docx, markdown
- **部署**: Docker, Docker Compose
- **監控**: Prometheus metrics

## 進階特性

### 1. 混合檢索策略

系統結合了多種檢索方法：
- 向量相似度檢索（語義理解）
- BM25 關鍵詞檢索（精確匹配）
- 重排序（Reranking）提高精度

### 2. Agent 決策流程

```python
1. 接收用戶問題
2. 問題分類（文檔查詢 / 計算 / 搜索 / 混合）
3. 選擇合適的工具
4. 執行工具並獲取結果
5. LLM 整合結果生成回答
6. 評估答案品質
7. 返回結果和建議
```

### 3. AI 輔助增強

- **問題改寫**: 自動優化用戶問題以提高檢索效果
- **答案評估**: 評估答案的完整性、準確性和相關性
- **追問建議**: 基於對話歷史生成智能追問
- **來源分析**: 評估來源的可信度和相關性

## 性能優化

- 向量檢索緩存
- 問答結果緩存
- 異步處理
- 批量 embedding
- 連接池管理

## 監控和日誌

系統提供完整的監控指標：
- 請求數量和延遲
- 工具使用統計
- 錯誤率追蹤
- 資源使用情況

訪問 `/metrics` 端點查看 Prometheus 格式的指標。

## 測試

```bash
# 運行所有測試
pytest tests/

# 運行特定測試
pytest tests/test_rag.py -v

# 測試覆蓋率
pytest --cov=src tests/
```

## 常見問題

### 1. 如何更換 LLM？

編輯 `config/config.yaml`，修改 `llm` 部分：

```yaml
llm:
  provider: "openai"  # 或 "anthropic", "local"
  model: "gpt-4"
  api_key: "your_key"
```

### 2. 如何自定義分塊策略？

修改 `src/document_processor.py` 中的 `ChunkingConfig`。

### 3. 如何添加新的 Agent 工具？

在 `src/agent_tools.py` 中添加新的工具類，並在 `RAGAgentSystem` 中註冊。

## 擴展方向

- [ ] 添加多模態支持（圖像、視頻）
- [ ] 實現流式響應
- [ ] 添加用戶認證和權限管理
- [ ] 集成更多外部工具（郵件、日曆等）
- [ ] 實現分佈式部署
- [ ] 添加 A/B 測試框架

## 參考資源

- [LangChain 文檔](https://python.langchain.com/)
- [ChromaDB 文檔](https://docs.trychroma.com/)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)

## 授權

MIT License
