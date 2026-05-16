# AI 文檔分析系統

基於 LLM 的智能文檔處理、分析和問答系統。支持多種文檔格式，提供文檔摘要、實體提取、關鍵詞分析、主題建模、情感分析等功能。

## ✨ 核心功能

### 📄 文檔處理
- **多格式支持**：PDF、DOCX、TXT、Markdown、HTML
- **智能分塊**：自動將長文檔分塊處理
- **向量化存儲**：使用 ChromaDB 進行語義搜索
- **元資料管理**：完整的文檔元資料追蹤

### 🔍 智能分析
1. **文檔摘要** - 生成不同長度的文檔摘要（短/中/長）
2. **實體提取** - 識別人名、地名、組織、日期等命名實體
3. **關鍵詞提取** - 提取核心關鍵詞和術語
4. **主題建模** - 識別主要主題和子主題
5. **情感分析** - 分析文檔整體和局部情感傾向
6. **結構分析** - 分析文檔結構和組織方式

### 💬 文檔問答
- 基於文檔內容的智能問答
- 提供答案置信度和引用來源
- 支持長文檔上下文理解

### 🔄 文檔比較
- 多文檔相似度分析
- 差異和共同點識別
- 主題對比分析

### 🔎 語義搜索
- 基於語義的文檔檢索
- 支持文件類型篩選
- 相關性排序

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI 應用                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐         ┌───────────────────┐    │
│  │ Document         │         │   Document        │    │
│  │ Processor        │◄────────┤   Analyzer        │    │
│  │                  │         │   (LLM-based)     │    │
│  │ - PDF 解析       │         │                   │    │
│  │ - DOCX 解析      │         │ - 摘要生成        │    │
│  │ - 文字分塊       │         │ - 實體提取        │    │
│  │ - 向量化         │         │ - 關鍵詞提取      │    │
│  └──────────────────┘         │ - 主題建模        │    │
│           │                    │ - 情感分析        │    │
│           ▼                    └───────────────────┘    │
│  ┌──────────────────┐                  │               │
│  │   ChromaDB       │                  ▼               │
│  │  (Vector Store)  │         ┌───────────────────┐    │
│  │                  │         │   OpenAI API      │    │
│  │ - 文檔嵌入       │         │   (GPT-3.5/4)     │    │
│  │ - 語義搜索       │         └───────────────────┘    │
│  └──────────────────┘                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🚀 快速開始

### 環境要求

- Python 3.11+
- OpenAI API 密鑰
- Docker（可選）

### 本地運行

1. **克隆項目並安裝依賴**

```bash
# 進入項目目錄
cd AI-Document-Analyzer

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

2. **配置環境變量**

```bash
# 複製設定檔
cp .env.example .env

# 編輯 .env，填入 OpenAI API 密鑰
OPENAI_API_KEY=your_api_key_here
```

3. **啟動服務**

```bash
# 開發模式
python main.py

# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

4. **訪問 API 文檔**

打開瀏覽器訪問：
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### Docker 部署

1. **使用 Docker Compose（推薦）**

```bash
# 配置環境變量
cp .env.example .env
# 編輯 .env 文件

# 啟動所有服務
docker-compose up -d

# 查看日誌
docker-compose logs -f document-analyzer

# 停止服務
docker-compose down
```

2. **僅運行主服務**

```bash
# 構建鏡像
docker build -t ai-document-analyzer .

# 運行容器
docker run -d \
  --name analyzer \
  -p 8001:8001 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/documents:/app/documents \
  ai-document-analyzer
```

## 📚 API 使用示例

### 1. 上傳文檔

```bash
curl -X POST "http://localhost:8001/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

**響應：**
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.pdf",
  "pages": 10,
  "word_count": 5000,
  "message": "Document uploaded successfully"
}
```

### 2. 生成文檔摘要

```bash
curl -X POST "http://localhost:8001/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "123e4567-e89b-12d3-a456-426614174000",
    "analysis_type": "summary",
    "options": {"length": "medium"}
  }'
```

**響應：**
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "analysis_type": "summary",
  "result": {
    "summary": "這份文檔主要討論了...",
    "length": "medium",
    "word_count": 120,
    "original_length": 5000
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 3. 提取關鍵詞

```bash
curl -X POST "http://localhost:8001/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "123e4567-e89b-12d3-a456-426614174000",
    "analysis_type": "keywords",
    "options": {"max_keywords": 10}
  }'
```

### 4. 文檔問答

```bash
curl -X POST "http://localhost:8001/api/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "123e4567-e89b-12d3-a456-426614174000",
    "question": "這份文檔的主要結論是什麼？",
    "context_window": 1000
  }'
```

**響應：**
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "question": "這份文檔的主要結論是什麼？",
  "answer": "根據文檔內容，主要結論是...",
  "confidence": 0.92,
  "sources": ["第3頁：...", "第7頁：..."]
}
```

### 5. 比較文檔

```bash
curl -X POST "http://localhost:8001/api/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc1_id", "doc2_id", "doc3_id"],
    "comparison_aspects": ["content", "topics", "structure"]
  }'
```

### 6. 語義搜索

```bash
curl -X GET "http://localhost:8001/api/search?query=machine+learning&limit=5"
```

### 7. 批量分析

```bash
curl -X POST "http://localhost:8001/api/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc1", "doc2", "doc3"],
    "analysis_types": ["summary", "keywords", "topics"]
  }'
```

## 🔧 高級配置

### 分析參數調整

```python
# 修改 analyzer.py 中的參數

# 摘要長度
length_prompts = {
    'short': "簡短摘要（2-3句）",
    'medium': "中等摘要（100-150字）",
    'long': "詳細摘要（200-300字）"
}

# LLM 溫度參數
temperature = 0.3  # 0-1，越低越確定

# 最大生成 tokens
max_tokens = 500
```

### 文檔分塊策略

```python
# 修改 document_processor.py

chunk_size = 500      # 每塊單詞數
overlap = 50          # 重疊單詞數
```

### 嵌入模型選擇

```env
# .env 文件

# 輕量級（快速，英文）
EMBEDDING_MODEL=all-MiniLM-L6-v2

# 高品質（較慢，英文）
EMBEDDING_MODEL=all-mpnet-base-v2

# 多語言支持
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

## 📊 性能優化

### 1. 使用更強大的 LLM

```env
# 在 .env 中配置
OPENAI_MODEL=gpt-4  # 更高品質但較慢
# 或
OPENAI_MODEL=gpt-3.5-turbo  # 平衡性能和成本
```

### 2. 啟用分析快取

分析結果會自動快取，避免重複計算。

### 3. 批處理文檔

使用批量分析 API 可以提高效率：

```python
# 一次分析多個文檔
POST /api/analyze/batch
```

### 4. 調整 Worker 數量

```bash
# 修改 docker-compose.yml
uvicorn main:app --workers 4  # 增加並發處理能力
```

## 🧪 測試

```bash
# 安裝測試依賴
pip install pytest pytest-asyncio pytest-cov httpx

# 運行所有測試
pytest

# 運行並查看覆蓋率
pytest --cov=. --cov-report=html

# 運行特定測試文件
pytest tests/test_analyzer.py -v
```

## 📈 監控與日誌

### 查看日誌

```bash
# Docker 日誌
docker-compose logs -f document-analyzer

# 本地日誌文件
tail -f logs/analyzer.log
```

### Prometheus + Grafana

Docker Compose 已包含監控服務：

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000（預設密碼：admin/admin）

## 🛡️ 安全注意事項

1. **API 密鑰保護**
   - 永遠不要將 `.env` 文件提交到 Git
   - 使用環境變量管理敏感資訊

2. **文件上傳限制**
   - 設置最大文件大小（預設 10MB）
   - 驗證文件類型

3. **CORS 配置**
   - 生產環境限制允許的域名
   - 不要使用 `allow_origins=["*"]`

4. **速率限制**
   - 實現 API 速率限制
   - 防止濫用

## 🐛 故障排除

### 問題：文檔上傳失敗

**解決方案：**
- 檢查文件格式是否支持
- 確認文件大小未超過限制
- 查看日誌文件獲取詳細錯誤

### 問題：OpenAI API 錯誤

**解決方案：**
- 驗證 API 密鑰是否正確
- 檢查 API 配額是否用盡
- 確認網絡連接正常

### 問題：ChromaDB 錯誤

**解決方案：**
- 刪除 `chroma_db` 目錄重新初始化
- 檢查磁盤空間是否充足

### 問題：內存不足

**解決方案：**
- 減少 Worker 數量
- 調整文檔分塊大小
- 使用更小的嵌入模型

## 📦 項目結構

```
AI-Document-Analyzer/
├── main.py                 # FastAPI 主應用
├── document_processor.py   # 文檔處理器
├── analyzer.py            # 文檔分析器
├── requirements.txt       # Python 依賴
├── Dockerfile            # Docker 配置
├── docker-compose.yml    # Docker Compose 配置
├── .env.example          # 環境變量範例
├── README.md             # 項目文檔
├── documents/            # 文檔存儲目錄
├── logs/                 # 日誌目錄
└── tests/                # 測試文件
    ├── test_analyzer.py
    └── test_processor.py
```

## 🔜 路線圖

- [ ] 支持更多文檔格式（PPT、Excel）
- [ ] 增加圖表和圖片分析
- [ ] 實現文檔版本控制
- [ ] 添加多用戶支持和權限管理
- [ ] Web UI 界面
- [ ] 支持更多 LLM 提供商（Anthropic、Cohere）
- [ ] 文檔註釋和標記功能
- [ ] 導出分析報告（PDF、Excel）

## 📄 許可證

MIT License

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📧 聯繫

如有問題或建議，請提交 Issue。
