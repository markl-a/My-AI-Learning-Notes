# 🚀 快速開始指南

本指南幫助你快速上手 LLM API 的使用和應用開發。

## 📁 項目結構

```
2.LLM as API/
├── .env.example              # 環境變數範例
├── requirements.txt          # Python 依賴
├── README.md                 # 詳細文檔
├── QUICK_START.md           # 快速開始（本文件）
├── examples/                 # 示例代碼
│   ├── basic_apis/          # 基礎 API 使用
│   │   ├── 01_openai_basic.py
│   │   ├── 02_anthropic_basic.py
│   │   ├── 03_gemini_basic.py
│   │   └── 04_api_comparison.py
│   └── frontend_integration/ # 前端整合
│       ├── streamlit_basic_chat.py    # 基礎聊天機器人
│       ├── streamlit_rag_chat.py      # RAG 問答系統
│       ├── streamlit_vision_chat.py   # 視覺理解助理
│       └── fastapi_backend/           # 後端服務
│           ├── main.py
│           ├── test_client.py
│           ├── Dockerfile
│           └── docker-compose.yml
└── utils/                    # 工具庫
    ├── cost_tracker.py       # 成本追踪
    └── performance_monitor.py # 性能監控
```

## ⚡ 5 分鐘快速開始

### 步驟 1：環境設定

```bash
# 1. 複製環境變數範例
cp .env.example .env

# 2. 編輯 .env，填入你的 API keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...

# 3. 安裝依賴
pip install -r requirements.txt
```

### 步驟 2：運行第一個示例

```bash
# 測試 OpenAI API
python examples/basic_apis/01_openai_basic.py

# 或者測試 Anthropic Claude
python examples/basic_apis/02_anthropic_basic.py

# 或者測試 Google Gemini
python examples/basic_apis/03_gemini_basic.py
```

### 步驟 3：啟動 Web 應用

```bash
# 基礎聊天機器人
streamlit run examples/frontend_integration/streamlit_basic_chat.py

# RAG 問答系統
streamlit run examples/frontend_integration/streamlit_rag_chat.py

# 視覺理解助理
streamlit run examples/frontend_integration/streamlit_vision_chat.py
```

## 🎯 使用場景選擇

### 場景 1：快速測試 API

**使用：** `examples/basic_apis/`

```python
# 最簡單的使用
from openai import OpenAI
client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

**相關文件：**
- [基礎 API README](examples/basic_apis/README.md)

### 場景 2：建立聊天機器人

**使用：** `streamlit_basic_chat.py`

**特點：**
- ✅ 支援多個 LLM 提供商切換
- ✅ 即時串流回應
- ✅ 對話歷史記錄
- ✅ 可調節參數

**啟動：**
```bash
streamlit run examples/frontend_integration/streamlit_basic_chat.py
```

**相關文件：**
- [Streamlit 整合 README](examples/frontend_integration/README.md)

### 場景 3：建立知識庫問答系統

**使用：** `streamlit_rag_chat.py`

**特點：**
- ✅ 文檔上傳（TXT, PDF, MD）
- ✅ 自動向量搜索
- ✅ 顯示來源文檔
- ✅ 精確的基於上下文回答

**啟動：**
```bash
streamlit run examples/frontend_integration/streamlit_rag_chat.py
```

**依賴：**
```bash
pip install langchain faiss-cpu pypdf pdfminer.six unstructured markdown
```

### 場景 4：建立圖片分析應用

**使用：** `streamlit_vision_chat.py`

**功能：**
- ✅ 圖片上傳和分析
- ✅ OCR 文字識別
- ✅ 物體檢測
- ✅ 情感分析
- ✅ 藝術評論

**啟動：**
```bash
streamlit run examples/frontend_integration/streamlit_vision_chat.py
```

### 場景 5：建立生產級 API 服務

**使用：** `fastapi_backend/`

**特點：**
- ✅ RESTful API
- ✅ 認證機制
- ✅ 錯誤處理
- ✅ Prometheus 監控
- ✅ 日誌記錄
- ✅ Docker 部署

**啟動：**
```bash
cd examples/frontend_integration/fastapi_backend

# 開發模式
python main.py

# 生產模式
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker 部署
docker-compose up -d
```

**相關文件：**
- [FastAPI 後端 README](examples/frontend_integration/fastapi_backend/README.md)

## 🔧 實用工具

### 成本追踪

```python
from utils.cost_tracker import CostTracker, print_statistics

# 創建追踪器
tracker = CostTracker("logs/costs.json")

# 追踪使用
tracker.track(
    provider="openai",
    model="gpt-4o-mini",
    prompt_tokens=100,
    completion_tokens=200,
    user_id="user123"
)

# 顯示統計
print_statistics(tracker)

# 導出 CSV
tracker.export_to_csv("costs_export.csv")
```

### 性能監控

```python
from utils.performance_monitor import PerformanceMonitor, print_performance_report

# 創建監控器
monitor = PerformanceMonitor("logs/performance.json")

# 記錄指標
monitor.record(
    provider="openai",
    model="gpt-4o-mini",
    operation="chat",
    latency=1.234,
    tokens=300,
    success=True
)

# 顯示報告
print_performance_report(monitor)
```

## 📊 API 比較工具

比較不同 LLM 的性能和成本：

```bash
python examples/basic_apis/04_api_comparison.py
```

**輸出：**
- ⚡ 延遲比較
- 💰 成本估算
- 📊 Token 使用分析
- 🎯 場景測試結果

## 🐳 Docker 快速部署

### 單個服務

```bash
cd examples/frontend_integration/fastapi_backend
docker build -t llm-api .
docker run -p 8000:8000 --env-file .env llm-api
```

### 完整棧（API + 監控）

```bash
cd examples/frontend_integration/fastapi_backend
docker-compose up -d
```

服務訪問：
- API: http://localhost:8000
- API 文檔: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## 📚 學習路徑

### 初學者

1. ✅ 運行基礎 API 示例
2. ✅ 啟動 Streamlit 聊天機器人
3. ✅ 了解 token 使用和成本
4. ✅ 嘗試不同的提示詞

### 中級

1. ✅ 建立 RAG 問答系統
2. ✅ 實作視覺理解應用
3. ✅ 使用成本追踪工具
4. ✅ 優化提示詞和參數

### 高級

1. ✅ 部署 FastAPI 後端服務
2. ✅ 設定 Prometheus 監控
3. ✅ 實作自定義錯誤處理
4. ✅ 優化性能和成本
5. ✅ Docker 容器化部署

## 💡 常見問題

### Q: 如何獲取 API Keys？

**OpenAI:**
- 訪問 https://platform.openai.com/
- 註冊帳號 → API Keys → Create new key

**Anthropic:**
- 訪問 https://console.anthropic.com/
- 註冊帳號 → API Keys → Create Key

**Google Gemini:**
- 訪問 https://makersuite.google.com/app/apikey
- 使用 Google 帳號登入 → Get API Key

### Q: 如何降低成本？

1. **選擇合適的模型：**
   - 開發測試：gpt-4o-mini
   - 生產環境：根據需求選擇

2. **優化 token 使用：**
   - 精簡提示詞
   - 設定 max_tokens 限制
   - 使用快取減少重複請求

3. **使用成本追踪：**
   ```python
   from utils.cost_tracker import CostTracker
   tracker = CostTracker()
   # 追踪每次使用
   ```

### Q: 如何提高性能？

1. **使用串流回應：**
   ```python
   response = client.chat.completions.create(..., stream=True)
   ```

2. **非同步請求：**
   ```python
   from openai import AsyncOpenAI
   client = AsyncOpenAI()
   response = await client.chat.completions.create(...)
   ```

3. **批次處理：**
   - 合併多個小請求
   - 使用並發處理

### Q: 錯誤處理最佳實踐？

```python
from openai import OpenAI, OpenAIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def robust_completion(messages):
    try:
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
    except RateLimitError:
        # 速率限制，重試
        raise
    except OpenAIError as e:
        # 其他錯誤，記錄並處理
        logger.error(f"API 錯誤: {e}")
        raise
```

## 🔗 相關資源

### 官方文檔
- [OpenAI API 文檔](https://platform.openai.com/docs)
- [Anthropic API 文檔](https://docs.anthropic.com/)
- [Google Gemini API 文檔](https://ai.google.dev/docs)

### 社群資源
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Awesome LLM](https://github.com/Hannibal046/Awesome-LLM)

### 本項目文檔
- [完整 README](README.md) - 詳細的理論和實作
- [基礎 API 示例](examples/basic_apis/README.md)
- [Streamlit 整合](examples/frontend_integration/README.md)
- [FastAPI 後端](examples/frontend_integration/fastapi_backend/README.md)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📝 更新日誌

**2025-01-15**
- ✅ 添加完整的基礎 API 示例
- ✅ 實作 3 個 Streamlit 應用
- ✅ 建立生產級 FastAPI 服務
- ✅ 添加 Docker 部署配置
- ✅ 創建成本追踪和性能監控工具

---

**開始你的 LLM 之旅！** 🚀

如有問題，請查看詳細文檔或提交 Issue。
