# Streamlit 前端整合示例

本目錄包含多個 Streamlit 應用示例，展示如何將 LLM API 整合到實用的 Web 應用中。

## 📁 文件說明

- `streamlit_basic_chat.py` - 基礎聊天機器人（支援多個 LLM 提供商）
- `streamlit_rag_chat.py` - RAG 智能問答系統（支援文檔上傳和向量搜索）
- `streamlit_vision_chat.py` - 視覺理解助理（支援圖片分析）

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd "3.LLM應用工程/2.LLM as API"
pip install -r requirements.txt

# 如果要使用 RAG 功能，還需要安裝：
pip install langchain faiss-cpu pypdf pdfminer.six unstructured markdown
```

### 2. 設定環境變數

確保已經設定好 `.env` 文件：

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### 3. 運行應用

```bash
# 基礎聊天機器人
streamlit run examples/frontend_integration/streamlit_basic_chat.py

# RAG 問答系統
streamlit run examples/frontend_integration/streamlit_rag_chat.py

# 視覺理解助理
streamlit run examples/frontend_integration/streamlit_vision_chat.py
```

應用會在瀏覽器中自動打開，預設地址為 `http://localhost:8501`

## 📚 應用詳解

### 1. 基礎聊天機器人 (`streamlit_basic_chat.py`)

**功能特點：**
- ✅ 支援多個 LLM 提供商切換（OpenAI, Anthropic, Gemini）
- ✅ 支援多種模型選擇
- ✅ 可調節 Temperature 參數
- ✅ 自定義系統提示
- ✅ 串流回應實時顯示
- ✅ 對話歷史記錄
- ✅ 預設範例問題
- ✅ 對話統計資訊

**使用場景：**
- 一般對話機器人
- AI 客服助理
- 教育問答系統
- 程式碼助手

**截圖：**

```
╔════════════════════════════════════════╗
║        🤖 AI 聊天助理                  ║
╠════════════════════════════════════════╣
║  選擇 AI: [OpenAI ▼]                   ║
║  選擇模型: [gpt-4o-mini ▼]            ║
║  Temperature: [━━●━━━] 0.7            ║
║  系統提示: [你是一個專業的助理...]     ║
║                                        ║
║  [🗑️ 清除對話]                        ║
║                                        ║
║  💡 建議問題                           ║
║  [解釋量子計算的基本原理]             ║
║  [寫一個 Python 快速排序]             ║
╚════════════════════════════════════════╝
```

### 2. RAG 智能問答系統 (`streamlit_rag_chat.py`)

**功能特點：**
- ✅ 支援文檔上傳（TXT, PDF, MD）
- ✅ 自動文檔切分和向量化
- ✅ FAISS 向量搜索
- ✅ 顯示來源文檔
- ✅ 可調節檢索數量
- ✅ 文檔統計資訊
- ✅ 基於上下文的精確回答

**使用場景：**
- 技術文檔問答
- 企業知識庫
- 學習資料整理
- 合約和法律文件分析

**工作流程：**

```
1. 上傳文檔
   ↓
2. 自動切分文檔為片段
   ↓
3. 生成向量嵌入
   ↓
4. 用戶提問
   ↓
5. 向量搜索相關片段
   ↓
6. 基於檢索結果生成答案
   ↓
7. 顯示答案和來源
```

**依賴套件：**

```bash
pip install langchain openai faiss-cpu
pip install pypdf pdfminer.six  # PDF 支援
pip install unstructured markdown  # Markdown 支援
```

### 3. 視覺理解助理 (`streamlit_vision_chat.py`)

**功能特點：**
- ✅ 支援圖片上傳和分析
- ✅ 支援多種格式（PNG, JPG, JPEG, WEBP）
- ✅ 多種預設分析任務
  - 詳細描述
  - OCR 文字識別
  - 物體檢測
  - 情感分析
  - 藝術評論
  - 技術分析
- ✅ 自定義分析提示
- ✅ 圖片自動壓縮
- ✅ 分析歷史記錄
- ✅ 範例圖片

**使用場景：**
- 圖片內容理解
- OCR 文字提取
- 圖片品質評估
- 藝術作品分析
- 醫學影像輔助
- 商品圖片分析

**支援的 AI：**
- OpenAI GPT-4o, GPT-4o-mini
- Anthropic Claude 3.5 Sonnet, Claude 3 Opus
- Google Gemini 1.5 Pro, Gemini 1.5 Flash

## 🎯 高級使用

### 自定義系統提示

在基礎聊天機器人中，你可以設定不同的系統提示來定制 AI 的行為：

**程式碼助手：**
```
你是一個專業的 Python 程式設計師。
- 總是提供完整可運行的程式碼
- 包含詳細的註解
- 遵循 PEP 8 規範
- 指出潛在的問題和最佳實踐
```

**翻譯助手：**
```
你是一個專業的翻譯專家。
- 保持原文的語氣和風格
- 使用自然流暢的目標語言
- 解釋文化背景差異
```

### RAG 參數調整

**提高準確度：**
- 增加檢索文檔數量（k=5）
- 降低 temperature（0.1-0.3）
- 使用更好的模型（gpt-4o）

**平衡成本和效果：**
- 適中的檢索數量（k=3）
- 適中的 temperature（0.3-0.5）
- 使用 gpt-4o-mini

### 視覺分析技巧

**OCR 最佳實踐：**
- 使用高解析度圖片
- 確保文字清晰可見
- 使用 Claude 3.5 Sonnet（OCR 性能優秀）

**詳細描述：**
- 提供具體的問題
- 使用 Gemini（多模態能力強）

## 🔧 自定義和擴展

### 添加新的 AI 提供商

在 `streamlit_basic_chat.py` 中添加新的 API：

```python
# 初始化新的客戶端
if os.getenv("NEW_API_KEY"):
    clients['new_provider'] = NewAPIClient(api_key=os.getenv("NEW_API_KEY"))

# 添加處理函數
def get_new_provider_response(client, messages, model, temperature):
    # 實作邏輯
    pass

# 在主函數中添加選項
if provider == 'new_provider':
    response = get_new_provider_response(...)
```

### 添加新的分析任務

在 `streamlit_vision_chat.py` 中添加：

```python
analysis_tasks = {
    # 現有任務...
    "新任務": "針對新任務的提示詞",
}
```

### 自定義 UI 樣式

修改 CSS：

```python
st.markdown("""
<style>
    .main-header {
        color: #your-color;
    }
    /* 添加更多自定義樣式 */
</style>
""", unsafe_allow_html=True)
```

## 🐛 常見問題

### 問題：Streamlit 無法啟動

**解決方案：**
```bash
# 確認已安裝 streamlit
pip install streamlit

# 檢查版本
streamlit --version

# 清除快取
streamlit cache clear
```

### 問題：API Key 錯誤

**解決方案：**
1. 確認 `.env` 文件在正確位置
2. 檢查環境變數名稱拼寫
3. 重啟 Streamlit 應用

### 問題：RAG 搜索不準確

**解決方案：**
1. 增加文檔片段的重疊（chunk_overlap）
2. 調整片段大小（chunk_size）
3. 增加檢索數量（k）
4. 使用更好的 embeddings 模型

### 問題：圖片上傳失敗

**解決方案：**
1. 檢查圖片格式是否支援
2. 確認圖片大小不超過限制
3. 嘗試壓縮圖片

## 🚀 部署

### 部署到 Streamlit Cloud

1. 將程式碼推送到 GitHub
2. 訪問 [streamlit.io](https://streamlit.io/)
3. 連接 GitHub repository
4. 在 Secrets 中設定環境變數
5. 部署！

### 部署到 Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_basic_chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

運行：
```bash
docker build -t streamlit-app .
docker run -p 8501:8501 --env-file .env streamlit-app
```

## 📊 性能優化

### 快取優化

使用 Streamlit 的快取功能：

```python
@st.cache_resource
def load_model():
    # 載入大型模型或資源
    pass

@st.cache_data
def process_data(data):
    # 處理資料
    pass
```

### 減少 API 呼叫

- 使用會話狀態保存結果
- 實作本地快取
- 批次處理請求

### UI 響應優化

- 使用串流回應
- 顯示進度指示器
- 非同步處理長時間操作

## 📚 延伸閱讀

- [Streamlit 官方文檔](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [LangChain 文檔](https://python.langchain.com/)
- [FAISS 文檔](https://github.com/facebookresearch/faiss)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

---

**最後更新：** 2025年1月
