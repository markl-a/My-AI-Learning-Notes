# 互動式演示 Demos

這個目錄包含了多個互動式演示，展示如何使用本項目中學習的技術構建實際應用。

## 📁 目錄結構

```
demos/
├── gradio/          # Gradio 演示
│   └── rag_demo.py  # RAG 系統演示
├── streamlit/       # Streamlit 演示
│   └── llm_chat.py  # LLM 聊天應用
└── chainlit/        # Chainlit 演示
    └── agent_demo.py # Agent 系統演示
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 從項目根目錄
pip install -r requirements-llm.txt

# 或安裝特定依賴
pip install gradio streamlit chainlit langchain langchain-openai chromadb
```

### 2. 設置環境變量

```bash
# 複製環境變量模板
cp .env.example .env

# 編輯 .env 文件，添加您的 API 金鑰
# OPENAI_API_KEY=your-key-here
```

### 3. 運行演示

```bash
# Gradio RAG 演示
python demos/gradio/rag_demo.py

# Streamlit LLM Chat
streamlit run demos/streamlit/llm_chat.py

# Chainlit Agent 演示
chainlit run demos/chainlit/agent_demo.py
```

## 📚 演示說明

### 1️⃣ Gradio RAG 演示

**功能：**
- 📤 文檔上傳（支援 TXT、PDF）
- 🔍 向量化和索引
- 💬 基於文檔的問答
- 📊 顯示相關文檔片段

**訪問：** http://localhost:7860

**技術棧：**
- Gradio (UI)
- LangChain (RAG Framework)
- ChromaDB (Vector Store)
- OpenAI / Ollama (LLM)

**使用場景：**
- 文檔問答系統
- 知識庫查詢
- 研究輔助工具

---

### 2️⃣ Streamlit LLM Chat

**功能：**
- 💬 多模型支持（OpenAI、Ollama）
- 🎛️ 參數調整（Temperature、Max Tokens）
- 💾 對話歷史管理
- 📥 對話導出

**訪問：** http://localhost:8501

**技術棧：**
- Streamlit (UI)
- LangChain (LLM Integration)
- OpenAI API / Ollama

**使用場景：**
- AI 聊天助手
- 代碼輔助工具
- 學習輔導系統

---

### 3️⃣ Chainlit Agent 演示

**功能：**
- 🤖 Agent 工作流
- 🔧 工具調用（搜尋、計算等）
- 🧠 推理過程可視化
- 📝 任務分解和執行

**訪問：** http://localhost:8000

**技術棧：**
- Chainlit (UI)
- LangGraph (Agent Framework)
- LangChain Tools

**使用場景：**
- 複雜任務自動化
- 研究助手
- 數據分析工具

---

## ⚙️ 配置選項

### 使用本地模型（Ollama）

如果您沒有 OpenAI API 金鑰，可以使用 Ollama 運行本地模型：

```bash
# 1. 安裝 Ollama
# 訪問 https://ollama.ai

# 2. 啟動 Ollama
ollama serve

# 3. 下載模型
ollama pull llama3.2
ollama pull mistral

# 4. 運行演示（會自動檢測並使用 Ollama）
python demos/gradio/rag_demo.py
```

### 自定義配置

每個演示都支持通過環境變量或配置文件自定義：

```bash
# .env 文件
OPENAI_API_KEY=your-key
DEFAULT_MODEL=gpt-4o-mini
VECTOR_DB_PATH=./custom_db
MAX_TOKENS=2000
TEMPERATURE=0.7
```

---

## 📖 詳細文檔

### RAG 演示使用指南

1. **準備文檔**
   - 支援格式：.txt, .pdf
   - 建議大小：每個文件 < 10MB
   - 編碼：UTF-8

2. **上傳步驟**
   - 點擊"選擇文檔"
   - 選擇一個或多個文件
   - 點擊"上傳並處理"
   - 等待處理完成（顯示綠色勾號）

3. **提問技巧**
   - 明確具體的問題
   - 可以要求總結、提取信息
   - 支援多輪對話
   - 系統會顯示相關文檔片段

4. **示例問題**
   - "這篇文檔的主要內容是什麼？"
   - "請總結關鍵要點"
   - "文檔中提到了哪些重要概念？"
   - "關於X的部分說了什麼？"

### LLM Chat 使用指南

1. **選擇模型**
   - OpenAI：gpt-4, gpt-3.5-turbo
   - Ollama：llama3.2, mistral, phi3

2. **調整參數**
   - **Temperature (0-2)**
     - 0.0-0.3：精確、一致的回答
     - 0.4-0.7：平衡
     - 0.8-2.0：創造性、多樣性

   - **Max Tokens (100-4000)**
     - 100-500：簡短回答
     - 1000-2000：標準對話
     - 3000-4000：長文生成

3. **系統提示詞**
   - 定義 AI 的角色和行為
   - 例如："你是一個 Python 專家"
   - 影響所有後續對話

4. **對話管理**
   - 清除：刪除所有歷史
   - 導出：保存為 Markdown 文件
   - 統計：查看對話輪數

---

## 🛠️ 開發指南

### 創建新演示

1. **選擇框架**
   ```bash
   # Gradio - 快速原型
   # Streamlit - 數據應用
   # Chainlit - Agent 系統
   ```

2. **項目結構**
   ```
   demos/your_demo/
   ├── app.py              # 主應用
   ├── requirements.txt    # 依賴
   ├── README.md           # 說明
   └── config.yaml         # 配置（可選）
   ```

3. **最佳實踐**
   - 使用環境變量管理敏感信息
   - 提供清晰的錯誤消息
   - 添加使用說明和示例
   - 實現錯誤處理和日誌記錄
   - 支援本地和雲端模型

### 貢獻演示

歡迎貢獻新的演示！請查看 [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 🔧 故障排除

### 常見問題

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements-llm.txt
   ```

2. **OpenAI API 錯誤**
   - 檢查 API 金鑰是否正確
   - 確認帳戶有餘額
   - 檢查網絡連接

3. **Ollama 連接失敗**
   ```bash
   # 檢查 Ollama 是否運行
   curl http://localhost:11434/api/tags

   # 如果沒有運行，啟動它
   ollama serve
   ```

4. **ChromaDB 錯誤**
   ```bash
   # 刪除舊數據庫
   rm -rf demos/gradio/chroma_db

   # 重新運行應用
   python demos/gradio/rag_demo.py
   ```

5. **端口被佔用**
   ```bash
   # 查找佔用端口的進程
   lsof -i :7860

   # 或使用不同端口
   python app.py --port 7861
   ```

---

## 📞 獲取幫助

- 📚 查看[主要文檔](../README.md)
- 🐛 報告[問題](https://github.com/yourusername/My-AI-Learning-Notes/issues)
- 💬 加入[討論](https://github.com/yourusername/My-AI-Learning-Notes/discussions)

---

## 📄 許可證

這些演示與主項目使用相同的 MIT 許可證。

---

**享受探索！🎉**
