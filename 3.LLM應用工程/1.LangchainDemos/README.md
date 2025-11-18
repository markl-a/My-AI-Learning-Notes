# LangChain Demos - LLM 應用實戰範例集

歡迎來到 LangChain Demos！這是一個全面的 LangChain 應用範例集合，涵蓋從基礎到進階的各種實際應用場景。

## 📚 專案目標

打造實用的 LLM 應用範例，使用 LangChain、OpenAI 以及其他流行的 AI 框架，幫助開發者快速上手 AI 應用開發。

## 🎯 涵蓋領域

### 1. 程式開發 💻
- ✅ API 呼叫生成程式碼
- ✅ 自動編譯與修復程式碼
- ✅ 調用本地程式庫撰寫函數
- ✅ 程式碼理解與分析
- ✅ 單元測試自動生成

### 2. 文字撰寫與處理 📝
- ✅ 網路資訊搜集與總結
- ✅ 文件閱讀與資訊提取
- ✅ 多語言文件翻譯
- ⏳ 郵件與訊息自動化（規劃中）

### 3. AI Agents 🤖
- ✅ ReAct Agent（推理+行動）
- ✅ Tool-Calling Agent（工具調用）
- ✅ Search Agent（網路搜尋）
- ✅ Conversational Agent（對話記憶）

### 4. RAG 應用 🔍
- ✅ 文件問答系統
- ✅ 對話式檢索
- ✅ 串流回應
- ✅ 來源追蹤

### 5. 進階應用 🚀
- ⏳ 圖像與 UI 生成（規劃中）
- ⏳ 個人時間行為分析（規劃中）
- ⏳ 語音與音樂生成（規劃中）

## 📁 專案結構

```
1.LangchainDemos/
├── README.md                           # 本文件
├── requirements.txt                    # Python 依賴套件
├── .env.example                        # 環境變數範例
├── .gitignore                          # Git 忽略檔案
├── utils.py                            # 通用工具模組
│
├── 1.langchain官網使用範例：RAG問答/
│   ├── README_實作指南.md              # RAG 詳細指南
│   ├── rag_demo_enhanced.py            # 增強版 RAG 範例
│   ├── Q&A_with_RAG_簡介以及快速開始.ipynb  # Jupyter 教學
│   └── [圖片檔案]
│
├── 2.AI Agents範例/
│   ├── README.md                       # Agent 使用指南
│   └── agent_demo.py                   # 完整 Agent 範例
│       ├── ReAct Agent
│       ├── Tool-Calling Agent
│       ├── Search Agent
│       └── Conversational Agent
│
├── 3.程式開發助手/
│   ├── README.md                       # 開發助手指南
│   └── code_assistant.py               # 程式碼助手
│       ├── 程式碼生成
│       ├── 程式碼修復
│       ├── 程式碼解釋
│       ├── 程式碼優化
│       └── 自動除錯
│
├── 4.文件處理與資訊提取/
│   ├── README.md                       # 文件處理指南
│   └── document_processor.py           # 文件處理器
│       ├── 文件載入（PDF/Word/TXT/網頁）
│       ├── 文件總結
│       ├── 資訊提取
│       ├── 文件問答
│       └── 文件翻譯
│
└── code_understanding_ipynb繁中翻譯.ipynb  # 程式碼理解範例
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 切換到專案目錄
cd "3.LLM應用工程/1.LangchainDemos"

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入你的 API Keys
```

### 2. 執行範例

```bash
# RAG 問答系統
cd "1.langchain官網使用範例：RAG問答"
python rag_demo_enhanced.py

# AI Agents
cd "../2.AI Agents範例"
python agent_demo.py

# 程式開發助手
cd "../3.程式開發助手"
python code_assistant.py

# 文件處理
cd "../4.文件處理與資訊提取"
python document_processor.py
```

### 3. 使用 Jupyter Notebook

```bash
jupyter notebook
# 開啟任何 .ipynb 檔案開始學習
```

## 💡 核心功能展示

### RAG 問答系統

```python
from rag_demo_enhanced import EnhancedRAG

rag = EnhancedRAG()
rag.load_from_web(["https://example.com/article"])
answer = rag.simple_query("這篇文章的主要觀點是什麼？")
```

### AI Agent

```python
from agent_demo import ConversationalAgent

agent = ConversationalAgent()
agent.chat("台北今天天氣如何？")
agent.chat("那邊溫度是幾度？")  # 會記住在討論台北
```

### 程式碼助手

```python
from code_assistant import CodeAssistant

assistant = CodeAssistant()
result = assistant.generate_code(
    "建立一個計算費氏數列的函數",
    include_tests=True
)
print(result["code"])
```

### 文件處理

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
documents = processor.load_document("report.pdf")
summary = processor.summarize(documents[0].page_content)
```

## 🛠️ 技術棧

- **LLM**: OpenAI GPT-3.5/GPT-4
- **框架**: LangChain, LangSmith
- **向量資料庫**: Chroma
- **文件處理**: PyPDF, python-docx, BeautifulSoup
- **Agent 工具**: DuckDuckGo Search, Wikipedia
- **程式語言**: Python 3.8+

## 📖 學習路徑建議

### 初學者
1. 從 **RAG 問答系統** 開始 → 理解基本概念
2. 嘗試 **文件處理** → 實際應用
3. 學習 **AI Agents** → 進階功能

### 進階開發者
1. 研究 **程式開發助手** → 了解複雜應用
2. 自訂 **Agent 工具** → 擴展功能
3. 優化 **RAG 系統** → 提升效能

## 🎓 實用資源

### 官方文件
- [LangChain 官方文件](https://python.langchain.com/)
- [OpenAI API 文件](https://platform.openai.com/docs/)
- [LangSmith 追蹤](https://docs.smith.langchain.com/)

### 推薦閱讀
- [RAG 最佳實踐](https://python.langchain.com/docs/use_cases/question_answering/)
- [Agent 設計模式](https://python.langchain.com/docs/modules/agents/)
- [Prompt Engineering 指南](https://www.promptingguide.ai/)

## ⚙️ 環境變數說明

```bash
# OpenAI API（必要）
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith 追蹤（選用）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langchain-demos

# 搜尋 API（選用，用於 Agent）
GOOGLE_API_KEY=your_google_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

## 🤝 如何貢獻

歡迎提出 Issue 或 Pull Request！

### 貢獻指南
1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 📝 未來計畫

- [ ] 加入更多實際應用範例
- [ ] 支援本地 LLM（Ollama, llama.cpp）
- [ ] 增加多模態範例（圖像+文字）
- [ ] 建立完整的專案範本
- [ ] 加入效能優化指南
- [ ] 製作視頻教學

## ⚠️ 注意事項

1. **API 成本**: 使用 OpenAI API 會產生費用，請注意使用量
2. **安全性**: 不要將 API Key 提交到 Git
3. **測試**: AI 生成的程式碼需要人工審查和測試
4. **隱私**: 不要將敏感資訊傳送給 AI

## 📄 授權

本專案採用 MIT 授權條款。

## 💬 聯絡方式

如有問題或建議，歡迎：
- 開啟 [Issue](../../issues)
- 發送 [Pull Request](../../pulls)

## 🌟 致謝

感謝 LangChain 社群和所有貢獻者！

本專案主要用於學習和研究目的，部分範例改編自官方文件和社群分享。

---

**開始你的 AI 應用開發之旅！** 🚀

如果這個專案對你有幫助，請給它一個 ⭐️
