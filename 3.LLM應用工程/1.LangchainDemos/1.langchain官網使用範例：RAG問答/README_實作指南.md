# RAG 問答系統實作指南

## 📚 目錄

1. [簡介](#簡介)
2. [環境設定](#環境設定)
3. [快速開始](#快速開始)
4. [範例說明](#範例說明)
5. [進階功能](#進階功能)

## 簡介

本資料夾包含 LangChain RAG（Retrieval-Augmented Generation）問答系統的完整實作範例。

### 什麼是 RAG？

RAG 是一種結合資訊檢索和生成式 AI 的技術：
1. **檢索（Retrieval）**：從文件庫中找出相關資訊
2. **增強（Augmented）**：將檢索到的資訊加入到提示詞中
3. **生成（Generation）**：使用 LLM 生成答案

### RAG 的優勢

- ✅ 減少幻覺（Hallucination）- LLM 基於實際文件回答
- ✅ 即時更新 - 不需重新訓練模型就能使用最新資訊
- ✅ 可追溯來源 - 能夠顯示答案的出處
- ✅ 領域專精 - 可針對特定領域建立知識庫

## 環境設定

### 1. 安裝套件

```bash
cd "3.LLM應用工程/1.LangchainDemos"
pip install -r requirements.txt
```

### 2. 設定環境變數

複製 `.env.example` 並重新命名為 `.env`：

```bash
cp .env.example .env
```

編輯 `.env` 檔案，填入你的 API 金鑰：

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 快速開始

### 方式 1: 使用 Python 腳本

```bash
cd "1.langchain官網使用範例：RAG問答"
python rag_demo_enhanced.py
```

### 方式 2: 使用 Jupyter Notebook

```bash
jupyter notebook
# 開啟 Q&A_with_RAG_簡介以及快速開始.ipynb
```

### 方式 3: 自己撰寫程式

```python
from rag_demo_enhanced import EnhancedRAG

# 1. 初始化 RAG 系統
rag = EnhancedRAG()

# 2. 載入文件
urls = ["https://example.com/article"]
rag.load_from_web(urls)

# 3. 提問
answer = rag.simple_query("你的問題？")
```

## 範例說明

### 範例檔案

| 檔案 | 說明 | 適合對象 |
|------|------|----------|
| `Q&A_with_RAG_簡介以及快速開始.ipynb` | 官方範例的繁體中文翻譯版 | 初學者 |
| `rag_demo_enhanced.py` | 增強版 RAG 系統（完整功能） | 進階使用者 |
| `README_實作指南.md` | 本文件 | 所有人 |

### 主要功能

#### 1. 簡單查詢

```python
rag = EnhancedRAG()
rag.load_from_web(["https://example.com"])
rag.simple_query("問題")
```

**特點：**
- 單次問答
- 顯示來源文件
- 適合快速查詢

#### 2. 對話式查詢

```python
rag.conversational_query("第一個問題")
rag.conversational_query("後續問題")  # 會參考前面的對話
```

**特點：**
- 保留對話歷史
- 支援上下文理解
- 適合連續討論

#### 3. 串流回應

```python
rag.stream_query("問題")  # 答案會即時顯示
```

**特點：**
- 即時回應
- 更好的使用者體驗
- 適合長篇回答

## 進階功能

### 自訂檢索參數

```python
# 調整檢索數量
rag.retriever = rag.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # 檢索 10 個文件
)
```

### 自訂模型參數

```python
# 使用不同的模型
rag = EnhancedRAG(
    model_name="gpt-4",
    temperature=0.3  # 降低隨機性
)
```

### 清除對話歷史

```python
rag.clear_history()  # 開始新的對話
```

## 常見問題

### Q1: 如何使用自己的文件？

目前 `rag_demo_enhanced.py` 支援網頁載入。要載入其他格式：

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()
rag._split_and_store(documents)

# 文字檔
loader = TextLoader("document.txt")
documents = loader.load()
rag._split_and_store(documents)
```

### Q2: 如何改善回答品質？

1. **調整分塊大小**：
```python
rag._split_and_store(documents, chunk_size=500, chunk_overlap=100)
```

2. **增加檢索數量**：
```python
search_kwargs={"k": 10}
```

3. **調整溫度參數**：
```python
rag = EnhancedRAG(temperature=0)  # 更確定性的回答
```

### Q3: 如何追蹤 LangSmith？

在 `.env` 中設定：

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=my-rag-project
```

然後在 https://smith.langchain.com 查看追蹤記錄。

## 架構圖

### RAG 流程

```
使用者問題
    ↓
向量化問題 (Embeddings)
    ↓
向量相似度搜尋 (Vector Store)
    ↓
檢索相關文件 (Retriever)
    ↓
組合問題 + 文件 (Prompt)
    ↓
LLM 生成答案
    ↓
返回答案給使用者
```

### 系統元件

```
EnhancedRAG
├── LLM (ChatOpenAI)
├── Embeddings (OpenAIEmbeddings)
├── VectorStore (Chroma)
├── Retriever
├── ChatHistory
└── Chains
    ├── Simple RAG Chain
    ├── Conversational Chain
    └── Streaming Chain
```

## 下一步

1. **探索其他範例**：查看 `code_understanding_ipynb繁中翻譯.ipynb` 了解程式碼分析
2. **建立自己的應用**：使用 `rag_demo_enhanced.py` 作為基礎
3. **進階主題**：
   - 多模態 RAG（圖片 + 文字）
   - 混合檢索（關鍵字 + 向量）
   - 重排序（Reranking）

## 參考資源

- [LangChain 官方文件](https://python.langchain.com/)
- [RAG 最佳實踐](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangSmith 追蹤](https://docs.smith.langchain.com/)

## 授權

本範例基於 LangChain 官方範例改編，供學習使用。
