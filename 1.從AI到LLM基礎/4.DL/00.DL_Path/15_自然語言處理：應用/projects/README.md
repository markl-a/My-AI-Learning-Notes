# NLP 實戰項目範例

> 三個完整的端到端項目，從原型到生產部署

## 🎯 項目概覽

本目錄包含三個漸進式的 NLP 實戰項目：

| 項目 | 難度 | 技術棧 | 學習重點 |
|------|------|--------|----------|
| [情感分析 API](#1-情感分析-api) | ⭐⭐ | FastAPI, DistilBERT, Docker | API 開發、模型部署 |
| [智能聊天機器人](#2-智能聊天機器人) | ⭐⭐⭐ | Rasa, BERT, Streamlit | 對話管理、意圖識別 |
| [文檔問答系統 (RAG)](#3-文檔問答系統-rag) | ⭐⭐⭐⭐ | LangChain, FAISS, LLM | 向量檢索、RAG 架構 |

---

## 1. 情感分析 API

### 📝 項目描述

構建一個生產級別的情感分析 REST API，支持：
- 單條和批量文本分類
- 實時推理（延遲 <100ms）
- Docker 容器化部署
- API 文檔和監控

### 🎯 學習目標

- ✅ FastAPI 框架使用
- ✅ BERT 模型微調和優化
- ✅ RESTful API 設計
- ✅ Docker 容器化
- ✅ 性能優化（批處理、緩存）
- ✅ 日誌和監控

### 🏗️ 項目結構

```
sentiment_api/
├── README.md                 # 項目文檔
├── requirements.txt          # Python 依賴
├── Dockerfile               # Docker 配置
├── docker-compose.yml       # 多容器編排
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI 主程序
│   ├── models.py           # 數據模型
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py        # ML 模型加載和推理
│   │   └── preprocessing.py # 文本預處理
│   └── utils/
│       ├── __init__.py
│       └── logger.py       # 日誌配置
├── training/
│   ├── train.py            # 訓練腳本
│   ├── evaluate.py         # 評估腳本
│   └── notebooks/          # 訓練 notebooks
├── tests/
│   ├── test_api.py         # API 測試
│   └── test_model.py       # 模型測試
└── models/                 # 保存的模型文件
```

### 🚀 快速開始

```bash
# 1. 克隆項目
cd projects/sentiment_api

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 訓練模型（可選，也可使用預訓練）
python training/train.py

# 4. 啟動 API
python app/main.py

# 或使用 Docker
docker-compose up -d
```

### 📡 API 使用示例

```bash
# 單條預測
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# 批量預測
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible.", "Okay"]}'
```

### 🎓 關鍵實現

#### 1. FastAPI 應用

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Sentiment Analysis API")

class PredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = model.predict(request.text)
        return {"sentiment": result["label"], "confidence": result["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 2. 模型優化

```python
# 使用 ONNX 加速推理
import onnxruntime as ort

class OptimizedModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def predict(self, text):
        inputs = self.preprocess(text)
        outputs = self.session.run(None, inputs)
        return self.postprocess(outputs)
```

### 📊 性能基準

| 指標 | 目標 | 實際 |
|------|------|------|
| 延遲 (P50) | <100ms | 45ms |
| 延遲 (P99) | <200ms | 120ms |
| 吞吐量 | >100 QPS | 150 QPS |
| 準確率 | >90% | 92.5% |

---

## 2. 智能聊天機器人

### 📝 項目描述

構建一個功能完整的任務型聊天機器人，支持：
- 意圖識別和槽位填充
- 多輪對話管理
- 上下文理解
- Web UI 界面

### 🎯 學習目標

- ✅ Rasa 框架使用
- ✅ NLU 和對話管理
- ✅ 意圖分類和實體提取
- ✅ 對話流設計
- ✅ Streamlit UI 開發

### 🏗️ 項目結構

```
chatbot/
├── README.md
├── requirements.txt
├── config.yml              # Rasa 配置
├── domain.yml              # 對話域定義
├── credentials.yml         # 憑證配置
├── endpoints.yml           # 端點配置
├── data/
│   ├── nlu.yml            # NLU 訓練數據
│   ├── stories.yml        # 對話故事
│   └── rules.yml          # 對話規則
├── actions/
│   ├── __init__.py
│   └── actions.py         # 自定義動作
├── models/                # 訓練好的模型
├── tests/
│   └── test_stories.yml   # 測試對話
└── ui/
    └── app.py             # Streamlit UI
```

### 🚀 快速開始

```bash
# 1. 安裝 Rasa
pip install rasa

# 2. 訓練模型
rasa train

# 3. 啟動 Rasa 服務器
rasa run --enable-api

# 4. 啟動 Action Server（另一個終端）
rasa run actions

# 5. 啟動 UI（第三個終端）
streamlit run ui/app.py
```

### 💬 對話示例

```
User: 你好
Bot: 你好！我是智能助手，有什麼可以幫你？

User: 我想訂一張去北京的火車票
Bot: 好的，請問您什麼時候出發？

User: 明天
Bot: 明天幾點？上午還是下午？

User: 上午 9 點
Bot: 好的，已為您查詢明天上午 9 點左右的火車票，共找到 5 個班次...
```

### 🎓 關鍵實現

#### 1. 意圖分類（nlu.yml）

```yaml
nlu:
- intent: greet
  examples: |
    - 你好
    - 嗨
    - 早上好
    - 您好

- intent: book_ticket
  examples: |
    - 我想買[火車票](ticket_type)
    - 訂一張去[北京](destination)的票
    - 幫我訂[明天](date)的[飛機票](ticket_type)
```

#### 2. 對話管理（stories.yml）

```yaml
stories:
- story: book ticket flow
  steps:
  - intent: greet
  - action: utter_greet
  - intent: book_ticket
    entities:
    - destination
  - action: action_check_availability
  - action: utter_ask_date
  - intent: inform
    entities:
    - date
  - action: action_book_ticket
  - action: utter_booking_confirmed
```

#### 3. 自定義動作（actions.py）

```python
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionCheckAvailability(Action):
    def name(self):
        return "action_check_availability"

    def run(self, dispatcher, tracker, domain):
        destination = tracker.get_slot('destination')

        # 查詢邏輯
        available = check_tickets(destination)

        if available:
            message = f"找到 {len(available)} 個可用班次"
        else:
            message = "抱歉，暫無可用班次"

        dispatcher.utter_message(text=message)
        return []
```

---

## 3. 文檔問答系統 (RAG)

### 📝 項目描述

構建一個基於 RAG (Retrieval-Augmented Generation) 的文檔問答系統：
- 支持 PDF、Word、Markdown 等格式
- 向量化檢索
- 基於 LLM 生成答案
- 引用來源

### 🎯 學習目標

- ✅ RAG 架構設計
- ✅ 文檔解析和分塊
- ✅ 向量數據庫（FAISS/ChromaDB）
- ✅ Embedding 模型使用
- ✅ LLM 集成（OpenAI/本地模型）
- ✅ LangChain 框架

### 🏗️ 項目結構

```
doc_qa/
├── README.md
├── requirements.txt
├── .env.example           # 環境變量示例
├── app/
│   ├── __init__.py
│   ├── main.py           # Streamlit 主應用
│   ├── document_loader.py # 文檔加載器
│   ├── embeddings.py     # Embedding 管理
│   ├── vector_store.py   # 向量存儲
│   ├── retriever.py      # 檢索器
│   └── qa_chain.py       # QA 鏈
├── data/
│   ├── documents/        # 原始文檔
│   └── vector_db/        # 向量數據庫
├── notebooks/
│   └── exploration.ipynb # 數據探索
└── tests/
    └── test_qa.py
```

### 🚀 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設置環境變量
cp .env.example .env
# 編輯 .env 填入 OPENAI_API_KEY 等

# 3. 上傳文檔
# 將文檔放入 data/documents/

# 4. 構建向量索引
python app/build_index.py

# 5. 啟動應用
streamlit run app/main.py
```

### 🎓 關鍵實現

#### 1. 文檔加載和分塊

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_and_split(self, file_path):
        # 加載文檔
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 分塊
        chunks = self.text_splitter.split_documents(documents)

        return chunks
```

#### 2. 向量化和檢索

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def build_index(self, documents):
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )

    def search(self, query, k=5):
        docs = self.vector_store.similarity_search(query, k=k)
        return docs
```

#### 3. QA 鏈

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class QASystem:
    def __init__(self, vector_store):
        self.llm = OpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

    def answer(self, question):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"],
        }
```

### 🔍 使用示例

```python
# 初始化系統
qa_system = QASystem(vector_store)

# 提問
result = qa_system.answer("什麼是 Transformer?")

# 輸出
print(f"答案: {result['answer']}")
print(f"\n來源文檔:")
for doc in result['sources']:
    print(f"- {doc.metadata['source']}, 第 {doc.metadata['page']} 頁")
```

---

## 📚 通用學習資源

### 推薦閱讀順序

1. **入門**: 先完成 `sentiment_api` 項目
2. **進階**: 再嘗試 `chatbot` 項目
3. **高級**: 最後挑戰 `doc_qa` 項目

### 相關技能

- Python 異步編程
- Docker 和容器化
- RESTful API 設計
- 數據庫操作
- 前端基礎（Streamlit）

### 延伸項目

完成這三個項目後，可以嘗試：

1. **多語言支持** - 支持中英文等多語言
2. **模型蒸餾** - 優化模型大小和速度
3. **A/B 測試** - 對比不同模型效果
4. **監控系統** - 添加 Prometheus + Grafana
5. **分布式部署** - Kubernetes 部署

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 許可證

MIT License

---

**開始構建你的第一個 NLP 項目吧！🚀**
