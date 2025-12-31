# 檢索增強生成 (Retrieval-Augmented Generation, RAG) 基礎

## 目錄
1. [RAG 流程與原理](#41-rag-流程與原理)
2. [文檔載入器與文件拆分](#42-文檔載入器與文件拆分)
3. [向量資料庫基礎與設置](#43-向量資料庫基礎與設置)
4. [實作示例](#44-實作示例)

---

## 4.1 RAG 流程與原理

### 什麼是 RAG？

RAG (Retrieval-Augmented Generation) 是一種結合資訊檢索和文本生成的技術，透過從外部知識庫檢索相關資訊來增強 LLM 的回答能力。

### 為什麼需要 RAG？

**LLM 的局限性**：
1. **知識截止**：只能回答訓練數據截止日期前的問題
2. **幻覺問題**：可能生成事實錯誤的內容
3. **缺乏專業知識**：對特定領域的深度知識有限
4. **無法訪問私有數據**：無法利用公司內部文檔

**RAG 的優勢**：
1. **實時更新**：可以檢索最新資訊
2. **事實準確**：基於檢索到的真實文檔
3. **領域專業化**：可添加專業領域知識
4. **可追溯性**：可以引用資料來源

### RAG 工作流程

```
1. 文檔準備階段（離線）：
   文檔 → 拆分 → 向量化 → 存儲到向量資料庫

2. 查詢階段（在線）：
   用戶查詢 → 向量化 → 向量檢索 → 相似度排序 → 選取 Top-K

3. 生成階段：
   檢索結果 + 用戶查詢 → 構建提示詞 → LLM 生成 → 返回答案
```

### 核心組件

#### 1. Embeddings (嵌入向量)

**定義**：將文本轉換為固定長度的數值向量，語義相似的文本會有相近的向量表示。

**常用模型**：
- **OpenAI Embeddings**：`text-embedding-ada-002`
- **Sentence Transformers**：`all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- **多語言模型**：`paraphrase-multilingual-MiniLM-L12-v2`
- **中文特化**：`shibing624/text2vec-base-chinese`

**特點**：
- 維度：384-1536
- 餘弦相似度計算
- 可捕捉語義關係

#### 2. 向量資料庫 (Vector Database)

**功能**：高效存儲和檢索向量數據

**檢索方法**：
- **暴力搜索 (Brute Force)**：精確但慢
- **近似最近鄰 (ANN)**：快速但近似
  - HNSW (Hierarchical Navigable Small World)
  - IVF (Inverted File Index)
  - FAISS

#### 3. 檢索器 (Retriever)

**類型**：
1. **密集檢索 (Dense Retrieval)**：使用向量相似度
2. **稀疏檢索 (Sparse Retrieval)**：BM25, TF-IDF
3. **混合檢索 (Hybrid)**：結合密集和稀疏

**相似度度量**：
- 餘弦相似度 (Cosine Similarity)
- 歐幾里得距離 (Euclidean Distance)
- 點積 (Dot Product)

---

## 4.2 文檔載入器與文件拆分

### 文檔載入

**常見格式**：
- PDF
- Word (DOC, DOCX)
- Markdown
- HTML
- JSON
- CSV
- 純文本

**載入工具**：
- **LangChain Document Loaders**
- **PyPDF2, pdfplumber**：PDF 處理
- **python-docx**：Word 文檔
- **BeautifulSoup**：HTML 解析

### 文本拆分策略

**為什麼需要拆分？**
- LLM 上下文長度限制
- 提高檢索精確度
- 減少噪音

**拆分方法**：

1. **固定長度拆分**：
   - 簡單但可能切斷語義
   - 適合結構簡單的文本

2. **遞歸字符拆分**：
   - 按段落、句子、單詞遞歸拆分
   - 保持語義完整性

3. **語義拆分**：
   - 基於主題或語義邊界
   - 效果最好但計算昂貴

**關鍵參數**：
- `chunk_size`：每個塊的大小（字符數或 token 數）
- `chunk_overlap`：塊之間的重疊（保證上下文連貫性）

**最佳實踐**：
- `chunk_size`: 200-1000 tokens
- `chunk_overlap`: 10-20% of chunk_size
- 保留文檔元數據（來源、頁碼等）

---

## 4.3 向量資料庫基礎與設置

### 主流向量資料庫比較

| 資料庫 | 類型 | 特點 | 適用場景 |
|--------|------|------|---------|
| **Chroma** | 嵌入式 | 輕量、易用、本地優先 | 原型開發、小規模應用 |
| **Pinecone** | 雲服務 | 完全托管、高性能、自動擴展 | 生產環境、大規模應用 |
| **Milvus** | 開源 | 高性能、分布式、功能豐富 | 企業級應用 |
| **Weaviate** | 開源 | 多模態、圖形化、RESTful API | 複雜查詢場景 |
| **Qdrant** | 開源 | Rust 實現、高性能 | 高吞吐量應用 |
| **FAISS** | 庫 | Facebook 開發、高效 | 研究、本地檢索 |

### Chroma 特點

- **簡單易用**：最小化配置
- **本地運行**：無需外部服務
- **持久化**：支持磁盤存儲
- **查詢靈活**：支持元數據過濾

### Pinecone 特點

- **完全托管**：無需維護基礎設施
- **高可用性**：自動備份和恢復
- **實時索引**：秒級更新
- **付費服務**：按使用量計費

### Milvus 特點

- **高性能**：億級向量毫秒檢索
- **混合搜索**：向量+標量過濾
- **可擴展**：水平擴展
- **雲原生**：Kubernetes 友好

---

## 4.4 實作示例

### 4.4.1 使用 Sentence Transformers 生成向量嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 載入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 文本列表
documents = [
    "機器學習是人工智慧的一個分支",
    "深度學習使用多層神經網絡",
    "自然語言處理幫助計算機理解人類語言",
    "計算機視覺處理圖像和視頻數據"
]

# 生成嵌入向量
embeddings = model.encode(documents)

print(f"嵌入向量形狀: {embeddings.shape}")
print(f"第一個文檔的向量 (前10維): {embeddings[0][:10]}")

# 計算相似度
from sklearn.metrics.pairwise import cosine_similarity

query = "什麼是神經網絡？"
query_embedding = model.encode([query])

# 計算與所有文檔的相似度
similarities = cosine_similarity(query_embedding, embeddings)[0]

# 排序並獲取最相似的文檔
most_similar_idx = np.argsort(similarities)[::-1]

print(f"\n查詢: {query}")
print("\n相似度排序:")
for idx in most_similar_idx:
    print(f"{documents[idx]} - 相似度: {similarities[idx]:.4f}")
```

### 4.4.2 文檔載入與拆分

```python
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 1. 載入單個 PDF
pdf_loader = PyPDFLoader("example.pdf")
pdf_documents = pdf_loader.load()

print(f"載入了 {len(pdf_documents)} 頁")
print(f"第一頁內容預覽: {pdf_documents[0].page_content[:200]}...")

# 2. 載入目錄中的所有文本文件
directory_loader = DirectoryLoader(
    "./documents",
    glob="**/*.txt",
    loader_cls=TextLoader
)
txt_documents = directory_loader.load()

print(f"\n載入了 {len(txt_documents)} 個文本文件")

# 3. 文本拆分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每塊最大字符數
    chunk_overlap=50,  # 重疊字符數
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# 拆分文檔
chunks = text_splitter.split_documents(pdf_documents + txt_documents)

print(f"\n拆分後共有 {len(chunks)} 個塊")
print(f"\n第一個塊:")
print(f"內容: {chunks[0].page_content[:200]}...")
print(f"元數據: {chunks[0].metadata}")
```

### 4.4.3 使用 Chroma 建立向量資料庫

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 準備數據
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 2. 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 3. 創建向量資料庫
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化路徑
)

print(f"向量資料庫已創建，包含 {len(chunks)} 個文檔塊")

# 4. 保存資料庫（可選，因為已指定 persist_directory）
vectorstore.persist()

# 5. 載入已有的資料庫
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 6. 相似度搜索
query = "什麼是機器學習？"
results = vectorstore.similarity_search(query, k=3)

print(f"\n查詢: {query}")
print("\n檢索結果:")
for i, doc in enumerate(results, 1):
    print(f"\n--- 結果 {i} ---")
    print(doc.page_content)
    print(f"元數據: {doc.metadata}")

# 7. 帶分數的相似度搜索
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

print("\n帶分數的檢索結果:")
for doc, score in results_with_scores:
    print(f"\n分數: {score:.4f}")
    print(doc.page_content[:200])
```

### 4.4.4 完整的 RAG 系統實現

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. 載入向量資料庫
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. 創建檢索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回前3個最相關的文檔
)

# 3. 定義提示詞模板
template = """使用以下上下文來回答問題。如果你不知道答案，就說不知道，不要試圖編造答案。

上下文: {context}

問題: {question}

詳細回答:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 4. 創建 LLM
llm = OpenAI(temperature=0)

# 5. 創建 RAG 鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# 6. 執行查詢
query = "深度學習的主要特點是什麼？"
result = qa_chain({"query": query})

print(f"問題: {query}")
print(f"\n答案: {result['result']}")
print("\n引用的文檔:")
for i, doc in enumerate(result['source_documents'], 1):
    print(f"\n來源 {i}:")
    print(doc.page_content[:200])
```

### 4.4.5 使用 LangChain 的簡化 RAG

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 載入並處理文檔
loader = TextLoader("knowledge.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 創建向量存儲
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 創建對話記憶
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 創建對話 RAG 鏈
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0),
    vectorstore.as_retriever(),
    memory=memory
)

# 多輪對話
questions = [
    "什麼是機器學習？",
    "它有哪些應用？",
    "深度學習和機器學習有什麼區別？"
]

for question in questions:
    result = qa({"question": question})
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

### 4.4.6 自定義 RAG 管道

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class CustomRAG:
    """自定義 RAG 系統"""

    def __init__(self, collection_name="knowledge_base"):
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # 初始化 Chroma 客戶端
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./custom_chroma_db"
        ))

        # 獲取或創建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, metadatas=None):
        """添加文檔到向量資料庫"""
        # 生成 ID
        ids = [f"doc_{i}" for i in range(len(documents))]

        # 生成嵌入
        embeddings = self.embedding_model.encode(documents).tolist()

        # 添加到集合
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )

        print(f"已添加 {len(documents)} 個文檔")

    def query(self, query_text, k=3):
        """查詢相關文檔"""
        # 生成查詢向量
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()

        # 檢索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results

    def rag_answer(self, question, llm_func):
        """使用 RAG 生成答案"""
        # 檢索相關文檔
        results = self.query(question, k=3)

        # 構建上下文
        context = "\n\n".join(results['documents'][0])

        # 構建提示詞
        prompt = f"""基於以下上下文回答問題。

上下文:
{context}

問題: {question}

回答:"""

        # 調用 LLM
        answer = llm_func(prompt)

        return {
            "answer": answer,
            "sources": results['documents'][0],
            "distances": results['distances'][0]
        }

# 使用示例
rag = CustomRAG()

# 添加知識
documents = [
    "機器學習是人工智慧的一個分支，通過數據學習模式。",
    "深度學習使用多層神經網絡來學習複雜的數據表示。",
    "自然語言處理幫助計算機理解和生成人類語言。"
]

rag.add_documents(documents)

# 查詢
results = rag.query("什麼是深度學習？")
print("\n檢索結果:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"\n{i+1}. (距離: {distance:.4f})")
    print(doc)
```

### 4.4.7 評估 RAG 系統

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)

# 準備評估數據
eval_data = {
    "question": ["什麼是機器學習？", "深度學習的應用有哪些？"],
    "answer": ["機器學習是...", "深度學習應用包括..."],
    "contexts": [
        ["機器學習是一種...", "它可以..."],
        ["深度學習在視覺...", "也用於NLP..."]
    ],
    "ground_truths": ["機器學習標準答案", "深度學習標準答案"]
}

# 評估
result = evaluate(
    eval_data,
    metrics=[
        faithfulness,  # 答案忠實度
        answer_relevancy,  # 答案相關性
        context_relevancy,  # 上下文相關性
        context_recall  # 上下文召回率
    ]
)

print("評估結果:")
print(result)
```

---

## 4.5 實戰範例程式

本資料夾包含多個可運行的 RAG 實作範例：

### 基礎範例

#### 1. 基礎嵌入向量 (`1_basic_embeddings.py`)

展示如何使用 Sentence Transformers 生成文本嵌入向量並計算相似度。

**功能**：
- 載入預訓練嵌入模型
- 生成文本嵌入向量
- 計算餘弦相似度
- 語義搜索演示
- 多語言嵌入向量

**運行**：
```bash
python 1_basic_embeddings.py
```

#### 2. 文檔處理與拆分 (`2_document_processing.py`)

展示如何載入和拆分不同格式的文檔。

**功能**：
- 文本拆分器（固定長度、遞歸拆分）
- 多格式文檔載入（TXT, JSON, Markdown）
- 元數據保留
- 智能斷句

**運行**：
```bash
python 2_document_processing.py
```

#### 3. 向量資料庫 (`3_vector_databases.py`)

展示如何使用不同的向量資料庫進行存儲和檢索。

**功能**：
- 簡單向量資料庫實現
- FAISS 向量資料庫
- 相似度搜索
- 持久化存儲
- 性能比較

**運行**：
```bash
python 3_vector_databases.py
```

### 進階範例

#### 4. 完整 RAG 系統 (`4_complete_rag_system.py`)

完整的端到端 RAG 問答系統實現。

**功能**：
- 文檔自動拆分和向量化
- 向量存儲和檢索
- 基於上下文的答案生成
- 來源引用和元數據
- 自定義知識庫構建

**運行**：
```bash
python 4_complete_rag_system.py
```

#### 5. 進階 RAG 技術 (`5_advanced_rag_techniques.py`)

展示進階的 RAG 優化技術。

**功能**：
- BM25 稀疏檢索
- 混合檢索（向量 + BM25）
- 重排序（Reranking）
- 查詢擴展
- 完整的進階 RAG 管道

**運行**：
```bash
python 5_advanced_rag_techniques.py
```

#### 6. 實戰問答系統 (`6_practical_qa_system.py`)

生產級別的多文檔問答系統。

**功能**：
- 多格式文檔載入（TXT, JSON, MD）
- 目錄批量載入
- 對話記憶功能
- 元數據過濾
- 置信度評分
- 統計信息

**運行**：
```bash
python 6_practical_qa_system.py
```

### 快速開始

1. **安裝依賴**：
```bash
pip install -r requirements.txt
```

2. **運行所有範例**：
```bash
chmod +x run_all_examples.sh
./run_all_examples.sh
```

3. **單獨運行某個範例**：
```bash
python <範例文件名>.py
```

### 範例特點

✅ **完全可運行**：所有範例都經過測試，可以直接運行
✅ **詳細註釋**：代碼包含詳細的中文註釋
✅ **逐步演示**：從基礎到進階，循序漸進
✅ **實用性強**：可以作為實際項目的起點
✅ **AI 輔助**：集成 AI 模型進行智能問答

### 進階擴展建議

基於這些範例，你可以進一步擴展：

1. **集成真實 LLM API**：
   - OpenAI GPT-4
   - Anthropic Claude
   - Google Gemini
   - 本地 Ollama 模型

2. **添加更多文件格式支持**：
   - PDF（使用 PyPDF2, pdfplumber）
   - Word（使用 python-docx）
   - Excel（使用 pandas）
   - HTML（使用 BeautifulSoup）

3. **實現更多進階功能**：
   - 多輪對話支持
   - 上下文壓縮
   - 自動摘要
   - 多模態檢索（文本 + 圖像）

4. **優化性能**：
   - 批量處理
   - 異步檢索
   - 緩存機制
   - 分布式部署

5. **添加用戶界面**：
   - Gradio Web UI
   - Streamlit 應用
   - FastAPI 後端
   - React 前端

---

## 參考資源

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG 論文: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [向量資料庫比較](https://github.com/erikbern/ann-benchmarks)
- [FAISS Documentation](https://faiss.ai/)
- [Hugging Face Models](https://huggingface.co/models)