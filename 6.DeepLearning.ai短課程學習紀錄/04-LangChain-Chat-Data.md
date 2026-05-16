# LangChain: Chat with Your Data

## 📋 課程概述

本課程深入探討如何使用 LangChain 建立與私有資料對話的 RAG（Retrieval-Augmented Generation）系統。

### 課程目標
- 掌握文檔載入與處理技術
- 理解向量嵌入和語意搜尋
- 學習文字分割的最佳實踐
- 建構完整的 RAG 問答系統

### 適合對象
- 已完成 LangChain 基礎課程
- 想要建立文檔問答系統的開發者
- 企業 AI 應用開發者

### 課程時長
約 1 小時

## 🎯 RAG 系統架構

```
┌─────────────────────────────────────────┐
│          RAG 系統流程                   │
├─────────────────────────────────────────┤
│  1. 文檔載入 (Document Loaders)         │
│     ↓                                   │
│  2. 文字分割 (Text Splitters)           │
│     ↓                                   │
│  3. 向量嵌入 (Embeddings)               │
│     ↓                                   │
│  4. 向量儲存 (Vector Stores)            │
│     ↓                                   │
│  5. 檢索器 (Retrievers)                 │
│     ↓                                   │
│  6. 問答鏈 (QA Chain)                   │
└─────────────────────────────────────────┘
```

## 1️⃣ 文檔載入（Document Loaders）

### 支援的文檔格式

LangChain 支援超過 80 種文檔載入器。

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
    DirectoryLoader
)
import os

# PDF 載入器
pdf_loader = PyPDFLoader("document.pdf")
pdf_pages = pdf_loader.load()

print(f"載入了 {len(pdf_pages)} 頁")
print(f"第一頁內容預覽：\n{pdf_pages[0].page_content[:200]}")
print(f"元資料：{pdf_pages[0].metadata}")
```

### 各種載入器範例

#### 1. 文字檔載入

```python
# TXT 檔案
txt_loader = TextLoader("notes.txt", encoding='utf-8')
txt_docs = txt_loader.load()

print(f"文檔數量：{len(txt_docs)}")
print(f"內容：{txt_docs[0].page_content}")
```

#### 2. CSV 載入

```python
# CSV 檔案
csv_loader = CSVLoader("data.csv", encoding='utf-8')
csv_docs = csv_loader.load()

# 每一行都會成為一個文檔
for doc in csv_docs[:3]:
    print(doc.page_content)
    print("---")
```

#### 3. Markdown 載入

```python
# Markdown 檔案
md_loader = UnstructuredMarkdownLoader("README.md")
md_docs = md_loader.load()

print(md_docs[0].page_content)
```

#### 4. 網頁載入

```python
# 從網頁載入
web_loader = WebBaseLoader("https://example.com/article")
web_docs = web_loader.load()

print(f"網頁標題：{web_docs[0].metadata.get('title')}")
print(f"內容長度：{len(web_docs[0].page_content)}")
```

#### 5. 目錄批次載入

```python
# 載入整個目錄的文件
loader = DirectoryLoader(
    './documents',
    glob="**/*.md",  # 只載入 markdown 檔案
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True
)

docs = loader.load()
print(f"共載入 {len(docs)} 個文檔")
```

### Notion 資料庫載入

```python
from langchain_community.document_loaders import NotionDirectoryLoader

# 從 Notion 匯出的資料夾載入
notion_loader = NotionDirectoryLoader("notion_export")
notion_docs = notion_loader.load()

print(f"Notion 頁面數：{len(notion_docs)}")
```

## 2️⃣ 文字分割（Text Splitting）

文字分割是 RAG 系統中最重要的步驟之一。

### 為什麼需要分割？

1. **模型限制**：LLM 有上下文長度限制
2. **相關性**：較小的文字塊更容易匹配查詢
3. **成本控制**：減少不必要的 token 使用

### CharacterTextSplitter

最基本的分割器，按字元數分割。

```python
from langchain.text_splitters import CharacterTextSplitter

text = """
台灣是位於東亞的島嶼，面積約 36,000 平方公里。
台灣有豐富的自然景觀，包括高山、森林、海岸線等。
台北 101 曾是世界最高的建築物之一。
台灣的夜市文化非常著名，各地都有特色夜市。
台灣美食享譽國際，小籠包、珍珠奶茶都是代表性食物。
"""

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,     # 每個塊的大小
    chunk_overlap=20,   # 塊之間的重疊
    length_function=len
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print(f"長度：{len(chunk)}\n")
```

### RecursiveCharacterTextSplitter（推薦）

智慧分割，優先使用自然分隔符。

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

# 繁體中文友善的分隔符設定
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=[
        "\n\n",   # 段落
        "\n",     # 行
        "。",     # 句號
        "！",     # 驚嘆號
        "？",     # 問號
        "；",     # 分號
        "，",     # 逗號
        " ",      # 空格
        ""        # 字元
    ]
)

# 載入並分割文檔
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load()

# 分割所有頁面
all_splits = text_splitter.split_documents(pages)

print(f"原始頁數：{len(pages)}")
print(f"分割後塊數：{len(all_splits)}")
print(f"\n第一塊：\n{all_splits[0].page_content}")
print(f"\n元資料：{all_splits[0].metadata}")
```

### TokenTextSplitter

按 token 數量分割（更精確的控制）。

```python
from langchain.text_splitters import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=100,    # token 數量
    chunk_overlap=10
)

texts = token_splitter.split_text(text)

for i, chunk in enumerate(texts):
    print(f"Chunk {i + 1}:\n{chunk}\n")
```

### MarkdownHeaderTextSplitter

根據 Markdown 標題結構分割。

```python
from langchain.text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# 台灣旅遊指南

## 北部地區

### 台北市
台北是台灣的首都，有許多著名景點。

#### 台北 101
世界知名的摩天大樓。

### 新北市
包圍台北市的直轄市。

## 中部地區

### 台中市
台灣第二大城市。

### 南投縣
台灣唯一不靠海的縣份。
"""

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_splits = markdown_splitter.split_text(markdown_text)

for split in md_splits:
    print(f"內容：{split.page_content}")
    print(f"元資料：{split.metadata}\n")
```

## 3️⃣ 向量嵌入（Embeddings）

將文字轉換為向量表示，用於語意搜尋。

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 或 text-embedding-3-large
)

# 嵌入單個文字
text = "台灣是一個美麗的島嶼"
vector = embeddings.embed_query(text)

print(f"向量維度：{len(vector)}")
print(f"向量前 5 個元素：{vector[:5]}")

# 嵌入多個文字
texts = [
    "台灣的夜市很有名",
    "台北 101 是著名地標",
    "機器學習是人工智慧的一個分支"
]

vectors = embeddings.embed_documents(texts)
print(f"\n嵌入了 {len(vectors)} 個文字")

# 計算相似度
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

query = "台灣的美食文化"
query_vector = embeddings.embed_query(query)

print(f"\n查詢：{query}")
for i, text in enumerate(texts):
    similarity = cosine_similarity(query_vector, vectors[i])
    print(f"與 '{text}' 的相似度：{similarity:.4f}")
```

### Hugging Face Embeddings（免費替代方案）

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# 使用開源模型
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

text = "這是一個測試文字"
vector = hf_embeddings.embed_query(text)

print(f"向量維度：{len(vector)}")
```

## 4️⃣ 向量儲存（Vector Stores）

### Chroma（本地向量資料庫）

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

# 1. 載入文檔
loader = TextLoader("taiwan_info.txt", encoding='utf-8')
documents = loader.load()

# 2. 分割文字
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 3. 建立向量儲存
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化儲存
)

print(f"向量資料庫已建立，包含 {vectorstore._collection.count()} 個向量")

# 4. 相似度搜尋
query = "台灣有哪些著名景點？"
docs = vectorstore.similarity_search(query, k=3)

print(f"\n查詢：{query}\n")
for i, doc in enumerate(docs, 1):
    print(f"結果 {i}:")
    print(doc.page_content)
    print(f"來源：{doc.metadata}\n")
```

### 載入已存在的向量資料庫

```python
# 從磁碟載入
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 搜尋
results = vectorstore.similarity_search("查詢文字", k=5)
```

### 相似度搜尋與評分

```python
# 返回相似度分數
docs_with_scores = vectorstore.similarity_search_with_score(
    "台灣的美食",
    k=3
)

for doc, score in docs_with_scores:
    print(f"分數：{score:.4f}")
    print(f"內容：{doc.page_content[:100]}...\n")
```

### MMR 搜尋（最大邊際相關性）

平衡相關性和多樣性。

```python
# MMR 搜尋
docs = vectorstore.max_marginal_relevance_search(
    "台灣旅遊",
    k=5,
    fetch_k=20,  # 先取出 20 個候選
    lambda_mult=0.5  # 0=最大多樣性，1=最大相關性
)

for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content[:100]}...")
```

### FAISS（Facebook AI Similarity Search）

更快速的向量搜尋。

```python
from langchain_community.vectorstores import FAISS

# 建立 FAISS 向量資料庫
faiss_db = FAISS.from_documents(splits, embeddings)

# 儲存到磁碟
faiss_db.save_local("faiss_index")

# 載入
faiss_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 搜尋
results = faiss_db.similarity_search("查詢", k=3)
```

## 5️⃣ 檢索器（Retrievers）

檢索器是向量儲存的抽象介面。

### 基本檢索器

```python
# 從向量儲存建立檢索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 使用檢索器
docs = retriever.invoke("台灣的文化特色")

for doc in docs:
    print(doc.page_content)
    print("---")
```

### MMR 檢索器

```python
# 使用 MMR 檢索
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

docs = mmr_retriever.invoke("台灣景點")
```

### 相似度閾值檢索器

```python
# 只返回相似度超過閾值的結果
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # 最小相似度
        "k": 5
    }
)

docs = threshold_retriever.invoke("查詢文字")
```

### 自訂檢索器

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# 基礎檢索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM 壓縮器
llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# 壓縮檢索器（只返回相關部分）
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 使用
compressed_docs = compression_retriever.invoke(
    "台灣最高的山是什麼？"
)

for doc in compressed_docs:
    print(doc.page_content)
    print("---")
```

## 6️⃣ 問答鏈（Question Answering）

### RetrievalQA

最常用的問答鏈。

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # stuff, map_reduce, refine, map_rerank
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# 提問
question = "台灣的首都是哪裡？有哪些著名景點？"
result = qa_chain.invoke({"query": question})

print(f"問題：{question}")
print(f"\n回答：{result['result']}")
print(f"\n來源文檔數：{len(result['source_documents'])}")

for i, doc in enumerate(result['source_documents'], 1):
    print(f"\n來源 {i}:")
    print(doc.page_content[:200])
```

### ConversationalRetrievalChain

支援多輪對話的問答鏈。

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 建立記憶
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 建立對話問答鏈
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)

# 多輪對話
print("對話 1:")
result1 = conversation_chain.invoke({"question": "台灣最高的山是什麼？"})
print(result1["answer"])

print("\n對話 2:")
result2 = conversation_chain.invoke({"question": "它有多高？"})
print(result2["answer"])

print("\n對話 3:")
result3 = conversation_chain.invoke({"question": "一般人可以爬嗎？"})
print(result3["answer"])
```

### 使用 LCEL 建立自訂 RAG 鏈

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 自訂提示模板
template = """
你是一位專業的台灣文化專家。請根據以下上下文回答問題。

上下文：
{context}

問題：{question}

回答時請：
1. 只使用上下文中的資訊
2. 如果上下文中沒有相關資訊，請說明
3. 使用繁體中文回答
4. 保持友善和專業的語氣

回答：
"""

prompt = ChatPromptTemplate.from_template(template)

# 建立檢索鏈
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("台灣有哪些特色美食？")
print(answer)
```

## 💡 完整 RAG 系統範例

```python
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class TaiwanKnowledgeBot:
    def __init__(self, docs_directory, persist_directory="./taiwan_db"):
        """初始化台灣知識機器人"""
        self.docs_directory = docs_directory
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.conversation_chain = None

        # 初始化
        self.setup()

    def setup(self):
        """設定系統"""
        # 檢查是否已有向量資料庫
        if os.path.exists(self.persist_directory):
            print("載入現有的向量資料庫...")
            self.load_vectorstore()
        else:
            print("建立新的向量資料庫...")
            self.create_vectorstore()

        # 建立對話鏈
        self.create_conversation_chain()

    def create_vectorstore(self):
        """建立向量資料庫"""
        # 載入文檔
        loader = DirectoryLoader(
            self.docs_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        print(f"載入了 {len(documents)} 個文檔")

        # 分割文字
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"分割成 {len(splits)} 個片段")

        # 建立向量儲存
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        print("向量資料庫建立完成")

    def load_vectorstore(self):
        """載入向量資料庫"""
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )

    def create_conversation_chain(self):
        """建立對話鏈"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

    def ask(self, question):
        """提問"""
        if not self.conversation_chain:
            return "系統尚未初始化"

        result = self.conversation_chain.invoke({"question": question})

        return {
            "answer": result["answer"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }

    def chat(self):
        """互動式對話"""
        print("=" * 60)
        print("台灣知識機器人已啟動！")
        print("輸入 'quit' 或 'exit' 結束對話")
        print("=" * 60)

        while True:
            question = input("\n👤 您：")

            if question.lower() in ['quit', 'exit', '退出', '結束']:
                print("👋 再見！")
                break

            if not question.strip():
                continue

            result = self.ask(question)

            print(f"\n🤖 機器人：{result['answer']}")

            if result['sources']:
                print(f"\n📚 參考來源（{len(result['sources'])} 個）：")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['content'][:100]}...")

# 使用範例
if __name__ == "__main__":
    # bot = TaiwanKnowledgeBot("./taiwan_docs")
    # bot.chat()
    pass
```

## ✅ 最佳實踐

### 1. 文字分割策略
- 使用 RecursiveCharacterTextSplitter
- chunk_size: 500-1000 字元
- chunk_overlap: 10-20% 的 chunk_size
- 針對中文設定合適的分隔符

### 2. 向量儲存選擇
- **開發測試**：Chroma（簡單易用）
- **生產環境**：Pinecone、Weaviate（雲端服務）
- **高效能**：FAISS（本地部署）

### 3. 檢索優化
- 使用 MMR 增加結果多樣性
- 設定合理的 k 值（3-5 個結果）
- 考慮使用混合搜尋（關鍵字+向量）

### 4. 提示工程
- 明確指示只使用提供的上下文
- 要求引用來源
- 處理「我不知道」的情況

## 📚 延伸學習

- **進階 RAG**：查詢重寫、混合搜尋、重新排序
- **評估系統**：使用 RAGAS 評估 RAG 效能
- **優化技術**：Parent Document Retriever、Self-Query Retriever

---

**課程連結**：[DeepLearning.ai - LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)

**完成日期**：2025-01-17
