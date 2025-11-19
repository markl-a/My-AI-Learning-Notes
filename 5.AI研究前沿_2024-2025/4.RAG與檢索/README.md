# RAG與檢索 - 10篇關鍵論文

> 2024-2025年檢索增強生成(RAG)的重大進展：GraphRAG、HyDE、混合檢索等核心技術

---

## 📋 論文列表

| # | 論文 | 機構 | 發布時間 | 代碼 | 影響力 |
|---|------|------|----------|------|--------|
| 1 | GraphRAG | Microsoft | 2024.04 | [GitHub](https://github.com/microsoft/graphrag) | ⭐⭐⭐⭐⭐ |
| 2 | HyDE | CMU | 2024 | [GitHub](https://github.com/texttron/hyde) | ⭐⭐⭐⭐ |
| 3 | BGE-Reranker | BAAI | 2024 | [HF](https://huggingface.co/BAAI/bge-reranker-large) | ⭐⭐⭐⭐ |
| 4 | Self-RAG | University of Washington | 2024 | [GitHub](https://github.com/AkariAsai/self-rag) | ⭐⭐⭐⭐ |
| 5 | RAPTOR | Stanford | 2024 | [GitHub](https://github.com/parthsarthi03/raptor) | ⭐⭐⭐⭐ |
| 6 | LongLLMLingua | Microsoft | 2024 | [GitHub](https://github.com/microsoft/LLMLingua) | ⭐⭐⭐ |
| 7 | Corrective RAG | Google | 2024 | [Paper](https://arxiv.org/abs/2401.15884) | ⭐⭐⭐⭐ |
| 8 | Multi-Vector Retriever | LangChain | 2024 | [Docs](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) | ⭐⭐⭐ |
| 9 | DSPy | Stanford | 2024 | [GitHub](https://github.com/stanfordnlp/dspy) | ⭐⭐⭐⭐ |
| 10 | RAG Fusion | Community | 2024 | [GitHub](https://github.com/Raudaschl/rag-fusion) | ⭐⭐⭐ |

---

## 核心技術與代碼實現

### 1. GraphRAG - 知識圖譜增強檢索

```python
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_relationships
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

# 加載知識圖譜
entities = read_indexer_entities("./output/artifacts")
relationships = read_indexer_relationships("./output/artifacts")

# 創建搜索引擎
llm = ChatOpenAI(model="gpt-4o-mini")
context = GlobalCommunityContext(entities, relationships)
search_engine = GlobalSearch(llm=llm, context=context)

# 執行查詢
result = await search_engine.search("What are the key AI trends?")
print(result.response)
```

### 2. HyDE - 假設文檔嵌入

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# HyDE提示模板
hyde_prompt = ChatPromptTemplate.from_template("""
Generate a hypothetical document that would answer this question:
{question}

Document:
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 生成假設文檔
def hyde_retrieval(question, vectorstore):
    # 1. 生成假設答案
    hypothetical_doc = llm.invoke(
        hyde_prompt.format(question=question)
    ).content

    # 2. 使用假設文檔進行檢索
    docs = vectorstore.similarity_search(hypothetical_doc, k=5)

    return docs

# 使用示例
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

results = hyde_retrieval("What is quantum computing?", vectorstore)
for doc in results:
    print(doc.page_content)
```

### 3. BGE-Reranker - 重排序優化

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加載Reranker
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

# 重排序
def rerank_documents(query, documents):
    pairs = [[query, doc] for doc in documents]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        scores = model(**inputs, return_dict=True).logits.view(-1,).float()

    # 按分數排序
    sorted_indices = torch.argsort(scores, descending=True)
    return [documents[i] for i in sorted_indices]

# 使用
query = "What is machine learning?"
docs = ["ML is...", "AI is...", "Deep learning..."]
ranked_docs = rerank_documents(query, docs)
```

### 4. Self-RAG - 自我反思檢索

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Self-RAG流程
def self_rag(question, vectorstore):
    # 1. 判斷是否需要檢索
    need_retrieval_prompt = ChatPromptTemplate.from_template("""
    Question: {question}
    Do you need to retrieve external information to answer this? (yes/no)
    """)

    need_retrieval = llm.invoke(
        need_retrieval_prompt.format(question=question)
    ).content.strip().lower()

    if "yes" in need_retrieval:
        # 2. 檢索文檔
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. 判斷文檔相關性
        relevance_prompt = ChatPromptTemplate.from_template("""
        Question: {question}
        Retrieved Context: {context}

        Is this context relevant? (yes/no)
        """)

        is_relevant = llm.invoke(
            relevance_prompt.format(question=question, context=context)
        ).content.strip().lower()

        if "yes" in is_relevant:
            # 4. 生成答案
            answer_prompt = ChatPromptTemplate.from_template("""
            Question: {question}
            Context: {context}

            Provide a detailed answer based on the context.
            """)

            answer = llm.invoke(
                answer_prompt.format(question=question, context=context)
            ).content

            # 5. 自我檢查
            check_prompt = ChatPromptTemplate.from_template("""
            Question: {question}
            Answer: {answer}

            Does this answer correctly address the question? (yes/no)
            """)

            is_correct = llm.invoke(
                check_prompt.format(question=question, answer=answer)
            ).content.strip().lower()

            if "yes" in is_correct:
                return answer

    # 直接回答
    return llm.invoke(question).content
```

### 5. RAPTOR - 遞歸摘要檢索

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import numpy as np

class RAPTOR:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings()

    def cluster_and_summarize(self, chunks, layer=0):
        """遞歸聚類和摘要"""
        if len(chunks) <= 10:  # 基礎情況
            return chunks

        # 1. 獲取嵌入
        embeddings = self.embeddings.embed_documents(chunks)

        # 2. 聚類（簡化版，實際使用KMeans等）
        num_clusters = len(chunks) // 5
        clusters = self._simple_cluster(embeddings, num_clusters)

        # 3. 為每個簇生成摘要
        summaries = []
        for cluster_chunks in clusters:
            summary = self.llm.invoke(f"""
            Summarize these related text chunks:
            {' '.join(cluster_chunks)}

            Summary:
            """).content
            summaries.append(summary)

        # 4. 遞歸處理
        return self.cluster_and_summarize(summaries, layer + 1)

    def _simple_cluster(self, embeddings, num_clusters):
        """簡化聚類"""
        embeddings_array = np.array(embeddings)
        # 實際應用中使用sklearn.cluster.KMeans
        # 這裡簡化處理
        clusters = []
        chunk_size = len(embeddings) // num_clusters
        for i in range(0, len(embeddings), chunk_size):
            clusters.append(embeddings[i:i+chunk_size])
        return clusters

# 使用
raptor = RAPTOR()
document = "..." # 長文檔
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_text(document)
hierarchical_summaries = raptor.cluster_and_summarize(chunks)
```

### 6. DSPy - RAG系統優化

```python
import dspy

# 配置LLM
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.settings.configure(lm=lm)

# 定義RAG Pipeline
class RAGPipeline(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate_answer(context=context, question=question)
        return answer

# 優化Pipeline
from dspy.teleprompt import BootstrapFewShot

trainset = [
    dspy.Example(question="What is AI?", answer="...").with_inputs('question'),
    # 更多訓練樣本
]

teleprompter = BootstrapFewShot(metric=lambda x, y: x.answer == y.answer)
optimized_rag = teleprompter.compile(RAGPipeline(), trainset=trainset)

# 使用優化後的Pipeline
result = optimized_rag(question="Explain machine learning")
print(result.answer)
```

---

## 📊 技術對比

| 技術 | 檢索準確率 | 計算成本 | 實現複雜度 | 適用場景 |
|------|-----------|---------|-----------|----------|
| GraphRAG | ⭐⭐⭐⭐⭐ | 🔥🔥🔥 | 🛠️🛠️🛠️ | 複雜知識推理 |
| HyDE | ⭐⭐⭐⭐ | 🔥🔥 | 🛠️🛠️ | 語義匹配困難 |
| Reranking | ⭐⭐⭐⭐⭐ | 🔥 | 🛠️ | 所有RAG場景 |
| Self-RAG | ⭐⭐⭐⭐ | 🔥🔥🔥 | 🛠️🛠️🛠️ | 需要高準確度 |
| RAPTOR | ⭐⭐⭐⭐ | 🔥🔥 | 🛠️🛠️🛠️ | 長文檔理解 |

---

## 🔬 最佳實踐

1. **混合檢索**: 向量 + 關鍵字 + 語義
2. **多階段Pipeline**: 檢索 → 重排序 → 生成 → 驗證
3. **上下文壓縮**: 使用LongLLMLingua壓縮無關信息
4. **結果驗證**: Self-RAG自我檢查機制
5. **持續優化**: DSPy自動化優化提示詞和參數

---

**最後更新**: 2025-01-19
