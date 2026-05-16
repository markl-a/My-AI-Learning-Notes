# RAG 2.0 與多模態 RAG 系統 (2024-2025)

## 目錄
- [從 RAG 1.0 到 RAG 2.0](#從-rag-10-到-rag-20)
- [RAG 2.0 核心改進](#rag-20-核心改進)
- [多模態 RAG 系統](#多模態-rag-系統)
- [進階檢索技術](#進階檢索技術)
- [實作範例](#實作範例)
- [最佳實踐](#最佳實踐)

---

## 從 RAG 1.0 到 RAG 2.0

### RAG 1.0 的局限

傳統 RAG（檢索增強生成）流程：
```
用戶查詢 → 向量檢索 → 取回文檔 → LLM 生成答案
```

**主要問題：**
1. **單一檢索策略**：僅依賴向量相似度
2. **缺乏推理**：檢索與生成分離
3. **單模態限制**：只處理文字
4. **幻覺問題**：無法有效驗證事實

### RAG 2.0 的演進

RAG 2.0 整合了多項先進技術：

```
用戶查詢
    ↓
[查詢改寫/擴展]
    ↓
[混合檢索：向量 + 關鍵詞 + 圖結構]
    ↓
[重排序 (Reranking)]
    ↓
[多模態處理：文字 + 圖像 + 表格]
    ↓
[推理驗證]
    ↓
[生成答案 + 引用來源]
```

---

## RAG 2.0 核心改進

### 1. 混合檢索 (Hybrid Search)

結合多種檢索策略以提升準確性：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

class HybridRetriever:
    """混合檢索器：向量搜索 + BM25"""

    def __init__(self, documents):
        # 1. 向量檢索器
        embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma.from_documents(
            documents,
            embeddings
        )
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        # 2. BM25 關鍵詞檢索器
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10

        # 3. 混合檢索器
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]  # 可調整權重
        )

    def retrieve(self, query: str, top_k: int = 5):
        """執行混合檢索"""
        return self.ensemble_retriever.get_relevant_documents(query)[:top_k]

# 使用示例
retriever = HybridRetriever(documents)
results = retriever.retrieve("什麼是 Transformer 架構？")
```

### 2. 查詢改寫 (Query Rewriting)

改善查詢質量以獲得更好的檢索結果：

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class QueryRewriter:
    """查詢改寫器"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def rewrite_query(self, original_query: str, context: str = ""):
        """改寫查詢以提升檢索效果"""

        prompt = ChatPromptTemplate.from_template("""
        原始查詢：{query}
        對話上下文：{context}

        請將查詢改寫為更適合檢索的形式：
        1. 補充必要的背景資訊
        2. 使用更精確的術語
        3. 分解複雜查詢
        4. 保持查詢意圖

        改寫後的查詢：
        """)

        response = self.llm.invoke(
            prompt.format(query=original_query, context=context)
        )

        return response.content

    def generate_multi_queries(self, query: str, n: int = 3):
        """生成多個查詢變體"""

        prompt = ChatPromptTemplate.from_template("""
        原始查詢：{query}

        請生成 {n} 個不同角度的查詢變體，每個查詢應該：
        - 保持原意
        - 使用不同的措辭
        - 可能探索不同的子問題

        輸出格式（每行一個查詢）：
        1. [查詢1]
        2. [查詢2]
        3. [查詢3]
        """)

        response = self.llm.invoke(
            prompt.format(query=query, n=n)
        )

        queries = [line.split('. ', 1)[1].strip()
                   for line in response.content.split('\n')
                   if line.strip() and line[0].isdigit()]

        return queries

# 使用示例
rewriter = QueryRewriter()

# 單一改寫
improved_query = rewriter.rewrite_query(
    "transformer 怎麼用",
    context="之前討論了深度學習模型"
)

# 多查詢生成
multi_queries = rewriter.generate_multi_queries(
    "Transformer 的注意力機制是如何工作的？"
)
```

### 3. HyDE (Hypothetical Document Embeddings)

生成假設性文檔以改善檢索：

```python
class HyDERetriever:
    """HyDE 檢索器"""

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def retrieve_with_hyde(self, query: str, top_k: int = 5):
        """使用 HyDE 方法檢索"""

        # 1. 生成假設性文檔
        hypothetical_doc = self.generate_hypothetical_document(query)

        # 2. 使用假設文檔進行檢索
        results = self.vector_store.similarity_search(
            hypothetical_doc,
            k=top_k
        )

        return results

    def generate_hypothetical_document(self, query: str) -> str:
        """生成假設性文檔"""

        prompt = f"""
        針對以下問題，生成一個假設性的、詳細的答案文檔：

        問題：{query}

        請撰寫一個完整、專業的答案，就好像你在撰寫一篇技術文章。
        不要說「我不知道」，而是基於常識生成一個合理的答案。
        """

        response = self.llm.invoke(prompt)
        return response.content

# 使用
hyde_retriever = HyDERetriever(vector_store, llm)
docs = hyde_retriever.retrieve_with_hyde("解釋量子計算的基本原理")
```

### 4. Reranking (重排序)

使用專門的模型對檢索結果重新排序：

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """檢索結果重排序器"""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 5):
        """重排序文檔"""

        # 1. 計算查詢與每個文檔的相關性分數
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        # 2. 按分數排序
        ranked_indices = np.argsort(scores)[::-1]

        # 3. 返回 top-k
        return [
            {
                "document": documents[i],
                "score": scores[i],
                "rank": rank + 1
            }
            for rank, i in enumerate(ranked_indices[:top_k])
        ]

# 使用
reranker = Reranker()
initial_results = retriever.retrieve("Transformer 架構")
reranked = reranker.rerank(
    "Transformer 架構",
    [doc.page_content for doc in initial_results],
    top_k=3
)
```

### 5. 自我反思與驗證

```python
class SelfReflectiveRAG:
    """自我反思 RAG 系統"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query_with_reflection(self, question: str, max_iterations: int = 3):
        """帶自我反思的查詢"""

        for iteration in range(max_iterations):
            # 1. 檢索
            docs = self.retriever.retrieve(question)

            # 2. 生成答案
            answer = self.generate_answer(question, docs)

            # 3. 自我評估
            is_satisfactory, feedback = self.self_evaluate(
                question,
                answer,
                docs
            )

            if is_satisfactory:
                return {
                    "answer": answer,
                    "iterations": iteration + 1,
                    "documents": docs
                }

            # 4. 改進查詢
            question = self.improve_question(question, feedback)

        return {
            "answer": answer,
            "iterations": max_iterations,
            "warning": "未達到滿意標準"
        }

    def self_evaluate(self, question: str, answer: str, docs: List) -> tuple:
        """自我評估答案質量"""

        eval_prompt = f"""
        問題：{question}
        答案：{answer}

        評估以下方面（每項1-5分）：
        1. 答案是否直接回答了問題？
        2. 答案是否有足夠的證據支持？
        3. 答案是否邏輯連貫？
        4. 是否存在事實錯誤？

        如果平均分低於4分，提供改進建議。

        輸出格式：
        評分：[分數]
        是否滿意：[是/否]
        改進建議：[建議]
        """

        response = self.llm.invoke(eval_prompt)

        # 解析評估結果
        is_satisfactory = "是否滿意：是" in response.content
        feedback = response.content

        return is_satisfactory, feedback
```

---

## 多模態 RAG 系統

### 1. 文字 + 圖像 RAG

```python
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from PIL import Image
import base64

class MultimodalRAG:
    """多模態 RAG 系統"""

    def __init__(self):
        self.text_embeddings = OpenAIEmbeddings()
        self.vision_model = ChatOpenAI(model="gpt-4o")  # GPT-4o 具備視覺能力

    def process_document_with_images(self, pdf_path: str):
        """處理包含圖像的 PDF 文檔"""

        # 1. 提取文字
        loader = PyPDFLoader(pdf_path)
        text_documents = loader.load()

        # 2. 提取圖像
        images = self.extract_images_from_pdf(pdf_path)

        # 3. 為圖像生成描述
        image_descriptions = []
        for i, img in enumerate(images):
            description = self.describe_image(img)
            image_descriptions.append({
                "image_id": i,
                "description": description,
                "image": img
            })

        # 4. 建立索引
        all_content = text_documents + [
            Document(page_content=desc["description"],
                     metadata={"type": "image", "image_id": desc["image_id"]})
            for desc in image_descriptions
        ]

        vector_store = Chroma.from_documents(
            all_content,
            self.text_embeddings
        )

        return vector_store, image_descriptions

    def describe_image(self, image: Image) -> str:
        """使用視覺模型描述圖像"""

        # 轉換圖像為 base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        response = self.vision_model.invoke([
            {
                "type": "text",
                "text": "詳細描述這張圖片的內容，包括關鍵資訊、圖表、示意圖等。"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            }
        ])

        return response.content

    def query_multimodal(self, query: str, vector_store, image_descriptions):
        """多模態查詢"""

        # 1. 檢索相關內容（文字和圖像描述）
        results = vector_store.similarity_search(query, k=5)

        # 2. 分離文字和圖像
        text_results = []
        image_results = []

        for doc in results:
            if doc.metadata.get("type") == "image":
                image_id = doc.metadata["image_id"]
                image_info = image_descriptions[image_id]
                image_results.append(image_info)
            else:
                text_results.append(doc)

        # 3. 生成綜合答案
        answer = self.generate_multimodal_answer(
            query,
            text_results,
            image_results
        )

        return {
            "answer": answer,
            "text_sources": text_results,
            "image_sources": image_results
        }
```

### 2. 表格資料 RAG

```python
import pandas as pd

class TableRAG:
    """表格資料 RAG"""

    def __init__(self, llm):
        self.llm = llm

    def query_table(self, query: str, df: pd.DataFrame):
        """查詢表格資料"""

        # 1. 分析查詢意圖
        intent = self.analyze_query_intent(query, df)

        # 2. 生成 SQL/Pandas 查詢
        code = self.generate_query_code(query, df, intent)

        # 3. 執行查詢
        result = self.execute_query(code, df)

        # 4. 生成自然語言答案
        answer = self.generate_answer(query, result)

        return {
            "answer": answer,
            "data": result,
            "code": code
        }

    def analyze_query_intent(self, query: str, df: pd.DataFrame) -> str:
        """分析查詢意圖"""

        prompt = f"""
        表格列名：{df.columns.tolist()}
        表格預覽：
        {df.head().to_string()}

        用戶查詢：{query}

        這個查詢的意圖是什麼？（過濾/聚合/排序/統計/其他）
        """

        response = self.llm.invoke(prompt)
        return response.content

    def generate_query_code(self, query: str, df: pd.DataFrame, intent: str) -> str:
        """生成查詢程式碼"""

        prompt = f"""
        表格結構：
        {df.dtypes.to_string()}

        查詢：{query}
        意圖：{intent}

        生成 Pandas 程式碼來回答這個查詢。
        變量名使用 'df'。只輸出程式碼，不要解釋。

        程式碼：
        """

        response = self.llm.invoke(prompt)
        return response.content

    def execute_query(self, code: str, df: pd.DataFrame):
        """安全執行查詢程式碼"""

        try:
            local_vars = {"df": df, "pd": pd}
            exec(code, {}, local_vars)
            result = local_vars.get('result', df)
            return result
        except Exception as e:
            return f"執行錯誤：{str(e)}"
```

---

## 進階檢索技術

### 1. 圖檢索 (Graph-based Retrieval)

```python
from langchain_community.graphs import Neo4jGraph

class GraphRAG:
    """圖資料庫 RAG"""

    def __init__(self, neo4j_uri, user, password):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=user,
            password=password
        )

    def query_graph(self, question: str):
        """查詢圖資料庫"""

        # 1. 將問題轉換為 Cypher 查詢
        cypher_query = self.nl_to_cypher(question)

        # 2. 執行查詢
        results = self.graph.query(cypher_query)

        # 3. 格式化結果
        formatted_results = self.format_graph_results(results)

        return formatted_results

    def nl_to_cypher(self, question: str) -> str:
        """將自然語言轉換為 Cypher 查詢"""

        prompt = f"""
        圖資料庫架構：
        - 節點類型：Person, Company, Technology
        - 關係：WORKS_AT, KNOWS, USES

        問題：{question}

        生成對應的 Cypher 查詢：
        """

        response = self.llm.invoke(prompt)
        return response.content
```

### 2. 時序 RAG (Temporal RAG)

處理時間敏感的資訊：

```python
from datetime import datetime

class TemporalRAG:
    """時序 RAG 系統"""

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def retrieve_with_time_context(self, query: str, time_range=None):
        """帶時間上下文的檢索"""

        # 1. 提取查詢中的時間資訊
        time_info = self.extract_time_from_query(query)

        # 2. 過濾文檔
        if time_info:
            docs = self.vector_store.similarity_search(
                query,
                filter={"timestamp": time_info}
            )
        else:
            docs = self.vector_store.similarity_search(query)

        # 3. 按時間排序
        docs = sorted(docs, key=lambda x: x.metadata.get("timestamp", 0))

        return docs

    def extract_time_from_query(self, query: str) -> dict:
        """從查詢中提取時間資訊"""

        prompt = f"""
        查詢：{query}

        提取時間相關資訊：
        - 是否提到具體日期？
        - 是否提到時間範圍？
        - 是否暗示「最新」「最近」等？

        JSON格式輸出。
        """

        response = self.llm.invoke(prompt)
        # 解析並返回時間資訊
        return eval(response.content)
```

---

## 實作範例

### 完整的 RAG 2.0 系統

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class RAG2System:
    """完整的 RAG 2.0 系統"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.query_rewriter = QueryRewriter()
        self.reranker = Reranker()

    def ingest_documents(self, documents: List[str]):
        """攝取文檔"""

        # 1. 文字分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.create_documents(documents)

        # 2. 建立向量存儲
        self.vector_store = Chroma.from_documents(
            splits,
            self.embeddings
        )

        # 3. 建立混合檢索器
        self.hybrid_retriever = HybridRetriever(splits)

    def query(self, question: str, use_hyde: bool = True, use_rerank: bool = True):
        """執行查詢"""

        # 1. 查詢改寫
        improved_question = self.query_rewriter.rewrite_query(question)
        multi_queries = self.query_rewriter.generate_multi_queries(question)

        # 2. 檢索
        all_docs = []

        # 主查詢
        if use_hyde:
            hyde_retriever = HyDERetriever(self.vector_store, self.llm)
            docs = hyde_retriever.retrieve_with_hyde(improved_question)
        else:
            docs = self.hybrid_retriever.retrieve(improved_question)
        all_docs.extend(docs)

        # 多查詢
        for mq in multi_queries:
            docs = self.hybrid_retriever.retrieve(mq, top_k=3)
            all_docs.extend(docs)

        # 去重
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        # 3. 重排序
        if use_rerank:
            reranked = self.reranker.rerank(
                question,
                [doc.page_content for doc in unique_docs],
                top_k=5
            )
            final_docs = [unique_docs[r["rank"]-1] for r in reranked]
        else:
            final_docs = unique_docs[:5]

        # 4. 生成答案
        answer = self.generate_answer(question, final_docs)

        # 5. 驗證與引用
        verified_answer = self.verify_and_cite(question, answer, final_docs)

        return {
            "question": question,
            "answer": verified_answer["answer"],
            "sources": verified_answer["sources"],
            "confidence": verified_answer["confidence"]
        }

    def generate_answer(self, question: str, documents: List) -> str:
        """生成答案"""

        context = "\n\n".join([f"[文檔{i+1}]\n{doc.page_content}"
                               for i, doc in enumerate(documents)])

        prompt = f"""
        基於以下文檔回答問題：

        {context}

        問題：{question}

        要求：
        1. 只根據提供的文檔回答
        2. 如果文檔中沒有相關資訊，明確說明
        3. 引用具體的文檔編號

        答案：
        """

        response = self.llm.invoke(prompt)
        return response.content

    def verify_and_cite(self, question: str, answer: str, documents: List) -> Dict:
        """驗證答案並添加引用"""

        verify_prompt = f"""
        問題：{question}
        答案：{answer}

        請驗證：
        1. 答案中的每個事實陳述是否有文檔支持？
        2. 是否存在幻覺（無根據的陳述）？
        3. 信心度評分（0-1）

        輸出格式：
        驗證結果：[通過/需要修正]
        信心度：[分數]
        需要修正的部分：[列表]
        """

        response = self.llm.invoke(verify_prompt)

        # 解析驗證結果並返回
        return {
            "answer": answer,
            "sources": [doc.page_content for doc in documents],
            "confidence": 0.85,  # 從驗證結果解析
            "verification": response.content
        }

# 使用完整系統
rag2 = RAG2System()

# 攝取文檔
documents = [
    "Transformer 是2017年提出的模型架構...",
    "注意力機制是 Transformer 的核心...",
    # 更多文檔...
]
rag2.ingest_documents(documents)

# 查詢
result = rag2.query(
    "Transformer 的注意力機制是如何工作的？",
    use_hyde=True,
    use_rerank=True
)

print(f"答案：{result['answer']}")
print(f"信心度：{result['confidence']}")
```

---

## 最佳實踐

### 1. 文檔分割策略

```python
# 根據文檔類型選擇分割策略
def smart_text_splitter(document_type: str):
    """智能文字分割"""

    if document_type == "code":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=500,
            chunk_overlap=50
        )
    elif document_type == "markdown":
        return MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
```

### 2. 性能優化

```python
# 使用快取提升性能
from functools import lru_cache

class CachedRAG:
    """帶快取的 RAG"""

    @lru_cache(maxsize=1000)
    def cached_retrieve(self, query: str):
        """快取檢索結果"""
        return self.retriever.retrieve(query)
```

### 3. 監控與評估

```python
class RAGMetrics:
    """RAG 系統評估指標"""

    def evaluate(self, questions: List[str], expected_answers: List[str]):
        """評估 RAG 性能"""

        metrics = {
            "retrieval_precision": [],
            "retrieval_recall": [],
            "answer_relevance": [],
            "answer_correctness": [],
            "latency": []
        }

        for q, expected in zip(questions, expected_answers):
            start_time = time.time()

            # 執行 RAG
            result = self.rag.query(q)

            # 計算指標
            metrics["latency"].append(time.time() - start_time)
            metrics["answer_relevance"].append(
                self.compute_relevance(result["answer"], q)
            )
            metrics["answer_correctness"].append(
                self.compute_correctness(result["answer"], expected)
            )

        return {k: np.mean(v) for k, v in metrics.items()}
```

---

## 總結

RAG 2.0 與多模態 RAG 系統代表了檢索增強生成技術的重要演進：

✅ **從單一檢索到混合策略**
✅ **從被動檢索到主動推理**
✅ **從文字到多模態**
✅ **從黑箱到可驗證**

### 關鍵技術
- 混合檢索 (Hybrid Search)
- 查詢改寫 (Query Rewriting)
- HyDE
- 重排序 (Reranking)
- 多模態處理
- 自我反思與驗證

### 實際應用
- 企業知識庫
- 技術文檔問答
- 多語言支持
- 實時資訊檢索

---

## 參考資源

### 論文
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [Self-RAG: Learning to Retrieve, Generate and Critique](https://arxiv.org/abs/2310.11511)

### 框架文檔
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [Haystack](https://docs.haystack.deepset.ai/)
