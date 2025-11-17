# Building and Evaluating Advanced RAG Applications

## 📋 課程概述

學習進階 RAG 技術，包括查詢優化、混合搜尋、重新排序和評估方法。

### 課程目標
- 掌握進階 RAG 技術和策略
- 學習 RAG 效能評估方法
- 理解查詢重寫和優化
- 實作企業級 RAG 應用

### 課程時長
約 1 小時

## 🎯 進階 RAG 架構

```
使用者查詢
    ↓
1. 查詢優化 (Query Optimization)
    ├─ 查詢重寫 (Query Rewriting)
    ├─ 查詢擴展 (Query Expansion)
    └─ 查詢分解 (Query Decomposition)
    ↓
2. 混合檢索 (Hybrid Retrieval)
    ├─ 向量搜尋 (Vector Search)
    ├─ 關鍵字搜尋 (Keyword Search)
    └─ 結果融合 (Fusion)
    ↓
3. 重新排序 (Reranking)
    ├─ Cross-encoder
    ├─ ColBERT
    └─ LLM-based Reranking
    ↓
4. 上下文壓縮 (Context Compression)
    ↓
5. 生成答案 (Answer Generation)
    ↓
6. 答案驗證 (Answer Validation)
```

## 1️⃣ 查詢優化技術

### Query Rewriting（查詢重寫）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def rewrite_for_retrieval(self, query: str) -> str:
        """重寫查詢以提升檢索效果"""
        prompt = ChatPromptTemplate.from_template("""
        你是一位搜尋專家。請將使用者的查詢重寫為更適合搜尋的格式。

        原始查詢：{query}

        重寫規則：
        1. 移除無關詞彙（如：請問、可以告訴我等）
        2. 轉換為陳述句或關鍵詞
        3. 保留重要的實體和概念
        4. 使用更精確的術語

        只輸出重寫後的查詢，不需要其他說明。
        """)

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def multi_query_generation(self, query: str, n: int = 3) -> list:
        """生成多個相關查詢"""
        prompt = ChatPromptTemplate.from_template("""
        根據原始問題，生成 {n} 個不同角度的相關查詢。
        這些查詢應該能幫助我們從不同角度檢索相關資訊。

        原始問題：{query}

        請以列表格式輸出，每行一個查詢：
        """)

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "n": n})

        # 解析結果
        queries = [q.strip().lstrip('123456789.-) ') for q in result.split('\n') if q.strip()]
        return queries[:n]

    def step_back_prompting(self, query: str) -> str:
        """後退提示：生成更抽象的查詢"""
        prompt = ChatPromptTemplate.from_template("""
        給定一個具體的問題，請生成一個更抽象、更general的問題。
        這個抽象問題應該能幫助我們檢索到更廣泛的背景知識。

        具體問題：{query}

        抽象問題：
        """)

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

# 使用範例
rewriter = QueryRewriter()

original_query = "請問台灣有什麼好吃的夜市小吃？"

print(f"原始查詢：{original_query}\n")

print(f"重寫查詢：{rewriter.rewrite_for_retrieval(original_query)}\n")

print("多角度查詢：")
multi_queries = rewriter.multi_query_generation(original_query, n=3)
for i, q in enumerate(multi_queries, 1):
    print(f"{i}. {q}")

print(f"\n後退提示：{rewriter.step_back_prompting(original_query)}")
```

### Query Decomposition（查詢分解）

```python
class QueryDecomposer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def decompose(self, complex_query: str) -> list:
        """將複雜查詢分解為子問題"""
        prompt = ChatPromptTemplate.from_template("""
        將以下複雜問題分解為多個簡單的子問題。
        每個子問題應該能夠獨立回答。

        複雜問題：{query}

        請以編號列表格式輸出子問題：
        """)

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": complex_query})

        sub_questions = [q.strip().lstrip('123456789.-) ') for q in result.split('\n') if q.strip()]
        return sub_questions

# 使用範例
decomposer = QueryDecomposer()

complex_query = "比較台北101和東京晴空塔的高度、建築年份和主要特色"
sub_questions = decomposer.decompose(complex_query)

print(f"原始問題：{complex_query}\n")
print("子問題：")
for i, q in enumerate(sub_questions, 1):
    print(f"{i}. {q}")
```

## 2️⃣ 混合搜尋（Hybrid Search）

結合向量搜尋和關鍵字搜尋的優點。

```python
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class HybridSearcher:
    def __init__(self, documents: List[str]):
        """
        Args:
            documents: 文檔列表
        """
        self.documents = documents

        # 向量搜尋
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings
        )

        # BM25 關鍵字搜尋
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def vector_search(self, query: str, k: int = 5) -> List[tuple]:
        """向量搜尋"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]

    def keyword_search(self, query: str, k: int = 5) -> List[tuple]:
        """關鍵字搜尋（BM25）"""
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)

        # 取得 top k
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        results = [(self.documents[i], doc_scores[i]) for i in top_k_indices]
        return results

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5
    ) -> List[tuple]:
        """
        混合搜尋

        Args:
            query: 查詢
            k: 返回結果數
            alpha: 向量搜尋權重（0-1），1-alpha 為關鍵字搜尋權重
        """
        # 獲取兩種搜尋結果
        vector_results = self.vector_search(query, k=k*2)
        keyword_results = self.keyword_search(query, k=k*2)

        # 標準化分數
        vector_scores = self._normalize_scores([score for _, score in vector_results])
        keyword_scores = self._normalize_scores([score for _, score in keyword_results])

        # 合併和重新評分
        doc_scores = {}

        for (doc, _), norm_score in zip(vector_results, vector_scores):
            doc_scores[doc] = doc_scores.get(doc, 0) + alpha * (1 - norm_score)  # Chroma 返回距離，需要反轉

        for (doc, _), norm_score in zip(keyword_results, keyword_scores):
            doc_scores[doc] = doc_scores.get(doc, 0) + (1 - alpha) * norm_score

        # 排序並返回 top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """標準化分數到 0-1 範圍"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

# 使用範例
documents = [
    "台北101是台灣最著名的地標建築",
    "台北101曾經是世界最高的建築物",
    "台北的夜市非常有名，特別是士林夜市",
    "珍珠奶茶是台灣最具代表性的飲品",
    "台灣的美食文化豐富多樣",
]

hybrid_searcher = HybridSearcher(documents)

query = "台北著名建築"

print(f"查詢：{query}\n")

print("向量搜尋結果：")
for i, (doc, score) in enumerate(hybrid_searcher.vector_search(query, k=3), 1):
    print(f"{i}. {doc} (分數: {score:.4f})")

print("\n關鍵字搜尋結果：")
for i, (doc, score) in enumerate(hybrid_searcher.keyword_search(query, k=3), 1):
    print(f"{i}. {doc} (分數: {score:.4f})")

print("\n混合搜尋結果：")
for i, (doc, score) in enumerate(hybrid_searcher.hybrid_search(query, k=3, alpha=0.5), 1):
    print(f"{i}. {doc} (分數: {score:.4f})")
```

## 3️⃣ 重新排序（Reranking）

使用更精確的模型對初步檢索結果重新排序。

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """初始化重排序模型"""
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        重新排序文檔

        Args:
            query: 查詢
            documents: 文檔列表
            top_k: 返回前 k 個

        Returns:
            重新排序後的 (文檔, 分數) 列表
        """
        # 準備輸入對
        pairs = [[query, doc] for doc in documents]

        # 預測分數
        scores = self.model.predict(pairs)

        # 排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

# 使用範例
reranker = Reranker()

query = "台灣最高的山"
candidate_docs = [
    "玉山是台灣最高的山，海拔3952公尺",
    "阿里山以日出和雲海聞名",
    "台灣有許多高山，是登山愛好者的天堂",
    "台北101曾經是世界最高建築",
    "雪山是台灣第二高峰"
]

print(f"查詢：{query}\n")
print("重新排序後：")

reranked_docs = reranker.rerank(query, candidate_docs, top_k=3)
for i, (doc, score) in enumerate(reranked_docs, 1):
    print(f"{i}. {doc} (分數: {score:.4f})")
```

### LLM-based Reranking

```python
class LLMReranker:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[str]:
        """使用 LLM 重新排序"""

        # 準備文檔列表
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])

        prompt = ChatPromptTemplate.from_template("""
        給定查詢和候選文檔，請根據相關性對文檔重新排序。
        只輸出文檔編號，用逗號分隔，從最相關到最不相關。

        查詢：{query}

        候選文檔：
        {documents}

        排序結果（只輸出編號，例如：1,3,2,5,4）：
        """)

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "documents": docs_text})

        # 解析結果
        try:
            indices = [int(i.strip()) - 1 for i in result.split(',')]
            reranked = [documents[i] for i in indices if 0 <= i < len(documents)]
            return reranked[:top_k]
        except:
            return documents[:top_k]

# 使用
llm_reranker = LLMReranker()
reranked = llm_reranker.rerank(query, candidate_docs, top_k=3)

print("\nLLM 重排序結果：")
for i, doc in enumerate(reranked, 1):
    print(f"{i}. {doc}")
```

## 4️⃣ RAG 評估

### RAGAS 評估框架

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# 準備評估資料
eval_data = {
    "question": [
        "台灣最高的山是什麼？",
        "台北有哪些著名夜市？"
    ],
    "answer": [
        "台灣最高的山是玉山，海拔3952公尺",
        "台北著名的夜市包括士林夜市、饒河街夜市等"
    ],
    "contexts": [
        ["玉山是台灣最高峰，海拔3952公尺，位於南投、嘉義、高雄交界"],
        ["士林夜市是台北最大的夜市", "饒河街夜市以小吃聞名"]
    ],
    "ground_truths": [
        ["玉山，3952公尺"],
        ["士林夜市、饒河街夜市"]
    ]
}

dataset = Dataset.from_dict(eval_data)

# 執行評估
result = evaluate(
    dataset,
    metrics=[
        faithfulness,        # 忠實度：答案是否基於上下文
        answer_relevancy,    # 相關性：答案是否回答問題
        context_recall,      # 召回率：上下文是否包含答案
        context_precision,   # 精確度：上下文的相關性
    ],
)

print(result)
```

### 自訂評估指標

```python
class RAGEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict:
        """評估答案品質"""

        prompt = ChatPromptTemplate.from_template("""
        評估以下 RAG 系統的回答品質。

        問題：{question}

        上下文：{context}

        回答：{answer}

        請從以下維度評分（1-5分）：
        1. 準確性：回答是否正確
        2. 完整性：是否充分回答問題
        3. 忠實度：是否基於提供的上下文
        4. 流暢度：語言是否通順自然

        請以 JSON 格式輸出：
        {{
            "accuracy": <分數>,
            "completeness": <分數>,
            "faithfulness": <分數>,
            "fluency": <分數>,
            "comments": "<簡短評語>"
        }}
        """)

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })

        import json
        return json.loads(result)

    def evaluate_retrieval(
        self,
        question: str,
        retrieved_docs: List[str],
        ground_truth_docs: List[str]
    ) -> Dict:
        """評估檢索品質"""

        # 計算精確率和召回率
        retrieved_set = set(retrieved_docs)
        ground_truth_set = set(ground_truth_docs)

        true_positives = len(retrieved_set & ground_truth_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved_count": len(retrieved_docs),
            "relevant_count": true_positives
        }

# 使用範例
evaluator = RAGEvaluator()

# 評估答案品質
quality_scores = evaluator.evaluate_answer_quality(
    question="台灣最高的山是什麼？",
    answer="台灣最高的山是玉山，海拔3952公尺",
    context="玉山是台灣最高峰，海拔3952公尺，位於中央山脈"
)

print("答案品質評估：")
print(json.dumps(quality_scores, indent=2, ensure_ascii=False))
```

## 💡 完整進階 RAG 系統

```python
class AdvancedRAGSystem:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.query_rewriter = QueryRewriter()
        self.evaluator = RAGEvaluator()

    def query(self, user_question: str, vectorstore) -> Dict:
        """完整的 RAG 流程"""

        # 1. 查詢優化
        print("🔧 查詢優化...")
        rewritten_query = self.query_rewriter.rewrite_for_retrieval(user_question)
        multi_queries = self.query_rewriter.multi_query_generation(user_question, n=2)
        all_queries = [rewritten_query] + multi_queries

        # 2. 多查詢檢索
        print("🔍 執行檢索...")
        all_docs = []
        for query in all_queries:
            docs = vectorstore.similarity_search(query, k=3)
            all_docs.extend(docs)

        # 去重
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        # 3. 重新排序（如果有多個文檔）
        if len(unique_docs) > 1:
            print("📊 重新排序...")
            reranker = Reranker()
            doc_contents = [doc.page_content for doc in unique_docs]
            reranked = reranker.rerank(user_question, doc_contents, top_k=3)
            context = "\n\n".join([doc for doc, _ in reranked])
        else:
            context = "\n\n".join([doc.page_content for doc in unique_docs])

        # 4. 生成答案
        print("✍️ 生成答案...")
        answer_prompt = ChatPromptTemplate.from_template("""
        根據以下上下文回答問題。如果上下文中沒有相關資訊，請說明無法回答。

        上下文：
        {context}

        問題：{question}

        回答：
        """)

        chain = answer_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": user_question
        })

        # 5. 評估（可選）
        print("📈 評估答案...")
        evaluation = self.evaluator.evaluate_answer_quality(
            user_question,
            answer,
            context
        )

        return {
            "question": user_question,
            "rewritten_query": rewritten_query,
            "context": context,
            "answer": answer,
            "evaluation": evaluation
        }
```

## ✅ 最佳實踐總結

1. **查詢優化**：重寫、擴展、分解複雜查詢
2. **混合搜尋**：結合向量和關鍵字搜尋
3. **重新排序**：使用更精確的模型
4. **評估驅動**：持續評估和優化系統
5. **迭代改進**：根據評估結果調整策略

---

**課程連結**：[DeepLearning.ai - Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)

**完成日期**：2025-01-17
