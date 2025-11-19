"""
Self-RAG (Self-Reflective Retrieval-Augmented Generation)
帶自我反思機制的RAG，通過評估檢索質量和生成質量來改進輸出
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalDecision(Enum):
    """檢索決策"""
    RETRIEVE = "需要檢索"
    NO_RETRIEVE = "不需要檢索"


class RelevanceScore(Enum):
    """相關性評分"""
    HIGHLY_RELEVANT = "高度相關"
    RELEVANT = "相關"
    IRRELEVANT = "不相關"


class SupportScore(Enum):
    """支持度評分"""
    FULLY_SUPPORTED = "完全支持"
    PARTIALLY_SUPPORTED = "部分支持"
    NOT_SUPPORTED = "不支持"


class UtilityScore(Enum):
    """有用性評分"""
    HIGHLY_USEFUL = "非常有用"
    USEFUL = "有用"
    NOT_USEFUL = "無用"


@dataclass
class ReflectionResult:
    """反思結果"""
    retrieval_needed: RetrievalDecision
    relevance: Optional[RelevanceScore] = None
    support: Optional[SupportScore] = None
    utility: Optional[UtilityScore] = None
    reasoning: str = ""


class SelfRAG:
    """Self-RAG 系統"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_generate_fn=None,
        llm_evaluate_fn=None
    ):
        """
        初始化 Self-RAG

        Args:
            embedding_model: 嵌入模型
            llm_generate_fn: LLM 生成函數
            llm_evaluate_fn: LLM 評估函數
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.llm_generate = llm_generate_fn or self._simple_generate
        self.llm_evaluate = llm_evaluate_fn or self._simple_evaluate

        self.documents: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

        print("Self-RAG 系統初始化完成")

    def add_documents(self, documents: List[str]):
        """添加文檔"""
        self.documents = documents
        self.doc_embeddings = self.encoder.encode(documents)
        print(f"已添加 {len(documents)} 個文檔")

    def _simple_generate(self, prompt: str) -> str:
        """簡單的生成函數（模擬）"""
        return f"基於提示生成的回覆：{prompt[:50]}..."

    def _simple_evaluate(self, prompt: str) -> str:
        """簡單的評估函數（模擬）"""
        if "檢索" in prompt or "retrieve" in prompt.lower():
            return "需要檢索"
        elif "相關" in prompt or "relevant" in prompt.lower():
            return "相關"
        elif "支持" in prompt or "support" in prompt.lower():
            return "部分支持"
        else:
            return "有用"

    def decide_retrieval(self, query: str) -> RetrievalDecision:
        """
        決定是否需要檢索

        Args:
            query: 用戶查詢

        Returns:
            檢索決策
        """
        prompt = f"""判斷以下問題是否需要檢索外部文檔來回答：

問題: {query}

如果問題需要特定的事實、數據或專業知識，回答"需要檢索"。
如果問題是常識性的或可以直接回答，回答"不需要檢索"。

決策:"""

        decision = self.llm_evaluate(prompt)

        if "需要" in decision or "retrieve" in decision.lower():
            return RetrievalDecision.RETRIEVE
        else:
            return RetrievalDecision.NO_RETRIEVE

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        檢索相關文檔

        Args:
            query: 查詢
            top_k: 返回數量

        Returns:
            [(文檔, 相似度分數), ...]
        """
        if self.doc_embeddings is None:
            return []

        query_embedding = self.encoder.encode([query])[0]

        # 計算相似度
        similarities = np.dot(self.doc_embeddings, query_embedding)

        # 獲取 top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results

    def evaluate_relevance(
        self,
        query: str,
        document: str
    ) -> RelevanceScore:
        """
        評估文檔相關性

        Args:
            query: 查詢
            document: 文檔

        Returns:
            相關性評分
        """
        prompt = f"""評估文檔與問題的相關性：

問題: {query}

文檔: {document}

請評估相關性（高度相關/相關/不相關）:"""

        evaluation = self.llm_evaluate(prompt)

        if "高度" in evaluation or "highly" in evaluation.lower():
            return RelevanceScore.HIGHLY_RELEVANT
        elif "不" in evaluation or "ir" in evaluation.lower():
            return RelevanceScore.IRRELEVANT
        else:
            return RelevanceScore.RELEVANT

    def evaluate_support(
        self,
        document: str,
        response: str
    ) -> SupportScore:
        """
        評估回覆是否被文檔支持

        Args:
            document: 文檔
            response: 生成的回覆

        Returns:
            支持度評分
        """
        prompt = f"""評估回覆是否被文檔支持：

文檔: {document}

回覆: {response}

請評估支持度（完全支持/部分支持/不支持）:"""

        evaluation = self.llm_evaluate(prompt)

        if "完全" in evaluation or "fully" in evaluation.lower():
            return SupportScore.FULLY_SUPPORTED
        elif "不" in evaluation or "not" in evaluation.lower():
            return SupportScore.NOT_SUPPORTED
        else:
            return SupportScore.PARTIALLY_SUPPORTED

    def evaluate_utility(
        self,
        query: str,
        response: str
    ) -> UtilityScore:
        """
        評估回覆的有用性

        Args:
            query: 查詢
            response: 回覆

        Returns:
            有用性評分
        """
        prompt = f"""評估回覆對問題的有用性：

問題: {query}

回覆: {response}

請評估有用性（非常有用/有用/無用）:"""

        evaluation = self.llm_evaluate(prompt)

        if "非常" in evaluation or "highly" in evaluation.lower():
            return UtilityScore.HIGHLY_USEFUL
        elif "無" in evaluation or "not" in evaluation.lower():
            return UtilityScore.NOT_USEFUL
        else:
            return UtilityScore.USEFUL

    def generate_with_reflection(
        self,
        query: str,
        max_iterations: int = 3
    ) -> Dict:
        """
        帶反思機制的生成

        Args:
            query: 用戶查詢
            max_iterations: 最大迭代次數

        Returns:
            包含最終答案和反思過程的字典
        """
        print(f"\n{'='*60}")
        print(f"問題: {query}")
        print(f"{'='*60}\n")

        reflections = []

        # Step 1: 決定是否需要檢索
        retrieval_decision = self.decide_retrieval(query)
        print(f"【步驟 1: 檢索決策】 {retrieval_decision.value}")

        if retrieval_decision == RetrievalDecision.NO_RETRIEVE:
            # 直接生成答案
            response = self.llm_generate(query)
            utility = self.evaluate_utility(query, response)

            return {
                "answer": response,
                "retrieval_used": False,
                "utility": utility.value,
                "reflections": []
            }

        # Step 2: 檢索文檔
        retrieved_docs = self.retrieve_documents(query, top_k=3)
        print(f"\n【步驟 2: 檢索文檔】 找到 {len(retrieved_docs)} 個文檔")

        best_response = None
        best_score = -1

        for i, (doc, similarity) in enumerate(retrieved_docs, 1):
            print(f"\n--- 文檔 {i} (相似度: {similarity:.4f}) ---")
            print(f"{doc[:150]}...")

            # Step 3: 評估相關性
            relevance = self.evaluate_relevance(query, doc)
            print(f"相關性: {relevance.value}")

            if relevance == RelevanceScore.IRRELEVANT:
                continue

            # Step 4: 生成回覆
            context_prompt = f"""基於以下文檔回答問題：

文檔: {doc}

問題: {query}

回答:"""

            response = self.llm_generate(context_prompt)
            print(f"生成回覆: {response[:100]}...")

            # Step 5: 評估支持度
            support = self.evaluate_support(doc, response)
            print(f"支持度: {support.value}")

            # Step 6: 評估有用性
            utility = self.evaluate_utility(query, response)
            print(f"有用性: {utility.value}")

            # 計算綜合得分
            score = 0
            if relevance == RelevanceScore.HIGHLY_RELEVANT:
                score += 3
            elif relevance == RelevanceScore.RELEVANT:
                score += 2

            if support == SupportScore.FULLY_SUPPORTED:
                score += 3
            elif support == SupportScore.PARTIALLY_SUPPORTED:
                score += 1

            if utility == UtilityScore.HIGHLY_USEFUL:
                score += 3
            elif utility == UtilityScore.USEFUL:
                score += 2

            # 記錄反思
            reflection = ReflectionResult(
                retrieval_needed=retrieval_decision,
                relevance=relevance,
                support=support,
                utility=utility,
                reasoning=f"文檔 {i}, 綜合得分: {score}"
            )
            reflections.append(reflection)

            # 更新最佳回覆
            if score > best_score:
                best_score = score
                best_response = response

        # 返回最佳回覆
        print(f"\n【最終答案】 (得分: {best_score})")
        print(best_response)

        return {
            "answer": best_response or "無法生成滿意的答案",
            "retrieval_used": True,
            "best_score": best_score,
            "reflections": reflections,
            "num_docs_evaluated": len(retrieved_docs)
        }


def example_basic_self_rag():
    """示例 1: 基本 Self-RAG"""
    print("=== 示例 1: 基本 Self-RAG ===\n")

    # 創建系統
    self_rag = SelfRAG()

    # 添加知識庫
    documents = [
        "Transformer 是 Google 在 2017 年提出的神經網絡架構，使用自注意力機制處理序列數據。它在 NLP 領域引發了革命。",
        "BERT 是一個雙向 Transformer 編碼器，通過遮罩語言建模進行預訓練。它在 2018 年發布後在多個 NLP 基準上刷新了記錄。",
        "GPT 系列是單向 Transformer 解碼器，擅長文本生成任務。GPT-3 擁有 1750 億參數，展現了驚人的少樣本學習能力。",
        "T5 (Text-to-Text Transfer Transformer) 將所有 NLP 任務統一為文本到文本的格式。這種方法簡化了模型架構和訓練流程。"
    ]

    self_rag.add_documents(documents)

    # 測試查詢
    query = "哪個模型最適合文本生成任務？"

    result = self_rag.generate_with_reflection(query)

    print(f"\n{'='*60}")
    print("總結:")
    print(f"使用檢索: {result['retrieval_used']}")
    if result['retrieval_used']:
        print(f"評估文檔數: {result['num_docs_evaluated']}")
        print(f"最佳得分: {result['best_score']}")
    print(f"最終答案: {result['answer']}")


def example_no_retrieval():
    """示例 2: 不需要檢索的查詢"""
    print("\n\n=== 示例 2: 不需要檢索的查詢 ===\n")

    self_rag = SelfRAG()

    # 常識性問題
    query = "1 + 1 等於多少？"

    result = self_rag.generate_with_reflection(query)

    print(f"\n使用檢索: {result['retrieval_used']}")
    print(f"答案: {result['answer']}")


if __name__ == "__main__":
    print("Self-RAG 自我反思檢索增強生成示例")
    print("=" * 60)
    print()

    # 運行示例
    example_basic_self_rag()
    example_no_retrieval()

    print("\n\nSelf-RAG 流程:")
    print("1. 【檢索決策】判斷是否需要檢索")
    print("2. 【文檔檢索】檢索相關文檔")
    print("3. 【相關性評估】評估每個文檔的相關性")
    print("4. 【生成回覆】基於文檔生成答案")
    print("5. 【支持度評估】檢查回覆是否被文檔支持")
    print("6. 【有用性評估】評估回覆的有用性")
    print("7. 【選擇最佳】根據評分選擇最佳回覆")

    print("\nSelf-RAG 優勢:")
    print("✓ 自適應檢索（只在需要時檢索）")
    print("✓ 質量控制（多層評估機制）")
    print("✓ 可解釋性（提供反思過程）")
    print("✓ 準確率提升 15-20%")

    print("\n評估維度:")
    print("1. Retrieval: 是否需要外部知識")
    print("2. Relevance: 文檔與查詢的相關性")
    print("3. Support: 生成內容是否有文檔支持")
    print("4. Utility: 回覆對用戶的有用性")

    print("\n實現建議:")
    print("1. 使用專門訓練的評估模型")
    print("2. 調整評分權重以適應應用場景")
    print("3. 緩存評估結果以提高效率")
    print("4. 設置質量閾值自動過濾低分回覆")
