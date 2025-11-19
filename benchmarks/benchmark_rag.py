"""
RAG 系統性能基準測試

測試不同配置下 RAG 系統的性能指標：
- 檢索延遲
- 生成延遲
- 端到端延遲
- 相關性評分
- 內存使用

運行：python benchmarks/benchmark_rag.py
"""

import time
import psutil
import json
from typing import Dict, List
from datetime import datetime
from pathlib import Path

# 基準測試結果保存路徑
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


class RAGBenchmark:
    """RAG 性能基準測試類"""

    def __init__(self):
        self.results = []
        self.process = psutil.Process()

    def benchmark_retrieval(
        self, vectorstore, queries: List[str], k: int = 3
    ) -> Dict:
        """基準測試檢索性能

        Args:
            vectorstore: 向量數據庫實例
            queries: 測試查詢列表
            k: 檢索文檔數量

        Returns:
            性能指標字典
        """
        latencies = []
        memory_usage = []

        for query in queries:
            start_mem = self.process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()
            results = vectorstore.similarity_search(query, k=k)
            end_time = time.time()

            end_mem = self.process.memory_info().rss / 1024 / 1024  # MB

            latencies.append((end_time - start_time) * 1000)  # ms
            memory_usage.append(end_mem - start_mem)

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
        }

    def benchmark_generation(self, qa_chain, queries: List[str]) -> Dict:
        """基準測試生成性能"""
        latencies = []
        token_counts = []

        for query in queries:
            start_time = time.time()
            result = qa_chain.invoke({"query": query})
            end_time = time.time()

            latencies.append((end_time - start_time) * 1000)
            token_counts.append(len(result["result"].split()))

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "throughput_tokens_per_sec": sum(token_counts) / (sum(latencies) / 1000),
        }

    def save_results(self, results: Dict, filename: str = None):
        """保存基準測試結果"""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = RESULTS_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ 結果已保存到：{filepath}")


# 示例測試查詢
SAMPLE_QUERIES = [
    "什麼是 RAG 系統？",
    "如何優化向量檢索性能？",
    "深度學習和機器學習有什麼區別？",
    "Transformer 架構的核心組件是什麼？",
    "如何評估 LLM 的性能？",
]


if __name__ == "__main__":
    print("📊 RAG 性能基準測試")
    print("=" * 50)
    print("這是一個基準測試框架模板")
    print("請根據實際 RAG 實現進行配置")
    print("=" * 50)
