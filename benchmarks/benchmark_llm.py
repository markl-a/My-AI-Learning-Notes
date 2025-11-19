"""
LLM 性能基準測試
測試不同 LLM 的性能、延遲和吞吐量
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any
import os
from openai import AsyncOpenAI
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class BenchmarkResult:
    """基準測試結果"""
    model_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: str


class LLMBenchmark:
    """LLM 基準測試類"""

    def __init__(self, api_key: str = None):
        """初始化基準測試"""
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.results: List[BenchmarkResult] = []

    async def benchmark_latency(
        self,
        model: str,
        prompt: str,
        num_requests: int = 10
    ) -> Dict[str, float]:
        """
        測試延遲性能

        Args:
            model: 模型名稱
            prompt: 測試提示
            num_requests: 請求次數

        Returns:
            延遲統計數據
        """
        print(f"\n🔍 測試 {model} 延遲性能...")

        latencies = []

        for i in range(num_requests):
            start_time = time.time()

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # 轉換為毫秒

                latencies.append(latency)
                print(f"  請求 {i+1}/{num_requests}: {latency:.2f}ms")

            except Exception as e:
                print(f"  請求 {i+1} 失敗: {str(e)}")
                continue

        if not latencies:
            return {"error": "All requests failed"}

        # 計算統計數據
        stats = {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99": sorted(latencies)[int(len(latencies) * 0.99)]
        }

        # 保存結果
        self.results.append(BenchmarkResult(
            model_name=model,
            metric_name="latency_mean",
            value=stats["mean"],
            unit="ms",
            timestamp=datetime.utcnow().isoformat()
        ))

        return stats

    async def benchmark_throughput(
        self,
        model: str,
        prompts: List[str],
        concurrent_requests: int = 5
    ) -> Dict[str, float]:
        """
        測試吞吐量性能

        Args:
            model: 模型名稱
            prompts: 測試提示列表
            concurrent_requests: 並發請求數

        Returns:
            吞吐量統計數據
        """
        print(f"\n🚀 測試 {model} 吞吐量性能...")

        async def send_request(prompt: str):
            """發送單個請求"""
            start_time = time.time()
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                end_time = time.time()
                return {
                    "success": True,
                    "latency": (end_time - start_time) * 1000,
                    "tokens": response.usage.total_tokens
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # 批量處理
        start_time = time.time()
        tasks = [send_request(p) for p in prompts[:concurrent_requests]]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # 計算統計
        successful_requests = [r for r in results if r.get("success")]
        total_time = end_time - start_time

        if not successful_requests:
            return {"error": "All requests failed"}

        total_tokens = sum(r["tokens"] for r in successful_requests)

        throughput_stats = {
            "requests_per_second": len(successful_requests) / total_time,
            "tokens_per_second": total_tokens / total_time,
            "total_time": total_time,
            "successful_requests": len(successful_requests),
            "failed_requests": len(results) - len(successful_requests)
        }

        # 保存結果
        self.results.append(BenchmarkResult(
            model_name=model,
            metric_name="throughput_rps",
            value=throughput_stats["requests_per_second"],
            unit="req/s",
            timestamp=datetime.utcnow().isoformat()
        ))

        return throughput_stats

    async def benchmark_token_efficiency(
        self,
        model: str,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        測試 Token 效率

        Args:
            model: 模型名稱
            test_cases: 測試案例列表

        Returns:
            Token 效率統計
        """
        print(f"\n📊 測試 {model} Token 效率...")

        token_stats = []

        for i, test_case in enumerate(test_cases):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_case["prompt"]}],
                    max_tokens=test_case.get("max_tokens", 200)
                )

                usage = response.usage
                token_stats.append({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "efficiency": usage.completion_tokens / usage.total_tokens
                })

                print(f"  測試 {i+1}: "
                      f"提示={usage.prompt_tokens}, "
                      f"完成={usage.completion_tokens}, "
                      f"效率={token_stats[-1]['efficiency']:.2%}")

            except Exception as e:
                print(f"  測試 {i+1} 失敗: {str(e)}")
                continue

        if not token_stats:
            return {"error": "All tests failed"}

        # 計算平均值
        efficiency_stats = {
            "avg_prompt_tokens": statistics.mean([s["prompt_tokens"] for s in token_stats]),
            "avg_completion_tokens": statistics.mean([s["completion_tokens"] for s in token_stats]),
            "avg_total_tokens": statistics.mean([s["total_tokens"] for s in token_stats]),
            "avg_efficiency": statistics.mean([s["efficiency"] for s in token_stats])
        }

        return efficiency_stats

    async def benchmark_quality(
        self,
        model: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        測試輸出質量

        Args:
            model: 模型名稱
            test_cases: 包含提示和預期答案的測試案例

        Returns:
            質量評分
        """
        print(f"\n✨ 測試 {model} 輸出質量...")

        quality_scores = []

        for i, test_case in enumerate(test_cases):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_case["prompt"]}],
                    max_tokens=200,
                    temperature=0.1
                )

                generated = response.choices[0].message.content
                expected = test_case.get("expected", "")

                # 簡單的相似度評分（實際應使用更複雜的評估）
                # 這裡使用關鍵詞匹配作為簡化
                keywords = set(expected.lower().split())
                generated_words = set(generated.lower().split())
                overlap = len(keywords & generated_words)
                score = overlap / len(keywords) if keywords else 0

                quality_scores.append(score)
                print(f"  測試 {i+1}: 質量評分={score:.2%}")

            except Exception as e:
                print(f"  測試 {i+1} 失敗: {str(e)}")
                continue

        if not quality_scores:
            return {"error": "All quality tests failed"}

        return {
            "average_quality_score": statistics.mean(quality_scores),
            "min_score": min(quality_scores),
            "max_score": max(quality_scores)
        }

    async def run_full_benchmark(
        self,
        models: List[str],
        test_prompts: List[str] = None
    ):
        """
        運行完整基準測試

        Args:
            models: 要測試的模型列表
            test_prompts: 測試提示列表
        """
        if test_prompts is None:
            test_prompts = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "What are the benefits of cloud computing?",
                "Describe the importance of data privacy.",
                "How does neural network work?"
            ]

        print("=" * 70)
        print("🔥 LLM 性能基準測試")
        print("=" * 70)

        all_results = {}

        for model in models:
            print(f"\n{'='*70}")
            print(f"📌 測試模型: {model}")
            print(f"{'='*70}")

            model_results = {}

            # 1. 延遲測試
            latency_stats = await self.benchmark_latency(
                model=model,
                prompt=test_prompts[0],
                num_requests=10
            )
            model_results["latency"] = latency_stats

            # 2. 吞吐量測試
            throughput_stats = await self.benchmark_throughput(
                model=model,
                prompts=test_prompts,
                concurrent_requests=3
            )
            model_results["throughput"] = throughput_stats

            # 3. Token 效率測試
            test_cases = [
                {"prompt": p, "max_tokens": 100}
                for p in test_prompts
            ]
            efficiency_stats = await self.benchmark_token_efficiency(
                model=model,
                test_cases=test_cases
            )
            model_results["token_efficiency"] = efficiency_stats

            all_results[model] = model_results

            # 等待一下避免速率限制
            await asyncio.sleep(2)

        # 保存結果
        self.save_results(all_results)

        # 打印摘要
        self.print_summary(all_results)

    def save_results(self, results: Dict):
        """保存結果到文件"""
        os.makedirs("results", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/llm_benchmark_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 結果已保存到: {filename}")

    def print_summary(self, results: Dict):
        """打印結果摘要"""
        print("\n" + "=" * 70)
        print("📊 基準測試摘要")
        print("=" * 70)

        for model, metrics in results.items():
            print(f"\n🔹 {model}")
            print("-" * 70)

            if "latency" in metrics and "mean" in metrics["latency"]:
                lat = metrics["latency"]
                print(f"  延遲:")
                print(f"    平均: {lat['mean']:.2f}ms")
                print(f"    P95:  {lat['p95']:.2f}ms")
                print(f"    P99:  {lat['p99']:.2f}ms")

            if "throughput" in metrics:
                tp = metrics["throughput"]
                print(f"  吞吐量:")
                print(f"    請求/秒: {tp['requests_per_second']:.2f}")
                print(f"    Tokens/秒: {tp['tokens_per_second']:.2f}")

            if "token_efficiency" in metrics:
                eff = metrics["token_efficiency"]
                print(f"  Token 效率:")
                print(f"    平均效率: {eff['avg_efficiency']:.2%}")
                print(f"    平均總 Tokens: {eff['avg_total_tokens']:.0f}")


async def main():
    """主函數"""
    # 要測試的模型
    models = [
        "gpt-3.5-turbo",
        # "gpt-4",  # 取消註釋以測試 GPT-4
        # "gpt-4-turbo-preview",
    ]

    # 創建基準測試實例
    benchmark = LLMBenchmark()

    # 運行完整測試
    await benchmark.run_full_benchmark(models=models)


if __name__ == "__main__":
    asyncio.run(main())
