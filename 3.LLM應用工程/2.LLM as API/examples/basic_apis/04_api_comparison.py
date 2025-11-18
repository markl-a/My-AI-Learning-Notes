"""
LLM API 比較示例
比較 OpenAI, Anthropic, Google Gemini 的性能、成本和輸出品質
"""

import os
import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import json

# API 客戶端
from openai import OpenAI
import anthropic
import google.generativeai as genai

# 載入環境變數
load_dotenv()

# 初始化客戶端
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def test_openai(prompt: str, model: str = "gpt-4o-mini") -> Dict:
    """測試 OpenAI API"""
    start_time = time.time()

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    end_time = time.time()

    return {
        "provider": "OpenAI",
        "model": model,
        "response": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency": end_time - start_time
    }


def test_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Dict:
    """測試 Anthropic API"""
    start_time = time.time()

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    end_time = time.time()

    return {
        "provider": "Anthropic",
        "model": model,
        "response": response.content[0].text,
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "latency": end_time - start_time
    }


def test_gemini(prompt: str, model_name: str = "gemini-1.5-pro") -> Dict:
    """測試 Google Gemini API"""
    start_time = time.time()

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    end_time = time.time()

    try:
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
    except:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

    return {
        "provider": "Google Gemini",
        "model": model_name,
        "response": response.text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency": end_time - start_time
    }


def compare_apis(prompt: str) -> List[Dict]:
    """比較所有 API"""
    print(f"\n{'='*60}")
    print(f"測試提示: {prompt}")
    print(f"{'='*60}\n")

    results = []

    # 測試 OpenAI
    try:
        print("測試 OpenAI...")
        result = test_openai(prompt)
        results.append(result)
        print(f"✓ 完成 (耗時: {result['latency']:.2f}秒)")
    except Exception as e:
        print(f"✗ OpenAI 錯誤: {e}")

    # 測試 Anthropic
    try:
        print("測試 Anthropic...")
        result = test_anthropic(prompt)
        results.append(result)
        print(f"✓ 完成 (耗時: {result['latency']:.2f}秒)")
    except Exception as e:
        print(f"✗ Anthropic 錯誤: {e}")

    # 測試 Gemini
    try:
        print("測試 Google Gemini...")
        result = test_gemini(prompt)
        results.append(result)
        print(f"✓ 完成 (耗時: {result['latency']:.2f}秒)")
    except Exception as e:
        print(f"✗ Gemini 錯誤: {e}")

    return results


def display_comparison(results: List[Dict]):
    """顯示比較結果"""
    print(f"\n{'='*60}")
    print("比較結果")
    print(f"{'='*60}\n")

    # 性能比較
    print("【性能指標】")
    print(f"{'提供商':<20} {'延遲(秒)':<15} {'總Tokens':<15}")
    print("-" * 50)

    for result in results:
        print(f"{result['provider']:<20} {result['latency']:<15.2f} {result['total_tokens']:<15}")

    # 找出最快的
    if results:
        fastest = min(results, key=lambda x: x['latency'])
        print(f"\n⚡ 最快: {fastest['provider']} ({fastest['latency']:.2f}秒)")

    # Token 使用比較
    print("\n\n【Token 使用】")
    for result in results:
        print(f"\n{result['provider']}:")
        print(f"  Prompt tokens: {result['prompt_tokens']}")
        print(f"  Completion tokens: {result['completion_tokens']}")
        print(f"  Total tokens: {result['total_tokens']}")

    # 回應品質（顯示前200字符）
    print("\n\n【回應內容預覽】")
    for result in results:
        print(f"\n{result['provider']} ({result['model']}):")
        print(f"{result['response'][:200]}...")


def cost_estimation(results: List[Dict]) -> Dict:
    """成本估算（基於2024年1月的價格）"""
    print(f"\n\n{'='*60}")
    print("成本估算")
    print(f"{'='*60}\n")

    # 價格（美元 per 1M tokens）
    pricing = {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},  # 128K 以下
    }

    cost_results = {}

    for result in results:
        model = result['model']
        if model in pricing:
            input_cost = (result['prompt_tokens'] / 1_000_000) * pricing[model]["input"]
            output_cost = (result['completion_tokens'] / 1_000_000) * pricing[model]["output"]
            total_cost = input_cost + output_cost

            cost_results[result['provider']] = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }

            print(f"{result['provider']}:")
            print(f"  Input cost: ${input_cost:.6f}")
            print(f"  Output cost: ${output_cost:.6f}")
            print(f"  Total cost: ${total_cost:.6f}")
            print()

    # 找出最便宜的
    if cost_results:
        cheapest = min(cost_results.items(), key=lambda x: x[1]["total_cost"])
        print(f"💰 最便宜: {cheapest[0]} (${cheapest[1]['total_cost']:.6f})")

    return cost_results


def batch_comparison(prompts: List[str]):
    """批次比較多個提示"""
    print(f"\n{'='*60}")
    print(f"批次測試 - 共 {len(prompts)} 個提示")
    print(f"{'='*60}")

    all_results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n\n【測試 {i}/{len(prompts)}】")
        results = compare_apis(prompt)
        all_results.append({
            "prompt": prompt,
            "results": results
        })

        # 顯示本次結果
        display_comparison(results)
        cost_estimation(results)

    # 總結
    print(f"\n\n{'='*60}")
    print("批次測試總結")
    print(f"{'='*60}")

    provider_stats = {}

    for test in all_results:
        for result in test['results']:
            provider = result['provider']
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "count": 0,
                    "total_latency": 0,
                    "total_tokens": 0
                }

            provider_stats[provider]["count"] += 1
            provider_stats[provider]["total_latency"] += result['latency']
            provider_stats[provider]["total_tokens"] += result['total_tokens']

    print(f"\n{'提供商':<20} {'平均延遲':<15} {'平均Tokens':<15}")
    print("-" * 50)

    for provider, stats in provider_stats.items():
        avg_latency = stats["total_latency"] / stats["count"]
        avg_tokens = stats["total_tokens"] / stats["count"]
        print(f"{provider:<20} {avg_latency:<15.2f} {avg_tokens:<15.0f}")

    return all_results


def scenario_tests():
    """不同場景測試"""
    print(f"\n{'='*60}")
    print("場景測試")
    print(f"{'='*60}")

    scenarios = {
        "程式碼生成": "寫一個 Python 函數來實現 LRU 快取，包含完整的類別定義和方法",
        "文本摘要": "請摘要這段文字：機器學習是人工智慧的一個分支，它使電腦能夠在沒有明確程式設計的情況下學習。機器學習專注於開發能夠存取資料並利用資料自主學習的演算法。學習過程從觀察資料開始，例如範例、直接經驗或指導，目的是在資料中尋找模式，並在未來做出更好的決策。",
        "創意寫作": "寫一個科幻短故事的開頭，關於一個發現時間旅行的科學家",
        "資料分析": "分析以下數據並給出洞察：銷售額 Q1: 100萬, Q2: 120萬, Q3: 95萬, Q4: 140萬",
        "翻譯": "將以下英文翻譯成中文：The future of artificial intelligence lies in the development of more efficient and ethical AI systems."
    }

    results_by_scenario = {}

    for scenario_name, prompt in scenarios.items():
        print(f"\n\n【場景: {scenario_name}】")
        results = compare_apis(prompt)
        results_by_scenario[scenario_name] = results

        display_comparison(results)

        # 等待一下避免 rate limit
        time.sleep(1)

    return results_by_scenario


def main():
    """主程式"""
    print("=" * 60)
    print("LLM API 完整比較測試")
    print("=" * 60)

    # 單一測試
    print("\n\n【單一提示測試】")
    prompt = "解釋什麼是遞迴，並提供一個 Python 範例"
    results = compare_apis(prompt)
    display_comparison(results)
    cost_estimation(results)

    # 批次測試
    print("\n\n【批次測試】")
    batch_prompts = [
        "什麼是機器學習？",
        "寫一個 Python 裝飾器範例",
        "解釋 REST API 的設計原則"
    ]
    batch_comparison(batch_prompts)

    # 場景測試
    print("\n\n【場景測試】")
    scenario_tests()

    print("\n\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)

    print("\n建議：")
    print("📊 根據您的使用場景選擇合適的 API：")
    print("  - 成本優先 → OpenAI GPT-4o-mini")
    print("  - 程式碼品質 → Anthropic Claude")
    print("  - 多模態需求 → Google Gemini")
    print("  - 長文本處理 → Anthropic Claude (200K) 或 Gemini (2M)")


if __name__ == "__main__":
    main()
