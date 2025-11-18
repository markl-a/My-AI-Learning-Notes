#!/usr/bin/env python3
"""
LLM 性能基准测试工具

测试和比较不同 LLM 模型的性能指标。
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from dotenv import load_dotenv
except ImportError:
    print("请安装依赖: pip install rich python-dotenv")
    sys.exit(1)

load_dotenv()
console = Console()


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model: str
    avg_latency: float  # 秒
    tokens_per_second: float
    total_tokens: int
    success_rate: float
    cost_per_1k_tokens: float
    memory_mb: float = 0


# 测试提示词集
TEST_PROMPTS = [
    "什么是机器学习？用一句话回答。",
    "解释什么是神经网络。",
    "列出 3 个常见的排序算法。",
    "Python 和 JavaScript 的主要区别是什么？",
    "什么是 REST API？",
]


def benchmark_model(model: str, prompts: List[str] = None) -> BenchmarkResult:
    """对单个模型进行基准测试"""

    if prompts is None:
        prompts = TEST_PROMPTS

    console.print(f"\n[cyan]测试模型: {model}[/cyan]")

    latencies = []
    total_tokens = 0
    successes = 0

    # 检测模型类型
    is_openai = model.startswith(("gpt-", "o1-"))
    is_anthropic = model.startswith("claude-")

    try:
        if is_openai:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            for prompt in track(prompts, description="执行测试"):
                start = time.time()
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100
                    )
                    latency = time.time() - start
                    latencies.append(latency)
                    total_tokens += response.usage.total_tokens
                    successes += 1
                except Exception as e:
                    console.print(f"[red]  错误: {e}[/red]")

        else:
            # Ollama (本地模型)
            import ollama

            for prompt in track(prompts, description="执行测试"):
                start = time.time()
                try:
                    response = ollama.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'num_predict': 100}
                    )
                    latency = time.time() - start
                    latencies.append(latency)
                    total_tokens += response.get('eval_count', 0) + response.get('prompt_eval_count', 0)
                    successes += 1
                except Exception as e:
                    console.print(f"[red]  错误: {e}[/red]")

    except Exception as e:
        console.print(f"[red]模型 {model} 测试失败: {e}[/red]")
        return None

    if not latencies:
        return None

    # 计算指标
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_second = total_tokens / sum(latencies) if sum(latencies) > 0 else 0
    success_rate = successes / len(prompts)

    # 估算成本 (基于模型类型)
    cost_per_1k = {
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.0004,
        "claude-3-5-sonnet": 0.006,
        "claude-3-haiku": 0.0007,
    }.get(model, 0.0)  # 本地模型免费

    return BenchmarkResult(
        model=model,
        avg_latency=avg_latency,
        tokens_per_second=tokens_per_second,
        total_tokens=total_tokens,
        success_rate=success_rate,
        cost_per_1k_tokens=cost_per_1k
    )


def compare_models(models: List[str]):
    """比较多个模型"""
    console.print(Panel.fit(
        "[bold cyan]🏁 LLM 性能基准测试[/bold cyan]\n"
        f"[dim]测试模型: {', '.join(models)}[/dim]",
        border_style="cyan"
    ))

    results = []

    for model in models:
        result = benchmark_model(model)
        if result:
            results.append(result)

    if not results:
        console.print("[red]没有成功的测试结果[/red]")
        return

    # 显示结果表格
    table = Table(title="\n性能比较结果", show_header=True, header_style="bold cyan")
    table.add_column("模型", style="cyan", width=20)
    table.add_column("平均延迟", style="green", width=12)
    table.add_column("速度", style="yellow", width=15)
    table.add_column("总 Tokens", style="magenta", width=12)
    table.add_column("成功率", style="blue", width=10)
    table.add_column("成本/1K", style="red", width=12)

    for result in results:
        table.add_row(
            result.model,
            f"{result.avg_latency:.2f}s",
            f"{result.tokens_per_second:.1f} tok/s",
            str(result.total_tokens),
            f"{result.success_rate * 100:.0f}%",
            f"${result.cost_per_1k_tokens:.4f}" if result.cost_per_1k_tokens > 0 else "免费"
        )

    console.print(table)

    # 显示推荐
    if len(results) > 1:
        fastest = min(results, key=lambda r: r.avg_latency)
        cheapest = min(results, key=lambda r: r.cost_per_1k_tokens)
        most_efficient = max(results, key=lambda r: r.tokens_per_second)

        console.print(f"\n[bold green]📊 推荐:[/bold green]")
        console.print(f"  • 最快: [cyan]{fastest.model}[/cyan] ({fastest.avg_latency:.2f}s)")
        console.print(f"  • 最便宜: [cyan]{cheapest.model}[/cyan] (${cheapest.cost_per_1k_tokens:.4f}/1K)")
        console.print(f"  • 最高效: [cyan]{most_efficient.model}[/cyan] ({most_efficient.tokens_per_second:.1f} tok/s)")
        console.print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM 性能基准测试工具")
    parser.add_argument("--model", type=str, help="单个模型测试")
    parser.add_argument("--compare", type=str, help="比较多个模型（逗号分隔）")
    parser.add_argument("--prompts", type=int, default=5, help="测试提示词数量")

    args = parser.parse_args()

    if args.model:
        result = benchmark_model(args.model, TEST_PROMPTS[:args.prompts])
        if result:
            console.print(f"\n[green]✅ 测试完成[/green]")
            console.print(f"  平均延迟: {result.avg_latency:.2f}s")
            console.print(f"  速度: {result.tokens_per_second:.1f} tokens/s")
            console.print(f"  成功率: {result.success_rate * 100:.0f}%")

    elif args.compare:
        models = [m.strip() for m in args.compare.split(",")]
        compare_models(models)

    else:
        # 默认比较常用模型
        console.print("[yellow]未指定模型，测试默认模型...[/yellow]")

        default_models = []
        if os.getenv("OPENAI_API_KEY"):
            default_models.append("gpt-4o-mini")

        try:
            import ollama
            ollama_models = ollama.list().get('models', [])
            if ollama_models:
                default_models.append(ollama_models[0]['name'])
        except:
            pass

        if default_models:
            compare_models(default_models)
        else:
            console.print("[red]没有可用的模型进行测试[/red]")
            console.print("[yellow]请使用 --model 或 --compare 参数指定模型[/yellow]")


if __name__ == "__main__":
    main()
