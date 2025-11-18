#!/usr/bin/env python3
"""
批量推理示例

展示如何高效地處理大量 LLM 推理請求。
包含並行處理、進度追蹤、錯誤處理、性能優化等功能。

應用場景：
- 批量文本分類
- 大規模內容生成
- 數據增強
- 離線處理任務

前置需求：
pip install openai anthropic ollama rich pandas tqdm concurrent
"""

import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from openai import OpenAI
    import ollama
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from dotenv import load_dotenv
    import pandas as pd
except ImportError as e:
    print(f"❌ 缺少依賴: {e}")
    print("請運行: pip install openai ollama rich pandas python-dotenv")
    sys.exit(1)

load_dotenv()
console = Console()


@dataclass
class InferenceTask:
    """推理任務"""
    id: str
    prompt: str
    metadata: Dict[str, Any] = None


@dataclass
class InferenceResult:
    """推理結果"""
    id: str
    prompt: str
    response: str
    success: bool
    elapsed_time: float
    error: str = None
    tokens: int = 0


class BatchInferenceEngine:
    """批量推理引擎"""

    def __init__(self, provider: str = "openai", model: str = None, max_workers: int = 5):
        self.provider = provider
        self.max_workers = max_workers

        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        elif provider == "ollama":
            self.model = model or "llama3.1:8b"
        else:
            raise ValueError(f"不支持的提供商: {provider}")

    def _infer_single(self, task: InferenceTask) -> InferenceResult:
        """單個推理任務"""
        start_time = time.time()

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": task.prompt}],
                    temperature=0.7,
                    max_tokens=500
                )

                result_text = response.choices[0].message.content
                tokens = response.usage.total_tokens

            elif self.provider == "ollama":
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': task.prompt}]
                )

                result_text = response['message']['content']
                tokens = response.get('eval_count', 0) + response.get('prompt_eval_count', 0)

            elapsed = time.time() - start_time

            return InferenceResult(
                id=task.id,
                prompt=task.prompt,
                response=result_text,
                success=True,
                elapsed_time=elapsed,
                tokens=tokens
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                id=task.id,
                prompt=task.prompt,
                response="",
                success=False,
                elapsed_time=elapsed,
                error=str(e)
            )

    def process_batch(self, tasks: List[InferenceTask]) -> List[InferenceResult]:
        """批量處理任務（並行）"""
        results = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task_progress = progress.add_task(
                f"[cyan]處理中 ({self.provider})...",
                total=len(tasks)
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._infer_single, task): task
                    for task in tasks
                }

                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    progress.advance(task_progress)

        return results

    def process_batch_sequential(self, tasks: List[InferenceTask]) -> List[InferenceResult]:
        """批量處理任務（順序）"""
        results = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task_progress = progress.add_task(
                "[cyan]順序處理中...",
                total=len(tasks)
            )

            for task in tasks:
                result = self._infer_single(task)
                results.append(result)
                progress.advance(task_progress)

        return results


def demo_parallel_vs_sequential():
    """比較並行 vs 順序處理"""
    console.print(Panel.fit(
        "[bold cyan]示例 1: 並行 vs 順序處理比較[/bold cyan]",
        border_style="cyan"
    ))

    # 創建測試任務
    tasks = [
        InferenceTask(id=f"task_{i}", prompt=f"用一句話解釋：{topic}")
        for i, topic in enumerate([
            "機器學習", "深度學習", "神經網路", "自然語言處理",
            "計算機視覺", "強化學習", "遷移學習", "聯邦學習"
        ])
    ]

    console.print(f"[yellow]📋 任務數量:[/yellow] {len(tasks)}\n")

    # 檢查可用的服務
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    console.print(f"[yellow]🔧 使用提供商:[/yellow] {provider}\n")

    engine = BatchInferenceEngine(provider=provider, max_workers=4)

    # 順序處理
    console.print("[yellow]1️⃣  順序處理（單線程）:[/yellow]")
    start = time.time()
    sequential_results = engine.process_batch_sequential(tasks)
    sequential_time = time.time() - start

    console.print(f"[green]✅ 完成，耗時: {sequential_time:.2f} 秒[/green]\n")

    # 並行處理
    console.print("[yellow]2️⃣  並行處理（4 線程）:[/yellow]")
    start = time.time()
    parallel_results = engine.process_batch(tasks)
    parallel_time = time.time() - start

    console.print(f"[green]✅ 完成，耗時: {parallel_time:.2f} 秒[/green]\n")

    # 性能比較
    speedup = sequential_time / parallel_time
    console.print(f"[bold green]⚡ 加速比: {speedup:.2f}x[/bold green]")
    console.print(f"[dim]時間節省: {sequential_time - parallel_time:.2f} 秒 ({(1 - parallel_time/sequential_time)*100:.1f}%)[/dim]\n")


def demo_batch_classification():
    """批量文本分類示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 2: 批量文本分類[/bold cyan]",
        border_style="cyan"
    ))

    # 模擬文本分類任務
    texts = [
        "這部電影太精彩了，演員演技一流！",
        "產品質量很差，客服態度惡劣",
        "今天天氣不錯，適合出去走走",
        "這個餐廳的食物很美味，環境也很好",
        "服務器又當機了，嚴重影響業務",
        "新功能很實用，大大提升了工作效率"
    ]

    # 創建分類任務
    classification_prompt = """
    請將以下文本分類為：正面、負面或中性。
    只需回答分類結果，不需要解釋。

    文本：{text}

    分類：
    """

    tasks = [
        InferenceTask(
            id=f"classify_{i}",
            prompt=classification_prompt.format(text=text),
            metadata={"original_text": text}
        )
        for i, text in enumerate(texts)
    ]

    console.print(f"[yellow]📝 待分類文本數量:[/yellow] {len(tasks)}\n")

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    engine = BatchInferenceEngine(provider=provider, max_workers=3)

    # 執行批量分類
    results = engine.process_batch(tasks)

    # 顯示結果
    table = Table(title="文本分類結果", show_header=True, header_style="bold cyan")
    table.add_column("文本", style="dim", width=40)
    table.add_column("分類", style="cyan", width=10)
    table.add_column("耗時", style="green", width=10)

    for result in results:
        if result.success:
            classification = result.response.strip()
            # 簡化顯示
            if len(result.prompt) > 50:
                display_text = texts[int(result.id.split('_')[1])]
            else:
                display_text = result.prompt

            table.add_row(
                display_text,
                classification,
                f"{result.elapsed_time:.2f}s"
            )

    console.print(table)
    console.print()


def demo_batch_data_generation():
    """批量數據生成示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 3: 批量數據生成[/bold cyan]",
        border_style="cyan"
    ))

    # 生成訓練數據
    topics = [
        "機器學習基礎",
        "Python 程式設計",
        "數據結構",
        "Web 開發",
        "雲端計算"
    ]

    generation_prompt = """
    為「{topic}」生成一個問答對。
    格式：
    問題：[問題內容]
    答案：[答案內容]
    """

    tasks = [
        InferenceTask(
            id=f"gen_{i}",
            prompt=generation_prompt.format(topic=topic),
            metadata={"topic": topic}
        )
        for i, topic in enumerate(topics)
    ]

    console.print(f"[yellow]🎲 生成主題數量:[/yellow] {len(tasks)}\n")

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    engine = BatchInferenceEngine(provider=provider, max_workers=3)

    results = engine.process_batch(tasks)

    # 顯示生成的數據
    for i, result in enumerate(results, 1):
        if result.success:
            topic = topics[int(result.id.split('_')[1])]
            console.print(f"[cyan]主題 {i}: {topic}[/cyan]")
            console.print(f"[green]{result.response}[/green]\n")


def demo_batch_with_error_handling():
    """帶錯誤處理的批量推理"""
    console.print(Panel.fit(
        "[bold cyan]示例 4: 錯誤處理和重試[/bold cyan]",
        border_style="cyan"
    ))

    tasks = [
        InferenceTask(id=f"task_{i}", prompt=f"解釋：{term}")
        for i, term in enumerate([
            "API", "REST", "JSON", "Docker", "Kubernetes"
        ])
    ]

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    engine = BatchInferenceEngine(provider=provider, max_workers=3)

    results = engine.process_batch(tasks)

    # 統計
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_time = sum(r.elapsed_time for r in results)
    avg_time = total_time / len(results) if results else 0

    console.print(f"[green]✅ 成功: {successful}/{len(tasks)}[/green]")
    if failed > 0:
        console.print(f"[red]❌ 失敗: {failed}/{len(tasks)}[/red]")

    console.print(f"\n[dim]📊 性能統計:[/dim]")
    console.print(f"[dim]  • 總耗時: {total_time:.2f} 秒[/dim]")
    console.print(f"[dim]  • 平均耗時: {avg_time:.2f} 秒/任務[/dim]")

    # 顯示失敗的任務
    if failed > 0:
        console.print("\n[yellow]失敗的任務:[/yellow]")
        for result in results:
            if not result.success:
                console.print(f"  • {result.id}: {result.error}")

    console.print()


def demo_save_results():
    """保存批量推理結果"""
    console.print(Panel.fit(
        "[bold cyan]示例 5: 保存結果到文件[/bold cyan]",
        border_style="cyan"
    ))

    tasks = [
        InferenceTask(id=f"task_{i}", prompt=f"用一句話描述：{item}")
        for i, item in enumerate(["GPU", "CPU", "TPU", "NPU"])
    ]

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    engine = BatchInferenceEngine(provider=provider, max_workers=2)

    results = engine.process_batch(tasks)

    # 保存為 JSON
    output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)

    json_file = output_dir / "results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            ensure_ascii=False,
            indent=2
        )

    console.print(f"[green]✅ 結果已保存到 JSON:[/green] {json_file}")

    # 保存為 CSV（使用 pandas）
    csv_file = output_dir / "results.csv"
    df = pd.DataFrame([
        {
            "ID": r.id,
            "提示": r.prompt[:50] + "..." if len(r.prompt) > 50 else r.prompt,
            "回答": r.response[:100] + "..." if len(r.response) > 100 else r.response,
            "成功": r.success,
            "耗時": f"{r.elapsed_time:.2f}s",
            "Tokens": r.tokens
        }
        for r in results
    ])

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    console.print(f"[green]✅ 結果已保存到 CSV:[/green] {csv_file}\n")


def demo_cost_estimation():
    """成本估算"""
    console.print(Panel.fit(
        "[bold cyan]示例 6: 成本估算[/bold cyan]",
        border_style="cyan"
    ))

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[yellow]⚠️  此示例需要 OpenAI API Key[/yellow]\n")
        return

    tasks = [
        InferenceTask(id=f"task_{i}", prompt=f"簡單介紹：{term}")
        for i, term in enumerate(["AI", "ML", "DL", "NLP", "CV"])
    ]

    engine = BatchInferenceEngine(provider="openai", model="gpt-4o-mini", max_workers=3)
    results = engine.process_batch(tasks)

    # 計算成本
    total_tokens = sum(r.tokens for r in results if r.success)
    input_tokens = total_tokens // 2  # 估算
    output_tokens = total_tokens // 2

    # GPT-4o-mini 價格 (2025年1月)
    cost_per_1m_input = 0.15
    cost_per_1m_output = 0.60

    total_cost = (input_tokens * cost_per_1m_input + output_tokens * cost_per_1m_output) / 1_000_000

    console.print(f"[yellow]💰 成本分析:[/yellow]")
    console.print(f"[dim]  • 總 tokens: {total_tokens}[/dim]")
    console.print(f"[dim]  • 輸入 tokens: {input_tokens} (${input_tokens * cost_per_1m_input / 1_000_000:.6f})[/dim]")
    console.print(f"[dim]  • 輸出 tokens: {output_tokens} (${output_tokens * cost_per_1m_output / 1_000_000:.6f})[/dim]")
    console.print(f"[bold green]  • 總成本: ${total_cost:.6f}[/bold green]")

    # 預估大規模成本
    scale_factors = [100, 1000, 10000, 100000]
    console.print(f"\n[yellow]📈 成本預估（按規模）:[/yellow]")

    for factor in scale_factors:
        scaled_cost = total_cost * factor
        console.print(f"[dim]  • {factor:>6}x: ${scaled_cost:>10.2f}[/dim]")

    console.print()


def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold magenta]📦 批量推理示例[/bold magenta]\n"
        "[dim]展示如何高效處理大量 LLM 推理任務[/dim]",
        border_style="magenta"
    ))

    try:
        demo_parallel_vs_sequential()
        demo_batch_classification()
        demo_batch_data_generation()
        demo_batch_with_error_handling()
        demo_save_results()
        demo_cost_estimation()

        console.print(Panel.fit(
            "[bold green]✅ 所有示例運行完成！[/bold green]\n"
            "[dim]提示: 批量處理可以顯著提升效率[/dim]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用戶中斷[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ 發生錯誤: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()
