#!/usr/bin/env python3
"""
模型比較工具

比較不同 LLM 模型的性能、輸出質量和成本。
幫助你選擇最適合你需求的模型。

比較維度：
- 輸出質量和一致性
- 推理速度
- 成本效益
- 特定任務表現

前置需求：
pip install openai anthropic ollama rich tabulate
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from openai import OpenAI
    from anthropic import Anthropic
    import ollama
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import track
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ 缺少依賴: {e}")
    print("請運行: pip install openai anthropic ollama rich python-dotenv")
    sys.exit(1)

load_dotenv()
console = Console()


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: str  # openai, anthropic, ollama
    model_id: str
    cost_per_1m_input: float = 0.0  # 每百萬輸入 tokens 的成本
    cost_per_1m_output: float = 0.0  # 每百萬輸出 tokens 的成本


@dataclass
class ComparisonResult:
    """比較結果"""
    model_name: str
    prompt: str
    response: str
    elapsed_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    success: bool = True
    error: Optional[str] = None


# 預定義模型配置
AVAILABLE_MODELS = {
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25
    ),
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00
    ),
    "llama3.1:8b": ModelConfig(
        name="Llama 3.1 8B",
        provider="ollama",
        model_id="llama3.1:8b",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0
    ),
    "mistral:7b": ModelConfig(
        name="Mistral 7B",
        provider="ollama",
        model_id="mistral:7b",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0
    )
}


class ModelComparator:
    """模型比較器"""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None

        # 初始化 API 客戶端
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI()

        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic()

    def infer(self, model_config: ModelConfig, prompt: str, temperature: float = 0.7) -> ComparisonResult:
        """執行推理"""
        start_time = time.time()

        try:
            if model_config.provider == "openai":
                result = self._infer_openai(model_config, prompt, temperature)
            elif model_config.provider == "anthropic":
                result = self._infer_anthropic(model_config, prompt, temperature)
            elif model_config.provider == "ollama":
                result = self._infer_ollama(model_config, prompt, temperature)
            else:
                raise ValueError(f"不支持的提供商: {model_config.provider}")

            result.elapsed_time = time.time() - start_time
            result.model_name = model_config.name

            return result

        except Exception as e:
            return ComparisonResult(
                model_name=model_config.name,
                prompt=prompt,
                response="",
                elapsed_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _infer_openai(self, config: ModelConfig, prompt: str, temperature: float) -> ComparisonResult:
        """OpenAI 推理"""
        if not self.openai_client:
            raise ValueError("OpenAI API Key 未設置")

        response = self.openai_client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )

        content = response.choices[0].message.content
        usage = response.usage

        cost = (
            usage.prompt_tokens * config.cost_per_1m_input +
            usage.completion_tokens * config.cost_per_1m_output
        ) / 1_000_000

        return ComparisonResult(
            model_name=config.name,
            prompt=prompt,
            response=content,
            elapsed_time=0,  # 將在外部設置
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=cost
        )

    def _infer_anthropic(self, config: ModelConfig, prompt: str, temperature: float) -> ComparisonResult:
        """Anthropic 推理"""
        if not self.anthropic_client:
            raise ValueError("Anthropic API Key 未設置")

        response = self.anthropic_client.messages.create(
            model=config.model_id,
            max_tokens=500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost = (
            input_tokens * config.cost_per_1m_input +
            output_tokens * config.cost_per_1m_output
        ) / 1_000_000

        return ComparisonResult(
            model_name=config.name,
            prompt=prompt,
            response=content,
            elapsed_time=0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )

    def _infer_ollama(self, config: ModelConfig, prompt: str, temperature: float) -> ComparisonResult:
        """Ollama 推理"""
        response = ollama.chat(
            model=config.model_id,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )

        content = response['message']['content']
        input_tokens = response.get('prompt_eval_count', 0)
        output_tokens = response.get('eval_count', 0)

        return ComparisonResult(
            model_name=config.name,
            prompt=prompt,
            response=content,
            elapsed_time=0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0  # 本地模型免費
        )

    def compare_models(
        self,
        model_keys: List[str],
        prompt: str,
        temperature: float = 0.7
    ) -> List[ComparisonResult]:
        """比較多個模型"""
        results = []

        console.print(f"\n[yellow]🔬 比較模型:[/yellow] {', '.join(model_keys)}")
        console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")

        for model_key in track(model_keys, description="執行推理..."):
            if model_key not in AVAILABLE_MODELS:
                console.print(f"[red]⚠️  未知模型: {model_key}[/red]")
                continue

            model_config = AVAILABLE_MODELS[model_key]
            result = self.infer(model_config, prompt, temperature)
            results.append(result)

        return results


def demo_basic_comparison():
    """基礎模型比較"""
    console.print(Panel.fit(
        "[bold cyan]示例 1: 基礎模型比較[/bold cyan]",
        border_style="cyan"
    ))

    comparator = ModelComparator()

    # 選擇可用的模型
    available_models = []
    if os.getenv("OPENAI_API_KEY"):
        available_models.append("gpt-4o-mini")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.append("claude-3-haiku")

    # 總是添加 Ollama 模型（假設已安裝）
    try:
        ollama.list()
        available_models.extend(["llama3.1:8b", "mistral:7b"])
    except:
        console.print("[yellow]⚠️  Ollama 不可用[/yellow]")

    if not available_models:
        console.print("[red]❌ 沒有可用的模型[/red]\n")
        return

    prompt = "解釋什麼是注意力機制（Attention Mechanism），用簡單的語言，100字以內"

    results = comparator.compare_models(available_models[:3], prompt)  # 限制為3個模型

    # 顯示結果
    for result in results:
        if result.success:
            console.print(Panel(
                f"[green]{result.response}[/green]\n\n"
                f"[dim]⏱️  耗時: {result.elapsed_time:.2f}s  |  "
                f"💰 成本: ${result.cost:.6f}  |  "
                f"📊 Tokens: {result.input_tokens + result.output_tokens}[/dim]",
                title=f"[bold cyan]{result.model_name}[/bold cyan]",
                border_style="cyan"
            ))
        else:
            console.print(f"[red]{result.model_name}: {result.error}[/red]\n")


def demo_performance_comparison():
    """性能比較"""
    console.print(Panel.fit(
        "[bold cyan]示例 2: 性能比較[/bold cyan]",
        border_style="cyan"
    ))

    comparator = ModelComparator()

    # 選擇模型
    models = []
    if os.getenv("OPENAI_API_KEY"):
        models.append("gpt-4o-mini")
    try:
        ollama.list()
        models.append("llama3.1:8b")
    except:
        pass

    if len(models) < 2:
        console.print("[yellow]⚠️  至少需要兩個模型進行比較[/yellow]\n")
        return

    prompt = "列出 3 個機器學習算法"

    # 多次運行取平均
    runs = 3
    all_results = []

    console.print(f"[yellow]🔁 每個模型運行 {runs} 次，計算平均性能...[/yellow]\n")

    for model_key in models:
        model_results = []
        for _ in range(runs):
            result = comparator.infer(AVAILABLE_MODELS[model_key], prompt)
            if result.success:
                model_results.append(result)

        if model_results:
            avg_time = sum(r.elapsed_time for r in model_results) / len(model_results)
            avg_tokens = sum(r.output_tokens for r in model_results) / len(model_results)

            all_results.append({
                "model": AVAILABLE_MODELS[model_key].name,
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "tokens_per_sec": avg_tokens / avg_time if avg_time > 0 else 0
            })

    # 顯示性能表格
    table = Table(title="性能比較", show_header=True, header_style="bold cyan")
    table.add_column("模型", style="cyan")
    table.add_column("平均耗時", style="green")
    table.add_column("平均 Tokens", style="yellow")
    table.add_column("速度 (tok/s)", style="magenta")

    for result in all_results:
        table.add_row(
            result["model"],
            f"{result['avg_time']:.2f}s",
            f"{result['avg_tokens']:.0f}",
            f"{result['tokens_per_sec']:.1f}"
        )

    console.print(table)
    console.print()


def demo_cost_comparison():
    """成本比較"""
    console.print(Panel.fit(
        "[bold cyan]示例 3: 成本比較[/bold cyan]",
        border_style="cyan"
    ))

    # 顯示成本對比表
    table = Table(title="模型成本對比 (每百萬 tokens)", show_header=True, header_style="bold cyan")
    table.add_column("模型", style="cyan", width=20)
    table.add_column("輸入成本", style="green", width=15)
    table.add_column("輸出成本", style="yellow", width=15)
    table.add_column("類型", style="magenta", width=10)

    for model_key, config in AVAILABLE_MODELS.items():
        model_type = "API" if config.provider in ["openai", "anthropic"] else "本地"

        table.add_row(
            config.name,
            f"${config.cost_per_1m_input:.2f}" if config.cost_per_1m_input > 0 else "免費",
            f"${config.cost_per_1m_output:.2f}" if config.cost_per_1m_output > 0 else "免費",
            model_type
        )

    console.print(table)

    # 成本預估示例
    console.print("\n[yellow]💰 成本預估示例（100萬次請求，每次平均 500 tokens）:[/yellow]\n")

    tokens_per_request = 500
    requests = 1_000_000
    total_tokens = tokens_per_request * requests
    input_tokens = total_tokens * 0.3  # 假設 30% 是輸入
    output_tokens = total_tokens * 0.7  # 假設 70% 是輸出

    for model_key in ["gpt-4o-mini", "claude-3-haiku", "llama3.1:8b"]:
        config = AVAILABLE_MODELS[model_key]
        cost = (
            (input_tokens / 1_000_000) * config.cost_per_1m_input +
            (output_tokens / 1_000_000) * config.cost_per_1m_output
        )

        console.print(f"[cyan]{config.name:20}[/cyan] [green]${cost:>10,.2f}[/green]")

    console.print()


def demo_quality_comparison():
    """輸出質量比較"""
    console.print(Panel.fit(
        "[bold cyan]示例 4: 輸出質量比較[/bold cyan]",
        border_style="cyan"
    ))

    comparator = ModelComparator()

    # 使用複雜任務測試
    prompt = """
    請為以下情境寫一個專業的郵件回覆（50字以內）：
    客戶抱怨產品延遲交貨，需要給予解釋和補償方案。
    """

    models = []
    if os.getenv("OPENAI_API_KEY"):
        models.append("gpt-4o-mini")
    try:
        ollama.list()
        models.append("llama3.1:8b")
    except:
        pass

    if not models:
        console.print("[yellow]⚠️  沒有可用的模型[/yellow]\n")
        return

    results = comparator.compare_models(models, prompt)

    console.print("[yellow]📝 並排比較輸出質量:[/yellow]\n")

    for result in results:
        if result.success:
            console.print(f"[bold cyan]>>> {result.model_name}[/bold cyan]")
            console.print(result.response)
            console.print()


def demo_save_comparison():
    """保存比較結果"""
    console.print(Panel.fit(
        "[bold cyan]示例 5: 保存比較結果[/bold cyan]",
        border_style="cyan"
    ))

    comparator = ModelComparator()

    models = []
    if os.getenv("OPENAI_API_KEY"):
        models.append("gpt-4o-mini")
    try:
        ollama.list()
        models.append("llama3.1:8b")
    except:
        pass

    if not models:
        console.print("[yellow]⚠️  沒有可用的模型[/yellow]\n")
        return

    prompt = "什麼是深度學習？用一句話回答。"

    results = comparator.compare_models(models, prompt)

    # 保存結果
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "prompt": prompt,
                "timestamp": timestamp,
                "results": [asdict(r) for r in results]
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    console.print(f"[green]✅ 比較結果已保存:[/green] {output_file}\n")


def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold magenta]⚖️  模型比較工具[/bold magenta]\n"
        "[dim]比較不同 LLM 模型的性能、質量和成本[/dim]",
        border_style="magenta"
    ))

    try:
        demo_basic_comparison()
        demo_performance_comparison()
        demo_cost_comparison()
        demo_quality_comparison()
        demo_save_comparison()

        console.print(Panel.fit(
            "[bold green]✅ 所有比較完成！[/bold green]\n"
            "[dim]提示: 根據你的需求選擇最合適的模型[/dim]",
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
