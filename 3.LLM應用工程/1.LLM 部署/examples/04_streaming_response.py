#!/usr/bin/env python3
"""
流式響應示例

展示如何實現 LLM 的流式輸出，提供類似 ChatGPT 的實時響應體驗。
流式輸出能顯著提升用戶體驗，讓用戶無需等待完整響應即可開始閱讀。

支持的提供商：
- OpenAI API
- Anthropic Claude
- Ollama
- vLLM 服務器

前置需求：
pip install openai anthropic ollama rich asyncio aiohttp
"""

import os
import sys
import time
import asyncio
from typing import AsyncGenerator, Generator

try:
    from openai import OpenAI
    import ollama
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.markdown import Markdown
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ 缺少依賴: {e}")
    print("請運行: pip install openai ollama rich python-dotenv")
    sys.exit(1)

load_dotenv()
console = Console()


def stream_openai(prompt: str, model: str = "gpt-4o-mini"):
    """OpenAI 流式輸出"""
    console.print(Panel.fit(
        "[bold cyan]OpenAI 流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠️  未設置 OPENAI_API_KEY，跳過此示例[/yellow]\n")
        return

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答:[/green] ", end="")

    client = OpenAI(api_key=api_key)
    start_time = time.time()
    full_response = ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.7
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                console.print(content, end="", style="green")

        elapsed = time.time() - start_time
        console.print(f"\n\n[dim]⏱️  耗時: {elapsed:.2f} 秒[/dim]")
        console.print(f"[dim]📝 總字數: {len(full_response)} 字元[/dim]\n")

    except Exception as e:
        console.print(f"\n[red]❌ 錯誤: {e}[/red]\n")


def stream_ollama(prompt: str, model: str = "llama3.1:8b"):
    """Ollama 流式輸出"""
    console.print(Panel.fit(
        "[bold cyan]Ollama 流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答:[/green] ", end="")

    start_time = time.time()
    full_response = ""

    try:
        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )

        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            console.print(content, end="", style="green")

        elapsed = time.time() - start_time
        console.print(f"\n\n[dim]⏱️  耗時: {elapsed:.2f} 秒[/dim]")
        console.print(f"[dim]📝 總字數: {len(full_response)} 字元[/dim]\n")

    except Exception as e:
        console.print(f"\n[red]❌ 錯誤: {e}[/red]")
        console.print("[yellow]提示: 確保 Ollama 正在運行並已下載該模型[/yellow]\n")


def stream_with_live_markdown(prompt: str):
    """使用 Rich Live 動態渲染 Markdown"""
    console.print(Panel.fit(
        "[bold cyan]動態 Markdown 渲染[/bold cyan]",
        border_style="cyan"
    ))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠️  未設置 OPENAI_API_KEY，使用 Ollama[/yellow]\n")
        use_openai = False
    else:
        use_openai = True

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")

    full_response = ""

    try:
        if use_openai:
            client = OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
        else:
            stream = ollama.chat(
                model="llama3.1:8b",
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )

        # 使用 Rich Live 動態更新顯示
        with Live(Markdown(full_response), refresh_per_second=4, console=console) as live:
            if use_openai:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        live.update(Markdown(full_response))
            else:
                for chunk in stream:
                    full_response += chunk['message']['content']
                    live.update(Markdown(full_response))

        console.print()

    except Exception as e:
        console.print(f"[red]❌ 錯誤: {e}[/red]\n")


async def async_stream_openai(prompt: str, model: str = "gpt-4o-mini") -> AsyncGenerator[str, None]:
    """異步流式輸出（適用於 Web 應用）"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield "❌ 未設置 OPENAI_API_KEY"
        return

    client = OpenAI(api_key=api_key)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                # 模擬網絡延遲
                await asyncio.sleep(0.01)

    except Exception as e:
        yield f"\n❌ 錯誤: {e}"


async def demo_async_streaming():
    """演示異步流式輸出"""
    console.print(Panel.fit(
        "[bold cyan]異步流式輸出 (適用於 Web 應用)[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "解釋什麼是異步編程，以及它的優勢"
    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答 (異步生成):[/green] ", end="")

    async for chunk in async_stream_openai(prompt):
        console.print(chunk, end="", style="green")

    console.print("\n")


def stream_with_token_counting(prompt: str):
    """帶 Token 計數的流式輸出"""
    console.print(Panel.fit(
        "[bold cyan]帶 Token 統計的流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠️  未設置 OPENAI_API_KEY，跳過此示例[/yellow]\n")
        return

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答:[/green]\n")

    client = OpenAI(api_key=api_key)
    start_time = time.time()
    full_response = ""
    chunk_count = 0

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True}  # 包含使用統計
        )

        for chunk in stream:
            chunk_count += 1

            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                console.print(content, end="", style="green")

            # 最後一個 chunk 包含使用統計
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = chunk.usage
                elapsed = time.time() - start_time

                console.print("\n\n[dim]📊 詳細統計:[/dim]")
                console.print(f"[dim]  • 接收 chunks: {chunk_count}[/dim]")
                console.print(f"[dim]  • 輸入 tokens: {usage.prompt_tokens}[/dim]")
                console.print(f"[dim]  • 輸出 tokens: {usage.completion_tokens}[/dim]")
                console.print(f"[dim]  • 總計 tokens: {usage.total_tokens}[/dim]")
                console.print(f"[dim]  • 耗時: {elapsed:.2f} 秒[/dim]")
                console.print(f"[dim]  • 速度: {usage.completion_tokens / elapsed:.1f} tokens/秒[/dim]")

                cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
                console.print(f"[dim]  • 成本: ${cost:.6f}[/dim]\n")

    except Exception as e:
        console.print(f"\n[red]❌ 錯誤: {e}[/red]\n")


def compare_streaming_vs_blocking():
    """比較流式 vs 阻塞式輸出"""
    console.print(Panel.fit(
        "[bold cyan]流式 vs 阻塞式比較[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "列出 5 個機器學習的應用場景"

    # 檢查可用的服務
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_ollama = True  # 假設 Ollama 可用

    if not has_openai and not has_ollama:
        console.print("[yellow]⚠️  需要至少一個可用的 LLM 服務[/yellow]\n")
        return

    # 測試阻塞式
    console.print("[yellow]1️⃣  阻塞式輸出（傳統方式）:[/yellow]")
    console.print(f"[dim]提示: {prompt}[/dim]\n")

    if has_openai:
        client = OpenAI()
        start = time.time()
        console.print("[cyan]⏳ 等待完整響應...[/cyan]")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            ttfr = time.time() - start  # Time To First Response
            content = response.choices[0].message.content

            console.print(f"[green]✅ 收到完整響應 (耗時 {ttfr:.2f} 秒)[/green]")
            console.print(content)
            console.print()

        except Exception as e:
            console.print(f"[red]❌ 錯誤: {e}[/red]\n")
            return

    # 測試流式
    console.print("[yellow]2️⃣  流式輸出（現代方式）:[/yellow]")
    console.print(f"[dim]提示: {prompt}[/dim]\n")

    if has_openai:
        client = OpenAI()
        start = time.time()
        first_token_time = None
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time() - start
                        console.print(f"[green]✅ 首個 token (耗時 {first_token_time:.2f} 秒)[/green]")

                    content = chunk.choices[0].delta.content
                    full_response += content
                    console.print(content, end="", style="green")

            total_time = time.time() - start
            console.print(f"\n\n[dim]📊 性能對比:[/dim]")
            console.print(f"[dim]  • 首 token 延遲: {first_token_time:.2f} 秒 ⚡[/dim]")
            console.print(f"[dim]  • 總耗時: {total_time:.2f} 秒[/dim]")
            console.print(f"[dim]  • 用戶感知時間減少: {((ttfr - first_token_time) / ttfr * 100):.1f}%[/dim]\n")

        except Exception as e:
            console.print(f"\n[red]❌ 錯誤: {e}[/red]\n")


def stream_with_retry():
    """帶重試機制的流式輸出"""
    console.print(Panel.fit(
        "[bold cyan]帶重試機制的流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠️  未設置 OPENAI_API_KEY，跳過此示例[/yellow]\n")
        return

    prompt = "解釋什麼是容錯設計"
    max_retries = 3

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}")
    console.print(f"[yellow]🔄 最大重試次數:[/yellow] {max_retries}\n")
    console.print("[green]🤖 回答:[/green] ", end="")

    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=30.0
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    console.print(chunk.choices[0].delta.content, end="", style="green")

            console.print("\n")
            break  # 成功，退出重試循環

        except Exception as e:
            if attempt < max_retries - 1:
                console.print(f"\n[yellow]⚠️  發生錯誤，重試中 ({attempt + 1}/{max_retries})...[/yellow]")
                time.sleep(2 ** attempt)  # 指數退避
            else:
                console.print(f"\n[red]❌ 達到最大重試次數: {e}[/red]\n")


def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold magenta]⚡ LLM 流式響應示例[/bold magenta]\n"
        "[dim]展示如何實現實時的流式輸出[/dim]",
        border_style="magenta"
    ))

    prompt = "用簡單的語言解釋什麼是量子計算"

    try:
        # OpenAI 流式輸出
        stream_openai(prompt)

        # Ollama 流式輸出
        stream_ollama(prompt)

        # 動態 Markdown 渲染
        markdown_prompt = "寫一個 Python 快速排序算法，包含註釋"
        stream_with_live_markdown(markdown_prompt)

        # 異步流式輸出
        asyncio.run(demo_async_streaming())

        # Token 計數
        stream_with_token_counting("什麼是注意力機制？")

        # 流式 vs 阻塞式比較
        compare_streaming_vs_blocking()

        # 帶重試機制
        stream_with_retry()

        console.print(Panel.fit(
            "[bold green]✅ 所有示例運行完成！[/bold green]\n"
            "[dim]提示: 流式輸出能顯著提升用戶體驗[/dim]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用戶中斷[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ 發生錯誤: {e}[/red]")


if __name__ == "__main__":
    main()
