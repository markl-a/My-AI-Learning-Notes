#!/usr/bin/env python3
"""
OpenAI API 使用示例

這個腳本展示如何使用 OpenAI API 進行各種常見任務。
包括基礎對話、結構化輸出、流式響應、函數調用等高級功能。

前置需求：
1. 獲取 API Key: https://platform.openai.com/api-keys
2. 設置環境變數: export OPENAI_API_KEY="your-key"
3. 安裝依賴: pip install openai python-dotenv rich

特點：
- 頂級模型性能
- 豐富的功能特性
- 按用量計費
- 無需本地 GPU
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.json import JSON
except ImportError as e:
    print(f"❌ 缺少依賴: {e}")
    print("請運行: pip install openai python-dotenv rich")
    sys.exit(1)

# 載入環境變數
load_dotenv()

console = Console()


def check_api_key() -> bool:
    """檢查 API Key 是否設置"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]❌ 未找到 OPENAI_API_KEY[/red]")
        console.print("[yellow]請設置環境變數或在 .env 文件中配置[/yellow]")
        return False
    return True


def basic_chat():
    """基礎對話示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 1: 基礎對話[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()

    messages = [
        {"role": "system", "content": "你是一位專業的技術寫作專家。"},
        {"role": "user", "content": "用三個段落解釋什麼是 Transformer 架構"}
    ]

    console.print("[yellow]💬 提示:[/yellow] 用三個段落解釋什麼是 Transformer 架構\n")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        elapsed = time.time() - start_time
        content = response.choices[0].message.content

        console.print("[green]🤖 回答:[/green]")
        console.print(Markdown(content))

        # 顯示使用統計
        usage = response.usage
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

        console.print(f"\n[dim]📊 統計信息:[/dim]")
        console.print(f"[dim]  • 輸入 tokens: {usage.prompt_tokens}[/dim]")
        console.print(f"[dim]  • 輸出 tokens: {usage.completion_tokens}[/dim]")
        console.print(f"[dim]  • 總計: {usage.total_tokens}[/dim]")
        console.print(f"[dim]  • 成本估算: ${cost:.6f}[/dim]")
        console.print(f"[dim]  • 耗時: {elapsed:.2f} 秒[/dim]\n")

    except Exception as e:
        console.print(f"[red]❌ API 調用失敗: {e}[/red]")


def streaming_chat():
    """流式輸出示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 2: 流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()

    prompt = "寫一個關於機器學習的創意故事開頭（100字內）"
    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答 (實時生成):[/green]\n")

    start_time = time.time()
    full_content = ""

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=200
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                console.print(content, end="", style="green")

        elapsed = time.time() - start_time
        console.print(f"\n\n[dim]⏱️  耗時: {elapsed:.2f} 秒[/dim]\n")

    except Exception as e:
        console.print(f"\n[red]❌ 流式輸出失敗: {e}[/red]")


def structured_output():
    """結構化輸出（JSON 模式）"""
    console.print(Panel.fit(
        "[bold cyan]示例 3: 結構化輸出 (JSON)[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()

    prompt = """
    請以 JSON 格式列出 3 個流行的機器學習框架。
    每個框架包含：name（名稱）、language（主要語言）、description（簡短描述）
    """

    console.print(f"[yellow]💬 提示:[/yellow] 請輸出 3 個機器學習框架的 JSON 數據\n")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一個數據格式化專家，總是返回有效的 JSON。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        console.print("[green]🤖 結構化輸出:[/green]")
        console.print(JSON(json.dumps(data, indent=2, ensure_ascii=False)))
        console.print()

    except json.JSONDecodeError as e:
        console.print(f"[red]❌ JSON 解析失敗: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ API 調用失敗: {e}[/red]")


def multi_turn_conversation():
    """多輪對話示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 4: 多輪對話[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()
    messages = [
        {"role": "system", "content": "你是一位資深的 Python 開發者。"}
    ]

    conversations = [
        "如何在 Python 中讀取 CSV 文件？",
        "如果文件很大怎麼辦？",
        "能給我一個完整的代碼示例嗎？"
    ]

    for i, user_msg in enumerate(conversations, 1):
        console.print(f"[yellow]👤 用戶 (第{i}輪):[/yellow] {user_msg}\n")

        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
                max_tokens=300
            )

            assistant_msg = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_msg})

            console.print("[green]🤖 助理:[/green]")
            console.print(Markdown(assistant_msg))
            console.print()

        except Exception as e:
            console.print(f"[red]❌ 對話失敗: {e}[/red]\n")
            break


def temperature_comparison():
    """溫度參數比較"""
    console.print(Panel.fit(
        "[bold cyan]示例 5: 溫度參數影響[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()
    prompt = "為一家 AI 初創公司想一個有創意的名字"

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")

    temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]

    table = Table(title="不同溫度下的輸出", show_header=True, header_style="bold cyan")
    table.add_column("溫度", style="cyan", width=10)
    table.add_column("輸出結果", style="green")

    for temp in temperatures:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=50
            )

            result = response.choices[0].message.content.strip()
            table.add_row(f"{temp:.1f}", result)

        except Exception as e:
            table.add_row(f"{temp:.1f}", f"[red]錯誤: {e}[/red]")

    console.print(table)
    console.print()


def function_calling():
    """函數調用示例（工具使用）"""
    console.print(Panel.fit(
        "[bold cyan]示例 6: 函數調用 (Function Calling)[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()

    # 定義可用的函數
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "獲取指定城市的天氣信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名稱，例如：台北、東京"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "溫度單位"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    user_query = "台北今天天氣如何？"
    console.print(f"[yellow]💬 用戶查詢:[/yellow] {user_query}\n")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_query}],
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            console.print("[green]🔧 函數調用檢測:[/green]")
            console.print(f"  • 函數名: {function_name}")
            console.print(f"  • 參數: {json.dumps(function_args, ensure_ascii=False)}")
            console.print("\n[dim]註：實際應用中，這裡會調用真實的天氣 API[/dim]\n")

            # 模擬函數返回結果
            mock_weather = {
                "city": function_args.get("city"),
                "temperature": 25,
                "condition": "晴朗",
                "humidity": 60
            }

            # 將函數結果發送回模型
            messages = [
                {"role": "user", "content": user_query},
                message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(mock_weather, ensure_ascii=False)
                }
            ]

            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            console.print("[green]🤖 最終回答:[/green]")
            console.print(final_response.choices[0].message.content)
            console.print()

        else:
            console.print("[yellow]ℹ️  未檢測到函數調用[/yellow]\n")

    except Exception as e:
        console.print(f"[red]❌ 函數調用失敗: {e}[/red]")


def batch_processing():
    """批量處理示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 7: 批量處理[/bold cyan]",
        border_style="cyan"
    ))

    client = OpenAI()

    prompts = [
        "用一句話總結：機器學習",
        "用一句話總結：深度學習",
        "用一句話總結：強化學習",
        "用一句話總結：遷移學習"
    ]

    console.print(f"[yellow]📋 批量處理 {len(prompts)} 個提示...[/yellow]\n")

    results = []
    total_tokens = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts, 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )

            result = response.choices[0].message.content
            tokens = response.usage.total_tokens
            total_tokens += tokens

            results.append({
                "prompt": prompt,
                "response": result,
                "tokens": tokens
            })

            console.print(f"[cyan]✓ {i}/{len(prompts)}[/cyan] {result}")

        except Exception as e:
            console.print(f"[red]✗ {i}/{len(prompts)} 失敗: {e}[/red]")

    elapsed = time.time() - start_time
    avg_time = elapsed / len(prompts)
    cost = (total_tokens * 0.60) / 1_000_000

    console.print(f"\n[dim]📊 批量處理統計:[/dim]")
    console.print(f"[dim]  • 總耗時: {elapsed:.2f} 秒[/dim]")
    console.print(f"[dim]  • 平均耗時: {avg_time:.2f} 秒/請求[/dim]")
    console.print(f"[dim]  • 總 tokens: {total_tokens}[/dim]")
    console.print(f"[dim]  • 估算成本: ${cost:.6f}[/dim]\n")


def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold magenta]🤖 OpenAI API 使用示例[/bold magenta]\n"
        "[dim]展示 OpenAI API 的各種功能和最佳實踐[/dim]",
        border_style="magenta"
    ))

    # 檢查 API Key
    if not check_api_key():
        return

    console.print("[green]✅ API Key 已配置[/green]\n")

    # 運行所有示例
    try:
        basic_chat()
        streaming_chat()
        structured_output()
        multi_turn_conversation()
        temperature_comparison()
        function_calling()
        batch_processing()

        console.print(Panel.fit(
            "[bold green]✅ 所有示例運行完成！[/bold green]\n"
            "[dim]提示: 查看你的 OpenAI 使用量: https://platform.openai.com/usage[/dim]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用戶中斷[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ 發生錯誤: {e}[/red]")


if __name__ == "__main__":
    main()
