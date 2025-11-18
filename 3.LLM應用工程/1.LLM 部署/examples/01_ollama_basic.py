#!/usr/bin/env python3
"""
Ollama 本地部署基礎示例

這個腳本展示如何使用 Ollama 在本地運行開源 LLM 模型。
Ollama 提供了類似 Docker 的使用體驗，讓本地部署變得極其簡單。

前置需求：
1. 安裝 Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. 下載模型: ollama pull llama3.1:8b
3. 安裝 Python 包: pip install ollama rich

特點：
- 零配置 GPU 設置
- 自動模型管理
- 支持流式輸出
- 內建 API 服務器
"""

import os
import sys
import time
from typing import List, Dict

try:
    import ollama
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
except ImportError as e:
    print(f"❌ 缺少依賴: {e}")
    print("請運行: pip install ollama rich")
    sys.exit(1)

console = Console()


def check_ollama_service() -> bool:
    """檢查 Ollama 服務是否運行"""
    try:
        ollama.list()
        return True
    except Exception as e:
        console.print(f"[red]❌ Ollama 服務未運行: {e}[/red]")
        console.print("[yellow]請先啟動 Ollama: ollama serve[/yellow]")
        return False


def list_available_models() -> List[Dict]:
    """列出已下載的模型"""
    try:
        models = ollama.list()
        return models.get('models', [])
    except Exception as e:
        console.print(f"[red]獲取模型列表失敗: {e}[/red]")
        return []


def basic_chat(model: str = "llama3.1:8b"):
    """基礎對話示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 1: 基礎對話[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "用三句話解釋什麼是 LLM（大型語言模型）"
    console.print(f"\n[yellow]💬 提示:[/yellow] {prompt}\n")

    start_time = time.time()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )

        elapsed = time.time() - start_time
        content = response['message']['content']

        console.print(f"[green]🤖 回答:[/green]")
        console.print(Markdown(content))
        console.print(f"\n[dim]⏱️  耗時: {elapsed:.2f} 秒[/dim]\n")

        return response

    except Exception as e:
        console.print(f"[red]❌ 對話失敗: {e}[/red]")
        return None


def streaming_chat(model: str = "llama3.1:8b"):
    """流式輸出示例 - 即時顯示生成內容"""
    console.print(Panel.fit(
        "[bold cyan]示例 2: 流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "寫一首關於人工智慧的四行短詩"
    console.print(f"\n[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答 (實時生成):[/green] ", end="")

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
        console.print(f"\n\n[dim]⏱️  耗時: {elapsed:.2f} 秒[/dim]\n")

    except Exception as e:
        console.print(f"\n[red]❌ 流式輸出失敗: {e}[/red]")


def chat_with_system_prompt(model: str = "llama3.1:8b"):
    """帶系統提示的對話 - 定義 AI 的角色和行為"""
    console.print(Panel.fit(
        "[bold cyan]示例 3: 系統提示詞[/bold cyan]",
        border_style="cyan"
    ))

    system_prompt = "你是一位專業的 Python 程式設計導師，擅長用簡單的例子解釋複雜的概念。"

    messages = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': '什麼是裝飾器（decorator）？請給我一個簡單的例子。',
        }
    ]

    console.print(f"[yellow]🎭 系統角色:[/yellow] {system_prompt}")
    console.print(f"[yellow]💬 用戶提問:[/yellow] {messages[1]['content']}\n")

    try:
        response = ollama.chat(model=model, messages=messages)
        content = response['message']['content']

        console.print("[green]🤖 回答:[/green]")
        console.print(Markdown(content))
        console.print()

    except Exception as e:
        console.print(f"[red]❌ 對話失敗: {e}[/red]")


def multi_turn_conversation(model: str = "llama3.1:8b"):
    """多輪對話 - 保持上下文"""
    console.print(Panel.fit(
        "[bold cyan]示例 4: 多輪對話（保持上下文）[/bold cyan]",
        border_style="cyan"
    ))

    messages = []

    # 第一輪對話
    user_msg_1 = "我正在學習機器學習，推薦一個入門算法"
    messages.append({'role': 'user', 'content': user_msg_1})

    console.print(f"[yellow]👤 用戶 (第1輪):[/yellow] {user_msg_1}\n")

    response_1 = ollama.chat(model=model, messages=messages)
    assistant_msg_1 = response_1['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_msg_1})

    console.print(f"[green]🤖 助理:[/green] {assistant_msg_1}\n")

    # 第二輪對話（基於第一輪的上下文）
    user_msg_2 = "能給我這個算法的 Python 實現嗎？"
    messages.append({'role': 'user', 'content': user_msg_2})

    console.print(f"[yellow]👤 用戶 (第2輪):[/yellow] {user_msg_2}\n")

    response_2 = ollama.chat(model=model, messages=messages)
    assistant_msg_2 = response_2['message']['content']

    console.print("[green]🤖 助理:[/green]")
    console.print(Markdown(assistant_msg_2))
    console.print()


def compare_temperatures(model: str = "llama3.1:8b"):
    """溫度參數比較 - 控制輸出的隨機性"""
    console.print(Panel.fit(
        "[bold cyan]示例 5: 溫度參數影響[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "給這個產品起一個創意的名字：一款智能筆記應用"
    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")

    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        console.print(f"[cyan]🌡️  Temperature = {temp}:[/cyan]")

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': temp,
                    'num_predict': 50,  # 限制輸出長度
                }
            )

            console.print(f"  {response['response'].strip()}\n")

        except Exception as e:
            console.print(f"[red]  ❌ 生成失敗: {e}[/red]\n")


def get_model_info(model: str = "llama3.1:8b"):
    """獲取模型信息"""
    console.print(Panel.fit(
        "[bold cyan]示例 6: 模型信息[/bold cyan]",
        border_style="cyan"
    ))

    try:
        info = ollama.show(model)

        console.print(f"[yellow]📊 模型:[/yellow] {model}")
        console.print(f"[yellow]🏷️  家族:[/yellow] {info.get('details', {}).get('family', 'N/A')}")
        console.print(f"[yellow]📐 參數量:[/yellow] {info.get('details', {}).get('parameter_size', 'N/A')}")
        console.print(f"[yellow]🔢 量化:[/yellow] {info.get('details', {}).get('quantization_level', 'N/A')}")
        console.print()

    except Exception as e:
        console.print(f"[red]❌ 獲取模型信息失敗: {e}[/red]")


def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold magenta]🦙 Ollama 本地部署示例[/bold magenta]\n"
        "[dim]本示例展示如何使用 Ollama 在本地運行 LLM[/dim]",
        border_style="magenta"
    ))

    # 檢查服務
    if not check_ollama_service():
        return

    # 列出可用模型
    models = list_available_models()
    if not models:
        console.print("[red]❌ 沒有找到已下載的模型[/red]")
        console.print("[yellow]請先下載模型，例如: ollama pull llama3.1:8b[/yellow]")
        return

    console.print("\n[green]✅ 已下載的模型:[/green]")
    for model in models:
        console.print(f"  • {model['name']} ({model.get('size', 'N/A')})")

    # 使用第一個模型
    model_name = models[0]['name']
    console.print(f"\n[cyan]🎯 使用模型: {model_name}[/cyan]\n")

    # 運行所有示例
    try:
        basic_chat(model_name)
        streaming_chat(model_name)
        chat_with_system_prompt(model_name)
        multi_turn_conversation(model_name)
        compare_temperatures(model_name)
        get_model_info(model_name)

        console.print(Panel.fit(
            "[bold green]✅ 所有示例運行完成！[/bold green]\n"
            "[dim]提示: 你可以修改代碼中的提示詞來嘗試不同的對話[/dim]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用戶中斷[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ 發生錯誤: {e}[/red]")


if __name__ == "__main__":
    main()
