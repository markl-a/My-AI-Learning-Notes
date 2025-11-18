#!/usr/bin/env python3
"""
Hugging Face 本地模型加載示例

展示如何使用 Transformers 庫在本地加載和運行開源 LLM。
包含量化技術、性能優化和記憶體管理等高級功能。

前置需求：
1. GPU (推薦): NVIDIA GPU with CUDA support
2. 安裝依賴: pip install transformers accelerate bitsandbytes torch rich
3. Hugging Face Token (可選): 用於下載受限模型

硬體需求：
- 最低: 12GB VRAM (4-bit 量化的 7B 模型)
- 推薦: 24GB VRAM (8-bit 量化的 13B 模型)
- 理想: 40GB+ VRAM (fp16 的大型模型)
"""

import os
import sys
import time
import torch
from typing import Optional

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
        TextStreamer
    )
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import psutil
except ImportError as e:
    print(f"❌ 缺少依賴: {e}")
    print("請運行: pip install transformers accelerate bitsandbytes torch rich psutil")
    sys.exit(1)

console = Console()


def check_gpu() -> bool:
    """檢查 GPU 可用性"""
    if not torch.cuda.is_available():
        console.print("[yellow]⚠️  未檢測到 CUDA GPU[/yellow]")
        console.print("[yellow]   將使用 CPU 運行（速度會較慢）[/yellow]")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    console.print(f"[green]✅ 檢測到 GPU: {gpu_name}[/green]")
    console.print(f"[green]   顯存: {gpu_memory:.1f} GB[/green]\n")

    return True


def get_memory_usage() -> dict:
    """獲取記憶體使用情況"""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        vram_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
    else:
        vram_mb = 0
        vram_reserved_mb = 0

    return {
        "ram_mb": ram_mb,
        "vram_mb": vram_mb,
        "vram_reserved_mb": vram_reserved_mb
    }


def load_model_4bit(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """
    加載 4-bit 量化模型（最節省記憶體）

    4-bit 量化可以將記憶體需求降低約 75%，
    對於 7B 模型，只需要約 4GB VRAM
    """
    console.print(Panel.fit(
        "[bold cyan]示例 1: 4-bit 量化模型加載[/bold cyan]",
        border_style="cyan"
    ))

    console.print(f"[yellow]📦 模型:[/yellow] {model_name}")
    console.print("[yellow]⚙️  量化:[/yellow] 4-bit (NF4)")
    console.print("[yellow]💾 預期記憶體:[/yellow] ~4GB VRAM (對於 7B 模型)\n")

    mem_before = get_memory_usage()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("正在加載模型...", total=None)

        try:
            # 配置 4-bit 量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # 加載 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 加載模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]❌ 模型加載失敗: {e}[/red]")
            return None, None

    mem_after = get_memory_usage()

    # 顯示記憶體使用
    console.print("\n[green]✅ 模型加載成功！[/green]")
    console.print(f"[dim]📊 RAM 使用: {mem_after['ram_mb']:.0f} MB (+{mem_after['ram_mb'] - mem_before['ram_mb']:.0f} MB)[/dim]")
    if torch.cuda.is_available():
        console.print(f"[dim]🎮 VRAM 使用: {mem_after['vram_mb']:.0f} MB[/dim]\n")

    return model, tokenizer


def basic_inference(model, tokenizer):
    """基礎推理示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 2: 基礎推理[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "解釋什麼是注意力機制（Attention Mechanism），用簡單的語言"

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答:[/green]\n")

    # 準備輸入
    messages = [
        {"role": "system", "content": "你是一位資深的機器學習專家。"},
        {"role": "user", "content": prompt}
    ]

    # 應用聊天模板
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )

    elapsed = time.time() - start_time

    # 解碼輸出
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    console.print(response)

    # 性能統計
    tokens_generated = outputs.shape[-1] - inputs['input_ids'].shape[-1]
    tokens_per_sec = tokens_generated / elapsed

    console.print(f"\n[dim]📊 性能統計:[/dim]")
    console.print(f"[dim]  • 生成 tokens: {tokens_generated}[/dim]")
    console.print(f"[dim]  • 耗時: {elapsed:.2f} 秒[/dim]")
    console.print(f"[dim]  • 速度: {tokens_per_sec:.2f} tokens/秒[/dim]\n")


def streaming_inference(model, tokenizer):
    """流式推理示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 3: 流式輸出[/bold cyan]",
        border_style="cyan"
    ))

    prompt = "寫一個 Python 函數來計算斐波那契數列"

    console.print(f"[yellow]💬 提示:[/yellow] {prompt}\n")
    console.print("[green]🤖 回答 (實時生成):[/green]\n")

    messages = [
        {"role": "user", "content": prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 使用 TextStreamer 實現流式輸出
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )

    elapsed = time.time() - start_time
    console.print(f"\n[dim]⏱️  耗時: {elapsed:.2f} 秒[/dim]\n")


def batch_inference(model, tokenizer):
    """批量推理示例"""
    console.print(Panel.fit(
        "[bold cyan]示例 4: 批量推理[/bold cyan]",
        border_style="cyan"
    ))

    prompts = [
        "什麼是機器學習？用一句話回答。",
        "什麼是深度學習？用一句話回答。",
        "什麼是神經網路？用一句話回答。"
    ]

    console.print(f"[yellow]📋 批量處理 {len(prompts)} 個提示...[/yellow]\n")

    # 設置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 批量編碼
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    start_time = time.time()

    # 批量生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    elapsed = time.time() - start_time

    # 解碼結果
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        response = tokenizer.decode(output, skip_special_tokens=True)
        # 移除原始提示，只保留回答
        if prompt in response:
            response = response.replace(prompt, "").strip()

        console.print(f"[cyan]{i}. 提示:[/cyan] {prompt}")
        console.print(f"[green]   回答:[/green] {response}\n")

    avg_time = elapsed / len(prompts)
    console.print(f"[dim]📊 批量處理統計:[/dim]")
    console.print(f"[dim]  • 總耗時: {elapsed:.2f} 秒[/dim]")
    console.print(f"[dim]  • 平均耗時: {avg_time:.2f} 秒/提示[/dim]\n")


def compare_quantization():
    """量化方法比較"""
    console.print(Panel.fit(
        "[bold cyan]示例 5: 量化方法比較[/bold cyan]",
        border_style="cyan"
    ))

    # 理論比較表
    table = Table(title="量化方法對比", show_header=True, header_style="bold cyan")
    table.add_column("量化方式", style="cyan", width=12)
    table.add_column("記憶體節省", style="green", width=12)
    table.add_column("品質保留", style="yellow", width=12)
    table.add_column("推理速度", style="magenta", width=12)
    table.add_column("適用場景", style="blue", width=30)

    table.add_row(
        "FP16",
        "基準 (100%)",
        "100%",
        "快",
        "高端 GPU，追求最佳品質"
    )
    table.add_row(
        "8-bit",
        "50%",
        "~98%",
        "中等",
        "中端 GPU，平衡性能和品質"
    )
    table.add_row(
        "4-bit (NF4)",
        "75%",
        "~95%",
        "稍慢",
        "消費級 GPU，記憶體受限"
    )
    table.add_row(
        "GPTQ",
        "75%",
        "~96%",
        "快",
        "生產環境，高吞吐量"
    )
    table.add_row(
        "AWQ",
        "75%",
        "~97%",
        "最快",
        "推理服務，低延遲需求"
    )

    console.print(table)
    console.print()

    console.print("[dim]💡 提示:[/dim]")
    console.print("[dim]  • 對於 7B 模型，4-bit 量化只需約 4GB VRAM[/dim]")
    console.print("[dim]  • 對於 13B 模型，4-bit 量化需要約 8GB VRAM[/dim]")
    console.print("[dim]  • 對於 70B 模型，4-bit 量化需要約 35GB VRAM[/dim]\n")


def memory_optimization_tips():
    """記憶體優化建議"""
    console.print(Panel.fit(
        "[bold cyan]示例 6: 記憶體優化技巧[/bold cyan]",
        border_style="cyan"
    ))

    tips = [
        ("使用梯度檢查點", "減少訓練時的記憶體使用"),
        ("啟用 Flash Attention", "減少注意力機制的記憶體"),
        ("使用 8-bit Adam", "減少優化器狀態的記憶體"),
        ("減小批次大小", "最直接的記憶體節省方法"),
        ("使用混合精度訓練", "FP16/BF16 可節省約 50% 記憶體"),
        ("使用模型並行", "將大模型分散到多個 GPU"),
        ("使用 CPU Offload", "將部分參數卸載到 CPU"),
        ("清理 CUDA 快取", "定期調用 torch.cuda.empty_cache()")
    ]

    for i, (tip, desc) in enumerate(tips, 1):
        console.print(f"[cyan]{i}. {tip}[/cyan]")
        console.print(f"   [dim]{desc}[/dim]\n")

    # 實際示例
    console.print("[yellow]💻 代碼示例:[/yellow]")
    code = """
# 清理 CUDA 快取
import torch
torch.cuda.empty_cache()

# 使用 gradient checkpointing
model.gradient_checkpointing_enable()

# 使用混合精度
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**inputs)
"""
    console.print(f"[dim]{code}[/dim]")


def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold magenta]🤗 Hugging Face 本地模型示例[/bold magenta]\n"
        "[dim]展示如何在本地加載和優化開源 LLM[/dim]",
        border_style="magenta"
    ))

    # 檢查 GPU
    has_gpu = check_gpu()

    if not has_gpu:
        console.print("[yellow]⚠️  建議使用 GPU 以獲得最佳性能[/yellow]")
        console.print("[yellow]   繼續執行將使用 CPU（可能非常慢）[/yellow]\n")

        user_input = input("是否繼續？ (y/N): ").strip().lower()
        if user_input != 'y':
            console.print("[yellow]已取消[/yellow]")
            return

    # 選擇一個較小的模型進行演示
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # 3B 模型更適合演示

    console.print(f"\n[cyan]📦 使用模型: {model_name}[/cyan]")
    console.print("[dim]註：此為演示用小模型，實際應用可選擇更大的模型[/dim]\n")

    # 加載模型
    model, tokenizer = load_model_4bit(model_name)

    if model is None or tokenizer is None:
        console.print("[red]❌ 模型加載失敗，無法繼續[/red]")
        return

    try:
        # 運行示例
        basic_inference(model, tokenizer)
        streaming_inference(model, tokenizer)
        batch_inference(model, tokenizer)
        compare_quantization()
        memory_optimization_tips()

        console.print(Panel.fit(
            "[bold green]✅ 所有示例運行完成！[/bold green]\n"
            "[dim]提示: 探索更多模型 https://huggingface.co/models[/dim]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用戶中斷[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ 發生錯誤: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    finally:
        # 清理資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
