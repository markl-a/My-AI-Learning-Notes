#!/usr/bin/env python3
"""
LLM 部署健康检查工具

检查部署环境的各个方面，确保一切正常运行。
"""

import os
import sys
from pathlib import Path

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("请安装 rich: pip install rich")
    sys.exit(1)

console = Console()


def check_python_version():
    """检查 Python 版本"""
    import sys
    version = sys.version_info
    required = (3, 9)

    is_ok = version >= required
    status = "✅" if is_ok else "❌"

    return {
        "name": "Python 版本",
        "status": status,
        "detail": f"{version.major}.{version.minor}.{version.micro}",
        "required": f">= {required[0]}.{required[1]}",
        "ok": is_ok
    }


def check_gpu():
    """检查 GPU 可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            detail = f"{gpu_name} ({gpu_memory:.1f} GB)"
            status = "✅"
        else:
            detail = "未检测到 CUDA GPU"
            status = "⚠️"

        return {
            "name": "GPU (CUDA)",
            "status": status,
            "detail": detail,
            "required": "推荐但非必需",
            "ok": True  # GPU 不是必需的
        }
    except ImportError:
        return {
            "name": "GPU (CUDA)",
            "status": "⚠️",
            "detail": "PyTorch 未安装",
            "required": "推荐但非必需",
            "ok": True
        }
    except Exception as e:
        return {
            "name": "GPU (CUDA)",
            "status": "❌",
            "detail": f"错误: {e}",
            "required": "推荐但非必需",
            "ok": False
        }


def check_api_keys():
    """检查 API Keys"""
    from dotenv import load_dotenv
    load_dotenv()

    keys_to_check = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "HUGGINGFACE_TOKEN": "Hugging Face (可选)"
    }

    results = []

    for env_var, service in keys_to_check.items():
        api_key = os.getenv(env_var)
        has_key = bool(api_key and len(api_key) > 10)

        status = "✅" if has_key else "⚠️"
        detail = "已配置" if has_key else "未配置"

        results.append({
            "name": f"{service} API Key",
            "status": status,
            "detail": detail,
            "required": "至少配置一个",
            "ok": True  # 只要有一个就行
        })

    return results


def check_ollama():
    """检查 Ollama 服务"""
    try:
        import ollama
        models = ollama.list()
        model_count = len(models.get('models', []))

        status = "✅" if model_count > 0 else "⚠️"
        detail = f"运行中，{model_count} 个模型"

        return {
            "name": "Ollama 服务",
            "status": status,
            "detail": detail,
            "required": "推荐",
            "ok": True
        }
    except Exception as e:
        return {
            "name": "Ollama 服务",
            "status": "⚠️",
            "detail": "未运行或未安装",
            "required": "推荐",
            "ok": True
        }


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        ("transformers", "Hugging Face Transformers"),
        ("openai", "OpenAI SDK"),
        ("ollama", "Ollama SDK"),
        ("fastapi", "FastAPI"),
        ("rich", "Rich 终端库")
    ]

    results = []

    for package, name in required_packages:
        try:
            __import__(package)
            status = "✅"
            detail = "已安装"
            ok = True
        except ImportError:
            status = "❌"
            detail = "未安装"
            ok = False

        results.append({
            "name": name,
            "status": status,
            "detail": detail,
            "required": "必需",
            "ok": ok
        })

    return results


def check_disk_space():
    """检查磁盘空间"""
    import shutil

    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    status = "✅" if free_gb > 50 else ("⚠️" if free_gb > 20 else "❌")
    detail = f"{free_gb:.1f} GB 可用 / {total_gb:.1f} GB 总计"

    return {
        "name": "磁盘空间",
        "status": status,
        "detail": detail,
        "required": "> 20 GB",
        "ok": free_gb > 20
    }


def check_memory():
    """检查系统内存"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)

        status = "✅" if total_gb >= 16 else ("⚠️" if total_gb >= 8 else "❌")
        detail = f"{available_gb:.1f} GB 可用 / {total_gb:.1f} GB 总计"

        return {
            "name": "系统内存",
            "status": status,
            "detail": detail,
            "required": ">= 16 GB 推荐",
            "ok": total_gb >= 8
        }
    except ImportError:
        return {
            "name": "系统内存",
            "status": "⚠️",
            "detail": "psutil 未安装",
            "required": ">= 16 GB 推荐",
            "ok": True
        }


def test_api_connectivity():
    """测试 API 连接"""
    from dotenv import load_dotenv
    load_dotenv()

    results = []

    # 测试 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # 简单测试：列出模型
            models = client.models.list()
            results.append({
                "name": "OpenAI API",
                "status": "✅",
                "detail": "连接正常",
                "required": "N/A",
                "ok": True
            })
        except Exception as e:
            results.append({
                "name": "OpenAI API",
                "status": "❌",
                "detail": f"连接失败: {str(e)[:50]}",
                "required": "N/A",
                "ok": False
            })

    # 测试 Ollama
    try:
        import ollama
        ollama.list()
        results.append({
            "name": "Ollama API",
            "status": "✅",
            "detail": "连接正常",
            "required": "N/A",
            "ok": True
        })
    except Exception as e:
        results.append({
            "name": "Ollama API",
            "status": "❌",
            "detail": "连接失败",
            "required": "N/A",
            "ok": False
        })

    return results


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold cyan]🏥 LLM 部署环境健康检查[/bold cyan]\n"
        "[dim]检查系统配置和依赖状态[/dim]",
        border_style="cyan"
    ))

    all_checks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # 系统检查
        task = progress.add_task("[cyan]检查系统环境...", total=None)
        all_checks.append(check_python_version())
        all_checks.append(check_gpu())
        all_checks.append(check_disk_space())
        all_checks.append(check_memory())
        progress.update(task, completed=True)

        # 依赖检查
        task = progress.add_task("[cyan]检查依赖包...", total=None)
        all_checks.extend(check_dependencies())
        progress.update(task, completed=True)

        # API Keys
        task = progress.add_task("[cyan]检查 API Keys...", total=None)
        all_checks.extend(check_api_keys())
        progress.update(task, completed=True)

        # 服务检查
        task = progress.add_task("[cyan]检查服务...", total=None)
        all_checks.append(check_ollama())
        progress.update(task, completed=True)

        # API 连接测试
        task = progress.add_task("[cyan]测试 API 连接...", total=None)
        all_checks.extend(test_api_connectivity())
        progress.update(task, completed=True)

    # 显示结果表格
    table = Table(title="\n健康检查结果", show_header=True, header_style="bold cyan")
    table.add_column("检查项", style="cyan", width=25)
    table.add_column("状态", width=8)
    table.add_column("详情", style="dim", width=35)
    table.add_column("要求", style="dim", width=20)

    for check in all_checks:
        table.add_row(
            check["name"],
            check["status"],
            check["detail"],
            check["required"]
        )

    console.print(table)

    # 总结
    total = len(all_checks)
    passed = sum(1 for c in all_checks if c["ok"])
    failed = total - passed

    if failed == 0:
        console.print(Panel(
            f"[bold green]✅ 所有检查通过！({passed}/{total})[/bold green]\n"
            "[dim]你的环境已准备就绪[/dim]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠️  发现 {failed} 个问题[/bold yellow]\n"
            f"[dim]通过: {passed}/{total}[/dim]\n\n"
            "[yellow]建议:[/yellow]\n"
            "1. 安装缺失的依赖: pip install -r requirements.txt\n"
            "2. 配置环境变量: cp .env.example .env\n"
            "3. 安装 Ollama: curl -fsSL https://ollama.com/install.sh | sh",
            border_style="yellow"
        ))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
