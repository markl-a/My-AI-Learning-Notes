"""命令行接口"""
import click
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from src.code_reviewer import AICodeReviewer
from src.models import ReviewConfig, Severity

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AI 代碼審查工具 - 使用 LLM 自動審查代碼質量"""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--model', default='gpt-4', help='LLM 模型 (default: gpt-4)')
@click.option('--output', '-o', type=click.Path(), help='輸出報告文件路徑')
@click.option('--format', type=click.Choice(['markdown', 'json', 'console']), default='console', help='輸出格式')
@click.option('--auto-fix', is_flag=True, help='自動生成修復後的代碼')
def review(file_path, model, output, format, auto_fix):
    """審查單個文件"""
    console.print(f"[bold cyan]正在審查文件:[/bold cyan] {file_path}")

    # 讀取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 檢測語言
    language = _detect_language(file_path)

    # 創建審查器
    config = ReviewConfig(auto_fix=auto_fix)
    reviewer = AICodeReviewer(model=model, config=config)

    # 執行審查
    with console.status("[bold green]審查中..."):
        result = reviewer.review_code(code, language, file_path)

    # 輸出結果
    if format == 'console':
        _display_result_console(result)
    elif format == 'markdown':
        markdown = reviewer.format_review_markdown(result)
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(markdown)
            console.print(f"[green]✓[/green] 報告已保存到: {output}")
        else:
            console.print(markdown)
    elif format == 'json':
        import json
        json_data = result.model_dump_json(indent=2)
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(json_data)
            console.print(f"[green]✓[/green] 報告已保存到: {output}")
        else:
            console.print(json_data)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='遞歸掃描子目錄')
@click.option('--pattern', default='*.py', help='文件匹配模式 (default: *.py)')
@click.option('--model', default='gpt-4', help='LLM 模型')
@click.option('--output', '-o', type=click.Path(), help='輸出報告目錄')
def review_dir(directory, recursive, pattern, model, output):
    """審查整個目錄"""
    console.print(f"[bold cyan]正在掃描目錄:[/bold cyan] {directory}")

    # 查找文件
    path = Path(directory)
    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))

    if not files:
        console.print("[yellow]未找到匹配的文件[/yellow]")
        return

    console.print(f"[green]找到 {len(files)} 個文件[/green]\n")

    # 創建審查器
    reviewer = AICodeReviewer(model=model)

    # 審查所有文件
    results = []
    for i, file_path in enumerate(files, 1):
        console.print(f"[{i}/{len(files)}] 審查: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            language = _detect_language(str(file_path))
            result = reviewer.review_code(code, language, str(file_path))
            results.append(result)

        except Exception as e:
            console.print(f"[red]✗ 錯誤:[/red] {e}")

    # 生成統計報告
    _display_batch_summary(results)

    # 保存報告
    if output:
        os.makedirs(output, exist_ok=True)
        for result in results:
            filename = Path(result.filename).stem + "_review.md"
            report_path = os.path.join(output, filename)
            markdown = reviewer.format_review_markdown(result)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
        console.print(f"\n[green]✓ 報告已保存到:[/green] {output}")


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='輸出測試文件路徑')
@click.option('--framework', default='pytest', help='測試框架 (default: pytest)')
@click.option('--model', default='gpt-4', help='LLM 模型')
def generate_tests(file_path, output, framework, model):
    """生成測試代碼"""
    console.print(f"[bold cyan]正在為文件生成測試:[/bold cyan] {file_path}")

    # 讀取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 創建審查器（復用 LLM 客戶端）
    reviewer = AICodeReviewer(model=model)

    # 生成測試
    with console.status("[bold green]生成測試中..."):
        test_prompt = f"""為以下代碼生成完整的 {framework} 測試：

```python
{code}
```

要求：
1. 測試所有主要函數
2. 包含正常情況和邊界情況
3. 使用 {framework} 語法
4. 添加清晰的測試文檔

只返回測試代碼，不要解釋。"""

        response = reviewer.llm.invoke(test_prompt)
        test_code = response.content

    # 提取代碼
    import re
    code_match = re.search(r'```(?:python)?\n([\s\S]*?)\n```', test_code)
    if code_match:
        test_code = code_match.group(1)

    # 輸出
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(test_code)
        console.print(f"[green]✓ 測試已保存到:[/green] {output}")
    else:
        syntax = Syntax(test_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--model', default='gpt-4', help='LLM 模型')
def generate_docs(file_path, model):
    """生成代碼文檔"""
    console.print(f"[bold cyan]正在生成文檔:[/bold cyan] {file_path}")

    # 讀取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 創建審查器
    reviewer = AICodeReviewer(model=model)

    # 生成文檔
    with console.status("[bold green]生成文檔中..."):
        doc_prompt = f"""為以下代碼生成詳細的文檔註釋：

```python
{code}
```

要求：
1. 為每個函數添加 docstring
2. 使用 Google 風格的文檔字符串
3. 包含參數說明、返回值和示例
4. 添加必要的行內註釋

返回添加了文檔後的完整代碼。"""

        response = reviewer.llm.invoke(doc_prompt)
        documented_code = response.content

    # 提取代碼
    import re
    code_match = re.search(r'```(?:python)?\n([\s\S]*?)\n```', documented_code)
    if code_match:
        documented_code = code_match.group(1)

    # 顯示結果
    syntax = Syntax(documented_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    # 詢問是否保存
    if click.confirm('\n是否要覆蓋原文件？'):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(documented_code)
        console.print(f"[green]✓ 已更新文件:[/green] {file_path}")


def _detect_language(file_path: str) -> str:
    """檢測編程語言"""
    ext = Path(file_path).suffix.lower()
    mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c'
    }
    return mapping.get(ext, 'python')


def _display_result_console(result):
    """在控制台顯示結果"""
    # 標題
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        f"[bold]代碼審查報告[/bold]\n"
        f"文件: {result.filename}\n"
        f"質量分數: [bold {'green' if result.score >= 7 else 'yellow' if result.score >= 5 else 'red'}]{result.score}/10[/bold]",
        border_style="cyan"
    ))

    # 總結
    console.print(f"\n[bold]總結:[/bold] {result.summary}\n")

    # 問題列表
    if result.issues:
        console.print(f"[bold red]發現 {len(result.issues)} 個問題:[/bold red]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("嚴重程度", style="dim", width=12)
        table.add_column("類型", width=15)
        table.add_column("位置", justify="right", width=10)
        table.add_column("問題描述", width=50)

        for issue in result.issues[:20]:  # 只顯示前20個
            severity_color = {
                Severity.CRITICAL: "red",
                Severity.HIGH: "orange1",
                Severity.MEDIUM: "yellow",
                Severity.LOW: "blue",
                Severity.INFO: "green"
            }.get(issue.severity, "white")

            location = f"第{issue.line_start}行" if issue.line_start else "-"

            table.add_row(
                f"[{severity_color}]{issue.severity.value}[/{severity_color}]",
                issue.type.value,
                location,
                issue.message[:47] + "..." if len(issue.message) > 50 else issue.message
            )

        console.print(table)

        if len(result.issues) > 20:
            console.print(f"\n[dim]...還有 {len(result.issues) - 20} 個問題未顯示[/dim]")

    # 複雜度
    if result.complexity_metrics:
        console.print(f"\n[bold]複雜度指標:[/bold]")
        cm = result.complexity_metrics
        console.print(f"  圈複雜度: {cm.cyclomatic_complexity}")
        console.print(f"  代碼行數: {cm.lines_of_code}")
        console.print(f"  函數數量: {cm.functions_count}")

    # 建議
    if result.suggestions:
        console.print(f"\n[bold]改進建議:[/bold]")
        for i, suggestion in enumerate(result.suggestions[:5], 1):
            console.print(f"  {i}. {suggestion}")

    console.print("\n" + "="*70 + "\n")


def _display_batch_summary(results):
    """顯示批量審查統計"""
    console.print("\n" + "="*70)
    console.print("[bold cyan]審查統計[/bold cyan]\n")

    total_issues = sum(len(r.issues) for r in results)
    avg_score = sum(r.score for r in results) / len(results) if results else 0

    # 統計表
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("指標", style="cyan")
    table.add_column("數值", justify="right")

    table.add_row("審查文件數", str(len(results)))
    table.add_row("總問題數", str(total_issues))
    table.add_row("平均分數", f"{avg_score:.1f}/10")

    # 按嚴重程度統計
    severity_counts = {}
    for result in results:
        for issue in result.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]:
        if severity in severity_counts:
            table.add_row(f"{severity.value} 問題", str(severity_counts[severity]))

    console.print(table)
    console.print("="*70 + "\n")


if __name__ == '__main__':
    cli()
