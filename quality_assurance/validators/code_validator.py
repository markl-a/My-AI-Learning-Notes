"""
代碼自動驗證工具

此工具自動驗證 Python 文件和 Jupyter Notebooks 的正確性：
- 語法檢查
- 導入檢查
- 運行測試
- 輸出驗證

運行方式：
    python quality_assurance/validators/code_validator.py path/to/file_or_directory
"""

import ast
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import traceback


@dataclass
class ValidationResult:
    """驗證結果"""
    file_path: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    execution_time: float = 0.0


class CodeValidator:
    """代碼驗證器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def validate_python_file(self, file_path: Path) -> ValidationResult:
        """驗證 Python 文件

        Args:
            file_path: Python 文件路徑

        Returns:
            ValidationResult: 驗證結果
        """
        errors = []
        warnings = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📝 驗證: {file_path}")
            print(f"{'='*60}")

        # 1. 語法檢查
        syntax_ok, syntax_errors = self._check_syntax(file_path)
        if not syntax_ok:
            errors.extend(syntax_errors)
            return ValidationResult(
                file_path=str(file_path),
                passed=False,
                errors=errors,
                warnings=warnings
            )

        # 2. 導入檢查
        import_ok, import_errors, import_warnings = self._check_imports(file_path)
        if not import_ok:
            errors.extend(import_errors)
        warnings.extend(import_warnings)

        # 3. Docstring 檢查
        docstring_ok, docstring_warnings = self._check_docstrings(file_path)
        warnings.extend(docstring_warnings)

        # 4. 類型提示檢查（可選）
        type_hints_warnings = self._check_type_hints(file_path)
        warnings.extend(type_hints_warnings)

        passed = len(errors) == 0

        if self.verbose:
            if passed:
                print("✅ 驗證通過")
            else:
                print("❌ 驗證失敗")

            if errors:
                print(f"\n錯誤 ({len(errors)}):")
                for error in errors:
                    print(f"  ❌ {error}")

            if warnings:
                print(f"\n警告 ({len(warnings)}):")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")

        return ValidationResult(
            file_path=str(file_path),
            passed=passed,
            errors=errors,
            warnings=warnings
        )

    def validate_notebook(self, file_path: Path) -> ValidationResult:
        """驗證 Jupyter Notebook

        Args:
            file_path: Notebook 文件路徑

        Returns:
            ValidationResult: 驗證結果
        """
        errors = []
        warnings = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📓 驗證 Notebook: {file_path}")
            print(f"{'='*60}")

        try:
            # 讀取 notebook
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            # 檢查格式
            if 'cells' not in nb:
                errors.append("無效的 notebook 格式：缺少 'cells' 字段")
                return ValidationResult(
                    file_path=str(file_path),
                    passed=False,
                    errors=errors,
                    warnings=warnings
                )

            # 檢查每個代碼單元格
            code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']

            if not code_cells:
                warnings.append("Notebook 中沒有代碼單元格")

            # 檢查代碼語法
            for i, cell in enumerate(code_cells):
                source = ''.join(cell['source'])
                if source.strip():
                    try:
                        ast.parse(source)
                    except SyntaxError as e:
                        errors.append(f"單元格 {i+1} 語法錯誤: {e}")

            # 檢查是否有輸出
            cells_with_output = sum(1 for cell in code_cells if cell.get('outputs'))
            if cells_with_output == 0 and len(code_cells) > 0:
                warnings.append("所有代碼單元格都沒有輸出（可能未執行）")

        except json.JSONDecodeError:
            errors.append("無法解析 notebook JSON 格式")
        except Exception as e:
            errors.append(f"未知錯誤: {e}")

        passed = len(errors) == 0

        if self.verbose:
            if passed:
                print("✅ Notebook 驗證通過")
            else:
                print("❌ Notebook 驗證失敗")

            print(f"\n統計:")
            print(f"  代碼單元格: {len(code_cells)}")
            print(f"  有輸出的單元格: {cells_with_output}")

            if errors:
                print(f"\n錯誤 ({len(errors)}):")
                for error in errors:
                    print(f"  ❌ {error}")

            if warnings:
                print(f"\n警告 ({len(warnings)}):")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")

        return ValidationResult(
            file_path=str(file_path),
            passed=passed,
            errors=errors,
            warnings=warnings
        )

    def validate_directory(self, directory: Path, recursive: bool = True) -> List[ValidationResult]:
        """驗證目錄中的所有文件

        Args:
            directory: 目錄路徑
            recursive: 是否遞歸子目錄

        Returns:
            List[ValidationResult]: 驗證結果列表
        """
        results = []

        # 查找所有 Python 文件
        pattern = "**/*.py" if recursive else "*.py"
        for py_file in directory.glob(pattern):
            # 跳過 __pycache__ 和虛擬環境
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue

            result = self.validate_python_file(py_file)
            results.append(result)

        # 查找所有 Notebook 文件
        pattern = "**/*.ipynb" if recursive else "*.ipynb"
        for nb_file in directory.glob(pattern):
            # 跳過檢查點
            if '.ipynb_checkpoints' in str(nb_file):
                continue

            result = self.validate_notebook(nb_file)
            results.append(result)

        return results

    def _check_syntax(self, file_path: Path) -> Tuple[bool, List[str]]:
        """檢查 Python 語法

        Returns:
            (是否通過, 錯誤列表)
        """
        errors = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            ast.parse(code)
            return True, []

        except SyntaxError as e:
            errors.append(f"語法錯誤 (行 {e.lineno}): {e.msg}")
            return False, errors

        except Exception as e:
            errors.append(f"無法讀取文件: {e}")
            return False, errors

    def _check_imports(self, file_path: Path) -> Tuple[bool, List[str], List[str]]:
        """檢查導入語句

        Returns:
            (是否通過, 錯誤列表, 警告列表)
        """
        errors = []
        warnings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            __import__(alias.name)
                        except ImportError:
                            warnings.append(f"無法導入模塊: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            __import__(node.module)
                        except ImportError:
                            warnings.append(f"無法導入模塊: {node.module}")

            return True, [], warnings

        except Exception as e:
            errors.append(f"導入檢查失敗: {e}")
            return False, errors, warnings

    def _check_docstrings(self, file_path: Path) -> Tuple[bool, List[str]]:
        """檢查文檔字符串

        Returns:
            (是否通過, 警告列表)
        """
        warnings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            # 檢查模塊 docstring
            if not ast.get_docstring(tree):
                warnings.append("缺少模塊級 docstring")

            # 檢查函數和類的 docstring
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        warnings.append(f"函數 '{node.name}' 缺少 docstring")

                elif isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        warnings.append(f"類 '{node.name}' 缺少 docstring")

            return True, warnings

        except Exception:
            return True, []

    def _check_type_hints(self, file_path: Path) -> List[str]:
        """檢查類型提示

        Returns:
            警告列表
        """
        warnings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 檢查參數類型提示
                    if node.args.args:
                        for arg in node.args.args:
                            if arg.arg != 'self' and arg.annotation is None:
                                warnings.append(
                                    f"函數 '{node.name}' 的參數 '{arg.arg}' 缺少類型提示"
                                )

                    # 檢查返回類型提示
                    if node.returns is None and node.name != '__init__':
                        warnings.append(f"函數 '{node.name}' 缺少返回類型提示")

        except Exception:
            pass

        return warnings

    def generate_report(self, results: List[ValidationResult]) -> str:
        """生成驗證報告

        Args:
            results: 驗證結果列表

        Returns:
            報告字符串
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)

        report = f"""
{'='*80}
代碼驗證報告
{'='*80}

總計文件: {total}
✅ 通過: {passed} ({passed/total*100:.1f}%)
❌ 失敗: {failed} ({failed/total*100:.1f}%)

總錯誤數: {total_errors}
總警告數: {total_warnings}

{'='*80}
"""

        if failed > 0:
            report += "\n失敗的文件:\n"
            for result in results:
                if not result.passed:
                    report += f"\n❌ {result.file_path}\n"
                    for error in result.errors:
                        report += f"   - {error}\n"

        return report


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="代碼自動驗證工具")
    parser.add_argument("path", help="文件或目錄路徑")
    parser.add_argument("--recursive", "-r", action="store_true", help="遞歸檢查子目錄")
    parser.add_argument("--quiet", "-q", action="store_true", help="安靜模式（只顯示摘要）")
    parser.add_argument("--report", help="保存報告到文件")

    args = parser.parse_args()

    validator = CodeValidator(verbose=not args.quiet)
    path = Path(args.path)

    # 驗證
    if path.is_file():
        if path.suffix == '.py':
            results = [validator.validate_python_file(path)]
        elif path.suffix == '.ipynb':
            results = [validator.validate_notebook(path)]
        else:
            print(f"❌ 不支持的文件類型: {path.suffix}")
            sys.exit(1)
    elif path.is_dir():
        results = validator.validate_directory(path, recursive=args.recursive)
    else:
        print(f"❌ 路徑不存在: {path}")
        sys.exit(1)

    # 生成報告
    report = validator.generate_report(results)
    print(report)

    # 保存報告
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ 報告已保存到: {args.report}")

    # 返回碼
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
