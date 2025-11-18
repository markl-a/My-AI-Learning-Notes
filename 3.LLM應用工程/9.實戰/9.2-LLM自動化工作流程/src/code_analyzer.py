"""代碼分析器 - 使用靜態分析工具"""
import ast
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models import ComplexityMetrics, CodeIssue, IssueType, Severity

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """代碼分析器 - 結合靜態分析和 AST"""

    def __init__(self, language: str = "python"):
        """初始化分析器

        Args:
            language: 編程語言
        """
        self.language = language
        self.supported_languages = ["python", "javascript", "typescript"]

    def analyze(self, code: str, filename: str = "untitled") -> Dict[str, Any]:
        """分析代碼

        Args:
            code: 代碼內容
            filename: 文件名

        Returns:
            分析結果字典
        """
        result = {
            "issues": [],
            "complexity": None,
            "ast_tree": None,
            "metadata": {}
        }

        if self.language == "python":
            result = self._analyze_python(code, filename)
        else:
            logger.warning(f"Language {self.language} not fully supported yet")

        return result

    def _analyze_python(self, code: str, filename: str) -> Dict[str, Any]:
        """分析 Python 代碼

        Args:
            code: Python 代碼
            filename: 文件名

        Returns:
            分析結果
        """
        issues = []
        complexity = ComplexityMetrics()
        ast_tree = None

        try:
            # 解析 AST
            ast_tree = ast.parse(code, filename=filename)

            # 基本指標
            complexity = self._calculate_complexity(code, ast_tree)

            # 靜態分析檢查
            issues.extend(self._check_syntax(code))
            issues.extend(self._check_style(code, ast_tree))
            issues.extend(self._check_complexity(complexity))
            issues.extend(self._check_best_practices(code, ast_tree))
            issues.extend(self._check_naming(ast_tree))

        except SyntaxError as e:
            issues.append(CodeIssue(
                type=IssueType.SYNTAX,
                severity=Severity.CRITICAL,
                message=f"語法錯誤: {str(e)}",
                line_start=e.lineno,
                confidence=1.0
            ))

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return {
            "issues": issues,
            "complexity": complexity,
            "ast_tree": ast_tree,
            "metadata": {
                "lines_of_code": len(code.splitlines()),
                "language": self.language
            }
        }

    def _calculate_complexity(self, code: str, ast_tree: ast.AST) -> ComplexityMetrics:
        """計算複雜度指標

        Args:
            code: 代碼
            ast_tree: AST 樹

        Returns:
            複雜度指標
        """
        metrics = ComplexityMetrics()

        # 代碼行數
        lines = code.splitlines()
        metrics.lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])

        # 統計函數和類
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                metrics.functions_count += 1
                # 計算圈複雜度
                cc = self._calculate_cyclomatic_complexity(node)
                metrics.cyclomatic_complexity = max(metrics.cyclomatic_complexity, cc)

            elif isinstance(node, ast.ClassDef):
                metrics.classes_count += 1

        # 計算最大嵌套深度
        metrics.max_nesting_depth = self._calculate_max_nesting_depth(ast_tree)

        return metrics

    def _calculate_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """計算圈複雜度

        Args:
            func_node: 函數節點

        Returns:
            圈複雜度
        """
        complexity = 1  # 基礎複雜度

        for node in ast.walk(func_node):
            # 決策點
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # 邏輯運算符
                complexity += len(node.values) - 1

        return complexity

    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """計算最大嵌套深度

        Args:
            tree: AST 樹

        Returns:
            最大嵌套深度
        """
        def get_depth(node, current_depth=0):
            max_depth = current_depth

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, depth)
                else:
                    depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, depth)

            return max_depth

        return get_depth(tree)

    def _check_syntax(self, code: str) -> List[CodeIssue]:
        """檢查語法問題

        Args:
            code: 代碼

        Returns:
            問題列表
        """
        issues = []

        # 基本語法檢查已在 AST 解析時完成
        # 這裡可以添加額外的語法相關檢查

        return issues

    def _check_style(self, code: str, ast_tree: ast.AST) -> List[CodeIssue]:
        """檢查代碼風格

        Args:
            code: 代碼
            ast_tree: AST 樹

        Returns:
            問題列表
        """
        issues = []

        lines = code.splitlines()

        # 檢查行長度
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(CodeIssue(
                    type=IssueType.STYLE,
                    severity=Severity.LOW,
                    message=f"行太長 ({len(line)} 字符)，建議不超過 120 字符",
                    line_start=i,
                    line_end=i,
                    confidence=1.0
                ))

        # 檢查空行
        blank_count = 0
        for i, line in enumerate(lines, 1):
            if not line.strip():
                blank_count += 1
                if blank_count > 2:
                    issues.append(CodeIssue(
                        type=IssueType.STYLE,
                        severity=Severity.INFO,
                        message="連續空行過多",
                        line_start=i,
                        confidence=0.8
                    ))
            else:
                blank_count = 0

        return issues

    def _check_complexity(self, complexity: ComplexityMetrics) -> List[CodeIssue]:
        """檢查複雜度問題

        Args:
            complexity: 複雜度指標

        Returns:
            問題列表
        """
        issues = []

        # 圈複雜度檢查
        if complexity.cyclomatic_complexity > 10:
            issues.append(CodeIssue(
                type=IssueType.COMPLEXITY,
                severity=Severity.HIGH if complexity.cyclomatic_complexity > 15 else Severity.MEDIUM,
                message=f"圈複雜度過高 ({complexity.cyclomatic_complexity})，建議重構",
                suggestion="將複雜函數拆分為多個小函數",
                confidence=0.9
            ))

        # 嵌套深度檢查
        if complexity.max_nesting_depth > 4:
            issues.append(CodeIssue(
                type=IssueType.COMPLEXITY,
                severity=Severity.MEDIUM,
                message=f"嵌套深度過深 ({complexity.max_nesting_depth})，影響可讀性",
                suggestion="考慮提前返回或提取嵌套邏輯",
                confidence=0.85
            ))

        return issues

    def _check_best_practices(self, code: str, ast_tree: ast.AST) -> List[CodeIssue]:
        """檢查最佳實踐

        Args:
            code: 代碼
            ast_tree: AST 樹

        Returns:
            問題列表
        """
        issues = []

        for node in ast.walk(ast_tree):
            # 檢查裸露的 except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(CodeIssue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.MEDIUM,
                        message="使用裸露的 except，應該捕獲特定異常",
                        line_start=node.lineno,
                        suggestion="使用 'except SpecificException:' 代替 'except:'",
                        confidence=0.9
                    ))

            # 檢查可變默認參數
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(CodeIssue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.HIGH,
                            message="使用可變對象作為默認參數可能導致意外行為",
                            line_start=node.lineno,
                            suggestion="使用 None 作為默認值，在函數內部初始化",
                            confidence=0.95
                        ))

        return issues

    def _check_naming(self, ast_tree: ast.AST) -> List[CodeIssue]:
        """檢查命名規範

        Args:
            ast_tree: AST 樹

        Returns:
            問題列表
        """
        issues = []

        for node in ast.walk(ast_tree):
            # 檢查函數命名
            if isinstance(node, ast.FunctionDef):
                name = node.name
                if not name.startswith('_') and not name.islower():
                    if not all(c.islower() or c == '_' for c in name):
                        issues.append(CodeIssue(
                            type=IssueType.NAMING,
                            severity=Severity.LOW,
                            message=f"函數名 '{name}' 應使用 snake_case 命名",
                            line_start=node.lineno,
                            confidence=0.85
                        ))

            # 檢查類命名
            elif isinstance(node, ast.ClassDef):
                name = node.name
                if not name[0].isupper():
                    issues.append(CodeIssue(
                        type=IssueType.NAMING,
                        severity=Severity.LOW,
                        message=f"類名 '{name}' 應使用 PascalCase 命名",
                        line_start=node.lineno,
                        confidence=0.85
                    ))

            # 檢查常量命名
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # 簡單的常量檢測（全大寫）
                        if name.isupper() and '_' not in name and len(name) > 1:
                            issues.append(CodeIssue(
                                type=IssueType.NAMING,
                                severity=Severity.INFO,
                                message=f"常量 '{name}' 建議使用下劃線分隔單詞",
                                line_start=node.lineno,
                                confidence=0.7
                            ))

        return issues

    def get_function_signatures(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """提取函數簽名

        Args:
            ast_tree: AST 樹

        Returns:
            函數簽名列表
        """
        functions = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None,
                    "docstring": ast.get_docstring(node),
                    "is_async": isinstance(node, ast.AsyncFunctionDef)
                }
                functions.append(func_info)

        return functions
