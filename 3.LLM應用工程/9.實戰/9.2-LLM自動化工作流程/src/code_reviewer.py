"""AI 代碼審查器 - 核心模塊"""
import logging
from typing import Optional, Dict, Any
import json

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.models import (
    ReviewResult, CodeIssue, IssueType, Severity,
    ReviewConfig
)
from src.code_analyzer import CodeAnalyzer

logger = logging.getLogger(__name__)


class AICodeReviewer:
    """AI 驅動的代碼審查器"""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.3,
        config: Optional[ReviewConfig] = None
    ):
        """初始化審查器

        Args:
            model: LLM 模型名稱
            temperature: 溫度參數
            config: 審查配置
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.config = config or ReviewConfig()
        self.analyzer = CodeAnalyzer()

    def review_code(
        self,
        code: str,
        language: str = "python",
        filename: str = "untitled",
        context: Optional[Dict[str, Any]] = None
    ) -> ReviewResult:
        """審查代碼

        Args:
            code: 要審查的代碼
            language: 編程語言
            filename: 文件名
            context: 額外的上下文信息

        Returns:
            ReviewResult: 審查結果
        """
        logger.info(f"Reviewing {filename} ({language})")

        # 1. 靜態分析
        analysis_result = self.analyzer.analyze(code, filename)
        static_issues = analysis_result["issues"]
        complexity = analysis_result["complexity"]

        # 2. LLM 審查
        llm_result = self._llm_review(code, language, context, static_issues)

        # 3. 合併結果
        all_issues = static_issues + llm_result.get("issues", [])

        # 4. 生成優化代碼（如果需要）
        optimized_code = None
        if self.config.auto_fix and llm_result.get("has_fixable_issues"):
            optimized_code = self._generate_fixed_code(code, all_issues)

        # 5. 計算分數
        score = self._calculate_score(all_issues, complexity)

        # 6. 構建結果
        result = ReviewResult(
            filename=filename,
            language=language,
            summary=llm_result.get("summary", "代碼審查完成"),
            score=score,
            issues=all_issues[:self.config.max_issues],
            complexity_metrics=complexity,
            suggestions=llm_result.get("suggestions", []),
            optimized_code=optimized_code,
            metadata={
                "context": context,
                "static_analysis_count": len(static_issues),
                "llm_analysis_count": len(llm_result.get("issues", []))
            }
        )

        logger.info(f"Review complete: {len(all_issues)} issues found, score: {score}/10")

        return result

    def _llm_review(
        self,
        code: str,
        language: str,
        context: Optional[Dict],
        static_issues: list
    ) -> Dict[str, Any]:
        """使用 LLM 審查代碼

        Args:
            code: 代碼
            language: 語言
            context: 上下文
            static_issues: 靜態分析發現的問題

        Returns:
            LLM 審查結果
        """
        # 構建系統提示
        system_prompt = self._build_system_prompt(language)

        # 構建用戶提示
        user_prompt = self._build_user_prompt(code, static_issues, context)

        try:
            # 調用 LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm.invoke(messages)
            result_text = response.content

            # 解析 LLM 響應
            parsed_result = self._parse_llm_response(result_text)

            return parsed_result

        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            return {
                "summary": "LLM 審查失敗",
                "issues": [],
                "suggestions": []
            }

    def _build_system_prompt(self, language: str) -> str:
        """構建系統提示

        Args:
            language: 編程語言

        Returns:
            系統提示
        """
        return f"""你是一個專業的 {language} 代碼審查專家。你的任務是：

1. 審查代碼質量、安全性、性能和最佳實踐
2. 識別潛在的 bug 和邏輯錯誤
3. 提供具體的改進建議和示例代碼
4. 評估代碼的可維護性和可讀性

請以結構化的 JSON 格式返回審查結果，包含：
- summary: 總體評價（字符串）
- issues: 問題列表（數組），每個問題包含：
  - type: 問題類型（security/performance/logic/style等）
  - severity: 嚴重程度（critical/high/medium/low/info）
  - message: 問題描述
  - line_start: 起始行號（如果適用）
  - suggestion: 改進建議
  - confidence: 置信度（0-1）
- suggestions: 總體改進建議列表（字符串數組）
- has_fixable_issues: 是否有可自動修復的問題（布爾值）

注意：
- 提供實用、可操作的建議
- 關注關鍵問題，不要過度關注風格細節
- 給出具體的代碼示例"""

    def _build_user_prompt(
        self,
        code: str,
        static_issues: list,
        context: Optional[Dict]
    ) -> str:
        """構建用戶提示

        Args:
            code: 代碼
            static_issues: 靜態分析問題
            context: 上下文

        Returns:
            用戶提示
        """
        prompt_parts = ["請審查以下代碼：\n\n```"]
        prompt_parts.append(code)
        prompt_parts.append("```\n")

        if static_issues:
            prompt_parts.append(f"\n靜態分析已發現 {len(static_issues)} 個問題。")
            prompt_parts.append("請重點關注邏輯錯誤、安全問題和性能優化。\n")

        if context:
            prompt_parts.append(f"\n額外上下文：{json.dumps(context, ensure_ascii=False)}\n")

        prompt_parts.append("\n請提供詳細的審查結果（JSON 格式）。")

        return "\n".join(prompt_parts)

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """解析 LLM 響應

        Args:
            response_text: LLM 響應文本

        Returns:
            解析後的結果
        """
        try:
            # 嘗試提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)

            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # 轉換 issues 為 CodeIssue 對象
                issues = []
                for issue_data in data.get("issues", []):
                    try:
                        issue = CodeIssue(
                            type=IssueType(issue_data.get("type", "best_practice")),
                            severity=Severity(issue_data.get("severity", "medium")),
                            message=issue_data.get("message", ""),
                            line_start=issue_data.get("line_start"),
                            suggestion=issue_data.get("suggestion"),
                            confidence=issue_data.get("confidence", 0.8)
                        )
                        issues.append(issue)
                    except Exception as e:
                        logger.warning(f"Failed to parse issue: {e}")

                return {
                    "summary": data.get("summary", ""),
                    "issues": issues,
                    "suggestions": data.get("suggestions", []),
                    "has_fixable_issues": data.get("has_fixable_issues", False)
                }

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")

        # 回退：提取文本信息
        return {
            "summary": response_text[:200],
            "issues": [],
            "suggestions": [],
            "has_fixable_issues": False
        }

    def _generate_fixed_code(self, code: str, issues: list) -> str:
        """生成修復後的代碼

        Args:
            code: 原始代碼
            issues: 問題列表

        Returns:
            修復後的代碼
        """
        try:
            # 構建修復提示
            fixable_issues = [
                issue for issue in issues
                if issue.suggestion and issue.severity in [Severity.CRITICAL, Severity.HIGH]
            ]

            if not fixable_issues:
                return code

            prompt = f"""請根據以下問題修復代碼：

原始代碼：
```
{code}
```

問題列表：
{chr(10).join(f"- {issue.message}: {issue.suggestion}" for issue in fixable_issues[:5])}

請提供修復後的完整代碼（只返回代碼，不要解釋）。"""

            response = self.llm.invoke(prompt)
            fixed_code = response.content

            # 提取代碼塊
            import re
            code_match = re.search(r'```(?:python)?\n([\s\S]*?)\n```', fixed_code)
            if code_match:
                return code_match.group(1)

            return fixed_code

        except Exception as e:
            logger.error(f"Failed to generate fixed code: {e}")
            return code

    def _calculate_score(self, issues: list, complexity) -> float:
        """計算代碼質量分數

        Args:
            issues: 問題列表
            complexity: 複雜度指標

        Returns:
            分數 (0-10)
        """
        base_score = 10.0

        # 根據問題嚴重程度扣分
        severity_penalties = {
            Severity.CRITICAL: 2.0,
            Severity.HIGH: 1.0,
            Severity.MEDIUM: 0.5,
            Severity.LOW: 0.2,
            Severity.INFO: 0.1
        }

        for issue in issues:
            base_score -= severity_penalties.get(issue.severity, 0.1)

        # 根據複雜度扣分
        if complexity:
            if complexity.cyclomatic_complexity > 15:
                base_score -= 1.0
            elif complexity.cyclomatic_complexity > 10:
                base_score -= 0.5

            if complexity.max_nesting_depth > 5:
                base_score -= 0.5

        # 確保分數在 0-10 範圍內
        return max(0.0, min(10.0, round(base_score, 1)))

    def format_review_markdown(self, result: ReviewResult) -> str:
        """格式化審查結果為 Markdown

        Args:
            result: 審查結果

        Returns:
            Markdown 格式的報告
        """
        lines = []

        lines.append(f"# 📊 代碼審查報告\n")
        lines.append(f"**文件**: {result.filename}")
        lines.append(f"**語言**: {result.language}")
        lines.append(f"**審查時間**: {result.review_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**質量分數**: {result.score}/10\n")

        lines.append(f"## 總結\n")
        lines.append(f"{result.summary}\n")

        if result.issues:
            lines.append(f"## 🔍 發現的問題 ({len(result.issues)})\n")

            # 按嚴重程度分組
            by_severity = {}
            for issue in result.issues:
                by_severity.setdefault(issue.severity, []).append(issue)

            for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
                if severity in by_severity:
                    lines.append(f"### {severity.value.upper()}")
                    for issue in by_severity[severity]:
                        lines.append(f"\n**{issue.type.value}**")
                        if issue.line_start:
                            lines.append(f"位置: 第 {issue.line_start} 行")
                        lines.append(f"問題: {issue.message}")
                        if issue.suggestion:
                            lines.append(f"建議: {issue.suggestion}")
                        lines.append("")

        if result.complexity_metrics:
            lines.append(f"## 📈 複雜度指標\n")
            cm = result.complexity_metrics
            lines.append(f"- 圈複雜度: {cm.cyclomatic_complexity}")
            lines.append(f"- 代碼行數: {cm.lines_of_code}")
            lines.append(f"- 函數數量: {cm.functions_count}")
            lines.append(f"- 類數量: {cm.classes_count}")
            lines.append(f"- 最大嵌套深度: {cm.max_nesting_depth}\n")

        if result.suggestions:
            lines.append(f"## 💡 改進建議\n")
            for i, suggestion in enumerate(result.suggestions, 1):
                lines.append(f"{i}. {suggestion}")
            lines.append("")

        if result.optimized_code:
            lines.append(f"## ✨ 優化後的代碼\n")
            lines.append(f"```{result.language}")
            lines.append(result.optimized_code)
            lines.append("```\n")

        return "\n".join(lines)
