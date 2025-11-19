"""
性能分析器
分析代碼性能並提供優化建議
"""

import os
import re
import logging
from typing import Dict, List, Any
from openai import AsyncOpenAI
import ast

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """性能分析器類"""

    def __init__(self, model_name: str = "gpt-4"):
        """初始化性能分析器"""
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def analyze(
        self,
        code: str,
        language: str,
        depth: str = "medium"
    ) -> Dict[str, Any]:
        """
        分析代碼性能

        Args:
            code: 代碼內容
            language: 編程語言
            depth: 分析深度（quick, medium, deep）

        Returns:
            性能分析結果
        """
        # 1. 靜態分析
        static_analysis = self._static_analysis(code, language)

        # 2. LLM 深度分析
        llm_analysis = await self._llm_analysis(code, language, depth)

        # 3. 合併結果
        result = self._merge_analysis(static_analysis, llm_analysis)

        return result

    def _static_analysis(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """靜態代碼分析"""
        analysis = {
            "loop_count": 0,
            "recursion_depth": 0,
            "db_queries": 0,
            "nested_loops": 0,
            "large_iterations": []
        }

        if language == "python":
            analysis = self._analyze_python_code(code)
        elif language == "javascript":
            analysis = self._analyze_javascript_code(code)

        return analysis

    def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """分析 Python 代碼"""
        analysis = {
            "loop_count": 0,
            "recursion_depth": 0,
            "db_queries": 0,
            "nested_loops": 0,
            "large_iterations": []
        }

        try:
            tree = ast.parse(code)

            # 計算循環
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    analysis["loop_count"] += 1

            # 檢測遞歸
            # 簡單檢測：函數內部調用自己
            func_calls = re.findall(r'def\s+(\w+).*?:\s*(?:.*?\1\s*\()', code, re.DOTALL)
            analysis["recursion_depth"] = len(func_calls)

            # 檢測數據庫查詢
            db_patterns = [
                r'\.execute\s*\(',
                r'\.query\s*\(',
                r'\.find\s*\(',
                r'\.filter\s*\('
            ]
            for pattern in db_patterns:
                matches = re.findall(pattern, code)
                analysis["db_queries"] += len(matches)

        except SyntaxError:
            logger.warning("Failed to parse Python code")

        return analysis

    def _analyze_javascript_code(self, code: str) -> Dict[str, Any]:
        """分析 JavaScript 代碼"""
        analysis = {
            "loop_count": 0,
            "recursion_depth": 0,
            "db_queries": 0,
            "nested_loops": 0,
            "large_iterations": []
        }

        # 計算循環
        for_loops = re.findall(r'\bfor\s*\(', code)
        while_loops = re.findall(r'\bwhile\s*\(', code)
        foreach_loops = re.findall(r'\.forEach\s*\(', code)

        analysis["loop_count"] = len(for_loops) + len(while_loops) + len(foreach_loops)

        # 檢測數據庫查詢
        db_patterns = [
            r'\.find\s*\(',
            r'\.query\s*\(',
            r'\.exec\s*\(',
            r'\.aggregate\s*\('
        ]
        for pattern in db_patterns:
            matches = re.findall(pattern, code)
            analysis["db_queries"] += len(matches)

        return analysis

    async def _llm_analysis(
        self,
        code: str,
        language: str,
        depth: str
    ) -> Dict[str, Any]:
        """使用 LLM 進行深度性能分析"""
        prompt = self._build_analysis_prompt(code, language, depth)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位 {language} 性能優化專家。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content

            # 解析結果
            return self._parse_llm_result(result_text)

        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return {}

    def _build_analysis_prompt(
        self,
        code: str,
        language: str,
        depth: str
    ) -> str:
        """構建分析提示"""
        if depth == "quick":
            return f"""
快速分析以下 {language} 代碼的性能。

代碼:
```{language}
{code}
```

提供：
1. 時間複雜度估計（O notation）
2. 主要性能瓶頸（top 3）
3. 快速優化建議

以 JSON 格式返回。
"""
        elif depth == "deep":
            return f"""
深度分析以下 {language} 代碼的性能。

代碼:
```{language}
{code}
```

詳細分析：
1. 時間複雜度（每個函數）
2. 空間複雜度
3. 性能瓶頸識別
4. I/O 操作分析
5. 緩存機會
6. 並發優化機會
7. 算法改進建議
8. 數據結構優化
9. 預期性能提升
10. 優化風險評估

以 JSON 格式返回詳細結果。
"""
        else:  # medium
            return f"""
分析以下 {language} 代碼的性能。

代碼:
```{language}
{code}
```

分析項目：
1. 整體時間複雜度
2. 空間複雜度
3. 主要性能瓶頸（top 5）
4. 優化建議（top 5）
5. 預期性能提升估計

以 JSON 格式返回：
{{
    "score": 75,
    "time_complexity": "O(n^2)",
    "space_complexity": "O(n)",
    "bottlenecks": [
        {{
            "location": "line 10",
            "issue": "Nested loops",
            "impact": "high"
        }}
    ],
    "optimizations": [
        {{
            "suggestion": "Use hash map",
            "expected_improvement": "50%",
            "difficulty": "medium"
        }}
    ],
    "estimated_speedup": "3x"
}}
"""

    def _parse_llm_result(self, result_text: str) -> Dict[str, Any]:
        """解析 LLM 分析結果"""
        try:
            import json
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM result as JSON")
            return {}

    def _merge_analysis(
        self,
        static: Dict[str, Any],
        llm: Dict[str, Any]
    ) -> Dict[str, Any]:
        """合併靜態分析和 LLM 分析結果"""
        # 計算性能評分
        score = llm.get("score", 75)

        # 調整評分基於靜態分析
        if static.get("nested_loops", 0) > 0:
            score -= 10
        if static.get("db_queries", 0) > 5:
            score -= 15
        if static.get("recursion_depth", 0) > 3:
            score -= 10

        score = max(0, min(100, score))

        return {
            "score": score,
            "time_complexity": llm.get("time_complexity", "O(n)"),
            "space_complexity": llm.get("space_complexity", "O(1)"),
            "bottlenecks": llm.get("bottlenecks", []),
            "optimizations": llm.get("optimizations", []),
            "estimated_speedup": llm.get("estimated_speedup", "1x"),
            "loop_count": static.get("loop_count", 0),
            "recursion_depth": static.get("recursion_depth", 0),
            "db_queries": static.get("db_queries", 0)
        }
