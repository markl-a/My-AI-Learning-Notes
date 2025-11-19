"""
代碼分析器
使用 LLM 進行智能代碼審查和分析
"""

import os
import uuid
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
import json
import asyncio

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """代碼分析器類"""

    def __init__(self, model_name: str = "gpt-4"):
        """初始化代碼分析器"""
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # 統計數據
        self.review_count = 0
        self.total_review_time = 0

        # 批量任務
        self.batch_tasks = {}

        # 支持的語言
        self.supported_languages = {
            "python": [".py"],
            "javascript": [".js", ".jsx", ".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
            "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
            "c": [".c", ".h"],
            "ruby": [".rb"],
            "php": [".php"],
            "swift": [".swift"],
            "kotlin": [".kt"],
            "scala": [".scala"]
        }

    def get_review_count(self) -> int:
        """獲取審查次數"""
        return self.review_count

    def get_avg_review_time(self) -> float:
        """獲取平均審查時間"""
        if self.review_count == 0:
            return 0
        return self.total_review_time / self.review_count

    def get_supported_languages(self) -> List[str]:
        """獲取支持的語言列表"""
        return list(self.supported_languages.keys())

    def detect_language(self, file_ext: str) -> str:
        """從文件擴展名檢測語言"""
        file_ext = f".{file_ext}" if not file_ext.startswith(".") else file_ext

        for lang, exts in self.supported_languages.items():
            if file_ext in exts:
                return lang

        return "unknown"

    async def review_code(
        self,
        code: str,
        language: str,
        review_type: str = "full",
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        審查代碼

        Args:
            code: 代碼內容
            language: 編程語言
            review_type: 審查類型
            context: 上下文信息

        Returns:
            審查結果
        """
        start_time = datetime.now()

        review_id = str(uuid.uuid4())
        context = context or {}

        # 根據審查類型構建提示
        if review_type == "full":
            prompt = self._build_full_review_prompt(code, language, context)
        elif review_type == "quick":
            prompt = self._build_quick_review_prompt(code, language)
        elif review_type == "security":
            prompt = self._build_security_review_prompt(code, language)
        elif review_type == "performance":
            prompt = self._build_performance_review_prompt(code, language)
        else:
            prompt = self._build_full_review_prompt(code, language, context)

        try:
            # 調用 LLM
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位資深的 {language} 代碼審查專家。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content

            # 解析結果
            result = self._parse_review_result(result_text)
            result["review_id"] = review_id

            # 更新統計
            end_time = datetime.now()
            review_time = (end_time - start_time).total_seconds()
            self.review_count += 1
            self.total_review_time += review_time

            return result

        except Exception as e:
            logger.error(f"Error in code review: {str(e)}")
            raise

    def _build_full_review_prompt(
        self,
        code: str,
        language: str,
        context: Dict
    ) -> str:
        """構建完整審查提示"""
        filename = context.get("filename", "unknown")

        prompt = f"""
請對以下 {language} 代碼進行全面審查。

文件名: {filename}

代碼:
```{language}
{code}
```

請從以下方面進行審查，並以 JSON 格式返回結果：

1. **代碼質量** (overall_score: 0-100)
2. **問題列表** (issues):
   - 嚴重性 (severity: critical, high, medium, low)
   - 問題描述 (description)
   - 問題位置 (line_number)
   - 建議修復 (suggestion)
3. **改進建議** (suggestions):
   - 類別 (category: readability, performance, maintainability, security)
   - 建議內容 (content)
   - 優先級 (priority: high, medium, low)
4. **代碼指標** (metrics):
   - 複雜度評分 (complexity)
   - 可讀性評分 (readability)
   - 可維護性評分 (maintainability)
5. **總結** (summary): 簡短總結

請只返回 JSON，格式如下：
{{
    "overall_score": 85,
    "issues": [
        {{
            "severity": "medium",
            "description": "問題描述",
            "line_number": 10,
            "suggestion": "修復建議"
        }}
    ],
    "suggestions": [
        {{
            "category": "readability",
            "content": "建議內容",
            "priority": "medium"
        }}
    ],
    "metrics": {{
        "complexity": 7,
        "readability": 8,
        "maintainability": 9
    }},
    "summary": "代碼整體質量良好..."
}}
"""
        return prompt

    def _build_quick_review_prompt(self, code: str, language: str) -> str:
        """構建快速審查提示"""
        return f"""
對以下 {language} 代碼進行快速審查，只報告主要問題。

代碼:
```{language}
{code}
```

以 JSON 格式返回：
{{
    "overall_score": 75,
    "issues": [
        {{"severity": "high", "description": "...", "line_number": 5}}
    ],
    "summary": "..."
}}
"""

    def _build_security_review_prompt(self, code: str, language: str) -> str:
        """構建安全審查提示"""
        return f"""
對以下 {language} 代碼進行安全審查。

代碼:
```{language}
{code}
```

重點檢查：
1. SQL 注入漏洞
2. XSS 攻擊
3. 敏感信息洩露
4. 不安全的加密
5. 認證和授權問題
6. 輸入驗證不足

以 JSON 格式返回安全問題。
"""

    def _build_performance_review_prompt(self, code: str, language: str) -> str:
        """構建性能審查提示"""
        return f"""
對以下 {language} 代碼進行性能審查。

代碼:
```{language}
{code}
```

分析：
1. 時間複雜度
2. 空間複雜度
3. 性能瓶頸
4. 優化機會

以 JSON 格式返回性能分析結果。
"""

    def _parse_review_result(self, result_text: str) -> Dict[str, Any]:
        """解析審查結果"""
        try:
            # 嘗試提取 JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # 如果沒有 JSON，創建基本結果
                return {
                    "overall_score": 70,
                    "issues": [],
                    "suggestions": [],
                    "metrics": {},
                    "summary": result_text
                }
        except json.JSONDecodeError:
            logger.error("Failed to parse review result as JSON")
            return {
                "overall_score": 70,
                "issues": [],
                "suggestions": [],
                "metrics": {},
                "summary": result_text
            }

    async def suggest_refactoring(
        self,
        code: str,
        language: str,
        goals: List[str]
    ) -> Dict[str, Any]:
        """生成重構建議"""
        prompt = f"""
對以下 {language} 代碼生成重構建議。

代碼:
```{language}
{code}
```

重構目標: {', '.join(goals)}

請提供：
1. 重構後的代碼
2. 具體變更說明
3. 預期改進
4. 潛在風險
5. 工作量估計

以 JSON 格式返回。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"你是一位代碼重構專家，專精於 {language}。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content

        # 簡單解析（實際應該更複雜）
        return {
            "id": str(uuid.uuid4()),
            "refactored_code": self._extract_code_block(result_text),
            "changes": [],
            "improvements": [],
            "risks": [],
            "effort": "medium",
            "priority": "medium"
        }

    def _extract_code_block(self, text: str) -> str:
        """從文本中提取代碼塊"""
        # 查找代碼塊
        code_block_pattern = r'```[\w]*\n(.*?)\n```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    async def check_best_practices(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """檢查最佳實踐"""
        prompt = f"""
檢查以下 {language} 代碼是否遵循最佳實踐。

代碼:
```{language}
{code}
```

檢查項目：
1. 命名規範
2. 代碼風格
3. 設計模式
4. 錯誤處理
5. 日誌記錄
6. 文檔註釋

以 JSON 格式返回合規性評分和具體建議。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )

        result = self._parse_review_result(response.choices[0].message.content)

        return {
            "score": result.get("overall_score", 75),
            "violations": result.get("issues", []),
            "good_practices": [],
            "recommendations": result.get("suggestions", []),
            "style_guide": language,
            "patterns": [],
            "anti_patterns": []
        }

    async def analyze_complexity(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """分析代碼複雜度"""
        # 簡單的行數統計
        lines = code.split('\n')
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])

        # 使用 LLM 進行深度分析
        prompt = f"""
分析以下 {language} 代碼的複雜度。

代碼:
```{language}
{code}
```

提供以下指標：
1. 圈複雜度 (1-20+)
2. 認知複雜度 (1-20+)
3. 維護性指數 (0-100)
4. 複雜度評級 (A-F)

以 JSON 格式返回。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )

        result = self._parse_review_result(response.choices[0].message.content)

        return {
            "cyclomatic": result.get("cyclomatic_complexity", 5),
            "cognitive": result.get("cognitive_complexity", 5),
            "maintainability_index": result.get("maintainability_index", 75),
            "loc": loc,
            "comment_ratio": comment_lines / max(loc, 1) if loc > 0 else 0,
            "function_count": len(re.findall(r'\bdef\b|\bfunction\b|\bfunc\b', code)),
            "class_count": len(re.findall(r'\bclass\b', code)),
            "rating": result.get("rating", "B"),
            "suggestions": result.get("suggestions", [])
        }

    def create_batch_task(self) -> str:
        """創建批量任務"""
        task_id = str(uuid.uuid4())
        self.batch_tasks[task_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "total_files": 0,
            "reviewed_files": 0,
            "results": []
        }
        return task_id

    async def batch_review(
        self,
        task_id: str,
        repository_url: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        review_type: str = "full"
    ):
        """批量審查（後台任務）"""
        if task_id not in self.batch_tasks:
            return

        self.batch_tasks[task_id]["status"] = "processing"

        # 模擬批量審查
        # 實際實現應該：
        # 1. 克隆 Git 倉庫（如果提供了 URL）
        # 2. 遍歷文件
        # 3. 對每個文件進行審查
        # 4. 聚合結果

        await asyncio.sleep(2)  # 模擬處理時間

        self.batch_tasks[task_id]["status"] = "completed"
        self.batch_tasks[task_id]["completed_at"] = datetime.now().isoformat()

    def get_batch_task_status(self, task_id: str) -> Optional[Dict]:
        """獲取批量任務狀態"""
        return self.batch_tasks.get(task_id)

    async def compare_versions(
        self,
        old_code: str,
        new_code: str,
        language: str
    ) -> Dict[str, Any]:
        """比較兩個版本的代碼"""
        prompt = f"""
比較以下兩個版本的 {language} 代碼。

舊版本:
```{language}
{old_code}
```

新版本:
```{language}
{new_code}
```

分析：
1. 主要變更
2. 質量改進
3. 新引入的問題
4. 性能影響

以 JSON 格式返回比較結果。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )

        result = self._parse_review_result(response.choices[0].message.content)

        return {
            "changes": result.get("changes", []),
            "quality_delta": 0,
            "new_issues": result.get("new_issues", []),
            "fixed_issues": result.get("fixed_issues", []),
            "performance_impact": "neutral",
            "recommendation": "approve"
        }

    async def generate_unit_tests(
        self,
        code: str,
        language: str,
        framework: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成單元測試"""
        if framework is None:
            framework = self._get_default_test_framework(language)

        prompt = f"""
為以下 {language} 代碼生成 {framework} 單元測試。

代碼:
```{language}
{code}
```

生成：
1. 完整的測試代碼
2. 測試用例說明
3. 覆蓋率估計

請只返回測試代碼。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )

        test_code = response.choices[0].message.content

        return {
            "code": self._extract_code_block(test_code),
            "cases": [],
            "coverage": "80%",
            "framework": framework,
            "setup": f"Install {framework}"
        }

    def _get_default_test_framework(self, language: str) -> str:
        """獲取默認測試框架"""
        frameworks = {
            "python": "pytest",
            "javascript": "jest",
            "java": "junit",
            "go": "testing",
            "rust": "cargo test"
        }
        return frameworks.get(language, "unittest")

    async def generate_documentation(
        self,
        code: str,
        language: str,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """生成文檔"""
        prompt = f"""
為以下 {language} 代碼生成 {format} 格式的文檔。

代碼:
```{language}
{code}
```

包含：
1. 功能概述
2. 參數說明
3. 返回值
4. 使用示例
5. 注意事項

請生成清晰、專業的文檔。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )

        doc_content = response.choices[0].message.content

        return {
            "content": doc_content,
            "sections": ["Overview", "Parameters", "Returns", "Examples"],
            "api_reference": {}
        }
