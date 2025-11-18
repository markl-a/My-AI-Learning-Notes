"""
程式碼開發助手
功能：
1. 程式碼生成
2. 程式碼修復
3. 程式碼解釋
4. 程式碼優化建議
5. 單元測試生成
"""

import os
import sys
from pathlib import Path
import subprocess
import traceback
from typing import Dict, List, Optional

# 添加父目錄到路徑以導入 utils
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_environment, get_llm, setup_langsmith

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain


class CodeAssistant:
    """程式碼開發助手"""

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.2):
        """
        初始化程式碼助手

        Args:
            model_name: 使用的模型（建議使用 gpt-4 以獲得更好的程式碼品質）
            temperature: 溫度參數（較低的值會產生更確定性的程式碼）
        """
        load_environment()
        setup_langsmith()

        self.llm = get_llm(model=model_name, temperature=temperature)
        print(f"✓ 程式碼助手初始化完成（模型: {model_name}）")

    def generate_code(
        self,
        description: str,
        language: str = "python",
        include_tests: bool = False,
        include_comments: bool = True
    ) -> Dict[str, str]:
        """
        生成程式碼

        Args:
            description: 功能描述
            language: 程式語言
            include_tests: 是否包含單元測試
            include_comments: 是否包含註解

        Returns:
            包含程式碼和測試的字典
        """
        print(f"\n正在生成 {language} 程式碼...")

        # 建立 prompt
        template = """你是一個專業的 {language} 程式設計師。
請根據以下描述生成高品質的程式碼。

要求：
- 程式碼要清晰、易讀
- 遵循 {language} 的最佳實踐和命名規範
- {comments_requirement}
- 包含適當的錯誤處理
- 如果適用，加入型別提示

功能描述：
{description}

請只返回程式碼，不要包含額外的解釋。
"""

        comments_requirement = (
            "加入詳細的註解說明每個部分的功能"
            if include_comments
            else "不需要註解"
        )

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        code = chain.invoke({
            "language": language,
            "description": description,
            "comments_requirement": comments_requirement
        })

        result = {"code": code.strip()}

        # 如果需要測試
        if include_tests:
            print("正在生成單元測試...")
            test_code = self._generate_tests(code, language)
            result["tests"] = test_code

        print("✓ 程式碼生成完成")
        return result

    def _generate_tests(self, code: str, language: str = "python") -> str:
        """
        生成單元測試

        Args:
            code: 要測試的程式碼
            language: 程式語言

        Returns:
            測試程式碼
        """
        template = """你是一個專業的測試工程師。
請為以下 {language} 程式碼生成完整的單元測試。

要求：
- 使用 {language} 的標準測試框架（Python 使用 pytest 或 unittest）
- 測試各種情況（正常情況、邊界情況、錯誤情況）
- 測試要全面且易於理解
- 包含適當的註解

程式碼：
{code}

請只返回測試程式碼，不要包含額外的解釋。
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        tests = chain.invoke({
            "language": language,
            "code": code
        })

        return tests.strip()

    def fix_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        language: str = "python"
    ) -> Dict[str, str]:
        """
        修復程式碼

        Args:
            code: 有問題的程式碼
            error_message: 錯誤訊息（如果有）
            language: 程式語言

        Returns:
            包含修復後程式碼和說明的字典
        """
        print("\n正在分析並修復程式碼...")

        template = """你是一個專業的 {language} 程式設計師和除錯專家。
請分析以下程式碼並修復其中的問題。

{error_context}

程式碼：
```{language}
{code}
```

請提供：
1. 修復後的完整程式碼
2. 問題說明（用繁體中文）
3. 修復說明（用繁體中文）

回應格式：
### 修復後的程式碼
```{language}
[修復後的程式碼]
```

### 問題說明
[問題描述]

### 修復說明
[如何修復的說明]
"""

        error_context = (
            f"錯誤訊息：\n{error_message}\n"
            if error_message
            else "請找出並修復程式碼中的潛在問題（語法錯誤、邏輯錯誤、效能問題等）。"
        )

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "language": language,
            "code": code,
            "error_context": error_context
        })

        # 解析回應
        parts = response.split("###")
        fixed_code = ""
        problem = ""
        fix_explanation = ""

        for part in parts:
            if "修復後的程式碼" in part:
                # 提取程式碼區塊
                code_block = part.split("```")
                if len(code_block) >= 2:
                    fixed_code = code_block[1].strip()
                    # 移除語言標記
                    if fixed_code.startswith(language):
                        fixed_code = fixed_code[len(language):].strip()
            elif "問題說明" in part:
                problem = part.replace("問題說明", "").strip()
            elif "修復說明" in part:
                fix_explanation = part.replace("修復說明", "").strip()

        print("✓ 程式碼修復完成")

        return {
            "fixed_code": fixed_code,
            "problem": problem,
            "fix_explanation": fix_explanation,
            "full_response": response
        }

    def explain_code(self, code: str, language: str = "python") -> str:
        """
        解釋程式碼

        Args:
            code: 要解釋的程式碼
            language: 程式語言

        Returns:
            程式碼解釋（繁體中文）
        """
        print("\n正在分析程式碼...")

        template = """你是一個程式教學專家。
請用繁體中文詳細解釋以下 {language} 程式碼的功能和實作方式。

程式碼：
```{language}
{code}
```

請包含：
1. 整體功能說明
2. 逐行或逐區塊的詳細解釋
3. 使用的重要概念或技術
4. 可能的使用場景

請用清晰易懂的方式解釋，適合初學者理解。
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        explanation = chain.invoke({
            "language": language,
            "code": code
        })

        print("✓ 程式碼解釋完成")
        return explanation.strip()

    def optimize_code(self, code: str, language: str = "python") -> Dict[str, str]:
        """
        優化程式碼並提供建議

        Args:
            code: 要優化的程式碼
            language: 程式語言

        Returns:
            包含優化建議和優化後程式碼的字典
        """
        print("\n正在分析並優化程式碼...")

        template = """你是一個資深的 {language} 程式設計師和效能優化專家。
請分析以下程式碼並提供優化建議。

程式碼：
```{language}
{code}
```

請提供：
1. 優化建議（列點說明，用繁體中文）
2. 優化後的程式碼
3. 優化說明（解釋為什麼這樣更好，用繁體中文）

回應格式：
### 優化建議
- [建議1]
- [建議2]
...

### 優化後的程式碼
```{language}
[優化後的程式碼]
```

### 優化說明
[詳細說明]
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "language": language,
            "code": code
        })

        # 解析回應
        parts = response.split("###")
        suggestions = ""
        optimized_code = ""
        explanation = ""

        for part in parts:
            if "優化建議" in part:
                suggestions = part.replace("優化建議", "").strip()
            elif "優化後的程式碼" in part:
                code_block = part.split("```")
                if len(code_block) >= 2:
                    optimized_code = code_block[1].strip()
                    if optimized_code.startswith(language):
                        optimized_code = optimized_code[len(language):].strip()
            elif "優化說明" in part:
                explanation = part.replace("優化說明", "").strip()

        print("✓ 程式碼優化完成")

        return {
            "suggestions": suggestions,
            "optimized_code": optimized_code,
            "explanation": explanation,
            "full_response": response
        }

    def auto_fix_with_execution(
        self,
        code: str,
        max_attempts: int = 3
    ) -> Dict[str, any]:
        """
        自動執行並修復程式碼（僅限 Python）

        Args:
            code: 要執行的程式碼
            max_attempts: 最大修復嘗試次數

        Returns:
            包含執行結果和修復過程的字典
        """
        print(f"\n開始自動修復流程（最多嘗試 {max_attempts} 次）...")

        attempts = []
        current_code = code

        for attempt in range(1, max_attempts + 1):
            print(f"\n--- 嘗試 {attempt}/{max_attempts} ---")

            # 執行程式碼
            result = self._execute_python_code(current_code)

            if result["success"]:
                print("✓ 程式碼執行成功！")
                return {
                    "success": True,
                    "final_code": current_code,
                    "output": result["output"],
                    "attempts": attempts,
                    "total_attempts": attempt
                }
            else:
                print(f"✗ 執行失敗：{result['error']}")
                attempts.append({
                    "attempt": attempt,
                    "code": current_code,
                    "error": result["error"]
                })

                if attempt < max_attempts:
                    # 嘗試修復
                    fix_result = self.fix_code(
                        current_code,
                        error_message=result["error"],
                        language="python"
                    )
                    current_code = fix_result["fixed_code"]
                    print(f"修復說明：{fix_result['fix_explanation']}")

        print(f"\n✗ 經過 {max_attempts} 次嘗試後仍無法修復")
        return {
            "success": False,
            "final_code": current_code,
            "attempts": attempts,
            "total_attempts": max_attempts
        }

    def _execute_python_code(self, code: str) -> Dict[str, any]:
        """
        執行 Python 程式碼

        Args:
            code: 要執行的程式碼

        Returns:
            執行結果
        """
        try:
            # 將程式碼寫入臨時檔案
            temp_file = Path("temp_code.py")
            temp_file.write_text(code)

            # 執行程式碼
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=5
            )

            # 清理臨時檔案
            temp_file.unlink()

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }

        except subprocess.TimeoutExpired:
            if temp_file.exists():
                temp_file.unlink()
            return {
                "success": False,
                "error": "程式碼執行超時（超過 5 秒）"
            }
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# 示範範例
# ============================================================================

def demo_code_generation():
    """示範程式碼生成"""
    print("=" * 80)
    print("示範 1: 程式碼生成")
    print("=" * 80)

    assistant = CodeAssistant(model_name="gpt-4", temperature=0.2)

    # 生成程式碼
    description = """
    建立一個函數，可以計算費氏數列的第 n 項。
    要求：
    - 使用動態規劃方法
    - 處理負數輸入
    - 加入型別提示
    """

    result = assistant.generate_code(
        description=description,
        include_tests=True,
        include_comments=True
    )

    print("\n生成的程式碼：")
    print("=" * 80)
    print(result["code"])

    if "tests" in result:
        print("\n生成的測試：")
        print("=" * 80)
        print(result["tests"])


def demo_code_fixing():
    """示範程式碼修復"""
    print("\n" + "=" * 80)
    print("示範 2: 程式碼修復")
    print("=" * 80)

    assistant = CodeAssistant(model_name="gpt-4")

    # 有問題的程式碼
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# 測試
result = calculate_average([])
print(result)
"""

    result = assistant.fix_code(buggy_code)

    print("\n問題說明：")
    print(result["problem"])

    print("\n修復後的程式碼：")
    print("=" * 80)
    print(result["fixed_code"])

    print("\n修復說明：")
    print(result["fix_explanation"])


def demo_code_explanation():
    """示範程式碼解釋"""
    print("\n" + "=" * 80)
    print("示範 3: 程式碼解釋")
    print("=" * 80)

    assistant = CodeAssistant(model_name="gpt-4")

    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

    explanation = assistant.explain_code(code)

    print("\n程式碼解釋：")
    print("=" * 80)
    print(explanation)


def demo_code_optimization():
    """示範程式碼優化"""
    print("\n" + "=" * 80)
    print("示範 4: 程式碼優化")
    print("=" * 80)

    assistant = CodeAssistant(model_name="gpt-4")

    code = """
def find_duplicates(numbers):
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j]:
                if numbers[i] not in duplicates:
                    duplicates.append(numbers[i])
    return duplicates
"""

    result = assistant.optimize_code(code)

    print("\n優化建議：")
    print(result["suggestions"])

    print("\n優化後的程式碼：")
    print("=" * 80)
    print(result["optimized_code"])

    print("\n優化說明：")
    print(result["explanation"])


def demo_auto_fix():
    """示範自動修復"""
    print("\n" + "=" * 80)
    print("示範 5: 自動執行並修復")
    print("=" * 80)

    assistant = CodeAssistant(model_name="gpt-4")

    # 有錯誤的程式碼
    buggy_code = """
def greet(name)
    print(f"Hello, {name}!")

greet("World")
"""

    result = assistant.auto_fix_with_execution(buggy_code, max_attempts=3)

    if result["success"]:
        print(f"\n✓ 成功修復！（總共嘗試 {result['total_attempts']} 次）")
        print("\n最終程式碼：")
        print("=" * 80)
        print(result["final_code"])
        print("\n執行輸出：")
        print(result["output"])
    else:
        print(f"\n✗ 修復失敗（已嘗試 {result['total_attempts']} 次）")


# ============================================================================
# 主程式
# ============================================================================

if __name__ == "__main__":
    try:
        # 執行所有示範
        demo_code_generation()
        demo_code_fixing()
        demo_code_explanation()
        demo_code_optimization()
        demo_auto_fix()

        print("\n" + "=" * 80)
        print("✓ 所有示範執行完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
