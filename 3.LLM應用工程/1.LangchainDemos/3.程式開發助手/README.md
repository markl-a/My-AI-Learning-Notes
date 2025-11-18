# 程式開發助手

## 📚 功能介紹

這是一個基於 LangChain 和 GPT-4 的智能程式開發助手，提供以下功能：

1. **程式碼生成** - 根據需求描述自動生成程式碼
2. **程式碼修復** - 自動檢測和修復程式碼錯誤
3. **程式碼解釋** - 詳細解釋程式碼的功能和實作
4. **程式碼優化** - 提供優化建議並生成更好的程式碼
5. **自動除錯** - 自動執行、檢測錯誤並修復

## 🎯 為什麼使用程式開發助手？

### 傳統開發 vs AI 輔助開發

| 情境 | 傳統方式 | AI 輔助方式 |
|------|---------|------------|
| 學習新函數 | 查文件 → 理解 → 實作 | 描述需求 → 自動生成 |
| 遇到錯誤 | 讀錯誤訊息 → 搜尋 → 嘗試修復 | 提供錯誤 → 自動修復 |
| 程式碼審查 | 人工閱讀 → 手動優化 | 自動分析 → 提供建議 |
| 寫單元測試 | 思考測試案例 → 手寫測試 | 自動生成完整測試 |

## 🚀 快速開始

### 1. 環境設定

```bash
cd "3.LLM應用工程/1.LangchainDemos"

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 OPENAI_API_KEY
```

### 2. 執行範例

```bash
cd "3.程式開發助手"
python code_assistant.py
```

## 💡 使用範例

### 範例 1: 生成程式碼

```python
from code_assistant import CodeAssistant

assistant = CodeAssistant(model_name="gpt-4")

# 描述你要的功能
description = """
建立一個函數，可以驗證電子郵件地址是否有效。
要求：
- 使用正則表達式
- 處理邊界情況
- 加入型別提示
"""

# 生成程式碼（包含測試）
result = assistant.generate_code(
    description=description,
    include_tests=True,
    include_comments=True
)

print(result["code"])
print(result["tests"])
```

**輸出範例：**

```python
import re
from typing import Optional

def validate_email(email: str) -> bool:
    """
    驗證電子郵件地址是否有效

    Args:
        email: 要驗證的電子郵件地址

    Returns:
        bool: 如果有效返回 True，否則返回 False
    """
    # 電子郵件的正則表達式模式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    # 檢查是否為空
    if not email:
        return False

    # 使用正則表達式驗證
    return bool(re.match(pattern, email))
```

### 範例 2: 修復程式碼

```python
# 有問題的程式碼
buggy_code = """
def divide_numbers(a, b):
    return a / b

result = divide_numbers(10, 0)
print(result)
"""

# 自動修復
result = assistant.fix_code(buggy_code)

print("問題：", result["problem"])
print("修復後：", result["fixed_code"])
print("說明：", result["fix_explanation"])
```

**輸出：**

```
問題： 程式碼存在除以零的錯誤，當 b 為 0 時會拋出 ZeroDivisionError

修復後：
def divide_numbers(a, b):
    if b == 0:
        raise ValueError("除數不能為零")
    return a / b

try:
    result = divide_numbers(10, 0)
    print(result)
except ValueError as e:
    print(f"錯誤：{e}")

說明： 加入了除數檢查，並使用 try-except 處理異常
```

### 範例 3: 解釋程式碼

```python
code = """
@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

explanation = assistant.explain_code(code)
print(explanation)
```

**輸出：**

```
這段程式碼實作了費氏數列的計算：

1. @lru_cache(maxsize=None)：
   - 這是一個裝飾器，來自 functools 模組
   - 功能是快取（memoization），儲存已計算過的結果
   - maxsize=None 表示快取無限制大小
   - 可以大幅提升效能，避免重複計算

2. 基本情況 (n < 2)：
   - 當 n 為 0 或 1 時，直接返回 n
   - 這是遞迴的終止條件

3. 遞迴計算：
   - fibonacci(n-1) + fibonacci(n-2)
   - 將問題分解為更小的子問題

使用場景：計算費氏數列、動態規劃問題
```

### 範例 4: 優化程式碼

```python
slow_code = """
def find_max(numbers):
    max_num = numbers[0]
    for i in range(len(numbers)):
        if numbers[i] > max_num:
            max_num = numbers[i]
    return max_num
"""

result = assistant.optimize_code(slow_code)
print("建議：", result["suggestions"])
print("優化後：", result["optimized_code"])
```

**輸出：**

```
建議：
- 使用 Python 內建的 max() 函數
- 加入空列表檢查
- 使用更 Pythonic 的寫法

優化後：
def find_max(numbers):
    if not numbers:
        raise ValueError("列表不能為空")
    return max(numbers)
```

### 範例 5: 自動執行並修復

```python
# 有語法錯誤的程式碼
buggy_code = """
def greet(name)
    print(f"Hello, {name}!")

greet("World")
"""

# 自動嘗試修復（最多 3 次）
result = assistant.auto_fix_with_execution(
    buggy_code,
    max_attempts=3
)

if result["success"]:
    print("✓ 修復成功！")
    print("最終程式碼：", result["final_code"])
    print("執行輸出：", result["output"])
else:
    print("✗ 修復失敗")
```

## 🎓 進階用法

### 自訂模型參數

```python
# 使用 gpt-3.5-turbo（更快、更便宜）
assistant = CodeAssistant(
    model_name="gpt-3.5-turbo",
    temperature=0.2  # 降低隨機性，提高一致性
)

# 使用 gpt-4（更準確、更強大）
assistant = CodeAssistant(
    model_name="gpt-4",
    temperature=0.1  # 非常確定性的輸出
)
```

### 批量處理

```python
assistant = CodeAssistant()

# 生成多個相關函數
functions = [
    "建立一個計算平均值的函數",
    "建立一個計算中位數的函數",
    "建立一個計算標準差的函數",
]

for desc in functions:
    result = assistant.generate_code(desc)
    print(f"\n{desc}:")
    print(result["code"])
```

### 整合到開發流程

```python
# 1. 生成初始程式碼
code = assistant.generate_code("建立一個用戶認證系統")["code"]

# 2. 自動生成測試
tests = assistant._generate_tests(code, "python")

# 3. 優化程式碼
optimized = assistant.optimize_code(code)["optimized_code"]

# 4. 加入文件
explanation = assistant.explain_code(optimized)

# 5. 儲存
with open("auth_system.py", "w") as f:
    f.write(optimized)
with open("test_auth_system.py", "w") as f:
    f.write(tests)
with open("README.md", "w") as f:
    f.write(explanation)
```

## 💻 支援的程式語言

目前主要支援：
- ✅ Python
- ✅ JavaScript
- ✅ Java
- ✅ C++
- ✅ Go
- ✅ Rust
- ✅ 其他主流語言

使用方式：

```python
# JavaScript
result = assistant.generate_code(
    description="建立一個 React 元件",
    language="javascript"
)

# Java
result = assistant.generate_code(
    description="建立一個單例模式",
    language="java"
)
```

## 📊 效能與成本

### 模型選擇建議

| 任務類型 | 建議模型 | 原因 |
|---------|---------|------|
| 簡單程式碼生成 | gpt-3.5-turbo | 速度快、成本低 |
| 複雜邏輯實作 | gpt-4 | 準確度高 |
| 程式碼修復 | gpt-4 | 理解能力強 |
| 程式碼解釋 | gpt-3.5-turbo | 足夠應付 |
| 程式碼優化 | gpt-4 | 需要深入理解 |

### 成本估算

以 OpenAI 定價為例：

- **gpt-3.5-turbo**: ~$0.002 / 1K tokens
- **gpt-4**: ~$0.03 / 1K tokens

一般程式碼生成任務：
- 簡單函數：500-1000 tokens → $0.001-0.03
- 複雜功能：2000-5000 tokens → $0.004-0.15

## 🛠️ 實際應用場景

### 1. 學習程式設計

```python
# 學習新概念
assistant = CodeAssistant()

# 生成範例
code = assistant.generate_code(
    "展示 Python 裝飾器的用法，包含多個範例"
)

# 理解範例
explanation = assistant.explain_code(code["code"])
```

### 2. 程式碼審查助手

```python
def code_review(code: str):
    assistant = CodeAssistant()

    # 檢查優化空間
    optimization = assistant.optimize_code(code)

    # 生成測試
    tests = assistant._generate_tests(code, "python")

    return {
        "suggestions": optimization["suggestions"],
        "optimized_code": optimization["optimized_code"],
        "tests": tests
    }
```

### 3. 快速原型開發

```python
# 快速實作功能
features = [
    "使用者註冊功能",
    "密碼加密功能",
    "JWT token 生成",
    "權限驗證中介層"
]

for feature in features:
    code = assistant.generate_code(feature, include_tests=True)
    # 儲存或整合到專案
```

### 4. 除錯助手

```python
# 自動除錯工作流程
def debug_workflow(buggy_code):
    assistant = CodeAssistant()

    # 嘗試自動修復
    result = assistant.auto_fix_with_execution(buggy_code)

    if not result["success"]:
        # 如果自動修復失敗，提供詳細分析
        explanation = assistant.explain_code(buggy_code)
        optimization = assistant.optimize_code(buggy_code)

        return {
            "auto_fix_failed": True,
            "explanation": explanation,
            "optimization": optimization
        }

    return result
```

## ⚠️ 注意事項

### 1. 安全性

```python
# ❌ 不要執行未審查的程式碼
result = assistant.generate_code("任意描述")
exec(result["code"])  # 危險！

# ✅ 先審查再執行
result = assistant.generate_code("任意描述")
print(result["code"])  # 審查程式碼
# 確認安全後再執行
```

### 2. 驗證輸出

AI 生成的程式碼可能有：
- 語法錯誤（較少見）
- 邏輯錯誤
- 效能問題
- 安全漏洞

**建議：**
- 總是審查生成的程式碼
- 執行測試
- 使用靜態分析工具
- 進行安全掃描

### 3. 版權與授權

- AI 生成的程式碼通常可以自由使用
- 但建議檢查具體的服務條款
- 避免將敏感或專有資訊傳送給 AI

## 🔍 常見問題

### Q1: 生成的程式碼品質如何？

**A:** 取決於：
- 模型選擇（gpt-4 > gpt-3.5-turbo）
- 需求描述的清晰度
- 問題的複雜度

建議：提供詳細、具體的描述

### Q2: 可以生成完整的專案嗎？

**A:** 可以，但建議：
- 分模組生成
- 逐步整合
- 持續測試
- 人工審查

### Q3: 如何提高生成品質？

**A:**
1. 提供清晰的需求
2. 指定具體的技術要求
3. 包含範例或參考
4. 使用更好的模型
5. 迭代優化

### Q4: 費用如何控制？

**A:**
- 簡單任務用 gpt-3.5-turbo
- 設定合理的 token 限制
- 快取常用結果
- 批量處理請求

## 📚 延伸閱讀

- [LangChain 官方文件](https://python.langchain.com/)
- [OpenAI API 文件](https://platform.openai.com/docs/)
- [程式碼品質最佳實踐](https://google.github.io/styleguide/)
- [單元測試指南](https://docs.python.org/3/library/unittest.html)

## 下一步

1. 嘗試生成你需要的程式碼
2. 整合到你的開發流程
3. 建立自己的程式碼範本庫
4. 探索更多 AI 輔助開發工具

---

Happy Coding! 🚀
