# AI 程式助手深度指南 (AI Coding Assistants)

## 概述

AI 程式助手在 2025 年已成為軟體開發的標準工具。從 GitHub Copilot 到 Claude Code，這些工具正在根本性地改變程式設計的方式。

## 主流 AI 程式助手比較

```
┌─────────────────────────────────────────────────────────────┐
│                 AI 程式助手生態系統 2025                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IDE 整合型                    獨立工具型                   │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ GitHub Copilot  │          │ Claude Code     │          │
│  │ (VS Code/JB)    │          │ (CLI/Terminal)  │          │
│  ├─────────────────┤          ├─────────────────┤          │
│  │ Cursor          │          │ Aider           │          │
│  │ (Fork VS Code)  │          │ (CLI)           │          │
│  ├─────────────────┤          ├─────────────────┤          │
│  │ Codeium         │          │ Continue        │          │
│  │ (多 IDE)        │          │ (開源)          │          │
│  └─────────────────┘          └─────────────────┘          │
│                                                             │
│  功能比較                                                   │
│  ┌──────────────┬─────────┬─────────┬─────────┬─────────┐  │
│  │ 功能         │ Copilot │ Cursor  │ Claude  │ Aider   │  │
│  ├──────────────┼─────────┼─────────┼─────────┼─────────┤  │
│  │ 程式碼補全   │ ✅      │ ✅      │ ✅      │ ✅      │  │
│  │ 聊天對話     │ ✅      │ ✅      │ ✅      │ ✅      │  │
│  │ 多檔案編輯   │ ⚠️      │ ✅      │ ✅      │ ✅      │  │
│  │ 程式碼庫理解 │ ✅      │ ✅      │ ✅      │ ⚠️      │  │
│  │ 終端機整合   │ ⚠️      │ ✅      │ ✅      │ ✅      │  │
│  │ MCP 支援     │ ❌      │ ❌      │ ✅      │ ❌      │  │
│  │ 開源         │ ❌      │ ❌      │ ❌      │ ✅      │  │
│  └──────────────┴─────────┴─────────┴─────────┴─────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. GitHub Copilot 進階使用

### 高效提示技巧

```python
# 技巧 1: 詳細的函數註解
def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    years: int,
    compounds_per_year: int = 12
) -> float:
    """
    計算複利終值。

    公式: A = P(1 + r/n)^(nt)

    Args:
        principal: 本金
        annual_rate: 年利率（小數形式，如 0.05 表示 5%）
        years: 投資年數
        compounds_per_year: 每年複利次數

    Returns:
        投資終值

    Example:
        >>> calculate_compound_interest(1000, 0.05, 10, 12)
        1647.01
    """
    # Copilot 會根據詳細註解生成正確的實作
    return principal * (1 + annual_rate / compounds_per_year) ** (compounds_per_year * years)


# 技巧 2: 使用範例引導
# 範例輸入: [3, 1, 4, 1, 5, 9, 2, 6]
# 範例輸出: [1, 1, 2, 3, 4, 5, 6, 9]
def quick_sort(arr: list[int]) -> list[int]:
    """使用快速排序演算法排序陣列"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# 技巧 3: 分步驟註解
def process_csv_file(file_path: str) -> dict:
    """處理 CSV 檔案並返回統計資訊"""
    # 步驟 1: 讀取 CSV 檔案
    import csv
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 步驟 2: 計算總行數
    total_rows = len(rows)

    # 步驟 3: 找出所有欄位
    columns = list(rows[0].keys()) if rows else []

    # 步驟 4: 計算每個數值欄位的統計
    stats = {}
    for col in columns:
        values = []
        for row in rows:
            try:
                values.append(float(row[col]))
            except (ValueError, TypeError):
                continue
        if values:
            stats[col] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }

    # 步驟 5: 返回結果
    return {
        'total_rows': total_rows,
        'columns': columns,
        'numeric_stats': stats
    }
```

### Copilot Chat 最佳實踐

```python
# 在 VS Code 中使用 Copilot Chat

# 1. 解釋程式碼
# 選取程式碼後，按 Ctrl+I 輸入：
# "解釋這段程式碼的作用"

# 2. 重構建議
# "重構這個函數，提高可讀性並添加錯誤處理"

# 3. 生成測試
# "為這個類別生成單元測試，覆蓋邊界情況"

# 4. 修復錯誤
# 貼上錯誤訊息，問：
# "這個錯誤是什麼原因？如何修復？"

# 5. 程式碼審查
# "審查這段程式碼，指出潛在問題和改進建議"
```

### Copilot 工作區命令

```python
# 使用 @workspace 進行專案級別查詢

# @workspace 這個專案使用什麼資料庫？
# @workspace 找出所有處理用戶認證的程式碼
# @workspace 這個 API 端點如何處理錯誤？
# @workspace 解釋這個專案的架構

# 使用 @terminal 處理終端機相關
# @terminal 如何執行這個專案的測試？
# @terminal 這個錯誤訊息是什麼意思？

# 使用 @vscode 處理編輯器相關
# @vscode 如何設定 Python 的 linter？
# @vscode 有什麼快捷鍵可以格式化程式碼？
```

## 2. Cursor IDE 深度使用

### Cursor 特有功能

```python
# Cursor 的 Composer 功能 - 多檔案編輯

# 使用 Cmd+K (Mac) 或 Ctrl+K (Windows) 開啟 Composer

# 範例提示：
"""
建立一個 FastAPI 應用程式，包含：
1. main.py - 主應用程式入口
2. models.py - Pydantic 模型
3. database.py - 資料庫連接
4. routes/users.py - 用戶路由
5. routes/items.py - 項目路由

實作基本的 CRUD 操作。
"""

# Cursor 會同時生成多個檔案

# Cursor 的 @ 參考功能
# @file:main.py - 參考特定檔案
# @folder:routes - 參考整個資料夾
# @code - 參考選取的程式碼
# @docs - 參考文件
# @web - 搜尋網路
```

### Cursor Rules 設定

```python
# .cursorrules 檔案範例
"""
# 專案規則

## 程式碼風格
- 使用 Python 3.11+ 語法
- 遵循 PEP 8 規範
- 使用 type hints
- 函數和類別必須有 docstring

## 架構規則
- 使用依賴注入
- 遵循 clean architecture
- API 路由放在 routes/ 目錄
- 業務邏輯放在 services/ 目錄
- 資料模型放在 models/ 目錄

## 命名規則
- 類別: PascalCase
- 函數: snake_case
- 常數: UPPER_SNAKE_CASE
- 私有方法: _leading_underscore

## 偏好
- 優先使用 async/await
- 使用 Pydantic 進行資料驗證
- 錯誤處理使用自定義異常類別
- 日誌使用 structlog

## 禁止
- 不要使用 print() 進行日誌
- 不要在函數中硬編碼配置值
- 不要使用 * import
"""
```

### Cursor 進階工作流程

```python
# 工作流程 1: 從設計文件生成程式碼

# 1. 準備設計文件 design.md
"""
# API 設計

## 用戶 API

### POST /users
建立新用戶
- 請求: {name: string, email: string, password: string}
- 回應: {id: int, name: string, email: string}

### GET /users/{id}
取得用戶資訊
- 回應: {id: int, name: string, email: string, created_at: datetime}
"""

# 2. 在 Cursor 中: @file:design.md 根據這個設計文件生成 FastAPI 實作


# 工作流程 2: 重構現有程式碼

# 1. 選取要重構的程式碼
# 2. Cmd+K 輸入：
"""
重構這段程式碼：
1. 提取重複邏輯到輔助函式
2. 添加適當的錯誤處理
3. 改善變數命名
4. 添加類型提示
5. 確保測試仍然通過
"""


# 工作流程 3: Debug 輔助

# 1. 遇到錯誤時，複製完整的錯誤堆疊
# 2. 在 Chat 中貼上並問：
"""
這個錯誤發生在我的程式碼中：
[錯誤堆疊]

@file:problematic_file.py

請：
1. 解釋錯誤原因
2. 提供修復方案
3. 建議如何避免類似問題
"""
```

## 3. Claude Code CLI 使用

### 基本使用

```bash
# 安裝
npm install -g @anthropic-ai/claude-code

# 基本對話
claude

# 帶有初始提示
claude "解釋這個專案的結構"

# 指定模型
claude --model claude-sonnet-4-20250514

# 繼續上次對話
claude --continue
```

### Claude Code 進階功能

```python
# Claude Code 的 MCP 整合

# 設定 MCP 伺服器 (~/.claude/config.json)
"""
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "/path/to/project"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
"""

# 使用 MCP 功能的提示範例：
# "讀取 src/main.py 的內容並解釋"
# "在 GitHub 上建立一個新的 issue"
# "搜尋專案中所有包含 'TODO' 的檔案"
```

### Claude Code 工作流程

```bash
# 工作流程 1: 程式碼審查
claude "審查 git diff 中的變更，指出問題和改進建議"

# 工作流程 2: 文件生成
claude "為這個專案生成 README.md，包含安裝、使用和 API 文件"

# 工作流程 3: 測試生成
claude "為 src/services/user_service.py 生成完整的單元測試"

# 工作流程 4: 重構
claude "重構 src/legacy/ 目錄中的程式碼，使用現代 Python 最佳實踐"

# 工作流程 5: Debug
claude "分析這個錯誤並提供修復方案：[貼上錯誤]"
```

## 4. Aider - 開源 AI 程式助手

### 安裝與設定

```bash
# 安裝
pip install aider-chat

# 設定 API Key
export ANTHROPIC_API_KEY=your-key
# 或
export OPENAI_API_KEY=your-key

# 基本使用
aider

# 指定模型
aider --model claude-sonnet-4-20250514

# 指定檔案
aider src/main.py src/utils.py
```

### Aider 進階使用

```bash
# 自動提交模式
aider --auto-commits

# 只讀模式（用於理解程式碼）
aider --read src/

# 使用 .aider.conf.yml 設定
cat > .aider.conf.yml << EOF
model: claude-sonnet-4-20250514
auto-commits: true
gitignore: true
EOF

# Aider 指令
/add file.py      # 添加檔案到對話
/drop file.py     # 移除檔案
/ls               # 列出對話中的檔案
/diff             # 顯示變更
/undo             # 撤銷上次變更
/commit           # 提交變更
/clear            # 清除對話歷史
/help             # 顯示幫助
```

### Aider 工作範例

```python
# 範例對話

# User: 添加一個用戶認證系統到這個 Flask 應用程式

# Aider 會：
# 1. 分析現有程式碼結構
# 2. 建議需要的檔案變更
# 3. 顯示 diff 預覽
# 4. 詢問是否應用變更

# 進階提示技巧
"""
請實作用戶認證系統：

要求：
- 使用 JWT tokens
- 包含註冊、登入、登出功能
- 密碼使用 bcrypt 加密
- 添加適當的錯誤處理
- 編寫單元測試

請一步一步實作，每步完成後等待我確認。
"""
```

## 5. AI 輔助程式設計最佳實踐

### 有效的提示工程

```python
# 糟糕的提示
# "寫一個函數處理資料"

# 好的提示
"""
寫一個 Python 函式處理 CSV 銷售資料：

輸入：
- file_path: str - CSV 檔案路徑
- CSV 格式: date,product,quantity,price

輸出：
- dict 包含:
  - total_revenue: float
  - top_products: list[tuple[str, float]] - 前 5 名產品
  - daily_revenue: dict[str, float]

要求：
- 使用 pandas
- 處理空值和格式錯誤
- 添加類型提示
- 包含 docstring 和使用範例
"""

def process_sales_data(file_path: str) -> dict:
    """
    處理銷售 CSV 資料並生成統計報告。

    Args:
        file_path: CSV 檔案路徑

    Returns:
        包含 total_revenue, top_products, daily_revenue 的字典

    Example:
        >>> result = process_sales_data("sales.csv")
        >>> print(f"總收入: {result['total_revenue']}")
    """
    import pandas as pd

    # 讀取資料
    df = pd.read_csv(file_path)

    # 清理資料
    df = df.dropna(subset=['quantity', 'price'])
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna()

    # 計算收入
    df['revenue'] = df['quantity'] * df['price']

    # 總收入
    total_revenue = df['revenue'].sum()

    # 前 5 名產品
    product_revenue = df.groupby('product')['revenue'].sum()
    top_products = list(product_revenue.nlargest(5).items())

    # 每日收入
    daily_revenue = df.groupby('date')['revenue'].sum().to_dict()

    return {
        'total_revenue': total_revenue,
        'top_products': top_products,
        'daily_revenue': daily_revenue
    }
```

### 迭代式開發

```python
# 步驟 1: 先實作基本功能
"""
實作一個基本的待辦事項 API：
- GET /todos - 列出所有待辦
- POST /todos - 建立待辦
使用 FastAPI 和記憶體儲存。
"""

# 步驟 2: 添加驗證
"""
為剛才的 API 添加：
- Pydantic 模型驗證
- 適當的錯誤回應
"""

# 步驟 3: 添加持久化
"""
將儲存從記憶體改為 SQLite：
- 使用 SQLAlchemy
- 添加資料庫遷移
"""

# 步驟 4: 添加認證
"""
添加 JWT 認證：
- 用戶註冊/登入
- 保護需要認證的端點
"""

# 步驟 5: 添加測試
"""
為所有端點添加測試：
- 單元測試
- 整合測試
- 使用 pytest
"""
```

### 程式碼審查與品質

```python
# 使用 AI 進行程式碼審查

review_prompt = """
審查以下程式碼，檢查：

1. 安全問題
   - SQL 注入
   - XSS
   - 敏感資料暴露

2. 效能問題
   - N+1 查詢
   - 不必要的迴圈
   - 記憶體洩漏

3. 程式碼品質
   - 命名清晰度
   - 函數長度
   - 重複程式碼

4. 錯誤處理
   - 異常處理完整性
   - 邊界情況

5. 可維護性
   - 文件完整性
   - 測試覆蓋率建議

請為每個發現的問題提供：
- 問題描述
- 嚴重程度 (高/中/低)
- 修復建議
- 修復後的程式碼範例
"""
```

## 6. 測試 AI 生成的程式碼

### 自動測試生成

```python
# 請 AI 生成測試的提示範例

test_generation_prompt = """
為以下函數生成完整的測試套件：

```python
def validate_email(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

測試需要包含：
1. 正常情況 - 有效的電子郵件
2. 邊界情況 - 最短有效郵件、特殊字元
3. 錯誤情況 - 無效格式、空值、None
4. 效能測試 - 超長輸入

使用 pytest，包含參數化測試。
"""

# 生成的測試
import pytest

class TestValidateEmail:
    """電子郵件驗證測試套件"""

    @pytest.mark.parametrize("email", [
        "user@example.com",
        "user.name@example.com",
        "user+tag@example.com",
        "user@subdomain.example.com",
        "user123@example.co.uk",
    ])
    def test_valid_emails(self, email):
        """測試有效的電子郵件"""
        assert validate_email(email) is True

    @pytest.mark.parametrize("email", [
        "invalid",
        "invalid@",
        "@example.com",
        "user@.com",
        "user@example",
        "",
        " user@example.com",
        "user@example.com ",
    ])
    def test_invalid_emails(self, email):
        """測試無效的電子郵件"""
        assert validate_email(email) is False

    def test_none_input(self):
        """測試 None 輸入"""
        with pytest.raises(TypeError):
            validate_email(None)

    def test_long_input(self):
        """測試超長輸入"""
        long_email = "a" * 1000 + "@example.com"
        # 應該返回 False 或在合理時間內完成
        result = validate_email(long_email)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("email,expected", [
        ("User@Example.COM", True),  # 大小寫
        ("user@例え.com", False),    # Unicode 域名
    ])
    def test_edge_cases(self, email, expected):
        """測試邊界情況"""
        assert validate_email(email) is expected
```

### 驗證 AI 輸出

```python
class AICodeValidator:
    """AI 生成程式碼驗證器"""

    @staticmethod
    def validate_syntax(code: str) -> tuple[bool, str]:
        """驗證語法"""
        import ast
        try:
            ast.parse(code)
            return True, "語法正確"
        except SyntaxError as e:
            return False, f"語法錯誤: {e}"

    @staticmethod
    def check_security(code: str) -> list[str]:
        """檢查安全問題"""
        import re
        issues = []

        dangerous_patterns = [
            (r'\beval\s*\(', "使用 eval() 有安全風險"),
            (r'\bexec\s*\(', "使用 exec() 有安全風險"),
            (r'__import__\s*\(', "動態導入可能有風險"),
            (r'subprocess\..*shell\s*=\s*True', "shell=True 有命令注入風險"),
            (r'os\.system\s*\(', "os.system 有命令注入風險"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(message)

        return issues

    @staticmethod
    def run_tests(code: str, tests: str) -> tuple[bool, str]:
        """執行測試"""
        import subprocess
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # 寫入程式碼
            code_file = os.path.join(tmpdir, "code.py")
            with open(code_file, "w") as f:
                f.write(code)

            # 寫入測試
            test_file = os.path.join(tmpdir, "test_code.py")
            with open(test_file, "w") as f:
                f.write(f"from code import *\n{tests}")

            # 執行測試
            result = subprocess.run(
                ["pytest", test_file, "-v"],
                capture_output=True,
                text=True,
                cwd=tmpdir
            )

            return result.returncode == 0, result.stdout + result.stderr

# 使用範例
validator = AICodeValidator()

ai_generated_code = """
def calculate_factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)
"""

# 驗證
is_valid, msg = validator.validate_syntax(ai_generated_code)
security_issues = validator.check_security(ai_generated_code)
```

## 7. 效能優化建議

### 使用 AI 優化程式碼

```python
optimization_prompt = """
分析這段程式碼的效能並提供優化建議：

```python
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
```

請提供：
1. 時間複雜度分析
2. 空間複雜度分析
3. 優化後的程式碼
4. 優化前後的效能比較
"""

# 優化後的版本
def find_duplicates_optimized(items):
    """
    找出列表中的重複元素。

    時間複雜度: O(n)
    空間複雜度: O(n)
    """
    seen = set()
    duplicates = set()

    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)
```

## 總結

### AI 程式助手選擇指南

```
如果你需要...                    推薦使用
─────────────────────────────────────────────
IDE 內快速補全                   GitHub Copilot
多檔案複雜重構                   Cursor
命令列工作流程                   Claude Code
開源自託管                       Aider
團隊協作                         GitHub Copilot Enterprise
```

### 最佳實踐總結

1. **清晰的提示** - 提供詳細的需求說明
2. **迭代開發** - 分步驟實作，逐步完善
3. **驗證輸出** - 始終審查和測試 AI 生成的程式碼
4. **學習模式** - 理解 AI 生成的程式碼，而非盲目使用
5. **安全意識** - 檢查安全漏洞和敏感資料

## 延伸閱讀

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [Cursor Documentation](https://cursor.sh/docs)
- [Claude Code Guide](https://docs.anthropic.com/claude-code)
- [Aider Documentation](https://aider.chat/)
