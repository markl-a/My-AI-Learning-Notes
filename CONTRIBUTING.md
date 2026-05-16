# 貢獻指南

首先，感謝您考慮為 My AI Learning Notes 做出貢獻！🎉

這份文檔將幫助您了解如何參與這個項目。

## 📋 目錄

1. [行為準則](#行為準則)
2. [如何貢獻](#如何貢獻)
3. [開發環境設置](#開發環境設置)
4. [開發流程](#開發流程)
5. [程式碼規範](#程式碼規範)
6. [提交規範](#提交規範)
7. [Pull Request 流程](#pull-request-流程)
8. [測試指南](#測試指南)
9. [文檔貢獻](#文檔貢獻)
10. [常見問題](#常見問題)

---

## 📜 行為準則

### 我們的承諾

為了營造開放和友好的環境，我們承諾：

- 使用友善和包容的語言
- 尊重不同的觀點和經驗
- 優雅地接受建設性批評
- 關注什麼對社區最有利
- 對其他社區成員表示同理心

### 不可接受的行為

- 使用性化的語言或圖像
- 侮辱性/貶損性評論和人身攻擊
- 公開或私下騷擾
- 未經許可發布他人的私人資訊
- 其他在專業環境中被認為不適當的行為

---

## 🤝 如何貢獻

### 貢獻類型

我們歡迎以下類型的貢獻：

#### 1. 報告 Bug 🐛
- 使用 GitHub Issues
- 提供詳細的重現步驟
- 包含系統資訊（OS、Python 版本等）
- 附上錯誤資訊和日誌

#### 2. 建議功能 💡
- 清楚描述功能需求
- 說明使用場景
- 考慮實現的可行性

#### 3. 改進文檔 📝
- 修正錯別字
- 改善說明清晰度
- 添加範例
- 翻譯文檔

#### 4. 提交程式碼 💻
- 修復 Bug
- 實現新功能
- 優化性能
- 重構程式碼

#### 5. 分享學習筆記 📚
- 添加新的學習材料
- 分享實戰經驗
- 貢獻 Jupyter Notebooks
- 整理學習資源

---

## 🛠️ 開發環境設置

### 1. Fork 項目

點擊 GitHub 頁面右上角的 "Fork" 按鈕

### 2. Clone 到本地

```bash
git clone https://github.com/markl-a/My-AI-Learning-Notes.git
cd My-AI-Learning-Notes
```

### 3. 添加上游倉庫

```bash
git remote add upstream https://github.com/markl-a/My-AI-Learning-Notes.git
```

### 4. 建立虛擬環境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 或使用 conda
conda create -n ai-learning python=3.11
conda activate ai-learning
```

### 5. 安裝依賴

```bash
# 安裝開發依賴
pip install -r requirements-full.txt -r requirements-dev.txt

# 或使用 pyproject.toml
pip install -e ".[dev]"
```

### 6. 設置 Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## 🔄 開發流程

### 1. 同步上游更新

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. 建立功能分支

```bash
# 功能分支命名規範：
# - feature/功能名稱 (新功能)
# - fix/問題描述 (修復 Bug)
# - docs/文檔主題 (文檔更新)
# - refactor/重構範圍 (程式碼重構)

git checkout -b feature/your-feature-name
```

### 3. 開發你的更改

```bash
# 編寫程式碼
# 添加測試
# 更新文檔
```

### 4. 提交更改

```bash
git add .
git commit -m "type: 簡短描述"
```

### 5. 推送到 Fork

```bash
git push origin feature/your-feature-name
```

### 6. 建立 Pull Request

在 GitHub 上建立 PR，詳見 [Pull Request 流程](#pull-request-流程)

---

## 📏 程式碼規範

### Python 程式碼風格

我們遵循 PEP 8 和以下工具的配置：

#### Black（程式碼格式化）

```python
# 配置：pyproject.toml
# 行長度：100
# 目標版本：Python 3.9+

# 運行
black .

# 檢查
black --check .
```

#### Ruff（快速 Linter）

```python
# 運行
ruff check .

# 自動修復
ruff check --fix .
```

#### MyPy（類型檢查）

```python
# 運行
mypy . --ignore-missing-imports
```

### 程式碼風格示例

```python
"""模塊文檔字串

詳細描述模塊功能。
"""

from typing import List, Dict, Optional
import numpy as np
from langchain_openai import OpenAI


class ExampleClass:
    """類文檔字串

    Args:
        param1: 參數描述
        param2: 參數描述

    Examples:
        >>> obj = ExampleClass("value")
        >>> obj.method()
        'result'
    """

    def __init__(self, param1: str, param2: Optional[int] = None):
        """初始化方法"""
        self.param1 = param1
        self.param2 = param2 or 10

    def method(self, input_data: List[str]) -> Dict[str, any]:
        """方法文檔字串

        Args:
            input_data: 輸入資料列表

        Returns:
            處理結果字典

        Raises:
            ValueError: 當輸入無效時
        """
        if not input_data:
            raise ValueError("輸入不能為空")

        result = {"data": input_data, "count": len(input_data)}
        return result


def example_function(param1: str, param2: int = 5) -> str:
    """函數文檔字串

    Args:
        param1: 第一個參數
        param2: 第二個參數，預設為 5

    Returns:
        處理後的字串
    """
    return f"{param1} - {param2}"
```

### Jupyter Notebook 規範

```python
# 每個 Notebook 應包含：
# 1. 標題和描述
# 2. 環境設置（imports）
# 3. 清晰的章節劃分
# 4. 充足的註釋
# 5. 結論和總結

# ===== Notebook 標題 =====
# 描述：這個 Notebook 的用途
# 作者：Your Name
# 日期：2024-01-01
# =========================

# 1. 環境設置
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. 資料加載
# ...

# 3. 資料處理
# ...

# 4. 可視化
# ...

# 5. 結論
# ...
```

---

## 📝 提交規範

### Commit Message 格式

我們使用 Conventional Commits 規範：

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type 類型

- `feat`: 新功能
- `fix`: Bug 修復
- `docs`: 文檔更新
- `style`: 程式碼格式（不影響功能）
- `refactor`: 程式碼重構
- `perf`: 性能優化
- `test`: 測試相關
- `chore`: 構建/工具相關
- `ci`: CI 配置

#### 範例

```bash
# 簡單格式
feat: 添加 RAG 多模態支援

# 完整格式
feat(rag): 添加圖像檢索功能

實現了基於 CLIP 的圖像嵌入和檢索功能，
支持文字到圖像的跨模態搜索。

Closes #123
```

### 好的 Commit Message

✅ **好例子：**

```
feat(llm): 添加 Anthropic Claude API 支援
fix(rag): 修復向量資料庫連接超時問題
docs(readme): 更新安裝指南
test(agent): 添加 ReAct 模式單元測試
```

❌ **壞例子：**

```
update
fix bug
改了一些東西
WIP
```

---

## 🔀 Pull Request 流程

### 1. PR 標題

使用與 Commit Message 相同的格式：

```
feat: 添加新功能簡短描述
fix: 修復具體問題
docs: 更新某某文檔
```

### 2. PR 描述模板

```markdown
## 📝 變更描述

簡要描述這個 PR 做了什麼。

## 🎯 相關 Issue

Closes #123
Fixes #456

## ✅ 變更類型

- [ ] 新功能 (feat)
- [ ] Bug 修復 (fix)
- [ ] 文檔更新 (docs)
- [ ] 程式碼重構 (refactor)
- [ ] 性能優化 (perf)
- [ ] 測試 (test)

## 🧪 測試

- [ ] 我已經測試了這些更改
- [ ] 我添加了新的測試
- [ ] 所有現有測試通過
- [ ] 程式碼符合項目的程式碼規範

## 📸 截圖（如適用）

添加截圖幫助解釋您的更改。

## 📚 文檔

- [ ] 我已更新相關文檔
- [ ] 我添加了程式碼註釋
- [ ] 我更新了 README（如需要）

## 💡 其他資訊

添加任何其他相關資訊。
```

### 3. PR 審查流程

1. **自動檢查**
   - CI/CD 自動運行
   - 程式碼品質檢查
   - 測試執行
   - 安全掃描

2. **程式碼審查**
   - 至少一位維護者審查
   - 回應審查意見
   - 進行必要的修改

3. **合併**
   - 所有檢查通過
   - 審查批准
   - 無衝突
   - Squash merge 到主分支

---

## 🧪 測試指南

### 運行測試

```bash
# 運行所有測試
pytest

# 運行特定測試文件
pytest tests/test_rag.py

# 運行特定測試函數
pytest tests/test_rag.py::test_vector_search

# 帶覆蓋率報告
pytest --cov=. --cov-report=html

# 並行測試
pytest -n auto
```

### 編寫測試

```python
# tests/test_example.py
import pytest
from your_module import your_function


class TestYourFunction:
    """測試 your_function 函數"""

    def test_basic_functionality(self):
        """測試基本功能"""
        result = your_function("input")
        assert result == "expected"

    def test_edge_case(self):
        """測試邊界情況"""
        result = your_function("")
        assert result == ""

    def test_error_handling(self):
        """測試錯誤處理"""
        with pytest.raises(ValueError):
            your_function(None)

    @pytest.mark.parametrize("input,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
        ("test3", "result3"),
    ])
    def test_multiple_inputs(self, input, expected):
        """參數化測試"""
        result = your_function(input)
        assert result == expected


# Fixtures
@pytest.fixture
def sample_data():
    """提供測試資料"""
    return {"key": "value"}


def test_with_fixture(sample_data):
    """使用 fixture 的測試"""
    assert sample_data["key"] == "value"
```

---

## 📖 文檔貢獻

### 文檔類型

1. **README 文件**
   - 項目概述
   - 快速開始
   - 功能介紹

2. **學習筆記**
   - Markdown 文件
   - Jupyter Notebooks
   - 程式碼示例

3. **API 文檔**
   - 函數/類文檔字串
   - 自動生成的 API 文檔

4. **指南和教程**
   - 安裝指南
   - 使用教程
   - 最佳實踐

### 文檔風格

```markdown
# 標題使用 ATX 風格（#）

## 二級標題

### 三級標題

- 列表項使用 `-`
- 保持一致的縮進
- 使用空行分隔段落

**粗體** 用於強調
*斜體* 用於術語
`程式碼` 用於程式碼片段

​```python
# 程式碼塊使用三個反引號
def example():
    pass
​```

> 引用使用 >

[鏈接文字](URL)

![圖片alt](圖片URL)
```

---

## ❓ 常見問題

### Q1: 我應該從哪裡開始？

**A:** 查看 [Good First Issues](https://github.com/markl-a/My-AI-Learning-Notes/labels/good%20first%20issue) 標籤，這些是適合新貢獻者的任務。

### Q2: 我發現了一個 Bug，但不知道如何修復

**A:** 沒關係！建立一個 Issue 詳細描述問題即可。提供重現步驟和錯誤資訊會很有幫助。

### Q3: 我可以貢獻非程式碼內容嗎？

**A:** 當然可以！文檔改進、學習筆記分享、問題回答都是非常有價值的貢獻。

### Q4: PR 審查需要多長時間？

**A:** 通常在 2-7 天內。如果超過一週沒有回應，可以在 PR 中留言提醒。

### Q5: 我的 PR 被拒絕了怎麼辦？

**A:** 不要灰心！閱讀審查意見，理解拒絕原因，可以修改後重新提交，或者開啟討論了解更多細節。

### Q6: 我需要簽署 CLA 嗎？

**A:** 目前不需要。我們使用 MIT 許可證，貢獻即表示您同意以該許可證發布您的程式碼。

### Q7: 如何同步我的 Fork 與上游？

**A:**
```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## 🎁 致謝

感謝所有貢獻者！您的努力讓這個項目變得更好。

### 貢獻者列表

查看 [Contributors](https://github.com/markl-a/My-AI-Learning-Notes/graphs/contributors)

---

## 📧 聯繫方式

- **Issues**: [GitHub Issues](https://github.com/markl-a/My-AI-Learning-Notes/issues)
- **Discussions**: [GitHub Discussions](https://github.com/markl-a/My-AI-Learning-Notes/discussions)
- **Email**: your.email@example.com

---

## 📄 許可證

通過貢獻本項目，您同意您的貢獻將以 [MIT License](LICENSE) 授權。

---

**再次感謝您的貢獻！🙏**

讓我們一起打造最好的 AI 學習資源！🚀
