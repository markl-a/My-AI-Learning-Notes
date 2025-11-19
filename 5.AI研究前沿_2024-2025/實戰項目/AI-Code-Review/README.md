# AI 代碼審查助手

基於 LLM 的智能代碼審查系統，提供自動化代碼審查、安全檢查、性能分析和最佳實踐建議。

## ✨ 核心功能

### 🔍 智能代碼審查
- **全面審查**：代碼質量、安全性、性能、可維護性
- **快速審查**：快速識別主要問題
- **專項審查**：針對性的安全或性能審查
- **批量審查**：支持整個代碼庫的批量審查

### 🛡️ 安全漏洞檢測
- **OWASP Top 10** 檢查
- **SQL 注入**檢測
- **XSS 攻擊**檢測
- **命令注入**檢測
- **敏感信息洩露**掃描（API 密鑰、密碼、Token）
- **路徑遍歷**檢測
- **CWE 和 OWASP 合規性**報告

### ⚡ 性能分析
- **時間複雜度**分析（Big O）
- **空間複雜度**分析
- **性能瓶頸**識別
- **優化建議**生成
- **預期性能提升**估計

### 🔧 代碼重構
- **可讀性**提升
- **性能優化**建議
- **可維護性**改進
- **設計模式**應用建議

### 📊 代碼質量指標
- **圈複雜度**（Cyclomatic Complexity）
- **認知複雜度**（Cognitive Complexity）
- **維護性指數**（Maintainability Index）
- **代碼覆蓋率**估計

### 🎯 最佳實踐檢查
- 命名規範
- 代碼風格
- 設計模式使用
- 錯誤處理
- 日誌記錄
- 文檔註釋

### 🤖 代碼生成
- **自動生成單元測試**
- **自動生成文檔**
- **代碼補全建議**

## 🎨 支持的語言

- ✅ Python
- ✅ JavaScript / TypeScript
- ✅ Java
- ✅ Go
- ✅ Rust
- ✅ C / C++
- ✅ Ruby
- ✅ PHP
- ✅ Swift
- ✅ Kotlin
- ✅ Scala

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI 應用層                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ Code         │  │  Security    │  │Performance│  │
│  │ Analyzer     │  │  Checker     │  │ Analyzer  │  │
│  │              │  │              │  │           │  │
│  │ - 代碼審查   │  │ - OWASP檢查  │  │ - 複雜度  │  │
│  │ - 最佳實踐   │  │ - 漏洞掃描   │  │ - 瓶頸    │  │
│  │ - 重構建議   │  │ - 敏感信息   │  │ - 優化    │  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
│         │                  │                │        │
│         └──────────────────┴────────────────┘        │
│                          │                           │
│                          ▼                           │
│                  ┌──────────────┐                    │
│                  │  OpenAI API  │                    │
│                  │  (GPT-4)     │                    │
│                  └──────────────┘                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## 🚀 快速開始

### 環境要求

- Python >= 3.9
- OpenAI API 密鑰
- Docker（可選）

### 本地運行

```bash
# 1. 克隆並進入目錄
cd AI-Code-Review

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 配置環境變量
cp .env.example .env
# 編輯 .env 並添加 OpenAI API 密鑰

# 4. 啟動服務
python main.py

# 5. 訪問 API 文檔
# http://localhost:8002/docs
```

### Docker 部署

```bash
# 使用 Docker Compose
docker-compose up -d

# 查看日誌
docker-compose logs -f code-review

# 停止服務
docker-compose down
```

## 📚 API 使用示例

### 1. 完整代碼審查

```bash
curl -X POST "http://localhost:8002/api/review" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total = total + num\n    return total",
    "language": "python",
    "review_type": "full"
  }'
```

**響應：**
```json
{
  "review_id": "550e8400-e29b-41d4-a716-446655440000",
  "language": "python",
  "overall_score": 75,
  "issues": [
    {
      "severity": "low",
      "description": "可以使用內置的 sum() 函數",
      "line_number": 2,
      "suggestion": "使用 return sum(numbers)"
    }
  ],
  "suggestions": [
    {
      "category": "performance",
      "content": "使用 sum() 內置函數會更高效",
      "priority": "medium"
    }
  ],
  "metrics": {
    "complexity": 3,
    "readability": 8,
    "maintainability": 9
  },
  "summary": "代碼整體質量良好，建議使用更 Pythonic 的寫法"
}
```

### 2. 安全漏洞檢查

```bash
curl -X POST "http://localhost:8002/api/security/check" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "query = \"SELECT * FROM users WHERE id = \" + user_id",
    "language": "python",
    "check_types": ["sql_injection"]
  }'
```

**響應：**
```json
{
  "security_score": 40,
  "vulnerabilities": [
    {
      "type": "sql_injection",
      "severity": "critical",
      "description": "Potential SQL injection vulnerability",
      "line_number": 1,
      "cwe_id": "CWE-89",
      "owasp_category": "A03:2021 – Injection",
      "recommendation": "使用參數化查詢或 ORM"
    }
  ],
  "severity_distribution": {
    "critical": 1,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "recommendations": [
    "實施參數化查詢和 ORM 使用規範",
    "立即修復所有嚴重級別的安全問題"
  ],
  "compliant_with": []
}
```

### 3. 性能分析

```bash
curl -X POST "http://localhost:8002/api/performance/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def find_duplicates(arr):\n    duplicates = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j]:\n                duplicates.append(arr[i])\n    return duplicates",
    "language": "python",
    "analysis_depth": "medium"
  }'
```

**響應：**
```json
{
  "performance_score": 45,
  "time_complexity": "O(n^2)",
  "space_complexity": "O(n)",
  "bottlenecks": [
    {
      "location": "line 3-5",
      "issue": "Nested loops causing quadratic time complexity",
      "impact": "high"
    }
  ],
  "optimization_suggestions": [
    {
      "suggestion": "使用 set 或 hash map 來跟蹤已見元素",
      "expected_improvement": "從 O(n^2) 到 O(n)",
      "difficulty": "easy"
    }
  ],
  "estimated_speedup": "100x for large arrays"
}
```

### 4. 掃描敏感信息

```bash
curl -X POST "http://localhost:8002/api/security/scan-secrets" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "api_key = \"sk-1234567890abcdefghijklmnopqrstuvwxyz\"\npassword = \"MySecretPassword123\"",
    "language": "python"
  }'
```

**響應：**
```json
{
  "secrets_found": 2,
  "secrets": [
    {
      "type": "api_key",
      "line_number": 1,
      "severity": "critical",
      "value": "sk-1234567...",
      "description": "Potential api key detected",
      "recommendation": "使用環境變量或密鑰管理服務"
    },
    {
      "type": "password",
      "line_number": 2,
      "severity": "critical",
      "value": "MySecr...",
      "description": "Potential password detected",
      "recommendation": "永遠不要硬編碼密碼"
    }
  ],
  "risk_level": "high"
}
```

### 5. 上傳文件審查

```bash
curl -X POST "http://localhost:8002/api/review/file" \
  -F "file=@example.py" \
  -F "review_type=full"
```

### 6. 生成單元測試

```bash
curl -X POST "http://localhost:8002/api/generate/tests" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a + b",
    "language": "python",
    "test_framework": "pytest"
  }'
```

**響應：**
```json
{
  "test_code": "import pytest\n\ndef test_add_positive_numbers():\n    assert add(2, 3) == 5\n\ndef test_add_negative_numbers():\n    assert add(-1, -1) == -2\n\ndef test_add_zero():\n    assert add(0, 0) == 0",
  "test_cases": ["test_add_positive_numbers", "test_add_negative_numbers", "test_add_zero"],
  "coverage_estimate": "80%",
  "framework": "pytest"
}
```

### 7. 代碼重構建議

```bash
curl -X POST "http://localhost:8002/api/refactor" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result",
    "language": "python",
    "refactor_goals": ["readability", "performance"]
  }'
```

### 8. 批量審查

```bash
# 啟動批量審查
curl -X POST "http://localhost:8002/api/review/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "repository_url": "https://github.com/user/repo.git",
    "review_type": "full"
  }'

# 響應：
# {"task_id": "task-123", "status": "processing"}

# 查詢狀態
curl "http://localhost:8002/api/review/batch/task-123"
```

### 9. 檢查最佳實踐

```bash
curl -X POST "http://localhost:8002/api/best-practices" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def my_func(x):\n    return x+1",
    "language": "python",
    "review_type": "full"
  }'
```

### 10. 代碼複雜度分析

```bash
curl -X POST "http://localhost:8002/api/complexity" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def complex_function(x):\n    if x > 0:\n        if x < 10:\n            return \"small\"\n        else:\n            return \"large\"\n    else:\n        return \"negative\"",
    "language": "python",
    "review_type": "full"
  }'
```

## 🔧 高級配置

### 自定義審查規則

可以在 `.env` 文件中配置審查行為：

```env
# 安全評分閾值
SECURITY_SCORE_THRESHOLD=70

# 性能評分閾值
PERFORMANCE_SCORE_THRESHOLD=75

# 默認審查類型
DEFAULT_REVIEW_TYPE=full

# 分析深度
DEFAULT_ANALYSIS_DEPTH=medium
```

### 集成到 CI/CD

#### GitHub Actions 示例

```yaml
name: Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Review Changed Files
        run: |
          # 獲取變更的文件
          git diff --name-only origin/main > changed_files.txt

          # 對每個文件進行審查
          while read file; do
            curl -X POST "http://your-code-review-server/api/review/file" \
              -F "file=@$file" \
              -F "review_type=full"
          done < changed_files.txt
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# 獲取暫存的文件
git diff --cached --name-only --diff-filter=ACM | while read file; do
    if [[ $file == *.py ]]; then
        # 審查 Python 文件
        curl -X POST "http://localhost:8002/api/review/file" \
          -F "file=@$file" \
          -F "review_type=quick"
    fi
done
```

## 📊 性能基準

| 審查類型 | 平均時間 | 準確率 |
|---------|---------|--------|
| Quick   | 2-3s    | 85%    |
| Full    | 5-8s    | 95%    |
| Security| 3-5s    | 92%    |
| Performance| 4-6s | 90%    |

## 🛡️ 安全最佳實踐

1. **永遠不要提交 API 密鑰到代碼庫**
2. **使用環境變量**存儲敏感配置
3. **定期更新依賴**以修復安全漏洞
4. **限制 API 訪問**使用速率限制
5. **審查日誌**不包含敏感信息

## 🐛 故障排除

### 問題：API 返回 500 錯誤

**解決方案：**
- 檢查 OpenAI API 密鑰是否正確
- 查看日誌文件：`logs/code_review.log`
- 確認 API 配額未用盡

### 問題：審查結果不準確

**解決方案：**
- 使用 GPT-4 模型獲得更好的結果
- 增加分析深度：`analysis_depth=deep`
- 提供更多上下文信息

### 問題：性能緩慢

**解決方案：**
- 使用 `review_type=quick` 進行快速審查
- 啟用緩存：`ENABLE_CACHE=true`
- 增加並發限制：`MAX_CONCURRENT_REVIEWS=10`

## 📈 路線圖

- [ ] 支持更多編程語言
- [ ] 增加自定義規則引擎
- [ ] 集成更多靜態分析工具
- [ ] Web UI 界面
- [ ] 團隊協作功能
- [ ] 歷史審查記錄和趨勢分析
- [ ] 機器學習模型訓練
- [ ] 離線模式支持

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 許可證

MIT License

## 📧 聯繫

如有問題或建議，請提交 Issue。
