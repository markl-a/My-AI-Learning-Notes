# 依賴版本更新報告

> **生成日期**: 2026-01-15
> **檢查範圍**: pyproject.toml, requirements*.txt

---

## 當前版本 vs 最新建議版本

### 核心依賴

| 套件 | 當前版本 | 建議更新 | 狀態 | 備註 |
|------|---------|---------|------|------|
| numpy | >=1.24.0 | >=2.0.0 | 🟡 可選 | NumPy 2.0 有 breaking changes |
| pandas | >=2.0.0 | >=2.2.0 | ✅ 建議 | 性能改進 |
| scipy | >=1.10.0 | >=1.14.0 | ✅ 建議 | 新功能 |
| matplotlib | >=3.7.0 | >=3.9.0 | ✅ 建議 | |
| pytest | >=7.4.0 | >=8.0.0 | ✅ 建議 | 新功能 |

### 機器學習

| 套件 | 當前版本 | 建議更新 | 狀態 |
|------|---------|---------|------|
| scikit-learn | >=1.3.0 | >=1.5.0 | ✅ 建議 |
| xgboost | >=2.0.0 | >=2.1.0 | ✅ OK |
| lightgbm | >=4.0.0 | >=4.5.0 | ✅ 建議 |
| mlflow | >=2.8.0 | >=2.18.0 | ✅ 建議 |

### 深度學習

| 套件 | 當前版本 | 建議更新 | 狀態 | 備註 |
|------|---------|---------|------|------|
| torch | >=2.5.0 | >=2.5.1 | ✅ OK | 最新穩定版 |
| tensorflow | >=2.20.0 | >=2.18.0 | ⚠️ 檢查 | 版本號需確認 |
| transformers | >=4.45.0 | >=4.47.0 | ✅ 建議 | 新模型支持 |
| accelerate | >=1.0.0 | >=1.2.0 | ✅ 建議 | |

### LLM 應用

| 套件 | 當前版本 | 建議更新 | 狀態 | 備註 |
|------|---------|---------|------|------|
| openai | >=1.50.0 | >=1.57.0 | ✅ 建議 | 新 API 功能 |
| anthropic | >=0.39.0 | >=0.40.0 | ✅ 建議 | Claude API 更新 |
| langchain | >=0.3.0 | >=0.3.11 | ✅ 建議 | Bug 修復 |
| langchain-openai | >=0.2.0 | >=0.2.12 | ✅ 建議 | |
| langgraph | >=0.2.0 | >=0.2.56 | ✅ 建議 | 新功能 |
| chromadb | >=0.5.0 | >=0.5.23 | ✅ 建議 | |
| gradio | >=5.0.0 | >=5.9.0 | ✅ 建議 | UI 改進 |
| streamlit | >=1.39.0 | >=1.41.0 | ✅ 建議 | |

### 開發工具

| 套件 | 當前版本 | 建議更新 | 狀態 |
|------|---------|---------|------|
| pytest | >=7.4.0 | >=8.0.0 | ✅ 建議 |
| pytest-cov | >=4.1.0 | >=6.0.0 | ✅ 建議 |
| ruff | >=0.1.0 | >=0.8.0 | ✅ 建議 |
| black | >=23.0.0 | >=24.10.0 | ✅ 建議 |
| mypy | >=1.6.0 | >=1.13.0 | ✅ 建議 |

---

## 建議的更新操作

### 優先級 1: 安全性更新 (立即執行)

```bash
pip install --upgrade \
    openai>=1.57.0 \
    anthropic>=0.40.0 \
    requests>=2.32.0
```

### 優先級 2: 功能性更新 (本週)

```bash
pip install --upgrade \
    langchain>=0.3.11 \
    langchain-openai>=0.2.12 \
    langgraph>=0.2.56 \
    transformers>=4.47.0
```

### 優先級 3: 開發工具更新 (本月)

```bash
pip install --upgrade \
    pytest>=8.0.0 \
    ruff>=0.8.0 \
    mypy>=1.13.0
```

---

## 新增建議套件

考慮新增以下套件以增強專案功能：

### 推理優化
```toml
# pyproject.toml [project.optional-dependencies.inference]
vllm = ">=0.6.0"           # 高效推理引擎
text-generation = ">=0.7.0" # TGI 客戶端
```

### MCP 協議
```toml
# pyproject.toml [project.optional-dependencies.mcp]
mcp = ">=1.0.0"           # MCP SDK
```

### 監控與追蹤
```toml
# pyproject.toml [project.optional-dependencies.monitoring]
langfuse = ">=2.0.0"      # LLM 追蹤
opentelemetry-api = ">=1.27.0"
opentelemetry-sdk = ">=1.27.0"
```

### 進階 RAG
```toml
# pyproject.toml [project.optional-dependencies.rag-advanced]
llama-index = ">=0.11.0"  # 替代 RAG 框架
cohere = ">=5.0.0"        # Reranking
```

---

## 注意事項

### Breaking Changes 警告

1. **NumPy 2.0**
   - 許多函式簽名變更
   - 建議先在測試環境驗證

2. **LangChain 0.3**
   - 已穩定，但與 0.2 有 API 差異
   - 確保使用正確的 import 路徑

3. **Transformers 4.46+**
   - 部分舊模型類被棄用
   - 檢查 deprecation warnings

### 相容性矩陣

| Python | PyTorch | TensorFlow | 建議 |
|--------|---------|------------|------|
| 3.9 | 2.5.x | 2.17.x | ✅ |
| 3.10 | 2.5.x | 2.18.x | ✅ 推薦 |
| 3.11 | 2.5.x | 2.18.x | ✅ 推薦 |
| 3.12 | 2.5.x | 2.18.x | ✅ |
| 3.13 | 待測試 | 待測試 | ⚠️ |

---

## 自動化更新建議

### 使用 Dependabot

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      python-packages:
        patterns:
          - "*"
```

### 使用 pre-commit 自動更新

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/python-poetry/poetry
    rev: "1.8.0"
    hooks:
      - id: poetry-check
      - id: poetry-lock
```

---

*報告自動生成，建議定期執行依賴審查*
