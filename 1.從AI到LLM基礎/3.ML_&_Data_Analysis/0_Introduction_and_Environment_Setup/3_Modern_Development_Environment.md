# 現代機器學習開發環境設置指南 (2024-2025)

本指南介紹如何設置現代化的機器學習開發環境，包括容器化、依賴管理、開發工具等最佳實踐。

## 目錄
1. [容器化環境 (Docker)](#1-容器化環境-docker)
2. [現代依賴管理工具](#2-現代依賴管理工具)
3. [開發工具與IDE](#3-開發工具與ide)
4. [AI 輔助開發工具](#4-ai-輔助開發工具)
5. [完整環境設置範例](#5-完整環境設置範例)

---

## 1. 容器化環境 (Docker)

### 1.1 為什麼使用 Docker？

**優勢**：
- ✅ **環境一致性**: "在我電腦上可以運行" 問題的終結者
- ✅ **隔離性**: 不同項目的依賴互不干擾
- ✅ **可重現性**: 確保團隊成員和生產環境一致
- ✅ **易於部署**: 容器可以輕鬆部署到任何支持 Docker 的平台
- ✅ **資源效率**: 比虛擬機更輕量

### 1.2 Docker 基礎概念

```
Image (鏡像)          Container (容器)
   📦       ------>      🏃
  藍圖                運行中的實例
```

### 1.3 安裝 Docker

**Linux**:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 添加當前用戶到 docker 組
sudo usermod -aG docker $USER
```

**macOS/Windows**:
- 下載並安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop)

**驗證安裝**:
```bash
docker --version
docker run hello-world
```

### 1.4 ML 項目 Dockerfile 範例

**基礎 Python ML 環境**:
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製項目文件
COPY . .

# 暴露 Jupyter 端口
EXPOSE 8888

# 啟動 Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

**包含 GPU 支持的 TensorFlow/PyTorch 環境**:
```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 安裝 Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安裝深度學習框架
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    tensorflow[and-cuda]

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### 1.5 Docker Compose 多服務編排

```yaml
# docker-compose.yml
version: '3.8'

services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
    networks:
      - ml-network

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000
    networks:
      - ml-network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpassword
      POSTGRES_DB: mldb
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge

volumes:
  postgres-data:
```

**啟動所有服務**:
```bash
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止所有服務
docker-compose down
```

### 1.6 Docker 最佳實踐

1. **使用 .dockerignore**:
```
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
*.md
.DS_Store
*.ipynb_checkpoints
data/raw/*
models/*.h5
```

2. **多階段構建減小鏡像大小**:
```dockerfile
# 構建階段
FROM python:3.11 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 運行階段
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

3. **使用健康檢查**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8888/ || exit 1
```

---

## 2. 現代依賴管理工具

### 2.1 Poetry - Python 依賴管理

**特點**:
- 🎯 自動解決依賴衝突
- 📦 統一管理依賴和虛擬環境
- 🔒 鎖定依賴版本 (poetry.lock)
- 📝 簡化項目打包和發布

**安裝**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**初始化項目**:
```bash
# 建立新項目
poetry new my-ml-project

# 在現有項目中初始化
poetry init
```

**pyproject.toml 範例**:
```toml
[tool.poetry]
name = "my-ml-project"
version = "0.1.0"
description = "機器學習項目範例"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
numpy = "^1.26.0"
pandas = "^2.1.0"
scikit-learn = "^1.3.0"
polars = "^0.19.0"
plotly = "^5.17.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.9.0"
ruff = "^0.1.0"
jupyterlab = "^4.0.0"

[tool.poetry.group.ml.dependencies]
torch = "^2.1.0"
transformers = "^4.35.0"
optuna = "^3.4.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**常用命令**:
```bash
# 安裝依賴
poetry install

# 添加新依賴
poetry add pandas numpy

# 添加開發依賴
poetry add --group dev pytest black

# 更新依賴
poetry update

# 運行腳本
poetry run python train.py
poetry run jupyter lab

# 進入虛擬環境
poetry shell

# 導出 requirements.txt
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### 2.2 uv - 極速 Python 包管理器

**特點**:
- ⚡ 用 Rust 編寫，比 pip 快 10-100 倍
- 🎯 完全兼容 pip 接口
- 💾 全局快取，節省磁盤空間
- 🔄 更好的依賴解析

**安裝**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**使用**:
```bash
# 替代 pip install
uv pip install pandas numpy scikit-learn

# 從 requirements.txt 安裝
uv pip install -r requirements.txt

# 建立虛擬環境
uv venv

# 編譯依賴
uv pip compile requirements.in -o requirements.txt

# 同步依賴（確保環境完全匹配 requirements.txt）
uv pip sync requirements.txt
```

**性能比較**:
```
包管理器         安裝 100 個包耗時
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip              ~2 分鐘
poetry           ~1.5 分鐘
uv               ~5 秒 ⚡
```

### 2.3 依賴管理最佳實踐

**1. 使用鎖定文件**:
```bash
# Poetry
poetry.lock

# pip with pip-tools
pip-compile requirements.in
# 生成 requirements.txt
```

**2. 分離不同環境的依賴**:
```
requirements/
├── base.txt          # 基礎依賴
├── dev.txt           # 開發依賴
├── prod.txt          # 生產依賴
└── test.txt          # 測試依賴
```

**3. 固定主要依賴版本**:
```txt
# requirements.in
numpy>=1.26.0,<2.0.0
pandas>=2.1.0,<3.0.0
scikit-learn~=1.3.0    # ~= 表示兼容版本
```

---

## 3. 開發工具與 IDE

### 3.1 JupyterLab - 現代 Notebook 環境

**安裝與配置**:
```bash
pip install jupyterlab

# 安裝有用的擴展
pip install jupyterlab-git jupyterlab-lsp python-lsp-server \
    jupyterlab-code-formatter black isort
```

**JupyterLab 配置 (~/.jupyter/jupyter_lab_config.py)**:
```python
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.token = ''  # 僅開發環境！
c.ServerApp.password = ''
c.ServerApp.allow_root = True

# 設置工作目錄
c.ServerApp.root_dir = '/path/to/your/notebooks'
```

**有用的 JupyterLab 快捷鍵**:
```
Ctrl/Cmd + Shift + C    打開命令面板
Ctrl/Cmd + Shift + L    切換行號
Ctrl/Cmd + B            切換側邊欄
Shift + M               合併單元格
A / B                   在上/下方插入單元格
```

### 3.2 VS Code - 通用程式碼編輯器

**必備擴展**:
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "ms-toolsai.vscode-jupyter-cell-tags",
    "charliermarsh.ruff",
    "ms-azuretools.vscode-docker",
    "eamodio.gitlens",
    "GitHub.copilot",
    "ms-python.black-formatter"
  ]
}
```

**settings.json 配置**:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

### 3.3 Cursor - AI 輔助開發工具

**特點**:
- 🤖 基於 VS Code，集成 AI 程式碼補全
- 💬 與 AI 對話式編程
- 🔍 AI 驅動的程式碼搜索和重構

**最佳實踐**:
1. 使用 `Cmd/Ctrl + K` 觸發 AI 編輯
2. 使用 `Cmd/Ctrl + L` 打開 AI 聊天
3. 選中程式碼後詢問解釋或優化建議

---

## 4. AI 輔助開發工具

### 4.1 GitHub Copilot

**功能**:
- 📝 實時程式碼建議
- 💡 根據註釋生成程式碼
- 🔄 程式碼重構建議

**最佳實踐**:
```python
# 技巧1: 寫清晰的註釋，讓 Copilot 生成程式碼
# 使用 pandas 讀取 CSV，處理缺失值，並進行標準化
# Copilot 會建議完整的實現

# 技巧2: 開始寫函式簽名
def train_xgboost_model(X_train, y_train, params=None):
    """
    訓練 XGBoost 模型並返回訓練好的模型

    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        params: 模型參數
    """
    # Copilot 會建議完整實現
```

### 4.2 Codeium - 免費 AI 編碼助手

**特點**:
- ✅ 完全免費
- 🚀 支持 70+ 編程語言
- 🔌 支持主流 IDE

**安裝**:
- VS Code: 搜索 "Codeium" 擴展
- JupyterLab: `pip install jupyter-codeium`

### 4.3 ChatGPT / Claude for Code

**使用場景**:
1. **程式碼生成**: 描述需求，生成初始程式碼
2. **程式碼審查**: 粘貼程式碼，要求審查和改進
3. **調試**: 粘貼錯誤資訊，獲取解決方案
4. **學習**: 詢問概念解釋和最佳實踐

**提示詞範例**:
```
"我需要一個 Python 函式來處理不平衡資料集，使用 SMOTE 進行過採樣，
並包含交叉驗證。請包含錯誤處理和詳細註釋。"

"審查以下程式碼並提出改進建議：[粘貼程式碼]"

"解釋 XGBoost 的 'learning_rate' 參數如何影響模型性能，
並提供調優建議。"
```

---

## 5. 完整環境設置範例

### 5.1 快速開始腳本

```bash
#!/bin/bash
# setup_ml_env.sh

echo "🚀 設置機器學習開發環境..."

# 1. 安裝 uv (快速包管理器)
echo "📦 安裝 uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 建立虛擬環境
echo "🐍 建立 Python 虛擬環境..."
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安裝依賴
echo "📚 安裝依賴..."
uv pip install -r requirements.txt

# 4. 安裝 pre-commit hooks
echo "🔧 設置 pre-commit hooks..."
uv pip install pre-commit
pre-commit install

# 5. 啟動 JupyterLab
echo "🚀 啟動 JupyterLab..."
jupyter lab

echo "✅ 環境設置完成！"
```

### 5.2 推薦的項目結構

```
my-ml-project/
├── .git/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
├── models/
│   └── .gitkeep
├── reports/
│   └── figures/
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

### 5.3 Makefile 自動化

```makefile
# Makefile

.PHONY: help install test format lint docker-build docker-run clean

help:
	@echo "可用命令："
	@echo "  make install      - 安裝依賴"
	@echo "  make test         - 運行測試"
	@echo "  make format       - 格式化程式碼"
	@echo "  make lint         - 程式碼檢查"
	@echo "  make docker-build - 構建 Docker 鏡像"
	@echo "  make docker-run   - 運行 Docker 容器"
	@echo "  make clean        - 清理快取"

install:
	uv pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src

format:
	black src/ tests/
	isort src/ tests/

lint:
	ruff check src/ tests/
	mypy src/

docker-build:
	docker build -t my-ml-project:latest .

docker-run:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov/
```

### 5.4 Pre-commit 配置

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 6. 總結與最佳實踐

### ✅ 環境設置檢查清單

- [ ] 安裝 Docker 並理解基本概念
- [ ] 選擇依賴管理工具 (Poetry 或 uv)
- [ ] 配置 IDE (VS Code / Cursor / JupyterLab)
- [ ] 設置 AI 輔助工具 (Copilot / Codeium)
- [ ] 建立標準化的項目結構
- [ ] 配置 pre-commit hooks
- [ ] 編寫 Dockerfile 和 docker-compose.yml
- [ ] 設置版本控制 (.gitignore)

### 🎯 推薦工具組合

**初學者**:
- IDE: VS Code + Python 擴展
- 依賴: pip + requirements.txt
- AI: Codeium (免費)

**進階用戶**:
- IDE: VS Code / Cursor + JupyterLab
- 依賴: Poetry
- 容器: Docker + Docker Compose
- AI: GitHub Copilot

**專業團隊**:
- IDE: Cursor / VS Code + JupyterLab
- 依賴: Poetry + uv
- 容器: Docker + Kubernetes
- CI/CD: GitHub Actions / GitLab CI
- AI: GitHub Copilot + ChatGPT API

### 📚 延伸學習資源

- **Docker**: [官方文檔](https://docs.docker.com/)
- **Poetry**: [官方文檔](https://python-poetry.org/)
- **uv**: [GitHub](https://github.com/astral-sh/uv)
- **VS Code Python**: [官方教程](https://code.visualstudio.com/docs/python/python-tutorial)
- **JupyterLab**: [官方文檔](https://jupyterlab.readthedocs.io/)

---

**下一步**: 設置好環境後，前往 `1_Data_Acquisition_and_Analysis/` 開始學習資料處理和分析！
