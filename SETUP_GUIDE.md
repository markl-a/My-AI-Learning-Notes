# 環境安裝指南

## 📋 目錄

1. [快速開始](#快速開始)
2. [詳細安裝步驟](#詳細安裝步驟)
3. [不同學習路徑的安裝建議](#不同學習路徑的安裝建議)
4. [常見問題](#常見問題)

---

## 🚀 快速開始

### 方式 1：使用 pip（推薦）

```bash
# 1. 克隆專案
git clone https://github.com/markl-a/My-AI-Learning-Notes.git
cd My-AI-Learning-Notes

# 2. 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安裝依賴（根據需求選擇）
# 選項 A - 基礎安裝
pip install -r requirements.txt

# 選項 B - 包含機器學習
pip install -r requirements.txt -r requirements-ml.txt

# 選項 C - 包含深度學習
pip install -r requirements.txt -r requirements-dl.txt

# 選項 D - 包含 LLM 應用
pip install -r requirements.txt -r requirements-llm.txt

# 選項 E - 完整安裝（所有功能）
pip install -r requirements-full.txt

# 選項 F - 開發環境
pip install -r requirements.txt -r requirements-dev.txt
```

### 方式 2：使用 pyproject.toml（現代化）

```bash
# 基礎安裝
pip install -e .

# 包含可選依賴
pip install -e ".[ml]"           # 機器學習
pip install -e ".[dl]"           # 深度學習
pip install -e ".[llm]"          # LLM 應用
pip install -e ".[dev]"          # 開發工具
pip install -e ".[full]"         # 完整安裝
```

---

## 📚 詳細安裝步驟

### Step 1: 檢查 Python 版本

確保您的 Python 版本 >= 3.9

```bash
python --version
```

如果版本過舊，請從 [python.org](https://www.python.org/downloads/) 下載最新版本。

### Step 2: 建立虛擬環境

**為什麼需要虛擬環境？**
- 隔離項目依賴
- 避免版本衝突
- 便於管理

**建立方式：**

#### 使用 venv（內建）
```bash
python -m venv venv
```

#### 使用 conda（推薦用於資料科學）
```bash
conda create -n ai-learning python=3.11
conda activate ai-learning
```

### Step 3: 激活虛擬環境

```bash
# Linux/Mac
source venv/bin/activate

# Windows (cmd)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Conda
conda activate ai-learning
```

### Step 4: 更新 pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 5: 安裝依賴

根據您的學習路徑選擇合適的安裝方式（見下節）。

---

## 🎯 不同學習路徑的安裝建議

### 1️⃣ 從 AI 到機器學習（初學者）

**推薦配置：**
```bash
pip install -r requirements.txt -r requirements-ml.txt
```

**包含的功能：**
- ✅ 數學基礎（NumPy, SciPy, Matplotlib）
- ✅ 機器學習（Scikit-learn, XGBoost, LightGBM）
- ✅ 資料處理（Pandas）
- ✅ 可視化（Matplotlib, Seaborn, Plotly）
- ✅ Jupyter Notebook

**適合：**
- 正在學習 1.從AI到LLM基礎/1.Math_4_ML
- 正在學習 1.從AI到LLM基礎/3.ML_&_Data_Analysis

---

### 2️⃣ 深度學習（進階）

**推薦配置：**
```bash
pip install -r requirements.txt -r requirements-dl.txt
```

**包含的功能：**
- ✅ PyTorch 生態系統
- ✅ TensorFlow/Keras
- ✅ Hugging Face Transformers
- ✅ 計算機視覺工具（OpenCV, Pillow）
- ✅ NLP 工具（NLTK, spaCy）
- ✅ 模型訓練工具（WandB, TensorBoard）

**適合：**
- 正在學習 1.從AI到LLM基礎/4.DL
- 正在學習 2.深入LLM模型工程與LLM運維

**注意：PyTorch GPU 版本**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ROCm (AMD GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

### 3️⃣ LLM 應用開發

**推薦配置：**
```bash
pip install -r requirements.txt -r requirements-llm.txt
```

**包含的功能：**
- ✅ LLM API（OpenAI, Anthropic）
- ✅ LangChain 生態系統
- ✅ 向量資料庫（ChromaDB, FAISS）
- ✅ RAG 工具
- ✅ Agent 框架
- ✅ Web 框架（FastAPI, Gradio, Streamlit）
- ✅ 文檔處理工具

**適合：**
- 正在學習 3.LLM應用工程/4.(RAG) 基礎
- 正在學習 3.LLM應用工程/3.Agent
- 正在學習 3.LLM應用工程/7.LLM應用部屬

**需要 API 金鑰：**
```bash
# 建立 .env 文件
cp .env.example .env

# 編輯 .env 文件，添加您的 API 金鑰
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

---

### 4️⃣ 開發者（貢獻程式碼）

**推薦配置：**
```bash
pip install -r requirements-full.txt -r requirements-dev.txt
```

**包含的功能：**
- ✅ 所有上述功能
- ✅ 測試工具（pytest, coverage）
- ✅ 程式碼品質工具（black, ruff, mypy）
- ✅ 文檔工具（Sphinx, MkDocs）
- ✅ 性能分析工具
- ✅ Pre-commit hooks

**設置 Pre-commit：**
```bash
pre-commit install
pre-commit run --all-files
```

---

### 5️⃣ 完整學習路徑

**推薦配置：**
```bash
pip install -r requirements-full.txt
```

**注意：**
- ⚠️ 這會安裝大量包（可能需要 10+ GB 空間）
- ⚠️ 安裝時間較長（可能需要 20-30 分鐘）
- ⚠️ 建議有足夠的網絡帶寬

---

## 🔧 常見問題

### Q1: 安裝失敗怎麼辦？

**A:** 常見解決方案：

```bash
# 1. 更新 pip
pip install --upgrade pip

# 2. 使用國內鏡像（中國大陸用戶）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 逐個安裝依賴
pip install -r requirements.txt --no-deps
```

### Q2: PyTorch 安裝問題

**A:** 訪問 [PyTorch 官網](https://pytorch.org/get-started/locally/) 獲取適合您系統的安裝命令。

### Q3: TensorFlow GPU 支援

**A:**
```bash
# TensorFlow 2.20+ 自動檢測 GPU
pip install tensorflow

# 驗證 GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Q4: 虛擬環境激活失敗（Windows PowerShell）

**A:**
```powershell
# 設置執行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q5: M1/M2 Mac 安裝問題

**A:**
```bash
# 使用 conda（推薦）
conda install pytorch torchvision torchaudio -c pytorch

# 某些包需要 Rosetta 2
arch -x86_64 pip install package-name
```

### Q6: 內存不足

**A:**
```bash
# 限制並行下載
pip install -r requirements.txt --no-cache-dir

# 逐個安裝
cat requirements.txt | xargs -n 1 pip install
```

### Q7: 依賴衝突

**A:**
```bash
# 使用 pip-tools 管理依賴
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

---

## 🧪 驗證安裝

### 快速驗證腳本

建立 `verify_installation.py`：

```python
#!/usr/bin/env python
"""驗證環境安裝"""

def verify_basic():
    """驗證基礎包"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ 基礎包安裝成功")
        return True
    except ImportError as e:
        print(f"❌ 基礎包安裝失敗: {e}")
        return False

def verify_ml():
    """驗證機器學習包"""
    try:
        import sklearn
        import xgboost
        import lightgbm
        print("✅ 機器學習包安裝成功")
        return True
    except ImportError as e:
        print(f"❌ 機器學習包安裝失敗: {e}")
        return False

def verify_dl():
    """驗證深度學習包"""
    try:
        import torch
        import tensorflow as tf
        import transformers
        print("✅ 深度學習包安裝成功")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   TensorFlow: {tf.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 深度學習包安裝失敗: {e}")
        return False

def verify_llm():
    """驗證 LLM 包"""
    try:
        import langchain
        import chromadb
        import openai
        print("✅ LLM 應用包安裝成功")
        return True
    except ImportError as e:
        print(f"❌ LLM 應用包安裝失敗: {e}")
        return False

if __name__ == "__main__":
    print("🔍 開始驗證環境...")
    print("-" * 50)

    verify_basic()
    verify_ml()
    verify_dl()
    verify_llm()

    print("-" * 50)
    print("✨ 驗證完成！")
```

執行驗證：
```bash
python verify_installation.py
```

---

## 📖 更多資源

- 📚 [主要 README](README.md)
- 🐛 [問題報告](https://github.com/markl-a/My-AI-Learning-Notes/issues)
- 💬 [討論區](https://github.com/markl-a/My-AI-Learning-Notes/discussions)
- 🤝 [貢獻指南](CONTRIBUTING.md)

---

## 📞 獲取幫助

如果遇到問題：

1. 查看 [常見問題](#常見問題)
2. 搜索 [Issues](https://github.com/markl-a/My-AI-Learning-Notes/issues)
3. 建立新的 Issue
4. 加入社群討論

---

**祝學習愉快！🎉**
