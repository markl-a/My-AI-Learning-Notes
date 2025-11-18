# 線性神經網路 (Linear Neural Networks)

## 📚 課程總覽

本章節介紹深度學習的基礎：線性神經網路。在介紹複雜的深度神經網路之前，我們需要掌握神經網路訓練的基礎知識，包括模型定義、數據處理、損失函數和訓練過程。

### 🎯 學習目標

- 理解線性回歸和 Softmax 回歸的數學原理
- 掌握從零實現和框架實現兩種方式
- 學會數據預處理和特徵工程
- 了解模型評估和正則化技術
- 學習使用 AI 工具輔助開發和調試

---

## 📖 章節內容

### 基礎理論

| 編號 | 主題 | 類型 | 難度 | 說明 |
|------|------|------|------|------|
| 0 | [目錄](0_index.ipynb) | 📑 導航 | - | 章節索引 |
| 1 | [線性回歸理論](1_linear-regression.ipynb) | 📖 理論 | ⭐ | 線性回歸數學基礎、損失函數、優化算法 |
| 4 | [Softmax 回歸理論](4_softmax-regression.ipynb) | 📖 理論 | ⭐⭐ | 分類問題、Softmax 函數、交叉熵損失 |
| 5 | [圖像分類數據集](5_image-classification-dataset.ipynb) | 📖 理論 | ⭐ | Fashion-MNIST 數據集介紹和使用 |

### 從零實現 (From Scratch)

| 編號 | 主題 | 類型 | 難度 | 說明 |
|------|------|------|------|------|
| 2 | [線性回歸從零實現](2_linear-regression-scratch.ipynb) | 💻 實作 | ⭐⭐ | 手動實現數據生成、模型、損失、優化器 |
| 6 | [Softmax 回歸從零實現](6_softmax-regression-scratch.ipynb) | 💻 實作 | ⭐⭐⭐ | 手動實現分類模型和訓練過程 |

### 框架實現 (Concise Implementation)

| 編號 | 主題 | 類型 | 難度 | 說明 |
|------|------|------|------|------|
| 3 | [線性回歸簡潔實現](3_linear-regression-concise.ipynb) | 💻 實作 | ⭐ | 使用 PyTorch 高級 API 實現 |
| 7 | [Softmax 回歸簡潔實現](7_softmax-regression-concise.ipynb) | 💻 實作 | ⭐⭐ | 使用 PyTorch 實現分類模型 |

### 進階內容 (New!)

| 編號 | 主題 | 類型 | 難度 | 說明 |
|------|------|------|------|------|
| 8 | [實用工具模組](utils.py) | 🔧 工具 | ⭐ | 通用函數庫 |
| 9 | [房價預測實戰](9_house-price-prediction.ipynb) | 🎯 實戰 | ⭐⭐ | 完整的回歸任務案例 |
| 10 | [正則化技術](10_regularization.ipynb) | 📖 進階 | ⭐⭐⭐ | Ridge、Lasso、Elastic Net |
| 11 | [模型評估與比較](11_model-evaluation.ipynb) | 📊 評估 | ⭐⭐ | R²、MAE、RMSE、交叉驗證 |
| 12 | [視覺化工具](12_visualization.ipynb) | 📊 視覺化 | ⭐⭐ | 訓練過程、梯度下降動畫 |
| 13 | [AI 輔助開發](13_ai-assisted-ml.ipynb) | 🤖 AI 輔助 | ⭐⭐⭐ | 自動調參、模型診斷 |
| 14 | [特徵工程](14_feature-engineering.ipynb) | 🔧 進階 | ⭐⭐⭐ | 特徵選擇、轉換、生成 |

---

## 🚀 學習路徑

### 入門路徑 (3-5 天)

```mermaid
graph LR
    A[線性回歸理論] --> B[線性回歸從零實現]
    B --> C[線性回歸簡潔實現]
    C --> D[Softmax 回歸理論]
    D --> E[圖像分類數據集]
    E --> F[Softmax 回歸簡潔實現]
```

**適合對象**：深度學習初學者

**學習步驟**：
1. **第 1 天**：學習線性回歸理論 (章節 1)
2. **第 2 天**：從零實現線性回歸 (章節 2)
3. **第 3 天**：框架實現線性回歸 (章節 3)
4. **第 4 天**：Softmax 回歸理論和數據集 (章節 4-5)
5. **第 5 天**：Softmax 回歸實現 (章節 7)

### 實戰路徑 (5-7 天)

```mermaid
graph LR
    A[完成入門路徑] --> B[房價預測實戰]
    B --> C[模型評估與比較]
    C --> D[特徵工程]
    D --> E[正則化技術]
    E --> F[視覺化分析]
```

**適合對象**：有基礎知識，想提升實戰能力

**學習步驟**：
1. 完成入門路徑的所有內容
2. 實踐房價預測案例 (章節 9)
3. 學習模型評估技術 (章節 11)
4. 掌握特徵工程 (章節 14)
5. 應用正則化防止過擬合 (章節 10)
6. 使用視覺化分析模型 (章節 12)

### 進階路徑 (7-10 天)

```mermaid
graph LR
    A[完成實戰路徑] --> B[深入從零實現]
    B --> C[AI 輔助開發]
    C --> D[超參數自動調優]
    D --> E[模型解釋性]
    E --> F[生產部署]
```

**適合對象**：追求深度理解和自動化

**學習步驟**：
1. 完成前兩個路徑
2. 深入理解從零實現的每個細節 (章節 2, 6)
3. 學習 AI 輔助工具 (章節 13)
4. 掌握自動化調參技術
5. 研究模型可解釋性方法
6. 了解生產環境部署

---

## 💡 核心概念

### 線性回歸

**數學形式**：
```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

**關鍵要素**：
- **模型**：線性假設 `y = Xw + b`
- **損失函數**：均方誤差 (MSE)
- **優化算法**：小批量隨機梯度下降 (SGD)
- **評估指標**：R²、MAE、RMSE

### Softmax 回歸

**數學形式**：
```
softmax(o)ᵢ = exp(oᵢ) / Σⱼ exp(oⱼ)
```

**關鍵要素**：
- **模型**：多類別分類
- **損失函數**：交叉熵損失
- **輸出**：概率分布
- **評估指標**：準確率、混淆矩陣

---

## 🛠️ 環境設置

### 必需套件

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 可選套件 (AI 輔助)

```bash
pip install optuna wandb shap lime plotly
```

### 驗證安裝

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📊 實戰案例

### 1. 房價預測 (章節 9)

**任務**：根據房屋特徵預測價格

**數據特徵**：
- 面積、房齡、臥室數量、位置等
- 包含數據清洗和特徵工程

**技術點**：
- 數據預處理
- 特徵標準化
- 交叉驗證
- 模型評估

### 2. 手寫數字分類 (章節 6-7)

**任務**：識別 Fashion-MNIST 圖像

**數據集**：
- 60,000 訓練圖像
- 10,000 測試圖像
- 10 個類別

**技術點**：
- 圖像數據處理
- Softmax 分類
- 準確率評估

---

## 🤖 AI 輔助功能

### 自動化工具

1. **超參數調優** (章節 13)
   - 使用 Optuna 自動搜索最佳參數
   - 學習率、批次大小、正則化係數

2. **模型診斷**
   - 自動檢測過擬合/欠擬合
   - 建議改進方向

3. **特徵重要性分析**
   - SHAP 值分析
   - 特徵選擇建議

### AI 編程助手使用建議

**推薦提示詞**：

```
"幫我優化這個線性回歸模型，目前 R² 只有 0.6"
"解釋為什麼我的損失不下降"
"建議一些特徵工程的方法"
"如何選擇合適的學習率？"
```

---

## 📈 性能優化技巧

### 數據處理

- ✅ 使用 DataLoader 批次處理
- ✅ 數據標準化/歸一化
- ✅ 使用 GPU 加速 (如果可用)

### 訓練優化

- ✅ 合適的學習率調度
- ✅ 早停法防止過擬合
- ✅ 批次大小權衡

### 記憶體優化

- ✅ 梯度累積
- ✅ 混合精度訓練
- ✅ 數據流式處理

---

## 🔍 常見問題

### Q1: 損失不下降怎麼辦？

**可能原因**：
- 學習率太大或太小
- 數據沒有標準化
- 模型過於簡單

**解決方案**：
1. 調整學習率 (試試 0.001, 0.01, 0.1)
2. 標準化輸入特徵
3. 檢查數據質量

### Q2: 如何選擇批次大小？

**建議**：
- **小批次** (32-64)：更好的泛化能力，但訓練較慢
- **大批次** (128-256)：訓練快，但可能過擬合

**經驗法則**：
```python
batch_size = min(256, len(dataset) // 10)
```

### Q3: 從零實現 vs 框架實現？

**從零實現優點**：
- 深入理解原理
- 靈活自定義

**框架實現優點**：
- 代碼簡潔
- 性能優化
- 生產級穩定性

**建議**：先學從零實現理解原理，實際項目用框架實現。

---

## 📚 延伸閱讀

### 推薦書籍

1. **《動手學深度學習》** - 本章節的理論基礎
2. **《Pattern Recognition and Machine Learning》** - Christopher Bishop
3. **《The Elements of Statistical Learning》** - Hastie et al.

### 在線資源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai 課程](https://www.fast.ai/)

### 相關論文

- **線性回歸**：Legendre (1805), Gauss (1809)
- **SGD 優化**：Robbins & Monro (1951)
- **Softmax**：Bridle (1990)

---

## 🤝 貢獻與反饋

### 如何貢獻

歡迎提交：
- 錯誤修正
- 新的實戰案例
- 代碼優化
- 文檔改進

### 反饋渠道

- 提交 Issue
- Pull Request
- 討論區交流

---

## 📝 更新日誌

### v2.0 (2024-11)

**新增內容**：
- ✨ 8 個進階 notebook
- ✨ 實用工具模組
- ✨ AI 輔助開發工具
- ✨ 完整的學習路徑圖
- ✨ 詳細的 README 文檔

**改進**：
- 📈 更豐富的視覺化
- 🔧 更完整的代碼注釋
- 📊 更多實戰案例
- 🤖 整合 AI 輔助工具

### v1.0 (2023)

- 基礎的 8 個 notebook
- 線性回歸和 Softmax 回歸
- 從零實現和框架實現

---

## 📞 聯繫方式

如有問題或建議，歡迎聯繫：

- 📧 Email: [your-email]
- 💬 討論區: [link]
- 🐛 Bug Report: [GitHub Issues]

---

## 📄 許可證

本教程採用 [MIT License](LICENSE) 或 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

**祝學習順利！💪**

_最後更新：2024-11-18_
