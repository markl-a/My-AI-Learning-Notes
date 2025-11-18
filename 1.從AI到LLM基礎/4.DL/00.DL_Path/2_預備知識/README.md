# 深度學習預備知識 - 完整學習指南

## 📚 目錄概覽

本資料夾包含深度學習所需的所有數學和程式設計基礎知識，從入門到熟練的完整學習路徑。

### 核心模組

| 模組 | 文件 | 難度 | 預計時間 | 描述 |
|------|------|------|----------|------|
| 📖 索引 | [0_index.ipynb](pytorch/0_index.ipynb) | ⭐ | 15分鐘 | 預備知識總覽與學習路線圖 |
| 🔢 數據操作 | [1_ndarray.ipynb](pytorch/1_ndarray.ipynb) | ⭐⭐ | 2-3小時 | PyTorch 張量基礎操作 |
| 📊 數據處理 | [2_pandas.ipynb](pytorch/2_pandas.ipynb) | ⭐⭐ | 2-3小時 | 使用 Pandas 進行數據預處理 |
| 📐 線性代數 | [3_linear-algebra.ipynb](pytorch/3_linear-algebra.ipynb) | ⭐⭐⭐ | 4-5小時 | 向量、矩陣運算與深度學習應用 |
| 📈 微積分 | [4_calculus.ipynb](pytorch/4_calculus.ipynb) | ⭐⭐⭐ | 4-5小時 | 導數、偏導數與梯度 |
| 🤖 自動微分 | [5_autograd.ipynb](pytorch/5_autograd.ipynb) | ⭐⭐⭐ | 3-4小時 | PyTorch 自動微分機制 |
| 🎲 概率統計 | [6_probability.ipynb](pytorch/6_probability.ipynb) | ⭐⭐⭐ | 4-5小時 | 概率論與統計基礎 |
| 📚 API查詢 | [7_lookup-api.ipynb](pytorch/7_lookup-api.ipynb) | ⭐ | 1小時 | PyTorch 文檔查詢技巧 |

## 🎯 學習目標

完成本模組後，你將能夠：

- ✅ 熟練使用 PyTorch 進行張量操作和數據處理
- ✅ 理解深度學習中的線性代數核心概念
- ✅ 掌握自動微分的原理和應用
- ✅ 具備必要的微積分和概率論基礎
- ✅ 能夠閱讀和理解深度學習論文中的數學公式

## 📖 學習路徑

### 🚀 入門路徑（1-2週）

適合完全新手，循序漸進學習：

```mermaid
graph LR
    A[0. 索引] --> B[1. 數據操作]
    B --> C[2. Pandas]
    C --> D[7. API查詢]
    D --> E[開始實踐項目]
```

**學習順序：**
1. 閱讀索引，了解整體框架
2. 學習張量基礎操作（ndarray）
3. 掌握數據預處理（pandas）
4. 學習如何查詢 PyTorch API
5. 完成基礎實踐項目

### ⚡ 進階路徑（2-3週）

已有程式設計基礎，深入數學原理：

```mermaid
graph LR
    A[1. 數據操作] --> B[3. 線性代數]
    B --> C[4. 微積分]
    C --> D[5. 自動微分]
    D --> E[6. 概率統計]
    E --> F[綜合實踐項目]
```

**學習順序：**
1. 快速回顧張量操作
2. 深入學習線性代數（向量、矩陣）
3. 理解微積分基礎（導數、梯度）
4. 掌握自動微分機制
5. 學習概率統計基礎
6. 完成進階綜合項目

### 🏆 精通路徑（3-4週）

追求深度理解，結合理論與實踐：

所有模組 + 額外的數學推導 + 實現自定義自動微分引擎

## 🛠️ 環境設置

### 基礎環境

```bash
# 安裝 PyTorch
pip install torch torchvision torchaudio

# 安裝數據處理庫
pip install numpy pandas matplotlib seaborn

# 安裝 Jupyter
pip install jupyter notebook

# 啟動 Jupyter Notebook
jupyter notebook
```

### AI 輔助學習環境

```bash
# 安裝額外的學習輔助工具
pip install plotly ipywidgets scikit-learn

# 啟用 Jupyter 擴展
jupyter nbextension enable --py widgetsnbextension
```

## 📝 學習建議

### ✨ 最佳實踐

1. **動手實踐**：每個概念都要親自編寫代碼驗證
2. **完成練習**：每個 notebook 末尾都有練習題，務必完成
3. **記錄筆記**：在 notebook 中添加自己的理解和註釋
4. **定期複習**：利用間隔重複法鞏固記憶
5. **討論交流**：加入學習社群，與他人討論問題

### 🎯 學習檢查清單

- [ ] 能夠創建和操作不同維度的張量
- [ ] 理解廣播機制的工作原理
- [ ] 能夠進行矩陣乘法和向量運算
- [ ] 理解梯度下降的數學原理
- [ ] 能夠使用自動微分計算複雜函數的梯度
- [ ] 理解常見概率分佈的特性
- [ ] 能夠使用 Pandas 進行數據清洗和轉換
- [ ] 熟練查閱 PyTorch 官方文檔

## 🧪 實踐項目

### 初級項目

1. **手寫數字數據預處理**
   - 使用 Pandas 和 NumPy 處理 MNIST 數據
   - 練習張量操作和數據可視化

2. **線性回歸實現**
   - 從零實現線性回歸
   - 理解梯度下降優化過程

### 中級項目

3. **多項式擬合與正則化**
   - 實現不同階數的多項式擬合
   - 探索過擬合和欠擬合

4. **自定義自動微分引擎**
   - 實現簡單的計算圖
   - 理解反向傳播的細節

### 高級項目

5. **貝葉斯推理實驗**
   - 實現貝葉斯線性回歸
   - 理解概率推理在機器學習中的應用

6. **矩陣分解應用**
   - 實現 SVD 和 PCA
   - 應用於數據降維和推薦系統

## 🤖 AI 輔助學習工具

本資料夾提供了多個 AI 輔助學習工具：

### 1. 智能練習生成器 `ai_exercise_generator.py`

自動生成個性化練習題，根據你的學習進度調整難度。

```python
python ai_exercise_generator.py --topic linear_algebra --difficulty medium
```

### 2. 概念可視化工具 `visualizer.py`

交互式可視化數學概念，幫助直觀理解。

```python
python visualizer.py --concept gradient_descent
```

### 3. 學習進度追蹤器 `progress_tracker.py`

記錄學習進度，分析薄弱環節，提供個性化建議。

```python
python progress_tracker.py --report
```

## 📚 推薦資源

### 書籍

- 《深度學習》- Ian Goodfellow (中文版)
- 《動手學深度學習》- 李沐等
- 《線性代數及其應用》- David C. Lay
- 《概率論與數理統計》- 茆詩松

### 在線課程

- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [3Blue1Brown - 線性代數的本質](https://www.3blue1brown.com/topics/linear-algebra)
- [StatQuest - 統計學習](https://statquest.org/)

### 工具與文檔

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [NumPy 文檔](https://numpy.org/doc/)
- [Pandas 文檔](https://pandas.pydata.org/docs/)

## 🎓 進階學習路線

完成本模組後，建議的學習路徑：

```
預備知識 (當前)
    ↓
線性神經網路 (../3_線性神經網路)
    ↓
多層感知機 (../4_多層感知機)
    ↓
卷積神經網路 (../13_計算機視覺)
    ↓
循環神經網路與注意力機制 (../10_注意力機制)
    ↓
Transformer 與預訓練模型 (../14_自然語言處理：預訓練)
```

## 🤝 貢獻與反饋

如果你發現任何錯誤或有改進建議，歡迎：

- 提交 Issue
- 發起 Pull Request
- 在討論區分享學習心得

## 📄 授權

本學習資料基於開源精神分享，遵循 MIT 授權條款。

---

## 🌟 重要提示

> **學習深度學習的數學基礎不是為了成為數學家，而是為了：**
> 1. 理解算法的工作原理
> 2. 調試模型時能夠追溯問題根源
> 3. 閱讀最新研究論文
> 4. 創新和改進現有方法
>
> **不要被數學嚇倒！** 本資料會循序漸進，從直觀理解開始，逐步深入數學細節。

---

**最後更新：** 2024-11
**維護者：** AI Learning Community
**版本：** v3.0

開始你的深度學習之旅吧！🚀
