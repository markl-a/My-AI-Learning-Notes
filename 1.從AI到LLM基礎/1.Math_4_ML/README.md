# 機器學習數學基礎 (Math for Machine Learning)

歡迎來到 **Math_4_ML** 專案目錄！本目錄系統性地整理了機器學習與深度學習所需的數學基礎知識，從理論到實作，幫助你建立紮實的數學基礎。

## 📚 目錄概覽

本目錄涵蓋四大數學領域，每個主題都包含詳細的理論說明、Python 實作範例和實際應用案例：

### 1. [線性代數 (Linear Algebra)](Linear_Algebra.md)
**為什麼重要：** 線性代數是深度學習的核心語言，從神經網路的前向傳播到注意力機制，都建立在矩陣運算之上。

**核心內容：**
- 向量、矩陣、張量的基本運算
- 特徵值分解與 SVD（奇異值分解）
- 主成分分析 (PCA) 與降維技術
- 神經網路權重初始化策略
- Transformer 中的矩陣運算

**實作範例：** 10+ 個完整的 Python 範例，包括 PCA 實現、SVD 圖像壓縮、權重初始化等。

### 2. [微積分 (Calculus)](Calculus.md)
**為什麼重要：** 理解梯度下降的本質、反向傳播的原理，以及如何優化神經網路參數。

**核心內容：**
- 導數、偏導數與梯度向量
- 多變數微積分與鏈式法則
- 反向傳播 (Backpropagation) 的數學推導
- 自動微分 (Automatic Differentiation)
- 梯度下降及其變種 (Momentum, Adam)

**實作範例：** 6+ 個範例，包括數值微分、梯度視覺化、從零實現反向傳播、優化器比較等。

### 3. [機率與統計 (Probability and Statistics)](Probability_and_Statistics.md)
**為什麼重要：** 機器學習本質上是在處理不確定性，理解機率分佈、貝葉斯推論和信息論是必須的。

**核心內容：**
- 機率分佈（高斯、伯努利、多項式等）
- 期望值、變異數、協方差矩陣
- 信息論：熵、KL 散度、交叉熵
- 最大似然估計 (MLE) 與貝葉斯推論
- 蒙地卡羅方法與 MCMC 採樣
- 變分推論 (Variational Inference)

**實作範例：** 6+ 個深入範例，涵蓋從基礎分佈到 MCMC 採樣、貝葉斯推論等。

### 4. [優化基礎 (Optimization Basics)](Optimization_Basics.md)
**為什麼重要：** 訓練神經網路就是解決優化問題，理解不同優化器的特性能幫助你更好地訓練模型。

**核心內容：**
- 梯度下降法及其變種
- 動量法 (Momentum) 與 Nesterov 加速
- 自適應學習率：AdaGrad, RMSProp, Adam
- 學習率調度策略
- 數值穩定性技巧（梯度裁剪、混合精度訓練）
- 二階方法簡介（牛頓法、BFGS）

**實作範例：** 豐富的實作範例，包括各種優化器實現、損失曲面視覺化、調參技巧等。

## 🎯 學習路徑建議

### 初學者路徑
1. **先學線性代數** → 理解向量、矩陣運算
2. **再學微積分** → 理解梯度、導數
3. **然後學優化** → 理解如何訓練模型
4. **最後學機率統計** → 理解模型的不確定性

### 進階學習者路徑
- 直接跳到你需要深化的主題
- 每個文檔都是獨立的，可以單獨學習
- 重點關注實作範例，動手實踐

## 💡 如何使用本目錄

### 理論學習
每個主題都包含：
- 📖 清晰的數學概念解釋
- 🔗 概念之間的聯繫
- 🎯 在機器學習中的實際應用

### 實作練習
每個文檔都提供：
- 💻 可執行的 Python 程式碼
- 📊 視覺化結果
- 🔍 詳細的註解說明

### 建議學習方式
1. **先閱讀理論** - 理解數學概念
2. **執行程式碼** - 看到實際效果
3. **修改參數** - 觀察變化
4. **應用到實際問題** - 鞏固理解

## 🛠️ 環境準備

所有程式碼範例使用以下 Python 套件：
```bash
pip install numpy scipy matplotlib seaborn scikit-learn torch
```

## 📖 推薦學習資源

### 書籍
- **Deep Learning** (Goodfellow, Bengio, Courville) - 深度學習聖經
- **Pattern Recognition and Machine Learning** (Bishop) - 機率觀點
- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) - 專為 ML 設計

### 線上課程
- [3Blue1Brown - 線性代數的本質](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown - 微積分的本質](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)

### 參考資料
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - 矩陣運算速查
- [Distill.pub](https://distill.pub/) - 視覺化機器學習概念

## 🚀 從這裡開始

建議按照以下順序開始學習：
1. 📐 [Linear_Algebra.md](Linear_Algebra.md) - 建立基礎
2. 📈 [Calculus.md](Calculus.md) - 理解優化
3. 🎲 [Probability_and_Statistics.md](Probability_and_Statistics.md) - 處理不確定性
4. ⚡ [Optimization_Basics.md](Optimization_Basics.md) - 訓練模型

---

**提示：** 所有程式碼範例都經過測試並可直接執行。建議在 Jupyter Notebook 中運行以獲得最佳學習體驗。

**持續更新：** 本目錄會隨著機器學習領域的發展持續更新，加入最新的技術和應用。

