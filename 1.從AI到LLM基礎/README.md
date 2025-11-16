# 從AI到LLM基礎

歡迎來到 **從AI到LLM基礎** 學習專區！本目錄匯集了從傳統機器學習到現代大型語言模型(LLM)所需的所有基礎知識與實作內容。

## 📖 目錄概覽

```
1.從AI到LLM基礎/
├── 1.Math_4_ML/              # 機器學習數學基礎
├── 2.AI_Intro/               # AI 簡介與發展歷史
├── 3.ML_&_Data_Analysis/     # 機器學習與資料分析
└── 4.DL/                     # 深度學習(包含CV和NLP)
```

## 🎯 學習目標

本專區旨在幫助學習者：

- ✅ 掌握機器學習所需的數學基礎（線性代數、微積分、機率統計）
- ✅ 理解 AI 的發展歷程與各個領域的關聯
- ✅ 熟練掌握 Python 資料科學工具鏈（NumPy、Pandas、Matplotlib）
- ✅ 實作傳統機器學習演算法並參與 Kaggle 競賽
- ✅ 深入學習深度學習框架（TensorFlow、PyTorch、Keras）
- ✅ 掌握電腦視覺(CV)和自然語言處理(NLP)的核心技術
- ✅ 了解分散式運算與 MLOps 基礎，為 LLM 工程打好基礎

## 📚 詳細內容導覽

### 1. Math_4_ML（機器學習數學基礎）

此資料夾匯集並整理與人工智慧（AI）相關的數學基礎與進階概念。

#### 📂 包含內容：

- **[Linear_Algebra.md](1.Math_4_ML/Linear_Algebra.md)**
  - 線性代數基礎：矩陣運算、特徵值特徵向量、奇異值分解（SVD）
  - 對於理解深度學習中的許多演算法至關重要

- **[Calculus.md](1.Math_4_ML/Calculus.md)**
  - 微分、偏微分、梯度與鏈式法則
  - 重點在深度學習的反向傳播中應用

- **[Probability_and_Statistics.md](1.Math_4_ML/Probability_and_Statistics.md)**
  - 機率分佈、期望值、變異數
  - 最大似然估計、Bayes定理

- **[Optimization_Basics.md](1.Math_4_ML/Optimization_Basics.md)**
  - 基礎優化概念：梯度下降法、Learning Rate、動量等

#### 🎓 推薦學習路徑：

1. **線性代數** → **微積分** → **機率統計** → **優化基礎**
2. 可以邊學邊用 Python 實作相關概念（使用 NumPy）
3. 建議搭配視覺化工具（Matplotlib）輔助理解

#### 📖 延伸資源：

- [3Blue1Brown - 線性代數的本質](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [StatQuest - 統計基礎知識](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
- [沉浸式線性代數](https://immersivemath.com/ila/learnmore.html)

---

### 2. AI_Intro（AI 簡介與相關領域）

AI（人工智慧）是一個相當廣泛的領域，它並不僅限於機器學習（Machine Learning）與深度學習（Deep Learning）。

#### 📂 包含內容：

- **AI 的廣泛領域介紹**
  - 知識表達與推理（Knowledge Representation and Reasoning）
  - 自然語言處理（Natural Language Processing, NLP）
  - 電腦視覺（Computer Vision）
  - 規劃與決策（Planning and Decision Making）
  - 強化學習（Reinforcement Learning）
  - 進化計算（Evolutionary Computation）
  - 機器人學（Robotics）
  - 多智能體系統（Multi-agent Systems）

- **[人工智慧到生成式AI的發展(2010 ~2024).md](2.AI_Intro/人工智慧到生成式AI的發展(2010%20~2024).md)**
  - 從深度學習革命到生成式 AI 的完整發展歷程
  - 涵蓋 2010-2024 年重要里程碑與突破

#### 🎓 推薦學習資源：

- [Harvard CS50's Artificial Intelligence with Python](https://youtu.be/5NgNicANyqM)
- [Stanford CS221: AI Principles and Techniques](https://youtu.be/ZiwogMtbjr4)
- [MIT 6.034 Artificial Intelligence, Fall 2010](https://youtu.be/TjZBTDzGeGg)

---

### 3. ML_&_Data_Analysis（機器學習與資料分析）

包含機器學習基礎、資料分析、傳統ML演算法、EDA、特徵工程、Kaggle案例、分散式運算、大數據處理與ETL流程整合。

#### 📂 子目錄結構：

##### 0_Introduction_and_Environment_Setup
- 環境設定指南（Python、Conda、Jupyter Notebook）
- 常用套件介紹（NumPy、Pandas、Matplotlib、Scikit-learn、Spark 等）

##### 1_Data_Acquisition_and_Analysis
- **[Data_Sources.md](3.ML_&_Data_Analysis/1_Data_Acquisition_and_Analysis/1_Data_Sources.md)**：常見資料來源（CSV、SQL、NoSQL、API）
- **資料清理與處理**：缺失值處理、異常值檢測、類別特徵編碼
- **探索性資料分析 (EDA)**：統計量分析、相關係數、資料視覺化
- **特徵工程**：特徵選擇、特徵縮放、特徵組合、One-Hot Encoding

##### 2_Traditional_ML_Algorithms
- 線性模型：線性回歸、羅吉斯迴歸
- 樹模型：決策樹、隨機森林、梯度提升樹 (XGBoost、LightGBM)
- SVM 和 KNN：支援向量機、k 最近鄰
- 樸素貝葉斯分類器
- 模型評估與指標：Cross-Validation、Accuracy、Precision、Recall、F1、AUC
- 模型選擇與調參：GridSearchCV、RandomSearch、Bayesian Optimization

##### 3_Kaggle_Case_Studies
- **[Kaggle_Tips_and_Tricks.md](3.ML_&_Data_Analysis/3_Kaggle_Case_Studies/Kaggle_Tips_and_Tricks.md)**：Kaggle 競賽常用技巧
- 實戰案例：從 EDA 到特徵工程到模型提交的完整流程

##### 4_Distributed_Computing_and_BigData_Processing
- **[Introduction_to_Distributed_Computing.md](3.ML_&_Data_Analysis/4_Distributed_Computing_and_BigData_Processing/Introduction_to_Distributed_Computing.md)**：分散式運算概念
- Spark 基礎與應用（PySpark）
- 分散式環境中訓練 ML 模型
- **[ETL_Pipeline_Integration.md](3.ML_&_Data_Analysis/4_Distributed_Computing_and_BigData_Processing/ETL_Pipeline_Integration.md)**：ETL 流程整合

##### 5_Best_Practices_and_MLOps_Basics
- **[Model_Deployment_Introduction.md](3.ML_&_Data_Analysis/5_Best_Practices_and_MLOps_Basics/Model_Deployment_Introduction.md)**：模型部署基礎
- **[Version_Control_and_Experiment_Tracking.md](3.ML_&_Data_Analysis/5_Best_Practices_and_MLOps_Basics/Version_Control_and_Experiment_Tracking.md)**：DVC、MLflow、Git
- **[Scalability_and_Performance_Tuning.md](3.ML_&_Data_Analysis/5_Best_Practices_and_MLOps_Basics/Scalability_and_Performance_Tuning.md)**：效能調校

#### 🎓 推薦學習路徑：

1. **環境設定** → **資料獲取與分析** → **傳統 ML 演算法**
2. **Kaggle 實戰** → **分散式運算** → **MLOps 基礎**

#### 📖 延伸資源：

- [Python 資料科學手冊](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Scikit-learn 官方文檔](https://scikit-learn.org/stable/)

---

### 4. DL（深度學習）

集中深度學習主題，區分「電腦視覺」與「NLP & 語音」兩大領域。包含理論、實作範例，以及經典與現代研究論文的摘要。

#### 📂 包含內容：

##### 00. DL_Path
- **[動手深度學習](4.DL/00.DL_Path/)**：完整的深度學習基礎教學
- 涵蓋從基礎到進階的所有概念

##### 01. Tensorflow2
- **[TensorFlow 2 學習紀錄](4.DL/01.Tensorflow2/)**
- Google 的深度學習框架
- 適合生產環境部署

##### 02. Keras3
- **[Keras 3 學習紀錄](4.DL/02.Keras3/)**
- 高階 API，支援多後端（TensorFlow、PyTorch、JAX）
- 快速原型開發

##### 03. PyTorch
- **[PyTorch 學習紀錄](4.DL/03.Pytorch/)**
- 目前最熱門的深度學習框架
- 研究與生產環境均適用
- **[Segment Anything 2 論文解讀](4.DL/03.Pytorch/3.Segment%20Anything%202/)**：最新的視覺分割模型

##### 04. Ultralytics
- **[YOLO 物件偵測](4.DL/04.Ultralytics/)**
- YOLOv8 訓練與部署
- 實戰項目：使用自製資料集訓練模型

##### 05. Transformer_lib
- **[Hugging Face Transformers](4.DL/05.Transformer_lib/)**
- 使用預訓練模型進行 NLP 任務
- 為 LLM 學習打基礎

##### 06. Paper_with_code
- **[論文閱讀與復現](4.DL/06.Paper_with_code/)**
- 閱讀、運作與分析學術論文
- 包含視訊品質評估等實際應用案例

#### 🎓 學習建議：

**電腦視覺 (CV) 學習路徑：**
1. CNN 基礎 → 圖像分類 → 物件檢測（YOLO）
2. 圖像分割（SAM2）→ GAN → Vision Transformer

**自然語言處理 (NLP) 學習路徑：**
1. 文字預處理 → 詞嵌入（Word2Vec, GloVe）
2. RNN/LSTM → Transformer → BERT/GPT
3. Hugging Face Transformers 實戰

#### 📖 推薦資源：

- [動手深度學習官網](https://zh.d2l.ai/)
- [PyTorch 官網教學](https://pytorch.org/tutorials/)
- [TensorFlow 官網教學](https://www.tensorflow.org/tutorials)
- [Stanford CS231n (CV)](http://cs231n.stanford.edu/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)

---

## 🚀 快速開始

### 1. 完全新手（零基礎）

```
數學基礎 → Python 基礎 → 資料科學套件 → 機器學習入門
```

**推薦時程：**
- 數學基礎：2-4 週
- Python 與資料科學：2-3 週
- 機器學習基礎：4-6 週

### 2. 有程式基礎，想學 ML/DL

```
數學基礎（複習）→ ML 演算法 → Kaggle 實戰 → 深度學習框架
```

**推薦時程：**
- 數學複習：1-2 週
- ML 演算法與實戰：3-4 週
- 深度學習：6-8 週

### 3. 有 ML/DL 基礎，準備學 LLM

```
Transformer 架構 → NLP 基礎 → Hugging Face → LLM 微調
```

**推薦時程：**
- Transformer 與 NLP：2-3 週
- Hugging Face 實戰：2 週
- 準備進入 LLM 領域

---

## 💡 實作項目建議

### 初級項目
1. **鳶尾花分類**：使用 Scikit-learn 實作經典分類問題
2. **房價預測**：線性回歸與特徵工程實戰
3. **手寫數字辨識**：使用 CNN 實作 MNIST

### 中級項目
1. **Kaggle 競賽參與**：選擇入門競賽（如 Titanic）
2. **圖像分類器**：使用遷移學習（ResNet、VGG）
3. **情感分析**：使用 LSTM 或 Transformer

### 進階項目
1. **物件檢測系統**：使用 YOLOv8 建立實時檢測應用
2. **圖像分割**：使用 SAM2 進行精確分割
3. **文本生成**：使用 GPT-2 或小型 LLM 進行微調

---

## 🔄 與其他章節的連結

- **往後學習**：[2.深入LLM模型工程與LLM運維](../2.深入LLM模型工程與LLM運維/)
  - 在掌握本章節的基礎後，可進入 LLM 的預訓練、微調與部署

- **應用開發**：[3.LLM應用工程](../3.LLM應用工程/)
  - 使用本章學到的知識開發實際的 AI 應用

---

## 📅 2024-2025 最新技術補充

### 深度學習框架更新
- **PyTorch 2.x**：引入 torch.compile() 提升效能
- **Keras 3**：多後端支援，統一 API
- **TensorFlow 2.16+**：改進的分散式訓練支援

### 電腦視覺最新進展
- **Segment Anything Model 2 (SAM2)**：通用影像與視訊分割
- **YOLOv10**：最新的實時物件檢測模型
- **Vision Transformer (ViT) 變體**：DeiT、Swin Transformer

### NLP 最新進展
- **Transformer 架構演進**：為理解 LLM 打基礎
- **多模態模型**：CLIP、BLIP 等視覺-語言模型
- **高效微調技術**：LoRA、QLoRA 等參數高效方法

---

## ⚡ 學習建議

1. **循序漸進**：不要跳過數學基礎，它們對理解演算法很重要
2. **動手實作**：每學一個概念都要自己寫程式碼驗證
3. **參與競賽**：Kaggle 是最好的實戰平台
4. **閱讀論文**：從經典論文開始，逐步了解領域發展
5. **建立作品集**：將學習成果整理成 GitHub 專案
6. **社群學習**：加入 AI/ML 相關社群，與他人交流

---

## 📞 相關連結

- [主目錄 README](../README.md)
- [深入LLM模型工程與LLM運維](../2.深入LLM模型工程與LLM運維/README.md)
- [LLM應用工程](../3.LLM應用工程/README.md)
- [相關的更新Blog](../4.相關的更新Blog/)

---

## 🤝 貢獻與反饋

如果您發現任何錯誤或有改進建議，歡迎提出 Issue 或 Pull Request！

---

**最後更新日期：2025-11-16**
