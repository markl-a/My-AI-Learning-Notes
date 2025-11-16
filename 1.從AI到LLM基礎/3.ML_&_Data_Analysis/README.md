## 機器學習與數據分析完整指南

本資料夾提供從基礎到進階的機器學習與數據分析學習路徑，結合傳統ML算法、現代工具鏈與最佳實踐。

**📌 2024-2025 更新亮點**
- ✅ 添加現代數據處理工具 (Polars, DuckDB, Vaex)
- ✅ 整合最新ML框架 (CatBoost, NGBoost, FLAML)
- ✅ 擴展分散式計算 (Ray, Dask, Modin)
- ✅ 強化MLOps實踐 (MLflow, Weights & Biases, DVC)
- ✅ 加入AutoML工具 (AutoGluon, H2O, PyCaret)
- ✅ 更新實戰案例與端到端項目

---

### 0_Introduction_and_Environment_Setup
- 說明文件：對本資料夾內容架構的簡介與學習路徑說明
- 環境設定指南（Python、Conda、Notebook 環境、資料庫連接）
- 常用套件介紹（NumPy、Pandas、Polars、Matplotlib、Seaborn、Plotly、Scikit-learn、Spark 等）
- **新增**：容器化環境設置 (Docker、Poetry、uv)
- **新增**：現代開發工具 (JupyterLab, VS Code, Cursor)

### 1_Data_Acquisition_and_Analysis
- **Data_Sources.md**：描述常見資料來源（CSV、SQL 資料庫、NoSQL、API 爬取、雲端資料湖）
- **新增**：現代數據處理工具 (Polars, DuckDB, Vaex) 高效處理大規模數據
- **Data_Cleaning_and_Processing.ipynb**：缺失值處理、異常值檢測、類別特徵編碼、數值轉換
- **Exploratory_Data_Analysis.ipynb** (EDA)：統計量、相關係數、繪圖（箱型圖、直方圖、散佈圖）、資料分佈、主成分分析 (PCA) 用於初步降維探索
- **新增**：互動式可視化 (Plotly, Altair, hvPlot)
- **Feature_Engineering.ipynb**：特徵選擇、特徵縮放、特徵組合、類別特徵 One-Hot Encoding、Target Encoding、Embeddings
- **新增**：自動化特徵工程 (FeatureTools, AutoFeat)

### 2_Traditional_ML_Algorithms
- **Linear_Models.ipynb**：線性回歸、羅吉斯迴歸、Ridge、Lasso、ElasticNet
- **Tree_Based_Models.ipynb**：決策樹、隨機森林、梯度提升樹 (XGBoost、LightGBM、**CatBoost**)
- **新增**：NGBoost、HistGradientBoosting 等現代提升算法
- **SVM_and_KNN.ipynb**：支援向量機、k 最近鄰、進階核函數技巧
- **Naive_Bayes.ipynb**：樸素貝葉斯分類器、變體與應用
- **新增**：集成學習進階 (Stacking, Blending, Voting)
- **Model_Evaluation_and_Metrics.ipynb**：訓練/驗證/測試集劃分、Cross-Validation、Accuracy、Precision、Recall、F1 Score、AUC、MSE、MAE、R²
- **新增**：進階評估指標 (Log Loss, Cohen's Kappa, MCC)
- **Model_Selection_and_Parameter_Tuning.ipynb**：GridSearchCV、RandomSearch、Bayesian Optimization (Optuna)
- **新增**：AutoML工具實戰 (AutoGluon, PyCaret, H2O AutoML, FLAML)

### 3_Kaggle_Case_Studies
- **Kaggle_Competition_1.ipynb**：以 Kaggle 常見入門競賽（如泰坦尼克生存預測）為例，從 EDA、Feature Engineering、模型訓練到提交預測結果
- **Kaggle_Competition_2.ipynb**：進階資料集（如房價預測），比較不同模型表現、使用集成方法提升準確率
- **新增**：表格數據競賽最佳實踐 (2024 Top Solutions)
- **新增**：時間序列預測案例 (Store Sales Forecasting)
- **Kaggle_Tips_and_Tricks.md**：整理常用方法、特徵工程想法、加強模型泛化的策略
- **新增**：Kaggle Notebooks GPU/TPU 加速技巧
- **新增**：模型融合與後處理技術

### 4_Distributed_Computing_and_BigData_Processing
- **Introduction_to_Distributed_Computing.md**：分散式運算的概念與大數據生態系（Hadoop、Spark）
- **Spark_Basics.ipynb**：使用 PySpark 進行資料讀取、轉換、簡單分析
- **新增**：現代分散式工具 - **Ray** (分散式ML與超參數調優)
- **新增**：**Dask** 入門 (Pandas-like並行計算)
- **新增**：**Modin** (加速Pandas操作)
- **Scaling_ML_Models.ipynb**：在分散式環境中訓練 ML 模型（Spark MLlib、Ray Train）
- **新增**：分散式超參數優化 (Ray Tune, Optuna Distributed)
- **ETL_Pipeline_Integration.md**：ETL 流程概念、將數據處理管線整合到日常工作中
- **新增**：現代數據編排工具 (Prefect, Dagster, Apache Airflow 2.0+)
- **Data_Pipeline_Demo.ipynb**：展示如何在分散式環境中執行從資料擷取、清理到模型訓練的流程
- **新增**：Delta Lake 與數據湖最佳實踐

### 5_Best_Practices_and_MLOps_Basics
- **Model_Deployment_Introduction.md**：基本部署模型觀念（非深度學習模型的 Web API 建置）
- **新增**：容器化部署實戰 (Docker, Docker Compose)
- **新增**：Kubernetes 入門與模型服務部署 (Seldon Core, KServe)
- **新增**：雲端部署選項 (AWS SageMaker, Google Vertex AI, Azure ML)
- **Version_Control_and_Experiment_Tracking.md**：DVC、MLflow、Git 等工具簡介
- **新增**：**Weights & Biases** 完整實踐
- **新增**：**Neptune.ai** 與 Comet.ml 替代方案
- **新增**：DVC 數據版本控制進階用法
- **Scalability_and_Performance_Tuning.md**：模型與數據的效能調校方法
- **新增**：模型優化技術 (Quantization, Pruning, Knowledge Distillation)
- **新增**：推理加速 (ONNX Runtime, TensorRT, OpenVINO)
- **新增**：CI/CD for ML (GitHub Actions, GitLab CI)
- **新增**：模型監控與漂移檢測 (Evidently AI, Alibi Detect)
- **新增**：特徵存儲 (Feast, Tecton)

---

## 學習路徑建議

### 🎯 初學者路徑 (0-3個月)
1. 從 `0_Introduction_and_Environment_Setup` 開始，設置開發環境
2. 學習 `1_Data_Acquisition_and_Analysis` 的基礎數據處理
3. 掌握 `2_Traditional_ML_Algorithms` 中的基本算法
4. 嘗試 `3_Kaggle_Case_Studies` 中的入門競賽

### 🚀 進階路徑 (3-6個月)
1. 深入學習進階特徵工程技巧
2. 掌握集成學習與模型融合
3. 學習 `4_Distributed_Computing` 處理大規模數據
4. 開始使用 AutoML 工具加速實驗

### 💼 專業路徑 (6-12個月)
1. 完整掌握 `5_Best_Practices_and_MLOps_Basics`
2. 實施端到端的 ML pipeline
3. 學習模型部署與監控
4. 參與進階 Kaggle 競賽並分析 Top Solutions

---

## 實用資源推薦

### 📚 書籍
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition, 2022)
- *Designing Machine Learning Systems* by Chip Huyen (2022)
- *Machine Learning Engineering* by Andriy Burkov (2020)
- *Feature Engineering for Machine Learning* by Alice Zheng & Amanda Casari

### 🎓 線上課程
- **Coursera**: Machine Learning Specialization (Andrew Ng, 2022 Updated)
- **Fast.ai**: Practical Deep Learning for Coders
- **DeepLearning.AI**: MLOps Specialization
- **Kaggle Learn**: Free micro-courses on ML topics

### 🛠️ 工具與平台
- **開發環境**: JupyterLab, VS Code + Jupyter Extension, Google Colab, Kaggle Notebooks
- **版本控制**: Git + GitHub/GitLab
- **實驗追蹤**: Weights & Biases, MLflow, Neptune.ai
- **雲端平台**: AWS, GCP, Azure (免費額度)
- **資料集**: Kaggle, UCI ML Repository, Hugging Face Datasets

### 🌟 社群與論壇
- **Kaggle Discussion Forums**: 學習競賽技巧與討論
- **Reddit**: r/MachineLearning, r/datascience, r/kaggle
- **Discord/Slack**: MLOps Community, Weights & Biases Community
- **Medium/Towards Data Science**: 最新技術文章
- **arXiv.org**: 機器學習最新論文

---

## 實戰項目建議

### 初級項目
1. **預測房價**: 使用回歸模型預測房價 (Boston Housing, California Housing)
2. **客戶流失預測**: 分類問題，預測客戶是否會流失
3. **產品推薦系統**: 基於協同過濾的簡單推薦系統

### 中級項目
1. **信用卡欺詐檢測**: 處理不平衡數據的分類問題
2. **時間序列預測**: 銷售預測或股價預測
3. **文本分類**: 新聞分類或情感分析 (結合 NLP)

### 高級項目
1. **端到端 ML Pipeline**: 從數據收集到模型部署的完整流程
2. **AutoML 框架比較**: 比較不同 AutoML 工具的性能
3. **生產級模型服務**: 使用 Docker + Kubernetes 部署模型
4. **A/B Testing 框架**: 建立模型 A/B 測試系統

---

## 常見問題 FAQ

**Q: 我應該選擇哪個 ML 庫？**
A: 對於表格數據，scikit-learn 是基礎；XGBoost/LightGBM/CatBoost 用於競賽和生產；AutoML 工具(如 AutoGluon)適合快速原型開發。

**Q: 如何選擇合適的算法？**
A: 從簡單模型開始(線性回歸/邏輯回歸)建立 baseline，然後嘗試樹模型(Random Forest, XGBoost)，最後考慮集成學習和 AutoML。

**Q: 學習 ML 需要多久？**
A: 基礎概念 2-3 個月，實戰能力 6-12 個月，精通需要持續實踐和學習。

**Q: 是否需要學習數學？**
A: 基本的線性代數、微積分和統計學有助於深入理解，但實作可以先從應用開始。

**Q: 如何在 Kaggle 上取得好成績？**
A: 1) 做好 EDA 和特徵工程 2) 嘗試多種模型 3) 使用集成學習 4) 學習 Top Solutions 5) 持續實踐

---

## 更新紀錄

### 2024-11 更新
- ✅ 添加 2024-2025 最新技術和工具
- ✅ 整合現代數據處理框架 (Polars, DuckDB)
- ✅ 擴展分散式計算內容 (Ray, Dask)
- ✅ 強化 MLOps 實踐和工具鏈
- ✅ 加入 AutoML 工具完整指南
- ✅ 更新實戰案例和最佳實踐

### 持續更新中...
歡迎提出建議和改進意見！

