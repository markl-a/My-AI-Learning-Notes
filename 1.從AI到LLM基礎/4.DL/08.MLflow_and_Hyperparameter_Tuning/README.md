# MLflow 與超參數調整

本資料夾包含關於 MLflow 實驗追蹤和超參數調整的完整學習資源，從基礎概念到實際應用的完整教程。

## 目錄結構

```
08.MLflow_and_Hyperparameter_Tuning/
├── 01.MLflow_Basics/              # MLflow 基礎介紹
├── 02.Experiment_Tracking/        # 實驗追蹤進階
├── 03.Hyperparameter_Tuning/      # 超參數調整方法
├── 04.Integration_Examples/       # MLflow 與調優工具整合
├── 05.Best_Practices/             # 最佳實踐
└── README.md                      # 本文件
```

## 學習路徑

### 初學者路徑（第1-2週）

1. **MLflow 基礎** ([01.MLflow_Basics](01.MLflow_Basics/README.md))
   - 了解 MLflow 的核心概念
   - 學習基本 API 使用
   - 實踐簡單的實驗追蹤
   - 啟動並使用 MLflow UI

2. **實驗追蹤** ([02.Experiment_Tracking](02.Experiment_Tracking/README.md))
   - 學習進階追蹤技巧
   - 掌握嵌套 runs
   - 記錄複雜的參數和指標
   - 實踐可視化和分析

### 進階路徑（第3-4週）

3. **超參數調整** ([03.Hyperparameter_Tuning](03.Hyperparameter_Tuning/README.md))
   - 理解超參數調整的重要性
   - 學習不同的優化方法：
     - 網格搜索
     - 隨機搜索
     - 貝葉斯優化
     - Hyperband
   - 掌握常用工具：
     - Optuna
     - Ray Tune
     - Keras Tuner

4. **整合實踐** ([04.Integration_Examples](04.Integration_Examples/README.md))
   - MLflow + Optuna 整合
   - MLflow + Ray Tune 整合
   - MLflow + Scikit-learn 整合
   - 完整專案範例

### 專家路徑（第5週及以後）

5. **最佳實踐** ([05.Best_Practices](05.Best_Practices/README.md))
   - 實驗組織規範
   - 團隊協作流程
   - 生產環境部署
   - 模型版本控制
   - 性能優化技巧

## 快速開始

### 環境設置

```bash
# 安裝必要的套件
pip install mlflow optuna scikit-learn pandas numpy matplotlib

# 啟動 MLflow UI
mlflow ui

# 瀏覽器訪問
# http://localhost:5000
```

### 第一個 MLflow 實驗

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 載入資料
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 設定實驗
mlflow.set_experiment("my_first_experiment")

# 開始 run
with mlflow.start_run():
    # 訓練模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 評估
    accuracy = model.score(X_test, y_test)

    # 記錄參數和指標
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # 記錄模型
    mlflow.sklearn.log_model(model, "model")

    print(f"準確率: {accuracy:.4f}")
```

### 第一個超參數調整實驗

```python
import optuna
import mlflow

def objective(trial):
    with mlflow.start_run(nested=True):
        # 建議超參數
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)

        # 記錄參數
        mlflow.log_params({
            'n_estimators': n_estimators,
            'max_depth': max_depth
        })

        # 訓練和評估
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # 記錄指標
        mlflow.log_metric('accuracy', accuracy)

        return accuracy

# 設定實驗
mlflow.set_experiment("my_first_tuning")

# 執行優化
with mlflow.start_run(run_name="optuna_optimization"):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f"最佳參數: {study.best_params}")
    print(f"最佳準確率: {study.best_value:.4f}")
```

## 核心概念

### MLflow 核心組件

1. **MLflow Tracking**
   - 記錄和查詢實驗
   - 參數、指標、產出物管理
   - 實驗對比和可視化

2. **MLflow Projects**
   - 打包程式碼為可重現格式
   - 環境管理
   - 遠程執行

3. **MLflow Models**
   - 標準化模型格式
   - 多框架支持
   - 部署工具

4. **MLflow Model Registry**
   - 集中式模型存儲
   - 版本控制
   - 階段管理（Staging、Production）

### 超參數調整方法比較

| 方法 | 效率 | 適用場景 | 優點 | 缺點 |
|------|------|----------|------|------|
| 手動調整 | 低 | 小規模探索 | 靈活 | 不系統 |
| 網格搜索 | 低 | 參數少 | 完整搜索 | 計算成本高 |
| 隨機搜索 | 中 | 參數多 | 效率較高 | 不保證最優 |
| 貝葉斯優化 | 高 | 評估成本高 | 智能搜索 | 實現複雜 |
| Hyperband | 高 | 深度學習 | 自動資源分配 | 需要定義資源 |

## 實際應用場景

### 1. 機器學習模型開發

```python
# 完整的模型開發流程
mlflow.set_experiment("customer_churn_prediction")

with mlflow.start_run():
    # 資料準備
    X_train, X_test, y_train, y_test = prepare_data()

    # 超參數調整
    best_params = tune_hyperparameters(X_train, y_train)

    # 訓練最終模型
    model = train_final_model(best_params, X_train, y_train)

    # 評估
    metrics = evaluate_model(model, X_test, y_test)

    # 記錄所有資訊
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
```

### 2. 深度學習訓練

```python
# PyTorch 深度學習訓練
mlflow.set_experiment("image_classification")

with mlflow.start_run():
    # 構建模型
    model = build_model(config)

    # 訓練循環
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model)
        val_loss, val_acc = validate(model)

        # 記錄每個 epoch 的指標
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, step=epoch)

    # 保存模型
    mlflow.pytorch.log_model(model, "model")
```

### 3. A/B 測試

```python
# 比較不同模型版本
models = {
    "baseline": RandomForestClassifier(),
    "optimized": GradientBoostingClassifier(),
    "ensemble": VotingClassifier([...])
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
```

## 常見問題

### Q1: MLflow 與 TensorBoard 有什麼不同？

**A**:
- **MLflow**：框架無關，支持整個 ML 生命週期，包括模型管理和部署
- **TensorBoard**：主要用於 TensorFlow，專注於訓練過程可視化

### Q2: 應該選擇哪種超參數調整方法？

**A**:
- 小規模（< 3 個參數）：網格搜索
- 中規模（3-10 個參數）：隨機搜索或貝葉斯優化
- 大規模/深度學習：Hyperband + Optuna
- 計算資源有限：隨機搜索 + 提前停止

### Q3: 如何在團隊中共享 MLflow 實驗？

**A**: 設置共享的 MLflow 追蹤伺服器：

```bash
# 伺服器端
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0

# 客戶端
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Q4: 如何確保實驗的可重現性？

**A**:
1. 記錄所有超參數
2. 設置隨機種子
3. 記錄資料版本/hash
4. 記錄環境資訊（Python版本、套件版本）
5. 使用 MLflow Projects

### Q5: MLflow 佔用多少存儲空間？

**A**: 取決於記錄的內容：
- 參數和指標：很小（KB 級別）
- 模型文件：根據模型大小（MB 到 GB）
- 產出物：根據文件大小

建議：
- 定期清理不需要的實驗
- 使用遠程存儲（S3、Azure Blob）
- 只保存重要模型

## 學習資源

### 官方文檔

- [MLflow 官方文檔](https://mlflow.org/docs/latest/index.html)
- [Optuna 文檔](https://optuna.readthedocs.io/)
- [Ray Tune 文檔](https://docs.ray.io/en/latest/tune/index.html)
- [Keras Tuner 文檔](https://keras.io/keras_tuner/)

### 推薦閱讀

- **論文**:
  - "Random Search for Hyper-Parameter Optimization" - Bergstra & Bengio (2012)
  - "Practical Bayesian Optimization of Machine Learning Algorithms" - Snoek et al. (2012)
  - "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" - Li et al. (2017)

- **書籍**:
  - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron
  - "Machine Learning Engineering" - Andriy Burkov

### 實踐專案

1. **Kaggle 競賽**：使用 MLflow 追蹤實驗
2. **開源貢獻**：為 MLflow/Optuna 貢獻程式碼
3. **個人專案**：構建完整的 ML 流水線

## 進階主題

### 1. 分散式訓練

- 使用 Ray 進行分散式超參數調整
- MLflow 追蹤分散式實驗
- 資源管理和調度

### 2. AutoML

- 結合 TPOT、Auto-sklearn
- 自動化特徵工程
- 神經架構搜索（NAS）

### 3. MLOps

- CI/CD 整合
- 模型監控和漂移檢測
- A/B 測試框架
- 模型治理

## 貢獻指南

歡迎貢獻！如果你發現任何問題或有改進建議：

1. 提出 Issue 描述問題
2. Fork 專案並建立分支
3. 提交 Pull Request

## 更新日誌

- **2024-11**: 初始版本
  - MLflow 基礎教程
  - 超參數調整方法
  - 整合範例
  - 最佳實踐

## 授權

本資料採用 MIT 授權。

---

## 快速導航

- **基礎學習**: [MLflow 基礎](01.MLflow_Basics/README.md) → [實驗追蹤](02.Experiment_Tracking/README.md)
- **進階技巧**: [超參數調整](03.Hyperparameter_Tuning/README.md) → [整合範例](04.Integration_Examples/README.md)
- **實踐應用**: [最佳實踐](05.Best_Practices/README.md)

開始你的 MLflow 和超參數調整學習之旅吧！🚀
