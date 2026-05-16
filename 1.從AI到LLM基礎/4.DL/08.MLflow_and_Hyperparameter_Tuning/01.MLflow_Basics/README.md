# MLflow 基礎介紹

## 目錄
- [什麼是 MLflow](#什麼是-mlflow)
- [核心概念](#核心概念)
- [MLflow 的四大組件](#mlflow-的四大組件)
- [安裝與設定](#安裝與設定)
- [基本使用流程](#基本使用流程)

## 什麼是 MLflow

MLflow 是一個開源的機器學習生命週期管理平台，由 Databricks 開發。它提供了一套完整的工具來管理機器學習專案的整個生命週期，包括實驗追蹤、模型管理、部署等。

### 主要特點

1. **開源免費**：完全開源，可自由使用和修改
2. **框架無關**：支援任何機器學習框架（TensorFlow、PyTorch、Scikit-learn 等）
3. **語言無關**：提供 Python、R、Java、REST API 等多種介面
4. **可擴展性**：從本地開發到大規模生產環境都適用
5. **社群活躍**：擁有龐大的開發者社群和豐富的文檔

### 解決的問題

在機器學習專案中，常見的挑戰包括：

- **實驗管理混亂**：難以追蹤不同實驗的參數、指標和結果
- **模型版本控制**：難以管理和比較不同版本的模型
- **可重現性差**：難以重現之前的實驗結果
- **部署複雜**：模型從開發到生產環境的部署流程複雜
- **協作困難**：團隊成員之間難以分享和協作

MLflow 為這些問題提供了系統化的解決方案。

## 核心概念

### 1. Experiment（實驗）

實驗是一組相關運行（Runs）的集合，用於組織和管理機器學習任務。每個實驗都有唯一的名稱和 ID。

```python
import mlflow

# 設定實驗
mlflow.set_experiment("my_experiment")
```

### 2. Run（運行）

Run 是單次模型訓練的執行過程，記錄了該次執行的所有相關資訊，包括：
- 參數（Parameters）
- 指標（Metrics）
- 標籤（Tags）
- 產出物（Artifacts）

```python
with mlflow.start_run():
    # 記錄參數
    mlflow.log_param("learning_rate", 0.01)

    # 記錄指標
    mlflow.log_metric("accuracy", 0.95)

    # 記錄模型
    mlflow.sklearn.log_model(model, "model")
```

### 3. Parameters（參數）

參數是影響模型訓練的輸入值，通常是超參數，如學習率、批次大小等。參數在一次 Run 中是固定不變的。

### 4. Metrics（指標）

指標是評估模型性能的數值，如準確率、損失值等。與參數不同，指標可以在訓練過程中隨時間變化。

### 5. Artifacts（產出物）

產出物是 Run 產生的任何文件，包括：
- 訓練好的模型
- 圖表和可視化
- 資料文件
- 其他任何相關文件

### 6. Models（模型）

MLflow 提供了模型註冊表，用於管理模型的版本、階段（Staging、Production 等）和生命週期。

## MLflow 的四大組件

### 1. MLflow Tracking

**功能**：記錄和查詢實驗、參數、指標和產出物

**使用場景**：
- 記錄訓練過程中的參數和指標
- 比較不同實驗的結果
- 可視化實驗結果

**核心 API**：
```python
import mlflow

# 開始一個 Run
with mlflow.start_run():
    # 記錄參數
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 32)

    # 記錄指標
    mlflow.log_metric("train_loss", 0.5)
    mlflow.log_metric("val_accuracy", 0.92)

    # 記錄產出物
    mlflow.log_artifact("model.pkl")
```

### 2. MLflow Projects

**功能**：將機器學習程式碼打包成可重現的格式

**使用場景**：
- 確保程式碼的可重現性
- 在不同環境中運行相同的實驗
- 與團隊分享項目

**專案結構**：
```
my_project/
├── MLproject          # 專案設定檔
├── conda.yaml         # 環境依賴
├── train.py          # 訓練腳本
└── data/             # 資料目錄
```

**MLproject 文件範例**：
```yaml
name: my_project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.01}
      epochs: {type: int, default: 100}
    command: "python train.py --lr {learning_rate} --epochs {epochs}"
```

### 3. MLflow Models

**功能**：以標準格式管理和部署模型

**使用場景**：
- 將模型部署到多種服務平台
- 管理模型版本
- 在不同框架之間轉換模型

**支援的框架**：
- Scikit-learn
- TensorFlow
- PyTorch
- Keras
- XGBoost
- LightGBM
- ONNX
- 自定義模型

**模型保存範例**：
```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# 訓練模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 保存模型
mlflow.sklearn.log_model(model, "random_forest_model")
```

### 4. MLflow Model Registry

**功能**：集中式模型存儲庫，提供模型版本控制和階段管理

**使用場景**：
- 註冊和版本化模型
- 管理模型生命週期（Staging → Production）
- 協作和審核模型

**模型階段**：
- **None**：初始狀態
- **Staging**：測試階段
- **Production**：生產環境
- **Archived**：歸檔狀態

**使用範例**：
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 註冊模型
model_uri = f"runs:/{run_id}/model"
mv = client.create_model_version(
    name="my_model",
    source=model_uri,
    run_id=run_id
)

# 轉換模型階段
client.transition_model_version_stage(
    name="my_model",
    version=1,
    stage="Production"
)
```

## 安裝與設定

### 安裝 MLflow

```bash
# 基本安裝
pip install mlflow

# 安裝特定版本
pip install mlflow==2.9.0

# 安裝額外依賴
pip install mlflow[extras]  # 包含所有可選依賴
```

### 驗證安裝

```bash
mlflow --version
```

### 啟動 MLflow UI

```bash
# 本地啟動
mlflow ui

# 指定端口
mlflow ui --port 5001

# 指定追蹤目錄
mlflow ui --backend-store-uri ./mlruns

# 訪問 UI
# 瀏覽器打開：http://localhost:5000
```

### 配置追蹤伺服器

```python
import mlflow

# 設定追蹤 URI
mlflow.set_tracking_uri("http://localhost:5000")

# 或使用本地目錄
mlflow.set_tracking_uri("file:///path/to/mlruns")

# 或使用遠程伺服器
mlflow.set_tracking_uri("https://my-mlflow-server.com")
```

## 基本使用流程

### 1. 完整範例：訓練並追蹤模型

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris

# 載入資料
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 設定實驗
mlflow.set_experiment("iris_classification")

# 開始 Run
with mlflow.start_run(run_name="random_forest_v1"):
    # 設定參數
    n_estimators = 100
    max_depth = 5

    # 記錄參數
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")

    # 訓練模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 預測並評估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 記錄指標
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # 記錄模型
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=mlflow.models.infer_signature(X_train, y_train)
    )

    # 記錄標籤
    mlflow.set_tag("dataset", "iris")
    mlflow.set_tag("framework", "sklearn")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

### 2. 查看實驗結果

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 獲取實驗
experiment = client.get_experiment_by_name("iris_classification")

# 搜索 runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

# 顯示結果
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics.get('accuracy', 'N/A')}")
    print(f"Parameters: {run.data.params}")
    print("-" * 50)
```

### 3. 載入並使用模型

```python
import mlflow.sklearn

# 方法 1: 使用 run_id 載入
run_id = "your_run_id_here"
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# 方法 2: 從 Model Registry 載入
model_name = "iris_classifier"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.sklearn.load_model(model_uri)

# 方法 3: 載入生產環境模型
model_uri = f"models:/{model_name}/Production"
loaded_model = mlflow.sklearn.load_model(model_uri)

# 使用模型進行預測
predictions = loaded_model.predict(X_test)
```

## 最佳實踐

### 1. 組織實驗

```python
# 使用有意義的實驗名稱
mlflow.set_experiment("image_classification_resnet50")

# 使用有意義的 run 名稱
with mlflow.start_run(run_name="baseline_model"):
    pass
```

### 2. 記錄關鍵資訊

```python
with mlflow.start_run():
    # 記錄所有超參數
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
        "epochs": 100
    })

    # 記錄多個指標
    for epoch in range(100):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        }, step=epoch)
```

### 3. 使用標籤組織

```python
with mlflow.start_run():
    mlflow.set_tags({
        "team": "ml_team",
        "project": "customer_churn",
        "model_type": "deep_learning",
        "experiment_type": "hyperparameter_tuning"
    })
```

### 4. 記錄環境資訊

```python
import sys
import mlflow

with mlflow.start_run():
    # 記錄 Python 版本
    mlflow.set_tag("python_version", sys.version)

    # 記錄依賴包版本
    import sklearn
    mlflow.set_tag("sklearn_version", sklearn.__version__)
```

### 5. 使用自動記錄（Autolog）

```python
import mlflow

# 啟用自動記錄
mlflow.sklearn.autolog()  # Scikit-learn
mlflow.tensorflow.autolog()  # TensorFlow
mlflow.pytorch.autolog()  # PyTorch

# 訓練模型時會自動記錄參數、指標和模型
model.fit(X_train, y_train)
```

## 常見問題

### Q1: MLflow 資料存儲在哪裡？

**A**: 預設情況下，MLflow 將資料存儲在本地的 `./mlruns` 目錄中。可以通過設定 `tracking_uri` 來改變存儲位置。

### Q2: 如何在團隊中共享實驗結果？

**A**: 可以設置一個共享的 MLflow 追蹤伺服器，團隊成員連接到同一個伺服器即可共享實驗。

### Q3: MLflow 支援分散式訓練嗎？

**A**: 是的，MLflow 可以追蹤分散式訓練任務，每個進程可以記錄到同一個 run 中。

### Q4: 如何刪除實驗或 run？

**A**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 刪除 run
client.delete_run(run_id)

# 刪除實驗
client.delete_experiment(experiment_id)
```

### Q5: MLflow 與其他工具的比較？

**A**:
- **vs TensorBoard**: MLflow 支援多種框架，而 TensorBoard 主要用於 TensorFlow
- **vs Weights & Biases**: MLflow 是開源的，可以自主託管
- **vs Neptune.ai**: MLflow 更輕量，適合小型團隊和個人使用

## 相關資源

- [MLflow 官方文檔](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [MLflow 教程](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
- [MLflow API 參考](https://mlflow.org/docs/latest/python_api/index.html)

## 下一步

學習完 MLflow 基礎後，建議繼續學習：

1. [實驗追蹤進階](../02.Experiment_Tracking/README.md)
2. [超參數調整](../03.Hyperparameter_Tuning/README.md)
3. [整合範例](../04.Integration_Examples/README.md)
4. [最佳實踐](../05.Best_Practices/README.md)
