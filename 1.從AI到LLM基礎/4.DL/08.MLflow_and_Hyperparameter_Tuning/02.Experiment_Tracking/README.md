# MLflow 實驗追蹤進階

## 目錄
- [實驗追蹤概述](#實驗追蹤概述)
- [進階追蹤技巧](#進階追蹤技巧)
- [實際範例](#實際範例)
- [最佳實踐](#最佳實踐)

## 實驗追蹤概述

實驗追蹤是 MLflow 最核心的功能之一，它幫助我們系統化地記錄、組織和比較機器學習實驗。

### 為什麼需要實驗追蹤？

1. **可重現性**：記錄所有參數和環境，確保實驗可以重現
2. **比較分析**：輕鬆比較不同實驗的結果
3. **協作**：團隊成員可以查看彼此的實驗結果
4. **決策支持**：基於資料做出模型選擇決策
5. **知識積累**：積累實驗經驗，避免重複工作

## 進階追蹤技巧

### 1. 記錄嵌套 Runs

嵌套 runs 適合記錄複雜的訓練流程，如交叉驗證或多階段訓練。

```python
import mlflow

# 父 run：整體實驗
with mlflow.start_run(run_name="cross_validation_experiment") as parent_run:
    mlflow.log_param("cv_folds", 5)

    # 子 run：每個 fold
    for fold in range(5):
        with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
            mlflow.log_param("fold_number", fold)

            # 訓練和評估
            train_model()
            accuracy = evaluate_model()

            mlflow.log_metric("accuracy", accuracy)

    # 在父 run 中記錄平均結果
    avg_accuracy = calculate_average()
    mlflow.log_metric("avg_accuracy", avg_accuracy)
```

### 2. 記錄時間序列指標

在訓練過程中記錄指標的變化，使用 `step` 參數。

```python
import mlflow

with mlflow.start_run():
    for epoch in range(100):
        # 訓練一個 epoch
        train_loss = train_one_epoch()
        val_loss = validate()

        # 記錄指標，指定 step
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
```

### 3. 記錄複雜參數

對於複雜的參數配置，可以使用 JSON 格式。

```python
import mlflow
import json

with mlflow.start_run():
    # 簡單參數
    mlflow.log_param("learning_rate", 0.001)

    # 複雜配置
    config = {
        "architecture": {
            "layers": [128, 64, 32],
            "activation": "relu",
            "dropout": 0.5
        },
        "optimizer": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999
        }
    }

    # 將配置轉為字串記錄
    mlflow.log_param("model_config", json.dumps(config))

    # 或者記錄為 artifact
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    mlflow.log_artifact("config.json")
```

### 4. 記錄圖表和可視化

```python
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

with mlflow.start_run():
    # 生成並保存圖表
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig("loss_curve.png")
    plt.close()

    # 記錄圖表
    mlflow.log_artifact("loss_curve.png")

    # 記錄混淆矩陣
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")
```

### 5. 記錄模型簽名

模型簽名定義了模型的輸入和輸出格式，確保模型部署時的正確性。

```python
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

with mlflow.start_run():
    # 訓練模型
    model.fit(X_train, y_train)

    # 推斷簽名
    predictions = model.predict(X_train[:5])
    signature = infer_signature(X_train, predictions)

    # 記錄模型with簽名
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature
    )
```

### 6. 使用標籤組織實驗

```python
import mlflow

with mlflow.start_run():
    # 設定多個標籤
    mlflow.set_tags({
        "team": "ml_team",
        "project": "recommendation_system",
        "model_type": "collaborative_filtering",
        "data_version": "v2.0",
        "experiment_type": "baseline",
        "priority": "high"
    })
```

## 實際範例

### 範例 1: 完整的深度學習訓練追蹤

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_with_mlflow(train_loader, val_loader):
    # 設定實驗
    mlflow.set_experiment("pytorch_classification")

    # 超參數
    input_size = 784
    hidden_size = 128
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 64

    with mlflow.start_run(run_name="simple_nn_baseline"):
        # 記錄參數
        mlflow.log_params({
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
        })

        # 初始化模型
        model = SimpleNN(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 訓練循環
        for epoch in range(num_epochs):
            # 訓練階段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.reshape(-1, input_size)

                # 前向傳播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 統計
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 計算訓練指標
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # 驗證階段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.reshape(-1, input_size)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # 計算驗證指標
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            # 記錄每個 epoch 的指標
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')

        # 記錄最終模型
        mlflow.pytorch.log_model(model, "model")

        # 記錄最終指標
        mlflow.log_metric("final_val_accuracy", val_accuracy)

        # 設定標籤
        mlflow.set_tags({
            "framework": "pytorch",
            "model_type": "feedforward_nn",
            "dataset": "mnist"
        })

        print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### 範例 2: 使用 Autolog 簡化追蹤

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

# 啟用自動記錄
mlflow.sklearn.autolog()

# 載入資料
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# 設定實驗
mlflow.set_experiment("sklearn_autolog_demo")

# 訓練模型 - autolog 會自動記錄所有內容
with mlflow.start_run(run_name="rf_autolog"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 預測並生成報告
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # 手動記錄額外資訊
    mlflow.log_text(report, "classification_report.txt")
```

### 範例 3: 比較多個模型

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 載入資料
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 設定實驗
mlflow.set_experiment("model_comparison")

# 定義要比較的模型
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

# 訓練和評估每個模型
results = []

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # 記錄模型類型
        mlflow.log_param("model_type", model_name)

        # 訓練模型
        model.fit(X_train, y_train)

        # 預測
        y_pred = model.predict(X_test)

        # 計算指標
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # 記錄所有指標
        mlflow.log_metrics(metrics)

        # 記錄模型
        mlflow.sklearn.log_model(model, "model")

        # 設定標籤
        mlflow.set_tag("dataset", "breast_cancer")

        # 保存結果
        results.append({
            "model": model_name,
            **metrics
        })

        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")

# 顯示比較結果
import pandas as pd
results_df = pd.DataFrame(results)
print("\n模型比較結果:")
print(results_df.to_string(index=False))
```

### 範例 4: 追蹤資料版本和特徵工程

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def get_data_hash(df):
    """計算資料的 hash 值，用於追蹤資料版本"""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def track_feature_engineering():
    mlflow.set_experiment("feature_engineering_tracking")

    with mlflow.start_run(run_name="feature_engineering_v1"):
        # 載入原始資料
        df = pd.read_csv("data.csv")

        # 記錄原始資料資訊
        mlflow.log_params({
            "raw_data_shape": str(df.shape),
            "raw_data_hash": get_data_hash(df),
            "raw_features": df.shape[1]
        })

        # 特徵工程
        # 1. 處理缺失值
        df_clean = df.dropna()
        mlflow.log_param("missing_handling", "dropna")

        # 2. 特徵標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean.drop('target', axis=1))

        # 3. 特徵選擇
        k_best = 10
        selector = SelectKBest(f_classif, k=k_best)
        X_selected = selector.fit_transform(X_scaled, df_clean['target'])

        # 記錄特徵工程參數
        mlflow.log_params({
            "scaling": "StandardScaler",
            "feature_selection": "SelectKBest",
            "k_best": k_best,
            "final_features": X_selected.shape[1]
        })

        # 記錄選中的特徵
        selected_features = df_clean.drop('target', axis=1).columns[
            selector.get_support()
        ].tolist()
        mlflow.log_param("selected_features", str(selected_features))

        # 保存並記錄預處理器
        import joblib
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(selector, "selector.pkl")

        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("selector.pkl")

        # 記錄處理後的資料統計
        mlflow.log_metrics({
            "data_reduction_ratio": X_selected.shape[0] / df.shape[0],
            "feature_reduction_ratio": X_selected.shape[1] / (df.shape[1] - 1)
        })
```

## 最佳實踐

### 1. 命名規範

```python
# 實驗命名：專案_任務_模型
mlflow.set_experiment("ecommerce_recommendation_collaborative_filtering")

# Run 命名：描述性名稱
with mlflow.start_run(run_name="baseline_user_based_cf"):
    pass

with mlflow.start_run(run_name="optimized_item_based_cf_v2"):
    pass
```

### 2. 參陣列織

```python
with mlflow.start_run():
    # 分組記錄參數
    # 資料相關
    mlflow.log_params({
        "data_version": "v1.2",
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1
    })

    # 模型相關
    mlflow.log_params({
        "model_type": "ResNet50",
        "pretrained": True,
        "freeze_layers": 40
    })

    # 訓練相關
    mlflow.log_params({
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })
```

### 3. 指標記錄策略

```python
with mlflow.start_run():
    # 訓練過程中的指標
    for epoch in range(epochs):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr
        }, step=epoch)

    # 最終評估指標
    mlflow.log_metrics({
        "final_test_accuracy": test_acc,
        "final_test_f1": test_f1,
        "inference_time_ms": inference_time
    })

    # 業務指標
    mlflow.log_metrics({
        "model_size_mb": model_size,
        "training_time_minutes": training_time
    })
```

### 4. 使用上下文管理器

```python
import mlflow

class MLflowLogger:
    def __init__(self, experiment_name, run_name=None):
        self.experiment_name = experiment_name
        self.run_name = run_name

    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

# 使用
with MLflowLogger("my_experiment", "my_run") as logger:
    logger.log_params({"lr": 0.01})
    logger.log_metrics({"accuracy": 0.95})
```

### 5. 錯誤處理

```python
import mlflow

mlflow.set_experiment("robust_experiment")

try:
    with mlflow.start_run(run_name="error_handling_example"):
        mlflow.log_param("status", "running")

        # 訓練程式碼
        model = train_model()

        mlflow.log_param("status", "success")
        mlflow.log_metric("accuracy", accuracy)

except Exception as e:
    # 記錄錯誤
    mlflow.log_param("status", "failed")
    mlflow.log_param("error_message", str(e))
    mlflow.set_tag("error", True)

    # 記錄錯誤日誌
    with open("error_log.txt", "w") as f:
        f.write(str(e))
    mlflow.log_artifact("error_log.txt")

    raise
finally:
    # 清理資源
    pass
```

## 查詢和分析實驗

### 使用 MLflow Client API

```python
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# 獲取所有實驗
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

# 搜索 runs
experiment_id = "1"
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# 轉換為 DataFrame
data = []
for run in runs:
    data.append({
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", "N/A"),
        "accuracy": run.data.metrics.get("accuracy", 0),
        "f1_score": run.data.metrics.get("f1_score", 0),
        **run.data.params
    })

df = pd.DataFrame(data)
print(df)
```

### 使用 MLflow Search API

```python
import mlflow

# 設定追蹤 URI
mlflow.set_tracking_uri("http://localhost:5000")

# 搜索實驗
experiments = mlflow.search_experiments(
    filter_string="name LIKE 'image_classification%'"
)

# 搜索 runs
runs = mlflow.search_runs(
    experiment_ids=["1", "2"],
    filter_string="params.model_type = 'CNN' AND metrics.accuracy > 0.85",
    order_by=["metrics.accuracy DESC"]
)

print(runs[["run_id", "params.learning_rate", "metrics.accuracy"]])
```

## 總結

實驗追蹤是機器學習工作流程中的關鍵環節。通過 MLflow 的實驗追蹤功能：

1. 可以系統化地記錄所有實驗
2. 輕鬆比較不同配置的效果
3. 確保實驗的可重現性
4. 促進團隊協作和知識分享

下一步建議學習：
- [超參數調整](../03.Hyperparameter_Tuning/README.md)
- [整合範例](../04.Integration_Examples/README.md)
