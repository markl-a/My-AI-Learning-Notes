# MLflow 和超參數調整最佳實踐

## 目錄
- [實驗組織](#實驗組織)
- [參數命名規範](#參數命名規範)
- [資源管理](#資源管理)
- [模型版本控制](#模型版本控制)
- [性能優化](#性能優化)
- [團隊協作](#團隊協作)
- [生產環境部署](#生產環境部署)

## 實驗組織

### 1. 使用有意義的實驗和 Run 名稱

```python
# 好的實踐
mlflow.set_experiment("ecommerce_recommendation_2024_q1")

with mlflow.start_run(run_name="baseline_collaborative_filtering"):
    pass

with mlflow.start_run(run_name="optimized_matrix_factorization_v2"):
    pass

# 不好的實踐
mlflow.set_experiment("test")
with mlflow.start_run(run_name="run1"):
    pass
```

### 2. 使用標籤進行分類

```python
# 按多個維度組織實驗
mlflow.set_tags({
    # 專案資訊
    "project": "customer_churn_prediction",
    "team": "ml_team",

    # 技術細節
    "framework": "pytorch",
    "model_family": "transformer",

    # 實驗類型
    "experiment_type": "hyperparameter_tuning",
    "optimization_method": "optuna",

    # 版本資訊
    "data_version": "v2.3",
    "code_version": "abc123",

    # 業務資訊
    "priority": "high",
    "status": "production_candidate"
})
```

### 3. 使用嵌套 Runs 組織複雜實驗

```python
# 父 Run：整體優化任務
with mlflow.start_run(run_name="hyperparameter_optimization"):
    mlflow.log_param("optimization_method", "optuna")
    mlflow.log_param("total_trials", 100)

    # 子 Run：每個試驗
    for trial_num in range(100):
        with mlflow.start_run(run_name=f"trial_{trial_num}", nested=True):
            # 記錄該試驗的參數和結果
            pass

    # 記錄最佳結果
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)
```

## 參數命名規範

### 1. 使用清晰描述性的名稱

```python
# 好的實踐
mlflow.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "optimizer_type": "adam",
    "dropout_rate": 0.5,
    "num_hidden_layers": 3,
    "hidden_layer_size": 128
})

# 不好的實踐
mlflow.log_params({
    "lr": 0.001,
    "bs": 32,
    "ep": 100,
    "opt": "adam",
    "dr": 0.5,
    "nl": 3,
    "hs": 128
})
```

### 2. 使用一致的命名風格

```python
# 選擇一種風格並保持一致

# 下劃線風格（推薦用於 Python）
mlflow.log_params({
    "max_depth": 10,
    "min_samples_split": 2,
    "learning_rate": 0.01
})

# 駝峰式（如果團隊使用）
mlflow.log_params({
    "maxDepth": 10,
    "minSamplesSplit": 2,
    "learningRate": 0.01
})
```

### 3. 分組相關參數

```python
# 使用前綴分組相關參數
mlflow.log_params({
    # 資料相關
    "data_train_size": 10000,
    "data_val_size": 2000,
    "data_test_size": 2000,
    "data_augmentation": True,

    # 模型架構
    "model_type": "resnet50",
    "model_pretrained": True,
    "model_freeze_layers": 40,

    # 訓練相關
    "train_batch_size": 32,
    "train_epochs": 100,
    "train_learning_rate": 0.001,
    "train_optimizer": "adam",

    # 正則化
    "reg_l2_lambda": 0.001,
    "reg_dropout_rate": 0.5,
    "reg_early_stopping_patience": 10
})
```

## 資源管理

### 1. 設定合理的資源限制

```python
import optuna

def objective(trial):
    # 估算資源需求
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 128, 1024)

    # 估算內存使用
    estimated_memory_gb = (batch_size * hidden_size * 4) / (1024**3)

    # 如果超過限制，直接剪枝
    if estimated_memory_gb > 16:  # 16GB 限制
        raise optuna.TrialPruned()

    # 繼續訓練
    return train_model(batch_size, hidden_size)
```

### 2. 使用提前停止節省資源

```python
def objective(trial):
    model = build_model(trial)

    for epoch in range(100):
        val_loss = train_epoch(model)

        # 報告中間結果
        trial.report(val_loss, epoch)

        # 如果表現不佳，提前停止
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss
```

### 3. 並行化優化

```python
# 使用多進程並行優化
study = optuna.create_study(
    direction='maximize',
    storage='sqlite:///optuna.db',  # 共享存儲
    study_name='parallel_optimization',
    load_if_exists=True
)

# 在多個進程中運行
# 進程 1
study.optimize(objective, n_trials=50)

# 進程 2（同時運行）
study.optimize(objective, n_trials=50)

# 進程 3（同時運行）
study.optimize(objective, n_trials=50)
```

## 模型版本控制

### 1. 使用 Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

with mlflow.start_run():
    # 訓練模型
    model = train_model()

    # 記錄模型
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="customer_churn_predictor"
    )

    run_id = mlflow.active_run().info.run_id

# 轉換模型階段
model_version = client.get_latest_versions("customer_churn_predictor")[0].version

client.transition_model_version_stage(
    name="customer_churn_predictor",
    version=model_version,
    stage="Staging"
)

# 添加描述
client.update_model_version(
    name="customer_churn_predictor",
    version=model_version,
    description="使用 Optuna 優化的 RandomForest 模型，驗證準確率 95%"
)
```

### 2. 記錄模型元資料

```python
with mlflow.start_run():
    # 訓練模型
    model = train_model()

    # 記錄完整的模型資訊
    mlflow.log_params({
        "model_type": type(model).__name__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version
    })

    # 記錄模型大小
    import joblib
    joblib.dump(model, "temp_model.pkl")
    model_size_mb = os.path.getsize("temp_model.pkl") / (1024 * 1024)
    mlflow.log_metric("model_size_mb", model_size_mb)

    # 記錄模型複雜度
    if hasattr(model, 'n_estimators'):
        mlflow.log_param("n_estimators", model.n_estimators)

    # 記錄訓練資料資訊
    mlflow.log_params({
        "n_training_samples": len(X_train),
        "n_features": X_train.shape[1],
        "data_hash": hashlib.md5(X_train.tobytes()).hexdigest()
    })
```

### 3. 版本比較

```python
def compare_models(model_name, versions):
    """比較不同版本的模型"""
    client = MlflowClient()
    results = []

    for version in versions:
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)

        # 評估模型
        metrics = evaluate_model(model, X_test, y_test)

        # 獲取模型元資料
        model_version = client.get_model_version(model_name, version)

        results.append({
            "version": version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            **metrics
        })

    return pd.DataFrame(results)

# 使用
comparison = compare_models("customer_churn_predictor", [1, 2, 3, 4])
print(comparison)
```

## 性能優化

### 1. 批量記錄指標

```python
# 好的實踐：批量記錄
metrics_batch = {
    f"fold_{i}_accuracy": acc
    for i, acc in enumerate(fold_accuracies)
}
mlflow.log_metrics(metrics_batch)

# 不好的實踐：逐個記錄
for i, acc in enumerate(fold_accuracies):
    mlflow.log_metric(f"fold_{i}_accuracy", acc)
```

### 2. 使用 Autolog

```python
# 啟用自動記錄，減少手動程式碼
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch

# Scikit-learn
mlflow.sklearn.autolog()

# TensorFlow/Keras
mlflow.tensorflow.autolog()

# PyTorch
mlflow.pytorch.autolog()

# 訓練模型時會自動記錄
model.fit(X_train, y_train)
```

### 3. 異步記錄

```python
import asyncio
import mlflow

async def log_metrics_async(metrics):
    """異步記錄指標"""
    await asyncio.to_thread(mlflow.log_metrics, metrics)

async def train_with_async_logging():
    for epoch in range(100):
        metrics = train_epoch()

        # 異步記錄，不阻塞訓練
        asyncio.create_task(log_metrics_async(metrics))
```

## 團隊協作

### 1. 共享 MLflow 追蹤伺服器

```bash
# 啟動 MLflow 伺服器
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

```python
# 團隊成員連接到共享伺服器
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("shared_project")
```

### 2. 使用標準化的實驗模板

```python
# experiment_template.py

import mlflow
from typing import Dict, Any

class ExperimentTemplate:
    """標準化的實驗模板"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def run_experiment(
        self,
        run_name: str,
        params: Dict[str, Any],
        train_fn,
        tags: Dict[str, str] = None
    ):
        """執行標準化的實驗"""
        with mlflow.start_run(run_name=run_name):
            # 記錄參數
            mlflow.log_params(params)

            # 記錄標籤
            if tags:
                mlflow.set_tags(tags)

            # 記錄環境資訊
            self._log_environment()

            # 訓練模型
            model, metrics = train_fn(params)

            # 記錄指標
            mlflow.log_metrics(metrics)

            # 記錄模型
            mlflow.sklearn.log_model(model, "model")

            return model, metrics

    def _log_environment(self):
        """記錄環境資訊"""
        import sys
        import platform

        mlflow.set_tags({
            "python_version": sys.version,
            "platform": platform.platform(),
            "user": os.getenv("USER")
        })

# 使用
template = ExperimentTemplate("team_project")

template.run_experiment(
    run_name="experiment_1",
    params={"learning_rate": 0.01},
    train_fn=my_train_function,
    tags={"team_member": "Alice"}
)
```

### 3. 程式碼審查檢查清單

建立實驗程式碼審查檢查清單：

```markdown
## MLflow 實驗程式碼審查檢查清單

### 必須項
- [ ] 使用有意義的實驗名稱
- [ ] 使用描述性的 run 名稱
- [ ] 記錄所有超參數
- [ ] 記錄關鍵指標
- [ ] 記錄模型
- [ ] 添加適當的標籤

### 推薦項
- [ ] 記錄資料版本/hash
- [ ] 記錄環境資訊
- [ ] 使用模型簽名
- [ ] 添加模型描述
- [ ] 記錄訓練時間
- [ ] 生成並記錄可視化

### 高級項
- [ ] 使用嵌套 runs 組織複雜實驗
- [ ] 實施錯誤處理
- [ ] 記錄完整的設定檔
- [ ] 添加單元測試
```

## 生產環境部署

### 1. 模型部署流程

```python
from mlflow.tracking import MlflowClient

class ModelDeploymentPipeline:
    """模型部署流水線"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = MlflowClient()

    def promote_to_staging(self, version: int):
        """提升模型到 Staging"""
        # 運行驗證測試
        if self._validate_model(version):
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Staging"
            )
            print(f"模型 {self.model_name} v{version} 已提升到 Staging")
        else:
            raise ValueError("模型驗證失敗")

    def promote_to_production(self, version: int):
        """提升模型到 Production"""
        # 獲取當前 Production 模型
        current_prod = self.client.get_latest_versions(
            self.model_name,
            stages=["Production"]
        )

        # 運行 A/B 測試
        if self._ab_test(version, current_prod):
            # 歸檔舊的 Production 模型
            if current_prod:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=current_prod[0].version,
                    stage="Archived"
                )

            # 提升新模型
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production"
            )
            print(f"模型 {self.model_name} v{version} 已提升到 Production")
        else:
            raise ValueError("A/B 測試失敗")

    def _validate_model(self, version: int) -> bool:
        """驗證模型"""
        model_uri = f"models:/{self.model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)

        # 運行驗證測試
        accuracy = evaluate_model(model, validation_data)

        # 檢查性能閾值
        return accuracy > 0.90

    def _ab_test(self, new_version: int, current_version) -> bool:
        """A/B 測試"""
        # 實施 A/B 測試邏輯
        pass

# 使用
pipeline = ModelDeploymentPipeline("customer_churn_predictor")
pipeline.promote_to_staging(version=5)
pipeline.promote_to_production(version=5)
```

### 2. 模型監控

```python
import mlflow
from datetime import datetime

class ModelMonitor:
    """生產環境模型監控"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def log_prediction(self, prediction, features, actual=None):
        """記錄預測結果"""
        with mlflow.start_run(run_name=f"prediction_{datetime.now()}"):
            # 記錄預測
            mlflow.log_metric("prediction", prediction)

            # 如果有實際值，記錄誤差
            if actual is not None:
                error = abs(prediction - actual)
                mlflow.log_metric("prediction_error", error)

            # 記錄輸入特徵
            for i, feature_value in enumerate(features):
                mlflow.log_param(f"feature_{i}", feature_value)

    def check_drift(self):
        """檢查模型漂移"""
        # 獲取最近的預測
        recent_predictions = self._get_recent_predictions()

        # 計算性能指標
        current_accuracy = calculate_accuracy(recent_predictions)

        # 與基準比較
        baseline_accuracy = self._get_baseline_accuracy()

        if current_accuracy < baseline_accuracy * 0.95:  # 下降超過5%
            self._alert_drift(current_accuracy, baseline_accuracy)

    def _alert_drift(self, current, baseline):
        """發送模型漂移警報"""
        print(f"警告：模型性能下降！")
        print(f"當前準確率: {current:.4f}")
        print(f"基準準確率: {baseline:.4f}")

        # 發送通知（郵件、Slack等）
        send_alert(f"模型 {self.model_name} 性能下降")
```

### 3. 回滾策略

```python
def rollback_model(model_name: str):
    """回滾模型到上一個版本"""
    client = MlflowClient()

    # 獲取當前 Production 模型
    current_prod = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )[0]

    # 獲取 Archived 中的上一個版本
    archived = client.get_latest_versions(
        model_name,
        stages=["Archived"]
    )

    if archived:
        previous_version = archived[0]

        # 歸檔當前 Production 模型
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod.version,
            stage="Archived"
        )

        # 恢復上一個版本到 Production
        client.transition_model_version_stage(
            name=model_name,
            version=previous_version.version,
            stage="Production"
        )

        print(f"已回滾到版本 {previous_version.version}")
    else:
        print("沒有可用的備份版本")
```

## 安全和合規

### 1. 敏感資訊管理

```python
# 不要直接記錄敏感資訊
# 錯誤做法
mlflow.log_param("api_key", "secret_key_123")
mlflow.log_param("database_password", "password")

# 正確做法：使用環境變量
import os
mlflow.set_tag("api_key_source", "env_var")
mlflow.set_tag("db_host", os.getenv("DB_HOST"))  # 只記錄主機名，不記錄密碼
```

### 2. 資料隱私

```python
# 記錄資料統計而非原始資料
mlflow.log_params({
    "n_samples": len(data),
    "n_features": data.shape[1],
    "data_hash": hashlib.sha256(str(data).encode()).hexdigest()
})

# 不要記錄原始資料
# mlflow.log_artifact("sensitive_data.csv")  # 錯誤
```

## 總結

遵循這些最佳實踐可以幫助你：

1. **組織性**：保持實驗井然有序，易於查找和比較
2. **可重現性**：確保所有實驗都可以完全重現
3. **協作性**：促進團隊成員之間的協作
4. **生產就緒**：確保模型可以安全地部署到生產環境
5. **可維護性**：使程式碼和實驗易於維護和擴展

記住：好的實踐需要時間建立，但長期來看會大大提高工作效率！
