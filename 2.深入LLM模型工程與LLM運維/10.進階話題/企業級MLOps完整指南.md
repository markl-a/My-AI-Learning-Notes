# 企業級 MLOps 完整指南

> 最後更新：2025-01

## 📋 概述

本指南涵蓋企業級 MLOps 的最佳實踐，包括模型生命週期管理、自動化部署、監控告警與治理框架。

## 🎯 MLOps 成熟度模型

### 等級 0：手動流程
```
特徵：
- 手動訓練與部署
- 無版本控制
- 無監控

問題：
- 難以復現
- 部署緩慢
- 無法擴展
```

### 等級 1：ML 流水線自動化
```
特徵：
- 自動化訓練流水線
- 基礎版本控制
- 簡單監控

改進：
- 可復現性提升
- 部署時間縮短
```

### 等級 2：CI/CD 整合
```
特徵：
- 持續整合/持續部署
- 完整版本控制
- 自動化測試

改進：
- 快速迭代
- 品質保證
```

### 等級 3：完全自動化
```
特徵：
- 自動重訓練
- 自動監控與告警
- 自動回滾

改進：
- 最小人工介入
- 7x24 運營
```

## 🏗️ 企業級 MLOps 架構

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps 平台架構                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  資料層  │→│  特徵層  │→│  訓練層  │→│  部署層  │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
│       ↓            ↓            ↓            ↓              │
│  ┌─────────────────────────────────────────────────┐        │
│  │              監控與治理層                        │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 1️⃣ 資料管理

### Feature Store 實作

```python
from feast import FeatureStore, Entity, Feature, FeatureView
from feast.types import Float32, Int64, String
from datetime import timedelta

# 定義實體
user = Entity(
    name="user_id",
    value_type=String,
    description="使用者唯一識別碼"
)

# 定義特徵視圖
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    features=[
        Feature(name="total_purchases", dtype=Float32),
        Feature(name="avg_order_value", dtype=Float32),
        Feature(name="days_since_last_order", dtype=Int64),
        Feature(name="customer_segment", dtype=String),
    ],
    online=True,
    source=user_data_source,
)

# 使用特徵
store = FeatureStore("feature_repo/")

# 訓練時獲取歷史特徵
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:total_purchases",
        "user_features:avg_order_value",
        "user_features:customer_segment"
    ]
).to_df()

# 推論時獲取線上特徵
online_features = store.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:customer_segment"
    ],
    entity_rows=[{"user_id": "user_123"}]
).to_dict()
```

### 資料品質檢查

```python
import great_expectations as gx
from great_expectations.core.batch import BatchRequest

# 建立資料上下文
context = gx.get_context()

# 定義期望
expectation_suite = context.create_expectation_suite(
    expectation_suite_name="training_data_suite"
)

# 添加驗證規則
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="training_data_suite"
)

# 資料完整性檢查
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between(
    "age", min_value=0, max_value=120
)

# 資料分佈檢查
validator.expect_column_mean_to_be_between(
    "purchase_amount", min_value=10, max_value=1000
)

# 執行驗證
results = validator.validate()

if not results.success:
    raise DataQualityError(f"資料品質檢查失敗: {results}")
```

## 2️⃣ 模型訓練流水線

### Kubeflow Pipeline

```python
from kfp import dsl
from kfp.components import create_component_from_func

@create_component_from_func
def preprocess_data(input_path: str, output_path: str):
    """資料預處理組件"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(input_path)

    # 清洗資料
    df = df.dropna()

    # 特徵工程
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.to_parquet(output_path)
    return output_path

@create_component_from_func
def train_model(
    data_path: str,
    model_path: str,
    hyperparameters: dict
):
    """模型訓練組件"""
    import mlflow
    from sklearn.ensemble import GradientBoostingClassifier

    with mlflow.start_run():
        # 載入資料
        df = pd.read_parquet(data_path)
        X = df.drop('target', axis=1)
        y = df['target']

        # 訓練模型
        model = GradientBoostingClassifier(**hyperparameters)
        model.fit(X, y)

        # 記錄指標
        mlflow.log_params(hyperparameters)
        mlflow.sklearn.log_model(model, "model")

        # 儲存模型
        joblib.dump(model, model_path)

    return model_path

@create_component_from_func
def evaluate_model(model_path: str, test_data_path: str) -> dict:
    """模型評估組件"""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_data_path)

    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "auc_roc": roc_auc_score(y_test, probabilities)
    }

    return metrics

@dsl.pipeline(
    name="enterprise-ml-pipeline",
    description="企業級機器學習訓練流水線"
)
def ml_pipeline(
    raw_data_path: str,
    model_output_path: str,
    hyperparameters: dict
):
    # 資料預處理
    preprocess_task = preprocess_data(
        input_path=raw_data_path,
        output_path="/tmp/processed_data.parquet"
    )

    # 模型訓練
    train_task = train_model(
        data_path=preprocess_task.output,
        model_path=model_output_path,
        hyperparameters=hyperparameters
    )
    train_task.after(preprocess_task)

    # 模型評估
    evaluate_task = evaluate_model(
        model_path=train_task.output,
        test_data_path="/tmp/test_data.parquet"
    )
    evaluate_task.after(train_task)
```

## 3️⃣ 模型版本管理

### MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 註冊模型
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "production-classifier")

# 模型版本管理
client.transition_model_version_stage(
    name="production-classifier",
    version=result.version,
    stage="Staging"
)

# 添加模型描述
client.update_model_version(
    name="production-classifier",
    version=result.version,
    description="v2.0 - 新增特徵，AUC 提升 5%"
)

# 模型標籤
client.set_model_version_tag(
    name="production-classifier",
    version=result.version,
    key="approved_by",
    value="ml-team-lead"
)

# 載入特定版本模型
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/production-classifier/Staging"
)
```

## 4️⃣ 自動化部署

### Kubernetes 部署配置

```yaml
# model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-service
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      containers:
      - name: model-server
        image: ml-model:v2.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MODEL_PATH
          value: "/models/production"
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 藍綠部署策略

```python
class BlueGreenDeployer:
    """藍綠部署管理器"""

    def __init__(self, kubernetes_client, namespace="production"):
        self.client = kubernetes_client
        self.namespace = namespace

    def deploy_new_version(self, model_version: str):
        """部署新版本到綠色環境"""
        green_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"ml-model-green-{model_version}",
                "labels": {"color": "green", "version": model_version}
            },
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"color": "green"}},
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": f"ml-model:{model_version}",
                            "ports": [{"containerPort": 8080}]
                        }]
                    }
                }
            }
        }

        self.client.create_deployment(
            namespace=self.namespace,
            body=green_deployment
        )

        # 等待就緒
        self._wait_for_ready(f"ml-model-green-{model_version}")

    def run_smoke_tests(self, endpoint: str) -> bool:
        """執行冒煙測試"""
        test_cases = [
            {"input": "test_input_1", "expected_type": "prediction"},
            {"input": "test_input_2", "expected_type": "prediction"},
        ]

        for test in test_cases:
            response = requests.post(
                f"{endpoint}/predict",
                json={"input": test["input"]}
            )
            if response.status_code != 200:
                return False

        return True

    def switch_traffic(self, to_color: str):
        """切換流量"""
        service_patch = {
            "spec": {
                "selector": {"color": to_color}
            }
        }

        self.client.patch_service(
            name="ml-model-service",
            namespace=self.namespace,
            body=service_patch
        )

    def rollback(self):
        """回滾到藍色環境"""
        self.switch_traffic("blue")
        print("已回滾到穩定版本")
```

## 5️⃣ 監控與告警

### Prometheus 指標

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定義指標
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction',
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5]
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_version']
)

DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Data drift detection score',
    ['feature']
)

class MonitoredPredictor:
    """帶監控的預測器"""

    def __init__(self, model, version: str):
        self.model = model
        self.version = version

    @PREDICTION_LATENCY.time()
    def predict(self, input_data):
        try:
            result = self.model.predict(input_data)
            PREDICTION_COUNT.labels(
                model_version=self.version,
                status="success"
            ).inc()
            return result
        except Exception as e:
            PREDICTION_COUNT.labels(
                model_version=self.version,
                status="error"
            ).inc()
            raise e

    def update_accuracy(self, accuracy: float):
        MODEL_ACCURACY.labels(model_version=self.version).set(accuracy)
```

### 資料漂移偵測

```python
from scipy import stats
import numpy as np
from typing import Dict, List

class DriftDetector:
    """資料漂移偵測器"""

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, data: pd.DataFrame) -> Dict:
        """計算統計量"""
        stats_dict = {}
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'distribution': data[col].values
                }
        return stats_dict

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """偵測資料漂移"""
        drift_scores = {}

        for col in current_data.columns:
            if col not in self.reference_stats:
                continue

            ref_dist = self.reference_stats[col]['distribution']
            cur_dist = current_data[col].values

            # KS 檢定
            statistic, p_value = stats.ks_2samp(ref_dist, cur_dist)
            drift_scores[col] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_drifted': p_value < self.threshold
            }

            # 更新 Prometheus 指標
            DATA_DRIFT_SCORE.labels(feature=col).set(statistic)

        return drift_scores

    def get_drift_report(self, current_data: pd.DataFrame) -> str:
        """生成漂移報告"""
        drift_results = self.detect_drift(current_data)

        report = ["=== 資料漂移偵測報告 ===\n"]

        drifted_features = [
            col for col, result in drift_results.items()
            if result['is_drifted']
        ]

        if drifted_features:
            report.append(f"⚠️ 偵測到 {len(drifted_features)} 個特徵發生漂移:\n")
            for col in drifted_features:
                result = drift_results[col]
                report.append(
                    f"  - {col}: KS統計量={result['statistic']:.4f}, "
                    f"p值={result['p_value']:.4f}\n"
                )
        else:
            report.append("✅ 未偵測到顯著資料漂移\n")

        return "".join(report)
```

## 6️⃣ 模型治理

### 模型審計追蹤

```python
from datetime import datetime
from typing import Optional
import json

class ModelAuditLogger:
    """模型審計日誌"""

    def __init__(self, storage_backend):
        self.storage = storage_backend

    def log_training(
        self,
        model_id: str,
        training_data_hash: str,
        hyperparameters: dict,
        metrics: dict,
        trained_by: str
    ):
        """記錄訓練事件"""
        event = {
            "event_type": "model_training",
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "training_data_hash": training_data_hash,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "trained_by": trained_by
        }
        self.storage.append(event)

    def log_deployment(
        self,
        model_id: str,
        version: str,
        environment: str,
        approved_by: str,
        approval_notes: Optional[str] = None
    ):
        """記錄部署事件"""
        event = {
            "event_type": "model_deployment",
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "version": version,
            "environment": environment,
            "approved_by": approved_by,
            "approval_notes": approval_notes
        }
        self.storage.append(event)

    def log_prediction(
        self,
        model_id: str,
        input_hash: str,
        output: dict,
        latency_ms: float
    ):
        """記錄預測事件（採樣）"""
        event = {
            "event_type": "model_prediction",
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "input_hash": input_hash,
            "output_summary": json.dumps(output)[:200],
            "latency_ms": latency_ms
        }
        self.storage.append(event)

    def get_model_lineage(self, model_id: str) -> List[dict]:
        """獲取模型血緣"""
        return self.storage.query(
            filter={"model_id": model_id},
            sort_by="timestamp"
        )
```

## 📚 最佳實踐總結

### 1. 資料管理
- 使用 Feature Store 統一特徵管理
- 實施資料品質檢查
- 保留資料版本與血緣

### 2. 模型開發
- 標準化訓練流水線
- 完整的版本控制
- 自動化測試

### 3. 部署策略
- 採用藍綠或金絲雀部署
- 實施健康檢查
- 配置自動擴展

### 4. 監控告警
- 追蹤關鍵指標
- 偵測資料漂移
- 設置告警閾值

### 5. 治理合規
- 完整審計追蹤
- 模型解釋性
- 符合法規要求

---

*本指南持續更新中，歡迎貢獻改進建議。*
