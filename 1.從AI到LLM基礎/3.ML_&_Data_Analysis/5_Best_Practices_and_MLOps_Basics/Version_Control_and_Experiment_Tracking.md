# MLOps 與版本控制完整指南

## 1. 版本控制基礎 (Git)
### 1.1 Git 基礎概念
- **Repository (儲存庫)**
  - 本地儲存庫 (Local Repository)
  - 遠端儲存庫 (Remote Repository)
  - Fork 與 Clone 的差異
  - Bare Repository vs Regular Repository

- **Git 內部運作機制**
  - Git 物件類型（blob、tree、commit、tag）
  - Git 參照（Reference）系統
  - Git 索引（Index）
  - Git 垃圾回收機制

- **.gitignore 進階設定**
  ```gitignore
  # 常見 ML 專案忽略項目
  *.pyc
  __pycache__/
  .ipynb_checkpoints/
  .env
  venv/
  *.log
  /data/*
  !/data/.gitkeep
  /models/*
  !/models/.gitkeep
  .DS_Store
  ```

### 1.2 Git 進階操作
- **分支管理策略**
  ```bash
  # 建立特性分支
  git checkout -b feature/new-model

  # 臨時保存修改
  git stash save "work in progress"
  git stash pop

  # 解決衝突
  git mergetool

  # 修改提交歷史
  git rebase -i HEAD~3
  ```

- **Git Hooks**
  ```bash
  #!/bin/sh
  # pre-commit hook 示例
  python tests/run_tests.py
  if [ $? -ne 0 ]; then
      echo "Tests must pass before commit!"
      exit 1
  fi
  ```

- **Git 子模組**
  ```bash
  # 添加子模組
  git submodule add https://github.com/user/repo libs/repo

  # 更新子模組
  git submodule update --init --recursive
  ```

### 1.3 ML 專案的 Git 最佳實踐
- **分支策略**
  ```
  main
    ├── develop
    │   ├── feature/model-v1
    │   ├── feature/data-preprocessing
    │   └── feature/evaluation-metrics
    ├── release/v1.0
    └── hotfix/model-bias
  ```

- **提交規範**
  ```
  feat: 新功能
  fix: 錯誤修復
  docs: 文檔更改
  style: 格式調整
  refactor: 重構程式碼
  test: 添加測試
  chore: 構建過程或輔助工具的變動
  ```

## 2. DVC (Data Version Control)
### 2.1 DVC 架構與概念
- **核心組件**
  - DVC Cache
  - DVC Remote
  - DVC Pipeline
  - DVC Metrics
  - DVC Parameters

- **支援的存儲後端**
  - Amazon S3
  - Google Cloud Storage
  - Azure Blob Storage
  - SSH/SFTP
  - Local Storage
  - NFS
  - HDFS

### 2.2 DVC 進階配置
```yaml
# .dvc/config
['remote "s3"']
    url = s3://bucket/path
    endpointurl = https://endpoint.com
    access_key_id = ${AWS_ACCESS_KEY_ID}
    secret_access_key = ${AWS_SECRET_ACCESS_KEY}
    region = us-east-1
```

### 2.3 DVC 進階管線
```yaml
# dvc.yaml
stages:
  data_preprocessing:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw
    params:
      - preprocessing.split_ratio
      - preprocessing.random_state
    outs:
      - data/processed
    metrics:
      - metrics/preprocessing.json:
          cache: false

  feature_engineering:
    cmd: python src/features.py
    deps:
      - src/features.py
      - data/processed
    params:
      - features.scaling_method
      - features.encoding_method
    outs:
      - data/features
    
  train_model:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features
    params:
      - training.model_type
      - training.hyperparameters
    outs:
      - models/model.pkl
    metrics:
      - metrics/training.json:
          cache: false
    plots:
      - plots/confusion_matrix.png
      - plots/roc_curve.png:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test
    metrics:
      - metrics/evaluation.json:
          cache: false
    plots:
      - plots/predictions.png:
          cache: false
```

### 2.4 DVC 性能優化
- **大型文件處理**
  ```bash
  # 啟用文件分塊
  dvc config cache.type hardlink,symlink

  # 配置快取大小
  dvc config cache.size_limit 10G
  ```

- **並行處理**
  ```bash
  # 設定並行執行數
  dvc config core.jobs 4
  ```

## 3. MLflow
### 3.1 MLflow 架構
- **MLflow Tracking Server**
  ```python
  # 啟動追蹤伺服器
  mlflow server \
      --backend-store-uri postgresql://user:pass@localhost/mlflow \
      --default-artifact-root s3://bucket/path \
      --host 0.0.0.0 \
      --port 5000
  ```

- **資料庫配置**
  ```python
  # SQLAlchemy 資料庫 URI
  mlflow.set_tracking_uri('postgresql://user:pass@localhost/mlflow')
  ```

### 3.2 進階實驗追蹤
```python
import mlflow
from mlflow.tracking import MlflowClient

# 自定義實驗管理
client = MlflowClient()
experiment = client.create_experiment(
    "experiment_name",
    artifact_location="s3://bucket/path",
    tags={"team": "ml-team", "project": "customer-churn"}
)

# 巢狀執行
with mlflow.start_run(run_name="parent_run") as parent_run:
    mlflow.log_param("parent_param", "value")
    
    with mlflow.start_run(run_name="child_run", nested=True) as child_run:
        mlflow.log_param("child_param", "value")
        mlflow.log_metric("child_metric", 0.85)

# 自動記錄
mlflow.sklearn.autolog()
mlflow.tensorflow.autolog()
mlflow.pytorch.autolog()

# 模型註冊
model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name="production_model"
)

# 模型版本轉換
client.transition_model_version_stage(
    name="production_model",
    version=1,
    stage="Production"
)
```

### 3.3 MLflow 專案進階配置
```yaml
# MLproject
name: advanced_ml_project

docker_env:
  image: mlflow-docker-example

entry_points:
  main:
    parameters:
      data_path: path
      model_config: path
      output_path: path
    command: "python train.py --data {data_path} --config {model_config} --output {output_path}"

  validate:
    parameters:
      model_path: path
      test_data: path
    command: "python validate.py --model {model_path} --data {test_data}"

  deploy:
    parameters:
      model_path: path
      deploy_config: path
    command: "python deploy.py --model {model_path} --config {deploy_config}"
```

## 4. Weights & Biases (WandB)
### 4.1 進階功能
- **超參數優化**
  ```python
  import wandb
  from wandb.keras import WandbCallback

  sweep_config = {
      'method': 'bayes',
      'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
      'parameters': {
          'learning_rate': {'min': 0.0001, 'max': 0.1},
          'batch_size': {'values': [16, 32, 64, 128]},
          'layers': {'values': [1, 2, 3, 4]},
          'activation': {'values': ['relu', 'tanh']}
      }
  }

  sweep_id = wandb.sweep(sweep_config, project="my_project")
  wandb.agent(sweep_id, function=train)
  ```

- **資料集版本控制**
  ```python
  # 建立資料集工件
  artifact = wandb.Artifact('dataset', type='dataset')
  artifact.add_dir('data/processed')
  run.log_artifact(artifact)

  # 使用資料集
  artifact = run.use_artifact('dataset:latest')
  artifact_dir = artifact.download()
  ```

### 4.2 自定義視覺化
```python
# 自定義圖表
wandb.log({
    "custom_plot": wandb.plot.line_series(
        xs=[[1, 2, 3], [1, 2, 3]],
        ys=[[1, 2, 3], [2, 4, 6]],
        keys=["metric1", "metric2"],
        title="Custom Metrics Comparison"
    )
})

# 記錄模型架構
wandb.watch(model, log="all", log_freq=100)
```

## 5. 整合實踐
### 5.1 完整 CI/CD 流程
```yaml
# .gitlab-ci.yml
image: python:3.8

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"
  MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI

cache:
  paths:
    - .pip-cache/
    - venv/

stages:
  - setup
  - data
  - train
  - evaluate
  - deploy

setup:
  stage: setup
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt

data_preparation:
  stage: data
  script:
    - source venv/bin/activate
    - dvc pull data/raw
    - dvc repro data_preparation
    - dvc push data/processed
  artifacts:
    paths:
      - data/processed/

model_training:
  stage: train
  script:
    - source venv/bin/activate
    - mlflow run . -P data_path=data/processed
    - dvc add models/
    - dvc push
  artifacts:
    paths:
      - models/
      - mlruns/

model_evaluation:
  stage: evaluate
  script:
    - source venv/bin/activate
    - python src/evaluate.py
    - mlflow ui
  artifacts:
    reports:
      metrics: metrics/evaluation.json

model_deployment:
  stage: deploy
  script:
    - source venv/bin/activate
    - python src/deploy.py
  only:
    - master
```

### 5.2 Docker 整合
```dockerfile
# Dockerfile
FROM python:3.8-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    git \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install -r requirements.txt

# 設定工作目錄
WORKDIR /app

# 複製專案文件
COPY . .

# 設定環境變數
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV WANDB_API_KEY=${WANDB_API_KEY}

# 啟動命令
CMD ["python", "src/train.py"]
```

### 5.3 Kubernetes 部署
```yaml
# kubernetes/mlflow.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-tracking-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: mlflow-server:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio:9000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: aws-access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: aws-secret-access-key
        volumeMounts:
        - name: mlflow-persistent-storage
          mountPath: /mlflow
      volumes:
      - name: mlflow-persistent-storage
        persistentVolumeClaim:
          claimName: mlflow-pvc
```

## 6. 監控與維護
### 6.1 監控系統
```python
# 模型監控
from prometheus_client import start_http_server, Gauge, Counter

# 定義指標
prediction_latency = Gauge('model_prediction_latency_seconds', 
                         'Time spent processing prediction')
prediction_counter = Counter('model_predictions_total', 
                           'Total number of predictions',
                           ['model_version', 'outcome'])

# 記錄指標
@prediction_latency.time()
def predict(data):
    result = model.predict(data)
    prediction_counter.labels(
        model_version='v1.0',
        outcome='success'
    ).inc()
    return result
```

### 6.2 日誌管理
```python
import logging
import json
from datetime import datetime

# 配置結構化日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_event(event_type, details):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': event_type,
        'details': details
    }
    logger.info(json.dumps(log_entry))
```

### 6.3 備份策略
```bash
#!/bin/bash
# backup_script.sh

# 備份 MLflow 資料庫
pg_dump mlflow > mlflow_backup_$(date