# MLflow 與超參數調整整合範例

## 目錄
- [整合概述](#整合概述)
- [MLflow + Optuna](#mlflow--optuna)
- [MLflow + Ray Tune](#mlflow--ray-tune)
- [MLflow + Scikit-learn](#mlflow--scikit-learn)
- [MLflow + Keras Tuner](#mlflow--keras-tuner)
- [完整專案範例](#完整專案範例)

## 整合概述

將 MLflow 與超參數調整工具整合，可以獲得以下好處：

1. **完整的實驗追蹤**：記錄每次超參數試驗的結果
2. **可視化對比**：輕鬆比較不同超參數組合的效果
3. **模型管理**：自動保存最佳模型
4. **可重現性**：記錄所有配置和環境資訊
5. **協作支持**：團隊成員可以查看和分享調優結果

## MLflow + Optuna

### 基礎整合

```python
import optuna
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

# 載入數據
iris = load_iris()
X, y = iris.data, iris.target

# 設定 MLflow 實驗
mlflow.set_experiment("optuna_sklearn_optimization")

def objective(trial):
    # 在 MLflow 中創建嵌套 run
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # 超參數採樣
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }

        # 記錄參數到 MLflow
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        # 訓練模型
        model = RandomForestClassifier(**params, random_state=42)

        # 交叉驗證評估
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()

        # 記錄指標到 MLflow
        mlflow.log_metric("cv_mean_accuracy", mean_score)
        mlflow.log_metric("cv_std_accuracy", std_score)

        # 訓練最終模型並記錄
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")

        # 設定標籤
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("model_type", "RandomForest")

    return mean_score

# 創建父 run 來組織所有試驗
with mlflow.start_run(run_name="optuna_optimization"):
    # 記錄優化配置
    mlflow.log_param("optimization_tool", "optuna")
    mlflow.log_param("n_trials", 50)
    mlflow.log_param("cv_folds", 5)

    # 創建 Optuna 研究
    study = optuna.create_study(
        direction='maximize',
        study_name='rf_optimization'
    )

    # 執行優化
    study.optimize(objective, n_trials=50)

    # 記錄最佳結果
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_accuracy", study.best_value)

    # 記錄 Optuna 可視化
    try:
        import optuna.visualization as vis

        # 優化歷史
        fig1 = vis.plot_optimization_history(study)
        fig1.write_image("optimization_history.png")
        mlflow.log_artifact("optimization_history.png")

        # 參數重要性
        fig2 = vis.plot_param_importances(study)
        fig2.write_image("param_importances.png")
        mlflow.log_artifact("param_importances.png")

        # 參數關係
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_image("parallel_coordinate.png")
        mlflow.log_artifact("parallel_coordinate.png")

    except Exception as e:
        print(f"可視化生成失敗: {e}")

    print(f"\n最佳參數: {study.best_params}")
    print(f"最佳交叉驗證準確率: {study.best_value:.4f}")
```

### 進階整合：使用 MLflow Callback

```python
from optuna.integration.mlflow import MLflowCallback

# MLflow callback 配置
mlflow_callback = MLflowCallback(
    tracking_uri="http://localhost:5000",
    metric_name="accuracy",
    create_experiment=False,
    mlflow_kwargs={
        "experiment_name": "optuna_with_callback",
        "tags": {"framework": "optuna", "version": "3.0"}
    }
)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }

    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5).mean()

    return score

# 創建研究並添加 callback
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])
```

### 完整的深度學習範例

```python
import optuna
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FlexibleNN(nn.Module):
    def __init__(self, trial, input_size, output_size):
        super(FlexibleNN, self).__init__()

        layers = []
        n_layers = trial.suggest_int('n_layers', 1, 4)

        in_features = input_size
        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 32, 256)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())

            dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            layers.append(nn.Dropout(dropout))

            in_features = out_features

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def objective(trial):
    with mlflow.start_run(nested=True):
        # 超參數
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

        # 記錄超參數
        mlflow.log_params({
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'trial_number': trial.number
        })

        # 構建模型
        model = FlexibleNN(trial, input_size=20, output_size=2)

        # 記錄模型架構資訊
        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param('total_parameters', n_params)

        # 優化器
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # 創建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        # 訓練
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5

        for epoch in range(50):
            # 訓練階段
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # 驗證階段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total

            # 記錄每個 epoch 的指標
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            }, step=epoch)

            # 報告給 Optuna（用於剪枝）
            trial.report(val_accuracy, epoch)

            # 檢查是否應該剪枝
            if trial.should_prune():
                mlflow.set_tag("pruned", True)
                raise optuna.TrialPruned()

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    mlflow.log_param("early_stopped_epoch", epoch)
                    break

        # 記錄最終模型
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("final_val_accuracy", val_accuracy)

        return val_accuracy

# 執行優化
mlflow.set_experiment("pytorch_optuna_optimization")

with mlflow.start_run(run_name="full_optimization"):
    # 創建帶剪枝的研究
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )

    mlflow.log_param("n_trials", 100)
    mlflow.log_param("pruner", "MedianPruner")

    study.optimize(objective, n_trials=100)

    # 記錄優化結果
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_accuracy", study.best_value)

    # 統計資訊
    mlflow.log_metric("n_completed_trials", len(study.trials))
    mlflow.log_metric("n_pruned_trials",
                      len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))

    print(f"最佳參數: {study.best_params}")
    print(f"最佳驗證準確率: {study.best_value:.4f}")
```

## MLflow + Ray Tune

```python
import ray
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
import mlflow

def train_function(config):
    # 訓練邏輯
    for epoch in range(10):
        accuracy = train_epoch(config)

        # 報告指標給 Ray Tune
        tune.report(accuracy=accuracy, epoch=epoch)

# MLflow callback 配置
mlflow_callback = MLflowLoggerCallback(
    tracking_uri="http://localhost:5000",
    experiment_name="ray_tune_experiment",
    save_artifact=True
)

# Ray Tune 配置
config = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "hidden_size": tune.randint(64, 512)
}

# 執行調優
analysis = tune.run(
    train_function,
    config=config,
    num_samples=50,
    callbacks=[mlflow_callback],
    metric="accuracy",
    mode="max"
)

print("最佳配置:", analysis.get_best_config(metric="accuracy", mode="max"))
```

## MLflow + Scikit-learn

### Grid Search 整合

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

mlflow.set_experiment("sklearn_gridsearch")

with mlflow.start_run(run_name="gridsearch_optimization"):
    # 定義參數網格
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    # 記錄搜索配置
    mlflow.log_param("search_method", "GridSearch")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("total_combinations",
                      len(param_grid['learning_rate']) *
                      len(param_grid['n_estimators']) *
                      len(param_grid['max_depth']) *
                      len(param_grid['subsample']))

    # 執行網格搜索
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # 記錄所有結果
    results_df = pd.DataFrame(grid_search.cv_results_)

    # 為每個參數組合創建子 run
    for idx, params in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(nested=True, run_name=f"config_{idx}"):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score",
                            grid_search.cv_results_['mean_test_score'][idx])
            mlflow.log_metric("std_test_score",
                            grid_search.cv_results_['std_test_score'][idx])
            mlflow.log_metric("mean_fit_time",
                            grid_search.cv_results_['mean_fit_time'][idx])

    # 記錄最佳結果
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # 評估測試集
    test_score = grid_search.score(X_test, y_test)
    mlflow.log_metric("test_score", test_score)

    # 記錄最佳模型
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

    # 記錄結果摘要
    results_df.to_csv("grid_search_results.csv", index=False)
    mlflow.log_artifact("grid_search_results.csv")

    print(f"最佳參數: {grid_search.best_params_}")
    print(f"最佳CV分數: {grid_search.best_score_:.4f}")
    print(f"測試分數: {test_score:.4f}")
```

### Random Search 整合

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

mlflow.set_experiment("sklearn_randomsearch")

with mlflow.start_run(run_name="randomsearch_optimization"):
    # 定義參數分佈
    param_distributions = {
        'learning_rate': uniform(0.01, 0.29),
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 15),
        'subsample': uniform(0.6, 0.4),
        'min_samples_split': randint(2, 20)
    }

    mlflow.log_param("search_method", "RandomSearch")
    mlflow.log_param("n_iter", 100)
    mlflow.log_param("cv_folds", 5)

    # 執行隨機搜索
    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    # 記錄結果（同 Grid Search）
    # ... 省略類似代碼 ...

    # 可視化參數分佈
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, param in enumerate(param_distributions.keys()):
        if idx < len(axes):
            values = [params[param] for params in random_search.cv_results_['params']]
            scores = random_search.cv_results_['mean_test_score']

            axes[idx].scatter(values, scores, alpha=0.5)
            axes[idx].set_xlabel(param)
            axes[idx].set_ylabel('CV Score')
            axes[idx].set_title(f'{param} vs CV Score')

    plt.tight_layout()
    plt.savefig("param_analysis.png")
    mlflow.log_artifact("param_analysis.png")
```

## MLflow + Keras Tuner

```python
import keras_tuner as kt
import mlflow
import mlflow.keras
from tensorflow import keras

class MLflowReporter(kt.Callback):
    """自定義 Keras Tuner callback 以整合 MLflow"""

    def on_trial_begin(self, trial):
        """每個試驗開始時啟動 MLflow run"""
        self.trial_run = mlflow.start_run(
            nested=True,
            run_name=f"trial_{trial.trial_id}"
        )

    def on_trial_end(self, trial):
        """每個試驗結束時記錄結果並結束 run"""
        # 記錄超參數
        mlflow.log_params(trial.hyperparameters.values)

        # 記錄最佳分數
        mlflow.log_metric("best_val_accuracy", trial.score)

        # 結束 run
        mlflow.end_run()

def build_model(hp):
    """構建可調優的模型"""
    model = keras.Sequential()

    # 調整層數
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(
            hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # 調整學習率
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# MLflow 實驗設定
mlflow.set_experiment("keras_tuner_optimization")

with mlflow.start_run(run_name="hyperband_search"):
    # 記錄調優配置
    mlflow.log_param("tuner_type", "Hyperband")
    mlflow.log_param("max_epochs", 50)

    # 創建調優器
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='tuning_dir',
        project_name='keras_mlflow'
    )

    # 搜索最佳超參數
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[
            MLflowReporter(),
            keras.callbacks.EarlyStopping(patience=5)
        ]
    )

    # 獲取最佳模型
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 記錄最佳超參數
    mlflow.log_params(best_hp.values)

    # 評估最佳模型
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    mlflow.log_metrics({
        "best_test_loss": test_loss,
        "best_test_accuracy": test_acc
    })

    # 記錄最佳模型
    mlflow.keras.log_model(best_model, "best_model")

    print(f"最佳超參數: {best_hp.values}")
    print(f"測試準確率: {test_acc:.4f}")
```

## 完整專案範例

### 專案結構

```
hyperparameter_tuning_project/
├── config/
│   ├── model_config.yaml
│   └── search_config.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── optimizer.py
│   └── utils/
│       ├── __init__.py
│       └── mlflow_utils.py
├── notebooks/
│   └── analysis.ipynb
├── main.py
├── requirements.txt
└── README.md
```

### main.py - 主要執行腳本

```python
"""
超參數調整主腳本
整合 MLflow 和 Optuna
"""

import argparse
import yaml
import mlflow
import optuna
from pathlib import Path

from src.data.data_loader import load_data
from src.models.model import create_model
from src.training.trainer import Trainer
from src.training.optimizer import OptunaOptimizer
from src.utils.mlflow_utils import setup_mlflow, log_optimization_results

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # 載入配置
    model_config = load_config(args.model_config)
    search_config = load_config(args.search_config)

    # 設定 MLflow
    experiment_id = setup_mlflow(
        experiment_name=args.experiment_name,
        tracking_uri=args.mlflow_uri
    )

    # 載入數據
    print("載入數據...")
    train_data, val_data, test_data = load_data(
        data_path=args.data_path,
        **model_config['data']
    )

    # 創建優化器
    optimizer = OptunaOptimizer(
        model_config=model_config,
        search_config=search_config,
        train_data=train_data,
        val_data=val_data
    )

    # 執行優化
    print("開始超參數優化...")
    with mlflow.start_run(run_name="optimization_run"):
        best_params, best_score = optimizer.optimize(
            n_trials=args.n_trials
        )

        # 記錄結果
        log_optimization_results(
            optimizer.study,
            best_params,
            best_score
        )

        # 使用最佳參數訓練最終模型
        print("\n使用最佳參數訓練最終模型...")
        final_model = create_model(best_params)
        trainer = Trainer(final_model, train_data, val_data)

        trainer.train(epochs=model_config['training']['epochs'])

        # 評估測試集
        test_metrics = trainer.evaluate(test_data)

        mlflow.log_metrics({
            f"final_test_{k}": v for k, v in test_metrics.items()
        })

        # 保存最終模型
        mlflow.pytorch.log_model(final_model, "final_model")

        print(f"\n優化完成!")
        print(f"最佳參數: {best_params}")
        print(f"最佳驗證分數: {best_score:.4f}")
        print(f"測試指標: {test_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="超參數優化訓練")

    parser.add_argument("--experiment-name", type=str, required=True,
                        help="MLflow 實驗名稱")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                        help="模型配置文件路徑")
    parser.add_argument("--search-config", type=str, default="config/search_config.yaml",
                        help="搜索配置文件路徑")
    parser.add_argument("--data-path", type=str, required=True,
                        help="數據路徑")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000",
                        help="MLflow 追蹤 URI")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="優化試驗次數")

    args = parser.parse_args()
    main(args)
```

### src/utils/mlflow_utils.py

```python
"""MLflow 工具函數"""

import mlflow
import matplotlib.pyplot as plt
import optuna.visualization as vis
from pathlib import Path

def setup_mlflow(experiment_name, tracking_uri="http://localhost:5000"):
    """設定 MLflow 實驗"""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id

def log_optimization_results(study, best_params, best_score, artifacts_dir="artifacts"):
    """記錄優化結果到 MLflow"""
    # 記錄最佳參數和分數
    mlflow.log_params(best_params)
    mlflow.log_metric("best_validation_score", best_score)

    # 記錄優化統計
    mlflow.log_metrics({
        "n_trials": len(study.trials),
        "n_complete_trials": len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned_trials": len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED])
    })

    # 生成並記錄可視化
    Path(artifacts_dir).mkdir(exist_ok=True)

    # 優化歷史
    try:
        fig = vis.plot_optimization_history(study)
        fig.write_image(f"{artifacts_dir}/optimization_history.png")
        mlflow.log_artifact(f"{artifacts_dir}/optimization_history.png")
    except Exception as e:
        print(f"無法生成優化歷史圖: {e}")

    # 參數重要性
    try:
        fig = vis.plot_param_importances(study)
        fig.write_image(f"{artifacts_dir}/param_importances.png")
        mlflow.log_artifact(f"{artifacts_dir}/param_importances.png")
    except Exception as e:
        print(f"無法生成參數重要性圖: {e}")

    # 平行座標圖
    try:
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(f"{artifacts_dir}/parallel_coordinate.png")
        mlflow.log_artifact(f"{artifacts_dir}/parallel_coordinate.png")
    except Exception as e:
        print(f"無法生成平行座標圖: {e}")

def log_model_performance(model, test_data, model_name="model"):
    """記錄模型性能"""
    # 評估模型
    metrics = model.evaluate(test_data)

    # 記錄指標
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    # 記錄模型
    mlflow.pytorch.log_model(model, model_name)

    return metrics
```

## 總結

整合 MLflow 與超參數調整工具的關鍵要點：

1. **嵌套 Runs**：使用嵌套 runs 組織優化過程
2. **完整記錄**：記錄所有試驗的參數和指標
3. **可視化**：生成並保存優化過程的可視化圖表
4. **最佳模型**：自動保存表現最好的模型
5. **可擴展性**：設計可重用的代碼結構

這些整合方法可以幫助你：
- 更好地管理超參數調整實驗
- 輕鬆比較不同配置的效果
- 確保實驗的可重現性
- 促進團隊協作和知識分享

下一步：
- [最佳實踐](../05.Best_Practices/README.md)
