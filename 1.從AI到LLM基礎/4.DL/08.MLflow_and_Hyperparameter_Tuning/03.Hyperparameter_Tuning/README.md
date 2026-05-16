# 超參數調整 (Hyperparameter Tuning)

## 目錄
- [什麼是超參數調整](#什麼是超參數調整)
- [超參數調整方法](#超參數調整方法)
- [常用工具和框架](#常用工具和框架)
- [實踐範例](#實踐範例)
- [最佳實踐](#最佳實踐)

## 什麼是超參數調整

### 超參數 vs 參數

**參數 (Parameters)**:
- 模型在訓練過程中學習得到的值
- 例如：神經網路的權重、線性迴歸的係數
- 通過訓練資料自動優化

**超參數 (Hyperparameters)**:
- 在訓練開始前設定的配置值
- 例如：學習率、批次大小、網路層數、正則化係數
- 需要人工設定或通過搜索方法確定

### 為什麼需要超參數調整？

1. **提升模型性能**：正確的超參數可以顯著提高模型效果
2. **防止過擬合/欠擬合**：通過調整正則化等參數平衡模型複雜度
3. **加快訓練速度**：優化學習率等參數可以加快收斂
4. **資源優化**：找到性能和成本的最佳平衡點

### 常見的超參數

#### 通用超參數
- **學習率 (Learning Rate)**：控制參數更新的步長
- **批次大小 (Batch Size)**：每次訓練使用的樣本數量
- **訓練輪數 (Epochs)**：訓練資料的完整遍歷次數
- **優化器 (Optimizer)**：SGD、Adam、RMSprop 等

#### 模型特定超參數
- **神經網路**：層數、每層神經元數、激活函式、Dropout 率
- **決策樹**：最大深度、最小分裂樣本數、最小葉子節點樣本數
- **隨機森林**：樹的數量、最大特徵數
- **SVM**：核函數、C 值、gamma 值

#### 正則化超參數
- **L1/L2 正則化係數**：控制權重懲罰強度
- **Dropout 率**：隨機丟棄神經元的比例
- **Early Stopping**：提前停止訓練的耐心值

## 超參數調整方法

### 1. 手動調整 (Manual Tuning)

**描述**：基於經驗和直覺手動調整超參數

**優點**：
- 靈活性高
- 可以結合領域知識
- 適合快速原型開發

**缺點**：
- 耗時費力
- 容易遺漏最佳配置
- 不夠系統化

**適用場景**：
- 初期探索階段
- 超參數空間較小
- 有經驗的從業者

```python
# 手動調整範例
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"Testing lr={lr}, batch_size={bs}")
        model = train_model(learning_rate=lr, batch_size=bs)
        accuracy = evaluate_model(model)
        print(f"Accuracy: {accuracy}")
```

### 2. 網格搜索 (Grid Search)

**描述**：窮舉搜索所有超參數組合

**優點**：
- 簡單易理解
- 保證找到搜索空間內的最佳組合
- 結果可重現

**缺點**：
- 計算成本隨維度指數增長
- 在高維空間效率低下
- 無法處理連續超參數

**適用場景**：
- 超參數數量較少（通常 ≤ 3-4 個）
- 每個超參數的候選值較少
- 計算資源充足

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 定義超參數空間
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 建立網格搜索
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5折交叉驗證
    scoring='accuracy',
    n_jobs=-1,  # 使用所有CPU核心
    verbose=2
)

# 執行搜索
grid_search.fit(X_train, y_train)

# 獲取最佳參數
print("最佳參數:", grid_search.best_params_)
print("最佳分數:", grid_search.best_score_)

# 使用最佳模型
best_model = grid_search.best_estimator_
```

**時間複雜度分析**：
```
假設有 3 個超參數，每個有 10 個候選值
總組合數 = 10 × 10 × 10 = 1000 次訓練
如果每次訓練 5 分鐘，總時間 = 5000 分鐘 ≈ 83 小時
```

### 3. 隨機搜索 (Random Search)

**描述**：在超參數空間中隨機採樣

**優點**：
- 比網格搜索效率高
- 可以處理連續超參數
- 更容易並行化
- 通常能更快找到較好的結果

**缺點**：
- 不保證找到最佳配置
- 需要設定合適的採樣次數

**適用場景**：
- 超參數空間較大
- 部分超參數對結果影響較小
- 計算資源有限

**理論基礎**：
Bergstra & Bengio (2012) 的研究表明，隨機搜索在高維空間中通常比網格搜索更有效，因為：
1. 通常只有少數超參數真正重要
2. 隨機搜索可以探索更多的超參數值

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

# 定義超參數分佈
param_distributions = {
    'n_estimators': randint(50, 300),  # 整數均勻分佈
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)  # 連續均勻分佈
}

# 建立隨機搜索
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # 採樣次數
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# 執行搜索
random_search.fit(X_train, y_train)

print("最佳參數:", random_search.best_params_)
print("最佳分數:", random_search.best_score_)
```

### 4. 貝葉斯優化 (Bayesian Optimization)

**描述**：使用貝葉斯推理建立超參數與性能的概率模型，智能地選擇下一組要嘗試的超參數

**核心概念**：
1. **代理模型 (Surrogate Model)**：通常使用高斯過程建模目標函式
2. **採集函數 (Acquisition Function)**：決定下一個要評估的點
   - Expected Improvement (EI)
   - Upper Confidence Bound (UCB)
   - Probability of Improvement (PI)

**優點**：
- 樣本效率高，適合評估成本高的場景
- 自適應搜索，智能平衡探索與利用
- 可以處理噪聲
- 支持並行評估

**缺點**：
- 實現複雜
- 在高維空間效率降低
- 需要更多計算開銷來維護代理模型

**適用場景**：
- 單次訓練成本很高
- 超參數維度中等（通常 < 20）
- 需要高品質結果

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import GradientBoostingClassifier

# 定義搜索空間
search_spaces = {
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'subsample': Real(0.5, 1.0)
}

# 建立貝葉斯搜索
bayes_search = BayesSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    search_spaces=search_spaces,
    n_iter=50,  # 迭代次數
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# 執行搜索
bayes_search.fit(X_train, y_train)

print("最佳參數:", bayes_search.best_params_)
print("最佳分數:", bayes_search.best_score_)
```

### 5. 超帶演算法 (Hyperband)

**描述**：自適應資源分配演算法，通過提前停止策略快速淘汰表現不佳的配置

**核心思想**：
- 用較少資源（如更少的訓練輪數）評估大量配置
- 逐步淘汰表現差的配置
- 將更多資源分配給有潛力的配置

**優點**：
- 樣本和計算效率高
- 自動平衡探索廣度和深度
- 適合深度學習場景

**缺點**：
- 實現較複雜
- 需要定義資源概念（如 epochs）

**適用場景**：
- 深度學習模型訓練
- 訓練時間長的模型
- 超參數空間很大

```python
# 使用 Optuna 實現 Hyperband
import optuna
from optuna.pruners import HyperbandPruner

def objective(trial):
    # 定義超參數
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 5)

    # 訓練模型（支持中途停止）
    for epoch in range(100):
        accuracy = train_epoch(lr, batch_size, n_layers)

        # 報告中間結果
        trial.report(accuracy, epoch)

        # 檢查是否應該剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy

# 建立研究
study = optuna.create_study(
    direction='maximize',
    pruner=HyperbandPruner(
        min_resource=1,
        max_resource=100,
        reduction_factor=3
    )
)

# 執行優化
study.optimize(objective, n_trials=100)

print("最佳參數:", study.best_params)
print("最佳分數:", study.best_value)
```

### 6. 進化演算法 (Evolutionary Algorithms)

**描述**：模擬生物進化過程，通過選擇、交叉和變異來搜索最優超參數

**優點**：
- 可以處理複雜的搜索空間
- 不需要梯度資訊
- 容易並行化

**缺點**：
- 收斂速度可能較慢
- 需要調整演算法自身的參數

```python
from tpot import TPOTClassifier

# TPOT 使用遺傳演算法自動化機器學習
tpot = TPOTClassifier(
    generations=10,  # 進化代數
    population_size=50,  # 每代個體數
    cv=5,
    scoring='accuracy',
    verbosity=2,
    random_state=42,
    n_jobs=-1
)

tpot.fit(X_train, y_train)
print("最佳準確率:", tpot.score(X_test, y_test))

# 導出最佳模型的程式碼
tpot.export('best_pipeline.py')
```

## 常用工具和框架

### 1. Optuna

**特點**：
- 現代化的超參數優化框架
- 支持多種採樣策略（TPE、CMA-ES、Grid、Random）
- 提供高效的剪枝機制
- 優秀的可視化功能
- 易於使用的 API

**安裝**：
```bash
pip install optuna
```

**基本使用**：
```python
import optuna

def objective(trial):
    # 建議超參數
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])

    if classifier_name == 'SVC':
        svc_c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
        classifier = SVC(C=svc_c, random_state=42)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        classifier = RandomForestClassifier(
            max_depth=rf_max_depth,
            random_state=42
        )

    # 訓練和評估
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)

    return accuracy

# 建立研究
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 查看結果
print("最佳參數:", study.best_params)
print("最佳分數:", study.best_value)

# 可視化
import optuna.visualization as vis

# 優化歷史
vis.plot_optimization_history(study).show()

# 參數重要性
vis.plot_param_importances(study).show()

# 參數關係
vis.plot_parallel_coordinate(study).show()
```

### 2. Ray Tune

**特點**：
- 支持大規模分散式調優
- 與 Ray 生態系統集成
- 支持多種搜索演算法
- 適合深度學習場景

**安裝**：
```bash
pip install ray[tune]
```

**基本使用**：
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = create_model(
        lr=config["lr"],
        batch_size=config["batch_size"]
    )

    for epoch in range(10):
        loss, accuracy = train_epoch(model)
        # 報告指標
        tune.report(loss=loss, accuracy=accuracy)

# 配置搜索空間
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128])
}

# 配置調度器
scheduler = ASHAScheduler(
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

# 執行調優
analysis = tune.run(
    train_model,
    config=config,
    num_samples=50,
    scheduler=scheduler,
    metric="accuracy",
    mode="max"
)

# 獲取最佳配置
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print("最佳配置:", best_config)
```

### 3. Keras Tuner

**特點**：
- 專門為 Keras/TensorFlow 設計
- 簡單易用
- 支持多種調優演算法

**安裝**：
```bash
pip install keras-tuner
```

**基本使用**：
```python
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()

    # 調整層數
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # 調整學習率
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# 建立調優器
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    directory='my_dir',
    project_name='keras_tuning'
)

# 執行搜索
tuner.search(X_train, y_train, epochs=50, validation_split=0.2)

# 獲取最佳模型
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("最佳超參數:", best_hp.values)
```

### 4. Scikit-optimize

**特點**：
- 基於貝葉斯優化
- 與 Scikit-learn 無縫集成
- 提供多種採集函數

**安裝**：
```bash
pip install scikit-optimize
```

**基本使用**：
```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# 定義搜索空間
space = [
    Real(1e-6, 1e-1, name='learning_rate', prior='log-uniform'),
    Integer(1, 5, name='num_layers'),
    Integer(16, 512, name='units'),
    Real(0.0, 0.5, name='dropout')
]

# 定義目標函式
@use_named_args(space)
def objective(**params):
    model = build_model(**params)
    score = evaluate_model(model)
    return -score  # 最小化

# 執行優化
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    random_state=42,
    verbose=True
)

print("最佳參數:", result.x)
print("最佳分數:", -result.fun)
```

## 實踐範例

### 範例 1: 完整的神經網路超參數調整流程

```python
import optuna
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras

# 定義目標函式
def objective(trial):
    # 啟動 MLflow run
    with mlflow.start_run(nested=True):
        # 超參數建議
        n_layers = trial.suggest_int('n_layers', 1, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

        # 記錄參數到 MLflow
        mlflow.log_params({
            'n_layers': n_layers,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate
        })

        # 構建模型
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        for i in range(n_layers):
            units = trial.suggest_int(f'units_layer_{i}', 32, 512, step=32)
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(10, activation='softmax'))

        # 選擇優化器
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 訓練模型（帶早停）
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # 評估模型
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        # 記錄指標到 MLflow
        mlflow.log_metrics({
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        })

        # 記錄模型
        mlflow.keras.log_model(model, "model")

        return test_accuracy

# 設定 MLflow 實驗
mlflow.set_experiment("hyperparameter_tuning_demo")

# 建立 Optuna 研究
study = optuna.create_study(
    direction='maximize',
    study_name='mnist_optimization'
)

# 執行優化
with mlflow.start_run(run_name="optuna_optimization"):
    study.optimize(objective, n_trials=50)

    # 記錄最佳結果
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)

    print("最佳參數:", study.best_params)
    print("最佳準確率:", study.best_value)

    # 可視化優化過程
    import optuna.visualization as vis

    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)

    fig1.write_image("optimization_history.png")
    fig2.write_image("param_importances.png")

    mlflow.log_artifact("optimization_history.png")
    mlflow.log_artifact("param_importances.png")
```

### 範例 2: PyTorch 模型的分散式超參數調整

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import mlflow.pytorch

class LightningModel(pl.LightningModule):
    def __init__(self, trial):
        super().__init__()

        # 超參數
        self.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # 構建網路
        layers = []
        in_features = 784

        for i in range(n_layers):
            out_features = trial.suggest_int(f"n_units_l{i}", 64, 512)
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_features = out_features

        layers.append(torch.nn.Linear(in_features, 10))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def objective(trial):
    model = LightningModel(trial)

    trainer = pl.Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_acc"),
            EarlyStopping(monitor="val_loss", patience=5)
        ]
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics["val_acc"].item()

# 執行分散式優化
study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna.db",  # 共享存儲
    study_name="distributed_optimization",
    load_if_exists=True
)

study.optimize(objective, n_trials=100)
```

## 最佳實踐

### 1. 搜索空間設計

```python
# 好的實踐：使用對數尺度搜索學習率
learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

# 不好的實踐：線性搜索
learning_rate = trial.suggest_float('lr', 0.00001, 0.1)

# 好的實踐：限制搜索範圍
n_layers = trial.suggest_int('n_layers', 1, 5)

# 不好的實踐：範圍太大
n_layers = trial.suggest_int('n_layers', 1, 100)
```

### 2. 評估策略

```python
# 使用交叉驗證獲得更可靠的評估
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }

    model = RandomForestClassifier(**params)

    # 5折交叉驗證
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    return scores.mean()
```

### 3. 提前停止

```python
def objective(trial):
    model = build_model(trial)

    for epoch in range(100):
        train_loss = train_one_epoch(model)
        val_accuracy = validate(model)

        # 報告中間結果
        trial.report(val_accuracy, epoch)

        # 剪枝不理想的試驗
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_accuracy
```

### 4. 多目標優化

```python
import optuna

def objective(trial):
    # 構建和訓練模型
    model = build_model(trial)

    # 返回多個目標
    accuracy = evaluate_accuracy(model)
    model_size = get_model_size(model)
    inference_time = measure_inference_time(model)

    return accuracy, -model_size, -inference_time

# 多目標優化
study = optuna.create_study(
    directions=['maximize', 'maximize', 'maximize']
)

study.optimize(objective, n_trials=100)

# 獲取帕累托前沿
pareto_front = study.best_trials

for trial in pareto_front:
    print(f"Trial {trial.number}:")
    print(f"  Accuracy: {trial.values[0]}")
    print(f"  Model Size: {-trial.values[1]} MB")
    print(f"  Inference Time: {-trial.values[2]} ms")
```

### 5. 資源管理

```python
# 使用資源約束
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_size = trial.suggest_int('hidden_size', 64, 512)

    # 估算內存使用
    estimated_memory = estimate_memory(batch_size, hidden_size)

    # 如果超過限制，直接剪枝
    if estimated_memory > MAX_MEMORY:
        raise optuna.TrialPruned()

    # 繼續訓練
    return train_model(batch_size, hidden_size)
```

### 6. 結果分析

```python
# 分析超參數重要性
import optuna.importance as importance

# 計算重要性
importances = importance.get_param_importances(study)

print("超參數重要性:")
for param, imp in importances.items():
    print(f"{param}: {imp:.4f}")

# 可視化
import optuna.visualization as vis

# 參數關係圖
vis.plot_parallel_coordinate(study).show()

# 超參數切片圖
vis.plot_slice(study).show()

# 等高線圖
vis.plot_contour(study, params=['learning_rate', 'batch_size']).show()
```

## 總結

超參數調整的關鍵要點：

1. **選擇合適的方法**：
   - 小規模：網格搜索
   - 中規模：隨機搜索或貝葉斯優化
   - 大規模/深度學習：Hyperband 或進化演算法

2. **合理設計搜索空間**：
   - 使用領域知識縮小範圍
   - 對數尺度搜索學習率等參數
   - 優先調整重要的超參數

3. **高效評估**：
   - 使用交叉驗證
   - 實施提前停止
   - 並行化評估

4. **記錄和追蹤**：
   - 使用 MLflow 記錄所有實驗
   - 可視化優化過程
   - 分析參數重要性

5. **實際考慮**：
   - 平衡性能和成本
   - 考慮模型部署需求
   - 多目標優化

下一步建議：
- [MLflow 與超參數調整整合](../04.Integration_Examples/README.md)
- [最佳實踐](../05.Best_Practices/README.md)
