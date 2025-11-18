"""
MLflow + PyTorch + Optuna 整合範例
展示如何將 MLflow 與 PyTorch 整合進行深度學習模型的超參數調整
"""

import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class FlexibleNN(nn.Module):
    """可配置的神經網路"""

    def __init__(self, input_size, output_size, hidden_sizes, dropout_rates):
        super(FlexibleNN, self).__init__()

        layers = []
        in_features = input_size

        # 構建隱藏層
        for hidden_size, dropout_rate in zip(hidden_sizes, dropout_rates):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        # 輸出層
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def prepare_data(n_samples=5000, n_features=20, random_state=42):
    """準備數據"""
    # 生成數據
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=random_state
    )

    # 分割數據
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 轉換為 PyTorch 張量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs, trial=None):
    """訓練模型"""
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # 記錄到 MLflow
        mlflow.log_metrics({
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }, step=epoch)

        # Optuna 剪枝
        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
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

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, '
                  f'Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}')

    return val_accuracy


def objective(trial, X_train, y_train, X_val, y_val, input_size, output_size):
    """Optuna 優化目標函數"""
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # 超參數採樣
        n_layers = trial.suggest_int('n_layers', 1, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

        # 為每層採樣參數
        hidden_sizes = []
        dropout_rates = []
        for i in range(n_layers):
            hidden_size = trial.suggest_int(f'hidden_size_l{i}', 32, 256)
            dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            hidden_sizes.append(hidden_size)
            dropout_rates.append(dropout_rate)

        # 記錄超參數
        mlflow.log_params({
            'n_layers': n_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'trial_number': trial.number
        })

        # 記錄每層的配置
        for i, (h, d) in enumerate(zip(hidden_sizes, dropout_rates)):
            mlflow.log_params({
                f'hidden_size_l{i}': h,
                f'dropout_l{i}': d
            })

        # 構建模型
        model = FlexibleNN(input_size, output_size, hidden_sizes, dropout_rates)

        # 記錄模型參數數量
        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param('total_parameters', n_params)

        # 選擇優化器
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # 創建 DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 訓練模型
        criterion = nn.CrossEntropyLoss()

        try:
            val_accuracy = train_model(
                model, train_loader, val_loader, optimizer,
                criterion, num_epochs=50, trial=trial
            )
        except optuna.TrialPruned:
            mlflow.set_tag("pruned", True)
            raise

        # 記錄最終模型
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("final_val_accuracy", val_accuracy)

        # 設定標籤
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("model_type", "feedforward_nn")

        return val_accuracy


def main():
    """主函數"""
    print("準備數據...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    input_size = X_train.shape[1]
    output_size = 2

    print(f"訓練集大小: {len(X_train)}")
    print(f"驗證集大小: {len(X_val)}")
    print(f"測試集大小: {len(X_test)}")

    # 設定 MLflow 實驗
    mlflow.set_experiment("pytorch_optuna_optimization")

    # 執行優化
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        # 記錄配置
        mlflow.log_params({
            "optimization_method": "optuna",
            "n_trials": 30,
            "pruner": "MedianPruner"
        })

        # 創建帶剪枝的研究
        print("\n開始超參數優化...")
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )

        # 執行優化
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val,
                                     input_size, output_size),
            n_trials=30,
            show_progress_bar=True
        )

        # 記錄優化結果
        print("\n優化完成！")
        print(f"最佳參數: {study.best_params}")
        print(f"最佳驗證準確率: {study.best_value:.4f}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_accuracy", study.best_value)

        # 記錄優化統計
        mlflow.log_metrics({
            "n_completed_trials": len(study.trials),
            "n_pruned_trials": len([t for t in study.trials
                                     if t.state == optuna.trial.TrialState.PRUNED])
        })

        # 使用最佳參數訓練最終模型
        print("\n使用最佳參數訓練最終模型...")

        # 重建最佳模型
        best_params = study.best_params
        n_layers = best_params['n_layers']
        hidden_sizes = [best_params[f'hidden_size_l{i}'] for i in range(n_layers)]
        dropout_rates = [best_params[f'dropout_l{i}'] for i in range(n_layers)]

        final_model = FlexibleNN(input_size, output_size, hidden_sizes, dropout_rates)

        # 配置優化器
        optimizer_name = best_params['optimizer']
        learning_rate = best_params['learning_rate']

        if optimizer_name == 'adam':
            optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(final_model.parameters(), lr=learning_rate, momentum=0.9)

        # 訓練最終模型
        batch_size = best_params['batch_size']
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()

        final_val_accuracy = train_model(
            final_model, train_loader, val_loader, optimizer,
            criterion, num_epochs=100
        )

        # 在測試集上評估
        print("\n在測試集上評估...")
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        final_model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = final_model(batch_X)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()

        test_accuracy = test_correct / test_total

        mlflow.log_metric("final_test_accuracy", test_accuracy)
        mlflow.pytorch.log_model(final_model, "final_model")

        print(f"測試集準確率: {test_accuracy:.4f}")

        # 生成可視化
        try:
            import optuna.visualization as vis

            print("\n生成可視化圖表...")

            fig1 = vis.plot_optimization_history(study)
            fig1.write_image("optimization_history.png")
            mlflow.log_artifact("optimization_history.png")

            fig2 = vis.plot_param_importances(study)
            fig2.write_image("param_importances.png")
            mlflow.log_artifact("param_importances.png")

            print("可視化圖表已保存")

        except Exception as e:
            print(f"可視化生成失敗: {e}")

    print("\n實驗完成！請在 MLflow UI 中查看結果：http://localhost:5000")


if __name__ == "__main__":
    main()
