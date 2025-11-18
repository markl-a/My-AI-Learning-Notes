"""
MLflow + Scikit-learn 整合範例
展示如何將 MLflow 與 Scikit-learn 整合進行超參數調整
"""

import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def load_data():
    """載入數據"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train):
    """Optuna 優化目標函數"""
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
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        # 5折交叉驗證
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        # 記錄指標到 MLflow
        mlflow.log_metric("cv_mean_accuracy", mean_score)
        mlflow.log_metric("cv_std_accuracy", std_score)

        # 訓練最終模型並記錄
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

        # 設定標籤
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("model_type", "RandomForest")

        print(f"Trial {trial.number}: CV Accuracy = {mean_score:.4f} (+/- {std_score:.4f})")

        return mean_score


def main():
    """主函數"""
    print("載入數據...")
    X_train, X_test, y_train, y_test = load_data()

    # 設定 MLflow 實驗
    mlflow.set_experiment("sklearn_optuna_optimization")

    # 創建父 run 來組織所有試驗
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        # 記錄優化配置
        mlflow.log_param("optimization_tool", "optuna")
        mlflow.log_param("n_trials", 50)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("dataset", "breast_cancer")

        # 創建 Optuna 研究
        print("開始超參數優化...")
        study = optuna.create_study(
            direction='maximize',
            study_name='rf_optimization'
        )

        # 執行優化
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=50,
            show_progress_bar=True
        )

        # 記錄最佳結果
        print("\n優化完成！")
        print(f"最佳參數: {study.best_params}")
        print(f"最佳交叉驗證準確率: {study.best_value:.4f}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_accuracy", study.best_value)

        # 使用最佳參數訓練最終模型
        print("\n使用最佳參數訓練最終模型...")
        best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)

        # 在測試集上評估
        y_pred = best_model.predict(X_test)

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred)
        }

        # 記錄測試集指標
        mlflow.log_metrics(test_metrics)

        # 記錄最終模型
        mlflow.sklearn.log_model(best_model, "best_model")

        print("\n測試集性能:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        # 生成並記錄 Optuna 可視化
        try:
            import optuna.visualization as vis

            print("\n生成可視化圖表...")

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

            print("可視化圖表已保存")

        except Exception as e:
            print(f"可視化生成失敗: {e}")

        # 記錄優化統計
        mlflow.log_metrics({
            "n_completed_trials": len(study.trials),
            "n_pruned_trials": len([t for t in study.trials
                                     if t.state == optuna.trial.TrialState.PRUNED])
        })

    print("\n實驗完成！請在 MLflow UI 中查看結果：http://localhost:5000")


if __name__ == "__main__":
    main()
