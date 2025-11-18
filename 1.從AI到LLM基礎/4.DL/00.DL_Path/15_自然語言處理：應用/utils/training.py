"""
訓練輔助工具模組
Training Utilities for NLP Models

提供訓練器、早停、學習率調度等功能
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
from pathlib import Path


class EarlyStopping:
    """早停機制"""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
    ):
        """
        Args:
            patience: 容忍的epoch數
            min_delta: 最小改進量
            mode: 'min' 或 'max'
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        檢查是否應該早停

        Args:
            score: 當前分數 (loss 或 metric)

        Returns:
            是否應該早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:  # max
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"✅ Validation score improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered!")

        return self.early_stop


class ModelCheckpoint:
    """模型檢查點保存"""

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            filepath: 保存路徑
            monitor: 監控指標
            mode: 'min' 或 'max'
            save_best_only: 是否只保存最佳模型
            verbose: 是否打印信息
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_score = None

        # 創建目錄
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: nn.Module, score: float, epoch: int):
        """保存模型"""
        if self.save_best_only:
            if self.best_score is None:
                improved = True
            elif self.mode == 'min':
                improved = score < self.best_score
            else:  # max
                improved = score > self.best_score

            if improved:
                self.best_score = score
                self._save_model(model, score, epoch)
        else:
            self._save_model(model, score, epoch)

    def _save_model(self, model: nn.Module, score: float, epoch: int):
        """保存模型"""
        filepath = str(self.filepath).format(epoch=epoch, score=score)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
        }, filepath)

        if self.verbose:
            print(f"💾 Model saved to {filepath}")


class MetricsTracker:
    """指標追蹤器"""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

    def update(self, **metrics):
        """更新指標"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_best(self, metric: str, mode: str = 'min'):
        """獲取最佳指標"""
        if metric not in self.history or not self.history[metric]:
            return None

        if mode == 'min':
            return min(self.history[metric])
        else:
            return max(self.history[metric])

    def get_history(self) -> Dict[str, List[float]]:
        """獲取歷史記錄"""
        return self.history


class Trainer:
    """統一訓練器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        metric_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: 模型
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            optimizer: 優化器
            loss_fn: 損失函數
            device: 設備
            metric_fn: 評估指標函數
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 默認優化器和損失函數
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metric_fn = metric_fn or self._accuracy

        # 指標追蹤
        self.metrics_tracker = MetricsTracker()

    def _accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """計算準確率"""
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y_true).sum().item()
        total = y_true.size(0)
        return correct / total

    def train_epoch(self) -> Tuple[float, float]:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(X)

            # 計算損失
            loss = self.loss_fn(outputs, y)

            # 反向傳播
            loss.backward()
            self.optimizer.step()

            # 記錄指標
            total_loss += loss.item()
            total_acc += self.metric_fn(outputs, y)
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc

    def validate(self) -> Tuple[float, float]:
        """驗證"""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)

                # 前向傳播
                outputs = self.model(X)

                # 計算損失
                loss = self.loss_fn(outputs, y)

                # 記錄指標
                total_loss += loss.item()
                total_acc += self.metric_fn(outputs, y)
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc

    def train(
        self,
        epochs: int,
        early_stopping: bool = False,
        patience: int = 5,
        save_best: bool = False,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        訓練模型

        Args:
            epochs: 訓練輪數
            early_stopping: 是否使用早停
            patience: 早停容忍輪數
            save_best: 是否保存最佳模型
            save_path: 模型保存路徑
            verbose: 是否打印訓練信息

        Returns:
            訓練歷史
        """
        # 早停和模型保存
        early_stopper = EarlyStopping(patience=patience, verbose=verbose) if early_stopping else None
        checkpointer = ModelCheckpoint(save_path, verbose=verbose) if save_best and save_path else None

        print(f"🚀 開始訓練 | 設備: {self.device} | Epochs: {epochs}")
        print("=" * 70)

        for epoch in range(epochs):
            start_time = time.time()

            # 訓練
            train_loss, train_acc = self.train_epoch()

            # 驗證
            val_loss, val_acc = self.validate()

            # 記錄指標
            self.metrics_tracker.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            # 打印信息
            if verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.2f}s")

            # 保存最佳模型
            if checkpointer:
                checkpointer(self.model, val_loss, epoch + 1)

            # 早停
            if early_stopper and early_stopper(val_loss):
                print(f"\n🛑 訓練在第 {epoch+1} 輪停止")
                break

        print("=" * 70)
        print("✅ 訓練完成!")
        print(f"📊 最佳驗證損失: {self.metrics_tracker.get_best('val_loss'):.4f}")
        print(f"📊 最佳驗證準確率: {self.metrics_tracker.get_best('val_acc', mode='max'):.4f}")

        return self.metrics_tracker.get_history()


class LearningRateScheduler:
    """學習率調度器"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'step',
        step_size: int = 10,
        gamma: float = 0.1,
        patience: int = 5,
        factor: float = 0.5,
    ):
        """
        Args:
            optimizer: 優化器
            mode: 調度模式 ('step', 'plateau', 'cosine')
            step_size: StepLR 的步長
            gamma: StepLR 的衰減因子
            patience: ReduceLROnPlateau 的容忍輪數
            factor: ReduceLROnPlateau 的衰減因子
        """
        self.optimizer = optimizer
        self.mode = mode

        if mode == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif mode == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=factor, verbose=True
            )
        elif mode == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=step_size
            )
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")

    def step(self, metric: Optional[float] = None):
        """更新學習率"""
        if self.mode == 'plateau':
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires metric")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        """獲取當前學習率"""
        return self.optimizer.param_groups[0]['lr']


def compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    計算準確率

    Args:
        y_pred: 預測值 (logits)
        y_true: 真實標籤

    Returns:
        準確率
    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    return correct / total


def compute_f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, average: str = 'binary') -> float:
    """
    計算 F1 分數

    Args:
        y_pred: 預測值 (logits)
        y_true: 真實標籤
        average: 'binary', 'micro', 'macro', 'weighted'

    Returns:
        F1 分數
    """
    from sklearn.metrics import f1_score

    _, predicted = torch.max(y_pred, 1)
    y_pred_np = predicted.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    return f1_score(y_true_np, y_pred_np, average=average)


def set_seed(seed: int = 42):
    """設置隨機種子以保證可重現性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to {seed}")


if __name__ == '__main__':
    print("=" * 50)
    print("訓練工具測試")
    print("=" * 50)

    # 創建簡單模型和數據
    from torch.utils.data import TensorDataset

    # 假數據
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 簡單模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    # 訓練
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    history = trainer.train(
        epochs=5,
        early_stopping=True,
        patience=3,
        verbose=True,
    )

    print("\n✅ 測試完成！")
    print(f"訓練歷史: {list(history.keys())}")
