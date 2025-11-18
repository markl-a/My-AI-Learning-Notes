"""
MLP 訓練實用工具集
提供訓練、評估、調試、可視化等實用功能

作者：Claude AI
日期：2025-01-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict


# ==================== 1. 訓練器類 ====================
class MLPTrainer:
    """
    MLP 訓練器
    提供完整的訓練流程，包括：
    - 訓練和驗證
    - 學習率調度
    - 早停
    - 檢查點保存
    - 訓練監控
    """

    def __init__(self, model, criterion, optimizer, device='cpu',
                 scheduler=None, early_stopping_patience=10):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

        # 訓練歷史
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向傳播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """驗證模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, num_epochs, verbose=True):
        """訓練模型"""
        for epoch in range(num_epochs):
            start_time = time.time()

            # 訓練和驗證
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 學習率調度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 早停
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

            # 打印進度
            if verbose:
                epoch_time = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1:3d}/{num_epochs} - '
                      f'Time: {epoch_time:.2f}s - '
                      f'LR: {lr:.6f} - '
                      f'Train Loss: {train_loss:.4f} - '
                      f'Train Acc: {train_acc:.2f}% - '
                      f'Val Loss: {val_loss:.4f} - '
                      f'Val Acc: {val_acc:.2f}%')

        # 載入最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def plot_history(self):
        """繪製訓練歷史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 損失曲線
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # 準確率曲線
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()


# ==================== 2. 模型分析工具 ====================
class ModelAnalyzer:
    """模型分析工具"""

    @staticmethod
    def count_parameters(model):
        """計算模型參數數量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"總參數數量: {total_params:,}")
        print(f"可訓練參數: {trainable_params:,}")

        # 按層統計
        print("\n各層參數數量:")
        for name, param in model.named_parameters():
            print(f"  {name:40s}: {param.numel():,}")

        return total_params, trainable_params

    @staticmethod
    def analyze_gradients(model):
        """分析模型梯度"""
        grad_dict = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'min': param.grad.min().item(),
                    'max': param.grad.max().item(),
                    'norm': param.grad.norm().item()
                }

        return grad_dict

    @staticmethod
    def plot_gradients(model):
        """可視化梯度分布"""
        gradients = []
        names = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.cpu().numpy().flatten())
                names.append(name)

        fig, axes = plt.subplots(len(gradients), 1, figsize=(12, 3 * len(gradients)))
        if len(gradients) == 1:
            axes = [axes]

        for idx, (grad, name) in enumerate(zip(gradients, names)):
            axes[idx].hist(grad, bins=50, alpha=0.7)
            axes[idx].set_title(f'Gradient Distribution - {name}')
            axes[idx].set_xlabel('Gradient Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_activations(model, dataloader, device='cpu'):
        """分析激活值分布"""
        model.eval()
        activations = defaultdict(list)

        # 註冊鉤子函數
        def get_activation(name):
            def hook(model, input, output):
                activations[name].append(output.detach().cpu())
            return hook

        hooks = []
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.GELU)):
                hooks.append(layer.register_forward_hook(get_activation(name)))

        # 前向傳播一個批次
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                model(data)
                break

        # 移除鉤子
        for hook in hooks:
            hook.remove()

        # 計算統計信息
        stats = {}
        for name, acts in activations.items():
            act_tensor = torch.cat(acts, dim=0)
            stats[name] = {
                'mean': act_tensor.mean().item(),
                'std': act_tensor.std().item(),
                'min': act_tensor.min().item(),
                'max': act_tensor.max().item()
            }

        return stats


# ==================== 3. 超參數搜索工具 ====================
class HyperparameterSearch:
    """超參數搜索工具"""

    @staticmethod
    def grid_search(model_fn, param_grid, train_loader, val_loader, num_epochs=5):
        """網格搜索"""
        results = []

        # 生成所有參數組合
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n測試組合 {idx+1}/{len(combinations)}: {params}")

            # 創建模型
            model = model_fn(**params)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 訓練
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
            trainer = MLPTrainer(model, criterion, optimizer, device)
            history = trainer.fit(train_loader, val_loader, num_epochs, verbose=False)

            # 記錄結果
            result = {
                'params': params,
                'best_val_acc': max(history['val_acc']),
                'final_val_acc': history['val_acc'][-1],
                'history': history
            }
            results.append(result)

            print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")

        # 排序結果
        results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
        return results

    @staticmethod
    def random_search(model_fn, param_ranges, train_loader, val_loader,
                      num_trials=10, num_epochs=5):
        """隨機搜索"""
        results = []

        for trial in range(num_trials):
            # 隨機採樣參數
            params = {}
            for key, (min_val, max_val, scale) in param_ranges.items():
                if scale == 'log':
                    params[key] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
                elif scale == 'int':
                    params[key] = np.random.randint(min_val, max_val + 1)
                else:  # linear
                    params[key] = np.random.uniform(min_val, max_val)

            print(f"\n試驗 {trial+1}/{num_trials}: {params}")

            # 訓練模型
            model = model_fn(**params)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
            trainer = MLPTrainer(model, criterion, optimizer, device)
            history = trainer.fit(train_loader, val_loader, num_epochs, verbose=False)

            # 記錄結果
            result = {
                'params': params,
                'best_val_acc': max(history['val_acc']),
                'final_val_acc': history['val_acc'][-1]
            }
            results.append(result)

            print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")

        # 排序結果
        results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
        return results


# ==================== 4. 模型可視化工具 ====================
class ModelVisualizer:
    """模型可視化工具"""

    @staticmethod
    def visualize_weights(model, layer_name=None):
        """可視化權重矩陣"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for name, param in model.named_parameters():
            if layer_name is None or layer_name in name:
                if 'weight' in name and len(param.shape) == 2:
                    weights = param.detach().cpu().numpy()

                    # 權重熱圖
                    im0 = axes[0].imshow(weights, cmap='viridis', aspect='auto')
                    axes[0].set_title(f'Weight Matrix - {name}')
                    axes[0].set_xlabel('Input Dimension')
                    axes[0].set_ylabel('Output Dimension')
                    plt.colorbar(im0, ax=axes[0])

                    # 權重分布
                    axes[1].hist(weights.flatten(), bins=50, alpha=0.7)
                    axes[1].set_title(f'Weight Distribution - {name}')
                    axes[1].set_xlabel('Weight Value')
                    axes[1].set_ylabel('Frequency')
                    axes[1].grid(True)

                    break

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_decision_boundary(model, X, y, device='cpu'):
        """可視化決策邊界（僅適用於2D輸入）"""
        if X.shape[1] != 2:
            print("僅支持2D輸入數據")
            return

        model.eval()
        h = 0.02  # 網格步長

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # 預測
        grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
        with torch.no_grad():
            Z = model(grid_tensor)
            Z = Z.argmax(dim=1).cpu().numpy()

        Z = Z.reshape(xx.shape)

        # 繪圖
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.colorbar()
        plt.show()


# ==================== 5. 調試工具 ====================
class DebugHelper:
    """訓練調試助手"""

    @staticmethod
    def check_gradient_flow(model):
        """檢查梯度流"""
        ave_grads = []
        max_grads = []
        layers = []

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label='max gradient')
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, label='mean gradient')
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=2, color='k')
        plt.xticks(range(len(ave_grads)), layers, rotation='vertical')
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001)
        plt.xlabel('Layers')
        plt.ylabel('Gradient Magnitude')
        plt.title('Gradient Flow')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def diagnose_training(history):
        """診斷訓練問題"""
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        train_acc = history['train_acc']
        val_acc = history['val_acc']

        print("訓練診斷報告")
        print("=" * 50)

        # 檢查過擬合
        if val_loss[-1] > train_loss[-1] * 1.5:
            print("⚠️ 檢測到過擬合問題")
            print("建議:")
            print("  1. 增加 Dropout")
            print("  2. 使用權重衰減")
            print("  3. 增加訓練數據")
            print("  4. 減少模型複雜度")

        # 檢查欠擬合
        if train_acc[-1] < 80:
            print("⚠️ 檢測到欠擬合問題")
            print("建議:")
            print("  1. 增加模型複雜度")
            print("  2. 訓練更多 epoch")
            print("  3. 調整學習率")
            print("  4. 檢查數據預處理")

        # 檢查訓練穩定性
        loss_std = np.std(train_loss[-10:])
        if loss_std > 0.5:
            print("⚠️ 訓練不穩定")
            print("建議:")
            print("  1. 降低學習率")
            print("  2. 使用梯度裁剪")
            print("  3. 使用批次正規化")

        # 檢查收斂
        if abs(train_loss[-1] - train_loss[-5]) < 0.001:
            print("✓ 模型已收斂")
        else:
            print("⚠️ 模型可能還需要更多訓練")

        print("=" * 50)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # 載入數據
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # 分割訓練集和驗證集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 創建模型
    from advanced_mlp_architectures import ModernMLP

    model = ModernMLP(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)

    # 分析模型
    print("模型結構分析:")
    ModelAnalyzer.count_parameters(model)

    # 訓練模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    trainer = MLPTrainer(
        model, criterion, optimizer, device,
        scheduler=scheduler, early_stopping_patience=10
    )

    print("\n開始訓練...")
    history = trainer.fit(train_loader, val_loader, num_epochs=20)

    # 可視化訓練過程
    trainer.plot_history()

    # 診斷訓練
    DebugHelper.diagnose_training(history)
