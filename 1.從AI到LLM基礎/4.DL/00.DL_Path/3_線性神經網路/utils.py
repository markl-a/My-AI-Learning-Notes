"""
線性神經網路實用工具模組
包含數據生成、視覺化、評估指標等通用函數
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import time
from typing import Tuple, List, Optional, Callable


# ==================== 數據生成 ====================

def synthetic_data(w: torch.Tensor, b: float, num_examples: int,
                  noise_std: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成線性回歸的合成數據

    參數：
        w: 權重向量 (d,)
        b: 偏置標量
        num_examples: 樣本數量
        noise_std: 噪聲標準差

    返回：
        features: 特徵矩陣 (num_examples, d)
        labels: 標籤向量 (num_examples, 1)
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, noise_std, y.shape)
    return X, y.reshape((-1, 1))


def generate_polynomial_data(num_examples: int = 100,
                             degree: int = 3,
                             noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成多項式回歸數據

    參數：
        num_examples: 樣本數量
        degree: 多項式階數
        noise_std: 噪聲標準差

    返回：
        X: 特徵矩陣
        y: 標籤向量
    """
    X = torch.sort(torch.rand(num_examples, 1) * 10 - 5)[0]
    y = 0
    for i in range(degree + 1):
        coef = torch.randn(1) * 2
        y = y + coef * (X ** i)
    y += torch.normal(0, noise_std, y.shape)
    return X, y


# ==================== 數據迭代器 ====================

def data_iter(batch_size: int, features: torch.Tensor,
             labels: torch.Tensor):
    """
    生成小批量數據迭代器

    參數：
        batch_size: 批次大小
        features: 特徵矩陣
        labels: 標籤向量

    生成：
        (X_batch, y_batch): 批次數據
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# ==================== 模型定義 ====================

def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """線性回歸模型"""
    return torch.matmul(X, w) + b


# ==================== 損失函數 ====================

def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """均方損失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def mse_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """均方誤差（平均）"""
    return ((y_hat - y.reshape(y_hat.shape)) ** 2).mean()


def mae_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """平均絕對誤差"""
    return torch.abs(y_hat - y.reshape(y_hat.shape)).mean()


# ==================== 優化器 ====================

def sgd(params: List[torch.Tensor], lr: float, batch_size: int):
    """小批量隨機梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ==================== 評估指標 ====================

def r_squared(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    計算 R² 分數（決定係數）

    R² = 1 - SS_res / SS_tot
    其中 SS_res = Σ(y - ŷ)²，SS_tot = Σ(y - ȳ)²
    """
    y = y.reshape(y_hat.shape)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot).item()


def mean_absolute_error(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """計算平均絕對誤差 (MAE)"""
    return torch.abs(y_hat - y.reshape(y_hat.shape)).mean().item()


def root_mean_squared_error(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """計算均方根誤差 (RMSE)"""
    return torch.sqrt(((y_hat - y.reshape(y_hat.shape)) ** 2).mean()).item()


def mean_absolute_percentage_error(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """計算平均絕對百分比誤差 (MAPE)"""
    y = y.reshape(y_hat.shape)
    return (torch.abs((y - y_hat) / y).mean() * 100).item()


# ==================== 視覺化工具 ====================

class Animator:
    """訓練過程動畫器"""

    def __init__(self, xlabel: str = 'epoch', ylabel: str = 'loss',
                 xlim: Optional[List] = None, ylim: Optional[List] = None,
                 legend: Optional[List[str]] = None, figsize: Tuple = (5, 3)):
        """
        初始化動畫器

        參數：
            xlabel: x 軸標籤
            ylabel: y 軸標籤
            xlim: x 軸範圍
            ylim: y 軸範圍
            legend: 圖例
            figsize: 圖形大小
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.legend = legend
        self.data = {}

    def add(self, x: float, *y_values: float):
        """添加數據點"""
        if not hasattr(self, 'lines'):
            self.lines = []
            for i, y in enumerate(y_values):
                if str(i) not in self.data:
                    self.data[str(i)] = {'x': [], 'y': []}
                line, = self.ax.plot([], [], label=self.legend[i] if self.legend else f'line {i}')
                self.lines.append(line)
            if self.legend:
                self.ax.legend()

        for i, y in enumerate(y_values):
            self.data[str(i)]['x'].append(x)
            self.data[str(i)]['y'].append(y)
            self.lines[i].set_data(self.data[str(i)]['x'], self.data[str(i)]['y'])

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)
        plt.draw()
        plt.pause(0.01)


def plot_regression_line(X: torch.Tensor, y: torch.Tensor,
                         w: torch.Tensor, b: torch.Tensor,
                         title: str = 'Linear Regression'):
    """
    繪製回歸直線

    參數：
        X: 特徵（僅支持1維）
        y: 標籤
        w: 權重
        b: 偏置
        title: 標題
    """
    plt.figure(figsize=(8, 6))

    # 繪製散點圖
    plt.scatter(X.detach().numpy(), y.detach().numpy(),
               alpha=0.5, label='Data points')

    # 繪製回歸直線
    X_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = linreg(X_line, w, b)
    plt.plot(X_line.detach().numpy(), y_line.detach().numpy(),
            'r-', linewidth=2, label='Regression line')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss_surface(X: torch.Tensor, y: torch.Tensor,
                     w_range: Tuple[float, float] = (-1, 5),
                     b_range: Tuple[float, float] = (-1, 5),
                     resolution: int = 50):
    """
    繪製損失函數表面（僅適用於單特徵）

    參數：
        X: 特徵矩陣 (n, 1)
        y: 標籤向量 (n, 1)
        w_range: 權重範圍
        b_range: 偏置範圍
        resolution: 網格解析度
    """
    from mpl_toolkits.mplot3d import Axes3D

    w_vals = np.linspace(*w_range, resolution)
    b_vals = np.linspace(*b_range, resolution)
    W, B = np.meshgrid(w_vals, b_vals)

    Z = np.zeros_like(W)
    for i in range(resolution):
        for j in range(resolution):
            w_temp = torch.tensor([[W[i, j]]], dtype=torch.float32)
            b_temp = torch.tensor([B[i, j]], dtype=torch.float32)
            y_hat = linreg(X, w_temp, b_temp)
            Z[i, j] = mse_loss(y_hat, y).item()

    fig = plt.figure(figsize=(12, 5))

    # 3D 表面圖
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(W, B, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('w')
    ax1.set_ylabel('b')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Surface (3D)')

    # 等高線圖
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(W, B, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('w')
    ax2.set_ylabel('b')
    ax2.set_title('Loss Contour')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_gradient_descent_path(X: torch.Tensor, y: torch.Tensor,
                               w_init: torch.Tensor, b_init: torch.Tensor,
                               lr: float = 0.1, num_steps: int = 50):
    """
    繪製梯度下降路徑

    參數：
        X: 特徵矩陣
        y: 標籤向量
        w_init: 初始權重
        b_init: 初始偏置
        lr: 學習率
        num_steps: 迭代步數
    """
    w = w_init.clone().requires_grad_(True)
    b = b_init.clone().requires_grad_(True)

    w_path = [w.item()]
    b_path = [b.item()]
    loss_path = []

    for _ in range(num_steps):
        y_hat = linreg(X, w, b)
        loss = mse_loss(y_hat, y)
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        w_path.append(w.item())
        b_path.append(b.item())
        loss_path.append(loss.item())

    # 繪製等高線和路徑
    from mpl_toolkits.mplot3d import Axes3D

    w_range = (min(w_path) - 0.5, max(w_path) + 0.5)
    b_range = (min(b_path) - 0.5, max(b_path) + 0.5)

    w_vals = np.linspace(*w_range, 50)
    b_vals = np.linspace(*b_range, 50)
    W, B = np.meshgrid(w_vals, b_vals)

    Z = np.zeros_like(W)
    for i in range(50):
        for j in range(50):
            w_temp = torch.tensor([[W[i, j]]], dtype=torch.float32)
            b_temp = torch.tensor([B[i, j]], dtype=torch.float32)
            y_hat = linreg(X, w_temp, b_temp)
            Z[i, j] = mse_loss(y_hat, y).item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 等高線 + 路徑
    contour = ax1.contour(W, B, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(w_path, b_path, 'ro-', markersize=4, linewidth=2,
            label='Gradient Descent Path')
    ax1.plot(w_path[0], b_path[0], 'go', markersize=10, label='Start')
    ax1.plot(w_path[-1], b_path[-1], 'r*', markersize=15, label='End')
    ax1.set_xlabel('w')
    ax1.set_ylabel('b')
    ax1.set_title('Gradient Descent Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 損失曲線
    ax2.plot(loss_path, linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Step')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 計時器 ====================

class Timer:
    """記錄多次運行時間"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """啟動計時器"""
        self.tik = time.time()

    def stop(self):
        """停止計時器並將時間記錄在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均時間"""
        return sum(self.times) / len(self.times) if self.times else 0

    def sum(self):
        """返回時間總和"""
        return sum(self.times)

    def cumsum(self):
        """返回累計時間"""
        return np.array(self.times).cumsum().tolist()


# ==================== 數據標準化 ====================

class StandardScaler:
    """標準化器（Z-score normalization）"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: torch.Tensor):
        """計算均值和標準差"""
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """標準化數據"""
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """擬合並轉換"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """反標準化"""
        return X * (self.std + 1e-8) + self.mean


class MinMaxScaler:
    """最小-最大標準化器"""

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, X: torch.Tensor):
        """計算最小值和最大值"""
        self.min = X.min(dim=0, keepdim=True)[0]
        self.max = X.max(dim=0, keepdim=True)[0]
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """標準化數據"""
        X_std = (X - self.min) / (self.max - self.min + 1e-8)
        return X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """擬合並轉換"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """反標準化"""
        X_std = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return X_std * (self.max - self.min + 1e-8) + self.min


# ==================== 交叉驗證 ====================

def k_fold_split(X: torch.Tensor, y: torch.Tensor, k: int = 5):
    """
    K折交叉驗證數據分割

    參數：
        X: 特徵矩陣
        y: 標籤向量
        k: 折數

    生成：
        (X_train, y_train, X_val, y_val): 訓練集和驗證集
    """
    n = len(X)
    fold_size = n // k
    indices = torch.randperm(n)

    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n

        val_indices = indices[val_start:val_end]
        train_indices = torch.cat([indices[:val_start], indices[val_end:]])

        yield X[train_indices], y[train_indices], X[val_indices], y[val_indices]


# ==================== 輔助函數 ====================

def set_random_seed(seed: int = 42):
    """設置隨機種子以確保可重現性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_model_summary(model: torch.nn.Module):
    """打印模型摘要"""
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name:30s} {str(param.shape):20s} {num_params:10d}")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)


def compare_models(models: dict, X: torch.Tensor, y: torch.Tensor):
    """
    比較多個模型的性能

    參數：
        models: 字典 {name: model}
        X: 特徵矩陣
        y: 標籤向量

    返回：
        results: 字典包含各模型的評估指標
    """
    results = {}

    for name, model in models.items():
        with torch.no_grad():
            y_hat = model(X)
            r2 = r_squared(y_hat, y)
            mae = mean_absolute_error(y_hat, y)
            rmse = root_mean_squared_error(y_hat, y)

            results[name] = {
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse
            }

    # 打印結果表格
    print("\n" + "=" * 70)
    print(f"{'Model':<20s} {'R²':<15s} {'MAE':<15s} {'RMSE':<15s}")
    print("=" * 70)
    for name, metrics in results.items():
        print(f"{name:<20s} {metrics['R²']:<15.4f} {metrics['MAE']:<15.4f} {metrics['RMSE']:<15.4f}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    # 測試工具函數
    print("Testing utils module...")

    # 測試數據生成
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    X, y = synthetic_data(true_w, true_b, 100)
    print(f"✓ Generated synthetic data: X.shape={X.shape}, y.shape={y.shape}")

    # 測試評估指標
    w = torch.tensor([[2.0], [-3.4]])
    b = torch.tensor([4.2])
    y_hat = linreg(X, w, b)
    r2 = r_squared(y_hat, y)
    print(f"✓ R² score: {r2:.4f}")

    # 測試標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Scaled data: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    print("\n所有測試通過！✓")
