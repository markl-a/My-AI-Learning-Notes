# Optimization_Basics.md

## 目錄
1. 前言
2. 最優化問題的基本概念
    - 目標函式 (Objective Function)、損失函式 (Loss Function)
    - 全域極小值 (Global Minimum) 與局部極小值 (Local Minimum)
    - 鞍點 (Saddle Point) 與平坦區域
    - 凸問題 (Convex Problem) 與非凸問題 (Non-Convex Problem)
3. 基於梯度的一階優化方法
    - 導數、偏導數與梯度 (Gradient)
    - 梯度下降法 (Gradient Descent)
    - 學習率 (Learning Rate) 選擇與調整策略
    - 動量 (Momentum) 與慣性概念
4. 數值計算考量與技巧
    - 浮點數表示與有限精度問題
    - 上溢 (Overflow) 與下溢 (Underflow) 與對數域計算 (Log-space Computation)
    - 病態條件 (Ill-conditioning) 與條件數 (Condition Number) 的影響
    - 數值穩定 (Numerical Stability) 的方法：如使用 `log-sum-exp` 技巧、對輸入正規化
    - 避免在優化過程中因數值誤差導致梯度爆炸 (Exploding Gradient) 或梯度消失 (Vanishing Gradient)
5. 加速收斂的策略與變形
    - 動量 (Momentum) 方法：Nesterov Accelerated Gradient (NAG)
    - 自適應學習率方法：Adagrad、RMSProp、Adam
    - 隨機梯度下降 (Stochastic Gradient Descent, SGD) 與 Mini-batch SGD
    - 隨機性 (Stochasticity) 在優化中的角色
6. 二階與高階方法
    - Hessian 矩陣 (Hessian Matrix) 與二階微分
    - 牛頓法 (Newton’s Method) 與擬牛頓法 (Quasi-Newton Methods)
    - 二階方法的數值問題與計算成本
7. 約束優化 (Constrained Optimization) 基礎
    - 以 Lagrange 乘子法 (Lagrange Multipliers) 處理約束
    - KKT 條件 (Karush–Kuhn–Tucker Conditions) 的概念
    - 實務中常用的簡單投影法 (Projection) 處理參數約束
8. 實務中優化策略的選擇與調適
    - 超參數 (Hyperparameter) 調整：學習率衰減策略 (Learning Rate Decay)、預熱 (Warmup)
    - 混合精度訓練 (Mixed Precision Training) 以改善數值穩定與加速運算
    - Early Stopping、正則化 (Regularization) 與模型優化的平衡
9. 實務案例與建議
    - 深度學習中常見使用 SGD with Momentum 或 Adam
    - 大規模資料集與分佈式訓練下的優化考量
    - 將數值穩定技巧融入損失函式與模型架構設計
10. 延伸閱讀與參考資源

---

## 1. 前言

優化 (Optimization) 是機器學習與深度學習中不可或缺的核心步驟。我們在訓練模型時，需透過對參數空間進行搜索，找到使損失函式 (例如交叉熵、均方誤差) 值最小的參數組合。深度學習中的優化問題通常極為複雜、非凸且含有許多局部極小點、鞍點以及高維度的平坦區域。

本章將介紹優化的基本概念與常用的一階法，重點在梯度下降及其變形（如動量法、自適應學習率法則等）。同時，本章也加入在數值計算上的考量，包括如何處理浮點誤差、上溢下溢問題，並簡述二階方法及約束優化的概念，最後提供實務選擇策略和學習資源。

## 2. 最優化問題的基本概念

- **目標函式 / 損失函式**：優化的目標為最小化或最大化一函數 f(x)。在機器學習中通常最小化訓練損失。

- **全域極小值 (Global Minimum)**：f(x*) ≤ f(x) 對任意 x 成立，x* 為全域最小點。但對複雜非凸問題，找到全域最小往往很難。

- **局部極小值 (Local Minimum)**：f(x*) ≤ f(x) 在 x 鄰近區域成立。在非凸問題中，我們常只能期望找到局部極小點或夠低的代價函式值即可。

- **鞍點 (Saddle Point)**：一點處梯度為 0，但該點同時在某些方向上彎曲向上、在某些方向上彎曲向下，不是極小或極大。在高維空間中，鞍點與平坦區域比局部極小點更加普遍。

- **凸問題 (Convex Problem)**：若目標函式為凸函數，任何局部極小點即為全域極小點。可惜深度學習中多為非凸問題。

## 3. 基於梯度的一階優化方法

- **梯度 (Gradient)**：目標函式對參數向量的偏導數組成的向量，指出上升最快方向。向相反方向走即下降最快。

- **梯度下降法 (Gradient Descent)**：反覆更新 x ← x − ϵ∇f(x)，其中 ϵ 為學習率。若 ϵ 適中且梯度方向可靠，能使 f(x) 遞減。

- **學習率 (Learning Rate)**：控制每步更新幅度。過大可能震盪或發散，過小則收斂慢。實務中可使用動態調整策略，如 Learning Rate Decay 或 Learning Rate Scheduler。

- **動量 (Momentum)**：引入速度概念。將前次梯度更新累積為動量項，可減少優化路徑中不必要的小震動，使收斂更平滑和快速。

## 4. 數值計算考量與技巧

在優化時需處理與數值相關的問題：

- **浮點數近似與有限精度**：電腦中實數以有限位元表示，逼近真實值，導致捨入誤差。

- **上溢 (Overflow) 與下溢 (Underflow)**：計算 exp(x) 等函數時，x 過大或過小會使結果逼近∞或0並喪失精度。  
  解決方法：  
  - 使用對數域計算 (log-space) 來避免下溢。  
  - 在 softmax 計算中先減去最大值，以確保數值穩定。

- **病態條件 (Ill-conditioning)**：若 Hessian 矩陣或問題本身條件數很大，則小幅參數改變也導致目標值巨大變化，需更保守的更新或使用更有彈性的優化器。

- **數值穩定性 (Numerical Stability)**：  
  - 使用 log-sum-exp 計算來避免概率歸一化時的下溢。  
  - 使用適度正則化、梯度裁剪 (Gradient Clipping) 來避免梯度爆炸。  
  - 適度初始化參數和正規化 (Normalization) 輔助收斂。

## 5. 加速收斂的策略與變形

- **Momentum**：透過動量項 (velocity) 將更新方向平滑化，如 v ← βv + (1−β)∇f(x)，x ← x−ϵv。

- **Nesterov Accelerated Gradient (NAG)**：先暫移動一步再計算梯度，能更精準預測下一步動向。

- **自適應學習率方法 (Adaptive Methods)**：  
  - Adagrad：對經常更新的參數給予較小的學習率；對很少更新的參數給予較大學習率。  
  - RMSProp：對梯度平方移動平均以調整學習率。  
  - Adam：同時考慮動量 (一階矩) 與梯度平方 (二階矩) 的估計，自適應校正更新步伐。

- **隨機梯度下降 (SGD)**：每次利用小批次 (mini-batch) 樣本估計梯度，可在高維大資料環境中更快速地進行參數更新。

## 6. 二階與高階方法

- **Hessian 矩陣**：包含二階偏導數，提供函數曲率資訊。

- **牛頓法 (Newton’s Method)**：利用 Hessian 逆矩陣找到臨界點，理想情況下一步即可到達局部極小。但實務上 Hessian 計算昂貴、對數值精度要求高。對非凸問題易陷於鞍點或不合適的臨界點。

- **擬牛頓法 (Quasi-Newton)**：如 L-BFGS，以較低代價近似 Hessian，但在深度學習大規模問題中仍不常使用。

## 7. 約束優化 (Constrained Optimization) 基礎

- 若有參數約束（例如參數必須非負或範數限制），可採  
  - Lagrange 乘子法導出 KKT 條件  
  - 投影法 (Projection) 將更新後的解投回可行域  
  - 在深度學習中較少明示使用 KKT，但某些正則化或特殊層結構等價於對參數的隱性約束。

## 8. 實務中優化策略的選擇與調適

- 選擇優化方法時，SGD with Momentum 或 Adam 是常見首選。  
- 對學習率做衰減，如在訓練中期降低學習率，加速前期收斂並在後期微調。
- 使用混合精度 (Mixed Precision) 計算，可加速訓練，並需留意數值穩定性。

- 過度擬合時，可以透過正則化策略（L2、Dropout）同時改善泛化能力與優化平穩度。

## 9. 實務案例與建議

- 絕大多數深度學習任務直接使用 Adam 或 SGD+Momentum 即可獲得不錯結果。
- 面對梯度消失，可在網路設計上引入殘差結構 (Residual Connections) 或適當初始化。
- 面對梯度爆炸，可嘗試梯度裁剪、正則化、或更小的學習率。

## 10. Python 實作範例

### 10.1 基礎梯度下降實現

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_1d(f, df, x0, learning_rate=0.1, n_iterations=50):
    """
    一維函數的梯度下降

    Args:
        f: 目標函式
        df: 目標函式的導數
        x0: 初始點
        learning_rate: 學習率
        n_iterations: 迭代次數
    """
    history = [x0]
    x = x0

    for i in range(n_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)

    return np.array(history)

# 測試：最小化 f(x) = (x - 3)^2
def f(x):
    return (x - 3)**2

def df(x):
    return 2 * (x - 3)

# 不同學習率的比較
learning_rates = [0.01, 0.1, 0.5, 0.9]
x_range = np.linspace(-1, 7, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    history = gradient_descent_1d(f, df, x0=0.0, learning_rate=lr, n_iterations=20)

    axes[idx].plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x)')
    axes[idx].plot(history, f(history), 'ro-', markersize=4, alpha=0.6, label='GD path')
    axes[idx].plot(3, f(3), 'g*', markersize=15, label='Minimum')
    axes[idx].set_title(f'Learning Rate = {lr}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('f(x)')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_learning_rates.png', dpi=150, bbox_inches='tight')
plt.close()

print("梯度下降結果:")
for lr in learning_rates:
    history = gradient_descent_1d(f, df, x0=0.0, learning_rate=lr, n_iterations=20)
    print(f"學習率 {lr:4.2f}: 最終值 x = {history[-1]:.6f}, f(x) = {f(history[-1]):.6f}")
```

### 10.2 從零實現各種優化器

```python
import numpy as np
import matplotlib.pyplot as plt

class Optimizer:
    """優化器基類"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, params, grads):
        """執行一步優化"""
        raise NotImplementedError

class SGD(Optimizer):
    """隨機梯度下降"""
    def step(self, params, grads):
        return params - self.lr * grads

class Momentum(Optimizer):
    """動量法"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params + self.velocity

class NAG(Optimizer):
    """Nesterov Accelerated Gradient"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        v_prev = self.velocity.copy()
        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params - self.momentum * v_prev + (1 + self.momentum) * self.velocity

class Adagrad(Optimizer):
    """Adagrad 優化器"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.sum_squared_grads = None

    def step(self, params, grads):
        if self.sum_squared_grads is None:
            self.sum_squared_grads = np.zeros_like(params)

        self.sum_squared_grads += grads**2
        adjusted_grad = grads / (np.sqrt(self.sum_squared_grads) + self.epsilon)
        return params - self.lr * adjusted_grad

class RMSprop(Optimizer):
    """RMSprop 優化器"""
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.squared_grads = None

    def step(self, params, grads):
        if self.squared_grads is None:
            self.squared_grads = np.zeros_like(params)

        self.squared_grads = (self.decay_rate * self.squared_grads +
                             (1 - self.decay_rate) * grads**2)
        adjusted_grad = grads / (np.sqrt(self.squared_grads) + self.epsilon)
        return params - self.lr * adjusted_grad

class Adam(Optimizer):
    """Adam 優化器"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一階矩估計
        self.v = None  # 二階矩估計
        self.t = 0     # 時間步

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新偏差一階矩和二階矩
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # 更新參數
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# 測試所有優化器
def rosenbrock(x):
    """Rosenbrock 函數（標準測試函數）"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Rosenbrock 函數的梯度"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def test_optimizer(optimizer, n_iterations=200):
    """測試優化器"""
    x = np.array([-1.0, 2.5])  # 起始點
    path = [x.copy()]

    for _ in range(n_iterations):
        grad = rosenbrock_grad(x)
        x = optimizer.step(x, grad)
        path.append(x.copy())

    return np.array(path)

# 運行所有優化器
optimizers = {
    'SGD': SGD(learning_rate=0.001),
    'Momentum': Momentum(learning_rate=0.001, momentum=0.9),
    'NAG': NAG(learning_rate=0.001, momentum=0.9),
    'Adagrad': Adagrad(learning_rate=0.1),
    'RMSprop': RMSprop(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.01)
}

paths = {}
for name, opt in optimizers.items():
    paths[name] = test_optimizer(opt)
    final_x = paths[name][-1]
    final_loss = rosenbrock(final_x)
    print(f"{name:10s}: 最終位置 = [{final_x[0]:.4f}, {final_x[1]:.4f}], "
          f"損失 = {final_loss:.6f}")

# 視覺化優化路徑
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

for idx, (name, path) in enumerate(paths.items()):
    ax = axes[idx]

    # 繪製等高線
    contour = ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20),
                         cmap='gray', alpha=0.4)

    # 繪製優化路徑
    ax.plot(path[:, 0], path[:, 1], color=colors[idx], marker='o',
            markersize=2, linewidth=2, alpha=0.7, label=name)

    # 標記起點和終點
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End')
    ax.plot(1, 1, 'yd', markersize=12, label='Global Min')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{name} Optimizer')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison_rosenbrock.png', dpi=150, bbox_inches='tight')
plt.close()
```

### 10.3 學習率調度策略

```python
import numpy as np
import matplotlib.pyplot as plt

class LearningRateScheduler:
    """學習率調度器基類"""
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def get_lr(self, epoch):
        raise NotImplementedError

class StepDecay(LearningRateScheduler):
    """階梯衰減"""
    def __init__(self, initial_lr, drop_rate=0.5, epochs_drop=10):
        super().__init__(initial_lr)
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def get_lr(self, epoch):
        return self.initial_lr * (self.drop_rate ** (epoch // self.epochs_drop))

class ExponentialDecay(LearningRateScheduler):
    """指數衰減"""
    def __init__(self, initial_lr, decay_rate=0.95):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_rate ** epoch)

class CosineAnnealing(LearningRateScheduler):
    """餘弦退火"""
    def __init__(self, initial_lr, T_max, eta_min=0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, epoch):
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max)) / 2

class WarmupCosine(LearningRateScheduler):
    """帶預熱的餘弦退火"""
    def __init__(self, initial_lr, warmup_epochs, T_max, eta_min=0):
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            # 線性預熱
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 餘弦退火
            progress = (epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            return self.eta_min + (self.initial_lr - self.eta_min) * \
                   (1 + np.cos(np.pi * progress)) / 2

class OneCycleLR(LearningRateScheduler):
    """One Cycle 學習率策略"""
    def __init__(self, max_lr, total_epochs, pct_start=0.3):
        super().__init__(max_lr)
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start

    def get_lr(self, epoch):
        if epoch < self.pct_start * self.total_epochs:
            # 上升階段
            progress = epoch / (self.pct_start * self.total_epochs)
            return self.max_lr * progress
        else:
            # 下降階段
            progress = (epoch - self.pct_start * self.total_epochs) / \
                      ((1 - self.pct_start) * self.total_epochs)
            return self.max_lr * (1 - progress)

# 視覺化所有調度策略
epochs = 100
schedulers = {
    'Step Decay': StepDecay(initial_lr=0.1, drop_rate=0.5, epochs_drop=20),
    'Exponential Decay': ExponentialDecay(initial_lr=0.1, decay_rate=0.96),
    'Cosine Annealing': CosineAnnealing(initial_lr=0.1, T_max=epochs),
    'Warmup + Cosine': WarmupCosine(initial_lr=0.1, warmup_epochs=10, T_max=epochs),
    'One Cycle': OneCycleLR(max_lr=0.1, total_epochs=epochs, pct_start=0.3)
}

plt.figure(figsize=(14, 8))

for name, scheduler in schedulers.items():
    lrs = [scheduler.get_lr(e) for e in range(epochs)]
    plt.plot(range(epochs), lrs, linewidth=2, label=name)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('learning_rate_schedules.png', dpi=150, bbox_inches='tight')
plt.close()

print("學習率調度策略已視覺化")
```

### 10.4 梯度裁剪演示

```python
import numpy as np
import matplotlib.pyplot as plt

def clip_gradient_norm(grads, max_norm):
    """基於範數的梯度裁剪"""
    total_norm = np.linalg.norm(grads)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return grads * clip_coef
    return grads

def clip_gradient_value(grads, clip_value):
    """基於值的梯度裁剪"""
    return np.clip(grads, -clip_value, clip_value)

# 演示梯度裁剪的效果
np.random.seed(42)
gradients = np.random.randn(1000) * 5  # 模擬梯度

# 添加一些異常大的梯度
gradients[::100] *= 10

# 不同的裁剪策略
clipped_norm_1 = np.array([clip_gradient_norm(np.array([g]), 1.0)[0] for g in gradients])
clipped_norm_5 = np.array([clip_gradient_norm(np.array([g]), 5.0)[0] for g in gradients])
clipped_value = clip_gradient_value(gradients, 5.0)

# 視覺化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(gradients, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title('Original Gradients')
axes[0, 0].set_xlabel('Gradient Value')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(clipped_norm_1, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title('Norm Clipping (max_norm=1.0)')
axes[0, 1].set_xlabel('Gradient Value')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(clipped_norm_5, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_title('Norm Clipping (max_norm=5.0)')
axes[1, 0].set_xlabel('Gradient Value')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(clipped_value, bins=50, alpha=0.7, color='red', edgecolor='black')
axes[1, 1].set_title('Value Clipping (clip_value=5.0)')
axes[1, 1].set_xlabel('Gradient Value')
axes[1, 1].set_ylabel('Frequency')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_clipping.png', dpi=150, bbox_inches='tight')
plt.close()

print("統計資訊:")
print(f"原始梯度 - 均值: {gradients.mean():.4f}, 標準差: {gradients.std():.4f}, "
      f"最大值: {gradients.max():.4f}, 最小值: {gradients.min():.4f}")
print(f"範數裁剪(1.0) - 均值: {clipped_norm_1.mean():.4f}, 標準差: {clipped_norm_1.std():.4f}, "
      f"最大值: {clipped_norm_1.max():.4f}, 最小值: {clipped_norm_1.min():.4f}")
print(f"值裁剪(5.0) - 均值: {clipped_value.mean():.4f}, 標準差: {clipped_value.std():.4f}, "
      f"最大值: {clipped_value.max():.4f}, 最小值: {clipped_value.min():.4f}")
```

### 10.5 損失曲面視覺化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualize_loss_surface():
    """視覺化不同類型的損失曲面"""

    # 1. 凸函數（容易優化）
    def convex_function(x, y):
        return x**2 + y**2

    # 2. 非凸函數（有局部最小值）
    def non_convex_function(x, y):
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

    # 3. Rosenbrock 函數（狹長的山谷）
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    # 4. 鞍點函數
    def saddle_function(x, y):
        return x**2 - y**2

    functions = [
        (convex_function, 'Convex Function (Easy)', (-2, 2), (-2, 2)),
        (non_convex_function, 'Non-Convex (Local Minima)', (-5, 5), (-5, 5)),
        (rosenbrock, 'Rosenbrock (Long Valley)', (-2, 2), (-1, 3)),
        (saddle_function, 'Saddle Point', (-2, 2), (-2, 2))
    ]

    fig = plt.figure(figsize=(16, 12))

    for idx, (func, title, x_range, y_range) in enumerate(functions, 1):
        # 3D 視圖
        ax1 = fig.add_subplot(4, 2, idx*2-1, projection='3d')

        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        # 避免 Rosenbrock 函數值過大
        if 'Rosenbrock' in title:
            Z = np.clip(Z, 0, 500)

        surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        ax1.set_title(f'{title} - 3D View')

        # 等高線視圖
        ax2 = fig.add_subplot(4, 2, idx*2)

        if 'Rosenbrock' in title:
            levels = np.logspace(-1, 2.5, 20)
            contour = ax2.contour(X, Y, Z, levels=levels, cmap='viridis')
        else:
            contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')

        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'{title} - Contour')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loss_surface_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("損失曲面視覺化完成")

visualize_loss_surface()
```

### 10.6 在神經網路上應用不同優化器

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成非線性分類資料
def generate_spiral_data(n_samples=100, n_classes=3):
    """生成螺旋狀資料"""
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)

    for class_num in range(n_classes):
        ix = range(n_samples * class_num, n_samples * (class_num + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_num * 4, (class_num + 1) * 4, n_samples) + \
            np.random.randn(n_samples) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_num

    return X, y

class TwoLayerNet:
    """兩層神經網路"""
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier 初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """前向傳播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2

        # Softmax
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return self.probs

    def compute_loss(self, X, y):
        """計算損失"""
        m = X.shape[0]
        probs = self.forward(X)

        # 交叉熵損失
        correct_logprobs = -np.log(probs[range(m), y] + 1e-10)
        loss = np.sum(correct_logprobs) / m

        return loss

    def backward(self, X, y):
        """反向傳播"""
        m = X.shape[0]

        # 輸出層梯度
        dscores = self.probs.copy()
        dscores[range(m), y] -= 1
        dscores /= m

        # 第二層梯度
        dW2 = self.a1.T @ dscores
        db2 = np.sum(dscores, axis=0, keepdims=True)

        # 隱藏層梯度
        dhidden = dscores @ self.W2.T
        dhidden[self.z1 <= 0] = 0  # ReLU 梯度

        # 第一層梯度
        dW1 = X.T @ dhidden
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def get_params(self):
        """獲取參數（展平）"""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel()
        ])

    def set_params(self, params):
        """設置參數"""
        W1_size = self.W1.size
        b1_size = self.b1.size
        W2_size = self.W2.size

        self.W1 = params[:W1_size].reshape(self.W1.shape)
        self.b1 = params[W1_size:W1_size+b1_size].reshape(self.b1.shape)
        self.W2 = params[W1_size+b1_size:W1_size+b1_size+W2_size].reshape(self.W2.shape)
        self.b2 = params[W1_size+b1_size+W2_size:].reshape(self.b2.shape)

def train_network(X, y, optimizer_name='Adam', n_epochs=1000):
    """訓練網路"""
    np.random.seed(42)
    net = TwoLayerNet(input_size=2, hidden_size=20, output_size=3)

    # 建立優化器
    if optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=0.1)
    elif optimizer_name == 'Momentum':
        optimizer = Momentum(learning_rate=0.1, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=0.01)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        # 計算損失
        loss = net.compute_loss(X, y)
        losses.append(loss)

        # 計算準確率
        probs = net.forward(X)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)

        # 反向傳播
        grads = net.backward(X, y)

        # 更新參數
        net.W1 = optimizer.step(net.W1, grads['W1'])
        net.b1 = optimizer.step(net.b1, grads['b1'])
        net.W2 = optimizer.step(net.W2, grads['W2'])
        net.b2 = optimizer.step(net.b2, grads['b2'])

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return net, losses, accuracies

# 生成資料
X, y = generate_spiral_data(n_samples=100, n_classes=3)

# 測試不同優化器
optimizers_to_test = ['SGD', 'Momentum', 'RMSprop', 'Adam']
results = {}

for opt_name in optimizers_to_test:
    print(f"\n訓練使用 {opt_name} 優化器:")
    print("=" * 50)
    net, losses, accuracies = train_network(X, y, optimizer_name=opt_name, n_epochs=500)
    results[opt_name] = {'net': net, 'losses': losses, 'accuracies': accuracies}

# 視覺化結果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 損失曲線
ax1 = axes[0, 0]
for opt_name in optimizers_to_test:
    ax1.plot(results[opt_name]['losses'], label=opt_name, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 準確率曲線
ax2 = axes[0, 1]
for opt_name in optimizers_to_test:
    ax2.plot(results[opt_name]['accuracies'], label=opt_name, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 決策邊界（Adam）
ax3 = axes[1, 0]
net = results['Adam']['net']
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = net.forward(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
ax3.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
ax3.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors='black')
ax3.set_title('Decision Boundary (Adam)')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')

# 最終準確率比較
ax4 = axes[1, 1]
final_accs = [results[opt]['accuracies'][-1] for opt in optimizers_to_test]
bars = ax4.bar(optimizers_to_test, final_accs, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax4.set_ylabel('Final Accuracy')
ax4.set_title('Final Accuracy Comparison')
ax4.set_ylim([0, 1])
ax4.grid(True, alpha=0.3, axis='y')

# 在柱狀圖上標註數值
for bar, acc in zip(bars, final_accs):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('optimizer_comparison_nn.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n最終結果:")
for opt_name in optimizers_to_test:
    final_loss = results[opt_name]['losses'][-1]
    final_acc = results[opt_name]['accuracies'][-1]
    print(f"{opt_name:10s}: Loss = {final_loss:.4f}, Accuracy = {final_acc:.4f}")
```

### 10.7 數值穩定性技巧

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax_naive(x):
    """樸素的 softmax（數值不穩定）"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_stable(x):
    """數值穩定的 softmax"""
    # 減去最大值避免溢出
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def log_sum_exp_naive(x):
    """樸素的 log-sum-exp"""
    return np.log(np.sum(np.exp(x)))

def log_sum_exp_stable(x):
    """數值穩定的 log-sum-exp"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# 測試數值穩定性
print("=" * 60)
print("數值穩定性演示")
print("=" * 60)

# 測試 1: Softmax
print("\n1. Softmax 穩定性測試")
print("-" * 60)

# 正常範圍的輸入
x_normal = np.array([1.0, 2.0, 3.0])
print(f"正常輸入: {x_normal}")
print(f"樸素 softmax: {softmax_naive(x_normal)}")
print(f"穩定 softmax: {softmax_stable(x_normal)}")

# 大數值輸入
x_large = np.array([1000.0, 1001.0, 1002.0])
print(f"\n大數值輸入: {x_large}")
try:
    result_naive = softmax_naive(x_large)
    print(f"樸素 softmax: {result_naive} (可能出現 nan)")
except:
    print(f"樸素 softmax: 計算失敗（溢出）")
print(f"穩定 softmax: {softmax_stable(x_large)}")

# 測試 2: Log-Sum-Exp
print("\n2. Log-Sum-Exp 穩定性測試")
print("-" * 60)

x_normal = np.array([1.0, 2.0, 3.0])
print(f"正常輸入: {x_normal}")
print(f"樸素 LSE: {log_sum_exp_naive(x_normal):.6f}")
print(f"穩定 LSE: {log_sum_exp_stable(x_normal):.6f}")

x_large = np.array([1000.0, 1001.0, 1002.0])
print(f"\n大數值輸入: {x_large}")
try:
    result_naive = log_sum_exp_naive(x_large)
    print(f"樸素 LSE: {result_naive} (可能為 inf)")
except:
    print(f"樸素 LSE: 計算失敗（溢出）")
print(f"穩定 LSE: {log_sum_exp_stable(x_large):.6f}")

# 測試 3: 梯度消失和爆炸
print("\n3. 梯度消失/爆炸演示")
print("-" * 60)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# 模擬深層網路的梯度傳播
def simulate_gradient_flow(n_layers, weight_scale, use_clipping=False, clip_value=1.0):
    """模擬梯度在深層網路中的流動"""
    gradient = 1.0
    gradients = [gradient]

    for layer in range(n_layers):
        # 權重梯度
        weight_grad = np.random.randn() * weight_scale

        # 激活函式梯度（sigmoid）
        activation_input = np.random.randn()
        activation_grad = sigmoid_grad(activation_input)

        # 鏈式法則
        gradient = gradient * weight_grad * activation_grad

        # 可選：梯度裁剪
        if use_clipping:
            gradient = np.clip(gradient, -clip_value, clip_value)

        gradients.append(gradient)

    return np.array(gradients)

# 不同權重初始化的效果
n_layers = 50

gradients_small = simulate_gradient_flow(n_layers, weight_scale=0.5)
gradients_normal = simulate_gradient_flow(n_layers, weight_scale=1.0)
gradients_large = simulate_gradient_flow(n_layers, weight_scale=2.0)
gradients_clipped = simulate_gradient_flow(n_layers, weight_scale=2.0,
                                          use_clipping=True, clip_value=1.0)

# 視覺化
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(gradients_small, 'b-', label='Small Init (0.5)', linewidth=2)
plt.plot(gradients_normal, 'g-', label='Normal Init (1.0)', linewidth=2)
plt.plot(gradients_large, 'r-', label='Large Init (2.0)', linewidth=2)
plt.plot(gradients_clipped, 'm--', label='Large Init + Clipping', linewidth=2)
plt.xlabel('Layer Depth')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Flow in Deep Networks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('symlog')  # 對稱對數刻度

plt.subplot(2, 1, 2)
plt.semilogy(np.abs(gradients_small), 'b-', label='Small Init (0.5)', linewidth=2)
plt.semilogy(np.abs(gradients_normal), 'g-', label='Normal Init (1.0)', linewidth=2)
plt.semilogy(np.abs(gradients_large), 'r-', label='Large Init (2.0)', linewidth=2)
plt.semilogy(np.abs(gradients_clipped), 'm--', label='Large Init + Clipping', linewidth=2)
plt.xlabel('Layer Depth')
plt.ylabel('|Gradient| (log scale)')
plt.title('Gradient Magnitude (Logarithmic Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_flow_numerical_stability.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"小權重初始化最終梯度: {gradients_small[-1]:.2e}")
print(f"正常權重初始化最終梯度: {gradients_normal[-1]:.2e}")
print(f"大權重初始化最終梯度: {gradients_large[-1]:.2e}")
print(f"大權重+裁剪最終梯度: {gradients_clipped[-1]:.2e}")
```

### 10.8 二階優化方法演示

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def quadratic_function(x):
    """二次函數"""
    A = np.array([[3, 0.5], [0.5, 1]])
    b = np.array([1, 1])
    return 0.5 * x @ A @ x - b @ x

def quadratic_gradient(x):
    """二次函數的梯度"""
    A = np.array([[3, 0.5], [0.5, 1]])
    b = np.array([1, 1])
    return A @ x - b

def quadratic_hessian(x):
    """二次函數的 Hessian"""
    return np.array([[3, 0.5], [0.5, 1]])

def newton_method(f, grad_f, hessian_f, x0, n_iterations=10):
    """牛頓法"""
    x = x0.copy()
    path = [x.copy()]

    for i in range(n_iterations):
        g = grad_f(x)
        H = hessian_f(x)

        try:
            # 解 H * delta_x = -g
            delta_x = -np.linalg.solve(H, g)
            x = x + delta_x
            path.append(x.copy())

            # 檢查收斂
            if np.linalg.norm(delta_x) < 1e-6:
                break
        except np.linalg.LinAlgError:
            print("Hessian 矩陣奇異，牛頓法失敗")
            break

    return np.array(path)

def bfgs_method(f, grad_f, x0, n_iterations=20):
    """BFGS 擬牛頓法"""
    x = x0.copy()
    path = [x.copy()]
    n = len(x)

    # 初始化 Hessian 逆矩陣估計為單位矩陣
    H_inv = np.eye(n)

    for i in range(n_iterations):
        g = grad_f(x)

        # 計算搜索方向
        p = -H_inv @ g

        # 線搜索（簡化版：固定步長）
        alpha = 0.1
        x_new = x + alpha * p

        # BFGS 更新
        s = x_new - x
        g_new = grad_f(x_new)
        y = g_new - g

        # 更新 H_inv
        rho = 1.0 / (y @ s + 1e-10)
        A1 = np.eye(n) - rho * np.outer(s, y)
        A2 = np.eye(n) - rho * np.outer(y, s)
        H_inv = A1 @ H_inv @ A2 + rho * np.outer(s, s)

        x = x_new
        path.append(x.copy())

        if np.linalg.norm(g_new) < 1e-6:
            break

    return np.array(path)

# 比較一階和二階方法
x0 = np.array([3.0, 3.0])

# 梯度下降
path_gd = []
x = x0.copy()
lr = 0.1
for _ in range(50):
    x = x - lr * quadratic_gradient(x)
    path_gd.append(x.copy())
path_gd = np.array(path_gd)

# 牛頓法
path_newton = newton_method(quadratic_function, quadratic_gradient,
                            quadratic_hessian, x0, n_iterations=10)

# BFGS
path_bfgs = bfgs_method(quadratic_function, quadratic_gradient,
                        x0, n_iterations=20)

# 使用 scipy 的 BFGS（參考）
result = minimize(quadratic_function, x0, method='BFGS', jac=quadratic_gradient)
print(f"Scipy BFGS 結果: {result.x}, 函數值: {result.fun:.6f}")

# 視覺化
x = np.linspace(-1, 4, 100)
y = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[quadratic_function(np.array([x, y])) for x in x] for y in y])

plt.figure(figsize=(14, 5))

# 子圖 1: 所有方法比較
plt.subplot(1, 3, 1)
plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
plt.plot(path_gd[:, 0], path_gd[:, 1], 'bo-', label='Gradient Descent',
         markersize=3, linewidth=2)
plt.plot(path_newton[:, 0], path_newton[:, 1], 'ro-', label='Newton',
         markersize=5, linewidth=2)
plt.plot(path_bfgs[:, 0], path_bfgs[:, 1], 'go-', label='BFGS',
         markersize=4, linewidth=2)
plt.plot(x0[0], x0[1], 'k*', markersize=15, label='Start')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimization Path Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# 子圖 2: 收斂速度
plt.subplot(1, 3, 2)
gd_values = [quadratic_function(x) for x in path_gd]
newton_values = [quadratic_function(x) for x in path_newton]
bfgs_values = [quadratic_function(x) for x in path_bfgs]

plt.semilogy(range(len(gd_values)), gd_values, 'b-', label='GD', linewidth=2)
plt.semilogy(range(len(newton_values)), newton_values, 'r-', label='Newton', linewidth=2)
plt.semilogy(range(len(bfgs_values)), bfgs_values, 'g-', label='BFGS', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Function Value (log scale)')
plt.title('Convergence Speed')
plt.legend()
plt.grid(True, alpha=0.3)

# 子圖 3: 迭代次數比較
plt.subplot(1, 3, 3)
methods = ['GD (50 iter)', f'Newton ({len(path_newton)} iter)',
           f'BFGS ({len(path_bfgs)} iter)']
final_values = [gd_values[-1], newton_values[-1], bfgs_values[-1]]
colors = ['blue', 'red', 'green']

bars = plt.bar(methods, final_values, color=colors, alpha=0.7)
plt.ylabel('Final Function Value')
plt.title('Final Optimization Result')
plt.xticks(rotation=15, ha='right')
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, final_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('second_order_methods.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n優化方法比較:")
print(f"梯度下降 - 迭代: {len(path_gd)}, 最終值: {gd_values[-1]:.6f}")
print(f"牛頓法 - 迭代: {len(path_newton)}, 最終值: {newton_values[-1]:.6f}")
print(f"BFGS - 迭代: {len(path_bfgs)}, 最終值: {bfgs_values[-1]:.6f}")
```

## 11. 實務建議與最佳實踐

### 11.1 選擇優化器的指南

| 場景 | 推薦優化器 | 原因 |
|------|-----------|------|
| 通用深度學習 | Adam | 自適應學習率，收斂快，對超參數不敏感 |
| 計算機視覺（CNN） | SGD + Momentum | 最終性能通常更好，泛化能力強 |
| 自然語言處理（Transformer） | AdamW | Adam 的改進版，更好的權重衰減 |
| 強化學習 | Adam 或 RMSprop | 處理非平穩目標函式 |
| 大批次訓練 | LARS 或 LAMB | 專為大批次設計 |
| 資源受限 | SGD | 記憶體佔用最小 |

### 11.2 超參數調整技巧

**學習率調整策略：**
1. **學習率範圍測試 (LR Range Test)**：從很小的學習率開始，逐步增大，記錄損失變化
2. **循環學習率 (Cyclical LR)**：在最小和最大學習率之間循環
3. **One-Cycle 策略**：先增大再減小學習率，配合動量反向變化

**其他技巧：**
- 使用學習率預熱 (Warmup) 避免初期不穩定
- 批次大小影響：大批次需要更大的學習率
- 梯度累積：模擬更大的批次大小

### 11.3 診斷優化問題

**損失不下降：**
- 檢查學習率（可能太小或太大）
- 檢查梯度（是否為零或 NaN）
- 檢查資料預處理和標準化
- 檢查模型初始化

**訓練不穩定：**
- 使用梯度裁剪
- 降低學習率
- 使用批次正規化或層正規化
- 檢查是否有數值溢出

**過擬合：**
- 增加正則化（L2, Dropout）
- 使用 Early Stopping
- 增加訓練資料或資料增強
- 簡化模型

### 11.4 混合精度訓練

```python
# PyTorch 混合精度訓練範例
import torch
from torch.cuda.amp import autocast, GradScaler

# 建立梯度縮放器
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # 前向傳播使用自動混合精度
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)

        # 縮放損失並反向傳播
        scaler.scale(loss).backward()

        # 更新權重（自動處理梯度縮放）
        scaler.step(optimizer)
        scaler.update()
```

## 12. 延伸閱讀與參考資源

### 書籍
- **《Deep Learning》** (Goodfellow, Bengio, Courville) - 第 8 章優化與第 5 章數值計算
- **《Numerical Optimization》** (Nocedal & Wright) - 傳統優化方法的經典教材
- **《Optimization Methods for Large-Scale Machine Learning》** (Bottou et al.) - 大規模機器學習優化綜述

### 論文
- **Adam**: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
- **LARS**: You et al., "Large Batch Training of Convolutional Networks" (2017)
- **One-Cycle**: Smith, "Super-Convergence: Very Fast Training of Neural Networks" (2018)

### 線上資源
- [Sebastian Ruder - An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [Distill - Why Momentum Really Works](https://distill.pub/2017/momentum/)
- [PyTorch Optimization Algorithms](https://pytorch.org/docs/stable/optim.html)
- [TensorFlow Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

### 工具和框架
- **Optuna** - 自動超參數優化
- **Ray Tune** - 可擴展的超參數調優
- **Weights & Biases** - 實驗追蹤和可視化

---

**總結：**優化是深度學習成功的關鍵。理解不同優化器的特性、掌握數值穩定性技巧、善用學習率調度策略，能夠大幅提升模型訓練的效率和效果。實踐中應根據具體任務選擇合適的優化方法，並通過實驗不斷調整超參數。

