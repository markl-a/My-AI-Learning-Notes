# Calculus.md

## 目錄
1. 前言
2. 基本概念回顧
    - 實數函數與連續性
    - 導數 (Derivative)
    - 偏導數 (Partial Derivative)
    - 梯度 (Gradient)
3. 多變數微分與向量微分算子
    - 偏微分的定義與計算
    - 梯度、Hessian 與高階偏導數
4. 鏈式法則 (Chain Rule)
    - 單變數鏈式法則
    - 多變數鏈式法則
    - 在神經網路中計算梯度與反向傳播的核心原理
5. 向量與矩陣微分
    - 對向量/矩陣函數求導的基本規則
    - 常見的向量與矩陣微分公式速查
    - Matrix Cookbook 作為參考資源
6. 深度學習中的應用
    - 損失函數對權重、偏置的梯度求解
    - 反向傳播 (Backpropagation) 演算法的數學推導
    - 自動微分 (Automatic Differentiation) 的概念
7. 最優化與學習率選擇
    - 梯度下降 (Gradient Descent) 與 Learning Rate 的角色
    - 基於梯度的優化方法：動量 (Momentum)、Adam、RMSProp
8. 數值穩定性與技巧
    - 避免浮點下溢/上溢的計算方法（如使用 Log-Sum-Exp）
    - 函數在邊界處的微分特性與數值行為
9. 延伸閱讀與參考資源

---

## 1. 前言

微積分是機器學習與深度學習的基石之一。在神經網路訓練中，我們需要透過優化演算法對網路參數 (weights, biases) 進行調整，而這些調整的方向與幅度端賴對「目標函數 (如損失函數)」的梯度資訊。

本章將從基礎導數、偏導數、梯度開始講起，介紹計算多變數函數導數的基本概念，並深入探討鏈式法則及其在神經網路反向傳播中的核心地位。我們也將討論自動微分工具的概念，以協助在實作上快速且精確地取得梯度。

## 2. 基本概念回顧

- **導數 (Derivative)**：對單變數函數 f(x)，f'(x) 表示 x 點處的瞬時變化率。  
  幾何詮釋為曲線在該點的切線斜率。

- **偏導數 (Partial Derivative)**：對多變數函數 f(x_1, x_2, ..., x_n)，固定其他變數不變，僅對某一變數求導。如  
  ∂f/∂x_1 表示當僅 x_1 微小變動時，f 的變化率。

- **梯度 (Gradient)**：將 f 對各變數的偏導數組合成向量，即 ∇f(x) = (∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n)。  
  梯度指出 f 增長最快的方向，而 −∇f 則為最快下降方向，在梯度下降中非常重要。

## 3. 多變數微分與向量微分算子

對於多變數函數，我們考慮：

- **Hessian 矩陣**：包含所有二階偏導數的矩陣，用於分析函數曲率 (Curvature)。

- 對高維度輸入，微分提供有用的線索來了解函數在局部的性質。機器學習中，理解 Hessian 有助於分析代價函數的優化性質和收斂行為。

## 4. 鏈式法則 (Chain Rule)

- **單變數鏈式法則**：若 y = f(u) 且 u = g(x)，則 dy/dx = f'(u)*g'(x)。
  
- **多變數鏈式法則**：若 y = f(u_1, u_2, ..., u_m)，而每個 u_i 又是其他變數的函數 u_i(x_1, ..., x_n)，則  
  ∂y/∂x_j = Σ_i (∂f/∂u_i)(∂u_i/∂x_j)。

- 在神經網路中，輸入經過多層線性/非線性變換，鏈式法則可將輸入到輸出間的偏導數拆解成每層的局部導數相乘，進而推導出反向傳播演算法。

## 5. 向量與矩陣微分

- **向量函數對向量求導**：常見於線性代數與機器學習的損失函數，如 L2 損失 L = ∥Ax - b∥²。求解 ∂L/∂x 時需用到向量/矩陣微分法則。

- 常用結果範例：  
  - 若 f(x) = a^T x，則 ∇_x f(x) = a。  
  - 若 f(x) = x^T A x (A 對稱)，則 ∇_x f(x) = (A + A^T)x = 2Ax。  
  - 若 f(X) = Tr(A^T X)，則 ∇_X f(X) = A。

- **Matrix Cookbook** 提供大量向量/矩陣微分的公式，是實務中常參考的資源。

## 6. 深度學習中的應用

- **損失函數對參數的梯度計算**：訓練神經網路時，我們必須計算權重 W 與偏置 b 對損失 L 的梯度，然後更新 W、b。

- **反向傳播 (Backpropagation)**：由輸出層開始，往回計算梯度的一種高效算法。透過鏈式法則，可將整個網路梯度計算分解為多個「局部偏導數」的連乘，極大化效率。

- **自動微分 (Automatic Differentiation)**：數值工具（例如 TensorFlow、PyTorch）可自動生成梯度，而無需手動推導公式。自動微分本質上是利用鏈式法則的程式實作。

## 7. 最優化與學習率選擇

- 利用梯度資訊，我們可以進行梯度下降 (Gradient Descent) 來最小化損失函數。

- **學習率 (Learning Rate)**：決定每次更新步伐大小。若步伐過大，可能不穩定；太小，則收斂太慢。

- 不同優化器（如 Momentum、Adam、RMSProp）運用梯度的歷史資訊或對梯度做平滑處理，提升訓練速度與穩定性。

## 8. 數值穩定性與技巧

- 在實務上計算梯度時，可能面臨指數函數的上溢/下溢等問題。例如計算 softmax 時，常須對輸入做 shift 處理以防數值不穩定。

- 對邏輯函數 (sigmoid) 或對數概率 (log-probability) 計算梯度時，若直接使用原生函數可能會發生數值下溢。透過數值穩定的技巧 (如使用 log-sum-exp 技巧) 可避免此類問題。

## 9. Python 實作範例

### 9.1 數值微分基礎

```python
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    """使用有限差分法計算數值導數"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 定義函數
def f(x):
    return x**2 + 3*x + 2

# 解析導數
def f_prime(x):
    return 2*x + 3

# 計算並比較
x_values = np.linspace(-5, 5, 100)
numerical_derivatives = [numerical_derivative(f, x) for x in x_values]
analytical_derivatives = [f_prime(x) for x in x_values]

plt.figure(figsize=(12, 5))

# 左圖：原函數
plt.subplot(1, 2, 1)
plt.plot(x_values, [f(x) for x in x_values], 'b-', linewidth=2, label='f(x) = x² + 3x + 2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('原函數')
plt.grid(True, alpha=0.3)
plt.legend()

# 右圖：導數比較
plt.subplot(1, 2, 2)
plt.plot(x_values, analytical_derivatives, 'r-', linewidth=2, label="解析導數 f'(x) = 2x + 3")
plt.plot(x_values, numerical_derivatives, 'b--', linewidth=1, label="數值導數", alpha=0.7)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('導數比較')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('numerical_derivative.png', dpi=100, bbox_inches='tight')
plt.close()

print("在 x=2 處：")
print(f"解析導數: {f_prime(2)}")
print(f"數值導數: {numerical_derivative(f, 2)}")
```

### 9.2 多變數函數的偏導數與梯度

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """二元函數: f(x,y) = x² + y²"""
    return x**2 + y**2

def gradient_f(x, y):
    """解析梯度"""
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

def numerical_gradient(f, x, y, h=1e-5):
    """數值計算梯度"""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# 創建網格
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3D 可視化
fig = plt.figure(figsize=(15, 5))

# 子圖1：3D 曲面
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('函數曲面 f(x,y) = x² + y²')

# 子圖2：等高線圖
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('等高線圖')
ax2.grid(True, alpha=0.3)

# 子圖3：梯度場
ax3 = fig.add_subplot(133)
ax3.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
# 繪製梯度向量
step = 5
for i in range(0, len(x), step):
    for j in range(0, len(y), step):
        grad = gradient_f(x[i], y[j])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm * 0.3  # 歸一化以便顯示
            ax3.arrow(x[i], y[j], -grad[0], -grad[1],
                     head_width=0.1, head_length=0.1,
                     fc='red', ec='red', alpha=0.7)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('梯度場 (紅色箭頭指向下降方向)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_visualization.png', dpi=100, bbox_inches='tight')
plt.close()

# 在特定點計算梯度
point = (1.5, 2.0)
analytical_grad = gradient_f(*point)
numerical_grad = numerical_gradient(f, *point)

print(f"在點 {point} 處：")
print(f"解析梯度: {analytical_grad}")
print(f"數值梯度: {numerical_grad}")
```

### 9.3 梯度下降優化

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """目標函數"""
    return x**4 - 3*x**3 + 2

def df(x):
    """目標函數的導數"""
    return 4*x**3 - 9*x**2

def gradient_descent(start_x, learning_rate, num_iterations):
    """梯度下降算法"""
    x = start_x
    history = [x]

    for i in range(num_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)

    return np.array(history)

# 執行梯度下降
start_point = 3.0
learning_rates = [0.01, 0.001, 0.0001]

plt.figure(figsize=(15, 5))

for idx, lr in enumerate(learning_rates, 1):
    history = gradient_descent(start_point, lr, 100)

    # 繪製優化過程
    plt.subplot(1, 3, idx)
    x_range = np.linspace(-1, 4, 200)
    plt.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x)')
    plt.plot(history, f(history), 'ro-', markersize=3, alpha=0.6,
             label=f'梯度下降軌跡 (lr={lr})')
    plt.plot(history[0], f(history[0]), 'go', markersize=10, label='起點')
    plt.plot(history[-1], f(history[-1]), 'r*', markersize=15, label='終點')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'學習率 = {lr}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    print(f"學習率 {lr}: 從 {start_point:.4f} 到 {history[-1]:.4f}")

plt.tight_layout()
plt.savefig('gradient_descent_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 9.4 反向傳播簡單實現

```python
import numpy as np

class SimpleNeuralNetwork:
    """簡單的兩層神經網路，展示反向傳播"""

    def __init__(self, input_size, hidden_size, output_size):
        # 初始化權重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        """Sigmoid 激活函數"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Sigmoid 導數"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        """前向傳播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        """反向傳播"""
        m = X.shape[0]

        # 輸出層梯度
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 隱藏層梯度（鏈式法則）
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新參數
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def compute_loss(self, y_pred, y_true):
        """計算損失（交叉熵）"""
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8) +
                       (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m
        return loss

# 生成簡單數據集（XOR 問題）
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 訓練網路
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = []

for epoch in range(10000):
    # 前向傳播
    y_pred = nn.forward(X)

    # 計算損失
    loss = nn.compute_loss(y_pred, y)
    losses.append(loss)

    # 反向傳播
    nn.backward(X, y, learning_rate=0.5)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 測試
print("\n測試結果：")
predictions = nn.forward(X)
for i in range(len(X)):
    print(f"輸入: {X[i]}, 預測: {predictions[i][0]:.4f}, 真實: {y[i][0]}")

# 繪製損失曲線
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel('訓練迭代次數')
plt.ylabel('損失')
plt.title('訓練過程中的損失變化')
plt.grid(True, alpha=0.3)
plt.savefig('backpropagation_loss.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 9.5 使用 PyTorch 自動微分

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定義簡單函數並計算梯度
x = torch.tensor([2.0], requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 3

print("函數值 y:", y.item())

# 自動計算梯度
y.backward()
print("在 x=2 處的導數:", x.grad.item())
print("解析導數 (3x² + 4x - 5):", 3*2**2 + 4*2 - 5)

# 多變數函數的梯度
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0]**2 + 3*x[0]*x[1] + x[1]**2

y.backward()
print("\n多變數函數梯度:", x.grad)

# 使用 PyTorch 訓練簡單模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 創建模型和訓練數據
model = SimpleModel()
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# 訓練
losses = []
for epoch in range(5000):
    # 前向傳播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向傳播和優化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 自動計算梯度
    optimizer.step()       # 更新參數

    losses.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel('訓練迭代次數')
plt.ylabel('損失')
plt.title('PyTorch 自動微分訓練過程')
plt.grid(True, alpha=0.3)
plt.savefig('pytorch_autograd.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 9.6 優化器比較 (SGD vs Momentum vs Adam)

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    """Rosenbrock 函數（優化測試函數）"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    """Rosenbrock 函數的梯度"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def sgd(start, grad_fn, lr, n_iter):
    """隨機梯度下降"""
    path = [start]
    pos = np.array(start, dtype=float)

    for _ in range(n_iter):
        grad = grad_fn(*pos)
        pos = pos - lr * grad
        path.append(pos.copy())

    return np.array(path)

def momentum(start, grad_fn, lr, momentum_coef, n_iter):
    """動量法"""
    path = [start]
    pos = np.array(start, dtype=float)
    velocity = np.zeros_like(pos)

    for _ in range(n_iter):
        grad = grad_fn(*pos)
        velocity = momentum_coef * velocity - lr * grad
        pos = pos + velocity
        path.append(pos.copy())

    return np.array(path)

def adam(start, grad_fn, lr, n_iter, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam 優化器"""
    path = [start]
    pos = np.array(start, dtype=float)
    m = np.zeros_like(pos)
    v = np.zeros_like(pos)

    for t in range(1, n_iter + 1):
        grad = grad_fn(*pos)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        pos = pos - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(pos.copy())

    return np.array(path)

# 設定起始點和參數
start_point = [-1.0, 2.5]
n_iterations = 100

# 執行不同優化器
path_sgd = sgd(start_point, rosenbrock_grad, lr=0.001, n_iter=n_iterations)
path_momentum = momentum(start_point, rosenbrock_grad, lr=0.001,
                         momentum_coef=0.9, n_iter=n_iterations)
path_adam = adam(start_point, rosenbrock_grad, lr=0.01, n_iter=n_iterations)

# 繪製優化路徑
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(15, 5))

optimizers = [
    (path_sgd, 'SGD', 'blue'),
    (path_momentum, 'Momentum', 'green'),
    (path_adam, 'Adam', 'red')
]

for idx, (path, name, color) in enumerate(optimizers, 1):
    plt.subplot(1, 3, idx)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.4)
    plt.plot(path[:, 0], path[:, 1], color=color, marker='o',
             markersize=3, linewidth=2, alpha=0.7, label=name)
    plt.plot(1, 1, 'r*', markersize=20, label='最優點 (1,1)')
    plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='起點')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{name} 優化路徑')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=100, bbox_inches='tight')
plt.close()

print("優化結果比較：")
for path, name, _ in optimizers:
    final_pos = path[-1]
    final_value = rosenbrock(*final_pos)
    print(f"{name}: 最終位置 = {final_pos}, 函數值 = {final_value:.6f}")
```

## 10. 延伸閱讀與參考資源

- **建議閱讀**：
  - "Deep Learning" (Goodfellow, Bengio, Courville) 中的數學基礎章節
  - 《Matrix Cookbook》：對向量與矩陣的微分規則作詳細整理
  - 有關自動微分的文獻與框架文件（TensorFlow、PyTorch 官方文件）

### 推薦資源

- [PyTorch Autograd 教學](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [3Blue1Brown - 微積分的本質](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [MIT 18.01 Single Variable Calculus](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

