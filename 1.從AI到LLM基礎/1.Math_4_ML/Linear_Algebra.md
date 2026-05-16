# Linear_Algebra.md

## 目錄
1. 前言
2. 基本定義與符號
    - 標量 (Scalar)
    - 向量 (Vector)
    - 矩陣 (Matrix)
    - 張量 (Tensor)
    - 符號與轉置運算
3. 基本矩陣運算
    - 矩陣相加、標量與矩陣之間的加乘
    - 矩陣與向量/矩陣的乘法
    - 單位矩陣 (Identity Matrix) 與逆矩陣 (Inverse)
    - 矩陣分解舉例：LU 分解、對角矩陣特性
4. 線性獨立與秩 (Rank)
    - 線性獨立 (Linear Independence)
    - 生成子空間 (Span) 與列空間 (Column Space)
    - 秩 (Rank) 的意義
    - 解線性方程組：可解性與有無限多解的判斷
5. 范數 (Norm) 與距離
    - 向量的 Lp 范數
    - 矩陣的 Frobenius 范數
    - 最大范數 (L∞ Norm)
    - 范數在機器學習中的應用（如正則化概念）
6. 特徵分解 (Eigen-Decomposition)
    - 特徵值 (Eigenvalue) 與特徵向量 (Eigenvector) 定義
    - 實對稱矩陣的特徵分解：A = QΛQ^T
    - 特徵值的性質：正定 (Positive Definite)、半正定、奇異、負定等分類
    - 利用特徵分解理解二次形式 (Quadratic Form)
    - PCA (主成分分析) 與特徵分解的關係簡介
7. 奇異值分解 (Singular Value Decomposition, SVD)
    - SVD 定義：A = U D V^T
    - 左奇異向量 (Left Singular Vector)、右奇異向量 (Right Singular Vector)、奇異值 (Singular Value)
    - SVD 的性質與應用（降維、壓縮、PCA）
    - SVD 與特徵分解的差異
8. 線性代數在深度學習與機器學習中的應用
    - 權重矩陣、特徵空間與模型參數化
    - 方程組求解在參數估計中的應用 (如最小二乘問題)
    - 矩陣分解、低秩近似在模型壓縮與加速上的運用
9. 數值穩定性與計算注意事項
    - 上溢 (Overflow) 與下溢 (Underflow)
    - 病態條件 (Ill-conditioning) 與數值方法的穩健性
    - 使用框架（如 NumPy、PyTorch、TensorFlow）執行矩陣運算的小技巧
10. 延伸閱讀與實務參考
    - 更深入的矩陣分解方式（如 Cholesky、QR、Hessenberg）
    - 高等線性代數主題

---

## 1. 前言

線性代數是現代數學與工程科學的核心工具之一，在深度學習和機器學習中更是無所不在。從最基礎的線性模型 (Linear Model) 到深度神經網路的權重矩陣 (Weight Matrix)，再到高階技巧如主成分分析 (PCA)、奇異值分解 (SVD) 與各種優化過程，都需要對線性代數有紮實的理解。

本章節將回顧線性代數的重要概念，涵蓋了向量、矩陣運算、特徵值特徵向量、奇異值分解 (SVD) 等主題，並在結尾連結到機器學習、深度學習實務中最常見的應用場景。

## 2. 基本定義與符號

- **標量 (Scalar)**：一個單獨的數字，如實數 R 中的元素。標量常以斜體小寫字母表示，例如 s。
- **向量 (Vector)**：一列有序數值的集合，如 x ∈ R^n。可將向量視為 n 維空間中的一個點或坐標。通常使用粗體小寫字母 (e.g. **x**)。
- **矩陣 (Matrix)**：一個二維陣列，含有 m × n 個元素。以粗體大寫字母 (e.g. **A**) 表示。A ∈ R^(m×n)。
- **張量 (Tensor)**：更高維度的泛化陣列（超過 2 維），如三維以上結構在深度學習的輸入圖像資料中常出現。
- **轉置 (Transpose)**：矩陣 A 的轉置 A^T 將行、列互換。

## 3. 基本矩陣運算

- **加法與標量乘法**：兩矩陣同形狀下可逐元素相加；標量乘法對每元素同時放大或縮小。
- **矩陣乘法**：若 A 為 m×n，B 為 n×p，則 C = A B 為 m×p 矩陣。C 的元素 C_(i,j) = Σ_k A_(i,k)*B_(k,j)。
- **單位矩陣 (Identity Matrix)**：記為 I，滿足 I x = x，常作為「不改變向量」的單位元素。I 是方陣且對角線為 1，其餘元素為 0。
- **逆矩陣 (Inverse Matrix)**：A 為可逆方陣，A^(-1) 滿足 A A^(-1) = I。若 A 不可逆，則可能需用偽逆 (Pseudo-Inverse)。
  
## 4. 線性獨立與秩 (Rank)

- **線性獨立**：一組向量中無法用其他向量的線性組合來表示該組內的某一成員時，即為線性獨立。
- **秩 (Rank)**：矩陣中最大線性獨立行或列向量的數目，rank(A) 表示 A 的維度「資訊量」或「非退化性」。
- 透過秩與線性獨立性，我們可判定 Ax=b 是否有唯一解、無解或無限多解。

## 5. 范數 (Norm) 與距離

- **向量范數**：如 L2 范數 ∥x∥2 = sqrt(Σ x_i^2)。L1、L∞、以及 Frobeinus 范數 (應用於矩陣) 在深度學習中也很常見。
- 范數用於衡量「大小」或「長度」，在優化、正則化 (Regularization) 與誤差衡量時相當重要。

## 6. 特徵分解 (Eigen-Decomposition)

- **特徵值與特徵向量**：給定方陣 A，若存在非零向量 v 與標量 λ，使得 A v = λ v，則 λ 為 A 的特徵值，v 為對應特徵向量。
- **特徵分解**：若 A 有 n 個線性獨立特徵向量，則 A = V Λ V^(-1)，其中 V 是由特徵向量組成的矩陣，Λ 是特徵值對角矩陣。
- 對實對稱矩陣，可得 A = Q Λ Q^T，其中 Q 為正交矩陣。此在機器學習中很常用，例如 PCA 中以特徵分解找出資料主軸方向。

## 7. 奇異值分解 (Singular Value Decomposition, SVD)

- **定義**：對任意 m×n 的實矩陣 A，可分解成 A = U D V^T，其中 U、V 為正交矩陣，D 為對角矩陣（奇異值在對角線上）。
- **奇異值 (Singular Value)**：D 的對角元素皆為非負實數，代表 A 沿各奇異向量方向的伸展尺度。
- SVD 是 PCA 的理論基礎，也是許多降維 (Dimensionality Reduction) 技術的關鍵，同時可用於矩陣近似、壓縮與正則化。

## 8. 線性代數在機器學習與深度學習中的應用

- 模型參數矩陣 W 在前饋運算 y = W x 中扮演關鍵角色。
- 解線性方程組 Ax = b 是最小二乘 (Least Squares) 問題的基礎，可推廣至迴歸模型參數估計。
- 特徵值、SVD 分解在資料降維、特徵提取 (Feature Extraction)、正則化與神經網路的權重初始化分析中扮演要角。

## 9. 數值穩定性與計算注意事項

- **上溢 (Overflow) 與下溢 (Underflow)**：計算 exp(x) 等函數時，若 x 太大或太小，數值易發生不穩定。
- **病態條件 (Ill-conditioned)**：矩陣條件數大時，微小的輸入誤差將被放大。需採用數值方法如 SVD、QR 分解穩定求解。
- 實務上多使用數值線代函式庫 (NumPy、TensorFlow、PyTorch) 提供高效穩定的矩陣計算。

## 10. Python 實作範例

### 10.1 基本矩陣運算

```python
import numpy as np
import matplotlib.pyplot as plt

# 建立向量和矩陣
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

matrix_A = np.array([[1, 2], [3, 4], [5, 6]])
matrix_B = np.array([[7, 8], [9, 10]])

print("向量 a:", vector_a)
print("向量 b:", vector_b)
print("矩陣 A:\n", matrix_A)
print("矩陣 B:\n", matrix_B)

# 向量操作
dot_product = np.dot(vector_a, vector_b)  # 點積
print("\n向量點積:", dot_product)

# 矩陣乘法
matrix_C = np.dot(matrix_A, matrix_B)
print("\n矩陣乘法 A @ B:\n", matrix_C)

# 轉置
print("\n矩陣 A 的轉置:\n", matrix_A.T)

# 單位矩陣
identity = np.eye(3)
print("\n3x3 單位矩陣:\n", identity)
```

### 10.2 計算逆矩陣和求解線性方程組

```python
import numpy as np

# 建立可逆矩陣
A = np.array([[4, 7], [2, 6]])
print("矩陣 A:\n", A)

# 計算逆矩陣
A_inv = np.linalg.inv(A)
print("\nA 的逆矩陣:\n", A_inv)

# 驗證 A @ A_inv = I
print("\nA @ A_inv (應該是單位矩陣):\n", np.dot(A, A_inv))

# 求解線性方程組 Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("\n解 x:", x)
print("驗證 Ax:", np.dot(A, x))
```

### 10.3 范數計算

```python
import numpy as np

vector = np.array([3, 4])

# L1 范數 (曼哈頓距離)
l1_norm = np.linalg.norm(vector, ord=1)
print("L1 范數:", l1_norm)

# L2 范數 (歐幾里得距離)
l2_norm = np.linalg.norm(vector, ord=2)
print("L2 范數:", l2_norm)

# L∞ 范數 (最大值)
linf_norm = np.linalg.norm(vector, ord=np.inf)
print("L∞ 范數:", linf_norm)

# 矩陣的 Frobenius 范數
matrix = np.array([[1, 2], [3, 4]])
frobenius_norm = np.linalg.norm(matrix, 'fro')
print("\nFrobenius 范數:", frobenius_norm)
```

### 10.4 特徵值與特徵向量

```python
import numpy as np
import matplotlib.pyplot as plt

# 建立對稱矩陣
A = np.array([[4, 2], [2, 3]])

# 計算特徵值和特徵向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("矩陣 A:\n", A)
print("\n特徵值:", eigenvalues)
print("\n特徵向量:\n", eigenvectors)

# 驗證 Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i]
    Av = np.dot(A, v)
    lambda_times_v = lambda_v * v
    print(f"\n特徵值 {i+1}: {lambda_v}")
    print(f"Av = {Av}")
    print(f"λv = {lambda_times_v}")
    print(f"是否相等: {np.allclose(Av, lambda_times_v)}")

# 視覺化特徵向量
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0],
           angles='xy', scale_units='xy', scale=1, color='r',
           label=f'特徵向量 1 (λ={eigenvalues[0]:.2f})')
plt.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1],
           angles='xy', scale_units='xy', scale=1, color='b',
           label=f'特徵向量 2 (λ={eigenvalues[1]:.2f})')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.legend()
plt.title('特徵向量視覺化')
plt.savefig('eigenvectors.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 10.5 奇異值分解 (SVD)

```python
import numpy as np

# 建立矩陣
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print("原始矩陣 A (4x3):\n", A)

# 進行 SVD 分解: A = U @ S @ V^T
U, s, VT = np.linalg.svd(A, full_matrices=False)

print("\nU 的形狀:", U.shape)
print("奇異值:", s)
print("V^T 的形狀:", VT.shape)

# 重建矩陣
S = np.diag(s)
A_reconstructed = U @ S @ VT

print("\n重建的矩陣:\n", A_reconstructed)
print("\n重建誤差:", np.linalg.norm(A - A_reconstructed))

# 低秩近似 (保留前2個奇異值)
k = 2
A_approx = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]
print(f"\n保留前 {k} 個奇異值的近似矩陣:\n", A_approx)
print(f"近似誤差:", np.linalg.norm(A - A_approx))
```

### 10.6 主成分分析 (PCA) 實作

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 載入資料
iris = load_iris()
X = iris.data
y = iris.target

# 資料標準化
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 計算協方差矩陣
cov_matrix = np.cov(X_centered.T)

# 特徵分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 按特徵值排序
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("特徵值:", eigenvalues)
print("解釋方差比:", eigenvalues / np.sum(eigenvalues))

# 投影到前2個主成分
n_components = 2
W = eigenvectors[:, :n_components]
X_pca = X_centered @ W

# 視覺化
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i, color in enumerate(colors):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, label=iris.target_names[i], alpha=0.7)
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('PCA 降維結果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pca_visualization.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n降維後的資料形狀:", X_pca.shape)
```

### 10.7 在神經網路中的應用：權重矩陣初始化

```python
import numpy as np
import matplotlib.pyplot as plt

def xavier_uniform(n_in, n_out):
    """Xavier/Glorot 均勻初始化"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def xavier_normal(n_in, n_out):
    """Xavier/Glorot 正態初始化"""
    std = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, std, (n_in, n_out))

def he_normal(n_in, n_out):
    """He 初始化 (適用於 ReLU)"""
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, (n_in, n_out))

# 建立不同初始化方法的權重
n_in, n_out = 128, 64

W_xavier_uniform = xavier_uniform(n_in, n_out)
W_xavier_normal = xavier_normal(n_in, n_out)
W_he = he_normal(n_in, n_out)

# 視覺化權重分佈
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(W_xavier_uniform.flatten(), bins=50, alpha=0.7, color='blue')
axes[0].set_title('Xavier Uniform')
axes[0].set_xlabel('權重值')
axes[0].set_ylabel('頻率')

axes[1].hist(W_xavier_normal.flatten(), bins=50, alpha=0.7, color='green')
axes[1].set_title('Xavier Normal')
axes[1].set_xlabel('權重值')

axes[2].hist(W_he.flatten(), bins=50, alpha=0.7, color='red')
axes[2].set_title('He Normal')
axes[2].set_xlabel('權重值')

plt.tight_layout()
plt.savefig('weight_initialization.png', dpi=100, bbox_inches='tight')
plt.close()

print("Xavier Uniform - 均值:", np.mean(W_xavier_uniform), "標準差:", np.std(W_xavier_uniform))
print("Xavier Normal - 均值:", np.mean(W_xavier_normal), "標準差:", np.std(W_xavier_normal))
print("He Normal - 均值:", np.mean(W_he), "標準差:", np.std(W_he))
```

### 10.8 線性迴歸的矩陣解法

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成資料
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加偏置項 (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X]

# 使用正規方程求解: θ = (X^T X)^(-1) X^T y
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("最佳參數 θ:", theta_best.ravel())

# 預測
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta_best

# 視覺化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='資料點')
plt.plot(X_new, y_predict, 'r-', linewidth=2, label='預測線')
plt.xlabel('X')
plt.ylabel('y')
plt.title('線性迴歸 - 矩陣解法')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# 使用 SVD 求解 (更穩定的方法)
U, s, VT = np.linalg.svd(X_b, full_matrices=False)
theta_svd = VT.T @ np.linalg.inv(np.diag(s)) @ U.T @ y

print("SVD 求解的參數:", theta_svd.ravel())
```

## 11. 實際應用案例

### 11.1 圖像壓縮 (使用 SVD)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 建立或載入圖像 (這裡用隨機資料示例)
# 實際使用時可以用: img = np.array(Image.open('image.jpg').convert('L'))
img = np.random.rand(100, 100) * 255

# 進行 SVD
U, s, VT = np.linalg.svd(img, full_matrices=False)

# 使用不同數量的奇異值重建
ranks = [5, 10, 20, 50]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始圖像')
axes[0, 0].axis('off')

for idx, k in enumerate(ranks, 1):
    # 重建圖像
    img_approx = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]

    # 計算壓縮率
    original_size = img.shape[0] * img.shape[1]
    compressed_size = k * (img.shape[0] + img.shape[1] + 1)
    compression_ratio = compressed_size / original_size * 100

    row = idx // 3
    col = idx % 3
    axes[row, col].imshow(img_approx, cmap='gray')
    axes[row, col].set_title(f'rank={k} ({compression_ratio:.1f}%)')
    axes[row, col].axis('off')

# 奇異值分佈
axes[1, 2].plot(s, 'b-', linewidth=2)
axes[1, 2].set_xlabel('索引')
axes[1, 2].set_ylabel('奇異值')
axes[1, 2].set_title('奇異值分佈')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svd_image_compression.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 10.9 Transformer 中的注意力機制

```python
import numpy as np
import matplotlib.pyplot as plt

class ScaledDotProductAttention:
    """縮放點積注意力 (Scaled Dot-Product Attention)"""

    def __init__(self):
        pass

    def forward(self, Q, K, V, mask=None):
        """
        計算注意力

        Args:
            Q: Query 矩陣 (batch_size, seq_len_q, d_k)
            K: Key 矩陣 (batch_size, seq_len_k, d_k)
            V: Value 矩陣 (batch_size, seq_len_v, d_v)
            mask: 遮罩矩陣 (可選)

        Returns:
            output: 注意力輸出 (batch_size, seq_len_q, d_v)
            attention_weights: 注意力權重 (batch_size, seq_len_q, seq_len_k)
        """
        d_k = Q.shape[-1]

        # 計算注意力分數: QK^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        # 應用遮罩（如果有）
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax 得到注意力權重
        attention_weights = self.softmax(scores)

        # 加權求和
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def softmax(self, x):
        """數值穩定的 softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    """多頭注意力 (Multi-Head Attention)"""

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: 模型維度
            num_heads: 注意力頭數
        """
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 初始化權重矩陣
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """分割最後一個維度到 (num_heads, d_k)"""
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        """
        多頭注意力前向傳播

        Args:
            Q, K, V: 形狀為 (batch_size, seq_len, d_model)
        """
        batch_size = Q.shape[0]

        # 線性投影
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)

        # 分割成多個頭
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 計算注意力
        # 對每個頭分別計算
        outputs = []
        attention_weights_list = []

        for i in range(self.num_heads):
            output, attn_weights = self.attention.forward(
                Q[:, i, :, :],
                K[:, i, :, :],
                V[:, i, :, :],
                mask
            )
            outputs.append(output)
            attention_weights_list.append(attn_weights)

        # 連接所有頭
        concat_output = np.concatenate(outputs, axis=-1)

        # 最終線性投影
        final_output = np.matmul(concat_output, self.W_o)

        return final_output, attention_weights_list

# 演示注意力機制
def demonstrate_attention():
    """演示注意力機制的工作原理"""
    np.random.seed(42)

    # 模擬輸入序列
    batch_size = 1
    seq_len = 5
    d_model = 8

    # 建立簡單的輸入（模擬詞嵌入）
    X = np.random.randn(batch_size, seq_len, d_model)

    # 自注意力（Q=K=V）
    attention = ScaledDotProductAttention()
    output, attention_weights = attention.forward(X, X, X)

    print("=" * 60)
    print("注意力機制演示")
    print("=" * 60)
    print(f"\n輸入形狀: {X.shape}")
    print(f"輸出形狀: {output.shape}")
    print(f"注意力權重形狀: {attention_weights.shape}")

    # 視覺化注意力權重
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0], cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weights Visualization')

    # 添加數值標註
    for i in range(seq_len):
        for j in range(seq_len):
            text = plt.text(j, i, f'{attention_weights[0, i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=10)

    plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 多頭注意力演示
    print("\n" + "=" * 60)
    print("多頭注意力演示")
    print("=" * 60)

    num_heads = 4
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output, mha_weights = mha.forward(X, X, X)

    print(f"\n多頭注意力輸出形狀: {mha_output.shape}")
    print(f"注意力頭數: {num_heads}")
    print(f"每個頭的維度: {d_model // num_heads}")

    # 視覺化多頭注意力
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(mha_weights[i][0], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n注意力權重可視化已保存")

demonstrate_attention()
```

### 10.10 張量操作與 Einstein Summation

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_tensor_operations():
    """演示張量操作"""
    print("=" * 60)
    print("張量操作演示")
    print("=" * 60)

    # 建立張量
    A = np.random.randn(3, 4, 5)  # 3D 張量
    B = np.random.randn(5, 6)     # 2D 張量 (矩陣)

    print(f"\n張量 A 形狀: {A.shape}")
    print(f"張量 B 形狀: {B.shape}")

    # 張量縮約 (Tensor Contraction)
    # 沿最後一個維度與 B 的第一個維度相乘
    C = np.tensordot(A, B, axes=([2], [0]))
    print(f"\n張量縮約 A ⊗ B 形狀: {C.shape}")

    # Einstein Summation 演示
    print("\n" + "=" * 60)
    print("Einstein Summation 演示")
    print("=" * 60)

    # 示例 1: 矩陣乘法
    X = np.random.randn(3, 4)
    Y = np.random.randn(4, 5)

    # 傳統方法
    Z_traditional = np.matmul(X, Y)

    # Einstein summation
    Z_einsum = np.einsum('ij,jk->ik', X, Y)

    print(f"\n矩陣乘法:")
    print(f"  傳統方法結果形狀: {Z_traditional.shape}")
    print(f"  Einsum 方法結果形狀: {Z_einsum.shape}")
    print(f"  結果是否相同: {np.allclose(Z_traditional, Z_einsum)}")

    # 示例 2: 批次矩陣乘法 (Batch Matrix Multiplication)
    batch_size = 10
    A_batch = np.random.randn(batch_size, 3, 4)
    B_batch = np.random.randn(batch_size, 4, 5)

    # 使用 einsum
    C_batch = np.einsum('bij,bjk->bik', A_batch, B_batch)
    print(f"\n批次矩陣乘法結果形狀: {C_batch.shape}")

    # 示例 3: 注意力機制中的操作
    print("\n" + "-" * 60)
    print("注意力機制中的 Einstein Summation")
    print("-" * 60)

    batch_size, seq_len, d_model = 2, 4, 8
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    # 計算注意力分數: Q @ K^T
    scores_traditional = np.matmul(Q, K.transpose(0, 2, 1))
    scores_einsum = np.einsum('bqd,bkd->bqk', Q, K)

    print(f"  注意力分數 (傳統): {scores_traditional.shape}")
    print(f"  注意力分數 (einsum): {scores_einsum.shape}")
    print(f"  結果相同: {np.allclose(scores_traditional, scores_einsum)}")

    # 常用的 Einstein Summation 模式
    print("\n" + "=" * 60)
    print("常用 Einstein Summation 模式")
    print("=" * 60)

    patterns = {
        '矩陣轉置': ("ij->ji", np.random.randn(3, 4)),
        '對角線求和 (跡)': ("ii->", np.random.randn(5, 5)),
        '逐元素相乘後求和': ("i,i->", np.random.randn(10), np.random.randn(10)),
        '外積': ("i,j->ij", np.random.randn(3), np.random.randn(4)),
        '批次點積': ("bi,bi->b", np.random.randn(10, 5), np.random.randn(10, 5)),
    }

    for name, (pattern, *arrays) in patterns.items():
        result = np.einsum(pattern, *arrays)
        print(f"\n{name}:")
        print(f"  模式: {pattern}")
        if len(arrays) == 1:
            print(f"  輸入形狀: {arrays[0].shape}")
        else:
            print(f"  輸入形狀: {[arr.shape for arr in arrays]}")
        print(f"  輸出形狀: {result.shape if isinstance(result, np.ndarray) else 'scalar'}")

demonstrate_tensor_operations()
```

### 10.11 模型壓縮：低秩分解應用

```python
import numpy as np
import matplotlib.pyplot as plt

class LowRankDecomposition:
    """低秩分解用於模型壓縮"""

    def __init__(self, rank):
        """
        Args:
            rank: 目標秩
        """
        self.rank = rank

    def decompose_weight(self, W):
        """
        使用 SVD 將權重矩陣分解為低秩近似

        Args:
            W: 原始權重矩陣 (m, n)

        Returns:
            W_approx: 低秩近似 (m, n)
            U_r: 左奇異向量 (m, rank)
            s_r: 奇異值 (rank,)
            VT_r: 右奇異向量 (rank, n)
        """
        # SVD 分解
        U, s, VT = np.linalg.svd(W, full_matrices=False)

        # 保留前 rank 個成分
        U_r = U[:, :self.rank]
        s_r = s[:self.rank]
        VT_r = VT[:self.rank, :]

        # 重建近似
        W_approx = U_r @ np.diag(s_r) @ VT_r

        return W_approx, U_r, s_r, VT_r

    def compression_ratio(self, original_shape):
        """
        計算壓縮比

        Args:
            original_shape: (m, n)
        """
        m, n = original_shape
        original_params = m * n
        compressed_params = self.rank * (m + n + 1)  # U_r + s_r + VT_r
        ratio = compressed_params / original_params

        return ratio, original_params, compressed_params

def demonstrate_model_compression():
    """演示模型壓縮"""
    print("=" * 60)
    print("神經網路權重低秩分解壓縮")
    print("=" * 60)

    # 模擬一個大的全連接層權重
    np.random.seed(42)
    m, n = 1000, 2000  # 輸入維度 -> 輸出維度
    W_original = np.random.randn(m, n) * 0.01

    print(f"\n原始權重形狀: {W_original.shape}")
    print(f"原始參數量: {W_original.size:,}")

    # 測試不同的秩
    ranks = [10, 50, 100, 200]

    results = {}
    for rank in ranks:
        decomp = LowRankDecomposition(rank)
        W_approx, U_r, s_r, VT_r = decomp.decompose_weight(W_original)

        # 計算重建誤差
        reconstruction_error = np.linalg.norm(W_original - W_approx, 'fro') / \
                              np.linalg.norm(W_original, 'fro')

        # 計算壓縮比
        ratio, orig_params, comp_params = decomp.compression_ratio(W_original.shape)

        results[rank] = {
            'error': reconstruction_error,
            'compression_ratio': ratio,
            'original_params': orig_params,
            'compressed_params': comp_params,
            'singular_values': s_r
        }

        print(f"\nRank {rank}:")
        print(f"  重建誤差: {reconstruction_error:.4%}")
        print(f"  壓縮參數量: {comp_params:,}")
        print(f"  壓縮比: {ratio:.2%}")
        print(f"  節省參數: {(1-ratio):.2%}")

    # 視覺化結果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子圖 1: 重建誤差 vs 秩
    ax1 = axes[0, 0]
    errors = [results[r]['error'] for r in ranks]
    ax1.plot(ranks, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Reconstruction Error vs Rank')
    ax1.grid(True, alpha=0.3)

    # 子圖 2: 壓縮比 vs 秩
    ax2 = axes[0, 1]
    ratios = [results[r]['compression_ratio'] for r in ranks]
    ax2.plot(ranks, ratios, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='原始大小')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Ratio vs Rank')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子圖 3: 參數量比較
    ax3 = axes[1, 0]
    params_original = [results[r]['original_params'] for r in ranks]
    params_compressed = [results[r]['compressed_params'] for r in ranks]

    x = np.arange(len(ranks))
    width = 0.35

    bars1 = ax3.bar(x - width/2, params_original, width, label='原始', alpha=0.7)
    bars2 = ax3.bar(x + width/2, params_compressed, width, label='壓縮後', alpha=0.7)

    ax3.set_xlabel('Rank')
    ax3.set_ylabel('參數量')
    ax3.set_title('參數量比較')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ranks)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 子圖 4: 奇異值分佈
    ax4 = axes[1, 1]

    # 計算完整 SVD 以顯示所有奇異值
    U, s_full, VT = np.linalg.svd(W_original, full_matrices=False)

    ax4.semilogy(s_full, 'b-', linewidth=2, label='所有奇異值')
    for rank in ranks:
        ax4.axvline(x=rank, linestyle='--', alpha=0.7, label=f'Rank {rank}')

    ax4.set_xlabel('Index')
    ax4.set_ylabel('Singular Value (log scale)')
    ax4.set_title('奇異值分佈')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('low_rank_compression.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n可視化結果已保存")

    # 實際應用示例：壓縮線性層
    print("\n" + "=" * 60)
    print("實際應用：壓縮神經網路層")
    print("=" * 60)

    class CompressedLinearLayer:
        """壓縮的線性層"""
        def __init__(self, W, rank):
            decomp = LowRankDecomposition(rank)
            _, self.U_r, self.s_r, self.VT_r = decomp.decompose_weight(W)

        def forward(self, x):
            """
            前向傳播: y = x @ W ≈ x @ (U_r @ diag(s_r) @ VT_r)
                     = ((x @ U_r) @ diag(s_r)) @ VT_r
            """
            # 分步計算以節省記憶體
            temp1 = x @ self.U_r
            temp2 = temp1 * self.s_r[np.newaxis, :]
            output = temp2 @ self.VT_r
            return output

    # 測試壓縮層
    batch_size = 32
    x_input = np.random.randn(batch_size, m)

    # 原始層
    y_original = x_input @ W_original

    # 壓縮層 (rank=100)
    compressed_layer = CompressedLinearLayer(W_original, rank=100)
    y_compressed = compressed_layer.forward(x_input)

    # 計算輸出差異
    output_error = np.linalg.norm(y_original - y_compressed) / np.linalg.norm(y_original)

    print(f"\n批次大小: {batch_size}")
    print(f"輸入維度: {m}")
    print(f"輸出維度: {n}")
    print(f"壓縮秩: 100")
    print(f"輸出誤差: {output_error:.4%}")
    print(f"\n結論: 使用低秩分解可以大幅減少參數量，同時保持合理的精度")

demonstrate_model_compression()
```

## 12. 延伸閱讀與實務參考

- 深入了解更多分解法：LU、QR、Cholesky、Jordan Normal Form 等。
- 高等線代主題：非齊次系統、廣義逆、張量分解 (Tensor Decomposition)。
- 推薦閱讀: 《Linear Algebra and Its Applications》、深度學習教科書中關於線代的章節，以及《Matrix Cookbook》作為公式速查表。

### 推薦資源

**書籍：**
- 《Linear Algebra and Its Applications》 (Gilbert Strang) - 經典線性代數教材
- 《Matrix Computations》 (Golub & Van Loan) - 數值線性代數聖經
- 《The Matrix Cookbook》 - 矩陣運算速查手冊

**線上課程：**
- [NumPy 線性代數文檔](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [3Blue1Brown - 線性代數的本質](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [Fast.ai Numerical Linear Algebra](https://github.com/fastai/numerical-linear-algebra)

**論文與文章：**
- **Attention Is All You Need** (Vaswani et al., 2017) - Transformer 架構
- **LoRA: Low-Rank Adaptation** (Hu et al., 2021) - 低秩適應用於大模型微調
- **Tensor Decompositions and Applications** (Kolda & Bader, 2009) - 張量分解綜述

**實用工具：**
- **NumPy** - Python 科學計算基礎
- **SciPy** - 科學計算工具包
- **PyTorch / TensorFlow** - 深度學習框架（內建張量運算）
- **Einops** - Einstein 記號的 Pythonic 實現

## 13. 練習題與挑戰 (Exercises & Challenges)

### 基礎練習 (Basics)

**練習 1: 矩陣運算**
給定矩陣 A = [[1, 2], [3, 4]] 和 B = [[5, 6], [7, 8]]
1. 計算 A + B
2. 計算 AB
3. 計算 A^T B
4. 驗證 (AB)^T = B^T A^T

**練習 2: 特徵值和特徵向量**
對於矩陣 A = [[4, 2], [2, 3]]
1. 計算特徵值和特徵向量
2. 驗證 Av = λv
3. 進行特徵分解 A = QΛQ^(-1)
4. 判斷 A 是否為正定矩陣

**練習 3: 范數計算**
對於向量 x = [3, -4, 0, 5]
1. 計算 L1 范數
2. 計算 L2 范數
3. 計算 L∞ 范數
4. 驗證 ||x||₁ ≥ ||x||₂ ≥ ||x||∞

### 中級練習 (Intermediate)

**練習 4: SVD 應用**
1. 對一個 4×3 矩陣進行 SVD 分解
2. 使用前 k 個奇異值重建矩陣
3. 計算重建誤差
4. 繪製重建誤差隨 k 變化的曲線

**練習 5: PCA 實作**
1. 從頭實現 PCA 演算法（不使用 sklearn）
2. 在 Iris 資料集上應用你的實現
3. 比較不同主成分數量下的方差解釋比例
4. 視覺化前兩個主成分

**練習 6: 線性變換視覺化**
1. 定義一個 2D 旋轉矩陣（旋轉 45 度）
2. 定義一個縮放矩陣（x 方向放大 2 倍，y 方向縮小 0.5 倍）
3. 將兩個變換組合
4. 視覺化原始向量和變換後向量

### 進階挑戰 (Advanced)

**挑戰 1: 矩陣條件數分析**
```python
# 建立一個病態矩陣（ill-conditioned matrix）
# 1. 計算其條件數
# 2. 分析條件數對線性方程組求解的影響
# 3. 使用 SVD 或 QR 分解改善數值穩定性
```

**挑戰 2: 低秩矩陣補全**
```python
# 實現矩陣補全演算法
# 1. 建立一個低秩矩陣並隨機移除部分元素
# 2. 使用 SVD 或梯度下降恢復缺失元素
# 3. 比較不同方法的效果
```

**挑戰 3: 實現 LoRA (Low-Rank Adaptation)**
```python
# 模擬 LoRA 在微調中的應用
# 1. 建立一個大型權重矩陣 W ∈ ℝ^(1000×1000)
# 2. 實現 LoRA: ΔW = BA，其中 B ∈ ℝ^(1000×r), A ∈ ℝ^(r×1000)
# 3. 比較完整微調和 LoRA 的參數量
# 4. 分析不同秩 r 的影響
```

**挑戰 4: Transformer 注意力機制分析**
```python
# 深入分析注意力機制的矩陣運算
# 1. 實現多頭自注意力（包含 Q, K, V 投影）
# 2. 分析注意力矩陣的秩
# 3. 視覺化注意力權重分佈
# 4. 實驗不同的注意力變體（如 Flash Attention 的分塊策略）
```

**挑戰 5: 大規模矩陣計算優化**
```python
# 優化大規模矩陣運算
# 1. 實現分塊矩陣乘法
# 2. 使用 Strassen 演算法加速
# 3. 比較與 NumPy 內建實現的性能
# 4. 分析記憶體使用和計算時間的權衡
```

### 應用項目 (Applied Projects)

**項目 1: 圖像壓縮系統**
開發一個基於 SVD 的圖像壓縮工具：
- 讀取任意大小的圖像
- 提供壓縮率控制
- 顯示壓縮前後對比
- 計算 PSNR 和壓縮比

**項目 2: 推薦系統原型**
使用矩陣分解實現協同過濾：
- 用戶-物品評分矩陣分解
- 處理稀疏矩陣
- 預測未知評分
- 評估推薦質量

**項目 3: 文字嵌入降維**
對高維詞向量進行降維：
- 載入預訓練詞向量（如 Word2Vec）
- 使用 PCA 和 t-SNE 降維
- 視覺化語義相似詞
- 分析降維對語義保持的影響

### 數學證明練習 (Proofs)

**證明 1:** 證明矩陣的秩等於非零奇異值的個數

**證明 2:** 證明對稱矩陣的特徵向量相互正交

**證明 3:** 證明 Frobenius 范數的性質：||AB||_F ≤ ||A||_F ||B||_F

**證明 4:** 證明 SVD 提供的低秩逼近在 Frobenius 范數意義下是最優的

### 實驗設計 (Experiments)

**實驗 1: 權重初始化策略**
```python
# 比較不同初始化方法對訓練的影響
# - 零初始化
# - 隨機初始化
# - Xavier 初始化
# - He 初始化
# 在簡單神經網路上測試並記錄收斂速度
```

**實驗 2: 正交性與梯度流**
```python
# 研究正交權重矩陣對梯度流的影響
# 1. 建立深層網路（10+ 層）
# 2. 比較正交初始化和隨機初始化
# 3. 測量梯度在每層的變化
# 4. 分析梯度消失/爆炸現象
```

### 理論問題 (Theoretical Questions)

1. **為什麼在深度學習中 SVD 比特徵分解更常用？**

2. **解釋為什麼批次正規化可以看作是對輸入進行線性變換？**

3. **Transformer 中的注意力矩陣為什麼要除以 √d_k？從線性代數角度解釋。**

4. **解釋模型壓縮中低秩分解的數學原理和侷限性。**

5. **為什麼協方差矩陣一定是半正定的？這對 PCA 有什麼影響？**

### 編程挑戰 (Coding Challenges)

**挑戰 1: 高效的批次矩陣運算**
```python
# 實現支持批次處理的矩陣運算庫
# 要求：
# - 支持廣播機制
# - 優化記憶體使用
# - 通過單元測試
# - 性能接近 NumPy
```

**挑戰 2: 稀疏矩陣運算**
```python
# 實現稀疏矩陣的基本運算
# - CSR/CSC 格式存儲
# - 稀疏矩陣乘法
# - 與 scipy.sparse 性能比較
```

### 解答提示 (Solution Hints)

部分練習的提示和參考實現可在以下位置找到：
- [GitHub Repository - Linear Algebra Solutions](https://github.com/example/ml-math-solutions)
- [練習題討論區](https://discourse.example.com)

### 自我評估檢查清單 (Self-Assessment)

完成本章學習後，你應該能夠：

- [ ] 熟練進行矩陣基本運算（加、乘、轉置、逆）
- [ ] 理解並計算特徵值和特徵向量
- [ ] 執行 SVD 分解並理解其應用
- [ ] 實現 PCA 演算法
- [ ] 理解不同范數的含義和應用場景
- [ ] 解釋深度學習中的矩陣運算（如注意力機制）
- [ ] 使用低秩分解進行模型壓縮
- [ ] 處理數值穩定性問題
- [ ] 優化大規模矩陣計算
- [ ] 將線性代數知識應用於實際問題

---

**總結：** 線性代數是深度學習的數學語言。從基礎的矩陣運算到 Transformer 的注意力機制，從 PCA 降維到模型壓縮，線性代數無處不在。掌握這些概念和工具，將幫助你更深入地理解現代深度學習技術。通過完成上述練習和挑戰，你將建立扎實的線性代數基礎，為深入學習機器學習打下堅實的數學基石。

