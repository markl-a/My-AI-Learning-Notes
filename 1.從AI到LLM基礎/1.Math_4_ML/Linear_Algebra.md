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

# 創建向量和矩陣
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

# 創建可逆矩陣
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

# 創建對稱矩陣
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

# 創建矩陣
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

# 載入數據
iris = load_iris()
X = iris.data
y = iris.target

# 數據標準化
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

print("\n降維後的數據形狀:", X_pca.shape)
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

# 創建不同初始化方法的權重
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

# 生成數據
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
plt.scatter(X, y, alpha=0.5, label='數據點')
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

# 創建或載入圖像 (這裡用隨機數據示例)
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

## 12. 延伸閱讀與實務參考

- 深入了解更多分解法：LU、QR、Cholesky、Jordan Normal Form 等。
- 高等線代主題：非齊次系統、廣義逆、張量分解 (Tensor Decomposition)。
- 推薦閱讀: 《Linear Algebra and Its Applications》、深度學習教科書中關於線代的章節，以及《Matrix Cookbook》作為公式速查表。

### 推薦資源

- [NumPy 線性代數文檔](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [3Blue1Brown - 線性代數的本質](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

