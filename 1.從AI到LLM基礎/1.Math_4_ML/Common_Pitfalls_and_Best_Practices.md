# 常見陷阱與最佳實踐 (Common Pitfalls & Best Practices)

本文檔總結了在學習和應用機器學習數學時的常見錯誤、陷阱和最佳實踐。

## 📑 目錄

1. [線性代數常見陷阱](#線性代數常見陷阱)
2. [微積分常見陷阱](#微積分常見陷阱)
3. [機率統計常見陷阱](#機率統計常見陷阱)
4. [優化常見陷阱](#優化常見陷阱)
5. [數值計算最佳實踐](#數值計算最佳實踐)
6. [程式碼實現最佳實踐](#程式碼實現最佳實踐)

---

## 線性代數常見陷阱

### ❌ 陷阱 1: 混淆矩陣乘法和逐元素乘法

**錯誤示例：**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 錯誤：使用 * 進行矩陣乘法
C = A * B  # 這是逐元素乘法！
```

**正確做法：**
```python
# 矩陣乘法
C = A @ B  # 或 np.dot(A, B) 或 np.matmul(A, B)

# 逐元素乘法（Hadamard 積）
C = A * B  # 或 np.multiply(A, B)
```

**最佳實踐：**
- 在深度學習中，明確區分矩陣乘法 `@` 和逐元素乘法 `*`
- 養成使用 `@` 的習慣，程式碼更清晰
- 注意 PyTorch 中 `*` 是逐元素乘法，`@` 或 `torch.matmul` 是矩陣乘法

### ❌ 陷阱 2: 忽略矩陣維度不匹配

**錯誤示例：**
```python
X = np.random.randn(100, 10)  # 100 samples, 10 features
W = np.random.randn(5, 10)    # Wrong shape!

# 這會出錯
Y = X @ W  # ValueError: shapes (100,10) and (5,10) not aligned
```

**正確做法：**
```python
W = np.random.randn(10, 5)  # (features, output_dim)
Y = X @ W  # (100, 10) @ (10, 5) = (100, 5) ✓
```

**最佳實踐：**
- 始終在紙上或註釋中寫出矩陣維度
- 使用 `assert` 檢查維度：`assert X.shape[1] == W.shape[0]`
- 養成檢查輸入輸出形狀的習慣

### ❌ 陷阱 3: 直接計算矩陣逆而不考慮數值穩定性

**錯誤示例：**
```python
# 線性回歸：θ = (X^T X)^(-1) X^T y
X = np.random.randn(100, 50)
y = np.random.randn(100, 1)

# 數值不穩定的做法
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

**正確做法：**
```python
# 方法 1: 使用 solve（更穩定）
theta = np.linalg.solve(X.T @ X, X.T @ y)

# 方法 2: 使用偽逆
theta = np.linalg.pinv(X) @ y

# 方法 3: 使用 lstsq（最穩定，推薦）
theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
```

**最佳實踐：**
- 盡量避免顯式計算逆矩陣
- 優先使用 `solve`、`lstsq` 或 SVD 分解
- 檢查矩陣的條件數：`np.linalg.cond(A)`

### ❌ 陷阱 4: 不理解廣播機制導致的隱藏錯誤

**錯誤示例：**
```python
A = np.array([[1, 2, 3]])  # Shape: (1, 3)
B = np.array([[1], [2], [3]])  # Shape: (3, 1)

# 預期是點積，但得到的是外積！
C = A + B  # Shape: (3, 3) - 廣播導致意外結果
```

**正確做法：**
```python
# 如果要點積
result = A @ B  # Shape: (1, 1)

# 如果確實需要廣播，明確註釋
C = A + B  # Intentional broadcasting: (1,3) + (3,1) -> (3,3)
```

**最佳實踐：**
- 理解 NumPy 廣播規則
- 使用 `np.newaxis` 或 `reshape` 明確維度
- 在複雜運算前檢查形狀

### ❌ 陷阱 5: SVD 後忘記考慮奇異值的順序

**錯誤示例：**
```python
U, s, VT = np.linalg.svd(A)

# 錯誤：假設奇異值是升序排列
low_rank_approx = U[:, -k:] @ np.diag(s[-k:]) @ VT[-k:, :]
```

**正確做法：**
```python
U, s, VT = np.linalg.svd(A)

# SVD 返回的奇異值是降序排列的
# 保留前 k 個最大的奇異值
low_rank_approx = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]
```

**最佳實踐：**
- 記住 SVD 返回降序奇異值
- 可視化奇異值分佈以理解矩陣結構
- 計算累積方差解釋比例

---

## 微積分常見陷阱

### ❌ 陷阱 6: 鏈式法則應用錯誤

**錯誤示例：**
```python
# 計算 d/dx [f(g(x))] 時忘記乘以內函數導數
def f(x):
    return x**2

def g(x):
    return 3*x + 1

# 錯誤：只對外函數求導
def wrong_derivative(x):
    return 2 * g(x)  # 忘記乘以 g'(x) = 3
```

**正確做法：**
```python
def correct_derivative(x):
    # d/dx [f(g(x))] = f'(g(x)) * g'(x)
    return 2 * g(x) * 3  # = 6 * (3x + 1)
```

**最佳實踐：**
- 反向傳播就是鏈式法則的應用
- 畫出計算圖有助於理解梯度流
- 使用自動微分工具驗證手工求導

### ❌ 陷阱 7: 混淆梯度、Jacobian 和 Hessian

**概念混淆：**
- **梯度 (Gradient)**：標量函數對向量的導數 → 向量
- **Jacobian**：向量函數對向量的導數 → 矩陣
- **Hessian**：梯度對向量的導數 → 矩陣（二階導數）

**正確理解：**
```python
import numpy as np

def f(x):
    # 標量函數 f: R^n -> R
    return np.sum(x**2)

def F(x):
    # 向量函數 F: R^n -> R^m
    return np.array([x[0]**2, x[0]*x[1], x[1]**2])

# 梯度：∇f ∈ R^n
def gradient_f(x):
    return 2 * x

# Jacobian：J_F ∈ R^(m×n)
def jacobian_F(x):
    return np.array([
        [2*x[0], 0],
        [x[1], x[0]],
        [0, 2*x[1]]
    ])

# Hessian：H_f ∈ R^(n×n)
def hessian_f(x):
    return 2 * np.eye(len(x))
```

**最佳實踐：**
- 明確函數的輸入輸出維度
- 檢查導數的形狀是否符合預期
- 使用自動微分驗證

### ❌ 陷阱 8: 數值微分步長選擇不當

**錯誤示例：**
```python
def numerical_gradient(f, x, h=1e-10):  # h 太小！
    return (f(x + h) - f(x - h)) / (2 * h)

# 或
def numerical_gradient(f, x, h=0.1):  # h 太大！
    return (f(x + h) - f(x - h)) / (2 * h)
```

**正確做法：**
```python
def numerical_gradient(f, x, h=1e-5):  # 平衡截斷誤差和捨入誤差
    return (f(x + h) - f(x - h)) / (2 * h)

# 或使用自適應步長
def adaptive_numerical_gradient(f, x):
    h = max(1e-5, abs(x) * 1e-5)  # 相對步長
    return (f(x + h) - f(x - h)) / (2 * h)
```

**最佳實踐：**
- 典型步長：`h = 1e-5` 到 `1e-7`
- 使用中心差分而非前向差分
- 驗證梯度時比較數值梯度和解析梯度

---

## 機率統計常見陷阱

### ❌ 陷阱 9: 混淆條件概率和聯合概率

**錯誤示例：**
```python
# 錯誤：將 P(A|B) 當作 P(A,B)
P_A_given_B = 0.8
P_B = 0.5

# 錯誤！P(A,B) ≠ P(A|B)
P_A_and_B = P_A_given_B  # Wrong!
```

**正確做法：**
```python
# P(A,B) = P(A|B) * P(B)
P_A_and_B = P_A_given_B * P_B  # = 0.4

# 或使用貝葉斯定理
# P(A|B) = P(B|A) * P(A) / P(B)
```

**最佳實踐：**
- 畫出概率樹或維恩圖
- 使用貝葉斯定理檢查一致性
- 驗證概率總和為 1

### ❌ 陷阱 10: 在對數空間計算時的數值問題

**錯誤示例：**
```python
# 計算 log(exp(a) + exp(b)) 時直接計算會溢出
a = 1000
b = 1001
result = np.log(np.exp(a) + np.exp(b))  # Overflow!
```

**正確做法：**
```python
def log_sum_exp(a, b):
    """數值穩定的 log-sum-exp"""
    max_val = max(a, b)
    return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))

result = log_sum_exp(1000, 1001)  # Works!
```

**最佳實踐：**
- 在對數空間進行概率計算
- 使用 `scipy.special.logsumexp`
- Softmax 計算時減去最大值

### ❌ 陷阱 11: 忽略獨立性假設

**錯誤示例：**
```python
# 錯誤：假設變量獨立但實際上不獨立
# P(A,B) = P(A) * P(B) 僅在 A ⊥ B 時成立

def joint_probability(P_A, P_B):
    return P_A * P_B  # 假設獨立，但可能不是！
```

**正確做法：**
```python
# 檢查獨立性
def check_independence(joint_prob, marginal_A, marginal_B):
    """檢查 P(A,B) = P(A) * P(B)"""
    expected_joint = marginal_A * marginal_B
    return np.allclose(joint_prob, expected_joint)

# 或使用互資訊
def mutual_information(joint_prob):
    """I(X;Y) = 0 當且僅當 X ⊥ Y"""
    # 實現互資訊計算
    pass
```

**最佳實踐：**
- 明確寫出獨立性假設
- 使用互資訊量化依賴程度
- 考慮條件獨立性

### ❌ 陷阱 12: KL 散度不對稱性

**錯誤示例：**
```python
# 錯誤：將 KL 散度當作距離度量
D_KL_PQ = kl_divergence(P, Q)
D_KL_QP = kl_divergence(Q, P)

# 錯誤假設：D_KL(P||Q) = D_KL(Q||P)
assert D_KL_PQ == D_KL_QP  # 通常不成立！
```

**正確理解：**
```python
# KL 散度不對稱
# D_KL(P||Q) ≠ D_KL(Q||P)

# 前向 KL: D_KL(P||Q) - Q 覆蓋 P 的支持域
# 後向 KL: D_KL(Q||P) - Q 集中在 P 的高概率區域

# 如果需要對稱距離，使用 JS 散度
def js_divergence(P, Q):
    M = (P + Q) / 2
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)
```

**最佳實踐：**
- 理解前向 KL 和後向 KL 的區別
- 變分推論中選擇合適的 KL 方向
- 必要時使用對稱距離（如 JS 散度）

---

## 優化常見陷阱

### ❌ 陷阱 13: 學習率設置不當

**錯誤示例：**
```python
# 學習率過大 → 發散
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# 學習率過小 → 收斂太慢
optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
```

**正確做法：**
```python
# 使用學習率範圍測試
from torch.optim.lr_scheduler import OneCycleLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = OneCycleLR(optimizer, max_lr=0.1,
                       steps_per_epoch=len(train_loader),
                       epochs=epochs)

# 或使用自適應優化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**最佳實踐：**
- 進行學習率範圍測試 (LR range test)
- 使用學習率調度器
- 監控訓練損失的變化
- 對於 Adam，通常 `lr=1e-3` 是好的起點

### ❌ 陷阱 14: 忘記梯度清零

**錯誤示例：**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)

        loss.backward()  # 梯度會累積！
        optimizer.step()  # 使用累積的梯度
        # 忘記 optimizer.zero_grad()
```

**正確做法：**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # 清零梯度

        outputs = model(batch)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
```

**最佳實踐：**
- 在每個訓練步驟開始時清零梯度
- 或在 `optimizer.step()` 後清零
- 梯度累積時才不清零（高級技巧）

### ❌ 陷阱 15: 不檢查梯度的數值穩定性

**錯誤示例：**
```python
# 不檢查梯度是否為 NaN 或 Inf
for epoch in range(num_epochs):
    loss.backward()
    optimizer.step()
    # 可能在訓練中途突然出現 NaN
```

**正確做法：**
```python
for epoch in range(num_epochs):
    loss.backward()

    # 檢查梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                break
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
                break

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

**最佳實踐：**
- 使用梯度裁剪防止梯度爆炸
- 定期檢查梯度統計（均值、標準差）
- 使用 TensorBoard 可視化梯度分佈

### ❌ 陷阱 16: 優化器選擇不當

**錯誤示例：**
```python
# 在所有情況下都使用同一個優化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**正確指導：**
```python
# 根據任務選擇優化器

# CV 任務（通常 SGD + Momentum 泛化更好）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# NLP 任務（通常 Adam 或 AdamW）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                              weight_decay=0.01)

# 大批次訓練（LARS 或 LAMB）
from apex.optimizers import FusedLAMB
optimizer = FusedLAMB(model.parameters(), lr=1e-3)
```

**最佳實踐：**
- 參考相關論文的優化器選擇
- 實驗比較不同優化器
- 考慮計算資源和記憶體限制

---

## 數值計算最佳實踐

### ✅ 實踐 1: 始終考慮數值穩定性

```python
# ❌ 不穩定
def softmax_unstable(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# ✅ 穩定
def softmax_stable(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

### ✅ 實踐 2: 使用對數空間進行概率計算

```python
# ❌ 直接計算可能下溢
probs = [1e-100, 1e-101, 1e-102]
total = sum(probs)  # 可能為 0

# ✅ 對數空間
log_probs = [-230, -232, -234]
log_total = logsumexp(log_probs)
```

### ✅ 實踐 3: 驗證實現的正確性

```python
def verify_gradient(f, grad_f, x, epsilon=1e-5):
    """梯度檢查"""
    numerical_grad = (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
    analytical_grad = grad_f(x)

    relative_error = abs(numerical_grad - analytical_grad) / \
                    (abs(numerical_grad) + abs(analytical_grad) + 1e-10)

    assert relative_error < 1e-5, f"Gradient check failed: {relative_error}"
    print("✓ Gradient check passed")
```

### ✅ 實踐 4: 使用斷言檢查維度

```python
def matrix_multiply(A, B):
    """帶維度檢查的矩陣乘法"""
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D"
    assert A.shape[1] == B.shape[0], \
        f"Incompatible shapes: {A.shape} and {B.shape}"

    C = A @ B

    assert C.shape == (A.shape[0], B.shape[1])
    return C
```

---

## 程式碼實現最佳實踐

### ✅ 實踐 5: 寫清晰的文檔字串

```python
def attention(Q, K, V, mask=None):
    """
    計算縮放點積注意力

    Args:
        Q: Query 矩陣，shape (batch_size, seq_len, d_k)
        K: Key 矩陣，shape (batch_size, seq_len, d_k)
        V: Value 矩陣，shape (batch_size, seq_len, d_v)
        mask: 可選的遮罩，shape (batch_size, seq_len, seq_len)

    Returns:
        output: 注意力輸出，shape (batch_size, seq_len, d_v)
        attention_weights: 注意力權重，shape (batch_size, seq_len, seq_len)

    Examples:
        >>> Q = torch.randn(2, 10, 64)
        >>> K = torch.randn(2, 10, 64)
        >>> V = torch.randn(2, 10, 64)
        >>> output, weights = attention(Q, K, V)
        >>> output.shape
        torch.Size([2, 10, 64])
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### ✅ 實踐 6: 單元測試

```python
import unittest

class TestMatrixOperations(unittest.TestCase):
    def test_matrix_multiply_shapes(self):
        """測試矩陣乘法的形狀"""
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        C = matrix_multiply(A, B)
        self.assertEqual(C.shape, (3, 5))

    def test_matrix_multiply_values(self):
        """測試矩陣乘法的值"""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(C, expected)

    def test_incompatible_shapes(self):
        """測試不兼容的形狀"""
        A = np.random.randn(3, 4)
        B = np.random.randn(5, 6)
        with self.assertRaises(AssertionError):
            matrix_multiply(A, B)

if __name__ == '__main__':
    unittest.main()
```

### ✅ 實踐 7: 性能分析

```python
import time
import numpy as np

def benchmark(func, *args, n_runs=100):
    """基準測試函數性能"""
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func(*args)
        times.append(time.time() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# 使用
A = np.random.randn(1000, 1000)
B = np.random.randn(1000, 1000)

stats = benchmark(np.matmul, A, B)
print(f"平均時間: {stats['mean']:.4f}s ± {stats['std']:.4f}s")
```

---

## 🎯 總結檢查清單

學習和應用數學時，記得：

**線性代數：**
- [ ] 明確區分矩陣乘法和逐元素乘法
- [ ] 始終檢查矩陣維度
- [ ] 避免直接計算逆矩陣
- [ ] 理解廣播機制
- [ ] 使用數值穩定的方法

**微積分：**
- [ ] 正確應用鏈式法則
- [ ] 區分梯度、Jacobian 和 Hessian
- [ ] 驗證數值梯度和解析梯度
- [ ] 使用自動微分工具

**機率統計：**
- [ ] 區分條件概率和聯合概率
- [ ] 在對數空間進行概率計算
- [ ] 明確獨立性假設
- [ ] 理解 KL 散度的不對稱性

**優化：**
- [ ] 合理設置學習率
- [ ] 記得清零梯度
- [ ] 檢查梯度數值穩定性
- [ ] 根據任務選擇合適的優化器

**數值計算：**
- [ ] 考慮數值穩定性
- [ ] 使用對數空間
- [ ] 驗證實現正確性
- [ ] 添加維度檢查

**程式碼實踐：**
- [ ] 寫清晰的文檔
- [ ] 編寫單元測試
- [ ] 性能分析
- [ ] 使用版本控制

---

**記住：** 避免陷阱的最好方法是：
1. 理解背後的數學原理
2. 多做實驗和測試
3. 查閱文檔和最佳實踐
4. 向社群學習和提問

**進一步學習：**
- [Math_Cheatsheet.md](Math_Cheatsheet.md) - 快速查找公式
- [Linear_Algebra.md](Linear_Algebra.md#練習題與挑戰) - 練習題
- [Calculus.md](Calculus.md) - 微積分詳解
- [Probability_and_Statistics.md](Probability_and_Statistics.md) - 機率統計
- [Optimization_Basics.md](Optimization_Basics.md) - 優化方法
