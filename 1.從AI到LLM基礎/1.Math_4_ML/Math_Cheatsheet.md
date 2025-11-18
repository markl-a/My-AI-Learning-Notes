# 機器學習數學速查表 (Math Cheatsheet for ML)

這是一份快速參考指南,包含機器學習中最常用的數學公式和概念。

## 📐 線性代數 (Linear Algebra)

### 基本運算

| 運算 | 符號 | 定義 | 維度 |
|------|------|------|------|
| 矩陣乘法 | C = AB | C_ij = Σ_k A_ik B_kj | (m×n)(n×p) → (m×p) |
| 轉置 | A^T | (A^T)_ij = A_ji | (m×n) → (n×m) |
| 點積 | a·b | Σ_i a_i b_i | 標量 |
| 外積 | a⊗b | (a⊗b)_ij = a_i b_j | (n×1)(1×m) → (n×m) |
| Hadamard 積 | A⊙B | (A⊙B)_ij = A_ij B_ij | 逐元素相乘 |
| Trace | tr(A) | Σ_i A_ii | 對角線元素和 |

### 重要分解

**特徵分解 (Eigendecomposition)**
```
A = QΛQ^(-1)
```
- Q: 特徵向量矩陣
- Λ: 特徵值對角矩陣
- 僅適用於方陣

**奇異值分解 (SVD)**
```
A = UΣV^T
```
- U: 左奇異向量 (m×m)
- Σ: 奇異值對角矩陣 (m×n)
- V: 右奇異向量 (n×n)
- 適用於任意矩陣

**Cholesky 分解**
```
A = LL^T
```
- 僅適用於對稱正定矩陣
- L: 下三角矩陣

### 矩陣性質

| 性質 | 定義 | 條件 |
|------|------|------|
| 對稱 | A = A^T | A_ij = A_ji |
| 正交 | Q^T Q = I | 列向量互相正交且單位化 |
| 正定 | x^T A x > 0 | 對所有非零 x |
| 半正定 | x^T A x ≥ 0 | 對所有 x |
| 滿秩 | rank(A) = min(m,n) | 列/列向量線性獨立 |

### 范數 (Norms)

**向量范數**
```
L0 范數: ||x||₀ = #{i : xᵢ ≠ 0}  (非零元素個數)
L1 范數: ||x||₁ = Σᵢ |xᵢ|
L2 范數: ||x||₂ = √(Σᵢ xᵢ²)
L∞ 范數: ||x||∞ = maxᵢ |xᵢ|
Lp 范數: ||x||p = (Σᵢ |xᵢ|^p)^(1/p)
```

**矩陣范數**
```
Frobenius 范數: ||A||_F = √(Σᵢⱼ Aᵢⱼ²) = √tr(A^T A)
核范數: ||A||_* = Σᵢ σᵢ  (奇異值和)
譜范數: ||A||₂ = σ_max  (最大奇異值)
```

### 常用恆等式

```
(AB)^T = B^T A^T
(AB)^(-1) = B^(-1) A^(-1)
tr(AB) = tr(BA)
tr(A) = tr(A^T)
det(AB) = det(A)det(B)
∂(x^T a)/∂x = a
∂(x^T A x)/∂x = (A + A^T)x = 2Ax  (若 A 對稱)
∂(a^T X b)/∂X = ab^T
```

## 📈 微積分 (Calculus)

### 導數規則

**基本規則**
```
(c)' = 0
(x^n)' = nx^(n-1)
(e^x)' = e^x
(ln x)' = 1/x
(sin x)' = cos x
(cos x)' = -sin x
```

**運算規則**
```
(f + g)' = f' + g'
(cf)' = cf'
(fg)' = f'g + fg'  (乘積法則)
(f/g)' = (f'g - fg')/g²  (商法則)
(f∘g)' = f'(g(x))·g'(x)  (鏈式法則)
```

### 多變數微積分

**梯度 (Gradient)**
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]^T
```

**Jacobian 矩陣**
```
J = [∂fᵢ/∂xⱼ]  (m×n 矩陣)
```

**Hessian 矩陣**
```
H = [∂²f/∂xᵢ∂xⱼ]  (n×n 對稱矩陣)
```

**方向導數**
```
D_v f(x) = ∇f(x)·v  (v 為單位向量)
```

### 常用梯度

```
∇(a^T x) = a
∇(x^T A x) = (A + A^T)x
∇(||x||²) = 2x
∇(||Ax - b||²) = 2A^T(Ax - b)
```

### Taylor 展開

**一維**
```
f(x) ≈ f(a) + f'(a)(x-a) + (1/2)f''(a)(x-a)² + ...
```

**多維**
```
f(x) ≈ f(a) + ∇f(a)^T(x-a) + (1/2)(x-a)^T H(a)(x-a)
```

## 🎲 機率與統計 (Probability & Statistics)

### 基本概念

**概率公理**
```
P(A) ∈ [0, 1]
P(S) = 1  (S 為樣本空間)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

**條件概率**
```
P(A|B) = P(A ∩ B) / P(B)
```

**貝葉斯定理**
```
P(A|B) = P(B|A)P(A) / P(B)
P(θ|D) = P(D|θ)P(θ) / P(D)
```

**全概率公式**
```
P(B) = Σᵢ P(B|Aᵢ)P(Aᵢ)
```

### 數值特徵

**期望值 (Expectation)**
```
E[X] = Σₓ x P(x)  (離散)
E[X] = ∫ x p(x) dx  (連續)
E[g(X)] = Σₓ g(x) P(x)
E[aX + b] = aE[X] + b
```

**變異數 (Variance)**
```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
Var(aX + b) = a² Var(X)
```

**協方差 (Covariance)**
```
Cov(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]
Cov(X,X) = Var(X)
```

**相關係數 (Correlation)**
```
ρ(X,Y) = Cov(X,Y) / (σₓ σᵧ)  ∈ [-1, 1]
```

### 常用分佈

**Bernoulli 分佈**
```
P(X=1) = p
E[X] = p
Var(X) = p(1-p)
```

**高斯/常態分佈**
```
N(μ, σ²): p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
E[X] = μ
Var(X) = σ²
```

**多變量高斯分佈**
```
N(μ, Σ): p(x) = (1/√((2π)^d|Σ|)) exp(-1/2(x-μ)^T Σ^(-1)(x-μ))
```

**指數分佈**
```
p(x|λ) = λe^(-λx),  x ≥ 0
E[X] = 1/λ
Var(X) = 1/λ²
```

### 信息論

**熵 (Entropy)**
```
H(X) = -Σₓ P(x) log P(x) = -E[log P(X)]
```

**交叉熵 (Cross-Entropy)**
```
H(P,Q) = -Σₓ P(x) log Q(x) = -E_P[log Q(X)]
```

**KL 散度 (KL Divergence)**
```
D_KL(P||Q) = Σₓ P(x) log(P(x)/Q(x))
            = H(P,Q) - H(P)
```

**互信息 (Mutual Information)**
```
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
       = D_KL(P(X,Y) || P(X)P(Y))
```

### 估計

**最大似然估計 (MLE)**
```
θ_MLE = argmax_θ P(D|θ) = argmax_θ Πᵢ P(xᵢ|θ)
      = argmax_θ Σᵢ log P(xᵢ|θ)
```

**最大後驗估計 (MAP)**
```
θ_MAP = argmax_θ P(θ|D) = argmax_θ P(D|θ)P(θ)
```

## ⚡ 優化 (Optimization)

### 梯度下降變體

**標準梯度下降 (GD)**
```
θ ← θ - η∇L(θ)
```

**隨機梯度下降 (SGD)**
```
θ ← θ - η∇L(θ; xᵢ, yᵢ)
```

**動量 (Momentum)**
```
v ← βv + (1-β)∇L(θ)
θ ← θ - ηv
```

**Nesterov 加速梯度 (NAG)**
```
v ← βv + (1-β)∇L(θ - ηβv)
θ ← θ - ηv
```

**AdaGrad**
```
Gₜ ← Gₜ₋₁ + (∇L(θ))²
θ ← θ - (η/√(Gₜ + ε))∇L(θ)
```

**RMSprop**
```
Eₜ ← βEₜ₋₁ + (1-β)(∇L(θ))²
θ ← θ - (η/√(Eₜ + ε))∇L(θ)
```

**Adam**
```
mₜ ← β₁mₜ₋₁ + (1-β₁)∇L(θ)
vₜ ← β₂vₜ₋₁ + (1-β₂)(∇L(θ))²
m̂ₜ ← mₜ/(1-β₁ᵗ)
v̂ₜ ← vₜ/(1-β₂ᵗ)
θ ← θ - η·m̂ₜ/(√v̂ₜ + ε)
```

### 凸優化條件

**一階條件 (First-order)**
```
∇f(x*) = 0
```

**二階條件 (Second-order)**
```
∇f(x*) = 0  且  H(x*) ⪰ 0  (半正定)
```

**KKT 條件 (約束優化)**
```
∇f(x*) + Σᵢ λᵢ∇gᵢ(x*) + Σⱼ μⱼ∇hⱼ(x*) = 0
gᵢ(x*) ≤ 0,  λᵢ ≥ 0,  λᵢgᵢ(x*) = 0
hⱼ(x*) = 0
```

## 🔥 深度學習特定公式

### 激活函數

**Sigmoid**
```
σ(x) = 1/(1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))
```

**Tanh**
```
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

**ReLU**
```
ReLU(x) = max(0, x)
ReLU'(x) = 1_{x>0}
```

**Leaky ReLU**
```
LeakyReLU(x) = max(αx, x),  α ∈ (0,1)
```

**GELU**
```
GELU(x) = x·Φ(x)  (Φ 為標準常態 CDF)
GELU(x) ≈ x·σ(1.702x)  (近似)
```

**Softmax**
```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))  (數值穩定版)
```

### 損失函數

**均方誤差 (MSE)**
```
L = (1/n)Σᵢ(yᵢ - ŷᵢ)²
∂L/∂ŷᵢ = (2/n)(ŷᵢ - yᵢ)
```

**交叉熵 (Cross-Entropy)**
```
L = -(1/n)Σᵢ yᵢ log(ŷᵢ)  (分類)
L = -(1/n)Σᵢ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]  (二元)
```

**KL 散度損失**
```
L = Σᵢ P(xᵢ) log(P(xᵢ)/Q(xᵢ))
```

### 正則化

**L1 正則化**
```
L_total = L_data + λ||θ||₁
```

**L2 正則化**
```
L_total = L_data + λ||θ||²
```

**Dropout**
```
訓練: h = m ⊙ f(Wx)  (m ~ Bernoulli(p))
測試: h = p·f(Wx)
```

### 批次正規化 (Batch Normalization)

```
μ_B = (1/m)Σᵢ xᵢ
σ²_B = (1/m)Σᵢ(xᵢ - μ_B)²
x̂ᵢ = (xᵢ - μ_B)/√(σ²_B + ε)
yᵢ = γx̂ᵢ + β
```

### 層正規化 (Layer Normalization)

```
μ = (1/H)Σᵢ xᵢ
σ² = (1/H)Σᵢ(xᵢ - μ)²
x̂ᵢ = (xᵢ - μ)/√(σ² + ε)
yᵢ = γx̂ᵢ + β
```

### 注意力機制 (Attention)

**Scaled Dot-Product Attention**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**多頭注意力 (Multi-Head Attention)**
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

## 📊 數值穩定性技巧

**Log-Sum-Exp 技巧**
```
log Σᵢ exp(xᵢ) = a + log Σᵢ exp(xᵢ - a)
其中 a = max(x)
```

**Softmax 穩定計算**
```
softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
```

**梯度裁剪 (Gradient Clipping)**
```
範數裁剪: g ← (c/||g||)·g  若 ||g|| > c
值裁剪: g ← clip(g, -c, c)
```

## 🎯 常用不等式

**Jensen 不等式**
```
f(E[X]) ≤ E[f(X)]  (f 為凸函數)
```

**Cauchy-Schwarz 不等式**
```
|⟨x,y⟩| ≤ ||x||·||y||
```

**Hölder 不等式**
```
||xy||₁ ≤ ||x||_p · ||y||_q  其中 1/p + 1/q = 1
```

**三角不等式**
```
||x + y|| ≤ ||x|| + ||y||
```

## 💡 實用技巧

### 矩陣求逆的替代方法

當需要計算 (A^T A)^(-1) A^T 時（如線性回歸）：
- 使用 QR 分解或 SVD 代替直接求逆
- 更數值穩定

### 大矩陣乘法優化

- 使用 Einstein Summation (`np.einsum`)
- 注意運算順序：(AB)C vs A(BC)
- 利用矩陣的稀疏性或低秩性質

### 記憶體優化

- 梯度檢查點 (Gradient Checkpointing)
- 混合精度訓練 (Mixed Precision)
- 動態計算圖 vs 靜態計算圖

## 📚 相關資源連結

- [線性代數詳解](Linear_Algebra.md)
- [微積分詳解](Calculus.md)
- [機率與統計詳解](Probability_and_Statistics.md)
- [優化基礎詳解](Optimization_Basics.md)

## 🔖 快速參考索引

按字母順序：

| 概念 | 位置 |
|------|------|
| Adam 優化器 | 優化 → 梯度下降變體 |
| Attention | 深度學習特定公式 |
| Batch Normalization | 深度學習特定公式 |
| Bayes 定理 | 機率與統計 → 基本概念 |
| Eigendecomposition | 線性代數 → 重要分解 |
| Entropy | 機率與統計 → 信息論 |
| Gradient | 微積分 → 多變數微積分 |
| Hessian | 微積分 → 多變數微積分 |
| KL Divergence | 機率與統計 → 信息論 |
| Layer Normalization | 深度學習特定公式 |
| MLE/MAP | 機率與統計 → 估計 |
| ReLU | 深度學習特定公式 → 激活函數 |
| Softmax | 深度學習特定公式 → 激活函數 |
| SVD | 線性代數 → 重要分解 |
| Taylor 展開 | 微積分 → Taylor 展開 |

---

**提示：** 本速查表僅提供快速參考，完整的理論推導和實作範例請參考各主題的詳細文檔。

**持續更新：** 隨著新技術和方法的出現，本速查表會持續更新。
