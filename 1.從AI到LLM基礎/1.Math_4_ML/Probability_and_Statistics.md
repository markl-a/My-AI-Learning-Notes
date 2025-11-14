# Probability_and_Statistics.md

## 目錄
1. 前言
2. 基本機率概念
    - 隨機變數 (Random Variable)
    - 機率質量函數 (PMF) 與 機率密度函數 (PDF)
    - 邊際分佈與條件分佈
    - 獨立性與條件獨立性
    - 貝葉斯機率觀點
3. 常用分佈介紹
    - Bernoulli 與 Multinoulli 分佈
    - 高斯 (正態) 分佈與多變量高斯
    - Exponential 分佈與 Laplace 分佈
    - Dirac delta 與 經驗分佈 (Empirical Distribution)
    - 混合分佈 (Mixture Distribution) 與高斯混合模型 (GMM)
4. 數值特徵：期望值、變異數、協方差
    - 期望值 (Expectation)
    - 變異數 (Variance) 與標準差 (Standard Deviation)
    - 協方差 (Covariance) 與相關係數 (Correlation)
    - 向量與矩陣的協方差矩陣
5. 機率論與信息論
    - 資訊量 (Information) 與 自信息 (Self-Information)
    - 熵 (Entropy)
    - KL 散度 (Kullback-Leibler Divergence)
    - 交叉熵 (Cross-Entropy) 與機器學習中常用的損失函數
6. 最大似然估計 (Maximum Likelihood Estimation, MLE) 與貝葉斯推論
    - Likelihood 函數定義
    - MLE 的求解思路與範例（如高斯分佈參數估計）
    - Maximum A Posteriori (MAP) 估計
    - 全貝葉斯 (Full Bayesian) 推論概念
    - 先驗 (Prior) 與後驗 (Posterior) 分佈
    - Bayes 定理及其在機器學習中的應用
7. 統計推論與取樣方法
    - 點估計與區間估計
    - 常見的估計量特性：偏差 (Bias)、一致性 (Consistency)、有效性 (Efficiency)
    - 蒙地卡羅 (Monte Carlo) 取樣
    - 馬可夫鏈蒙地卡羅 (MCMC)
    - 近似推論 (Approximate Inference) 與變分推論 (Variational Inference)
8. 在機器學習與深度學習的應用
    - 機率模型在分類、回歸問題中的角色：預測輸出分佈與不確定性衡量
    - 確率性梯度下降 (Stochastic Gradient Descent) 的理論基礎與隨機取樣
    - 使用最大似然原則來推導損失函數（如 cross-entropy 對應於訓練分類模型時最大化觀測資料的 likelihood）
    - 經驗分佈與訓練資料集：從資料估計分佈
    - 正則化 (Regularization) 與先驗分佈的關係
9. 數值計算與機率計算的注意事項
    - 數值穩定性：計算對數機率 (log probability) 以避免下溢 (Underflow)
    - 使用 log-sum-exp 技巧來計算歸一化常數
    - 大數法則 (Law of Large Numbers) 與中心極限定理 (Central Limit Theorem) 在估計平均值、方差時的重要性
10. 延伸閱讀與資源

---

## 1. 前言

在機器學習與深度學習中，我們常面對不確定性。為了表達與處理不確定性，機率論 (Probability Theory) 是最自然的框架。透過機率，我們可以描述資料、模型參數與預測結果的不確定性。

同時，統計學 (Statistics) 提供了從數據中估計模型參數、衡量不確定性並進行推論的工具。無論是簡單的線性回歸模型或是複雜的深度神經網路，機率與統計都是底層的支柱。

本章將回顧機率論與統計的基本概念，介紹常用的分佈模型、資訊理論以及最大似然估計等方法，並深入探討這些概念在機器學習實務中扮演的關鍵角色。

## 2. 基本機率概念

### 2.1 隨機變數 (Random Variable)

隨機變數是從樣本空間到實數的函數，用來描述可能結果的數值映射。

- **離散型隨機變數**：取值為可數集合（如擲骰子結果 {1,2,3,4,5,6}）
- **連續型隨機變數**：取值為連續區間（如身高、溫度）

### 2.2 機率質量函數 (PMF) 與機率密度函數 (PDF)

**PMF (Probability Mass Function)**：對離散型隨機變數 X，PMF 定義為：
```
P(X = x) = P(x)
```
滿足：
- P(x) ≥ 0 對所有 x
- Σ P(x) = 1

**PDF (Probability Density Function)**：對連續型隨機變數 X，PDF 定義為：
```
P(a ≤ X ≤ b) = ∫[a,b] p(x) dx
```
滿足：
- p(x) ≥ 0 對所有 x
- ∫ p(x) dx = 1

**注意**：對連續型隨機變數，單點機率 P(X=x) = 0，必須透過積分求區間機率。

### 2.3 邊際分佈與條件分佈

**邊際分佈 (Marginal Distribution)**：從聯合分佈中得到單一變數的分佈

- 離散型：P(X = x) = Σ_y P(X = x, Y = y)
- 連續型：p(x) = ∫ p(x, y) dy

**條件分佈 (Conditional Distribution)**：在已知某變數值時，另一變數的分佈
```
P(Y = y | X = x) = P(X = x, Y = y) / P(X = x)
```

### 2.4 獨立性與條件獨立性

**獨立性 (Independence)**：
```
P(X, Y) = P(X) P(Y)
```
等價於：P(Y|X) = P(Y)

**條件獨立性 (Conditional Independence)**：
```
P(X, Y | Z) = P(X | Z) P(Y | Z)
```
記為 X ⊥ Y | Z

**鏈式法則 (Chain Rule)**：
```
P(x₁, x₂, ..., xₙ) = P(x₁) P(x₂|x₁) P(x₃|x₁,x₂) ... P(xₙ|x₁,...,xₙ₋₁)
```

### 2.5 貝葉斯定理 (Bayes' Theorem)

貝葉斯定理描述後驗概率與先驗概率、似然的關係：

```
P(θ|D) = P(D|θ) P(θ) / P(D)
```

其中：
- P(θ|D)：後驗概率 (Posterior)
- P(D|θ)：似然 (Likelihood)
- P(θ)：先驗概率 (Prior)
- P(D)：證據 (Evidence) = Σ_θ P(D|θ) P(θ)

**貝葉斯機率觀點**：將機率視為主觀信念的強度，而非僅是長期頻率。這種觀點在機器學習中非常重要，允許我們量化不確定性。

## 3. 常用分佈介紹

### 3.1 Bernoulli 分佈

二值型隨機變數 X ∈ {0, 1}，參數為 ϕ ∈ [0,1]：
```
P(X = 1) = ϕ
P(X = 0) = 1 - ϕ
```

PMF: P(X = x) = ϕˣ (1-ϕ)^(1-x)

**性質**：
- E[X] = ϕ
- Var(X) = ϕ(1 - ϕ)

**應用**：二元分類問題（如垃圾郵件檢測）

### 3.2 Multinoulli (Categorical) 分佈

k 類別隨機變數，參數為 p = (p₁, p₂, ..., p_k)，其中 Σpᵢ = 1：
```
P(X = i) = pᵢ
```

通常用 one-hot 向量表示：x = [0, 0, ..., 1, ..., 0]

**應用**：多類別分類問題（如圖像分類）

### 3.3 高斯 (正態) 分佈

**單變量高斯分佈** N(μ, σ²)：
```
p(x) = (1 / √(2πσ²)) exp(-(x-μ)² / (2σ²))
```

**性質**：
- E[X] = μ
- Var(X) = σ²
- 68-95-99.7 法則：約 68% 數據在 μ±σ 範圍內

**多變量高斯分佈** N(μ, Σ)：
```
p(x) = (1 / √((2π)^d |Σ|)) exp(-1/2 (x-μ)ᵀ Σ⁻¹ (x-μ))
```

其中：
- μ ∈ ℝ^d：均值向量
- Σ ∈ ℝ^(d×d)：協方差矩陣（對稱正定）
- |Σ|：Σ 的行列式

**應用**：
- 神經網路權重初始化
- 變分自編碼器 (VAE) 的潛在空間
- 高斯過程

### 3.4 指數族分佈 (Exponential Family)

許多常見分佈可寫成指數族形式：
```
p(x|θ) = h(x) exp(θᵀ T(x) - A(θ))
```

其中：
- T(x)：充分統計量
- A(θ)：log 配分函數
- h(x)：基礎測度

**指數分佈**：
```
p(x|λ) = λ exp(-λx),  x ≥ 0
```
E[X] = 1/λ，常用於等待時間建模

**拉普拉斯分佈**：
```
p(x|μ,b) = (1/2b) exp(-|x-μ|/b)
```
特點：在均值處有尖峰，常用於 L1 正則化的先驗

### 3.5 混合分佈 (Mixture Distribution)

將多個分佈加權組合：
```
p(x) = Σᵢ πᵢ pᵢ(x)
```

其中 πᵢ 為混合係數，滿足 Σπᵢ = 1, πᵢ ≥ 0

**高斯混合模型 (GMM)**：
```
p(x) = Σᵢ πᵢ N(x | μᵢ, Σᵢ)
```

**應用**：
- 聚類分析
- 密度估計
- 語音識別

### 3.6 經驗分佈 (Empirical Distribution)

給定訓練數據 {x₁, x₂, ..., xₙ}，經驗分佈定義為：
```
p̂(x) = (1/n) Σᵢ δ(x - xᵢ)
```

其中 δ 為 Dirac delta 函數。

**意義**：訓練集可視為真實數據分佈的經驗估計，最小化訓練誤差等價於最小化與經驗分佈的期望損失。

## 4. 數值特徵：期望值、變異數、協方差

### 4.1 期望值 (Expectation)

期望值是隨機變數的加權平均值：

**離散型**：
```
E[X] = Σₓ x P(x)
```

**連續型**：
```
E[X] = ∫ x p(x) dx
```

**性質**：
1. 線性性：E[aX + bY] = aE[X] + bE[Y]
2. 獨立變數乘積：若 X⊥Y，則 E[XY] = E[X]E[Y]
3. 函數的期望：E[g(X)] = Σₓ g(x)P(x) 或 ∫ g(x)p(x)dx

**條件期望**：
```
E[Y|X=x] = Σᵧ y P(y|x)  或  ∫ y p(y|x) dy
```

**全期望定理 (Law of Total Expectation)**：
```
E[Y] = E[E[Y|X]]
```

### 4.2 變異數 (Variance)

變異數衡量隨機變數與其期望值的偏離程度：

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

**標準差 (Standard Deviation)**：
```
σ = √Var(X)
```

**性質**：
1. Var(aX + b) = a² Var(X)
2. 若 X⊥Y：Var(X + Y) = Var(X) + Var(Y)
3. Var(X) ≥ 0，且 Var(X) = 0 當且僅當 X 為常數

### 4.3 協方差 (Covariance)

協方差描述兩個隨機變數的線性相關性：

```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

**性質**：
1. Cov(X, X) = Var(X)
2. Cov(X, Y) = Cov(Y, X) (對稱性)
3. Cov(aX + b, Y) = a Cov(X, Y)
4. 若 X⊥Y，則 Cov(X, Y) = 0（但反向不一定成立）

**相關係數 (Correlation Coefficient)**：

標準化的協方差，範圍在 [-1, 1]：
```
ρ(X, Y) = Cov(X, Y) / (σₓ σᵧ)
```

- ρ = 1：完全正相關
- ρ = -1：完全負相關
- ρ = 0：無線性相關（但可能有非線性相關）

### 4.4 協方差矩陣 (Covariance Matrix)

對隨機向量 X = [X₁, X₂, ..., X_d]ᵀ，協方差矩陣定義為：

```
Σ = E[(X - μ)(X - μ)ᵀ]
```

其中 Σᵢⱼ = Cov(Xᵢ, Xⱼ)

**性質**：
1. 對稱性：Σ = Σᵀ
2. 半正定：xᵀΣx ≥ 0 對所有 x
3. 對角元素為各變數的變異數：Σᵢᵢ = Var(Xᵢ)

**在機器學習中的應用**：
- 多變量高斯分佈的參數
- PCA (主成分分析) 的基礎
- 白化 (Whitening) 變換
- 馬氏距離 (Mahalanobis Distance) 計算

## 5. 機率論與信息論

信息論提供了量化不確定性和信息量的數學框架，在機器學習中扮演核心角色。

### 5.1 自信息 (Self-Information)

事件 x 的自信息定義為：
```
I(x) = -log P(x) = log(1/P(x))
```

**直覺理解**：
- 概率越小的事件，發生時攜帶的信息量越大
- P(x) = 1（確定事件）→ I(x) = 0（無信息）
- P(x) → 0（罕見事件）→ I(x) → ∞（大量信息）

通常使用 log₂（單位為 bit）或 ln（單位為 nat）。

### 5.2 熵 (Entropy)

熵是自信息的期望值，衡量隨機變數的平均不確定性：

```
H(X) = E[I(X)] = -Σₓ P(x) log P(x) = -E[log P(X)]
```

對連續型：
```
H(X) = -∫ p(x) log p(x) dx
```

**性質**：
1. H(X) ≥ 0
2. 均勻分佈熵最大：H(X) ≤ log|X|
3. 確定性分佈熵最小：H(X) = 0

**條件熵 (Conditional Entropy)**：
```
H(Y|X) = E[H(Y|X=x)] = -Σₓ P(x) Σᵧ P(y|x) log P(y|x)
```

**聯合熵 (Joint Entropy)**：
```
H(X,Y) = -Σₓ,ᵧ P(x,y) log P(x,y)
```

**鏈式法則**：
```
H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
```

### 5.3 互信息 (Mutual Information)

互信息衡量兩個隨機變數的相互依賴程度：

```
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
```

也可表示為 KL 散度：
```
I(X;Y) = DKL(P(X,Y) || P(X)P(Y))
```

**性質**：
1. I(X;Y) ≥ 0
2. I(X;Y) = 0 當且僅當 X⊥Y
3. 對稱性：I(X;Y) = I(Y;X)

**應用**：特徵選擇、變分信息瓶頸理論

### 5.4 KL 散度 (Kullback-Leibler Divergence)

KL 散度衡量兩個分佈的差異（也稱相對熵）：

```
DKL(P||Q) = Σₓ P(x) log(P(x)/Q(x)) = E_P[log P(X) - log Q(X)]
```

對連續型：
```
DKL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
```

**性質**：
1. DKL(P||Q) ≥ 0，當且僅當 P=Q 時等於 0（Gibbs 不等式）
2. **不對稱**：DKL(P||Q) ≠ DKL(Q||P)
3. 不滿足三角不等式（非距離度量）

**直覺理解**：
- 前向 KL：DKL(P||Q) → 使 Q 覆蓋 P 的所有支持域（避免 zero-forcing）
- 後向 KL：DKL(Q||P) → 使 Q 集中在 P 的高概率區域（zero-forcing）

**應用**：
- 變分推論（最小化 DKL(Q||P)）
- 模型選擇（Akaike Information Criterion）
- 策略梯度方法（KL 約束）

### 5.5 交叉熵 (Cross-Entropy)

交叉熵衡量用分佈 Q 編碼來自分佈 P 的樣本所需的平均編碼長度：

```
H(P,Q) = -Σₓ P(x) log Q(x) = -E_P[log Q(X)]
```

**與 KL 散度的關係**：
```
H(P,Q) = H(P) + DKL(P||Q)
```

**在機器學習中的應用**：

對於分類問題，設真實標籤分佈為 P（通常為 one-hot），模型預測分佈為 Q：
```
Loss = -Σᵢ Σc yᵢc log ŷᵢc
```

其中 yᵢc 為真實標籤（0或1），ŷᵢc 為預測概率。

**二元交叉熵 (Binary Cross-Entropy)**：
```
Loss = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**多類別交叉熵**：
```
Loss = -Σc yc log(ŷc)
```

**最小化交叉熵等價於最大化似然**：
```
min H(P,Q) ⟺ max log P(data|θ)
```

## 6. 最大似然估計 (MLE) 與貝葉斯推論

### 6.1 似然函數 (Likelihood Function)

給定參數 θ 的模型 p(x|θ)，對觀察數據 D = {x⁽¹⁾, x⁽²⁾, ..., x⁽ⁿ⁾}：

**似然函數**：
```
L(θ) = p(D|θ) = ∏ᵢ p(x⁽ⁱ⁾|θ)
```

假設樣本獨立同分佈 (i.i.d.)。

**對數似然 (Log-Likelihood)**：
```
ℓ(θ) = log L(θ) = Σᵢ log p(x⁽ⁱ⁾|θ)
```

使用對數的優點：
1. 將乘積轉為求和，數值更穩定
2. 單調變換不改變最優解
3. 方便計算梯度

### 6.2 最大似然估計 (MLE)

MLE 尋找使似然函數最大的參數：

```
θ_MLE = argmax_θ L(θ) = argmax_θ ℓ(θ)
```

**求解步驟**：
1. 寫出對數似然函數 ℓ(θ)
2. 計算梯度：∇_θ ℓ(θ)
3. 令梯度為零：∇_θ ℓ(θ) = 0
4. 解方程得 θ_MLE

**範例：高斯分佈的 MLE**

給定數據 D = {x₁, ..., xₙ}，假設 x ~ N(μ, σ²)：

對數似然：
```
ℓ(μ,σ²) = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ(xᵢ-μ)²
```

求導令為零：
```
μ_MLE = (1/n) Σᵢ xᵢ  (樣本均值)
σ²_MLE = (1/n) Σᵢ(xᵢ-μ_MLE)²  (樣本方差)
```

**MLE 的性質**：
1. **一致性 (Consistency)**：n→∞ 時，θ_MLE → θ_true
2. **漸近正態性**：大樣本下近似高斯分佈
3. **漸近有效性**：達到 Cramér-Rao 下界
4. 可能有偏差（但漸近無偏）

### 6.3 最大後驗估計 (MAP)

MAP 將參數視為隨機變數，結合先驗信息：

```
θ_MAP = argmax_θ p(θ|D) = argmax_θ p(D|θ) p(θ)
```

等價於：
```
θ_MAP = argmax_θ [log p(D|θ) + log p(θ)]
```

**與 MLE 的關係**：
- MLE：θ_MLE = argmax_θ log p(D|θ)
- MAP：θ_MAP = argmax_θ [log p(D|θ) + log p(θ)]
- 當 p(θ) 為均勻分佈時，MAP = MLE

**MAP 與正則化的關係**：

1. L2 正則化（Ridge）對應高斯先驗：
   ```
   p(θ) = N(0, σ²I)  →  log p(θ) = -λ||θ||²
   ```

2. L1 正則化（Lasso）對應拉普拉斯先驗：
   ```
   p(θ) = Laplace(0, b)  →  log p(θ) = -λ||θ||₁
   ```

### 6.4 貝葉斯推論 (Bayesian Inference)

完整的貝葉斯方法不只找點估計，而是計算完整的後驗分佈：

**後驗分佈**：
```
p(θ|D) = p(D|θ) p(θ) / p(D)
```

其中：
```
p(D) = ∫ p(D|θ) p(θ) dθ  (邊際似然/證據)
```

**貝葉斯預測分佈**：

對新數據點 x*：
```
p(x*|D) = ∫ p(x*|θ) p(θ|D) dθ
```

這個積分考慮了參數不確定性，提供更可靠的預測。

**共軛先驗 (Conjugate Prior)**：

若先驗和後驗屬於同一分佈族，稱為共軛先驗：

| 似然 | 共軛先驗 | 後驗 |
|------|---------|------|
| Bernoulli | Beta | Beta |
| Multinomial | Dirichlet | Dirichlet |
| Gaussian (已知σ²) | Gaussian | Gaussian |
| Gaussian (已知μ) | Inverse-Gamma | Inverse-Gamma |

**優點**：
- 後驗有閉式解
- 易於計算和解釋
- 可遞歸更新

### 6.5 MLE vs MAP vs 全貝葉斯

| 方法 | 輸出 | 不確定性 | 先驗 | 計算複雜度 |
|------|------|---------|------|----------|
| MLE | 點估計 θ_MLE | 無 | 不需要 | 低 |
| MAP | 點估計 θ_MAP | 無 | 需要 | 低-中 |
| 全貝葉斯 | 分佈 p(θ\|D) | 完整 | 需要 | 高 |

**何時使用**：
- MLE：數據充足，無先驗知識
- MAP：有先驗知識，需要正則化
- 全貝葉斯：需要量化不確定性，數據稀少

## 7. 統計推論與取樣方法

### 7.1 統計推論基礎

**點估計 (Point Estimation)**：用單一數值估計參數
- 範例：樣本均值估計母體均值

**區間估計 (Interval Estimation)**：給出參數的可能範圍
- 信賴區間 (Confidence Interval)：以一定置信水平包含真實參數的區間

**假設檢定 (Hypothesis Testing)**：
- 虛無假設 (H₀) vs 對立假設 (H₁)
- Type I 錯誤（假陽性）：拒絕正確的 H₀
- Type II 錯誤（假陰性）：接受錯誤的 H₀
- p-value：在 H₀ 為真時，觀察到當前或更極端結果的概率

### 7.2 估計量的性質

**無偏性 (Unbiasedness)**：
```
E[θ̂] = θ
```
估計量的期望值等於真實值

**偏差 (Bias)**：
```
Bias(θ̂) = E[θ̂] - θ
```

**均方誤差 (MSE)**：
```
MSE(θ̂) = E[(θ̂ - θ)²] = Var(θ̂) + Bias²(θ̂)
```

**一致性 (Consistency)**：
```
lim_{n→∞} P(|θ̂ₙ - θ| > ε) = 0  對所有 ε > 0
```
樣本數增加時，估計量機率收斂到真值

**有效性 (Efficiency)**：在無偏估計量中，方差最小者稱為有效估計量

**Cramér-Rao 下界**：無偏估計量方差的理論下界
```
Var(θ̂) ≥ 1 / I(θ)
```
其中 I(θ) 為 Fisher 信息量

### 7.3 大數法則與中心極限定理

**大數法則 (Law of Large Numbers)**：

樣本均值收斂到期望值：
```
X̄ₙ = (1/n)Σᵢ Xᵢ  →  E[X]  當 n→∞
```

**中心極限定理 (Central Limit Theorem)**：

獨立同分佈隨機變數的和，標準化後趨近標準常態分佈：
```
√n (X̄ₙ - μ) / σ  →  N(0, 1)  當 n→∞
```

**意義**：
- 解釋了為何許多自然現象呈常態分佈
- 為許多統計推論方法提供理論基礎
- 使得樣本均值的信賴區間可用常態分佈近似

### 7.4 蒙地卡羅方法 (Monte Carlo Methods)

用隨機取樣近似期望值或積分：

**基本思想**：
```
E[f(X)] = ∫ f(x) p(x) dx ≈ (1/n) Σᵢ f(xᵢ)
```
其中 xᵢ ~ p(x)

**應用**：
1. 計算高維積分
2. 估計期望值
3. 貝葉斯推論
4. 強化學習中的策略評估

**重要性採樣 (Importance Sampling)**：

當無法直接從 p(x) 採樣時，使用提議分佈 q(x)：
```
E_p[f(X)] = ∫ f(x) p(x) dx = ∫ f(x) (p(x)/q(x)) q(x) dx
          ≈ (1/n) Σᵢ f(xᵢ) w(xᵢ)
```
其中 w(x) = p(x)/q(x) 為重要性權重，xᵢ ~ q(x)

### 7.5 馬可夫鏈蒙地卡羅 (MCMC)

當無法直接從複雜分佈 p(x) 採樣時，構造馬可夫鏈使其平穩分佈為 p(x)。

**Metropolis-Hastings 算法**：

1. 初始化 x₀
2. 對 t = 1, 2, ..., T：
   - 從提議分佈採樣：x' ~ q(x'|xₜ₋₁)
   - 計算接受率：
     ```
     α = min(1, [p(x')q(xₜ₋₁|x')] / [p(xₜ₋₁)q(x'|xₜ₋₁)])
     ```
   - 以概率 α 接受：xₜ = x'，否則 xₜ = xₜ₋₁

**Gibbs Sampling**：

特殊的 MCMC 方法，輪流從條件分佈採樣：

對多維隨機變數 x = (x₁, ..., x_d)：
1. 初始化 x⁽⁰⁾
2. 對 t = 1, 2, ..., T：
   - x₁⁽ᵗ⁾ ~ p(x₁ | x₂⁽ᵗ⁻¹⁾, ..., x_d⁽ᵗ⁻¹⁾)
   - x₂⁽ᵗ⁾ ~ p(x₂ | x₁⁽ᵗ⁾, x₃⁽ᵗ⁻¹⁾, ..., x_d⁽ᵗ⁻¹⁾)
   - ...
   - x_d⁽ᵗ⁾ ~ p(x_d | x₁⁽ᵗ⁾, ..., x_{d-1}⁽ᵗ⁾)

**應用**：貝葉斯推論、隱變數模型、後驗採樣

### 7.6 變分推論 (Variational Inference)

將推論問題轉化為優化問題：用簡單分佈 q(θ) 近似複雜的後驗 p(θ|D)。

**ELBO (Evidence Lower Bound)**：

```
log p(D) ≥ E_q[log p(D,θ)] - E_q[log q(θ)]
         = E_q[log p(D|θ)] - DKL(q(θ)||p(θ))
         := ELBO(q)
```

**優化目標**：
```
q*(θ) = argmax_q ELBO(q) = argmin_q DKL(q(θ)||p(θ|D))
```

**平均場變分推論 (Mean Field VI)**：

假設 q 可分解：
```
q(θ) = ∏ᵢ qᵢ(θᵢ)
```

**優點 vs MCMC**：
- 速度快（優化 vs 採樣）
- 容易診斷收斂
- 可擴展到大數據

**缺點**：
- 可能欠擬合（低估不確定性）
- 需要選擇變分族

## 8. 在機器學習與深度學習的應用

- **模型不確定性**：以機率模型對預測結果給出不確定性量化。

- **損失函數連結最大似然**：例如交叉熵作為訓練分類模型的損失函數，可視為最大化資料在模型下的對數似然。

- **SGD (Stochastic Gradient Descent)**：隨機抽樣小批量 (mini-batch) 訓練，透過資料的邊際分佈估計目標函數的梯度。

- **貝葉斯深度學習**：透過先驗與後驗估計，提升模型對新情境的適應，及對不確定性的量化能力。

- **正則化與先驗**：將偏好稀疏參數或平滑解的偏見以先驗形式表達，有助於防止過度擬合。

## 9. 數值計算與機率計算的注意事項

- **數值穩定性**：計算非常小的機率時採用對數空間 (log probabilities) 避免下溢問題。對計算 softmax、歸一化常數時使用 log-sum-exp 技巧。

- **大數法則與中心極限定理**：保證在樣本數充足時，樣本平均接近母體平均，且合適情況下和高斯分佈產生關聯。這些理論支撐了以樣本近似期望的計算方法與漸近分析。

## 10. Python 實作範例

### 10.1 機率分佈實作

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 設定繪圖風格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Bernoulli 分佈
def bernoulli_example():
    """Bernoulli 分佈範例"""
    p = 0.7  # 成功概率

    # 生成樣本
    samples = np.random.binomial(1, p, size=1000)

    # 理論 PMF
    x = [0, 1]
    pmf = [1-p, p]

    # 視覺化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 理論分佈
    axes[0].bar(x, pmf, alpha=0.7, color='blue')
    axes[0].set_title(f'Bernoulli PMF (p={p})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('P(X=x)')

    # 經驗分佈
    axes[1].hist(samples, bins=[-0.5, 0.5, 1.5], density=True,
                 alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Empirical Distribution (1000 samples)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig('bernoulli_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 統計量
    print(f"理論均值: {p:.3f}, 樣本均值: {samples.mean():.3f}")
    print(f"理論方差: {p*(1-p):.3f}, 樣本方差: {samples.var():.3f}")

# 2. 多變量高斯分佈
def multivariate_gaussian_example():
    """多變量高斯分佈範例"""
    # 參數設定
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.8],
                      [0.8, 1]])

    # 生成樣本
    samples = np.random.multivariate_normal(mu, Sigma, size=1000)

    # 視覺化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 散點圖
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3)
    axes[0].set_title('Samples from Bivariate Gaussian')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    axes[0].axis('equal')
    axes[0].grid(True)

    # 等高線圖
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    rv = stats.multivariate_normal(mu, Sigma)
    Z = rv.pdf(pos)

    axes[1].contour(X1, X2, Z, levels=10)
    axes[1].set_title('PDF Contours')
    axes[1].set_xlabel('X₁')
    axes[1].set_ylabel('X₂')
    axes[1].axis('equal')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('multivariate_gaussian.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 計算樣本協方差矩陣
    sample_cov = np.cov(samples.T)
    print("真實協方差矩陣:\n", Sigma)
    print("樣本協方差矩陣:\n", sample_cov)

# 3. 常見分佈比較
def distribution_comparison():
    """比較常見的連續分佈"""
    x = np.linspace(-5, 5, 1000)

    # 定義不同分佈
    distributions = {
        'Gaussian': stats.norm(0, 1),
        'Laplace': stats.laplace(0, 1/np.sqrt(2)),
        'Student-t (df=3)': stats.t(3),
        'Uniform': stats.uniform(-2, 4)
    }

    plt.figure(figsize=(12, 6))

    for name, dist in distributions.items():
        if name == 'Uniform':
            x_plot = np.linspace(-3, 3, 1000)
        else:
            x_plot = x
        plt.plot(x_plot, dist.pdf(x_plot), label=name, linewidth=2)

    plt.title('Common Probability Distributions', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

# 執行範例
print("=" * 50)
print("1. Bernoulli 分佈範例")
print("=" * 50)
bernoulli_example()

print("\n" + "=" * 50)
print("2. 多變量高斯分佈範例")
print("=" * 50)
multivariate_gaussian_example()

print("\n" + "=" * 50)
print("3. 常見分佈比較")
print("=" * 50)
distribution_comparison()
```

### 10.2 信息論計算

```python
import numpy as np
from scipy.special import rel_entr

def compute_entropy(probs):
    """計算熵"""
    # 過濾零概率（避免 log(0)）
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_cross_entropy(p, q):
    """計算交叉熵"""
    # 過濾零概率
    mask = (p > 0) & (q > 0)
    return -np.sum(p[mask] * np.log2(q[mask]))

def compute_kl_divergence(p, q):
    """計算 KL 散度"""
    # 使用 scipy 的實作（處理數值穩定性）
    return np.sum(rel_entr(p, q)) / np.log(2)  # 轉換為 base-2

def information_theory_demo():
    """信息論概念演示"""
    print("=" * 60)
    print("信息論計算範例")
    print("=" * 60)

    # 範例 1: 均勻分佈 vs 不均勻分佈的熵
    print("\n1. 熵的計算")
    print("-" * 60)

    # 均勻分佈
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    H_uniform = compute_entropy(p_uniform)
    print(f"均勻分佈 P = {p_uniform}")
    print(f"熵 H(P) = {H_uniform:.4f} bits")
    print(f"最大熵（4個類別）= {np.log2(4):.4f} bits")

    # 不均勻分佈
    p_skewed = np.array([0.7, 0.2, 0.08, 0.02])
    H_skewed = compute_entropy(p_skewed)
    print(f"\n不均勻分佈 P = {p_skewed}")
    print(f"熵 H(P) = {H_skewed:.4f} bits")
    print(f"熵減少了 {H_uniform - H_skewed:.4f} bits")

    # 範例 2: KL 散度
    print("\n2. KL 散度計算")
    print("-" * 60)

    p = np.array([0.4, 0.3, 0.2, 0.1])
    q1 = np.array([0.35, 0.35, 0.2, 0.1])  # 接近 p
    q2 = np.array([0.1, 0.2, 0.3, 0.4])    # 遠離 p

    kl_pq1 = compute_kl_divergence(p, q1)
    kl_pq2 = compute_kl_divergence(p, q2)

    print(f"真實分佈 P = {p}")
    print(f"近似分佈 Q1 = {q1}")
    print(f"KL(P||Q1) = {kl_pq1:.4f} bits")
    print(f"\n遠離分佈 Q2 = {q2}")
    print(f"KL(P||Q2) = {kl_pq2:.4f} bits")

    # 驗證不對稱性
    kl_q1p = compute_kl_divergence(q1, p)
    print(f"\n不對稱性驗證:")
    print(f"KL(P||Q1) = {kl_pq1:.4f}")
    print(f"KL(Q1||P) = {kl_q1p:.4f}")

    # 範例 3: 交叉熵與 KL 散度的關係
    print("\n3. 交叉熵與 KL 散度的關係")
    print("-" * 60)

    H_p = compute_entropy(p)
    H_pq1 = compute_cross_entropy(p, q1)
    kl_pq1_check = H_pq1 - H_p

    print(f"H(P) = {H_p:.4f}")
    print(f"H(P, Q1) = {H_pq1:.4f}")
    print(f"KL(P||Q1) = H(P,Q1) - H(P) = {kl_pq1_check:.4f}")
    print(f"直接計算 KL(P||Q1) = {kl_pq1:.4f}")
    print(f"差異: {abs(kl_pq1_check - kl_pq1):.6f}")

# 互信息計算
def compute_mutual_information(joint_prob):
    """計算互信息"""
    # 計算邊際分佈
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)

    # 計算互信息
    mi = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (p_x[i] * p_y[j])
                )
    return mi

def mutual_information_demo():
    """互信息演示"""
    print("\n4. 互信息計算")
    print("-" * 60)

    # 獨立情況
    p_x = np.array([0.5, 0.5])
    p_y = np.array([0.4, 0.6])
    joint_independent = np.outer(p_x, p_y)

    mi_independent = compute_mutual_information(joint_independent)
    print("獨立變數的聯合分佈:")
    print(joint_independent)
    print(f"互信息 I(X;Y) = {mi_independent:.6f} bits")

    # 相依情況
    joint_dependent = np.array([[0.3, 0.1],
                                 [0.1, 0.5]])

    mi_dependent = compute_mutual_information(joint_dependent)
    print("\n相依變數的聯合分佈:")
    print(joint_dependent)
    print(f"互信息 I(X;Y) = {mi_dependent:.4f} bits")

# 執行演示
information_theory_demo()
mutual_information_demo()
```

### 10.3 最大似然估計 (MLE) 實作

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

class GaussianMLE:
    """高斯分佈的最大似然估計"""

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X):
        """使用 MLE 估計參數"""
        # 解析解
        self.mu = np.mean(X)
        self.sigma = np.std(X, ddof=0)  # ddof=0 for MLE (有偏估計)
        return self

    def log_likelihood(self, X):
        """計算對數似然"""
        n = len(X)
        ll = -n/2 * np.log(2 * np.pi * self.sigma**2)
        ll -= 1/(2 * self.sigma**2) * np.sum((X - self.mu)**2)
        return ll

    def pdf(self, x):
        """計算 PDF"""
        return stats.norm.pdf(x, self.mu, self.sigma)

def mle_gaussian_demo():
    """高斯分佈 MLE 演示"""
    print("=" * 60)
    print("高斯分佈 MLE 範例")
    print("=" * 60)

    # 真實參數
    true_mu = 3.0
    true_sigma = 1.5

    # 生成數據
    np.random.seed(42)
    X = np.random.normal(true_mu, true_sigma, size=100)

    # MLE 估計
    model = GaussianMLE()
    model.fit(X)

    print(f"真實參數: μ = {true_mu}, σ = {true_sigma}")
    print(f"MLE 估計: μ̂ = {model.mu:.4f}, σ̂ = {model.sigma:.4f}")
    print(f"對數似然: {model.log_likelihood(X):.4f}")

    # 視覺化
    x_range = np.linspace(X.min()-1, X.max()+1, 200)

    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=20, density=True, alpha=0.5,
             color='skyblue', edgecolor='black', label='Data')
    plt.plot(x_range, model.pdf(x_range), 'r-', linewidth=2,
             label=f'MLE: N({model.mu:.2f}, {model.sigma:.2f}²)')
    plt.plot(x_range, stats.norm.pdf(x_range, true_mu, true_sigma),
             'g--', linewidth=2, label=f'True: N({true_mu}, {true_sigma}²)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Gaussian MLE Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mle_gaussian.png', dpi=150, bbox_inches='tight')
    plt.close()

# Bernoulli 分佈 MLE
class BernoulliMLE:
    """Bernoulli 分佈 MLE"""

    def __init__(self):
        self.p = None

    def fit(self, X):
        """MLE 估計"""
        self.p = np.mean(X)  # 解析解
        return self

    def log_likelihood(self, X):
        """對數似然"""
        return np.sum(X * np.log(self.p) + (1-X) * np.log(1-self.p))

def mle_bernoulli_demo():
    """Bernoulli MLE 演示"""
    print("\n" + "=" * 60)
    print("Bernoulli 分佈 MLE 範例")
    print("=" * 60)

    # 真實參數
    true_p = 0.7

    # 生成數據
    np.random.seed(42)
    X = np.random.binomial(1, true_p, size=100)

    # MLE 估計
    model = BernoulliMLE()
    model.fit(X)

    print(f"真實參數: p = {true_p}")
    print(f"MLE 估計: p̂ = {model.p:.4f}")
    print(f"樣本中 1 的比例: {X.sum()}/{len(X)} = {X.mean():.4f}")

# 執行演示
mle_gaussian_demo()
mle_bernoulli_demo()
```

### 10.4 貝葉斯推論實作

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bayesian_coin_flip():
    """硬幣翻轉的貝葉斯推論"""
    print("=" * 60)
    print("貝葉斯推論：硬幣翻轉範例")
    print("=" * 60)

    # 先驗：Beta(2, 2) - 輕微偏向公平硬幣
    alpha_prior = 2
    beta_prior = 2

    # 觀察數據：10 次翻轉，7 次正面
    n_heads = 7
    n_tails = 3

    # 後驗：Beta(alpha_prior + n_heads, beta_prior + n_tails)
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails

    # 繪製先驗和後驗
    p_values = np.linspace(0, 1, 1000)
    prior = stats.beta(alpha_prior, beta_prior)
    posterior = stats.beta(alpha_post, beta_post)

    # MLE 估計（僅用於比較）
    p_mle = n_heads / (n_heads + n_tails)

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, prior.pdf(p_values), 'b-',
             label=f'Prior: Beta({alpha_prior}, {beta_prior})', linewidth=2)
    plt.plot(p_values, posterior.pdf(p_values), 'r-',
             label=f'Posterior: Beta({alpha_post}, {beta_post})', linewidth=2)
    plt.axvline(p_mle, color='g', linestyle='--',
                label=f'MLE: p={p_mle:.2f}', linewidth=2)
    plt.axvline(posterior.mean(), color='orange', linestyle='--',
                label=f'Posterior Mean: p={posterior.mean():.2f}', linewidth=2)

    plt.xlabel('p (probability of heads)')
    plt.ylabel('Density')
    plt.title(f'Bayesian Inference: {n_heads} heads, {n_tails} tails')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bayesian_coin_flip.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 統計量
    print(f"\n觀察數據: {n_heads} 個正面, {n_tails} 個反面")
    print(f"MLE 估計: p̂ = {p_mle:.4f}")
    print(f"後驗均值: E[p|D] = {posterior.mean():.4f}")
    print(f"後驗中位數: {posterior.median():.4f}")
    print(f"95% 信賴區間: [{posterior.ppf(0.025):.4f}, {posterior.ppf(0.975):.4f}]")

def bayesian_linear_regression():
    """貝葉斯線性回歸"""
    print("\n" + "=" * 60)
    print("貝葉斯線性回歸範例")
    print("=" * 60)

    # 生成數據
    np.random.seed(42)
    n = 20
    X = np.linspace(0, 10, n)
    true_w = 2.5
    true_b = 1.0
    noise_std = 1.0
    y = true_w * X + true_b + np.random.normal(0, noise_std, n)

    # 貝葉斯線性回歸（共軛先驗）
    # 先驗: w ~ N(0, σ²_prior)
    sigma_prior = 10.0

    # 設計矩陣
    X_design = np.column_stack([np.ones(n), X])

    # 後驗參數（已知 noise_std）
    Lambda_prior = (1/sigma_prior**2) * np.eye(2)
    Lambda_post = Lambda_prior + (1/noise_std**2) * X_design.T @ X_design
    Sigma_post = np.linalg.inv(Lambda_post)
    mu_post = Sigma_post @ (1/noise_std**2) * X_design.T @ y

    # MLE（用於比較）
    w_mle = np.linalg.lstsq(X_design, y, rcond=None)[0]

    print(f"真實參數: w = {true_w}, b = {true_b}")
    print(f"MLE 估計: w = {w_mle[1]:.4f}, b = {w_mle[0]:.4f}")
    print(f"後驗均值: w = {mu_post[1]:.4f}, b = {mu_post[0]:.4f}")
    print(f"後驗標準差: σ_w = {np.sqrt(Sigma_post[1,1]):.4f}, "
          f"σ_b = {np.sqrt(Sigma_post[0,0]):.4f}")

    # 視覺化
    X_test = np.linspace(-1, 11, 200)
    X_test_design = np.column_stack([np.ones(len(X_test)), X_test])

    # 後驗預測均值和不確定性
    y_pred_mean = X_test_design @ mu_post
    y_pred_std = np.sqrt(np.diag(X_test_design @ Sigma_post @ X_test_design.T) + noise_std**2)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', s=50, alpha=0.6, label='Data')
    plt.plot(X_test, y_pred_mean, 'r-', linewidth=2, label='Posterior Mean')
    plt.fill_between(X_test,
                     y_pred_mean - 2*y_pred_std,
                     y_pred_mean + 2*y_pred_std,
                     alpha=0.2, color='red', label='95% Credible Interval')
    plt.plot(X_test, X_test_design @ w_mle, 'g--', linewidth=2, label='MLE')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Bayesian Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bayesian_linear_regression.png', dpi=150, bbox_inches='tight')
    plt.close()

# 執行演示
bayesian_coin_flip()
bayesian_linear_regression()
```

### 10.5 蒙地卡羅與 MCMC 實作

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def monte_carlo_integration():
    """蒙地卡羅積分範例"""
    print("=" * 60)
    print("蒙地卡羅積分範例")
    print("=" * 60)

    # 目標：計算 E[X²]，其中 X ~ N(0, 1)
    # 理論值 = 1（因為 Var(X) = 1, E[X] = 0）

    sample_sizes = [10, 100, 1000, 10000, 100000]
    estimates = []

    np.random.seed(42)

    for n in sample_sizes:
        # 從 N(0,1) 採樣
        samples = np.random.normal(0, 1, n)
        # 計算 f(X) = X² 的期望
        estimate = np.mean(samples**2)
        estimates.append(estimate)

        print(f"n = {n:6d}: E[X²] ≈ {estimate:.6f}, "
              f"Error = {abs(estimate - 1.0):.6f}")

    # 視覺化收斂
    plt.figure(figsize=(10, 6))
    plt.semilogx(sample_sizes, estimates, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='True Value')
    plt.xlabel('Number of Samples')
    plt.ylabel('Estimate of E[X²]')
    plt.title('Monte Carlo Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('monte_carlo_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()

def importance_sampling_demo():
    """重要性採樣演示"""
    print("\n" + "=" * 60)
    print("重要性採樣範例")
    print("=" * 60)

    # 目標：計算 E_p[X²]，其中 p(x) = N(5, 1)
    # 理論值 = Var + μ² = 1 + 25 = 26

    true_value = 26.0
    n_samples = 10000
    np.random.seed(42)

    # 方法 1: 標準 MC（直接從 p 採樣）
    samples_direct = np.random.normal(5, 1, n_samples)
    estimate_direct = np.mean(samples_direct**2)

    # 方法 2: 重要性採樣（從 q = N(0, 3) 採樣）
    samples_q = np.random.normal(0, 3, n_samples)

    # 計算重要性權重 w(x) = p(x) / q(x)
    log_p = stats.norm.logpdf(samples_q, 5, 1)
    log_q = stats.norm.logpdf(samples_q, 0, 3)
    weights = np.exp(log_p - log_q)

    # 加權估計
    estimate_is = np.sum(weights * samples_q**2) / np.sum(weights)

    print(f"真實值: {true_value}")
    print(f"直接 MC 估計: {estimate_direct:.4f}, "
          f"Error = {abs(estimate_direct - true_value):.4f}")
    print(f"重要性採樣估計: {estimate_is:.4f}, "
          f"Error = {abs(estimate_is - true_value):.4f}")
    print(f"有效樣本數: {(np.sum(weights)**2 / np.sum(weights**2)):.1f} / {n_samples}")

def metropolis_hastings_demo():
    """Metropolis-Hastings MCMC 演示"""
    print("\n" + "=" * 60)
    print("Metropolis-Hastings MCMC 範例")
    print("=" * 60)

    # 目標分佈：混合高斯 0.3*N(-2,0.5²) + 0.7*N(3,1²)
    def target_pdf(x):
        return (0.3 * stats.norm.pdf(x, -2, 0.5) +
                0.7 * stats.norm.pdf(x, 3, 1.0))

    # Metropolis-Hastings 算法
    def metropolis_hastings(n_samples, proposal_std=1.0):
        samples = np.zeros(n_samples)
        current = 0.0  # 初始點
        n_accepted = 0

        for i in range(n_samples):
            # 提議新點（對稱提議分佈）
            proposed = current + np.random.normal(0, proposal_std)

            # 計算接受率
            acceptance_ratio = target_pdf(proposed) / target_pdf(current)

            # 接受/拒絕
            if np.random.uniform() < acceptance_ratio:
                current = proposed
                n_accepted += 1

            samples[i] = current

        return samples, n_accepted / n_samples

    # 執行 MCMC
    n_samples = 20000
    burn_in = 2000
    samples, acceptance_rate = metropolis_hastings(n_samples)

    # 去除 burn-in 期間
    samples_final = samples[burn_in:]

    print(f"接受率: {acceptance_rate:.2%}")
    print(f"樣本均值: {samples_final.mean():.4f}")
    print(f"樣本標準差: {samples_final.std():.4f}")

    # 視覺化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 軌跡圖
    axes[0, 0].plot(samples[:1000], alpha=0.7)
    axes[0, 0].axvline(burn_in, color='r', linestyle='--', label='Burn-in')
    axes[0, 0].set_title('MCMC Trace (first 1000 samples)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 直方圖 vs 真實分佈
    x_range = np.linspace(-5, 6, 500)
    axes[0, 1].hist(samples_final, bins=50, density=True, alpha=0.5,
                    color='skyblue', edgecolor='black', label='MCMC samples')
    axes[0, 1].plot(x_range, target_pdf(x_range), 'r-',
                    linewidth=2, label='True PDF')
    axes[0, 1].set_title('Sampled Distribution vs True Distribution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 自相關圖
    from numpy import correlate
    lags = range(0, 200)
    autocorr = [np.corrcoef(samples_final[:-lag if lag > 0 else None],
                            samples_final[lag:])[0, 1] if lag > 0
                else 1.0 for lag in lags]

    axes[1, 0].plot(lags, autocorr)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Autocorrelation')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 累積均值
    cumulative_mean = np.cumsum(samples_final) / np.arange(1, len(samples_final) + 1)
    true_mean = 0.3 * (-2) + 0.7 * 3  # 理論均值

    axes[1, 1].plot(cumulative_mean, alpha=0.7)
    axes[1, 1].axhline(y=true_mean, color='r', linestyle='--',
                      linewidth=2, label=f'True Mean = {true_mean:.2f}')
    axes[1, 1].set_title('Cumulative Mean')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Cumulative Mean')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcmc_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()

# 執行演示
monte_carlo_integration()
importance_sampling_demo()
metropolis_hastings_demo()
```

### 10.6 應用：分類問題中的交叉熵損失

```python
import numpy as np
import matplotlib.pyplot as plt

def binary_cross_entropy_demo():
    """二元交叉熵損失演示"""
    print("=" * 60)
    print("二元交叉熵損失函數")
    print("=" * 60)

    # 真實標籤 y ∈ {0, 1}
    # 預測概率 ŷ ∈ [0, 1]

    y_true = 1  # 真實標籤
    y_pred_range = np.linspace(0.01, 0.99, 100)

    # 計算損失
    bce_loss = -(y_true * np.log(y_pred_range) +
                 (1 - y_true) * np.log(1 - y_pred_range))

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred_range, bce_loss, linewidth=2)
    plt.xlabel('Predicted Probability ŷ')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title(f'BCE Loss for True Label y = {y_true}')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=y_true, color='r', linestyle='--',
                label=f'True Label = {y_true}')
    plt.legend()
    plt.savefig('binary_cross_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 展示不同預測的損失
    predictions = [0.1, 0.5, 0.9, 0.99]
    print(f"\n真實標籤 y = {y_true}")
    for pred in predictions:
        loss = -(y_true * np.log(pred) + (1-y_true) * np.log(1-pred))
        print(f"預測 ŷ = {pred:.2f} → Loss = {loss:.4f}")

def categorical_cross_entropy_demo():
    """多類別交叉熵演示"""
    print("\n" + "=" * 60)
    print("多類別交叉熵損失函數")
    print("=" * 60)

    # 真實標籤（one-hot編碼）
    y_true = np.array([0, 1, 0, 0])  # 類別 1

    # 不同的預測分佈
    predictions = {
        'Perfect': np.array([0.01, 0.97, 0.01, 0.01]),
        'Good': np.array([0.05, 0.80, 0.10, 0.05]),
        'Poor': np.array([0.25, 0.25, 0.25, 0.25]),
        'Wrong': np.array([0.05, 0.05, 0.05, 0.85])
    }

    print(f"真實標籤: {y_true} (類別 {np.argmax(y_true)})")
    print("\n預測結果和對應的交叉熵損失:")
    print("-" * 60)

    for name, y_pred in predictions.items():
        # 計算交叉熵
        ce_loss = -np.sum(y_true * np.log(y_pred + 1e-10))
        print(f"{name:10s}: ŷ = {y_pred}, Loss = {ce_loss:.4f}")

# 執行演示
binary_cross_entropy_demo()
categorical_cross_entropy_demo()

print("\n" + "=" * 60)
print("所有範例執行完成！")
print("=" * 60)
```

## 11. 延伸閱讀與資源

- 建議閱讀：《Pattern Recognition and Machine Learning》(Bishop) 中的機率分佈與貝葉斯推論章節。
- 《Deep Learning》 (Goodfellow, Bengio, Courville) 對機率論與信息論有清晰的介紹，並在深度學習應用上深入探討。
- 線上資源如 The Matrix Cookbook（對機率與線代部分有助記公式）與統計學入門課程可補充基礎概念。
- 推薦實作練習：
  - 使用 PyMC3 或 Stan 進行貝葉斯建模
  - 實作變分自編碼器 (VAE) 理解變分推論
  - 探索 scipy.stats 模組中的各種分佈
