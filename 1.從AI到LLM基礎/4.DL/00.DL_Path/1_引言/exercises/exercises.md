# 深度學習引言 - 練習題

> 💪 通過練習鞏固你的理解
>
> 📝 建議先嘗試自己解答，再查看答案

---

## 📚 概念理解題

### 問題 1：機器學習 vs 傳統編程
**題目**：解釋機器學習與傳統編程的主要區別，並舉例說明。

<details>
<summary>點擊查看答案</summary>

**答案**：

**主要區別**：
1. **傳統編程**：
   - 程序員明確編寫規則和邏輯
   - 給定輸入和規則，產生輸出
   - 適合邏輯明確的任務

2. **機器學習**：
   - 從數據中學習規則和模式
   - 給定輸入和輸出，學習規則
   - 適合難以明確編程的任務

**例子**：

**傳統編程**：
```python
# 垃圾郵件過濾 - 傳統方法
def is_spam(email):
    if "中獎" in email or "免費" in email:
        return True
    return False
```

**機器學習**：
```python
# 垃圾郵件過濾 - ML 方法
model = train_on_labeled_emails()
is_spam = model.predict(email)  # 從數據中學習規則
```

**優勢**：
- ML 可以發現人類未注意到的模式
- 自動適應新的垃圾郵件策略
- 無需手動更新規則

</details>

---

### 問題 2：監督學習 vs 無監督學習
**題目**：說明監督學習和無監督學習的區別，各舉兩個應用例子。

<details>
<summary>點擊查看答案</summary>

**答案**：

| 特性 | 監督學習 | 無監督學習 |
|------|---------|-----------|
| **數據** | 帶標籤（輸入-輸出對） | 無標籤（只有輸入） |
| **目標** | 學習輸入到輸出的映射 | 發現數據的內在結構 |
| **訓練** | 利用標籤指導學習 | 自主發現模式 |

**監督學習例子**：
1. **垃圾郵件分類**
   - 輸入：郵件文本
   - 標籤：垃圾/正常
   - 任務：預測新郵件類別

2. **房價預測**
   - 輸入：房屋特徵（面積、位置等）
   - 標籤：房價
   - 任務：預測新房屋價格

**無監督學習例子**：
1. **客戶細分**
   - 輸入：客戶購買行為
   - 無標籤
   - 任務：發現相似客戶群體

2. **異常檢測**
   - 輸入：系統日誌
   - 無標籤
   - 任務：識別異常行為模式

</details>

---

### 問題 3：過擬合問題
**題目**：什麼是過擬合？它為什麼會發生？如何檢測和解決？

<details>
<summary>點擊查看答案</summary>

**答案**：

**定義**：
過擬合指模型在訓練數據上表現很好，但在新數據（測試數據）上表現差。

**發生原因**：
1. **模型過於複雜**
   - 參數太多
   - 網絡層數太深

2. **訓練數據不足**
   - 樣本量太小
   - 不能代表真實分布

3. **訓練時間過長**
   - 學習了噪音和細節
   - 失去泛化能力

**檢測方法**：
1. **觀察訓練/驗證曲線**
   ```
   訓練損失：持續下降 ✓
   驗證損失：開始上升 ✗ <- 過擬合開始
   ```

2. **性能差距**
   ```python
   train_acc = 0.98  # 98% 訓練準確率
   test_acc = 0.65   # 65% 測試準確率
   # 差距過大 -> 過擬合
   ```

**解決方法**：
1. **增加數據**
   - 收集更多訓練樣本
   - 數據增強

2. **簡化模型**
   - 減少參數數量
   - 減少層數

3. **正則化**
   - L1/L2 正則化
   - Dropout
   - Early Stopping

4. **交叉驗證**
   - K-fold 交叉驗證
   - 更可靠的性能評估

</details>

---

### 問題 4：參數 vs 超參數
**題目**：解釋參數和超參數的區別，各舉3個例子。

<details>
<summary>點擊查看答案</summary>

**答案**：

| 特性 | 參數 (Parameters) | 超參數 (Hyperparameters) |
|------|------------------|------------------------|
| **定義** | 模型內部變量 | 訓練前設置的配置 |
| **學習** | 通過訓練自動學習 | 需要手動設置或調優 |
| **數量** | 通常很多 | 相對較少 |
| **存儲** | 保存在模型文件中 | 保存在配置中 |

**參數例子**：
1. **神經網絡權重 (w)**
   ```python
   # 自動通過反向傳播學習
   layer.weight  # 例如：shape (128, 64)
   ```

2. **神經網絡偏置 (b)**
   ```python
   layer.bias  # 例如：shape (64,)
   ```

3. **線性回歸係數**
   ```python
   # y = w1*x1 + w2*x2 + b
   # w1, w2, b 都是參數
   ```

**超參數例子**：
1. **學習率 (Learning Rate)**
   ```python
   optimizer = Adam(lr=0.001)  # 需要手動設置
   ```

2. **批次大小 (Batch Size)**
   ```python
   DataLoader(dataset, batch_size=32)  # 手動選擇
   ```

3. **網絡架構**
   ```python
   # 層數、每層神經元數量
   model = nn.Sequential(
       nn.Linear(10, 64),   # 64 是超參數
       nn.ReLU(),
       nn.Linear(64, 32),   # 32 是超參數
       nn.Linear(32, 1)
   )
   ```

4. **正則化強度**
   ```python
   # L2 正則化係數
   optimizer = Adam(weight_decay=0.01)
   ```

5. **訓練輪數 (Epochs)**
   ```python
   for epoch in range(100):  # 100 是超參數
       train_one_epoch()
   ```

**調優方法**：
- 網格搜索 (Grid Search)
- 隨機搜索 (Random Search)
- 貝葉斯優化
- 經驗和實驗

</details>

---

## 💻 編程練習

### 練習 1：實現簡單的線性回歸

**任務**：從零實現一個簡單的線性回歸模型（不使用 scikit-learn）

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成數據
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 3 * X + 7 + np.random.randn(100, 1) * 5

# TODO: 實現 LinearRegression 類
class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        """
        訓練模型
        提示：使用梯度下降
        """
        # TODO: 你的代碼
        pass

    def predict(self, X):
        """
        預測
        """
        # TODO: 你的代碼
        pass

# 使用你的模型
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# 可視化結果
plt.scatter(X, y, alpha=0.5)
plt.plot(X, predictions, 'r-', linewidth=2)
plt.show()
```

<details>
<summary>點擊查看參考答案</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
        self.history = {'loss': []}

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        """訓練模型"""
        # 初始化參數
        self.w = np.random.randn()
        self.b = np.random.randn()

        m = len(X)

        for epoch in range(epochs):
            # 前向傳播
            y_pred = self.w * X + self.b

            # 計算損失
            loss = np.mean((y_pred - y) ** 2)
            self.history['loss'].append(loss)

            # 計算梯度
            dw = (2/m) * np.sum((y_pred - y) * X)
            db = (2/m) * np.sum(y_pred - y)

            # 更新參數
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        """預測"""
        return self.w * X + self.b

# 生成數據
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 3 * X + 7 + np.random.randn(100, 1) * 5

# 訓練模型
model = LinearRegression()
model.fit(X, y, learning_rate=0.01, epochs=1000)

print(f"\n學習到的參數: w={model.w[0]:.4f}, b={model.b[0]:.4f}")
print(f"真實參數: w=3.0000, b=7.0000")

# 預測
predictions = model.predict(X)

# 可視化結果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 擬合結果
axes[0].scatter(X, y, alpha=0.5, label='Data')
axes[0].plot(X, predictions, 'r-', linewidth=2,
             label=f'Fit: y={model.w[0]:.2f}x+{model.b[0]:.2f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].legend()
axes[0].set_title('Linear Regression Fit')

# 損失曲線
axes[1].plot(model.history['loss'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Training Loss')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

</details>

---

### 練習 2：數據預處理

**任務**：實現常見的數據預處理技術

```python
import numpy as np

# 原始數據（不同特徵的尺度差異很大）
data = np.array([
    [1, 1000, 0.5],
    [2, 2000, 0.6],
    [3, 1500, 0.4],
    [4, 3000, 0.7],
    [5, 2500, 0.5]
])

# TODO: 實現以下預處理函數

def min_max_normalize(X):
    """
    最小-最大歸一化：將數據縮放到 [0, 1]
    公式：x_norm = (x - x_min) / (x_max - x_min)
    """
    # TODO: 你的代碼
    pass

def standardize(X):
    """
    標準化：將數據轉換為均值0，標準差1
    公式：x_std = (x - mean) / std
    """
    # TODO: 你的代碼
    pass

# 測試
print("原始數據:")
print(data)

print("\n歸一化後:")
print(min_max_normalize(data))

print("\n標準化後:")
print(standardize(data))
```

<details>
<summary>點擊查看參考答案</summary>

```python
import numpy as np

def min_max_normalize(X):
    """最小-最大歸一化"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm

def standardize(X):
    """標準化（Z-score）"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_std = (X - mean) / std
    return X_std

# 測試
data = np.array([
    [1, 1000, 0.5],
    [2, 2000, 0.6],
    [3, 1500, 0.4],
    [4, 3000, 0.7],
    [5, 2500, 0.5]
])

print("原始數據:")
print(data)
print(f"均值: {data.mean(axis=0)}")
print(f"標準差: {data.std(axis=0)}")

print("\n歸一化後 (Min-Max):")
normalized = min_max_normalize(data)
print(normalized)
print(f"最小值: {normalized.min(axis=0)}")
print(f"最大值: {normalized.max(axis=0)}")

print("\n標準化後 (Z-score):")
standardized = standardize(data)
print(standardized)
print(f"均值: {standardized.mean(axis=0)}")
print(f"標準差: {standardized.std(axis=0)}")

# 可視化比較
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (title, d) in enumerate([
    ('Original', data),
    ('Min-Max Normalized', normalized),
    ('Standardized', standardized)
]):
    for j in range(d.shape[1]):
        axes[i].scatter([j]*len(d), d[:, j], alpha=0.6, s=100)
    axes[i].set_title(title)
    axes[i].set_xlabel('Feature Index')
    axes[i].set_ylabel('Value')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**解釋**：
- **歸一化**：將所有特徵縮放到相同範圍 [0, 1]
  - 優點：簡單直觀
  - 缺點：對異常值敏感

- **標準化**：將數據轉換為均值0、標準差1的分布
  - 優點：對異常值較穩健
  - 缺點：不保證固定範圍

**使用場景**：
- 神經網絡：通常使用標準化
- 樹模型：通常不需要預處理
- 距離相關算法（如KNN）：需要預處理

</details>

---

### 練習 3：訓練/驗證/測試集分割

**任務**：實現數據集的正確分割

```python
import numpy as np

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    將數據集分割為訓練集、驗證集和測試集

    參數:
        X: 特徵矩陣
        y: 標籤
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        seed: 隨機種子

    返回:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # TODO: 你的代碼
    pass

# 測試
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

print(f"原始數據: {len(X)} 樣本")
print(f"訓練集: {len(X_train)} 樣本 ({len(X_train)/len(X)*100:.1f}%)")
print(f"驗證集: {len(X_val)} 樣本 ({len(X_val)/len(X)*100:.1f}%)")
print(f"測試集: {len(X_test)} 樣本 ({len(X_test)/len(X)*100:.1f}%)")
```

<details>
<summary>點擊查看參考答案</summary>

```python
import numpy as np

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """將數據集分割為訓練集、驗證集和測試集"""

    # 檢查比例總和
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例總和必須為 1.0"

    # 設置隨機種子
    np.random.seed(seed)

    # 獲取樣本數量
    n_samples = len(X)

    # 生成隨機索引
    indices = np.random.permutation(n_samples)

    # 計算分割點
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # 分割索引
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # 分割數據
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

# 測試
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

print(f"✓ 數據集分割完成！")
print(f"\n原始數據: {len(X)} 樣本")
print(f"訓練集: {len(X_train)} 樣本 ({len(X_train)/len(X)*100:.1f}%)")
print(f"驗證集: {len(X_val)} 樣本 ({len(X_val)/len(X)*100:.1f}%)")
print(f"測試集: {len(X_test)} 樣本 ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n特徵形狀: {X_train.shape}")
print(f"標籤形狀: {y_train.shape}")

# 檢查類別分布
print(f"\n原始數據類別分布:")
for i in range(2):
    count = np.sum(y == i)
    print(f"  類別 {i}: {count} ({count/len(y)*100:.1f}%)")

print(f"\n訓練集類別分布:")
for i in range(2):
    count = np.sum(y_train == i)
    print(f"  類別 {i}: {count} ({count/len(y_train)*100:.1f}%)")
```

**重要概念**：

1. **為什麼需要三個集合？**
   - **訓練集**：訓練模型參數
   - **驗證集**：調整超參數，選擇模型
   - **測試集**：最終評估，評估泛化能力

2. **常見比例**：
   - 小數據集：60/20/20 或 70/15/15
   - 大數據集：98/1/1（如有百萬樣本）

3. **注意事項**：
   - 測試集永遠不參與訓練
   - 驗證集用於模型選擇，不用於訓練
   - 使用隨機種子確保可重現性

</details>

---

## 🎯 應用題

### 問題 5：設計機器學習解決方案

**場景**：一家電商公司想要預測用戶是否會購買推薦的商品。

**任務**：設計一個完整的機器學習解決方案，包括：
1. 問題定義
2. 數據收集
3. 特徵工程
4. 模型選擇
5. 評估指標
6. 部署考慮

<details>
<summary>點擊查看參考答案</summary>

**1. 問題定義**
- **類型**：監督學習 - 二元分類
- **輸入**：用戶特徵、商品特徵、行為特徵
- **輸出**：是否購買（0/1）
- **目標**：提高轉化率，增加銷售

**2. 數據收集**

需要的數據：
```python
# 用戶特徵
- 用戶ID
- 年齡
- 性別
- 地理位置
- 會員等級
- 歷史購買次數
- 平均訂單金額

# 商品特徵
- 商品ID
- 類別
- 價格
- 評分
- 評論數
- 庫存狀態

# 行為特徵
- 瀏覽時長
- 點擊次數
- 加入購物車
- 收藏狀態
- 搜索關鍵詞

# 標籤
- 是否購買（目標變量）
```

**3. 特徵工程**

```python
# 創建新特徵
- 價格敏感度 = 歷史平均訂單金額 / 商品價格
- 品類興趣 = 該品類歷史購買次數
- 時間特徵 = 星期幾、是否節假日
- 交互特徵 = 瀏覽時長 * 點擊次數

# 特徵編碼
- 類別特徵：One-Hot 或 Embedding
- 數值特徵：標準化
- 文本特徵：TF-IDF 或 詞嵌入
```

**4. 模型選擇**

候選模型：
```python
# 基線模型
1. Logistic Regression（可解釋性好）
2. Decision Tree（易於理解）

# 進階模型
3. Random Forest（集成學習）
4. Gradient Boosting（XGBoost, LightGBM）
5. Neural Network（深度學習）

# 選擇策略
- 先從簡單模型開始
- 逐步嘗試複雜模型
- 使用交叉驗證比較
```

**5. 評估指標**

由於這是轉化預測（可能類別不平衡）：

```python
主要指標：
- Precision（精確率）：推薦的商品中實際購買的比例
- Recall（召回率）：實際購買的商品中被推薦的比例
- F1-Score：Precision 和 Recall 的調和平均
- AUC-ROC：綜合性能

業務指標：
- 轉化率提升
- GMV（成交總額）增長
- 用戶體驗（不要過度推薦）

# 示例代碼
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f'AUC: {roc_auc_score(y_test, y_pred_proba):.4f}')
```

**6. 部署考慮**

```python
# A/B 測試
- 對照組：原有推薦系統
- 實驗組：新ML模型
- 監控指標：轉化率、用戶參與度

# 在線更新
- 定期重新訓練（如每週）
- 監控模型性能下降
- 準備回滾機制

# 性能優化
- 模型推理延遲 < 100ms
- 批量預測
- 模型壓縮（如量化）

# 監控
- 預測分布監控
- 特徵漂移檢測
- 性能指標告警
```

**完整代碼框架**：

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. 數據準備
def prepare_data():
    # 加載數據
    df = pd.read_csv('user_behavior.csv')

    # 特徵工程
    df['price_sensitivity'] = df['avg_order_value'] / df['product_price']
    df['category_interest'] = df.groupby(['user_id', 'category'])['purchase'].transform('sum')

    # 分離特徵和標籤
    X = df.drop('purchase', axis=1)
    y = df['purchase']

    return train_test_split(X, y, test_size=0.2, stratify=y)

# 2. 訓練模型
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

# 3. 評估模型
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f'AUC: {roc_auc_score(y_test, y_pred_proba):.4f}')

# 4. 主流程
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test)
```

</details>

---

## 📝 思考題

### 問題 6：深度學習 vs 傳統機器學習

在什麼情況下應該使用深度學習，什麼情況下傳統機器學習就足夠了？

<details>
<summary>點擊查看分析</summary>

**使用深度學習的情況**：

1. **大規模數據**
   - 有數百萬樣本
   - 深度學習能更好利用數據

2. **複雜特徵**
   - 圖像、語音、文本
   - 自動特徵提取很重要

3. **端到端學習**
   - 從原始數據到結果
   - 無需手動特徵工程

4. **表示學習**
   - 需要學習數據的抽象表示
   - 遷移學習

**使用傳統ML的情況**：

1. **小數據集**
   - 少於幾千樣本
   - 傳統ML更穩定

2. **結構化數據**
   - 表格數據
   - 特徵明確

3. **可解釋性重要**
   - 醫療診斷
   - 金融風控
   - 決策樹、線性模型

4. **資源限制**
   - 計算資源有限
   - 訓練時間要求

5. **快速原型**
   - 需要快速驗證想法
   - scikit-learn 更簡單

**決策流程圖**：

```
數據量大(>10萬樣本)？
├─ 是 → 數據類型？
│         ├─ 圖像/語音/文本 → 深度學習 ✓
│         └─ 結構化表格 → 嘗試 XGBoost
└─ 否 → 可解釋性重要？
          ├─ 是 → 傳統ML（如邏輯回歸、決策樹）
          └─ 否 → 從簡單模型開始，逐步嘗試
```

**實踐建議**：
- 總是從簡單模型開始
- 建立基線模型
- 逐步增加複雜度
- 根據實際需求選擇

</details>

---

## 🏆 挑戰題

### 挑戰 1：實現 K-折交叉驗證

從零實現 K-折交叉驗證，不使用 scikit-learn 的 `KFold`。

```python
def k_fold_cross_validation(X, y, k=5, model_class=None):
    """
    實現 K-折交叉驗證

    參數:
        X: 特徵矩陣
        y: 標籤
        k: 折數
        model_class: 模型類

    返回:
        scores: 每折的分數列表
    """
    # TODO: 你的代碼
    pass
```

<details>
<summary>點擊查看提示</summary>

**提示**：
1. 將數據分成 k 個大致相等的部分
2. 每次使用一個部分作為驗證集，其餘作為訓練集
3. 訓練模型並評估
4. 返回所有折的平均性能

</details>

---

## ✅ 自我檢查清單

完成練習後，檢查是否達到以下標準：

- [ ] 能解釋機器學習的基本概念
- [ ] 理解監督學習和無監督學習的區別
- [ ] 知道過擬合的原因和解決方法
- [ ] 能區分參數和超參數
- [ ] 會實現簡單的線性回歸
- [ ] 掌握數據預處理技術
- [ ] 理解訓練/驗證/測試集的作用
- [ ] 能設計機器學習解決方案
- [ ] 知道何時使用深度學習 vs 傳統ML

---

**完成所有練習後，你可以：**
1. 使用 `quiz_generator.py` 進行測驗
2. 進入第2章學習數學基礎
3. 開始你的第一個實際項目！

**祝練習順利！💪**
