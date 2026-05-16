# 人工神經網路（Artificial Neural Network, ANN）

> 📚 **章節狀態：** 已完成
> 🎯 **學習目標：** 掌握 Keras 3 中 ANN 的基本概念和實作
> ⏱️ **預計學習時間：** 4-6 小時

---

## 📖 本章節學習內容

### ✅ 已完成的主題

1. **MNIST 資料集探索**
   - 載入和可視化手寫數字
   - 理解資料集結構
   - ASCII 藝術視覺化

2. **基本神經網路實作**
   - Sequential API 的使用
   - 簡單的分類模型
   - 使用不同的損失函式

3. **模型優化技巧**
   - 添加隱藏層
   - 使用不同的激活函式
   - 正規化處理

4. **二維分類器**
   - 視覺化決策邊界
   - 比較不同的隱藏層大小
   - 理解模型容量

5. **實際應用：心臟病預測**
   - 處理真實世界資料
   - 比較多種機器學習演算法
   - 模型性能評估

---

## 🧠 人工神經網路基礎理論

### 什麼是 ANN？

人工神經網路是受生物神經系統啟發的計算模型。它由多層互相連接的「神經元」（節點）組成：

```
輸入層 → 隱藏層(1...N) → 輸出層
```

### 核心組件

#### 1. 神經元（Neuron）

每個神經元執行以下計算：

```
y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

其中：
- `x₁, x₂, ..., xₙ` 是輸入
- `w₁, w₂, ..., wₙ` 是權重
- `b` 是偏差（bias）
- `f` 是激活函式
- `y` 是輸出

#### 2. 激活函式（Activation Functions）

**常見激活函式比較：**

| 激活函式 | 公式 | 用途 | 優點 | 缺點 |
|---------|------|------|------|------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | 二元分類輸出層 | 輸出範圍 [0,1] | 梯度消失 |
| **ReLU** | f(x) = max(0,x) | 隱藏層（最常用） | 計算快速 | Dead ReLU |
| **Tanh** | tanh(x) | 隱藏層 | 輸出範圍 [-1,1] | 梯度消失 |
| **Softmax** | σ(xᵢ) = e^xᵢ/Σe^xⱼ | 多類別分類輸出層 | 輸出概率分佈 | 只用於輸出層 |

**Keras 3 實作範例：**

```python
import keras
from keras import layers

# ReLU 激活函式（隱藏層）
model.add(layers.Dense(64, activation='relu'))

# Sigmoid 激活函式（二元分類）
model.add(layers.Dense(1, activation='sigmoid'))

# Softmax 激活函式（多類別分類）
model.add(layers.Dense(10, activation='softmax'))
```

#### 3. 損失函式（Loss Functions）

**常見損失函式：**

| 損失函式 | 適用場景 | Keras 程式碼 |
|---------|---------|-----------|
| **Binary Crossentropy** | 二元分類 | `'binary_crossentropy'` |
| **Categorical Crossentropy** | 多類別分類（one-hot） | `'categorical_crossentropy'` |
| **Sparse Categorical Crossentropy** | 多類別分類（整數標籤） | `'sparse_categorical_crossentropy'` |
| **Mean Squared Error (MSE)** | 回歸問題 | `'mse'` 或 `'mean_squared_error'` |
| **Mean Absolute Error (MAE)** | 回歸問題 | `'mae'` 或 `'mean_absolute_error'` |

#### 4. 優化器（Optimizers）

**常用優化器比較：**

| 優化器 | 特點 | 適用場景 | 學習率建議 |
|-------|------|---------|----------|
| **SGD** | 基礎梯度下降 | 簡單問題 | 0.01 - 0.1 |
| **Adam** | 自適應學習率 | 大多數問題（推薦） | 0.001 |
| **RMSprop** | 適合 RNN | 序列資料 | 0.001 |
| **AdaGrad** | 稀疏資料 | NLP 任務 | 0.01 |

```python
# Adam 優化器（最常用）
model.compile(optimizer='adam', ...)

# 自定義學習率
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...
)
```

---

## 🛠️ Keras 3 實作模式

### 模式 1：Sequential API（推薦初學者）

```python
import keras
from keras import layers

# 建立模型
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

### 模式 2：Functional API（更靈活）

```python
# 定義輸入
inputs = keras.Input(shape=(28, 28))

# 建立層
x = layers.Flatten()(inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# 建立模型
model = keras.Model(inputs=inputs, outputs=outputs)
```

### 模式 3：Model Subclassing（最靈活）

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

model = MyModel()
```

---

## 📊 資料預處理最佳實踐

### 1. 資料正規化

```python
# 方法 1：Min-Max 正規化（縮放到 [0, 1]）
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 方法 2：標準化（均值0，標準差1）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 784))
x_test = scaler.transform(x_test.reshape(-1, 784))
```

### 2. One-Hot 編碼

```python
from keras.utils import to_categorical

# 將整數標籤轉換為 one-hot 向量
# 例如：3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

### 3. 訓練/驗證/測試集分割

```python
from sklearn.model_selection import train_test_split

# 分割資料：70% 訓練，15% 驗證，15% 測試
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

---

## 🎯 實用訓練技巧

### 1. 使用 Callbacks

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    # 早停：驗證損失不再改善時停止訓練
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    # 模型檢查點：保存最佳模型
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    ),

    # 學習率衰減：自動降低學習率
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=callbacks
)
```

### 2. 防止過擬合

```python
from keras import layers, regularizers

model = keras.Sequential([
    # Dropout：隨機丟棄神經元
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 丟棄 50%

    # L2 正則化
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),

    # 批次正規化
    layers.BatchNormalization(),

    layers.Dense(10, activation='softmax')
])
```

### 3. 批次大小選擇

```python
# 小批次（8-32）：更好的泛化，訓練較慢
model.fit(x_train, y_train, batch_size=16)

# 中批次（32-128）：平衡性能和速度（推薦）
model.fit(x_train, y_train, batch_size=64)

# 大批次（128+）：訓練更快，可能過擬合
model.fit(x_train, y_train, batch_size=256)
```

---

## 📈 模型評估與可視化

### 1. 訓練歷史可視化

```python
import matplotlib.pyplot as plt

# 訓練模型並記錄歷史
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=20
)

# 繪製準確率
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 繪製損失
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 2. 混淆矩陣

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 預測
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 分類報告
print(classification_report(y_true, y_pred_classes))
```

---

## 📁 本資料夾檔案說明

### `ANN.ipynb`

這個 Jupyter Notebook 包含以下範例：

1. **範例 0：MNIST 資料集探索**
   - 理解資料集結構
   - 視覺化手寫數字

2. **範例 1：簡易神經網路（使用 MSE）**
   - 使用 WandB 追蹤訓練
   - 基本的 Sequential 模型

3. **範例 2：改進的神經網路（使用 Softmax）**
   - 更適合分類的損失函式
   - 模型性能比較

4. **範例 3：添加隱藏層**
   - 增加模型複雜度
   - 資料正規化

5. **範例 4：二維分類器**
   - 視覺化決策邊界
   - 理解隱藏層大小的影響

6. **範例 5：心臟病預測**
   - 真實世界應用
   - 多種演算法比較（Logistic Regression, SVM, Random Forest, Neural Network 等）

### `heart.csv`

心臟病資料集，包含以下特徵：
- **age**：年齡
- **sex**：性別
- **cp**：胸痛類型
- **trestbps**：靜息血壓
- **chol**：膽固醇
- **fbs**：空腹血糖
- **restecg**：靜息心電圖結果
- **thalach**：最大心率
- **exang**：運動誘發心絞痛
- **oldpeak**：ST 段壓低
- **slope**：運動 ST 段斜率
- **ca**：主要血管數量
- **thal**：地中海貧血
- **target**：是否有心臟病（0/1）

---

## 🎓 練習題

### 初級練習

1. **修改激活函式**
   - 將 ReLU 改為 tanh，觀察性能變化
   - 嘗試使用 LeakyReLU

2. **調整網路架構**
   - 增加/減少隱藏層數量
   - 改變每層的神經元數量

3. **實驗優化器**
   - 比較 Adam、SGD、RMSprop 的性能

### 中級練習

4. **實作 Dropout**
   - 添加 Dropout 層防止過擬合
   - 找出最佳的 dropout 率

5. **學習率調整**
   - 實作學習率衰減
   - 使用 ReduceLROnPlateau

6. **資料增強**
   - 對 MNIST 資料進行旋轉、平移
   - 觀察模型泛化能力的變化

### 進階練習

7. **自定義損失函式**
   ```python
   def custom_loss(y_true, y_pred):
       # 實作你的損失函式
       pass
   ```

8. **實作早停（Early Stopping）**
   - 使用 callbacks 實現早停
   - 保存最佳模型

9. **超參數優化**
   - 使用 Keras Tuner 尋找最佳超參數
   - 比較不同配置的性能

---

## 🐛 常見問題

### Q1: 為什麼我的模型準確率一直是 0.1？

**可能原因：**
- 學習率過大或過小
- 損失函式選擇錯誤
- 資料沒有正規化
- 標籤編碼問題

**解決方案：**
```python
# 檢查資料範圍
print("X range:", x_train.min(), x_train.max())
print("y shape:", y_train.shape)

# 正規化資料
x_train = x_train / 255.0

# 檢查損失函式和標籤的匹配
# sparse_categorical_crossentropy → 整數標籤
# categorical_crossentropy → one-hot 標籤
```

### Q2: 模型過擬合怎麼辦？

**症狀：** 訓練準確率高，驗證準確率低

**解決方案：**
1. 添加 Dropout 層
2. 使用 L1/L2 正則化
3. 減少模型複雜度
4. 增加訓練資料
5. 使用資料增強

### Q3: 訓練速度太慢？

**優化方法：**
```python
# 1. 增加批次大小
model.fit(x_train, y_train, batch_size=128)

# 2. 使用 GPU
# 確保安裝了 GPU 版本的後端

# 3. 減少模型大小
# 使用較少的層或神經元

# 4. 使用混合精度訓練
keras.mixed_precision.set_global_policy('mixed_float16')
```

---

## 📚 延伸學習資源

### 官方文檔
- [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/)
- [Keras Functional API Guide](https://keras.io/guides/functional_api/)
- [Keras Training & Evaluation](https://keras.io/guides/training_with_built_in_methods/)

### 推薦教程
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### 書籍
- *Deep Learning with Python* by François Chollet（Keras 創始人）
- *Hands-On Machine Learning* by Aurélien Géron

---

## ✅ 學習檢查清單

完成本章節後，你應該能夠：

- [ ] 解釋 ANN 的基本原理和組成
- [ ] 使用 Keras Sequential API 建立模型
- [ ] 選擇合適的激活函式和損失函式
- [ ] 正確地預處理資料（正規化、編碼）
- [ ] 訓練模型並監控性能
- [ ] 使用 callbacks 優化訓練過程
- [ ] 評估模型並視覺化結果
- [ ] 診斷和解決常見問題（過擬合、欠擬合）
- [ ] 比較不同的模型架構和超參數

---

## 🔗 相關章節

- **上一章：** [Keras 3 總覽](../README.md)
- **下一章：** 2. 計算機視覺（CV）- 規劃中

---

**Happy Learning! 🚀**

> 💡 **學習建議：** 不要只是閱讀程式碼，親手實作每個範例，並嘗試修改參數觀察變化！
