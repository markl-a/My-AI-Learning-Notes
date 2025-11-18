# 卷積神經網路 (Convolutional Neural Network, CNN)

> 📚 **章節狀態:** 全新完整版
> 🎯 **學習目標:** 掌握 Keras 3 中 CNN 的原理和實作
> ⏱️ **預計學習時間:** 8-12 小時
> 🔄 **最後更新:** 2025-01-18

---

## 📖 本章節學習內容

### ✅ 完整主題

1. **CNN 基礎理論**
   - 卷積層 (Convolutional Layer)
   - 池化層 (Pooling Layer)
   - 特徵圖 (Feature Maps)
   - 感受野 (Receptive Field)

2. **圖像分類實作**
   - CIFAR-10 數據集
   - 模型架構設計
   - 訓練和評估

3. **數據增強技術**
   - 旋轉、翻轉、縮放
   - 顏色調整
   - Keras 數據增強層

4. **遷移學習**
   - 預訓練模型使用
   - 特徵提取
   - 微調 (Fine-tuning)

5. **現代 CNN 架構**
   - VGG
   - ResNet
   - EfficientNet
   - ConvNeXt

---

## 🧠 CNN 基礎理論

### 為什麼使用 CNN?

傳統的全連接神經網路 (ANN) 處理圖像時存在以下問題:

1. **參數太多**: 一張 224x224x3 的圖像展平後有 150,528 個輸入,連接到 1000 個神經元就需要 1.5 億個參數
2. **無法捕捉空間結構**: 展平操作破壞了圖像的 2D 結構
3. **無法實現平移不變性**: 同樣的物體出現在不同位置需要重新學習

**CNN 的優勢:**
- ✅ 參數共享,大幅減少參數量
- ✅ 局部連接,保留空間結構
- ✅ 平移不變性,同一個特徵檢測器可以檢測圖像任意位置的特徵

### CNN 核心組件

#### 1. 卷積層 (Convolutional Layer)

卷積是 CNN 的核心操作,使用小的濾波器 (filter/kernel) 在圖像上滑動,提取局部特徵。

**工作原理:**

```
輸入圖像 (5x5)        卷積核 (3x3)        特徵圖 (3x3)
┌─────────┐          ┌───┐            ┌─────┐
│1 1 1 0 0│          │1 0 1│           │4 3 4│
│0 1 1 1 0│    ×     │0 1 0│    =     │2 4 3│
│0 0 1 1 1│          │1 0 1│           │2 3 4│
│0 0 1 1 0│          └───┘            └─────┘
│0 1 1 0 0│
└─────────┘
```

**Keras 3 實作:**

```python
from keras import layers

# 基本卷積層
model.add(layers.Conv2D(
    filters=32,              # 輸出通道數 (濾波器數量)
    kernel_size=(3, 3),      # 濾波器大小
    strides=(1, 1),          # 步幅
    padding='same',          # 填充方式: 'same' 或 'valid'
    activation='relu',       # 激活函數
    input_shape=(28, 28, 1)  # 輸入形狀 (高, 寬, 通道)
))
```

**重要參數:**

| 參數 | 說明 | 常用值 |
|------|------|--------|
| **filters** | 卷積核數量 | 32, 64, 128, 256... |
| **kernel_size** | 卷積核大小 | (3,3), (5,5), (7,7) |
| **strides** | 步幅 | (1,1), (2,2) |
| **padding** | 填充方式 | 'same', 'valid' |
| **activation** | 激活函數 | 'relu', 'leaky_relu' |

#### 2. 池化層 (Pooling Layer)

池化用於降採樣,減少特徵圖的空間維度,同時保留重要信息。

**最大池化 (Max Pooling):**

```
輸入 (4x4)           輸出 (2x2)
┌───────┐          ┌───┐
│1 3 2 4│          │3 4│
│2 1 3 2│   →     │4 5│
│3 4 1 5│          └───┘
│1 2 3 2│
└───────┘
```

**Keras 3 實作:**

```python
# 最大池化 (最常用)
model.add(layers.MaxPooling2D(
    pool_size=(2, 2),  # 池化窗口大小
    strides=(2, 2)     # 步幅
))

# 平均池化
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

# 全局平均池化 (常用於分類層之前)
model.add(layers.GlobalAveragePooling2D())
```

#### 3. 批次正規化 (Batch Normalization)

標準化每個批次的激活值,加速訓練並提高穩定性。

```python
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())  # 在激活函數之前或之後
model.add(layers.Activation('relu'))
```

#### 4. Dropout

在訓練時隨機丟棄神經元,防止過擬合。

```python
model.add(layers.Dropout(0.5))  # 丟棄 50% 的神經元
```

---

## 🏗️ CNN 架構設計模式

### 模式 1: 基本 CNN (適合簡單任務)

```python
import keras
from keras import layers

model = keras.Sequential([
    # 第一個卷積塊
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 第二個卷積塊
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 分類器
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

**適用場景:** MNIST, Fashion MNIST 等簡單數據集

### 模式 2: 深度 CNN (VGG 風格)

```python
def create_vgg_style_model(input_shape=(224, 224, 3), num_classes=10):
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
```

**適用場景:** CIFAR-10, CIFAR-100, 自定義圖像分類

### 模式 3: 殘差網路 (ResNet 風格)

```python
def residual_block(x, filters, kernel_size=(3, 3)):
    """殘差塊"""
    shortcut = x

    # 主路徑
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 殘差連接
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1))(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def create_resnet_style_model(input_shape=(224, 224, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    # Initial Conv
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = layers.Conv2D(128, (1, 1), strides=(2, 2))(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = layers.Conv2D(256, (1, 1), strides=(2, 2))(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```

**適用場景:** 複雜數據集, 需要更深網路時

---

## 📊 數據增強技術

數據增強可以:
- ✅ 增加訓練數據的多樣性
- ✅ 防止過擬合
- ✅ 提高模型泛化能力

### Keras 3 數據增強層

```python
from keras import layers

data_augmentation = keras.Sequential([
    # 隨機翻轉
    layers.RandomFlip("horizontal"),

    # 隨機旋轉 (±20 度)
    layers.RandomRotation(0.2),

    # 隨機縮放 (±20%)
    layers.RandomZoom(0.2),

    # 隨機平移
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),

    # 隨機對比度調整
    layers.RandomContrast(0.2),
])

# 使用方式
model = keras.Sequential([
    data_augmentation,  # 在模型開始處添加
    layers.Conv2D(32, (3, 3), activation='relu'),
    # ... 其他層
])
```

### 進階數據增強

```python
import keras
from keras import layers

# Mixup 增強
class Mixup(layers.Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, training=False):
        if not training:
            return inputs

        batch_size = keras.ops.shape(inputs)[0]
        lam = keras.random.beta([batch_size, 1, 1, 1], self.alpha, self.alpha)

        # 隨機排列
        indices = keras.random.shuffle(keras.ops.arange(batch_size))
        mixed = lam * inputs + (1 - lam) * keras.ops.take(inputs, indices, axis=0)

        return mixed

# CutMix 增強
class CutMix(layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, training=False):
        if not training:
            return inputs

        # 實作 CutMix 邏輯
        # (省略詳細實作)
        return inputs
```

---

## 🎯 CIFAR-10 圖像分類完整範例

```python
import keras
from keras import layers
from keras.datasets import cifar10
import numpy as np

# 1. 載入數據
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 數據預處理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. 創建模型
model = keras.Sequential([
    # 數據增強
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    # CNN 特徵提取
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # 分類器
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. 編譯模型
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 訓練模型
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('best_cifar10_model.keras',
                                       save_best_only=True)
    ]
)

# 6. 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"測試準確率: {test_acc*100:.2f}%")
```

---

## 🔄 遷移學習 (Transfer Learning)

使用預訓練模型可以:
- ✅ 節省訓練時間
- ✅ 在小數據集上獲得更好性能
- ✅ 利用大規模數據集的知識

### 使用預訓練模型

```python
from keras.applications import EfficientNetB0, ResNet50, VGG16

# 方法 1: 特徵提取 (凍結預訓練層)
base_model = EfficientNetB0(
    include_top=False,  # 不包含分類層
    weights='imagenet',  # 使用 ImageNet 預訓練權重
    input_shape=(224, 224, 3)
)

# 凍結基礎模型
base_model.trainable = False

# 添加自定義分類層
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 類分類
])

# 編譯和訓練
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

### 微調 (Fine-tuning)

```python
# 先進行特徵提取訓練
# ... (如上)

# 然後解凍部分層進行微調
base_model.trainable = True

# 只訓練最後幾層
for layer in base_model.layers[:-20]:
    layer.trainable = False

# 使用較小的學習率重新編譯
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 更小的學習率
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 繼續訓練
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

---

## 🏆 常用預訓練模型比較

| 模型 | 參數量 | Top-1 準確率 | 適用場景 | 推薦用途 |
|------|--------|-------------|---------|---------|
| **EfficientNetB0** | 5.3M | 77.1% | 資源受限 | 移動端、嵌入式 |
| **EfficientNetB7** | 66M | 84.3% | 高精度需求 | 服務器端 |
| **ResNet50** | 25.6M | 76.0% | 通用 | 特徵提取 |
| **ResNet152** | 60.2M | 78.3% | 深度網路 | 複雜任務 |
| **VGG16** | 138M | 71.3% | 簡單架構 | 教學用途 |
| **MobileNetV2** | 3.5M | 71.8% | 移動端 | 實時應用 |
| **InceptionV3** | 23.9M | 77.9% | 多尺度特徵 | 多樣化數據 |
| **ConvNeXt** | 28M | 82.1% | 最新架構 | 研究前沿 |

### Keras 3 中使用預訓練模型

```python
from keras.applications import (
    EfficientNetB0,
    ResNet50,
    VGG16,
    MobileNetV2,
    InceptionV3,
    ConvNeXtTiny
)

# 創建模型
model = EfficientNetB0(weights='imagenet')

# 預處理輸入
from keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)

# 預測
predictions = model.predict(x)
```

---

## 📈 模型評估與可視化

### 1. 訓練歷史可視化

```python
import matplotlib.pyplot as plt

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 準確率
    ax1.plot(history.history['accuracy'], label='訓練')
    ax1.plot(history.history['val_accuracy'], label='驗證')
    ax1.set_title('模型準確率')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # 損失
    ax2.plot(history.history['loss'], label='訓練')
    ax2.plot(history.history['val_loss'], label='驗證')
    ax2.set_title('模型損失')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)
```

### 2. 特徵圖可視化

```python
def visualize_feature_maps(model, img, layer_name):
    """可視化卷積層的特徵圖"""
    # 創建特徵提取模型
    feature_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )

    # 提取特徵
    features = feature_model.predict(img[np.newaxis, ...])

    # 可視化
    n_features = features.shape[-1]
    size = features.shape[1]

    n_cols = 8
    n_rows = n_features // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    for i in range(n_features):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(features[0, :, :, i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 使用範例
visualize_feature_maps(model, x_test[0], 'conv2d_1')
```

### 3. Grad-CAM 可視化

```python
import numpy as np
import keras

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """生成 Grad-CAM 熱力圖"""
    # 創建梯度模型
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 計算梯度
    with keras.backend.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = keras.ops.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 計算類別通道相對於特徵圖的梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 池化梯度
    pooled_grads = keras.ops.mean(grads, axis=(0, 1, 2))

    # 加權特徵圖
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., None]
    heatmap = keras.ops.squeeze(heatmap)

    # 正規化
    heatmap = keras.ops.maximum(heatmap, 0) / keras.ops.max(heatmap)
    return heatmap.numpy()

# 使用範例
heatmap = make_gradcam_heatmap(img_array, model, 'conv2d_last')

# 疊加到原圖
import cv2
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
```

---

## 🎓 練習題

### 初級練習

1. **基本 CNN 實作**
   - 在 Fashion MNIST 上構建 CNN 模型
   - 達到 90% 以上準確率

2. **數據增強實驗**
   - 比較有/無數據增強的性能差異
   - 嘗試不同的增強組合

3. **架構調整**
   - 實驗不同的濾波器數量
   - 比較不同的池化策略

### 中級練習

4. **CIFAR-10 挑戰**
   - 從頭訓練模型達到 85% 準確率
   - 實作學習率衰減策略

5. **遷移學習應用**
   - 使用 ResNet50 進行特徵提取
   - 在自定義數據集上微調

6. **模型可視化**
   - 實作 Grad-CAM
   - 可視化不同層的特徵圖

### 進階練習

7. **自定義架構**
   - 設計殘差塊 (Residual Block)
   - 實作注意力機制 (Attention)

8. **優化技巧**
   - 實作混合精度訓練
   - 使用多 GPU 訓練

9. **實際應用**
   - 構建貓狗分類器
   - 部署模型到移動端

---

## 🐛 常見問題

### Q1: 為什麼我的 CNN 準確率很低?

**可能原因:**
- 學習率設置不當
- 數據預處理錯誤
- 網路架構不合理
- 訓練時間不足

**解決方案:**
```python
# 1. 檢查數據範圍
print(x_train.min(), x_train.max())  # 應該在 [0, 1] 或 [-1, 1]

# 2. 使用學習率調度
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# 3. 添加 BatchNormalization
model.add(layers.BatchNormalization())

# 4. 增加訓練 epochs
model.fit(..., epochs=100)
```

### Q2: 模型過擬合怎麼辦?

**症狀:** 訓練準確率高,驗證準確率低

**解決方案:**
```python
# 1. 增加 Dropout
model.add(layers.Dropout(0.5))

# 2. 使用數據增強
data_augmentation = keras.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 3. 添加 L2 正則化
model.add(layers.Conv2D(
    64, (3, 3),
    kernel_regularizer=keras.regularizers.l2(0.001)
))

# 4. 使用早停
callbacks = [keras.callbacks.EarlyStopping(patience=10)]

# 5. 減少模型容量
# 使用更少的濾波器或更少的層
```

### Q3: 訓練速度太慢?

**優化方法:**
```python
# 1. 使用混合精度訓練
keras.mixed_precision.set_global_policy('mixed_float16')

# 2. 增加批次大小
model.fit(x_train, y_train, batch_size=128)  # 從 32 增加到 128

# 3. 使用更快的後端 (JAX)
os.environ['KERAS_BACKEND'] = 'jax'

# 4. 確保使用 GPU
print(keras.backend.backend())  # 檢查是否使用 GPU

# 5. 使用數據預取
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
```

### Q4: 如何選擇合適的架構?

**決策樹:**

```
數據集大小 < 1000 張?
├─ 是 → 使用遷移學習 (EfficientNet, ResNet)
└─ 否 → 數據集大小 < 10000 張?
    ├─ 是 → 小型 CNN + 數據增強
    └─ 否 → 可以從頭訓練較大模型

圖像尺寸 < 64x64?
├─ 是 → 淺層 CNN (2-3 個卷積塊)
└─ 否 → 深層 CNN (4-5 個卷積塊)

需要實時推理?
├─ 是 → 輕量級模型 (MobileNet, EfficientNet-B0)
└─ 否 → 可以使用更大模型 (ResNet, EfficientNet-B7)
```

---

## 📚 延伸學習資源

### 官方文檔
- [Keras CNN 指南](https://keras.io/guides/convnets/)
- [Keras 數據增強](https://keras.io/guides/data_augmentation/)
- [Keras 遷移學習](https://keras.io/guides/transfer_learning/)

### 推薦教程
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Deep Learning for Computer Vision (Coursera)](https://www.coursera.org/learn/convolutional-neural-networks)

### 經典論文
- **LeNet-5** (1998): [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- **AlexNet** (2012): [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **VGG** (2014): [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
- **ResNet** (2015): [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **EfficientNet** (2019): [Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)

### 書籍
- *Deep Learning* by Ian Goodfellow (Chapter 9: Convolutional Networks)
- *Hands-On Computer Vision with TensorFlow 2* by Benjamin Planche

---

## ✅ 學習檢查清單

完成本章節後,你應該能夠:

- [ ] 解釋 CNN 的工作原理和優勢
- [ ] 理解卷積、池化、批次正規化的作用
- [ ] 設計並實作基本的 CNN 架構
- [ ] 使用數據增強防止過擬合
- [ ] 應用遷移學習解決實際問題
- [ ] 可視化和解釋 CNN 學到的特徵
- [ ] 選擇合適的預訓練模型
- [ ] 優化 CNN 的訓練過程
- [ ] 診斷和解決常見問題

---

## 🔗 相關章節

- **上一章:** [1. 人工神經網路 (ANN)](../1.ANN/README.md)
- **下一章:** 3. 循環神經網路 (RNN) - 規劃中
- **返回:** [Keras 3 總覽](../README.md)

---

**Happy Learning! 🚀**

> 💡 **學習建議:** CNN 的學習需要大量實踐,建議從簡單的 MNIST 開始,逐步挑戰 CIFAR-10,最後嘗試實際應用!
