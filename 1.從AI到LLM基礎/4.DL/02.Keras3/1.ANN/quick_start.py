"""
Keras 3 ANN 快速入門範例
========================

這個腳本展示了使用 Keras 3 建立和訓練一個簡單神經網路的完整流程。
適合初學者快速上手。

🎯 學習目標:
  - 掌握 Keras 3 的基本工作流程
  - 了解數據預處理方法
  - 學習模型建立、訓練和評估
  - 掌握模型保存和載入

📦 Keras 3 新特性:
  - 多後端支持 (TensorFlow/JAX/PyTorch)
  - 統一的 API
  - 更快的訓練速度
  - 新的 .keras 保存格式

作者: AI Learning Notes
日期: 2025-01
最後更新: 2025-01-18
Keras 版本: 3.0+
"""

import os
# ============================================================================
# Keras 3 後端選擇 (必須在 import keras 之前設置)
# ============================================================================
# 可選: 'tensorflow', 'jax', 'torch'
# 建議初學者使用 'tensorflow'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers, ops
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 打印 Keras 版本和後端資訊
print(f"Keras 版本: {keras.__version__}")
print(f"使用後端: {keras.backend.backend()}")
print("-" * 70)

# =============================================================================
# 1. 載入和預處理數據
# =============================================================================

print("=" * 70)
print("步驟 1: 載入 MNIST 數據集")
print("=" * 70)

# 載入 MNIST 手寫數字數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"訓練數據形狀: {x_train.shape}")  # (60000, 28, 28)
print(f"測試數據形狀: {x_test.shape}")   # (10000, 28, 28)
print(f"訓練標籤形狀: {y_train.shape}")  # (60000,)
print(f"標籤範圍: {y_train.min()} 到 {y_train.max()}")

# 數據正規化: 將像素值從 [0, 255] 縮放到 [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"\n正規化後的數據範圍: {x_train.min():.2f} 到 {x_train.max():.2f}")

# One-hot 編碼標籤
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

print(f"One-hot 編碼後的標籤形狀: {y_train_cat.shape}")
print(f"範例: 標籤 {y_train[0]} → {y_train_cat[0]}")

# =============================================================================
# 2. 建立模型
# =============================================================================

print("\n" + "=" * 70)
print("步驟 2: 建立神經網路模型")
print("=" * 70)

model = keras.Sequential([
    # 輸入層: 將 28x28 的圖像展平為 784 維向量
    layers.Flatten(input_shape=(28, 28), name='flatten'),

    # 第一個隱藏層: 128 個神經元,使用 ReLU 激活函數
    layers.Dense(128, activation='relu', name='hidden1'),

    # Dropout 層: 隨機丟棄 20% 的神經元,防止過擬合
    layers.Dropout(0.2, name='dropout'),

    # 輸出層: 10 個神經元(對應 0-9),使用 Softmax 激活函數
    layers.Dense(10, activation='softmax', name='output')
], name='MNIST_Classifier')

# 顯示模型結構
model.summary()

# =============================================================================
# 3. 編譯模型
# =============================================================================

print("\n" + "=" * 70)
print("步驟 3: 編譯模型")
print("=" * 70)

model.compile(
    # 優化器: Adam 是最常用的優化器
    optimizer='adam',

    # 損失函數: 多類別分類使用 categorical_crossentropy
    loss='categorical_crossentropy',

    # 評估指標: 準確率
    metrics=['accuracy']
)

print("✓ 模型編譯完成")
print(f"  - 優化器: Adam")
print(f"  - 損失函數: Categorical Crossentropy")
print(f"  - 評估指標: Accuracy")

# =============================================================================
# 4. 訓練模型
# =============================================================================

print("\n" + "=" * 70)
print("步驟 4: 訓練模型")
print("=" * 70)

# 訓練模型
history = model.fit(
    x_train, y_train_cat,
    epochs=10,                    # 訓練 10 個 epoch
    batch_size=128,               # 每批次 128 個樣本
    validation_split=0.2,         # 使用 20% 的訓練數據作為驗證集
    verbose=1                     # 顯示訓練進度
)

# =============================================================================
# 5. 評估模型
# =============================================================================

print("\n" + "=" * 70)
print("步驟 5: 評估模型")
print("=" * 70)

# 在測試集上評估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

print(f"測試集損失: {test_loss:.4f}")
print(f"測試集準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# =============================================================================
# 6. 進行預測
# =============================================================================

print("\n" + "=" * 70)
print("步驟 6: 進行預測")
print("=" * 70)

# 隨機選擇 5 個測試樣本進行預測
num_samples = 5
random_indices = np.random.choice(len(x_test), num_samples, replace=False)

predictions = model.predict(x_test[random_indices], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test[random_indices]

print("\n預測結果:")
print("-" * 70)
for i, idx in enumerate(random_indices):
    print(f"樣本 {i+1}:")
    print(f"  真實標籤: {true_labels[i]}")
    print(f"  預測標籤: {predicted_labels[i]}")
    print(f"  預測置信度: {predictions[i][predicted_labels[i]]:.4f}")
    print(f"  預測正確: {'✓' if predicted_labels[i] == true_labels[i] else '✗'}")
    print()

# =============================================================================
# 7. 視覺化結果
# =============================================================================

print("=" * 70)
print("步驟 7: 視覺化訓練歷史")
print("=" * 70)

# 繪製準確率曲線
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.title('模型準確率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 繪製損失曲線
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.title('模型損失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ 訓練歷史圖表已保存為 'training_history.png'")

# 視覺化預測結果
plt.figure(figsize=(15, 3))
for i, idx in enumerate(random_indices):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"真實: {true_labels[i]}\n預測: {predicted_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("✓ 預測結果圖表已保存為 'predictions.png'")

# =============================================================================
# 8. 保存模型
# =============================================================================

print("\n" + "=" * 70)
print("步驟 8: 保存模型")
print("=" * 70)

# 保存完整模型（推薦）
model.save('mnist_model.keras')
print("✓ 模型已保存為 'mnist_model.keras'")

# 載入模型示例
loaded_model = keras.models.load_model('mnist_model.keras')
print("✓ 模型載入成功")

# =============================================================================
# 總結
# =============================================================================

print("\n" + "=" * 70)
print("✓ 訓練完成!")
print("=" * 70)
print(f"""
總結:
  - 訓練樣本數: {len(x_train)}
  - 測試樣本數: {len(x_test)}
  - 訓練 Epochs: 10
  - 最終測試準確率: {test_accuracy*100:.2f}%

生成的文件:
  - training_history.png (訓練歷史圖表)
  - predictions.png (預測結果示例)
  - mnist_model.keras (保存的模型)

下一步建議:
  1. 嘗試修改模型架構（增加/減少層數或神經元數量）
  2. 實驗不同的激活函數
  3. 調整學習率和批次大小
  4. 閱讀 best_practices.py 學習更多技巧
""")

# =============================================================================
# 可選: 顯示圖表（如果在支持 GUI 的環境中）
# =============================================================================

try:
    plt.show()
except:
    print("\n注意: 無法顯示圖表（可能在無 GUI 環境中運行）")
    print("     請查看保存的 PNG 文件")
