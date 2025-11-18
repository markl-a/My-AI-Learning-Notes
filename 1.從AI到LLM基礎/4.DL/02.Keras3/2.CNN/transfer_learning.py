"""
Keras 3 遷移學習 (Transfer Learning) 完整範例
=============================================

這個腳本展示了如何使用預訓練模型進行遷移學習,包括特徵提取和微調兩種方法。

🎯 學習目標:
  - 理解遷移學習的概念和優勢
  - 掌握如何使用預訓練模型
  - 學習特徵提取 (Feature Extraction)
  - 掌握模型微調 (Fine-tuning)

📦 預訓練模型: EfficientNetB0
  - 在 ImageNet 上預訓練 (1.4M 張圖像, 1000 個類別)
  - 5.3M 參數
  - 77.1% Top-1 準確率

作者: AI Learning Notes
日期: 2025-01
最後更新: 2025-01-18
Keras 版本: 3.0+
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers
from keras.applications import EfficientNetB0
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# 設置隨機種子
np.random.seed(42)
keras.utils.set_random_seed(42)

print("=" * 70)
print("Keras 3 遷移學習範例")
print("=" * 70)
print(f"Keras 版本: {keras.__version__}")
print(f"使用後端: {keras.backend.backend()}")
print("=" * 70)

# =============================================================================
# 1. 載入和預處理數據
# =============================================================================

print("\n步驟 1: 載入和預處理數據")
print("-" * 70)

# 載入 CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['飛機', '汽車', '鳥', '貓', '鹿',
               '狗', '青蛙', '馬', '船', '卡車']

print(f"訓練數據: {x_train.shape}")
print(f"測試數據: {x_test.shape}")

# EfficientNet 需要至少 32x32 的輸入,CIFAR-10 正好是 32x32
# 但為了獲得更好的性能,我們可以將圖像調整到更大的尺寸

# 方法 1: 使用原始 32x32 (更快)
target_size = 32

# 方法 2: 調整到 224x224 (更好的性能,但更慢)
# target_size = 224

if target_size != 32:
    print(f"\n調整圖像大小到 {target_size}x{target_size}...")
    import cv2
    x_train_resized = np.array([
        cv2.resize(img, (target_size, target_size)) for img in x_train
    ])
    x_test_resized = np.array([
        cv2.resize(img, (target_size, target_size)) for img in x_test
    ])
else:
    x_train_resized = x_train
    x_test_resized = x_test

# 預處理: EfficientNet 需要特定的預處理
from keras.applications.efficientnet import preprocess_input

x_train_preprocessed = preprocess_input(x_train_resized.astype('float32'))
x_test_preprocessed = preprocess_input(x_test_resized.astype('float32'))

print(f"✓ 數據預處理完成")
print(f"  輸入形狀: {x_train_preprocessed.shape}")

# =============================================================================
# 2. 方法 1: 特徵提取 (Feature Extraction)
# =============================================================================

print("\n步驟 2: 方法 1 - 特徵提取")
print("-" * 70)
print("特徵提取: 凍結預訓練模型的權重,只訓練新增的分類層")

# 載入預訓練模型 (不包含頂部分類層)
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(target_size, target_size, 3)
)

# 凍結基礎模型
base_model.trainable = False

print(f"\n基礎模型: EfficientNetB0")
print(f"  總層數: {len(base_model.layers)}")
print(f"  可訓練: {base_model.trainable}")
print(f"  參數量: {base_model.count_params():,}")

# 構建完整模型
model_feature_extraction = keras.Sequential([
    # 數據增強 (可選)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    # 預訓練基礎模型
    base_model,

    # 全局平均池化
    layers.GlobalAveragePooling2D(),

    # 批次正規化
    layers.BatchNormalization(),

    # 全連接層
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    # 輸出層
    layers.Dense(10, activation='softmax')
], name='Feature_Extraction_Model')

print("\n完整模型架構:")
model_feature_extraction.summary()

# 計算可訓練和不可訓練參數
trainable_params = sum([
    keras.backend.count_params(w)
    for w in model_feature_extraction.trainable_weights
])
non_trainable_params = sum([
    keras.backend.count_params(w)
    for w in model_feature_extraction.non_trainable_weights
])

print(f"\n參數統計:")
print(f"  總參數: {trainable_params + non_trainable_params:,}")
print(f"  可訓練參數: {trainable_params:,}")
print(f"  不可訓練參數: {non_trainable_params:,}")
print(f"  可訓練比例: {trainable_params/(trainable_params+non_trainable_params)*100:.2f}%")

# 編譯模型
model_feature_extraction.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
print("\n開始訓練 (特徵提取模式)...")
print("這通常比從頭訓練快得多!")

history_feature = model_feature_extraction.fit(
    x_train_preprocessed, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
    ],
    verbose=1
)

# 評估
test_loss_feature, test_acc_feature = model_feature_extraction.evaluate(
    x_test_preprocessed, y_test, verbose=0
)

print(f"\n特徵提取模型測試準確率: {test_acc_feature*100:.2f}%")

# =============================================================================
# 3. 方法 2: 微調 (Fine-tuning)
# =============================================================================

print("\n步驟 3: 方法 2 - 微調")
print("-" * 70)
print("微調: 解凍部分預訓練層,進行進一步訓練")

# 創建新模型 (或使用之前的模型)
base_model_finetune = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(target_size, target_size, 3)
)

model_finetune = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    base_model_finetune,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='Fine_Tuning_Model')

# 先進行特徵提取訓練
base_model_finetune.trainable = False
model_finetune.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("階段 1: 特徵提取預訓練...")
model_finetune.fit(
    x_train_preprocessed, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# 解凍部分層進行微調
print("\n階段 2: 微調...")
base_model_finetune.trainable = True

# 只訓練最後 20 層
print(f"解凍最後 20 層進行微調...")
for layer in base_model_finetune.layers[:-20]:
    layer.trainable = False

# 檢查哪些層是可訓練的
trainable_layers = [layer.name for layer in base_model_finetune.layers if layer.trainable]
print(f"可訓練層數: {len(trainable_layers)}")

# 重新編譯 (使用更小的學習率)
model_finetune.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 更小的學習率
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 繼續訓練
history_finetune = model_finetune.fit(
    x_train_preprocessed, y_train,
    batch_size=64,
    epochs=15,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-8)
    ],
    verbose=1
)

# 評估
test_loss_finetune, test_acc_finetune = model_finetune.evaluate(
    x_test_preprocessed, y_test, verbose=0
)

print(f"\n微調模型測試準確率: {test_acc_finetune*100:.2f}%")

# =============================================================================
# 4. 比較結果
# =============================================================================

print("\n步驟 4: 比較不同方法的性能")
print("-" * 70)

# 為了公平比較,我們也訓練一個從頭開始的小型 CNN
print("\n訓練基準模型 (從頭開始)...")

baseline_model = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(target_size, target_size, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='Baseline_CNN')

baseline_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_baseline = baseline_model.fit(
    x_train_preprocessed, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ],
    verbose=1
)

test_loss_baseline, test_acc_baseline = baseline_model.evaluate(
    x_test_preprocessed, y_test, verbose=0
)

# 打印比較結果
print("\n" + "=" * 70)
print("性能比較")
print("=" * 70)

results = {
    '從頭訓練 (Baseline)': test_acc_baseline,
    '特徵提取': test_acc_feature,
    '微調': test_acc_finetune
}

for name, acc in results.items():
    print(f"{name:25s}: {acc*100:.2f}%")

# 可視化比較
plt.figure(figsize=(12, 5))

# 準確率比較
plt.subplot(1, 2, 1)
methods = list(results.keys())
accuracies = [v * 100 for v in results.values()]
colors = ['#ff9999', '#66b3ff', '#99ff99']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.8)
plt.ylabel('測試準確率 (%)')
plt.title('不同方法的測試準確率比較', fontweight='bold')
plt.ylim([0, 100])

# 在柱狀圖上添加數值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom')

# 訓練歷史比較
plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['val_accuracy'],
         label='從頭訓練', linewidth=2)
plt.plot(history_feature.history['val_accuracy'],
         label='特徵提取', linewidth=2)
plt.plot(history_finetune.history['val_accuracy'],
         label='微調', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('驗證準確率')
plt.title('訓練過程比較', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ 比較圖表已保存為 'transfer_learning_comparison.png'")

# =============================================================================
# 5. 保存模型
# =============================================================================

print("\n步驟 5: 保存模型")
print("-" * 70)

model_finetune.save('transfer_learning_model.keras')
print("✓ 微調模型已保存為 'transfer_learning_model.keras'")

# =============================================================================
# 6. 使用保存的模型進行推理
# =============================================================================

print("\n步驟 6: 模型推理示例")
print("-" * 70)

# 載入模型
loaded_model = keras.models.load_model('transfer_learning_model.keras')
print("✓ 模型載入成功")

# 隨機選擇測試樣本
n_samples = 5
indices = np.random.choice(len(x_test), n_samples, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    # 預測
    img = x_test_preprocessed[idx:idx+1]
    pred = loaded_model.predict(img, verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]

    # 顯示
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(x_test[idx])
    plt.title(
        f"真實: {class_names[y_test[idx][0]]}\n"
        f"預測: {class_names[pred_class]}\n"
        f"置信度: {confidence:.2%}",
        fontsize=9
    )
    plt.axis('off')

plt.tight_layout()
plt.savefig('inference_examples.png', dpi=150, bbox_inches='tight')
print("✓ 推理示例已保存為 'inference_examples.png'")

# =============================================================================
# 總結
# =============================================================================

print("\n" + "=" * 70)
print("總結與建議")
print("=" * 70)

print(f"""
遷移學習優勢:
  ✓ 訓練速度更快
  ✓ 需要更少的數據
  ✓ 通常獲得更好的性能
  ✓ 可以利用大規模數據集的知識

實驗結果:
  - 從頭訓練:  {test_acc_baseline*100:.2f}%
  - 特徵提取:  {test_acc_feature*100:.2f}%
  - 微調:      {test_acc_finetune*100:.2f}%

最佳實踐:
  1. 先進行特徵提取訓練
  2. 然後進行微調 (使用較小學習率)
  3. 只解凍最後幾層
  4. 使用數據增強
  5. 監控過擬合

常用預訓練模型:
  - EfficientNet (推薦): 速度和精度平衡
  - ResNet: 經典可靠
  - MobileNet: 移動端部署
  - VGG: 簡單易懂 (教學用)

下一步:
  1. 嘗試不同的預訓練模型
  2. 實驗不同的微調策略
  3. 在自定義數據集上應用
  4. 優化模型大小和速度
""")

print("=" * 70)
print("Happy Learning! 🚀")
print("=" * 70)
