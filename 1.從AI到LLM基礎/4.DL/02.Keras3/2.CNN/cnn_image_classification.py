"""
Keras 3 CNN 圖像分類完整範例
============================

這個腳本展示了使用 Keras 3 建立和訓練 CNN 模型進行圖像分類的完整流程。
使用 CIFAR-10 數據集作為示例。

🎯 學習目標:
  - 掌握 CNN 模型構建
  - 學習數據增強技術
  - 理解訓練最佳實踐
  - 掌握模型評估方法

📦 數據集: CIFAR-10
  - 60,000 張 32x32 彩色圖像
  - 10 個類別 (飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車)
  - 50,000 張訓練圖像, 10,000 張測試圖像

作者: AI Learning Notes
日期: 2025-01
最後更新: 2025-01-18
Keras 版本: 3.0+
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 設置隨機種子
np.random.seed(42)
keras.utils.set_random_seed(42)

# 打印版本資訊
print("=" * 70)
print(f"Keras 版本: {keras.__version__}")
print(f"使用後端: {keras.backend.backend()}")
print("=" * 70)

# =============================================================================
# 1. 載入和預處理數據
# =============================================================================

print("\n步驟 1: 載入 CIFAR-10 數據集")
print("-" * 70)

# 載入數據
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 類別名稱
class_names = ['飛機', '汽車', '鳥', '貓', '鹿',
               '狗', '青蛙', '馬', '船', '卡車']

print(f"訓練數據形狀: {x_train.shape}")  # (50000, 32, 32, 3)
print(f"測試數據形狀: {x_test.shape}")   # (10000, 32, 32, 3)
print(f"訓練標籤形狀: {y_train.shape}")  # (50000, 1)
print(f"類別數量: {len(class_names)}")

# 數據正規化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"\n正規化後的數據範圍: {x_train.min():.2f} 到 {x_train.max():.2f}")

# 展示部分訓練數據
print("\n可視化部分訓練數據...")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]], fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
print("✓ 樣本圖片已保存為 'cifar10_samples.png'")

# =============================================================================
# 2. 定義數據增強
# =============================================================================

print("\n步驟 2: 定義數據增強策略")
print("-" * 70)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name='data_augmentation')

print("數據增強層:")
print("  - 隨機水平翻轉")
print("  - 隨機旋轉 (±10%)")
print("  - 隨機縮放 (±10%)")
print("  - 隨機平移 (±10%)")

# 可視化數據增強效果
sample_img = x_train[0:1]
plt.figure(figsize=(10, 4))

plt.subplot(1, 5, 1)
plt.imshow(sample_img[0])
plt.title('原始圖像')
plt.axis('off')

for i in range(4):
    augmented = data_augmentation(sample_img, training=True)
    plt.subplot(1, 5, i + 2)
    plt.imshow(augmented[0])
    plt.title(f'增強 {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('data_augmentation_demo.png', dpi=150, bbox_inches='tight')
print("✓ 數據增強示例已保存為 'data_augmentation_demo.png'")

# =============================================================================
# 3. 構建 CNN 模型
# =============================================================================

print("\n步驟 3: 構建 CNN 模型")
print("-" * 70)

def create_cnn_model():
    """
    創建 CNN 模型

    架構:
      - 3 個卷積塊 (Conv2D + BatchNorm + MaxPooling + Dropout)
      - 全連接分類器
    """
    model = keras.Sequential([
        # 數據增強 (僅在訓練時應用)
        data_augmentation,

        # 卷積塊 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # 卷積塊 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 卷積塊 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # 全連接分類器
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name='CIFAR10_CNN')

    return model

model = create_cnn_model()
model.summary()

# 計算參數量
total_params = model.count_params()
print(f"\n總參數量: {total_params:,}")

# =============================================================================
# 4. 編譯模型
# =============================================================================

print("\n步驟 4: 編譯模型")
print("-" * 70)

# 學習率調度
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ 模型編譯完成")
print(f"  - 優化器: Adam (lr=0.001)")
print(f"  - 損失函數: Sparse Categorical Crossentropy")
print(f"  - 評估指標: Accuracy")

# =============================================================================
# 5. 訓練模型
# =============================================================================

print("\n步驟 5: 訓練模型")
print("-" * 70)

# 回調函數
callbacks = [
    # 早停
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    # 模型檢查點
    keras.callbacks.ModelCheckpoint(
        'best_cifar10_cnn.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # 學習率衰減
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # TensorBoard
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# 訓練模型
print("開始訓練...")
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=100,  # 使用早停,可以設置較大的值
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ 訓練完成!")

# =============================================================================
# 6. 評估模型
# =============================================================================

print("\n步驟 6: 評估模型")
print("-" * 70)

# 在測試集上評估
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\n測試集性能:")
print(f"  損失: {test_loss:.4f}")
print(f"  準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 生成預測
print("\n生成預測...")
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# 分類報告
print("\n分類報告:")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names,
    digits=4
))

# =============================================================================
# 7. 可視化結果
# =============================================================================

print("\n步驟 7: 可視化訓練結果")
print("-" * 70)

# 訓練歷史
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 準確率
axes[0].plot(history.history['accuracy'], label='訓練', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='驗證', linewidth=2)
axes[0].set_title('模型準確率', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 損失
axes[1].plot(history.history['loss'], label='訓練', linewidth=2)
axes[1].plot(history.history['val_loss'], label='驗證', linewidth=2)
axes[1].set_title('模型損失', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ 訓練歷史圖表已保存為 'training_history.png'")

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('混淆矩陣', fontsize=16, fontweight='bold')
plt.ylabel('真實標籤')
plt.xlabel('預測標籤')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ 混淆矩陣已保存為 'confusion_matrix.png'")

# =============================================================================
# 8. 預測示例
# =============================================================================

print("\n步驟 8: 預測示例")
print("-" * 70)

# 隨機選擇一些測試樣本
n_samples = 10
indices = np.random.choice(len(x_test), n_samples, replace=False)

plt.figure(figsize=(20, 4))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx])

    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    confidence = y_pred[idx][y_pred_classes[idx]]

    color = 'green' if y_true[idx] == y_pred_classes[idx] else 'red'

    plt.title(
        f"真實: {true_label}\n"
        f"預測: {pred_label}\n"
        f"置信度: {confidence:.2%}",
        color=color,
        fontsize=9
    )
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("✓ 預測結果已保存為 'predictions.png'")

# 打印詳細預測
print("\n詳細預測:")
print("-" * 70)
for i, idx in enumerate(indices):
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    confidence = y_pred[idx][y_pred_classes[idx]]
    status = "✓" if y_true[idx] == y_pred_classes[idx] else "✗"

    print(f"樣本 {i+1}: {status}")
    print(f"  真實標籤: {true_label}")
    print(f"  預測標籤: {pred_label}")
    print(f"  預測置信度: {confidence:.2%}")

    # 顯示 top-3 預測
    top3_idx = np.argsort(y_pred[idx])[-3:][::-1]
    print(f"  Top-3 預測:")
    for j, top_idx in enumerate(top3_idx):
        print(f"    {j+1}. {class_names[top_idx]}: {y_pred[idx][top_idx]:.2%}")
    print()

# =============================================================================
# 9. 錯誤分析
# =============================================================================

print("\n步驟 9: 錯誤分析")
print("-" * 70)

# 找出預測錯誤的樣本
errors = y_pred_classes != y_true
error_indices = np.where(errors)[0]

# 找出最自信的錯誤預測
confidences = np.max(y_pred[errors], axis=1)
most_confident_errors = error_indices[np.argsort(confidences)[-9:][::-1]]

print(f"總錯誤數: {len(error_indices)} / {len(y_test)}")
print(f"錯誤率: {len(error_indices)/len(y_test)*100:.2f}%")

# 可視化最自信的錯誤預測
plt.figure(figsize=(15, 6))
for i, idx in enumerate(most_confident_errors):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx])

    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    confidence = y_pred[idx][y_pred_classes[idx]]

    plt.title(
        f"真實: {true_label}\n"
        f"預測: {pred_label}\n"
        f"置信度: {confidence:.2%}",
        color='red',
        fontsize=9
    )
    plt.axis('off')

plt.suptitle('最自信的錯誤預測', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
print("✓ 錯誤分析圖已保存為 'error_analysis.png'")

# =============================================================================
# 10. 模型保存
# =============================================================================

print("\n步驟 10: 保存模型")
print("-" * 70)

# 保存最終模型
model.save('final_cifar10_cnn.keras')
print("✓ 最終模型已保存為 'final_cifar10_cnn.keras'")

# 保存權重
model.save_weights('model_weights.weights.h5')
print("✓ 模型權重已保存為 'model_weights.weights.h5'")

# =============================================================================
# 總結
# =============================================================================

print("\n" + "=" * 70)
print("✓ 所有步驟完成!")
print("=" * 70)

# 計算訓練統計
total_epochs = len(history.history['loss'])
best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

print(f"""
訓練總結:
  - 訓練樣本數: {len(x_train)}
  - 測試樣本數: {len(x_test)}
  - 總 Epochs: {total_epochs}
  - 最佳驗證準確率: {best_val_acc:.4f} (Epoch {best_epoch})
  - 最終測試準確率: {test_accuracy:.4f}
  - 模型參數量: {total_params:,}

生成的文件:
  - cifar10_samples.png (數據樣本)
  - data_augmentation_demo.png (數據增強示例)
  - training_history.png (訓練歷史)
  - confusion_matrix.png (混淆矩陣)
  - predictions.png (預測結果)
  - error_analysis.png (錯誤分析)
  - final_cifar10_cnn.keras (最終模型)
  - best_cifar10_cnn.keras (最佳模型)
  - logs/ (TensorBoard 日誌)

查看 TensorBoard:
  tensorboard --logdir=logs

下一步建議:
  1. 嘗試更深的網路架構
  2. 實驗不同的數據增強策略
  3. 使用遷移學習 (ResNet, EfficientNet)
  4. 嘗試其他數據集 (CIFAR-100, ImageNet subset)
""")

print("=" * 70)
print("Happy Learning! 🚀")
print("=" * 70)
