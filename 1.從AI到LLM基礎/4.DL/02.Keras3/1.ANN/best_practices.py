"""
Keras 3 ANN 最佳實踐範例
========================

這個腳本展示了訓練神經網路時的各種最佳實踐,包括:
- 數據預處理技巧
- 模型架構設計
- 訓練優化策略
- 過擬合防止
- 模型評估方法

作者: AI Learning Notes
日期: 2025-01
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers, regularizers, callbacks
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# 最佳實踐 1: 設置隨機種子以確保可重現性
# =============================================================================

def set_seeds(seed=42):
    """設置所有相關的隨機種子"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    # TensorFlow 的隨機種子
    import tensorflow as tf
    tf.random.set_seed(seed)

set_seeds(42)
print("✓ 隨機種子已設置,確保結果可重現")

# =============================================================================
# 最佳實踐 2: 使用數據管道和預處理
# =============================================================================

class DataPreprocessor:
    """數據預處理器類"""

    def __init__(self):
        self.mean = None
        self.std = None

    def load_and_preprocess(self, validation_split=0.2):
        """載入並預處理數據"""
        # 載入數據
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 展平圖像
        x_train = x_train.reshape(-1, 28 * 28).astype('float32')
        x_test = x_test.reshape(-1, 28 * 28).astype('float32')

        # 標準化（零均值,單位方差）
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0) + 1e-8  # 避免除以零

        x_train = (x_train - self.mean) / self.std
        x_test = (x_test - self.mean) / self.std

        # 分割驗證集
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

preprocessor = DataPreprocessor()
(x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocessor.load_and_preprocess()

print(f"訓練集大小: {len(x_train)}")
print(f"驗證集大小: {len(x_val)}")
print(f"測試集大小: {len(x_test)}")

# =============================================================================
# 最佳實踐 3: 使用 Functional API 建立靈活的模型
# =============================================================================

def create_model(input_dim=784, num_classes=10, dropout_rate=0.3, l2_reg=0.001):
    """
    使用 Functional API 創建模型

    參數:
        input_dim: 輸入維度
        num_classes: 類別數量
        dropout_rate: Dropout 比率
        l2_reg: L2 正則化強度
    """
    # 定義輸入
    inputs = keras.Input(shape=(input_dim,), name='input')

    # 第一個隱藏層
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense1'
    )(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)

    # 第二個隱藏層
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(dropout_rate, name='dropout2')(x)

    # 第三個隱藏層
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense3'
    )(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout3')(x)  # 較低的 dropout

    # 輸出層
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)

    # 創建模型
    model = keras.Model(inputs=inputs, outputs=outputs, name='OptimizedMNIST')

    return model

model = create_model()
model.summary()

# =============================================================================
# 最佳實踐 4: 使用自定義學習率和優化器配置
# =============================================================================

# 創建自定義優化器
initial_learning_rate = 0.001
optimizer = keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# 編譯模型
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # 使用整數標籤
    metrics=['accuracy']
)

print("✓ 模型已編譯,使用 Adam 優化器")

# =============================================================================
# 最佳實踐 5: 使用完整的 Callbacks 套件
# =============================================================================

# 創建回調函數列表
callbacks_list = [
    # 早停: 驗證損失不再改善時停止訓練
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    # 模型檢查點: 保存最佳模型
    callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # 學習率衰減: 驗證損失停滯時降低學習率
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # TensorBoard: 視覺化訓練過程
    callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True
    ),

    # 自定義回調: 打印訓練進度
    callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"\nEpoch {epoch + 1}: "
            f"loss={logs['loss']:.4f}, "
            f"acc={logs['accuracy']:.4f}, "
            f"val_loss={logs['val_loss']:.4f}, "
            f"val_acc={logs['val_accuracy']:.4f}"
        )
    )
]

# =============================================================================
# 最佳實踐 6: 訓練模型並記錄詳細歷史
# =============================================================================

print("\n開始訓練...")
print("=" * 70)

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=100,  # 使用早停,所以可以設置較大的值
    validation_data=(x_val, y_val),
    callbacks=callbacks_list,
    verbose=0  # 使用自定義回調來控制輸出
)

print("\n✓ 訓練完成!")

# =============================================================================
# 最佳實踐 7: 全面的模型評估
# =============================================================================

print("\n" + "=" * 70)
print("模型評估")
print("=" * 70)

# 在測試集上評估
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n測試集性能:")
print(f"  損失: {test_loss:.4f}")
print(f"  準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 生成預測
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# 分類報告
print("\n分類報告:")
print(classification_report(y_test, y_pred_classes, digits=4))

# =============================================================================
# 最佳實踐 8: 視覺化訓練歷史和結果
# =============================================================================

def plot_training_history(history, save_path='training_metrics.png'):
    """繪製詳細的訓練歷史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 準確率
    axes[0, 0].plot(history.history['accuracy'], label='訓練')
    axes[0, 0].plot(history.history['val_accuracy'], label='驗證')
    axes[0, 0].set_title('模型準確率', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 損失
    axes[0, 1].plot(history.history['loss'], label='訓練')
    axes[0, 1].plot(history.history['val_loss'], label='驗證')
    axes[0, 1].set_title('模型損失', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 學習率變化
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('學習率變化', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # 訓練 vs 驗證準確率差距（過擬合指標）
    train_val_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 1].plot(train_val_gap)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('過擬合指標 (訓練-驗證準確率)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 訓練指標圖已保存: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.title('混淆矩陣', fontsize=16, fontweight='bold')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 混淆矩陣已保存: {save_path}")

# 生成視覺化
plot_training_history(history)
plot_confusion_matrix(y_test, y_pred_classes)

# =============================================================================
# 最佳實踐 9: 錯誤分析
# =============================================================================

def analyze_errors(x_test, y_test, y_pred, top_n=10):
    """分析模型的錯誤預測"""
    # 找出錯誤預測
    y_pred_classes = np.argmax(y_pred, axis=1)
    errors = y_pred_classes != y_test

    # 計算每個錯誤預測的置信度
    error_indices = np.where(errors)[0]
    confidences = np.max(y_pred[errors], axis=1)

    # 按置信度排序（找出最自信的錯誤預測）
    most_confident_errors = error_indices[np.argsort(confidences)[-top_n:][::-1]]

    print("\n" + "=" * 70)
    print(f"最自信的 {top_n} 個錯誤預測:")
    print("=" * 70)

    for i, idx in enumerate(most_confident_errors, 1):
        true_label = y_test[idx]
        pred_label = y_pred_classes[idx]
        confidence = np.max(y_pred[idx])

        print(f"\n錯誤 {i}:")
        print(f"  真實標籤: {true_label}")
        print(f"  預測標籤: {pred_label}")
        print(f"  預測置信度: {confidence:.4f}")

    # 視覺化這些錯誤
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, idx in enumerate(most_confident_errors):
        ax = axes[i // 5, i % 5]
        img = x_test[idx].reshape(28, 28)
        # 反標準化以便視覺化
        img = img * preprocessor.std.reshape(28, 28) + preprocessor.mean.reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"真實:{y_test[idx]}\n預測:{y_pred_classes[idx]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 錯誤分析圖已保存: error_analysis.png")

analyze_errors(x_test, y_test, y_pred)

# =============================================================================
# 最佳實踐 10: 模型總結和建議
# =============================================================================

print("\n" + "=" * 70)
print("訓練總結與建議")
print("=" * 70)

# 計算統計數據
total_epochs = len(history.history['loss'])
best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
overfitting_gap = final_train_acc - final_val_acc

print(f"""
訓練統計:
  - 總 Epochs: {total_epochs}
  - 最佳驗證準確率: {best_val_acc:.4f} (Epoch {best_epoch})
  - 最終訓練準確率: {final_train_acc:.4f}
  - 最終驗證準確率: {final_val_acc:.4f}
  - 測試準確率: {test_accuracy:.4f}
  - 過擬合程度: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)

模型診斷:
""")

# 提供診斷建議
if overfitting_gap > 0.1:
    print("  ⚠️  檢測到明顯過擬合!")
    print("  建議:")
    print("    - 增加 Dropout 比率")
    print("    - 增加 L2 正則化強度")
    print("    - 減少模型複雜度")
    print("    - 增加訓練數據")
elif overfitting_gap < 0.02:
    print("  ✓ 模型泛化良好")
    print("  可以考慮:")
    print("    - 增加模型複雜度以提高性能")
    print("    - 訓練更多 epochs")
else:
    print("  ✓ 模型表現正常")

if test_accuracy < 0.95:
    print("\n  測試準確率可以進一步提高:")
    print("    - 嘗試更深的網路")
    print("    - 使用數據增強")
    print("    - 調整超參數")
else:
    print("\n  ✓ 測試準確率優秀!")

print(f"""
生成的文件:
  - best_model.keras (最佳模型)
  - training_metrics.png (訓練指標)
  - confusion_matrix.png (混淆矩陣)
  - error_analysis.png (錯誤分析)
  - logs/ (TensorBoard 日誌)

查看 TensorBoard:
  tensorboard --logdir=logs
""")

print("=" * 70)
print("✓ 所有最佳實踐演示完成!")
print("=" * 70)
