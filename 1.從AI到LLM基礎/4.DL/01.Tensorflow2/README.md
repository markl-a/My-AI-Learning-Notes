# TensorFlow 2 個人複習筆記

> 🔄 **最後更新：** 2025-11-18
> 📊 **完成度：** 約 25% → 持續更新中
> 🎯 **推薦版本：** TensorFlow 2.18+ (最新穩定版: 2.20.0) - Python 3.9-3.12
> 💻 **測試環境：** TensorFlow 2.20.0, CUDA 12.6, cuDNN 9.x
> 📚 **教程數量：** 6 個完整教程（基礎到進階）
> ⚡ **重要更新：** 支援 NumPy 2.0、改進的 CUDA 支援、Keras 3.x 整合

---

## ⚠️ 內容狀態說明

本筆記規劃了完整的 TensorFlow 2 學習路徑，涵蓋從基礎到進階的實用內容。

**已完成的內容：**
- ✅ **1. TensorFlow 快速入門** (`1.Tensorflow_JumpStart.ipynb`) - 完整的初學者教程，涵蓋基本概念和操作
- ✅ **2. CNN 基礎實作** (`2.CNN/CNN.ipynb`) - 包含傳統 CNN、Dropout 正則化、AutoKeras 和 Optuna 超參數調優
- ✅ **3. RNN 實作** (`3.RNNs/RNN.ipynb`) - 基於文字生成的 RNN 實現，使用 Alice in Wonderland 資料集
- ✅ **4. LSTM & GRU 實作** (`4.LSTM/LSTM&GRU.ipynb`) - 股票價格預測案例，展示 LSTM 和 GRU 的差異與應用
- ✅ **5. tf.data 最佳實踐** (`5.TF_Data_Best_Practices.ipynb`) - 高效資料載入、預處理和性能優化技巧
- ✅ **6. 模型保存與部署** (`6.Model_Saving_and_Deployment.ipynb`) - 完整的模型保存、載入、轉換和部署指南

**最近更新（2025-11-18）：**
- 🆕 **版本更新至 TensorFlow 2.20+** - 支援最新的 TensorFlow 2.20.0 穩定版
- 🆕 **NumPy 2.0 兼容性** - 完整支援 NumPy 2.0 API 和性能改進
- 🆕 **Keras 3.x 整合** - 多後端支援（TensorFlow、JAX、PyTorch）
- 🆕 **LiteRT 遷移提醒** - tf.lite 將逐步遷移至獨立的 LiteRT 項目
- 🆕 **改進的 CUDA 支援** - Hermetic CUDA 實現更好的構建可重現性
- 🔄 更新所有程式碼示例以確保與最新版本兼容

**歷史更新（2025-01-18）：**
- 新增 **tf.data API 最佳實踐**教程 - 涵蓋 prefetch、cache、並行處理等優化技術
- 新增 **模型保存與部署**完整指南 - 包含 TFLite、ONNX、TF Serving 等多種部署方式
- 大幅增強 README - 添加 FAQ、最佳實踐、常見陷阱、性能優化等實用章節
- 添加快速開始指南和推薦學習路徑

其餘內容正在逐步補充中。歡迎先學習已完成的部分，或參考官方 [TensorFlow 教程](https://www.tensorflow.org/tutorials)。

---

## 📂 已完成教程總覽

| 編號 | 標題 | 檔案 | 主要內容 | 難度 |
|------|------|------|----------|------|
| 1 | TensorFlow 快速入門 | `1.Tensorflow_JumpStart.ipynb` | 基本操作、張量、模型建立 | ⭐ 入門 |
| 2 | CNN 基礎實作 | `2.CNN/CNN.ipynb` | 卷積神經網路、Dropout、AutoKeras、Optuna | ⭐⭐ 初級 |
| 3 | RNN 實作 | `3.RNNs/RNN.ipynb` | 循環神經網路、文字生成 | ⭐⭐ 初級 |
| 4 | LSTM & GRU | `4.LSTM/LSTM&GRU.ipynb` | 長短期記憶網路、時間序列預測 | ⭐⭐⭐ 中級 |
| 5 | tf.data 最佳實踐 | `5.TF_Data_Best_Practices.ipynb` | 資料管道、性能優化、並行處理 | ⭐⭐⭐ 中級 |
| 6 | 模型保存與部署 | `6.Model_Saving_and_Deployment.ipynb` | SavedModel、TFLite、部署策略 | ⭐⭐⭐⭐ 進階 |

### 🎯 推薦學習順序

**第一週：** 基礎入門
1. TensorFlow 快速入門 → 2. CNN 基礎實作

**第二週：** 序列模型
3. RNN 實作 → 4. LSTM & GRU

**第三週：** 工程實踐
5. tf.data 最佳實踐 → 6. 模型保存與部署

---

## 📚 完整學習規劃

### ✅ 1. 快速入門
**狀態：已完成**

- ✅ 初學者的 TensorFlow 2.0 教程
- ⏳ 針對專業人員的 TensorFlow 2.0 入門

**檔案位置：** `1.Tensorflow_JumpStart.ipynb`

---

### 🔨 2. 初學者基礎（使用 Keras API）
**狀態：規劃中**

#### 機器學習基礎
- ⏳ 基本圖像分類
- ⏳ 基本文字分類
- ⏳ 使用 TF Hub 的文字分類
- ⏳ 回歸問題
- ⏳ 過擬合與欠擬合
- ⏳ 保存和加載模型
- ⏳ 使用 Keras Tuner 調整超參數

**推薦學習順序：** 圖像分類 → 文字分類 → 回歸 → 模型調優

---

### 📦 3. 資料載入與預處理
**狀態：部分完成**

#### 3.1 圖像資料
- ✅ 使用 `tf.data` 載入圖像
- ✅ 圖像增強技術
- ✅ 批次處理與預取

#### 3.2 文字資料
- ✅ Unicode 處理
- ✅ 子詞標記化（Subword Tokenization）
- ✅ TextVectorization 層

#### 3.3 其他資料格式
- ✅ CSV 文件處理
- ✅ NumPy 陣列轉換
- ✅ pandas.DataFrame 整合
- ⏳ TFRecord 和 tf.Example
- ⏳ 影片資料處理

**檔案位置：** `5.TF_Data_Best_Practices.ipynb`

**重要技能：** `tf.data.Dataset` API 是 TensorFlow 高效訓練的核心

**已涵蓋內容：**
- 完整的 tf.data API 使用教程
- 性能優化技巧（prefetch, cache, parallel map）
- 不同資料源的處理方法
- 實戰範例和基準測試

---

### 🚀 4. 進階技術
**狀態：規劃中**

#### 4.1 自定義開發
- ⏳ 張量操作基礎
- ⏳ 自定義層開發
- ⏳ 自定義訓練迴圈
- ⏳ 自定義損失函式與指標

#### 4.2 分散式訓練
- ⏳ 使用 Keras 的分散式訓練
- ⏳ `tf.distribute.Strategy` 介紹
- ⏳ 多 GPU 訓練
- ⏳ 多工作者訓練（Multi-worker Training）
- ⏳ 參數伺服器架構
- ⏳ DTensors 使用

**適用場景：** 大規模資料集、大型模型訓練

---

### 🎨 5. 計算機視覺
**狀態：部分完成**

- ✅ CNN 基礎（已完成）
- ⏳ KerasCV 庫介紹
- ⏳ 圖像分類實戰
- ⏳ 遷移學習與微調
- ⏳ 使用 TF Hub 的預訓練模型
- ⏳ 資料增強策略
- ⏳ 圖像分割（Semantic Segmentation）
- ⏳ 目標檢測
- ⏳ 影片分類
- ⏳ MoViNet 遷移學習

**檔案位置：** `2.CNN/CNN.ipynb`

**推薦專案：**
- 貓狗分類
- 人臉識別
- 風格遷移

---

### 💬 6. 自然語言處理（NLP）
**狀態：部分完成**

- ✅ RNN 基礎（已完成）
- ✅ LSTM & GRU（已完成）
- ⏳ KerasNLP 庫介紹
- ⏳ 文字分類
- ⏳ 情感分析
- ⏳ 命名實體識別（NER）
- ⏳ 機器翻譯
- ⏳ 文字生成

**檔案位置：**
- `3.RNNs/RNN.ipynb`
- `4.LSTM/LSTM&GRU.ipynb`

**現代替代方案：** 對於 2024+ 的 NLP 任務，建議優先考慮 Transformer 架構（參見 `05.Transformer_lib`）

---

### 🎵 7. 音頻處理
**狀態：規劃中**

- ⏳ 簡單的音頻識別
- ⏳ 音頻特徵提取
- ⏳ 音頻分類遷移學習
- ⏳ 使用 RNN 生成音樂

---

### 📊 8. 結構化資料
**狀態：規劃中**

- ⏳ 使用預處理層分類結構化資料
- ⏳ 處理不平衡資料集
- ⏳ 時間序列預測
- ⏳ TensorFlow Decision Forests
- ⏳ 推薦系統基礎

**應用場景：** 商業分析、金融預測、使用者行為預測

---

### 🎭 9. 生成式模型
**狀態：規劃中**

#### 9.1 GAN 系列
- ⏳ DCGAN（深度卷積 GAN）
- ⏳ Pix2Pix（圖像到圖像轉換）
- ⏳ CycleGAN（無配對圖像轉換）

#### 9.2 自編碼器
- ⏳ 自編碼器入門
- ⏳ 變分自編碼器（VAE）
- ⏳ 有損資料壓縮

#### 9.3 現代生成模型
- ⏳ Stable Diffusion 基礎
- ⏳ 神經風格遷移
- ⏳ Deep Dream

#### 9.4 對抗性訓練
- ⏳ FGSM 攻擊
- ⏳ 模型魯棒性

**熱門應用：** 圖像生成、藝術創作、資料增強

---

### ⚡ 10. 模型優化與部署
**狀態：已完成**

#### 10.1 模型保存與載入
- ✅ SavedModel 格式（推薦）
- ✅ HDF5 格式
- ✅ Checkpoint 格式
- ✅ ModelCheckpoint Callback

#### 10.2 模型優化
- ✅ TensorFlow Model Optimization Toolkit
- ✅ 模型剪枝（Pruning）
- ✅ 量化（Quantization）
- ✅ 知識蒸餾

#### 10.3 模型部署
- ✅ TensorFlow Serving（伺服器端）
- ✅ TensorFlow Lite（移動端/嵌入式）
- ✅ TensorFlow.js（瀏覽器/Node.js）
- ✅ ONNX 轉換
- ✅ Android/iOS 部署示例

**檔案位置：** `6.Model_Saving_and_Deployment.ipynb`

**已涵蓋內容：**
- 完整的模型保存和載入流程
- 多種格式轉換（TFLite、ONNX、TF.js）
- 模型優化技術實戰
- 不同平台的部署策略
- 版本管理與實驗追蹤（MLflow）

**目標：** 將模型部署到生產環境

---

### 🔍 11. 模型可解釋性
**狀態：規劃中**

- ⏳ 整合梯度（Integrated Gradients）
- ⏳ Grad-CAM 視覺化
- ⏳ SHAP 值分析
- ⏳ 使用 SNGP 的不確定性量化
- ⏳ 機率性回歸

**重要性：** 理解模型決策，建立信任

---

### 🎮 12. 強化學習
**狀態：規劃中**

- ⏳ Actor-Critic 方法
- ⏳ TensorFlow Agents 庫
- ⏳ Q-Learning 基礎
- ⏳ 深度 Q 網路（DQN）
- ⏳ 策略梯度方法

---

## 🆕 TensorFlow 2.18-2.20 新特性亮點（2024-2025）

### TensorFlow 2.20 (2025年8月)
- **LiteRT 獨立化：** `tf.lite` 模組將遷移至獨立的 LiteRT 專案，專注於設備端推理
- **Keras 3.x 完全整合：** 多後端架構支援（TensorFlow、JAX、PyTorch）
- **性能優化：** 改進的記憶體管理和執行效率
- **API 清理：** 棄用舊版 API，簡化開發體驗

### TensorFlow 2.19 (2025年3月)
- **LiteRT C++ API 改進：** 更好的設備端部署支援
- **bfloat16 支援增強：** TFLite 型別轉換支援 bfloat16
- **效能提升：** 編譯時優化和運行時性能改進

### TensorFlow 2.18 (2024年10月) - 重要里程碑
- **✅ NumPy 2.0 支援：** 預設編譯支援 NumPy 2.0，帶來顯著性能提升
- **🔧 Hermetic CUDA：** 更好的構建可重現性，避免本地 CUDA 版本衝突
- **📦 Keras 增強：**
  ```python
  # 新增 Pipeline 層，用於構建預處理管道
  from tensorflow import keras

  preprocessing_pipeline = keras.layers.Pipeline([
      keras.layers.Rescaling(1./255),
      keras.layers.RandomFlip("horizontal"),
      keras.layers.RandomRotation(0.2)
  ])
  ```
- **🚀 TFLite 改進：** SignatureRunner 支援無簽名的模型
- **⚠️ 重大變更：** 為了程式碼健康，CUDA 構建中禁用了 TensorRT 支援

### 升級建議

```python
# 檢查當前版本
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

# 升級到最新版本
# pip install --upgrade tensorflow>=2.18.0

# 檢查新特性兼容性
if hasattr(tf.keras.layers, 'Pipeline'):
    print("✅ 支援 Keras Pipeline 層")
else:
    print("⚠️ 請升級到 TensorFlow 2.18+")
```

---

## 🛠️ 現代 TensorFlow 2 最佳實踐（2024-2025）

### 推薦工作流程
```python
# 1. 使用 tf.data 進行高效資料載入
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 2. 使用 Keras Functional API 或 Model Subclassing
model = tf.keras.Sequential([...])

# 3. 使用 Mixed Precision 加速訓練
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 4. 使用 Callbacks 監控訓練
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(...),
    tf.keras.callbacks.EarlyStopping(...),
    tf.keras.callbacks.TensorBoard(...)
]

# 5. 編譯並訓練
model.compile(optimizer='adam', loss='...', metrics=['...'])
model.fit(dataset, epochs=epochs, callbacks=callbacks)
```

### 效能優化技巧
- ✅ 使用 `tf.data.AUTOTUNE` 自動調整參數
- ✅ 啟用 XLA 編譯（`jit_compile=True`）
- ✅ 使用混合精度訓練
- ✅ 預取（Prefetch）與快取（Cache）資料
- ✅ 使用 `tf.function` 進行圖執行

---

## 📚 學習資源

### 官方資源
- [TensorFlow 官方教程](https://www.tensorflow.org/tutorials)
- [TensorFlow 官方文檔](https://www.tensorflow.org/api_docs/python/tf)
- [TensorFlow Hub](https://tfhub.dev/)
- [Keras 官方指南](https://keras.io/guides/)

### 進階資源
- [TensorFlow Model Garden](https://github.com/tensorflow/models)
- [TensorFlow Examples](https://github.com/tensorflow/examples)
- [TensorFlow Blog](https://blog.tensorflow.org/)

### 社群資源
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Keras Code Examples](https://keras.io/examples/)

---

## 🎯 學習建議

### 對於初學者：
1. 從快速入門教程開始
2. 練習基本的圖像分類和文字分類
3. 了解 `tf.data` 和 Keras API
4. 完成至少 2-3 個小專案

### 對於中級學習者：
1. 深入理解自定義訓練迴圈
2. 學習模型優化技術
3. 實作遷移學習專案
4. 探索 TensorFlow Hub 預訓練模型

### 對於進階學習者：
1. 掌握分散式訓練
2. 研究模型部署策略
3. 實作生成式模型
4. 貢獻開源專案

---

## 🔗 相關章節

- **Keras 3 筆記：** 查看 `../02.Keras3/`
- **PyTorch 筆記：** 查看 `../03.Pytorch/`
- **深度學習基礎：** 查看 `../00.DL_Path/`

---

## 📝 注意事項

> **注意：`tensorflow-gpu` 套件自 TF 2.0 起已合併到 `tensorflow`，2023 起 PyPI 已下架，請直接 `pip install tensorflow`。**

1. **版本兼容性：** 建議使用 TensorFlow 2.15 或更高版本
2. **GPU 支援：** 安裝統一的 `tensorflow` 套件（2.0+ 起 CPU/GPU 已合併，不再使用 `tensorflow-gpu`）
3. **環境配置：** 推薦使用 Conda 或 venv 建立虛擬環境
4. **雲端訓練：** 可使用 Google Colab 或 Kaggle Notebooks 免費 GPU

---

## 🚀 快速開始指南

### 環境安裝

```bash
# 使用 pip 安裝最新版 TensorFlow (推薦)
pip install tensorflow>=2.18.0

# 或安裝最新穩定版
pip install tensorflow==2.20.0

# 驗證安裝
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Keras version:', tf.keras.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')))"

# 檢查 NumPy 版本（建議 NumPy 2.0+）
python -c "import numpy as np; print('NumPy version:', np.__version__)"

# 可選：安裝額外套件
pip install tensorflow-datasets tensorflow-hub keras-tuner

# 如果需要 GPU 支援，確保安裝匹配的 CUDA 版本
# TensorFlow 2.18+ 需要 CUDA 12.3+ 和 cuDNN 8.9+
```

### 重要版本兼容性說明（2025）

| TensorFlow 版本 | Python 版本 | CUDA 版本 | cuDNN 版本 | NumPy 版本 |
|----------------|------------|----------|-----------|-----------|
| 2.20.0 | 3.9-3.12 | 12.3+ | 9.x | 1.26+ / 2.0+ |
| 2.19.0 | 3.9-3.12 | 12.3+ | 8.9+ | 1.26+ / 2.0+ |
| 2.18.0 | 3.9-3.12 | 12.3+ | 8.9+ | 1.26+ / 2.0+ |
| 2.15.0 | 3.9-3.11 | 12.2+ | 8.9+ | 1.23-1.26 |

**注意事項：**
- TensorFlow 2.18+ 預設編譯支援 NumPy 2.0，獲得更好的性能
- 使用 Hermetic CUDA 可以避免本地 CUDA 版本衝突
- Keras 3.x 提供多後端支援，但預設使用 TensorFlow 後端

### 第一個 TensorFlow 程式

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 建立簡單的神經網路
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 載入 MNIST 資料集
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 訓練模型
model.fit(x_train.reshape(-1, 784), y_train, epochs=5, validation_split=0.2)

# 評估模型
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 784), y_test)
print(f'Test accuracy: {test_acc:.4f}')
```

---

## ❓ 常見問題 (FAQ)

### Q1: TensorFlow 2.x 和 1.x 有什麼主要差異？

**A:** 主要差異包括：
- **預設 Eager Execution：** TensorFlow 2.x 預設啟用 eager execution，程式碼更直觀易懂
- **Keras 整合：** Keras 成為官方高階 API (`tf.keras`)
- **移除 Session：** 不再需要 `tf.Session()`
- **簡化 API：** 移除重複的 API，統一命名規範
- **更好的錯誤訊息：** 更清晰的錯誤提示

### Q2: 如何選擇 CNN、RNN 還是 Transformer？

**A:** 選擇建議：
- **CNN：** 圖像處理、局部特徵提取、空間資料
- **RNN/LSTM/GRU：** 短序列、時間序列預測、需要記憶的任務
- **Transformer：** 長序列、NLP 任務、需要並行處理的場景（推薦用於現代 NLP）

### Q3: 訓練時 GPU 記憶體不足怎麼辦？

**A:** 解決方案：
```python
# 1. 減少批次大小
model.fit(x_train, y_train, batch_size=32)  # 降低 batch_size

# 2. 使用混合精度訓練
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# 3. 限制 GPU 記憶體增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 4. 使用梯度累積
# 5. 考慮模型量化或剪枝
```

### Q4: 如何避免過擬合？

**A:** 常用技術：
1. **Dropout：** `keras.layers.Dropout(0.5)`
2. **L2 正則化：** `keras.regularizers.l2(0.01)`
3. **Early Stopping：** 監控驗證損失提前停止
4. **資料增強：** 增加訓練資料多樣性
5. **Batch Normalization：** 穩定訓練過程
6. **減少模型複雜度：** 減少層數或神經元數量

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train,
          validation_split=0.2,
          callbacks=[early_stop])
```

### Q5: 如何保存和載入模型？

**A:** 推薦使用 SavedModel 格式：
```python
# 保存整個模型
model.save('my_model')  # SavedModel 格式（推薦）
model.save('my_model.h5')  # HDF5 格式

# 載入模型
loaded_model = keras.models.load_model('my_model')

# 只保存權重
model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')
```

---

## ⚠️ 常見陷阱與最佳實踐

### 陷阱 1: 忘記正規化資料
```python
# ❌ 錯誤：未正規化
model.fit(x_train, y_train)

# ✅ 正確：正規化到 [0, 1]
x_train = x_train / 255.0
model.fit(x_train, y_train)
```

### 陷阱 2: 訓練/測試資料洩漏
```python
# ❌ 錯誤：在分割前正規化
scaler.fit(X)  # 使用全部資料
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ 正確：只用訓練集擬合
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # 只使用訓練資料
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### 陷阱 3: 類別不平衡未處理
```python
# ✅ 使用 class_weight
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

model.fit(x_train, y_train, class_weight=class_weight_dict)
```

### 陷阱 4: 忘記設置隨機種子
```python
# ✅ 確保可重現性
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)
```

---

## 🔧 效能優化技巧

### 1. 使用 tf.data API 進行高效資料載入

```python
# 建立高效的資料管道
AUTOTUNE = tf.data.AUTOTUNE

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = (dataset
    .shuffle(buffer_size=10000)
    .batch(batch_size=32)
    .prefetch(buffer_size=AUTOTUNE)  # 預取資料
    .cache()  # 快取資料
)

model.fit(dataset, epochs=10)
```

### 2. 啟用 XLA 編譯加速

```python
# 使用 XLA 編譯
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=True  # 啟用 XLA
)
```

### 3. 使用 Mixed Precision 訓練

```python
from tensorflow.keras import mixed_precision

# 啟用混合精度
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 確保最後一層使用 float32
outputs = Dense(10, activation='softmax', dtype='float32')(x)
```

### 4. 使用 @tf.function 裝飾器

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

---

## 📊 模型評估與調試

### TensorBoard 使用

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 設置 TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    callbacks=[tensorboard_callback]
)

# 啟動 TensorBoard
# 在終端執行: tensorboard --logdir logs/fit
```

### 模型性能分析

```python
# 使用 profiler
from tensorflow.python.profiler import profiler_v2 as profiler

# 開始 profiling
profiler.start('logdir')

# 訓練幾個步驟
model.fit(x_train, y_train, epochs=1, steps_per_epoch=10)

# 停止 profiling
profiler.stop()
```

---

## 🎓 推薦學習路徑

### 初學者路徑（1-2 個月）
1. ✅ 完成「TensorFlow 快速入門」
2. ✅ 實作基本圖像分類（MNIST, CIFAR-10）
3. ✅ 學習「CNN 基礎實作」
4. 🎯 完成 2-3 個小專案（建議：貓狗分類、手寫數字識別）
5. 📚 學習 tf.data 和資料預處理

### 中級路徑（2-3 個月）
1. ✅ 深入學習「RNN」、「LSTM & GRU」
2. 🎯 實作時間序列預測專案
3. 📚 學習遷移學習和預訓練模型
4. 🎯 使用 TensorFlow Hub 實作專案
5. 📚 學習模型調優（Keras Tuner, Optuna）

### 進階路徑（3-6 個月）
1. 📚 自定義訓練迴圈和損失函式
2. 📚 分散式訓練策略
3. 📚 模型優化（剪枝、量化）
4. 📚 模型部署（TF Serving, TFLite, TF.js）
5. 🎯 完成端到端專案（訓練到部署）

---

## 🔗 進階學習資源

### 官方資源
- [TensorFlow 官方教程](https://www.tensorflow.org/tutorials) - 最權威的學習資源
- [TensorFlow API 文檔](https://www.tensorflow.org/api_docs/python/tf)
- [Keras 官方指南](https://keras.io/guides/) - 深入理解 Keras API
- [TensorFlow Hub](https://tfhub.dev/) - 預訓練模型庫

### 實用工具
- [TensorFlow Model Garden](https://github.com/tensorflow/models) - 官方模型實現
- [TensorFlow Datasets](https://www.tensorflow.org/datasets) - 常用資料集
- [Keras Code Examples](https://keras.io/examples/) - 高品質範例程式碼
- [TensorFlow.js](https://www.tensorflow.org/js) - 瀏覽器端 ML

### 社群與支援
- [TensorFlow Forum](https://discuss.tensorflow.org/) - 官方論壇
- [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow) - 問題解答
- [TensorFlow Blog](https://blog.tensorflow.org/) - 最新動態和技術文章

### 推薦書籍
- **《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》** by Aurélien Géron
- **《Deep Learning with Python》** by François Chollet (Keras 創始人)
- **《TensorFlow 2.0 深度學習實戰》**

---

## 💡 實用技巧集錦

### 調試模型不收斂

```python
# 1. 檢查資料是否正確
print("資料範圍:", x_train.min(), x_train.max())
print("標籤分布:", np.bincount(y_train))

# 2. 從簡單模型開始
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# 3. 使用較大的學習率
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# 4. 監控梯度
from tensorflow.keras.callbacks import Callback

class GradientCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            for layer in self.model.layers:
                weights = layer.get_weights()
                if weights:
                    print(f"{layer.name} weight mean: {np.mean(np.abs(weights[0]))}")
```

### 處理不同資料類型

```python
# 圖像資料
image_dataset = tf.data.Dataset.from_tensor_slices(images)
image_dataset = image_dataset.map(
    lambda x: tf.image.resize(x, [224, 224])
)

# 文字資料
from tensorflow.keras.layers import TextVectorization

vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)
vectorize_layer.adapt(text_dataset)

# 時間序列資料
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
```

---

**持續更新中...** 📖

> 💬 **回饋與建議：** 如果你在學習過程中有任何問題或建議，歡迎提出 Issue！
