# TensorFlow 2 個人複習筆記

> 🔄 **最後更新：** 2025-01
> 📊 **完成度：** 約 5%
> 🎯 **推薦版本：** TensorFlow 2.15+

---

## ⚠️ 內容狀態說明

本筆記規劃了完整的 TensorFlow 2 學習路徑，但**目前僅完成少量基礎內容**。

**已完成的內容：**
- ✅ 1. TensorFlow 快速入門
- ✅ 2. CNN 基礎實作
- ✅ 3. RNN 實作
- ✅ 4. LSTM & GRU 實作

其餘內容正在逐步補充中。歡迎先學習已完成的部分，或參考官方 [TensorFlow 教程](https://www.tensorflow.org/tutorials)。

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
- ⏳ 基本文本分類
- ⏳ 使用 TF Hub 的文本分類
- ⏳ 回歸問題
- ⏳ 過擬合與欠擬合
- ⏳ 保存和加載模型
- ⏳ 使用 Keras Tuner 調整超參數

**推薦學習順序：** 圖像分類 → 文本分類 → 回歸 → 模型調優

---

### 📦 3. 資料載入與預處理
**狀態：規劃中**

#### 3.1 圖像資料
- ⏳ 使用 `tf.data` 載入圖像
- ⏳ 圖像增強技術
- ⏳ 批次處理與預取

#### 3.2 文本資料
- ⏳ Unicode 處理
- ⏳ 子詞標記化（Subword Tokenization）
- ⏳ TextVectorization 層

#### 3.3 其他資料格式
- ⏳ CSV 文件處理
- ⏳ NumPy 數組轉換
- ⏳ pandas.DataFrame 整合
- ⏳ TFRecord 和 tf.Example
- ⏳ 影片資料處理

**重要技能：** `tf.data.Dataset` API 是 TensorFlow 高效訓練的核心

---

### 🚀 4. 進階技術
**狀態：規劃中**

#### 4.1 自定義開發
- ⏳ 張量操作基礎
- ⏳ 自定義層開發
- ⏳ 自定義訓練迴圈
- ⏳ 自定義損失函數與指標

#### 4.2 分散式訓練
- ⏳ 使用 Keras 的分散式訓練
- ⏳ `tf.distribute.Strategy` 介紹
- ⏳ 多 GPU 訓練
- ⏳ 多工作者訓練（Multi-worker Training）
- ⏳ 參數服務器架構
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
- ⏳ 文本分類
- ⏳ 情感分析
- ⏳ 命名實體識別（NER）
- ⏳ 機器翻譯
- ⏳ 文本生成

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
**狀態：規劃中**

#### 10.1 模型優化
- ⏳ TensorFlow Model Optimization Toolkit
- ⏳ 模型剪枝（Pruning）
- ⏳ 量化（Quantization）
- ⏳ 知識蒸餾

#### 10.2 模型部署
- ⏳ TensorFlow Serving
- ⏳ TensorFlow Lite（移動端/嵌入式）
- ⏳ TensorFlow.js（瀏覽器/Node.js）
- ⏳ ONNX 轉換

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
2. 練習基本的圖像分類和文本分類
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

1. **版本兼容性：** 建議使用 TensorFlow 2.15 或更高版本
2. **GPU 支援：** 安裝 `tensorflow-gpu` 或使用統一的 `tensorflow` 套件（2.11+）
3. **環境配置：** 推薦使用 Conda 或 venv 建立虛擬環境
4. **雲端訓練：** 可使用 Google Colab 或 Kaggle Notebooks 免費 GPU

---

**持續更新中...** 📖
