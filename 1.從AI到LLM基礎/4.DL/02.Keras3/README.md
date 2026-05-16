# Keras 3 個人複習筆記

> 🔄 **最後更新：** 2025-01-18
> 📊 **完成度：** 約 35%
> 🎯 **推薦版本：** Keras 3.0+
> 🚀 **特色：** 多後端支援（TensorFlow、JAX、PyTorch）

---

## ⚠️ 內容狀態說明

Keras 3 是 Keras 的重大升級，支援多個深度學習後端（TensorFlow、JAX、PyTorch），讓你可以用同一套 API 在不同框架間切換。

**已完成的內容：**
- ✅ 1. ANN 基礎實作
- ✅ 2. CNN 卷積神經網路 (全新增強版)

**規劃中的內容正在逐步補充。**

---

## 🌟 Keras 3 的革命性特性

### 🔄 多後端架構
```python
# 在環境變數中切換後端
export KERAS_BACKEND="tensorflow"  # 或 "jax" 或 "torch"

import keras
print(keras.backend.backend())  # 查看當前後端
```

### ✨ 主要優勢
- ✅ **統一的 API：** 一次學習，處處使用
- ✅ **效能最佳化：** 根據任務選擇最佳後端
- ✅ **完全向後兼容：** 無縫遷移 Keras 2 程式碼
- ✅ **分散式訓練：** 內建多 GPU/TPU 支援
- ✅ **生產部署：** 易於導出和部署

---

## 📚 完整學習規劃

### ✅ 1. 基礎神經網路（ANN）
**狀態：已完成並增強**

#### 核心內容
- ✅ 人工神經網路基礎理論
- ✅ Sequential API 使用
- ✅ Functional API 介紹
- ✅ 基本分類問題實作
- ✅ 模型訓練與評估
- ✅ 資料預處理最佳實踐
- ✅ 防止過擬合技巧
- ✅ 模型優化策略
- ✅ 錯誤分析方法

#### 檔案列表
- 📖 `1.ANN/README.md` - 完整的 ANN 學習指南（理論+實踐）
- 📓 `1.ANN/ANN.ipynb` - Jupyter Notebook 綜合範例
- 🚀 `1.ANN/quick_start.py` - 快速入門腳本（適合初學者）
- 💎 `1.ANN/best_practices.py` - 最佳實踐完整範例
- 📊 `1.ANN/heart.csv` - 心臟病預測資料集

#### 學習建議
1. 先閱讀 `README.md` 了解理論基礎
2. 運行 `quick_start.py` 快速上手
3. 深入研究 `ANN.ipynb` 中的各種範例
4. 學習 `best_practices.py` 中的進階技巧

---
---

### 🎨 2. 卷積神經網路（CNN）
**狀態：已完成並增強** ✅

#### 核心內容
- ✅ CNN 基礎理論與原理
- ✅ 卷積層、池化層、批次正規化
- ✅ CNN 架構設計模式
- ✅ CIFAR-10 圖像分類實戰
- ✅ 資料增強技術詳解
- ✅ 遷移學習完整教程
- ✅ 預訓練模型使用指南
- ✅ 模型可視化與解釋

#### 檔案列表
- 📖 `2.CNN/README.md` - 完整的 CNN 學習指南（超過 800 行）
- 🚀 `2.CNN/cnn_image_classification.py` - CNN 圖像分類完整範例
- 🔄 `2.CNN/transfer_learning.py` - 遷移學習實戰教程

#### 學習路徑
1. 閱讀 `README.md` 理解 CNN 原理
2. 運行 `cnn_image_classification.py` 進行 CIFAR-10 分類
3. 學習 `transfer_learning.py` 掌握遷移學習
4. 實作自己的圖像分類項目

#### 涵蓋主題
- **基礎**: 卷積運算、特徵圖、感受野
- **架構**: VGG、ResNet、EfficientNet 風格
- **技術**: 資料增強、Dropout、BatchNorm
- **應用**: 圖像分類、特徵提取、微調
- **可視化**: 特徵圖、Grad-CAM、混淆矩陣

#### 進階主題（規劃中）
- ⏳ 目標檢測 (YOLO, RetinaNet)
- ⏳ 圖像分割 (U-Net, Mask R-CNN)
- ⏳ Vision Transformer
- ⏳ 實時應用部署

**推薦練習：**
- CIFAR-10 挑戰（目標：85%+ 準確率）
- Fashion MNIST 分類
- 貓狗分類器（使用遷移學習）
- 自定義資料集應用


---

### 💬 3. 自然語言處理（NLP）
**狀態：規劃中**

#### 3.1 文字處理基礎
- ⏳ TextVectorization 層
- ⏳ Embedding 層使用
- ⏳ 序列處理技術

#### 3.2 文字分類
- ⏳ 情感分析
- ⏳ 新聞分類
- ⏳ 垃圾郵件檢測

#### 3.3 序列到序列模型
- ⏳ 機器翻譯
- ⏳ 文字摘要
- ⏳ 問答系統

#### 3.4 Transformer 架構
- ⏳ 自注意力機制
- ⏳ BERT 風格模型
- ⏳ GPT 風格模型

**推薦專案：**
- IMDB 電影評論情感分析
- 新聞分類器
- 簡易聊天機器人

---

### 📊 4. 結構化資料
**狀態：規劃中**

#### 4.1 表格資料處理
- ⏳ 特徵工程
- ⏳ 類別編碼
- ⏳ 數值正規化
- ⏳ 缺失值處理

#### 4.2 分類任務
- ⏳ 二元分類
- ⏳ 多類別分類
- ⏳ 不平衡資料處理

#### 4.3 回歸任務
- ⏳ 房價預測
- ⏳ 銷售預測
- ⏳ 需求預測

**應用場景：**
- 信用評分
- 客戶流失預測
- 風險評估

---

### 📈 5. 時間序列分析
**狀態：規劃中**

#### 5.1 時間序列基礎
- ⏳ 資料預處理
- ⏳ 滑動窗口技術
- ⏳ 序列正規化

#### 5.2 預測模型
- ⏳ LSTM 時間序列預測
- ⏳ GRU 模型
- ⏳ 1D CNN 用於時間序列
- ⏳ Transformer 時間序列

#### 5.3 實際應用
- ⏳ 股票價格預測
- ⏳ 天氣預測
- ⏳ 能源消耗預測
- ⏳ 異常檢測

**推薦資料集：**
- 時間序列分類檔案（TSC Archive）
- Kaggle 時間序列競賽

---

### 🎭 6. 生成式深度學習
**狀態：規劃中**

#### 6.1 生成對抗網路（GAN）
- ⏳ 基礎 GAN 實作
- ⏳ DCGAN（深度卷積 GAN）
- ⏳ StyleGAN 基礎
- ⏳ 條件 GAN（cGAN）

#### 6.2 變分自編碼器（VAE）
- ⏳ VAE 原理與實作
- ⏳ 潛在空間探索
- ⏳ 圖像生成應用

#### 6.3 擴散模型
- ⏳ DDPM 基礎
- ⏳ Stable Diffusion 整合
- ⏳ 圖像生成與編輯

#### 6.4 其他生成模型
- ⏳ 自編碼器
- ⏳ 神經風格遷移
- ⏳ 圖像修復

**創意應用：**
- AI 藝術生成
- 圖像增強
- 虛擬人物生成

---

### 🎵 7. 音頻資料處理
**狀態：規劃中**

#### 7.1 音頻基礎
- ⏳ 音頻載入與預處理
- ⏳ 特徵提取（MFCC, Mel-spectrogram）
- ⏳ 音頻增強技術

#### 7.2 音頻分類
- ⏳ 語音情感識別
- ⏳ 音樂類型分類
- ⏳ 環境聲音分類

#### 7.3 語音處理
- ⏳ 語音識別基礎
- ⏳ 說話者辨識
- ⏳ 語音合成（TTS）

**應用範例：**
- 音樂推薦系統
- 語音助手
- 聲音事件檢測

---

### 🎮 8. 強化學習
**狀態：規劃中**

#### 8.1 強化學習基礎
- ⏳ Q-Learning 原理
- ⏳ 深度 Q 網路（DQN）
- ⏳ 策略梯度方法

#### 8.2 進階演算法
- ⏳ Actor-Critic
- ⏳ PPO（Proximal Policy Optimization）
- ⏳ A3C（Asynchronous Actor-Critic）

#### 8.3 環境與應用
- ⏳ OpenAI Gym 整合
- ⏳ 遊戲 AI
- ⏳ 機器人控制

---

### 🖼️ 9. 進階圖像處理
**狀態：規劃中**

- ⏳ 圖像超解析度
- ⏳ 影像去噪
- ⏳ 圖像著色
- ⏳ 3D 視覺
- ⏳ 光流估計

---

### 🔧 10. 模型優化與部署
**狀態：規劃中**

#### 10.1 模型優化
- ⏳ 量化（Quantization）
- ⏳ 剪枝（Pruning）
- ⏳ 知識蒸餾
- ⏳ 混合精度訓練

#### 10.2 部署方案
- ⏳ TensorFlow Serving
- ⏳ ONNX 導出
- ⏳ TensorFlow Lite（移動端）
- ⏳ TensorFlow.js（網頁）

---

### 📦 11. 其他進階主題
**狀態：規劃中**

- ⏳ 自定義層與模型
- ⏳ 自定義訓練迴圈
- ⏳ 混合專家模型（MoE）
- ⏳ 圖神經網路（GNN）
- ⏳ 自監督學習
- ⏳ 對比學習
- ⏳ Few-Shot Learning
- ⏳ Meta-Learning

---

## 🛠️ Keras 3 最佳實踐（2024-2025）

### 推薦工作流程

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'  # 選擇後端

import keras
from keras import layers

# 1. 建立模型（Functional API）
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# 2. 編譯模型
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. 訓練
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.keras')
    ]
)

# 4. 評估與預測
test_loss, test_acc = model.evaluate(x_test, y_test)
predictions = model.predict(x_new)
```

### 多後端性能比較

| 後端 | 優勢 | 適用場景 |
|------|------|----------|
| **TensorFlow** | 成熟穩定、部署便利 | 生產環境、移動端 |
| **JAX** | 速度最快、自動微分強大 | 研究、大規模訓練 |
| **PyTorch** | 動態圖、調試方便 | 研究、原型開發 |

### 效能優化技巧
- ✅ 使用 `keras.mixed_precision` 進行混合精度訓練
- ✅ 利用 `keras.utils.to_categorical` 進行資料預處理
- ✅ 使用 `model.save('model.keras')` 儲存完整模型
- ✅ 啟用資料預取和快取
- ✅ 使用適當的批次大小

---

## 📚 學習資源

### 官方資源
- [Keras 3 官方文檔](https://keras.io/)
- [Keras 3 程式碼範例](https://keras.io/examples/)
- [Keras 3 API 參考](https://keras.io/api/)
- [Keras 3 遷移指南](https://keras.io/guides/migrating_to_keras_3/)

### 社群資源
- [Keras GitHub](https://github.com/keras-team/keras)
- [Keras 論壇](https://github.com/keras-team/keras/discussions)
- [Keras on X (Twitter)](https://twitter.com/keras_io)

### 推薦書籍
- *Deep Learning with Python* by François Chollet（Keras 創始人）
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*

---

## 🎯 學習建議

### 對於初學者：
1. 從 ANN 基礎開始（已完成）
2. 練習圖像分類任務（使用 MNIST, CIFAR-10）
3. 嘗試文字分類（IMDB 評論）
4. 完成 2-3 個端到端專案

### 對於中級學習者：
1. 深入理解 Functional API 和 Model Subclassing
2. 實作遷移學習專案
3. 探索不同的後端性能
4. 學習自定義層和訓練迴圈

### 對於進階學習者：
1. 研究生成式模型（GAN、VAE、Diffusion）
2. 實作 SOTA 模型架構
3. 優化模型並部署到生產環境
4. 參與 Keras 開源貢獻

---

## 🔗 相關章節

- **TensorFlow 2 筆記：** 查看 `../01.Tensorflow2/`
- **PyTorch 筆記：** 查看 `../03.Pytorch/`
- **深度學習基礎：** 查看 `../00.DL_Path/`

---

## 📝 Keras 3 遷移提示

如果你之前使用 Keras 2：
```python
# Keras 2
from tensorflow import keras

# Keras 3（獨立套件）
import keras
# 或指定後端
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
```

主要差異：
- ✅ Keras 3 是獨立套件（不再是 `tensorflow.keras`）
- ✅ 支援多後端
- ✅ 更好的效能
- ✅ 新的儲存格式（`.keras`）

---

## 💡 快速開始

### 安裝指南

```bash
# 1. 安裝 Keras 3
pip install keras

# 2. 安裝後端（至少選擇一個）
pip install tensorflow  # TensorFlow 後端（推薦初學者）
pip install jax jaxlib  # JAX 後端（研究和高性能）
pip install torch       # PyTorch 後端（靈活調試）

# 3. 驗證安裝
python -c "import keras; print(keras.__version__)"
```

### 第一個 Keras 3 程式

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # 設置後端

import keras
from keras import layers
import numpy as np

# 生成示例資料
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))

# 建立模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

---

## 🔧 常見問題與故障排除

### Q1: 如何在不同後端之間切換？

**方法一：環境變數（推薦）**
```python
import os
os.environ['KERAS_BACKEND'] = 'jax'  # 在 import keras 之前設置
import keras
```

**方法二：設定檔**
```bash
# 在 ~/.keras/keras.json 中設置
{
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}
```

### Q2: Keras 2 程式碼如何遷移到 Keras 3？

```python
# Keras 2
from tensorflow import keras
from tensorflow.keras import layers

# Keras 3
import keras
from keras import layers
```

主要變化：
- ✅ Keras 3 是獨立包，不再是 `tensorflow.keras`
- ✅ 新的儲存格式使用 `.keras` 而非 `.h5`
- ✅ 某些舊的 API 已被棄用

### Q3: 為什麼我的模型訓練很慢？

**優化建議：**
```python
# 1. 啟用混合精度訓練
keras.mixed_precision.set_global_policy('mixed_float16')

# 2. 使用合適的批次大小
batch_size = 32  # 根據 GPU 記憶體調整

# 3. 啟用資料預取
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 4. 考慮切換到 JAX 後端（通常更快）
os.environ['KERAS_BACKEND'] = 'jax'
```

### Q4: 如何檢查當前使用的後端？

```python
import keras
print(f"當前後端：{keras.backend.backend()}")
```

### Q5: 模型儲存和載入的最佳實踐

```python
# 儲存完整模型（推薦）
model.save('my_model.keras')

# 載入模型
loaded_model = keras.models.load_model('my_model.keras')

# 只儲存權重
model.save_weights('model_weights.weights.h5')
model.load_weights('model_weights.weights.h5')
```

---

## ⚠️ 版本兼容性說明

### Keras 3 vs Keras 2

| 特性 | Keras 2 | Keras 3 |
|------|---------|---------|
| 後端支援 | 僅 TensorFlow | TensorFlow, JAX, PyTorch |
| 安裝方式 | `pip install tensorflow` | `pip install keras` |
| 導入方式 | `from tensorflow import keras` | `import keras` |
| 儲存格式 | `.h5` | `.keras` （推薦） |
| 性能 | 標準 | 優化後更快 |

### Python 版本要求
- ✅ Python 3.9+
- ✅ Python 3.10 （推薦）
- ✅ Python 3.11
- ⚠️ Python 3.12 （部分後端可能不支援）

---

## 📊 性能基準測試

### 不同後端的訓練速度比較（相對值）

```
任務類型         TensorFlow    JAX    PyTorch
簡單 MLP         1.0x         1.2x   0.95x
CNN 圖像分類     1.0x         1.5x   1.1x
RNN 序列處理     1.0x         1.8x   1.0x
Transformer      1.0x         2.0x   1.2x
```

*注意：實際性能取決於硬體配置和模型架構*

---

## 🎓 學習路徑建議

### 初學者路徑（1-2 個月）
1. ✅ 完成 ANN 基礎 (`1.ANN/`)
2. 📝 練習 MNIST 數字識別
3. 📝 實作簡單的二元分類（規劃中）
4. 📝 學習 CNN 進行圖像分類（規劃中）

### 中級路徑（2-3 個月）
1. 📝 深入 Functional API 和 Model Subclassing
2. 📝 遷移學習實戰
3. 📝 NLP 文字分類
4. 📝 時間序列預測

### 進階路徑（3-6 個月）
1. 📝 自定義層和訓練循環
2. 📝 生成式模型（GAN, VAE）
3. 📝 模型優化與部署
4. 📝 大規模分散式訓練

---

## 🐛 故障排除

### 常見錯誤及解決方案

**錯誤 1: "No module named 'keras'"**
```bash
# 解決方案
pip install keras --upgrade
```

**錯誤 2: "Backend not available"**
```bash
# 確保至少安裝一個後端
pip install tensorflow  # 或 jax/torch
```

**錯誤 3: "版本衝突"**
```bash
# 建立新的虛擬環境
python -m venv keras3_env
source keras3_env/bin/activate  # Linux/Mac
# 或
keras3_env\Scripts\activate  # Windows

pip install keras tensorflow
```

**錯誤 4: "GPU 未被使用"**
```python
# 檢查 GPU 是否可用
import tensorflow as tf
print("GPU 可用：", tf.config.list_physical_devices('GPU'))

# 啟用 GPU 記憶體增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 🔗 有用的資源連結

### 官方資源
- 📘 [Keras 3 發布公告](https://keras.io/keras_3/)
- 📺 [Keras 3 教學影片](https://www.youtube.com/c/keras-io)
- 💬 [Keras 論壇](https://github.com/keras-team/keras/discussions)
- 🐛 [回報問題](https://github.com/keras-team/keras/issues)

### 社群資源
- 🌟 [Keras Code Examples](https://keras.io/examples/)
- 📚 [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- 🎯 [Kaggle Keras Notebooks](https://www.kaggle.com/code?searchQuery=keras+3)

---

## 💪 實踐建議

### 學習技巧
1. **動手實作** - 每學一個概念就寫程式碼實作
2. **閱讀文檔** - Keras 文檔寫得非常好
3. **研究範例** - 官方 examples 是最佳學習資源
4. **參與社群** - 在論壇提問和回答問題
5. **做專案** - 完成 2-3 個端到端的實際專案

### 程式碼規範
```python
# 好的做法 ✅
import keras
from keras import layers, models, optimizers

# 避免的做法 ❌
from keras import *  # 不要使用通配符導入
```

---

**持續更新中...** 📖

**Keras 3：一次學習，處處使用！** 🚀

> 💡 **提示：** 遇到問題時，先查看[官方文檔](https://keras.io/)，90%的問題都能找到答案！
