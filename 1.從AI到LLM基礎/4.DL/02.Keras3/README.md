# Keras 3 個人複習筆記

> 🔄 **最後更新：** 2025-01
> 📊 **完成度：** 約 8%
> 🎯 **推薦版本：** Keras 3.0+
> 🚀 **特色：** 多後端支援（TensorFlow、JAX、PyTorch）

---

## ⚠️ 內容狀態說明

Keras 3 是 Keras 的重大升級，支援多個深度學習後端（TensorFlow、JAX、PyTorch），讓你可以用同一套 API 在不同框架間切換。

**已完成的內容：**
- ✅ 1. ANN 基礎實作

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
**狀態：已完成**

- ✅ 人工神經網路基礎
- ✅ Sequential API 使用
- ✅ 基本分類問題
- ✅ 模型訓練與評估

**檔案位置：** `1.ANN/ANN.ipynb`

---

### 🎨 2. 計算機視覺（CV）
**狀態：規劃中**

#### 2.1 圖像分類
- ⏳ 使用 CNN 進行圖像分類
- ⏳ 資料增強技術
- ⏳ 遷移學習（VGG, ResNet, EfficientNet）
- ⏳ 模型微調策略

#### 2.2 目標檢測
- ⏳ YOLO 整合
- ⏳ RetinaNet 實作
- ⏳ 邊界框處理

#### 2.3 圖像分割
- ⏳ U-Net 架構
- ⏳ Mask R-CNN
- ⏳ 語義分割應用

#### 2.4 現代視覺架構
- ⏳ Vision Transformer (ViT)
- ⏳ ConvNeXt
- ⏳ EfficientNetV2

**推薦專案：**
- 貓狗分類器
- 醫療影像分割
- 人臉識別系統

---

### 💬 3. 自然語言處理（NLP）
**狀態：規劃中**

#### 3.1 文本處理基礎
- ⏳ TextVectorization 層
- ⏳ Embedding 層使用
- ⏳ 序列處理技術

#### 3.2 文本分類
- ⏳ 情感分析
- ⏳ 新聞分類
- ⏳ 垃圾郵件檢測

#### 3.3 序列到序列模型
- ⏳ 機器翻譯
- ⏳ 文本摘要
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
3. 嘗試文本分類（IMDB 評論）
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

```bash
# 安裝 Keras 3
pip install keras

# 安裝後端（選擇一個或多個）
pip install tensorflow  # TensorFlow 後端
pip install jax jaxlib  # JAX 後端
pip install torch       # PyTorch 後端
```

---

**持續更新中...** 📖

**Keras 3：一次學習，處處使用！** 🚀
