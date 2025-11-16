# 深度學習完整學習指南

> 最後更新：2025-01

本目錄包含從基礎到進階的深度學習完整學習資源，涵蓋理論、實作與最新技術。

## 📚 學習路徑建議

**初學者路徑：** 00.DL_Path → 03.PyTorch 基礎 → 04.Ultralytics (實戰)

**進階學習：** 05.Transformer_lib → 06.Paper_with_code → 現代深度學習技術

**框架專精：** 根據需求選擇 TensorFlow2 / Keras3 / PyTorch 深入學習

---

## 📖 目錄內容

### ✅ 00.DL_Path - 動手深度學習（完整）
**完成度：95%** | **推薦指數：⭐⭐⭐⭐⭐**

基於《動手學深度學習》的完整筆記，包含 16 個章節，從基礎到高級主題。

**內容涵蓋：**
- ✅ 預備知識（線性代數、微積分、自動微分）
- ✅ 線性神經網路與多層感知機
- ✅ 深度學習計算（GPU、參數管理、自定義層）
- ✅ 卷積神經網路（CNN）與現代架構（ResNet、DenseNet）
- ✅ 循環神經網路（RNN、LSTM、GRU）
- ✅ 注意力機制與 Transformer
- ✅ 優化算法（SGD、Adam、學習率調度）
- ✅ 計算機視覺（目標檢測、語義分割、風格遷移）
- ✅ 自然語言處理（Word2Vec、BERT、情感分析）
- ✅ 計算性能優化（多GPU訓練、分散式訓練）

**⚠️ 注意：** 部分舊版筆記使用 MXNet/Gluon（已停止維護），建議參考 PyTorch 版本。

**詳細內容：** 查看 [00.DL_Path/README.md](./00.DL_Path/README.md)

---

### 🔨 01.Tensorflow2 - TensorFlow 2 實作
**完成度：約 5%** | **狀態：規劃中**

TensorFlow 2 的學習筆記與實作範例。

**目前已完成：**
- ✅ TensorFlow 快速入門
- ✅ CNN 基礎實作
- ✅ RNN 實作
- ✅ LSTM & GRU 實作

**規劃中的內容：**
- ⏳ 資料載入與預處理（tf.data）
- ⏳ 自定義訓練迴圈
- ⏳ 分散式訓練
- ⏳ 模型優化與部署
- ⏳ TensorBoard 整合
- ⏳ 遷移學習與微調

**詳細內容：** 查看 [01.Tensorflow2/README.md](./01.Tensorflow2/README.md)

---

### 🎯 02.Keras3 - Keras 3 多後端實作
**完成度：約 8%** | **狀態：規劃中**

Keras 3 的程式碼與筆記（支援 TensorFlow、JAX、PyTorch 後端）。

**目前已完成：**
- ✅ ANN 基礎實作

**規劃中的內容：**
- ⏳ 計算機視覺（CV）
- ⏳ 自然語言處理（NLP）
- ⏳ 結構化資料
- ⏳ 時間序列分析
- ⏳ 生成式深度學習
- ⏳ 音頻資料處理
- ⏳ 強化學習

**詳細內容：** 查看 [02.Keras3/README.md](./02.Keras3/README.md)

---

### 🔥 03.PyTorch - PyTorch 完整學習
**完成度：約 30%** | **推薦指數：⭐⭐⭐⭐**

PyTorch 的完整學習紀錄，從基礎到進階應用。

**已完成內容：**
- ✅ ResNet 實作
- ✅ Transformer 程式碼詳解
- ✅ Segment Anything 2 (SAM2) - **2024 最新技術** ⭐

**規劃內容：**
- ⏳ PyTorch 核心概念
- ⏳ 資料處理模組
- ⏳ 模型建構與訓練
- ⏳ 優化技巧
- ⏳ 視覺化工具
- ⏳ 圖像專案實戰
- ⏳ NLP 專案實戰
- ⏳ 大語言模型應用

**詳細內容：** 查看 [03.Pytorch/README.md](./03.Pytorch/README.md)

---

### 🎓 04.Ultralytics - YOLOv8 物件偵測
**完成度：良好** | **推薦指數：⭐⭐⭐⭐⭐**

使用 YOLOv8 進行物件偵測的完整教學，從資料準備到模型部署。

**內容包含：**
- ✅ 自製資料集訓練
- ✅ YOLOv8 模型訓練
- ✅ 模型評估與優化
- ✅ iOS 部署範例
- ✅ 實際應用案例

**特色：** 實戰導向，包含完整的端到端流程。

**詳細內容：** 查看 [04.Ultralytics/README.md](./04.Ultralytics/README.md)

---

### 🤗 05.Transformer_lib - Hugging Face Transformers
**完成度：良好** | **推薦指數：⭐⭐⭐⭐⭐**

使用 Hugging Face Transformers 庫進行大語言模型（LLM）訓練與應用。

**內容涵蓋：**
- ✅ Transformers 庫介紹
- ✅ 預訓練模型使用
- ✅ 模型微調（Fine-tuning）
- ✅ 實際應用範例

**詳細內容：** 查看 [05.Transformer_lib/README.md](./05.Transformer_lib/README.md)

---

### 📄 06.Paper_with_code - 論文實作
**完成度：初期** | **狀態：持續更新**

閱讀、實作與分析深度學習經典論文及其對應程式碼。

**已完成：**
- ✅ 影片品質評估（從美學與技術角度）

**規劃中：**
- ⏳ Attention Is All You Need (Transformer)
- ⏳ Vision Transformer (ViT)
- ⏳ CLIP
- ⏳ Stable Diffusion
- ⏳ 更多經典論文實作

---

### 🚀 07.現代深度學習技術 (2024-2025)
**狀態：新增規劃**

涵蓋最新的深度學習技術與趨勢。

**規劃內容：**

#### 🎨 計算機視覺
- ⏳ Vision Transformer (ViT) 實作
- ⏳ Diffusion Models（擴散模型）
- ⏳ EfficientNet V2/V3
- ⏳ ConvNeXt
- ⏳ 現代物件檢測（DETR, DINO）

#### 💬 自然語言處理與 LLM
- ⏳ 參數高效微調（LoRA, QLoRA, Adapter）
- ⏳ 指令微調（Instruction Tuning）
- ⏳ RLHF 基礎
- ⏳ 提示工程（Prompt Engineering）

#### ⚡ 訓練優化技術
- ⏳ 混合精度訓練（AMP）
- ⏳ Flash Attention
- ⏳ 梯度檢查點（Gradient Checkpointing）
- ⏳ 現代資料增強（RandAugment, CutMix, MixUp）

#### 🎭 多模態學習
- ⏳ CLIP 與變體
- ⏳ 圖文模型
- ⏳ 音視頻模型

#### 🛠️ 部署與生產
- ⏳ ONNX 轉換與優化
- ⏳ TensorRT 部署
- ⏳ 模型量化（INT8, FP16）
- ⏳ 邊緣裝置部署（TFLite, CoreML）

---

## 🎯 額外資源

### MLflow 入門
- 📓 [MLFLOW入門介紹：通過COLAB, NGROK, PYCARET.ipynb](./MLFLOW入門介紹：通過COLAB,%20NGROK,%20PYCARET.ipynb)

---

## 📝 學習建議

### 對於初學者：
1. 從 **00.DL_Path** 的預備知識開始
2. 學習基礎的線性神經網路和 CNN
3. 通過 **04.Ultralytics** 進行實戰練習
4. 根據興趣選擇框架深入學習

### 對於中級學習者：
1. 深入學習 **Transformer** 架構
2. 探索 **05.Transformer_lib** 的 LLM 應用
3. 閱讀並實作 **06.Paper_with_code** 中的經典論文
4. 學習模型優化與部署技術

### 對於進階學習者：
1. 研究最新的 **2024-2025 現代深度學習技術**
2. 實作參數高效微調方法
3. 探索多模態學習
4. 研究生產環境部署方案

---

## 🔗 相關連結

- [PyTorch 官方文檔](https://pytorch.org/docs/)
- [TensorFlow 官方文檔](https://www.tensorflow.org/)
- [Hugging Face 文檔](https://huggingface.co/docs)
- [動手學深度學習](https://zh.d2l.ai/)
- [Papers with Code](https://paperswithcode.com/)

---

## 📅 更新日誌

- **2025-01**: 大幅更新 README，添加完成度指標、學習路徑建議和現代深度學習技術規劃
- **2024**: 添加 SAM2、YOLOv8 等最新技術實作
- **2023**: 建立基礎內容結構

---

## 🤝 貢獻

歡迎提出問題和建議！如果您發現任何錯誤或有改進建議，請隨時提出 Issue。

---

**祝學習愉快！💪**



