# 深度學習經典與前沿論文復現 (Papers with Code)

> 📚 **目標**: 深入理解深度學習領域的經典論文和最新研究，通過代碼復現加深理解
>
> 💡 **特色**: 每個論文都包含詳細的理論解析、代碼實作和實驗結果

---

## 📋 目錄

- [簡介](#簡介)
- [論文分類](#論文分類)
  - [計算機視覺 (Computer Vision)](#計算機視覺-computer-vision)
  - [自然語言處理 (NLP)](#自然語言處理-nlp)
  - [生成模型 (Generative Models)](#生成模型-generative-models)
  - [視頻理解 (Video Understanding)](#視頻理解-video-understanding)
  - [多模態學習 (Multimodal Learning)](#多模態學習-multimodal-learning)
- [使用指南](#使用指南)
- [貢獻指南](#貢獻指南)
- [學習路徑](#學習路徑)

---

## 🎯 簡介

本目錄收錄了深度學習領域的**經典論文**和**前沿研究**的詳細解析與代碼復現。每個論文項目都包含：

- ✅ **論文詳細解說**: 逐章節分析，深入淺出
- ✅ **核心創新點**: 提煉關鍵技術和突破
- ✅ **代碼實作**: 從零開始的完整實現
- ✅ **實驗結果**: 復現論文中的主要實驗
- ✅ **應用場景**: 實際應用案例和最佳實踐
- ✅ **延伸閱讀**: 相關資源和進階材料

---

## 📚 論文分類

### 計算機視覺 (Computer Vision)

#### 🏆 經典基礎模型 (2012-2018)

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **AlexNet** - ImageNet Classification with Deep CNNs | 2012 | NIPS | 開啟深度學習時代 | 📝 計劃中 |
| **VGGNet** - Very Deep Convolutional Networks | 2014 | ICLR | 小卷積核堆疊設計 | 📝 計劃中 |
| **GoogLeNet (Inception v1)** | 2014 | CVPR | Inception 模塊 | 📝 計劃中 |
| **ResNet** - Deep Residual Learning | 2015 | CVPR | 殘差連接，突破深度瓶頸 | ✅ [已完成](./ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/) |
| **DenseNet** - Densely Connected Networks | 2017 | CVPR | 密集連接，特徵重用 | 📝 計劃中 |
| **MobileNet** - Efficient CNNs | 2017 | arXiv | 深度可分離卷積 | 📝 計劃中 |
| **EfficientNet** - Rethinking Model Scaling | 2019 | ICML | 複合縮放策略 | 📝 計劃中 |

#### 🎯 目標檢測 (Object Detection)

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **R-CNN** - Rich feature hierarchies | 2014 | CVPR | 區域提議 + CNN | 📝 計劃中 |
| **Fast R-CNN** | 2015 | ICCV | ROI pooling，端到端訓練 | 📝 計劃中 |
| **Faster R-CNN** | 2015 | NIPS | RPN 網絡 | 📝 計劃中 |
| **YOLO** - You Only Look Once | 2016 | CVPR | 實時單階段檢測 | ✅ [已完成](./YOLO%20-%20You%20Only%20Look%20Once/) |
| **SSD** - Single Shot Detector | 2016 | ECCV | 多尺度特徵檢測 | 📝 計劃中 |
| **RetinaNet** - Focal Loss | 2017 | ICCV | 類別不平衡解決方案 | 📝 計劃中 |
| **YOLOv3** | 2018 | arXiv | 多尺度預測改進 | 📝 計劃中 |
| **EfficientDet** | 2020 | CVPR | 高效目標檢測 | 📝 計劃中 |
| **DETR** - Detection Transformer | 2020 | ECCV | Transformer 用於檢測 | 📝 計劃中 |

#### 🖼️ 語義分割 (Semantic Segmentation)

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **FCN** - Fully Convolutional Networks | 2015 | CVPR | 端到端像素級預測 | 📝 計劃中 |
| **U-Net** | 2015 | MICCAI | 編碼器-解碼器架構 | ✅ [已完成](./U-Net%20-%20Convolutional%20Networks%20for%20Biomedical%20Image%20Segmentation/) |
| **SegNet** | 2016 | TPAMI | 高效分割網絡 | 📝 計劃中 |
| **DeepLab v3+** | 2018 | ECCV | 空洞卷積 + ASPP | 📝 計劃中 |
| **Mask R-CNN** | 2017 | ICCV | 實例分割 | 📝 計劃中 |

#### 🌟 Vision Transformer

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **ViT** - Vision Transformer | 2020 | ICLR | Transformer 用於視覺 | 📝 計劃中 |
| **Swin Transformer** | 2021 | ICCV | 滑動窗口 Transformer | 📝 計劃中 |
| **DeiT** - Data-efficient ViT | 2021 | ICML | 知識蒸餾訓練 ViT | 📝 計劃中 |
| **BEiT** - BERT Pre-Training of ViT | 2021 | ICLR | 視覺 BERT 預訓練 | 📝 計劃中 |

---

### 自然語言處理 (NLP)

#### 📝 經典模型

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **Word2Vec** | 2013 | NIPS | 詞向量表示 | 📝 計劃中 |
| **GloVe** | 2014 | EMNLP | 全局向量表示 | 📝 計劃中 |
| **Seq2Seq** | 2014 | NIPS | 序列到序列學習 | 📝 計劃中 |
| **Attention Mechanism** | 2015 | ICLR | 注意力機制 | 📝 計劃中 |

#### 🔥 Transformer 時代

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **Transformer** - Attention is All You Need | 2017 | NIPS | 純注意力架構 | ✅ [已完成](./Transformer%20-%20Attention%20Is%20All%20You%20Need/) |
| **BERT** | 2018 | NAACL | 雙向預訓練 | 📝 計劃中 |
| **GPT** | 2018 | OpenAI | 自回歸預訓練 | 📝 計劃中 |
| **GPT-2** | 2019 | OpenAI | 大規模語言模型 | 📝 計劃中 |
| **RoBERTa** | 2019 | arXiv | BERT 訓練優化 | 📝 計劃中 |
| **T5** - Text-to-Text Transfer | 2020 | JMLR | 統一文本生成框架 | 📝 計劃中 |
| **GPT-3** | 2020 | NeurIPS | 1750 億參數模型 | 📝 計劃中 |

---

### 生成模型 (Generative Models)

#### 🎨 GAN (生成對抗網絡)

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **GAN** - Generative Adversarial Networks | 2014 | NIPS | 對抗訓練框架 | ✅ [已完成](./GAN%20-%20Generative%20Adversarial%20Networks/) |
| **DCGAN** - Deep Convolutional GAN | 2015 | ICLR | 穩定的 GAN 訓練 | 📝 計劃中 |
| **Pix2Pix** | 2017 | CVPR | 條件圖像生成 | 📝 計劃中 |
| **CycleGAN** | 2017 | ICCV | 無配對圖像轉換 | 📝 計劃中 |
| **StyleGAN** | 2019 | CVPR | 高質量人臉生成 | 📝 計劃中 |
| **StyleGAN2** | 2020 | CVPR | 改進的生成質量 | 📝 計劃中 |

#### 🌈 Diffusion Models (擴散模型)

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **DDPM** - Denoising Diffusion Probabilistic | 2020 | NeurIPS | 擴散模型基礎 | 📝 計劃中 |
| **DDIM** - Denoising Diffusion Implicit | 2021 | ICLR | 加速採樣 | 📝 計劃中 |
| **Stable Diffusion** | 2022 | CVPR | Latent Diffusion 模型 | 📝 計劃中 |
| **ControlNet** | 2023 | ICCV | 可控圖像生成 | 📝 計劃中 |

#### 🎬 VAE 與其他生成模型

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **VAE** - Variational Autoencoder | 2013 | ICLR | 變分推斷生成 | 📝 計劃中 |
| **VQ-VAE** | 2017 | NeurIPS | 離散潛在表示 | 📝 計劃中 |
| **VQ-VAE-2** | 2019 | NeurIPS | 層次化生成 | 📝 計劃中 |

---

### 視頻理解 (Video Understanding)

#### 🎥 視頻分類與動作識別

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **Two-Stream CNNs** | 2014 | NIPS | 雙流架構 | 📝 計劃中 |
| **C3D** - 3D ConvNets | 2015 | ICCV | 3D 卷積用於視頻 | 📝 計劃中 |
| **I3D** - Inflated 3D ConvNet | 2017 | CVPR | 膨脹 3D 卷積 | 📝 計劃中 |
| **SlowFast Networks** | 2019 | ICCV | 雙路徑時空建模 | 📝 計劃中 |
| **TimeSformer** | 2021 | ICML | 視頻 Transformer | 📝 計劃中 |

#### 📊 視頻質量評估

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **DOVER** - Video Quality Assessment | 2023 | CVPR | 美學與技術雙維度評估 | ✅ [已完成](./Exploring%20Video%20Quality%20Assessment%20on%20User%20Generated%20Contents%20from%20Aesthetic%20and%20Technical%20Perspectives/) |

---

### 多模態學習 (Multimodal Learning)

#### 🌐 視覺-語言模型

| 論文名稱 | 年份 | 會議 | 主要貢獻 | 狀態 |
|---------|------|------|---------|------|
| **CLIP** - Contrastive Language-Image | 2021 | ICML | 對比學習視覺語言 | 📝 計劃中 |
| **ALIGN** | 2021 | ICML | 大規模視覺語言對齊 | 📝 計劃中 |
| **DALL-E** | 2021 | OpenAI | 文本到圖像生成 | 📝 計劃中 |
| **Flamingo** | 2022 | NeurIPS | Few-shot 視覺語言 | 📝 計劃中 |
| **BLIP** | 2022 | ICML | 統一視覺語言理解 | 📝 計劃中 |
| **LLaVA** | 2023 | NeurIPS | 視覺指令調優 | 📝 計劃中 |

---

## 🚀 使用指南

### 目錄結構說明

每個論文項目通常包含以下文件結構：

```
論文名稱/
├── README.md                    # 論文概述和快速導航
├── 論文詳細解說.md              # 逐章節深入分析
├── 代碼實作.ipynb               # Jupyter Notebook 實作
├── 核心代碼/                    # 核心算法實現
│   ├── model.py                # 模型定義
│   ├── train.py                # 訓練腳本
│   ├── evaluate.py             # 評估腳本
│   └── utils.py                # 工具函數
├── 實驗結果/                    # 實驗數據和可視化
├── 數據集/                      # 數據集說明或鏈接
└── 參考資源.md                  # 延伸閱讀和相關資源
```

### 學習建議

1. **循序漸進**: 建議從經典論文開始，逐步深入到前沿研究
2. **理論結合實踐**: 先閱讀論文解說，再動手實作代碼
3. **對比學習**: 比較同類型論文的異同，理解演進脈絡
4. **動手實驗**: 嘗試修改參數，觀察結果變化
5. **應用導向**: 思考如何將論文技術應用到實際問題

### 推薦學習路徑

#### 初學者路徑
1. AlexNet → VGGNet → ResNet (理解 CNN 演進)
2. Word2Vec → Attention → Transformer (理解 NLP 基礎)
3. GAN → VAE (理解生成模型)

#### 進階路徑
1. ViT → Swin Transformer (視覺 Transformer)
2. BERT → GPT → T5 (大規模預訓練)
3. DDPM → Stable Diffusion (擴散模型)

#### 應用路徑
1. YOLO → Faster R-CNN (目標檢測)
2. U-Net → DeepLab (分割任務)
3. CLIP → DALL-E (多模態應用)

---

## 💡 如何選擇論文學習

### 按任務類型選擇

| 任務 | 推薦論文 | 難度 |
|-----|---------|------|
| 圖像分類 | ResNet, EfficientNet, ViT | ⭐⭐ |
| 目標檢測 | YOLO, Faster R-CNN, DETR | ⭐⭐⭐ |
| 語義分割 | U-Net, DeepLab, Mask R-CNN | ⭐⭐⭐ |
| 圖像生成 | GAN, VAE, Diffusion Models | ⭐⭐⭐⭐ |
| 文本理解 | BERT, GPT, T5 | ⭐⭐⭐ |
| 視頻分析 | I3D, SlowFast, TimeSformer | ⭐⭐⭐⭐ |
| 多模態 | CLIP, DALL-E, LLaVA | ⭐⭐⭐⭐⭐ |

### 按創新點學習

- **架構創新**: ResNet (殘差), Transformer (注意力), U-Net (編碼解碼)
- **訓練技巧**: BERT (掩碼預訓練), GAN (對抗訓練), Focal Loss (類別不平衡)
- **效率優化**: MobileNet (輕量化), EfficientNet (縮放), DDIM (加速採樣)
- **應用突破**: YOLO (實時檢測), Stable Diffusion (可控生成), CLIP (零樣本)

---

## 🛠️ 環境配置

### 基礎環境

```bash
# Python 版本
Python 3.8+

# 核心依賴
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install numpy pandas matplotlib seaborn
pip install jupyter notebook
pip install opencv-python pillow
pip install scikit-learn scipy

# 可選依賴（根據具體論文）
pip install transformers timm accelerate
pip install diffusers controlnet-aux
pip install decord av moviepy
```

### GPU 支持

```bash
# CUDA 環境（推薦）
# 查看 CUDA 版本
nvcc --version

# 安裝對應版本的 PyTorch
# 訪問: https://pytorch.org/get-started/locally/
```

---

## 📈 學習進度追蹤

### 完成狀態說明

- ✅ **已完成**: 包含詳細解說、代碼實作和實驗結果
- 🚧 **進行中**: 正在編寫或實作
- 📝 **計劃中**: 已列入計劃，待開始
- 💡 **待定**: 考慮中但尚未確定

### 更新日誌

- **2024-11**: 創建目錄結構，完成 DOVER 論文復現
- **2024-12**: 計劃添加經典 CNN 論文（AlexNet, VGGNet, ResNet）
- **2025-01**: 計劃添加 Transformer 系列論文

---

## 🤝 貢獻指南

歡迎貢獻新的論文復現或改進現有內容！

### 貢獻方式

1. **新增論文復現**
   - 選擇一篇影響力大的論文
   - 按照標準目錄結構組織文件
   - 包含詳細的理論解析和代碼實作
   - 確保代碼可運行並有清晰的註釋

2. **改進現有內容**
   - 修正錯誤或補充缺失內容
   - 優化代碼實作
   - 添加更多實驗結果
   - 提供更好的可視化

3. **質量標準**
   - 代碼風格一致，遵循 PEP 8
   - 充分的註釋和文檔
   - 可復現的實驗結果
   - 引用原始論文和相關資源

---

## 📚 參考資源

### 論文搜索平台

- [Papers with Code](https://paperswithcode.com/): 論文 + 代碼 + 排行榜
- [arXiv](https://arxiv.org/): 預印本論文
- [Google Scholar](https://scholar.google.com/): 學術搜索
- [Semantic Scholar](https://www.semanticscholar.org/): AI 驅動的論文搜索

### 學習資源

- [Deep Learning Book](https://www.deeplearningbook.org/): Ian Goodfellow 等著
- [Dive into Deep Learning](https://d2l.ai/): 動手學深度學習
- [Stanford CS231n](http://cs231n.stanford.edu/): 計算機視覺課程
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/): NLP 課程

### 代碼資源

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Hugging Face](https://huggingface.co/): 預訓練模型庫
- [Timm](https://github.com/huggingface/pytorch-image-models): PyTorch 圖像模型庫

### 會議與期刊

- **頂級會議**: CVPR, ICCV, ECCV (視覺), NeurIPS, ICML (機器學習), ACL, EMNLP (NLP)
- **頂級期刊**: TPAMI, IJCV, JMLR

---

## 📊 統計信息

- **總論文數**: 80+ (計劃中)
- **已完成**: 6 (DOVER, ResNet, Transformer, YOLO, U-Net, GAN)
- **進行中**: 0
- **計劃中**: 74+
- **涵蓋年份**: 2012-2025
- **涵蓋領域**: 計算機視覺、NLP、生成模型、多模態等

---

## 🎓 學習目標

通過本目錄的學習，你將能夠：

1. ✅ **深入理解** 深度學習領域的經典論文和前沿研究
2. ✅ **掌握實作** 從零開始實現複雜的深度學習模型
3. ✅ **追蹤前沿** 了解最新的研究動態和技術趨勢
4. ✅ **應用實踐** 將論文技術應用到實際問題中
5. ✅ **培養能力** 提升閱讀論文、復現算法的能力

---

## 📝 許可證

本項目僅供學習和研究使用。所有論文版權歸原作者所有，請遵守相應的授權協議。

---

## 📧 聯繫方式

如有問題或建議，歡迎通過以下方式聯繫：

- 提交 Issue
- 提交 Pull Request
- Email: [待補充]

---

<div align="center">
  <p><strong>⭐ 持續更新中，敬請期待！</strong></p>
  <p>📚 深度學習 | 💡 論文復現 | 🚀 知識分享</p>
  <p><i>最後更新: 2024-11-18</i></p>
</div>
