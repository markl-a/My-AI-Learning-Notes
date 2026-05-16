# 現代卷積神經網路 (Modern Convolutional Neural Networks)

> 從經典架構到最新技術的完整學習路徑

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📚 目錄

- [概述](#概述)
- [學習路徑](#學習路徑)
- [內容結構](#內容結構)
- [環境設置](#環境設置)
- [快速開始](#快速開始)
- [AI 輔助學習工具](#ai-輔助學習工具)
- [學習建議](#學習建議)
- [參考資源](#參考資源)

---

## 🎯 概述

本章節深入探討現代卷積神經網路（CNN）的發展歷程與核心架構。從2012年AlexNet開啟深度學習革命開始，到2024-2025年的最新架構，我們將系統性地學習：

- **經典架構**：AlexNet, VGG, NiN, GoogLeNet, ResNet, DenseNet
- **現代架構**：EfficientNet, MobileNet系列, Vision Transformer
- **核心技術**：批量歸一化、殘差連接、注意力機制
- **實戰技能**：遷移學習、模型優化、可視化解釋

### 為什麼學習這些架構？

1. **歷史脈絡**：理解CNN演進的思想歷程
2. **設計理念**：掌握網路設計的核心原則
3. **實用價值**：這些架構至今仍廣泛應用
4. **創新基礎**：為設計自己的架構打下基礎

---

## 🗺️ 學習路徑

### 階段一：經典架構基礎（第1-7節）

```
AlexNet (2012) → VGG (2014) → NiN (2014) → GoogLeNet (2014)
    ↓
Batch Normalization (2015) → ResNet (2015) → DenseNet (2017)
```

**學習重點**：
- 理解每個架構的創新點
- 掌握核心概念（卷積、池化、批歸一化等）
- 實作基礎網路結構

**預計時間**：1-2週

### 階段二：遷移學習與實戰（第8節）

```
預訓練模型 → Fine-tuning → 實際應用
```

**學習重點**：
- 使用預訓練模型
- 特徵提取 vs Fine-tuning
- 實戰項目實作

**預計時間**：1週

### 階段三：現代架構（第9節）

```
EfficientNet (2019) → MobileNet系列 (2017-2019) → 效率優化
```

**學習重點**：
- 模型效率與性能平衡
- 移動端部署
- AutoML與神經架構搜索

**預計時間**：1週

### 階段四：進階技術（第10-12節）

```
模型可視化 → 實用技巧 → 完整項目
```

**學習重點**：
- 模型解釋性
- 調優技巧
- 端到端項目

**預計時間**：1-2週

---

## 📖 內容結構

### 經典架構系列

| 編號 | 主題 | 年份 | 核心創新 | 難度 |
|------|------|------|----------|------|
| 01 | [AlexNet](1_alexnet.ipynb) | 2012 | 深度CNN、ReLU、Dropout | ⭐⭐ |
| 02 | [VGG](2_vgg.ipynb) | 2014 | 小卷積核堆疊、網路深度 | ⭐⭐ |
| 03 | [NiN](3_nin.ipynb) | 2014 | 1×1卷積、全局平均池化 | ⭐⭐⭐ |
| 04 | [GoogLeNet](4_googlenet.ipynb) | 2014 | Inception模塊、多尺度 | ⭐⭐⭐ |
| 05 | [Batch Normalization](5_batch-norm.ipynb) | 2015 | 批歸一化、訓練穩定性 | ⭐⭐⭐ |
| 06 | [ResNet](6_resnet.ipynb) | 2015 | 殘差連接、超深網路 | ⭐⭐⭐⭐ |
| 07 | [DenseNet](7_densenet.ipynb) | 2017 | 密集連接、特徵重用 | ⭐⭐⭐⭐ |

### 實戰與進階系列

| 編號 | 主題 | 內容重點 | 難度 |
|------|------|----------|------|
| 08 | [遷移學習](8_transfer_learning.ipynb) | 預訓練模型、Fine-tuning | ⭐⭐⭐ |
| 09 | [現代架構](9_modern_architectures.ipynb) | EfficientNet、MobileNet | ⭐⭐⭐⭐ |
| 10 | [模型可視化](10_model_visualization.ipynb) | GradCAM、特徵可視化 | ⭐⭐⭐ |
| 11 | [實用技巧](11_practical_tips.ipynb) | 調優、優化、部署 | ⭐⭐⭐⭐ |
| 12 | [完整項目](12_complete_project.ipynb) | 端到端實戰項目 | ⭐⭐⭐⭐⭐ |

---

## 🛠️ 環境設置

### 基礎環境

```bash
# 建立虛擬環境
conda create -n modern-cnn python=3.10
conda activate modern-cnn

# 安裝 PyTorch（根據你的CUDA版本）
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU版本（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 必要套件

```bash
# 核心深度學習套件
pip install torchvision torchsummary tensorboard

# 資料處理與視覺化
pip install numpy pandas matplotlib seaborn opencv-python pillow

# 模型工具
pip install timm  # PyTorch Image Models（預訓練模型庫）
pip install torchinfo  # 模型資訊顯示

# 可視化與解釋性
pip install grad-cam pytorch-grad-cam captum

# 實驗追蹤
pip install wandb tensorboard

# 其他工具
pip install tqdm scikit-learn jupyter ipywidgets
```

### 驗證安裝

```python
import torch
import torchvision
import timm

print(f"PyTorch版本: {torch.__version__}")
print(f"Torchvision版本: {torchvision.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU設備: {torch.cuda.get_device_name(0)}")
print(f"Timm版本: {timm.__version__}")
```

---

## 🚀 快速開始

### 1. 克隆並進入目錄

```bash
cd My-AI-Learning-Notes/1.從AI到LLM基礎/4.DL/00.DL_Path/7_現代卷積神經網路
```

### 2. 啟動 Jupyter Notebook

```bash
jupyter notebook
```

### 3. 按順序學習

建議按照以下順序學習：

**初學者路徑**：
1. 先閱讀本 README
2. 從 AlexNet (01) 開始
3. 逐步學習到 ResNet (06)
4. 實作遷移學習 (08)
5. 完成一個完整項目 (12)

**進階路徑**：
1. 快速瀏覽經典架構 (01-07)
2. 深入現代架構 (09)
3. 學習可視化技術 (10)
4. 掌握實用技巧 (11)
5. 完成進階項目 (12)

### 4. 使用預訓練模型（快速實驗）

```python
import timm
import torch

# 查看可用的預訓練模型
models = timm.list_models('*resnet*', pretrained=True)
print(f"可用的ResNet模型: {len(models)}個")

# 載入預訓練的ResNet50
model = timm.create_model('resnet50', pretrained=True, num_classes=10)
model.eval()

# 測試模型
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"輸出形狀: {output.shape}")
```

---

## 🤖 AI 輔助學習工具

### 1. 自動化模型搜索與訓練

使用 `timm` 庫快速實驗不同架構：

```python
import timm

# 建立模型（自動下載預訓練權重）
model = timm.create_model('resnet50', pretrained=True)

# 查看模型配置
print(timm.models.resnet50().default_cfg)

# 列出所有可用模型
all_models = timm.list_models(pretrained=True)
print(f"總共有 {len(all_models)} 個預訓練模型可用！")
```

### 2. 超參數調優

使用 Optuna 進行自動超參數搜索：

```python
import optuna

def objective(trial):
    # 定義超參數搜索空間
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # 訓練並返回驗證準確率
    val_acc = train_and_evaluate(lr, batch_size)
    return val_acc

# 建立研究並優化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"最佳超參數: {study.best_params}")
```

### 3. 實驗追蹤與視覺化

使用 Weights & Biases 追蹤實驗：

```python
import wandb

# 初始化實驗
wandb.init(project="modern-cnn", name="resnet50-experiment")

# 記錄超參數
wandb.config.update({
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
})

# 在訓練循環中記錄指標
for epoch in range(epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()

    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "epoch": epoch
    })
```

### 4. 模型解釋與可視化

使用 GradCAM 可視化模型關注區域：

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 初始化 GradCAM
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# 生成熱力圖
grayscale_cam = cam(input_tensor=input_image, targets=None)

# 疊加到原圖
visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
```

### 5. 自動資料增強

使用 AutoAugment 和 RandAugment：

```python
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform

# RandAugment
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    rand_augment_transform(
        config_str='rand-m9-mstd0.5',
        hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
    ),
    transforms.ToTensor(),
])
```

---

## 💡 學習建議

### 對於初學者

1. **循序漸進**
   - 不要跳過基礎章節
   - 每個架構都要親手實作一遍
   - 理解為什麼而不只是怎麼做

2. **動手實踐**
   - 運行每一個程式碼示例
   - 修改參數觀察變化
   - 嘗試在不同資料集上實驗

3. **建立直覺**
   - 可視化網路結構
   - 觀察特徵圖變化
   - 理解每層的作用

4. **記錄筆記**
   - 記錄遇到的問題和解決方法
   - 總結每個架構的優缺點
   - 建立自己的知識圖譜

### 對於進階學習者

1. **深入原理**
   - 閱讀原始論文
   - 理解數學推導
   - 掌握實現細節

2. **對比分析**
   - 比較不同架構的性能
   - 分析計算效率
   - 研究適用場景

3. **創新實踐**
   - 嘗試改進現有架構
   - 結合最新技術
   - 解決實際問題

4. **社群參與**
   - 參加競賽（Kaggle等）
   - 閱讀最新論文
   - 貢獻開源項目

### 常見問題與解決方案

#### Q1: 訓練太慢怎麼辦？
- 使用更小的模型（如 MobileNet）
- 減少批次大小或圖像分辨率
- 使用混合精度訓練（AMP）
- 考慮使用預訓練模型

#### Q2: 記憶體不足？
- 減小批次大小
- 使用梯度累積
- 啟用梯度檢查點
- 使用更小的模型

#### Q3: 過擬合嚴重？
- 增加資料增強
- 使用 Dropout 和正則化
- 減少模型複雜度
- 早停（Early Stopping）

#### Q4: 如何選擇架構？
- 考慮任務需求（準確率 vs 速度）
- 評估資源限制（GPU、內存）
- 參考論文和競賽結果
- 從預訓練模型開始

---

## 📊 架構對比速查表

| 模型 | 年份 | 參數量 | ImageNet Top-1 | 特點 | 適用場景 |
|------|------|--------|----------------|------|----------|
| AlexNet | 2012 | 60M | 57.1% | 深度學習開端 | 教學、理解基礎 |
| VGG-16 | 2014 | 138M | 71.5% | 簡單結構 | 特徵提取 |
| GoogLeNet | 2014 | 7M | 69.8% | Inception模塊 | 效率優先 |
| ResNet-50 | 2015 | 25M | 76.1% | 殘差連接 | 通用backbone |
| DenseNet-121 | 2017 | 8M | 74.9% | 密集連接 | 特徵重用 |
| EfficientNet-B0 | 2019 | 5M | 77.1% | 複合縮放 | 移動端 |
| ResNet-152 | 2015 | 60M | 78.3% | 超深網路 | 高精度需求 |

---

## 🔗 參考資源

### 重要論文

1. **AlexNet** (2012)
   - ImageNet Classification with Deep Convolutional Neural Networks
   - [論文連結](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

2. **VGG** (2014)
   - Very Deep Convolutional Networks for Large-Scale Image Recognition
   - [論文連結](https://arxiv.org/abs/1409.1556)

3. **GoogLeNet/Inception** (2014)
   - Going Deeper with Convolutions
   - [論文連結](https://arxiv.org/abs/1409.4842)

4. **ResNet** (2015)
   - Deep Residual Learning for Image Recognition
   - [論文連結](https://arxiv.org/abs/1512.03385)

5. **DenseNet** (2017)
   - Densely Connected Convolutional Networks
   - [論文連結](https://arxiv.org/abs/1608.06993)

6. **EfficientNet** (2019)
   - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
   - [論文連結](https://arxiv.org/abs/1905.11946)

### 線上資源

- **官方文檔**
  - [PyTorch官方教程](https://pytorch.org/tutorials/)
  - [Torchvision模型](https://pytorch.org/vision/stable/models.html)
  - [Timm文檔](https://timm.fast.ai/)

- **互動教程**
  - [Dive into Deep Learning](https://d2l.ai/)
  - [Fast.ai課程](https://course.fast.ai/)
  - [Stanford CS231n](http://cs231n.stanford.edu/)

- **實用工具**
  - [Papers with Code](https://paperswithcode.com/)
  - [Model Zoo](https://modelzoo.co/)
  - [Netron (可視化工具)](https://netron.app/)

### 資料集

- **CIFAR-10/100**: 小型圖像分類（60,000張圖片）
- **Fashion-MNIST**: 時尚物品分類（70,000張圖片）
- **ImageNet**: 大規模圖像分類（1400萬張圖片）
- **COCO**: 目標檢測與分割（330,000張圖片）

---

## 📝 學習檢查清單

完成以下檢查項，確保你已經掌握核心知識：

### 基礎知識
- [ ] 理解卷積操作的原理
- [ ] 掌握池化層的作用
- [ ] 了解批歸一化的必要性
- [ ] 理解激活函式的選擇

### 架構理解
- [ ] 能說明 AlexNet 的創新點
- [ ] 理解 VGG 的設計哲學
- [ ] 掌握 Inception 模塊的思想
- [ ] 深刻理解殘差連接的作用
- [ ] 了解 DenseNet 的特徵重用機制

### 實作能力
- [ ] 能從零實現基礎 CNN
- [ ] 會使用預訓練模型
- [ ] 掌握遷移學習技術
- [ ] 能進行模型調優
- [ ] 會使用可視化工具

### 進階技能
- [ ] 了解現代架構（EfficientNet等）
- [ ] 掌握模型壓縮技術
- [ ] 能進行模型部署
- [ ] 理解 AutoML 概念
- [ ] 完成端到端項目

---

## 🎓 學習成果展示

完成本章節學習後，你應該能夠：

1. ✅ **理論掌握**
   - 深入理解CNN的演進歷史
   - 掌握各種架構的設計理念
   - 理解為什麼某些設計更有效

2. ✅ **實作能力**
   - 從零實現經典CNN架構
   - 使用預訓練模型解決實際問題
   - 進行模型優化和調優

3. ✅ **項目經驗**
   - 完成至少一個圖像分類項目
   - 掌握完整的開發流程
   - 能夠部署模型到生產環境

4. ✅ **創新思維**
   - 能夠改進現有架構
   - 針對特定問題設計網路
   - 具備閱讀和理解最新論文的能力

---

## 🤝 貢獻指南

歡迎貢獻！如果你發現錯誤或有改進建議：

1. Fork 本倉庫
2. 建立你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個 Pull Request

---

## 📄 授權

本項目採用 MIT 授權 - 查看 [LICENSE](../LICENSE) 文件了解詳情

---

## 📧 聯繫方式

如有問題或建議，歡迎：
- 提交 Issue
- 發送 Pull Request
- 在討論區參與討論

---

**最後更新**: 2024年11月
**維護者**: AI Learning Community

---

<div align="center">

### ⭐ 如果這個資源對你有幫助，請給個星星！⭐

**祝學習愉快！🚀**

[返回頂部](#現代卷積神經網路-modern-convolutional-neural-networks)

</div>
