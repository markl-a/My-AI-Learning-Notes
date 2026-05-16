# ResNet - 深度殘差網絡

> **論文**: Deep Residual Learning for Image Recognition
>
> **作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
>
> **發表**: CVPR 2016 (Best Paper Award)
>
> **論文鏈接**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
>
> **官方程式碼**: [GitHub - KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)

---

## 📋 目錄

- [簡介](#簡介)
- [核心創新](#核心創新)
- [為什麼 ResNet 如此重要](#為什麼-resnet-如此重要)
- [文件說明](#文件說明)
- [快速開始](#快速開始)
- [模型架構](#模型架構)
- [實驗結果](#實驗結果)
- [應用場景](#應用場景)
- [參考資源](#參考資源)

---

## 🎯 簡介

**ResNet (Residual Network)** 是深度學習歷史上最具影響力的論文之一，徹底改變了深度神經網絡的設計思路。它通過引入**殘差連接 (Residual Connection)** 或稱**跳躍連接 (Skip Connection)**，成功訓練了超過 1000 層的極深網絡，並在 ImageNet 2015 競賽中取得冠軍。

### 核心問題

在 ResNet 之前，深度學習領域面臨一個重要挑戰：

**❓ 為什麼更深的網絡反而性能下降？**

直覺上，更深的網絡應該至少和淺層網絡一樣好（因為可以將額外的層設為恆等映射）。但實際上：

- ❌ **梯度消失/爆炸**: 反向傳播時梯度逐層衰減或放大
- ❌ **退化問題 (Degradation)**: 深度網絡訓練誤差反而上升
- ❌ **優化困難**: 深層網絡難以優化

### ResNet 的解決方案

ResNet 提出了**殘差學習 (Residual Learning)** 的概念：

```
傳統方式: H(x) = F(x)           # 直接學習目標映射
ResNet: H(x) = F(x) + x          # 學習殘差映射

其中:
- H(x): 目標映射
- F(x): 殘差映射（要學習的部分）
- x: 輸入（通過跳躍連接直接傳遞）
```

**核心洞察**: 學習殘差 F(x) = H(x) - x 比直接學習 H(x) 更容易！

---

## 💡 核心創新

### 1. 殘差塊 (Residual Block)

ResNet 的基本構建單元：

```
Input x
   │
   ├────────────────────┐  (跳躍連接)
   │                    │
   ▼                    │
 Conv 3x3               │
   │                    │
   ▼                    │
 BatchNorm              │
   │                    │
   ▼                    │
  ReLU                  │
   │                    │
   ▼                    │
 Conv 3x3               │
   │                    │
   ▼                    │
 BatchNorm              │
   │                    │
   ▼                    │
   ├────────────────────┘  (元素相加)
   │
   ▼
  ReLU
   │
   ▼
 Output
```

**Python 實現**:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 主路徑
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳躍連接（需要調整維度時）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路徑
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 殘差連接
        out += self.shortcut(x)
        out = self.relu(out)

        return out
```

### 2. Bottleneck 設計（針對更深網絡）

對於 ResNet-50/101/152，使用 Bottleneck 塊以降低計算量：

```
Input x (256 維)
   │
   ├────────────────────┐
   │                    │
   ▼                    │
 Conv 1x1 (降維到 64)   │
   │                    │
   ▼                    │
 BatchNorm + ReLU       │
   │                    │
   ▼                    │
 Conv 3x3 (保持 64)     │
   │                    │
   ▼                    │
 BatchNorm + ReLU       │
   │                    │
   ▼                    │
 Conv 1x1 (升維到 256)  │
   │                    │
   ▼                    │
 BatchNorm              │
   │                    │
   ├────────────────────┘
   │
   ▼
  ReLU
```

**優勢**:
- ✅ 先降維再升維，減少計算量
- ✅ 參數量減少約 50%
- ✅ 適合構建極深網絡（50+ 層）

### 3. 批量歸一化 (Batch Normalization)

每個卷積層後都使用 BN：
- ✅ 加速訓練
- ✅ 允許更大的學習率
- ✅ 減少對初始化的依賴

### 4. 恆等映射的優化

**關鍵設計選擇**:
- ✅ 使用 1x1 卷積調整維度（而非 padding）
- ✅ 保持跳躍連接的簡潔性
- ✅ 激活函式放在加法之後

---

## 🌟 為什麼 ResNet 如此重要？

### 1. 理論突破

**解決了深度學習的根本問題**:
- ✅ 證明了可以訓練極深的網絡（1000+ 層）
- ✅ 解決了退化問題（degradation problem）
- ✅ 提供了新的優化視角（學習殘差而非目標函式）

### 2. 實踐影響

**廣泛應用於各個領域**:
- 🏆 **ImageNet 2015**: Top-5 錯誤率 3.57%（超越人類水平 ~5.1%）
- 🎯 **目標檢測**: Faster R-CNN, Mask R-CNN 的 backbone
- 🖼️ **分割**: DeepLab, U-Net 的殘差變體
- 🎬 **影片**: 3D ResNet 用於動作識別
- 🌐 **NLP**: Transformer 中的殘差連接受其啟發

### 3. 架構影響

**啟發了後續研究**:
- ResNeXt (2017): 增加基數維度
- DenseNet (2017): 密集連接
- EfficientNet (2019): 複合縮放
- Vision Transformer (2020): 殘差連接在 Transformer 中

### 4. 工程價值

- ✅ **易於實現**: 概念簡單，程式碼清晰
- ✅ **穩定訓練**: 收斂快，不易過擬合
- ✅ **可擴展性**: 容易調整深度和寬度
- ✅ **遷移學習**: 預訓練模型效果好

---

## 📁 文件說明

本目錄包含以下文件：

### 1. `README.md` (本文件)
**快速導航和概述**，包含：
- 📖 論文核心思想
- 💡 關鍵創新點
- 🎯 應用場景
- 🚀 快速開始

### 2. `ResNet論文詳細解說.md`
**深入解析**，包含：
- 📚 逐章節論文分析
- 🔬 數學推導和理論
- 📊 實驗結果詳解
- 🎓 延伸閱讀

### 3. `ResNet完整實作.ipynb`
**Jupyter Notebook 教程**，包含：
- 🛠️ 從零實現 ResNet-18/34/50/101/152
- 📊 在 CIFAR-10/ImageNet 上訓練
- 🎨 可視化特徵和激活
- 🔍 消融實驗

### 4. `核心程式碼/`
**模塊化實現**，包含：
- `resnet.py`: ResNet 模型定義
- `train.py`: 訓練腳本
- `evaluate.py`: 評估腳本
- `utils.py`: 工具函數
- `config.py`: 設定檔

---

## 🚀 快速開始

### 環境需求

```bash
# Python 版本
Python 3.8+

# 主要依賴
pip install torch torchvision
pip install numpy matplotlib
pip install tqdm tensorboard
```

### 基本使用

#### 1. 使用 PyTorch 官方實現

```python
import torch
import torchvision.models as models

# 載入預訓練模型
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet152 = models.resnet152(pretrained=True)

# 推理
model = resnet50
model.eval()

with torch.no_grad():
    output = model(input_tensor)  # input: [batch, 3, 224, 224]
    predictions = torch.softmax(output, dim=1)
```

#### 2. 從零實現

```python
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 殘差塊組
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # 分類頭
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始處理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 殘差塊
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分類
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 使用
model = ResNet18(num_classes=10)  # CIFAR-10
```

#### 3. 訓練範例

```python
import torch.optim as optim

# 設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 訓練循環
for epoch in range(100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    scheduler.step()

    # 評估
    model.eval()
    # ... 評估程式碼
```

---

## 🏗️ 模型架構

### ResNet 系列模型對比

| 模型 | 層數 | 參數量 | FLOPs | Top-1 準確率 (ImageNet) | Top-5 準確率 |
|------|------|--------|-------|-------------------------|--------------|
| **ResNet-18** | 18 | 11.7M | 1.8B | 69.76% | 89.08% |
| **ResNet-34** | 34 | 21.8M | 3.7B | 73.31% | 91.42% |
| **ResNet-50** | 50 | 25.6M | 4.1B | 76.13% | 92.86% |
| **ResNet-101** | 101 | 44.5M | 7.8B | 77.37% | 93.56% |
| **ResNet-152** | 152 | 60.2M | 11.6B | 78.31% | 94.06% |

### ResNet-50 詳細架構

| Stage | Output Size | Layers | Blocks | Channels |
|-------|-------------|--------|--------|----------|
| **Conv1** | 112×112 | conv 7×7, stride=2 | - | 64 |
| **Conv2_x** | 56×56 | max pool 3×3, stride=2<br>bottleneck×3 | 3 | 256 |
| **Conv3_x** | 28×28 | bottleneck×4 | 4 | 512 |
| **Conv4_x** | 14×14 | bottleneck×6 | 6 | 1024 |
| **Conv5_x** | 7×7 | bottleneck×3 | 3 | 2048 |
| **FC** | 1×1 | avg pool + fc | - | 1000 |

**總參數**: 25.6M
**總FLOPs**: 4.1B (for 224×224 input)

### 不同深度的區別

**ResNet-18/34** (淺層):
- 使用基本殘差塊（2 個 3×3 卷積）
- 適合資源受限場景
- 訓練快，推理快

**ResNet-50/101/152** (深層):
- 使用 Bottleneck 塊（1×1 → 3×3 → 1×1）
- 更高的表示能力
- 適合大規模資料集

---

## 📊 實驗結果

### ImageNet 分類 (2015)

| 方法 | 層數 | Top-1 錯誤率 | Top-5 錯誤率 |
|------|------|--------------|--------------|
| VGGNet-16 | 16 | - | 7.32% |
| GoogLeNet | 22 | - | 6.67% |
| **ResNet-34** | 34 | 26.70% | 8.58% |
| **ResNet-50** | 50 | 24.01% | 7.02% |
| **ResNet-101** | 101 | 23.14% | 6.63% |
| **ResNet-152** | 152 | 22.16% | 6.16% |
| **ResNet-152 (ensemble)** | - | - | **3.57%** 🏆 |

### CIFAR-10 分類

| 模型 | 參數量 | 準確率 |
|------|--------|--------|
| ResNet-20 | 0.27M | 91.25% |
| ResNet-32 | 0.46M | 92.49% |
| ResNet-44 | 0.66M | 92.83% |
| ResNet-56 | 0.85M | 93.03% |
| ResNet-110 | 1.7M | 93.57% |
| ResNet-1202 | 19.4M | 92.07% (過擬合) |

### 計算效率

**推論速度** (單 GPU, batch_size=1):

| 模型 | GPU | 推理時間 (ms) | 吞吐量 (imgs/s) |
|------|-----|---------------|----------------|
| ResNet-18 | V100 | 2.1 | 476 |
| ResNet-50 | V100 | 4.5 | 222 |
| ResNet-101 | V100 | 7.8 | 128 |
| ResNet-152 | V100 | 11.2 | 89 |

---

## 🎯 應用場景

### 1. 圖像分類

**標準任務**:
```python
# 使用預訓練 ResNet 進行分類
model = models.resnet50(pretrained=True)
model.eval()

# 預處理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 推理
img = Image.open('cat.jpg')
img_tensor = transform(img).unsqueeze(0)
output = model(img_tensor)
```

### 2. 遷移學習

**微調預訓練模型**:
```python
# 載入預訓練模型
model = models.resnet50(pretrained=True)

# 凍結早期層
for param in model.parameters():
    param.requires_grad = False

# 替換最後的 FC 層
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # 自定義類別數

# 只訓練新的 FC 層
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### 3. 特徵提取

**作為 Backbone**:
```python
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # 移除最後的 FC 層
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # [batch, 2048]
        return x

# 提取特徵用於下游任務
extractor = FeatureExtractor()
features = extractor(images)  # [batch, 2048]
```

### 4. 目標檢測

ResNet 作為 Faster R-CNN, Mask R-CNN 的 backbone：
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 使用 ResNet-50 + FPN 的 Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 檢測
predictions = model(images)
```

### 5. 語義分割

ResNet 用於 DeepLab, FCN 等分割模型：
```python
from torchvision.models.segmentation import deeplabv3_resnet50

# DeepLab v3 + ResNet-50
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# 分割
output = model(image)['out']
```

---

## 📚 參考資源

### 論文與程式碼

- 📄 **原始論文**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 💻 **官方實現 (Caffe)**: [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)
- 🔥 **PyTorch 官方**: [torchvision.models.resnet](https://pytorch.org/vision/stable/models.html#resnet)
- 📦 **TensorFlow/Keras**: [keras.applications.ResNet](https://keras.io/api/applications/resnet/)

### 相關論文

1. **Identity Mappings in Deep Residual Networks** (2016)
   - arXiv:1603.05027
   - ResNet v2，改進的殘差塊設計

2. **Aggregated Residual Transformations (ResNeXt)** (2017)
   - arXiv:1611.05431
   - 引入"基數"維度

3. **Wide Residual Networks** (2016)
   - arXiv:1605.07146
   - 增加寬度而非深度

4. **Densely Connected Networks (DenseNet)** (2017)
   - arXiv:1608.06993
   - 密集連接的變體

### 學習資源

- 🎥 **Kaiming He 演講**: [CVPR 2016 Keynote](https://www.youtube.com/watch?v=1PGLj-uKT1w)
- 📖 **論文解讀**: [ResNet 論文精讀 - 李沐](https://www.bilibili.com/video/BV1P3411y7nn)
- 📚 **博客文章**:
  - [Understanding ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
  - [ResNet 詳解](https://zhuanlan.zhihu.com/p/31852747)

### 資料集

- **ImageNet**: [http://www.image-net.org/](http://www.image-net.org/)
- **CIFAR-10/100**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Places365**: [http://places2.csail.mit.edu/](http://places2.csail.mit.edu/)

---

## 🔬 深入理解

### 為什麼殘差連接有效？

**1. 梯度流動**:
```
傳統網絡: ∂L/∂x = ∂L/∂H · ∂H/∂x
ResNet: ∂L/∂x = ∂L/∂H · (∂F/∂x + 1)

恆等項 "+1" 確保梯度至少有一條直通路徑！
```

**2. 集成效應**:
- ResNet 可以看作多個淺層網絡的集成
- n 個殘差塊 → 2^n 條路徑
- 類似 dropout 的正則化效果

**3. 優化視角**:
- 學習恆等映射比學習零映射更難
- 殘差學習將問題轉化為學習零映射
- 更容易優化，收斂更快

### 關鍵設計決策

**Q1: 為什麼使用 1×1 卷積而非 padding？**
- A: 1×1 卷積可以學習最優的維度映射，而 padding 是固定的

**Q2: 激活函式應該放在加法前還是後？**
- A: 論文實驗表明放在加法後效果更好（ResNet v2 有不同設計）

**Q3: BN 應該放在哪裡？**
- A: 每個卷積層後，加法前

---

## 🤝 貢獻

歡迎提出問題、建議或改進！

如果您發現任何錯誤或有改進建議，請：
1. 提交 Issue
2. 提交 Pull Request
3. 聯繫維護者

---

## 📝 引用

如果您在研究中使用了 ResNet，請引用原始論文：

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

**ResNet v2** (改進版):
```bibtex
@article{he2016identity,
  title={Identity mappings in deep residual networks},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={European conference on computer vision},
  pages={630--645},
  year={2016}
}
```

---

## 📄 授權

本教程遵循原論文和官方程式碼的授權協議。ResNet 論文和程式碼開源免費使用。

---

## 🏆 影響力

**統計資料**:
- 📄 **引用次數**: 150,000+ (截至 2024)
- ⭐ **GitHub Stars**: 100,000+ (各種實現總和)
- 🏅 **獎項**: CVPR 2016 Best Paper Award

**里程碑**:
- 🥇 ImageNet 2015 冠軍
- 🎯 首次超越人類水平的圖像分類
- 🌟 開啟極深網絡時代
- 💡 啟發無數後續研究

---

<div align="center">
  <p><strong>⭐ ResNet 改變了深度學習的歷史！</strong></p>
  <p>📚 深入理解 | 💡 實踐應用 | 🚀 持續創新</p>
  <p><i>最後更新: 2024-11-18</i></p>
</div>
