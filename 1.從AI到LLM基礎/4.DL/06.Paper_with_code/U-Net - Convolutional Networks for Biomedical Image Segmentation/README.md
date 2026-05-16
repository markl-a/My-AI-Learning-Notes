# U-Net - 醫學圖像分割

> **論文**: U-Net: Convolutional Networks for Biomedical Image Segmentation
>
> **作者**: Olaf Ronneberger, Philipp Fischer, Thomas Brox (University of Freiburg)
>
> **發表**: MICCAI 2015
>
> **論文鏈接**: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
>
> **引用次數**: 90,000+ (截至 2024)

---

## 🎯 簡介

**U-Net** 是醫學圖像分割領域最具影響力的論文之一，其優雅的 **U 型編碼器-解碼器架構** 和 **跳躍連接** 設計，不僅在醫學圖像領域取得突破，更廣泛應用於各類分割任務。

### 核心問題

醫學圖像分割面臨的挑戰：
- ❌ **訓練資料少**: 醫學圖像標註成本高
- ❌ **細節重要**: 需要精確的邊界定位
- ❌ **上下文**: 需要全局和局部資訊

### U-Net 的解決方案

```
編碼器-解碼器 + 跳躍連接

編碼器 (Contracting Path):
↓ 捕捉上下文資訊
↓ 逐步降低分辨率

解碼器 (Expanding Path):
↑ 精確定位
↑ 逐步恢復分辨率

跳躍連接:
→ 融合低層細節和高層語義
```

---

## 💡 核心創新

### 1. U 型架構

```
Input (572×572)
    ↓
┌───────── Encoder ─────────┐
│ Conv 3×3 (64)              │
│ Conv 3×3 (64)              │ → 跳躍連接 1 →
│ MaxPool 2×2                │                ↓
│ Conv 3×3 (128)             │                ↓
│ Conv 3×3 (128)             │ → 跳躍連接 2 → UpConv + Concat
│ MaxPool 2×2                │                ↓
│ Conv 3×3 (256)             │                ↓
│ Conv 3×3 (256)             │ → 跳躍連接 3 → UpConv + Concat
│ MaxPool 2×2                │                ↓
│ Conv 3×3 (512)             │                ↓
│ Conv 3×3 (512)             │ → 跳躍連接 4 → UpConv + Concat
│ MaxPool 2×2                │                ↓
└────────────────────────────┘                │
    ↓                                         │
Bottleneck                                    │
Conv 3×3 (1024)                               │
Conv 3×3 (1024)                               │
    ↓                                         │
┌───────── Decoder ─────────┐                │
│ UpConv 2×2 (512)           │ ←─────────────┘
│ Conv 3×3 (512)             │
│ Conv 3×3 (512)             │
│ UpConv 2×2 (256)           │
│ ...                        │
└────────────────────────────┘
    ↓
Output (388×388)
```

### 2. 跳躍連接 (Skip Connections)

**為什麼重要？**
- ✅ 保留空間細節資訊
- ✅ 緩解梯度消失
- ✅ 融合多尺度特徵

```python
# 跳躍連接的實現
def forward(self, x):
    # Encoder
    enc1 = self.enc_conv1(x)
    enc2 = self.enc_conv2(self.pool(enc1))
    enc3 = self.enc_conv3(self.pool(enc2))
    enc4 = self.enc_conv4(self.pool(enc3))

    # Bottleneck
    bottleneck = self.bottleneck(self.pool(enc4))

    # Decoder (帶跳躍連接)
    dec4 = self.dec_conv4(torch.cat([self.up4(bottleneck), enc4], dim=1))
    dec3 = self.dec_conv3(torch.cat([self.up3(dec4), enc3], dim=1))
    dec2 = self.dec_conv2(torch.cat([self.up2(dec3), enc2], dim=1))
    dec1 = self.dec_conv1(torch.cat([self.up1(dec2), enc1], dim=1))

    return self.final_conv(dec1)
```

### 3. 資料增強策略

**彈性形變 (Elastic Deformation)**:
- 模擬組織變形
- 大幅增加訓練樣本多樣性

**加權損失函式**:
```python
# 邊界加權損失
weight_map = compute_weight_map(labels)
loss = weighted_cross_entropy(predictions, labels, weight_map)
```

---

## 🏗️ 完整實現

### PyTorch 實現

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """兩個連續的 3×3 卷積"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (下採樣路徑)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder (上採樣路徑)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # 最終輸出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反轉順序

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # UpConv
            skip_connection = skip_connections[idx//2]

            # 處理尺寸不匹配
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:])

            # 連接
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # DoubleConv

        return self.final_conv(x)

# 使用
model = UNet(in_channels=3, out_channels=1)
```

---

## 🚀 快速開始

### 使用 segmentation_models_pytorch

```bash
pip install segmentation-models-pytorch
```

```python
import segmentation_models_pytorch as smp

# 建立模型
model = smp.Unet(
    encoder_name="resnet34",        # 使用 ResNet34 作為 encoder
    encoder_weights="imagenet",     # 使用 ImageNet 預訓練權重
    in_channels=3,                  # 輸入通道數
    classes=1,                      # 輸出類別數
)

# 訓練
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = smp.losses.DiceLoss(mode='binary')

for epoch in range(num_epochs):
    for images, masks in train_loader:
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向傳播
        loss.backward()
        optimizer.step()
```

### 從零訓練

```python
# 資料增強
from albumentations import (
    Compose, RandomRotate90, Flip, Transpose,
    ElasticTransform, GridDistortion, OpticalDistortion
)

transform = Compose([
    RandomRotate90(),
    Flip(),
    Transpose(),
    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    GridDistortion(),
    OpticalDistortion(distort_limit=2, shift_limit=0.5),
])

# 訓練循環
model = UNet(in_channels=1, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(epochs):
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 📊 實驗結果

### ISBI Cell Tracking Challenge 2015

| 指標 | U-Net | 第二名 |
|------|-------|--------|
| **IOU** | **92%** | 83% |
| **Warping Error** | **0.0003** | 0.0006 |

### 醫學圖像分割基準

| 資料集 | 任務 | Dice 係數 |
|--------|------|-----------|
| EM Segmentation | 神經元分割 | 0.98 |
| DRIVE | 視網膜血管 | 0.95 |
| ISIC 2018 | 皮膚病變 | 0.86 |

---

## 🎯 應用場景

### 1. 醫學圖像分割

```python
# 肺部 CT 分割
model = UNet(in_channels=1, out_channels=3)  # 背景、左肺、右肺

# 預測
with torch.no_grad():
    prediction = model(ct_image)
    segmentation = torch.argmax(prediction, dim=1)
```

**應用領域**:
- 🫁 肺部分割
- 🧠 腦腫瘤分割
- ❤️ 心臟分割
- 🦴 骨骼分割

### 2. 細胞分割與追蹤

- 顯微鏡圖像分析
- 細胞計數
- 細胞形態分析

### 3. 衛星圖像分割

```python
# 建築物提取
model = UNet(in_channels=3, out_channels=2)  # RGB → 建築物/背景
```

- 建築物檢測
- 道路提取
- 土地利用分類

### 4. 自然圖像分割

- 人像分割（背景虛化）
- 物體分割
- 語義分割

---

## 🌟 U-Net 變體

### 1. U-Net++

**改進**: 重新設計的跳躍連接
```
密集連接 + 深度監督
```

### 2. Attention U-Net

**改進**: 注意力門控機制
```python
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi
```

### 3. ResUNet

**改進**: 殘差塊替換普通卷積

### 4. U-Net 3D

**改進**: 擴展到 3D 醫學圖像
```python
nn.Conv3d(in_channels, out_channels, kernel_size=3)
```

---

## 📚 參考資源

### 論文

- 📄 **原始論文**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- 📄 **U-Net++**: [arXiv:1807.10165](https://arxiv.org/abs/1807.10165)
- 📄 **Attention U-Net**: [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)

### 程式碼實現

- 🔥 **segmentation_models_pytorch**: [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- 🐍 **milesial/Pytorch-UNet**: [github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- 📦 **TensorFlow 實現**: [zhixuhao/unet](https://github.com/zhixuhao/unet)

### 學習資源

- 📖 **U-Net 詳解**: [towardsdatascience.com/unet](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)
- 🎥 **影片教程**: [YouTube - U-Net Explained](https://www.youtube.com/watch?v=oLvmLJkmXuc)

---

## 📝 引用

```bibtex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2015},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

---

<div align="center">
  <p><strong>⭐ U-Net: 簡單而優雅的分割架構！</strong></p>
  <p>🏥 醫學圖像 | 🛰️ 遙感圖像 | 📸 自然圖像</p>
  <p><i>最後更新: 2024-11-18</i></p>
</div>
