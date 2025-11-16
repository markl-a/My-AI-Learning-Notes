# PyTorch 個人複習筆記

> 🔄 **最後更新：** 2025-01
> 📊 **完成度：** 約 30%
> 🎯 **推薦版本：** PyTorch 2.0+
> 🔥 **特色：** 動態計算圖、研究友好、生產就緒

---

## ⚠️ 內容狀態說明

PyTorch 是目前深度學習研究和生產的主流框架之一，以其靈活性和易用性著稱。

**已完成的內容：**
- ✅ 1. ResNet 實作
- ✅ 2. Transformer 程式碼詳解
- ✅ 3. Segment Anything 2 (SAM2) - **2024 最新技術** ⭐

**規劃中的內容正在逐步補充。**

---

## 🌟 為什麼選擇 PyTorch？

### 核心優勢
- ✅ **動態計算圖：** 更直覺的編程體驗
- ✅ **Python 原生：** 與 Python 生態完美整合
- ✅ **研究首選：** 頂級會議論文的主要框架
- ✅ **生產部署：** TorchServe、ONNX、TorchScript
- ✅ **社群活躍：** 豐富的預訓練模型和工具

### PyTorch 2.0+ 新特性
- 🚀 **`torch.compile`：** 自動優化，速度提升達 2x
- 🚀 **改進的分散式訓練**
- 🚀 **更好的 ONNX 支援**
- 🚀 **原生 AOTAutograd**

---

## 📚 完整學習規劃

### 🎓 1. PyTorch 簡介與安裝
**狀態：規劃中**

#### 1.1 環境設置
- ⏳ PyTorch 安裝（CPU/GPU）
- ⏳ CUDA 配置
- ⏳ 虛擬環境管理
- ⏳ IDE 配置（VS Code, PyCharm, Jupyter）

#### 1.2 快速入門
- ⏳ Tensor 基礎操作
- ⏳ 自動微分（Autograd）
- ⏳ 第一個神經網路

---

### 🔥 2. PyTorch 核心概念
**狀態：規劃中**

#### 2.1 Tensor 操作
- ⏳ Tensor 創建與初始化
- ⏳ Tensor 運算（加減乘除、矩陣運算）
- ⏳ Tensor 形狀操作（reshape, view, transpose）
- ⏳ 索引與切片
- ⏳ GPU 加速

#### 2.2 自動微分
- ⏳ `torch.autograd` 機制
- ⏳ 梯度計算
- ⏳ 反向傳播
- ⏳ `torch.no_grad()` 和 `requires_grad`

#### 2.3 神經網路模組（nn.Module）
- ⏳ 層定義
- ⏳ 前向傳播
- ⏳ 參數管理
- ⏳ 模型組合

---

### 📊 3. 資料處理模組
**狀態：規劃中**

#### 3.1 Dataset 與 DataLoader
- ⏳ `torch.utils.data.Dataset` 自定義
- ⏳ `DataLoader` 批次處理
- ⏳ 資料增強（Transforms）
- ⏳ 多進程資料載入

#### 3.2 常用資料集
- ⏳ MNIST/FashionMNIST
- ⏳ CIFAR-10/100
- ⏳ ImageNet
- ⏳ 自定義資料集

#### 3.3 資料增強
- ⏳ `torchvision.transforms`
- ⏳ `albumentations` 整合
- ⏳ 進階增強技術

---

### 🏗️ 4. 模型建構模組
**狀態：部分完成**

#### 4.1 常用層
- ⏳ Linear（全連接層）
- ⏳ Conv2d（卷積層）
- ⏳ BatchNorm（批次正規化）
- ⏳ Dropout
- ⏳ 激活函數（ReLU, GELU, SiLU）

#### 4.2 經典架構實作
- ✅ ResNet（已完成）
- ✅ Transformer（已完成）
- ⏳ VGG
- ⏳ DenseNet
- ⏳ MobileNet

#### 4.3 現代架構
- ✅ Segment Anything 2 (SAM2)
- ⏳ Vision Transformer (ViT)
- ⏳ Swin Transformer
- ⏳ ConvNeXt
- ⏳ EfficientNet V2

**檔案位置：**
- `1.ResNet/`
- `2.Transformer程式碼詳解/`
- `3.Segment Anything 2/`

---

### ⚡ 5. 優化模組
**狀態：規劃中**

#### 5.1 損失函數
- ⏳ CrossEntropyLoss
- ⏳ MSELoss
- ⏳ BCELoss
- ⏳ 自定義損失函數

#### 5.2 優化器
- ⏳ SGD
- ⏳ Adam/AdamW
- ⏳ RMSprop
- ⏳ 學習率調度器（Scheduler）

#### 5.3 進階優化技術
- ⏳ 混合精度訓練（AMP）
- ⏳ 梯度累積
- ⏳ 梯度裁剪
- ⏳ 權重衰減

---

### 📈 6. 視覺化與監控
**狀態：規劃中**

#### 6.1 訓練監控
- ⏳ TensorBoard 整合
- ⏳ Weights & Biases (wandb)
- ⏳ 損失與準確率曲線

#### 6.2 模型視覺化
- ⏳ 模型結構視覺化
- ⏳ 特徵圖視覺化
- ⏳ 注意力圖視覺化
- ⏳ Grad-CAM

---

### 🛠️ 7. PyTorch 技巧彙總
**狀態：規劃中**

#### 7.1 訓練技巧
- ⏳ 早停（Early Stopping）
- ⏳ 模型檢查點（Checkpoint）
- ⏳ 學習率查找
- ⏳ 遷移學習策略

#### 7.2 除錯技巧
- ⏳ 梯度檢查
- ⏳ 過擬合小批次資料
- ⏳ 記憶體優化
- ⏳ 常見錯誤排查

#### 7.3 效能優化
- ⏳ 混合精度訓練
- ⏳ 資料載入優化
- ⏳ 模型編譯（`torch.compile`）
- ⏳ 分散式訓練

---

### 🖼️ 8. 計算機視覺專案
**狀態：部分完成**

#### 8.1 圖像分類
- ⏳ MNIST 手寫數字識別
- ⏳ CIFAR-10 圖像分類
- ⏳ 遷移學習實戰

#### 8.2 目標檢測
- ⏳ YOLO 系列
- ⏳ Faster R-CNN
- ⏳ DETR（Detection Transformer）

#### 8.3 圖像分割
- ✅ SAM2（Segment Anything 2）
- ⏳ U-Net
- ⏳ DeepLab
- ⏳ Mask R-CNN

#### 8.4 其他視覺任務
- ⏳ 圖像生成（GAN, Diffusion）
- ⏳ 風格遷移
- ⏳ 超解析度
- ⏳ 影片理解

---

### 💬 9. 自然語言處理專案
**狀態：規劃中**

#### 9.1 文本分類
- ⏳ 情感分析
- ⏳ 新聞分類
- ⏳ 垃圾郵件檢測

#### 9.2 序列建模
- ⏳ RNN/LSTM/GRU
- ⏳ Transformer
- ⏳ 命名實體識別（NER）

#### 9.3 生成任務
- ⏳ 文本生成
- ⏳ 機器翻譯
- ⏳ 文本摘要

---

### 🤖 10. 大語言模型應用
**狀態：規劃中**

#### 10.1 預訓練模型使用
- ⏳ Hugging Face Transformers 整合
- ⏳ BERT/GPT 系列
- ⏳ T5, BART

#### 10.2 模型微調
- ⏳ 全參數微調
- ⏳ LoRA（低秩適應）
- ⏳ QLoRA（量化 LoRA）
- ⏳ Adapter 方法

#### 10.3 推理優化
- ⏳ 模型量化
- ⏳ KV Cache 優化
- ⏳ Flash Attention
- ⏳ 批次推理

**建議參考：** `../05.Transformer_lib/`

---

### 🚀 11. 進階主題
**狀態：規劃中**

#### 11.1 分散式訓練
- ⏳ DataParallel (DP)
- ⏳ DistributedDataParallel (DDP)
- ⏳ Fully Sharded Data Parallel (FSDP)
- ⏳ DeepSpeed 整合

#### 11.2 模型部署
- ⏳ TorchScript
- ⏳ ONNX 導出
- ⏳ TorchServe
- ⏳ 邊緣裝置部署

#### 11.3 其他進階技術
- ⏳ 自定義 CUDA 核心
- ⏳ JIT 編譯
- ⏳ 圖神經網路（PyG）
- ⏳ 強化學習（PyTorch RL）

---

## 🛠️ PyTorch 2.0+ 最佳實踐

### 推薦工作流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. 定義模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()

# 2. 編譯模型（PyTorch 2.0+）
model = torch.compile(model)  # 自動優化！

# 3. 設置優化器和損失函數
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. 訓練循環
for epoch in range(epochs):
    for batch in dataloader:
        # 前向傳播
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. 儲存模型
torch.save(model.state_dict(), 'model.pth')
```

### 混合精度訓練（AMP）

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 自動混合精度
    with autocast(device_type='cuda'):
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])

    # 梯度縮放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 分散式訓練（DDP）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化進程組
dist.init_process_group(backend='nccl')

# 包裝模型
model = DDP(model, device_ids=[local_rank])

# 訓練（與單卡相同）
for batch in dataloader:
    loss = train_step(model, batch)
```

---

## 📚 學習資源

### 官方資源
- [PyTorch 官方文檔](https://pytorch.org/docs/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 論壇](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)

### 預訓練模型
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [Hugging Face](https://huggingface.co/models)
- [PyTorch Hub](https://pytorch.org/hub/)

### 社群資源
- [Papers with Code](https://paperswithcode.com/)
- [PyTorch Lightning](https://lightning.ai/)（高階封裝）
- [timm](https://github.com/huggingface/pytorch-image-models)（圖像模型庫）

### 推薦課程
- [Stanford CS231n](http://cs231n.stanford.edu/)（計算機視覺）
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/)（NLP）
- [Fast.ai](https://www.fast.ai/)（實戰導向）

---

## 🎯 學習建議

### 對於初學者：
1. 掌握 Tensor 基礎操作
2. 理解自動微分機制
3. 實作簡單的 MNIST 分類器
4. 學習使用 DataLoader
5. 完成 2-3 個小專案

### 對於中級學習者：
1. 深入理解 nn.Module
2. 實作經典模型（ResNet, Transformer）
3. 掌握訓練技巧（學習率調度、正規化）
4. 學習遷移學習
5. 探索混合精度訓練

### 對於進階學習者：
1. 掌握分散式訓練（DDP, FSDP）
2. 研究最新模型架構
3. 優化推理效能
4. 貢獻開源專案
5. 發表研究論文

---

## 🔗 相關章節

- **深度學習基礎：** 查看 `../00.DL_Path/`
- **TensorFlow 2：** 查看 `../01.Tensorflow2/`
- **Transformer 庫：** 查看 `../05.Transformer_lib/`
- **實戰項目：** 查看 `../04.Ultralytics/`

---

## 🌟 特色內容

### Segment Anything 2 (SAM2)
**2024 年 Meta 最新發布的通用分割模型**

- ✅ 論文解讀
- ✅ 影片分割實作
- ✅ 物體識別應用
- ✅ 自動遮罩生成

**檔案位置：** `3.Segment Anything 2/`

這是本筆記的亮點內容，展示了如何使用最新的計算機視覺技術！

---

## 💡 快速開始

```bash
# 安裝 PyTorch（CPU 版本）
pip install torch torchvision torchaudio

# 安裝 PyTorch（GPU 版本 - CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 驗證安裝
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 📝 注意事項

1. **版本選擇：** 建議使用 PyTorch 2.0 或更高版本以獲得最佳性能
2. **CUDA 配置：** GPU 訓練需要正確安裝 CUDA 和 cuDNN
3. **記憶體管理：** 注意批次大小，避免 OOM 錯誤
4. **可重現性：** 設置隨機種子確保實驗可重現

```python
# 設置隨機種子
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

**持續更新中...** 📖

**PyTorch：讓深度學習研究變得簡單！** 🔥
