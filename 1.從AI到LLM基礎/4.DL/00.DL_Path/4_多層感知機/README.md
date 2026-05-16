# 多層感知機 (MLP) 完整學習指南

## 📖 目錄

- [學習路徑](#學習路徑)
- [內容概覽](#內容概覽)
- [快速開始](#快速開始)
- [進階主題](#進階主題)
- [實戰項目](#實戰項目)
- [AI 輔助學習建議](#ai-輔助學習建議)
- [常見問題](#常見問題)
- [學習資源](#學習資源)

---

## 🎯 學習路徑

### 初學者路徑（1-2週）
```
第1-2天：理解基礎概念
├─ 1_mlp.ipynb                    # MLP 基礎理論
├─ 激活函式（ReLU、Sigmoid、Tanh）
└─ 前向傳播機制

第3-4天：動手實作
├─ 2_mlp-scratch.ipynb            # 從零實作
├─ 理解每個組件的作用
└─ 3_mlp-concise.ipynb            # 使用框架實作

第5-7天：過擬合問題
├─ 4_underfit-overfit.ipynb       # 理解過擬合和欠擬合
├─ 5_weight-decay.ipynb           # 權重衰減（L2正則化）
└─ 6_dropout.ipynb                # Dropout 正則化

第8-10天：深入理論
├─ 7_backprop.ipynb               # 反向傳播原理
├─ 8_numerical-stability-and-init.ipynb  # 數值穩定性
└─ 梯度消失和爆炸問題

第11-14天：實戰練習
└─ 10_kaggle-house-price.ipynb   # 完整實戰項目
```

### 進階路徑（1-2週）
```
進階架構技術
├─ 殘差連接（Residual Connections）
├─ 批次正規化（Batch Normalization）
├─ Layer Normalization
└─ 注意力機制基礎

超參數調優
├─ 系統化調參策略
├─ 學習率調度
├─ 優化器選擇（SGD、Adam、AdamW）
└─ 早停策略

模型診斷與優化
├─ 學習曲線分析
├─ 梯度監控
├─ 權重可視化
└─ 性能瓶頸分析
```

---

## 📚 內容概覽

### 核心文件

| 文件 | 難度 | 學習時間 | 主要內容 |
|------|------|----------|----------|
| `0_index.ipynb` | ⭐ | 10分鐘 | 章節總覽 |
| `1_mlp.ipynb` | ⭐⭐ | 2-3小時 | MLP基礎、激活函式 |
| `2_mlp-scratch.ipynb` | ⭐⭐⭐ | 3-4小時 | 從零實作MLP |
| `3_mlp-concise.ipynb` | ⭐⭐ | 1-2小時 | 框架實作MLP |
| `4_underfit-overfit.ipynb` | ⭐⭐⭐ | 2-3小時 | 模型選擇、過擬合 |
| `5_weight-decay.ipynb` | ⭐⭐⭐ | 2-3小時 | L2正則化 |
| `6_dropout.ipynb` | ⭐⭐⭐ | 2-3小時 | Dropout技術 |
| `7_backprop.ipynb` | ⭐⭐⭐⭐ | 3-4小時 | 反向傳播原理 |
| `8_numerical-stability-and-init.ipynb` | ⭐⭐⭐⭐ | 2-3小時 | 數值穩定性 |
| `10_kaggle-house-price.ipynb` | ⭐⭐⭐ | 4-6小時 | 完整實戰項目 |

---

## 🚀 快速開始

### 環境設置

```bash
# 1. 安裝必要的套件
pip install torch torchvision matplotlib numpy pandas scikit-learn

# 2. 驗證安裝
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 3. 檢查 GPU 可用性（可選）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 第一個 MLP 模型（5分鐘實作）

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 準備資料
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# 2. 定義模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 3. 訓練模型
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} completed')

print("✅ 第一個模型訓練完成！")
```

---

## 🎓 進階主題

### 1. 現代 MLP 架構技術

#### 殘差連接（Residual Connections）
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return x + self.activation(self.linear(x))  # 殘差連接
```

#### 批次正規化（Batch Normalization）
```python
class MLPWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),  # 批次正規化
            nn.ReLU(),
            nn.Linear(256, 10)
        )
```

### 2. 超參數調優策略

#### 學習率調度
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 方法1：餘弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 方法2：基於驗證損失調整
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
```

#### 系統化調參
```python
# 推薦的調參順序
hyperparams = {
    '1_學習率': [0.001, 0.01, 0.1],
    '2_批次大小': [32, 64, 128, 256],
    '3_隱藏層數': [1, 2, 3],
    '4_隱藏單元數': [64, 128, 256, 512],
    '5_正則化': ['dropout', 'weight_decay', 'both'],
}
```

### 3. 模型診斷工具

#### 梯度監控
```python
def monitor_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
```

#### 學習曲線可視化
```python
import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
```

---

## 💡 實戰項目

### 項目1：圖像分類（Fashion-MNIST）
- **難度**：⭐⭐
- **目標**：達到 90% 測試準確率
- **文件**：`2_mlp-scratch.ipynb`, `3_mlp-concise.ipynb`

### 項目2：房價預測（Kaggle）
- **難度**：⭐⭐⭐
- **目標**：理解完整的機器學習流程
- **文件**：`10_kaggle-house-price.ipynb`

### 項目3：手寫數字識別（MNIST）
- **難度**：⭐⭐
- **目標**：達到 98% 測試準確率
- **建議**：作為入門練習

### 項目4：多標籤分類
- **難度**：⭐⭐⭐
- **目標**：處理複雜的分類問題
- **擴展**：可以使用真實資料集

---

## 🤖 AI 輔助學習建議

### 使用 ChatGPT/Claude 加速學習

#### 1. 概念理解
```
提示詞範例：
"請用簡單的例子解釋反向傳播演算法，包括：
1. 基本原理
2. 數學推導
3. 程式碼實現
4. 常見誤區"
```

#### 2. 程式碼調試
```
提示詞範例：
"我的 MLP 模型出現梯度消失問題，以下是我的程式碼：
[貼上程式碼]
請幫我：
1. 診斷問題
2. 提供解決方案
3. 解釋原因"
```

#### 3. 練習題解答
```
提示詞範例：
"請幫我檢查以下練習題的解答是否正確：
[貼上問題和解答]
如果有錯誤，請：
1. 指出錯誤處
2. 提供正確解答
3. 解釋為什麼"
```

#### 4. 程式碼優化
```
提示詞範例：
"請幫我優化以下 MLP 訓練程式碼：
[貼上程式碼]
優化方向：
1. 訓練速度
2. 記憶體使用
3. 程式碼可讀性
4. 最佳實踐"
```

### 使用 GitHub Copilot

```python
# 提示：只需要寫註釋，Copilot 會自動生成程式碼

# 建立一個包含 Dropout 和 Batch Normalization 的 MLP 模型
# 輸入維度：784，隱藏層：[512, 256, 128]，輸出維度：10
# [Copilot 會自動生成完整程式碼]
```

### 使用 Cursor AI

1. **程式碼補全**：智能預測下一步要寫的程式碼
2. **錯誤修復**：自動檢測並修復常見錯誤
3. **文檔生成**：自動生成程式碼註釋和文檔

---

## ❓ 常見問題

### Q1: 為什麼我的模型不收斂？

**可能原因和解決方案**：

```python
# 1. 學習率太大
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 降低學習率

# 2. 沒有正規化資料
transform = transforms.Normalize((0.5,), (0.5,))  # 添加正規化

# 3. 權重初始化不當
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
model.apply(init_weights)

# 4. 梯度爆炸
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
```

### Q2: 如何選擇隱藏層的數量和大小？

**經驗法則**：

```python
# 起點建議
input_dim = 784
output_dim = 10

# 保守選擇（通常有效）
hidden_dims = [
    input_dim // 2,      # 第一層：392
    input_dim // 4,      # 第二層：196
]

# 實驗建議的範圍
experiments = {
    'small': [128],
    'medium': [256, 128],
    'large': [512, 256, 128],
    'very_large': [1024, 512, 256],
}
```

### Q3: 過擬合了怎麼辦？

**解決策略（按優先級）**：

```python
# 1. 添加 Dropout
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # ← 添加 Dropout
    nn.Linear(256, 10)
)

# 2. 使用權重衰減
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    weight_decay=1e-4  # ← L2 正則化
)

# 3. 早停
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(max_epochs):
    val_loss = validate(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

# 4. 資料增強（如果適用）
# 5. 減少模型複雜度
# 6. 增加訓練資料
```

### Q4: 訓練太慢怎麼辦？

**加速策略**：

```python
# 1. 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. 增加批次大小
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# 3. 使用更快的優化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 使用混合精度訓練（GPU）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 5. 資料加載優化
train_loader = DataLoader(
    train_data,
    batch_size=256,
    shuffle=True,
    num_workers=4,     # ← 多線程加載
    pin_memory=True    # ← 加速 GPU 傳輸
)
```

---

## 📚 學習資源

### 官方文檔
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [D2L - Dive into Deep Learning](https://d2l.ai/)

### 推薦書籍
1. **《深度學習》** - Ian Goodfellow（深度學習聖經）
2. **《動手學深度學習》** - 李沐等（本資料夾基於此書）
3. **《神經網絡與深度學習》** - 邱錫鵬

### 在線課程
1. **Fast.ai** - Practical Deep Learning for Coders
2. **Stanford CS231n** - Convolutional Neural Networks
3. **吳恩達深度學習專項課程** - Coursera

### 實踐平台
1. **Kaggle** - 實戰競賽和資料集
2. **Google Colab** - 免費 GPU 訓練環境
3. **Paperspace Gradient** - 雲端機器學習平台

### 社群資源
- [PyTorch 論壇](https://discuss.pytorch.org/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)

---

## 🎯 學習檢查清單

### 基礎概念 ✓
- [ ] 理解 MLP 的基本架構
- [ ] 掌握常見激活函式（ReLU、Sigmoid、Tanh）
- [ ] 理解前向傳播和反向傳播
- [ ] 能夠從零實現一個簡單的 MLP

### 實踐能力 ✓
- [ ] 使用 PyTorch 實現 MLP
- [ ] 能夠訓練模型並評估性能
- [ ] 理解並應用正則化技術
- [ ] 能夠調試常見的訓練問題

### 進階技能 ✓
- [ ] 理解梯度消失和爆炸問題
- [ ] 掌握不同的初始化策略
- [ ] 能夠進行系統化的超參數調優
- [ ] 完成至少一個完整的實戰項目

### 理論深度 ✓
- [ ] 理解反向傳播的數學原理
- [ ] 理解不同正則化方法的理論基礎
- [ ] 能夠分析模型的學習曲線
- [ ] 理解優化演算法的工作原理

---

## 🤝 貢獻指南

如果你發現任何錯誤或有改進建議，歡迎：
1. 提交 Issue
2. 發起 Pull Request
3. 分享你的學習心得

---

## 📄 授權

本資料基於 D2L（Dive into Deep Learning）開源教材，遵循其原始授權。

---

## 📮 聯繫方式

如有問題或建議，歡迎通過以下方式聯繫：
- GitHub Issues
- Email: [你的郵箱]

---

**最後更新時間**：2025-01-18

**祝學習順利！🎉**
