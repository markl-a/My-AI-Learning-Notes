# 深度學習計算 (Deep Learning Computation)

> **更新日期**: 2024-2025
> **版本**: v3.0
> **難度**: ⭐⭐⭐ 中級到進階

## 📚 章節概述

深度學習計算是從「基礎用戶」進階到「高級用戶」的關鍵章節。本章深入探索深度學習框架的內部工作原理，掌握這些知識將使你能夠：

- 🏗️ 靈活構建複雜的神經網絡架構
- 🎛️ 精確控制模型參數的初始化和管理
- 🔧 創建自定義層和模型組件
- 💾 高效地保存和加載模型
- ⚡ 充分利用GPU/TPU等加速器
- 🚀 優化模型的計算性能

## 📖 學習路徑

### 基礎篇 (Foundation)

#### 1. 層和區塊 (Layers and Blocks)
📄 **文件**: `1_model-construction.ipynb`

**學習目標**:
- 理解PyTorch中的模塊化設計理念
- 掌握`nn.Module`和`nn.Sequential`的使用
- 學會構建自定義區塊
- 理解模型的層次化組織

**核心概念**:
```python
# 自定義區塊的基本結構
class CustomBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
```

**實際應用**:
- ResNet的殘差塊設計
- Transformer的多頭注意力塊
- UNet的編碼器-解碼器結構

**練習題**: ✅ 已完成
- 實現平行區塊 (Parallel Block)
- 創建可複製的網路實例
- 探索不同的模塊組合方式

---

#### 2. 參數管理 (Parameter Management)
📄 **文件**: `2_parameters.ipynb`

**學習目標**:
- 訪問和修改模型參數
- 理解參數初始化的重要性
- 掌握參數共享技術
- 實現自定義初始化策略

**核心技能**:
```python
# 參數訪問模式
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# 自定義初始化
def custom_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

model.apply(custom_init)
```

**重要概念**:
- **Xavier/Glorot初始化**: 適用於tanh/sigmoid激活函數
- **Kaiming/He初始化**: 適用於ReLU激活函數
- **參數共享**: 減少模型大小，提高泛化能力
- **參數綁定**: 在多個層之間共享權重

**練習題**: ✅ 已完成
- 探索不同初始化方法的效果
- 訓練包含共享參數的模型
- 觀察參數和梯度的變化

---

#### 3. 延遲初始化 (Deferred Initialization)
📄 **文件**: `3_deferred-init.ipynb`

**學習目標**:
- 理解延遲初始化的工作原理
- 掌握動態形狀推斷機制
- 學會處理可變輸入維度

**核心概念**:
- 框架自動推斷層的輸入輸出維度
- 簡化模型定義過程
- 提高代碼的靈活性和可維護性

**優勢**:
```python
# 不需要明確指定每一層的輸入維度
net = nn.Sequential(
    nn.LazyLinear(256),  # 自動推斷輸入維度
    nn.ReLU(),
    nn.LazyLinear(10)
)
```

**練習題**: ✅ 已完成
- 處理不匹配的維度情況
- 實現動態維度的模型

---

### 進階篇 (Advanced)

#### 4. 自定義層 (Custom Layers)
📄 **文件**: `4_custom-layer.ipynb`

**學習目標**:
- 創建無參數的自定義層
- 實現帶參數的自定義層
- 集成自定義層到複雜模型中

**實用案例**:
```python
# 張量降維層
class TensorReduction(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_features, in_features, out_features)
        )

    def forward(self, x):
        return torch.einsum('bi,bj,ijk->bk', x, x, self.weight)

# 傅立葉變換層
class FFTLayer(nn.Module):
    def forward(self, x):
        fft_result = torch.fft.fft(x, dim=-1)
        return fft_result[..., :(x.shape[-1] + 1) // 2]
```

**應用場景**:
- 🔬 科學計算中的特殊變換
- 🎨 圖像處理中的自定義濾波器
- 🎵 音頻處理中的頻域操作
- 📊 特徵工程中的非線性變換

**練習題**: ✅ 已完成
- 實現張量降維層
- 實現傅立葉係數提取層

---

#### 5. 文件讀寫 (File I/O)
📄 **文件**: `5_read-write.ipynb`

**學習目標**:
- 保存和加載模型參數
- 實現模型檢查點機制
- 處理跨平台模型部署

**最佳實踐**:
```python
# 完整的檢查點保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc
}
torch.save(checkpoint, 'checkpoint.pth')

# 加載檢查點
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

**生產環境考慮**:
- ✅ 版本控制和向後兼容性
- ✅ 模型元數據的保存
- ✅ 跨設備的模型遷移
- ✅ 安全的序列化和反序列化

**練習題**: ✅ 已完成
- 實現帶版本控制的模型保存
- 復用預訓練模型的部分層

---

#### 6. GPU計算 (GPU Computing) 🔥
📄 **文件**: `6_use-gpu.ipynb`

**學習目標**:
- 掌握多平台加速器的使用
- 優化GPU內存使用
- 理解數據傳輸開銷
- 實現高效的設備管理

**2024-2025年更新重點**:

🎯 **多平台支持**:
- **CUDA** - NVIDIA GPU (最成熟)
- **MPS** - Apple Silicon (M1/M2/M3/M4)
- **ROCm** - AMD GPU
- **XLA** - Google TPU

**跨平台代碼模式**:
```python
def get_device():
    """自動選擇最佳設備"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
model = model.to(device)
data = data.to(device)
```

**性能優化技巧**:
1. **最小化設備間傳輸**
   ```python
   # ❌ 不好的做法
   for batch in dataloader:
       data = batch.to(device)
       output = model(data)
       loss = loss.cpu()  # 頻繁傳輸

   # ✅ 好的做法
   for batch in dataloader:
       data = batch.to(device)
       output = model(data)
       # 在GPU上累積損失，最後一次性傳輸
   ```

2. **使用混合精度訓練**
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   for data, target in dataloader:
       with autocast():
           output = model(data)
           loss = criterion(output, target)

       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

3. **固定記憶體 (Pinned Memory)**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       pin_memory=True  # 加速CPU到GPU傳輸
   )
   ```

**Apple Silicon特別說明**:
```python
# MPS設備檢測
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("✅ MPS可用 - 使用Apple Silicon GPU加速")
else:
    device = torch.device('cpu')
    print("ℹ️ MPS不可用 - 使用CPU")
```

**性能對比參考**:
- **CPU**: 基準性能
- **MPS** (Apple Silicon): 3-10x 加速
- **CUDA** (NVIDIA GPU): 10-100x 加速

**練習題**: ✅ 已完成
- 比較CPU vs GPU的計算速度
- 測量數據傳輸開銷
- 實現多GPU並行計算
- 在MPS上訓練模型

---

## 🚀 新增進階主題

### 7. 模型部署與生產環境 (Coming Soon)
📄 **文件**: `7_deployment.ipynb`

**計劃內容**:
- 模型導出 (ONNX, TorchScript)
- 模型服務化 (TorchServe, FastAPI)
- 邊緣設備部署
- 移動端優化

### 8. 分布式訓練 (Coming Soon)
📄 **文件**: `8_distributed-training.ipynb`

**計劃內容**:
- 數據並行 (DataParallel, DistributedDataParallel)
- 模型並行 (Pipeline Parallelism)
- 混合並行策略
- 多節點訓練

### 9. 模型壓縮與優化 (Coming Soon)
📄 **文件**: `9_model-optimization.ipynb`

**計劃內容**:
- 量化 (Quantization)
- 剪枝 (Pruning)
- 知識蒸餾 (Knowledge Distillation)
- 神經架構搜索 (NAS)

### 10. 調試與性能分析 (Coming Soon)
📄 **文件**: `10_profiling-debugging.ipynb`

**計劃內容**:
- PyTorch Profiler使用
- 內存分析工具
- 梯度檢查
- 性能瓶頸診斷

---

## 🤖 AI輔助學習工具

### 1. GitHub Copilot
**用途**: 代碼補全和生成
```python
# Copilot可以幫助你快速生成模型架構
# 只需輸入註釋，它會建議完整的實現

# 創建一個帶殘差連接的自定義塊
# [讓Copilot幫你完成]
```

### 2. ChatGPT / Claude
**用途**: 概念解釋和代碼審查
- 解釋複雜的深度學習概念
- 審查你的代碼並提供改進建議
- 生成測試用例
- 調試錯誤信息

### 3. PyTorch官方文檔的AI搜索
**鏈接**: https://pytorch.org/docs/stable/index.html
- 使用語義搜索快速找到相關API
- 查看實際使用示例

### 4. Weights & Biases
**用途**: 實驗追踪和可視化
```python
import wandb

wandb.init(project="dl-computation")
wandb.config.update({"learning_rate": 0.001, "epochs": 10})

for epoch in range(epochs):
    train_loss = train(model)
    wandb.log({"train_loss": train_loss, "epoch": epoch})
```

### 5. TensorBoard
**用途**: 模型可視化
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_graph(model, input_sample)
writer.add_scalar('Loss/train', train_loss, epoch)
```

---

## 💡 學習建議

### 初學者路線 (4-6週)
1. **Week 1-2**: 層和區塊、參數管理
   - 完成所有基礎練習
   - 實現至少3個自定義模型

2. **Week 3-4**: 延遲初始化、自定義層
   - 創建2-3個自定義層
   - 集成到實際項目中

3. **Week 5-6**: 文件讀寫、GPU計算
   - 實現完整的訓練循環
   - 優化GPU使用效率

### 進階路線 (2-3週)
1. 深入研究一個特定主題（如自定義層或GPU優化）
2. 閱讀相關論文並實現算法
3. 貢獻到開源項目

### 實戰建議
- ✅ 每個概念都親手實現一遍
- ✅ 對比不同實現的性能
- ✅ 記錄遇到的問題和解決方案
- ✅ 與社區分享你的學習心得

---

## 📚 推薦資源

### 書籍
- 📖 **Deep Learning** by Ian Goodfellow (中文版: 深度學習)
- 📖 **Dive into Deep Learning** by Aston Zhang et al. (d2l.ai)
- 📖 **PyTorch Documentation** (官方文檔)

### 在線課程
- 🎓 **Fast.ai Practical Deep Learning for Coders**
- 🎓 **Stanford CS231n: Convolutional Neural Networks**
- 🎓 **MIT 6.S191: Introduction to Deep Learning**

### 論文
- 📄 **Batch Normalization** (Ioffe & Szegedy, 2015)
- 📄 **Layer Normalization** (Ba et al., 2016)
- 📄 **Mixed Precision Training** (Micikevicius et al., 2018)

### 社區
- 💬 **PyTorch Forums**: discuss.pytorch.org
- 💬 **Reddit r/MachineLearning**
- 💬 **Stack Overflow**
- 💬 **GitHub Issues & Discussions**

---

## 🔧 開發環境設置

### 基礎環境
```bash
# 創建虛擬環境
conda create -n dl-compute python=3.10
conda activate dl-compute

# 安裝PyTorch (根據你的平台選擇)
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch torchvision torchaudio

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 常用工具
```bash
# 實驗追踪
pip install wandb tensorboard

# 代碼質量
pip install black isort flake8 mypy

# Jupyter支持
pip install jupyter ipywidgets

# 性能分析
pip install py-spy memory_profiler
```

### VS Code擴展推薦
- Python
- Pylance
- Jupyter
- GitHub Copilot
- PyTorch Snippets

---

## 🎯 實際項目示例

### 項目1: 自定義ResNet實現
**難度**: ⭐⭐⭐
**知識點**: 層和區塊、參數管理、GPU計算

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### 項目2: 模型訓練管道
**難度**: ⭐⭐⭐⭐
**知識點**: 所有核心概念的綜合應用

完整代碼見: `examples/training_pipeline.py`

### 項目3: 跨平台部署方案
**難度**: ⭐⭐⭐⭐⭐
**知識點**: 模型導出、優化、部署

完整代碼見: `examples/deployment_example.py`

---

## 🐛 常見問題與解決方案

### Q1: CUDA out of memory錯誤
**解決方案**:
```python
# 1. 減小batch size
# 2. 使用梯度累積
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 清理GPU緩存
torch.cuda.empty_cache()
```

### Q2: MPS設備不支持某些操作
**解決方案**:
```python
try:
    output = model(data.to('mps'))
except RuntimeError:
    print("操作不支持MPS，回退到CPU")
    output = model(data.to('cpu'))
```

### Q3: 模型加載後性能下降
**檢查清單**:
- ✅ 確保使用`model.eval()`進行推理
- ✅ 檢查是否正確加載了優化器狀態
- ✅ 驗證數據預處理流程一致性

### Q4: 梯度爆炸/消失
**解決方案**:
```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用批標準化
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    # ...
)

# 選擇合適的初始化
model.apply(lambda m: nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) else None)
```

---

## 📊 學習檢查清單

### 基礎知識 ✓
- [ ] 理解`nn.Module`的工作原理
- [ ] 能夠構建自定義區塊
- [ ] 掌握參數訪問和初始化
- [ ] 了解延遲初始化的機制

### 進階技能 ✓
- [ ] 創建複雜的自定義層
- [ ] 實現高效的模型保存/加載
- [ ] 優化GPU內存使用
- [ ] 處理多設備計算

### 實戰能力 ✓
- [ ] 完成至少一個端到端的項目
- [ ] 能夠調試性能問題
- [ ] 理解生產環境的考慮因素
- [ ] 能夠閱讀和理解開源代碼

---

## 🤝 貢獻指南

歡迎貢獻！你可以：
- 📝 改進文檔和註釋
- 🐛 報告和修復錯誤
- ✨ 添加新的示例和練習
- 💡 分享學習心得和最佳實踐

**提交流程**:
1. Fork本倉庫
2. 創建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個Pull Request

---

## 📜 更新日誌

### v3.0 (2024-2025)
- ✨ 新增多平台GPU支持 (CUDA/MPS/ROCm)
- 📚 完善所有練習題的答案
- 🎯 添加AI輔助學習工具指南
- 🔧 更新代碼示例到PyTorch 2.0+

### v2.0 (2023)
- 📖 重構文檔結構
- ✅ 添加完整的練習題
- 🚀 優化代碼示例

### v1.0 (2022)
- 🎉 初始版本發布
- 📝 基礎內容完成

---

## 📞 聯繫方式

- **討論**: [GitHub Discussions](https://discuss.d2l.ai)
- **問題**: [GitHub Issues](https://github.com/your-repo/issues)
- **郵件**: your-email@example.com

---

## 📄 許可證

本項目採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件

---

## 🙏 致謝

感謝以下資源和社區：
- PyTorch團隊
- D2L.ai項目
- 所有貢獻者和學習者

---

**⭐ 如果這個項目對你有幫助，請給我們一個Star！**

**📖 開始學習**: 從 `1_model-construction.ipynb` 開始你的深度學習計算之旅！
