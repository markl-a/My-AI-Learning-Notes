# AI輔助CNN開發指南

本文檔介紹如何使用AI工具來輔助CNN模型的開發、調試和優化，以及CNN開發的最佳實踐。

## 目錄

1. [AI輔助開發工具](#ai輔助開發工具)
2. [自動化超參數調優](#自動化超參數調優)
3. [模型調試技巧](#模型調試技巧)
4. [性能優化](#性能優化)
5. [部署最佳實踐](#部署最佳實踐)
6. [實戰checklist](#實戰checklist)

---

## AI輔助開發工具

### 1. 程式碼輔助工具

#### GitHub Copilot / ChatGPT / Claude
**用途：**
- 快速生成模型架構程式碼
- 編寫資料加載和預處理程式碼
- 調試錯誤和異常
- 優化程式碼性能

**示例提示詞：**
```
# 生成ResNet塊
"幫我實現一個ResNet的殘差塊，包含批量歸一化和跳躍連接"

# 資料增強
"為圖像分類任務建立一個資料增強pipeline，包含翻轉、旋轉、裁剪和顏色抖動"

# 調試幫助
"我的模型訓練時損失不下降，可能的原因和解決方案是什麼？"

# 性能優化
"如何優化這段PyTorch程式碼的性能？[貼上程式碼]"
```

#### Cursor / Windsurf
**特點：**
- AI原生編輯器
- 實時程式碼建議
- 上下文感知
- 多文件編輯

**最佳實踐：**
1. 編寫清晰的註釋和文檔字串
2. 使用AI生成測試用例
3. 讓AI解釋複雜程式碼片段
4. 用AI重構和優化程式碼

### 2. 實驗管理工具

#### Weights & Biases (wandb)
```python
import wandb

# 初始化實驗
wandb.init(
    project="cnn-classification",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "architecture": "ResNet-18"
    }
)

# 訓練循環中記錄
for epoch in range(epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()

    # 記錄指標
    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "epoch": epoch
    })

    # 記錄圖像（可選）
    wandb.log({"predictions": wandb.Image(image, caption=f"Pred: {pred}, True: {label}")})
```

**優勢：**
- 自動記錄所有實驗
- 可視化訓練過程
- 超參數對比
- 團隊協作
- AI輔助分析（wandb.ai的AI功能）

#### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# 記錄標量
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)

# 記錄圖像
writer.add_images('predictions', images, epoch)

# 記錄模型圖
writer.add_graph(model, input_tensor)

# 記錄嵌入向量
writer.add_embedding(features, metadata=labels, label_img=images)

writer.close()
```

### 3. 自動模型設計

#### Neural Architecture Search (NAS)
使用AutoML工具自動搜索最優架構：

**AutoKeras示例：**
```python
import autokeras as ak

# 自動搜索圖像分類器
clf = ak.ImageClassifier(
    max_trials=10,  # 最大嘗試次數
    overwrite=True
)

# 訓練
clf.fit(x_train, y_train, epochs=50)

# 評估
accuracy = clf.evaluate(x_test, y_test)

# 獲取最佳模型
best_model = clf.export_model()
```

**NNI (Neural Network Intelligence) 示例：**
```python
import nni

# 在訓練程式碼中獲取超參數
params = nni.get_next_parameter()
lr = params['learning_rate']
batch_size = params['batch_size']

# 訓練模型
model = train_model(lr, batch_size)

# 報告結果
nni.report_final_result(accuracy)
```

---

## 自動化超參數調優

### 1. 使用Optuna進行超參數搜索

```python
import optuna

def objective(trial):
    # 定義搜索空間
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    # 建立模型
    model = create_model(dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 訓練和驗證
    val_acc = train_and_validate(model, optimizer, batch_size)

    return val_acc

# 建立研究
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 獲取最佳參數
print("Best parameters:", study.best_params)
print("Best validation accuracy:", study.best_value)

# 可視化
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### 2. 學習率查找器

```python
from torch_lr_finder import LRFinder

model = create_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

# 執行LR查找
lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

# 繪製圖表
lr_finder.plot()

# 獲取建議的學習率
best_lr = lr_finder.suggest_lr()
print(f"Suggested learning rate: {best_lr}")

# 重置模型
lr_finder.reset()
```

### 3. 使用Ray Tune進行分佈式超參數調優

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_func(config):
    model = create_model(
        hidden_size=config["hidden_size"],
        dropout=config["dropout"]
    )

    lr = config["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        train_loss = train_epoch(model, optimizer)
        val_loss = validate(model)

        # 向Ray Tune報告結果
        tune.report(loss=val_loss)

# 配置搜索空間
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "hidden_size": tune.choice([64, 128, 256]),
    "dropout": tune.uniform(0.2, 0.5)
}

# 運行調優
scheduler = ASHAScheduler(
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

result = tune.run(
    train_func,
    resources_per_trial={"gpu": 1},
    config=config,
    num_samples=20,
    scheduler=scheduler
)

# 獲取最佳配置
best_trial = result.get_best_trial("loss", "min", "last")
print("Best config:", best_trial.config)
```

---

## 模型調試技巧

### 1. 使用AI助手調試

**常見問題和AI提示詞：**

#### 問題1：損失不下降
```
提示詞：
"我的CNN模型訓練時損失一直不下降，保持在初始值附近。
模型架構：[描述架構]
學習率：0.001
優化器：Adam
資料集：[描述資料集]
可能的原因和解決方案是什麼？"
```

**AI可能的回答：**
1. 檢查資料預處理是否正確（歸一化、標準化）
2. 學習率可能太小，嘗試1e-2
3. 檢查損失函式是否適配任務
4. 驗證梯度是否為0（梯度消失）
5. 檢查標籤是否正確

#### 問題2：過擬合
```
提示詞：
"我的模型在訓練集上準確率95%，但驗證集只有70%，出現嚴重過擬合。
已經使用了：Dropout(0.5), 資料增強, Batch Normalization
還有什麼其他方法可以改善？"
```

#### 問題3：記憶體溢出
```
提示詞：
"訓練時CUDA out of memory錯誤，如何優化記憶體使用？
當前配置：batch_size=32, 圖像大小=224x224, 模型=ResNet-50"
```

### 2. 梯度檢查

```python
def check_gradients(model):
    """檢查模型的梯度"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.6f}")

            if grad_norm == 0:
                print(f"  ⚠️  警告：{name} 的梯度為0！")
            elif grad_norm > 100:
                print(f"  ⚠️  警告：{name} 的梯度爆炸！")
        else:
            print(f"{name}: 沒有梯度")

# 訓練循環中使用
loss.backward()
check_gradients(model)
optimizer.step()
```

### 3. 激活值監控

```python
def register_activation_hooks(model):
    """註冊hook監控激活值"""
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # 為所有ReLU層註冊hook
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(get_activation(name))

    return activations

# 使用
activations = register_activation_hooks(model)
output = model(input_tensor)

# 檢查激活值
for name, act in activations.items():
    print(f"{name}: mean={act.mean():.4f}, std={act.std():.4f}")
    dead_neurons = (act == 0).float().mean().item()
    print(f"  死神經元比例: {dead_neurons*100:.2f}%")
```

### 4. 使用AI分析訓練曲線

**提示詞示例：**
```
"分析這個訓練曲線，給出改進建議：
[貼上loss和accuracy的資料或圖表]

觀察到的現象：
1. 訓練損失穩定下降
2. 驗證損失在epoch 20後開始上升
3. 訓練準確率達到98%
4. 驗證準確率停留在75%"
```

---

## 性能優化

### 1. 使用AI優化程式碼

**提示詞：**
```
"優化以下PyTorch訓練程式碼的性能：
[貼上程式碼]

要求：
1. 減少記憶體使用
2. 提高訓練速度
3. 保持程式碼可讀性"
```

### 2. 混合精度訓練

```python
from torch.cuda.amp import autocast, GradScaler

# 建立GradScaler
scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()

        # 使用autocast進行前向傳播
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # 縮放損失並反向傳播
        scaler.scale(loss).backward()

        # 更新權重
        scaler.step(optimizer)
        scaler.update()

# 記憶體使用減少約40-50%
# 訓練速度提升2-3倍（在支持Tensor Core的GPU上）
```

### 3. 資料加載優化

```python
# 優化DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # 使用多進程
    pin_memory=True,  # 加速GPU資料傳輸
    persistent_workers=True,  # PyTorch 1.7+
    prefetch_factor=2  # 預取批次數
)

# 使用更快的圖像讀取庫
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

# DALI pipeline（GPU上的資料增強）
class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root=data_dir)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return output, labels
```

### 4. 模型編譯優化

```python
# PyTorch 2.0的compile功能
import torch._dynamo as dynamo

# 編譯模型以獲得更好的性能
model = torch.compile(model, mode="max-autotune")

# 速度提升：10-30%（取決於模型）
```

---

## 部署最佳實踐

### 1. 模型量化

```python
import torch.quantization

# 動態量化（最簡單）
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 靜態量化（最佳性能）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 校準
with torch.no_grad():
    for data, _ in calibration_loader:
        model(data)

torch.quantization.convert(model, inplace=True)

# 模型大小減少4倍
# 推論速度提升2-4倍
```

### 2. 模型剪枝

```python
import torch.nn.utils.prune as prune

# 對卷積層進行剪枝
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# 移除剪枝重參數化
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# 參數減少30%
# 推論速度提升約20-30%
```

### 3. ONNX導出

```python
# 導出為ONNX格式
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)

# 使用ONNX Runtime進行推理
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'input': input_data.numpy()})
```

### 4. TorchScript

```python
# 腳本化模型
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 或追蹤模型
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# 載入和使用
loaded_model = torch.jit.load("model_scripted.pt")
output = loaded_model(input_tensor)
```

---

## 實戰Checklist

### 開始新項目時

- [ ] 定義問題和評估指標
- [ ] 收集和分析資料
- [ ] 建立baseline（簡單模型）
- [ ] 選擇預訓練模型（如果適用）
- [ ] 設置實驗跟蹤（wandb/tensorboard）
- [ ] 建立資料增強pipeline
- [ ] 實現訓練和驗證循環
- [ ] 設置檢查點保存

### 訓練過程中

- [ ] 監控訓練和驗證曲線
- [ ] 檢查梯度和激活值
- [ ] 嘗試不同的超參數
- [ ] 使用學習率調度器
- [ ] 定期在驗證集上評估
- [ ] 保存最佳模型
- [ ] 可視化預測結果

### 優化階段

- [ ] 分析錯誤樣本
- [ ] 調整資料增強策略
- [ ] 嘗試不同的架構
- [ ] 使用集成方法
- [ ] 進行超參數調優
- [ ] 考慮遷移學習

### 部署前

- [ ] 在測試集上評估
- [ ] 測試邊緣情況
- [ ] 優化模型（量化/剪枝）
- [ ] 測試推論速度
- [ ] 準備模型文檔
- [ ] 設置監控和日誌

### 使用AI助手的最佳實踐

**1. 程式碼生成**
- ✅ 提供清晰的需求描述
- ✅ 指定框架和版本
- ✅ 要求添加註釋
- ❌ 盲目複製粘貼

**2. 調試**
- ✅ 提供完整的錯誤資訊
- ✅ 描述已經嘗試的方法
- ✅ 說明預期和實際行為
- ❌ 只提供錯誤類型

**3. 優化建議**
- ✅ 提供性能瓶頸資訊
- ✅ 說明硬體限制
- ✅ 指定優化目標
- ❌ 要求"讓程式碼更快"而不提供上下文

**4. 學習**
- ✅ 要求解釋原理
- ✅ 請求提供參考資料
- ✅ 詢問最佳實踐
- ❌ 只要答案不要理解

---

## AI工具推薦

### 程式碼開發
- **Cursor / Windsurf**: AI原生編輯器
- **GitHub Copilot**: 程式碼補全
- **ChatGPT / Claude**: 技術諮詢和調試

### 實驗管理
- **Weights & Biases**: 最全面的實驗跟蹤
- **TensorBoard**: PyTorch官方支持
- **Neptune.ai**: 適合大型團隊

### 自動化
- **Optuna**: 超參數優化
- **Ray Tune**: 分佈式調優
- **AutoKeras**: 自動模型搜索
- **NNI**: Microsoft的AutoML工具

### 部署
- **ONNX Runtime**: 跨平台推理
- **TorchServe**: PyTorch官方部署
- **Triton**: NVIDIA推理伺服器
- **TensorRT**: NVIDIA GPU優化

### 監控和調試
- **Netron**: 模型可視化
- **torch-tb-profiler**: PyTorch性能分析
- **Captum**: 模型解釋性

---

## 總結

**AI輔助開發的核心原則：**
1. **AI是工具，不是替代**：理解原理比快速生成程式碼更重要
2. **驗證和測試**：始終驗證AI生成的程式碼
3. **持續學習**：使用AI作為學習助手
4. **記錄實驗**：使用工具記錄所有嘗試
5. **自動化重複任務**：讓AI處理繁瑣的工作

**關鍵要點：**
- 使用AI助手快速原型設計和調試
- 使用自動化工具進行超參數優化
- 使用實驗跟蹤工具記錄所有實驗
- 在部署前進行充分的優化和測試
- 建立系統的開發和部署流程

通過合理使用AI工具，可以大幅提升開發效率和模型品質！
