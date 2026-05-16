# 混合精度策略 (Mixed-Precision Strategies)

## 目錄
1. [基本概念](#1-基本概念)
2. [混合精度訓練](#2-混合精度訓練)
3. [混合精度推理](#3-混合精度推理)
4. [敏感層識別](#4-敏感層識別)
5. [實作範例](#5-實作範例)
6. [最佳實踐](#6-最佳實踐)

---

## 1. 基本概念

### 1.1 什麼是混合精度？

**混合精度**是在同一個模型中使用多種數值精度的策略，對不同層或操作使用不同的位元寬度。

**核心思想**：
```
關鍵層（敏感）    → 高精度（FP16/FP32）
非關鍵層（穩健）   → 低精度（INT8/INT4）
```

**動機**：
1. **精度敏感性差異**：不同層對量化的容忍度不同
2. **性能權衡**：在精度損失和效率提升間找平衡
3. **硬體限制**：充分利用現代GPU的混合精度能力

### 1.2 為什麼需要混合精度？

**均勻量化的問題**：
```
所有層 INT8：
✓ 最快、最小
✗ 某些層精度損失嚴重

所有層 FP16：
✓ 高精度
✗ 未充分壓縮
```

**混合精度的優勢**：
```
敏感層 FP16 + 其他層 INT8：
✓ 保持關鍵層精度
✓ 大部分層仍壓縮
✓ 性能與精度的最佳平衡
```

### 1.3 精度層次

**常見精度格式**：
```
FP32 (32-bit)   ████████ 最高精度，訓練預設
  ↓
BF16 (16-bit)   ████     訓練穩定，範圍大
  ↓
FP16 (16-bit)   ████     訓練/推理常用
  ↓
INT8 (8-bit)    ██       推理常用，4x 壓縮
  ↓
INT4 (4-bit)    █        極限壓縮，8x 壓縮
  ↓
INT2 (2-bit)    ▌        實驗性，16x 壓縮
```

---

## 2. 混合精度訓練

### 2.1 自動混合精度 (AMP)

**Automatic Mixed Precision** 是最常見的混合精度訓練方法。

**核心策略**：
```
前向傳播：
  - 大部分操作：FP16（快速）
  - 某些操作：FP32（穩定）

反向傳播：
  - 梯度計算：FP16
  - 梯度累積：FP32

權重更新：
  - Master weights：FP32（精確）
  - Working weights：FP16（推理用）
```

**Loss Scaling**：
```
問題：FP16 範圍小，梯度可能下溢

解決：放大損失和梯度
  loss_scaled = loss × scale_factor
  gradient_scaled = gradient × scale_factor
  gradient_unscaled = gradient_scaled / scale_factor

動態 scaling：
  - 無溢出：增加 scale
  - 有溢出：減少 scale
```

### 2.2 PyTorch AMP 實作

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 模型、優化器、資料
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

# 訓練循環
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch

        # 清除梯度
        optimizer.zero_grad()

        # 混合精度前向傳播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 縮放損失，反向傳播
        scaler.scale(loss).backward()

        # 更新權重（自動 unscale 梯度）
        scaler.step(optimizer)

        # 更新 scaler
        scaler.update()
```

**優勢**：
```
✓ 訓練速度提升 2-3x（在 Tensor Core GPU）
✓ 顯存佔用減少 ~50%
✓ 精度損失極小（< 0.1%）
✓ 易於實現（幾行程式碼）
```

### 2.3 Transformers 中的 AMP

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,              # 啟用 FP16 訓練
    fp16_opt_level="O1",    # Apex O1 級別（可選）
    # bf16=True,            # 或使用 BF16（Ampere+ GPU）
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 2.4 BF16 vs FP16

**BF16 (Brain Float 16)**：
```
優勢：
✓ 與 FP32 相同的指數範圍（不易溢出）
✓ 不需要 loss scaling
✓ 訓練更穩定

劣勢：
✗ 精度略低於 FP16（尾數短）
✗ 需要較新硬體（Ampere+, TPU）

使用場景：
- 大模型訓練（GPT, LLaMA）
- 不穩定的訓練任務
- 有 BF16 硬體支援時
```

**FP16**：
```
優勢：
✓ 更廣泛的硬體支援
✓ 精度略高於 BF16

劣勢：
✗ 需要 loss scaling
✗ 更容易溢出

使用場景：
- Volta/Turing GPU
- 訓練相對穩定的模型
```

**選擇建議**：
```python
# 檢測 BF16 支援
if torch.cuda.is_bf16_supported():
    use_bf16 = True
else:
    use_fp16 = True
```

---

## 3. 混合精度推理

### 3.1 層級混合精度

**策略**：為不同層分配不同精度。

**典型配置**：
```
Transformer 模型：

Layer Type              Precision   Reason
Embedding               FP16        查表操作，不量化
LayerNorm               FP16        對量化敏感
Attention Q/K/V         INT8        量化友好
Attention Softmax       FP16        對量化敏感
Attention Output        INT8        量化友好
FFN (FC1, FC2)          INT8/INT4   量化友好
LM Head                 FP16        輸出層，保持精度
```

**效果**：
```
全 FP16：14 GB,  1.0x 速度,  PPL 5.68
全 INT8：7 GB,   2.5x 速度,  PPL 5.75
混合（上述配置）：9 GB, 2.0x 速度, PPL 5.69
```

### 3.2 SmoothQuant

**問題**：激活值的離群值（outliers）導致量化困難。

**核心思想**：
```
通過數學變換，將激活值的難度轉移到權重：

Y = (Xdiag(s)^(-1)) · (diag(s)W)
  = X' · W'

其中：
- s：平滑因子
- X'：平滑後的激活值（更易量化）
- W'：調整後的權重
```

**實作**：
```python
def smooth_quant(weight, activation, alpha=0.5):
    """
    SmoothQuant 變換

    Args:
        weight: 權重矩陣 [out_features, in_features]
        activation: 激活值統計 [in_features]
        alpha: 平滑因子（0.5 為平衡）
    """
    # 計算 scale
    s = activation.abs().max(dim=0).values ** alpha / \
        weight.abs().max(dim=0).values ** (1 - alpha)

    # 應用變換
    weight_smoothed = weight * s.unsqueeze(0)
    activation_scale = 1.0 / s

    return weight_smoothed, activation_scale

# 使用
weight_smoothed, act_scale = smooth_quant(weight, activation_stats)

# 量化
weight_quant = quantize(weight_smoothed)
activation_quant = quantize(activation * act_scale)
```

**優勢**：
```
✓ 解決激活值離群值問題
✓ INT8 量化精度損失 < 0.5%
✓ 無需重訓練
✓ 適用於 LLaMA, OPT 等模型
```

### 3.3 動態精度選擇

**運行時動態調整**：

```python
class DynamicPrecisionModel(nn.Module):
    """動態精度推理模型"""

    def __init__(self, model, sensitivity_map):
        super().__init__()
        self.model = model
        self.sensitivity = sensitivity_map  # 層級敏感度

    def forward(self, x, quality_target="balanced"):
        """
        quality_target:
        - "fast": 更多低精度層
        - "balanced": 平衡
        - "accurate": 更多高精度層
        """
        precision_thresholds = {
            "fast": 0.7,
            "balanced": 0.5,
            "accurate": 0.3,
        }

        threshold = precision_thresholds[quality_target]

        for name, layer in self.model.named_modules():
            if self.sensitivity[name] > threshold:
                # 高敏感度：使用 FP16
                layer.use_fp16()
            else:
                # 低敏感度：使用 INT8
                layer.use_int8()

        return self.model(x)
```

---

## 4. 敏感層識別

### 4.1 為什麼需要？

**觀察**：
```
不同層對量化的敏感度差異巨大：

Layer 1 (Embedding)     : FP16→INT8, PPL 5.68→6.82 ❌
Layer 5 (Attention)     : FP16→INT8, PPL 5.68→5.71 ✓
Layer 10 (FFN)          : FP16→INT8, PPL 5.68→5.69 ✓
Layer 15 (LayerNorm)    : FP16→INT8, PPL 5.68→7.45 ❌
```

**策略**：
- 識別敏感層 → 保持高精度
- 識別穩健層 → 激進量化

### 4.2 敏感度分析方法

#### 方法 1：逐層量化測試

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def layer_sensitivity_analysis(model, tokenizer, dataset, metric="perplexity"):
    """逐層分析量化敏感度"""

    # 基準性能（全 FP16）
    baseline = evaluate_model(model, tokenizer, dataset, metric)
    print(f"Baseline {metric}: {baseline:.4f}")

    sensitivity_map = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # 量化單層
        original_weight = module.weight.data.clone()
        module.weight.data = quantize_to_int8(original_weight)

        # 評估
        score = evaluate_model(model, tokenizer, dataset, metric)
        sensitivity = abs(score - baseline) / baseline

        sensitivity_map[name] = sensitivity

        # 恢復
        module.weight.data = original_weight

        print(f"{name}: {metric}={score:.4f}, sensitivity={sensitivity:.4f}")

    return sensitivity_map

# 使用
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

sensitivity = layer_sensitivity_analysis(model, tokenizer, dataset)

# 排序
sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)

print("\n最敏感的 10 層：")
for name, sens in sorted_layers[:10]:
    print(f"{name}: {sens:.4f}")
```

#### 方法 2：Hessian 追蹤

**基於二階資訊**：
```python
def hessian_sensitivity(model, dataloader):
    """使用 Hessian 對角線估計敏感度"""

    sensitivity = {}

    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue

        # 累積梯度平方（Hessian 對角線近似）
        param.grad = None
        hessian_diag = torch.zeros_like(param)

        for batch in dataloader:
            output = model(batch)
            loss = output.loss

            # 計算梯度
            grad = torch.autograd.grad(loss, param, create_graph=True)[0]

            # 累積梯度平方
            hessian_diag += grad ** 2

        # 平均敏感度
        sensitivity[name] = hessian_diag.mean().item()

    return sensitivity
```

#### 方法 3：激活值統計

```python
def activation_based_sensitivity(model, dataloader):
    """基於激活值範圍和離群值分析敏感度"""

    activation_stats = {}
    hooks = []

    def hook_fn(module, input, output, name):
        """記錄激活值統計"""
        act = output.detach()

        stats = {
            'mean': act.mean().item(),
            'std': act.std().item(),
            'max': act.abs().max().item(),
            'outlier_ratio': (act.abs() > 6 * act.std()).float().mean().item(),
        }

        if name not in activation_stats:
            activation_stats[name] = []
        activation_stats[name].append(stats)

    # 註冊 hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            hooks.append(hook)

    # 運行模型
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch)

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    # 計算敏感度（離群值比例高 = 敏感）
    sensitivity = {
        name: sum(s['outlier_ratio'] for s in stats) / len(stats)
        for name, stats in activation_stats.items()
    }

    return sensitivity
```

### 4.3 自動混合精度決策

```python
def auto_mixed_precision_config(sensitivity_map, target_compression=2.0):
    """
    自動生成混合精度配置

    Args:
        sensitivity_map: 層敏感度字典
        target_compression: 目標壓縮比（如 2.0 表示模型大小減半）

    Returns:
        precision_config: 層精度配置
    """
    # 排序層（敏感度降序）
    sorted_layers = sorted(
        sensitivity_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 計算參數量
    layer_params = {name: get_layer_params(model, name) for name, _ in sorted_layers}
    total_params = sum(layer_params.values())

    # 目標參數預算（FP16 = 2 bytes, INT8 = 1 byte）
    budget = total_params * 2 / target_compression  # bytes

    precision_config = {}
    used_budget = 0

    for name, sensitivity in sorted_layers:
        params = layer_params[name]

        # 高敏感層優先分配 FP16
        if used_budget + params * 2 <= budget:
            precision_config[name] = "fp16"
            used_budget += params * 2
        else:
            precision_config[name] = "int8"
            used_budget += params * 1

    return precision_config

# 使用
config = auto_mixed_precision_config(sensitivity, target_compression=2.0)

print("混合精度配置：")
print(f"FP16 層數: {sum(1 for p in config.values() if p == 'fp16')}")
print(f"INT8 層數: {sum(1 for p in config.values() if p == 'int8')}")
```

---

## 5. 實作範例

### 5.1 手動混合精度推理

```python
import torch
import torch.nn as nn

class MixedPrecisionLinear(nn.Module):
    """混合精度線性層"""

    def __init__(self, linear, precision="fp16"):
        super().__init__()
        self.precision = precision

        if precision == "int8":
            # 量化權重
            self.weight_int8, self.scale, self.zero_point = quantize_int8(linear.weight)
            self.bias = linear.bias
        else:
            # FP16
            self.weight = linear.weight.half()
            self.bias = linear.bias.half() if linear.bias is not None else None

    def forward(self, x):
        if self.precision == "int8":
            # INT8 推理
            x_quant, x_scale, x_zp = quantize_int8(x)
            output_quant = F.linear(x_quant, self.weight_int8, None)
            output = dequantize_int8(output_quant, self.scale * x_scale)
            if self.bias is not None:
                output += self.bias
        else:
            # FP16 推理
            output = F.linear(x, self.weight, self.bias)

        return output

def convert_to_mixed_precision(model, precision_config):
    """轉換模型為混合精度"""

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if name in precision_config:
            precision = precision_config[name]

            # 替換模塊
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)

            mixed_layer = MixedPrecisionLinear(module, precision)
            setattr(parent, child_name, mixed_layer)

    return model

# 使用範例
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 定義精度配置
precision_config = {
    "transformer.h.0.attn.c_attn": "fp16",      # 敏感
    "transformer.h.0.attn.c_proj": "int8",      # 穩健
    "transformer.h.0.mlp.c_fc": "int8",         # 穩健
    "transformer.h.0.mlp.c_proj": "int8",       # 穩健
    # ... 其他層
}

# 轉換
model = convert_to_mixed_precision(model, precision_config)

# 推理
output = model.generate(input_ids, max_length=50)
```

### 5.2 使用 ONNX Runtime 混合精度

```python
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 載入模型
model = ORTModelForCausalLM.from_pretrained("gpt2", export=True)

# 混合精度配置
qconfig = AutoQuantizationConfig.arm64(
    is_static=False,
    per_channel=True,
    # 指定不量化的層（敏感層）
    nodes_to_exclude=[
        "/transformer/h.0/ln_1/Add",
        "/transformer/h.0/attn/Softmax",
        "/transformer/ln_f/Add",
        "/lm_head/MatMul",
    ]
)

# 量化
model.quantize(save_directory="./gpt2-mixed-int8", quantization_config=qconfig)

# 載入量化模型
quantized_model = ORTModelForCausalLM.from_pretrained("./gpt2-mixed-int8")

# 推理
output = quantized_model.generate(input_ids, max_length=50)
```

### 5.3 TensorRT 混合精度

```python
import tensorrt as trt

# 建立 builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# 啟用 INT8 和 FP16
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)

# 為敏感層設置精度
for layer in network:
    layer_name = layer.name

    if layer_name in sensitive_layers:
        # 強制使用 FP16
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)
    else:
        # 允許使用 INT8
        layer.precision = trt.int8

# 構建引擎
engine = builder.build_engine(network, config)
```

---

## 6. 最佳實踐

### 6.1 混合精度配置指南

**1. 通用規則**：
```
始終使用高精度：
- Embedding 層
- LayerNorm / BatchNorm
- Softmax / Attention 計算
- 輸出層（LM Head）

可以使用低精度：
- FFN 中間層
- 大部分 Attention 投影（Q/K/V/O）
- 激活函式（ReLU, GELU）
```

**2. 不同模型架構**：
```
BERT / RoBERTa：
- 保持 Embedding 和 Pooler 為 FP16
- Attention 可用 INT8
- FFN 可用 INT8

GPT / LLaMA：
- Embedding 和 LM Head 為 FP16
- 早期層（1-5層）保守量化
- 中間層（6-28層）激進量化
- 最後幾層（29-32層）保守量化

Vision Transformer：
- Patch Embedding 為 FP16
- Position Embedding 為 FP16
- 其他類似 BERT
```

**3. 資料類型選擇**：
```
精度      訓練    推理    硬體需求
FP32      ✓       ✓       任意
BF16      ✓       ✓       Ampere+, TPU
FP16      ✓       ✓       Volta+
INT8      ✗       ✓       任意（軟體）
INT4      ✗       ✓       專用硬體或軟體
```

### 6.2 性能調優

**1. 批次大小**：
```python
# 混合精度允許更大批次
if mixed_precision:
    batch_size *= 2  # 顯存佔用減半
```

**2. 梯度累積**：
```python
# 混合精度訓練時的梯度累積
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with autocast():
        loss = model(batch) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**3. 精度驗證**：
```python
def validate_precision(model, test_data):
    """驗證混合精度模型精度"""

    # 測試困惑度
    ppl = evaluate_perplexity(model, test_data)

    # 測試生成品質
    gen_quality = evaluate_generation(model, test_prompts)

    # 測試特定任務
    task_acc = evaluate_task(model, task_dataset)

    return {
        "perplexity": ppl,
        "generation_quality": gen_quality,
        "task_accuracy": task_acc
    }
```

### 6.3 常見陷阱

**陷阱 1：忽視資料類型轉換開銷**
```python
# ❌ 錯誤：頻繁轉換
for layer in model:
    x = x.float()  # FP32
    x = layer(x)
    x = x.half()   # FP16

# ✓ 正確：批量轉換
x = x.half()
for layer in model:
    x = layer(x)
```

**陷阱 2：未處理 NaN/Inf**
```python
# ✓ 檢測並處理
with autocast():
    loss = model(inputs)

if torch.isnan(loss) or torch.isinf(loss):
    print("檢測到數值不穩定，跳過此批次")
    continue
```

**陷阱 3：不當的層選擇**
```python
# ❌ 錯誤：量化所有層
quantize_all_layers(model)

# ✓ 正確：基於敏感度選擇
sensitive_layers = identify_sensitive_layers(model)
quantize_non_sensitive_layers(model, exclude=sensitive_layers)
```

### 6.4 調試技巧

**1. 逐層對比**：
```python
def compare_layer_outputs(model_fp32, model_mixed, test_input):
    """對比混合精度模型與全精度模型的逐層輸出"""

    outputs_fp32 = {}
    outputs_mixed = {}

    # 註冊 hooks
    for name, module in model_fp32.named_modules():
        module.register_forward_hook(
            lambda m, i, o, n=name: outputs_fp32.update({n: o.detach()})
        )

    for name, module in model_mixed.named_modules():
        module.register_forward_hook(
            lambda m, i, o, n=name: outputs_mixed.update({n: o.detach()})
        )

    # 運行
    with torch.no_grad():
        model_fp32(test_input)
        model_mixed(test_input)

    # 比較
    for name in outputs_fp32:
        diff = (outputs_fp32[name] - outputs_mixed[name]).abs().mean()
        print(f"{name}: mean_abs_diff = {diff:.6f}")
```

**2. 激活值分佈**：
```python
import matplotlib.pyplot as plt

def plot_activation_distribution(model, test_input, layer_name):
    """繪製激活值分佈"""

    activation = {}

    def hook(module, input, output):
        activation['output'] = output.detach().cpu()

    layer = dict(model.named_modules())[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        model(test_input)

    hook_handle.remove()

    # 繪圖
    plt.hist(activation['output'].flatten().numpy(), bins=100)
    plt.title(f"Activation Distribution: {layer_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
```

---

## 總結

### 關鍵要點

1. **混合精度是精度與效率的最佳平衡**：
   - 關鍵層保持高精度
   - 穩健層激進壓縮
   - 整體性能最優

2. **訓練與推理策略不同**：
   - 訓練：AMP（FP16/BF16 + FP32）
   - 推理：層級混合（FP16 + INT8/INT4）

3. **敏感層識別至關重要**：
   - 逐層測試
   - Hessian 分析
   - 激活值統計

4. **硬體考量**：
   - Tensor Core（NVIDIA）：FP16/BF16
   - VNNI（Intel）：INT8
   - 專用加速器：INT4/INT2

5. **自動化工具**：
   - PyTorch AMP
   - TensorRT
   - ONNX Runtime
   - Optimum

### 延伸閱讀

**論文**：
- Mixed Precision Training: "Mixed Precision Training" (Micikevicius et al., 2017)
- SmoothQuant: "SmoothQuant: Accurate and Efficient Post-Training Quantization" (Xiao et al., 2022)
- LLM.int8(): "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al., 2022)

**工具文檔**：
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- NVIDIA TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
- ONNX Runtime: https://onnxruntime.ai/docs/performance/quantization.html

**實用資源**：
- Hugging Face Optimum: https://huggingface.co/docs/optimum/
- NVIDIA Apex: https://github.com/NVIDIA/apex
