# 量化技術 (Quantization)

> 深入理解 LLM 量化技術，從理論到實踐

## 目錄

- [什麼是量化](#什麼是量化)
- [量化原理](#量化原理)
- [量化方法對比](#量化方法對比)
- [實作範例](#實作範例)
- [最佳實踐](#最佳實踐)

## 什麼是量化

量化是將模型權重和激活值從高精度數值（如 FP32、FP16）轉換為低精度數值（如 INT8、INT4）的過程，以減少模型大小和加速推論。

### 為什麼需要量化？

1. **減少記憶體使用**
   - FP32 → FP16：記憶體減少 50%
   - FP32 → INT8：記憶體減少 75%
   - FP32 → INT4：記憶體減少 87.5%

2. **加速推論**
   - 低精度計算更快（INT8 運算 > FP16 運算 > FP32 運算）
   - 減少記憶體帶寬需求
   - 更好的快取利用率

3. **降低部署成本**
   - 可使用更小的 GPU
   - 支援更大的批次大小
   - 降低雲端運算成本

### 量化的挑戰

- **精度損失**：低精度表示可能導致模型性能下降
- **量化誤差**：捨入誤差累積影響輸出品質
- **硬體支援**：需要硬體支援低精度運算（如 Tensor Cores）

## 量化原理

### 量化公式

**線性量化（對稱）**：
```
Q = round(X / scale)
X_reconstructed = Q × scale
```

**線性量化（非對稱）**：
```
Q = round(X / scale + zero_point)
X_reconstructed = (Q - zero_point) × scale
```

其中：
- `X`：原始浮點數值
- `Q`：量化後的整數值
- `scale`：縮放因子
- `zero_point`：零點偏移（非對稱量化）

### 量化類型

#### 1. 對稱量化 (Symmetric Quantization)

零點固定為 0，量化範圍關於零對稱。

**優點**：
- 計算簡單，不需要零點偏移
- 硬體實現更高效

**缺點**：
- 如果數據分佈不對稱，會浪費表示範圍

#### 2. 非對稱量化 (Asymmetric Quantization)

零點可調整，更好地適應數據分佈。

**優點**：
- 更好地利用量化範圍
- 適合不對稱分佈的數據

**缺點**：
- 計算複雜度稍高

### 量化粒度

#### 1. 逐張量量化 (Per-Tensor Quantization)

整個張量使用同一個 scale。

**優點**：簡單、快速
**缺點**：精度較低

#### 2. 逐通道量化 (Per-Channel Quantization)

每個輸出通道使用不同的 scale。

**優點**：精度更高
**缺點**：計算稍慢

#### 3. 逐組量化 (Group-wise Quantization)

將權重分組，每組使用不同的 scale。

**優點**：平衡精度和效率
**缺點**：需要特殊硬體支援

## 量化方法對比

### 1. 動態量化 (Dynamic Quantization)

推論時動態計算激活值的量化參數。

```python
# 權重預先量化，激活值動態量化
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**特點**：
- ✅ 實施簡單，無需校準數據
- ✅ 激活值量化更精確
- ❌ 推論時有額外計算開銷
- **適用**：記憶體受限，權重佔主導

### 2. 靜態量化 (Static Quantization / PTQ)

使用校準數據預先計算量化參數。

```python
# 權重和激活值都預先量化
model = torch.quantization.quantize_static(
    model, calibration_data, ...
)
```

**特點**：
- ✅ 推論速度最快
- ✅ 記憶體佔用最小
- ❌ 需要代表性校準數據
- ❌ 可能有精度損失
- **適用**：有校準數據，追求極致性能

### 3. 量化感知訓練 (QAT)

訓練過程中模擬量化效果。

```python
# 訓練時模擬量化
model = torch.quantization.prepare_qat(model)
train(model)  # 微調
model = torch.quantization.convert(model)
```

**特點**：
- ✅ 精度損失最小
- ✅ 可達到接近全精度的性能
- ❌ 需要重新訓練
- ❌ 計算成本高
- **適用**：追求最佳精度，有訓練資源

### 4. GPTQ (GPT Quantization)

專為 Transformer 設計的訓練後量化方法。

**核心思想**：
- 逐層量化，最小化量化誤差
- 使用 Hessian 矩陣的二階信息
- 無需反向傳播

**特點**：
- ✅ 4-bit 量化精度損失小
- ✅ 支援大模型（LLaMA、GPT）
- ✅ 量化速度較快
- ❌ 需要校準數據
- **適用**：大模型壓縮，生產部署

**論文**：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

### 5. AWQ (Activation-aware Weight Quantization)

考慮激活值分佈的權重量化方法。

**核心思想**：
- 保護重要的權重通道（激活值大的通道）
- 動態調整量化 scale
- 1% 的突出權重保持高精度

**特點**：
- ✅ 4-bit 量化精度最佳
- ✅ 推論速度快（相比 GPTQ）
- ✅ 記憶體效率高
- ❌ 量化時間較長
- **適用**：追求精度，生產環境

**論文**：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

### 6. GGUF (GPT-Generated Unified Format)

llama.cpp 使用的量化格式。

**特點**：
- ✅ 支援 2-8 bit 多種精度
- ✅ CPU 推論優化
- ✅ 跨平台支援（Mac、Windows、Linux）
- ✅ 混合精度（不同層不同精度）
- ❌ 主要針對 CPU
- **適用**：本地部署，無 GPU 環境

**格式**：Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0

## 量化方法詳細對比

| 方法 | 精度 | 量化速度 | 推論速度 | 記憶體 | 精度保持 | 校準數據 | 硬體要求 |
|------|------|---------|---------|--------|---------|---------|---------|
| FP16 | 16-bit | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ | 50% | ⭐⭐⭐⭐⭐ | ❌ | GPU |
| Dynamic INT8 | 8-bit | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | 75% | ⭐⭐⭐⭐ | ❌ | CPU/GPU |
| Static INT8 | 8-bit | ⚡⚡⚡⚡ | ⚡⚡⚡⚡⚡ | 75% | ⭐⭐⭐⭐ | ✅ | CPU/GPU |
| GPTQ | 4-bit | ⚡⚡⚡ | ⚡⚡⚡⚡ | 87.5% | ⭐⭐⭐⭐ | ✅ | GPU |
| AWQ | 4-bit | ⚡⚡ | ⚡⚡⚡⚡⚡ | 87.5% | ⭐⭐⭐⭐⭐ | ✅ | GPU |
| GGUF Q4 | 4-bit | ⚡⚡⚡⚡ | ⚡⚡⚡ | 87.5% | ⭐⭐⭐ | ❌ | CPU |
| GGUF Q8 | 8-bit | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | 75% | ⭐⭐⭐⭐ | ❌ | CPU |

## 實作範例

### 範例 1：基礎量化

[01_basic_quantization.py](./01_basic_quantization.py)

- FP16 量化
- INT8 動態量化
- 記憶體和速度對比

### 範例 2：GPTQ 量化

[02_gptq_quantization.py](./02_gptq_quantization.py)

- 使用 AutoGPTQ 量化 LLaMA 模型
- 4-bit GPTQ 量化
- 精度評估

### 範例 3：AWQ 量化

[03_awq_quantization.py](./03_awq_quantization.py)

- 使用 AutoAWQ 量化模型
- 對比 AWQ vs GPTQ
- 推論速度測試

### 範例 4：GGUF 量化

[04_gguf_quantization.py](./04_gguf_quantization.py)

- 轉換模型為 GGUF 格式
- 多種量化精度（Q4_K_M, Q5_K_M, Q8_0）
- CPU 推論測試

### 範例 5：量化方法綜合對比

[05_quantization_comparison.py](./05_quantization_comparison.py)

- 對比所有量化方法
- 生成詳細報告
- 視覺化對比圖表

## 量化工具生態

### Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM
import torch

# FP16
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype=torch.float16
)

# 8-bit (需要 bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit (需要 bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_4bit=True,
    device_map="auto"
)
```

### AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)
model.quantize(calibration_data)
model.save_quantized(output_dir)
```

### AutoAWQ

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(tokenizer, quant_config={
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
})
model.save_quantized(output_dir)
```

### llama.cpp

```bash
# 轉換為 GGUF 格式
python convert.py model_name --outtype f16

# 量化為不同精度
./quantize model.gguf model_q4_k_m.gguf Q4_K_M
./quantize model.gguf model_q5_k_m.gguf Q5_K_M
./quantize model.gguf model_q8_0.gguf Q8_0
```

## 最佳實踐

### 1. 選擇量化方法

根據你的需求選擇：

```
追求最佳精度？
├─ 有訓練資源 → QAT (量化感知訓練)
└─ 只能 PTQ → AWQ

平衡精度和速度？
├─ GPU 部署 → GPTQ 或 AWQ
└─ CPU 部署 → GGUF Q5/Q6

追求極致壓縮？
├─ GPU 可用 → GPTQ/AWQ 4-bit
└─ 僅 CPU → GGUF Q4

快速實驗？
└─ bitsandbytes 4/8-bit
```

### 2. 量化流程

**標準流程**：
1. 準備校準數據（代表性樣本，512-1024 條）
2. 選擇量化方法和配置
3. 執行量化
4. 評估精度（perplexity、準確率等）
5. 測試推論速度和記憶體
6. 調整參數優化

### 3. 精度評估

```python
# 計算 perplexity
from torch.nn import CrossEntropyLoss

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# 對比量化前後
ppl_original = calculate_perplexity(original_model, tokenizer, test_text)
ppl_quantized = calculate_perplexity(quantized_model, tokenizer, test_text)
print(f"Perplexity 變化: {ppl_original:.2f} → {ppl_quantized:.2f}")
```

### 4. 常見陷阱

❌ **不要**：
- 使用過少的校準數據（<100 樣本）
- 忽略精度評估直接部署
- 混用不同量化方法的權重
- 在不支援的硬體上強行使用

✅ **應該**：
- 使用代表性的校準數據
- 始終評估量化後的模型性能
- 根據硬體選擇合適的量化方法
- 保留原始模型用於對比

### 5. 效能調優

```python
# 1. 調整 group_size（GPTQ/AWQ）
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,  # 嘗試 64, 128, 256
)

# 2. 使用更多校準數據
calibration_data = dataset[:2048]  # 增加樣本數

# 3. 啟用 desc_act（GPTQ）
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # 可能提升精度
)

# 4. 混合精度量化
# 對敏感層使用更高精度
sensitive_layers = ["lm_head", "embed_tokens"]
# 其他層使用 4-bit，敏感層使用 8-bit
```

## 量化效果示例

### LLaMA-7B 量化對比

| 方法 | 模型大小 | WikiText PPL | MMLU | 推論速度 | GPU 記憶體 |
|------|---------|--------------|------|---------|-----------|
| FP16 | 13.5 GB | 5.68 | 45.3% | 1.0x | 13.5 GB |
| INT8 | 7.0 GB | 5.70 | 45.1% | 1.8x | 7.2 GB |
| GPTQ 4-bit | 3.5 GB | 5.82 | 44.2% | 2.1x | 4.0 GB |
| AWQ 4-bit | 3.5 GB | 5.74 | 44.9% | 2.3x | 4.0 GB |
| GGUF Q4_K_M | 3.8 GB | 5.89 | 43.8% | 1.2x (CPU) | - |

*註：推論速度相對於 FP16，測試環境：A100 40GB (GPU) / AMD EPYC 7742 (CPU)*

## 參考資源

### 論文

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)

### 開源專案

- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Optimum](https://github.com/huggingface/optimum)

### 教程和文檔

- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [vLLM Quantization Support](https://docs.vllm.ai/en/latest/quantization/supported_hardware.html)

---

**下一步**：動手實作 [01_basic_quantization.py](./01_basic_quantization.py) 🚀
