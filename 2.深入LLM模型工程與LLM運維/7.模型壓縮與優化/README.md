# 模型壓縮與優化

## 目錄

### 📚 主要章節
1. [前言](#1-前言)
2. [量化技術 (Quantization)](#2-量化技術-quantization)
   - 2.1 [量化基礎原理](#21-量化基礎原理)
   - 2.2 [常見量化格式](#22-常見量化格式)
   - 2.3 [PTQ vs QAT](#23-ptq-vs-qat)
   - 2.4 [進階量化方法](#24-進階量化方法)
3. [剪枝技術 (Pruning)](#3-剪枝技術-pruning)
4. [知識蒸餾 (Knowledge Distillation)](#4-知識蒸餾-knowledge-distillation)
5. [量化工具與方法比較](#5-量化工具與方法比較)
6. [Python 實作範例](#6-python-實作範例)
7. [性能評估與基準測試](#7-性能評估與基準測試)
8. [延伸閱讀](#8-延伸閱讀)

### 🚀 進階專題
- [低秩分解 (Low-Rank Factorization)](./advanced/low_rank_factorization.md) - LoRA 詳解與應用
- [混合精度策略 (Mixed-Precision)](./advanced/mixed_precision.md) - 混合精度訓練與推理
- [實際部署案例](./advanced/deployment_cases.md) - 5 個真實部署案例分析

### 📖 實用指南
- [速查表 (Quick Reference)](./QUICK_REFERENCE.md) - 常用命令與配置速查
- [練習題 (Exercises)](./EXERCISES.md) - 從基礎到進階的實作練習
- [故障排除 (Troubleshooting)](./guides/troubleshooting.md) - 常見問題解決方案
- [硬體選擇指南](./guides/hardware_guide.md) - GPU/CPU 選擇建議
- [最佳實踐](./guides/best_practices.md) - 生產環境部署最佳實踐

---

## 1. 前言

隨著大型語言模型 (LLM) 的參數規模不斷增長，模型的部署和推論成本成為重要挑戰：

- **GPT-3 (175B)**：需要 ~350GB 顯存 (FP16)
- **LLaMA-2-70B**：需要 ~140GB 顯存 (FP16)
- **LLaMA-2-7B**：需要 ~14GB 顯存 (FP16)

**模型壓縮與優化的目標**：
- 減少模型大小（降低存儲需求）
- 降低記憶體佔用（適應消費級硬體）
- 加速推論速度（降低延遲）
- 保持模型性能（最小化精度損失）

**主要技術**：
1. **量化 (Quantization)**：降低數值精度
2. **剪枝 (Pruning)**：移除不重要的參數/連接
3. **知識蒸餾 (Distillation)**：訓練小模型模仿大模型
4. **低秩分解 (Low-Rank Factorization)**：參數矩陣分解（詳見 [LoRA 專題](./advanced/low_rank_factorization.md)）

本章主要介紹**量化、剪枝和知識蒸餾**技術。**低秩分解**（包括 LoRA、QLoRA 等）請參見 [進階專題](./advanced/low_rank_factorization.md)。

---

## 2. 量化技術 (Quantization)

### 2.1 量化基礎原理

**量化**是將高精度浮點數（如 FP32）映射到低精度表示（如 INT8）的過程。

#### 數值表示回顧

| 格式 | 位元數 | 範圍 | 精度 | 典型用途 |
|------|--------|------|------|---------|
| FP32 (Float32) | 32 bits | ±3.4×10³⁸ | ~7 位數 | 訓練、高精度推理 |
| FP16 (Half) | 16 bits | ±65,504 | ~3 位數 | 訓練、推理 |
| BF16 (BFloat16) | 16 bits | ±3.4×10³⁸ | ~2 位數 | 訓練（與 FP32 範圍相同）|
| INT8 | 8 bits | -128 ~ 127 | 整數 | 推理 |
| INT4 | 4 bits | -8 ~ 7 | 整數 | 極限壓縮 |

#### 基本量化公式

**對稱量化 (Symmetric Quantization)**：
```
x_quant = round(x_float / scale)
x_float ≈ x_quant × scale
```

其中 `scale = max(|x_float|) / (2^(bits-1) - 1)`

**非對稱量化 (Asymmetric Quantization)**：
```
x_quant = round((x_float - zero_point) / scale)
x_float ≈ x_quant × scale + zero_point
```

更靈活，可處理非對稱分佈。

#### 量化顆粒度

1. **逐張量量化 (Per-Tensor)**：整個張量共用一組量化參數
2. **逐通道量化 (Per-Channel)**：每個輸出通道有獨立量化參數（更精確）
3. **逐群組量化 (Per-Group)**：將參數分組，每組獨立量化

### 2.2 常見量化格式

#### 2.2.1 FP16 (Half Precision)

**特性**：
- 16 位元浮點數
- 模型大小減半（相對 FP32）
- GPU 原生支援，速度快
- 精度損失極小

**應用**：
- 幾乎所有現代深度學習訓練
- 高品質推理

**PyTorch 使用**：
```python
model = model.half()  # 轉換為 FP16
model = model.to(torch.float16)
```

#### 2.2.2 BF16 (BFloat16)

**特性**：
- 與 FP32 相同的指數範圍，但尾數較短
- 訓練時更穩定（不易溢出）
- Google TPU、NVIDIA Ampere+ 架構支援

**優勢**：
- 可直接替代 FP32 進行訓練
- 不需要 loss scaling

#### 2.2.3 INT8 量化

**特性**：
- 8 位元整數
- 模型大小減少 4 倍（相對 FP32）
- 推論速度提升 2-4 倍
- 精度損失 < 1%（適當校準後）

**應用場景**：
- 生產環境推理
- 邊緣設備部署

**挑戰**：
- 需要校準資料
- 量化感知訓練可能需要較長時間

#### 2.2.4 4-bit 量化

**特性**：
- 4 位元整數
- 模型大小減少 8 倍
- 精度損失 2-5%

**代表技術**：
- **QLoRA**：4-bit NormalFloat (NF4) + LoRA
- **GPTQ**：Group-wise quantization
- **AWQ**：Activation-aware Weight Quantization

### 2.3 PTQ vs QAT

#### 2.3.1 訓練後量化 (Post-Training Quantization, PTQ)

在已訓練好的模型上直接應用量化，無需重新訓練。

**優點**：
- 快速：幾分鐘到幾小時
- 簡單：無需訓練基礎設施
- 適合大型預訓練模型

**缺點**：
- 精度損失較大（特別是低於 INT8）
- 對激活值分佈敏感

**流程**：
```
1. 準備校準資料集（通常 100-1000 樣本）
2. 運行前向傳播，收集激活值統計
3. 計算量化參數（scale, zero_point）
4. 應用量化
5. 評估量化後模型
```

#### 2.3.2 量化感知訓練 (Quantization-Aware Training, QAT)

在訓練過程中模擬量化效果，讓模型適應量化誤差。

**優點**：
- 精度損失最小
- 可達到接近全精度性能

**缺點**：
- 需要完整訓練流程
- 計算成本高
- 需要訓練資料和算力

**偽量化**：
```
forward pass:
    x_float → quantize → x_quant → dequantize → x_float

backward pass:
    gradient 通過 "straight-through estimator" 反向傳播
```

**何時使用**：
- 需要極低精度（INT4, INT2）
- 對精度要求嚴格
- 有充足訓練資源

### 2.4 進階量化方法

#### 2.4.1 QLoRA (4-bit Quantization + LoRA)

**核心思想**：
- 基礎模型用 4-bit NormalFloat (NF4) 量化
- LoRA 適配器保持 FP16/BF16
- 僅訓練 LoRA 參數

**NF4 (4-bit NormalFloat)**：
- 專為正態分佈設計的量化格式
- 理論上最優化資訊熵

**優勢**：
- 7B 模型僅需 ~5GB 顯存（含 LoRA）
- 微調質量接近全精度 LoRA
- 大幅降低微調門檻

**實作**：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # 嵌套量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

#### 2.4.2 GPTQ (Accurate Post-Training Quantization)

**核心思想**：
- 逐層量化，最小化重建誤差
- 使用二階資訊（Hessian 矩陣）
- Group-wise quantization

**特點**：
- INT4/INT3 量化，精度損失 < 1%
- 僅需少量校準資料（128 樣本）
- 推論速度快（搭配專用 kernel）

**適用場景**：
- 推理部署
- 需要極限壓縮

**工具**：AutoGPTQ, ExLlama

#### 2.4.3 AWQ (Activation-aware Weight Quantization)

**核心思想**：
- 識別對激活值影響大的權重
- 對重要權重使用更高精度或跳過量化
- 基於激活值分佈動態調整量化

**優勢**：
- 相比 GPTQ，精度更高
- 特別適合 3-4 bit 量化
- 推論速度與 GPTQ 相當

**適用場景**：
- 超低精度量化（INT3/INT4）
- 對精度要求嚴格的應用

#### 2.4.4 GGUF / llama.cpp

**GGUF (GPT-Generated Unified Format)**：
- llama.cpp 生態系統的量化格式
- 支持多種量化方案（Q4_0, Q5_1, Q8_0 等）
- CPU 推論優化

**特點**：
- 針對 CPU 優化（支援 AVX2, ARM NEON）
- 支援 Metal (macOS), CUDA, OpenCL
- 內存映射，支援大模型

**量化方案**：
```
Q2_K: 2.5-3 bpw (bits per weight) - 極限壓縮
Q3_K_M: ~3.5 bpw - 平衡
Q4_K_M: ~4.5 bpw - 推薦
Q5_K_M: ~5.5 bpw - 高品質
Q6_K: ~6 bpw - 幾乎無損
Q8_0: 8 bpw - 高精度
```

**使用場景**：
- 本地部署（消費級硬體）
- 隱私敏感應用
- 離線推理

---

## 3. 剪枝技術 (Pruning)

### 3.1 基本概念

**剪枝**：移除神經網路中不重要的權重或神經元。

**理論基礎**：
- Lottery Ticket Hypothesis：稀疏子網路可達到相似性能
- 冗餘假設：大模型存在大量冗餘參數

### 3.2 剪枝類型

#### 非結構化剪枝 (Unstructured Pruning)
- 移除單個權重
- 稀疏度高，但需要專用硬體/軟體支援
- 壓縮率高，但加速有限

#### 結構化剪枝 (Structured Pruning)
- 移除整個神經元、通道或層
- 直接減少模型大小
- 硬體友好，加速明顯

### 3.3 剪枝方法

#### 3.3.1 Magnitude Pruning
- 移除絕對值最小的權重
- 簡單有效
- 需要微調恢復性能

#### 3.3.2 Attention Head Pruning
- 移除不重要的 attention heads
- 針對 Transformer 架構
- 可減少 10-20% 參數，精度損失 < 1%

#### 3.3.3 Layer Dropping
- 移除整個 Transformer 層
- 最激進，但可能影響性能
- 適合深層模型

### 3.4 剪枝流程

```
1. 訓練全模型
2. 分析重要性（權重大小、梯度、Hessian等）
3. 移除不重要參數
4. 微調 (Fine-tuning)
5. 重複 2-4（迭代剪枝）
```

---

## 4. 知識蒸餾 (Knowledge Distillation)

### 4.1 基本原理

訓練小模型（學生）模仿大模型（教師）的行為。

**損失函式**：
```
L = α × L_hard + (1-α) × L_soft

L_hard: 學生模型與真實標籤的交叉熵
L_soft: 學生與教師輸出分佈的 KL 散度
```

**Soft Targets**：
```
p_soft = softmax(logits / T)
```
其中 T 為溫度，T > 1 使分佈更平滑，包含更多資訊。

### 4.2 蒸餾方法

#### 4.2.1 Response-based Distillation
- 匹配最終輸出層
- 最簡單，但資訊有限

#### 4.2.2 Feature-based Distillation
- 匹配中間層特徵
- 需要對齊層維度

#### 4.2.3 Relation-based Distillation
- 匹配樣本間關係
- 更豐富的知識轉移

### 4.3 LLM 蒸餾特殊考慮

- **序列級蒸餾**：匹配生成序列分佈
- **On-policy distillation**：使用學生模型生成資料
- **Chain-of-thought distillation**：遷移推理能力

---

## 5. 量化工具與方法比較

### 5.1 工具對比表

| 工具 | 量化精度 | 推理後端 | 微調支援 | 適用場景 |
|------|---------|---------|---------|---------|
| **bitsandbytes** | 8bit, 4bit (NF4) | PyTorch | QLoRA | 微調、研究 |
| **GPTQ** | 4bit, 3bit, 2bit | ExLlama, CUDA | 否 | 推理部署 |
| **AWQ** | 4bit | AutoAWQ, vLLM | 否 | 推理部署 |
| **llama.cpp** | Q2_K ~ Q8_0 | CPU, Metal, CUDA | 否 | 本地部署 |
| **ONNX Runtime** | INT8, FP16 | ONNX | 否 | 生產環境 |
| **TensorRT** | INT8, FP16 | TensorRT | 否 | NVIDIA GPU 推理 |

### 5.2 選擇指南

#### 場景 1：微調 7B-70B 模型
```
推薦：QLoRA (bitsandbytes)
配置：
- 4-bit NF4 量化
- bf16 compute dtype
- double quantization
優勢：顯存效率高，微調質量好
```

#### 場景 2：生產環境推理（GPU）
```
推薦：GPTQ 或 AWQ
配置：
- INT4 量化
- 搭配 vLLM 或 TGI
優勢：推論速度快，吞吐量高
```

#### 場景 3：本地/邊緣設備
```
推薦：llama.cpp (GGUF)
配置：
- Q4_K_M 或 Q5_K_M
- Metal (Mac) / CPU
優勢：跨平台，硬體要求低
```

#### 場景 4：雲端大規模部署
```
推薦：TensorRT-LLM
配置：
- INT8 或 FP16
- NVIDIA GPU
優勢：極致性能，多GPU優化
```

### 5.3 精度與速度權衡

**經驗資料（LLaMA-7B）**：

| 格式 | 模型大小 | 推論速度 | 困惑度 (PPL) | 顯存 (推理) |
|------|---------|---------|-------------|-----------|
| FP32 | 28 GB | 1.0x | 5.68 | 30 GB |
| FP16 | 14 GB | 1.8x | 5.68 | 16 GB |
| INT8 (GPTQ) | 7 GB | 2.5x | 5.75 | 10 GB |
| INT4 (GPTQ) | 3.5 GB | 3.5x | 6.02 | 8 GB |
| Q4_K_M (GGUF) | 4 GB | 2.8x (CPU) | 5.98 | 6 GB |

**註**：實際性能取決於硬體、批次大小、序列長度等因素。

---

## 6. Python 實作範例

### 6.1 基礎 PTQ 實作（PyTorch）

```python
import torch
import torch.nn as nn
import torch.quantization as quant

# 定義簡單模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================================
# 1. 動態量化（最簡單）
# ============================================================================
def dynamic_quantization_demo():
    """動態量化範例 - 僅量化權重，激活值保持 FP32"""
    print("=" * 60)
    print("動態量化 (Dynamic Quantization)")
    print("=" * 60)

    # 建立模型
    model = SimpleModel()
    model.eval()

    # 應用動態量化
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear},  # 量化的層類型
        dtype=torch.qint8
    )

    # 測試
    x = torch.randn(1, 128)
    output_fp32 = model(x)
    output_int8 = quantized_model(x)

    print(f"FP32 模型大小: {get_model_size(model):.2f} MB")
    print(f"INT8 模型大小: {get_model_size(quantized_model):.2f} MB")
    print(f"輸出差異: {torch.abs(output_fp32 - output_int8).mean():.6f}")

    return quantized_model

def get_model_size(model):
    """計算模型大小（MB）"""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size

# ============================================================================
# 2. 靜態量化（需要校準）
# ============================================================================
def static_quantization_demo():
    """靜態量化範例 - 權重和激活值都量化"""
    print("\n" + "=" * 60)
    print("靜態量化 (Static Quantization)")
    print("=" * 60)

    # 建立模型
    model = SimpleModel()
    model.eval()

    # 設定量化配置
    model.qconfig = quant.get_default_qconfig('fbgemm')  # x86 CPU
    # model.qconfig = quant.get_default_qconfig('qnnpack')  # ARM CPU

    # 插入觀察器
    model_prepared = quant.prepare(model)

    # 校準（用代表性資料）
    print("開始校準...")
    calibration_data = [torch.randn(1, 128) for _ in range(100)]

    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data)

    # 轉換為量化模型
    quantized_model = quant.convert(model_prepared)

    # 測試
    x = torch.randn(1, 128)
    with torch.no_grad():
        output_fp32 = model(x)
        output_int8 = quantized_model(x)

    print(f"FP32 模型大小: {get_model_size(model):.2f} MB")
    print(f"INT8 模型大小: {get_model_size(quantized_model):.2f} MB")
    print(f"輸出差異: {torch.abs(output_fp32 - output_int8).mean():.6f}")

    return quantized_model

# 執行範例
import os
dynamic_quantization_demo()
static_quantization_demo()
```

### 6.2 使用 bitsandbytes 進行 8-bit/4-bit 量化

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# 1. 8-bit 量化（LLM.int8()）
# ============================================================================
def load_8bit_model():
    """載入 8-bit 量化模型"""
    print("=" * 60)
    print("8-bit 量化（LLM.int8()）")
    print("=" * 60)

    model_name = "gpt2"

    # FP16 基準
    print("\n載入 FP16 模型...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 8-bit 量化
    print("載入 8-bit 量化模型...")
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )

    # 比較
    print(f"\nFP16 顯存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 釋放 FP16 模型
    del model_fp16
    torch.cuda.empty_cache()

    model_int8_mem = torch.cuda.memory_allocated() / 1e9
    print(f"INT8 顯存: {model_int8_mem:.2f} GB")

    # 測試推理
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model_int8.generate(**inputs, max_length=50)

    print(f"\n生成結果:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return model_int8

# ============================================================================
# 2. 4-bit 量化（QLoRA）
# ============================================================================
def load_4bit_model():
    """載入 4-bit NF4 量化模型（QLoRA）"""
    print("\n" + "=" * 60)
    print("4-bit NF4 量化（QLoRA）")
    print("=" * 60)

    model_name = "gpt2"

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # 嵌套量化（進一步壓縮）
    )

    print("載入 4-bit 量化模型...")
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model_4bit_mem = torch.cuda.memory_allocated() / 1e9
    print(f"INT4 顯存: {model_4bit_mem:.2f} GB")

    # 測試推理
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model_4bit.generate(**inputs, max_length=50)

    print(f"\n生成結果:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return model_4bit

# 執行範例
if torch.cuda.is_available():
    load_8bit_model()
    load_4bit_model()
else:
    print("需要 CUDA GPU 來運行 bitsandbytes 量化")
```

### 6.3 使用 GPTQ 量化

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig

def gptq_quantization():
    """使用 GPTQ 進行 4-bit 量化"""
    print("=" * 60)
    print("GPTQ 4-bit 量化")
    print("=" * 60)

    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 準備校準資料
    calibration_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large amounts of data.",
        # ... 更多樣本
    ]

    # GPTQ 配置
    gptq_config = GPTQConfig(
        bits=4,  # 4-bit 量化
        dataset="c4",  # 或提供自定義資料
        group_size=128,  # Group size
        desc_act=False  # 是否量化 activation
    )

    print("開始量化（這可能需要幾分鐘）...")

    # 載入並量化
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=gptq_config,
        device_map="auto"
    )

    # 保存量化模型
    model.save_pretrained("./opt-125m-gptq-4bit")
    tokenizer.save_pretrained("./opt-125m-gptq-4bit")

    print("量化完成！模型已保存。")

    # 測試
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=50)
    print(f"\n生成結果:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 執行
# gptq_quantization()  # 需要較長時間
```

### 6.4 剪枝實作

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def pruning_demo():
    """神經網路剪枝範例"""
    print("=" * 60)
    print("神經網路剪枝 (Pruning)")
    print("=" * 60)

    # 建立簡單模型
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleNN()

    # 計算原始參數數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"原始參數數量: {total_params:,}")

    # ========================================================================
    # 1. L1 非結構化剪枝（移除權重絕對值最小的）
    # ========================================================================
    print("\n1. L1 非結構化剪枝")

    # 對 fc1 進行 30% 剪枝
    prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

    # 檢查稀疏度
    sparsity_fc1 = 100. * float(torch.sum(model.fc1.weight == 0)) / float(model.fc1.weight.nelement())
    print(f"fc1 稀疏度: {sparsity_fc1:.2f}%")

    # ========================================================================
    # 2. 全局剪枝（跨層）
    # ========================================================================
    print("\n2. 全局剪枝")

    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,  # 全局 50% 剪枝
    )

    # 計算剪枝後的參數數量
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_zeros += float(torch.sum(module.weight == 0))
            total_params += float(module.weight.nelement())

    global_sparsity = 100. * total_zeros / total_params
    print(f"全局稀疏度: {global_sparsity:.2f}%")

    # ========================================================================
    # 3. 結構化剪枝（移除整個神經元）
    # ========================================================================
    print("\n3. 結構化剪枝（移除神經元）")

    # 建立新模型
    model2 = SimpleNN()

    # 對 fc1 進行結構化剪枝（移除輸出神經元）
    prune.ln_structured(
        model2.fc1,
        name="weight",
        amount=0.3,  # 移除 30% 的輸出神經元
        n=2,  # L2 norm
        dim=0  # 沿輸出維度
    )

    print(f"fc1 權重形狀: {model2.fc1.weight.shape}")
    print(f"非零行數: {torch.sum(torch.sum(model2.fc1.weight, dim=1) != 0).item()}")

    # ========================================================================
    # 4. 永久移除（使剪枝生效）
    # ========================================================================
    print("\n4. 永久應用剪枝")

    # 移除 prune 的重新參數化，使剪枝永久生效
    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')

    print("剪枝已永久應用到模型")

    # 可以正常保存模型
    torch.save(model.state_dict(), "pruned_model.pth")
    print("剪枝模型已保存")

# 執行
pruning_demo()
```

### 6.5 知識蒸餾實作

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def knowledge_distillation_demo():
    """知識蒸餾範例"""
    print("=" * 60)
    print("知識蒸餾 (Knowledge Distillation)")
    print("=" * 60)

    # 教師模型（大模型）
    class TeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # 學生模型（小模型）
    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    # 蒸餾損失函式
    def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
        """
        組合硬標籤損失和軟標籤損失

        Args:
            student_logits: 學生模型輸出
            teacher_logits: 教師模型輸出
            labels: 真實標籤
            temperature: 溫度參數
            alpha: 平衡係數
        """
        # 硬標籤損失（與真實標籤的交叉熵）
        hard_loss = F.cross_entropy(student_logits, labels)

        # 軟標籤損失（與教師模型的 KL 散度）
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

        # 組合損失
        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return total_loss, hard_loss, soft_loss

    # 訓練函數
    def train_student(teacher, student, train_loader, epochs=10, temperature=3.0, alpha=0.5):
        """訓練學生模型"""
        teacher.eval()  # 教師模型固定
        student.train()

        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # 前向傳播
                with torch.no_grad():
                    teacher_logits = teacher(data)

                student_logits = student(data)

                # 計算蒸餾損失
                loss, hard_loss, soft_loss = distillation_loss(
                    student_logits, teacher_logits, target,
                    temperature=temperature, alpha=alpha
                )

                # 反向傳播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 建立模型
    teacher = TeacherModel()
    student = StudentModel()

    # 模型大小比較
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"\n教師模型參數: {teacher_params:,}")
    print(f"學生模型參數: {student_params:,}")
    print(f"壓縮比: {teacher_params / student_params:.2f}x")

    # 模擬訓練（實際使用時需要真實資料）
    print("\n開始蒸餾訓練...")
    print("（這是模擬，實際需要真實資料集）")

    # 建立假資料用於演示
    fake_data = [(torch.randn(32, 128), torch.randint(0, 10, (32,))) for _ in range(10)]

    train_student(teacher, student, fake_data, epochs=3)

    print("\n蒸餾完成！")

# 執行
knowledge_distillation_demo()
```

---

## 7. 性能評估與基準測試

### 7.1 評估指標

#### 模型品質指標
- **困惑度 (Perplexity)**：語言模型的標準指標
- **準確率**：分類任務
- **BLEU/ROUGE**：生成任務
- **Human Evaluation**：人工評估

#### 效率指標
- **模型大小**：磁碟空間（MB/GB）
- **推論速度**：tokens/second
- **延遲 (Latency)**：首token時間 (TTFT), 平均token時間
- **吞吐量 (Throughput)**：requests/second
- **顯存佔用**：推理時的峰值顯存

### 7.2 基準測試腳本

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_model(model_path, quantization=None):
    """基準測試模型性能"""
    print("=" * 60)
    print(f"基準測試: {model_path}")
    if quantization:
        print(f"量化: {quantization}")
    print("=" * 60)

    # 載入模型
    start_time = time.time()

    kwargs = {"device_map": "auto"}
    if quantization == "8bit":
        kwargs["load_in_8bit"] = True
    elif quantization == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    load_time = time.time() - start_time
    print(f"載入時間: {load_time:.2f} 秒")

    # 顯存使用
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"顯存使用: {memory_used:.2f} GB")

    # 模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"參數數量: {total_params / 1e9:.2f}B")

    # 推論速度測試
    prompt = "Once upon a time, in a land far away,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 預熱
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    # 測試生成速度
    num_tokens = 100
    num_runs = 5

    total_time = 0
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time += time.time() - start

    avg_time = total_time / num_runs
    tokens_per_sec = num_tokens / avg_time

    print(f"\n推理性能:")
    print(f"  生成 {num_tokens} tokens: {avg_time:.2f} 秒")
    print(f"  速度: {tokens_per_sec:.2f} tokens/秒")
    print(f"  平均每 token: {1000 * avg_time / num_tokens:.2f} ms")

    # 生成樣本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n生成樣本:\n{generated_text}")

    return {
        "load_time": load_time,
        "memory_gb": memory_used if torch.cuda.is_available() else None,
        "params_b": total_params / 1e9,
        "tokens_per_sec": tokens_per_sec
    }

# 執行基準測試
if __name__ == "__main__":
    model_name = "gpt2"

    # 測試不同量化方法
    results = {}

    print("\n" + "=" * 80)
    print("開始基準測試")
    print("=" * 80)

    # FP16 (基準)
    results['fp16'] = benchmark_model(model_name)

    # 8-bit
    if torch.cuda.is_available():
        results['8bit'] = benchmark_model(model_name, quantization="8bit")

    # 4-bit
    if torch.cuda.is_available():
        results['4bit'] = benchmark_model(model_name, quantization="4bit")

    # 總結
    print("\n" + "=" * 80)
    print("基準測試總結")
    print("=" * 80)
    print(f"{'方法':<10} {'顯存(GB)':<12} {'速度(tok/s)':<15} {'參數(B)':<12}")
    print("-" * 80)
    for method, result in results.items():
        mem = f"{result['memory_gb']:.2f}" if result['memory_gb'] else "N/A"
        print(f"{method:<10} {mem:<12} {result['tokens_per_sec']:<15.2f} {result['params_b']:<12.2f}")
```

### 7.3 困惑度評估

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", subset="wikitext-2-raw-v1", split="test", max_samples=100):
    """評估模型困惑度"""
    print("=" * 60)
    print("困惑度評估 (Perplexity Evaluation)")
    print("=" * 60)

    # 載入資料集
    dataset = load_dataset(dataset_name, subset, split=split)

    # 計算困惑度
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        text = example["text"]
        if len(text.strip()) == 0:
            continue

        # Tokenize
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(model.device)

        # 計算損失
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

        if (i + 1) % 10 == 0:
            print(f"已處理 {i+1}/{max_samples} 樣本...")

    # 計算困惑度
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"\n結果:")
    print(f"  平均損失: {avg_loss:.4f}")
    print(f"  困惑度 (PPL): {perplexity:.2f}")

    return perplexity

# 使用範例
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# ppl = evaluate_perplexity(model, tokenizer)
```

---

## 8. 延伸閱讀

### 論文

#### 量化
1. **LLM.int8()**: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al., 2022)
2. **QLoRA**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
3. **GPTQ**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)
4. **AWQ**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)
5. **SmoothQuant**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2022)

#### 剪枝
6. **Lottery Ticket Hypothesis**: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, 2019)
7. **Magnitude Pruning**: "Learning both Weights and Connections for Efficient Neural Networks" (Han et al., 2015)

#### 蒸餾
8. **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
9. **DistilBERT**: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (Sanh et al., 2019)

### 工具與庫

- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **AutoGPTQ**: https://github.com/PanQiWei/AutoGPTQ
- **AutoAWQ**: https://github.com/casper-hansen/AutoAWQ
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **ONNX Runtime**: https://onnxruntime.ai/
- **TensorRT**: https://developer.nvidia.com/tensorrt
- **vLLM**: https://github.com/vllm-project/vllm

### 實用資源

- **Hugging Face Quantization Guide**: https://huggingface.co/docs/transformers/main/en/quantization
- **PyTorch Quantization Tutorial**: https://pytorch.org/docs/stable/quantization.html
- **NVIDIA TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM

### 社群與討論

- **r/LocalLLaMA** (Reddit): 本地 LLM 部署討論
- **Hugging Face Forums**: 量化技術討論
- **llama.cpp Discussions**: CPU 推論優化

---

## 總結

模型壓縮與優化是將 LLM 部署到實際應用的關鍵技術：

### 核心要點

1. **量化是首選方法**：
   - 8-bit：幾乎無損，顯存減半
   - 4-bit：輕微損失，顯存減至 1/4
   - 2-bit：顯著損失，僅極限場景使用

2. **選擇合適工具**：
   - 微調：bitsandbytes (QLoRA)
   - GPU 推理：GPTQ, AWQ
   - CPU 推理：llama.cpp (GGUF)
   - 生產部署：TensorRT, vLLM

3. **權衡三角**：
   - 模型大小 vs 精度 vs 速度
   - 沒有銀彈，需根據應用場景選擇

4. **實踐建議**：
   - 從 FP16 開始作為基準
   - 嘗試 8-bit 評估精度損失
   - 需要更激進壓縮才考慮 4-bit
   - 始終在目標任務上評估性能

5. **未來趨勢**：
   - 更低精度量化（INT2, 1-bit）
   - 混合精度策略
   - 硬體-軟體協同優化
   - 量化感知預訓練

**起點建議**：
- 研究/實驗：QLoRA (4-bit) + LoRA 微調（詳見 [LoRA 專題](./advanced/low_rank_factorization.md)）
- 生產推理（GPU）：GPTQ/AWQ (4-bit) + vLLM（參考 [部署案例](./advanced/deployment_cases.md)）
- 本地部署：llama.cpp Q4_K_M/Q5_K_M（參考 [硬體指南](./guides/hardware_guide.md)）

**更多資源**：
- 📋 [速查表](./QUICK_REFERENCE.md)：常用命令和配置
- 🎯 [練習題](./EXERCISES.md)：動手實踐
- 🔧 [故障排除](./guides/troubleshooting.md)：解決常見問題
- ⚡ [最佳實踐](./guides/best_practices.md)：生產環境建議
