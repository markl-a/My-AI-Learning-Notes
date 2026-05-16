# 模型壓縮與優化速查表

## 📋 快速決策表

### 1. 量化方法選擇

| 場景 | 推薦方法 | 精度 | 原因 |
|------|---------|------|------|
| **微調 7B-70B 模型** | QLoRA | 4-bit | 顯存效率最高，支持微調 |
| **GPU 推理部署** | GPTQ / AWQ | 4-bit | 速度快，精度好，支持 vLLM |
| **CPU 推理** | llama.cpp (GGUF) | Q4_K_M | CPU 優化，跨平台 |
| **雲端大規模部署** | TensorRT-LLM | INT8 / FP16 | 極致性能，NVIDIA 優化 |
| **移動端設備** | llama.cpp (GGUF) | Q4_0 / INT4 | 內存小，功耗低 |
| **邊緣設備** | TFLite / ONNX | INT8 | 硬體加速，功耗敏感 |

### 2. 精度格式對比

| 格式 | 大小 | 速度 | 精度 | 用途 | 硬體要求 |
|------|------|------|------|------|---------|
| FP32 | 4x | 1.0x | ⭐⭐⭐⭐⭐ | 訓練基準 | 任意 |
| FP16 | 2x | 1.8x | ⭐⭐⭐⭐⭐ | 訓練/推理 | Volta+ GPU |
| BF16 | 2x | 1.8x | ⭐⭐⭐⭐ | 訓練 | Ampere+ GPU, TPU |
| INT8 | 1x | 2.5x | ⭐⭐⭐⭐ | 推理 | 任意 (軟體) |
| INT4 | 0.5x | 3.5x | ⭐⭐⭐ | 推理 | 專用/軟體 |
| INT2 | 0.25x | ~5x | ⭐⭐ | 實驗性 | 專用硬體 |

### 3. 工具選擇矩陣

| 工具 | 量化精度 | 訓練支援 | 推理後端 | 最佳場景 |
|------|---------|---------|----------|----------|
| **bitsandbytes** | 8-bit, 4-bit | ✅ QLoRA | PyTorch | 微調、研究 |
| **AutoGPTQ** | 2-4 bit | ❌ | ExLlama, vLLM | GPU 推理 |
| **AutoAWQ** | 4-bit | ❌ | vLLM, TGI | GPU 推理 |
| **llama.cpp** | Q2_K ~ Q8_0 | ❌ | CPU, Metal, CUDA | 本地/CPU |
| **ONNX Runtime** | INT8, FP16 | ❌ | ONNX | 跨平台 |
| **TensorRT-LLM** | INT8, FP16, INT4 | ❌ | TensorRT | NVIDIA GPU |
| **Optimum** | INT8, INT4 | ❌ | ONNX, TRT | Hugging Face |

---

## 🔧 常用命令速查

### bitsandbytes (QLoRA)

```python
# 4-bit 量化
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 8-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

### AutoGPTQ

```bash
# 安裝
pip install auto-gptq

# 量化
python -m auto_gptq.cli quantize \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir ./llama-2-7b-gptq \
    --bits 4 \
    --group_size 128 \
    --desc_act

# 載入
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./llama-2-7b-gptq",
    device_map="auto"
)
```

### llama.cpp (GGUF)

```bash
# 安裝
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 轉換模型
python convert.py /path/to/llama-2-7b \
    --outfile llama-2-7b-f16.gguf \
    --outtype f16

# 量化
./quantize llama-2-7b-f16.gguf llama-2-7b-Q4_K_M.gguf Q4_K_M

# 推理
./main -m llama-2-7b-Q4_K_M.gguf \
       -p "Once upon a time" \
       -n 128 \
       -t 8
```

### vLLM

```bash
# 安裝
pip install vllm

# 啟動服務
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-Chat-GPTQ \
    --quantization gptq \
    --tensor-parallel-size 1 \
    --port 8000

# Python API
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", quantization="awq")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

### TensorRT-LLM

```bash
# 安裝
pip install tensorrt_llm

# 轉換模型
python convert_checkpoint.py \
    --model_dir ./llama-2-7b-hf \
    --output_dir ./llama-2-7b-trt \
    --dtype float16

# 構建引擎
trtllm-build \
    --checkpoint_dir ./llama-2-7b-trt \
    --output_dir ./llama-2-7b-engine \
    --gemm_plugin float16

# 推理
python run.py \
    --engine_dir ./llama-2-7b-engine \
    --input_text "Hello, world!"
```

---

## 📊 性能指標參考

### LLaMA-7B 量化性能

| 方法 | 大小 | 速度 | PPL ↓ | 顯存 (推理) |
|------|------|------|-------|------------|
| FP32 | 28 GB | 1.0x | 5.68 | 30 GB |
| FP16 | 14 GB | 1.8x | 5.68 | 16 GB |
| INT8 (GPTQ) | 7 GB | 2.5x | 5.75 | 10 GB |
| INT4 (GPTQ) | 3.5 GB | 3.5x | 6.02 | 8 GB |
| Q4_K_M (GGUF) | 4 GB | 2.8x (CPU) | 5.98 | 6 GB |

### LLaMA-2-13B 量化性能

| 方法 | 大小 | 速度 (GPU) | PPL ↓ | 顯存 |
|------|------|-----------|-------|------|
| FP16 | 26 GB | 1.0x | 5.09 | 28 GB |
| INT8 (GPTQ) | 13 GB | 2.2x | 5.15 | 15 GB |
| INT4 (GPTQ) | 6.5 GB | 3.2x | 5.38 | 9 GB |
| INT4 (AWQ) | 6.5 GB | 3.5x | 5.22 | 9 GB |

---

## ⚙️ 超參數速查

### LoRA 超參數

| 參數 | 推薦值 | 範圍 | 說明 |
|------|--------|------|------|
| `r` (秩) | 8 | 4-64 | 低秩矩陣秩，越大越接近全微調 |
| `lora_alpha` | 16 | r ~ 2r | 縮放因子，通常設為 r 或 2r |
| `lora_dropout` | 0.05 | 0-0.1 | Dropout 率，防止過擬合 |
| `target_modules` | `["q_proj", "v_proj"]` | - | 最小配置 |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | - | 推薦配置 |

### 量化超參數

| 參數 | GPTQ | AWQ | 說明 |
|------|------|-----|------|
| `bits` | 4 | 4 | 量化位元數 |
| `group_size` | 128 | 128 | 分組大小，越小精度越高但越慢 |
| `desc_act` | False | - | 降序激活值順序 |
| `sym` | True | True | 對稱量化 |
| `damp_percent` | 0.01 | - | Hessian 阻尼 |

### 訓練超參數

| 參數 | 全微調 | LoRA | 說明 |
|------|--------|------|------|
| `learning_rate` | 2e-5 | 2e-4 | LoRA 可用更高學習率 |
| `batch_size` | 1-2 | 4-8 | LoRA 可用更大批次 |
| `gradient_accumulation` | 16 | 4 | 模擬更大批次 |
| `warmup_ratio` | 0.03 | 0.03 | 預熱比例 |
| `weight_decay` | 0.01 | 0.01 | 權重衰減 |

---

## 💾 顯存估算

### 訓練顯存 (Full Fine-tuning)

```
總顯存 ≈ 模型參數 × (
    2 (FP16參數) +
    2 (梯度) +
    8 (優化器狀態, AdamW) +
    激活值
) + 開銷

例如 LLaMA-7B (FP16):
= 7B × (2 + 2 + 8) + 激活值
≈ 84 GB + 30 GB (激活值)
≈ 114 GB
```

### 訓練顯存 (LoRA)

```
總顯存 ≈ 模型參數 × 2 (FP16, 凍結) +
    LoRA 參數 × 12 (可訓練) +
    激活值

例如 LLaMA-7B + LoRA (r=8):
= 7B × 2 + 4M × 12 + 激活值
≈ 14 GB + 0.048 GB + 6 GB
≈ 20 GB
```

### 訓練顯存 (QLoRA)

```
總顯存 ≈ 模型參數 × 0.5 (4-bit) +
    LoRA 參數 × 12 (FP16) +
    激活值

例如 LLaMA-7B + QLoRA (r=8):
= 7B × 0.5 + 4M × 12 + 激活值
≈ 3.5 GB + 0.048 GB + 2.5 GB
≈ 6 GB
```

### 推理顯存

```
推理顯存 ≈ 模型參數大小 + KV Cache

FP16: 參數量 × 2 bytes
INT8: 參數量 × 1 byte
INT4: 參數量 × 0.5 bytes

KV Cache ≈ batch_size × seq_len × n_layers × hidden_dim × 4 bytes

例如 LLaMA-7B (INT4), batch=1, seq=2048:
= 3.5 GB + 1 × 2048 × 32 × 4096 × 4 bytes
≈ 3.5 GB + 1 GB
≈ 4.5 GB
```

---

## 🎯 常見場景推薦配置

### 場景 1：本地開發 (RTX 4090 24GB)

```python
# LLaMA-13B + QLoRA
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### 場景 2：生產推理 (多 GPU)

```python
# vLLM + GPTQ/AWQ
llm = LLM(
    model="TheBloke/Llama-2-70B-Chat-GPTQ",
    quantization="gptq",
    tensor_parallel_size=4,
    max_num_batched_tokens=8192,
)
```

### 場景 3：CPU 推理 (無 GPU)

```bash
# llama.cpp + GGUF Q4_K_M
./main -m llama-2-13b-Q4_K_M.gguf \
       -t 8 \           # 8 線程
       -c 2048 \        # 上下文長度
       -b 512 \         # 批次大小
       --mlock \        # 鎖定內存
       -p "Prompt"
```

### 場景 4：邊緣設備 (Raspberry Pi)

```python
# llama.cpp + TinyLlama/Phi-2 Q4_0
from llama_cpp import Llama

llm = Llama(
    model_path="tinyllama-1.1b-Q4_0.gguf",
    n_ctx=512,        # 較小上下文
    n_threads=4,      # 4 核心
    n_batch=128,
    use_mlock=True,
)
```

---

## 🐛 問題診斷速查

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| **OOM (顯存不足)** | 批次太大 | 減小 `batch_size` 或 `max_length` |
| | 模型太大 | 使用更激進量化 (4-bit) 或更小模型 |
| | KV cache 太大 | 減小 `n_ctx` 或 `max_num_seqs` |
| **精度下降嚴重** | 量化過於激進 | 使用更高精度 (INT8 而非 INT4) |
| | 校準資料不足 | 增加校準樣本數量 |
| | 敏感層被量化 | 使用混合精度，保留敏感層 |
| **推論速度慢** | 未使用量化 | 應用 INT8/INT4 量化 |
| | CPU 推理 | 遷移到 GPU 或使用 llama.cpp 優化 |
| | 批次太小 | 增加批次大小（吞吐量） |
| **訓練不收斂** | 學習率過高 | 降低學習率 (e.g., 2e-4 → 1e-4) |
| | LoRA 秩太低 | 增加 `r` (e.g., 8 → 16) |
| | 梯度爆炸 | 啟用梯度裁剪 `max_grad_norm=1.0` |

---

## 📚 快速鏈接

### 官方文檔
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT (LoRA): https://huggingface.co/docs/peft
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- vLLM: https://docs.vllm.ai/
- llama.cpp: https://github.com/ggerganov/llama.cpp

### 模型權重
- Hugging Face Hub: https://huggingface.co/models
- TheBloke (量化模型): https://huggingface.co/TheBloke

### 基準測試
- Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- MMLU: https://github.com/hendrycks/test

### 社群
- r/LocalLLaMA: https://reddit.com/r/LocalLLaMA
- Hugging Face Forums: https://discuss.huggingface.co/

---

## 🔖 常用程式碼片段

### 1. 快速檢測 GPU 能力

```python
import torch

print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 數量: {torch.cuda.device_count()}")
print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
print(f"總顯存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"BF16 支援: {torch.cuda.is_bf16_supported()}")

# 計算可用顯存
torch.cuda.empty_cache()
print(f"可用顯存: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
```

### 2. 顯存監控

```python
import torch

def print_gpu_memory():
    """打印 GPU 顯存使用情況"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.2f} GB / {reserved:.2f} GB")

# 使用
model = load_model()
print_gpu_memory()
```

### 3. 模型大小計算

```python
def get_model_size(model):
    """計算模型參數量和大小"""
    param_size = 0
    param_count = 0

    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2

    print(f"參數量: {param_count / 1e9:.2f}B")
    print(f"模型大小: {size_mb:.2f} MB")

    return param_count, size_mb
```

### 4. 量化前後對比

```python
def compare_models(model_fp16, model_quant, test_prompts):
    """對比量化前後的模型"""
    results = []

    for prompt in test_prompts:
        # FP16
        output_fp16 = model_fp16.generate(prompt)

        # 量化
        output_quant = model_quant.generate(prompt)

        # 比較
        similarity = calculate_similarity(output_fp16, output_quant)
        results.append({
            "prompt": prompt,
            "fp16": output_fp16,
            "quant": output_quant,
            "similarity": similarity
        })

    avg_similarity = sum(r["similarity"] for r in results) / len(results)
    print(f"平均相似度: {avg_similarity:.2%}")

    return results
```

---

這份速查表涵蓋了模型壓縮與優化的核心要點，可以快速查閱常用命令、配置和解決方案。
