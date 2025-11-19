# 訓練與優化 - 10篇關鍵論文

> 2024-2025年模型訓練與推理優化的重大進展：FlashAttention 3、vLLM、QLoRA等關鍵技術

---

## 📋 論文列表

| # | 論文/項目 | 機構 | 發布時間 | 代碼 | 影響力 |
|---|-----------|------|----------|------|--------|
| 1 | FlashAttention 3 | Stanford | 2024.07 | [GitHub](https://github.com/Dao-AILab/flash-attention) | ⭐⭐⭐⭐⭐ |
| 2 | vLLM | UC Berkeley | 2024 | [GitHub](https://github.com/vllm-project/vllm) | ⭐⭐⭐⭐⭐ |
| 3 | QLoRA | UW | 2024 | [GitHub](https://github.com/artidoro/qlora) | ⭐⭐⭐⭐⭐ |
| 4 | AWQ | MIT | 2024 | [GitHub](https://github.com/mit-han-lab/llm-awq) | ⭐⭐⭐⭐ |
| 5 | SGLang | UC Berkeley | 2024 | [GitHub](https://github.com/sgl-project/sglang) | ⭐⭐⭐⭐ |
| 6 | Medusa Decoding | Together AI | 2024 | [GitHub](https://github.com/FasterDecoding/Medusa) | ⭐⭐⭐⭐ |
| 7 | DeepSpeed ZeRO++ | Microsoft | 2024 | [GitHub](https://github.com/microsoft/DeepSpeed) | ⭐⭐⭐⭐ |
| 8 | DoRA | NVIDIA | 2024 | [GitHub](https://github.com/NVlabs/DoRA) | ⭐⭐⭐ |
| 9 | GPTQ | IST Austria | 2024 | [GitHub](https://github.com/IST-DASLab/gptq) | ⭐⭐⭐⭐ |
| 10 | TensorRT-LLM | NVIDIA | 2024 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) | ⭐⭐⭐⭐⭐ |

---

## 核心技術與代碼實現

### 1. FlashAttention 3 - 注意力計算革命

```python
import torch
from transformers import AutoModelForCausalLM

# 使用FlashAttention 2/3
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 或 "sdpa" (PyTorch 2.0+)
    device_map="auto"
)

# 自定義FlashAttention配置
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# FlashAttention自動應用於長序列
inputs = tokenizer("Long text " * 1000, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)

# 性能對比
import time

def benchmark_attention(model, input_ids, implementation):
    model.config.attn_implementation = implementation
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    return time.time() - start

input_ids = torch.randint(0, 32000, (1, 8192)).cuda()

time_eager = benchmark_attention(model, input_ids, "eager")
time_flash = benchmark_attention(model, input_ids, "flash_attention_2")

print(f"Eager: {time_eager:.2f}s")
print(f"FlashAttention: {time_flash:.2f}s")
print(f"Speedup: {time_eager/time_flash:.2f}x")
```

### 2. vLLM - 高吞吐推理引擎

```python
from vllm import LLM, SamplingParams

# 初始化vLLM
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # 使用4個GPU
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.95,
    enable_chunked_prefill=True,  # 分塊預填充
    max_num_batched_tokens=8192
)

# 批量推理
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "Describe neural networks"
] * 100  # 300個請求

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# 高吞吐批處理
import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

print(f"Generated {len(outputs)} responses in {elapsed:.2f}s")
print(f"Throughput: {len(outputs)/elapsed:.2f} requests/s")

# 與標準Transformers對比
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

start = time.time()
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
elapsed_baseline = time.time() - start

print(f"\nBaseline: {elapsed_baseline:.2f}s")
print(f"vLLM speedup: {elapsed_baseline/elapsed:.2f}x")
```

### 3. QLoRA - 量化低秩微調

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 加載量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# 準備模型for QLoRA
model = prepare_model_for_kbit_training(model)

# LoRA配置
lora_config = LoraConfig(
    r=16,  # LoRA秩
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 應用LoRA
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# 訓練
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()
```

### 4. AWQ - 激活感知量化

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 量化模型
model_path = "meta-llama/Meta-Llama-3.1-8B"
quant_path = "llama-3.1-8b-awq"

# 加載模型並量化
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 準備校準數據
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 執行量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# 加載並使用量化模型
model_awq = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=True,
    trust_remote_code=False,
    safetensors=True
)

# 推理
inputs = tokenizer("Explain AI", return_tensors="pt").to("cuda")
outputs = model_awq.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 5. SGLang - 結構化生成優化

```python
import sglang as sgl

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

# 運行
state = multi_turn_question.run(
    question_1="What is machine learning?",
    question_2="Give me an example",
    backend="openai",
    model="gpt-4o-mini"
)

print("Q1:", state["answer_1"])
print("Q2:", state["answer_2"])

# 批量並行
states = multi_turn_question.run_batch([
    {"question_1": "What is AI?", "question_2": "Explain more"},
    {"question_1": "What is ML?", "question_2": "Give examples"}
],
    backend="openai",
    model="gpt-4o-mini"
)
```

### 6. Medusa Decoding - 並行解碼加速

```python
from transformers import AutoTokenizer
from medusa import MedusaModel

# 加載Medusa模型
model = MedusaModel.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")

# Medusa並行解碼
inputs = tokenizer("Explain the theory of relativity", return_tensors="pt").to("cuda")

# 比較標準解碼vs Medusa
import time

# 標準解碼
start = time.time()
outputs_standard = model.generate(**inputs, max_new_tokens=256, do_sample=False)
time_standard = time.time() - start

# Medusa解碼
start = time.time()
outputs_medusa = model.medusa_generate(**inputs, max_new_tokens=256, temperature=0)
time_medusa = time.time() - start

print(f"Standard: {time_standard:.2f}s")
print(f"Medusa: {time_medusa:.2f}s")
print(f"Speedup: {time_standard/time_medusa:.2f}x")
```

### 7. DeepSpeed ZeRO++ - 分布式訓練

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# DeepSpeed配置
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}

# 初始化模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# DeepSpeed初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# 訓練循環
for batch in train_dataloader:
    outputs = model_engine(**batch)
    loss = outputs.loss

    model_engine.backward(loss)
    model_engine.step()
```

### 8-10. DoRA, GPTQ, TensorRT-LLM

**DoRA**: LoRA改進版本，分解權重更新
**GPTQ**: 後訓練量化技術
**TensorRT-LLM**: NVIDIA推理優化框架

```python
# TensorRT-LLM示例
from tensorrt_llm import LLM

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")

outputs = llm.generate(
    ["Explain quantum physics"],
    sampling_params={"max_tokens": 256, "temperature": 0.7}
)

print(outputs[0].outputs[0].text)
```

---

## 📊 性能對比

| 技術 | 推理加速 | 記憶體節省 | 訓練加速 | 易用性 |
|------|---------|-----------|---------|--------|
| FlashAttention 3 | 2-4x | 3-5x | 2-3x | ⭐⭐⭐⭐ |
| vLLM | 5-15x | 2-3x | - | ⭐⭐⭐⭐⭐ |
| QLoRA | - | 4x | 1.5x | ⭐⭐⭐⭐ |
| AWQ | 2-3x | 4x | - | ⭐⭐⭐ |
| Medusa | 2-3x | 1x | - | ⭐⭐⭐ |
| TensorRT-LLM | 3-6x | 2x | - | ⭐⭐⭐ |

---

## 🔬 選擇指南

**推理優化**: vLLM (生產) / TensorRT-LLM (NVIDIA GPU)
**微調**: QLoRA (記憶體受限) / LoRA (性能優先)
**量化**: AWQ (平衡) / GPTQ (極致壓縮)
**分布式訓練**: DeepSpeed ZeRO++ (大規模) / FSDP (PyTorch原生)

---

**最後更新**: 2025-01-19
