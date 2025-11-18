# 進階主題

探索 Transformers 的進階功能和最新技術。

## 📚 學習目標

- 掌握模型優化技術
- 學習分布式訓練
- 了解最新的推理優化方法
- 探索模型對齊和RLHF技術

## 目錄

1. [模型量化](#模型量化)
2. [分布式訓練](#分布式訓練)
3. [推理優化](#推理優化)
4. [模型對齊 (RLHF/DPO)](#模型對齊)
5. [多模態模型](#多模態模型)

---

## 模型量化

### 靜態量化 (GPTQ)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Llama-3-8B"

# GPTQ 量化配置
gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",
    group_size=128,
    desc_act=False,
)

# 載入並量化
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config,
    device_map="auto",
)

# 保存量化模型
model.save_pretrained("./llama3-8b-gptq")
```

### 動態量化 (AWQ)

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3-8B"

# 載入模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# AWQ 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized("./llama3-8b-awq")
```

---

## 分布式訓練

### 使用 Accelerate

```python
from accelerate import Accelerator
from transformers import AdamW

accelerator = Accelerator()

# 準備模型、優化器、數據載入器
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 訓練循環
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### DeepSpeed 整合

```python
# deepspeed_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

```python
# 使用 DeepSpeed
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="./deepspeed_config.json",
    ...
)

trainer = Trainer(
    model=model,
    args=training_args,
    ...
)

trainer.train()
```

### FSDP (PyTorch 原生)

```python
training_args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"
    },
    ...
)
```

---

## 推理優化

### Flash Attention 2

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# 速度提升 2-8 倍，記憶體減少 50%
```

### vLLM 部署

```python
from vllm import LLM, SamplingParams

# 初始化 vLLM
llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=2,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

# 批次推理
prompts = ["Explain AI:", "What is Python?"]
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### TensorRT-LLM

```bash
# 1. 轉換模型
python convert_checkpoint.py \
    --model_dir ./llama-3-8b \
    --output_dir ./trt_ckpt \
    --dtype float16

# 2. 構建引擎
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16

# 3. 運行推理（速度提升 3-6 倍）
```

---

## 模型對齊

### RLHF (Reinforcement Learning from Human Feedback)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# 1. 載入模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. PPO 配置
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
)

# 3. 創建 PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# 4. 訓練循環
for epoch in range(num_epochs):
    for batch in dataloader:
        query_tensors = batch["input_ids"]

        # 生成回應
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )

        # 計算獎勵
        rewards = [compute_reward(q, r) for q, r in zip(queries, responses)]

        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

### DPO (Direct Preference Optimization)

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 載入模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. DPO 配置
dpo_config = DPOConfig(
    beta=0.1,  # DPO 的 KL 懲罰係數
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)

# 3. 準備偏好數據
# 格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
train_dataset = load_preference_dataset()

# 4. 創建 DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 5. 訓練
dpo_trainer.train()
```

### ORPO (Odds Ratio Preference Optimization)

```python
from trl import ORPOTrainer, ORPOConfig

# ORPO 配置
orpo_config = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    per_device_train_batch_size=2,
)

# ORPO Trainer
orpo_trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

orpo_trainer.train()
```

---

## 多模態模型

### LLaVA (大語言視覺助手)

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# 載入模型
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 準備輸入
image = Image.open("image.jpg")
prompt = "USER: <image>\nWhat's in this image? ASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 生成回應
outputs = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### CLIP (對比語言-圖像預訓練)

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 圖像和文本
image = Image.open("photo.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# 處理輸入
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 獲取相似度
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("Label probabilities:", probs)
```

### Whisper (多語言語音識別)

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# 載入 Whisper V3
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# 轉錄音頻
def transcribe(audio_path):
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to("cuda").to(torch.float16) for k, v in inputs.items()}

    # 生成轉錄
    generated_ids = model.generate(**inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription

# 使用
result = transcribe("audio.mp3")
print(result)
```

---

## 最佳實踐

### 1. 選擇合適的量化方法

- **GPTQ**: 最廣泛支援，速度快
- **AWQ**: 準確度最高，稍慢
- **GGUF**: 適合 CPU 推理

### 2. 分布式訓練選擇

- **小模型 (< 1B)**: 數據並行
- **中型模型 (1B-10B)**: DeepSpeed ZeRO-2
- **大型模型 (> 10B)**: FSDP 或 DeepSpeed ZeRO-3

### 3. 推理優化策略

- **延遲優先**: Flash Attention 2 + TensorRT-LLM
- **吞吐量優先**: vLLM + Continuous Batching
- **記憶體受限**: 量化 (GPTQ/AWQ) + PagedAttention

---

## 延伸閱讀

- [PEFT 文檔](https://huggingface.co/docs/peft/)
- [TRL 文檔](https://huggingface.co/docs/trl/)
- [Accelerate 文檔](https://huggingface.co/docs/accelerate/)
- [Optimum 文檔](https://huggingface.co/docs/optimum/)
- [vLLM 文檔](https://docs.vllm.ai/)

## 下一步

- 查看 [實戰項目](../05.實戰項目/) 應用所學知識
- 探索最新的研究論文和技術
