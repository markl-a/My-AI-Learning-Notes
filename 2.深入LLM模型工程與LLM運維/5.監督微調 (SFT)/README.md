# 監督微調 (Supervised Fine-Tuning, SFT)

## 目錄

### 基礎內容
1. [SFT 概念：由 Next-token 預測到特定任務微調](#51-sft-概念)
2. [全參數微調 vs. PEFT 方法](#52-全參數微調-vs-peft-方法)
3. [微調實務](#53-微調實務)
4. [不同模板格式對微調結果的影響](#54-不同模板格式對微調結果的影響)
5. [實作範例](#55-實作範例)

### 實用工具與資源
- 📦 [**數據準備工具集**](./data_preparation_tools/) - AI 輔助數據生成、質量檢查、格式轉換
- 🎓 [**從入門到熟練學習路徑**](./LEARNING_PATH.md) - 完整的學習路線圖、故障排除、最佳實踐
- 🚀 [**端到端實戰項目**](./hands_on_project/) - 電商客服機器人完整項目
- 🧠 [**進階主題**](./advanced_topics/) - 多任務學習、持續學習、災難性遺忘

---

## 5.1 SFT 概念

### 什麼是監督微調？

監督微調 (SFT) 是將預訓練的語言模型適配到特定任務或領域的過程。透過在標註數據上進行訓練，模型學習特定的輸入輸出映射關係。

### 從預訓練到微調

1. **預訓練階段**：
   - 目標：Next Token Prediction
   - 數據：大規模無標註文本
   - 學習：語言的統計規律和通用知識

2. **微調階段**：
   - 目標：特定任務的輸入輸出映射
   - 數據：高質量的任務相關標註數據
   - 學習：任務特定的模式和行為

### SFT 的訓練目標

**指令微調 (Instruction Tuning)**：

```
輸入：指令 + 上下文（可選）
輸出：期望的回答

損失函數：L = -Σ log P(y_t | x, y_<t)
```

只對輸出部分計算損失，輸入部分不參與梯度更新。

### SFT 的重要性

1. **提升任務性能**：針對特定任務優化
2. **行為對齊**：使模型遵循指令
3. **格式標準化**：統一輸出格式
4. **安全性提升**：減少有害輸出

---

## 5.2 全參數微調 vs. PEFT 方法

### 全參數微調 (Full Fine-Tuning)

**特點**：
- 更新模型的所有參數
- 需要大量 GPU 內存
- 性能通常最好
- 需要保存完整的模型副本

**內存需求**：
- 模型參數：`params × 4` bytes (FP32)
- 優化器狀態：`params × 8` bytes (AdamW)
- 梯度：`params × 4` bytes
- 總計：`params × 16` bytes

**適用場景**：
- 有充足計算資源
- 需要最佳性能
- 數據量充足

### 參數高效微調 (Parameter-Efficient Fine-Tuning, PEFT)

#### 1. LoRA (Low-Rank Adaptation)

**核心思想**：
- 凍結預訓練權重
- 添加低秩分解矩陣進行訓練

**數學表示**：
```
W' = W + ΔW
ΔW = B × A
```
其中：
- W：原始權重矩陣 (d×k)
- A：低秩矩陣 (d×r)
- B：低秩矩陣 (r×k)
- r：秩 (r << min(d,k))

**優點**：
- 大幅減少可訓練參數
- 推理時可合併權重，無額外開銷
- 易於切換不同任務的適配器

**參數**：
- `r`：秩（通常 4-64）
- `alpha`：縮放因子
- `target_modules`：應用 LoRA 的模塊

#### 2. QLoRA (Quantized LoRA)

**核心思想**：
- 將基座模型量化到 4-bit
- 在量化模型上應用 LoRA

**技術要點**：
- **NF4 量化**：專為神經網絡設計的 4-bit 格式
- **雙量化**：量化量化常數
- **分頁優化器**：處理內存峰值

**內存優勢**：
- 7B 模型：從 28GB 降至 ~5GB
- 13B 模型：可在單張 24GB GPU 上訓練

#### 3. 其他 PEFT 方法

**Adapter Tuning**：
- 在每層插入小型適配器模塊
- 只訓練適配器參數

**Prefix Tuning**：
- 在輸入前添加可學習的前綴向量
- 凍結其他所有參數

**Prompt Tuning**：
- 只優化軟提示 (soft prompt) 的嵌入
- 極度參數高效

### 方法對比

| 方法 | 可訓練參數 | 內存需求 | 性能 | 推理開銷 |
|------|-----------|---------|------|---------|
| 全參數微調 | 100% | 極高 | 最佳 | 無 |
| LoRA | ~0.1-1% | 低 | 接近全參數 | 無（合併後） |
| QLoRA | ~0.1-1% | 極低 | 接近全參數 | 無（合併後） |
| Adapter | ~1-5% | 中 | 良好 | 有 |
| Prefix Tuning | ~0.01-0.1% | 極低 | 中等 | 有 |

---

## 5.3 微調實務

### 數據準備

**數據格式**：

```json
{
  "instruction": "解釋什麼是機器學習",
  "input": "",
  "output": "機器學習是人工智慧的一個分支..."
}
```

**數據質量要點**：
1. **多樣性**：涵蓋不同類型的任務
2. **準確性**：輸出必須正確
3. **一致性**：格式和風格統一
4. **平衡性**：避免類別不平衡

### 超參數選擇

**關鍵超參數**：

1. **學習率**：
   - 全參數：1e-5 ~ 5e-5
   - LoRA：1e-4 ~ 5e-4
   - 通常需要 warmup

2. **Batch Size**：
   - 根據 GPU 內存調整
   - 使用梯度累積增加有效 batch size

3. **Epochs**：
   - 通常 1-3 epochs
   - 過度訓練會導致過擬合

4. **LoRA 特定參數**：
   - `r`：8-64（越大越接近全參數）
   - `lora_alpha`：通常設為 r 的 2 倍
   - `lora_dropout`：0.05-0.1

### 訓練技巧

1. **梯度累積**：模擬更大的 batch size
2. **混合精度訓練**：FP16/BF16 減少內存
3. **梯度檢查點**：以時間換空間
4. **DeepSpeed/FSDP**：分布式訓練

### 評估指標

1. **困惑度 (Perplexity)**：衡量語言建模能力
2. **任務特定指標**：準確率、F1、BLEU、ROUGE 等
3. **人工評估**：質量、有用性、安全性

---

## 5.4 不同模板格式對微調結果的影響

### 常見模板格式

#### 1. Alpaca 格式

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
```

**特點**：
- 簡單直接
- 適合單輪對話
- 廣泛使用

#### 2. ChatML 格式

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{assistant_response}
<|im_end|>
```

**特點**：
- 結構化
- 支持多輪對話
- 角色明確

#### 3. Vicuna 格式

```
A chat between a curious user and an artificial intelligence assistant.

USER: {user_message}
ASSISTANT: {assistant_response}
```

**特點**：
- 對話式
- 適合聊天場景

#### 4. Custom 格式

根據任務需求自定義格式。

### 模板選擇建議

1. **與預訓練一致**：使用模型預訓練時的格式
2. **明確角色區分**：幫助模型理解輸入輸出邊界
3. **特殊 token**：使用模型詞彙表中的特殊 token
4. **一致性**：訓練和推理時保持一致

---

## 5.5 實作範例

### 5.5.1 使用 Hugging Face Transformers 進行全參數微調

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 載入模型和 tokenizer
model_name = "gpt2"  # 或其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 添加 pad token（如果沒有）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 準備數據
def preprocess_function(examples):
    # 假設數據格式為 {"text": "..."}
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# 載入數據集（這裡使用示例數據集）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # 混合精度訓練
    report_to="none"
)

# 數據整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果語言建模
)

# 創建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 開始訓練
trainer.train()

# 保存模型
trainer.save_model("./fine_tuned_model")
```

### 5.5.2 使用 LoRA 進行參數高效微調

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

# 載入模型
model_name = "meta-llama/Llama-2-7b-hf"  # 需要訪問權限
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 8-bit 量化
    device_map="auto",
    torch_dtype=torch.float16
)

# 準備模型進行訓練
model = prepare_model_for_kbit_training(model)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,  # 秩
    lora_alpha=32,  # 縮放因子
    target_modules=["q_proj", "v_proj"],  # 應用 LoRA 的模塊
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 應用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 準備數據
def format_instruction(example):
    """格式化指令數據"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": prompt}

# 載入並處理數據集
dataset = load_dataset("json", data_files="your_data.json", split="train")
dataset = dataset.map(format_instruction)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # LoRA 使用較大學習率
    fp16=True,
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_8bit"  # 內存高效的優化器
)

# 訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# 保存 LoRA 權重
model.save_pretrained("./lora_weights")
```

### 5.5.3 使用 QLoRA 在單 GPU 上微調大模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # 雙量化
    bnb_4bit_quant_type="nf4",  # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16  # 計算時使用 BF16
)

# 載入量化模型
model_name = "meta-llama/Llama-2-13b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 準備模型
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./qlora_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # QLoRA 允許更小的 batch size
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,  # 使用 BF16
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,  # 啟用梯度檢查點
    max_grad_norm=0.3
)

# 使用 SFTTrainer（專為監督微調設計）
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args
)

# 訓練
trainer.train()

# 保存
trainer.save_model("./qlora_final")
```

### 5.5.4 推理時使用 LoRA 適配器

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 載入基座模型
base_model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 載入 LoRA 適配器
lora_weights_path = "./lora_weights"
model = PeftModel.from_pretrained(base_model, lora_weights_path)

# 合併權重（可選，提升推理速度）
model = model.merge_and_unload()

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 生成文本
prompt = "### Instruction:\n解釋什麼是量子計算\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 5.5.6 數據集準備腳本

```python
import json

def create_instruction_dataset(examples):
    """創建指令微調數據集"""
    dataset = []

    for example in examples:
        formatted_example = {
            "instruction": example["instruction"],
            "input": example.get("input", ""),
            "output": example["output"]
        }
        dataset.append(formatted_example)

    return dataset

# 示例數據
examples = [
    {
        "instruction": "將以下句子翻譯成英文",
        "input": "機器學習是人工智慧的一個重要分支。",
        "output": "Machine learning is an important branch of artificial intelligence."
    },
    {
        "instruction": "解釋以下概念",
        "input": "深度學習",
        "output": "深度學習是機器學習的一個子領域，使用多層神經網絡來學習數據的表示..."
    }
]

# 創建數據集
dataset = create_instruction_dataset(examples)

# 保存為 JSON
with open("instruction_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"已創建 {len(dataset)} 條訓練樣本")
```

### 5.5.7 監控訓練過程

```python
from transformers import TrainerCallback
import wandb

class CustomCallback(TrainerCallback):
    """自定義回調函數監控訓練"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step: {state.global_step}")
            print(f"Loss: {logs.get('loss', 'N/A')}")
            print(f"Learning Rate: {logs.get('learning_rate', 'N/A')}")
            print("-" * 50)

# 使用 Weights & Biases 進行監控
wandb.init(project="llm-fine-tuning", name="experiment-1")

training_args = TrainingArguments(
    ...
    report_to="wandb",  # 啟用 wandb 日誌
    logging_steps=10
)

trainer = Trainer(
    ...
    callbacks=[CustomCallback()]
)
```

---

## 快速開始指南

### 新手入門

如果你是 SFT 的新手，建議按以下步驟開始：

1. **學習基礎概念** (1-2 天)
   - 閱讀上面的基礎內容章節
   - 理解 SFT 的基本原理

2. **動手實踐** (3-5 天)
   - 跟隨 [端到端實戰項目](./hands_on_project/) 的快速開始指南
   - 使用示例數據訓練第一個模型

3. **系統學習** (2-4 週)
   - 跟隨 [完整學習路徑](./LEARNING_PATH.md)
   - 完成每個階段的練習和項目

### 工具使用

我們提供了完整的工具集來加速你的 SFT 工作流程：

```bash
# 1. 使用 AI 生成訓練數據
cd data_preparation_tools
python ai_assisted_data_generator.py

# 2. 檢查數據質量
python data_quality_checker.py your_data.json

# 3. 訓練模型
cd ../hands_on_project
python scripts/3_train_model.py \
    --model_name gpt2 \
    --train_data data/train.json \
    --use_qlora
```

### 學習路徑建議

- **完全新手** → [學習路徑階段一](./LEARNING_PATH.md#階段一基礎入門)
- **有基礎知識** → [學習路徑階段二](./LEARNING_PATH.md#階段二實踐應用)
- **需要進階技術** → [進階主題](./advanced_topics/)
- **準備生產部署** → [學習路徑階段四](./LEARNING_PATH.md#階段四生產部署)

---

## 參考資源

### 論文

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [InstructGPT: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)
- [FLAN: Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

### 工具和庫

- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [Hugging Face TRL Library](https://github.com/huggingface/trl)
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Efficient Training Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)

### 社區和討論

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
- [LLM Discord Communities](https://discord.gg/hugging-face)