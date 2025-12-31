# 微調策略進階指南 (Advanced Fine-tuning Strategies)

## 概述

模型微調是將預訓練模型適應特定任務的關鍵技術。2025 年，隨著 LoRA、QLoRA 等高效方法的成熟，微調變得更加經濟實惠和可行。

## 微調方法比較

```
┌─────────────────────────────────────────────────────────────┐
│                    微調方法光譜                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  資源需求低 ←────────────────────────────────→ 資源需求高   │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Prompt  │  │  LoRA   │  │ QLoRA   │  │  Full   │       │
│  │ Tuning  │  │         │  │         │  │Fine-tune│       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│       │            │            │            │             │
│       ▼            ▼            ▼            ▼             │
│   只訓練        低秩適應      量化+LoRA    全參數更新       │
│   提示向量      ~1% 參數      ~0.5% 參數   100% 參數       │
│                                                             │
│  記憶體需求                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 7B 模型:  ~1GB     ~8GB      ~4GB        ~28GB       │  │
│  │ 70B 模型: ~4GB     ~40GB     ~20GB       ~280GB      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. LoRA (Low-Rank Adaptation)

### LoRA 原理

```python
# LoRA 核心概念
# 原始權重: W (d × k)
# LoRA 分解: W' = W + BA
#   B: (d × r) - 低秩矩陣
#   A: (r × k) - 低秩矩陣
#   r << min(d, k) - 秩遠小於原始維度

# 例如: d=4096, k=4096, r=8
# 原始參數: 16,777,216
# LoRA 參數: 4096*8 + 8*4096 = 65,536 (0.4%)
```

### 使用 PEFT 實作 LoRA

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import torch

class LoRAFineTuner:
    """LoRA 微調器"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list[str] = None
    ):
        self.model_name = model_name
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.tokenizer = None
        self.model = None

    def load_model(self):
        """載入模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 應用 LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(
        self,
        dataset_name: str,
        text_column: str = "text",
        max_length: int = 512
    ):
        """準備資料集"""
        dataset = load_dataset(dataset_name)

        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        return tokenized

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./lora_output",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4
    ):
        """訓練模型"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            report_to="tensorboard"
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        trainer.train()
        return trainer

    def save_model(self, output_dir: str):
        """儲存 LoRA 權重"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def merge_and_save(self, output_dir: str):
        """合併 LoRA 權重到基礎模型"""
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# 使用範例
tuner = LoRAFineTuner(
    model_name="meta-llama/Llama-2-7b-hf",
    lora_r=16,
    lora_alpha=32
)

tuner.load_model()
dataset = tuner.prepare_dataset("tatsu-lab/alpaca")
tuner.train(dataset["train"])
tuner.save_model("./my_lora_model")
```

### LoRA 超參數調優

```python
# LoRA 超參數指南

lora_configs = {
    # 通用任務（對話、指令遵循）
    "general": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    },

    # 特定領域（法律、醫療）
    "domain_specific": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },

    # 程式碼生成
    "code_generation": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },

    # 快速適應（少量資料）
    "few_shot": {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
    }
}

# 選擇技巧
"""
1. r (秩):
   - 較低 (4-8): 快速訓練，較少參數，適合簡單任務
   - 較高 (16-64): 更強表達能力，適合複雜任務
   - 經驗法則: 從 8 開始，根據效果調整

2. lora_alpha:
   - 通常設為 2*r
   - 控制 LoRA 更新的縮放
   - 較高值 = 更強的適應能力

3. target_modules:
   - 最小: ["q_proj", "v_proj"] - 最快，效果一般
   - 標準: ["q_proj", "k_proj", "v_proj", "o_proj"] - 平衡
   - 完整: 包含 MLP 層 - 最強，最慢

4. lora_dropout:
   - 0.05-0.1 適用於大多數情況
   - 資料量少時可增加到 0.2
"""
```

## 2. QLoRA (Quantized LoRA)

### QLoRA 實作

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

class QLoRAFineTuner:
    """QLoRA 微調器"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_4bit: bool = True
    ):
        self.model_name = model_name

        # 4-bit 量化配置
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",  # NormalFloat4
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True  # 雙重量化
        )

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """載入量化模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 準備模型進行 k-bit 訓練
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA 配置
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train_with_sft(
        self,
        dataset,
        output_dir: str = "./qlora_output",
        num_epochs: int = 3,
        batch_size: int = 4,
        max_seq_length: int = 512
    ):
        """使用 SFTTrainer 訓練"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",  # 8-bit 優化器
            gradient_checkpointing=True
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            dataset_text_field="text"
        )

        trainer.train()
        return trainer

# 使用範例
qlora_tuner = QLoRAFineTuner("meta-llama/Llama-2-7b-hf")
qlora_tuner.load_model()

# 假設有準備好的資料集
# qlora_tuner.train_with_sft(dataset)
```

### QLoRA vs LoRA 比較

```python
# 資源比較（以 7B 模型為例）

comparison = """
┌─────────────────────┬──────────────┬──────────────┐
│ 指標                │ LoRA         │ QLoRA        │
├─────────────────────┼──────────────┼──────────────┤
│ GPU 記憶體          │ ~16 GB       │ ~6 GB        │
│ 訓練速度            │ 基準         │ ~1.3x 慢     │
│ 模型品質            │ 基準         │ ~98% 基準    │
│ 可訓練參數          │ ~0.1%        │ ~0.1%        │
│ 推論速度            │ 基準         │ 需要反量化   │
└─────────────────────┴──────────────┴──────────────┘

選擇建議：
- GPU < 16GB: 使用 QLoRA
- GPU >= 24GB: 使用 LoRA（更快）
- 追求最佳品質: 使用 LoRA
- 資源受限: 使用 QLoRA
"""
```

## 3. 資料準備與格式

### 指令微調資料格式

```python
from datasets import Dataset
import json

class InstructionDataset:
    """指令資料集準備"""

    # 常見格式

    # Alpaca 格式
    ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    # ChatML 格式
    CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""

    # Llama 2 Chat 格式
    LLAMA2_TEMPLATE = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant} </s>"""

    @staticmethod
    def format_alpaca(
        instruction: str,
        input_text: str = "",
        output: str = ""
    ) -> str:
        """格式化 Alpaca 樣本"""
        return InstructionDataset.ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text if input_text else "",
            output=output
        )

    @staticmethod
    def prepare_dataset(
        data: list[dict],
        format_type: str = "alpaca"
    ) -> Dataset:
        """準備資料集"""
        formatted_data = []

        for item in data:
            if format_type == "alpaca":
                text = InstructionDataset.format_alpaca(
                    instruction=item.get("instruction", ""),
                    input_text=item.get("input", ""),
                    output=item.get("output", "")
                )
            elif format_type == "chatml":
                text = InstructionDataset.CHATML_TEMPLATE.format(
                    system=item.get("system", "You are a helpful assistant."),
                    user=item.get("user", ""),
                    assistant=item.get("assistant", "")
                )
            else:
                text = item.get("text", "")

            formatted_data.append({"text": text})

        return Dataset.from_list(formatted_data)

    @staticmethod
    def load_and_prepare(
        file_path: str,
        format_type: str = "alpaca"
    ) -> Dataset:
        """從檔案載入並準備資料集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return InstructionDataset.prepare_dataset(data, format_type)

# 使用範例
# 準備訓練資料
training_data = [
    {
        "instruction": "將以下英文翻譯成中文",
        "input": "Hello, how are you?",
        "output": "你好，你好嗎？"
    },
    {
        "instruction": "總結以下文章的重點",
        "input": "人工智能（AI）正在改變各行各業...",
        "output": "AI 正在各領域引發變革，主要影響..."
    }
]

dataset = InstructionDataset.prepare_dataset(training_data, "alpaca")
```

### 資料品質檢查

```python
from typing import List, Dict
import re

class DataQualityChecker:
    """資料品質檢查器"""

    @staticmethod
    def check_length(
        data: List[Dict],
        min_length: int = 10,
        max_length: int = 2048
    ) -> Dict:
        """檢查長度"""
        issues = []
        for i, item in enumerate(data):
            text = item.get("text", "") or item.get("output", "")
            if len(text) < min_length:
                issues.append(f"樣本 {i}: 太短 ({len(text)} 字元)")
            elif len(text) > max_length:
                issues.append(f"樣本 {i}: 太長 ({len(text)} 字元)")

        return {
            "total": len(data),
            "issues": len(issues),
            "details": issues[:10]  # 只顯示前 10 個
        }

    @staticmethod
    def check_duplicates(data: List[Dict], key: str = "instruction") -> Dict:
        """檢查重複"""
        seen = {}
        duplicates = []

        for i, item in enumerate(data):
            value = item.get(key, "")
            if value in seen:
                duplicates.append((i, seen[value]))
            else:
                seen[value] = i

        return {
            "total": len(data),
            "unique": len(seen),
            "duplicates": len(duplicates),
            "examples": duplicates[:5]
        }

    @staticmethod
    def check_formatting(data: List[Dict]) -> Dict:
        """檢查格式問題"""
        issues = []

        for i, item in enumerate(data):
            # 檢查必要欄位
            if "instruction" not in item and "text" not in item:
                issues.append(f"樣本 {i}: 缺少 instruction 或 text 欄位")

            # 檢查空白
            for key, value in item.items():
                if isinstance(value, str):
                    if value.strip() != value:
                        issues.append(f"樣本 {i}: {key} 有多餘空白")

            # 檢查特殊字元
            text = str(item)
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
                issues.append(f"樣本 {i}: 包含控制字元")

        return {
            "total": len(data),
            "issues": len(issues),
            "details": issues[:10]
        }

    @staticmethod
    def full_check(data: List[Dict]) -> Dict:
        """完整檢查"""
        return {
            "length": DataQualityChecker.check_length(data),
            "duplicates": DataQualityChecker.check_duplicates(data),
            "formatting": DataQualityChecker.check_formatting(data)
        }

# 使用範例
checker = DataQualityChecker()
report = checker.full_check(training_data)
print(json.dumps(report, indent=2, ensure_ascii=False))
```

## 4. 訓練策略與技巧

### 學習率調度

```python
from transformers import get_scheduler
import torch

# 常用調度器

schedulers_config = {
    # 餘弦退火（推薦）
    "cosine": {
        "type": "cosine",
        "num_warmup_steps": 100,
        "num_training_steps": 1000
    },

    # 線性衰減
    "linear": {
        "type": "linear",
        "num_warmup_steps": 100,
        "num_training_steps": 1000
    },

    # 常數學習率（帶預熱）
    "constant_with_warmup": {
        "type": "constant_with_warmup",
        "num_warmup_steps": 100
    }
}

def create_scheduler(
    optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 100,
    num_training_steps: int = 1000
):
    """建立學習率調度器"""
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

# 學習率建議
"""
模型大小    | 建議學習率  | Batch Size
-----------+------------+-----------
1-3B       | 2e-4       | 4-8
7B         | 1e-4       | 4
13B        | 5e-5       | 2-4
70B        | 2e-5       | 1-2

注意事項：
1. 使用梯度累積來模擬更大的 batch size
2. 預熱步數通常設為總步數的 3-10%
3. 過擬合時降低學習率或增加 dropout
"""
```

### 梯度累積與混合精度

```python
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler

class EfficientTrainer:
    """高效訓練器"""

    def __init__(
        self,
        model,
        optimizer,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16"
    ):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision
        )

        self.model, self.optimizer = self.accelerator.prepare(
            model, optimizer
        )

        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train_step(self, batch, step: int):
        """單步訓練"""
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            loss = outputs.loss

            self.accelerator.backward(loss)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss.item()

    def save_checkpoint(self, output_dir: str):
        """儲存檢查點"""
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            output_dir,
            save_function=self.accelerator.save
        )
```

### 早停與最佳模型選擇

```python
from typing import Optional
import numpy as np

class EarlyStopping:
    """早停機制"""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

# 使用範例
early_stopping = EarlyStopping(patience=3, mode="min")

for epoch in range(100):
    val_loss = evaluate_model()

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## 5. 成本與效益分析

### 微調成本估算

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FineTuningCost:
    """微調成本估算"""
    gpu_hours: float
    gpu_cost_per_hour: float
    total_cost: float
    cost_per_sample: float

class CostEstimator:
    """成本估算器"""

    # GPU 每小時成本（雲端）
    GPU_COSTS = {
        "A100_40GB": 3.0,   # $/hour
        "A100_80GB": 4.0,
        "H100": 5.0,
        "RTX_4090": 0.5,   # 本地電費估算
        "T4": 0.5
    }

    # 訓練速度估算（samples/second）
    TRAINING_SPEEDS = {
        "7B_LoRA_A100": 10,
        "7B_QLoRA_A100": 7,
        "7B_LoRA_4090": 5,
        "13B_LoRA_A100": 5,
        "70B_QLoRA_A100": 1
    }

    @classmethod
    def estimate(
        cls,
        model_size: str,
        method: str,
        gpu_type: str,
        num_samples: int,
        num_epochs: int
    ) -> FineTuningCost:
        """估算成本"""
        config_key = f"{model_size}_{method}_{gpu_type}"
        speed = cls.TRAINING_SPEEDS.get(config_key, 5)
        gpu_cost = cls.GPU_COSTS.get(gpu_type, 3.0)

        total_samples = num_samples * num_epochs
        training_seconds = total_samples / speed
        training_hours = training_seconds / 3600

        # 加上 20% 的額外時間（驗證、checkpointing 等）
        total_hours = training_hours * 1.2
        total_cost = total_hours * gpu_cost
        cost_per_sample = total_cost / num_samples

        return FineTuningCost(
            gpu_hours=total_hours,
            gpu_cost_per_hour=gpu_cost,
            total_cost=total_cost,
            cost_per_sample=cost_per_sample
        )

# 使用範例
cost = CostEstimator.estimate(
    model_size="7B",
    method="LoRA",
    gpu_type="A100_40GB",
    num_samples=10000,
    num_epochs=3
)

print(f"""
微調成本估算:
- GPU 時數: {cost.gpu_hours:.2f} 小時
- GPU 成本: ${cost.gpu_cost_per_hour}/小時
- 總成本: ${cost.total_cost:.2f}
- 每樣本成本: ${cost.cost_per_sample:.4f}
""")
```

### 微調 vs API 成本比較

```python
def compare_costs(
    num_samples: int,
    avg_tokens_per_sample: int,
    fine_tuning_cost: float,
    inference_volume_monthly: int
):
    """比較微調與 API 成本"""

    # API 成本（以 GPT-4o-mini 為例）
    api_input_cost = 0.15 / 1_000_000  # $0.15 per 1M tokens
    api_output_cost = 0.60 / 1_000_000

    # 假設輸入輸出 token 比例 1:1
    cost_per_request = (
        avg_tokens_per_sample * api_input_cost +
        avg_tokens_per_sample * api_output_cost
    )

    monthly_api_cost = inference_volume_monthly * cost_per_request

    # 計算回本月數
    if monthly_api_cost > 0:
        payback_months = fine_tuning_cost / (monthly_api_cost * 0.3)  # 假設微調後成本降 70%
    else:
        payback_months = float('inf')

    return {
        "monthly_api_cost": monthly_api_cost,
        "fine_tuning_cost": fine_tuning_cost,
        "payback_months": payback_months,
        "recommendation": "微調" if payback_months < 6 else "API"
    }

# 使用範例
comparison = compare_costs(
    num_samples=10000,
    avg_tokens_per_sample=500,
    fine_tuning_cost=100,
    inference_volume_monthly=100000
)
print(comparison)
```

## 6. 評估與驗證

### 微調效果評估

```python
from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class FineTuneEvaluator:
    """微調評估器"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """評估生成品質"""
        results = {
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }

        for pred, ref in zip(predictions, references):
            # BLEU
            bleu = sentence_bleu([ref.split()], pred.split())
            results["bleu"].append(bleu)

            # ROUGE
            rouge = self.rouge_scorer.score(ref, pred)
            results["rouge1"].append(rouge["rouge1"].fmeasure)
            results["rouge2"].append(rouge["rouge2"].fmeasure)
            results["rougeL"].append(rouge["rougeL"].fmeasure)

        return {
            metric: np.mean(scores)
            for metric, scores in results.items()
        }

    def evaluate_task_performance(
        self,
        model,
        tokenizer,
        test_data: List[Dict],
        task_type: str = "classification"
    ) -> Dict:
        """評估任務表現"""
        predictions = []
        labels = []

        for item in test_data:
            # 生成預測
            inputs = tokenizer(
                item["input"],
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=50)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

            predictions.append(pred)
            labels.append(item["label"])

        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score
            # 簡單匹配
            pred_labels = [p.strip().lower() for p in predictions]
            true_labels = [l.strip().lower() for l in labels]

            return {
                "accuracy": accuracy_score(true_labels, pred_labels),
                "f1": f1_score(true_labels, pred_labels, average="macro")
            }

        return {"predictions": predictions}

# 使用範例
evaluator = FineTuneEvaluator()

predictions = ["這是一個很好的產品", "服務態度不錯"]
references = ["這是一個優秀的產品", "服務態度很好"]

metrics = evaluator.evaluate_generation(predictions, references)
print(metrics)
```

## 最佳實踐總結

```markdown
## 微調檢查清單

### 資料準備
- [ ] 清理和標準化資料格式
- [ ] 檢查資料品質（長度、重複、格式）
- [ ] 準備驗證集（10-20%）
- [ ] 確認資料多樣性

### 模型選擇
- [ ] 評估基礎模型能力
- [ ] 選擇適當的微調方法
- [ ] 計算資源需求

### 訓練設定
- [ ] 設定合適的學習率
- [ ] 配置梯度累積
- [ ] 啟用混合精度訓練
- [ ] 設定早停機制

### 評估驗證
- [ ] 定義評估指標
- [ ] 準備測試案例
- [ ] 比較微調前後效果

### 部署準備
- [ ] 合併 LoRA 權重（可選）
- [ ] 測試推論效能
- [ ] 準備模型版本管理
```

## 延伸閱讀

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
