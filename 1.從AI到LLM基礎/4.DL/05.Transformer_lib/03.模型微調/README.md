# 模型微調 (Fine-tuning)

學習如何將預訓練模型微調到你的特定任務上。

## 📚 學習目標

- 理解微調的原理和流程
- 掌握使用 Trainer API 進行微調
- 學習使用 PEFT 進行高效微調
- 了解如何準備數據和評估模型

## 目錄

1. [微調基礎](#微調基礎)
2. [使用 Trainer API](#使用-trainer-api)
3. [PEFT 高效微調](#peft-高效微調)
4. [數據準備](#數據準備)
5. [模型評估](#模型評估)

---

## 微調基礎

### 什麼是微調？

微調是在預訓練模型的基礎上，使用特定任務的數據進行進一步訓練，使模型適應新任務。

### 微調 vs 從頭訓練

| 特徵 | 微調 | 從頭訓練 |
|------|------|---------|
| 訓練時間 | 短（小時） | 長（天/週） |
| 數據需求 | 少（數千） | 多（百萬） |
| 計算資源 | 低 | 高 |
| 效果 | 通常更好 | 視數據量而定 |

### 微調流程

```
1. 選擇預訓練模型
        ↓
2. 準備任務數據
        ↓
3. 數據預處理
        ↓
4. 設定訓練參數
        ↓
5. 開始訓練
        ↓
6. 評估和優化
        ↓
7. 保存模型
```

---

## 使用 Trainer API

### 基本微調示例

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 1. 載入數據
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. 數據預處理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. 載入模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 4. 設定訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 5. 創建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 6. 開始訓練
trainer.train()

# 7. 評估模型
trainer.evaluate()

# 8. 保存模型
trainer.save_model("./final_model")
```

### TrainingArguments 重要參數

```python
training_args = TrainingArguments(
    # 基本參數
    output_dir="./results",              # 輸出目錄
    num_train_epochs=3,                  # 訓練輪數
    per_device_train_batch_size=16,      # 每設備批次大小

    # 學習率
    learning_rate=2e-5,                  # 學習率
    lr_scheduler_type="linear",          # 學習率調度器
    warmup_steps=500,                    # 預熱步數

    # 優化器
    optim="adamw_torch",                 # 優化器類型
    weight_decay=0.01,                   # 權重衰減

    # 評估
    evaluation_strategy="steps",         # 評估策略
    eval_steps=500,                      # 每 500 步評估一次

    # 保存
    save_strategy="steps",               # 保存策略
    save_steps=500,                      # 每 500 步保存一次
    save_total_limit=3,                  # 最多保留 3 個檢查點
    load_best_model_at_end=True,         # 訓練結束載入最佳模型

    # 混合精度
    fp16=True,                           # 啟用 FP16

    # 日誌
    logging_dir="./logs",                # TensorBoard 日誌目錄
    logging_steps=100,                   # 每 100 步記錄一次

    # 梯度
    gradient_accumulation_steps=4,       # 梯度累積
    max_grad_norm=1.0,                   # 梯度裁剪

    # 其他
    seed=42,                             # 隨機種子
    dataloader_num_workers=4,            # 數據載入器工作進程數
)
```

---

## PEFT 高效微調

### LoRA 微調（2025 推薦）

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 載入模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. 準備模型
model = prepare_model_for_kbit_training(model)

# 4. LoRA 配置
lora_config = LoraConfig(
    r=64,                               # LoRA 秩
    lora_alpha=128,                     # LoRA alpha
    target_modules=[                    # 目標模組
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,                  # Dropout
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 應用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. 訓練（使用 Trainer 或 SFTTrainer）
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
)

trainer.train()
```

### QLoRA 微調（極致記憶體優化）

```python
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import torch

# 4-bit QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all-linear",  # 針對所有線性層
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",  # 8-bit Adam 優化器
    num_train_epochs=3,
)
```

---

## 數據準備

### 1. 載入數據

```python
from datasets import load_dataset

# 從 Hub 載入
dataset = load_dataset("imdb")

# 從本地文件載入
dataset = load_dataset("csv", data_files="train.csv")
dataset = load_dataset("json", data_files="train.json")

# 從 Pandas DataFrame
import pandas as pd
from datasets import Dataset

df = pd.read_csv("data.csv")
dataset = Dataset.from_pandas(df)
```

### 2. 數據預處理

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def preprocess_function(examples):
    # 分詞
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

# 批次處理
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,  # 使用 4 個進程
    remove_columns=dataset.column_names,  # 移除原始列
)
```

### 3. 數據增強

```python
import random

def augment_text(examples):
    texts = examples["text"]
    augmented_texts = []

    for text in texts:
        # 隨機插入、刪除、替換等
        augmented_texts.append(augment(text))

    examples["text"] = augmented_texts
    return examples

augmented_dataset = dataset.map(augment_text, batched=True)
```

---

## 模型評估

### 1. 定義評估指標

```python
import evaluate
import numpy as np

# 載入指標
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }
```

### 2. 評估模型

```python
# 在測試集上評估
results = trainer.evaluate(eval_dataset=test_dataset)
print(results)

# 預測
predictions = trainer.predict(test_dataset)
print(predictions.metrics)
```

### 3. 混淆矩陣

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 獲取預測
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

# 繪製混淆矩陣
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

---

## 最佳實踐

### 1. 選擇合適的學習率

```python
# 使用學習率查找器
from transformers import TrainerCallback

class LRFinderCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        lr = trainer.optimizer.param_groups[0]["lr"]
        loss = state.log_history[-1]["loss"]
        print(f"LR: {lr}, Loss: {loss}")
```

### 2. 使用早停

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

### 3. 保存檢查點

```python
training_args = TrainingArguments(
    output_dir="./checkpoints",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # 只保留最近 3 個檢查點
    load_best_model_at_end=True,
)
```

---

## 常見問題

**Q: 微調需要多少數據？**
A: 通常幾千到幾萬條數據即可，視任務複雜度而定。

**Q: 學習率如何設置？**
A: 一般使用 1e-5 到 5e-5，比預訓練時小 10-100 倍。

**Q: 如何避免過擬合？**
A: 使用 dropout、早停、數據增強、減少訓練輪數。

**Q: GPU 記憶體不足怎麼辦？**
A: 減小 batch size、使用梯度累積、使用 FP16、使用 PEFT。

---

## 延伸閱讀

- [Trainer API 文檔](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PEFT 文檔](https://huggingface.co/docs/peft/)
- [訓練技巧](https://huggingface.co/docs/transformers/performance)

## 下一步

- 查看 [示例代碼](./examples/) 學習完整微調流程
- 前往 [04. 進階主題](../04.進階主題/) 學習更多高級技術
