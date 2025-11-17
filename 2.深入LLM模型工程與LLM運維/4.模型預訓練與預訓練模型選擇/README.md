# 模型預訓練與預訓練模型選擇

## 目錄
1. [預訓練的基礎概念](#41-預訓練的基礎概念)
2. [開源預訓練模型](#42-開源預訓練模型)
3. [預訓練流程](#43-預訓練流程)
4. [Scaling Laws 與高效預訓練技術](#44-scaling-laws-與高效預訓練技術)
5. [實作範例](#45-實作範例)

---

## 4.1 預訓練的基礎概念

### 4.1.1 什麼是預訓練？

**預訓練 (Pre-training)** 是在大規模無標註文字資料上訓練語言模型的過程，目的是讓模型學習語言的統計規律、語法結構和世界知識。

**預訓練的重要性**：
- 學習通用語言表示
- 捕捉語法和語義知識
- 建立世界知識基礎
- 為下游任務提供良好的初始化

**預訓練 vs 微調**：

| 階段 | 資料 | 目標 | 規模 | 成本 |
|------|------|------|------|------|
| 預訓練 | 海量無標註文字 | 語言建模 | 數百億到數兆 tokens | 極高 |
| 微調 | 少量標註資料 | 特定任務 | 數千到數百萬 tokens | 中等 |

### 4.1.2 語言建模目標

#### Causal Language Modeling (因果語言建模)

用於 GPT 等自回歸模型：

```
給定前文 x₁, x₂, ..., xₜ，預測下一個 token xₜ₊₁

損失函數：
L = -∑ log P(xₜ | x₁, ..., xₜ₋₁)
```

**特點**：
- 單向注意力（只看前文）
- 適合文字生成任務
- 訓練與推理一致

**範例**：
```
輸入: "機器學習是"
目標: "人工智慧"
```

#### Masked Language Modeling (遮蔽語言建模)

用於 BERT 等雙向模型：

```
隨機遮蔽部分 token，預測被遮蔽的 token

輸入: "機器[MASK]是人工智慧的[MASK]分支"
目標: 預測 [MASK] 位置的詞（"學習"、"一個"）
```

**特點**：
- 雙向注意力（可看前後文）
- 適合理解任務（分類、問答）
- 訓練與推理有差異（MLM 預訓練，但推理時沒有 [MASK]）

### 4.1.3 預訓練資料

**資料規模趨勢**：

| 模型 | 發布年份 | 參數量 | 訓練 Tokens |
|------|---------|--------|-------------|
| GPT-2 | 2019 | 1.5B | 40B |
| GPT-3 | 2020 | 175B | 300B |
| GPT-4 | 2023 | ~1.8T (推測) | ~13T (推測) |
| LLaMA | 2023 | 7B-65B | 1T-1.4T |
| LLaMA 2 | 2023 | 7B-70B | 2T |
| DeepSeek-V3 | 2024 | 671B | 14.8T |

**資料來源**：
1. **網路爬取資料**
   - Common Crawl
   - Reddit
   - StackOverflow
   - Wikipedia

2. **書籍與學術資料**
   - Books3
   - arXiv
   - PubMed

3. **程式碼**
   - GitHub
   - GitLab
   - StackExchange

4. **對話資料**
   - 論壇討論
   - 社交媒體

**資料品質考量**：
- 去重（deduplication）
- 過濾低品質內容
- 移除個人隱私資訊
- 平衡不同領域比例

---

## 4.2 開源預訓練模型

### 4.2.1 LLaMA 系列

**LLaMA (Large Language Model Meta AI)**

**LLaMA 1 (2023.02)**：
- **規模**：7B, 13B, 33B, 65B
- **訓練資料**：1T-1.4T tokens
- **特點**：
  - 開源權重（研究用途）
  - 高效架構設計
  - SwiGLU 啟用函數
  - RoPE 位置編碼

**LLaMA 2 (2023.07)**：
- **規模**：7B, 13B, 70B
- **訓練資料**：2T tokens（較 LLaMA 1 增加 40%）
- **改進**：
  - 上下文長度從 2K 增加到 4K
  - Grouped-Query Attention (GQA)
  - 更好的安全性對齊
  - 商業友好授權

**LLaMA 3 (2024)**：
- **規模**：8B, 70B, 405B
- **訓練資料**：15T+ tokens
- **改進**：
  - 上下文長度 8K（可擴展到 128K）
  - 更大的詞彙表（128K）
  - 多語言能力顯著提升

**使用範例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入 LLaMA 2 模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 生成文字
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 4.2.2 Mistral 系列

**Mistral 7B (2023.09)**：
- **參數量**：7.3B
- **特點**：
  - Sliding Window Attention（滑動窗口注意力）
  - GQA (Grouped-Query Attention)
  - 性能媲美 13B-34B 模型
  - 上下文長度：8K（可擴展到 32K）

**Mixtral 8x7B (2023.12)**：
- **架構**：Mixture of Experts (MoE)
- **總參數**：47B
- **啟用參數**：12.9B（每次只啟用 2 個專家）
- **特點**：
  - 8 個專家模型
  - 稀疏激活提升效率
  - 多語言能力強

**Mixtral 8x22B (2024.04)**：
- **總參數**：141B
- **啟用參數**：39B
- **上下文長度**：64K

**使用範例**：

```python
# 載入 Mistral 模型
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Mistral 支援更長的上下文
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

### 4.2.3 BLOOM

**BLOOM (BigScience Large Open-science Open-access Multilingual Language Model)**

- **發布**：2022.07
- **規模**：176B 參數
- **訓練資料**：366B tokens，46 種語言
- **特點**：
  - 真正的多語言模型
  - 社群協作開發
  - 完全開源
  - ALiBi 位置編碼

**語言支援**：
- 英語、中文、法語、西班牙語等 46 種語言
- 包含程式碼（13 種程式語言）

### 4.2.4 Qwen (通義千問)

**Qwen 1.0 (2023)**：
- **規模**：1.8B, 7B, 14B, 72B
- **訓練資料**：~3T tokens
- **特點**：
  - 中英雙語優化
  - 長上下文支援（8K-32K）
  - 程式碼能力強

**Qwen 2 (2024)**：
- **規模**：0.5B, 1.5B, 7B, 72B
- **改進**：
  - 更大的詞彙表
  - 更好的多語言能力
  - 支援 128K 上下文

### 4.2.5 DeepSeek 系列

**DeepSeek-V2 (2024.05)**：
- **規模**：236B 總參數，21B 啟用參數
- **架構**：MoE（混合專家）
- **特點**：
  - 極低的訓練成本
  - Multi-head Latent Attention (MLA)
  - 128K 上下文窗口

**DeepSeek-V3 (2024.12)**：
- **規模**：671B 總參數，37B 啟用參數
- **訓練資料**：14.8T tokens
- **訓練成本**：僅 $5.576M
- **特點**：
  - 無輔助損失的 MoE 訓練
  - Multi-Token Prediction (MTP)
  - FP8 混合精度訓練
  - 開源權重

### 4.2.6 模型選擇指南

**根據規模選擇**：

| 規模 | 適用場景 | 推薦模型 | 硬體需求 |
|------|---------|---------|---------|
| 小型 (< 10B) | 邊緣設備、快速推理 | Mistral 7B, Qwen 7B | 單卡 GPU (16GB+) |
| 中型 (10B-70B) | 一般應用、微調 | LLaMA 2 70B, Qwen 72B | 多卡 GPU (80GB+) |
| 大型 (> 70B) | 複雜推理、研究 | LLaMA 3 405B, DeepSeek-V3 | 多機多卡 |

**根據任務選擇**：

| 任務類型 | 推薦模型 | 原因 |
|---------|---------|------|
| 中文對話 | Qwen, DeepSeek | 中文優化 |
| 程式碼生成 | DeepSeek, LLaMA 3 | 程式碼訓練資料豐富 |
| 多語言 | BLOOM, LLaMA 3 | 多語言支援 |
| 長上下文 | Qwen 2, Mistral | 支援 32K-128K 上下文 |
| 成本敏感 | DeepSeek-V3 (MoE) | 稀疏激活降低成本 |

---

## 4.3 預訓練流程

### 4.3.1 資料處理流程

#### 步驟 1：資料收集

```python
from datasets import load_dataset
import os

class PretrainingDataCollector:
    """預訓練資料收集器"""

    def __init__(self, output_dir="./pretraining_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def collect_web_data(self):
        """收集網路資料（使用 Common Crawl）"""
        # Common Crawl 資料非常大，這裡僅示範
        dataset = load_dataset(
            "c4",
            "en",
            split="train",
            streaming=True  # 串流模式避免記憶體不足
        )
        return dataset

    def collect_wikipedia(self, language="en"):
        """收集 Wikipedia 資料"""
        dataset = load_dataset(
            "wikipedia",
            f"20231101.{language}",
            split="train"
        )
        return dataset

    def collect_code_data(self):
        """收集程式碼資料"""
        dataset = load_dataset(
            "codeparrot/github-code",
            split="train",
            streaming=True
        )
        return dataset

    def collect_books(self):
        """收集書籍資料"""
        # 注意：某些資料集可能有版權問題
        dataset = load_dataset(
            "bookcorpus",
            split="train"
        )
        return dataset

# 使用範例
collector = PretrainingDataCollector()

# 收集不同來源的資料
wiki_data = collector.collect_wikipedia(language="zh")
print(f"Wikipedia 資料量: {len(wiki_data)}")
```

#### 步驟 2：資料清理

```python
import re
from typing import List, Dict

class DataCleaner:
    """預訓練資料清理器"""

    @staticmethod
    def remove_duplicates(texts: List[str]) -> List[str]:
        """移除重複內容"""
        seen = set()
        unique_texts = []

        for text in texts:
            # 使用雜湊避免記憶體問題
            text_hash = hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def filter_low_quality(text: str) -> bool:
        """過濾低品質文字"""
        # 基本品質檢查
        if len(text) < 100:  # 太短
            return False

        if len(text) > 100000:  # 太長
            return False

        # 檢查標點符號比例
        punct_ratio = sum(c in '.,!?;:' for c in text) / len(text)
        if punct_ratio > 0.3:  # 標點符號過多
            return False

        # 檢查大寫字母比例
        upper_ratio = sum(c.isupper() for c in text) / len(text)
        if upper_ratio > 0.5:  # 大寫字母過多
            return False

        return True

    @staticmethod
    def normalize_text(text: str) -> str:
        """規範化文字"""
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊控制字元
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # 規範化引號
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

# 使用範例
cleaner = DataCleaner()

raw_texts = [
    "這是一段正常的文字...",
    "太短",  # 會被過濾
    "!!!!!!!!!!!!!!!!!",  # 標點符號過多，會被過濾
]

# 過濾低品質文字
high_quality = [t for t in raw_texts if cleaner.filter_low_quality(t)]

# 規範化
normalized = [cleaner.normalize_text(t) for t in high_quality]

print(f"原始: {len(raw_texts)}, 高品質: {len(high_quality)}")
```

#### 步驟 3：Tokenization

```python
from transformers import AutoTokenizer
from datasets import Dataset

class PretrainingTokenizer:
    """預訓練資料 Tokenization"""

    def __init__(self, tokenizer_name="gpt2", max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 設定 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_function(self, examples):
        """Tokenize 文字"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

    def prepare_dataset(self, texts: List[str]):
        """準備訓練資料集"""
        # 創建 Dataset
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

# 使用範例
tokenizer_tool = PretrainingTokenizer(max_length=512)

texts = [
    "這是第一段文字，用於預訓練。",
    "這是第二段文字，包含更多內容。"
]

tokenized_data = tokenizer_tool.prepare_dataset(texts)
print(f"Tokenized dataset: {tokenized_data}")
```

### 4.3.2 大規模分散式訓練

#### DeepSpeed 配置

```python
# deepspeed_config.json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 100000,
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-4,
      "warmup_num_steps": 2000
    }
  }
}
```

#### 訓練腳本

```python
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import deepspeed

class PretrainingPipeline:
    """預訓練流程"""

    def __init__(self, config_path="model_config.json"):
        # 模型配置
        self.model_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )

        # 建立模型
        self.model = GPT2LMHeadModel(self.model_config)

        # 顯示模型大小
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型參數數量: {total_params / 1e6:.2f}M")

    def setup_training_args(self, output_dir="./pretrained_model"):
        """設定訓練參數"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,

            # 訓練設定
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,

            # 優化器
            learning_rate=6e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,

            # 學習率調度
            lr_scheduler_type="cosine",
            warmup_steps=2000,

            # 混合精度
            fp16=True,

            # 保存與記錄
            save_steps=1000,
            logging_steps=100,
            save_total_limit=3,

            # DeepSpeed
            deepspeed="deepspeed_config.json",
        )

        return training_args

    def train(self, train_dataset, output_dir="./pretrained_model"):
        """執行預訓練"""
        training_args = self.setup_training_args(output_dir)

        # 資料整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # 開始訓練
        print("開始預訓練...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        print(f"模型已保存至 {output_dir}")

        return trainer

# 使用範例（需要大量計算資源）
# pipeline = PretrainingPipeline()
# trainer = pipeline.train(tokenized_dataset)
```

### 4.3.3 訓練監控

```python
from transformers import TrainerCallback
import wandb

class PretrainingMonitor(TrainerCallback):
    """預訓練監控回調"""

    def __init__(self):
        # 初始化 Weights & Biases
        wandb.init(project="llm-pretraining", name="my-model")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """記錄訓練指標"""
        if logs:
            # 記錄損失
            if "loss" in logs:
                wandb.log({"train/loss": logs["loss"]}, step=state.global_step)

            # 記錄學習率
            if "learning_rate" in logs:
                wandb.log({"train/lr": logs["learning_rate"]}, step=state.global_step)

            # 計算困惑度
            if "loss" in logs:
                perplexity = torch.exp(torch.tensor(logs["loss"]))
                wandb.log({"train/perplexity": perplexity}, step=state.global_step)

            print(f"Step {state.global_step}: Loss={logs.get('loss', 'N/A'):.4f}")

    def on_save(self, args, state, control, **kwargs):
        """保存檢查點時的回調"""
        print(f"檢查點已保存在 step {state.global_step}")

# 使用
# trainer = Trainer(
#     ...
#     callbacks=[PretrainingMonitor()]
# )
```

---

## 4.4 Scaling Laws 與高效預訓練技術

### 4.4.1 Scaling Laws

**Chinchilla Scaling Laws (2022)**：

研究發現，模型性能取決於三個因素：
1. 模型參數量 (N)
2. 訓練資料量 (D)
3. 計算量 (C)

**關鍵發現**：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中：
- L：損失
- N：參數量
- D：訓練 tokens 數量
- E, A, B, α, β：擬合常數

**最優配置**：

對於給定的計算預算 C：
- 模型大小和訓練資料應該**同時增長**
- 最優比例：每增加 1 倍參數，增加約 20 倍訓練資料

**實際應用**：

| 模型 | 參數 | 訓練 Tokens | 是否符合 Chinchilla |
|------|------|-------------|-------------------|
| GPT-3 | 175B | 300B | ❌ 訓練不足 |
| Chinchilla | 70B | 1.4T | ✅ 最優配置 |
| LLaMA | 7B-65B | 1T-1.4T | ✅ 符合 |
| LLaMA 2 | 7B-70B | 2T | ✅ 符合 |

**啟示**：
- 不要盲目增大模型規模
- 充足的訓練資料同樣重要
- 小模型 + 更多資料可能更有效

### 4.4.2 高效預訓練技術

#### Flash Attention

**原理**：
- 優化注意力計算的記憶體訪問模式
- 減少 HBM (High Bandwidth Memory) 訪問
- 使用 SRAM 進行快取

**效果**：
- 速度提升 2-4 倍
- 記憶體使用減少
- 支援更長的序列

**使用**：

```python
from flash_attn import flash_attn_qkvpacked_func

# 在模型中使用 Flash Attention
# transformers 庫已經整合了 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

#### Gradient Checkpointing

**原理**：
- 不保存所有中間啟用值
- 需要時重新計算
- 以時間換空間

**使用**：

```python
model.gradient_checkpointing_enable()

# 或在訓練參數中設定
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
)
```

**效果**：
- 記憶體使用減少 30-50%
- 訓練速度降低 15-25%
- 允許使用更大的 batch size

#### Mixed Precision Training

**FP16 訓練**：

```python
training_args = TrainingArguments(
    ...
    fp16=True,  # 啟用 FP16
)
```

**BF16 訓練**（更穩定）：

```python
training_args = TrainingArguments(
    ...
    bf16=True,  # 啟用 BF16（需要 Ampere+ GPU）
)
```

**效果**：
- 速度提升 2-3 倍
- 記憶體減半
- BF16 數值穩定性更好

#### ZeRO (Zero Redundancy Optimizer)

**DeepSpeed ZeRO 三階段**：

**Stage 1**：
- 分割優化器狀態
- 記憶體減少 4 倍

**Stage 2**：
- 分割優化器狀態 + 梯度
- 記憶體減少 8 倍

**Stage 3**：
- 分割優化器狀態 + 梯度 + 模型參數
- 記憶體減少 64 倍以上

**配置**：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}
```

---

## 4.5 實作範例

### 4.5.1 小規模預訓練實驗

```python
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def small_scale_pretraining():
    """小規模預訓練實驗（教學目的）"""

    print("=" * 60)
    print("小規模 GPT 預訓練實驗")
    print("=" * 60)

    # 1. 建立小型模型配置
    config = GPT2Config(
        vocab_size=50257,      # GPT-2 詞彙表大小
        n_positions=512,       # 最大序列長度
        n_embd=256,            # 嵌入維度
        n_layer=6,             # 層數
        n_head=8,              # 注意力頭數
    )

    model = GPT2LMHeadModel(config)

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數數量: {total_params / 1e6:.2f}M")

    # 2. 準備資料
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 使用小型資料集（WikiText）
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # 3. 訓練配置
    training_args = TrainingArguments(
        output_dir="./tiny_gpt",
        overwrite_output_dir=True,

        # 訓練設定
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,

        # 優化器
        learning_rate=5e-4,
        weight_decay=0.1,

        # 學習率調度
        lr_scheduler_type="cosine",
        warmup_steps=100,

        # 記錄
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,

        # 混合精度
        fp16=torch.cuda.is_available(),
    )

    # 4. 資料整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM，不是 Masked LM
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. 訓練
    print("\n開始訓練...")
    trainer.train()

    # 7. 保存模型
    trainer.save_model("./tiny_gpt_final")
    tokenizer.save_pretrained("./tiny_gpt_final")

    print("\n訓練完成！模型已保存至 ./tiny_gpt_final")

    # 8. 測試生成
    print("\n測試文字生成:")
    model.eval()
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")

    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n生成結果:\n{generated_text}")

# 執行
# small_scale_pretraining()
```

### 4.5.2 從預訓練模型微調

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def finetune_from_pretrained():
    """從預訓練模型開始微調"""

    # 1. 載入預訓練模型
    model_name = "gpt2"  # 可替換為其他模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 應用 LoRA（參數高效微調）
    lora_config = LoraConfig(
        r=8,                          # LoRA 秩
        lora_alpha=32,                # 縮放因子
        target_modules=["c_attn"],    # 應用 LoRA 的模組
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 準備資料（使用指令資料集）
    dataset = load_dataset("json", data_files="your_instruction_data.json", split="train")

    def format_instruction(example):
        """格式化指令資料"""
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}

    dataset = dataset.map(format_instruction)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # 4. 訓練配置
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,          # LoRA 可使用較大學習率
        logging_steps=10,
        save_steps=100,
        fp16=True,
    )

    # 5. 訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # 6. 保存 LoRA 權重
    model.save_pretrained("./lora_weights")
    print("LoRA 權重已保存")

# 使用
# finetune_from_pretrained()
```

### 4.5.3 評估預訓練模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def evaluate_pretrained_model(model_name="gpt2"):
    """評估預訓練模型的困惑度"""

    print(f"評估模型: {model_name}")

    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()

    # 載入測試資料
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # 計算困惑度
    total_loss = 0
    total_tokens = 0

    for i, example in enumerate(test_dataset):
        if i >= 100:  # 只評估前 100 個樣本（示範）
            break

        text = example["text"]
        if len(text.strip()) == 0:
            continue

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 計算損失
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

        if (i + 1) % 10 == 0:
            print(f"已處理 {i+1} 個樣本...")

    # 計算平均困惑度
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"\n結果:")
    print(f"平均損失: {avg_loss:.4f}")
    print(f"困惑度 (Perplexity): {perplexity:.2f}")

    return perplexity

# 使用
# evaluate_pretrained_model("gpt2")
# evaluate_pretrained_model("meta-llama/Llama-2-7b-hf")
```

---

## 參考資源

### 論文

1. **Scaling Laws**: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
2. **Chinchilla**: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
3. **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
4. **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
5. **Mistral**: "Mistral 7B" (Jiang et al., 2023)
6. **Mixtral**: "Mixtral of Experts" (Jiang et al., 2024)
7. **BLOOM**: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" (BigScience, 2022)
8. **DeepSeek-V3**: "DeepSeek-V3 Technical Report" (2024)

### 工具與框架

- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **DeepSpeed**: https://github.com/microsoft/DeepSpeed
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **PyTorch FSDP**: https://pytorch.org/docs/stable/fsdp.html
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention

### 開源模型

- **LLaMA**: https://github.com/facebookresearch/llama
- **Mistral**: https://huggingface.co/mistralai
- **Qwen**: https://huggingface.co/Qwen
- **BLOOM**: https://huggingface.co/bigscience/bloom
- **DeepSeek**: https://huggingface.co/deepseek-ai

### 資料集

- **The Pile**: https://pile.eleuther.ai/
- **Common Crawl**: https://commoncrawl.org/
- **Wikipedia**: https://huggingface.co/datasets/wikipedia
- **C4**: https://huggingface.co/datasets/c4
- **BookCorpus**: https://huggingface.co/datasets/bookcorpus

---

## 總結

預訓練是 LLM 開發的基礎階段：

### 核心要點

1. **選擇合適的預訓練模型**
   - 根據任務需求選擇模型規模
   - 考慮語言支援和領域適配
   - 評估硬體和成本限制

2. **理解 Scaling Laws**
   - 模型大小和訓練資料需要平衡
   - 不要盲目追求大模型
   - Chinchilla 最優比例：N : D ≈ 1 : 20

3. **高效訓練技術**
   - Flash Attention 提升速度
   - ZeRO 減少記憶體需求
   - 混合精度訓練加速
   - Gradient Checkpointing 節省記憶體

4. **實務建議**
   - 小團隊：使用開源預訓練模型 + 微調
   - 中型團隊：領域特定的持續預訓練
   - 大型團隊：從頭預訓練

5. **開源生態**
   - LLaMA 系列：性能優秀，廣泛使用
   - Mistral：效率高，MoE 架構
   - Qwen：中文優化
   - DeepSeek：極低成本，開源

### 未來趨勢

1. **更高效的架構**：MoE、State Space Models
2. **更長的上下文**：百萬 token 級別
3. **多模態整合**：文字、圖像、音訊統一模型
4. **更低的訓練成本**：新的訓練技術和硬體
5. **開源化**：更多高品質開源模型
