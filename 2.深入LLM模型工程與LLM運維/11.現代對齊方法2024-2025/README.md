# 現代LLM對齊方法 2024-2025

> **最後更新**: 2025-12-14
> **狀態**: 涵蓋RLHF之後的新一代對齊技術

---

## 📋 目錄

1. [對齊技術演進](#1-對齊技術演進)
2. [DPO: Direct Preference Optimization](#2-dpo-direct-preference-optimization)
3. [IPO: Identity Preference Optimization](#3-ipo-identity-preference-optimization)
4. [SimPO: Simple Preference Optimization](#4-simpo-simple-preference-optimization)
5. [KTO: Kahneman-Tversky Optimization](#5-kto-kahneman-tversky-optimization)
6. [ORPO: Odds Ratio Preference Optimization](#6-orpo-odds-ratio-preference-optimization)
7. [方法對比與選擇指南](#7-方法對比與選擇指南)
8. [實戰案例](#8-實戰案例)

---

## 1. 對齊技術演進

### 1.1 從RLHF到直接偏好學習

```
┌─────────────────────────────────────────────────────────────────┐
│                    對齊技術演進時間線                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2020          2022          2023          2024          2025   │
│    │            │             │             │             │     │
│    ▼            ▼             ▼             ▼             ▼     │
│  ┌────┐      ┌────┐       ┌────┐       ┌────┐       ┌────┐    │
│  │RLHF│  →   │RLHF│   →   │DPO │   →   │IPO │   →   │KTO │    │
│  │基礎│      │成熟│       │    │       │SimPO│      │ORPO│    │
│  └────┘      └────┘       └────┘       └────┘       └────┘    │
│                                                                 │
│  特點:                                                          │
│  RLHF: 需要獎勵模型 + PPO訓練，複雜度高                         │
│  DPO:  直接從偏好學習，無需獎勵模型                             │
│  IPO:  解決DPO的過擬合問題                                      │
│  SimPO: 簡化DPO，無需參考模型                                   │
│  KTO:  使用前景理論，支持非配對數據                             │
│  ORPO: 整合SFT和對齊，一步完成                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 方法核心對比

| 方法 | 需要獎勵模型 | 需要參考模型 | 數據需求 | 訓練複雜度 | 穩定性 |
|------|------------|------------|---------|-----------|--------|
| RLHF | ✅ | ✅ | 配對偏好 | 🔴 高 | 🟡 中 |
| DPO | ❌ | ✅ | 配對偏好 | 🟢 低 | 🟢 高 |
| IPO | ❌ | ✅ | 配對偏好 | 🟢 低 | 🟢 高 |
| SimPO | ❌ | ❌ | 配對偏好 | 🟢 最低 | 🟢 高 |
| KTO | ❌ | ✅ | 非配對 | 🟢 低 | 🟢 高 |
| ORPO | ❌ | ❌ | 配對偏好 | 🟢 低 | 🟢 高 |

---

## 2. DPO: Direct Preference Optimization

### 2.1 核心原理

DPO (Direct Preference Optimization) 是2023年由Stanford團隊提出的方法，通過數學推導將RLHF的目標函數轉化為簡單的分類損失，避免了訓練獎勵模型和使用PPO的複雜性。

**核心公式**:

```
L_DPO = -E[(x, y_w, y_l)] [log σ(β * (log π_θ(y_w|x) / π_ref(y_w|x)
                                    - log π_θ(y_l|x) / π_ref(y_l|x)))]
```

其中:
- `y_w`: 優選回答 (winner)
- `y_l`: 劣選回答 (loser)
- `π_θ`: 當前模型
- `π_ref`: 參考模型 (通常是SFT後的模型)
- `β`: 溫度參數，控制與參考模型的偏離程度

### 2.2 實現代碼

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 參考模型 (通常是SFT後的checkpoint)
ref_model = AutoModelForCausalLM.from_pretrained(
    "path/to/sft-checkpoint",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 準備數據集
# 格式: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("your_preference_dataset")

# DPO配置
dpo_config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,  # 關鍵超參數
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    max_prompt_length=512,
)

# 訓練
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

### 2.3 從頭實現DPO損失

```python
def dpo_loss(
    model_logps_chosen: torch.Tensor,
    model_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    計算DPO損失

    Args:
        model_logps_chosen: 當前模型對優選回答的log概率
        model_logps_rejected: 當前模型對劣選回答的log概率
        ref_logps_chosen: 參考模型對優選回答的log概率
        ref_logps_rejected: 參考模型對劣選回答的log概率
        beta: 溫度參數

    Returns:
        DPO損失值
    """
    # 計算log ratio
    chosen_logratios = model_logps_chosen - ref_logps_chosen
    rejected_logratios = model_logps_rejected - ref_logps_rejected

    # DPO損失 = -log(sigmoid(beta * (chosen_ratio - rejected_ratio)))
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 計算準確率 (chosen分數是否高於rejected)
    accuracy = (logits > 0).float().mean()

    return loss, accuracy

def compute_log_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """計算序列的log概率"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 計算每個token的log概率
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 選擇對應label的log概率
    per_token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 使用mask過濾padding
    mask = (shift_labels != -100).float()
    log_prob_sum = (per_token_logps * mask).sum(dim=-1)

    return log_prob_sum
```

### 2.4 DPO最佳實踐

```python
# 超參數建議
dpo_hyperparams = {
    "beta": {
        "range": [0.05, 0.5],
        "default": 0.1,
        "notes": "較小的beta允許更大的偏離，較大的beta保守學習"
    },
    "learning_rate": {
        "range": [1e-7, 5e-6],
        "default": 5e-7,
        "notes": "比SFT低10-100倍"
    },
    "epochs": {
        "range": [1, 3],
        "default": 1,
        "notes": "通常1-2 epochs足夠，過多會過擬合"
    },
    "batch_size": {
        "effective": 32,  # gradient_accumulation * per_device
        "notes": "較大的batch size更穩定"
    }
}

# 數據質量檢查
def validate_preference_data(dataset):
    """驗證偏好數據質量"""
    issues = []

    for idx, example in enumerate(dataset):
        # 檢查必需字段
        if "prompt" not in example or "chosen" not in example or "rejected" not in example:
            issues.append(f"樣本 {idx}: 缺少必需字段")
            continue

        # 檢查chosen和rejected是否相同
        if example["chosen"] == example["rejected"]:
            issues.append(f"樣本 {idx}: chosen和rejected相同")

        # 檢查長度
        if len(example["chosen"]) < 10 or len(example["rejected"]) < 10:
            issues.append(f"樣本 {idx}: 回答過短")

    return issues
```

---

## 3. IPO: Identity Preference Optimization

### 3.1 核心改進

IPO (Identity Preference Optimization) 解決了DPO的一個關鍵問題：當偏好數據確定性很高時（幾乎總是選擇y_w），DPO會過擬合。

**IPO損失函數**:

```
L_IPO = E[(log π_θ(y_w|x) / π_ref(y_w|x)
         - log π_θ(y_l|x) / π_ref(y_l|x) - 1/2τ)²]
```

### 3.2 實現代碼

```python
def ipo_loss(
    model_logps_chosen: torch.Tensor,
    model_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    """
    計算IPO損失

    Args:
        tau: 正則化參數
    """
    chosen_logratios = model_logps_chosen - ref_logps_chosen
    rejected_logratios = model_logps_rejected - ref_logps_rejected

    # IPO使用MSE而非log sigmoid
    logits = chosen_logratios - rejected_logratios
    loss = (logits - 1 / (2 * tau)) ** 2

    return loss.mean()

# TRL中使用IPO
from trl import DPOConfig

ipo_config = DPOConfig(
    output_dir="./ipo-output",
    loss_type="ipo",  # 關鍵：指定IPO損失
    beta=0.1,
    # ... 其他參數
)
```

---

## 4. SimPO: Simple Preference Optimization

### 4.1 核心創新

SimPO (Simple Preference Optimization) 的主要創新是**不需要參考模型**，通過使用平均log概率作為隱式獎勵，簡化了訓練流程。

**SimPO損失函數**:

```
L_SimPO = -log σ(β/|y_w| * log π_θ(y_w|x) - β/|y_l| * log π_θ(y_l|x) - γ)
```

其中:
- `|y_w|`, `|y_l|`: 回答長度（用於長度歸一化）
- `γ`: margin參數，確保優選和劣選之間有足夠差距

### 4.2 實現代碼

```python
def simpo_loss(
    model_logps_chosen: torch.Tensor,
    model_logps_rejected: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5
) -> torch.Tensor:
    """
    計算SimPO損失

    Args:
        beta: 溫度參數 (SimPO通常使用較大的beta)
        gamma: margin參數
    """
    # 長度歸一化
    chosen_rewards = beta * model_logps_chosen / chosen_lengths
    rejected_rewards = beta * model_logps_rejected / rejected_lengths

    # 帶margin的損失
    logits = chosen_rewards - rejected_rewards - gamma
    loss = -F.logsigmoid(logits).mean()

    return loss

# 使用TRL的SimPO
from trl import DPOConfig

simpo_config = DPOConfig(
    output_dir="./simpo-output",
    loss_type="simpo",
    beta=2.0,  # SimPO推薦較大的beta
    simpo_gamma=0.5,
    # 注意: SimPO不需要ref_model
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # SimPO不需要參考模型！
    args=simpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

### 4.3 SimPO vs DPO

| 特性 | DPO | SimPO |
|------|-----|-------|
| 參考模型 | 需要 | 不需要 |
| 記憶體使用 | 2x模型 | 1x模型 |
| 訓練速度 | 較慢 | 更快 |
| 長度偏見 | 可能存在 | 內建歸一化 |
| 推薦beta | 0.1 | 2.0 |

---

## 5. KTO: Kahneman-Tversky Optimization

### 5.1 核心理念

KTO (Kahneman-Tversky Optimization) 基於行為經濟學的**前景理論**，主要創新是：
1. **不需要配對數據** - 只需要標記好/壞的回答
2. **損失厭惡** - 對壞回答的懲罰大於對好回答的獎勵

**KTO損失函數**:

```
L_KTO = E_chosen[-λ_w * σ(-β * (r_θ(x, y_w) - z_0))]
      + E_rejected[-λ_l * σ(β * (r_θ(x, y_l) - z_0))]

其中 r_θ(x, y) = log π_θ(y|x) - log π_ref(y|x)
```

### 5.2 實現代碼

```python
def kto_loss(
    model_logps_chosen: torch.Tensor,
    model_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float = 0.1,
    lambda_w: float = 1.0,
    lambda_l: float = 1.0
) -> torch.Tensor:
    """
    計算KTO損失

    Args:
        lambda_w: 優選回答的權重
        lambda_l: 劣選回答的權重 (損失厭惡時 lambda_l > lambda_w)
    """
    # 計算獎勵
    chosen_rewards = model_logps_chosen - ref_logps_chosen
    rejected_rewards = model_logps_rejected - ref_logps_rejected

    # KL散度作為baseline (z_0)
    kl_chosen = (ref_logps_chosen - model_logps_chosen).mean().detach()
    kl_rejected = (ref_logps_rejected - model_logps_rejected).mean().detach()
    z_0 = (kl_chosen + kl_rejected) / 2

    # KTO損失
    chosen_loss = -lambda_w * F.logsigmoid(beta * (chosen_rewards - z_0))
    rejected_loss = -lambda_l * F.logsigmoid(-beta * (rejected_rewards - z_0))

    loss = chosen_loss.mean() + rejected_loss.mean()

    return loss

# TRL配置
kto_config = DPOConfig(
    output_dir="./kto-output",
    loss_type="kto",
    beta=0.1,
    desirable_weight=1.0,     # lambda_w
    undesirable_weight=1.33,  # lambda_l (損失厭惡)
)
```

### 5.3 KTO的優勢場景

```python
# KTO特別適合的數據格式
# 不需要配對，只需要單獨標記好/壞

kto_dataset = [
    {"prompt": "問題1", "completion": "好的回答1", "label": True},
    {"prompt": "問題2", "completion": "壞的回答1", "label": False},
    {"prompt": "問題3", "completion": "好的回答2", "label": True},
    # 注意: prompt可以不同，不需要同一個prompt有好壞配對
]

# 轉換現有的人類反饋數據
def convert_feedback_to_kto(feedback_data):
    """
    將用戶反饋數據轉換為KTO格式

    原始格式: [{"prompt": ..., "response": ..., "rating": 1-5}]
    """
    kto_data = []

    for item in feedback_data:
        kto_data.append({
            "prompt": item["prompt"],
            "completion": item["response"],
            "label": item["rating"] >= 4  # 4-5分視為好回答
        })

    return kto_data
```

---

## 6. ORPO: Odds Ratio Preference Optimization

### 6.1 核心創新

ORPO (Odds Ratio Preference Optimization) 的創新是**整合SFT和對齊為一步**，通過在SFT損失中加入對比項。

**ORPO損失函數**:

```
L_ORPO = L_SFT + λ * L_OR

L_OR = -log σ(log odds_θ(y_w|x) / odds_θ(y_l|x))
```

### 6.2 實現代碼

```python
def orpo_loss(
    model_logps_chosen: torch.Tensor,
    model_logps_rejected: torch.Tensor,
    chosen_nll: torch.Tensor,  # SFT損失部分
    lambda_orpo: float = 1.0
) -> torch.Tensor:
    """
    計算ORPO損失

    Args:
        chosen_nll: 優選回答的負對數似然 (SFT損失)
        lambda_orpo: 對比項權重
    """
    # 計算odds ratio
    log_odds_chosen = model_logps_chosen - torch.log1p(-torch.exp(model_logps_chosen).clamp(max=0.9999))
    log_odds_rejected = model_logps_rejected - torch.log1p(-torch.exp(model_logps_rejected).clamp(max=0.9999))

    # Odds ratio損失
    or_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    # 總損失 = SFT + lambda * OR
    total_loss = chosen_nll.mean() + lambda_orpo * or_loss

    return total_loss

# TRL配置
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    output_dir="./orpo-output",
    beta=0.1,
    learning_rate=5e-6,  # ORPO通常可以用較高的學習率
    per_device_train_batch_size=4,
    num_train_epochs=1,
    # ORPO不需要參考模型
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

### 6.3 ORPO的優勢

1. **一步完成** - 無需先SFT再對齊
2. **無需參考模型** - 節省記憶體
3. **更快收斂** - 同時學習任務和偏好

---

## 7. 方法對比與選擇指南

### 7.1 決策樹

```
                    開始
                      │
                      ▼
            ┌─────────────────┐
            │ 是否有配對數據？ │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
        是                      否
         │                       │
         ▼                       ▼
    ┌─────────┐            ┌─────────┐
    │需要SFT嗎│            │  KTO    │
    └────┬────┘            └─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
   是        否
    │         │
    ▼         ▼
 ┌─────┐  ┌─────────────┐
 │ORPO │  │ 記憶體受限？  │
 └─────┘  └──────┬──────┘
                 │
            ┌────┴────┐
            │         │
            ▼         ▼
           是        否
            │         │
            ▼         ▼
        ┌──────┐  ┌─────┐
        │SimPO │  │ DPO │
        └──────┘  └─────┘
```

### 7.2 場景推薦

| 場景 | 推薦方法 | 原因 |
|------|---------|------|
| **資源有限** | SimPO | 無需參考模型，記憶體減半 |
| **數據質量高** | DPO | 標準方法，效果穩定 |
| **數據可能有噪音** | IPO | 抗過擬合能力強 |
| **只有單獨標記** | KTO | 不需要配對數據 |
| **從頭訓練** | ORPO | 一步完成SFT+對齊 |
| **生產環境** | DPO/SimPO | 成熟穩定 |

### 7.3 超參數速查表

```python
hyperparams_by_method = {
    "DPO": {
        "beta": 0.1,
        "learning_rate": 5e-7,
        "epochs": 1,
        "batch_size": 32
    },
    "IPO": {
        "tau": 0.1,  # 替代beta
        "learning_rate": 5e-7,
        "epochs": 1
    },
    "SimPO": {
        "beta": 2.0,  # 較大
        "gamma": 0.5,
        "learning_rate": 5e-7
    },
    "KTO": {
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.33  # 損失厭惡
    },
    "ORPO": {
        "beta": 0.1,
        "learning_rate": 5e-6  # 較高
    }
}
```

---

## 8. 實戰案例

### 8.1 完整訓練流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

# 1. 載入基礎模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 2. 添加LoRA (可選，節省記憶體)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. 載入偏好數據
dataset = load_dataset("your_preference_dataset")

def format_dataset(example):
    """格式化數據"""
    return {
        "prompt": f"問題: {example['question']}\n回答: ",
        "chosen": example["chosen_response"],
        "rejected": example["rejected_response"]
    }

dataset = dataset.map(format_dataset)

# 4. 選擇對齊方法
# 方法A: DPO (需要參考模型)
if USE_DPO:
    ref_model = AutoModelForCausalLM.from_pretrained(
        "path/to/sft-model",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    config = DPOConfig(
        output_dir="./dpo-output",
        loss_type="sigmoid",  # DPO默認
        beta=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

# 方法B: SimPO (不需要參考模型)
elif USE_SIMPO:
    config = DPOConfig(
        output_dir="./simpo-output",
        loss_type="simpo",
        beta=2.0,
        simpo_gamma=0.5,
        learning_rate=5e-7,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # SimPO不需要
        args=config,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

# 方法C: KTO (非配對數據)
elif USE_KTO:
    # KTO數據格式不同
    kto_dataset = convert_to_kto_format(dataset)

    config = DPOConfig(
        output_dir="./kto-output",
        loss_type="kto",
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.33,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=kto_dataset,
        tokenizer=tokenizer,
    )

# 5. 訓練
trainer.train()

# 6. 保存模型
trainer.save_model("./final-aligned-model")
```

### 8.2 評估對齊效果

```python
from datasets import load_dataset
import numpy as np

def evaluate_alignment(model, tokenizer, eval_dataset, method="pairwise"):
    """
    評估對齊效果

    Args:
        method: "pairwise" (配對比較) 或 "rating" (絕對評分)
    """
    if method == "pairwise":
        wins, losses, ties = 0, 0, 0

        for example in eval_dataset:
            prompt = example["prompt"]

            # 生成回答
            response = generate(model, tokenizer, prompt)

            # 使用GPT-4評判
            judge_result = judge_preference(
                prompt=prompt,
                response_a=response,
                response_b=example["baseline_response"]
            )

            if judge_result == "A":
                wins += 1
            elif judge_result == "B":
                losses += 1
            else:
                ties += 1

        win_rate = wins / (wins + losses + ties)
        return {"win_rate": win_rate, "wins": wins, "losses": losses, "ties": ties}

    elif method == "rating":
        ratings = []

        for example in eval_dataset:
            response = generate(model, tokenizer, example["prompt"])

            # 使用GPT-4打分
            rating = rate_response(
                prompt=example["prompt"],
                response=response,
                criteria=["helpfulness", "harmlessness", "honesty"]
            )
            ratings.append(rating)

        return {
            "mean_rating": np.mean(ratings),
            "std_rating": np.std(ratings)
        }

def judge_preference(prompt: str, response_a: str, response_b: str) -> str:
    """使用GPT-4作為judge"""
    judge_prompt = f"""
    請比較以下兩個回答，選出更好的一個。

    問題: {prompt}

    回答A: {response_a}

    回答B: {response_b}

    請回答 "A" 或 "B" 或 "Tie"。只需要回答字母，不需要解釋。
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=1
    )

    return response.choices[0].message.content.strip()
```

---

## 📚 參考文獻

1. **DPO**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
2. **IPO**: Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback" (2023)
3. **SimPO**: Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024)
4. **KTO**: Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2024)
5. **ORPO**: Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" (2024)

---

## 🔗 相關章節

- [監督微調 (SFT)](../5.監督微調%20(SFT)/README.md)
- [偏好對齊技術](../6.偏好對齊%20(Alignment)%20技術/README.md)
- [模型評估](../9.模型評估/README.md)
