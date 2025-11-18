# 偏好對齊 (Alignment) 技術

## 目錄
- [6.1 RLHF 基本概念](#61-rlhf-基本概念)
  - [什麼是 RLHF](#什麼是-rlhf)
  - [RLHF 的三階段流程](#rlhf-的三階段流程)
  - [人類回饋的重要性](#人類回饋的重要性)
  - [獎勵模型訓練](#獎勵模型訓練)
  - [PPO 演算法](#ppo-演算法)
- [6.2 DPO 與其他替代方案](#62-dpo-與其他替代方案)
- [6.3 偏好資料集的建立與收集](#63-偏好資料集的建立與收集)
- [6.4 StackLLaMA 實踐範例](#64-stackllama-實踐範例)
- [6.5 對齊技術比較與選擇](#65-對齊技術比較與選擇)
- [6.6 最佳實踐與常見問題](#66-最佳實踐與常見問題)

---

## 6.1 RLHF 基本概念

### 什麼是 RLHF

**RLHF (Reinforcement Learning with Human Feedback)** 是一種將人類偏好融入大型語言模型的訓練方法，目的是讓模型生成更符合人類期望、更有幫助、更安全的輸出。

#### 為什麼需要 RLHF？

傳統的語言模型訓練（如預訓練和監督微調）存在以下問題：

1. **預測下一個 token ≠ 有用的輸出**
   - 模型學會預測文本，但不一定學會「有幫助」
   - 可能生成語法正確但無用的內容

2. **難以定義「好」的目標函數**
   - 什麼是有幫助的回答？
   - 什麼是安全的輸出？
   - 這些很難用傳統的損失函數表達

3. **人類偏好難以量化**
   - 有些偏好是主觀的
   - 需要人類判斷來指導

**RLHF 的解決方案：**
- 通過人類反饋訓練獎勵模型
- 使用強化學習優化模型，使其最大化獎勵
- 將「好」的定義交給人類標註者

#### RLHF 的成功案例

- **ChatGPT**：OpenAI 使用 RLHF 訓練 InstructGPT 和 ChatGPT
- **Claude**：Anthropic 使用 RLHF + Constitutional AI
- **Llama 2-Chat**：Meta 的對話模型使用 RLHF
- **Gemini**：Google 的多模態模型使用 RLHF

### RLHF 的三階段流程

RLHF 是一個包含三個階段的訓練流程：

```
階段 1: 監督微調 (SFT)
   ↓
階段 2: 獎勵模型訓練 (Reward Model)
   ↓
階段 3: 強化學習優化 (PPO)
```

#### 階段 1：監督微調（Supervised Fine-Tuning, SFT）

**目標：** 讓預訓練模型學會遵循指令和對話格式

**數據格式：**
```json
{
  "prompt": "解釋量子計算的基本原理",
  "response": "量子計算是一種利用量子力學原理進行計算的技術..."
}
```

**訓練過程：**
1. 收集高質量的指令-回答配對數據（通常數萬條）
2. 使用標準的監督學習訓練模型
3. 目標是最小化交叉熵損失

**實現示例：**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 載入預訓練模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# 準備指令數據集
dataset = load_dataset("your_instruction_dataset")

def formatting_func(example):
    text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    return tokenizer(text, truncation=True, max_length=2048)

tokenized_dataset = dataset.map(formatting_func)

# 訓練配置
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

# 訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()
```

**關鍵要點：**
- SFT 數據質量 > 數量（幾千到幾萬條高質量數據即可）
- 需要多樣化的指令類型
- 回答應該是高質量的示範

#### 階段 2：獎勵模型訓練（Reward Model Training）

**目標：** 訓練一個模型來預測人類偏好，作為強化學習的獎勵信號

**數據收集流程：**

1. **生成候選回答**
   - 使用 SFT 模型對同一個 prompt 生成多個不同的回答（通常 4-9 個）

2. **人類排序**
   - 標註者根據質量對這些回答進行排序
   - 形成偏好對：(chosen, rejected)

3. **數據格式**
```json
{
  "prompt": "如何學習機器學習？",
  "chosen": "學習機器學習建議從以下步驟開始：1) 掌握數學基礎...",
  "rejected": "學機器學習很簡單，看幾本書就行了。"
}
```

**獎勵模型架構：**
```python
# 獎勵模型 = SFT 模型 + 線性層（輸出標量獎勵）
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # SFT 模型
        self.value_head = nn.Linear(hidden_size, 1)  # 獎勵頭

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # 取最後一個 token 的隱藏狀態
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        # 輸出獎勵分數
        reward = self.value_head(last_hidden_state)
        return reward
```

**訓練目標：**

使用配對損失（Pairwise Ranking Loss）：

```python
# 損失函數：讓 chosen 的獎勵高於 rejected
loss = -log(sigmoid(reward_chosen - reward_rejected))
```

**完整訓練代碼：**
```python
from transformers import AutoModelForSequenceClassification, Trainer

# 載入 SFT 模型作為基礎
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/sft_model",
    num_labels=1  # 輸出單一獎勵分數
)

# 準備偏好數據
def prepare_reward_data(example):
    chosen_encoding = tokenizer(
        example["prompt"] + example["chosen"],
        truncation=True,
        max_length=2048
    )
    rejected_encoding = tokenizer(
        example["prompt"] + example["rejected"],
        truncation=True,
        max_length=2048
    )
    return {
        "input_ids_chosen": chosen_encoding["input_ids"],
        "attention_mask_chosen": chosen_encoding["attention_mask"],
        "input_ids_rejected": rejected_encoding["input_ids"],
        "attention_mask_rejected": rejected_encoding["attention_mask"]
    }

# 自定義 Trainer 來計算配對損失
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs):
        reward_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"]
        ).logits
        reward_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"]
        ).logits

        # 配對損失
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
        return loss

# 訓練獎勵模型
reward_trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=reward_dataset
)

reward_trainer.train()
```

**關鍵要點：**
- 需要大量高質量的偏好數據（通常 10 萬對以上）
- 標註者的一致性很重要
- 獎勵模型容易過擬合，需要正則化

### 人類回饋的重要性

#### 為什麼人類回饋如此重要？

1. **捕捉複雜的人類價值觀**
   - 有些概念難以用規則定義（如「有幫助」、「安全」）
   - 人類判斷可以捕捉微妙的偏好

2. **對齊現實世界需求**
   - 模型需要在實際應用中有用
   - 人類反饋確保模型符合實際需求

3. **動態調整**
   - 隨著技術發展，人類偏好可能變化
   - 通過人類反饋可以持續調整模型

#### 人類回饋的挑戰

**1. 標註成本高**
- 需要大量人力
- 標註者需要培訓
- 成本可達數十萬到數百萬美元

**2. 標註質量不一致**
- 不同標註者有不同偏好
- 需要建立詳細的標註指南
- 需要質量控制機制

**3. 偏見問題**
- 標註者可能帶有自己的偏見
- 需要多樣化的標註團隊
- 需要識別和減輕偏見

**4. 可擴展性問題**
- 人類標註速度有限
- 難以處理海量數據
- 需要考慮自動化方案（如 RLAIF）

### 獎勵模型訓練

#### 獎勵模型的作用

獎勵模型是 RLHF 的核心組件，它的作用是：
1. 將人類偏好轉化為可計算的獎勵信號
2. 在 RL 訓練時評估模型輸出的質量
3. 指導模型朝著人類偏好的方向優化

#### 獎勵模型訓練的關鍵技術

**1. 數據增強**
```python
# 通過採樣溫度生成多樣化的回答
responses = []
for temp in [0.7, 0.8, 0.9, 1.0]:
    response = model.generate(
        prompt,
        temperature=temp,
        do_sample=True
    )
    responses.append(response)
```

**2. 獎勵模型正則化**
```python
# 防止獎勵模型過度自信
loss = pairwise_loss + regularization_term

# 常見正則化方法：
# - L2 正則化
# - Dropout
# - 早停
```

**3. 獎勵裁剪**
```python
# 避免獎勵值過大導致訓練不穩定
reward = torch.clamp(reward, min=-10, max=10)
```

#### 獎勵模型的評估

**1. 準確率**
- 在測試集上，獎勵模型對偏好對的預測準確率
- 通常目標：> 65-70%

**2. 與人類判斷的一致性**
```python
# 計算獎勵排序與人類排序的相關性
from scipy.stats import spearmanr

human_rankings = [3, 1, 2, 4]  # 人類排序
model_rewards = [0.8, 1.2, 1.0, 0.6]  # 模型獎勵
correlation, p_value = spearmanr(human_rankings, model_rewards)
```

**3. 獎勵分佈**
- 檢查獎勵分佈是否合理
- 避免獎勵值塌陷或爆炸

### PPO 演算法

#### 什麼是 PPO？

**PPO (Proximal Policy Optimization)** 是一種強化學習算法，用於 RLHF 的第三階段優化。

**核心思想：**
- 限制每次更新的策略變化幅度
- 避免訓練過程中的性能崩潰
- 平衡探索和利用

#### PPO 的數學原理

**1. 策略梯度**

目標是最大化期望獎勵：
```
J(θ) = E[R(τ)]
```

其中 τ 是軌跡（prompt + response），R 是總獎勵。

**2. PPO 的目標函數**

```
L^CLIP(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
```

其中：
- `r(θ) = π_θ(a|s) / π_old(a|s)` 是概率比
- `A` 是優勢函數（Advantage）
- `ε` 是裁剪參數（通常 0.2）

**3. 裁剪機制**

```python
# 限制策略更新幅度
ratio = new_policy / old_policy
clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
```

#### PPO 在 RLHF 中的應用

**完整訓練流程：**

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. 載入 SFT 模型和獎勵模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("path/to/sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("path/to/sft_model")
reward_model = AutoModelForSequenceClassification.from_pretrained("path/to/reward_model")

# 2. PPO 配置
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    init_kl_coef=0.2,  # KL 散度懲罰係數
    cliprange=0.2,  # PPO 裁剪參數
    vf_coef=0.1,  # Value function 係數
)

# 3. 初始化 PPO 訓練器
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=prompt_dataset
)

# 4. 訓練循環
for epoch in range(num_epochs):
    for batch in ppo_trainer.dataloader:
        # 生成回答
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )

        # 計算獎勵
        texts = [tokenizer.decode(r) for r in response_tensors]
        rewards = [reward_model(text).item() for text in texts]

        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 記錄訓練指標
        print(f"Epoch {epoch}, Reward: {stats['ppo/mean_scores']}")
```

#### PPO 訓練的關鍵技巧

**1. KL 散度懲罰**

防止模型偏離原始 SFT 模型太遠：

```python
# 計算 KL 散度
kl_div = torch.distributions.kl_divergence(
    new_policy_dist,
    ref_policy_dist
)

# 懲罰項
kl_penalty = -kl_coef * kl_div

# 總獎勵
total_reward = reward_from_reward_model + kl_penalty
```

**2. 價值函數（Value Function）**

預測未來獎勵，減少方差：

```python
# 價值函數損失
value_loss = (returns - values).pow(2).mean()

# 總損失
total_loss = policy_loss + vf_coef * value_loss
```

**3. 優勢函數（Advantage Function）**

```python
# GAE (Generalized Advantage Estimation)
advantages = rewards + gamma * next_values - values
```

**4. 自適應 KL 控制**

```python
# 動態調整 KL 係數
if kl_div > target_kl * 1.5:
    kl_coef *= 1.5  # 增加懲罰
elif kl_div < target_kl * 0.5:
    kl_coef /= 1.5  # 減少懲罰
```

#### PPO 的挑戰和解決方案

**挑戰 1：訓練不穩定**
- **解決方案：**
  - 降低學習率
  - 增加 KL 懲罰
  - 使用梯度裁剪

**挑戰 2：獎勵黑客（Reward Hacking）**
- **問題：** 模型學會利用獎勵模型的漏洞獲得高獎勵
- **解決方案：**
  - KL 散度約束
  - 獎勵模型持續更新
  - 使用更多樣化的獎勵信號

**挑戰 3：計算成本高**
- **解決方案：**
  - 使用較小的批次大小
  - 結合 LoRA 等參數高效方法
  - 考慮使用 DPO 替代

**挑戰 4：樣本效率低**
- **解決方案：**
  - 增加 PPO epochs
  - 使用經驗回放
  - 改進獎勵函數

#### RLHF 完整流程總結

```
預訓練模型
    ↓
【階段 1】監督微調 (SFT)
    ├─ 數據：10K-100K 指令-回答對
    ├─ 方法：標準監督學習
    └─ 輸出：SFT 模型
    ↓
【階段 2】獎勵模型訓練
    ├─ 數據：100K+ 偏好對 (chosen, rejected)
    ├─ 方法：配對排序損失
    └─ 輸出：獎勵模型
    ↓
【階段 3】PPO 強化學習
    ├─ 輸入：SFT 模型 + 獎勵模型
    ├─ 方法：PPO 算法
    └─ 輸出：對齊後的模型
```

**訓練時間和成本（以 Llama 3.1 8B 為例）：**
- SFT：2-4 小時（單 GPU）
- 獎勵模型：4-8 小時（單 GPU）
- PPO：數天到數週（多 GPU）
- 總成本：數千到數萬美元（取決於規模）

---

## 6.2 DPO 與其他替代方案

### DPO (Direct Preference Optimization)

#### 為什麼需要 DPO？

RLHF 雖然有效，但存在以下問題：
1. **複雜的三階段流程**：需要訓練 SFT 模型、獎勵模型和 PPO 優化
2. **訓練不穩定**：PPO 算法難以調優，容易崩潰
3. **計算成本高**：需要同時運行多個大模型
4. **樣本效率低**：PPO 需要大量的訓練樣本

**DPO 的創新：** 繞過獎勵模型和 PPO，直接從偏好數據優化策略。

#### DPO 的核心思想

DPO 將 RLHF 的隱式獎勵函數直接用策略模型本身來表示：

**RLHF 流程：**
```
偏好數據 → 獎勵模型 → PPO 優化 → 對齊模型
```

**DPO 流程：**
```
偏好數據 → 直接優化策略 → 對齊模型
```

#### DPO 的數學原理

**1. 獎勵重參數化**

在 RLHF 中，最優策略 π* 與獎勵模型 r 的關係：
```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

DPO 反轉這個關係，直接用策略表示獎勵：
```
r(x,y) = β · log(π*(y|x) / π_ref(y|x))
```

**2. DPO 損失函數**

給定偏好對 (x, y_w, y_l)，其中 y_w 是更好的回答，y_l 是較差的回答：

```python
loss = -log(σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x))))
```

其中：
- `σ` 是 sigmoid 函數
- `β` 是溫度參數，控制 KL 散度懲罰
- `π_θ` 是要訓練的策略
- `π_ref` 是參考策略（通常是 SFT 模型）

**簡化形式：**
```python
loss = -log(σ(β · (log_ratio_w - log_ratio_l)))
```

#### DPO 完整實現

**1. 數據準備**

```python
from datasets import load_dataset

# DPO 需要的數據格式
dataset = load_dataset("your_preference_dataset")

# 每條數據包含：
# {
#   "prompt": "用戶的問題",
#   "chosen": "更好的回答",
#   "rejected": "較差的回答"
# }

# 示例數據
example = {
    "prompt": "解釋什麼是量子計算",
    "chosen": "量子計算是一種利用量子力學原理（如疊加和糾纏）進行計算的技術...",
    "rejected": "量子計算就是很快的計算機。"
}
```

**2. 模型設置**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 載入 SFT 模型作為基礎
model = AutoModelForCausalLM.from_pretrained(
    "path/to/sft_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 創建參考模型（凍結的 SFT 模型）
ref_model = AutoModelForCausalLM.from_pretrained(
    "path/to/sft_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("path/to/sft_model")

# 可選：使用 LoRA 減少內存
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

**3. DPO 訓練**

```python
from trl import DPOTrainer, DPOConfig

# DPO 配置
dpo_config = DPOConfig(
    # 基本訓練參數
    output_dir="./dpo_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,  # DPO 通常使用較小的學習率

    # DPO 特定參數
    beta=0.1,  # KL 散度懲罰係數（關鍵參數）
    loss_type="sigmoid",  # 損失函數類型

    # 優化參數
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,

    # 日誌和保存
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=100
)

# DPO 訓練器
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    max_length=2048,
    max_prompt_length=1024
)

# 開始訓練
dpo_trainer.train()

# 保存模型
dpo_trainer.save_model("./final_dpo_model")
```

**4. DPO 損失函數詳解**

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO 損失函數實現

    Args:
        policy_chosen_logps: 策略模型對 chosen 回答的 log 概率
        policy_rejected_logps: 策略模型對 rejected 回答的 log 概率
        reference_chosen_logps: 參考模型對 chosen 回答的 log 概率
        reference_rejected_logps: 參考模型對 rejected 回答的 log 概率
        beta: KL 散度懲罰係數
    """
    # 計算 log 概率比
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps

    # DPO 損失
    logits = beta * (policy_log_ratios - reference_log_ratios)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # 計算準確率（用於監控）
    accuracy = (logits > 0).float().mean()

    return loss, accuracy
```

#### DPO 的關鍵參數調優

**1. Beta (β) 參數**

Beta 控制模型偏離參考模型的程度：

| Beta 值 | 效果 | 適用場景 |
|---------|------|---------|
| 0.01-0.05 | 允許較大變化 | 需要顯著改變模型行為 |
| 0.1 | 平衡（推薦起點） | 大多數情況 |
| 0.3-0.5 | 保守變化 | 已經很好的 SFT 模型 |

```python
# Beta 參數對比實驗
for beta in [0.05, 0.1, 0.2, 0.3]:
    dpo_config = DPOConfig(beta=beta, ...)
    dpo_trainer = DPOTrainer(...)
    dpo_trainer.train()
    # 評估模型性能
```

**2. 學習率**

DPO 通常使用比 SFT 更小的學習率：

```python
# 推薦學習率範圍
learning_rates = {
    "7B 模型": 5e-7,
    "13B 模型": 3e-7,
    "70B 模型": 1e-7
}
```

**3. 訓練輪數**

```python
# DPO 通常不需要太多 epochs
num_epochs = 1-3  # 大多數情況下 1-3 個 epoch 足夠
```

#### DPO 與 RLHF 的詳細對比

| 特性 | RLHF (PPO) | DPO |
|-----|-----------|-----|
| **訓練階段** | 3 階段 (SFT → RM → PPO) | 2 階段 (SFT → DPO) |
| **需要的模型** | 4 個 (Policy, Ref, Reward, Value) | 2 個 (Policy, Ref) |
| **穩定性** | 低（PPO 難調） | 高（監督學習） |
| **計算成本** | 高（~3-4x SFT） | 中（~1.5-2x SFT） |
| **內存需求** | 非常高 | 中等 |
| **實現複雜度** | 高 | 低 |
| **訓練時間** | 長（數天到數週） | 短（數小時到數天） |
| **超參數調優** | 困難（多個超參數） | 簡單（主要是 beta） |
| **性能** | 優秀 | 優秀（comparable） |
| **適用場景** | 大規模、複雜獎勵 | 大多數對齊任務 |

#### DPO 的優勢

**1. 簡單性**
```python
# RLHF 需要的組件
components_rlhf = [
    "SFT 模型",
    "獎勵模型",
    "參考模型",
    "價值模型",
    "PPO 訓練器"
]

# DPO 需要的組件
components_dpo = [
    "SFT 模型",
    "參考模型",
    "DPO 訓練器"
]
```

**2. 穩定性**
- DPO 是監督學習，不需要複雜的 RL 算法
- 訓練曲線平滑，容易監控
- 不會出現 PPO 的崩潰問題

**3. 效率**
```python
# 內存需求對比（以 7B 模型為例）
memory_requirements = {
    "RLHF (PPO)": "~120GB",  # 4 個模型
    "DPO": "~40GB",          # 2 個模型
    "DPO + LoRA": "~24GB"    # LoRA 優化
}
```

#### DPO 的局限性

**1. 需要高質量的偏好數據**
- 偏好數據的質量直接影響 DPO 效果
- 需要清晰的 chosen vs rejected 區分

**2. 對 SFT 模型的依賴**
- DPO 的效果依賴於好的 SFT 基礎
- SFT 模型需要已經能夠生成合理的回答

**3. 可能不適合複雜獎勵**
- 對於難以用配對比較表達的獎勵，RLHF 可能更好
- 例如：需要多步推理驗證的任務

#### DPO 實戰技巧

**1. 數據質量檢查**

```python
def check_preference_quality(dataset):
    """檢查偏好數據質量"""
    quality_issues = []

    for example in dataset:
        chosen = example["chosen"]
        rejected = example["rejected"]

        # 檢查長度差異
        if len(chosen) < len(rejected):
            quality_issues.append("Chosen shorter than rejected")

        # 檢查重複
        if chosen == rejected:
            quality_issues.append("Chosen equals rejected")

        # 檢查是否有實質性差異
        similarity = compute_similarity(chosen, rejected)
        if similarity > 0.9:
            quality_issues.append("Too similar")

    return quality_issues
```

**2. 訓練監控指標**

```python
# 關鍵指標
metrics_to_monitor = {
    "loss": "訓練損失（應該下降）",
    "accuracy": "偏好預測準確率（應該 > 60%）",
    "rewards/chosen": "Chosen 回答的獎勵（應該上升）",
    "rewards/rejected": "Rejected 回答的獎勵（應該下降）",
    "rewards/margin": "獎勵差距（應該擴大）",
    "kl_divergence": "KL 散度（不應太大）"
}
```

**3. 評估方法**

```python
# DPO 訓練後的評估
def evaluate_dpo_model(model, test_dataset):
    results = {
        "preference_accuracy": 0,
        "generation_quality": 0,
        "safety_score": 0
    }

    # 1. 偏好準確率
    for example in test_dataset:
        score_chosen = model.score(example["prompt"], example["chosen"])
        score_rejected = model.score(example["prompt"], example["rejected"])
        if score_chosen > score_rejected:
            results["preference_accuracy"] += 1

    # 2. 生成質量評估
    for prompt in test_prompts:
        response = model.generate(prompt)
        quality = evaluate_response_quality(response)
        results["generation_quality"] += quality

    # 3. 安全性評估
    for adversarial_prompt in safety_test_set:
        response = model.generate(adversarial_prompt)
        safety = check_safety(response)
        results["safety_score"] += safety

    return results
```

**4. Beta 參數調優策略**

```python
def find_optimal_beta(model, dataset, beta_range=[0.05, 0.1, 0.2, 0.3]):
    """找到最優的 beta 參數"""
    results = {}

    for beta in beta_range:
        # 訓練模型
        dpo_config = DPOConfig(beta=beta, num_train_epochs=1)
        trainer = DPOTrainer(model=model, args=dpo_config, ...)
        trainer.train()

        # 評估
        eval_results = evaluate_dpo_model(trainer.model, test_dataset)
        results[beta] = eval_results

    # 選擇最佳 beta
    best_beta = max(results.items(), key=lambda x: x[1]["preference_accuracy"])[0]
    return best_beta, results
```

### 其他對齊技術

除了 RLHF 和 DPO，還有多種新興的對齊技術：

#### 1. ORPO (Odds Ratio Preference Optimization)

**核心創新：** 單階段對齊，無需 SFT

**特點：**
- 直接在預訓練模型上進行對齊
- 同時進行指令微調和偏好對齊
- 減少了一個訓練階段

**損失函數：**
```python
# ORPO 同時優化兩個目標
loss_orpo = loss_sft + λ * loss_preference

# SFT 損失：標準的語言模型損失
loss_sft = -log P(y_chosen | x)

# 偏好損失：基於 odds ratio
odds_ratio = (P(y_chosen|x) / (1-P(y_chosen|x))) / (P(y_rejected|x) / (1-P(y_rejected|x)))
loss_preference = -log(sigmoid(log(odds_ratio)))
```

**實現示例：**
```python
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    num_train_epochs=3,
    per_device_train_batch_size=2
)

orpo_trainer = ORPOTrainer(
    model=base_model,  # 直接使用預訓練模型
    args=orpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer
)

orpo_trainer.train()
```

**優勢：**
- 更簡單的流程（1 個階段 vs DPO 的 2 個階段）
- 更高效的訓練
- 較少的計算資源需求

**劣勢：**
- 較新的方法，實踐經驗較少
- 對數據質量要求更高

#### 2. RLAIF (Reinforcement Learning from AI Feedback)

**核心創新：** 用 AI 反饋替代人類反饋

**為什麼需要 RLAIF？**
1. **成本問題**：人類標註非常昂貴
2. **擴展性問題**：人類標註速度有限
3. **一致性問題**：不同標註者的判斷可能不一致

**RLAIF 流程：**
```
1. 使用強大的 AI 模型（如 GPT-4）作為評判者
   ↓
2. AI 模型對回答進行評分和排序
   ↓
3. 生成偏好數據
   ↓
4. 使用 DPO 或 PPO 訓練
```

**實現示例：**
```python
from openai import OpenAI

client = OpenAI()

def generate_ai_preferences(prompt, responses):
    """使用 AI 模型生成偏好數據"""

    # 構建評判提示
    evaluation_prompt = f"""
    Given the following question and two responses, determine which response is better.

    Question: {prompt}

    Response A: {responses[0]}
    Response B: {responses[1]}

    Evaluate based on:
    1. Accuracy and correctness
    2. Helpfulness
    3. Safety and harmlessness
    4. Clarity and coherence

    Output your judgment as: "A" or "B"
    Also provide a brief explanation.
    """

    # 調用 AI 評判者
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": evaluation_prompt}]
    )

    judgment = response.choices[0].message.content

    # 解析判斷結果
    if "A" in judgment:
        return {"chosen": responses[0], "rejected": responses[1]}
    else:
        return {"chosen": responses[1], "rejected": responses[0]}

# 批量生成偏好數據
preference_dataset = []
for prompt in prompts:
    # 生成多個回答
    responses = [model.generate(prompt) for _ in range(2)]

    # AI 評判
    preference = generate_ai_preferences(prompt, responses)
    preference["prompt"] = prompt

    preference_dataset.append(preference)
```

**RLAIF 的優勢：**
- **成本低**：AI 評判比人類便宜
- **可擴展**：可以快速生成大量偏好數據
- **一致性好**：同一 AI 模型的判斷標準一致

**RLAIF 的挑戰：**
- **AI 評判的質量**：依賴於評判模型的能力
- **偏見傳播**：評判模型的偏見會傳遞給訓練模型
- **循環問題**：可能導致模型過度擬合評判者的偏好

**Constitutional AI (Anthropic 的方法)：**

Constitutional AI 是 RLAIF 的一種變體：

```python
# Constitutional AI 流程
def constitutional_ai_training():
    """
    Constitutional AI 包含兩個階段
    """

    # 階段 1: Constitutional AI Critique and Revision (CAR)
    # 使用 AI 自我批評和修正
    def critique_and_revise(model, prompt, response):
        # AI 自我批評
        critique_prompt = f"""
        Review this response for harmful content:
        Prompt: {prompt}
        Response: {response}

        Critique: Is this response harmful, biased, or problematic?
        """
        critique = model.generate(critique_prompt)

        # AI 自我修正
        if is_problematic(critique):
            revision_prompt = f"""
            Revise the following response to be more helpful and harmless:
            Original: {response}
            Critique: {critique}

            Revised response:
            """
            revised_response = model.generate(revision_prompt)
            return revised_response

        return response

    # 階段 2: RLAIF
    # 使用 AI 評判者進行偏好學習
    # ... (與上述 RLAIF 流程類似)
```

#### 3. KTO (Kahneman-Tversky Optimization)

**核心創新：** 基於前景理論的對齊方法

**特點：**
- 不需要配對的偏好數據
- 只需要二元反饋（好/不好）
- 更容易收集數據

**數據格式：**
```json
{
  "prompt": "問題",
  "response": "回答",
  "label": true  // true 表示好，false 表示不好
}
```

**實現示例：**
```python
from trl import KTOTrainer, KTOConfig

kto_config = KTOConfig(
    learning_rate=5e-7,
    num_train_epochs=3,
    per_device_train_batch_size=4
)

kto_trainer = KTOTrainer(
    model=model,
    args=kto_config,
    train_dataset=binary_feedback_dataset,
    tokenizer=tokenizer
)

kto_trainer.train()
```

#### 4. IPO (Identity Preference Optimization)

**核心創新：** 簡化的 DPO 變體

**特點：**
- 移除 DPO 中的 sigmoid 函數
- 使用均方誤差損失
- 訓練更穩定

**損失函數：**
```python
# IPO 損失
loss_ipo = (preference_diff - 1).pow(2).mean()
```

#### 對齊技術選擇指南

| 方法 | 最佳使用場景 | 數據需求 | 計算成本 | 難度 |
|------|------------|---------|---------|------|
| **RLHF (PPO)** | 複雜獎勵、大規模部署 | 大量偏好數據 | 非常高 | 高 |
| **DPO** | 通用對齊任務 | 中等偏好數據 | 中等 | 低 |
| **ORPO** | 資源受限、快速實驗 | 中等偏好數據 | 低 | 低 |
| **RLAIF** | 缺乏人類標註、快速迭代 | 可自動生成 | 中等 | 中 |
| **KTO** | 只有二元反饋 | 簡單反饋數據 | 低 | 低 |
| **IPO** | 需要穩定訓練 | 中等偏好數據 | 中等 | 低 |

#### 2025 年推薦方案

**小型團隊/研究者：**
```
DPO + LoRA
或
ORPO（如果想省略 SFT）
```

**中型團隊：**
```
DPO 全量微調
+ RLAIF（擴充數據）
```

**大型企業：**
```
RLHF (PPO)
+ Constitutional AI（提升安全性）
+ 持續的人類反饋
```

---

## 6.3 偏好資料集的建立與收集

偏好數據是對齊訓練的基礎，數據質量直接決定模型對齊的效果。本節深入探討如何建立高質量的偏好資料集。

### 偏好數據的類型

#### 1. 配對偏好數據（Pairwise Preference）

**格式：**
```json
{
  "prompt": "用戶問題或指令",
  "chosen": "更好的回答",
  "rejected": "較差的回答"
}
```

**使用場景：**
- DPO 訓練
- 獎勵模型訓練
- 大多數對齊方法

**示例：**
```json
{
  "prompt": "如何學習 Python？",
  "chosen": "學習 Python 建議遵循以下步驟：\n1. 從基礎語法開始...\n2. 通過實際項目練習...",
  "rejected": "直接看文檔就可以了。"
}
```

#### 2. 排序偏好數據（Ranking Preference）

**格式：**
```json
{
  "prompt": "用戶問題",
  "responses": [
    "第一好的回答",
    "第二好的回答",
    "第三好的回答",
    "第四好的回答"
  ],
  "ranking": [0, 2, 1, 3]  // 排序索引
}
```

**使用場景：**
- 更精細的偏好學習
- 獎勵模型訓練
- 可以轉換為多個配對數據

**轉換為配對數據：**
```python
def ranking_to_pairs(ranking_data):
    """將排序數據轉換為配對數據"""
    pairs = []
    responses = ranking_data["responses"]
    ranking = ranking_data["ranking"]

    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            if ranking[i] < ranking[j]:  # i 更好
                pairs.append({
                    "prompt": ranking_data["prompt"],
                    "chosen": responses[i],
                    "rejected": responses[j]
                })

    return pairs
```

#### 3. 二元反饋數據（Binary Feedback）

**格式：**
```json
{
  "prompt": "用戶問題",
  "response": "模型回答",
  "label": true  // true=好, false=不好
}
```

**使用場景：**
- KTO 訓練
- 簡單的反饋收集
- 更容易標註

### 偏好數據收集方法

#### 方法 1：人類標註（Human Annotation）

**流程：**

```python
# 1. 生成候選回答
def generate_candidate_responses(model, prompt, num_samples=4):
    """生成多個候選回答"""
    responses = []

    for temp in [0.7, 0.8, 0.9, 1.0]:
        response = model.generate(
            prompt,
            temperature=temp,
            max_length=512,
            do_sample=True
        )
        responses.append(response)

    return responses

# 2. 人類標註界面
def annotate_preferences(prompt, responses):
    """
    展示標註界面供人類選擇

    標註者看到：
    - 問題/指令
    - 多個候選回答
    - 選擇最好的和最差的
    """
    print(f"Prompt: {prompt}\n")

    for i, response in enumerate(responses):
        print(f"Response {i+1}:\n{response}\n")

    best_idx = int(input("選擇最好的回答 (1-4): ")) - 1
    worst_idx = int(input("選擇最差的回答 (1-4): ")) - 1

    return {
        "prompt": prompt,
        "chosen": responses[best_idx],
        "rejected": responses[worst_idx]
    }

# 3. 批量標註
preference_dataset = []
for prompt in prompts:
    responses = generate_candidate_responses(model, prompt)
    preference = annotate_preferences(prompt, responses)
    preference_dataset.append(preference)
```

**優點：**
- 高質量的偏好標註
- 能捕捉細微的人類判斷
- 可以處理複雜場景

**缺點：**
- 成本高（每條數據 $0.1-$1）
- 速度慢（每人每小時 20-50 條）
- 可擴展性差

**質量控制：**

```python
def quality_control(annotations, min_agreement=0.7):
    """標註質量控制"""

    # 1. 多人標註同一數據
    multi_annotations = collect_multiple_annotations(
        data_point, num_annotators=3
    )

    # 2. 計算一致性
    agreement_rate = calculate_agreement(multi_annotations)

    # 3. 過濾低一致性數據
    if agreement_rate < min_agreement:
        flag_for_review(data_point)

    # 4. 黃金標準測試
    test_annotator_quality(annotator, golden_set)

    # 5. 標註指南培訓
    train_annotators_with_guidelines()
```

#### 方法 2：AI 輔助標註（AI-Assisted Annotation）

使用強大的 AI 模型（如 GPT-4）輔助人類標註：

```python
from openai import OpenAI

client = OpenAI()

def ai_assisted_annotation(prompt, responses):
    """AI 輔助標註"""

    # 1. AI 初步評分
    ai_scores = []
    for response in responses:
        evaluation_prompt = f"""
        Rate the following response from 1-10:

        Question: {prompt}
        Response: {response}

        Criteria:
        - Accuracy (0-3)
        - Helpfulness (0-3)
        - Safety (0-2)
        - Clarity (0-2)

        Output only the total score.
        """

        result = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )

        score = int(result.choices[0].message.content.strip())
        ai_scores.append(score)

    # 2. 找出 AI 認為最好和最差的
    best_idx = ai_scores.index(max(ai_scores))
    worst_idx = ai_scores.index(min(ai_scores))

    # 3. 人類審核（只審核不確定的）
    if max(ai_scores) - min(ai_scores) < 3:  # 分數差異小
        # 需要人類審核
        return human_review(prompt, responses)
    else:
        # AI 判斷可信
        return {
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx],
            "confidence": "high"
        }
```

**優點：**
- 顯著降低成本（節省 50-80%）
- 提高標註速度
- 保持質量

**混合策略：**
```python
# AI 標註 + 人類抽樣審核
def hybrid_annotation(dataset, human_review_rate=0.2):
    """
    80% AI 標註，20% 人類審核

    適合大規模數據標註
    """
    ai_annotated = []
    for data in dataset:
        # AI 標註
        annotation = ai_assisted_annotation(data)
        ai_annotated.append(annotation)

    # 隨機抽樣人類審核
    sample_size = int(len(ai_annotated) * human_review_rate)
    review_samples = random.sample(ai_annotated, sample_size)

    for sample in review_samples:
        human_label = human_review(sample)
        if not agrees(human_label, sample):
            # 發現不一致，需要調整 AI 評判提示
            refine_ai_prompt()
```

#### 方法 3：合成偏好數據（Synthetic Preference Data）

使用技術生成偏好數據，無需人類標註：

**3.1 對比生成法**

```python
def generate_synthetic_preferences(prompt, model):
    """生成合成偏好數據"""

    # 生成高質量回答（低溫度）
    good_response = model.generate(
        prompt,
        temperature=0.7,
        top_p=0.9,
        max_length=512
    )

    # 生成低質量回答（高溫度 + 截斷）
    bad_response = model.generate(
        prompt,
        temperature=1.5,  # 更隨機
        top_p=0.5,        # 限制詞彙
        max_length=200    # 更短
    )

    return {
        "prompt": prompt,
        "chosen": good_response,
        "rejected": bad_response
    }
```

**3.2 錯誤注入法**

```python
def inject_errors(good_response):
    """在好的回答中注入錯誤，生成差的回答"""

    error_types = [
        "add_factual_error",    # 添加事實錯誤
        "make_incoherent",      # 破壞連貫性
        "add_harmful_content",  # 添加有害內容
        "make_too_short",       # 截短
        "add_irrelevant_info"   # 添加無關信息
    ]

    error_type = random.choice(error_types)
    bad_response = apply_error(good_response, error_type)

    return bad_response

# 示例
good = "Python 是一種高級編程語言，以其簡潔的語法和強大的庫而著稱。"
bad = inject_errors(good)
# 可能結果："Python 是一種低級編程語言..." (事實錯誤)
```

**3.3 Self-Play 法**

```python
def self_play_preference_generation(model):
    """使用自對弈生成偏好數據"""

    # 模型生成問題
    prompt = model.generate_question()

    # 生成多個回答
    responses = []
    for i in range(4):
        response = model.generate(prompt)
        responses.append(response)

    # 模型自己評判（使用更強的模型）
    evaluator = load_model("gpt-4")
    rankings = evaluator.rank(prompt, responses)

    # 生成偏好對
    return {
        "prompt": prompt,
        "chosen": responses[rankings[0]],
        "rejected": responses[rankings[-1]]
    }
```

**3.4 蒸餾法（Distillation）**

```python
def distillation_preference_generation(teacher_model, student_model, prompt):
    """從教師模型蒸餾偏好數據"""

    # 教師模型生成（好的回答）
    teacher_response = teacher_model.generate(prompt, temperature=0.7)

    # 學生模型生成（較差的回答）
    student_response = student_model.generate(prompt, temperature=0.9)

    return {
        "prompt": prompt,
        "chosen": teacher_response,  # 教師的回答作為 chosen
        "rejected": student_response  # 學生的回答作為 rejected
    }
```

**合成數據的優缺點：**

優點：
- 成本極低
- 可快速生成大量數據
- 高度可擴展

缺點：
- 質量可能不如人類標註
- 可能引入模型偏見
- 難以捕捉細微的人類偏好

**最佳實踐：**
```python
# 混合使用人類標註和合成數據
final_dataset = {
    "human_annotated": 10000,   # 10% 人類標註（核心數據）
    "synthetic": 90000          # 90% 合成數據（擴充）
}
```

### 偏好數據收集的困難點

#### 困難 1：標註不一致（Inter-Annotator Disagreement）

**問題：** 不同標註者對同一數據的判斷不一致

**解決方案：**

```python
def handle_disagreement(annotations):
    """處理標註不一致"""

    # 1. 計算 Fleiss' Kappa（多人一致性）
    kappa = calculate_fleiss_kappa(annotations)

    if kappa < 0.4:  # 一致性差
        # 2. 召集專家討論
        expert_annotation = expert_review(annotations)
        return expert_annotation

    elif kappa < 0.6:  # 一致性中等
        # 3. 多數投票
        majority_vote = get_majority_annotation(annotations)
        return majority_vote

    else:  # 一致性好
        # 4. 直接使用
        return annotations[0]

# 示例：計算一致性
from sklearn.metrics import cohen_kappa_score

annotator1 = [1, 0, 1, 1, 0]  # 1=chosen, 0=rejected
annotator2 = [1, 0, 0, 1, 0]

kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa:.2f}")  # 0.75 = 高一致性
```

**改進標註指南：**

```markdown
# 標註指南範例

## 評判標準

### 1. 準確性（最重要）
- ✅ 事實正確
- ❌ 包含錯誤信息

### 2. 有幫助性
- ✅ 直接回答問題
- ✅ 提供具體細節
- ❌ 過於籠統

### 3. 安全性
- ✅ 不包含有害內容
- ❌ 可能造成傷害

### 4. 連貫性
- ✅ 邏輯清晰
- ❌ 前後矛盾

## 困難案例處理

**兩個回答都很好？**
→ 選擇更詳細、更有幫助的

**兩個回答都不好？**
→ 選擇危害較小的

**不確定？**
→ 標記為"需要審核"
```

#### 困難 2：主觀性（Subjectivity）

**問題：** 有些偏好是主觀的（如幽默感、寫作風格）

**解決方案：**

```python
def handle_subjectivity(prompt, responses):
    """處理主觀偏好"""

    # 1. 識別主觀性類型
    subjectivity_type = classify_prompt(prompt)

    if subjectivity_type == "factual":
        # 事實性問題：有客觀答案
        return objective_evaluation(responses)

    elif subjectivity_type == "creative":
        # 創意性問題：多樣化標註
        return diverse_annotation(responses, num_annotators=5)

    elif subjectivity_type == "personal":
        # 個人偏好：記錄多種偏好
        return multi_preference_annotation(responses)
```

**多元偏好建模：**

```python
# 不強制一個"正確"答案，而是記錄偏好分佈
preference_distribution = {
    "prompt": "寫一首關於春天的詩",
    "responses": [response_a, response_b, response_c],
    "preferences": {
        "response_a": 0.5,  # 50% 標註者喜歡
        "response_b": 0.3,  # 30% 標註者喜歡
        "response_c": 0.2   # 20% 標註者喜歡
    }
}
```

#### 困難 3：偏見（Bias）

**問題：** 標註者可能帶有文化、性別、政治等偏見

**檢測偏見：**

```python
def detect_bias(dataset):
    """檢測數據集中的偏見"""

    biases = {
        "length_bias": 0,      # 偏好更長的回答
        "format_bias": 0,      # 偏好特定格式
        "politeness_bias": 0,  # 偏好更禮貌的回答
        "demographic_bias": 0  # 人口統計偏見
    }

    # 1. 長度偏見
    chosen_lengths = [len(d["chosen"]) for d in dataset]
    rejected_lengths = [len(d["rejected"]) for d in dataset]

    if mean(chosen_lengths) > mean(rejected_lengths) * 1.5:
        biases["length_bias"] = 1
        print("⚠️ 檢測到長度偏見：chosen 平均比 rejected 長 50%+")

    # 2. 格式偏見（檢查是否總是偏好列表格式）
    chosen_has_list = sum("1." in d["chosen"] or "-" in d["chosen"] for d in dataset)
    if chosen_has_list / len(dataset) > 0.8:
        biases["format_bias"] = 1
        print("⚠️ 檢測到格式偏見：80%+ chosen 包含列表")

    # 3. 禮貌性偏見
    polite_words = ["請", "謝謝", "抱歉"]
    chosen_polite = sum(any(w in d["chosen"] for w in polite_words) for d in dataset)

    if chosen_polite / len(dataset) > 0.7:
        biases["politeness_bias"] = 1
        print("⚠️ 檢測到禮貌性偏見")

    return biases
```

**減輕偏見：**

```python
def mitigate_bias(dataset):
    """減輕數據集偏見"""

    # 1. 多樣化標註團隊
    annotators = recruit_diverse_annotators(
        demographics=["age", "gender", "culture", "education"]
    )

    # 2. 平衡數據
    balanced_dataset = balance_dataset(
        dataset,
        factors=["response_length", "format", "tone"]
    )

    # 3. 去偏見訓練
    debiased_model = train_with_debiasing(
        model, balanced_dataset, bias_mitigation_loss
    )

    return debiased_model
```

#### 困難 4：規模和成本（Scale and Cost）

**問題：** 高質量偏好數據成本高，難以大規模收集

**成本分析：**

```python
# 典型成本估算
cost_per_annotation = {
    "crowdworker": 0.10,      # 眾包平台
    "专业annotator": 0.50,     # 專業標註者
    "domain_expert": 5.00      # 領域專家
}

dataset_size = 100000  # 需要 10 萬條數據

total_costs = {
    "crowdworker": dataset_size * 0.10,       # $10,000
    "professional": dataset_size * 0.50,      # $50,000
    "expert": dataset_size * 5.00             # $500,000
}
```

**降低成本策略：**

```python
def cost_effective_data_collection(budget=10000):
    """成本效益的數據收集策略"""

    # 1. 核心數據用專業標註（20%）
    core_data_size = int(budget * 0.5 / 0.50)  # 10,000 條
    core_data = professional_annotation(prompts[:core_data_size])

    # 2. 擴充數據用合成方法（60%）
    synthetic_data_size = 30000
    synthetic_data = generate_synthetic_preferences(prompts, model)

    # 3. 驗證數據用眾包（20%）
    validation_size = int(budget * 0.3 / 0.10)  # 3,000 條
    validation_data = crowdsource_annotation(prompts)

    # 4. 混合數據集
    final_dataset = core_data + synthetic_data + validation_data

    return final_dataset
```

### 常見的偏好數據集

#### 公開數據集

| 數據集 | 大小 | 語言 | 領域 | 使用 |
|-------|------|------|------|------|
| **Anthropic HH-RLHF** | 169K | EN | 通用對話 | RLHF, DPO |
| **OpenAssistant** | 161K | 多語言 | 對話助手 | SFT, RLHF |
| **WebGPT** | 20K | EN | 網絡問答 | 獎勵模型 |
| **SHP (StackExchange)** | 385K | EN | 技術問答 | 偏好學習 |
| **UltraFeedback** | 64K | EN | 指令跟隨 | DPO |
| **HelpSteer** | 37K | EN | 多維度評分 | 精細對齊 |

**使用示例：**

```python
from datasets import load_dataset

# 載入 Anthropic HH-RLHF
dataset = load_dataset("Anthropic/hh-rlhf")

# 數據格式
example = dataset["train"][0]
print(example.keys())  # ['chosen', 'rejected']

# 載入 UltraFeedback
ultra_feedback = load_dataset("openbmb/UltraFeedback")

# 包含多維度評分
example = ultra_feedback["train"][0]
print(example["score"])  # {'helpfulness': 4, 'honesty': 5, ...}
```

#### 自建數據集建議

```python
def build_custom_dataset(domain, size=10000):
    """構建特定領域的偏好數據集"""

    # 1. 收集領域特定的提示
    prompts = collect_domain_prompts(domain)

    # 2. 使用領域專家生成高質量回答
    expert_responses = expert_generation(prompts[:1000])

    # 3. 使用模型生成候選回答
    model_responses = model_generation(prompts, num_per_prompt=4)

    # 4. 領域專家標註偏好
    preferences = expert_annotation(
        prompts,
        model_responses,
        num_experts=3
    )

    # 5. 質量控制
    filtered_preferences = quality_filter(
        preferences,
        min_agreement=0.7
    )

    # 6. 數據增強
    augmented_data = augment_with_synthetic(
        filtered_preferences,
        target_size=size
    )

    return augmented_data

# 示例：構建醫療領域數據集
medical_dataset = build_custom_dataset(
    domain="medical",
    size=50000
)
```

### 偏好數據質量驗證

```python
def validate_preference_dataset(dataset):
    """全面驗證偏好數據集質量"""

    metrics = {}

    # 1. 基本統計
    metrics["size"] = len(dataset)
    metrics["avg_prompt_length"] = np.mean([len(d["prompt"]) for d in dataset])
    metrics["avg_chosen_length"] = np.mean([len(d["chosen"]) for d in dataset])
    metrics["avg_rejected_length"] = np.mean([len(d["rejected"]) for d in dataset])

    # 2. 質量檢查
    metrics["duplicates"] = count_duplicates(dataset)
    metrics["identical_pairs"] = sum(
        d["chosen"] == d["rejected"] for d in dataset
    )

    # 3. 多樣性檢查
    metrics["unique_prompts"] = len(set(d["prompt"] for d in dataset))
    metrics["diversity_score"] = calculate_diversity(dataset)

    # 4. 偏見檢測
    metrics["biases"] = detect_bias(dataset)

    # 5. 一致性檢查（如果有多標註者）
    if "annotations" in dataset[0]:
        metrics["inter_annotator_agreement"] = calculate_agreement(dataset)

    # 生成報告
    generate_quality_report(metrics)

    return metrics
```

---