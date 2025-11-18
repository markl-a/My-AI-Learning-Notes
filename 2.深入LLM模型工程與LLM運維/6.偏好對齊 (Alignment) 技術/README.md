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