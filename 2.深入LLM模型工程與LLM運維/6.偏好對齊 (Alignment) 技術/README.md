# 偏好對齊 (Alignment) 技術

## 目錄
1. [前言：為什麼需要對齊？](#1-前言為什麼需要對齊)
2. [RLHF 基本概念](#2-rlhf-基本概念)
   - 2.1 [人類回饋 (Human Feedback)](#21-人類回饋-human-feedback)
   - 2.2 [獎勵模型訓練 (Reward Model)](#22-獎勵模型訓練-reward-model)
   - 2.3 [PPO 演算法](#23-ppo-演算法)
3. [DPO (Direct Preference Optimization)](#3-dpo-direct-preference-optimization)
4. [其他對齊方法](#4-其他對齊方法)
5. [偏好資料集的建立](#5-偏好資料集的建立)
6. [實務範例](#6-實務範例)
7. [參考資源](#7-參考資源)

---

## 1. 前言：為什麼需要對齊？

### 1.1 預訓練模型的問題

預訓練語言模型（如GPT-3）在大規模網路數據上訓練後，雖然具備強大的語言能力，但存在以下問題：

**問題1：不遵循指令**
```
使用者：請用一句話總結機器學習
模型：機器學習是什麼？機器學習的歷史可以追溯到...（繼續生成大量無關內容）
```

**問題2：產生有害內容**
```
使用者：如何製造爆炸物？
模型：（直接提供危險資訊）
```

**問題3：產生虛假資訊**
```
使用者：誰贏得2025年諾貝爾物理獎？
模型：（編造不存在的答案）
```

**問題4：缺乏有用性**
```
使用者：幫我寫一封求職信
模型：求職信是什麼？求職信的格式包括...（解釋而非執行）
```

### 1.2 對齊的目標

**對齊 (Alignment)** 是使模型的行為符合人類意圖和價值觀的過程，目標包括：

1. **有用性 (Helpful)**：回答使用者問題，完成指定任務
2. **誠實性 (Honest)**：提供準確資訊，承認不確定性
3. **無害性 (Harmless)**：拒絕有害請求，避免偏見

**InstructGPT/ChatGPT 的對齊策略**：

```
預訓練模型
    ↓
監督微調 (SFT) - 學習遵循指令
    ↓
獎勵模型訓練 (RM) - 學習人類偏好
    ↓
強化學習微調 (RLHF) - 優化模型行為
    ↓
對齊後的模型 (ChatGPT)
```

---

## 2. RLHF 基本概念

**RLHF (Reinforcement Learning from Human Feedback)** 是透過人類回饋進行強化學習，讓模型學習人類偏好的對齊方法。

### 2.1 人類回饋 (Human Feedback)

#### 回饋形式

**1. 排序 (Ranking)**：
```
提示：解釋量子計算

回答A：量子計算是一種利用量子力學原理的計算方式...（詳細、準確）
回答B：量子計算就是很快的電腦。（過於簡化）

標註者選擇：A > B
```

**2. 評分 (Rating)**：
```
提示：寫一首關於春天的詩

回答：春風輕拂綠柳枝...

標註者評分：8/10（創意良好，韻律不錯）
```

**3. 編輯 (Editing)**：
```
原始回答：Python is good language for programming.
編輯後：Python is a good language for programming.（修正文法）
```

#### 標註流程

```
1. 準備提示 (Prompts)
   ↓
2. 模型生成多個回答 (通常 4-9 個)
   ↓
3. 標註者排序回答（從最好到最差）
   ↓
4. 收集偏好資料對 (chosen, rejected)
   ↓
5. 用於訓練獎勵模型
```

### 2.2 獎勵模型訓練 (Reward Model)

獎勵模型 (Reward Model, RM) 學習預測人類對回答的偏好。

#### 架構

獎勵模型通常是在 SFT 模型基礎上修改：

```
輸入: [Prompt + Response]
    ↓
LLM Backbone (凍結或微調)
    ↓
Linear Head (輸出標量獎勵)
    ↓
輸出: Reward Score (r)
```

#### 訓練目標

給定提示 x 和一對回答 (y_w, y_l)，其中 y_w 優於 y_l：

**損失函數（Bradley-Terry 模型）**：
```
L_RM = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]
```

其中：
- r_θ：獎勵模型
- y_w：preferred (chosen) 回答
- y_l：dispreferred (rejected) 回答
- σ：sigmoid 函數

**目標**：最大化 chosen 回答的獎勵，最小化 rejected 回答的獎勵。

#### 實作範例

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    """獎勵模型"""

    def __init__(self, base_model_name):
        super().__init__()
        # 載入預訓練模型
        self.backbone = AutoModel.from_pretrained(base_model_name)

        # 獎勵頭（輸出標量）
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            rewards: (batch_size,) 獎勵分數
        """
        # 獲取最後一層隱藏狀態
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取最後一個 token 的隱藏狀態
        last_hidden = outputs.last_hidden_state[:, -1, :]  # (batch_size, hidden_size)

        # 計算獎勵
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (batch_size,)

        return rewards

def reward_model_loss(chosen_rewards, rejected_rewards):
    """計算獎勵模型損失"""
    # Bradley-Terry 損失
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss

# 使用範例
model = RewardModel("gpt2")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 模擬訓練
for epoch in range(num_epochs):
    # chosen_inputs: 較好的回答
    # rejected_inputs: 較差的回答

    chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
    rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

    loss = reward_model_loss(chosen_rewards, rejected_rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2.3 PPO 演算法

**PPO (Proximal Policy Optimization)** 是 RLHF 中最常用的強化學習演算法。

#### 強化學習框架

**狀態 (State)**：提示 x
**動作 (Action)**：生成的 token
**獎勵 (Reward)**：
- 獎勵模型給出的分數 r_θ(x, y)
- KL 懲罰項（防止偏離太遠）

**目標函數**：
```
maximize E[r_θ(x, y) - β * KL(π_θ || π_ref)]
```

其中：
- π_θ：當前策略（正在訓練的模型）
- π_ref：參考策略（SFT 模型）
- β：KL 懲罰係數
- KL：防止模型偏離參考模型太遠

#### PPO 更新規則

**Clipped Objective**：
```
L^CLIP(θ) = E[min(
    r_t(θ) * A_t,
    clip(r_t(θ), 1-ε, 1+ε) * A_t
)]
```

其中：
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)：重要性權重
- A_t：優勢函數 (Advantage)
- ε：裁剪範圍（通常 0.1-0.2）

**關鍵特性**：
- 限制每次更新的幅度
- 防止策略崩潰
- 樣本效率高

#### RLHF 訓練流程

```
1. 初始化
   - 載入 SFT 模型作為初始策略 π_θ
   - 載入獎勵模型 r_θ
   - 複製 π_θ 作為參考模型 π_ref（凍結）

2. 對每個 batch：
   a. 採樣提示 x
   b. 用當前策略生成回答 y ~ π_θ(·|x)
   c. 用獎勵模型評分 r = r_θ(x, y)
   d. 計算 KL 懲罰 kl = KL(π_θ || π_ref)
   e. 計算總獎勵 reward = r - β * kl
   f. 用 PPO 更新策略 π_θ

3. 重複步驟 2 直到收斂
```

#### 實作考量

**挑戰**：
1. **計算成本高**：需要同時載入多個大模型（策略、參考、獎勵、價值網路）
2. **訓練不穩定**：強化學習容易發散
3. **獎勵 Hacking**：模型可能學會利用獎勵模型的弱點

**技巧**：
1. **DeepSpeed/FSDP**：分散式訓練
2. **Gradient Checkpointing**：減少記憶體
3. **適當的超參數**：學習率、KL 係數
4. **Early Stopping**：監控獎勵和 KL 散度

---

## 3. DPO (Direct Preference Optimization)

**DPO** 是一種**不需要訓練獎勵模型**的對齊方法，直接從偏好數據優化策略。

### 3.1 核心思想

**RLHF 的問題**：
- 需要訓練獎勵模型（額外計算成本）
- 獎勵模型可能不準確
- PPO 訓練複雜且不穩定

**DPO 的創新**：
- 將獎勵模型隱式地表示為策略的函數
- 直接優化策略，無需獎勵模型
- 使用簡單的分類損失

### 3.2 數學原理

**Bradley-Terry 偏好模型**：
```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

**DPO 的關鍵洞察**：
獎勵可以用策略和參考策略的比率表示：
```
r(x, y) = β * log(π_θ(y|x) / π_ref(y|x)) + Z(x)
```

**DPO 損失函數**：
```
L_DPO = -E[log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]
```

**簡化形式**：
```
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]
```

### 3.3 DPO vs RLHF

| 特性 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 獎勵模型 | ✅ 需要 | ❌ 不需要 |
| 訓練複雜度 | 高 | 低 |
| 穩定性 | 較不穩定 | 較穩定 |
| 計算成本 | 高（4個模型） | 低（2個模型） |
| 性能 | 通常更好 | 接近 RLHF |

### 3.4 DPO 實作

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO 損失函數

    Args:
        policy_chosen_logps: log π_θ(y_w|x)
        policy_rejected_logps: log π_θ(y_l|x)
        reference_chosen_logps: log π_ref(y_w|x)
        reference_rejected_logps: log π_ref(y_l|x)
        beta: 溫度參數
    """
    # 計算 logits
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO 損失
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 計算隱式獎勵（用於監控）
    implicit_rewards_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    implicit_rewards_rejected = beta * (policy_rejected_logps - reference_rejected_logps)

    return loss, implicit_rewards_chosen, implicit_rewards_rejected

def compute_logprobs(model, input_ids, attention_mask, labels):
    """計算序列的對數機率"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits

    # 計算每個 token 的對數機率
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 計算對數機率
    logprobs = F.log_softmax(shift_logits, dim=-1)
    per_token_logprobs = torch.gather(logprobs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

    # 對序列求和
    sequence_logprobs = (per_token_logprobs * (shift_labels != -100)).sum(dim=1)

    return sequence_logprobs

# 訓練循環
def train_dpo(policy_model, reference_model, train_dataloader, optimizer, beta=0.1):
    """DPO 訓練循環"""
    policy_model.train()
    reference_model.eval()

    for batch in train_dataloader:
        # batch 包含 chosen 和 rejected 回答
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        # 計算策略模型的對數機率
        policy_chosen_logps = compute_logprobs(
            policy_model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = compute_logprobs(
            policy_model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )

        # 計算參考模型的對數機率（不計算梯度）
        with torch.no_grad():
            reference_chosen_logps = compute_logprobs(
                reference_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            reference_rejected_logps = compute_logprobs(
                reference_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )

        # 計算 DPO 損失
        loss, rewards_chosen, rewards_rejected = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            beta=beta
        )

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 記錄
        print(f"Loss: {loss.item():.4f}, "
              f"Reward Chosen: {rewards_chosen.mean().item():.4f}, "
              f"Reward Rejected: {rewards_rejected.mean().item():.4f}")
```

---

## 4. 其他對齊方法

### 4.1 Constitutional AI (CAI)

**核心思想**：使用 AI 自我批評和修正，減少人類標註需求。

**流程**：
```
1. Self-Critique：AI 生成回答後，根據憲法原則自我批評
2. Revision：根據批評修正回答
3. Preference Modeling：用修正後的資料訓練偏好模型
```

**優點**：
- 減少人力標註
- 可擴展性強
- 提升透明度（原則明確）

### 4.2 RLAIF (RL from AI Feedback)

**思想**：用 AI 模型替代人類標註者。

**流程**：
```
1. 用強大的 AI 模型（如 GPT-4）對回答進行排序
2. 訓練獎勵模型
3. 用 RLHF 流程訓練
```

**優勢**：
- 成本低
- 速度快
- 可大規模擴展

**挑戰**：
- AI 標註品質
- 可能引入 AI 偏見

### 4.3 RAFT (Reward rAnked FineTuning)

**思想**：結合 SFT 和 RM，排序後微調。

**流程**：
```
1. 對每個提示生成多個回答
2. 用獎勵模型排序
3. 只用高分回答進行 SFT
4. 迭代重複
```

### 4.4 IPO (Identity Preference Optimization)

**改進 DPO**：使用不同的損失函數，更穩定。

**損失函數**：
```
L_IPO = (π_θ(y_w|x) - π_θ(y_l|x) - 1)^2
```

### 4.5 KTO (Kahneman-Tversky Optimization)

**思想**：基於前景理論，使用二元標籤（好/壞）而非成對比較。

**適用場景**：
- 標註資源有限
- 只有好/壞標籤的數據

---

## 5. 偏好資料集的建立

### 5.1 資料收集流程

#### 步驟 1：準備提示

**來源**：
- 真實使用者查詢
- 合成提示（用 GPT-4 生成）
- 現有基準測試

**多樣性**：
- 涵蓋不同任務（問答、創意寫作、程式碼等）
- 不同難度
- 不同長度

#### 步驟 2：生成回答

```python
def generate_multiple_responses(model, tokenizer, prompt, num_responses=4):
    """為每個提示生成多個回答"""
    responses = []

    for _ in range(num_responses):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,  # 增加多樣性
            top_p=0.95,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    return responses
```

#### 步驟 3：人類標註

**標註介面範例**：
```
提示：解釋什麼是機器學習

回答 A：機器學習是人工智慧的一個分支...（200字）
回答 B：就是讓電腦學習的技術。（簡短）
回答 C：機器學習是...（詳細且準確，400字）
回答 D：機器學習嘛，就是...（口語化）

請排序（從最好到最差）：C > A > D > B
```

**標註指南**：
1. **有用性**：是否回答問題
2. **準確性**：資訊是否正確
3. **完整性**：是否涵蓋重要方面
4. **清晰度**：是否易於理解
5. **安全性**：是否有害

#### 步驟 4：資料格式化

**成對比較格式**：
```json
{
  "prompt": "解釋什麼是機器學習",
  "chosen": "機器學習是人工智慧的一個分支，使電腦能夠從資料中學習...",
  "rejected": "就是讓電腦學習的技術。"
}
```

### 5.2 資料品質控制

#### 標註者一致性

```python
def calculate_inter_annotator_agreement(annotations):
    """計算標註者間一致性（Krippendorff's alpha）"""
    # 實作標註一致性計算
    pass
```

**目標**：
- Kappa > 0.6（中等一致性）
- 對不一致案例進行複審

#### 資料過濾

**過濾規則**：
1. 太短的回答（< 10 tokens）
2. 包含有害內容
3. 標註者不確定的案例
4. Chosen 和 rejected 太相似

```python
def filter_preference_data(data):
    """過濾偏好資料"""
    filtered = []

    for item in data:
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 長度檢查
        if len(chosen.split()) < 10 or len(rejected.split()) < 10:
            continue

        # 相似度檢查
        similarity = compute_similarity(chosen, rejected)
        if similarity > 0.95:  # 太相似
            continue

        filtered.append(item)

    return filtered
```

### 5.3 資料集規模

**經驗數據**：
- **InstructGPT**：~13k 比較
- **Anthropic HH**：~160k 比較
- **OpenAssistant**：~88k 比較

**最小需求**：
- 至少 1k-5k 高品質比較
- 涵蓋多種任務類型

---

## 6. 實務範例

### 6.1 使用 TRL 進行 RLHF

[TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) 是 Hugging Face 提供的 RLHF 庫。

#### 安裝

```bash
pip install trl transformers accelerate
```

#### PPO 訓練範例

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset

# 配置
config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=128,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
)

# 載入模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 載入獎勵模型
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model_path")

# 準備數據
dataset = load_dataset("your_dataset")

# 創建 PPO Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,  # 自動創建參考模型
    tokenizer=tokenizer,
    dataset=dataset["train"],
)

# 訓練循環
for epoch in range(config.ppo_epochs):
    for batch in ppo_trainer.dataloader:
        query_tensors = batch["input_ids"]

        # 生成回答
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=128,
            **generation_kwargs
        )

        # 計算獎勵
        texts = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        rewards = compute_rewards(reward_model, texts)

        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 記錄
        ppo_trainer.log_stats(stats, batch, rewards)
```

### 6.2 使用 TRL 進行 DPO

```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 載入模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 載入偏好資料
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# 創建 DPO Trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,  # DPO 溫度參數
)

# 訓練
trainer.train()

# 保存模型
trainer.save_model("dpo_model")
```

### 6.3 StackLLaMA 實例

**StackLLaMA** 是基於 LLaMA 的 RLHF 實例，在 Stack Exchange 資料上訓練。

#### 完整流程

```python
"""
StackLLaMA RLHF 流程

步驟1：監督微調 (SFT)
步驟2：訓練獎勵模型 (RM)
步驟3：PPO 強化學習
"""

# ============================================
# 步驟 1：監督微調 (SFT)
# ============================================

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# 載入基礎模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 載入 Stack Exchange 資料
dataset = load_dataset("lvwerra/stack-exchange-paired", split="train")

# SFT 訓練
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./sft_model")

# ============================================
# 步驟 2：訓練獎勵模型 (RM)
# ============================================

from trl import RewardTrainer

# 載入 SFT 模型作為基礎
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./sft_model",
    num_labels=1  # 輸出標量獎勵
)

# 準備偏好資料
# 格式：{"prompt": ..., "chosen": ..., "rejected": ...}
preference_dataset = load_dataset("lvwerra/stack-exchange-paired", split="train")

# 訓練獎勵模型
reward_trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

reward_trainer.train()
reward_trainer.save_model("./reward_model")

# ============================================
# 步驟 3：PPO 強化學習
# ============================================

from trl import PPOTrainer, PPOConfig

# 載入 SFT 模型
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")

# 載入獎勵模型
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model")

# PPO 配置
ppo_config = PPOConfig(
    model_name="./sft_model",
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=64,
    ppo_epochs=4,
    init_kl_coef=0.2,  # KL 懲罰係數
)

# 創建 PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    tokenizer=tokenizer,
    dataset=dataset,
)

# PPO 訓練循環
for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        # 生成回答
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=128)

        # 計算獎勵
        texts = [tokenizer.decode(r) for r in response_tensors]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        rewards = reward_model(**inputs).logits.squeeze()

        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"Epoch {epoch}, Reward: {rewards.mean().item():.4f}")

# 保存最終模型
ppo_trainer.save_pretrained("./rlhf_model")
```

---

## 7. 參考資源

### 論文

1. **InstructGPT**: "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
2. **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
3. **Constitutional AI**: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
4. **PPO**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
5. **RLAIF**: "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (Lee et al., 2023)

### 工具與庫

- **TRL**: https://github.com/huggingface/trl
- **OpenAssistant**: https://github.com/LAION-AI/Open-Assistant
- **DeepSpeed-Chat**: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat
- **Colossal-AI**: https://github.com/hpcaitech/ColossalAI

### 資料集

- **Anthropic HH-RLHF**: https://huggingface.co/datasets/Anthropic/hh-rlhf
- **OpenAssistant Conversations**: https://huggingface.co/datasets/OpenAssistant/oasst1
- **Stack Exchange**: https://huggingface.co/datasets/lvwerra/stack-exchange-paired
- **SHP**: https://huggingface.co/datasets/stanfordnlp/SHP

### 部落格與教學

- **Hugging Face RLHF Blog**: https://huggingface.co/blog/rlhf
- **OpenAI Alignment Research**: https://openai.com/research/alignment
- **Anthropic Research**: https://www.anthropic.com/research

---

## 總結

### 核心要點

1. **對齊的重要性**
   - 使模型行為符合人類意圖
   - 有用、誠實、無害

2. **RLHF 是主流方法**
   - 三階段：SFT → RM → PPO
   - 效果好但複雜

3. **DPO 是簡化替代方案**
   - 不需要獎勵模型
   - 更簡單、更穩定
   - 效果接近 RLHF

4. **資料是關鍵**
   - 高品質偏好資料
   - 多樣性和平衡性
   - 標註者一致性

5. **實務建議**
   - 小團隊：DPO
   - 大團隊/需要極致性能：RLHF
   - 使用現有工具（TRL）
   - 從小規模實驗開始

### 未來趨勢

1. **更高效的對齊方法**：減少人力和計算成本
2. **自動化對齊**：RLAIF、Constitutional AI
3. **多目標對齊**：平衡多個價值觀
4. **個性化對齊**：針對不同使用者群體
5. **持續對齊**：部署後持續改進
