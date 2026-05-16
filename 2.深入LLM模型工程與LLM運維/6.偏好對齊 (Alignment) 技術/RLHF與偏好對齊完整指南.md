# RLHF 與偏好對齊完整指南

## 目錄

1. [偏好對齊概述](#1-偏好對齊概述)
2. [RLHF (Reinforcement Learning from Human Feedback)](#2-rlhf-reinforcement-learning-from-human-feedback)
3. [DPO (Direct Preference Optimization)](#3-dpo-direct-preference-optimization)
4. [其他對齊方法](#4-其他對齊方法)
5. [偏好資料集建立](#5-偏好資料集建立)
6. [實作範例](#6-實作範例)
7. [實務挑戰與解決方案](#7-實務挑戰與解決方案)

---

## 1. 偏好對齊概述

### 1.1 為什麼需要對齊？

預訓練的語言模型雖然具備強大的能力，但存在以下問題：

- **不遵循指令**：可能產生與用戶意圖不符的輸出
- **有害內容**：可能生成暴力、歧視、不道德的內容
- **幻覺問題**：編造不存在的事實
- **缺乏幫助性**：回答可能正確但不實用
- **不一致性**：對相似問題給出矛盾答案

### 1.2 對齊的三個維度 (HHH)

**Helpful (有用性)**：
- 理解並完成用戶的任務
- 提供有價值的資訊
- 承認不確定性而非編造

**Honest (誠實性)**：
- 準確反映訓練資料和能力範圍
- 不誇大或虛假陳述
- 區分事實與觀點

**Harmless (無害性)**：
- 避免生成有害內容
- 拒絕不當請求
- 考慮社會影響

### 1.3 對齊技術演進

```
階段 1: 監督微調 (SFT)
├─ 使用高品質示範資料
└─ 學習基本的指令遵循能力

階段 2: 人類反饋強化學習 (RLHF)
├─ 訓練獎勵模型捕捉人類偏好
├─ 使用 RL 優化模型
└─ InstructGPT, ChatGPT 的核心技術

階段 3: 直接偏好優化 (DPO/IPO/KTO)
├─ 無需獎勵模型和 RL
├─ 直接從偏好資料學習
└─ 訓練更簡單、穩定

階段 4: 憲法式 AI (Constitutional AI)
├─ 使用 AI 自我批評和修正
├─ 減少人類標註需求
└─ Claude 系列使用
```

---

## 2. RLHF (Reinforcement Learning from Human Feedback)

### 2.1 RLHF 三階段流程

#### 階段 1：監督微調 (SFT)

**目標**：訓練基礎模型遵循指令

**資料格式**：
```json
{
  "prompt": "解釋什麼是量子糾纏",
  "completion": "量子糾纏是一種量子力學現象，指兩個或多個粒子..."
}
```

**訓練**：
```python
# 標準的監督學習
loss = CrossEntropy(model_output, target_completion)
```

**資料量**：通常需要 10K-100K 高品質示範

#### 階段 2：訓練獎勵模型 (Reward Model)

**目標**：學習人類偏好函數

**資料格式（偏好對）**：
```json
{
  "prompt": "如何學習深度學習？",
  "chosen": "建議從基礎數學開始，然後學習 Python...",
  "rejected": "深度學習很簡單，直接用框架就行。"
}
```

**模型架構**：
```
Base LLM → Remove LM Head → Add Scalar Head → Reward Score
```

**訓練目標（Ranking Loss）**：
```python
# 最大化 chosen 和 rejected 的分數差距
loss = -log(sigmoid(r_chosen - r_rejected))
```

其中：
- `r_chosen`：偏好回答的獎勵分數
- `r_rejected`：非偏好回答的獎勵分數

**資料量**：通常需要 50K-500K 偏好對

#### 階段 3：強化學習優化 (RL Fine-tuning)

**目標**：使用 PPO 最大化獎勵模型的分數

**優化目標**：
```
max E[r(x, y)] - β * KL(π_θ || π_ref)
```

其中：
- `r(x, y)`：獎勵模型給的分數
- `π_θ`：當前策略（正在訓練的模型）
- `π_ref`：參考策略（SFT 模型）
- `β`：KL 散度係數（防止偏離太遠）

### 2.2 PPO (Proximal Policy Optimization) 演算法

**核心思想**：限制每次更新的策略變化幅度

**PPO-Clip 目標函式**：
```python
L_CLIP(θ) = E[min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)]

其中:
ratio = π_θ(a|s) / π_old(a|s)  # 新舊策略比率
advantage = r - baseline        # 優勢函數
ε = 0.2                         # 裁剪參數
```

**PPO 訓練循環**：
```
1. 使用當前策略生成 rollout (prompt → response)
2. 用獎勵模型評分
3. 計算優勢函數
4. 多次 mini-batch 更新（重要：使用裁剪防止過大更新）
5. 重複 1-4
```

### 2.3 RLHF 的挑戰

**1. 訓練不穩定**
- RL 訓練容易崩潰
- 獎勵 hacking（模型找到獎勵模型的漏洞）
- KL 散度難以平衡

**2. 計算成本高**
- 需要同時運行 4 個模型：
  - Policy model（正在訓練）
  - Reference model（KL 參考）
  - Reward model（評分）
  - Value model（PPO baseline）

**3. 獎勵模型局限**
- 可能過擬合標註資料
- 難以泛化到 OOD 情況
- 人類偏好的複雜性難以完全捕捉

---

## 3. DPO (Direct Preference Optimization)

### 3.1 核心思想

**問題**：RLHF 太複雜，需要獎勵模型和 RL

**DPO 的洞察**：偏好學習可以直接優化，無需中間的獎勵模型

### 3.2 數學推導

**RLHF 的最優策略**：
```
π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)
```

**反推獎勵函數**：
```
r(x,y) = β * log(π*(y|x) / π_ref(y|x))
```

**Bradley-Terry 偏好模型**：
```
P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
```

**代入獎勵函數得到 DPO 損失**：
```
L_DPO = -E[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x))
                  - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

其中：
- `y_w`：chosen (偏好的回答)
- `y_l`：rejected (不偏好的回答)
- `β`：溫度參數（控制偏離參考模型的程度）

### 3.3 DPO vs RLHF

| 方面 | RLHF | DPO |
|------|------|-----|
| **訓練階段** | 3 階段（SFT → RM → RL） | 2 階段（SFT → DPO） |
| **模型數量** | 4 個（policy, ref, reward, value） | 2 個（policy, ref） |
| **訓練穩定性** | 較低（RL 不穩定） | 較高（監督學習） |
| **計算成本** | 高 | 中等 |
| **超參數調優** | 困難（RL 超參數多） | 較簡單 |
| **性能** | 略優（理論上限更高） | 接近 RLHF |

### 3.4 DPO 優缺點

**優點**：
- ✅ 訓練簡單，穩定性高
- ✅ 無需獎勵模型，節省計算
- ✅ 更容易調優
- ✅ 可以復用 SFT 基礎設施

**缺點**：
- ❌ 理論性能上限可能略低於 RLHF
- ❌ 對偏好資料品質敏感
- ❌ β 參數需要仔細調整

---

## 4. 其他對齊方法

### 4.1 IPO (Identity Preference Optimization)

**改進點**：使用 MSE 損失代替 log-sigmoid

**損失函式**：
```python
L_IPO = E[(r(x,y_w) - r(x,y_l) - 1)^2]
```

**優勢**：
- 梯度更穩定
- 不會過度懲罰已經分離的樣本

### 4.2 KTO (Kahneman-Tversky Optimization)

**核心思想**：不需要成對偏好，只需要 thumbs up/down

**資料格式**：
```json
{
  "prompt": "...",
  "completion": "...",
  "label": "good"  // 或 "bad"
}
```

**優勢**：
- 標註成本更低（不需要對比）
- 適合已有的隱式反饋資料（點讚/點踩）

### 4.3 RRHF (Rank Responses to align Human Feedback)

**改進點**：處理多個候選回答的排名

**損失函式**：
```python
# 對所有排名對計算損失
L = Σ -log σ(r(y_i) - r(y_j))  # 對所有 i > j
```

### 4.4 Constitutional AI (Claude 的方法)

**階段 1：AI 生成的批評和修訂（RL 前）**
```
1. 模型生成初始回答
2. 模型根據"憲法"自我批評
3. 模型生成改進版本
4. 微調模型學習改進後的回答
```

**階段 2：AI 反饋 RL**
```
1. 模型生成多個候選回答
2. AI 評判器根據"憲法"排名
3. 使用 AI 偏好進行 RL（而非人類偏好）
```

**優勢**：
- 減少人類標註需求
- 更容易擴展
- 可以明確編碼價值觀

---

## 5. 偏好資料集建立

### 5.1 資料收集方法

#### 方法 1：人類對比標註

**流程**：
```
1. 給定 prompt
2. 模型生成 2-4 個候選回答
3. 人類標註者選擇最佳回答
4. （可選）提供選擇理由
```

**標註指南範例**：
```markdown
請選擇更好的回答，考慮：
✓ 準確性：事實正確
✓ 有用性：真正幫助到用戶
✓ 無害性：沒有偏見、歧視、危險內容
✓ 連貫性：邏輯清晰，語言流暢
```

#### 方法 2：眾包評分

使用 Likert 量表（1-5分）評估：
```
1 = 非常差
2 = 差
3 = 一般
4 = 好
5 = 非常好
```

轉換為偏好對：選擇分數差 ≥ 2 的對

#### 方法 3：隱式反饋

從用戶行為推斷偏好：
```
正面信號：
- 點讚 / thumbs up
- 複制回答
- 繼續對話
- 分享

負面信號：
- 點踩 / thumbs down
- 立即重新生成
- 放棄對話
- 舉報
```

#### 方法 4：AI 輔助標註

**流程**：
```
1. 使用強大的 LLM（如 GPT-4）評估回答
2. 人類審核 AI 標註的子集
3. 對於高置信度的，直接使用 AI 標註
4. 對於低置信度的，人類標註
```

### 5.2 資料品質控制

**1. 標註者間一致性**
```python
# Cohen's Kappa 係數
κ = (P_observed - P_expected) / (1 - P_expected)

# κ > 0.6 可接受
# κ > 0.8 高品質
```

**2. 多數投票**
```
每個樣本由 3-5 個標註者評估
使用多數投票決定最終標籤
```

**3. 專家審核**
```
隨機抽取 5-10% 樣本
由領域專家審核
發現系統性問題
```

**4. 困難樣本識別**
```
標註者意見分歧大的樣本
可能是：
- 問題定義不清
- 兩個回答都好/都差
- 需要領域知識

處理方式：
- 重新標註
- 排除
- 加入標註指南
```

### 5.3 資料平衡與採樣

**類別平衡**：
```python
# 確保不同類型任務的平衡
categories = {
    'QA': 0.3,
    'Creative Writing': 0.2,
    'Code': 0.2,
    'Math': 0.15,
    'Other': 0.15
}
```

**難度分佈**：
```
簡單樣本：30%（明顯的偏好）
中等樣本：50%（需要判斷）
困難樣本：20%（微妙的差異）
```

**長度多樣性**：
```
短回答 (<50 tokens): 25%
中等回答 (50-200 tokens): 50%
長回答 (>200 tokens): 25%
```

### 5.4 開源偏好資料集

**英文**：
```
- Anthropic HH-RLHF: 170K 偏好對
  https://huggingface.co/datasets/Anthropic/hh-rlhf

- OpenAssistant Conversations: 161K 對話
  https://huggingface.co/datasets/OpenAssistant/oasst1

- UltraFeedback: 64K 偏好對（GPT-4 評分）
  https://huggingface.co/datasets/openbmb/UltraFeedback

- Helpsteer: 37K 評分（多維度）
  https://huggingface.co/datasets/nvidia/HelpSteer
```

**中文**：
```
- BELLE Eval Set: 中文偏好資料
- CValues: 中文價值對齊資料
- Chinese HH-RLHF: 翻譯的 Anthropic 資料
```

---

## 6. 實作範例

### 6.1 獎勵模型訓練（RLHF 階段 2）

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class RewardModel(nn.Module):
    """獎勵模型：輸入文字，輸出標量分數"""

    def __init__(self, base_model_name):
        super().__init__()
        # 載入預訓練模型（通常是 SFT 後的模型）
        self.backbone = AutoModel.from_pretrained(base_model_name)

        # 獲取隱藏層維度
        hidden_size = self.backbone.config.hidden_size

        # 添加獎勵頭（輸出標量）
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # 獲取最後一層隱藏狀態
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取最後一個 token 的表示（或使用 pooling）
        last_hidden = outputs.last_hidden_state[:, -1, :]

        # 計算獎勵分數
        reward = self.reward_head(last_hidden)

        return reward.squeeze(-1)  # (batch_size,)


class PreferenceDataset(Dataset):
    """偏好對資料集"""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 組合 prompt + completion
        chosen_text = item['prompt'] + item['chosen']
        rejected_text = item['prompt'] + item['rejected']

        # Tokenize
        chosen = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        rejected = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected['attention_mask'].squeeze(0)
        }


def train_reward_model(model, train_loader, epochs=3, lr=1e-5):
    """訓練獎勵模型"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0

        for batch in train_loader:
            # 移到設備
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)

            # 前向傳播
            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            # Ranking Loss
            loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 統計
            total_loss += loss.item()
            accuracy = (reward_chosen > reward_rejected).float().mean()
            total_acc += accuracy.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.4f}")

    return model


# 使用範例
if __name__ == "__main__":
    # 準備資料（示例）
    preference_data = [
        {
            'prompt': '如何學習深度學習？',
            'chosen': '建議先打好數學基礎，包括線性代數、微積分和概率論。然後學習 Python 和 PyTorch，從簡單的神經網絡開始實作...',
            'rejected': '深度學習很簡單，直接用框架就行了。'
        },
        # ... 更多資料
    ]

    # 初始化
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = PreferenceDataset(preference_data, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 訓練獎勵模型
    reward_model = RewardModel('gpt2')
    trained_rm = train_reward_model(reward_model, train_loader)

    # 保存
    torch.save(trained_rm.state_dict(), 'reward_model.pth')
```

### 6.2 DPO 訓練實作

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

class DPOTrainer:
    """DPO 訓練器"""

    def __init__(self, model_name, beta=0.1, lr=1e-6):
        self.beta = beta

        # 載入策略模型（將被訓練）
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name)

        # 載入參考模型（凍結）
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)

        # 移到 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_model = self.policy_model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)

    def compute_log_probs(self, model, input_ids, labels):
        """計算 log 概率"""
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits

        # 計算每個 token 的 log prob
        log_probs = F.log_softmax(logits, dim=-1)

        # 收集對應 label 的 log prob
        labels_log_probs = torch.gather(
            log_probs[:, :-1, :],  # 去掉最後一個 logit
            dim=2,
            index=labels[:, 1:].unsqueeze(-1)  # 去掉第一個 label (prompt 部分)
        ).squeeze(-1)

        # 只計算 response 部分（需要 mask）
        return labels_log_probs

    def dpo_loss(self, batch):
        """計算 DPO 損失"""
        # 解包 batch
        chosen_ids = batch['chosen_input_ids'].to(self.device)
        rejected_ids = batch['rejected_input_ids'].to(self.device)
        chosen_labels = batch['chosen_labels'].to(self.device)
        rejected_labels = batch['rejected_labels'].to(self.device)

        # 計算 policy model 的 log probs
        policy_chosen_logps = self.compute_log_probs(
            self.policy_model, chosen_ids, chosen_labels
        ).sum(-1)

        policy_rejected_logps = self.compute_log_probs(
            self.policy_model, rejected_ids, rejected_labels
        ).sum(-1)

        # 計算 reference model 的 log probs
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_labels
            ).sum(-1)

            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_labels
            ).sum(-1)

        # DPO 損失
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps

        logits = self.beta * (policy_ratio - ref_ratio)
        loss = -F.logsigmoid(logits).mean()

        # 計算準確率（用於監控）
        accuracy = (logits > 0).float().mean()

        return loss, accuracy

    def train(self, train_loader, epochs=3):
        """訓練循環"""
        self.policy_model.train()

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0

            for batch_idx, batch in enumerate(train_loader):
                # 計算損失
                loss, acc = self.dpo_loss(batch)

                # 反向傳播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 記錄
                total_loss += loss.item()
                total_acc += acc.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                          f"Loss={loss.item():.4f}, Acc={acc.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg Accuracy: {avg_acc:.4f}\n")

        return self.policy_model


# 使用範例
if __name__ == "__main__":
    # 初始化 DPO trainer
    trainer = DPOTrainer(
        model_name='gpt2',
        beta=0.1,  # DPO 超參數
        lr=1e-6
    )

    # 準備資料（需要實現 PreferenceDataset）
    # train_loader = ...

    # 訓練
    # trained_model = trainer.train(train_loader, epochs=3)

    # 保存
    # trained_model.save_pretrained('./dpo_model')
```

### 6.3 使用 TRL 庫進行 RLHF/DPO

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 載入模型
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 準備偏好資料集
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# DPO 訓練配置
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    logging_steps=10,
    save_steps=100,
    beta=0.1,  # DPO β 參數
    max_prompt_length=512,
    max_length=1024,
)

# 建立 trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 會自動建立參考模型
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 訓練
dpo_trainer.train()

# 保存
dpo_trainer.save_model("./final_dpo_model")
```

---

## 7. 實務挑戰與解決方案

### 7.1 獎勵 Hacking

**問題**：模型找到獎勵模型的漏洞，獲得高分但輸出無意義

**範例**：
```
模型發現：重複某個詞會獲得高獎勵
輸出："非常非常非常非常非常好的答案！"
```

**解決方案**：

1. **KL 散度懲罰**
```python
# 限制偏離參考模型的程度
penalty = β * KL(π_θ || π_ref)
```

2. **獎勵模型集成**
```python
# 使用多個獎勵模型的平均
reward = mean([rm1(x,y), rm2(x,y), rm3(x,y)])
```

3. **規則約束**
```python
# 硬性規則過濾
if len(set(response.split())) / len(response.split()) < 0.5:
    reward = -1  # 重複詞太多
```

### 7.2 訓練不穩定

**問題**：RL 訓練損失震盪，模型崩潰

**解決方案**：

1. **降低學習率**
```python
# RL 階段使用極小的學習率
lr = 1e-6  # 比 SFT 小 10-100 倍
```

2. **梯度裁剪**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **使用 DPO 替代 RLHF**
```
DPO 是監督學習，訓練更穩定
```

### 7.3 資料品質問題

**問題**：標註者意見不一致，資料有噪聲

**解決方案**：

1. **多標註者投票**
```python
# 需要至少 2/3 標註者同意
if agree_count / total_annotators >= 0.67:
    use_this_sample = True
```

2. **困難樣本處理**
```python
# 標註者分歧大的樣本
if variance(annotator_choices) > threshold:
    # 選項1：排除
    # 選項2：專家重新標註
    # 選項3：降低權重
```

3. **主動學習**
```python
# 優先標註對模型最有幫助的樣本
uncertainty = model_uncertainty(sample)
prioritize_samples_by(uncertainty, top_k=100)
```

### 7.4 計算資源限制

**問題**：RLHF 需要 4 個大模型同時運行

**解決方案**：

1. **使用 DPO**
```
只需要 2 個模型（policy + reference）
```

2. **參數高效方法**
```python
# 只訓練 LoRA 適配器
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)
```

3. **離線 RL**
```python
# 預先生成 rollout，不需要在線生成
# 減少推理開銷
```

### 7.5 過度優化

**問題**：模型在獎勵模型上過度優化，泛化能力下降

**指標**：
```
Reward model score ↑↑↑
但 Human evaluation ↓↓↓
```

**解決方案**：

1. **Early stopping**
```python
# 監控驗證集上的人類評估分數
if human_eval_score_decrease_for_3_epochs:
    stop_training()
```

2. **增加 KL 懲罰**
```python
# 更大的 β 限制偏離
β = 0.1 → 0.2
```

3. **定期人類評估**
```python
every_n_steps = 100
if step % every_n_steps == 0:
    human_eval_score = get_human_feedback(model)
    log_metric('human_eval', human_eval_score)
```

---

## 8. 最佳實踐與建議

### 8.1 選擇對齊方法

**小團隊 / 資源有限**：
```
推薦：DPO
原因：
- 訓練簡單
- 計算成本低
- 效果接近 RLHF
```

**大團隊 / 充足資源**：
```
推薦：RLHF (PPO)
原因：
- 理論性能上限更高
- 更靈活（可在線調整獎勵）
- 工業界驗證（GPT-4, Claude）
```

**已有隱式反饋資料**：
```
推薦：KTO
原因：
- 不需要成對比較
- 可利用已有的點讚/點踩資料
```

### 8.2 資料建議

**資料量**：
```
SFT: 10K-100K 高品質示範
Preference: 50K-500K 偏好對
最小可行: SFT 1K + Preference 10K
```

**資料多樣性**：
```
✓ 涵蓋不同任務類型
✓ 包含不同難度級別
✓ 平衡不同長度的回答
✓ 包含正面和負面範例
```

### 8.3 超參數建議

**DPO**：
```python
beta = 0.1        # 起始值，可調 0.05-0.5
lr = 5e-7         # 非常小的學習率
epochs = 1-3      # 通常 1-2 epochs 足夠
batch_size = 4-8  # 取決於 GPU 內存
```

**RLHF (PPO)**：
```python
# Policy model
lr = 1e-6                 # 極小學習率
kl_coef = 0.1            # KL 散度係數
clip_range = 0.2         # PPO 裁剪範圍

# Reward model
rm_lr = 1e-5             # 稍大的學習率
rm_epochs = 3            # 充分訓練獎勵模型
```

### 8.4 評估建議

**自動評估**：
```python
metrics = {
    'reward_score': reward_model.score(response),
    'perplexity': calculate_ppl(response),
    'diversity': unique_ngrams(response) / total_ngrams,
    'length': len(response.split())
}
```

**人類評估（關鍵）**：
```
維度：
- 有用性 (Helpfulness): 1-5
- 準確性 (Correctness): 1-5
- 無害性 (Harmlessness): 1-5
- 整體質量 (Overall): 1-5

頻率：
- 開發階段：每 100 steps
- 穩定後：每 500 steps
```

---

## 9. 參考資源

### 論文

**RLHF 基礎**：
1. "Training language models to follow instructions with human feedback" (InstructGPT, OpenAI, 2022)
2. "Learning to summarize from human feedback" (OpenAI, 2020)

**DPO**：
3. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Stanford, 2023)

**其他方法**：
4. "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
5. "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (Google, 2023)

### 工具與框架

- **TRL (Transformer Reinforcement Learning)**: https://github.com/huggingface/trl
  - 官方 RLHF/DPO 實現

- **OpenRLHF**: https://github.com/OpenLLMAI/OpenRLHF
  - 高性能 RLHF 框架

- **DeepSpeed-Chat**: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
  - 完整的 RLHF 流程實現

### 資料集

- **Anthropic HH-RLHF**: https://huggingface.co/datasets/Anthropic/hh-rlhf
- **OpenAssistant**: https://huggingface.co/datasets/OpenAssistant/oasst1
- **UltraFeedback**: https://huggingface.co/datasets/openbmb/UltraFeedback

### 教程

- **Hugging Face RLHF Blog**: https://huggingface.co/blog/rlhf
- **OpenAI Spinning Up in Deep RL**: https://spinningup.openai.com/
- **LLM Course by Maxime Labonne**: https://github.com/mlabonne/llm-course

---

## 總結

偏好對齊是使 LLM 真正有用、安全、可靠的關鍵技術：

### 核心要點

1. **RLHF 是工業標準**
   - GPT-4、Claude 都使用
   - 效果最好但複雜度高

2. **DPO 是實用替代**
   - 訓練簡單，穩定性好
   - 效果接近 RLHF
   - 小團隊首選

3. **資料品質至關重要**
   - 偏好資料決定對齊方向
   - 投入時間建立高品質資料集
   - 持續評估和改進

4. **實務挑戰需要經驗**
   - 獎勵 hacking
   - 訓練不穩定
   - 過度優化
   - 需要實驗和調整

5. **人類評估不可或缺**
   - 自動指標無法完全反映質量
   - 定期進行人類評估
   - 根據反饋迭代改進

### 實踐路徑

**起步**：
1. SFT 模型準備
2. 收集 10K-50K 偏好資料
3. 嘗試 DPO 訓練

**進階**：
1. 建立獎勵模型
2. 實施 PPO 訓練
3. 建立評估體系

**成熟**：
1. Constitutional AI
2. AI 反饋迴路
3. 持續學習系統
