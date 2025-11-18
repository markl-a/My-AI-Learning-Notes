# LLM 訓練流程詳解 - 從預訓練到對齊

## 目錄
- [概述](#概述)
- [預訓練 (Pre-training)](#預訓練-pre-training)
- [監督式微調 (Supervised Fine-Tuning)](#監督式微調-supervised-fine-tuning)
- [強化學習人類反饋 (RLHF)](#強化學習人類反饋-rlhf)
- [直接偏好優化 (DPO)](#直接偏好優化-dpo)
- [訓練技巧與優化](#訓練技巧與優化)
- [實作範例](#實作範例)
- [成本估算](#成本估算)

---

## 概述

現代大型語言模型的訓練是一個多階段流程,每個階段都有特定的目標和技術:

```
原始模型
   ↓
1. 預訓練 (Pre-training)
   ↓ [基礎語言能力]
2. 監督式微調 (SFT)
   ↓ [任務適配]
3. 獎勵模型訓練 (RM)
   ↓ [人類偏好學習]
4. 強化學習 (RLHF/DPO)
   ↓ [對齊優化]
最終模型
```

### 訓練階段對比

| 階段 | 目標 | 數據量 | 訓練時間 | 成本 |
|------|------|--------|---------|------|
| 預訓練 | 學習語言基礎 | 數 TB (數兆 tokens) | 數週到數月 | 極高 ($數百萬) |
| SFT | 任務適配 | 10K-100K 樣本 | 數小時到數天 | 中等 ($數千) |
| RM 訓練 | 學習人類偏好 | 10K-100K 對比樣本 | 數小時 | 中等 |
| RLHF | 策略優化 | 與 SFT 相當 | 數天 | 高 ($數萬) |

---

## 預訓練 (Pre-training)

### 核心概念

預訓練是 LLM 訓練的基礎階段,模型在海量文本數據上學習語言的統計規律。

**訓練目標:**
```
最大化 P(x_t | x_1, ..., x_{t-1})
```

即預測序列中下一個 token 的概率。

### 預訓練流程

```python
# 概念性預訓練代碼
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PreTraining:
    """預訓練管理器"""

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        # 優化器配置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # 學習率調度器
        self.scheduler = self.get_lr_scheduler()

    def get_lr_scheduler(self):
        """Cosine 學習率調度器 + Warmup"""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Warmup 步數: 通常是總步數的 1-5%
        warmup_steps = int(0.01 * self.config.total_steps)

        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_steps,
            eta_min=self.config.learning_rate * 0.1
        )

    def train_step(self, batch):
        """單個訓練步驟"""

        # 前向傳播
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 計算損失 (Next Token Prediction)
        logits = outputs.logits
        labels = input_ids[:, 1:]  # 向右移動一位作為標籤
        logits = logits[:, :-1, :]  # 調整 logits 維度

        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

        # 反向傳播
        loss.backward()

        # 梯度裁剪 (重要!)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        # 更新參數
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self):
        """完整訓練循環"""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            for batch in dataloader:
                loss = self.train_step(batch)
                global_step += 1

                # 日誌記錄
                if global_step % self.config.log_interval == 0:
                    print(f"Step {global_step}, Loss: {loss:.4f}, "
                          f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

                # 保存檢查點
                if global_step % self.config.save_interval == 0:
                    self.save_checkpoint(global_step)

        return self.model

# 使用示例
config = {
    'learning_rate': 6e-4,
    'batch_size': 512,  # 實際會用梯度累積
    'num_epochs': 1,  # 通常只訓練 1 epoch
    'total_steps': 1000000,
    'log_interval': 100,
    'save_interval': 10000,
}

# trainer = PreTraining(model, dataset, config)
# trained_model = trainer.train()
```

### 預訓練數據處理

```python
class TextDataset:
    """預訓練文本數據集"""

    def __init__(self, data_paths, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.documents = self.load_documents(data_paths)

    def load_documents(self, paths):
        """加載並處理文檔"""
        documents = []

        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                # 按文檔分割
                doc = f.read()
                documents.append(doc)

        return documents

    def __getitem__(self, idx):
        """獲取單個樣本"""

        # 獲取文檔
        doc = self.documents[idx]

        # Tokenize
        encodings = self.tokenizer(
            doc,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0)
        }

    def __len__(self):
        return len(self.documents)
```

### 預訓練關鍵技術

#### 1. 數據混合 (Data Mixing)

```python
# 不同來源數據的混合策略
data_sources = {
    'common_crawl': 0.60,      # 60% 網頁數據
    'books': 0.15,             # 15% 書籍
    'wikipedia': 0.10,         # 10% 維基百科
    'github': 0.10,            # 10% 代碼
    'papers': 0.05             # 5% 學術論文
}

# 高質量過濾
def quality_filter(text):
    """數據質量過濾"""

    # 1. 長度過濾
    if len(text.split()) < 50:
        return False

    # 2. 語言檢測
    if not is_target_language(text):
        return False

    # 3. 去重
    if is_duplicate(text):
        return False

    # 4. 有害內容過濾
    if contains_harmful_content(text):
        return False

    return True
```

#### 2. 梯度累積 (Gradient Accumulation)

```python
def train_with_gradient_accumulation(model, dataloader, accumulation_steps=8):
    """使用梯度累積訓練"""

    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # 前向傳播
        loss = model(batch)

        # 縮放損失
        loss = loss / accumulation_steps

        # 反向傳播
        loss.backward()

        # 每 N 步更新一次
        if (i + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新參數
            optimizer.step()
            optimizer.zero_grad()
```

#### 3. 混合精度訓練

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, dataloader):
    """混合精度訓練"""

    scaler = GradScaler()

    for batch in dataloader:
        optimizer.zero_grad()

        # 自動混合精度
        with autocast():
            loss = model(batch)

        # 縮放損失並反向傳播
        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新參數
        scaler.step(optimizer)
        scaler.update()
```

---

## 監督式微調 (Supervised Fine-Tuning)

### 核心概念

SFT 將預訓練模型適配到特定任務,使用高質量的指令-回應對進行訓練。

### SFT 數據格式

```python
# 典型的 SFT 數據格式
sft_data = [
    {
        "instruction": "解釋什麼是量子糾纏",
        "input": "",
        "output": "量子糾纏是一種量子力學現象,當兩個或多個粒子以某種方式相互作用後..."
    },
    {
        "instruction": "將以下文本翻譯成英文",
        "input": "你好,世界!",
        "output": "Hello, World!"
    },
    {
        "instruction": "寫一個 Python 函數來計算斐波那契數列",
        "input": "n=10",
        "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    }
]
```

### SFT 訓練實現

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class SFTTrainer:
    """監督式微調訓練器"""

    def __init__(self, model_name, sft_data):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 準備數據集
        self.dataset = self.prepare_dataset(sft_data)

    def prepare_dataset(self, sft_data):
        """準備 SFT 數據集"""

        formatted_data = []

        for example in sft_data:
            # 格式化提示
            if example['input']:
                prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"

            # 完整文本
            full_text = prompt + example['output']

            # Tokenize
            encodings = self.tokenizer(
                full_text,
                truncation=True,
                max_length=2048,
                padding='max_length',
                return_tensors='pt'
            )

            formatted_data.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()  # 標籤與輸入相同
            })

        return Dataset.from_list(formatted_data)

    def train(self, output_dir='./sft_model'):
        """執行微調"""

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            fp16=True,  # 混合精度
            optim='adamw_torch',
            lr_scheduler_type='cosine',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
        )

        trainer.train()

        # 保存模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return self.model

# 使用示例
# sft_trainer = SFTTrainer('gpt2', sft_data)
# fine_tuned_model = sft_trainer.train()
```

### LoRA 高效微調

```python
from peft import LoraConfig, get_peft_model, TaskType

def sft_with_lora(base_model_name, sft_data):
    """使用 LoRA 進行高效微調"""

    # 加載基礎模型
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA 秩
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # 目標模塊
    )

    # 應用 LoRA
    model = get_peft_model(model, lora_config)

    # 顯示可訓練參數
    model.print_trainable_parameters()
    # 輸出: trainable params: 4.7M || all params: 7B || trainable%: 0.067%

    # 正常訓練流程...

    return model
```

---

## 強化學習人類反饋 (RLHF)

### 核心概念

RLHF 使用強化學習將模型與人類偏好對齊,是 ChatGPT 成功的關鍵。

### RLHF 三步流程

```
1. 收集人類反饋數據
   ↓
2. 訓練獎勵模型 (Reward Model)
   ↓
3. 使用 PPO 優化策略
```

### 1. 獎勵模型訓練

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """獎勵模型"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # 獎勵頭
        self.reward_head = nn.Linear(
            base_model.config.hidden_size,
            1,
            bias=False
        )

    def forward(self, input_ids, attention_mask):
        # 獲取基礎模型輸出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 使用最後一個 token 的隱藏狀態
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]

        # 計算獎勵分數
        reward = self.reward_head(last_token_hidden)

        return reward

class RewardModelTrainer:
    """獎勵模型訓練器"""

    def __init__(self, base_model, comparison_data):
        self.model = RewardModel(base_model)
        self.data = comparison_data
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5
        )

    def train_step(self, batch):
        """訓練步驟"""

        # batch 包含: chosen (更好的回應) 和 rejected (較差的回應)
        chosen_ids = batch['chosen_input_ids']
        chosen_mask = batch['chosen_attention_mask']
        rejected_ids = batch['rejected_input_ids']
        rejected_mask = batch['rejected_attention_mask']

        # 計算獎勵
        reward_chosen = self.model(chosen_ids, chosen_mask)
        reward_rejected = self.model(rejected_ids, rejected_mask)

        # 損失: 鼓勵 chosen > rejected
        # 使用 sigmoid 確保差異為正
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, eval_data):
        """評估獎勵模型準確率"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_data:
                reward_chosen = self.model(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask']
                )
                reward_rejected = self.model(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask']
                )

                # 正確預測: chosen 獎勵 > rejected 獎勵
                correct += (reward_chosen > reward_rejected).sum().item()
                total += len(reward_chosen)

        accuracy = correct / total
        return accuracy
```

### 2. PPO 策略優化

```python
class PPOTrainer:
    """PPO 訓練器用於 RLHF"""

    def __init__(self, policy_model, reward_model, ref_model):
        """
        Args:
            policy_model: 要訓練的策略模型
            reward_model: 已訓練的獎勵模型
            ref_model: 參考模型 (SFT 模型的凍結副本)
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model

        # 凍結參考模型和獎勵模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=1e-6
        )

        # PPO 超參數
        self.clip_epsilon = 0.2
        self.kl_coef = 0.1
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def compute_advantages(self, rewards, values):
        """計算優勢函數 (Advantage)"""
        advantages = []
        advantage = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            advantage = r - v + 0.99 * advantage
            advantages.insert(0, advantage)

        return torch.tensor(advantages)

    def ppo_step(self, queries, responses):
        """PPO 訓練步驟"""

        # 1. 生成回應並計算舊的 log probs
        with torch.no_grad():
            old_log_probs = self.policy.compute_log_probs(queries, responses)
            ref_log_probs = self.ref_model.compute_log_probs(queries, responses)
            rewards = self.reward_model.compute_rewards(queries, responses)

        # 2. 計算 KL 散度懲罰
        kl_penalty = old_log_probs - ref_log_probs
        rewards = rewards - self.kl_coef * kl_penalty

        # 3. 計算新的 log probs
        new_log_probs = self.policy.compute_log_probs(queries, responses)

        # 4. 計算概率比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 5. PPO 裁剪目標
        advantages = self.compute_advantages(rewards, values=None)

        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        )

        # 6. 計算損失
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # 7. 反向傳播
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return policy_loss.item()

# 完整 RLHF 流程
def rlhf_pipeline(sft_model, comparison_data, prompts):
    """完整 RLHF 管道"""

    # 步驟 1: 訓練獎勵模型
    print("訓練獎勵模型...")
    reward_model = RewardModelTrainer(sft_model, comparison_data)
    reward_model.train()

    # 步驟 2: 創建參考模型
    ref_model = copy.deepcopy(sft_model)

    # 步驟 3: PPO 訓練
    print("PPO 優化...")
    ppo_trainer = PPOTrainer(sft_model, reward_model, ref_model)

    for epoch in range(num_epochs):
        for batch in prompts:
            # 生成回應
            responses = sft_model.generate(batch)

            # PPO 更新
            loss = ppo_trainer.ppo_step(batch, responses)

            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return sft_model
```

---

## 直接偏好優化 (DPO)

### 核心優勢

DPO 是 RLHF 的簡化替代方案,無需訓練獎勵模型和使用 RL。

**關鍵思想:**
直接優化策略,使其更傾向於人類偏好的回應。

### DPO 損失函數

```
L_DPO = -E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]
```

其中:
- `y_w`: 更好的回應 (chosen)
- `y_l`: 較差的回應 (rejected)
- `β`: 溫度參數
- `σ`: sigmoid 函數

### DPO 實現

```python
import torch
import torch.nn.functional as F

class DPOTrainer:
    """Direct Preference Optimization 訓練器"""

    def __init__(self, model, ref_model, beta=0.1):
        """
        Args:
            model: 要訓練的模型
            ref_model: 參考模型 (通常是 SFT 模型)
            beta: 溫度參數
        """
        self.model = model
        self.ref_model = ref_model
        self.beta = beta

        # 凍結參考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-7
        )

    def compute_log_probs(self, model, input_ids, labels):
        """計算 log probabilities"""

        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        # 獲取每個 token 的 log probability
        log_probs = F.log_softmax(logits, dim=-1)

        # 收集實際標籤的 log probs
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        return gathered_log_probs.sum(dim=-1)

    def dpo_loss(self, batch):
        """計算 DPO 損失"""

        # 提取 chosen 和 rejected 數據
        chosen_input_ids = batch['chosen_input_ids']
        chosen_labels = batch['chosen_labels']
        rejected_input_ids = batch['rejected_input_ids']
        rejected_labels = batch['rejected_labels']

        # 計算策略模型的 log probs
        policy_chosen_log_probs = self.compute_log_probs(
            self.model, chosen_input_ids, chosen_labels
        )
        policy_rejected_log_probs = self.compute_log_probs(
            self.model, rejected_input_ids, rejected_labels
        )

        # 計算參考模型的 log probs
        with torch.no_grad():
            ref_chosen_log_probs = self.compute_log_probs(
                self.ref_model, chosen_input_ids, chosen_labels
            )
            ref_rejected_log_probs = self.compute_log_probs(
                self.ref_model, rejected_input_ids, rejected_labels
            )

        # 計算 log ratios
        chosen_log_ratios = policy_chosen_log_probs - ref_chosen_log_probs
        rejected_log_ratios = policy_rejected_log_probs - ref_rejected_log_probs

        # DPO 損失
        logits = self.beta * (chosen_log_ratios - rejected_log_ratios)
        loss = -F.logsigmoid(logits).mean()

        # 計算準確率 (chosen 是否獲得更高的獎勵)
        accuracy = (chosen_log_ratios > rejected_log_ratios).float().mean()

        return loss, accuracy

    def train_step(self, batch):
        """訓練步驟"""

        self.optimizer.zero_grad()

        loss, accuracy = self.dpo_loss(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

# 使用示例
def train_with_dpo(sft_model, preference_data):
    """使用 DPO 訓練"""

    # 創建參考模型
    ref_model = copy.deepcopy(sft_model)

    # 創建 DPO 訓練器
    dpo_trainer = DPOTrainer(sft_model, ref_model, beta=0.1)

    # 訓練循環
    for epoch in range(3):
        total_loss = 0
        total_acc = 0

        for batch in preference_data:
            metrics = dpo_trainer.train_step(batch)
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']

        avg_loss = total_loss / len(preference_data)
        avg_acc = total_acc / len(preference_data)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

    return sft_model
```

### RLHF vs DPO 對比

| 特性 | RLHF | DPO |
|------|------|-----|
| 復雜度 | 高 (需要 RM + PPO) | 低 (直接優化) |
| 訓練穩定性 | 較差 | 較好 |
| 計算成本 | 高 | 中等 |
| 性能 | 略高 | 接近 RLHF |
| 實現難度 | 困難 | 簡單 |
| 推薦場景 | 大規模生產 | 快速迭代 |

---

## 訓練技巧與優化

### 1. 學習率調度

```python
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """線性學習率 + Warmup"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup 階段: 線性增加
            return float(current_step) / float(max(1, num_warmup_steps))

        # 訓練階段: 線性衰減
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 使用示例
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

### 2. 梯度檢查點 (Gradient Checkpointing)

```python
from torch.utils.checkpoint import checkpoint

class TransformerWithCheckpointing(nn.Module):
    """使用梯度檢查點的 Transformer"""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.use_checkpointing = config.use_checkpointing

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # 使用檢查點節省內存
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x

# 優點: 減少內存使用 ~50%
# 缺點: 增加訓練時間 ~20%
```

### 3. ZeRO 優化 (DeepSpeed)

```python
import deepspeed

# DeepSpeed 配置
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO-3: 分片優化器狀態、梯度和參數
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# 訓練循環
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 4. 數據並行與模型並行

```python
# 數據並行 (DP)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# 分佈式數據並行 (DDP) - 推薦
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup_distributed()
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 模型並行 (Pipeline Parallelism)
from torch.distributed.pipeline.sync import Pipe

model = nn.Sequential(
    layer1,
    layer2,
    layer3,
    layer4
)

# 分割到 4 個 GPU
model = Pipe(model, chunks=8, balance=[1, 1, 1, 1])
```

---

## 實作範例

### 完整的微調流程

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def complete_finetuning_pipeline():
    """完整的微調管道"""

    # 1. 加載模型和 tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 準備數據
    dataset = load_dataset("your_dataset")

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
        remove_columns=dataset["train"].column_names
    )

    # 3. 數據整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # 4. 訓練參數
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        fp16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # 5. 創建訓練器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # 6. 訓練
    trainer.train()

    # 7. 保存模型
    trainer.save_model("./final_model")

    return model, tokenizer

# 執行
# model, tokenizer = complete_finetuning_pipeline()
```

---

## 成本估算

### GPU 成本計算

```python
def estimate_training_cost(
    num_parameters,
    num_tokens,
    gpu_type="A100",
    gpu_hours_rate=2.0
):
    """估算訓練成本"""

    # GPU FLOPs (每秒浮點運算次數)
    gpu_flops = {
        "A100": 312e12,  # 312 TFLOPS
        "H100": 1000e12,  # 1 PFLOPS
        "V100": 125e12   # 125 TFLOPS
    }

    # 計算 FLOPs
    # 公式: 6 * num_parameters * num_tokens
    total_flops = 6 * num_parameters * num_tokens

    # 計算所需 GPU 時間
    gpu_hours = total_flops / (gpu_flops[gpu_type] * 3600)

    # 計算成本
    total_cost = gpu_hours * gpu_hours_rate

    return {
        "total_flops": total_flops,
        "gpu_hours": gpu_hours,
        "total_cost_usd": total_cost
    }

# 示例: 訓練一個 7B 參數模型
cost = estimate_training_cost(
    num_parameters=7e9,
    num_tokens=2e12,  # 2 trillion tokens
    gpu_type="A100",
    gpu_hours_rate=2.0
)

print(f"預估 GPU 小時數: {cost['gpu_hours']:,.0f}")
print(f"預估成本: ${cost['total_cost_usd']:,.0f}")
# 輸出:
# 預估 GPU 小時數: 37,500
# 預估成本: $75,000
```

### 不同模型規模的訓練成本

| 模型規模 | 訓練數據 | GPU (A100) | 時間 | 估算成本 |
|---------|---------|------------|------|---------|
| 125M | 300B tokens | 8 | 3 天 | $1,200 |
| 1.3B | 300B tokens | 64 | 4 天 | $12,000 |
| 7B | 2T tokens | 256 | 14 天 | $170,000 |
| 70B | 2T tokens | 2048 | 21 天 | $2,000,000 |
| 175B (GPT-3) | 300B tokens | 1024 | 34 天 | $4,600,000 |

---

## 總結

### 訓練流程選擇指南

```
場景                     → 推薦方法
─────────────────────────────────────
從零預訓練                → 完整預訓練流程
任務適配                  → SFT (標準 / LoRA)
提升對話質量              → RLHF / DPO
有限預算                  → DPO
需要最佳性能              → RLHF
快速原型                  → LoRA 微調
```

### 關鍵要點

✅ **預訓練**: 海量數據 + 長時間 + 高成本
✅ **SFT**: 高質量樣本 + 快速適配
✅ **RLHF**: 人類反饋 + 最佳對齊
✅ **DPO**: 簡化 RLHF + 更穩定
✅ **LoRA**: 高效微調 + 低成本

---

## 參考資源

### 論文

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始論文
2. [InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF 詳解
3. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO 論文
4. [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - LoRA 技術
5. [ZeRO](https://arxiv.org/abs/1910.02054) - 內存優化

### 框架與工具

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **DeepSpeed**: https://www.deepspeed.ai/
- **TRL (Transformer Reinforcement Learning)**: https://github.com/huggingface/trl
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl

### 教程

- [Hugging Face RLHF Course](https://huggingface.co/learn/rlhf-course/unit0/1)
- [DeepSpeed Tutorials](https://www.deepspeed.ai/tutorials/)
