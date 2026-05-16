# 強化學習與 LLM 整合指南

## 概述

強化學習（Reinforcement Learning, RL）在 LLM 時代扮演關鍵角色，特別是 RLHF（人類反饋強化學習）已成為訓練對齊 AI 的核心技術。

```
┌─────────────────────────────────────────────────────────────────┐
│                    強化學習在 LLM 中的應用                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   RLHF     │    │    DPO     │    │   PPO      │        │
│  │ 人類反饋對齊 │    │ 直接偏好優化 │    │ 策略梯度    │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              對齊的語言模型                          │      │
│  │         (安全、有幫助、誠實)                         │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                 │
│  應用場景:                                                      │
│  • 模型對齊與安全                                               │
│  • 獎勵建模                                                     │
│  • Agent 決策優化                                               │
│  • 程式碼生成優化                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 強化學習基礎

### 核心概念

```python
"""
強化學習核心概念實現
"""
from dataclasses import dataclass
from typing import List, Tuple, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Experience:
    """經驗元組"""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


class Environment(ABC):
    """環境抽象類"""

    @abstractmethod
    def reset(self) -> Any:
        """重置環境，返回初始狀態"""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """執行動作，返回 (下一狀態, 獎勵, 是否結束, 資訊)"""
        pass

    @abstractmethod
    def get_action_space(self) -> List[Any]:
        """獲取可用動作空間"""
        pass


class Agent(ABC):
    """智能體抽象類"""

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """根據狀態選擇動作"""
        pass

    @abstractmethod
    def update(self, experience: Experience):
        """根據經驗更新策略"""
        pass


class ReplayBuffer:
    """經驗回放緩衝區"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0

    def push(self, experience: Experience):
        """添加經驗"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        """隨機採樣"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class EpsilonGreedyPolicy:
    """ε-貪婪策略"""

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(
        self,
        q_values: np.ndarray,
        action_space: List[Any]
    ) -> Any:
        """選擇動作"""
        if np.random.random() < self.epsilon:
            # 探索：隨機選擇
            return np.random.choice(action_space)
        else:
            # 利用：選擇最大 Q 值
            return action_space[np.argmax(q_values)]

    def decay(self):
        """衰減 epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

### Q-Learning 實現

```python
"""
Q-Learning 演算法實現
"""
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class QLearningAgent(Agent):
    """Q-Learning 智能體"""

    def __init__(
        self,
        action_space: List[Any],
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1
    ):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q 表：state -> action -> value
        self.q_table: Dict[Any, Dict[Any, float]] = defaultdict(
            lambda: {a: 0.0 for a in action_space}
        )

    def select_action(self, state: Any) -> Any:
        """ε-貪婪選擇動作"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        q_values = self.q_table[state]
        max_q = max(q_values.values())
        # 處理多個最大值的情況
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

    def update(self, experience: Experience):
        """Q-Learning 更新"""
        state = experience.state
        action = experience.action
        reward = experience.reward
        next_state = experience.next_state
        done = experience.done

        # 當前 Q 值
        current_q = self.q_table[state][action]

        # 目標 Q 值
        if done:
            target_q = reward
        else:
            max_next_q = max(self.q_table[next_state].values())
            target_q = reward + self.gamma * max_next_q

        # 更新 Q 值
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def get_policy(self) -> Dict[Any, Any]:
        """獲取當前策略"""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = max(q_values, key=q_values.get)
        return policy


class SARSAAgent(Agent):
    """SARSA 智能體（On-Policy）"""

    def __init__(
        self,
        action_space: List[Any],
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1
    ):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[Any, Dict[Any, float]] = defaultdict(
            lambda: {a: 0.0 for a in action_space}
        )
        self.last_action = None

    def select_action(self, state: Any) -> Any:
        """ε-貪婪選擇動作"""
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = np.random.choice(best_actions)

        self.last_action = action
        return action

    def update(self, experience: Experience, next_action: Any = None):
        """SARSA 更新"""
        state = experience.state
        action = experience.action
        reward = experience.reward
        next_state = experience.next_state
        done = experience.done

        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            # 使用實際選擇的下一個動作
            if next_action is None:
                next_action = self.select_action(next_state)
            target_q = reward + self.gamma * self.q_table[next_state][next_action]

        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
```

## RLHF（人類反饋強化學習）

### RLHF 完整流程

```python
"""
RLHF 訓練流程實現
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PreferenceData:
    """人類偏好資料"""
    prompt: str
    chosen: str  # 人類偏好的回應
    rejected: str  # 人類不偏好的回應


class PreferenceDataset(Dataset):
    """偏好資料集"""

    def __init__(
        self,
        data: List[PreferenceData],
        tokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 編碼 chosen 回應
        chosen_text = f"{item.prompt}\n{item.chosen}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 編碼 rejected 回應
        rejected_text = f"{item.prompt}\n{item.rejected}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(),
        }


class RewardModel(nn.Module):
    """獎勵模型"""

    def __init__(self, base_model_name: str = "gpt2"):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(
            self.base_model.config.hidden_size, 1
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """計算獎勵分數"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 使用最後一層隱藏狀態的最後一個 token
        last_hidden_state = outputs.hidden_states[-1]
        # 找到每個序列的最後一個非 padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]

        last_token_hidden = last_hidden_state[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths
        ]

        reward = self.reward_head(last_token_hidden)
        return reward.squeeze(-1)


class RewardModelTrainer:
    """獎勵模型訓練器"""

    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """計算偏好損失"""
        # 計算 chosen 的獎勵
        chosen_rewards = self.model(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )

        # 計算 rejected 的獎勵
        rejected_rewards = self.model(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        # Bradley-Terry 模型損失
        # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        return loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> dict:
        """評估獎勵模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                chosen_rewards = self.model(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                rejected_rewards = self.model(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )

                # 計算準確率
                correct += (chosen_rewards > rejected_rewards).sum().item()
                total += chosen_rewards.shape[0]

        return {"accuracy": correct / total}
```

### PPO 訓練

```python
"""
PPO（Proximal Policy Optimization）用於 RLHF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class PPOConfig:
    """PPO 配置"""
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    batch_size: int = 64
    kl_target: float = 0.02


class PPOMemory:
    """PPO 經驗存儲"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算 GAE（Generalized Advantage Estimation）"""
        advantages = []
        returns = []
        gae = 0

        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = (
                    self.rewards[t]
                    + gamma * values[t + 1]
                    - values[t]
                )
                gae = delta + gamma * gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        # 標準化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


class PPOTrainer:
    """PPO 訓練器用於語言模型"""

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: RewardModel,
        tokenizer,
        config: PPOConfig = None
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config or PPOConfig()

        # 保存參考模型（用於 KL 懲罰）
        self.ref_policy = self._clone_model(policy_model)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=1e-5
        )

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """克隆模型作為參考"""
        import copy
        ref_model = copy.deepcopy(model)
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def generate_response(
        self,
        prompt: str,
        max_length: int = 128
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """生成回應並返回 log_probs"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )

        self.policy.eval()
        with torch.no_grad():
            outputs = self.policy.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = outputs.sequences[0]
        response = self.tokenizer.decode(
            generated_ids[inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 計算 log probabilities
        log_probs = self._compute_log_probs(
            inputs["input_ids"],
            generated_ids.unsqueeze(0)
        )

        return response, generated_ids, log_probs

    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor
    ) -> torch.Tensor:
        """計算生成序列的 log probabilities"""
        self.policy.eval()
        with torch.no_grad():
            outputs = self.policy(output_ids)
            logits = outputs.logits

        # 計算每個 token 的 log prob
        log_probs = F.log_softmax(logits, dim=-1)

        # 獲取實際生成 token 的 log prob
        generated_log_probs = torch.gather(
            log_probs[:, :-1],
            dim=-1,
            index=output_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # 只計算生成部分的 log prob
        prompt_length = input_ids.shape[1]
        response_log_probs = generated_log_probs[:, prompt_length - 1:]

        return response_log_probs.sum(dim=-1)

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """使用獎勵模型計算獎勵"""
        rewards = []

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}\n{response}"
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                reward = self.reward_model(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
            rewards.append(reward.item())

        return torch.tensor(rewards)

    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor
    ) -> torch.Tensor:
        """計算 KL 散度懲罰"""
        # 當前策略的 log prob
        current_log_probs = self._compute_log_probs(input_ids, output_ids)

        # 參考策略的 log prob
        with torch.no_grad():
            ref_outputs = self.ref_policy(output_ids)
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            ref_generated_log_probs = torch.gather(
                ref_log_probs[:, :-1],
                dim=-1,
                index=output_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            prompt_length = input_ids.shape[1]
            ref_response_log_probs = ref_generated_log_probs[:, prompt_length - 1:]
            ref_total_log_prob = ref_response_log_probs.sum(dim=-1)

        # KL = log(p/q) = log_p - log_q
        kl = current_log_probs - ref_total_log_prob

        return kl

    def ppo_update(
        self,
        old_log_probs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """PPO 更新步驟"""
        self.policy.train()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.config.ppo_epochs):
            # 計算新的 log probs
            outputs = self.policy(states)
            logits = outputs.logits

            # 策略損失
            new_log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = torch.gather(
                new_log_probs[:, :-1],
                dim=-1,
                index=actions[:, 1:].unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1)

            ratio = torch.exp(action_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # 熵獎勵（鼓勵探索）
            entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=-1).mean()

            # 總損失
            loss = (
                policy_loss
                - self.config.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()

        return {
            "policy_loss": total_policy_loss / self.config.ppo_epochs,
            "entropy": total_entropy / self.config.ppo_epochs,
        }
```

## DPO（直接偏好優化）

### DPO 實現

```python
"""
DPO（Direct Preference Optimization）實現
DPO 是 RLHF 的簡化替代方案，直接從偏好資料訓練
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class DPOConfig:
    """DPO 配置"""
    beta: float = 0.1  # KL 懲罰係數
    learning_rate: float = 1e-6
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4


class DPOTrainer:
    """DPO 訓練器"""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        config: DPOConfig = None
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()

        # 凍結參考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """計算序列的 log probability"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        # 計算每個 token 的 log prob
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)

        # 獲取實際 token 的 log prob
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # 使用 attention mask 遮蔽 padding
        mask = attention_mask[:, 1:].float()
        token_log_probs = token_log_probs * mask

        # 返回序列的總 log prob
        return token_log_probs.sum(dim=-1)

    def compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 DPO 損失

        DPO Loss = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x)
                                        - log_ref(y_w|x) + log_ref(y_l|x))))
        """
        # 策略模型的 log probs
        pi_chosen_logps = self.compute_log_probs(
            self.model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_input_ids
        )
        pi_rejected_logps = self.compute_log_probs(
            self.model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_input_ids
        )

        # 參考模型的 log probs
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_input_ids
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_input_ids
            )

        # 計算 log ratio
        pi_log_ratio = pi_chosen_logps - pi_rejected_logps
        ref_log_ratio = ref_chosen_logps - ref_rejected_logps

        # DPO 損失
        logits = self.config.beta * (pi_log_ratio - ref_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # 計算額外指標
        with torch.no_grad():
            chosen_rewards = self.config.beta * (pi_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.beta * (pi_rejected_logps - ref_rejected_logps)
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss, {
            "reward_margin": reward_margin.item(),
            "accuracy": accuracy.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item()
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """訓練步驟"""
        self.model.train()

        loss, metrics = self.compute_dpo_loss(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        loss.backward()

        metrics["loss"] = loss.item()
        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """訓練一個 epoch"""
        total_metrics = {}
        num_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            metrics = self.train_step(batch)

            # 梯度累積
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 累積指標
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

        # 平均指標
        return {k: v / num_batches for k, v in total_metrics.items()}


class IPOTrainer(DPOTrainer):
    """
    IPO（Identity Preference Optimization）訓練器
    IPO 是 DPO 的變體，使用更簡單的損失函式
    """

    def compute_ipo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        IPO 損失 = (log_pi(y_w|x) - log_pi(y_l|x)
                    - log_ref(y_w|x) + log_ref(y_l|x) - 1/(2*beta))^2
        """
        # 獲取 log probs
        pi_chosen_logps = self.compute_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask, chosen_input_ids
        )
        pi_rejected_logps = self.compute_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask, rejected_input_ids
        )

        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_input_ids
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_input_ids
            )

        log_ratio_diff = (
            (pi_chosen_logps - ref_chosen_logps)
            - (pi_rejected_logps - ref_rejected_logps)
        )

        # IPO 損失
        target = 1 / (2 * self.config.beta)
        loss = ((log_ratio_diff - target) ** 2).mean()

        return loss
```

## Agent 強化學習

### LLM Agent 訓練

```python
"""
LLM Agent 的強化學習訓練
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class AgentState:
    """Agent 狀態"""
    conversation_history: List[Dict[str, str]]
    current_task: str
    available_tools: List[str]
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 10


@dataclass
class AgentAction:
    """Agent 動作"""
    action_type: str  # "tool_call", "respond", "think"
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    reasoning: Optional[str] = None


class ToolEnvironment:
    """工具執行環境"""

    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.execution_history = []

    def execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行工具"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }

        try:
            result = self.tools[tool_name](**tool_args)
            execution = {
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "success": True
            }
            self.execution_history.append(execution)
            return execution
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def reset(self):
        self.execution_history = []


class AgentRewardCalculator:
    """Agent 獎勵計算器"""

    def __init__(
        self,
        task_completion_reward: float = 10.0,
        step_penalty: float = -0.1,
        tool_error_penalty: float = -1.0,
        helpful_response_reward: float = 2.0
    ):
        self.task_completion_reward = task_completion_reward
        self.step_penalty = step_penalty
        self.tool_error_penalty = tool_error_penalty
        self.helpful_response_reward = helpful_response_reward

    def calculate_reward(
        self,
        state: AgentState,
        action: AgentAction,
        tool_result: Optional[Dict[str, Any]],
        task_completed: bool,
        human_feedback: Optional[float] = None
    ) -> float:
        """計算獎勵"""
        reward = 0.0

        # 步驟懲罰
        reward += self.step_penalty

        # 任務完成獎勵
        if task_completed:
            reward += self.task_completion_reward

        # 工具執行結果
        if tool_result:
            if tool_result.get("success"):
                reward += 0.5  # 成功執行工具
            else:
                reward += self.tool_error_penalty

        # 人類反饋
        if human_feedback is not None:
            reward += human_feedback * self.helpful_response_reward

        return reward


class AgentPolicyNetwork(nn.Module):
    """Agent 策略網絡"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_actions: int = 10,  # 工具數量 + 回應 + 思考
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # 動作類型選擇頭
        self.action_type_head = nn.Linear(hidden_size, 3)  # tool, respond, think

        # 工具選擇頭
        self.tool_head = nn.Linear(hidden_size, num_actions - 2)

        # 價值頭
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        encoded = self.state_encoder(state_embedding)

        action_type_logits = self.action_type_head(encoded)
        tool_logits = self.tool_head(encoded)
        value = self.value_head(encoded)

        return {
            "action_type_logits": action_type_logits,
            "tool_logits": tool_logits,
            "value": value
        }


class AgentRLTrainer:
    """Agent RL 訓練器"""

    def __init__(
        self,
        policy_network: AgentPolicyNetwork,
        llm_backbone,  # 用於生成的 LLM
        tool_env: ToolEnvironment,
        reward_calculator: AgentRewardCalculator
    ):
        self.policy = policy_network
        self.llm = llm_backbone
        self.env = tool_env
        self.reward_calc = reward_calculator

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=1e-4
        )

        self.episode_buffer = []

    def encode_state(self, state: AgentState) -> torch.Tensor:
        """編碼狀態為向量"""
        # 簡化實現：實際應使用 LLM 編碼
        state_text = json.dumps({
            "task": state.current_task,
            "history_length": len(state.conversation_history),
            "tools_available": state.available_tools,
            "step": state.step_count
        })

        # 這裡應該用 LLM 編碼，簡化為隨機向量
        return torch.randn(1, 768)

    def select_action(
        self,
        state: AgentState,
        epsilon: float = 0.1
    ) -> AgentAction:
        """選擇動作"""
        state_embedding = self.encode_state(state)

        with torch.no_grad():
            outputs = self.policy(state_embedding)

        # ε-貪婪探索
        if np.random.random() < epsilon:
            action_type = np.random.choice(["tool_call", "respond", "think"])
        else:
            action_type_probs = torch.softmax(
                outputs["action_type_logits"], dim=-1
            )
            action_type_idx = torch.argmax(action_type_probs, dim=-1).item()
            action_types = ["tool_call", "respond", "think"]
            action_type = action_types[action_type_idx]

        if action_type == "tool_call":
            if np.random.random() < epsilon:
                tool_idx = np.random.randint(len(state.available_tools))
            else:
                tool_probs = torch.softmax(outputs["tool_logits"], dim=-1)
                tool_idx = torch.argmax(tool_probs, dim=-1).item()

            tool_name = state.available_tools[tool_idx]
            # 實際應該用 LLM 生成參數
            tool_args = {}

            return AgentAction(
                action_type="tool_call",
                tool_name=tool_name,
                tool_args=tool_args
            )
        elif action_type == "respond":
            # 用 LLM 生成回應
            response = "Generated response"
            return AgentAction(
                action_type="respond",
                response=response
            )
        else:
            return AgentAction(
                action_type="think",
                reasoning="Thinking about the problem..."
            )

    def run_episode(
        self,
        initial_state: AgentState,
        max_steps: int = 10
    ) -> List[Dict]:
        """運行一個 episode"""
        state = initial_state
        trajectory = []

        for step in range(max_steps):
            # 選擇動作
            action = self.select_action(state)

            # 執行動作
            tool_result = None
            task_completed = False

            if action.action_type == "tool_call":
                tool_result = self.env.execute_tool(
                    action.tool_name,
                    action.tool_args
                )
                state.tool_results.append(tool_result)
            elif action.action_type == "respond":
                # 檢查任務是否完成
                task_completed = self._check_task_completion(state, action)

            # 計算獎勵
            reward = self.reward_calc.calculate_reward(
                state, action, tool_result, task_completed
            )

            # 記錄軌跡
            trajectory.append({
                "state": state,
                "action": action,
                "reward": reward,
                "tool_result": tool_result,
                "done": task_completed
            })

            if task_completed:
                break

            # 更新狀態
            state.step_count += 1

        return trajectory

    def _check_task_completion(
        self,
        state: AgentState,
        action: AgentAction
    ) -> bool:
        """檢查任務是否完成（簡化實現）"""
        # 實際應該使用更複雜的判斷邏輯
        return action.action_type == "respond" and len(state.tool_results) > 0

    def update_policy(self, trajectories: List[List[Dict]]):
        """更新策略（簡化的 REINFORCE）"""
        self.policy.train()

        total_loss = 0

        for trajectory in trajectories:
            # 計算回報
            returns = []
            G = 0
            for step in reversed(trajectory):
                G = step["reward"] + 0.99 * G
                returns.insert(0, G)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # 計算損失
            for step, R in zip(trajectory, returns):
                state_embedding = self.encode_state(step["state"])
                outputs = self.policy(state_embedding)

                # 簡化：只計算動作類型的損失
                action_types = ["tool_call", "respond", "think"]
                action_idx = action_types.index(step["action"].action_type)

                log_prob = torch.log_softmax(
                    outputs["action_type_logits"], dim=-1
                )[0, action_idx]

                loss = -log_prob * R
                total_loss += loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

## 實用工具與框架整合

### 使用 TRL 庫

```python
"""
使用 Hugging Face TRL 庫進行 RLHF 訓練
"""
from trl import (
    PPOTrainer as TRLPPOTrainer,
    PPOConfig as TRLPPOConfig,
    AutoModelForCausalLMWithValueHead,
    DPOTrainer as TRLDPOTrainer,
    DPOConfig as TRLDPOConfig
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch


def setup_ppo_training():
    """設置 PPO 訓練"""

    # 配置
    config = TRLPPOConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        max_grad_norm=0.5,
        target_kl=0.02,
        kl_penalty="kl",
        seed=42,
    )

    # 載入模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 載入參考模型
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    # 建立訓練器
    ppo_trainer = TRLPPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    return ppo_trainer


def ppo_training_loop(
    ppo_trainer,
    reward_model,
    prompts: list,
    num_epochs: int = 10
):
    """PPO 訓練循環"""

    for epoch in range(num_epochs):
        for prompt in prompts:
            # 生成回應
            query_tensors = ppo_trainer.tokenizer.encode(
                prompt,
                return_tensors="pt"
            )

            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7
            )

            # 計算獎勵
            response_text = ppo_trainer.tokenizer.decode(
                response_tensors[0],
                skip_special_tokens=True
            )

            with torch.no_grad():
                reward = reward_model.compute_reward(prompt, response_text)

            # PPO 更新
            stats = ppo_trainer.step(
                [query_tensors[0]],
                [response_tensors[0]],
                [torch.tensor([reward])]
            )

            print(f"Epoch {epoch}, Reward: {reward:.3f}")

    return ppo_trainer.model


def setup_dpo_training():
    """設置 DPO 訓練"""

    # 載入模型
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 載入參考模型
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 載入偏好資料集
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

    def process_sample(sample):
        """處理資料樣本"""
        return {
            "prompt": sample["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:",
            "chosen": sample["chosen"].split("\n\nAssistant:")[-1],
            "rejected": sample["rejected"].split("\n\nAssistant:")[-1]
        }

    dataset = dataset.map(process_sample)

    # DPO 配置
    config = TRLDPOConfig(
        beta=0.1,
        learning_rate=1e-6,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_length=512,
        max_prompt_length=256,
        num_train_epochs=1,
    )

    # 建立訓練器
    dpo_trainer = TRLDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    return dpo_trainer


def run_dpo_training(dpo_trainer):
    """運行 DPO 訓練"""

    dpo_trainer.train()

    # 保存模型
    dpo_trainer.save_model("./dpo_model")

    return dpo_trainer.model
```

### ORPO 實現

```python
"""
ORPO（Odds Ratio Preference Optimization）實現
ORPO 不需要參考模型，是更簡化的偏好優化方法
"""
import torch
import torch.nn.functional as F
from typing import Dict


class ORPOTrainer:
    """ORPO 訓練器"""

    def __init__(
        self,
        model,
        tokenizer,
        lambda_weight: float = 0.1,
        learning_rate: float = 1e-6
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_weight = lambda_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """計算 log probability"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        mask = attention_mask[:, 1:].float()
        return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)

    def compute_orpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        ORPO 損失
        L = L_NLL + λ * L_OR
        其中 L_OR = -log(sigmoid(log(odds_chosen / odds_rejected)))
        """
        # 計算 log probs
        chosen_log_probs = self.compute_log_probs(
            chosen_input_ids,
            chosen_attention_mask,
            chosen_input_ids
        )
        rejected_log_probs = self.compute_log_probs(
            rejected_input_ids,
            rejected_attention_mask,
            rejected_input_ids
        )

        # NLL 損失（只在 chosen 上）
        outputs = self.model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            labels=chosen_input_ids
        )
        nll_loss = outputs.loss

        # 計算 odds ratio
        # odds = p / (1 - p) = exp(log_p) / (1 - exp(log_p))
        chosen_probs = torch.exp(chosen_log_probs)
        rejected_probs = torch.exp(rejected_log_probs)

        # 避免數值問題
        eps = 1e-7
        chosen_odds = chosen_probs / (1 - chosen_probs + eps)
        rejected_odds = rejected_probs / (1 - rejected_probs + eps)

        # Odds ratio 損失
        log_odds_ratio = torch.log(chosen_odds + eps) - torch.log(rejected_odds + eps)
        or_loss = -F.logsigmoid(log_odds_ratio).mean()

        # 總損失
        total_loss = nll_loss + self.lambda_weight * or_loss

        return total_loss, {
            "nll_loss": nll_loss.item(),
            "or_loss": or_loss.item(),
            "total_loss": total_loss.item()
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """訓練步驟"""
        self.model.train()

        loss, metrics = self.compute_orpo_loss(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics
```

## 評估與監控

### RL 訓練監控

```python
"""
強化學習訓練監控
"""
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import json
import time


@dataclass
class RLMetrics:
    """RL 訓練指標"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)
    entropy_values: List[float] = field(default_factory=list)


class RLMonitor:
    """RL 訓練監控器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = RLMetrics()
        self.reward_window = deque(maxlen=window_size)
        self.start_time = time.time()

    def log_episode(
        self,
        reward: float,
        length: int,
        info: Dict[str, Any] = None
    ):
        """記錄 episode"""
        self.metrics.episode_rewards.append(reward)
        self.metrics.episode_lengths.append(length)
        self.reward_window.append(reward)

        if info:
            if "policy_loss" in info:
                self.metrics.policy_losses.append(info["policy_loss"])
            if "value_loss" in info:
                self.metrics.value_losses.append(info["value_loss"])
            if "kl" in info:
                self.metrics.kl_divergences.append(info["kl"])
            if "entropy" in info:
                self.metrics.entropy_values.append(info["entropy"])

    def get_stats(self) -> Dict[str, float]:
        """獲取統計資料"""
        stats = {
            "total_episodes": len(self.metrics.episode_rewards),
            "mean_reward": np.mean(self.metrics.episode_rewards[-self.window_size:]),
            "std_reward": np.std(self.metrics.episode_rewards[-self.window_size:]),
            "max_reward": max(self.metrics.episode_rewards[-self.window_size:]),
            "min_reward": min(self.metrics.episode_rewards[-self.window_size:]),
            "mean_length": np.mean(self.metrics.episode_lengths[-self.window_size:]),
            "elapsed_time": time.time() - self.start_time
        }

        if self.metrics.policy_losses:
            stats["mean_policy_loss"] = np.mean(
                self.metrics.policy_losses[-self.window_size:]
            )

        if self.metrics.kl_divergences:
            stats["mean_kl"] = np.mean(
                self.metrics.kl_divergences[-self.window_size:]
            )

        return stats

    def print_status(self, episode: int):
        """打印訓練狀態"""
        stats = self.get_stats()

        print(f"\n{'='*50}")
        print(f"Episode: {episode}")
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Mean Reward (last {self.window_size}): {stats['mean_reward']:.3f}")
        print(f"Std Reward: {stats['std_reward']:.3f}")
        print(f"Mean Length: {stats['mean_length']:.1f}")

        if "mean_policy_loss" in stats:
            print(f"Mean Policy Loss: {stats['mean_policy_loss']:.4f}")
        if "mean_kl" in stats:
            print(f"Mean KL Divergence: {stats['mean_kl']:.4f}")

        print(f"Elapsed Time: {stats['elapsed_time']:.1f}s")
        print(f"{'='*50}\n")

    def save_metrics(self, path: str):
        """保存指標"""
        data = {
            "episode_rewards": self.metrics.episode_rewards,
            "episode_lengths": self.metrics.episode_lengths,
            "policy_losses": self.metrics.policy_losses,
            "value_losses": self.metrics.value_losses,
            "kl_divergences": self.metrics.kl_divergences,
            "entropy_values": self.metrics.entropy_values
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class RewardShaping:
    """獎勵塑造工具"""

    @staticmethod
    def scale_reward(reward: float, scale: float = 1.0) -> float:
        """縮放獎勵"""
        return reward * scale

    @staticmethod
    def clip_reward(reward: float, min_val: float = -10, max_val: float = 10) -> float:
        """裁剪獎勵"""
        return np.clip(reward, min_val, max_val)

    @staticmethod
    def normalize_reward(
        reward: float,
        mean: float,
        std: float
    ) -> float:
        """標準化獎勵"""
        if std > 0:
            return (reward - mean) / std
        return reward - mean

    @staticmethod
    def add_exploration_bonus(
        reward: float,
        state_count: int,
        beta: float = 0.1
    ) -> float:
        """添加探索獎勵"""
        exploration_bonus = beta / np.sqrt(state_count + 1)
        return reward + exploration_bonus
```

## 最佳實踐

### 訓練建議

```yaml
# RLHF 訓練最佳實踐

資料準備:
  偏好資料:
    - 確保多樣性（不同主題、風格）
    - 標註一致性（多人標註取共識）
    - 資料平衡（避免偏見）
    - 質量檢查（過濾噪聲資料）

  資料量建議:
    - 獎勵模型: 10K-100K 偏好對
    - PPO 訓練: 根據任務複雜度調整
    - DPO 訓練: 通常需要較少資料

獎勵模型訓練:
  架構:
    - 使用與策略模型相同或相似的基礎模型
    - 添加簡單的值頭（線性層）

  訓練技巧:
    - 學習率: 1e-5 到 5e-5
    - 批次大小: 較大更穩定（32-128）
    - 早停: 監控驗證集準確率
    - 正則化: 防止過擬合

PPO 訓練:
  超參數:
    - clip_epsilon: 0.1-0.2
    - KL 目標: 0.01-0.05
    - 批次大小: 較小可以（8-32）
    - PPO epochs: 2-4

  穩定性:
    - 使用 KL 懲罰控制偏離
    - 梯度裁剪（max_norm=0.5）
    - 學習率預熱和衰減
    - 監控 KL 散度

DPO 訓練:
  優勢:
    - 不需要訓練獎勵模型
    - 訓練更穩定
    - 計算效率更高

  注意事項:
    - beta 參數敏感（通常 0.1-0.5）
    - 需要高品質偏好資料
    - 參考模型選擇重要

評估:
  指標:
    - 人類偏好率
    - 獎勵模型分數
    - KL 散度（與參考模型）
    - 任務特定指標

  方法:
    - A/B 測試
    - 自動評估（GPT-4 評分）
    - 紅隊測試
```

## 相關資源

- [TRL 文檔](https://huggingface.co/docs/trl)
- [RLHF 原始論文](https://arxiv.org/abs/2203.02155)
- [DPO 論文](https://arxiv.org/abs/2305.18290)
- [PPO 論文](https://arxiv.org/abs/1707.06347)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
