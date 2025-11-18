# 持續學習與災難性遺忘 (Continual Learning & Catastrophic Forgetting)

## 目錄

1. [什麼是災難性遺忘](#什麼是災難性遺忘)
2. [為什麼會發生災難性遺忘](#為什麼會發生災難性遺忘)
3. [持續學習的挑戰](#持續學習的挑戰)
4. [緩解災難性遺忘的方法](#緩解災難性遺忘的方法)
5. [實作範例](#實作範例)
6. [評估和監控](#評估和監控)
7. [最佳實踐](#最佳實踐)

---

## 什麼是災難性遺忘

### 定義

**災難性遺忘 (Catastrophic Forgetting)** 是指神經網絡在學習新任務時，會迅速且顯著地遺忘先前學習的知識。

### 典型場景

在 LLM 微調中的表現：

```
階段 1: 預訓練模型
  ✓ 通用語言能力
  ✓ 知識廣度
  ✓ 多樣化任務能力

階段 2: 在任務 A 上微調
  ✓ 任務 A 性能優異
  ✗ 通用能力下降
  ✗ 其他任務性能降低

階段 3: 在任務 B 上繼續微調
  ✓ 任務 B 性能優異
  ✗ 任務 A 性能大幅下降 ← 災難性遺忘
  ✗ 通用能力進一步下降
```

### 示例

```python
# 災難性遺忘的簡單示例
model = load_pretrained_model("gpt2")

# 初始性能
print("初始性能:")
print(f"通用問答: {evaluate(model, general_qa_data)}")  # 85%
print(f"代碼生成: {evaluate(model, code_data)}")  # 75%

# 在客服對話上微調
fine_tune(model, customer_service_data, epochs=5)

print("\n微調後性能:")
print(f"客服對話: {evaluate(model, customer_service_data)}")  # 95% ↑
print(f"通用問答: {evaluate(model, general_qa_data)}")  # 60% ↓↓
print(f"代碼生成: {evaluate(model, code_data)}")  # 45% ↓↓↓
```

---

## 為什麼會發生災難性遺忘

### 神經網絡的本質

1. **權重覆蓋**
   - 新任務的訓練會直接修改網絡權重
   - 舊任務的知識被新知識覆蓋

2. **梯度干擾**
   - 新任務的梯度更新與舊任務的最優解衝突
   - 權重被推離舊任務的最優區域

3. **表示空間重組**
   - 網絡的內部表示被重新組織以適應新任務
   - 舊任務的表示結構被破壞

### 數學視角

假設模型參數為 θ：

```
任務 A 的最優參數: θ*_A
任務 B 的最優參數: θ*_B

問題: θ*_A ≠ θ*_B

在 B 上微調後: θ → θ*_B
但在 A 上的性能: L_A(θ*_B) >> L_A(θ*_A)
```

### LLM 特有因素

1. **過度專精化**
   - 模型快速適應新任務的特定模式
   - 失去原有的通用性

2. **數據分佈偏移**
   - 微調數據與預訓練數據分佈差異大
   - 模型過度擬合微調數據

3. **小數據集效應**
   - 微調數據量通常遠小於預訓練數據
   - 容易過擬合並遺忘通用知識

---

## 持續學習的挑戰

### 挑戰 1: Stability-Plasticity 困境

**穩定性 (Stability)**：保持舊知識的能力

**可塑性 (Plasticity)**：學習新知識的能力

```
高穩定性 ← → 高可塑性
   ↓               ↓
難以學習新知識   容易忘記舊知識
```

**目標**：在兩者之間找到平衡

### 挑戰 2: 無法訪問舊數據

在實際應用中，通常無法訪問所有歷史訓練數據：

- **隱私限制**：無法保存用戶數據
- **存儲限制**：無法存儲所有歷史數據
- **許可限制**：預訓練數據不可用

### 挑戰 3: 計算成本

持續學習需要額外的計算資源：

- **重放 (Replay)**：需要存儲和重訓歷史數據
- **正則化**：需要計算額外的正則項
- **架構方法**：需要更複雜的模型結構

---

## 緩解災難性遺忘的方法

### 方法 1: 經驗重放 (Experience Replay)

#### 核心思想

在訓練新任務時，混入舊任務的樣本

#### 實現策略

**1.1 簡單重放**

```python
class ReplayBuffer:
    """經驗重放緩衝區"""

    def __init__(self, max_size=1000):
        self.buffer = []
        self.max_size = max_size

    def add(self, examples):
        """添加樣本到緩衝區"""
        self.buffer.extend(examples)

        # 如果超過容量，隨機移除舊樣本
        if len(self.buffer) > self.max_size:
            import random
            self.buffer = random.sample(self.buffer, self.max_size)

    def sample(self, n):
        """從緩衝區採樣"""
        import random
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def get_all(self):
        """獲取所有樣本"""
        return self.buffer


# 使用示例
replay_buffer = ReplayBuffer(max_size=1000)

# 任務 A 訓練後，保存樣本
replay_buffer.add(task_a_data)

# 訓練任務 B 時，混入重放樣本
for epoch in range(num_epochs):
    # 混合新舊數據
    replay_samples = replay_buffer.sample(batch_size // 2)
    new_samples = sample_from(task_b_data, batch_size // 2)

    mixed_batch = replay_samples + new_samples
    train_step(model, mixed_batch)
```

**1.2 基於重要性的重放**

```python
import numpy as np

class ImportanceReplayBuffer:
    """基於重要性的重放緩衝區"""

    def __init__(self, max_size=1000):
        self.buffer = []
        self.importances = []
        self.max_size = max_size

    def add(self, examples, importances=None):
        """添加樣本及其重要性分數"""
        if importances is None:
            importances = [1.0] * len(examples)

        self.buffer.extend(examples)
        self.importances.extend(importances)

        # 如果超過容量，移除重要性低的樣本
        if len(self.buffer) > self.max_size:
            # 按重要性排序
            sorted_indices = np.argsort(self.importances)[::-1]
            keep_indices = sorted_indices[:self.max_size]

            self.buffer = [self.buffer[i] for i in keep_indices]
            self.importances = [self.importances[i] for i in keep_indices]

    def sample(self, n, temperature=1.0):
        """按重要性採樣"""
        # 使用重要性作為採樣權重
        probs = np.array(self.importances) ** (1/temperature)
        probs = probs / probs.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=min(n, len(self.buffer)),
            p=probs,
            replace=False
        )

        return [self.buffer[i] for i in indices]


# 計算樣本重要性
def calculate_importance(model, example):
    """計算樣本的重要性（例如：損失值）"""
    loss = model.compute_loss(example)
    return float(loss)

# 使用
buffer = ImportanceReplayBuffer(max_size=1000)
importances = [calculate_importance(model, ex) for ex in task_a_data]
buffer.add(task_a_data, importances)
```

**1.3 生成式重放**

```python
class GenerativeReplay:
    """使用生成模型生成舊任務的偽樣本"""

    def __init__(self, generator_model):
        self.generator = generator_model

    def generate_pseudo_samples(self, n, task_description):
        """生成偽樣本"""
        pseudo_samples = []

        for _ in range(n):
            # 使用生成模型創建樣本
            prompt = f"Generate an example for task: {task_description}"
            generated = self.generator.generate(prompt)

            pseudo_samples.append({
                "instruction": generated["instruction"],
                "output": generated["output"],
                "is_pseudo": True
            })

        return pseudo_samples

# 使用 AI 生成舊任務的樣本
from ai_assisted_data_generator import AIDataGenerator

gen = AIDataGenerator()
pseudo_samples = gen.generate_examples_from_topic(
    topic="Previous task examples",
    num_examples=100
)

# 混入訓練
mixed_data = current_task_data + pseudo_samples
```

### 方法 2: 正則化方法

#### 2.1 Elastic Weight Consolidation (EWC)

**核心思想**：為重要的參數添加正則化，防止其大幅改變

**數學公式**：
```
L_total = L_new_task + λ Σ F_i (θ_i - θ*_i)^2
```

其中：
- L_new_task：新任務損失
- F_i：參數 i 的 Fisher 信息（重要性）
- θ*_i：舊任務的最優參數
- λ：正則化強度

**實現**：

```python
import torch
import torch.nn.functional as F

class EWC:
    """Elastic Weight Consolidation"""

    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.device = device

        # 保存舊任務的參數
        self.saved_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # 計算 Fisher 信息矩陣
        self.fisher_matrix = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        """計算 Fisher 信息矩陣"""
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向傳播
            outputs = self.model(**batch)
            loss = outputs.loss

            # 計算梯度
            self.model.zero_grad()
            loss.backward()

            # 累積梯度的平方作為 Fisher 信息
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        # 平均
        n_samples = len(dataloader)
        for name in fisher:
            fisher[name] /= n_samples

        return fisher

    def penalty(self):
        """計算 EWC 懲罰項"""
        loss = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                saved = self.saved_params[name]
                fisher = self.fisher_matrix[name]

                # EWC 損失
                loss += (fisher * (param - saved).pow(2)).sum()

        return loss


# 使用示例
# 1. 在任務 A 上訓練
model = train_on_task_a(model, task_a_data)

# 2. 計算 EWC
ewc = EWC(model, task_a_dataloader)

# 3. 在任務 B 上訓練，加入 EWC 懲罰
def train_with_ewc(model, dataloader, ewc, lambda_ewc=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for batch in dataloader:
        # 正常損失
        outputs = model(**batch)
        loss = outputs.loss

        # 添加 EWC 懲罰
        ewc_loss = ewc.penalty()
        total_loss = loss + lambda_ewc * ewc_loss

        # 反向傳播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

train_with_ewc(model, task_b_dataloader, ewc)
```

#### 2.2 Learning without Forgetting (LwF)

**核心思想**：使用知識蒸餾，保持舊任務的輸出分佈

```python
class LwF:
    """Learning without Forgetting"""

    def __init__(self, model, temperature=2.0):
        self.model = model
        self.temperature = temperature

        # 保存舊模型（用於蒸餾）
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()

        # 凍結舊模型
        for param in self.old_model.parameters():
            param.requires_grad = False

    def distillation_loss(self, outputs_new, outputs_old):
        """計算蒸餾損失"""
        # 使用溫度軟化輸出
        soft_targets = F.softmax(outputs_old.logits / self.temperature, dim=-1)
        soft_outputs = F.log_softmax(outputs_new.logits / self.temperature, dim=-1)

        # KL 散度
        loss = F.kl_div(
            soft_outputs,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss


# 使用示例
import copy

# 1. 訓練任務 A 後，保存舊模型
old_model = copy.deepcopy(model)
lwf = LwF(model)

# 2. 訓練任務 B，加入蒸餾損失
def train_with_lwf(model, dataloader, lwf, alpha=0.5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for batch in dataloader:
        # 新任務損失
        outputs_new = model(**batch)
        task_loss = outputs_new.loss

        # 蒸餾損失（在舊任務數據或新任務數據上）
        with torch.no_grad():
            outputs_old = lwf.old_model(**batch)

        distill_loss = lwf.distillation_loss(outputs_new, outputs_old)

        # 總損失
        total_loss = alpha * task_loss + (1 - alpha) * distill_loss

        # 更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### 方法 3: 參數隔離方法

#### 3.1 Progressive Neural Networks

**思想**：為每個新任務添加新的網絡列

```python
class ProgressiveNN:
    """漸進式神經網絡"""

    def __init__(self, base_model):
        self.columns = [base_model]  # 每個任務一列
        self.adapters = []  # 跨列的適配器

    def add_task_column(self, new_model):
        """為新任務添加網絡列"""
        # 凍結所有舊列
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False

        # 添加新列
        self.columns.append(new_model)

        # 創建適配器連接新列和舊列
        adapter = self._create_adapter(len(self.columns))
        self.adapters.append(adapter)

    def forward(self, x, task_id):
        """前向傳播"""
        # 使用對應任務的列
        return self.columns[task_id](x)
```

#### 3.2 Adapter Tuning

**思想**：為每個任務添加小型適配器模塊，凍結主模型

```python
class TaskAdapter(torch.nn.Module):
    """任務特定的適配器"""

    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = torch.nn.Linear(hidden_size, adapter_size)
        self.up_project = torch.nn.Linear(adapter_size, hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, hidden_states):
        """適配器前向傳播"""
        down = self.down_project(hidden_states)
        activated = self.activation(down)
        up = self.up_project(activated)

        # 殘差連接
        return hidden_states + up


class ModelWithAdapters(torch.nn.Module):
    """帶適配器的模型"""

    def __init__(self, base_model, num_tasks, adapter_size=64):
        super().__init__()
        self.base_model = base_model

        # 凍結基礎模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 為每個任務創建適配器
        hidden_size = base_model.config.hidden_size
        self.task_adapters = torch.nn.ModuleList([
            TaskAdapter(hidden_size, adapter_size)
            for _ in range(num_tasks)
        ])

    def forward(self, input_ids, task_id):
        """前向傳播"""
        # 基礎模型
        outputs = self.base_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        # 應用任務適配器
        adapted = self.task_adapters[task_id](hidden_states)

        return adapted
```

### 方法 4: 混合策略（推薦）

結合多種方法以獲得最佳效果：

```python
class ContinualLearningTrainer:
    """持續學習訓練器"""

    def __init__(
        self,
        model,
        replay_buffer_size=1000,
        ewc_lambda=1000,
        lwf_alpha=0.5,
        use_replay=True,
        use_ewc=True,
        use_lwf=True
    ):
        self.model = model
        self.use_replay = use_replay
        self.use_ewc = use_ewc
        self.use_lwf = use_lwf

        # 經驗重放
        if use_replay:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # EWC
        self.ewc = None
        self.ewc_lambda = ewc_lambda

        # LwF
        self.lwf = None
        self.lwf_alpha = lwf_alpha

    def train_on_new_task(self, task_data, task_name):
        """在新任務上訓練"""

        # 1. 如果有舊任務，設置防遺忘機制
        if hasattr(self, 'previous_task'):
            if self.use_lwf:
                self.lwf = LwF(self.model)

            if self.use_ewc:
                self.ewc = EWC(self.model, self.previous_dataloader)

        # 2. 準備訓練數據
        train_data = task_data

        if self.use_replay and len(self.replay_buffer.buffer) > 0:
            # 混入重放數據
            replay_samples = self.replay_buffer.get_all()
            train_data = task_data + replay_samples

        # 3. 訓練
        for epoch in range(num_epochs):
            for batch in dataloader(train_data):
                # 計算損失
                outputs = self.model(**batch)
                loss = outputs.loss

                # 添加 EWC 懲罰
                if self.ewc is not None:
                    loss += self.ewc_lambda * self.ewc.penalty()

                # 添加 LwF 蒸餾損失
                if self.lwf is not None:
                    with torch.no_grad():
                        old_outputs = self.lwf.old_model(**batch)
                    distill_loss = self.lwf.distillation_loss(outputs, old_outputs)
                    loss = self.lwf_alpha * loss + (1 - self.lwf_alpha) * distill_loss

                # 反向傳播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # 4. 訓練後，保存樣本到重放緩衝區
        if self.use_replay:
            self.replay_buffer.add(task_data[:100])  # 保存部分樣本

        # 5. 記錄當前任務
        self.previous_task = task_name
        self.previous_dataloader = dataloader(task_data)


# 使用示例
trainer = ContinualLearningTrainer(
    model=model,
    replay_buffer_size=1000,
    use_replay=True,
    use_ewc=True,
    use_lwf=True
)

# 依次訓練多個任務
trainer.train_on_new_task(task_a_data, "task_a")
trainer.train_on_new_task(task_b_data, "task_b")
trainer.train_on_new_task(task_c_data, "task_c")
```

---

## 實作範例

### 完整的持續學習管道

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import copy


class ComprehensiveContinualLearner:
    """綜合持續學習框架"""

    def __init__(
        self,
        model_name,
        strategies=["replay", "ewc", "lwf"],
        replay_size=1000,
        ewc_lambda=1000,
        lwf_alpha=0.5
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.strategies = strategies
        self.task_history = []

        # 初始化組件
        if "replay" in strategies:
            self.replay_buffer = ImportanceReplayBuffer(replay_size)

        if "ewc" in strategies:
            self.ewc_components = []
            self.ewc_lambda = ewc_lambda

        if "lwf" in strategies:
            self.old_models = []
            self.lwf_alpha = lwf_alpha

    def train_task(
        self,
        task_data,
        task_name,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5
    ):
        """訓練新任務"""

        print(f"\n訓練任務: {task_name}")
        print(f"數據量: {len(task_data)}")

        # 準備數據
        train_data = self._prepare_training_data(task_data)

        # 如果是持續學習（不是第一個任務）
        is_continual = len(self.task_history) > 0

        if is_continual:
            # 設置防遺忘機制
            self._setup_forgetting_prevention(task_data)

        # 訓練
        self._train(train_data, num_epochs, batch_size, learning_rate, is_continual)

        # 訓練後處理
        self._post_training(task_data, task_name)

        # 評估所有歷史任務
        self._evaluate_all_tasks()

    def _prepare_training_data(self, task_data):
        """準備訓練數據"""

        # 混入重放樣本
        if "replay" in self.strategies and len(self.task_history) > 0:
            n_replay = min(len(task_data), len(self.replay_buffer.buffer))
            replay_samples = self.replay_buffer.sample(n_replay)

            print(f"混入 {len(replay_samples)} 個重放樣本")
            train_data = task_data + replay_samples
        else:
            train_data = task_data

        return train_data

    def _setup_forgetting_prevention(self, task_data):
        """設置防遺忘機制"""

        # LwF: 保存舊模型
        if "lwf" in self.strategies:
            old_model = copy.deepcopy(self.model)
            old_model.eval()
            for param in old_model.parameters():
                param.requires_grad = False
            self.old_models.append(old_model)
            print("已保存舊模型用於 LwF")

        # EWC: 計算 Fisher 信息
        if "ewc" in self.strategies:
            # 使用上一個任務的數據計算 Fisher
            prev_task_data = self.task_history[-1]["data"]
            ewc = EWC(self.model, self._create_dataloader(prev_task_data))
            self.ewc_components.append(ewc)
            print("已計算 Fisher 信息用於 EWC")

    def _train(self, train_data, num_epochs, batch_size, learning_rate, is_continual):
        """執行訓練"""

        # 如果是持續學習，使用自定義訓練循環
        if is_continual and len(self.strategies) > 0:
            self._continual_training_loop(train_data, num_epochs, batch_size, learning_rate)
        else:
            # 標準訓練
            self._standard_training(train_data, num_epochs, batch_size, learning_rate)

    def _continual_training_loop(self, train_data, num_epochs, batch_size, lr):
        """持續學習訓練循環"""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        dataloader = self._create_dataloader(train_data, batch_size)

        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            task_loss_sum = 0
            ewc_loss_sum = 0
            lwf_loss_sum = 0

            for batch in dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                # 1. 任務損失
                outputs = self.model(**batch)
                task_loss = outputs.loss

                # 2. EWC 損失
                ewc_loss = 0
                if "ewc" in self.strategies and len(self.ewc_components) > 0:
                    for ewc in self.ewc_components:
                        ewc_loss += ewc.penalty()
                    ewc_loss *= self.ewc_lambda

                # 3. LwF 損失
                lwf_loss = 0
                if "lwf" in self.strategies and len(self.old_models) > 0:
                    for old_model in self.old_models:
                        with torch.no_grad():
                            old_outputs = old_model(**batch)

                        # KL 散度
                        lwf_loss += self._compute_kl_div(outputs.logits, old_outputs.logits)

                # 總損失
                if "lwf" in self.strategies and len(self.old_models) > 0:
                    loss = self.lwf_alpha * task_loss + (1 - self.lwf_alpha) * lwf_loss + ewc_loss
                else:
                    loss = task_loss + ewc_loss

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 記錄
                total_loss += loss.item()
                task_loss_sum += task_loss.item()
                if isinstance(ewc_loss, torch.Tensor):
                    ewc_loss_sum += ewc_loss.item()
                if isinstance(lwf_loss, torch.Tensor):
                    lwf_loss_sum += lwf_loss.item()

            # 打印 epoch 統計
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  總損失: {total_loss/len(dataloader):.4f}")
            print(f"  任務損失: {task_loss_sum/len(dataloader):.4f}")
            if ewc_loss_sum > 0:
                print(f"  EWC 損失: {ewc_loss_sum/len(dataloader):.4f}")
            if lwf_loss_sum > 0:
                print(f"  LwF 損失: {lwf_loss_sum/len(dataloader):.4f}")

    def _standard_training(self, train_data, num_epochs, batch_size, lr):
        """標準訓練（第一個任務）"""

        # 使用 Hugging Face Trainer
        dataset = Dataset.from_list(train_data)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized = dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir="./tmp",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=100
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized
        )

        trainer.train()

    def _post_training(self, task_data, task_name):
        """訓練後處理"""

        # 添加樣本到重放緩衝區
        if "replay" in self.strategies:
            # 計算樣本重要性
            importances = self._compute_sample_importance(task_data)

            # 選擇重要樣本添加到緩衝區
            n_samples = min(200, len(task_data))  # 每個任務保存 200 個樣本
            top_indices = np.argsort(importances)[-n_samples:]

            selected_samples = [task_data[i] for i in top_indices]
            selected_importances = [importances[i] for i in top_indices]

            self.replay_buffer.add(selected_samples, selected_importances)
            print(f"添加了 {len(selected_samples)} 個樣本到重放緩衝區")

        # 記錄任務
        self.task_history.append({
            "name": task_name,
            "data": task_data[:100]  # 保存部分數據用於評估
        })

    def _evaluate_all_tasks(self):
        """評估所有歷史任務"""

        print("\n評估所有任務:")
        print("-" * 50)

        for task_info in self.task_history:
            task_name = task_info["name"]
            task_data = task_info["data"]

            # 計算困惑度或其他指標
            metric = self._evaluate_task(task_data)
            print(f"{task_name}: {metric:.2f}")

        print("-" * 50)

    def _evaluate_task(self, task_data):
        """評估單個任務"""
        # 簡化版：計算平均損失
        dataloader = self._create_dataloader(task_data, batch_size=4)

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        self.model.train()

        return total_loss / len(dataloader)

    def _compute_sample_importance(self, task_data):
        """計算樣本重要性"""
        importances = []

        self.model.eval()
        with torch.no_grad():
            for example in task_data:
                # 使用損失作為重要性
                inputs = self.tokenizer(example["text"], return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                importances.append(outputs.loss.item())

        self.model.train()

        return importances

    def _create_dataloader(self, data, batch_size=4):
        """創建數據加載器"""
        from torch.utils.data import DataLoader

        dataset = Dataset.from_list(data)

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )

        tokenized = dataset.map(tokenize, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

        return DataLoader(tokenized, batch_size=batch_size, shuffle=True)

    def _compute_kl_div(self, logits_new, logits_old, temperature=2.0):
        """計算 KL 散度"""
        soft_targets = F.softmax(logits_old / temperature, dim=-1)
        soft_outputs = F.log_softmax(logits_new / temperature, dim=-1)

        return F.kl_div(soft_outputs, soft_targets, reduction='batchmean') * (temperature ** 2)


# 使用示例
if __name__ == "__main__":
    # 初始化持續學習器
    learner = ComprehensiveContinualLearner(
        model_name="gpt2",
        strategies=["replay", "ewc", "lwf"],
        replay_size=1000,
        ewc_lambda=1000,
        lwf_alpha=0.5
    )

    # 準備任務數據
    task_a_data = [{"text": "..."} for _ in range(1000)]
    task_b_data = [{"text": "..."} for _ in range(1000)]
    task_c_data = [{"text": "..."} for _ in range(1000)]

    # 依次訓練
    learner.train_task(task_a_data, "客服對話", num_epochs=3)
    learner.train_task(task_b_data, "代碼生成", num_epochs=3)
    learner.train_task(task_c_data, "文本摘要", num_epochs=3)
```

---

## 評估和監控

### 評估指標

1. **平均準確率 (Average Accuracy)**
   ```
   AA = (1/T) Σ acc_i
   ```
   所有任務的平均性能

2. **遺忘度 (Forgetting Measure)**
   ```
   F = (1/(T-1)) Σ (acc_i,max - acc_i,T)
   ```
   測量性能下降程度

3. **向後遷移 (Backward Transfer)**
   ```
   BWT = (1/(T-1)) Σ (acc_i,T - acc_i,i)
   ```
   新任務對舊任務的影響

4. **向前遷移 (Forward Transfer)**
   ```
   FWT = (1/(T-1)) Σ (acc_i,i-1 - acc_i,random)
   ```
   舊任務對新任務的幫助

### 監控腳本

```python
class ContinualLearningMonitor:
    """持續學習監控器"""

    def __init__(self):
        self.task_metrics = {}  # {task_name: [acc_after_task_1, acc_after_task_2, ...]}

    def record(self, current_task_id, task_accuracies):
        """記錄當前時刻所有任務的準確率"""
        for task_name, acc in task_accuracies.items():
            if task_name not in self.task_metrics:
                self.task_metrics[task_name] = []
            self.task_metrics[task_name].append(acc)

    def compute_forgetting(self):
        """計算遺忘度"""
        forgetting_scores = {}

        for task_name, accs in self.task_metrics.items():
            if len(accs) > 1:
                max_acc = max(accs)
                current_acc = accs[-1]
                forgetting = max_acc - current_acc
                forgetting_scores[task_name] = forgetting

        return forgetting_scores

    def compute_average_accuracy(self):
        """計算平均準確率"""
        current_accs = [accs[-1] for accs in self.task_metrics.values()]
        return np.mean(current_accs)

    def plot_learning_curve(self):
        """繪製學習曲線"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        for task_name, accs in self.task_metrics.items():
            plt.plot(accs, marker='o', label=task_name)

        plt.xlabel('訓練任務數')
        plt.ylabel('準確率')
        plt.title('持續學習曲線')
        plt.legend()
        plt.grid(True)
        plt.savefig('continual_learning_curve.png')
        plt.close()


# 使用示例
monitor = ContinualLearningMonitor()

# 訓練任務 A
train(task_a)
monitor.record(0, {"task_a": 0.95})

# 訓練任務 B
train(task_b)
accs = {"task_a": 0.88, "task_b": 0.92}  # 任務 A 性能下降
monitor.record(1, accs)

# 訓練任務 C
train(task_c)
accs = {"task_a": 0.85, "task_b": 0.89, "task_c": 0.94}
monitor.record(2, accs)

# 分析
print(f"平均準確率: {monitor.compute_average_accuracy():.2f}")
print(f"遺忘度: {monitor.compute_forgetting()}")
monitor.plot_learning_curve()
```

---

## 最佳實踐

### 1. 選擇合適的策略

根據場景選擇：

| 場景 | 推薦策略 | 原因 |
|------|---------|------|
| 可以保存少量歷史數據 | Replay + EWC | 平衡性能和成本 |
| 完全無法保存歷史數據 | EWC + LwF | 純正則化方法 |
| 有充足計算資源 | Replay + LwF + EWC | 最佳性能 |
| 任務差異大 | Adapter Tuning | 避免干擾 |
| 在線學習場景 | Replay + 動態權重 | 適應流式數據 |

### 2. 超參數調優

關鍵超參數：

```python
# 重放緩衝區大小
replay_size = 1000  # 每個任務 100-200 個樣本

# EWC lambda
ewc_lambda = 1000  # 範圍：100-10000

# LwF alpha
lwf_alpha = 0.5  # 範圍：0.3-0.7

# 學習率
learning_rate = 1e-5  # 比標準微調更小
```

### 3. 數據管理

- **重放數據選擇**：選擇困難樣本或代表性樣本
- **數據質量**：確保重放數據質量高
- **數據平衡**：保持任務間的平衡

### 4. 定期評估

- **每個任務訓練後**：評估所有歷史任務
- **監控遺忘度**：及時發現災難性遺忘
- **調整策略**：根據評估結果調整超參數

### 5. 漸進式方法

- **從簡單到複雜**：先訓練簡單任務
- **相關任務優先**：先訓練相關任務
- **定期鞏固**：定期重新訓練關鍵任務

---

## 參考資源

- [Continual Learning Survey](https://arxiv.org/abs/1909.08383)
- [Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796)
- [Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)
