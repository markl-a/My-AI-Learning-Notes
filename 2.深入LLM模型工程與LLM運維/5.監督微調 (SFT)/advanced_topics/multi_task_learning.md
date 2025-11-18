# 多任務學習 (Multi-Task Learning) in SFT

## 目錄

1. [什麼是多任務學習](#什麼是多任務學習)
2. [多任務學習的優勢](#多任務學習的優勢)
3. [實現策略](#實現策略)
4. [數據準備](#數據準備)
5. [訓練技巧](#訓練技巧)
6. [實作範例](#實作範例)
7. [常見挑戰與解決方案](#常見挑戰與解決方案)

---

## 什麼是多任務學習

多任務學習 (MTL) 是指在單一模型上同時訓練多個相關任務，讓模型學習不同任務之間的共享知識。

### 核心概念

在 SFT 的場景中，多任務學習意味著：

- **單一模型處理多種任務**：問答、摘要、翻譯、代碼生成等
- **共享表示學習**：不同任務共享底層的語言理解能力
- **任務特定適配**：為不同任務學習特定的模式

### 與單任務學習的對比

| 特性 | 單任務學習 | 多任務學習 |
|------|-----------|-----------|
| 訓練數據 | 單一任務 | 多個任務混合 |
| 模型專精度 | 高（單任務） | 中等（多任務平衡） |
| 泛化能力 | 有限 | 更強 |
| 訓練成本 | 低 | 中等 |
| 部署複雜度 | 每個任務一個模型 | 單一模型服務多個任務 |

---

## 多任務學習的優勢

### 1. 提升泛化能力

不同任務的訓練信號可以互相補充，減少過擬合：

```
任務 1: 問答 → 學習理解問題和給出答案
任務 2: 摘要 → 學習提取關鍵信息
任務 3: 翻譯 → 學習語言對應關係

共享學習: 更好的語言理解和生成能力
```

### 2. 數據效率

即使某個任務的數據較少，也能從其他任務中獲益：

- **知識遷移**：從數據豐富的任務遷移知識到數據稀缺的任務
- **正則化效果**：多任務約束防止模型過度擬合單一任務

### 3. 降低部署成本

- **統一服務**：一個模型處理多個任務
- **資源節省**：減少 GPU 內存和計算資源需求
- **簡化維護**：只需維護一個模型

### 4. 任務間正遷移

相關任務可以互相促進：

```
問答 + 對話 → 提升交互理解能力
代碼生成 + 代碼解釋 → 增強代碼理解
翻譯 + 文本改寫 → 改善語言表達
```

---

## 實現策略

### 策略 1: 任務混合訓練 (Task Mixing)

**方法**：在訓練數據中混合不同任務的樣本

**實現**：
```python
# 準備不同任務的數據
qa_data = [...]  # 問答數據
summary_data = [...]  # 摘要數據
translation_data = [...]  # 翻譯數據

# 混合數據
mixed_data = qa_data + summary_data + translation_data

# 打亂順序
import random
random.shuffle(mixed_data)

# 訓練
trainer.train(mixed_data)
```

**優點**：
- 實現簡單
- 任務自然混合

**缺點**：
- 難以控制任務比例
- 可能被大數據集任務主導

### 策略 2: 任務採樣 (Task Sampling)

**方法**：為每個任務設置採樣權重

**實現**：
```python
from torch.utils.data import WeightedRandomSampler

# 定義任務權重
task_weights = {
    "qa": 0.4,        # 40%
    "summary": 0.3,   # 30%
    "translation": 0.3  # 30%
}

# 創建加權採樣器
weights = [task_weights[example["task_type"]] for example in dataset]
sampler = WeightedRandomSampler(weights, len(dataset))

# 使用採樣器訓練
dataloader = DataLoader(dataset, sampler=sampler)
```

**優點**：
- 精確控制任務比例
- 平衡不同大小的數據集

**缺點**：
- 需要手動調整權重
- 可能需要實驗確定最佳比例

### 策略 3: 溫度採樣 (Temperature Sampling)

**方法**：使用溫度參數平滑任務分佈

**公式**：
```
P(task_i) = (N_i)^(1/T) / Σ(N_j)^(1/T)
```

其中：
- N_i：任務 i 的數據量
- T：溫度參數（T=1 按原始比例，T→∞ 均勻分佈）

**實現**：
```python
import numpy as np

def temperature_sampling(task_sizes, temperature=0.7):
    """使用溫度採樣計算任務權重"""
    probs = np.array([size ** (1/temperature) for size in task_sizes])
    probs = probs / probs.sum()
    return probs

# 示例
task_sizes = [10000, 5000, 2000]  # QA, Summary, Translation
weights = temperature_sampling(task_sizes, temperature=0.7)
print(f"任務採樣權重: {weights}")
# 輸出: [0.52, 0.31, 0.17]（相比原始比例 [0.59, 0.29, 0.12] 更平衡）
```

**優點**：
- 自動平衡任務
- T 參數直觀易調

**缺點**：
- 需要實驗確定最佳溫度

### 策略 4: 任務提示 (Task Prompting)

**方法**：在輸入中明確標識任務類型

**示例**：
```python
# 方式 1: 任務前綴
def add_task_prefix(instruction, task_type):
    task_prefixes = {
        "qa": "[問答] ",
        "summary": "[摘要] ",
        "translation": "[翻譯] "
    }
    return task_prefixes[task_type] + instruction

# 方式 2: 任務 Token
def add_task_token(text, task_type):
    task_tokens = {
        "qa": "<|qa|>",
        "summary": "<|summary|>",
        "translation": "<|translation|>"
    }
    return task_tokens[task_type] + text

# 方式 3: 自然語言描述
def add_task_description(instruction, task_type):
    descriptions = {
        "qa": "請回答以下問題：",
        "summary": "請總結以下內容：",
        "translation": "請翻譯以下文本："
    }
    return descriptions[task_type] + "\n" + instruction
```

**優點**：
- 明確任務邊界
- 便於任務切換

**缺點**：
- 增加輸入長度
- 需要在推理時也使用相同格式

---

## 數據準備

### 1. 統一數據格式

所有任務使用相同的數據結構：

```python
{
    "task_type": "qa",  # 任務類型
    "instruction": "解釋什麼是深度學習",
    "input": "",
    "output": "深度學習是機器學習的一個子領域...",
    "metadata": {
        "domain": "AI",
        "difficulty": "medium"
    }
}
```

### 2. 任務標註

為每個樣本添加任務標籤：

```python
def annotate_task_type(dataset, task_type):
    """為數據集添加任務類型標註"""
    for example in dataset:
        example["task_type"] = task_type
        example["task_id"] = get_task_id(task_type)
    return dataset

# 示例
qa_data = annotate_task_type(qa_data, "qa")
summary_data = annotate_task_type(summary_data, "summary")
```

### 3. 數據平衡

確保任務之間的平衡：

```python
def balance_multi_task_dataset(datasets, strategy="undersample"):
    """平衡多任務數據集"""

    if strategy == "undersample":
        # 下採樣：所有任務使用最小數據集的大小
        min_size = min(len(d) for d in datasets)
        balanced = [d[:min_size] for d in datasets]

    elif strategy == "oversample":
        # 上採樣：重複樣本使所有任務數據量一致
        max_size = max(len(d) for d in datasets)
        balanced = []
        for d in datasets:
            if len(d) < max_size:
                # 重複採樣
                repeats = max_size // len(d) + 1
                d_extended = (d * repeats)[:max_size]
                balanced.append(d_extended)
            else:
                balanced.append(d)

    elif strategy == "weighted":
        # 加權採樣：按比例採樣
        # 實現見上文的溫度採樣

    return balanced
```

### 4. 質量過濾

確保每個任務的數據質量：

```python
def filter_low_quality(dataset, task_type):
    """過濾低質量樣本"""
    filtered = []

    for example in dataset:
        # 基本檢查
        if not example["output"].strip():
            continue

        # 任務特定檢查
        if task_type == "qa":
            # 問答：答案不能太短
            if len(example["output"]) < 10:
                continue
        elif task_type == "summary":
            # 摘要：摘要應短於原文
            if len(example["output"]) >= len(example["input"]):
                continue

        filtered.append(example)

    return filtered
```

---

## 訓練技巧

### 1. 分階段訓練

**策略**：先單任務預熱，再多任務混合訓練

```python
# 階段 1: 各任務獨立預熱 (warm-up)
for task_data in [qa_data, summary_data, translation_data]:
    trainer.train(task_data, epochs=1)

# 階段 2: 多任務混合訓練
mixed_data = qa_data + summary_data + translation_data
trainer.train(mixed_data, epochs=3)
```

**優勢**：
- 避免任務衝突
- 確保每個任務都得到學習

### 2. 梯度累積策略

**策略**：為不同任務使用不同的梯度累積步數

```python
task_accumulation_steps = {
    "qa": 4,
    "summary": 2,
    "translation": 1
}

def train_with_task_accumulation(model, dataloader):
    for batch in dataloader:
        task_type = batch["task_type"][0]
        accum_steps = task_accumulation_steps[task_type]

        # 計算損失
        loss = model(batch)
        loss = loss / accum_steps
        loss.backward()

        if step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### 3. 任務特定學習率

**策略**：為不同任務設置不同的學習率

```python
from torch.optim import AdamW

# 為不同任務的參數組設置不同學習率
optimizer = AdamW([
    {'params': qa_params, 'lr': 1e-4},
    {'params': summary_params, 'lr': 2e-4},
    {'params': translation_params, 'lr': 1.5e-4},
])
```

### 4. 動態任務權重

**策略**：根據任務性能動態調整採樣權重

```python
def adjust_task_weights(task_losses, alpha=0.5):
    """根據損失動態調整任務權重"""
    # 損失高的任務獲得更多訓練
    inv_losses = [1 / (loss + 1e-8) for loss in task_losses]
    weights = np.array(inv_losses) ** alpha
    weights = weights / weights.sum()
    return weights

# 訓練循環中
task_losses = evaluate_tasks(model, val_data)
task_weights = adjust_task_weights(task_losses)
sampler = create_weighted_sampler(train_data, task_weights)
```

### 5. 任務難度課程學習

**策略**：從簡單任務逐步過渡到困難任務

```python
# 定義任務難度
task_difficulty = {
    "qa_easy": 1,
    "summary": 2,
    "qa_hard": 3,
    "translation": 4
}

# 課程學習調度
def curriculum_schedule(epoch, total_epochs):
    """返回當前 epoch 應該訓練的任務"""
    progress = epoch / total_epochs
    max_difficulty = 1 + progress * 3  # 從 1 逐步增加到 4

    available_tasks = [
        task for task, diff in task_difficulty.items()
        if diff <= max_difficulty
    ]
    return available_tasks
```

---

## 實作範例

### 完整的多任務訓練腳本

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset, concatenate_datasets
from torch.utils.data import WeightedRandomSampler
import numpy as np


class MultiTaskDataset:
    """多任務數據集"""

    def __init__(self, task_datasets, task_weights=None, temperature=1.0):
        """
        Args:
            task_datasets: 字典 {task_name: dataset}
            task_weights: 任務權重（如果為 None，使用溫度採樣）
            temperature: 溫度參數
        """
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())

        # 計算任務權重
        if task_weights is None:
            task_sizes = [len(d) for d in task_datasets.values()]
            self.task_weights = self._temperature_sampling(task_sizes, temperature)
        else:
            self.task_weights = task_weights

        # 合併數據集並添加任務標籤
        self.dataset = self._merge_datasets()

    def _temperature_sampling(self, sizes, temperature):
        """溫度採樣"""
        probs = np.array([s ** (1/temperature) for s in sizes])
        return probs / probs.sum()

    def _merge_datasets(self):
        """合併數據集"""
        merged = []
        for task_name, dataset in self.task_datasets.items():
            for example in dataset:
                example["task_type"] = task_name
                merged.append(example)
        return merged

    def get_weighted_sampler(self):
        """獲取加權採樣器"""
        # 為每個樣本分配權重
        task_to_weight = dict(zip(self.task_names, self.task_weights))
        weights = [task_to_weight[ex["task_type"]] for ex in self.dataset]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        return sampler


def format_multi_task_example(example):
    """格式化多任務樣本"""
    task_type = example["task_type"]

    # 任務前綴
    task_prefixes = {
        "qa": "問答任務",
        "summary": "摘要任務",
        "translation": "翻譯任務",
        "code": "代碼生成任務"
    }

    prefix = task_prefixes.get(task_type, "")

    # 構建提示
    if example.get("input"):
        prompt = f"{prefix}\n\n指令: {example['instruction']}\n輸入: {example['input']}\n回答: {example['output']}"
    else:
        prompt = f"{prefix}\n\n指令: {example['instruction']}\n回答: {example['output']}"

    return {"text": prompt}


def train_multi_task_model(
    model_name,
    task_datasets,
    output_dir="./multi_task_model",
    temperature=0.7,
    num_epochs=3
):
    """訓練多任務模型"""

    # 載入模型和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 創建多任務數據集
    mt_dataset = MultiTaskDataset(task_datasets, temperature=temperature)

    # 格式化數據
    formatted_data = [format_multi_task_example(ex) for ex in mt_dataset.dataset]

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 創建採樣器
    sampler = mt_dataset.get_weighted_sampler()

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=100,
        save_steps=500,
        fp16=True,
        report_to="none"
    )

    # 訓練
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # 注意：Trainer 不直接支持自定義 sampler
        # 需要使用 DataLoader 或自定義 Trainer
    )

    trainer.train()
    trainer.save_model(output_dir)

    return model, tokenizer


# 使用示例
if __name__ == "__main__":
    # 準備不同任務的數據
    task_datasets = {
        "qa": [
            {"instruction": "什麼是 Python?", "input": "", "output": "Python 是一種高級編程語言..."},
            # ... 更多 QA 數據
        ],
        "summary": [
            {"instruction": "總結以下文章", "input": "文章內容...", "output": "摘要..."},
            # ... 更多摘要數據
        ],
        "translation": [
            {"instruction": "翻譯成英文", "input": "你好世界", "output": "Hello World"},
            # ... 更多翻譯數據
        ]
    }

    # 訓練多任務模型
    model, tokenizer = train_multi_task_model(
        model_name="gpt2",
        task_datasets=task_datasets,
        temperature=0.7,
        num_epochs=3
    )
```

---

## 常見挑戰與解決方案

### 挑戰 1: 任務衝突 (Task Conflict)

**問題**：不同任務的優化目標可能相互矛盾

**症狀**：
- 某些任務性能下降
- 訓練不穩定
- 損失震盪

**解決方案**：

1. **梯度手術 (Gradient Surgery)**
   ```python
   def project_conflicting_gradients(grad1, grad2):
       """投影衝突的梯度"""
       # 如果梯度方向衝突（夾角 > 90°）
       if torch.dot(grad1.flatten(), grad2.flatten()) < 0:
           # 投影到正交方向
           grad2 = grad2 - (torch.dot(grad1, grad2) / torch.dot(grad1, grad1)) * grad1
       return grad2
   ```

2. **任務分組**：將相似任務分組訓練

3. **使用更大的模型**：增加模型容量以容納多個任務

### 挑戰 2: 負遷移 (Negative Transfer)

**問題**：某個任務的加入反而降低其他任務的性能

**解決方案**：

1. **任務相關性分析**：只組合相關任務
2. **任務特定層**：為每個任務保留專門的層
3. **自適應任務權重**：動態調整任務重要性

### 挑戰 3: 數據不平衡

**問題**：不同任務的數據量差異大

**解決方案**：

1. **溫度採樣**（如前所述）
2. **分階段訓練**：先訓練數據少的任務
3. **數據增強**：為數據少的任務生成更多樣本

### 挑戰 4: 評估複雜性

**問題**：需要評估多個任務的性能

**解決方案**：

```python
def evaluate_multi_task_model(model, task_val_datasets):
    """評估多任務模型"""
    results = {}

    for task_name, val_data in task_val_datasets.items():
        # 任務特定評估
        if task_name == "qa":
            metric = evaluate_qa(model, val_data)
        elif task_name == "summary":
            metric = evaluate_summary(model, val_data)
        # ...

        results[task_name] = metric

    # 綜合分數
    results["average"] = np.mean(list(results.values()))

    return results
```

---

## 最佳實踐總結

1. **任務選擇**：
   - 選擇相關任務進行多任務學習
   - 避免完全不相關的任務組合

2. **數據準備**：
   - 統一數據格式
   - 平衡任務數據量
   - 確保數據質量

3. **訓練策略**：
   - 使用溫度採樣平衡任務
   - 監控各任務的性能
   - 必要時使用分階段訓練

4. **超參數調整**：
   - 從較低的學習率開始
   - 調整溫度參數 (0.5-1.0)
   - 根據任務調整 batch size

5. **評估和監控**：
   - 分別評估每個任務
   - 關注任務間的平衡
   - 警惕負遷移現象

---

## 參考資源

- [An Overview of Multi-Task Learning in Deep Neural Networks](https://ruder.io/multi-task/)
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
- [FLAN: Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
