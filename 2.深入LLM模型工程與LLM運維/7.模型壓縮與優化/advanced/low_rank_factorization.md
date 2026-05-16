# 低秩分解 (Low-Rank Factorization)

## 目錄
1. [基本概念](#1-基本概念)
2. [理論基礎](#2-理論基礎)
3. [常見低秩分解方法](#3-常見低秩分解方法)
4. [LoRA 詳解](#4-lora-詳解)
5. [其他變體](#5-其他變體)
6. [實作範例](#6-實作範例)
7. [性能分析](#7-性能分析)
8. [最佳實踐](#8-最佳實踐)

---

## 1. 基本概念

### 1.1 什麼是低秩分解？

**低秩分解**是將大型參數矩陣分解為多個較小矩陣的乘積，利用矩陣的低秩特性來減少參數量。

**核心思想**：
```
原始權重矩陣：W ∈ ℝ^(d×k)  （d×k 個參數）
低秩分解：W ≈ A·B
  其中 A ∈ ℝ^(d×r), B ∈ ℝ^(r×k)
  參數量：d×r + r×k << d×k （當 r << min(d,k)）
```

**壓縮比**：
```
壓縮比 = (d×k) / (d×r + r×k)

例如：d=4096, k=4096, r=64
原始參數：4096 × 4096 = 16,777,216
低秩參數：4096 × 64 + 64 × 4096 = 524,288
壓縮比：32x
```

### 1.2 為什麼有效？

**理論依據**：
1. **內在維度假設**：深度神經網路的有效參數空間遠小於實際參數空間
2. **矩陣秩特性**：訓練後的權重矩陣往往是低秩或近似低秩的
3. **冗餘性**：大型模型存在大量參數冗餘

**經驗證據**：
- BERT 的注意力權重矩陣秩通常遠小於其維度
- GPT 模型的 FFN 層存在低秩結構
- LoRA 實驗表明秩 r=8 就能達到良好效果

---

## 2. 理論基礎

### 2.1 奇異值分解 (SVD)

**定義**：
```
對於任意矩陣 W ∈ ℝ^(m×n)，存在分解：
W = UΣV^T

其中：
- U ∈ ℝ^(m×m)：左奇異向量矩陣（正交矩陣）
- Σ ∈ ℝ^(m×n)：奇異值對角矩陣
- V ∈ ℝ^(n×n)：右奇異向量矩陣（正交矩陣）
```

**低秩近似**：
```
保留前 r 個最大奇異值：
W_r = U_r Σ_r V_r^T

其中：
- U_r ∈ ℝ^(m×r)：前 r 個左奇異向量
- Σ_r ∈ ℝ^(r×r)：前 r 個奇異值
- V_r ∈ ℝ^(n×r)：前 r 個右奇異向量
```

**Eckart-Young 定理**：
SVD 給出的低秩近似是最優的（Frobenius 範數意義下）。

### 2.2 矩陣的秩

**定義**：
```
rank(W) = 矩陣 W 的線性獨立行（或列）的最大數量
```

**性質**：
```
1. rank(W) ≤ min(m, n)
2. rank(AB) ≤ min(rank(A), rank(B))
3. 大部分實際矩陣是滿秩或近似滿秩的
4. 但有效秩（effective rank）可能遠小於矩陣維度
```

**有效秩**：
```
考慮奇異值衰減：σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0

有效秩定義為能量集中的奇異值數量：
r_eff = argmin_r { Σᵢ₌₁ʳ σᵢ² / Σᵢ₌₁ⁿ σᵢ² ≥ 0.9 }

即保留 90% 能量所需的最少奇異值數量
```

### 2.3 參數效率

**參數效率比較**：
```
方法                    參數量
全微調                  n (所有參數)
低秩分解 (r=8)          (d+k)×r ≈ 0.1% ~ 1% 全微調
Adapter                 2×d×r + r + d ≈ 0.5% 全微調
Prefix Tuning          l×h×2 ≈ 0.1% 全微調
```

---

## 3. 常見低秩分解方法

### 3.1 直接 SVD 分解

**方法**：
1. 對預訓練權重 W 進行 SVD
2. 保留前 r 個奇異值
3. 重構低秩矩陣

**優點**：
- 理論最優（Frobenius 範數）
- 不需要訓練

**缺點**：
- 靜態方法，不能適應新任務
- 精度損失可能較大
- 計算成本高（對大矩陣）

**PyTorch 實作**：
```python
import torch

def svd_decomposition(weight, rank):
    """使用 SVD 進行低秩分解"""
    # SVD 分解
    U, S, V = torch.svd(weight)

    # 保留前 rank 個成分
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]

    # 重構
    A = U_r @ torch.diag(torch.sqrt(S_r))
    B = torch.diag(torch.sqrt(S_r)) @ V_r.T

    return A, B

# 使用範例
weight = torch.randn(4096, 4096)
A, B = svd_decomposition(weight, rank=64)

print(f"原始參數: {weight.numel():,}")
print(f"分解後參數: {A.numel() + B.numel():,}")
print(f"壓縮比: {weight.numel() / (A.numel() + B.numel()):.2f}x")
```

### 3.2 Tucker 分解

**方法**：
高階張量的多線性分解。

**應用**：
- 卷積層壓縮
- 對 LLM 較少使用

### 3.3 CP 分解

**方法**：
張量的 CANDECOMP/PARAFAC 分解。

**應用**：
- 主要用於卷積神經網路
- LLM 中應用有限

---

## 4. LoRA 詳解

### 4.1 LoRA 原理

**Low-Rank Adaptation (LoRA)** 是目前最流行的參數高效微調方法。

**核心思想**：
```
凍結預訓練權重 W₀，添加低秩更新：
h = W₀x + ΔWx = W₀x + BAx

其中：
- W₀ ∈ ℝ^(d×k)：凍結的預訓練權重
- B ∈ ℝ^(d×r)：可訓練的低秩矩陣
- A ∈ ℝ^(r×k)：可訓練的低秩矩陣
- r << min(d, k)：秩（超參數）
```

**關鍵設計**：
1. **初始化**：
   - A：高斯隨機初始化
   - B：零初始化
   - 確保訓練開始時 ΔW = BA = 0

2. **縮放因子**：
   ```
   ΔW = (α/r) × BA
   ```
   其中 α 是可調整的超參數（通常設為 r）

3. **應用位置**：
   - 僅應用於注意力層的 Q, K, V, O 投影
   - 可選：FFN 層

### 4.2 LoRA 優勢

**1. 參數效率高**：
```
7B 模型全微調：7B 參數
7B 模型 LoRA (r=8)：~4M 參數（0.06%）
```

**2. 推理無額外開銷**：
```
訓練：h = W₀x + BAx（兩個矩陣）
推理：W = W₀ + BA（合併後一個矩陣）
```

**3. 任務切換靈活**：
```
- 可以保存多個 LoRA 適配器
- 快速切換不同任務（僅交換 BA）
- 基座模型保持不變
```

**4. 顯存效率**：
- 不需要存儲大部分梯度
- 優化器狀態僅針對少量參數
- 可與量化結合（QLoRA）

### 4.3 LoRA 超參數

**秩 (r)**：
```
r=1~4   : 極限壓縮，性能可能下降
r=8     : 推薦起點，多數任務表現良好
r=16~32 : 複雜任務，接近全微調性能
r=64+   : 高性能要求，參數效率降低
```

**Alpha (α)**：
```
α = r   : 預設設置
α = 2r  : 更大的學習率
α = r/2 : 更保守的更新
```

**目標模塊**：
```
最小配置：query, value (Q, V)
推薦配置：query, key, value, output (Q, K, V, O)
最大配置：所有線性層（包括 FFN）
```

**Dropout**：
```
lora_dropout = 0.05  # 輕微正則化
lora_dropout = 0.1   # 標準設置
```

### 4.4 LoRA 實作

**使用 PEFT 庫**：
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 載入基座模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # 秩
    lora_alpha=16,                # 縮放因子
    lora_dropout=0.05,            # Dropout
    target_modules=[              # 目標模塊
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj",  # 可選：FFN 層
        # "up_proj",
        # "down_proj",
    ],
    bias="none",                  # 偏置處理
)

# 應用 LoRA
model = get_peft_model(model, lora_config)

# 查看可訓練參數
model.print_trainable_parameters()
# 輸出：trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

**手動實作 LoRA**：
```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """LoRA 層實作"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 低秩矩陣
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        """前向傳播"""
        # ΔW = BA
        delta_w = self.lora_B @ self.lora_A
        return self.dropout(x) @ delta_w.T * self.scaling

class LinearWithLoRA(nn.Module):
    """帶 LoRA 的線性層"""
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # 凍結原始權重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        """h = W₀x + BAx"""
        return self.linear(x) + self.lora(x)

# 使用範例
original_layer = nn.Linear(4096, 4096)
lora_layer = LinearWithLoRA(original_layer, rank=8, alpha=16)

# 測試
x = torch.randn(2, 4096)
output = lora_layer(x)
print(f"輸出形狀: {output.shape}")
```

---

## 5. 其他變體

### 5.1 AdaLoRA

**自適應 LoRA**：動態調整不同層和模塊的秩。

**核心思想**：
```
- 重要模塊：分配更高的秩
- 不重要模塊：分配更低的秩或完全剪枝
- 訓練過程中動態調整
```

**優勢**：
- 更好的參數分配
- 可能獲得更好性能
- 自動化超參數調整

**實作**：
```python
from peft import AdaLoraConfig, get_peft_model

adalora_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    init_r=12,                    # 初始秩
    target_r=8,                   # 目標秩
    beta1=0.85,                   # 重要性估計參數
    beta2=0.85,
    tinit=200,                    # 開始剪枝的步數
    tfinal=1000,                  # 完成剪枝的步數
    deltaT=10,                    # 更新頻率
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, adalora_config)
```

### 5.2 QLoRA

**量化 + LoRA**：結合 4-bit 量化和 LoRA。

**詳細內容**：參見主 README 的 QLoRA 章節。

### 5.3 LoRA-FA

**Frozen-A LoRA**：凍結 A 矩陣，僅訓練 B。

**優勢**：
- 進一步減少參數
- 加速訓練

**缺點**：
- 可能降低性能

### 5.4 DoRA

**Weight-Decomposed LoRA**：將權重分解為幅度和方向。

**公式**：
```
W = m · (W₀ + BA) / ||W₀ + BA||

其中 m 是可學習的幅度向量
```

**優勢**：
- 更接近全微調性能
- 更穩定的訓練

### 5.5 LoRA+

**改進的 LoRA**：使用不同的學習率策略。

**關鍵**：
```
learning_rate_B = λ × learning_rate_A

通常 λ = 16
```

**優勢**：
- 更快收斂
- 更好性能

---

## 6. 實作範例

### 6.1 完整微調流程

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ============================================================================
# 1. 載入模型和資料
# ============================================================================
model_name = "meta-llama/Llama-2-7b-hf"

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 載入資料
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# ============================================================================
# 2. 配置 LoRA
# ============================================================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                         # 秩
    lora_alpha=32,                # alpha = 2 * r
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================================
# 3. 資料預處理
# ============================================================================
def preprocess_function(examples):
    """預處理資料"""
    texts = [
        f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]

    model_inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# ============================================================================
# 4. 訓練配置
# ============================================================================
training_args = TrainingArguments(
    output_dir="./lora-llama2-7b-alpaca",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    warmup_ratio=0.03,
)

# ============================================================================
# 5. 訓練
# ============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ============================================================================
# 6. 保存和載入
# ============================================================================
# 保存 LoRA 適配器（僅 4MB）
model.save_pretrained("./lora-adapter")

# 載入
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# 合併權重（可選）
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### 6.2 多任務 LoRA

```python
from peft import PeftModel

# 訓練多個任務的 LoRA 適配器
tasks = ["summarization", "translation", "qa"]

for task in tasks:
    # 訓練特定任務的 LoRA
    # ... 訓練程式碼 ...
    model.save_pretrained(f"./lora-{task}")

# 使用時動態切換
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 任務 1：摘要
model = PeftModel.from_pretrained(base_model, "./lora-summarization")
# ... 執行摘要 ...

# 任務 2：翻譯
model = PeftModel.from_pretrained(base_model, "./lora-translation")
# ... 執行翻譯 ...
```

---

## 7. 性能分析

### 7.1 參數效率

**LLaMA-7B 範例**：
```
模型大小：7B 參數
全微調參數：7,000,000,000

LoRA (r=8):
- Q, K, V, O 每層：4 × (4096×8 + 8×4096) = 262,144
- 32 層：32 × 262,144 = 8,388,608
- 佔比：0.12%

LoRA (r=16):
- 單層：4 × (4096×16 + 16×4096) = 524,288
- 32 層：16,777,216
- 佔比：0.24%
```

### 7.2 顯存佔用

**訓練顯存對比**（LLaMA-7B，batch_size=1）：
```
全微調 (FP16)：~120 GB
  - 模型參數：14 GB
  - 梯度：14 GB
  - 優化器狀態（AdamW）：28 GB × 2 = 56 GB
  - 激活值：~36 GB

LoRA (r=8, FP16)：~20 GB
  - 模型參數（凍結）：14 GB
  - LoRA 參數：~16 MB
  - LoRA 梯度：~16 MB
  - LoRA 優化器：~32 MB
  - 激活值：~6 GB

QLoRA (r=8, 4-bit)：~6 GB
  - 模型參數（4-bit）：3.5 GB
  - LoRA 參數（FP16）：~16 MB
  - 其他：~2.5 GB
```

### 7.3 性能對比

**GLUE 基準測試（RoBERTa-large）**：
```
方法              參數量        平均分數
全微調            355M (100%)   90.2
BitFit            0.1%          89.5
Adapter           0.5%          89.8
Prefix Tuning     0.1%          89.6
LoRA (r=8)        0.3%          90.0
```

**LLaMA-7B 指令微調（Alpaca）**：
```
方法              參數量        ROUGE-L    推論速度
全微調            7B (100%)     0.432      1.0x
LoRA (r=8)        4M (0.06%)    0.428      1.0x
LoRA (r=16)       8M (0.11%)    0.430      1.0x
```

**觀察**：
- LoRA 能達到接近全微調的性能
- 秩 r=8 通常足夠
- 推論速度無差異（合併後）

---

## 8. 最佳實踐

### 8.1 選擇合適的秩

**指導原則**：
```
任務複雜度低（分類、情感分析）：r=4~8
任務複雜度中（NER、QA）：r=8~16
任務複雜度高（生成、推理）：r=16~32
資料量小：較小的 r（避免過擬合）
資料量大：可以使用較大的 r
```

**實驗策略**：
1. 從 r=8 開始
2. 觀察驗證集性能
3. 如果欠擬合，增加 r
4. 如果過擬合，減少 r 或增加 dropout

### 8.2 目標模塊選擇

**推薦配置**：
```
最小（最快）：["q_proj", "v_proj"]
平衡（推薦）：["q_proj", "k_proj", "v_proj", "o_proj"]
最大（最好）：所有線性層
```

**經驗**：
- Q 和 V 最重要
- 添加 K 和 O 通常能提升性能
- FFN 層收益遞減

### 8.3 超參數調優

**學習率**：
```
LoRA 通常需要更高的學習率：
- 基座模型（凍結）：不需要
- LoRA 參數：2e-4 ~ 3e-4（比全微調高 5-10 倍）
```

**Alpha**：
```
經驗公式：α = 2r
- r=8  → α=16
- r=16 → α=32
```

**批次大小**：
```
由於參數少，可以使用更大的批次：
- 全微調：batch_size=1~2
- LoRA：batch_size=4~8
```

### 8.4 常見陷阱

**陷阱 1：忘記合併權重**
```python
# 錯誤：保存整個模型（包括基座）
model.save_pretrained("./model")  # 保存 7GB+

# 正確：僅保存 LoRA 適配器
lora_model.save_pretrained("./lora-adapter")  # 僅 4MB
```

**陷阱 2：推理時未合併**
```python
# 低效：推理時仍計算 W₀x + BAx
output = model(input)

# 高效：合併為單個矩陣 W = W₀ + BA
merged_model = model.merge_and_unload()
output = merged_model(input)
```

**陷阱 3：秩過大**
```python
# 過大的秩失去參數效率優勢
config = LoraConfig(r=128)  # 可能不如全微調

# 合理的秩
config = LoraConfig(r=8)    # 通常足夠
```

### 8.5 與量化結合

**QLoRA 最佳實踐**：
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 載入量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 準備 LoRA 訓練
model = prepare_model_for_kbit_training(model)

# 應用 LoRA
lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)
```

---

## 總結

### 關鍵要點

1. **低秩分解利用矩陣的低秩結構**：
   - 大幅減少參數量
   - 保持模型性能
   - 降低計算和存儲成本

2. **LoRA 是當前最佳實踐**：
   - 參數效率：僅 0.1% 可訓練參數
   - 性能接近全微調
   - 易於實現和部署

3. **合理選擇超參數**：
   - 秩 r：從 8 開始
   - Alpha：通常設為 2r
   - 目標模塊：至少包含 Q, V

4. **與量化結合威力巨大**：
   - QLoRA 使得消費級 GPU 可微調大模型
   - 7B 模型僅需 6GB 顯存

5. **靈活的部署方式**：
   - 多任務切換
   - 參數合併
   - 增量更新

### 延伸閱讀

**論文**：
- LoRA: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- QLoRA: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- AdaLoRA: "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (Zhang et al., 2023)

**工具**：
- PEFT: https://github.com/huggingface/peft
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes

**資源**：
- Hugging Face PEFT 文檔: https://huggingface.co/docs/peft
- LoRA 論文解讀: https://arxiv.org/abs/2106.09685
