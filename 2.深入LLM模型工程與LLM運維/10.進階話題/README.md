# 進階話題

## 概述

本章涵蓋 LLM 領域的前沿技術和研究方向，包括位置編碼創新、模型融合策略、多模態整合以及開源實踐案例。這些進階話題代表了當前 LLM 研究的最新進展，對於構建高性能、長上下文、多功能的 AI 系統至關重要。

**本章重點**：
- 位置編碼技術（RoPE、ALiBi、YaRN）及其對上下文長度的影響
- 模型融合和 Mixture of Experts (MoE) 架構
- 多模態模型設計與實現
- 開源 LLM 預訓練項目的實踐經驗

---

## 10.1 不同位置嵌入 (RoPE、ALiBi、YaRN) 與上下文長度延展方法

### 10.1.1 位置編碼的重要性

**為何需要位置編碼**：
- Transformer 架構本身不具備位置感知能力
- 位置信息對於理解序列順序至關重要
- 不同的位置編碼方法影響模型的外推能力

**傳統絕對位置編碼的限制**：
- 固定長度限制（如 GPT-2 的 1024 tokens）
- 外推能力差：超出訓練長度後性能急劇下降
- 參數效率低

### 10.1.2 RoPE (Rotary Position Embedding)

**核心思想**：
通過旋轉變換將位置信息編碼到 Query 和 Key 向量中，使得注意力分數自然地包含相對位置信息。

**數學原理**：

對於位置 $m$ 的向量，RoPE 應用旋轉矩陣 $R_m$：

$$
\begin{pmatrix}
q_{m,2i} \\
q_{m,2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
q_{2i} \\
q_{2i+1}
\end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$

**實現範例**：

```python
import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    """RoPE 位置編碼"""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 預計算頻率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 預計算位置編碼
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """預計算旋轉矩陣"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]

        # 拼接 sin 和 cos
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def rotate_half(self, x):
        """旋轉一半維度"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        """
        應用 RoPE
        q, k: [batch, seq_len, num_heads, head_dim]
        """
        if seq_len is None:
            seq_len = q.shape[1]

        # 擴展緩存（如果需要）
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]

        # 應用旋轉
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

# 使用範例
dim = 128
rope = RotaryPositionEmbedding(dim, max_seq_len=2048)

# 假設輸入
batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 128
q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn(batch_size, seq_len, num_heads, head_dim)

# 應用 RoPE
q_rope, k_rope = rope(q, k, seq_len=seq_len)
print(f"Q shape: {q_rope.shape}")  # [2, 512, 8, 128]
print(f"K shape: {k_rope.shape}")  # [2, 512, 8, 128]
```

**高效實現（GPU 優化）**：

```python
def apply_rotary_emb(q, k, cos, sin):
    """
    高效應用 RoPE（向量化版本）
    """
    # Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 旋轉一半
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

**RoPE 的優勢**：
- **相對位置感知**：注意力分數自然包含相對位置
- **外推能力強**：可以處理比訓練時更長的序列
- **參數高效**：無需額外的位置參數
- **性能優異**：LLaMA、PaLM 等模型廣泛採用

**長度外推技術 - Position Interpolation**：

```python
class ExtendedRoPE(RotaryPositionEmbedding):
    """支持位置插值的 RoPE"""

    def __init__(self, dim, max_seq_len=2048, base=10000, scaling_factor=1.0):
        """
        scaling_factor: 縮放因子
        - scaling_factor=2.0 可將上下文長度從 2048 擴展到 4096
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_seq_len, base)

    def _build_cache(self, seq_len):
        """使用插值構建緩存"""
        # 縮放位置索引
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype) / self.scaling_factor
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

# 將 2048 上下文擴展到 8192
extended_rope = ExtendedRoPE(dim=128, max_seq_len=8192, scaling_factor=4.0)
```

---

### 10.1.3 ALiBi (Attention with Linear Biases)

**核心思想**：
在注意力分數上添加線性偏置，而不使用顯式的位置嵌入。

**數學原理**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot [-(i-j)]\right) V
$$

其中 $m$ 是 head-specific 的斜率，$[-(i-j)]$ 是相對距離矩陣。

**實現範例**：

```python
class ALiBiAttention(nn.Module):
    """ALiBi 注意力機制"""

    def __init__(self, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # 計算每個 head 的斜率
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # 預計算偏置矩陣
        alibi = self._build_alibi_bias(max_seq_len, slopes)
        self.register_buffer('alibi', alibi)

    def _get_slopes(self, num_heads):
        """
        計算 ALiBi 斜率
        使用幾何序列：2^(-8/n), 2^(-16/n), ..., 2^(-8)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(torch.arange(n).float() / n * 3))
            return start

        if num_heads <= 8:
            return get_slopes_power_of_2(num_heads)
        else:
            # 如果 heads 數量不是 2 的冪，使用插值
            extra_slopes = self._get_slopes(2 * num_heads)[:num_heads]
            return extra_slopes

    def _build_alibi_bias(self, max_seq_len, slopes):
        """構建 ALiBi 偏置矩陣"""
        # 相對位置矩陣：[i - j]
        position = torch.arange(max_seq_len).unsqueeze(0)
        relative_position = position - position.transpose(0, 1)  # [seq_len, seq_len]

        # 應用斜率（每個 head 不同）
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position.unsqueeze(0)
        # alibi shape: [num_heads, seq_len, seq_len]

        return alibi

    def forward(self, query, key, value, attention_mask=None):
        """
        query, key, value: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # 標準注意力分數
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        # [batch, num_heads, seq_len, seq_len]

        # 添加 ALiBi 偏置
        alibi_bias = self.alibi[:num_heads, :seq_len, :seq_len]
        attention_scores = attention_scores + alibi_bias.unsqueeze(0)

        # 應用掩碼（如果有）
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 計算輸出
        context = torch.matmul(attention_probs, value)
        return context

# 使用範例
num_heads = 8
alibi_attn = ALiBiAttention(num_heads=num_heads, max_seq_len=2048)

# 假設輸入
batch_size, seq_len, head_dim = 2, 512, 64
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)
v = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 計算注意力
output = alibi_attn(q, k, v)
print(f"Output shape: {output.shape}")  # [2, 8, 512, 64]
```

**可視化 ALiBi 偏置**：

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_alibi_bias(num_heads=8, seq_len=64):
    """可視化 ALiBi 偏置矩陣"""
    alibi_attn = ALiBiAttention(num_heads=num_heads, max_seq_len=seq_len)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(min(num_heads, 8)):
        bias = alibi_attn.alibi[i, :seq_len, :seq_len].cpu().numpy()
        sns.heatmap(bias, ax=axes[i], cmap='RdBu_r', center=0,
                    xticklabels=False, yticklabels=False)
        axes[i].set_title(f'Head {i+1} (slope={alibi_attn.slopes[i]:.4f})')

    plt.tight_layout()
    plt.savefig('alibi_bias_visualization.png')
    plt.show()

# visualize_alibi_bias()
```

**ALiBi 的優勢**：
- **零外推開銷**：訓練時使用短序列，推理時可用於長序列
- **參數高效**：無需學習位置嵌入
- **線性外推**：在長序列上表現優於絕對位置編碼
- **簡單實現**：僅需在注意力分數上加偏置

---

### 10.1.4 YaRN (Yet another RoPE extensioN method)

**核心思想**：
結合頻率插值和注意力縮放，實現 RoPE 的高效長度擴展。

**關鍵技術**：

1. **NTK-aware 插值**：
   - 調整 RoPE 的 base 參數而非直接插值位置
   - 更好地保留高頻和低頻信息

2. **動態縮放**：
   - 根據序列長度動態調整注意力溫度

**實現範例**：

```python
class YaRNRoPE(nn.Module):
    """YaRN: 改進的 RoPE 長度擴展"""

    def __init__(self, dim, max_seq_len=2048, original_max_len=2048,
                 base=10000, scale=1.0, extrapolation_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_len = original_max_len
        self.scale = scale
        self.extrapolation_factor = extrapolation_factor

        # NTK-aware base 調整
        if max_seq_len > original_max_len:
            # 調整 base 以適應更長序列
            alpha = max_seq_len / original_max_len
            base = base * (alpha ** (dim / (dim - 2)))

        # 計算頻率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """構建緩存"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # 應用縮放
        if seq_len > self.original_max_len:
            # 外推部分使用不同的縮放
            t_scaled = t.clone()
            extrapolation_range = seq_len - self.original_max_len
            t_scaled[self.original_max_len:] = (
                self.original_max_len +
                (t[self.original_max_len:] - self.original_max_len) / self.extrapolation_factor
            )
            t = t_scaled

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[1]

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # 動態注意力縮放
        if seq_len > self.original_max_len:
            scale_factor = (seq_len / self.original_max_len) ** 0.5
            q = q / scale_factor
            k = k / scale_factor

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

# 將 4096 上下文擴展到 32768
yarn_rope = YaRNRoPE(
    dim=128,
    max_seq_len=32768,
    original_max_len=4096,
    scale=8.0,
    extrapolation_factor=2.0
)
```

**YaRN vs RoPE vs ALiBi 比較**：

```python
def compare_position_encodings():
    """比較不同位置編碼方法"""
    dim = 128
    train_len = 2048
    test_len = 8192  # 4x 外推

    # 1. 原始 RoPE（無擴展）
    rope_original = RotaryPositionEmbedding(dim, max_seq_len=train_len)

    # 2. RoPE + Position Interpolation
    rope_pi = ExtendedRoPE(dim, max_seq_len=test_len, scaling_factor=4.0)

    # 3. YaRN
    yarn = YaRNRoPE(
        dim=dim,
        max_seq_len=test_len,
        original_max_len=train_len,
        extrapolation_factor=2.0
    )

    # 4. ALiBi
    alibi = ALiBiAttention(num_heads=8, max_seq_len=test_len)

    print("Position Encoding Comparison:")
    print(f"  Original RoPE: max_len={train_len}")
    print(f"  RoPE + PI: max_len={test_len} (scaling={4.0})")
    print(f"  YaRN: max_len={test_len} (NTK-aware + dynamic scaling)")
    print(f"  ALiBi: max_len={test_len} (linear bias)")

    return {
        'rope_original': rope_original,
        'rope_pi': rope_pi,
        'yarn': yarn,
        'alibi': alibi
    }

# encodings = compare_position_encodings()
```

**性能對比表**：

| 方法 | 外推能力 | 訓練成本 | 推理效率 | 實現複雜度 |
|------|----------|----------|----------|------------|
| 絕對位置編碼 | 差 | 低 | 高 | 低 |
| RoPE | 中等 | 低 | 高 | 中等 |
| RoPE + PI | 好 | 低（需微調）| 高 | 中等 |
| YaRN | 優秀 | 中等 | 高 | 高 |
| ALiBi | 優秀 | 低 | 高 | 低 |

---

### 10.1.5 上下文長度延展的實踐技巧

**1. 漸進式擴展訓練**：

```python
def progressive_length_training(model, dataset, stages):
    """
    漸進式長度擴展訓練
    stages: [(length, epochs), ...]
    """
    for target_length, epochs in stages:
        print(f"Training stage: max_length={target_length}, epochs={epochs}")

        # 調整位置編碼
        model.update_position_encoding(max_length=target_length)

        # 過濾數據集
        filtered_data = [x for x in dataset if len(x) <= target_length]

        # 訓練
        for epoch in range(epochs):
            train_epoch(model, filtered_data)

        # 評估
        eval_results = evaluate(model, test_data_length=target_length)
        print(f"Eval at {target_length}: {eval_results}")

# 範例
# stages = [
#     (2048, 1),   # 先在 2K 上訓練
#     (4096, 1),   # 擴展到 4K
#     (8192, 2),   # 擴展到 8K
#     (16384, 2),  # 最終 16K
# ]
# progressive_length_training(model, dataset, stages)
```

**2. 長上下文微調策略**：

```python
from transformers import Trainer, TrainingArguments

def finetune_for_long_context(
    model,
    tokenizer,
    train_dataset,
    target_length=8192,
    original_length=2048
):
    """長上下文微調"""

    # 1. 更新模型配置
    model.config.max_position_embeddings = target_length

    # 2. 擴展位置編碼
    if hasattr(model, 'resize_position_embeddings'):
        model.resize_position_embeddings(target_length)

    # 3. 使用較低學習率
    training_args = TrainingArguments(
        output_dir="./long_context_model",
        learning_rate=1e-5,  # 降低學習率
        per_device_train_batch_size=1,  # 長序列需要更少 batch size
        gradient_accumulation_steps=16,
        max_steps=1000,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        fp16=True,  # 使用混合精度節省內存
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    return model

# 範例
# model = finetune_for_long_context(
#     model, tokenizer, long_context_dataset,
#     target_length=8192, original_length=2048
# )
```

**3. 內存優化技巧**：

```python
class MemoryEfficientLongContextAttention(nn.Module):
    """內存高效的長上下文注意力"""

    def __init__(self, config):
        super().__init__()
        self.chunk_size = 1024  # 分塊處理

    def forward(self, query, key, value, attention_mask=None):
        """
        分塊計算注意力以節省內存
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # 如果序列較短，直接計算
        if seq_len <= self.chunk_size:
            return self._standard_attention(query, key, value, attention_mask)

        # 分塊處理
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            chunk_end = min(i + self.chunk_size, seq_len)
            q_chunk = query[:, :, i:chunk_end, :]

            # 計算這個 chunk 與所有 key 的注意力
            chunk_output = self._standard_attention(
                q_chunk, key, value,
                attention_mask[:, :, i:chunk_end, :] if attention_mask is not None else None
            )
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)

    def _standard_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
```

---

## 10.2 模型融合 (Model Merging)、Mixture of Experts (MoE)

### 10.2.1 模型融合 (Model Merging)

**核心概念**：
將多個獨立訓練的模型合併，創造出具有綜合能力的新模型。

**常見融合方法**：

**1. 線性插值 (Linear Interpolation)**：

```python
def linear_model_merge(model_a, model_b, alpha=0.5):
    """
    線性插值融合兩個模型
    merged = alpha * model_a + (1-alpha) * model_b
    """
    merged_state_dict = {}

    for key in model_a.state_dict():
        merged_state_dict[key] = (
            alpha * model_a.state_dict()[key] +
            (1 - alpha) * model_b.state_dict()[key]
        )

    # 創建新模型並加載參數
    merged_model = type(model_a)(model_a.config)
    merged_model.load_state_dict(merged_state_dict)

    return merged_model

# 範例：融合通用模型和領域模型
# merged = linear_model_merge(general_model, domain_model, alpha=0.7)
```

**2. SLERP (Spherical Linear Interpolation)**：

```python
import torch

def slerp_merge(model_a, model_b, alpha=0.5):
    """
    球面線性插值融合（更適合神經網絡）
    """
    merged_state_dict = {}

    for key in model_a.state_dict():
        param_a = model_a.state_dict()[key].float()
        param_b = model_b.state_dict()[key].float()

        # SLERP 公式
        dot_product = torch.sum(param_a * param_b) / (
            torch.norm(param_a) * torch.norm(param_b)
        )
        omega = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

        if omega.abs() < 1e-6:
            # 參數幾乎相同，使用線性插值
            merged_param = alpha * param_a + (1 - alpha) * param_b
        else:
            # SLERP
            merged_param = (
                torch.sin((1 - alpha) * omega) / torch.sin(omega) * param_a +
                torch.sin(alpha * omega) / torch.sin(omega) * param_b
            )

        merged_state_dict[key] = merged_param

    merged_model = type(model_a)(model_a.config)
    merged_model.load_state_dict(merged_state_dict)

    return merged_model
```

**3. Task Arithmetic (任務向量)**：

```python
def task_arithmetic_merge(base_model, finetuned_models, weights=None):
    """
    任務算術：通過任務向量組合模型
    task_vector = finetuned_model - base_model
    merged = base + sum(weight_i * task_vector_i)
    """
    if weights is None:
        weights = [1.0] * len(finetuned_models)

    # 計算任務向量
    task_vectors = []
    for ft_model in finetuned_models:
        task_vec = {}
        for key in base_model.state_dict():
            task_vec[key] = ft_model.state_dict()[key] - base_model.state_dict()[key]
        task_vectors.append(task_vec)

    # 組合任務向量
    merged_state_dict = {}
    for key in base_model.state_dict():
        merged_state_dict[key] = base_model.state_dict()[key].clone()

        for task_vec, weight in zip(task_vectors, weights):
            merged_state_dict[key] += weight * task_vec[key]

    merged_model = type(base_model)(base_model.config)
    merged_model.load_state_dict(merged_state_dict)

    return merged_model

# 範例：組合多個專家模型
# math_expert = finetune(base_model, math_data)
# code_expert = finetune(base_model, code_data)
# writing_expert = finetune(base_model, writing_data)
#
# merged = task_arithmetic_merge(
#     base_model,
#     [math_expert, code_expert, writing_expert],
#     weights=[0.5, 0.3, 0.2]
# )
```

**4. TIES-Merging (Trim, Elect, Merge)**：

```python
def ties_merging(base_model, finetuned_models, k=0.2, weights=None):
    """
    TIES-Merging：更智能的模型融合
    1. Trim: 移除不重要的參數變化
    2. Elect: 解決符號衝突
    3. Merge: 合併參數
    """
    if weights is None:
        weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

    # 計算任務向量
    task_vectors = []
    for ft_model in finetuned_models:
        task_vec = {}
        for key in base_model.state_dict():
            task_vec[key] = ft_model.state_dict()[key] - base_model.state_dict()[key]
        task_vectors.append(task_vec)

    merged_state_dict = {}

    for key in base_model.state_dict():
        # 1. Trim: 保留 top-k% 重要參數
        all_values = torch.cat([tv[key].flatten() for tv in task_vectors])
        threshold = torch.quantile(all_values.abs(), 1 - k)

        trimmed_vectors = []
        for tv in task_vectors:
            trimmed = tv[key].clone()
            trimmed[trimmed.abs() < threshold] = 0
            trimmed_vectors.append(trimmed)

        # 2. Elect: 解決符號衝突（保留同號多數）
        signs = torch.stack([torch.sign(tv) for tv in trimmed_vectors])
        majority_sign = torch.sign(signs.sum(dim=0))

        # 3. Merge: 加權平均
        merged_param = base_model.state_dict()[key].clone()
        for tv, weight in zip(trimmed_vectors, weights):
            # 只保留與多數符號一致的值
            mask = (torch.sign(tv) == majority_sign) | (majority_sign == 0)
            merged_param += weight * tv * mask.float()

        merged_state_dict[key] = merged_param

    merged_model = type(base_model)(base_model.config)
    merged_model.load_state_dict(merged_state_dict)

    return merged_model
```

**模型融合評估**：

```python
def evaluate_merged_model(merged_model, test_datasets):
    """評估融合模型在多個任務上的表現"""
    results = {}

    for task_name, dataset in test_datasets.items():
        # 評估任務
        accuracy = evaluate_task(merged_model, dataset)
        results[task_name] = accuracy
        print(f"{task_name}: {accuracy:.4f}")

    # 計算平均性能
    avg_performance = sum(results.values()) / len(results)
    print(f"Average Performance: {avg_performance:.4f}")

    return results

# 範例
# test_datasets = {
#     'math': math_test_data,
#     'code': code_test_data,
#     'writing': writing_test_data,
# }
# results = evaluate_merged_model(merged_model, test_datasets)
```

---

### 10.2.2 Mixture of Experts (MoE)

**核心概念**：
使用多個專家網絡和一個門控機制，根據輸入動態選擇專家。

**MoE 架構**：

```python
class MixtureOfExpertsLayer(nn.Module):
    """Mixture of Experts Layer"""

    def __init__(self, d_model, num_experts=8, expert_capacity=None, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        # 門控網絡
        self.gate = nn.Linear(d_model, num_experts)

        # 專家網絡（每個是 FFN）
        self.experts = nn.ModuleList([
            FeedForwardNetwork(d_model, d_model * 4)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]

        # 計算門控分數
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Top-K 門控
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        # top_k_scores: [batch_size * seq_len, top_k]
        # top_k_indices: [batch_size * seq_len, top_k]

        # 規範化 top-k 分數
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # 計算專家輸出
        expert_outputs = torch.zeros_like(x_flat)

        for i in range(self.num_experts):
            # 找到選擇了專家 i 的 tokens
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue

            # 提取這些 tokens
            expert_input = x_flat[expert_mask]

            # 通過專家處理
            expert_output = self.experts[i](expert_input)

            # 加權累加到輸出
            # 獲取這些 tokens 對專家 i 的權重
            weights = top_k_scores[expert_mask]
            weights = weights[top_k_indices[expert_mask] == i]

            expert_outputs[expert_mask] += weights.unsqueeze(-1) * expert_output

        # Reshape 回原始形狀
        output = expert_outputs.view(batch_size, seq_len, d_model)

        return output, gate_logits

class FeedForwardNetwork(nn.Module):
    """標準 FFN 專家"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

# 使用範例
moe_layer = MixtureOfExpertsLayer(d_model=768, num_experts=8, top_k=2)
x = torch.randn(2, 128, 768)  # [batch, seq_len, d_model]
output, gate_logits = moe_layer(x)
print(f"Output shape: {output.shape}")  # [2, 128, 768]
```

**負載均衡損失**：

```python
def compute_load_balancing_loss(gate_logits, num_experts):
    """
    計算負載均衡損失，鼓勵專家均勻使用
    """
    # gate_logits: [batch * seq_len, num_experts]

    # 計算每個專家被選中的頻率
    gate_probs = F.softmax(gate_logits, dim=-1)
    expert_usage = gate_probs.mean(dim=0)  # [num_experts]

    # 目標是均勻分布
    target_usage = torch.ones_like(expert_usage) / num_experts

    # 負載均衡損失（KL 散度）
    load_loss = F.kl_div(
        expert_usage.log(),
        target_usage,
        reduction='batchmean'
    )

    return load_loss

# 在訓練中使用
# output, gate_logits = moe_layer(x)
# main_loss = compute_main_loss(output, targets)
# load_loss = compute_load_balancing_loss(gate_logits, num_experts=8)
# total_loss = main_loss + 0.01 * load_loss  # 加權負載損失
```

**Switch Transformer 風格的 MoE**：

```python
class SwitchTransformerMoE(nn.Module):
    """
    Switch Transformer: 每個 token 只路由到一個專家 (top-1)
    """

    def __init__(self, d_model, num_experts=8, capacity_factor=1.25):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            FeedForwardNetwork(d_model, d_model * 4)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # 門控（top-1）
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 選擇最佳專家
        expert_indices = torch.argmax(gate_probs, dim=-1)
        expert_weights = gate_probs.gather(1, expert_indices.unsqueeze(-1)).squeeze(-1)

        # 容量限制
        capacity = int(self.capacity_factor * x_flat.size(0) / self.num_experts)

        # 處理每個專家
        output = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            # 找到分配給該專家的 tokens
            expert_mask = (expert_indices == expert_id)
            num_assigned = expert_mask.sum().item()

            if num_assigned == 0:
                continue

            # 應用容量限制
            if num_assigned > capacity:
                # 選擇權重最高的 tokens
                expert_probs = gate_probs[:, expert_id]
                _, top_indices = torch.topk(expert_probs, capacity)
                expert_mask = torch.zeros_like(expert_mask)
                expert_mask[top_indices] = True

            # 處理
            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_id](expert_input)

            # 加權輸出
            weights = expert_weights[expert_mask]
            output[expert_mask] = weights.unsqueeze(-1) * expert_output

        return output.view(batch_size, seq_len, d_model)

# 使用
switch_moe = SwitchTransformerMoE(d_model=768, num_experts=8)
output = switch_moe(x)
```

**MoE 優勢與挑戰**：

**優勢**：
- 參數效率高：增加專家數量而不增加計算量
- 專業化：不同專家可學習不同領域知識
- 可擴展性：可輕鬆擴展到數百個專家

**挑戰**：
- 負載均衡：確保專家均勻使用
- 通信開銷：分布式訓練時專家間通信
- 穩定性：訓練初期可能不穩定

---

## 10.3 多模態模型 (CLIP、LLaVA) 與整合多種輸入管道的應用

### 10.3.1 CLIP (Contrastive Language-Image Pre-training)

**核心思想**：
通過對比學習將圖像和文本映射到共同的嵌入空間。

**架構**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class SimpleCLIP(nn.Module):
    """簡化的 CLIP 實現"""

    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 投影頭
        self.image_projection = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, embed_dim)

        # 可學習的溫度參數
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        """編碼圖像"""
        image_features = self.image_encoder(images)
        image_embeds = self.image_projection(image_features)
        # L2 規範化
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def encode_text(self, text_tokens):
        """編碼文本"""
        text_features = self.text_encoder(text_tokens)
        text_embeds = self.text_projection(text_features)
        # L2 規範化
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    def forward(self, images, text_tokens):
        """
        計算對比損失
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(text_tokens)

        # 計算相似度矩陣
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image, logits_per_text):
    """
    對比學習損失（InfoNCE）
    """
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # 圖像到文本損失
    loss_img_to_text = F.cross_entropy(logits_per_image, labels)

    # 文本到圖像損失
    loss_text_to_img = F.cross_entropy(logits_per_text, labels)

    # 總損失
    loss = (loss_img_to_text + loss_text_to_img) / 2

    return loss

# 訓練範例
# images: [batch, 3, 224, 224]
# text_tokens: [batch, max_len]
# logits_per_image, logits_per_text = clip_model(images, text_tokens)
# loss = contrastive_loss(logits_per_image, logits_per_text)
```

**CLIP 應用 - 零樣本圖像分類**：

```python
def zero_shot_classification(clip_model, image, class_names):
    """
    使用 CLIP 進行零樣本圖像分類
    """
    # 編碼圖像
    image_embed = clip_model.encode_image(image.unsqueeze(0))

    # 為每個類別創建文本描述
    text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
    text_tokens = tokenize(text_prompts)

    # 編碼文本
    text_embeds = clip_model.encode_text(text_tokens)

    # 計算相似度
    similarities = (image_embed @ text_embeds.t()).squeeze(0)
    probs = F.softmax(similarities, dim=-1)

    # 返回預測
    pred_idx = probs.argmax().item()
    pred_class = class_names[pred_idx]
    pred_prob = probs[pred_idx].item()

    return pred_class, pred_prob, probs

# 範例
# class_names = ["cat", "dog", "bird", "car"]
# pred_class, pred_prob, all_probs = zero_shot_classification(
#     clip_model, image, class_names
# )
# print(f"Prediction: {pred_class} ({pred_prob:.2%})")
```

**CLIP 應用 - 圖像檢索**：

```python
def image_text_retrieval(clip_model, images, texts):
    """
    圖像-文本檢索
    """
    # 編碼所有圖像和文本
    image_embeds = clip_model.encode_image(images)  # [num_images, embed_dim]
    text_embeds = clip_model.encode_text(texts)      # [num_texts, embed_dim]

    # 計算相似度矩陣
    similarity_matrix = image_embeds @ text_embeds.t()  # [num_images, num_texts]

    return similarity_matrix

def retrieve_images_by_text(query_text, images, clip_model, top_k=5):
    """根據文本檢索最相關的圖像"""
    query_tokens = tokenize([query_text])
    text_embed = clip_model.encode_text(query_tokens)

    image_embeds = clip_model.encode_image(images)

    similarities = (text_embed @ image_embeds.t()).squeeze(0)
    top_indices = similarities.topk(top_k).indices

    return top_indices, similarities[top_indices]

# 範例
# query = "a cute cat playing with a ball"
# top_image_indices, scores = retrieve_images_by_text(
#     query, all_images, clip_model, top_k=5
# )
```

---

### 10.3.2 LLaVA (Large Language and Vision Assistant)

**核心思想**：
將視覺編碼器與 LLM 結合，實現視覺問答和多模態對話。

**架構**：

```python
class LLaVA(nn.Module):
    """LLaVA: 視覺語言助手"""

    def __init__(self, vision_encoder, vision_projector, language_model):
        super().__init__()
        self.vision_encoder = vision_encoder  # 例如 CLIP 圖像編碼器
        self.vision_projector = vision_projector  # 投影到 LLM 嵌入空間
        self.language_model = language_model  # LLaMA/Mistral 等

    def encode_images(self, images):
        """
        編碼圖像為視覺 tokens
        images: [batch, 3, H, W]
        returns: [batch, num_visual_tokens, llm_dim]
        """
        # 通過視覺編碼器
        visual_features = self.vision_encoder(images)  # [batch, vision_dim]

        # 投影到 LLM 維度
        visual_tokens = self.vision_projector(visual_features)  # [batch, num_tokens, llm_dim]

        return visual_tokens

    def forward(self, images, input_ids, attention_mask=None):
        """
        前向傳播
        images: [batch, 3, H, W]
        input_ids: [batch, text_seq_len]
        """
        batch_size = images.size(0)

        # 1. 編碼圖像
        visual_tokens = self.encode_images(images)  # [batch, num_visual_tokens, llm_dim]

        # 2. 獲取文本嵌入
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        # [batch, text_seq_len, llm_dim]

        # 3. 合併視覺和文本 tokens
        # 在文本序列開頭插入視覺 tokens
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        # [batch, num_visual_tokens + text_seq_len, llm_dim]

        # 4. 通過 LLM
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=self._create_attention_mask(
                visual_tokens.size(1),
                input_ids.size(1),
                batch_size
            ) if attention_mask is None else attention_mask
        )

        return outputs

    def _create_attention_mask(self, num_visual_tokens, num_text_tokens, batch_size):
        """創建注意力掩碼"""
        visual_mask = torch.ones(batch_size, num_visual_tokens)
        text_mask = torch.ones(batch_size, num_text_tokens)
        return torch.cat([visual_mask, text_mask], dim=1)

    def generate(self, images, prompt, max_new_tokens=256):
        """
        生成回答
        """
        # 編碼圖像
        visual_tokens = self.encode_images(images)

        # Tokenize prompt
        prompt_ids = tokenize(prompt)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids)

        # 合併
        combined_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)

        # 生成
        outputs = self.language_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        # 解碼
        response = detokenize(outputs[0, visual_tokens.size(1) + prompt_ids.size(1):])

        return response

# 使用範例
vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
vision_projector = nn.Linear(1024, 4096)  # CLIP dim -> LLaMA dim
language_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

llava = LLaVA(vision_encoder, vision_projector, language_model)

# 視覺問答
# image = load_image("cat.jpg")
# response = llava.generate(image, "What is in this image?")
# print(response)  # "There is a cat sitting on a mat..."
```

**LLaVA 訓練流程**：

```python
def train_llava(llava_model, vision_text_dataset, epochs=3):
    """
    LLaVA 兩階段訓練
    """
    optimizer = torch.optim.AdamW(llava_model.parameters(), lr=2e-5)

    # 階段 1: 預訓練投影層（凍結視覺編碼器和 LLM）
    print("Stage 1: Pretraining vision projector...")
    llava_model.vision_encoder.requires_grad_(False)
    llava_model.language_model.requires_grad_(False)
    llava_model.vision_projector.requires_grad_(True)

    for epoch in range(epochs // 2):
        for batch in vision_text_dataset:
            images = batch['images']
            captions = batch['captions']  # 圖像描述

            # 前向傳播
            outputs = llava_model(images, captions['input_ids'])
            loss = outputs.loss

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 階段 2: 微調整個模型（視覺問答）
    print("Stage 2: Finetuning on VQA...")
    llava_model.language_model.requires_grad_(True)

    optimizer = torch.optim.AdamW(llava_model.parameters(), lr=2e-6)  # 更低學習率

    for epoch in range(epochs // 2, epochs):
        for batch in vision_text_dataset:
            images = batch['images']
            questions = batch['questions']
            answers = batch['answers']

            # 構建 prompt
            input_text = f"Question: {questions}\nAnswer: {answers}"
            input_ids = tokenize(input_text)

            # 前向傳播
            outputs = llava_model(images, input_ids)
            loss = outputs.loss

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return llava_model

# llava_trained = train_llava(llava, vqa_dataset)
```

**多模態應用範例**：

```python
class MultimodalAssistant:
    """多模態助手應用"""

    def __init__(self, llava_model):
        self.model = llava_model

    def visual_question_answering(self, image, question):
        """視覺問答"""
        prompt = f"USER: {question}\nASSISTANT:"
        response = self.model.generate(image, prompt)
        return response

    def image_captioning(self, image):
        """圖像描述"""
        prompt = "USER: Describe this image in detail.\nASSISTANT:"
        return self.model.generate(image, prompt)

    def visual_reasoning(self, image, task_description):
        """視覺推理"""
        prompt = f"USER: {task_description}\nASSISTANT:"
        return self.model.generate(image, prompt)

    def multi_turn_conversation(self, image, conversation_history):
        """多輪對話"""
        conversation = "\n".join([
            f"{turn['role']}: {turn['content']}"
            for turn in conversation_history
        ])
        prompt = conversation + "\nASSISTANT:"
        return self.model.generate(image, prompt)

# 使用範例
assistant = MultimodalAssistant(llava)

# 1. 視覺問答
# answer = assistant.visual_question_answering(image, "What color is the cat?")

# 2. 圖像描述
# caption = assistant.image_captioning(image)

# 3. 視覺推理
# reasoning = assistant.visual_reasoning(
#     image,
#     "Is this a safe environment for children? Explain why."
# )

# 4. 多輪對話
# conversation = [
#     {"role": "USER", "content": "What's in this image?"},
#     {"role": "ASSISTANT", "content": "There's a cat on a mat."},
#     {"role": "USER", "content": "What breed might it be?"}
# ]
# response = assistant.multi_turn_conversation(image, conversation)
```

---

### 10.3.3 其他多模態架構

**Flamingo (DeepMind)**：

```python
class FlamingoPerceiverResampler(nn.Module):
    """Flamingo 的 Perceiver Resampler"""

    def __init__(self, vision_dim, llm_dim, num_latents=64, num_layers=6):
        super().__init__()
        self.num_latents = num_latents

        # 可學習的潛在查詢
        self.latents = nn.Parameter(torch.randn(num_latents, llm_dim))

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(llm_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])

        # Projection
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, visual_features):
        """
        visual_features: [batch, num_patches, vision_dim]
        returns: [batch, num_latents, llm_dim]
        """
        batch_size = visual_features.size(0)

        # 投影視覺特徵
        visual_features = self.proj(visual_features)

        # 重複潛在查詢for batch
        latents = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)

        # 多層 cross-attention
        for layer in self.layers:
            latents, _ = layer(latents, visual_features, visual_features)

        return latents

# flamingo_resampler = FlamingoPerceiverResampler(vision_dim=1024, llm_dim=4096)
# compressed_visual_tokens = flamingo_resampler(visual_patches)
```

---

## 10.4 預訓練全開源專案 (OLMo, Dolly) 與社群實踐案例

### 10.4.1 OLMo (Open Language Model)

**特點**：
- Allen AI 的完全開源 LLM 項目
- 開放訓練數據、代碼、模型權重、訓練日誌

**訓練框架參考**：

```python
# OLMo 風格的訓練配置
from dataclasses import dataclass

@dataclass
class OLMoTrainingConfig:
    """OLMo 訓練配置"""

    # 模型配置
    model_name: str = "olmo-7b"
    vocab_size: int = 50280
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32

    # 數據配置
    train_data_path: str = "dolma/train"  # Dolma 數據集
    val_data_path: str = "dolma/val"
    seq_length: int = 2048

    # 訓練配置
    batch_size: int = 2048  # 全局 batch size
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    max_steps: int = 500000
    weight_decay: float = 0.1

    # 優化器配置
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # 其他
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"

def train_olmo_style(config):
    """OLMo 風格的訓練流程"""
    import torch
    from torch.utils.data import DataLoader

    # 1. 初始化模型
    model = build_transformer_model(config)

    # 2. 初始化優化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )

    # 3. 學習率調度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    # 4. 加載數據
    train_loader = DataLoader(
        load_dolma_dataset(config.train_data_path),
        batch_size=config.batch_size,
        shuffle=True
    )

    # 5. 訓練循環
    model.train()
    for step, batch in enumerate(train_loader):
        if step >= config.max_steps:
            break

        # 前向傳播
        outputs = model(batch['input_ids'], labels=batch['input_ids'])
        loss = outputs.loss

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # 記錄
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        # 保存檢查點
        if step % 10000 == 0:
            save_checkpoint(model, optimizer, step)

    return model

# config = OLMoTrainingConfig()
# model = train_olmo_style(config)
```

---

### 10.4.2 Dolly (Databricks)

**特點**：
- 基於 GPT-J 6B 微調的開源指令模型
- 使用內部標注的 15K 指令數據集

**複製 Dolly 訓練流程**：

```python
def create_dolly_training_dataset():
    """創建 Dolly 風格的指令數據集"""
    instructions = [
        {
            "instruction": "Explain quantum computing to a 10-year-old",
            "context": "",
            "response": "Quantum computing is like a super powerful computer..."
        },
        {
            "instruction": "Summarize the following article",
            "context": "Article text here...",
            "response": "Summary here..."
        },
        # ... 更多範例
    ]

    return instructions

def format_dolly_prompt(instruction, context="", response=""):
    """格式化 Dolly 風格的 prompt"""
    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return prompt

def finetune_dolly(base_model, instruction_dataset):
    """微調 Dolly 模型"""
    from transformers import Trainer, TrainingArguments

    # 格式化數據集
    formatted_dataset = [
        format_dolly_prompt(item['instruction'], item['context'], item['response'])
        for item in instruction_dataset
    ]

    # 訓練配置
    training_args = TrainingArguments(
        output_dir="./dolly-model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=1000,
        fp16=True,
    )

    # 訓練
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=formatted_dataset,
    )

    trainer.train()
    return base_model

# 使用範例
# base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
# instruction_data = create_dolly_training_dataset()
# dolly_model = finetune_dolly(base_model, instruction_data)
```

---

### 10.4.3 社群實踐案例

**案例 1: 從零訓練小型 LLM**：

```python
def train_small_llm_from_scratch():
    """從零訓練小型 LLM (125M 參數)"""

    # 1. 配置
    config = {
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'vocab_size': 50257,
        'max_seq_len': 1024,
    }

    # 2. 初始化模型
    model = GPTModel(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 3. 準備數據（使用公開數據集）
    from datasets import load_dataset
    dataset = load_dataset("openwebtext")

    # 4. 訓練（需要約 100GB 數據，數天 GPU 時間）
    train(model, dataset, epochs=1, batch_size=256)

    return model

# small_llm = train_small_llm_from_scratch()
```

**案例 2: 領域特化微調**：

```python
def domain_specific_finetuning(base_model, domain_data):
    """領域特化微調（例如醫療、法律）"""

    # 使用 LoRA 進行高效微調
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    peft_model = get_peft_model(base_model, lora_config)

    # 微調
    trainer = Trainer(
        model=peft_model,
        train_dataset=domain_data,
        args=TrainingArguments(
            output_dir="./domain_model",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            learning_rate=3e-4
        )
    )

    trainer.train()
    return peft_model

# 範例
# medical_llm = domain_specific_finetuning(base_llm, medical_dataset)
```

---

## 參考資源

- **RoPE & 位置編碼**
  - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  - "Train Short, Test Long: Attention with Linear Biases (ALiBi)"
  - "YaRN: Efficient Context Window Extension"

- **模型融合**
  - "Model Soups: Averaging Weights of Multiple Fine-tuned Models"
  - "Editing Models with Task Arithmetic"
  - "TIES-Merging: Resolving Interference in Model Merging"

- **多模態模型**
  - "CLIP: Learning Transferable Visual Models From Natural Language"
  - "LLaVA: Visual Instruction Tuning"
  - "Flamingo: a Visual Language Model for Few-Shot Learning"

- **開源項目**
  - [OLMo](https://github.com/allenai/OLMo)
  - [Dolly](https://github.com/databrickslabs/dolly)
  - [Pythia](https://github.com/EleutherAI/pythia)
