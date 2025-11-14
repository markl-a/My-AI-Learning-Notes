# LLM 基礎與架構介紹

## 目錄
1. [LLM 的概念與 Transformer 基本架構](#11-llm-的概念與-transformer-基本架構)
2. [解碼器為主的模型 (GPT 系列) 介紹](#12-解碼器為主的模型-gpt-系列-介紹)
3. [Tokenization 與語言模型輸入/輸出形式](#13-tokenization-與語言模型輸入輸出形式)
4. [注意力機制及各種變體](#14-注意力機制及各種變體)
5. [Python 實作範例](#15-python-實作範例)

---

## 1.1 LLM 的概念與 Transformer 基本架構

### 什麼是 LLM？

大型語言模型 (Large Language Models, LLM) 是基於深度學習的自然語言處理模型，通常包含數億到數千億個參數。這些模型通過在海量文本數據上進行預訓練，學習語言的統計規律和語義關係。

### Transformer 架構核心概念

Transformer 是現代 LLM 的基礎架構，由 Vaswani 等人在 2017 年的論文 "Attention is All You Need" 中提出。其核心創新包括：

1. **自注意力機制 (Self-Attention)**：允許模型關注輸入序列中的不同位置
2. **多頭注意力 (Multi-Head Attention)**：並行計算多個注意力表示
3. **位置編碼 (Positional Encoding)**：為序列添加位置信息
4. **前饋神經網絡 (Feed-Forward Networks)**：對每個位置獨立應用
5. **層歸一化 (Layer Normalization)**：穩定訓練過程

### Transformer 架構組成

**編碼器 (Encoder)**：
- 多層堆疊
- 每層包含：自注意力層 + 前饋網絡
- 用於理解輸入序列

**解碼器 (Decoder)**：
- 多層堆疊
- 每層包含：自注意力層 + 交叉注意力層 + 前饋網絡
- 用於生成輸出序列

### 數學表示

**自注意力機制**：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- Q (Query)：查詢矩陣
- K (Key)：鍵矩陣
- V (Value)：值矩陣
- d_k：鍵向量的維度

**多頭注意力**：

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

---

## 1.2 解碼器為主的模型 (GPT 系列) 介紹

### GPT 系列演進

1. **GPT-1 (2018)**
   - 117M 參數
   - 12 層 Transformer 解碼器
   - 引入「預訓練 + 微調」範式

2. **GPT-2 (2019)**
   - 1.5B 參數
   - 48 層
   - 展示了 zero-shot 學習能力

3. **GPT-3 (2020)**
   - 175B 參數
   - 96 層
   - 強大的 few-shot 學習能力

4. **GPT-4 (2023)**
   - 多模態能力
   - 更強的推理能力

### 解碼器架構特點

- **單向注意力**：只能看到當前位置之前的內容 (Causal Masking)
- **自回歸生成**：逐個 token 生成文本
- **上下文窗口**：有固定的最大輸入長度限制

### GPT 的訓練目標

**Next Token Prediction（下一個 token 預測）**：

```
L = -Σ log P(x_t | x_1, ..., x_{t-1})
```

模型學習預測序列中下一個 token 的概率分布。

---

## 1.3 Tokenization 與語言模型輸入/輸出形式

### 什麼是 Tokenization？

Tokenization 是將文本轉換為模型可處理的數字序列的過程。

### 常見的 Tokenization 方法

1. **詞級別 (Word-level)**
   - 優點：直觀，易於理解
   - 缺點：詞彙表太大，無法處理未見過的詞

2. **字符級別 (Character-level)**
   - 優點：詞彙表小
   - 缺點：序列太長，難以捕捉語義

3. **子詞級別 (Subword-level)**
   - **BPE (Byte Pair Encoding)**：GPT 系列使用
   - **WordPiece**：BERT 使用
   - **SentencePiece**：多語言模型常用
   - 優點：平衡詞彙表大小和序列長度

### Tokenization 流程

```
原始文本 → [分詞] → Token序列 → [映射] → ID序列 → [嵌入] → 向量序列
```

### 特殊 Token

- `<bos>` / `<s>`：句子開始
- `<eos>` / `</s>`：句子結束
- `<pad>`：填充
- `<unk>`：未知詞
- `<mask>`：掩碼（BERT等使用）

---

## 1.4 注意力機制及各種變體

### 基礎注意力機制

**縮放點積注意力 (Scaled Dot-Product Attention)**：

計算步驟：
1. 計算 Q 和 K 的點積
2. 除以 √d_k 進行縮放
3. 應用 softmax 獲得注意力權重
4. 與 V 相乘得到輸出

### 多頭注意力的優勢

- 捕捉不同類型的關係
- 增加模型表達能力
- 並行計算，提高效率

### 注意力機制的變體

1. **Causal (Masked) Self-Attention**
   - 用於解碼器
   - 掩蓋未來信息

2. **Cross-Attention**
   - 用於編碼器-解碼器架構
   - Query 來自解碼器，Key/Value 來自編碼器

3. **Relative Positional Encoding**
   - 使用相對位置而非絕對位置
   - 提升對長序列的泛化能力

4. **Flash Attention**
   - 優化 GPU 內存訪問
   - 顯著提升計算效率

5. **Multi-Query Attention (MQA)**
   - 所有頭共享同一組 K 和 V
   - 減少推理時的內存消耗

6. **Grouped-Query Attention (GQA)**
   - MQA 和標準多頭注意力的折中
   - 平衡性能和效率

### 位置編碼方案

1. **絕對位置編碼 (Absolute Positional Encoding)**
   - 原始 Transformer 使用的正弦/余弦編碼

2. **相對位置編碼 (Relative Positional Encoding)**
   - Transformer-XL, T5 使用

3. **旋轉位置編碼 (RoPE - Rotary Position Embedding)**
   - LLaMA, GPT-NeoX 使用
   - 性能優秀，支持長上下文

4. **ALiBi (Attention with Linear Biases)**
   - 在注意力分數上添加線性偏置
   - 訓練效率高，外推能力強

---

## 1.5 Python 實作範例

### 1.5.1 簡單的自注意力機制實現

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """縮放點積注意力"""

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query (batch_size, seq_len, d_k)
            K: Key (batch_size, seq_len, d_k)
            V: Value (batch_size, seq_len, d_v)
            mask: 注意力掩碼 (batch_size, seq_len, seq_len)
        """
        # 計算注意力分數
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 應用掩碼（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 獲得注意力權重
        attention_weights = F.softmax(scores, dim=-1)

        # 加權求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# 測試
d_k = 64
seq_len = 10
batch_size = 2

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

attention = ScaledDotProductAttention(d_k)
output, weights = attention(Q, K, V)

print(f"輸出形狀: {output.shape}")
print(f"注意力權重形狀: {weights.shape}")
print(f"注意力權重總和（每行）: {weights.sum(dim=-1)[0]}")
```

### 1.5.2 多頭注意力實現

```python
class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 線性變換層
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """將最後一個維度分割成 (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """合併多頭"""
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 線性變換
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 分割成多頭
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 縮放點積注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # 合併多頭
        output = self.combine_heads(output)

        # 最終線性變換
        output = self.W_o(output)

        return output, attention_weights

# 測試
d_model = 512
num_heads = 8
seq_len = 20
batch_size = 4

x = torch.randn(batch_size, seq_len, d_model)

mha = MultiHeadAttention(d_model, num_heads)
output, weights = mha(x, x, x)

print(f"輸入形狀: {x.shape}")
print(f"輸出形狀: {output.shape}")
print(f"注意力權重形狀: {weights.shape}")
```

### 1.5.3 Causal Mask 實現（用於 GPT 解碼器）

```python
def create_causal_mask(seq_len):
    """創建因果掩碼，使模型只能看到當前和之前的 token"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# 視覺化掩碼
import matplotlib.pyplot as plt

seq_len = 10
mask = create_causal_mask(seq_len)

plt.figure(figsize=(8, 8))
plt.imshow(mask[0, 0].numpy(), cmap='Blues')
plt.title('Causal Attention Mask')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar(label='Attention Allowed')
plt.savefig('causal_mask.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"掩碼形狀: {mask.shape}")
print("掩碼矩陣（1表示可見，0表示掩蓋）:")
print(mask[0, 0].numpy().astype(int))
```

### 1.5.4 位置編碼實現

```python
class PositionalEncoding(nn.Module):
    """正弦/余弦位置編碼"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 創建位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]

# 測試和視覺化
d_model = 128
max_len = 100

pos_enc = PositionalEncoding(d_model, max_len)

# 視覺化位置編碼
pe_matrix = pos_enc.pe[0].numpy()

plt.figure(figsize=(12, 6))
plt.imshow(pe_matrix.T, aspect='auto', cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding')
plt.colorbar()
plt.savefig('positional_encoding.png', dpi=100, bbox_inches='tight')
plt.close()

# 繪製特定維度的位置編碼
plt.figure(figsize=(12, 4))
for i in [0, 1, 2, 3]:
    plt.plot(pe_matrix[:, i], label=f'dim {i}')
plt.xlabel('Position')
plt.ylabel('Value')
plt.title('Positional Encoding - First 4 Dimensions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('positional_encoding_dims.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 1.5.5 Tokenization 實例

```python
from transformers import AutoTokenizer

# 載入不同的 tokenizer
tokenizers = {
    'GPT-2': 'gpt2',
    'BERT': 'bert-base-uncased',
    'LLaMA': 'meta-llama/Llama-2-7b-hf'  # 需要訪問權限
}

text = "Hello, how are you doing today? I'm learning about Large Language Models!"

print("原始文本:", text)
print("\n" + "="*80 + "\n")

for name, model_name in tokenizers.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        print(f"{name} Tokenizer:")
        print(f"  Token 數量: {len(tokens)}")
        print(f"  Tokens: {tokens[:10]}...")  # 只顯示前10個
        print(f"  Token IDs: {token_ids[:10]}...")
        print()

        # 解碼回文本
        decoded_text = tokenizer.decode(token_ids)
        print(f"  解碼後: {decoded_text}")
        print("\n" + "-"*80 + "\n")

    except Exception as e:
        print(f"{name}: 無法載入 - {e}\n")
```

### 1.5.6 簡單的 Transformer 解碼器層

```python
class TransformerDecoderLayer(nn.Module):
    """Transformer 解碼器層"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 多頭自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 前饋網絡
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # 層歸一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 殘差連接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前饋網絡 + 殘差連接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x

# 測試
d_model = 512
num_heads = 8
d_ff = 2048
seq_len = 10
batch_size = 4

x = torch.randn(batch_size, seq_len, d_model)
mask = create_causal_mask(seq_len)

decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff)
output = decoder_layer(x, mask)

print(f"輸入形狀: {x.shape}")
print(f"輸出形狀: {output.shape}")
```

### 1.5.7 完整的小型 GPT 模型

```python
class TinyGPT(nn.Module):
    """簡化版的 GPT 模型"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置編碼
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer 解碼器層
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最終層歸一化
        self.norm = nn.LayerNorm(d_model)

        # 輸出層
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        # 創建因果掩碼
        mask = create_causal_mask(seq_len).to(input_ids.device)

        # Token 嵌入 + 位置編碼
        x = self.token_embedding(input_ids) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通過所有解碼器層
        for layer in self.layers:
            x = layer(x, mask)

        # 最終歸一化
        x = self.norm(x)

        # 輸出 logits
        logits = self.lm_head(x)

        return logits

# 創建模型
vocab_size = 50257  # GPT-2 詞彙表大小
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 1024
max_len = 512

model = TinyGPT(vocab_size, d_model, num_heads, num_layers, d_ff, max_len)

# 測試
batch_size = 2
seq_len = 20
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

logits = model(input_ids)

print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
print(f"輸入形狀: {input_ids.shape}")
print(f"輸出 logits 形狀: {logits.shape}")
```

---

---

## 2024-2025 年最新進展

### 新增內容

本目錄新增了以下2024-2025年LLM領域的重要內容：

1. **[2024-2025 LLM 模型重大突破](./2024-2025_LLM模型重大突破.md)**
   - OpenAI GPT-4o、o1 系列（推理模型）
   - Anthropic Claude 3.5 系列（Computer Use 功能）
   - Google Gemini 1.5/2.5 系列（超長上下文）
   - DeepSeek-V3（671B MoE，極低成本）
   - 各大模型的詳細對比與實作範例

2. **[Mixture of Experts (MoE) 架構詳解](./Mixture_of_Experts_架構詳解.md)**
   - MoE 核心概念與工作原理
   - DeepSeek-MoE 創新架構（細粒度專家、共享專家）
   - Mixtral 8x7B、8x22B 實現
   - 完整的 Python 實作範例
   - 訓練技巧與最佳實踐

### 重要技術趨勢（2024-2025）

#### 1. 推理能力突破
- **OpenAI o1**：逐步推理，數學/科學任務顯著提升
- **DeepSeek-R1**：強化學習實現低成本高性能推理

#### 2. 超長上下文窗口
- **百萬 Token**：Claude 4、Gemini 2.5、GPT-4.1
- **200萬 Token**：Gemini 1.5 Pro
- 能處理整本書籍、長篇文檔、複雜多輪對話

#### 3. MoE 架構普及
- **稀疏激活**：只激活相關專家，大幅降低計算成本
- **DeepSeek-V3**：671B 參數僅激活 37B（5.5%）
- **經濟效益**：訓練成本僅 $5.576M

#### 4. 多模態整合
- **GPT-4o**：實時處理文字、視覺、音頻
- **Gemini Pro Vision**：強大的視覺理解
- **Claude 3.5**：Computer Use 功能

#### 5. 注意力機制優化
- **Flash Attention 2/3**：更快的推理速度
- **Multi-Query Attention (MQA)**：減少推理內存
- **Grouped-Query Attention (GQA)**：平衡性能與效率

### 架構演進圖

```
2017: Transformer
  ↓
2018-2019: GPT-1/2, BERT
  ↓
2020: GPT-3 (175B Dense)
  ↓
2021-2023: 規模擴大 + 對齊
  ↓
2024-2025: 多路徑演進
  ├─ 推理模型 (o1, DeepSeek-R1)
  ├─ 超長上下文 (Gemini 2M tokens)
  ├─ MoE 架構 (DeepSeek-V3 671B)
  ├─ 多模態 (GPT-4o, Claude 3.5)
  └─ 經濟高效訓練 (<$6M)
```

### 實用建議

**選擇模型時的考量：**

1. **任務類型**
   - 推理任務 → o1, DeepSeek-R1
   - 長文檔 → Gemini 1.5 Pro, Claude 4
   - 多模態 → GPT-4o, Gemini Pro Vision
   - 代碼生成 → Claude 3.5, GPT-4o

2. **成本預算**
   - 高預算：Claude Opus, GPT-4o
   - 中預算：Claude Sonnet, Gemini Flash
   - 低預算：開源模型（LLaMA, DeepSeek）

3. **部署場景**
   - 雲端 API：OpenAI, Anthropic, Google
   - 自部署：開源 MoE 模型
   - 邊緣設備：量化模型

---

## 參考資源

### 基礎資源
- [Attention is All You Need (原始論文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Transformers 文檔](https://huggingface.co/docs/transformers)

### 經典論文
- [GPT-3 論文](https://arxiv.org/abs/2005.14165)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

### 2024-2025 最新論文
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [Flash Attention 論文](https://arxiv.org/abs/2205.14135)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)