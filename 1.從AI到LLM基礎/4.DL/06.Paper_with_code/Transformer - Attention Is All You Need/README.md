# Transformer - Attention Is All You Need

> **論文**: Attention Is All You Need
>
> **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (Google Brain & Google Research)
>
> **發表**: NIPS 2017
>
> **論文鏈接**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
>
> **官方程式碼**: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

---

## 📋 目錄

- [簡介](#簡介)
- [核心創新](#核心創新)
- [為什麼 Transformer 如此重要](#為什麼-transformer-如此重要)
- [架構詳解](#架構詳解)
- [快速開始](#快速開始)
- [實驗結果](#實驗結果)
- [應用場景](#應用場景)
- [參考資源](#參考資源)

---

## 🎯 簡介

**Transformer** 是深度學習歷史上最具革命性的論文之一，徹底改變了序列建模和自然語言處理的範式。它拋棄了傳統的循環神經網絡（RNN）和卷積神經網絡（CNN），**僅使用注意力機制（Attention Mechanism）** 就達到了最先進的性能。

### 核心問題

在 Transformer 之前，序列建模主要依賴於：

**RNN/LSTM/GRU 的問題**:
- ❌ **序列依賴**: 必須按順序處理，無法並行化
- ❌ **長距離依賴**: 梯度消失，難以捕捉長程關係
- ❌ **計算效率**: 訓練和推論速度慢

**CNN 的問題**:
- ❌ **固定感受野**: 需要堆疊多層才能捕捉全局資訊
- ❌ **位置資訊**: 難以建模遠距離位置關係

### Transformer 的革命性突破

**核心思想: Attention is All You Need**

```
傳統方式:
- RNN: 順序處理，隱藏狀態傳遞
- CNN: 局部卷積，堆疊增加感受野

Transformer:
- 自注意力機制: 直接建模任意位置之間的關係
- 完全並行化: 所有位置同時處理
- 多頭注意力: 從多個角度捕捉資訊
```

**關鍵優勢**:
- ✅ **並行化**: 訓練速度提升 10-100 倍
- ✅ **長距離依賴**: O(1) 複雜度捕捉全局資訊
- ✅ **可解釋性**: 注意力權重可視化
- ✅ **可擴展性**: 易於擴展到超大規模模型

---

## 💡 核心創新

### 1. 自注意力機制 (Self-Attention)

**核心思想**: 計算序列中每個位置與所有其他位置的關係

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q (Query): [batch, seq_len, d_k]
    K (Key): [batch, seq_len, d_k]
    V (Value): [batch, seq_len, d_v]

    輸出: [batch, seq_len, d_v]
    """
    d_k = Q.size(-1)

    # 1. 計算注意力分數 (QK^T)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 應用掩碼（可選）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 歸一化
    attention_weights = F.softmax(scores, dim=-1)

    # 4. 加權求和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

**數學公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

其中:
- Q, K, V 分別是 Query, Key, Value 矩陣
- d_k 是 Key 的維度
- √d_k 是縮放因子（防止梯度消失）
```

**直觀理解**:
1. **Query**: "我想找什麼？"
2. **Key**: "我能提供什麼資訊？"
3. **Value**: "具體的資訊內容"
4. **Attention**: Query 和 Key 的相似度決定了對 Value 的加權

### 2. 多頭注意力 (Multi-Head Attention)

**為什麼需要多頭？**
- 從不同的表示子空間學習不同的關係
- 類似於 CNN 的多個卷積核

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 線性投影層
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1. 線性投影並分割成多頭
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 應用縮放點積注意力
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合併多頭
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        # 4. 最終線性投影
        output = self.W_O(attn_output)

        return output
```

**多頭注意力公式**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. 位置編碼 (Positional Encoding)

**問題**: 注意力機制本身沒有位置資訊（置換不變性）

**解決方案**: 添加位置編碼到輸入嵌入

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 建立位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        # 偶數位置使用 sin，奇數位置使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x
```

**位置編碼公式**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中:
- pos: 位置索引
- i: 維度索引
```

**為什麼使用 sin/cos？**
- ✅ 確定性函數（無需學習參數）
- ✅ 可以外推到更長的序列
- ✅ 相對位置關係可以通過線性變換表示

### 4. 前饋網絡 (Feed-Forward Network)

每個 Transformer 塊包含一個位置獨立的前饋網絡：

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = self.fc1(x)        # [batch, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)        # [batch, seq_len, d_model]
        return x
```

**公式**:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

通常 d_ff = 4 * d_model
```

### 5. 層歸一化與殘差連接

```python
# 每個子層都包含殘差連接和層歸一化
output = LayerNorm(x + Sublayer(x))

# 完整的 Encoder 層
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多頭自注意力 + 殘差 + 層歸一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前饋網絡 + 殘差 + 層歸一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

---

## 🏗️ 架構詳解

### Transformer 整體架構

```
                Transformer
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
     Encoder                  Decoder

Encoder (N=6 層):
Input Embedding + Positional Encoding
    ↓
┌───────────────────────┐ ×6
│ Multi-Head Attention  │
│         ↓             │
│   Add & Norm          │
│         ↓             │
│ Feed Forward Network  │
│         ↓             │
│   Add & Norm          │
└───────────────────────┘
    ↓
Encoder Output

Decoder (N=6 層):
Output Embedding + Positional Encoding
    ↓
┌────────────────────────────────┐ ×6
│ Masked Multi-Head Attention    │
│         ↓                      │
│   Add & Norm                   │
│         ↓                      │
│ Cross-Attention (Encoder-Decoder)│
│         ↓                      │
│   Add & Norm                   │
│         ↓                      │
│ Feed Forward Network           │
│         ↓                      │
│   Add & Norm                   │
└────────────────────────────────┘
    ↓
Linear + Softmax
    ↓
Output Probabilities
```

### Encoder vs Decoder

| 組件 | Encoder | Decoder |
|------|---------|---------|
| **Self-Attention** | 雙向（可以看到整個輸入） | 單向（只能看到之前的輸出，masked） |
| **Cross-Attention** | ❌ 無 | ✅ 有（關注 Encoder 輸出） |
| **用途** | 理解輸入序列 | 生成輸出序列 |

### 完整 Transformer 實現

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        # 嵌入層
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 輸出層
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src = self.dropout(self.positional_encoding(
            self.encoder_embedding(src)))

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decoder
        tgt = self.dropout(self.positional_encoding(
            self.decoder_embedding(tgt)))

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        # 輸出
        output = self.fc_out(tgt)
        return output
```

---

## 🚀 快速開始

### 環境需求

```bash
pip install torch torchvision
pip install transformers  # Hugging Face
pip install numpy matplotlib
pip install tokenizers sentencepiece
```

### 使用預訓練模型

```python
from transformers import AutoModel, AutoTokenizer

# 載入 BERT (基於 Transformer Encoder)
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 編碼文字
text = "Attention is all you need!"
inputs = tokenizer(text, return_tensors='pt')

# 前向傳播
outputs = model(**inputs)
hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
```

### 機器翻譯範例

```python
from transformers import MarianMTModel, MarianTokenizer

# 載入預訓練翻譯模型
model_name = 'Helsinki-NLP/opus-mt-en-zh'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# 翻譯
text = "Attention is all you need."
inputs = tokenizer(text, return_tensors='pt', padding=True)

# 生成翻譯
translated = model.generate(**inputs)
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(translated_text)  # 輸出: 注意力就是你所需要的。
```

---

## 📊 實驗結果

### WMT 2014 英德翻譯

| 模型 | BLEU | 訓練成本 (FLOPs) |
|------|------|-----------------|
| GNMT (Google) | 24.6 | - |
| ConvS2S | 25.16 | 9.6×10^18 |
| **Transformer (base)** | **27.3** | 3.3×10^18 |
| **Transformer (big)** | **28.4** | 2.3×10^19 |

### WMT 2014 英法翻譯

| 模型 | BLEU |
|------|------|
| Previous SOTA | 40.4 |
| **Transformer (big)** | **41.8** |

### 訓練效率

**Transformer 優勢**:
- ⚡ **訓練速度**: 比 LSTM/GRU 快 10-100 倍
- 💾 **內存效率**: 可並行處理整個序列
- 🎯 **質量**: 更高的 BLEU 分數

**單個訓練步驟時間** (WMT En-De):
- RNN: ~6 秒
- CNN: ~3 秒
- **Transformer**: ~0.4 秒 (使用 8 個 P100 GPU)

---

## 🎯 應用場景

### 1. 機器翻譯

**原始應用**，論文的主要任務：
```python
# 使用 T5 進行翻譯
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# 翻譯任務
input_text = "translate English to German: How are you?"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))  # "Wie geht es dir?"
```

### 2. 文字生成 (GPT 系列)

**Decoder-only Transformer**:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 文字生成
prompt = "Artificial intelligence will"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 3. 文字理解 (BERT 系列)

**Encoder-only Transformer**:
```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 文字分類
text = "This movie is amazing!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
```

### 4. 視覺 Transformer (ViT)

**將 Transformer 應用於計算機視覺**:
```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 圖像分類
# 將圖像切分為 patch，視為 "token"
```

### 5. 多模態應用 (CLIP, DALL-E)

**結合視覺和語言**:
```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# 圖文匹配
```

---

## 🌟 為什麼 Transformer 如此重要？

### 1. 範式轉變

**從 RNN 到 Transformer**:
- ✅ 擺脫順序依賴，實現完全並行化
- ✅ 直接建模長距離依賴（O(1) vs O(n)）
- ✅ 更好的可解釋性（注意力權重可視化）

### 2. 催生了現代 AI 時代

**Transformer 後裔**:
- 📝 **BERT** (2018): 預訓練語言模型
- 🤖 **GPT** (2018-2023): GPT-2, GPT-3, ChatGPT, GPT-4
- 🎨 **Vision Transformer** (2020): ViT, Swin Transformer
- 🌐 **多模態**: CLIP, DALL-E, Flamingo, GPT-4V
- 🎵 **音頻**: Whisper, MusicGen
- 🧬 **生物**: AlphaFold2, ESM

### 3. 可擴展性

**Scaling Laws**:
- 模型可以擴展到數千億參數（GPT-3: 175B）
- 性能隨規模提升（幾乎呈冪律關係）
- 訓練穩定，易於優化

### 4. 統一架構

**一個架構統治所有任務**:
- 文字、視覺、語音、多模態
- 理解、生成、翻譯、推理
- 少樣本學習、零樣本學習

---

## 📚 參考資源

### 論文與程式碼

- 📄 **原始論文**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 💻 **官方實現**: [tensor2tensor](https://github.com/tensorflow/tensor2tensor)
- 🤗 **Hugging Face Transformers**: [transformers](https://github.com/huggingface/transformers)
- 📖 **Annotated Transformer**: [harvardnlp/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### 經典實現

```python
# PyTorch 官方實現
import torch.nn as nn

model = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)
```

### 學習資源

- 🎥 **論文講解**: [Attention is All You Need - 李沐](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
- 📖 **The Illustrated Transformer**: [jalammar.github.io](http://jalammar.github.io/illustrated-transformer/)
- 📚 **Dive into Deep Learning**: [d2l.ai/chapter_attention-mechanisms](https://d2l.ai/chapter_attention-mechanisms/index.html)

### 相關論文

1. **BERT** (2018): Bidirectional Encoder Representations from Transformers
2. **GPT** (2018): Improving Language Understanding by Generative Pre-Training
3. **T5** (2019): Exploring the Limits of Transfer Learning
4. **ViT** (2020): An Image is Worth 16x16 Words
5. **Swin Transformer** (2021): Hierarchical Vision Transformer

---

## 🔬 核心概念深入

### 為什麼自注意力有效？

**1. 全局感受野**:
- 每個位置都能直接訪問所有其他位置
- 不需要堆疊多層就能捕捉長距離依賴

**2. 動態權重**:
- 注意力權重根據輸入動態計算
- 相比 CNN 的固定卷積核更靈活

**3. 並行計算**:
- 矩陣乘法可以高度並行化
- GPU 友好，訓練效率高

### Transformer 的限制

**1. 計算複雜度**:
- 自注意力複雜度: O(n²·d)
- 長序列時內存和計算開銷大

**2. 位置資訊**:
- 需要顯式編碼位置資訊
- 對於超長序列的外推能力有限

**3. 資料需求**:
- 需要大量資料才能發揮優勢
- 小資料集上可能不如 RNN/CNN

### 改進方向

**1. 高效 Transformer**:
- Linformer: 線性複雜度
- Reformer: LSH 注意力
- Performer: FAVOR+ 機制

**2. 長序列 Transformer**:
- Longformer: 局部+全局注意力
- BigBird: 稀疏注意力模式

**3. 視覺優化**:
- Swin: 滑動窗口注意力
- Pyramid Vision Transformer

---

## 📝 引用

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---

## 🏆 影響力

**統計資料** (截至 2024):
- 📄 **引用次數**: 100,000+
- ⭐ **GitHub Stars**: 200,000+ (所有實現總和)
- 🏅 **獎項**: Test of Time Award (候選)

**里程碑**:
- 🥇 WMT 2017 翻譯任務最佳性能
- 🌟 開啟了預訓練大模型時代
- 💡 啟發了 AI 的新範式
- 🚀 使 ChatGPT 等應用成為可能

---

<div align="center">
  <p><strong>⭐ Transformer 徹底改變了 AI 的未來！</strong></p>
  <p>📚 從 NLP 到 Vision | 💡 從理解到生成 | 🚀 從研究到應用</p>
  <p><i>最後更新: 2024-11-18</i></p>
</div>
