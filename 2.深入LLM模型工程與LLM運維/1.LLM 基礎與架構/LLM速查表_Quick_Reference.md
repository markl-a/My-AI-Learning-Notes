# LLM 基礎與架構速查表 (Quick Reference)

> 快速查閱 LLM 關鍵概念、公式、程式碼片段和最佳實踐

---

## 📐 核心公式

### 自注意力機制

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **Q**: Query 矩陣 [seq_len, d_k]
- **K**: Key 矩陣 [seq_len, d_k]
- **V**: Value 矩陣 [seq_len, d_v]
- **d_k**: Key 向量維度

### 多頭注意力

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Transformer 前饋網絡

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

通常 `d_ff = 4 × d_model`

### 位置編碼

**正弦/余弦編碼:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**RoPE (旋轉位置編碼):**
```
RoPE(x_m, m) = R_Θ,m · x_m
```
其中 R 是旋轉矩陣

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

---

## 🏗️ 模型架構對比

### Transformer 變體

| 架構 | Encoder | Decoder | 代表模型 | 用途 |
|------|---------|---------|---------|------|
| Encoder-only | ✅ | ❌ | BERT | 理解任務 |
| Decoder-only | ❌ | ✅ | GPT, LLaMA | 生成任務 |
| Encoder-Decoder | ✅ | ✅ | T5, BART | 翻譯, 總結 |

### GPT 系列演進

| 模型 | 參數 | 層數 | 上下文 | 發布時間 |
|------|------|------|--------|---------|
| GPT-1 | 117M | 12 | 512 | 2018 |
| GPT-2 | 1.5B | 48 | 1024 | 2019 |
| GPT-3 | 175B | 96 | 2048 | 2020 |
| GPT-3.5 | ~175B | - | 4096 | 2022 |
| GPT-4 | ~1.8T (MoE) | - | 32K/128K | 2023 |
| GPT-4o | - | - | 128K | 2024 |
| GPT-5 (preview) | 未公開 | 未公開 | 標稱長 context | 2025/Q4 開始 preview，規格未官方完整發布 |

### 開源模型對比 (2024-2025)

| 模型 | 參數 | 架構 | 上下文 | 特點 |
|------|------|------|--------|------|
| LLaMA 3 | 8B, 70B, 405B | Decoder | 128K | Meta 開源,強大基座 |
| LLaMA 4 | - | MoE | 1M | 首次採用 MoE |
| Mistral 7B | 7B | Decoder + SW | 32K | 滑動窗口注意力 |
| Mixtral 8x7B | 47B (激活 12.8B) | MoE | 32K | 稀疏專家混合 |
| DeepSeek-V3 | 671B (激活 37B) | MoE | 128K | 極低成本 ($5.5M) |
| Qwen 2.5 | 0.5B-72B | Decoder | 128K | 中文優秀 |

---

## 🔢 參數規模速算

### 計算 Transformer 參數量

**嵌入層:**
```
Embedding = vocab_size × d_model
```

**單層 Transformer:**
```
Attention = 4 × d_model² (QKV 投影 + 輸出投影)
FFN = 2 × d_model × d_ff (兩個線性層)
Layer = Attention + FFN
```

**總參數:**
```
Total ≈ Embedding + num_layers × (4 × d_model² + 2 × d_model × d_ff)
```

**示例: LLaMA 7B**
```
d_model = 4096
num_layers = 32
d_ff = 11008
vocab_size = 32000

Embedding = 32000 × 4096 = 131M
Per Layer = 4 × 4096² + 2 × 4096 × 11008 ≈ 157M
Total = 131M + 32 × 157M ≈ 5.15B

實際: 6.7B (包含 LN 等)
```

### 訓練成本估算

**FLOPs 計算:**
```
FLOPs = 6 × parameters × tokens
```

**GPU 時間估算:**
```
GPU_hours = FLOPs / (GPU_TFLOPS × 3600)
```

**示例:**
```
7B 模型, 2T tokens, A100 (312 TFLOPS)
FLOPs = 6 × 7B × 2T = 84e21
GPU_hours = 84e21 / (312e12 × 3600) ≈ 75,000 小時
```

---

## 💾 內存計算

### 模型內存占用

**參數存儲:**
```
FP32: 4 bytes/param
FP16/BF16: 2 bytes/param
INT8: 1 byte/param
```

**示例:**
```
7B 模型
FP32: 7B × 4 = 28 GB
FP16: 7B × 2 = 14 GB
INT8: 7B × 1 = 7 GB
```

### KV 快取占用

```
KV_cache = 2 × batch × seq_len × num_layers × d_model × bytes_per_element
```

**示例: LLaMA 7B, FP16**
```
batch = 1, seq_len = 2048
KV_cache = 2 × 1 × 2048 × 32 × 4096 × 2 ≈ 1 GB
```

### 訓練內存估算

**總內存 (混合精度訓練):**
```
Total = Model (FP16) + Gradients (FP16) + Optimizer States (FP32) + Activations
      ≈ 2 × params + 2 × params + 12 × params + activations
      ≈ 16 × params + activations
```

**7B 模型:**
```
16 × 7B × 2 = 224 GB (不含激活值)
```

需要 **4+ A100 (80GB)** GPU

---

## ⚡ 注意力機制對比

| 機制 | 時間復雜度 | 空間復雜度 | KV 快取 | 推理加速 | 適用場景 |
|------|-----------|-----------|---------|---------|---------|
| Standard MHA | O(n²d) | O(n²) | 100% | 1.0x | 通用 |
| Flash Attention 2 | O(n²d) | O(n) | 100% | 5x | 訓練 |
| Flash Attention 3 | O(n²d) | O(n) | 100% | 9x | H100 訓練 |
| MQA | O(n²d) | O(n²) | 3% | 1.5x | 推論優化 |
| GQA (8組) | O(n²d) | O(n²) | 25% | 1.3x | 平衡 |
| Sliding Window | O(nWd) | O(nW) | 100% | 1.0x | 長序列 |
| Paged Attention | O(n²d) | O(n) | 100%* | 1.2x | 推論服務 |

---

## 🎯 Tokenization 對比

### 主流 Tokenizer

| Tokenizer | 詞彙表大小 | 演算法 | 使用模型 | 特點 |
|-----------|-----------|------|---------|------|
| GPT-2 BPE | 50,257 | BPE | GPT-2/3 | 字節級 BPE |
| GPT-4 tiktoken | ~100,000 | BPE | GPT-4 | 更高效 |
| BERT WordPiece | 30,522 | WordPiece | BERT | 子詞分割 |
| LLaMA SentencePiece | 32,000 | BPE | LLaMA | 多語言 |
| Qwen2 | 151,643 | BPE | Qwen 2 | 大詞彙表 |

### Token 數量估算

**英文:**
```
1 token ≈ 0.75 words
1000 words ≈ 1333 tokens
```

**中文:**
```
1 字符 ≈ 1-2 tokens (取決於 tokenizer)
```

### 程式碼示例

```python
from transformers import AutoTokenizer

# 快速使用
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, how are you?"
tokens = tokenizer.encode(text)
# 輸出: [15496, 11, 703, 389, 345, 30]

# 解碼
decoded = tokenizer.decode(tokens)
# 輸出: "Hello, how are you?"

# Token 數量
num_tokens = len(tokens)
```

---

## 🛠️ 常用程式碼片段

### 加載預訓練模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加載模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # 自動分配到 GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 文字生成

```python
# 準備輸入
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 解碼
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### LoRA 微調

```python
from peft import LoraConfig, get_peft_model

# LoRA 配置
lora_config = LoraConfig(
    r=8,                          # LoRA 秩
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 目標層
    task_type="CAUSAL_LM"
)

# 應用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4.2M || all params: 6.7B || trainable%: 0.063%
```

### 量化加載 (4-bit)

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加載量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 內存占用: 7B 模型 ~3.5 GB
```

### Flash Attention 使用

```python
from transformers import AutoModelForCausalLM

# 啟用 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # 使用 FA2
    device_map="auto"
)
```

### 批次推理

```python
# 準備多個提示
prompts = [
    "What is AI?",
    "Explain quantum computing",
    "Write a poem about stars"
]

# Tokenize
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(model.device)

# 批次生成
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    pad_token_id=tokenizer.pad_token_id
)

# 解碼所有結果
results = [tokenizer.decode(output, skip_special_tokens=True)
           for output in outputs]
```

---

## 📊 訓練超參數參考

### 預訓練

| 參數 | 推薦值 | 說明 |
|------|--------|------|
| Learning Rate | 6e-4 to 3e-4 | 大模型用較小值 |
| Batch Size | 512-4096 | 使用梯度累積 |
| Warmup Steps | 1-5% 總步數 | 線性增長 |
| Weight Decay | 0.1 | AdamW 正則化 |
| Gradient Clipping | 1.0 | 防止梯度爆炸 |
| β₁, β₂ | 0.9, 0.95 | AdamW 預設 |

### 監督式微調 (SFT)

| 參數 | 推薦值 | 說明 |
|------|--------|------|
| Learning Rate | 1e-5 to 5e-5 | 比預訓練小 |
| Batch Size | 4-32 | 取決於 GPU 內存 |
| Epochs | 3-5 | 避免過擬合 |
| Warmup Ratio | 0.03 | 3% 總步數 |
| LR Scheduler | Cosine | 平滑衰減 |

### LoRA 微調

| 參數 | 推薦值 | 說明 |
|------|--------|------|
| r (秩) | 8, 16, 32 | 越大越強但成本高 |
| alpha | 16, 32 | 通常是 r 的 2 倍 |
| dropout | 0.05-0.1 | 防止過擬合 |
| target_modules | q_proj, v_proj | 最常用 |
| Learning Rate | 3e-4 | 可以比 SFT 大 |

### RLHF/DPO

| 參數 | 推薦值 | 說明 |
|------|--------|------|
| Learning Rate | 5e-7 to 1e-6 | 非常小 |
| β (DPO) | 0.1-0.5 | 溫度參數 |
| PPO clip ε | 0.2 | PPO 裁剪範圍 |
| KL coef | 0.1 | KL 散度權重 |

---

## 🚀 推論優化技巧

### 1. KV 快取

```python
# 啟用 KV 快取加速自回歸生成
past_key_values = None

for _ in range(max_new_tokens):
    outputs = model(
        input_ids,
        past_key_values=past_key_values,
        use_cache=True
    )

    # 獲取下一個 token
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # 更新快取
    past_key_values = outputs.past_key_values

    # 只需要處理新 token
    input_ids = next_token
```

### 2. 投機解碼 (Speculative Decoding)

```python
# 使用小模型草稿 + 大模型驗證
draft_model = AutoModelForCausalLM.from_pretrained("small-model")
target_model = AutoModelForCausalLM.from_pretrained("large-model")

# 草稿模型快速生成 K 個 token
draft_tokens = draft_model.generate(input_ids, max_new_tokens=5)

# 大模型並行驗證
with torch.no_grad():
    logits = target_model(draft_tokens).logits

# 接受正確的 tokens, 拒絕錯誤的
# 實際加速: 2-3x
```

### 3. 連續批處理 (Continuous Batching)

由 vLLM 等推理框架自動處理:
- 動態添加/移除請求
- 最大化 GPU 利用率
- 提升 2-4x 吞吐量

---

## 📈 性能基準參考

### MMLU (Massive Multitask Language Understanding)

| 模型 | MMLU 分數 | 參數規模 |
|------|----------|---------|
| GPT-3.5 | 70.0 | ~175B |
| GPT-4 | 86.4 | ~1.8T MoE |
| Claude 3 Opus | 86.8 | - |
| Claude 3.5 Sonnet | 88.7 | - |
| Gemini 1.5 Pro | 85.9 | - |
| LLaMA 3 70B | 79.2 | 70B |
| LLaMA 3 405B | 86.0 | 405B |
| DeepSeek-V3 | 88.5 | 671B MoE |

### 程式碼生成 (HumanEval)

| 模型 | Pass@1 | 參數規模 |
|------|--------|---------|
| GPT-3.5 | 48.1 | ~175B |
| GPT-4 | 67.0 | - |
| Claude 3.5 Sonnet | 92.0 | - |
| CodeLlama 34B | 48.8 | 34B |
| DeepSeek-Coder 33B | 56.1 | 33B |

### 數學推理 (MATH-500)

| 模型 | 準確率 | 類型 |
|------|--------|------|
| GPT-4 | 52.9 | 通用 |
| o1-preview | 74.4 | 推理 |
| o1 | 83.3 | 推理 |
| Claude 3.5 Sonnet | 71.1 | 通用 |
| DeepSeek-V3 | 90.2 | 通用 + 推論優化 |
| DeepSeek-R1 | 79.8 | 推理 |

---

## 🔧 常見問題速查

### Q: 如何選擇模型大小?

**任務類型決定:**
- 簡單分類/QA: 1B-7B
- 複雜推理: 13B-70B
- 專業領域/極致性能: 70B+

**資源限制:**
- 1x A100 (80GB): 最大 70B (INT8)
- 1x RTX 4090 (24GB): 最大 13B (FP16) 或 30B (INT4)
- CPU 推理: 最大 7B (量化)

### Q: 訓練還是微調?

| 場景 | 方法 | 成本 |
|------|------|------|
| 有大量資料 (>100B tokens) | 預訓練 | 極高 |
| 任務適配 (10K-100K 樣本) | SFT | 中 |
| 領域適配 | LoRA | 低 |
| 改善對話質量 | RLHF/DPO | 中-高 |
| 提示工程 | 零樣本/少樣本 | 極低 |

### Q: 如何減少推論延遲?

**方法優先級:**
1. Flash Attention (5-9x)
2. 量化 INT8/INT4 (2-4x)
3. KV 快取 (必須)
4. 批次處理 (提升吞吐量)
5. 投機解碼 (2-3x)
6. 使用推理框架 (vLLM, TensorRT-LLM)

### Q: OOM (Out of Memory) 怎麼辦?

**訓練時:**
1. 減小 batch size
2. 梯度累積
3. 梯度檢查點
4. 混合精度 (FP16)
5. DeepSpeed ZeRO
6. 使用 LoRA 而非全參數微調

**推理時:**
1. 量化模型 (INT8/INT4)
2. 減小 batch size
3. 縮短最大序列長度
4. 使用 Paged Attention (vLLM)
5. 使用 MQA/GQA 模型

---

## 🎓 學習路徑建議

### 初學者 (0-3 個月)

1. ✅ 理解 Transformer 架構
2. ✅ 掌握注意力機制原理
3. ✅ 學會使用 Hugging Face
4. ✅ 實現簡單的文字生成
5. ✅ 了解 tokenization

### 進階 (3-6 個月)

1. ✅ 微調開源模型 (LoRA)
2. ✅ 理解訓練流程 (SFT, RLHF)
3. ✅ 掌握提示工程
4. ✅ 學習 RAG 技術
5. ✅ 部署推論服務

### 專家 (6+ 個月)

1. ✅ 從零預訓練小型模型
2. ✅ 實現自定義架構改進
3. ✅ 分佈式訓練優化
4. ✅ 深入推論優化
5. ✅ 貢獻開源項目

---

## 📚 推薦資源

### 教程與課程
- [Hugging Face Course](https://huggingface.co/learn)
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)
- [DeepLearning.AI LLM Course](https://www.deeplearning.ai/)

### 論文必讀
- Attention Is All You Need (Transformer)
- BERT, GPT-2, GPT-3 系列
- Flash Attention 1/2/3
- LoRA, RLHF, DPO

### 開源項目
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

### 實用工具
- **模型下載**: Hugging Face Hub, ModelScope
- **訓練框架**: Axolotl, TRL, LLaMA-Factory
- **推論服務**: vLLM, TGI, TensorRT-LLM
- **監控工具**: Weights & Biases, TensorBoard

---

## 🔗 快速鏈接

- [Hugging Face 模型庫](https://huggingface.co/models)
- [Papers With Code](https://paperswithcode.com/)
- [Ar5iv (論文瀏覽器)](https://ar5iv.org/)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)

---

**最後更新**: 2025-01

**說明**: 本速查表包含 LLM 領域的核心概念和實用資訊,建議收藏備用。
