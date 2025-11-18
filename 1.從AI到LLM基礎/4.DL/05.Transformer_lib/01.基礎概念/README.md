# Transformers 基礎概念

本章節介紹 Hugging Face Transformers 庫的基礎概念和核心組件。

## 📚 學習目標

- 理解 Transformers 庫的整體架構
- 掌握 AutoModel 和 AutoTokenizer 的使用
- 了解模型的載入、保存和配置
- 學習不同模型類型的選擇
- 理解 Tokenizer 的工作原理

## 目錄

1. [核心組件介紹](#核心組件介紹)
2. [AutoModel 類詳解](#automodel-類詳解)
3. [Tokenizer 詳解](#tokenizer-詳解)
4. [模型配置](#模型配置)
5. [實踐範例](#實踐範例)

---

## 核心組件介紹

### 1. Transformers 庫的三大核心

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 1. Model（模型）：執行推理和訓練的神經網路
model = AutoModel.from_pretrained("bert-base-chinese")

# 2. Tokenizer（分詞器）：將文本轉換為模型可理解的數字
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 3. Config（配置）：模型的超參數和設定
config = AutoConfig.from_pretrained("bert-base-chinese")
```

### 2. Transformers 的設計哲學

Hugging Face Transformers 遵循以下設計原則：

- **統一接口**：所有模型使用相同的 API
- **模型無關**：支援 PyTorch、TensorFlow、JAX
- **易於使用**：3 行代碼即可使用任何模型
- **高度可擴展**：支援自定義模型和訓練邏輯

---

## AutoModel 類詳解

### 1. AutoModel 家族

根據不同任務選擇對應的 AutoModel 類：

#### NLP 任務
```python
from transformers import (
    AutoModel,                              # 基礎模型（獲取隱藏狀態）
    AutoModelForSequenceClassification,     # 文本分類
    AutoModelForTokenClassification,        # 標記分類（NER）
    AutoModelForQuestionAnswering,          # 問答
    AutoModelForCausalLM,                   # 因果語言模型（GPT 風格）
    AutoModelForMaskedLM,                   # 遮罩語言模型（BERT 風格）
    AutoModelForSeq2SeqLM,                  # 序列到序列（T5 風格）
)

# 範例：載入文本分類模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2,  # 二分類
)
```

#### Computer Vision 任務
```python
from transformers import (
    AutoModelForImageClassification,    # 圖像分類
    AutoModelForObjectDetection,        # 目標檢測
    AutoModelForImageSegmentation,      # 圖像分割
    AutoModelForDepthEstimation,        # 深度估計
)

# 範例：載入圖像分類模型
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

#### Audio 任務
```python
from transformers import (
    AutoModelForSpeechSeq2Seq,          # 語音識別（Whisper）
    AutoModelForAudioClassification,    # 音頻分類
    AutoModelForTextToSpectrogram,      # 文本轉語音
)

# 範例：載入語音識別模型
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")
```

#### Multimodal 任務
```python
from transformers import (
    AutoModelForVision2Seq,             # 圖像描述
    AutoModelForVisualQuestionAnswering, # 視覺問答
)

# 範例：載入視覺問答模型
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
```

### 2. 模型載入選項

```python
from transformers import AutoModelForCausalLM
import torch

# 基礎載入
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 指定設備
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto",  # 自動分配到可用設備
)

# 指定精度
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,  # 使用 FP16
)

# 量化載入（需要 bitsandbytes）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    load_in_8bit=True,  # 8-bit 量化
    device_map="auto",
)

# 從本地載入
model = AutoModelForCausalLM.from_pretrained(
    "./my_model",
    local_files_only=True,  # 僅使用本地文件
)

# 使用特定版本（commit hash 或 tag）
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    revision="main",  # 或具體的 commit hash
)
```

### 3. 模型保存

```python
# 保存完整模型
model.save_pretrained("./my_saved_model")

# 同時保存 tokenizer
tokenizer.save_pretrained("./my_saved_model")

# 保存為 safetensors 格式（更安全、更快）
model.save_pretrained("./my_saved_model", safe_serialization=True)
```

---

## Tokenizer 詳解

### 1. Tokenizer 的工作流程

```
原始文本 → 規範化 → 預分詞 → 模型分詞 → 後處理 → Token IDs
```

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 完整的分詞流程
text = "我愛使用 Hugging Face Transformers！"
encoding = tokenizer(text, return_tensors="pt")

print("Token IDs:", encoding['input_ids'])
print("Attention Mask:", encoding['attention_mask'])
print("Tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
```

### 2. Tokenizer 的重要參數

```python
# 基本參數
encoding = tokenizer(
    text,
    add_special_tokens=True,        # 添加 [CLS], [SEP] 等特殊標記
    padding=True,                   # 填充到批次最大長度
    truncation=True,                # 截斷超長序列
    max_length=512,                 # 最大長度
    return_tensors="pt",            # 返回 PyTorch tensors
    return_attention_mask=True,     # 返回 attention mask
    return_token_type_ids=True,     # 返回 token type IDs
)

# 批次處理
texts = ["第一句話", "第二句話", "第三句話"]
batch_encoding = tokenizer(
    texts,
    padding="max_length",           # 填充策略
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
```

### 3. 不同類型的 Tokenizer

#### BPE (Byte-Pair Encoding)
```python
# GPT-2, RoBERTa 使用 BPE
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Vocab size:", len(tokenizer))
```

#### WordPiece
```python
# BERT 使用 WordPiece
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unbelievable")
print(tokens)  # ['un', '##believable']
```

#### SentencePiece
```python
# T5, XLNet 使用 SentencePiece
tokenizer = AutoTokenizer.from_pretrained("t5-small")
```

#### Unigram
```python
# ALBERT, mBART 使用 Unigram
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
```

### 4. Fast Tokenizer vs Slow Tokenizer

```python
from transformers import AutoTokenizer

# Fast Tokenizer（Rust 實現，速度快 10-100 倍）
fast_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)

# Slow Tokenizer（Python 實現）
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)

# Fast Tokenizer 的額外功能
encoding = fast_tokenizer("測試文本", return_offsets_mapping=True)
print("Offsets:", encoding['offset_mapping'])  # 字符級別的偏移量
```

### 5. 特殊標記處理

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 查看特殊標記
print("PAD token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)
print("UNK token:", tokenizer.unk_token, "ID:", tokenizer.unk_token_id)
print("CLS token:", tokenizer.cls_token, "ID:", tokenizer.cls_token_id)
print("SEP token:", tokenizer.sep_token, "ID:", tokenizer.sep_token_id)
print("MASK token:", tokenizer.mask_token, "ID:", tokenizer.mask_token_id)

# 添加自定義特殊標記
tokenizer.add_special_tokens({'additional_special_tokens': ['[USER]', '[ASSISTANT]']})
model.resize_token_embeddings(len(tokenizer))  # 調整模型的 embedding 層
```

---

## 模型配置

### 1. Config 類

```python
from transformers import AutoConfig

# 載入配置
config = AutoConfig.from_pretrained("bert-base-chinese")

# 查看配置
print("Hidden size:", config.hidden_size)
print("Num attention heads:", config.num_attention_heads)
print("Num hidden layers:", config.num_hidden_layers)
print("Vocab size:", config.vocab_size)

# 修改配置
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

# 使用自定義配置創建模型
from transformers import AutoModel
model = AutoModel.from_config(config)
```

### 2. 常見配置選項

```python
from transformers import AutoConfig, AutoModelForSequenceClassification

# 創建分類模型的配置
config = AutoConfig.from_pretrained(
    "bert-base-chinese",
    num_labels=3,                   # 分類類別數
    hidden_dropout_prob=0.1,        # Dropout 比例
    attention_probs_dropout_prob=0.1,
    classifier_dropout=0.1,         # 分類器 Dropout
)

# 從配置創建模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    config=config,
)
```

---

## 實踐範例

### 範例 1：完整的推理流程

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 載入模型和 tokenizer
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 準備輸入
text = "這是一個很棒的產品！"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 3. 執行推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1)

# 4. 解釋結果
predicted_class = torch.argmax(predictions, dim=-1).item()
confidence = predictions[0][predicted_class].item()

print(f"預測類別: {predicted_class}")
print(f"信心度: {confidence:.4f}")
```

### 範例 2：批次處理

```python
# 批次處理多個文本
texts = [
    "這個產品很好用",
    "質量太差了",
    "價格合理，性價比高",
    "完全不推薦",
]

# 批次編碼
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

# 批次推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)

# 輸出結果
for i, text in enumerate(texts):
    pred_class = torch.argmax(predictions[i]).item()
    confidence = predictions[i][pred_class].item()
    print(f"文本: {text}")
    print(f"  預測: {pred_class}, 信心度: {confidence:.4f}\n")
```

### 範例 3：使用不同框架

```python
# PyTorch
from transformers import AutoModelForCausalLM
model_pt = AutoModelForCausalLM.from_pretrained("gpt2")

# TensorFlow
from transformers import TFAutoModelForCausalLM
model_tf = TFAutoModelForCausalLM.from_pretrained("gpt2")

# JAX/Flax
from transformers import FlaxAutoModelForCausalLM
model_flax = FlaxAutoModelForCausalLM.from_pretrained("gpt2")
```

---

## 常見問題

### Q1: 如何選擇合適的模型？

**答**：根據任務和資源選擇：
- **資源有限**：選擇小型模型（DistilBERT, TinyBERT）
- **中文任務**：選擇中文預訓練模型（BERT-wwm-Chinese, MacBERT）
- **多語言**：選擇多語言模型（XLM-RoBERTa, mBERT）
- **生成任務**：選擇因果語言模型（GPT, LLaMA）

### Q2: 為什麼需要 attention_mask？

**答**：`attention_mask` 告訴模型哪些 token 是真實的（值為 1），哪些是填充的（值為 0）。這樣模型在計算注意力時會忽略填充的部分。

### Q3: 如何處理超長文本？

**答**：
1. **截斷**：使用 `truncation=True` 截取前 N 個 token
2. **分段處理**：將長文本切分為多個片段分別處理
3. **使用長序列模型**：如 Longformer, BigBird

### Q4: Fast Tokenizer 和 Slow Tokenizer 的區別？

**答**：
- **Fast**：Rust 實現，速度快，支援更多功能（offset mapping, word_ids）
- **Slow**：Python 實現，兼容性更好
- 大多數情況下優先使用 Fast Tokenizer

---

## 延伸閱讀

- [Transformers 官方文檔](https://huggingface.co/docs/transformers/)
- [Tokenizers 文檔](https://huggingface.co/docs/tokenizers/)
- [模型配置參數詳解](https://huggingface.co/docs/transformers/main_classes/configuration)
- [AutoClass 使用指南](https://huggingface.co/docs/transformers/model_doc/auto)

---

## 下一步

完成本章節後，你應該：
- ✅ 理解 Transformers 庫的核心組件
- ✅ 掌握模型和 tokenizer 的載入、保存
- ✅ 了解不同類型的 tokenizer 和模型

接下來，請前往：
- [02. Pipeline API](../02.Pipeline_API/) - 學習快速使用預訓練模型
- [03. 模型微調](../03.模型微調/) - 學習如何微調模型以適應特定任務
