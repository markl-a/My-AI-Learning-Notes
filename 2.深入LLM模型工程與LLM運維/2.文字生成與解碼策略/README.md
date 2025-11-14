# 文字生成與解碼策略

## 目錄
1. [前言](#1-前言)
2. [基本解碼策略](#2-基本解碼策略)
   - 2.1 [貪婪搜尋 (Greedy Decoding)](#21-貪婪搜尋-greedy-decoding)
   - 2.2 [Beam Search](#22-beam-search)
3. [隨機採樣策略](#3-隨機採樣策略)
   - 3.1 [基礎隨機採樣](#31-基礎隨機採樣)
   - 3.2 [Temperature Sampling](#32-temperature-sampling)
   - 3.3 [Top-k Sampling](#33-top-k-sampling)
   - 3.4 [Top-p (Nucleus) Sampling](#34-top-p-nucleus-sampling)
   - 3.5 [Top-a Sampling](#35-top-a-sampling)
4. [進階解碼技術](#4-進階解碼技術)
   - 4.1 [Contrastive Search](#41-contrastive-search)
   - 4.2 [Typical Sampling](#42-typical-sampling)
   - 4.3 [Mirostat Sampling](#43-mirostat-sampling)
5. [策略比較與選擇指南](#5-策略比較與選擇指南)
6. [Python 實作範例](#6-python-實作範例)
7. [參數調優指南](#7-參數調優指南)
8. [延伸閱讀](#8-延伸閱讀)

---

## 1. 前言

在大型語言模型 (LLM) 中，文字生成 (Text Generation) 是將模型的輸出機率分佈轉換為實際文字序列的過程。給定上下文 x₁, ..., xₜ，模型預測下一個 token 的機率分佈 P(xₜ₊₁ | x₁, ..., xₜ)。

**解碼 (Decoding)** 是從這個機率分佈中選擇下一個 token 的策略。不同的解碼策略會產生：
- 不同的文字品質
- 不同的多樣性程度
- 不同的生成速度
- 不同的應用場景適配性

選擇合適的解碼策略對於生成高質量、符合需求的文字至關重要。

---

## 2. 基本解碼策略

### 2.1 貪婪搜尋 (Greedy Decoding)

#### 原理

每一步選擇機率最高的 token：

```
xₜ₊₁ = argmax P(x | x₁, ..., xₜ)
         x
```

#### 優點
- **簡單高效**：計算複雜度 O(1)
- **確定性**：相同輸入產生相同輸出
- **速度快**：無需比較多個候選

#### 缺點
- **缺乏多樣性**：總是選擇最可能的詞
- **局部最優**：可能錯過全局最優序列
- **重複問題**：容易產生重複的詞或短語

#### 適用場景
- 事實性問答
- 翻譯任務（準確性優先）
- 需要確定性輸出的應用

#### 數學範例

假設模型輸出分佈：
```
P("the") = 0.5
P("a") = 0.3
P("an") = 0.2
```

貪婪解碼選擇："the"

### 2.2 Beam Search

#### 原理

維護 k 個最可能的序列（beam），每步擴展所有 beam 並保留 top-k：

```
在時間步 t：
1. 對每個 beam 生成所有可能的下一個 token
2. 計算累積對數機率：log P(x₁...xₜ) = Σᵢ log P(xᵢ|x₁...xᵢ₋₁)
3. 保留總機率最高的 k 個序列
```

#### 參數

- **beam_size (k)**：保留的候選序列數量
  - k = 1：等同於貪婪搜尋
  - k = 5~10：常用範圍
  - k 越大：搜尋越全面，但計算成本越高

- **length_penalty (α)**：調整序列長度偏好
  ```
  score = log P(x₁...xₜ) / tᵅ
  ```
  - α = 0：無懲罰
  - α > 0：鼓勵較長序列
  - α < 0：鼓勵較短序列

#### 優點
- **更好的全局最優性**：考慮多個候選路徑
- **適合目標明確的任務**：翻譯、摘要
- **可控的探索範圍**：通過 beam_size 調節

#### 缺點
- **計算成本高**：O(k × vocab_size)
- **缺乏多樣性**：beam 內的序列趨於相似
- **曝光偏差 (Exposure Bias)**：訓練和推理分佈不匹配
- **重複問題**：仍可能生成重複短語

#### 適用場景
- 機器翻譯
- 文本摘要
- 圖像描述生成
- 需要高質量但不需要創意的任務

---

## 3. 隨機採樣策略

### 3.1 基礎隨機採樣

#### 原理

根據模型輸出的機率分佈隨機採樣：

```
xₜ₊₁ ~ P(x | x₁, ..., xₜ)
```

#### 優缺點

**優點**：
- 生成多樣性高
- 避免確定性帶來的重複問題

**缺點**：
- 可能採樣到低機率、不連貫的詞
- 輸出質量不穩定

### 3.2 Temperature Sampling

#### 原理

通過溫度參數 T 調整機率分佈的"平滑度"：

```
P'(xᵢ) = exp(logits_i / T) / Σⱼ exp(logits_j / T)
```

其中 logits 是模型的原始輸出（softmax 之前）。

#### 溫度效果

- **T = 1.0**：原始分佈，模型的自然輸出
- **T → 0**：接近貪婪搜尋，選擇最可能的詞
- **T < 1.0** (e.g., 0.7)：更確定，降低隨機性
  - 分佈更尖銳
  - 高機率詞更可能被選中
  - 適合需要準確性的任務

- **T > 1.0** (e.g., 1.5)：更隨機，增加多樣性
  - 分佈更平滑
  - 低機率詞也有機會被選中
  - 適合創意生成

#### 數學範例

原始 logits: [3.0, 2.0, 1.0]

**T = 1.0**：
```
P = softmax([3.0, 2.0, 1.0]) = [0.665, 0.245, 0.090]
```

**T = 0.5** (更確定)：
```
P = softmax([6.0, 4.0, 2.0]) = [0.843, 0.114, 0.043]
```

**T = 2.0** (更隨機)：
```
P = softmax([1.5, 1.0, 0.5]) = [0.506, 0.307, 0.187]
```

#### 適用場景

- **低溫 (0.1-0.7)**：程式碼生成、數學推理、事實性問答
- **中溫 (0.7-1.0)**：一般對話、文章寫作
- **高溫 (1.0-2.0)**：創意寫作、腦力激盪、故事生成

### 3.3 Top-k Sampling

#### 原理

只從機率最高的 k 個 token 中採樣：

```
1. 選出機率最高的 k 個 token
2. 重新歸一化這 k 個 token 的機率
3. 從重新歸一化的分佈中採樣
```

#### 參數選擇

- **k = 1**：等同於貪婪搜尋
- **k = 10-50**：常用範圍
- **k = vocab_size**：等同於基礎隨機採樣

#### 優點
- **避免低質量詞**：排除極低機率的詞
- **保持多樣性**：允許合理的隨機性
- **簡單直觀**：易於理解和實現

#### 缺點
- **固定 k 不靈活**：
  - 機率分佈陡峭時（一個詞概率很高），k 可能過大
  - 機率分佈平坦時（多個詞概率相近），k 可能過小
- 無法動態適應機率分佈形狀

#### 範例

模型輸出：
```
P("the") = 0.40
P("a") = 0.30
P("an") = 0.15
P("this") = 0.10
P("that") = 0.05
...（其餘詞總和 = 0.00）
```

**Top-3 採樣**：
```
候選：["the", "a", "an"]
重新歸一化：[0.40, 0.30, 0.15] / 0.85 = [0.471, 0.353, 0.176]
從這個分佈中採樣
```

### 3.4 Top-p (Nucleus) Sampling

#### 原理

選擇累積機率超過閾值 p 的最小詞集：

```
1. 將詞按機率降序排列
2. 累加機率直到總和 ≥ p
3. 從這個子集中採樣
```

#### 動態適應性

Top-p 的核心優勢是動態調整候選集大小：

- **機率分佈陡峭**（模型很確定）：
  - 只需少數幾個詞就能達到 p
  - 候選集小，輸出更確定

- **機率分佈平坦**（模型不確定）：
  - 需要更多詞才能達到 p
  - 候選集大，保持多樣性

#### 參數選擇

- **p = 0.9-0.95**：最常用，平衡質量和多樣性
- **p = 1.0**：包含所有詞（等同於基礎採樣）
- **p = 0.5**：更保守，只選擇高機率詞

#### 優點
- **自適應**：根據機率分佈動態調整
- **避免低質量詞**：自動排除長尾低機率詞
- **保持多樣性**：在合理範圍內允許隨機性

#### 缺點
- **p 值調優**：需要根據任務調整
- **計算稍複雜**：需要排序和累積和

#### 範例

**情境 1：模型很確定**
```
P("Paris") = 0.70
P("France") = 0.20
P("London") = 0.05
P("Berlin") = 0.03
...
```
Top-p=0.9：選擇 ["Paris", "France"]（累積 0.90）

**情境 2：模型不確定**
```
P("might") = 0.15
P("could") = 0.14
P("would") = 0.13
P("may") = 0.12
P("can") = 0.11
P("should") = 0.10
...
```
Top-p=0.9：選擇前 6-7 個詞

### 3.5 Top-a Sampling

#### 原理

選擇機率大於 `max_prob / a` 的所有 token：

```
threshold = max(P(x | context)) / a
候選集 = {x : P(x | context) > threshold}
```

#### 特性

- **相對閾值**：基於最高機率動態設定
- **適應性強**：自動調整候選集大小
- **a = 10-100** 為常用範圍

---

## 4. 進階解碼技術

### 4.1 Contrastive Search

#### 原理

結合模型信心度和與已生成文本的差異性：

```
xₜ₊₁ = argmax [(1-α) × P(x|context) - α × max_similarity(x, context)]
         x
```

其中：
- α ∈ [0, 1]：平衡因子
- similarity：通常使用餘弦相似度

#### 優點
- **減少重複**：明確懲罰與歷史相似的詞
- **保持連貫性**：仍考慮機率分佈
- **生成流暢**：平衡創新和合理性

#### 適用場景
- 長文本生成
- 開放式對話
- 需要避免重複的創意寫作

### 4.2 Typical Sampling

#### 原理

選擇"典型"的 token，即信息量接近條件熵的 token：

```
典型性 = |log P(x|context) + H(P)|

選擇典型性最低的 token（最接近期望信息量）
```

#### 理論基礎

基於信息論，"典型集"包含最可能的序列，這些序列的信息量接近期望值。

#### 優點
- **理論基礎強**：信息論支撐
- **避免極端**：既不太確定也不太隨機
- **提高連貫性**：生成更自然的文本

### 4.3 Mirostat Sampling

#### 原理

動態調整採樣過程以維持目標困惑度 (perplexity)：

```
1. 設定目標困惑度 τ
2. 監控生成文本的實際困惑度
3. 動態調整溫度或閾值以維持困惑度在目標附近
```

#### 優點
- **自適應**：根據生成質量動態調整
- **穩定輸出**：維持一致的文本複雜度
- **理論優雅**：基於困惑度的控制

---

## 5. 策略比較與選擇指南

### 5.1 策略對比表

| 策略 | 多樣性 | 質量穩定性 | 計算成本 | 重複風險 | 適用場景 |
|------|--------|----------|---------|---------|---------|
| Greedy | 無 | 高 | 極低 | 高 | 事實問答、翻譯 |
| Beam Search | 低 | 高 | 高 | 中 | 翻譯、摘要 |
| Temperature | 可調 | 中 | 低 | 中 | 通用對話 |
| Top-k | 中 | 中-高 | 低 | 低 | 一般生成 |
| Top-p | 中-高 | 中 | 低 | 低 | 創意寫作、對話 |
| Contrastive | 高 | 中 | 中 | 極低 | 長文本生成 |

### 5.2 任務導向選擇指南

#### 1. **事實性任務**（翻譯、問答、摘要）
```
推薦：Beam Search 或 Greedy
配置：
- beam_size = 4-6
- temperature = 0.0-0.3
- length_penalty = 1.0
```

#### 2. **對話系統**
```
推薦：Top-p + Temperature
配置：
- temperature = 0.7-0.9
- top_p = 0.9
- repetition_penalty = 1.2
```

#### 3. **創意寫作**
```
推薦：Top-p + 高 Temperature
配置：
- temperature = 1.0-1.3
- top_p = 0.95
- top_k = 50
```

#### 4. **程式碼生成**
```
推薦：低 Temperature + Top-p
配置：
- temperature = 0.2
- top_p = 0.95
- 使用語法約束
```

#### 5. **長文本生成**
```
推薦：Contrastive Search 或 Top-p
配置：
- temperature = 0.8
- top_p = 0.92
- repetition_penalty = 1.3
```

### 5.3 組合策略

實務中常組合多種策略：

**組合 1：Temperature + Top-p + Repetition Penalty**
```python
# 平衡質量、多樣性和避免重複
temperature = 0.8
top_p = 0.9
repetition_penalty = 1.2
```

**組合 2：Beam Search + Length Penalty + No Repeat N-gram**
```python
# 結構化任務的高質量輸出
beam_size = 5
length_penalty = 1.2
no_repeat_ngram_size = 3
```

---

## 6. Python 實作範例

### 6.1 基礎解碼策略實現

```python
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# 載入模型（使用小型模型作為示範）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# 設定設備
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def prepare_input(prompt):
    """準備輸入"""
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

# ============================================================================
# 1. 貪婪解碼 (Greedy Decoding)
# ============================================================================

def greedy_decoding(prompt, max_length=50):
    """貪婪解碼實現"""
    print("=" * 60)
    print("貪婪解碼 (Greedy Decoding)")
    print("=" * 60)

    inputs = prepare_input(prompt)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()

    for _ in range(max_length):
        # 獲取模型輸出
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits

        # 取最後一個 token 的 logits
        next_token_logits = logits[:, -1, :]

        # 選擇概率最高的 token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 添加到生成序列
        generated = torch.cat([generated, next_token], dim=1)

        # 如果生成 EOS token，停止
        if next_token.item() == tokenizer.eos_token_id:
            break

    # 解碼
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"輸入: {prompt}")
    print(f"輸出: {output_text}")
    return output_text

# ============================================================================
# 2. Beam Search
# ============================================================================

def beam_search(prompt, beam_size=5, max_length=50, length_penalty=1.0):
    """Beam Search 實現"""
    print("\n" + "=" * 60)
    print(f"Beam Search (beam_size={beam_size}, length_penalty={length_penalty})")
    print("=" * 60)

    inputs = prepare_input(prompt)
    input_ids = inputs["input_ids"]

    # 初始化 beams: (序列, 累積對數概率)
    beams = [(input_ids, 0.0)]
    completed_beams = []

    for step in range(max_length):
        candidates = []

        for seq, score in beams:
            # 如果已經生成 EOS，加入完成列表
            if seq[0, -1].item() == tokenizer.eos_token_id:
                completed_beams.append((seq, score))
                continue

            # 獲取下一個 token 的概率
            with torch.no_grad():
                outputs = model(seq)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)

            # 獲取 top-k 個候選
            top_log_probs, top_indices = torch.topk(log_probs, beam_size)

            # 擴展每個候選
            for i in range(beam_size):
                next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                next_log_prob = top_log_probs[0, i].item()

                new_seq = torch.cat([seq, next_token], dim=1)

                # 計算新的累積分數（帶長度懲罰）
                seq_length = new_seq.size(1)
                new_score = (score + next_log_prob) / (seq_length ** length_penalty)

                candidates.append((new_seq, new_score * (seq_length ** length_penalty)))

        # 選擇 top beam_size 個候選
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

        # 如果所有 beams 都完成了，提前停止
        if len(completed_beams) >= beam_size:
            break

    # 如果沒有完成的 beam，使用當前的 beams
    if not completed_beams:
        completed_beams = beams

    # 選擇分數最高的序列
    best_seq, best_score = max(completed_beams, key=lambda x: x[1])
    output_text = tokenizer.decode(best_seq[0], skip_special_tokens=True)

    print(f"輸入: {prompt}")
    print(f"輸出: {output_text}")
    print(f"分數: {best_score:.4f}")

    return output_text

# ============================================================================
# 3. Temperature Sampling
# ============================================================================

def temperature_sampling(prompt, temperature=1.0, max_length=50):
    """Temperature Sampling 實現"""
    print("\n" + "=" * 60)
    print(f"Temperature Sampling (T={temperature})")
    print("=" * 60)

    inputs = prepare_input(prompt)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

        # 應用 temperature
        logits = logits / temperature

        # 計算概率分佈
        probs = F.softmax(logits, dim=-1)

        # 從分佈中採樣
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"輸入: {prompt}")
    print(f"輸出: {output_text}")
    return output_text

# ============================================================================
# 4. Top-k Sampling
# ============================================================================

def top_k_sampling(prompt, k=50, temperature=1.0, max_length=50):
    """Top-k Sampling 實現"""
    print("\n" + "=" * 60)
    print(f"Top-k Sampling (k={k}, T={temperature})")
    print("=" * 60)

    inputs = prepare_input(prompt)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

        # 應用 temperature
        logits = logits / temperature

        # Top-k 過濾
        top_k_logits, top_k_indices = torch.topk(logits, k)

        # 計算 top-k 的概率分佈
        probs = F.softmax(top_k_logits, dim=-1)

        # 從 top-k 中採樣
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices.gather(-1, next_token_idx)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"輸入: {prompt}")
    print(f"輸出: {output_text}")
    return output_text

# ============================================================================
# 5. Top-p (Nucleus) Sampling
# ============================================================================

def top_p_sampling(prompt, p=0.9, temperature=1.0, max_length=50):
    """Top-p (Nucleus) Sampling 實現"""
    print("\n" + "=" * 60)
    print(f"Top-p Sampling (p={p}, T={temperature})")
    print("=" * 60)

    inputs = prepare_input(prompt)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

        # 應用 temperature
        logits = logits / temperature

        # 計算概率並排序
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 計算累積概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累積概率超過 p 的位置
        sorted_indices_to_remove = cumulative_probs > p

        # 保留至少一個 token
        sorted_indices_to_remove[..., 0] = False

        # 將要移除的 token 概率設為 0
        sorted_probs[sorted_indices_to_remove] = 0.0

        # 重新歸一化
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # 採樣
        next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token_idx)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"輸入: {prompt}")
    print(f"輸出: {output_text}")
    return output_text

# ============================================================================
# 執行範例
# ============================================================================

if __name__ == "__main__":
    prompt = "Once upon a time"

    print("測試不同的解碼策略")
    print("=" * 60)

    # 1. 貪婪解碼
    greedy_output = greedy_decoding(prompt, max_length=30)

    # 2. Beam Search
    beam_output = beam_search(prompt, beam_size=5, max_length=30)

    # 3. Temperature Sampling (不同溫度)
    for temp in [0.5, 1.0, 1.5]:
        temperature_sampling(prompt, temperature=temp, max_length=30)

    # 4. Top-k Sampling
    topk_output = top_k_sampling(prompt, k=50, temperature=0.8, max_length=30)

    # 5. Top-p Sampling
    topp_output = top_p_sampling(prompt, p=0.9, temperature=0.8, max_length=30)
```

### 6.2 使用 Transformers 庫的高階 API

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 載入模型
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

print("=" * 80)
print("使用 Transformers 庫的 generate() 方法")
print("=" * 80)

# ============================================================================
# 1. 貪婪解碼
# ============================================================================
print("\n1. 貪婪解碼")
print("-" * 80)

output = model.generate(
    **inputs,
    max_length=50,
    do_sample=False,  # 關閉採樣，使用貪婪
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 2. Beam Search
# ============================================================================
print("\n2. Beam Search")
print("-" * 80)

output = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    length_penalty=1.0,
    early_stopping=True,
    no_repeat_ngram_size=2,  # 避免重複 2-gram
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 3. Temperature Sampling
# ============================================================================
print("\n3. Temperature Sampling (不同溫度)")
print("-" * 80)

for temp in [0.3, 0.7, 1.0, 1.5]:
    output = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=temp,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"\nTemperature = {temp}:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 4. Top-k Sampling
# ============================================================================
print("\n4. Top-k Sampling")
print("-" * 80)

output = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_k=50,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 5. Top-p Sampling
# ============================================================================
print("\n5. Top-p (Nucleus) Sampling")
print("-" * 80)

output = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 6. 組合策略: Top-k + Top-p + Temperature
# ============================================================================
print("\n6. 組合策略: Top-k + Top-p + Temperature")
print("-" * 80)

output = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.2,  # 懲罰重複
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 7. Contrastive Search
# ============================================================================
print("\n7. Contrastive Search")
print("-" * 80)

output = model.generate(
    **inputs,
    max_length=50,
    penalty_alpha=0.6,  # 對比度參數
    top_k=4,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# ============================================================================
# 8. 使用 GenerationConfig
# ============================================================================
print("\n8. 使用 GenerationConfig 統一管理參數")
print("-" * 80)

generation_config = GenerationConfig(
    max_length=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id
)

output = model.generate(
    **inputs,
    generation_config=generation_config
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 6.3 策略效果視覺化

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def visualize_sampling_strategies():
    """視覺化不同採樣策略的效果"""

    # 模擬一個機率分佈（詞彙表大小 = 20）
    np.random.seed(42)
    logits = np.random.randn(20)
    logits[0] = 5.0  # 讓第一個詞有很高的機率
    logits[1] = 3.0
    logits[2] = 2.0

    # 原始概率分佈
    probs = F.softmax(torch.tensor(logits), dim=0).numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('不同解碼策略的概率分佈比較', fontsize=16)

    x = np.arange(len(probs))

    # 1. 原始分佈
    axes[0, 0].bar(x, probs, alpha=0.7, color='blue')
    axes[0, 0].set_title('原始分佈 (T=1.0)')
    axes[0, 0].set_xlabel('Token ID')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 貪婪（選擇最高的）
    greedy_probs = np.zeros_like(probs)
    greedy_probs[np.argmax(probs)] = 1.0
    axes[0, 1].bar(x, greedy_probs, alpha=0.7, color='red')
    axes[0, 1].set_title('貪婪解碼')
    axes[0, 1].set_xlabel('Token ID')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Temperature = 0.5 (更確定)
    temp_low = 0.5
    probs_temp_low = F.softmax(torch.tensor(logits) / temp_low, dim=0).numpy()
    axes[0, 2].bar(x, probs_temp_low, alpha=0.7, color='green')
    axes[0, 2].set_title(f'Temperature = {temp_low}')
    axes[0, 2].set_xlabel('Token ID')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Temperature = 2.0 (更隨機)
    temp_high = 2.0
    probs_temp_high = F.softmax(torch.tensor(logits) / temp_high, dim=0).numpy()
    axes[1, 0].bar(x, probs_temp_high, alpha=0.7, color='orange')
    axes[1, 0].set_title(f'Temperature = {temp_high}')
    axes[1, 0].set_xlabel('Token ID')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Top-k = 5
    k = 5
    topk_probs = probs.copy()
    topk_indices = np.argsort(topk_probs)[::-1]
    topk_probs[topk_indices[k:]] = 0
    topk_probs = topk_probs / topk_probs.sum()
    axes[1, 1].bar(x, topk_probs, alpha=0.7, color='purple')
    axes[1, 1].set_title(f'Top-k (k={k})')
    axes[1, 1].set_xlabel('Token ID')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Top-p = 0.9
    p = 0.9
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.where(cumsum_probs > p)[0][0] + 1

    topp_probs = np.zeros_like(probs)
    topp_probs[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
    topp_probs = topp_probs / topp_probs.sum()
    axes[1, 2].bar(x, topp_probs, alpha=0.7, color='brown')
    axes[1, 2].set_title(f'Top-p (p={p})')
    axes[1, 2].set_xlabel('Token ID')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sampling_strategies_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("視覺化已儲存: sampling_strategies_comparison.png")

    # 列印統計資訊
    print("\n" + "=" * 60)
    print("各策略選擇的候選詞數量:")
    print("=" * 60)
    print(f"原始分佈: 所有 {len(probs)} 個詞")
    print(f"貪婪: 1 個詞")
    print(f"Temperature={temp_low}: 所有 {len(probs)} 個詞 (但分佈更尖銳)")
    print(f"Temperature={temp_high}: 所有 {len(probs)} 個詞 (但分佈更平滑)")
    print(f"Top-k (k={k}): {k} 個詞")
    print(f"Top-p (p={p}): {cutoff_idx} 個詞")

# 執行視覺化
visualize_sampling_strategies()
```

### 6.4 實際應用範例：對話生成

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

def chatbot_demo():
    """對話機器人範例：使用不同策略生成回應"""

    # 使用對話模型（如 DialoGPT）
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 設定 pad_token
    tokenizer.pad_token = tokenizer.eos_token

    print("=" * 80)
    print("對話生成範例：比較不同解碼策略")
    print("=" * 80)

    # 對話歷史
    conversation_history = ""

    # 用戶輸入
    user_inputs = [
        "Hello! How are you?",
        "What do you think about artificial intelligence?",
        "Can you write a poem?"
    ]

    for user_input in user_inputs:
        print(f"\n{'=' * 80}")
        print(f"用戶: {user_input}")
        print(f"{'=' * 80}")

        # 更新對話歷史
        conversation_history += user_input + tokenizer.eos_token
        inputs = tokenizer(conversation_history, return_tensors="pt")

        # 策略 1: 貪婪 (確定性，適合事實性回答)
        print("\n策略 1: 貪婪解碼")
        output = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        print(f"回應: {response}")

        # 策略 2: Top-p + Temperature (平衡，適合一般對話)
        print("\n策略 2: Top-p + Temperature (p=0.9, T=0.8)")
        output = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        print(f"回應: {response}")

        # 策略 3: 高 Temperature (創意性，適合開放性問題)
        print("\n策略 3: 高 Temperature + Top-p (p=0.95, T=1.2)")
        output = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 50,
            do_sample=True,
            top_p=0.95,
            temperature=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        print(f"回應: {response}")

# 執行對話範例
chatbot_demo()
```

---

## 7. 參數調優指南

### 7.1 Temperature 調優

| 溫度範圍 | 效果 | 適用場景 |
|---------|------|---------|
| 0.1 - 0.3 | 極確定，接近貪婪 | 程式碼生成、數學問題、需要精確答案 |
| 0.4 - 0.6 | 較確定，稍有變化 | 技術寫作、專業文檔 |
| 0.7 - 0.9 | 平衡質量與創意 | 一般對話、文章寫作 |
| 1.0 - 1.3 | 高創意，較隨機 | 創意寫作、腦力激盪 |
| > 1.5 | 極隨機，可能不連貫 | 實驗性生成 |

### 7.2 Top-p 調優

| p 值 | 效果 | 適用場景 |
|------|------|---------|
| 0.5 - 0.7 | 保守，只選高機率詞 | 事實性任務 |
| 0.8 - 0.9 | 平衡，最常用 | 一般對話、寫作 |
| 0.95 - 0.99 | 包容，允許更多變化 | 創意任務 |
| 1.0 | 無過濾 | 等同於基礎採樣 |

### 7.3 Repetition Penalty 調優

```python
# 避免重複的參數設定
repetition_penalty = 1.2  # 1.0 = 無懲罰，> 1.0 懲罰重複
no_repeat_ngram_size = 3  # 禁止重複的 n-gram 長度
```

**經驗法則**：
- 短文本（<100 tokens）：`repetition_penalty = 1.1-1.2`
- 長文本（>100 tokens）：`repetition_penalty = 1.2-1.5`
- 對話：`repetition_penalty = 1.2`, `no_repeat_ngram_size = 2-3`

### 7.4 組合策略建議

#### 配置 1：平衡配置（通用）
```python
generation_config = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "do_sample": True
}
```

#### 配置 2：保守配置（事實性任務）
```python
generation_config = {
    "temperature": 0.3,
    "top_p": 0.85,
    "repetition_penalty": 1.1,
    "do_sample": True
}
```

#### 配置 3：創意配置（創作任務）
```python
generation_config = {
    "temperature": 1.1,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.3,
    "do_sample": True
}
```

#### 配置 4：結構化配置（翻譯/摘要）
```python
generation_config = {
    "num_beams": 5,
    "length_penalty": 1.0,
    "early_stopping": True,
    "no_repeat_ngram_size": 3
}
```

---

## 8. 延伸閱讀

### 論文

1. **Beam Search**
   - "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)

2. **Top-k Sampling**
   - "Hierarchical Neural Story Generation" (Fan et al., 2018)

3. **Top-p (Nucleus) Sampling**
   - "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019)
   - 重要貢獻：提出 Nucleus Sampling 解決重複問題

4. **Contrastive Search**
   - "A Contrastive Framework for Neural Text Generation" (Su et al., 2022)
   - GitHub: https://github.com/yxuansu/SimCTG

5. **Typical Sampling**
   - "Typical Decoding for Natural Language Generation" (Meister et al., 2022)

6. **Mirostat**
   - "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity" (Basu et al., 2020)

### 實用資源

- **Hugging Face 文檔**: https://huggingface.co/docs/transformers/main_classes/text_generation
- **OpenAI API 文檔**: https://platform.openai.com/docs/api-reference/completions
- **LangChain Generation 文檔**: https://python.langchain.com/docs/modules/model_io/llms/

### 實戰建議

1. **從簡單開始**：先用 Greedy 或 Beam Search 建立 baseline
2. **迭代調優**：根據任務特性逐步調整參數
3. **人工評估**：自動指標（如 BLEU）可能不反映真實質量
4. **A/B 測試**：在實際應用中比較不同策略
5. **監控重複**：使用 `repetition_penalty` 和 `no_repeat_ngram_size`
6. **考慮效率**：Beam Search 計算成本高，權衡質量與速度

### 開源工具

- **vLLM**: 高效推理引擎，支援各種採樣策略
- **TensorRT-LLM**: NVIDIA 的優化推理框架
- **llama.cpp**: 輕量級 LLM 推理工具
- **text-generation-webui**: 互動式文本生成 UI

---

## 總結

選擇合適的解碼策略是 LLM 應用開發的關鍵環節：

1. **理解任務需求**：事實性 vs 創意性
2. **平衡質量與多樣性**：貪婪太死板，純隨機太混亂
3. **善用組合策略**：Temperature + Top-p + Repetition Penalty
4. **實驗與迭代**：沒有萬能配置，需要根據具體任務調整
5. **監控生成質量**：使用自動和人工評估

**推薦起點**：對於大多數應用，`temperature=0.8 + top_p=0.9 + repetition_penalty=1.2` 是一個不錯的起點，然後根據實際效果進行調整。
