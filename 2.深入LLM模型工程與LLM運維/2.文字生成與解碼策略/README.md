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
   - 4.4 [Speculative Decoding (推測解碼)](#44-speculative-decoding-推測解碼)
   - 4.5 [Self-Consistency (自洽性解碼)](#45-self-consistency-自洽性解碼)
   - 4.6 [Classifier-Free Guidance (CFG)](#46-classifier-free-guidance-cfg)
   - 4.7 [Diverse Beam Search](#47-diverse-beam-search)
5. [約束解碼與結構化生成](#5-約束解碼與結構化生成)
   - 5.1 [為什麼需要約束解碼？](#51-為什麼需要約束解碼)
   - 5.2 [基於語法的約束解碼](#52-基於語法的約束解碼)
   - 5.3 [使用專門函式庫](#53-使用專門函式庫)
   - 5.4 [正則表達式約束](#54-正則表達式約束)
   - 5.5 [長度控制技術](#55-長度控制技術)
   - 5.6 [實戰案例：生成有效的 Python 程式碼](#56-實戰案例生成有效的-python-程式碼)
6. [策略比較與選擇指南](#6-策略比較與選擇指南)
   - 6.1 [策略對比表](#61-策略對比表)
   - 6.2 [任務導向選擇指南](#62-任務導向選擇指南)
   - 6.3 [組合策略](#63-組合策略)
7. [評估指標與性能比較](#7-評估指標與性能比較)
   - 7.1 [自動評估指標](#71-自動評估指標)
   - 7.2 [人工評估維度](#72-人工評估維度)
   - 7.3 [性能基準測試](#73-性能基準測試)
8. [實戰問題診斷與解決方案](#8-實戰問題診斷與解決方案)
   - 8.1 [常見問題與解決方案](#81-常見問題與解決方案)
   - 8.2 [調試技巧](#82-調試技巧)
9. [生產環境考量](#9-生產環境考量)
   - 9.1 [延遲 vs 質量權衡](#91-延遲-vs-質量權衡)
   - 9.2 [批次處理策略](#92-批次處理策略)
   - 9.3 [快取策略](#93-快取策略)
   - 9.4 [監控與日誌](#94-監控與日誌)
10. [Python 實作範例](#10-python-實作範例)
    - 10.1 [基礎解碼策略實現](#101-基礎解碼策略實現)
    - 10.2 [使用 Transformers 庫的高階 API](#102-使用-transformers-庫的高階-api)
    - 10.3 [策略效果視覺化](#103-策略效果視覺化)
    - 10.4 [實際應用範例：對話生成](#104-實際應用範例對話生成)
11. [參數調優指南](#11-參數調優指南)
    - 11.1 [Temperature 調優](#111-temperature-調優)
    - 11.2 [Top-p 調優](#112-top-p-調優)
    - 11.3 [Repetition Penalty 調優](#113-repetition-penalty-調優)
    - 11.4 [組合策略建議](#114-組合策略建議)
12. [延伸閱讀](#12-延伸閱讀)

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

### 4.4 Speculative Decoding (推測解碼)

#### 原理

使用較小的"草稿模型"快速生成候選 token，然後用大型目標模型並行驗證：

```
1. 草稿模型生成 k 個候選 token
2. 目標模型並行評估這 k 個 token
3. 接受前 n 個符合標準的 token（n ≤ k）
4. 重複上述過程
```

#### 數學原理

接受條件：
```
接受 token x 當且僅當：
P_target(x | context) ≥ P_draft(x | context)

或以機率 P_target(x | context) / P_draft(x | context) 接受
```

#### 優點
- **顯著加速**：在不損失質量的前提下提速 2-3 倍
- **無損方法**：輸出分佈與標準採樣完全相同
- **並行化**：充分利用現代硬體的並行能力

#### 實作考量
- 草稿模型應該是目標模型的小版本（如 1.5B vs 70B）
- 需要相同的詞彙表和 tokenizer
- GPU 記憶體需求：同時載入兩個模型

#### 適用場景
- 線上推理服務（降低延遲）
- 需要大模型質量但受限於延遲的應用
- 批次生成任務

### 4.5 Self-Consistency (自洽性解碼)

#### 原理

生成多個不同的推理路徑，然後選擇最一致的答案：

```
1. 使用採樣策略生成 N 個不同的回答
2. 提取每個回答的最終答案
3. 透過投票選出最常見的答案
```

#### 數學表示

```
最終答案 = argmax Σ 1[answer(path_i) = a]
           a     i=1...N
```

#### 優點
- **提高準確性**：特別對推理任務有效
- **魯棒性強**：減少單次生成的隨機錯誤
- **簡單有效**：無需額外訓練

#### 缺點
- **計算成本高**：需要生成 N 次（通常 N = 5-40）
- **只適合答案可枚舉的任務**：如數學題、選擇題

#### 適用場景
- 數學推理（GSM8K, MATH）
- 邏輯推理
- 程式碼生成（選擇通過最多測試的版本）
- 多步驟問答

#### 實作範例

```python
def self_consistency_decoding(prompt, model, tokenizer, num_samples=10, temperature=0.7):
    """自洽性解碼實現"""
    from collections import Counter

    # 生成多個候選答案
    answers = []
    for i in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=temperature,
            top_p=0.95
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取答案（這裡需要根據具體任務定義提取邏輯）
        answer = extract_answer(text)  # 自定義函數
        answers.append(answer)

    # 投票選出最常見答案
    answer_counts = Counter(answers)
    final_answer = answer_counts.most_common(1)[0][0]

    print(f"生成了 {num_samples} 個答案:")
    for ans, count in answer_counts.items():
        print(f"  {ans}: {count} 次")
    print(f"最終答案: {final_answer}")

    return final_answer

def extract_answer(text):
    """從生成的文本中提取答案（示例）"""
    # 數學題示例：提取最後一個數字
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None
```

### 4.6 Classifier-Free Guidance (CFG)

#### 原理

在生成過程中，同時考慮條件化和無條件化的輸出，透過引導強度控制生成：

```
logits_guided = logits_uncond + guidance_scale × (logits_cond - logits_uncond)
```

其中：
- `logits_cond`：給定條件（prompt）的輸出
- `logits_uncond`：無條件（空 prompt）的輸出
- `guidance_scale`：引導強度（通常 1.0-2.0）

#### 效果

- **guidance_scale = 1.0**：等同於正常條件生成
- **guidance_scale > 1.0**：增強對 prompt 的遵循度
- **guidance_scale < 1.0**：減弱條件影響

#### 優點
- **更好的提示遵循**：生成更符合 prompt 的內容
- **控制生成方向**：靈活調整條件影響程度
- **提高生成質量**：特別是在圖像和音頻生成中表現優異

#### 在 LLM 中的應用

雖然 CFG 最初用於擴散模型（如 Stable Diffusion），但也可應用於 LLM：

```python
def classifier_free_guidance_sampling(
    prompt,
    model,
    tokenizer,
    guidance_scale=1.5,
    max_length=50
):
    """CFG 採樣實現"""
    import torch
    import torch.nn.functional as F

    # 準備條件輸入
    cond_inputs = tokenizer(prompt, return_tensors="pt")

    # 準備無條件輸入（空 prompt）
    uncond_inputs = tokenizer("", return_tensors="pt")

    # 初始化生成序列
    generated = cond_inputs["input_ids"].clone()

    for _ in range(max_length):
        # 獲取條件化 logits
        with torch.no_grad():
            cond_outputs = model(generated)
            cond_logits = cond_outputs.logits[:, -1, :]

        # 獲取無條件 logits（使用較短的序列長度）
        uncond_len = min(generated.shape[1], uncond_inputs["input_ids"].shape[1])
        with torch.no_grad():
            uncond_outputs = model(generated[:, -uncond_len:])
            uncond_logits = uncond_outputs.logits[:, -1, :]

        # 應用 CFG
        guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

        # 從引導後的分佈採樣
        probs = F.softmax(guided_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

#### 適用場景
- 需要嚴格遵循 prompt 的生成任務
- 創意寫作（控制風格強度）
- 特定主題的內容生成

### 4.7 Diverse Beam Search

#### 原理

改進標準 Beam Search，促進生成多樣化的候選序列：

```
1. 將 beams 分成 G 組（groups）
2. 每組內進行標準 beam search
3. 添加多樣性懲罰，使不同組生成不同的序列

分數 = log P(x) - λ × diversity_penalty
```

#### 多樣性懲罰

```
diversity_penalty = Σ sim(x_t, x'_t)
                    x'_t ∈ previous_groups

其中 sim() 通常使用 hamming 距離或餘弦相似度
```

#### 優點
- **增加多樣性**：生成多個不同的候選
- **保持質量**：仍然是基於機率的搜尋
- **可控性**：透過參數調整多樣性程度

#### 參數
- `num_beams`：總 beam 數量
- `num_beam_groups`：分組數量
- `diversity_penalty`：多樣性懲罰係數（通常 0.5-1.0）

#### Transformers 實作

```python
output = model.generate(
    **inputs,
    max_length=50,
    num_beams=6,
    num_beam_groups=3,  # 分成 3 組，每組 2 個 beams
    diversity_penalty=1.0,
    num_return_sequences=3  # 返回 3 個不同的候選
)
```

#### 適用場景
- 需要多個候選答案的任務
- 創意寫作（生成多個版本）
- 機器翻譯（提供多個翻譯選項）

---

## 5. 約束解碼與結構化生成

### 5.1 為什麼需要約束解碼？

在許多實際應用中，我們需要 LLM 生成特定格式的輸出：
- **結構化數據**：JSON、XML、YAML
- **程式碼**：符合語法規則的程式碼
- **特定詞彙**：醫療術語、法律用語
- **格式限制**：固定長度、特定模式

標準解碼方法無法保證這些約束，約束解碼通過在生成過程中強制執行規則來解決這個問題。

### 5.2 基於語法的約束解碼

#### 原理

使用形式語法（如 CFG - Context-Free Grammar）來約束生成：

```
1. 定義目標格式的語法規則
2. 在每一步生成時，只考慮符合語法的 token
3. 動態維護當前合法的 token 集合
```

#### JSON 生成範例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

def json_constrained_generation(prompt, model, tokenizer, schema):
    """
    約束 LLM 生成符合指定 schema 的 JSON

    Args:
        schema: dict, JSON schema 定義
    """
    # 簡化示例：生成包含特定鍵的 JSON

    required_keys = schema.get("required", [])

    # 添加格式約束到 prompt
    schema_str = json.dumps(schema, indent=2)
    constrained_prompt = f"""{prompt}

Please respond with a valid JSON object matching this schema:
{schema_str}

JSON:"""

    inputs = tokenizer(constrained_prompt, return_tensors="pt")

    # 使用低溫度確保格式正確
    output = model.generate(
        **inputs,
        max_length=512,
        temperature=0.2,  # 低溫度提高準確性
        do_sample=True,
        top_p=0.95
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 提取 JSON 部分
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            print("生成的 JSON 無效，重試...")
            return None

    return None

# 使用範例
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

result = json_constrained_generation(
    "Generate information about a user named John",
    model,
    tokenizer,
    schema
)
print(json.dumps(result, indent=2))
```

### 5.3 使用專門函式庫

#### Outlines - 結構化文本生成

```python
# 安裝: pip install outlines

import outlines

# 1. 使用 JSON schema 約束
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

model = outlines.models.transformers("gpt2")

# 生成符合 schema 的 JSON
generator = outlines.generate.json(model, Person)
result = generator("Generate a person profile: ")

print(result)
# 輸出保證是符合 Person schema 的 JSON

# 2. 使用正則表達式約束
generator = outlines.generate.regex(
    model,
    r"\d{3}-\d{3}-\d{4}"  # 電話號碼格式
)
phone = generator("My phone number is ")
print(phone)  # 保證符合格式：xxx-xxx-xxxx

# 3. 使用選擇約束
generator = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)
sentiment = generator("This movie is great! Sentiment: ")
print(sentiment)  # 保證是三個選項之一
```

#### Guidance - 可控生成框架

```python
# 安裝: pip install guidance

from guidance import models, gen, select

# 載入模型
lm = models.Transformers("gpt2")

# 定義生成模板
lm += f"The capital of France is {gen('capital', max_tokens=10)}\n"
lm += f"Is it in Europe? {select(['Yes', 'No'], name='in_europe')}\n"

print(lm["capital"])
print(lm["in_europe"])

# 複雜範例：生成結構化對話
lm = models.Transformers("gpt2")

with lm:
    lm += "Q: What is 2+2?\n"
    lm += f"A: {gen('answer', max_tokens=10)}\n"
    lm += f"Correct: {select(['True', 'False'], name='is_correct')}"

print(f"Answer: {lm['answer']}")
print(f"Correct: {lm['is_correct']}")
```

#### LMQL - 查詢語言

```python
# 安裝: pip install lmql

import lmql

@lmql.query
def generate_person():
    '''lmql
    "Generate a person's information:\n"
    "Name: [NAME]\n"
    "Age: [AGE]\n"
    "City: [CITY]\n"

    WHERE
        STOPS_AT(NAME, "\n") and
        INT(AGE) and AGE < 100 and
        STOPS_AT(CITY, "\n") and
        CITY in ["New York", "London", "Paris", "Tokyo"]
    '''

result = generate_person()
print(result)
```

### 5.4 正則表達式約束

```python
import torch
import torch.nn.functional as F
import re

def regex_constrained_sampling(
    prompt,
    model,
    tokenizer,
    regex_pattern,
    max_length=50
):
    """
    使用正則表達式約束生成
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"].clone()

    compiled_regex = re.compile(regex_pattern)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

        # 獲取所有可能的 token
        probs = F.softmax(logits, dim=-1)

        # 測試每個 token 是否符合正則表達式
        valid_token_mask = torch.zeros_like(probs)

        current_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        # 測試 top-k 個 token
        top_k_probs, top_k_indices = torch.topk(probs, k=100)

        for i, token_id in enumerate(top_k_indices[0]):
            # 嘗試添加這個 token
            test_text = current_text + tokenizer.decode([token_id.item()])

            # 檢查是否可能匹配正則表達式（部分匹配）
            if is_partial_match(test_text, compiled_regex):
                valid_token_mask[0, token_id] = probs[0, token_id]

        # 如果沒有有效 token，放寬約束
        if valid_token_mask.sum() == 0:
            valid_token_mask = probs

        # 重新歸一化
        valid_probs = valid_token_mask / valid_token_mask.sum()

        # 採樣
        next_token = torch.multinomial(valid_probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        # 檢查是否完全匹配
        final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        if compiled_regex.fullmatch(final_text.strip()):
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

def is_partial_match(text, regex):
    """檢查文本是否可能匹配正則表達式（前綴匹配）"""
    # 簡化實現：檢查是否是某個完整匹配的前綴
    try:
        # 嘗試匹配
        if regex.match(text):
            return True
        # 檢查是否可能是有效前綴
        # （實際實現需要更複雜的邏輯）
        return True
    except:
        return False
```

### 5.5 長度控制技術

#### 精確長度控制

```python
def length_controlled_generation(
    prompt,
    model,
    tokenizer,
    target_length,
    tolerance=5
):
    """
    生成接近目標長度的文本
    """
    output = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_length=target_length + tolerance,
        min_length=target_length - tolerance,
        length_penalty=2.0,  # 鼓勵生成目標長度
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
```

#### 基於 Token 預算的生成

```python
def budget_aware_generation(
    prompt,
    model,
    tokenizer,
    max_tokens,
    stop_sequences=["\n\n", "###"]
):
    """
    在 token 預算內生成，遇到停止序列時提前結束
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class TokenBudgetStoppingCriteria(StoppingCriteria):
        def __init__(self, max_tokens, stop_token_ids):
            self.max_tokens = max_tokens
            self.stop_token_ids = stop_token_ids

        def __call__(self, input_ids, scores, **kwargs):
            # 檢查是否超過預算
            if input_ids.shape[1] >= self.max_tokens:
                return True

            # 檢查是否遇到停止序列
            for stop_id in self.stop_token_ids:
                if input_ids[0, -1] == stop_id:
                    return True

            return False

    # 將停止序列轉換為 token IDs
    stop_token_ids = []
    for seq in stop_sequences:
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        stop_token_ids.extend(tokens)

    stopping_criteria = StoppingCriteriaList([
        TokenBudgetStoppingCriteria(max_tokens, stop_token_ids)
    ])

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_length=max_tokens,
        stopping_criteria=stopping_criteria,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
```

### 5.6 實戰案例：生成有效的 Python 程式碼

```python
import ast

def generate_valid_python(
    prompt,
    model,
    tokenizer,
    max_attempts=5
):
    """
    生成語法正確的 Python 程式碼
    """
    for attempt in range(max_attempts):
        # 使用低溫度和適當的 prompt 工程
        full_prompt = f"""{prompt}

```python
"""

        inputs = tokenizer(full_prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_length=512,
            temperature=0.2,  # 低溫度提高準確性
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取程式碼
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
        else:
            code = text.split(full_prompt)[1].strip() if full_prompt in text else text

        # 驗證語法
        try:
            ast.parse(code)
            print(f"成功生成有效的 Python 程式碼 (嘗試 {attempt + 1})")
            return code
        except SyntaxError as e:
            print(f"嘗試 {attempt + 1} 失敗: {e}")
            if attempt < max_attempts - 1:
                # 添加錯誤訊息到 prompt 進行修正
                prompt = f"{prompt}\n\n# Previous attempt had error: {e}\n# Please fix and generate valid Python code:"

    return None

# 使用範例
code = generate_valid_python(
    "Write a function to calculate fibonacci numbers",
    model,
    tokenizer
)

if code:
    print("Generated code:")
    print(code)
    exec(code)  # 執行生成的程式碼
```

---

## 6. 策略比較與選擇指南

### 6.1 策略對比表

| 策略 | 多樣性 | 質量穩定性 | 計算成本 | 重複風險 | 適用場景 |
|------|--------|----------|---------|---------|---------|
| Greedy | 無 | 高 | 極低 | 高 | 事實問答、翻譯 |
| Beam Search | 低 | 高 | 高 | 中 | 翻譯、摘要 |
| Temperature | 可調 | 中 | 低 | 中 | 通用對話 |
| Top-k | 中 | 中-高 | 低 | 低 | 一般生成 |
| Top-p | 中-高 | 中 | 低 | 低 | 創意寫作、對話 |
| Contrastive | 高 | 中 | 中 | 極低 | 長文本生成 |

### 6.2 任務導向選擇指南

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

### 6.3 組合策略

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

## 7. 評估指標與性能比較

### 7.1 自動評估指標

#### 文本質量指標

**1. Perplexity (困惑度)**
```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, tokenizer, text):
    """
    計算生成文本的困惑度
    困惑度越低，模型對文本的"驚訝程度"越低，表示文本越符合預期
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    perplexity = torch.exp(loss)
    return perplexity.item()

# 使用範例
text = "The quick brown fox jumps over the lazy dog"
ppl = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity: {ppl:.2f}")
```

**2. BLEU (Bilingual Evaluation Understudy)**
- 主要用於機器翻譯
- 比較生成文本與參考文本的 n-gram 重疊

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# 單句 BLEU
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'test']
score = sentence_bleu(reference, candidate)
print(f"BLEU score: {score:.4f}")

# 使用 SacreBLEU (更標準的實現)
from sacrebleu import corpus_bleu

references = [['The cat is on the mat', 'There is a cat on the mat']]
candidates = ['The cat is on the mat']
bleu = corpus_bleu(candidates, references)
print(f"BLEU: {bleu.score:.2f}")
```

**3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- 主要用於摘要任務
- 測量 n-gram 重疊的召回率

```python
from rouge import Rouge

rouge = Rouge()

hypothesis = "the cat is on the mat"
reference = "the cat sat on the mat"

scores = rouge.get_scores(hypothesis, reference)
print(f"ROUGE-1: {scores[0]['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {scores[0]['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {scores[0]['rouge-l']['f']:.4f}")
```

**4. BERTScore**
- 使用 BERT embeddings 計算語義相似度
- 比 BLEU/ROUGE 更能捕捉語義

```python
from bert_score import score

candidates = ["The cat sits on the mat"]
references = ["A cat is on the mat"]

P, R, F1 = score(candidates, references, lang="en", verbose=True)
print(f"BERTScore F1: {F1.mean():.4f}")
```

#### 多樣性指標

**1. Distinct-n**
- 測量 n-gram 的多樣性
- distinct-n = (unique n-grams) / (total n-grams)

```python
def calculate_distinct_n(text, n=2):
    """
    計算 distinct-n 分數
    """
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    if len(ngrams) == 0:
        return 0.0

    unique_ngrams = set(ngrams)
    distinct_score = len(unique_ngrams) / len(ngrams)

    return distinct_score

text = "the cat sat on the mat and the dog sat on the rug"
print(f"Distinct-1: {calculate_distinct_n(text, 1):.4f}")
print(f"Distinct-2: {calculate_distinct_n(text, 2):.4f}")
```

**2. Self-BLEU**
- 生成多個樣本，計算它們之間的 BLEU
- 分數越低，多樣性越高

```python
from sacrebleu import sentence_bleu

def calculate_self_bleu(generated_texts):
    """
    計算 Self-BLEU 分數
    """
    scores = []
    for i, candidate in enumerate(generated_texts):
        references = [text for j, text in enumerate(generated_texts) if j != i]
        score = sentence_bleu(candidate, references).score
        scores.append(score)

    return sum(scores) / len(scores)
```

#### 連貫性指標

```python
def calculate_repetition_rate(text, n=3):
    """
    計算 n-gram 重複率
    """
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    if len(ngrams) == 0:
        return 0.0

    unique_ngrams = set(ngrams)
    repetition_rate = 1 - (len(unique_ngrams) / len(ngrams))

    return repetition_rate
```

### 7.2 人工評估維度

#### 評估框架

```python
from dataclasses import dataclass
from typing import List

@dataclass
class HumanEvaluation:
    """人工評估結構"""
    fluency: int  # 1-5: 流暢度
    coherence: int  # 1-5: 連貫性
    relevance: int  # 1-5: 相關性
    informativeness: int  # 1-5: 資訊量
    overall: int  # 1-5: 總體質量
    comments: str  # 評論

def aggregate_human_eval(evaluations: List[HumanEvaluation]):
    """
    聚合人工評估結果
    """
    n = len(evaluations)
    return {
        'fluency': sum(e.fluency for e in evaluations) / n,
        'coherence': sum(e.coherence for e in evaluations) / n,
        'relevance': sum(e.relevance for e in evaluations) / n,
        'informativeness': sum(e.informativeness for e in evaluations) / n,
        'overall': sum(e.overall for e in evaluations) / n,
    }
```

### 7.3 性能基準測試

```python
import time
from typing import Dict, Any

def benchmark_decoding_strategy(
    model,
    tokenizer,
    prompts: List[str],
    generation_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    基準測試解碼策略
    """
    results = {
        'total_time': 0,
        'avg_time_per_sample': 0,
        'tokens_per_second': 0,
        'avg_length': 0,
        'avg_perplexity': 0
    }

    total_tokens = 0
    start_time = time.time()

    generated_texts = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        step_start = time.time()
        output = model.generate(**inputs, **generation_config)
        step_time = time.time() - step_start

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append(text)

        total_tokens += len(output[0])

    end_time = time.time()
    total_time = end_time - start_time

    results['total_time'] = total_time
    results['avg_time_per_sample'] = total_time / len(prompts)
    results['tokens_per_second'] = total_tokens / total_time
    results['avg_length'] = total_tokens / len(prompts)

    # 計算平均困惑度
    perplexities = [
        calculate_perplexity(model, tokenizer, text)
        for text in generated_texts
    ]
    results['avg_perplexity'] = sum(perplexities) / len(perplexities)

    return results, generated_texts

# 使用範例
prompts = ["Once upon a time"] * 10

configs = {
    'greedy': {'do_sample': False, 'max_length': 50},
    'top_p': {'do_sample': True, 'top_p': 0.9, 'temperature': 0.8, 'max_length': 50},
    'beam': {'num_beams': 5, 'max_length': 50}
}

for name, config in configs.items():
    print(f"\n測試策略: {name}")
    results, texts = benchmark_decoding_strategy(model, tokenizer, prompts, config)
    print(f"  平均時間: {results['avg_time_per_sample']:.3f}s")
    print(f"  Tokens/秒: {results['tokens_per_second']:.1f}")
    print(f"  平均困惑度: {results['avg_perplexity']:.2f}")
```

---

## 8. 實戰問題診斷與解決方案

### 8.1 常見問題與解決方案

#### 問題 1：生成重複內容

**症狀**：
- 模型反覆生成相同的詞或短語
- 出現"the the the"或"I think, I think, I think"

**原因**：
- Temperature 太低（接近貪婪）
- Beam search 的固有問題
- 訓練數據中有重複模式

**解決方案**：

```python
# 方案 1：使用 repetition_penalty
output = model.generate(
    **inputs,
    max_length=100,
    repetition_penalty=1.2,  # > 1.0 懲罰重複
    do_sample=True,
    temperature=0.8
)

# 方案 2：禁止重複 n-gram
output = model.generate(
    **inputs,
    max_length=100,
    no_repeat_ngram_size=3,  # 禁止重複 3-gram
    num_beams=5
)

# 方案 3：使用 Contrastive Search
output = model.generate(
    **inputs,
    max_length=100,
    penalty_alpha=0.6,
    top_k=4
)

# 方案 4：後處理去重
def remove_repetitions(text, max_repeat=2):
    """移除連續重複的詞"""
    words = text.split()
    result = []
    count = 1

    for i, word in enumerate(words):
        if i > 0 and word == words[i-1]:
            count += 1
            if count <= max_repeat:
                result.append(word)
        else:
            count = 1
            result.append(word)

    return ' '.join(result)
```

#### 問題 2：生成不連貫或無意義的文本

**症狀**：
- 文本語法正確但語義不連貫
- 主題頻繁跳轉
- 包含不相關的內容

**原因**：
- Temperature 太高
- Top-p/Top-k 太寬鬆
- 上下文不足

**解決方案**：

```python
# 方案 1：降低溫度和 top_p
output = model.generate(
    **inputs,
    max_length=100,
    temperature=0.5,  # 降低到 0.5-0.7
    top_p=0.85,  # 降低到 0.85-0.9
    do_sample=True
)

# 方案 2：增加上下文
# 確保 prompt 包含足夠的上下文信息
better_prompt = f"""Context: {context}
Task: {task}
Requirements: {requirements}

Response:"""

# 方案 3：使用 Beam Search
output = model.generate(
    **inputs,
    max_length=100,
    num_beams=5,
    early_stopping=True
)
```

#### 問題 3：生成速度太慢

**症狀**：
- 推理延遲高
- 無法滿足實時要求

**原因**：
- 使用 Beam Search（計算成本高）
- 模型太大
- Batch size 太小
- 沒有使用加速技術

**解決方案**：

```python
# 方案 1：使用更快的採樣策略
# 避免 Beam Search，使用 Greedy 或 Sampling
output = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 方案 2：減少 max_length
output = model.generate(
    **inputs,
    max_new_tokens=50,  # 限制新生成的 tokens
    do_sample=True
)

# 方案 3：使用 KV Cache（默認開啟）
# 確保 use_cache=True
output = model.generate(
    **inputs,
    max_length=100,
    use_cache=True,  # 默認 True
    do_sample=True
)

# 方案 4：批次處理
# 一次處理多個 prompts
batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(
    **batch_inputs,
    max_length=100,
    do_sample=True
)

# 方案 5：使用 Speculative Decoding
# 見 4.4 Speculative Decoding 章節

# 方案 6：模型量化
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

#### 問題 4：輸出不符合指定格式

**症狀**：
- 要求 JSON 但生成純文本
- 格式不完整或不正確
- 包含額外的解釋文字

**原因**：
- Prompt 不夠明確
- 沒有使用約束解碼
- Temperature 太高導致發散

**解決方案**：

```python
# 方案 1：改進 Prompt
prompt = """Generate a JSON object with the following fields: name, age, city.
Output ONLY the JSON, no explanations.

{"""

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_length=100,
    temperature=0.2,  # 低溫度
    do_sample=True
)

# 在結果前加上 "{"
result = "{" + tokenizer.decode(output[0], skip_special_tokens=True)

# 方案 2：使用約束解碼（見第 5 節）
# 使用 Outlines、Guidance 等庫

# 方案 3：後處理提取
import json
import re

def extract_json(text):
    """從文本中提取 JSON"""
    # 嘗試提取花括號或方括號之間的內容
    json_pattern = r'\{[^{}]*\}|\[[^\[\]]*\]'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None

# 方案 4：重試機制
def generate_with_retry(prompt, model, tokenizer, validator, max_retries=3):
    """帶驗證的生成"""
    for i in range(max_retries):
        output = model.generate(
            **tokenizer(prompt, return_tensors="pt"),
            max_length=200,
            temperature=0.2 + i * 0.1,  # 逐步增加溫度
            do_sample=True
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        if validator(text):
            return text

        print(f"嘗試 {i+1} 失敗，重試...")

    return None
```

### 8.2 調試技巧

#### 1. 可視化 Token 概率

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_token_probabilities(model, tokenizer, prompt, top_k=10):
    """
    可視化下一個 token 的概率分佈
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=0)

    # 獲取 top-k 概率
    top_probs, top_indices = torch.topk(probs, top_k)

    # 獲取對應的 tokens
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), top_probs.cpu().numpy())
    plt.yticks(range(top_k), top_tokens)
    plt.xlabel('Probability')
    plt.title(f'Top-{top_k} Token Probabilities')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 使用
visualize_token_probabilities(model, tokenizer, "The capital of France is")
```

#### 2. 逐步生成追蹤

```python
def generate_with_trace(model, tokenizer, prompt, max_length=20):
    """
    逐步生成並追蹤每一步的選擇
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"].clone()

    print(f"Prompt: {prompt}")
    print("\n生成過程:")

    for step in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

        # 獲取 top-3 候選
        top_probs, top_indices = torch.topk(probs[0], 3)

        print(f"\nStep {step + 1}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (p={prob:.4f})")

        # 選擇最高概率的 token（貪婪）
        next_token = top_indices[0].unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        chosen_token = tokenizer.decode([next_token[0, 0]])
        print(f"  選擇: '{chosen_token}'")

        if next_token[0, 0] == tokenizer.eos_token_id:
            break

    final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\n最終輸出: {final_text}")
    return final_text
```

---

## 9. 生產環境考量

### 9.1 延遲 vs 質量權衡

#### 延遲預算分配

```python
class LatencyBudgetManager:
    """延遲預算管理器"""

    def __init__(self, total_budget_ms=1000):
        self.total_budget = total_budget_ms
        self.used = 0

    def select_strategy(self, remaining_budget_ms):
        """根據剩餘預算選擇策略"""
        if remaining_budget_ms > 500:
            # 充足時間，使用高質量策略
            return {
                'num_beams': 5,
                'do_sample': False,
                'max_length': 100
            }
        elif remaining_budget_ms > 200:
            # 中等時間，使用平衡策略
            return {
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'max_length': 100
            }
        else:
            # 時間緊迫，使用快速策略
            return {
                'do_sample': False,  # Greedy
                'max_length': 50  # 減少長度
            }
```

### 9.2 批次處理策略

```python
from typing import List
import asyncio

class BatchGenerator:
    """批次生成器"""

    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.queue = []

    async def add_request(self, prompt, generation_config):
        """添加生成請求"""
        request = {
            'prompt': prompt,
            'config': generation_config,
            'future': asyncio.Future()
        }
        self.queue.append(request)

        # 如果達到 batch size，執行生成
        if len(self.queue) >= self.batch_size:
            await self.process_batch()

        return await request['future']

    async def process_batch(self):
        """處理一個批次"""
        if not self.queue:
            return

        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]

        # 準備批次輸入
        prompts = [req['prompt'] for req in batch]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 生成（使用第一個請求的配置）
        outputs = self.model.generate(
            **inputs,
            **batch[0]['config']
        )

        # 分配結果
        for i, req in enumerate(batch):
            text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            req['future'].set_result(text)
```

### 9.3 快取策略

```python
from functools import lru_cache
import hashlib

class GenerationCache:
    """生成結果快取"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt, config):
        """生成快取鍵"""
        config_str = str(sorted(config.items()))
        content = f"{prompt}:{config_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt, config):
        """獲取快取結果"""
        key = self._make_key(prompt, config)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, prompt, config, result):
        """設置快取"""
        if len(self.cache) >= self.max_size:
            # 簡單的 FIFO 淘汰策略
            self.cache.pop(next(iter(self.cache)))

        key = self._make_key(prompt, config)
        self.cache[key] = result

    def stats(self):
        """快取統計"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }
```

### 9.4 監控與日誌

```python
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class GenerationMetrics:
    """生成指標"""
    timestamp: str
    prompt_length: int
    output_length: int
    latency_ms: float
    strategy: str
    temperature: float
    success: bool
    error: str = None

class GenerationMonitor:
    """生成監控器"""

    def __init__(self, log_file="generation_metrics.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger("generation")

        # 設置日誌
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_generation(self, metrics: GenerationMetrics):
        """記錄生成指標"""
        self.logger.info(json.dumps(asdict(metrics)))

    def analyze_logs(self):
        """分析日誌"""
        metrics_list = []

        with open(self.log_file, 'r') as f:
            for line in f:
                metrics_list.append(json.loads(line))

        # 計算統計
        total = len(metrics_list)
        successful = sum(1 for m in metrics_list if m['success'])
        avg_latency = sum(m['latency_ms'] for m in metrics_list) / total

        return {
            'total_requests': total,
            'success_rate': successful / total,
            'avg_latency_ms': avg_latency,
            'avg_output_length': sum(m['output_length'] for m in metrics_list) / total
        }

# 使用範例
monitor = GenerationMonitor()

start = time.time()
try:
    output = model.generate(**inputs, do_sample=True, temperature=0.7)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    latency = (time.time() - start) * 1000

    metrics = GenerationMetrics(
        timestamp=datetime.now().isoformat(),
        prompt_length=len(inputs["input_ids"][0]),
        output_length=len(output[0]),
        latency_ms=latency,
        strategy="sampling",
        temperature=0.7,
        success=True
    )
except Exception as e:
    metrics = GenerationMetrics(
        timestamp=datetime.now().isoformat(),
        prompt_length=len(inputs["input_ids"][0]),
        output_length=0,
        latency_ms=(time.time() - start) * 1000,
        strategy="sampling",
        temperature=0.7,
        success=False,
        error=str(e)
    )

monitor.log_generation(metrics)
```

---

## 10. Python 實作範例

### 10.1 基礎解碼策略實現

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

## 11. 參數調優指南

### 11.1 Temperature 調優

| 溫度範圍 | 效果 | 適用場景 |
|---------|------|---------|
| 0.1 - 0.3 | 極確定，接近貪婪 | 程式碼生成、數學問題、需要精確答案 |
| 0.4 - 0.6 | 較確定，稍有變化 | 技術寫作、專業文檔 |
| 0.7 - 0.9 | 平衡質量與創意 | 一般對話、文章寫作 |
| 1.0 - 1.3 | 高創意，較隨機 | 創意寫作、腦力激盪 |
| > 1.5 | 極隨機，可能不連貫 | 實驗性生成 |

### 11.2 Top-p 調優

| p 值 | 效果 | 適用場景 |
|------|------|---------|
| 0.5 - 0.7 | 保守，只選高機率詞 | 事實性任務 |
| 0.8 - 0.9 | 平衡，最常用 | 一般對話、寫作 |
| 0.95 - 0.99 | 包容，允許更多變化 | 創意任務 |
| 1.0 | 無過濾 | 等同於基礎採樣 |

### 11.3 Repetition Penalty 調優

```python
# 避免重複的參數設定
repetition_penalty = 1.2  # 1.0 = 無懲罰，> 1.0 懲罰重複
no_repeat_ngram_size = 3  # 禁止重複的 n-gram 長度
```

**經驗法則**：
- 短文本（<100 tokens）：`repetition_penalty = 1.1-1.2`
- 長文本（>100 tokens）：`repetition_penalty = 1.2-1.5`
- 對話：`repetition_penalty = 1.2`, `no_repeat_ngram_size = 2-3`

### 11.4 組合策略建議

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

## 12. 延伸閱讀

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

本文全面介紹了 LLM 文字生成與解碼策略，從基礎到進階，從理論到實踐：

### 核心要點

1. **基礎策略**
   - **貪婪解碼**：簡單快速，適合事實性任務
   - **Beam Search**：質量更高，適合翻譯和摘要
   - **Temperature/Top-k/Top-p**：平衡質量與多樣性的採樣方法

2. **進階技術**
   - **Speculative Decoding**：無損加速 2-3 倍
   - **Self-Consistency**：提高推理任務準確性
   - **Contrastive Search**：減少重複，提升流暢度
   - **Diverse Beam Search**：生成多樣化候選

3. **結構化生成**
   - **約束解碼**：確保輸出符合特定格式（JSON、XML、程式碼）
   - **專門函式庫**：Outlines、Guidance、LMQL
   - **長度控制**：精確控制輸出長度

4. **評估與優化**
   - **自動指標**：Perplexity、BLEU、ROUGE、BERTScore
   - **多樣性指標**：Distinct-n、Self-BLEU
   - **人工評估**：流暢度、連貫性、相關性

5. **實戰經驗**
   - **常見問題**：重複、不連貫、速度慢、格式錯誤
   - **解決方案**：參數調整、策略組合、後處理
   - **調試技巧**：可視化概率分佈、逐步追蹤生成

6. **生產部署**
   - **延遲優化**：根據預算選擇策略
   - **批次處理**：提高吞吐量
   - **快取策略**：減少重複計算
   - **監控日誌**：追蹤性能指標

### 選擇指南

**事實性任務**（翻譯、問答）：
```python
temperature=0.3, top_p=0.85, num_beams=5
```

**平衡任務**（對話、寫作）：
```python
temperature=0.8, top_p=0.9, repetition_penalty=1.2
```

**創意任務**（故事、腦力激盪）：
```python
temperature=1.1, top_p=0.95, top_k=100
```

**結構化輸出**（JSON、程式碼）：
```python
temperature=0.2, 使用約束解碼或專門函式庫
```

**速度優先**（即時應用）：
```python
do_sample=False (Greedy), max_new_tokens=50, 或使用 Speculative Decoding
```

### 最佳實踐

1. **從簡單開始**：先用 Greedy 或基礎採樣建立 baseline
2. **理解任務特性**：事實性優先還是創意性優先？
3. **善用組合策略**：Temperature + Top-p + Repetition Penalty 通常效果最好
4. **持續實驗**：沒有萬能配置，需要根據具體任務和數據調整
5. **監控評估**：結合自動指標和人工評估
6. **考慮效率**：在質量和速度之間找到平衡點
7. **使用約束**：需要特定格式時，使用約束解碼而非依賴 prompt
8. **生產就緒**：實現快取、批次處理、監控和錯誤處理

### 推薦起點

**通用配置**（適合大多數應用）：
```python
generation_config = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "max_new_tokens": 512
}
```

從這個配置開始，然後根據實際效果進行調整。記住：**沒有完美的配置，只有最適合你任務的配置**。

### 延伸學習

- 實踐各種策略，觀察差異
- 閱讀相關論文，理解理論基礎
- 參與開源項目，學習最佳實踐
- 關注最新研究，掌握前沿技術

文字生成是 LLM 應用的核心，掌握解碼策略將大大提升應用質量！
