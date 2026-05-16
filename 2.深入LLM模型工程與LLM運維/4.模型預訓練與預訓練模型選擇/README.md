# 模型預訓練與預訓練模型選擇

## 目錄
1. [預訓練的基礎概念](#41-預訓練的基礎概念)
2. [開源預訓練模型](#42-開源預訓練模型)
3. [預訓練流程](#43-預訓練流程)
4. [Scaling Laws 與高效預訓練技術](#44-scaling-laws-與高效預訓練技術)
5. [實作範例](#45-實作範例)

---

## 4.1 預訓練的基礎概念

### 4.1.1 什麼是預訓練？

**預訓練 (Pre-training)** 是在大規模無標註文字資料上訓練語言模型的過程，目的是讓模型學習語言的統計規律、語法結構和世界知識。

**預訓練的重要性**：
- 學習通用語言表示
- 捕捉語法和語義知識
- 建立世界知識基礎
- 為下游任務提供良好的初始化

**預訓練 vs 微調**：

| 階段 | 資料 | 目標 | 規模 | 成本 |
|------|------|------|------|------|
| 預訓練 | 海量無標註文字 | 語言建模 | 數百億到數兆 tokens | 極高 |
| 微調 | 少量標註資料 | 特定任務 | 數千到數百萬 tokens | 中等 |

### 4.1.2 語言建模目標

#### Causal Language Modeling (因果語言建模)

用於 GPT 等自回歸模型：

```
給定前文 x₁, x₂, ..., xₜ，預測下一個 token xₜ₊₁

損失函式：
L = -∑ log P(xₜ | x₁, ..., xₜ₋₁)
```

**特點**：
- 單向注意力（只看前文）
- 適合文字生成任務
- 訓練與推理一致

**範例**：
```
輸入: "機器學習是"
目標: "人工智慧"
```

#### Masked Language Modeling (遮蔽語言建模)

用於 BERT 等雙向模型：

```
隨機遮蔽部分 token，預測被遮蔽的 token

輸入: "機器[MASK]是人工智慧的[MASK]分支"
目標: 預測 [MASK] 位置的詞（"學習"、"一個"）
```

**特點**：
- 雙向注意力（可看前後文）
- 適合理解任務（分類、問答）
- 訓練與推理有差異（MLM 預訓練，但推理時沒有 [MASK]）

### 4.1.3 預訓練資料

**資料規模趨勢**：

| 模型 | 發布年份 | 參數量 | 訓練 Tokens |
|------|---------|--------|-------------|
| GPT-2 | 2019 | 1.5B | 40B |
| GPT-3 | 2020 | 175B | 300B |
| GPT-4 | 2023 | ~1.8T (推測) | ~13T (推測) |
| LLaMA | 2023 | 7B-65B | 1T-1.4T |
| LLaMA 2 | 2023 | 7B-70B | 2T |
| DeepSeek-V3 | 2024 | 671B | 14.8T |

**資料來源**：
1. **網路爬取資料**
   - Common Crawl
   - Reddit
   - StackOverflow
   - Wikipedia

2. **書籍與學術資料**
   - Books3
   - arXiv
   - PubMed

3. **程式碼**
   - GitHub
   - GitLab
   - StackExchange

4. **對話資料**
   - 論壇討論
   - 社交媒體

**資料品質考量**：
- 去重（deduplication）
- 過濾低品質內容
- 移除個人隱私資訊
- 平衡不同領域比例

---

## 4.2 開源預訓練模型

### 4.2.1 LLaMA 系列

**LLaMA (Large Language Model Meta AI)**

**LLaMA 1 (2023.02)**：
- **規模**：7B, 13B, 33B, 65B
- **訓練資料**：1T-1.4T tokens
- **特點**：
  - 開源權重（研究用途）
  - 高效架構設計
  - SwiGLU 啟用函數
  - RoPE 位置編碼

**LLaMA 2 (2023.07)**：
- **規模**：7B, 13B, 70B
- **訓練資料**：2T tokens（較 LLaMA 1 增加 40%）
- **改進**：
  - 上下文長度從 2K 增加到 4K
  - Grouped-Query Attention (GQA)
  - 更好的安全性對齊
  - 商業友好授權

**LLaMA 3 (2024)**：
- **規模**：8B, 70B, 405B
- **訓練資料**：15T+ tokens
- **改進**：
  - 上下文長度 8K（可擴展到 128K）
  - 更大的詞彙表（128K）
  - 多語言能力顯著提升

**使用範例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入 LLaMA 2 模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 生成文字
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 4.2.2 Mistral 系列

**Mistral 7B (2023.09)**：
- **參數量**：7.3B
- **特點**：
  - Sliding Window Attention（滑動窗口注意力）
  - GQA (Grouped-Query Attention)
  - 性能媲美 13B-34B 模型
  - 上下文長度：8K（可擴展到 32K）

**Mixtral 8x7B (2023.12)**：
- **架構**：Mixture of Experts (MoE)
- **總參數**：47B
- **啟用參數**：12.9B（每次只啟用 2 個專家）
- **特點**：
  - 8 個專家模型
  - 稀疏激活提升效率
  - 多語言能力強

**Mixtral 8x22B (2024.04)**：
- **總參數**：141B
- **啟用參數**：39B
- **上下文長度**：64K

**使用範例**：

```python
# 載入 Mistral 模型
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Mistral 支援更長的上下文
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

### 4.2.3 BLOOM

**BLOOM (BigScience Large Open-science Open-access Multilingual Language Model)**

- **發布**：2022.07
- **規模**：176B 參數
- **訓練資料**：366B tokens，46 種語言
- **特點**：
  - 真正的多語言模型
  - 社群協作開發
  - 完全開源
  - ALiBi 位置編碼

**語言支援**：
- 英語、中文、法語、西班牙語等 46 種語言
- 包含程式碼（13 種程式語言）

### 4.2.4 Qwen (通義千問)

**Qwen 1.0 (2023)**：
- **規模**：1.8B, 7B, 14B, 72B
- **訓練資料**：~3T tokens
- **特點**：
  - 中英雙語優化
  - 長上下文支援（8K-32K）
  - 程式碼能力強

**Qwen 2 (2024)**：
- **規模**：0.5B, 1.5B, 7B, 72B
- **改進**：
  - 更大的詞彙表
  - 更好的多語言能力
  - 支援 128K 上下文

### 4.2.5 DeepSeek 系列

**DeepSeek-V2 (2024.05)**：
- **規模**：236B 總參數，21B 啟用參數
- **架構**：MoE（混合專家）
- **特點**：
  - 極低的訓練成本
  - Multi-head Latent Attention (MLA)
  - 128K 上下文窗口

**DeepSeek-V3 (2024.12)**：
- **規模**：671B 總參數，37B 啟用參數
- **訓練資料**：14.8T tokens
- **訓練成本**：僅 $5.576M
- **特點**：
  - 無輔助損失的 MoE 訓練
  - Multi-Token Prediction (MTP)
  - FP8 混合精度訓練
  - 開源權重

### 4.2.6 Gemma 系列

**Gemma (Google)**

**Gemma 1 (2024.02)**：
- **規模**：2B, 7B
- **特點**：
  - 基於 Gemini 技術
  - 完全開源且商業友好
  - 優秀的安全性對齊
  - Multi-Query Attention (MQA)
  - RoPE 位置編碼

**Gemma 2 (2024.06)**：
- **規模**：9B, 27B
- **改進**：
  - 滑動窗口注意力
  - 局部-全局注意力混合
  - Logit soft-capping
  - 更好的知識蒸餾

**使用範例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入 Gemma 模型
model_name = "google/gemma-2-9b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 生成
prompt = "Explain machine learning in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### 4.2.7 Phi 系列

**Phi (Microsoft)**

**Phi-1 (2023.06)**：
- **規模**：1.3B
- **訓練資料**：7B tokens（高品質篩選）
- **特點**：小模型，專注程式碼和數學

**Phi-2 (2023.12)**：
- **規模**：2.7B
- **訓練資料**：1.4T tokens
- **特點**：
  - 性能媲美 7B-13B 模型
  - 資料品質優於數量
  - 合成資料增強

**Phi-3 (2024.04)**：
- **規模**：3.8B (mini), 7B (small), 14B (medium)
- **上下文長度**：4K-128K
- **特點**：
  - 長上下文支援
  - 多語言能力
  - 安全性對齊

**使用範例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入 Phi-3 模型
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Phi-3 使用特定的對話格式
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 4.2.8 其他重要模型

#### Falcon

**Falcon-180B (2023.09)**：
- **規模**：180B
- **訓練資料**：3.5T tokens
- **特點**：
  - 高品質的 RefinedWeb 資料集
  - Multi-Query Attention
  - Apache 2.0 授權

#### Yi 系列

**Yi (01.AI)**：
- **規模**：6B, 9B, 34B
- **訓練資料**：3.1T tokens
- **特點**：
  - 雙語能力（中英）
  - 200K 上下文（Yi-34B-200K）
  - 商業友好授權

#### Baichuan 系列

**Baichuan 2 (2023.09)**：
- **規模**：7B, 13B
- **特點**：
  - 中文優化
  - 2.6T tokens 訓練資料
  - 商業授權

### 4.2.9 模型選擇指南

**根據規模選擇**：

| 規模 | 適用場景 | 推薦模型 | 硬體需求 |
|------|---------|---------|---------|
| 超小型 (< 3B) | 邊緣設備、移動端 | Phi-3 mini, Gemma 2B | 單卡 GPU (8GB+) |
| 小型 (3B-10B) | 快速推理、資源受限 | Mistral 7B, Gemma 9B, Qwen 7B | 單卡 GPU (16GB+) |
| 中型 (10B-70B) | 一般應用、微調 | LLaMA 2 70B, Qwen 72B | 多卡 GPU (80GB+) |
| 大型 (> 70B) | 複雜推理、研究 | LLaMA 3 405B, DeepSeek-V3 | 多機多卡 |

**根據任務選擇**：

| 任務類型 | 推薦模型 | 原因 |
|---------|---------|------|
| 中文對話 | Qwen, DeepSeek | 中文優化 |
| 程式碼生成 | DeepSeek, LLaMA 3 | 程式碼訓練資料豐富 |
| 多語言 | BLOOM, LLaMA 3 | 多語言支援 |
| 長上下文 | Qwen 2, Mistral | 支援 32K-128K 上下文 |
| 成本敏感 | DeepSeek-V3 (MoE) | 稀疏激活降低成本 |

---

## 4.3 預訓練流程

### 4.3.1 資料處理流程

#### 步驟 1：資料收集

```python
from datasets import load_dataset
import os

class PretrainingDataCollector:
    """預訓練資料收集器"""

    def __init__(self, output_dir="./pretraining_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def collect_web_data(self):
        """收集網路資料（使用 Common Crawl）"""
        # Common Crawl 資料非常大，這裡僅示範
        dataset = load_dataset(
            "c4",
            "en",
            split="train",
            streaming=True  # 串流模式避免記憶體不足
        )
        return dataset

    def collect_wikipedia(self, language="en"):
        """收集 Wikipedia 資料"""
        dataset = load_dataset(
            "wikipedia",
            f"20231101.{language}",
            split="train"
        )
        return dataset

    def collect_code_data(self):
        """收集程式碼資料"""
        dataset = load_dataset(
            "codeparrot/github-code",
            split="train",
            streaming=True
        )
        return dataset

    def collect_books(self):
        """收集書籍資料"""
        # 注意：某些資料集可能有版權問題
        dataset = load_dataset(
            "bookcorpus",
            split="train"
        )
        return dataset

# 使用範例
collector = PretrainingDataCollector()

# 收集不同來源的資料
wiki_data = collector.collect_wikipedia(language="zh")
print(f"Wikipedia 資料量: {len(wiki_data)}")
```

#### 步驟 1.5：資料混合策略

**為什麼需要資料混合？**

預訓練資料的配方（data mixture）對模型性能至關重要。不同類型資料的比例會直接影響模型的能力平衡。

**常見資料配方**：

```python
from typing import Dict, List
from datasets import Dataset, concatenate_datasets
import numpy as np

class DataMixtureManager:
    """資料混合管理器"""

    def __init__(self):
        self.datasets = {}
        self.mixture_weights = {}

    def add_dataset(self, name: str, dataset: Dataset, weight: float = 1.0):
        """
        添加資料集

        Args:
            name: 資料集名稱
            dataset: 資料集
            weight: 混合權重
        """
        self.datasets[name] = dataset
        self.mixture_weights[name] = weight

    def create_mixture(self, total_samples: int = None) -> Dataset:
        """
        建立混合資料集

        Args:
            total_samples: 總樣本數（None 則使用所有資料）

        Returns:
            混合後的資料集
        """
        # 計算每個資料集應該取樣的數量
        total_weight = sum(self.mixture_weights.values())
        sample_counts = {}

        for name, weight in self.mixture_weights.items():
            if total_samples:
                count = int(total_samples * (weight / total_weight))
            else:
                count = len(self.datasets[name])

            sample_counts[name] = min(count, len(self.datasets[name]))

        print("資料混合配方:")
        print("=" * 60)
        for name, count in sample_counts.items():
            percentage = (count / sum(sample_counts.values())) * 100
            print(f"{name}: {count:,} samples ({percentage:.1f}%)")

        # 取樣並混合
        sampled_datasets = []
        for name, count in sample_counts.items():
            dataset = self.datasets[name]

            if count < len(dataset):
                # 隨機取樣
                indices = np.random.choice(len(dataset), count, replace=False)
                sampled = dataset.select(indices)
            else:
                sampled = dataset

            sampled_datasets.append(sampled)

        # 合併資料集
        mixed_dataset = concatenate_datasets(sampled_datasets)

        # 打亂
        mixed_dataset = mixed_dataset.shuffle(seed=42)

        print(f"\n總計: {len(mixed_dataset):,} samples")
        return mixed_dataset


# 使用範例：建立預訓練資料配方
def create_pretraining_mixture():
    """建立預訓練資料混合"""
    from datasets import load_dataset

    manager = DataMixtureManager()

    # 1. 網路爬取資料（占比最高）
    # web_data = load_dataset("c4", "en", split="train", streaming=True)
    # manager.add_dataset("web", web_data, weight=0.5)

    # 2. 書籍（高品質內容）
    # books = load_dataset("bookcorpus", split="train")
    # manager.add_dataset("books", books, weight=0.15)

    # 3. Wikipedia（結構化知識）
    wiki = load_dataset("wikipedia", "20231101.en", split="train[:10000]")  # 示範用
    manager.add_dataset("wikipedia", wiki, weight=0.15)

    # 4. 學術論文
    # papers = load_dataset("arxiv", split="train")
    # manager.add_dataset("papers", papers, weight=0.1)

    # 5. 程式碼
    # code = load_dataset("codeparrot/github-code", split="train", streaming=True)
    # manager.add_dataset("code", code, weight=0.1)

    # 建立混合資料集
    mixed_dataset = manager.create_mixture(total_samples=50000)

    return mixed_dataset

# mixed_data = create_pretraining_mixture()
```

**不同模型的資料配方參考**：

| 資料類型 | GPT-3 | LLaMA | LLaMA 2 | DeepSeek-V3 |
|---------|-------|-------|---------|-------------|
| 網路爬取 | 60% | 67% | 70% | 52% |
| 書籍 | 16% | 4.5% | 8% | 5% |
| Wikipedia | 3% | 4.5% | 4.5% | 3% |
| 學術論文 | 12% | 2.5% | 2.5% | 15% |
| 程式碼 | 9% | 4.5% | 8% | 25% |
| 其他 | - | 17% | 7% | - |

**領域特定的資料配方**：

```python
# 程式碼優化模型的配方
CODE_HEAVY_MIXTURE = {
    "code": 0.60,           # 程式碼占 60%
    "technical_docs": 0.20, # 技術文檔
    "web": 0.15,            # 通用網路資料
    "books": 0.05,          # 技術書籍
}

# 對話模型的配方
DIALOGUE_MIXTURE = {
    "dialogue": 0.40,       # 對話資料
    "web": 0.30,            # 網路資料
    "books": 0.15,          # 書籍
    "wikipedia": 0.10,      # 百科知識
    "qa": 0.05,             # 問答資料
}

# 多語言模型的配方
MULTILINGUAL_MIXTURE = {
    "english": 0.40,        # 英文
    "chinese": 0.30,        # 中文
    "other_languages": 0.20,# 其他語言
    "code": 0.10,           # 程式碼（多語言通用）
}
```

**動態資料混合（課程學習）**：

```python
class CurriculumDataMixture:
    """課程學習資料混合"""

    def __init__(self, datasets: Dict[str, Dataset]):
        self.datasets = datasets
        self.current_phase = 0

    def get_mixture_for_phase(self, phase: int) -> Dict[str, float]:
        """
        根據訓練階段返回不同的混合配方

        Args:
            phase: 訓練階段（0: 早期, 1: 中期, 2: 後期）
        """
        if phase == 0:
            # 早期：更多高品質、結構化資料
            return {
                "wikipedia": 0.3,
                "books": 0.3,
                "papers": 0.2,
                "web": 0.2,
            }
        elif phase == 1:
            # 中期：平衡的混合
            return {
                "web": 0.5,
                "books": 0.2,
                "wikipedia": 0.15,
                "papers": 0.15,
            }
        else:
            # 後期：更多樣化，包含更多網路資料
            return {
                "web": 0.65,
                "books": 0.15,
                "wikipedia": 0.1,
                "papers": 0.1,
            }

    def create_curriculum_mixture(self, phase: int, samples: int) -> Dataset:
        """建立特定階段的資料混合"""
        weights = self.get_mixture_for_phase(phase)

        print(f"\n訓練階段 {phase + 1} 的資料配方:")
        for name, weight in weights.items():
            print(f"  {name}: {weight * 100:.1f}%")

        # 使用 DataMixtureManager 建立混合
        manager = DataMixtureManager()
        for name, weight in weights.items():
            if name in self.datasets:
                manager.add_dataset(name, self.datasets[name], weight)

        return manager.create_mixture(samples)

# 使用範例
# curriculum = CurriculumDataMixture(all_datasets)
# phase_0_data = curriculum.create_curriculum_mixture(phase=0, samples=100000)
# phase_1_data = curriculum.create_curriculum_mixture(phase=1, samples=100000)
# phase_2_data = curriculum.create_curriculum_mixture(phase=2, samples=100000)
```

**資料去重策略**：

```python
from datasketch import MinHash, MinHashLSH
from typing import Set

class DataDeduplicator:
    """資料去重器（使用 MinHash LSH）"""

    def __init__(self, threshold=0.8, num_perm=128):
        """
        初始化去重器

        Args:
            threshold: 相似度閾值（0-1）
            num_perm: MinHash 排列數
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_ids = set()

    def get_minhash(self, text: str) -> MinHash:
        """計算文字的 MinHash"""
        minhash = MinHash(num_perm=self.num_perm)

        # 使用字符級 n-gram
        for i in range(len(text) - 3):
            minhash.update(text[i:i+3].encode('utf8'))

        return minhash

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """
        檢查文字是否重複

        Args:
            text: 待檢查文字
            doc_id: 文檔 ID

        Returns:
            是否重複
        """
        minhash = self.get_minhash(text)

        # 查詢相似文檔
        similar = self.lsh.query(minhash)

        if similar:
            return True

        # 添加到索引
        self.lsh.insert(doc_id, minhash)
        self.seen_ids.add(doc_id)

        return False

    def deduplicate_dataset(self, dataset: Dataset) -> Dataset:
        """去重資料集"""
        print("開始去重...")

        unique_indices = []

        for idx, example in enumerate(dataset):
            text = example.get("text", "")
            doc_id = f"doc_{idx}"

            if not self.is_duplicate(text, doc_id):
                unique_indices.append(idx)

            if (idx + 1) % 1000 == 0:
                print(f"已處理 {idx + 1} 個樣本，保留 {len(unique_indices)} 個")

        # 選擇唯一樣本
        deduplicated = dataset.select(unique_indices)

        print(f"\n去重完成:")
        print(f"  原始: {len(dataset):,} 樣本")
        print(f"  去重後: {len(deduplicated):,} 樣本")
        print(f"  去除率: {(1 - len(deduplicated)/len(dataset)) * 100:.2f}%")

        return deduplicated

# 使用
# deduplicator = DataDeduplicator(threshold=0.8)
# unique_dataset = deduplicator.deduplicate_dataset(mixed_dataset)
```

#### 步驟 2：資料清理

```python
import re
from typing import List, Dict

class DataCleaner:
    """預訓練資料清理器"""

    @staticmethod
    def remove_duplicates(texts: List[str]) -> List[str]:
        """移除重複內容"""
        seen = set()
        unique_texts = []

        for text in texts:
            # 使用雜湊避免記憶體問題
            text_hash = hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def filter_low_quality(text: str) -> bool:
        """過濾低品質文字"""
        # 基本品質檢查
        if len(text) < 100:  # 太短
            return False

        if len(text) > 100000:  # 太長
            return False

        # 檢查標點符號比例
        punct_ratio = sum(c in '.,!?;:' for c in text) / len(text)
        if punct_ratio > 0.3:  # 標點符號過多
            return False

        # 檢查大寫字母比例
        upper_ratio = sum(c.isupper() for c in text) / len(text)
        if upper_ratio > 0.5:  # 大寫字母過多
            return False

        return True

    @staticmethod
    def normalize_text(text: str) -> str:
        """規範化文字"""
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊控制字元
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # 規範化引號
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

# 使用範例
cleaner = DataCleaner()

raw_texts = [
    "這是一段正常的文字...",
    "太短",  # 會被過濾
    "!!!!!!!!!!!!!!!!!",  # 標點符號過多，會被過濾
]

# 過濾低品質文字
high_quality = [t for t in raw_texts if cleaner.filter_low_quality(t)]

# 規範化
normalized = [cleaner.normalize_text(t) for t in high_quality]

print(f"原始: {len(raw_texts)}, 高品質: {len(high_quality)}")
```

#### 步驟 3：訓練自定義 Tokenizer

**為什麼需要訓練自定義 Tokenizer？**

1. **語言優化**：現有 tokenizer 對特定語言（如中文）效率低
2. **領域適配**：包含領域特定詞彙和術語
3. **效率提升**：減少 token 數量，提高推論速度
4. **成本降低**：更少的 token 意味著更低的 API 成本

**Tokenizer 訓練實作**：

```python
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    processors,
    decoders,
    normalizers
)
from transformers import PreTrainedTokenizerFast
from typing import List, Iterator
import os

class TokenizerTrainer:
    """自定義 Tokenizer 訓練器"""

    def __init__(self, vocab_size=32000, model_type="bpe"):
        """
        初始化 Tokenizer 訓練器

        Args:
            vocab_size: 詞彙表大小
            model_type: 模型類型 ("bpe", "wordpiece", "unigram")
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.tokenizer = None

    def create_tokenizer(self):
        """建立 tokenizer"""

        if self.model_type == "bpe":
            # Byte-Pair Encoding（GPT 系列使用）
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        elif self.model_type == "wordpiece":
            # WordPiece（BERT 系列使用）
            tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

        elif self.model_type == "unigram":
            # Unigram（T5, LLaMA 系列使用）
            tokenizer = Tokenizer(models.Unigram())

        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")

        # 設定 normalizer（標準化）
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),  # Unicode 正規化
            normalizers.StripAccents(),  # 移除重音符號
        ])

        # 設定 pre-tokenizer（預分詞）
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False)
        ])

        # 設定 decoder
        tokenizer.decoder = decoders.ByteLevel()

        self.tokenizer = tokenizer
        return tokenizer

    def train_from_iterator(
        self,
        text_iterator: Iterator[str],
        special_tokens: List[str] = None
    ):
        """
        從文字迭代器訓練 tokenizer

        Args:
            text_iterator: 文字迭代器
            special_tokens: 特殊 token 列表
        """
        if self.tokenizer is None:
            self.create_tokenizer()

        # 設定特殊 tokens
        if special_tokens is None:
            special_tokens = [
                "<unk>",  # 未知 token
                "<s>",    # 開始 token
                "</s>",   # 結束 token
                "<pad>",  # 填充 token
            ]

        # 建立訓練器
        if self.model_type == "bpe":
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
                min_frequency=2
            )
        elif self.model_type == "wordpiece":
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
                min_frequency=2
            )
        elif self.model_type == "unigram":
            trainer = trainers.UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
                unk_token="<unk>"
            )

        # 訓練
        print(f"開始訓練 {self.model_type} tokenizer...")
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        print("訓練完成！")

        # 設定 post-processor
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def save(self, save_directory: str):
        """保存 tokenizer"""
        os.makedirs(save_directory, exist_ok=True)

        # 保存 tokenizer
        self.tokenizer.save(os.path.join(save_directory, "tokenizer.json"))

        # 轉換為 Hugging Face tokenizer 並保存
        tokenizer_hf = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>"
        )
        tokenizer_hf.save_pretrained(save_directory)

        print(f"Tokenizer 已保存至 {save_directory}")

    def test_tokenizer(self, test_texts: List[str]):
        """測試 tokenizer"""
        print("\n測試 tokenizer:")
        print("=" * 60)

        for text in test_texts:
            encoding = self.tokenizer.encode(text)
            tokens = encoding.tokens
            ids = encoding.ids

            print(f"\n原文: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {ids}")
            print(f"Token 數量: {len(tokens)}")


# 使用範例：訓練中文 tokenizer
def train_chinese_tokenizer():
    """訓練中文優化的 tokenizer"""

    # 準備訓練資料（使用迭代器節省記憶體）
    def get_training_corpus():
        # 這裡應該是大規模中文文字
        texts = [
            "人工智慧是計算機科學的一個分支。",
            "機器學習是實現人工智慧的一種方法。",
            "深度學習使用多層神經網絡進行學習。",
            # ... 更多文字
        ]
        for text in texts:
            yield text

    # 建立訓練器
    trainer = TokenizerTrainer(vocab_size=32000, model_type="bpe")

    # 訓練
    trainer.train_from_iterator(
        get_training_corpus(),
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
    )

    # 保存
    trainer.save("./chinese_tokenizer")

    # 測試
    trainer.test_tokenizer([
        "人工智慧快速發展",
        "機器學習應用廣泛"
    ])

# train_chinese_tokenizer()
```

**比較不同 Tokenizer 的效率**：

```python
from transformers import AutoTokenizer

def compare_tokenizers(text: str):
    """比較不同 tokenizer 的效率"""

    tokenizers = {
        "GPT-2 (英文)": "gpt2",
        "LLaMA (多語言)": "meta-llama/Llama-2-7b-hf",
        "Qwen (中文優化)": "Qwen/Qwen-7B",
    }

    print(f"測試文字: {text}")
    print("=" * 80)

    results = []

    for name, model_name in tokenizers.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokens = tokenizer.tokenize(text)
            token_count = len(tokens)

            results.append({
                "name": name,
                "token_count": token_count,
                "tokens": tokens[:10]  # 只顯示前 10 個
            })

            print(f"\n{name}:")
            print(f"  Token 數量: {token_count}")
            print(f"  前 10 個 tokens: {tokens[:10]}")

        except Exception as e:
            print(f"\n{name}: 載入失敗 - {e}")

    # 找出最有效的 tokenizer
    best = min(results, key=lambda x: x["token_count"])
    print(f"\n最有效: {best['name']} ({best['token_count']} tokens)")

# 測試
# compare_tokenizers("人工智慧技術正在快速發展，機器學習和深度學習是其核心技術。")
```

#### 步驟 4：Tokenization

```python
from transformers import AutoTokenizer
from datasets import Dataset

class PretrainingTokenizer:
    """預訓練資料 Tokenization"""

    def __init__(self, tokenizer_name="gpt2", max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 設定 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_function(self, examples):
        """Tokenize 文字"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

    def prepare_dataset(self, texts: List[str]):
        """準備訓練資料集"""
        # 建立 Dataset
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

# 使用範例
tokenizer_tool = PretrainingTokenizer(max_length=512)

texts = [
    "這是第一段文字，用於預訓練。",
    "這是第二段文字，包含更多內容。"
]

tokenized_data = tokenizer_tool.prepare_dataset(texts)
print(f"Tokenized dataset: {tokenized_data}")
```

### 4.3.2 大規模分散式訓練

#### DeepSpeed 配置

```python
# deepspeed_config.json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 100000,
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-4,
      "warmup_num_steps": 2000
    }
  }
}
```

#### 訓練腳本

```python
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import deepspeed

class PretrainingPipeline:
    """預訓練流程"""

    def __init__(self, config_path="model_config.json"):
        # 模型配置
        self.model_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )

        # 建立模型
        self.model = GPT2LMHeadModel(self.model_config)

        # 顯示模型大小
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型參數數量: {total_params / 1e6:.2f}M")

    def setup_training_args(self, output_dir="./pretrained_model"):
        """設定訓練參數"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,

            # 訓練設定
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,

            # 優化器
            learning_rate=6e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,

            # 學習率調度
            lr_scheduler_type="cosine",
            warmup_steps=2000,

            # 混合精度
            fp16=True,

            # 保存與記錄
            save_steps=1000,
            logging_steps=100,
            save_total_limit=3,

            # DeepSpeed
            deepspeed="deepspeed_config.json",
        )

        return training_args

    def train(self, train_dataset, output_dir="./pretrained_model"):
        """執行預訓練"""
        training_args = self.setup_training_args(output_dir)

        # 資料整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # 開始訓練
        print("開始預訓練...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        print(f"模型已保存至 {output_dir}")

        return trainer

# 使用範例（需要大量計算資源）
# pipeline = PretrainingPipeline()
# trainer = pipeline.train(tokenized_dataset)
```

### 4.3.3 訓練監控

```python
from transformers import TrainerCallback
import wandb

class PretrainingMonitor(TrainerCallback):
    """預訓練監控回調"""

    def __init__(self):
        # 初始化 Weights & Biases
        wandb.init(project="llm-pretraining", name="my-model")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """記錄訓練指標"""
        if logs:
            # 記錄損失
            if "loss" in logs:
                wandb.log({"train/loss": logs["loss"]}, step=state.global_step)

            # 記錄學習率
            if "learning_rate" in logs:
                wandb.log({"train/lr": logs["learning_rate"]}, step=state.global_step)

            # 計算困惑度
            if "loss" in logs:
                perplexity = torch.exp(torch.tensor(logs["loss"]))
                wandb.log({"train/perplexity": perplexity}, step=state.global_step)

            print(f"Step {state.global_step}: Loss={logs.get('loss', 'N/A'):.4f}")

    def on_save(self, args, state, control, **kwargs):
        """保存檢查點時的回調"""
        print(f"檢查點已保存在 step {state.global_step}")

# 使用
# trainer = Trainer(
#     ...
#     callbacks=[PretrainingMonitor()]
# )
```

### 4.3.4 持續預訓練 (Continual Pretraining)

**什麼是持續預訓練？**

持續預訓練是在已有的預訓練模型基礎上，使用特定領域或語言的資料繼續訓練，以適應特定場景。

**為什麼需要持續預訓練？**

1. **領域適配**：讓通用模型適應特定領域（醫療、法律、金融等）
2. **語言擴展**：增強模型對特定語言的理解能力
3. **知識更新**：更新模型的知識到最新時間點
4. **成本效益**：比從頭訓練便宜，比微調效果更好

**持續預訓練的類型**：

| 類型 | 目的 | 資料量 | 訓練步數 | 成本 |
|------|------|--------|---------|------|
| 領域適配 | 適應特定領域 | 10B-100B tokens | 10K-50K | 中 |
| 語言擴展 | 擴展語言能力 | 50B-200B tokens | 50K-100K | 高 |
| 知識更新 | 更新時效知識 | 5B-50B tokens | 5K-20K | 低 |

**實作範例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

class ContinualPretraining:
    """持續預訓練實作"""

    def __init__(self, base_model_name="meta-llama/Llama-2-7b-hf"):
        """
        初始化持續預訓練

        Args:
            base_model_name: 基礎預訓練模型名稱
        """
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None

    def load_base_model(self):
        """載入基礎模型"""
        print(f"載入基礎模型: {self.base_model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"模型參數量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

    def prepare_domain_data(self, domain_texts):
        """
        準備領域資料

        Args:
            domain_texts: 領域特定文字列表
        """
        from datasets import Dataset

        # 建立資料集
        dataset = Dataset.from_dict({"text": domain_texts})

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def continual_pretrain(
        self,
        train_dataset,
        output_dir="./continual_pretrained",
        num_epochs=1,
        learning_rate=1e-5,  # 比初始預訓練小
        warmup_steps=500
    ):
        """
        執行持續預訓練

        Args:
            train_dataset: 訓練資料集
            output_dir: 輸出目錄
            num_epochs: 訓練輪數
            learning_rate: 學習率（通常比初始預訓練小）
            warmup_steps: 熱身步數
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,

            # 訓練設定
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,

            # 優化器設定（使用較小的學習率）
            learning_rate=learning_rate,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,

            # 學習率調度
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,

            # 混合精度
            bf16=True,

            # 記錄與保存
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,

            # 啟用梯度檢查點以節省記憶體
            gradient_checkpointing=True,
        )

        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        print("開始持續預訓練...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"持續預訓練完成！模型已保存至 {output_dir}")

# 使用範例：醫療領域適配
# medical_texts = [
#     "心肌梗塞是一種嚴重的心臟疾病...",
#     "糖尿病的主要症狀包括...",
#     # ... 更多醫療文字
# ]
#
# cp = ContinualPretraining("meta-llama/Llama-2-7b-hf")
# cp.load_base_model()
# train_data = cp.prepare_domain_data(medical_texts)
# cp.continual_pretrain(train_data, output_dir="./llama2-medical")
```

**持續預訓練的最佳實踐**：

1. **學習率設定**
   ```python
   # 持續預訓練使用較小的學習率
   initial_pretraining_lr = 3e-4
   continual_pretraining_lr = 1e-5  # 約為初始預訓練的 1/30
   ```

2. **資料混合策略**
   ```python
   # 混合通用資料和領域資料
   general_data_ratio = 0.3
   domain_data_ratio = 0.7

   # 避免災難性遺忘
   mixed_dataset = mix_datasets(
       general_dataset,
       domain_dataset,
       weights=[general_data_ratio, domain_data_ratio]
   )
   ```

3. **訓練步數控制**
   ```python
   # 根據資料量和模型大小調整
   # 經驗法則：每 1B tokens 訓練約 1000-2000 steps
   tokens_count = 10e9  # 10B tokens
   estimated_steps = int(tokens_count / (batch_size * seq_length))
   ```

4. **評估與驗證**
   ```python
   def evaluate_domain_adaptation(model, domain_eval_data, general_eval_data):
       """評估領域適配效果"""
       domain_ppl = calculate_perplexity(model, domain_eval_data)
       general_ppl = calculate_perplexity(model, general_eval_data)

       print(f"領域困惑度: {domain_ppl:.2f}")
       print(f"通用困惑度: {general_ppl:.2f}")

       # 確保通用能力沒有顯著下降
       assert general_ppl < baseline_general_ppl * 1.2
   ```

**避免災難性遺忘**：

```python
class ContinualPretrainingWithRegularization:
    """帶正則化的持續預訓練（避免災難性遺忘）"""

    def __init__(self, base_model):
        self.model = base_model
        # 保存原始模型參數用於正則化
        self.original_params = {
            name: param.clone().detach()
            for name, param in base_model.named_parameters()
        }

    def elastic_weight_consolidation_loss(self, lambda_ewc=0.4):
        """
        彈性權重鞏固損失

        防止重要參數發生大幅變化
        """
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                ewc_loss += torch.sum(
                    (param - self.original_params[name]) ** 2
                )
        return lambda_ewc * ewc_loss
```

---

## 4.4 Scaling Laws 與高效預訓練技術

### 4.4.1 Scaling Laws

**Chinchilla Scaling Laws (2022)**：

研究發現，模型性能取決於三個因素：
1. 模型參數量 (N)
2. 訓練資料量 (D)
3. 計算量 (C)

**關鍵發現**：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中：
- L：損失
- N：參數量
- D：訓練 tokens 數量
- E, A, B, α, β：擬合常數

**最優配置**：

對於給定的計算預算 C：
- 模型大小和訓練資料應該**同時增長**
- 最優比例：每增加 1 倍參數，增加約 20 倍訓練資料

**實際應用**：

| 模型 | 參數 | 訓練 Tokens | 是否符合 Chinchilla |
|------|------|-------------|-------------------|
| GPT-3 | 175B | 300B | ❌ 訓練不足 |
| Chinchilla | 70B | 1.4T | ✅ 最優配置 |
| LLaMA | 7B-65B | 1T-1.4T | ✅ 符合 |
| LLaMA 2 | 7B-70B | 2T | ✅ 符合 |

**啟示**：
- 不要盲目增大模型規模
- 充足的訓練資料同樣重要
- 小模型 + 更多資料可能更有效

### 4.4.2 高效預訓練技術

#### Flash Attention

**原理**：
- 優化注意力計算的記憶體訪問模式
- 減少 HBM (High Bandwidth Memory) 訪問
- 使用 SRAM 進行快取

**效果**：
- 速度提升 2-4 倍
- 記憶體使用減少
- 支援更長的序列

**使用**：

```python
from flash_attn import flash_attn_qkvpacked_func

# 在模型中使用 Flash Attention
# transformers 庫已經整合了 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

#### Gradient Checkpointing

**原理**：
- 不保存所有中間啟用值
- 需要時重新計算
- 以時間換空間

**使用**：

```python
model.gradient_checkpointing_enable()

# 或在訓練參數中設定
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
)
```

**效果**：
- 記憶體使用減少 30-50%
- 訓練速度降低 15-25%
- 允許使用更大的 batch size

#### Mixed Precision Training

**FP16 訓練**：

```python
training_args = TrainingArguments(
    ...
    fp16=True,  # 啟用 FP16
)
```

**BF16 訓練**（更穩定）：

```python
training_args = TrainingArguments(
    ...
    bf16=True,  # 啟用 BF16（需要 Ampere+ GPU）
)
```

**效果**：
- 速度提升 2-3 倍
- 記憶體減半
- BF16 數值穩定性更好

#### ZeRO (Zero Redundancy Optimizer)

**DeepSpeed ZeRO 三階段**：

**Stage 1**：
- 分割優化器狀態
- 記憶體減少 4 倍

**Stage 2**：
- 分割優化器狀態 + 梯度
- 記憶體減少 8 倍

**Stage 3**：
- 分割優化器狀態 + 梯度 + 模型參數
- 記憶體減少 64 倍以上

**配置**：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}
```

### 4.4.3 訓練穩定性技術

**預訓練過程中常見的穩定性問題**：

1. **梯度爆炸/消失**
2. **損失突增（Loss Spike）**
3. **訓練發散**
4. **數值不穩定**

**解決方案**：

#### 1. 梯度裁剪

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...
    max_grad_norm=1.0,  # 梯度裁剪閾值
)

# 或手動實作
import torch.nn as nn

def clip_gradients(model, max_norm=1.0):
    """手動梯度裁剪"""
    total_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm
    )
    return total_norm
```

#### 2. 學習率預熱與衰減

```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5
):
    """
    帶預熱的餘弦退火學習率調度器

    Args:
        optimizer: 優化器
        num_warmup_steps: 預熱步數
        num_training_steps: 總訓練步數
        num_cycles: 餘弦週期數
    """
    def lr_lambda(current_step):
        # 預熱階段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 餘弦退火階段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda)


# 使用範例
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=6e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000
)
```

#### 3. 損失監控與檢查點恢復

```python
import torch
from collections import deque
import numpy as np

class LossMonitor:
    """訓練損失監控器"""

    def __init__(self, window_size=100, spike_threshold=1.5):
        """
        初始化監控器

        Args:
            window_size: 滑動窗口大小
            spike_threshold: 損失突增閾值（倍數）
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.loss_history = deque(maxlen=window_size)
        self.best_checkpoint = None
        self.best_loss = float('inf')

    def check_loss_spike(self, current_loss: float) -> bool:
        """
        檢查是否發生損失突增

        Args:
            current_loss: 當前損失值

        Returns:
            是否發生突增
        """
        if len(self.loss_history) < self.window_size // 2:
            self.loss_history.append(current_loss)
            return False

        # 計算歷史平均
        avg_loss = np.mean(self.loss_history)

        # 檢查是否突增
        if current_loss > avg_loss * self.spike_threshold:
            print(f"⚠️  警告：損失突增！")
            print(f"   當前損失: {current_loss:.4f}")
            print(f"   平均損失: {avg_loss:.4f}")
            print(f"   突增倍數: {current_loss / avg_loss:.2f}x")
            return True

        self.loss_history.append(current_loss)
        return False

    def save_checkpoint_if_best(self, model, loss: float, path: str):
        """保存最佳檢查點"""
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': loss,
            }, path)
            print(f"✓ 保存最佳檢查點 (loss: {loss:.4f})")

    def recover_from_spike(self, model, checkpoint_path: str):
        """從損失突增中恢復"""
        print("正在從最佳檢查點恢復...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已恢復到損失為 {checkpoint['loss']:.4f} 的檢查點")
        return checkpoint['loss']

# 使用範例
# monitor = LossMonitor(window_size=100, spike_threshold=1.5)
#
# for step, batch in enumerate(train_dataloader):
#     loss = train_step(model, batch)
#
#     if monitor.check_loss_spike(loss):
#         # 發生損失突增，恢復檢查點
#         monitor.recover_from_spike(model, "best_checkpoint.pt")
#         # 可能需要降低學習率
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= 0.5
#
#     monitor.save_checkpoint_if_best(model, loss, "best_checkpoint.pt")
```

#### 4. 權重初始化

```python
import torch.nn as nn
import math

def initialize_model_weights(model, init_std=0.02):
    """
    初始化模型權重

    使用截斷正態分佈初始化，常用於 Transformer 模型

    Args:
        model: 模型
        init_std: 初始化標準差
    """
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # 線性層：截斷正態分佈
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            # 嵌入層
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            # Layer Normalization
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    model.apply(_init_weights)

    # 特殊處理：殘差連接的縮放
    # GPT-2 風格：按層深度縮放
    for name, param in model.named_parameters():
        if "c_proj.weight" in name or "out_proj.weight" in name:
            # 殘差投影層使用更小的初始化
            param.data.normal_(mean=0.0, std=init_std / math.sqrt(2 * model.config.n_layer))


# 使用
# initialize_model_weights(model, init_std=0.02)
```

#### 5. 數值穩定性技巧

```python
import torch

class StablePretraining:
    """數值穩定的預訓練技巧"""

    @staticmethod
    def stable_softmax(logits, dim=-1):
        """數值穩定的 softmax"""
        # 減去最大值避免溢出
        logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
        logits_shifted = logits - logits_max
        return torch.softmax(logits_shifted, dim=dim)

    @staticmethod
    def log_sum_exp_stable(logits, dim=-1):
        """數值穩定的 log-sum-exp"""
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        return max_logits + torch.log(
            torch.sum(torch.exp(logits - max_logits), dim=dim, keepdim=True)
        )

    @staticmethod
    def check_tensor_health(tensor, name="tensor"):
        """檢查張量健康狀況"""
        if torch.isnan(tensor).any():
            print(f"❌ {name} 包含 NaN!")
            return False

        if torch.isinf(tensor).any():
            print(f"❌ {name} 包含 Inf!")
            return False

        if (tensor.abs() > 1e6).any():
            print(f"⚠️  {name} 包含異常大的值!")
            print(f"   最大值: {tensor.max().item()}")
            print(f"   最小值: {tensor.min().item()}")

        return True

# 在訓練循環中使用
# if not StablePretraining.check_tensor_health(loss, "loss"):
#     print("檢測到數值問題，跳過此批次")
#     continue
```

### 4.4.4 預訓練最佳實踐

**1. 計算資源規劃**

```python
def estimate_training_time(
    model_params: int,
    training_tokens: int,
    gpu_tflops: float,
    num_gpus: int,
    mfu: float = 0.5  # Model FLOPs Utilization
) -> dict:
    """
    估算訓練時間

    Args:
        model_params: 模型參數量（單位：個）
        training_tokens: 訓練 token 數（單位：個）
        gpu_tflops: 單 GPU 算力（TFLOPs）
        num_gpus: GPU 數量
        mfu: 模型 FLOPs 利用率（通常 0.3-0.6）

    Returns:
        包含訓練時間估算的字典
    """
    # 每個 token 的 FLOPs 計算（前向 + 反向）
    # 前向：6 * model_params
    # 反向：2 * 前向 = 12 * model_params
    flops_per_token = 6 * model_params + 12 * model_params

    # 總 FLOPs
    total_flops = flops_per_token * training_tokens

    # 有效算力
    effective_tflops = gpu_tflops * num_gpus * mfu

    # 訓練時間（秒）
    training_time_seconds = total_flops / (effective_tflops * 1e12)

    # 轉換為不同單位
    training_time_hours = training_time_seconds / 3600
    training_time_days = training_time_hours / 24

    return {
        "total_flops": total_flops,
        "effective_tflops": effective_tflops,
        "time_seconds": training_time_seconds,
        "time_hours": training_time_hours,
        "time_days": training_time_days,
    }

# 範例：估算 LLaMA-7B 規模模型的訓練時間
result = estimate_training_time(
    model_params=7e9,        # 7B 參數
    training_tokens=1e12,    # 1T tokens
    gpu_tflops=312,          # A100 40GB
    num_gpus=64,             # 64 張 A100
    mfu=0.45                 # 45% MFU
)

print(f"訓練時間估算:")
print(f"  總 FLOPs: {result['total_flops']:.2e}")
print(f"  有效算力: {result['effective_tflops']:.2f} TFLOPs")
print(f"  預計時間: {result['time_days']:.1f} 天")
```

**2. 檢查點策略**

```python
class CheckpointStrategy:
    """檢查點管理策略"""

    def __init__(
        self,
        save_dir: str,
        save_interval_steps: int = 1000,
        keep_last_n: int = 3,
        keep_best_n: int = 2
    ):
        self.save_dir = save_dir
        self.save_interval_steps = save_interval_steps
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n

        self.checkpoints = []
        self.best_checkpoints = []

    def should_save(self, step: int) -> bool:
        """判斷是否應該保存檢查點"""
        return step % self.save_interval_steps == 0

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        loss: float
    ):
        """保存檢查點"""
        checkpoint_path = f"{self.save_dir}/checkpoint-{step}.pt"

        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        self.checkpoints.append((step, loss, checkpoint_path))

        # 清理舊檢查點
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """清理舊檢查點，只保留最近的 N 個"""
        if len(self.checkpoints) > self.keep_last_n:
            # 按步數排序
            self.checkpoints.sort(key=lambda x: x[0])

            # 刪除最舊的檢查點
            to_remove = self.checkpoints[:-self.keep_last_n]
            for step, loss, path in to_remove:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"已刪除舊檢查點: {path}")

            self.checkpoints = self.checkpoints[-self.keep_last_n:]
```

**3. 資源監控**

```python
import psutil
import torch

class ResourceMonitor:
    """資源監控器"""

    @staticmethod
    def get_gpu_memory_usage():
        """獲取 GPU 記憶體使用情況"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization": allocated / torch.cuda.get_device_properties(0).total_memory * 100
            }
        return None

    @staticmethod
    def get_cpu_memory_usage():
        """獲取 CPU 記憶體使用情況"""
        memory = psutil.virtual_memory()
        return {
            "used_gb": memory.used / 1024**3,
            "total_gb": memory.total / 1024**3,
            "percent": memory.percent
        }

    @staticmethod
    def log_resource_usage():
        """記錄資源使用情況"""
        gpu_mem = ResourceMonitor.get_gpu_memory_usage()
        cpu_mem = ResourceMonitor.get_cpu_memory_usage()

        print(f"\n資源使用:")
        if gpu_mem:
            print(f"  GPU 記憶體: {gpu_mem['allocated_gb']:.2f}/{gpu_mem['reserved_gb']:.2f} GB")
        print(f"  CPU 記憶體: {cpu_mem['used_gb']:.2f}/{cpu_mem['total_gb']:.2f} GB ({cpu_mem['percent']:.1f}%)")

# 在訓練循環中定期呼叫
# if step % 100 == 0:
#     ResourceMonitor.log_resource_usage()
```

### 4.4.5 模型評估基準

**常用評估基準**：

| 基準 | 任務類型 | 語言 | 說明 |
|------|---------|------|------|
| **MMLU** | 知識問答 | 英文 | 57 個學科的多選題，評估廣泛知識 |
| **HellaSwag** | 常識推理 | 英文 | 句子補全，評估常識推理 |
| **TruthfulQA** | 真實性 | 英文 | 評估模型是否產生真實答案 |
| **HumanEval** | 程式碼生成 | Python | 評估程式碼生成能力 |
| **GSM8K** | 數學推理 | 英文 | 小學數學應用題 |
| **C-Eval** | 知識問答 | 中文 | 中文綜合能力評估 |
| **CMMLU** | 知識問答 | 中文 | 中文多任務語言理解 |

**評估實作**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

class ModelEvaluator:
    """模型評估器"""

    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def evaluate_perplexity(self, dataset_name="wikitext", split="test"):
        """
        評估困惑度

        Args:
            dataset_name: 資料集名稱
            split: 資料集分割

        Returns:
            困惑度
        """
        print(f"評估困惑度: {dataset_name}")

        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)

        total_loss = 0
        total_tokens = 0

        self.model.eval()

        for i, example in enumerate(dataset):
            text = example["text"]
            if len(text.strip()) == 0:
                continue

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

            if (i + 1) % 10 == 0:
                print(f"  處理 {i+1} 個樣本...")

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        print(f"  困惑度: {perplexity:.2f}\n")
        return perplexity

    def evaluate_mmlu(self):
        """評估 MMLU 基準"""
        # 載入 MMLU 資料集
        dataset = load_dataset("cais/mmlu", "all")

        correct = 0
        total = 0

        for example in dataset["test"]:
            question = example["question"]
            choices = example["choices"]
            answer = example["answer"]

            # 構建提示
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"

            # 獲取模型預測
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            pred = self.tokenizer.decode(outputs[0][-1])

            if pred.strip() == chr(65 + answer):
                correct += 1

            total += 1

            if total % 100 == 0:
                print(f"MMLU 進度: {total} questions, 準確率: {correct/total*100:.2f}%")

        accuracy = correct / total * 100
        print(f"\nMMLU 最終準確率: {accuracy:.2f}%")
        return accuracy

# 使用
# evaluator = ModelEvaluator("./my_pretrained_model")
# ppl = evaluator.evaluate_perplexity()
# mmlu_score = evaluator.evaluate_mmlu()
```

---

## 4.5 實作範例

### 4.5.1 小規模預訓練實驗

```python
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def small_scale_pretraining():
    """小規模預訓練實驗（教學目的）"""

    print("=" * 60)
    print("小規模 GPT 預訓練實驗")
    print("=" * 60)

    # 1. 建立小型模型配置
    config = GPT2Config(
        vocab_size=50257,      # GPT-2 詞彙表大小
        n_positions=512,       # 最大序列長度
        n_embd=256,            # 嵌入維度
        n_layer=6,             # 層數
        n_head=8,              # 注意力頭數
    )

    model = GPT2LMHeadModel(config)

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數數量: {total_params / 1e6:.2f}M")

    # 2. 準備資料
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 使用小型資料集（WikiText）
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # 3. 訓練配置
    training_args = TrainingArguments(
        output_dir="./tiny_gpt",
        overwrite_output_dir=True,

        # 訓練設定
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,

        # 優化器
        learning_rate=5e-4,
        weight_decay=0.1,

        # 學習率調度
        lr_scheduler_type="cosine",
        warmup_steps=100,

        # 記錄
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,

        # 混合精度
        fp16=torch.cuda.is_available(),
    )

    # 4. 資料整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM，不是 Masked LM
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. 訓練
    print("\n開始訓練...")
    trainer.train()

    # 7. 保存模型
    trainer.save_model("./tiny_gpt_final")
    tokenizer.save_pretrained("./tiny_gpt_final")

    print("\n訓練完成！模型已保存至 ./tiny_gpt_final")

    # 8. 測試生成
    print("\n測試文字生成:")
    model.eval()
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")

    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n生成結果:\n{generated_text}")

# 執行
# small_scale_pretraining()
```

### 4.5.2 從預訓練模型微調

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def finetune_from_pretrained():
    """從預訓練模型開始微調"""

    # 1. 載入預訓練模型
    model_name = "gpt2"  # 可替換為其他模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 應用 LoRA（參數高效微調）
    lora_config = LoraConfig(
        r=8,                          # LoRA 秩
        lora_alpha=32,                # 縮放因子
        target_modules=["c_attn"],    # 應用 LoRA 的模組
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 準備資料（使用指令資料集）
    dataset = load_dataset("json", data_files="your_instruction_data.json", split="train")

    def format_instruction(example):
        """格式化指令資料"""
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}

    dataset = dataset.map(format_instruction)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # 4. 訓練配置
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,          # LoRA 可使用較大學習率
        logging_steps=10,
        save_steps=100,
        fp16=True,
    )

    # 5. 訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # 6. 保存 LoRA 權重
    model.save_pretrained("./lora_weights")
    print("LoRA 權重已保存")

# 使用
# finetune_from_pretrained()
```

### 4.5.3 評估預訓練模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def evaluate_pretrained_model(model_name="gpt2"):
    """評估預訓練模型的困惑度"""

    print(f"評估模型: {model_name}")

    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()

    # 載入測試資料
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # 計算困惑度
    total_loss = 0
    total_tokens = 0

    for i, example in enumerate(test_dataset):
        if i >= 100:  # 只評估前 100 個樣本（示範）
            break

        text = example["text"]
        if len(text.strip()) == 0:
            continue

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 計算損失
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

        if (i + 1) % 10 == 0:
            print(f"已處理 {i+1} 個樣本...")

    # 計算平均困惑度
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"\n結果:")
    print(f"平均損失: {avg_loss:.4f}")
    print(f"困惑度 (Perplexity): {perplexity:.2f}")

    return perplexity

# 使用
# evaluate_pretrained_model("gpt2")
# evaluate_pretrained_model("meta-llama/Llama-2-7b-hf")
```

---

## 參考資源

### 論文

1. **Scaling Laws**: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
2. **Chinchilla**: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
3. **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
4. **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
5. **Mistral**: "Mistral 7B" (Jiang et al., 2023)
6. **Mixtral**: "Mixtral of Experts" (Jiang et al., 2024)
7. **BLOOM**: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" (BigScience, 2022)
8. **DeepSeek-V3**: "DeepSeek-V3 Technical Report" (2024)

### 工具與框架

- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **DeepSpeed**: https://github.com/microsoft/DeepSpeed
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **PyTorch FSDP**: https://pytorch.org/docs/stable/fsdp.html
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention

### 開源模型

- **LLaMA**: https://github.com/facebookresearch/llama
- **Mistral**: https://huggingface.co/mistralai
- **Qwen**: https://huggingface.co/Qwen
- **BLOOM**: https://huggingface.co/bigscience/bloom
- **DeepSeek**: https://huggingface.co/deepseek-ai

### 資料集

- **The Pile**: https://pile.eleuther.ai/
- **Common Crawl**: https://commoncrawl.org/
- **Wikipedia**: https://huggingface.co/datasets/wikipedia
- **C4**: https://huggingface.co/datasets/c4
- **BookCorpus**: https://huggingface.co/datasets/bookcorpus

---

## 預訓練速查表

### 超參數快速參考

| 參數 | 推薦值 | 說明 |
|------|--------|------|
| **學習率** | | |
| 初始預訓練 | 3e-4 ~ 6e-4 | 大模型可用較大學習率 |
| 持續預訓練 | 1e-5 ~ 5e-5 | 約為初始預訓練的 1/30 |
| 微調 | 1e-5 ~ 2e-4 | 依任務調整 |
| **批次大小** | | |
| 小模型 (< 10B) | 256 ~ 512 | 全局批次大小 |
| 中型 (10-70B) | 512 ~ 2048 | |
| 大型 (> 70B) | 2048 ~ 4096 | |
| **優化器** | | |
| AdamW β1 | 0.9 | 動量係數 |
| AdamW β2 | 0.95 ~ 0.999 | 二階動量係數 |
| Weight Decay | 0.1 | 權重衰減 |
| Gradient Clipping | 1.0 | 梯度裁剪閾值 |
| **學習率調度** | | |
| 預熱步數 | 2000 ~ 5000 | 總步數的 1-2% |
| 調度器 | Cosine | 餘弦退火 |
| **序列長度** | | |
| 標準 | 2048 | GPT-2, LLaMA |
| 長上下文 | 4096 ~ 8192 | LLaMA 2, Mistral |
| 超長 | 32K ~ 128K | Qwen 2, Yi |

### 資料配方參考

```
通用模型：
  網路資料: 60-70%
  書籍: 8-15%
  Wikipedia: 4-5%
  程式碼: 8-10%
  學術論文: 2-5%

程式碼模型：
  程式碼: 50-60%
  技術文檔: 20-30%
  網路資料: 10-20%

中文模型：
  中文資料: 40-50%
  英文資料: 30-40%
  程式碼: 10-20%
```

### 硬體需求參考

| 模型規模 | 訓練 (FP16) | 訓練 (BF16+ZeRO3) | 推理 (FP16) | 推理 (INT4) |
|---------|------------|------------------|------------|------------|
| 1B | 8 GB | 4 GB | 2 GB | 1 GB |
| 7B | 56 GB | 14 GB | 14 GB | 4 GB |
| 13B | 104 GB | 26 GB | 26 GB | 7 GB |
| 33B | 264 GB | 66 GB | 66 GB | 17 GB |
| 70B | 560 GB | 140 GB | 140 GB | 35 GB |

### 訓練時間估算

```python
# 簡化公式
訓練天數 = (6 * 參數量 * Token數) / (GPU算力 * GPU數量 * MFU * 86400)

# 範例：7B 模型，1T tokens，64 張 A100
訓練天數 = (6 * 7e9 * 1e12) / (312e12 * 64 * 0.45 * 86400) ≈ 55 天
```

---

## 常見問題 (FAQ)

### Q1: 什麼時候應該從頭預訓練？什麼時候應該持續預訓練？

**從頭預訓練**適用於：
- 資源充足（數百萬美元預算）
- 有大規模獨特資料
- 需要完全控制模型架構和訓練過程
- 目標是建立基礎模型

**持續預訓練**適用於：
- 資源有限
- 需要領域適配（醫療、法律等）
- 需要語言擴展
- 希望保留通用能力

**建議**：對於大多數應用，持續預訓練 + 微調是最經濟高效的選擇。

### Q2: 如何選擇合適的基礎模型？

考慮因素：
1. **任務需求**：
   - 中文對話 → Qwen, DeepSeek
   - 程式碼生成 → DeepSeek, CodeLLaMA
   - 多語言 → BLOOM, LLaMA 3
   - 邊緣部署 → Phi-3, Gemma 2B

2. **資源限制**：
   - 單卡 16GB → ≤ 7B 模型
   - 單卡 40GB → ≤ 13B 模型
   - 多卡 → 70B+ 模型

3. **授權要求**：
   - 商業使用 → LLaMA 2/3, Mistral, Gemma
   - 研究使用 → 大多數開源模型

### Q3: 訓練過程中出現損失突增（Loss Spike）怎麼辦？

**原因**：
- 學習率過大
- 批次中出現異常資料
- 數值不穩定
- 梯度爆炸

**解決方法**：
1. **立即處理**：
   - 從最近的穩定檢查點恢復
   - 降低學習率（減半）
   - 跳過當前批次

2. **預防措施**：
   - 使用梯度裁剪（max_grad_norm=1.0）
   - 適當的學習率預熱
   - 監控損失曲線
   - 定期保存檢查點

3. **程式碼範例**：
   ```python
   if loss > avg_loss * 1.5:  # 檢測突增
       model.load_state_dict(best_checkpoint)  # 恢復
       for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.5  # 降低學習率
   ```

### Q4: 預訓練資料應該去重嗎？去重的程度如何把握？

**應該去重**：
- 精確去重：移除完全相同的文檔（必須）
- 近似去重：移除高度相似的文檔（推薦）

**去重程度**：
- **保守**（相似度閾值 > 0.9）：保留更多資料，模型可能記憶訓練資料
- **激進**（相似度閾值 > 0.7）：去除更多資料，減少記憶，但可能損失資訊

**建議**：
- 網路爬取資料：激進去重（0.7-0.8）
- 書籍、論文：保守去重（0.9-0.95）
- Wikipedia：精確去重即可

### Q5: 如何判斷預訓練是否收斂？

**收斂指標**：

1. **訓練損失**：
   - 穩定下降後趨於平穩
   - 不應該持續上升

2. **驗證困惑度**：
   - 在驗證集上評估
   - 持續降低或穩定
   - 如果上升則可能過擬合

3. **下游任務性能**：
   - 定期在代表性任務上評估
   - 性能不再顯著提升

**何時停止訓練**：
- 達到預定的 token 數量（推薦）
- 驗證困惑度不再下降（早停）
- 預算/時間限制

### Q6: BF16 和 FP16 哪個更好？

**FP16（Half Precision）**：
- ✅ 廣泛支援（V100+）
- ✅ 速度快，省記憶體
- ❌ 數值範圍小，容易溢出
- ❌ 需要 Loss Scaling

**BF16（Brain Float 16）**：
- ✅ 數值範圍與 FP32 相同
- ✅ 訓練穩定，不需要 Loss Scaling
- ✅ Google、Meta 推薦
- ❌ 需要新硬體（A100, H100, TPU）

**建議**：
- 如果有 A100/H100：使用 BF16
- 如果只有 V100：使用 FP16 + 仔細調參
- 小模型：兩者差異不大
- 大模型：BF16 明顯更穩定

### Q7: 如何平衡模型大小和訓練資料量？

**Chinchilla Scaling Laws**：

對於給定的計算預算 C：
- 模型參數 N 和訓練 tokens D 應該同時增長
- 最優比例：**N : D ≈ 1 : 20**

**實際建議**：

| 模型參數 | 最少訓練 Tokens | 推薦訓練 Tokens |
|---------|----------------|----------------|
| 1B | 20B | 200B |
| 7B | 140B | 1.4T |
| 13B | 260B | 2.6T |
| 70B | 1.4T | 14T |

**啟示**：
- GPT-3 (175B, 300B tokens) → 訓練不足
- LLaMA (7B, 1T tokens) → 更優配置
- 不要盲目追求大模型，資料同樣重要

### Q8: 訓練中如何檢測和處理 GPU OOM（記憶體不足）？

**預防措施**：

1. **減少批次大小**：
   ```python
   per_device_batch_size = 4  # 從 8 降到 4
   gradient_accumulation_steps = 8  # 增加累積步數保持總批次
   ```

2. **啟用梯度檢查點**：
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **使用 DeepSpeed ZeRO**：
   ```json
   "zero_optimization": {
       "stage": 3,  // 最激進的記憶體優化
       "offload_optimizer": {"device": "cpu"},
       "offload_param": {"device": "cpu"}
   }
   ```

4. **減少序列長度**：
   ```python
   max_length = 1024  # 從 2048 降到 1024
   ```

**記憶體估算**：
```python
# 模型參數記憶體（訓練）
model_memory = params * 16  # FP16: 2 bytes/param, 包含梯度等

# 範例：7B 模型
model_memory = 7e9 * 16 ≈ 112 GB
```

### Q9: 如何複現論文中的預訓練結果？

**關鍵因素**：

1. **資料**（最重要）：
   - 相同的資料來源和配方
   - 相同的預處理流程
   - 相同的去重策略

2. **超參數**：
   - 學習率、批次大小
   - 優化器參數
   - 學習率調度

3. **隨機種子**：
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   ```

4. **硬體和精度**：
   - FP16 vs BF16 會有差異
   - 不同 GPU 可能有數值差異

**現實**：
- 完全複現幾乎不可能（資料不公開）
- 目標是接近而非完全相同
- 關注相對性能而非絕對數字

### Q10: 預訓練完成後如何保存和分享模型？

**標準格式 - Hugging Face**：

```python
# 保存模型
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 上傳到 Hugging Face Hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./my_model",
    repo_id="your-username/model-name",
    repo_type="model"
)
```

**包含的文件**：
```
my_model/
├── config.json           # 模型配置
├── pytorch_model.bin     # 模型權重
├── tokenizer.json        # Tokenizer
├── tokenizer_config.json # Tokenizer 配置
├── special_tokens_map.json
├── README.md             # 模型卡片
└── training_args.json    # 訓練參數（可選）
```

**模型卡片建議包含**：
- 模型描述和用途
- 訓練資料來源
- 已知限制
- 授權資訊
- 使用範例
- 評估結果

---

## 總結

預訓練是 LLM 開發的基礎階段：

### 核心要點

1. **選擇合適的預訓練模型**
   - 根據任務需求選擇模型規模
   - 考慮語言支援和領域適配
   - 評估硬體和成本限制

2. **理解 Scaling Laws**
   - 模型大小和訓練資料需要平衡
   - 不要盲目追求大模型
   - Chinchilla 最優比例：N : D ≈ 1 : 20

3. **高效訓練技術**
   - Flash Attention 提升速度
   - ZeRO 減少記憶體需求
   - 混合精度訓練加速
   - Gradient Checkpointing 節省記憶體

4. **實務建議**
   - 小團隊：使用開源預訓練模型 + 微調
   - 中型團隊：領域特定的持續預訓練
   - 大型團隊：從頭預訓練

5. **開源生態**
   - LLaMA 系列：性能優秀，廣泛使用
   - Mistral：效率高，MoE 架構
   - Qwen：中文優化
   - DeepSeek：極低成本，開源

### 未來趨勢

1. **更高效的架構**：MoE、State Space Models
2. **更長的上下文**：百萬 token 級別
3. **多模態整合**：文字、圖像、音訊統一模型
4. **更低的訓練成本**：新的訓練技術和硬體
5. **開源化**：更多高品質開源模型
