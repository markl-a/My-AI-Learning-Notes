# Magpie × distilabel 合成資料實作筆記

> 對應 [全景圖 #15](../2024-2026_AI完整領域全景圖.md);實作前置請見 [`../2.深入LLM模型工程與LLM運維/3.資料集準備與建立/`](../2.深入LLM模型工程與LLM運維/3.資料集準備與建立/)

2024 年後 SFT / DPO 的瓶頸已經從「算力」轉移到「**指令資料的品質與多樣性**」。Magpie(Xu et al., 2024)與 Argilla 的 **distilabel** 框架,把「從 instruct 模型蒸餾自我對齊資料」這件事從一次性的 Notebook 腳本,升級成可重現、可驗證的 DAG pipeline。本文整理一份從零到上 Hub 的完整 recipe。

---

## 1. distilabel 框架介紹

[distilabel](https://github.com/argilla-io/distilabel) 是 Argilla(現已併入 Hugging Face)推出的 synthetic data pipeline 框架,核心抽象有三:

- **Step**:資料流節點,負責 I/O 與 batching(`LoadDataFromHub`、`LoadDataFromDicts`、`PushToHub`)。
- **Task**:封裝 prompt template + LLM 呼叫的特化 Step,內建 `TextGeneration`、`UltraFeedback`、`EvolInstruct`、`SelfInstruct`、`Magpie`、`PrometheusEval` 等數十種。
- **Pipeline**:用 `>>` operator 接成有向圖,執行時自動處理 retry、cache、checkpoint、batch size、GPU 並行。

跟 LangChain 比起來,distilabel 不做 agent / RAG,而是專注於「**離線批次生成**」場景:幾百萬筆 prompt 灌進去、幾百萬筆對齊資料出來,過程中允許 OOM、API 429、模型擲骰失敗,中斷後可以 resume。

---

## 2. Magpie 原理

傳統 self-instruct(Wang et al., 2022)要先準備幾百筆 human-written seed,讓 LLM 模仿擴寫。Magpie 的洞察是:**對 instruct 模型,連 seed 都不需要**。

機制非常直觀:

1. 把 chat template 的「user 開頭」prefix 餵給模型,例如 Llama-3 的 `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n`。
2. 模型已經被 RLHF 訓練成「user 開頭後必須是一個合理問題」,於是它會**自己幻想出一個 user prompt**(這一步是 free-form generation,只到 `<|eot_id|>` 停止)。
3. 把生成出來的 prompt 再餵回去,正常跑一次 `assistant` 回應。

於是一筆 `(instruction, response)` 對誕生,且**完全不需要人類 seed**。原論文用這個方法從 Llama-3-70B-Instruct 蒸餾出 **Magpie-Pro-1M**,在 AlpacaEval 2 / Arena-Hard 上訓出來的 SFT 模型可超過 ShareGPT、OpenHermes 等人工資料集。

關鍵假設:**instruct 模型的對齊本身就是一個 prompt 分布的近似**,只要 prefix 切得對,就能反推出這個分布的樣本。

---

## 3. 環境準備

```bash
# Python 3.10+
pip install "distilabel[hf-inference-endpoints,vllm,sentence-transformers]>=1.4.0"
pip install argilla>=2.0
pip install sentence-transformers faiss-cpu langdetect

# HF token(需開啟 read + write,write 給 push_to_hub)
export HF_TOKEN="hf_xxx"
# 若用 HF Inference Endpoints
export HF_INFERENCE_ENDPOINT_URL="https://xxx.endpoints.huggingface.cloud"
```

本地若有 GPU(>= 24 GB)直接用 `vLLM`;沒有就用 `InferenceEndpointsLLM` 打 serverless / dedicated endpoint。

---

## 4. 完整 pipeline 範例

下面這份 ~120 行的 pipeline 做完一條龍:Magpie 自生 prompt → 雙模型回應 → UltraFeedback 評分 → SemDeDup 去重 → push 上 Hub。

```python
# magpie_pipeline.py
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromDicts,
    KeepColumns,
    PushToHub,
)
from distilabel.steps.tasks import (
    Magpie,
    UltraFeedback,
    EvolInstruct,
)
from distilabel.steps.filtering import EmbeddingDedup
from distilabel.embeddings import SentenceTransformerEmbeddings
from distilabel.llms import vLLM, InferenceEndpointsLLM

# ---- 1. 目標模型(用來「被蒸餾」的 teacher) ----
teacher = vLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
    magpie_pre_query_template="llama3",  # 內建 chat template prefix
    generation_kwargs={
        "temperature": 1.0,          # 高溫度提升多樣性
        "top_p": 0.95,
        "max_new_tokens": 1024,
    },
    extra_kwargs={"tensor_parallel_size": 1, "max_model_len": 4096},
)

# ---- 2. 第二位 responder(可換成 70B 提升回應品質) ----
responder = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 1024},
)

# ---- 3. Judge model(評分用,通常用更強的) ----
judge = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={"temperature": 0.1, "max_new_tokens": 1024},
)

with Pipeline(name="magpie-sft-dpo-corpus") as pipeline:
    # 4a. 種子(只用來控制 batch 數量,內容會被 Magpie 忽略)
    seed = LoadDataFromDicts(
        name="seed",
        data=[{"_dummy": i} for i in range(10_000)],  # 跑 1 萬筆
        batch_size=64,
    )

    # 4b. Magpie:自生 (instruction, response)
    magpie = Magpie(
        name="magpie_gen",
        llm=teacher,
        n_turns=1,                    # 1 = 單輪;設 2/3 可生 multi-turn
        only_instruction=False,       # False = 同時生 instruction 與 response
        system_prompt=(
            "You are a helpful AI assistant. The user will ask you a wide "
            "variety of questions across STEM, coding, reasoning, and creative writing."
        ),
        num_generations=1,
        input_batch_size=32,
    )

    # 4c. Evol-Instruct:把 Magpie 出來的 prompt 升難度
    evol = EvolInstruct(
        name="evol_difficulty",
        llm=judge,
        num_evolutions=2,             # 升 2 級
        store_evolutions=False,
        generate_answers=False,
        input_batch_size=16,
    )

    # 4d. 用 responder 對「進化後的 prompt」重新生成 chosen response
    from distilabel.steps.tasks import TextGeneration
    regen = TextGeneration(
        name="regen_chosen",
        llm=responder,
        input_batch_size=16,
    )

    # 4e. UltraFeedback:給 (instruction, response_pair) 評分,產生 DPO 用 chosen/rejected
    score = UltraFeedback(
        name="ultrafeedback",
        llm=judge,
        aspects=["instruction-following", "truthfulness", "honesty", "helpfulness"],
        input_batch_size=8,
    )

    # 4f. SemDeDup:embedding 去重(threshold 越大保留越多)
    embed = SentenceTransformerEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    dedup = EmbeddingDedup(
        name="semdedup",
        threshold=0.92,
        embeddings=embed,
        input_batch_size=128,
    )

    keep = KeepColumns(
        name="keep_cols",
        columns=["instruction", "generation", "evolved_instruction",
                 "ratings", "rationales", "model_name"],
    )

    push = PushToHub(
        name="push",
        repo_id="your-username/magpie-llama3-sft-10k",
        private=True,
        token=None,                   # 從 HF_TOKEN env 讀
    )

    # ---- 5. DAG 接線 ----
    seed >> magpie >> evol >> regen >> score >> dedup >> keep >> push


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            magpie.name: {"llm": {"generation_kwargs": {"temperature": 1.0}}},
        },
        use_cache=True,               # 中斷可 resume
    )
    distiset.push_to_hub("your-username/magpie-llama3-sft-10k")
```

跑 1 萬筆,8B teacher + 70B responder 在 1×A100 + HF endpoint 上大約 4–6 小時。

---

## 5. 品質控制策略

跑完先別開心,raw output 通常有 20–40% 是雜訊。常規後處理:

| 過濾 | 規則 | 工具 |
|---|---|---|
| Reward threshold | UltraFeedback 平均分 < 3.5 全丟 | `dataset.filter(lambda x: mean(x["ratings"]) >= 3.5)` |
| Length filter | response < 30 token 或 > 1500 token 丟 | tokenizer count |
| Language | 非英文 / 非中文丟(視目標而定) | `langdetect` |
| Refusal detect | 含 "I cannot" / "as an AI" 的丟 | regex / classifier |
| N-gram dedup | 4-gram Jaccard > 0.7 丟 | datasketch MinHash |
| Toxicity | Llama-Guard-2 標 unsafe 丟 | distilabel `ArgillaLabeller` |

實務上 reward threshold + SemDeDup + length filter 三個就能砍掉 ~35%,留下的 65% 品質會明顯高於 raw Magpie。

---

## 6. Evol-Instruct 整合

WizardLM 的 Evol-Instruct(Xu et al., 2023)用 LLM 把簡單 prompt 改寫成更難版本。Magpie 出來的 prompt 偏向「中等難度的常見問題」,接 Evol 可以拉高分布右尾:

```python
# 升難度的 prompt template(distilabel 內建)
EVOL_OPS = [
    "Add new constraints to the original prompt",
    "Replace common requirements with more rare ones",
    "Deepen the depth and breadth of the inquiry",
    "Concretize the prompt with specific examples",
    "Increase reasoning steps required",
]
```

建議:每筆 Magpie prompt 跑 1–2 次 Evol 即可;>3 次容易產生「人類也看不懂」的怪題,反而被 judge 打低分。

---

## 7. PersonaHub 整合

Tencent PersonaHub(Chan et al., 2024)提供 **10 億個合成 persona**(`proj-persona/PersonaHub`)。把 persona 注入 Magpie 的 system prompt,能大幅提升話題多樣性:

```python
import random
from datasets import load_dataset

personas = load_dataset("proj-persona/PersonaHub", "persona", split="train")

def make_seed(n=10000):
    samples = personas.shuffle().select(range(n))
    return [
        {"persona": s["persona"],
         "system_prompt": (
             f"You are an AI assistant. The next user is: {s['persona']}. "
             "Generate a question they would realistically ask."
         )}
        for s in samples
    ]
```

然後把 `system_prompt` 從 Magpie 的固定字串改成從 dataset row 讀。Magpie + PersonaHub 在 Nemotron-4 340B 的 report 裡是 instruction diversity 的主力。

---

## 8. 產出資料用於 SFT / DPO 的最小流程

UltraFeedback 會給每筆 sample 多個 response + 分數,最高分當 `chosen`、最低分當 `rejected`,直接餵 DPO:

```python
from datasets import load_dataset

ds = load_dataset("your-username/magpie-llama3-sft-10k", split="train")

def to_dpo(example):
    ratings = example["ratings"]
    gens = example["generations"]
    chosen_idx = ratings.index(max(ratings))
    rejected_idx = ratings.index(min(ratings))
    return {
        "prompt": example["instruction"],
        "chosen": gens[chosen_idx],
        "rejected": gens[rejected_idx],
    }

dpo_ds = ds.map(to_dpo).filter(lambda x: x["chosen"] != x["rejected"])
dpo_ds.push_to_hub("your-username/magpie-dpo-10k")
```

接下來請接到 [`../2.深入LLM模型工程與LLM運維/`](../2.深入LLM模型工程與LLM運維/) 的 SFT(TRL `SFTTrainer`)與 DPO(`DPOTrainer`)章節。

---

## 9. 真實案例

- **Phi-4(Microsoft, 2024)**:training corpus 大量使用「textbook-quality synthetic data」,搭配嚴格 reward model 過濾。14B 在 MMLU / GSM8K 打贏多數 70B 模型,佐證合成資料 + 強過濾 > 純爬蟲。
- **Nemotron-4 340B(NVIDIA, 2024)**:**對齊資料 98% 是合成**,Reward Model + Constitutional AI + Magpie-style self-generation 三管齊下,RewardBench 一度登頂。
- **Llama-3.1 405B**:後訓練文件明說使用 synthetic SFT data,並用 405B 自己當 judge 反向蒸餾到 8B/70B。
- **SmolLM2 / SmolTalk(HF, 2024)**:全 distilabel pipeline 開源,可直接抄。

---

## 10. 生產陷阱

1. **Reward model 偏誤**:同一個模型當 generator + judge,會偏好自己的風格(self-preference bias),建議 judge 用不同家族(e.g. teacher=Llama,judge=Qwen)。
2. **Persona collapse**:不加 PersonaHub 時,Magpie 在 100k+ 規模容易反覆出現「explain X」、「what is Y」這類 head distribution。SemDeDup threshold 設太鬆會留下大量近似樣本。
3. **語言污染**:Llama-3 即使指定英文 system prompt 偶爾會吐中文 / 日文片段,DPO 訓完模型會「無故切換語言」,務必加 `langdetect` 過濾。
4. **Mode collapse to refusal**:若 teacher 安全對齊太強,Magpie 會生大量「Sorry, I can't help with that」自問自答,占資料的 5–10%。
5. **License 陷阱**:用 Llama 系列生資料,output 仍受 Llama community license 約束;商用前先讀條款。GPT-4 生資料則違反 OpenAI ToS,不能用於訓練競品模型。
6. **Token cost 失控**:8B teacher × 70B judge × 100k samples 在 HF endpoint 上大約 $300–$800。先用 1k sample dry run 估成本再放大。

---

**延伸閱讀**:Magpie 原論文(arXiv:2406.08464)、distilabel docs、HuggingFaceH4/ultrafeedback_binarized、argilla/distilabel-intel-orca-dpo-pairs。
