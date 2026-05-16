> 對應 [全景圖 #4 推理模型](../../2024-2026_AI完整領域全景圖.md);搭配閱讀 [`./GRPO_DAPO_RLVR_實戰.md`](./GRPO_DAPO_RLVR_實戰.md);[`../12.推理模型應用/`](../12.推理模型應用/)

# PRM 訓練實作:從 Math-Shepherd 到 ReasonFlux-PRM

Process Reward Model (PRM) 是 2024-2026 推理模型訓練的關鍵組件之一。本文聚焦「怎麼把 PRM 訓出來、怎麼接到 GRPO、什麼時候該換 verifier」這三件事。

---

## 1. PRM vs ORM:為什麼推理任務必須 step-level

**ORM (Outcome Reward Model)** 只看最終答案對不對,回傳一個 scalar reward。它的問題在長 CoT 場景非常明顯:

- **稀疏訊號**:一條 20 步的推理,只有最後一步拿到 reward,中間 19 步的 credit assignment 全靠 policy gradient 自己推。對 7B 以下的模型幾乎學不動。
- **獎勵錯誤推理**:模型可能用錯誤步驟「碰巧」算出正確答案 (例如錯誤抵消、抄答案、subscribers hack),ORM 仍給滿分。這在 MATH / AIME 這種 numeric answer 的 dataset 特別嚴重。
- **無法 best-of-N 重排序**:ORM 只能事後判斷對錯,沒辦法在生成中途終止錯誤分支。

**PRM (Process Reward Model)** 對每一個 reasoning step 給分 $r_t \in [0,1]$,代表「從這步繼續走、最終答對的機率」。優點:

- **稠密 reward**,credit assignment 直接、policy gradient variance 顯著下降。
- 可做 **step-wise best-of-N**:beam search 時剪掉低分分支,inference compute 可控。
- 偵測 reward hacking:若中間出現「答案正確但 PRM 分數低」的步驟,通常代表抄捷徑。

代價是標註成本高 — 原始 PRM800K (OpenAI, 2023) 雇人逐步標,規模有限。後續所有 PRM 工作幾乎都在解「怎麼自動標」。

---

## 2. Math-Shepherd 路線:MCTS rollout 自動標 process label

Math-Shepherd (Wang et al., ACL 2024) 提出 **automatic process annotation**,核心想法是:

> 一步的「品質」 = 從這步繼續走 N 次 rollout,有幾次能到達正確答案。

具體做法:

1. 給定問題 $q$ 與一條 partial reasoning trace $s_1, s_2, \ldots, s_t$。
2. 用 completer model 從 $s_t$ rollout $K$ 次 (典型 $K = 8 \sim 16$),每次 sample 到結束。
3. 統計這 $K$ 條 rollout 裡有 $k$ 條最終答案正確,則 $\text{label}(s_t) = k/K$ (soft label) 或 $\mathbb{1}[k > 0]$ (hard label)。
4. 重複所有 step → 得到 (trace, step-labels) pair 訓練資料。

Math-Shepherd 用此法在 MATH/GSM8K 標了 **445k samples**,訓出的 PRM 把 Mistral-7B 從 77.9 → 84.1% (GSM8K) 與 28.6 → 33.0% (MATH),搭配 verifier 進一步推到 89.1 / 43.5%。後續 **OmegaPRM** (Google, 2024) 用 divide-and-conquer MCTS 把標註規模拉到 1.5M。

關鍵 trick:**只在「分歧步」做 rollout**。若所有 K 條 rollout 都對或都錯,該步沒鑑別力,跳過省算力。

---

## 3. 2025 進展:R-PRM / ReasonFlux-PRM

**R-PRM (Reasoning-Driven PRM, EMNLP 2025)** 解決傳統 PRM「只輸出分數、不解釋」的問題:

- 改用 **generative PRM** — 讓 PRM 自己 CoT 評估每一步,輸出 reasoning trace + verdict。
- 三段式訓練:(a) 強模型 distill 種子資料 → (b) preference optimization 自我進化 → (c) inference-time scaling (多次評估投票)。
- ProcessBench / PRMBench F1 +13.9 / +8.5,六個推理 benchmark 平均 +8.6 acc。

**ReasonFlux-PRM (2025)** 針對 long CoT 場景:

- 同時做 **step-level** 與 **trajectory-level** 監督,前者抓單步錯誤、後者評整條推理結構。
- 7B 版本在 SFT data selection +12.1%、RL +4.5%、test-time scaling +6.3%。
- 與 DeepSeek-R1 風格的 long CoT 對齊,適合 reasoning model 後訓練 pipeline。

兩者共通方向:PRM 本身越來越像「會推理的 judge」,而不是單純的 classifier。

---

## 4. Weak-to-strong 標註:小模型標、大模型驗

人標太貴、強模型 rollout 太慢時,**weak-to-strong labeling** 是經濟解:

- **Llama-3.1-8B** 當 completer,負責大量 rollout 與初步 process label。
- **Llama-3.1-70B** 當 verifier/judge,只審「不確定」樣本 (active learning,以 entropy 或 disagreement 篩選)。
- VersaPRM (2025) 用此 pipeline 訓多領域 PRM;ActPRM 用 uncertainty 估計把 70B judge 的呼叫量壓到 20% 以下。

實務 ratio:8B 標 100%、70B 抽審 10-20%、人工 spot-check 1% — 標 1M 樣本約 $2-5k API/GPU 成本。

---

## 5. 訓練 pipeline 全貌

```
[1] 收 reasoning trace
    └─ MATH / GSM8K / AIME / Olympiad → 用 policy model (e.g. Qwen2.5-Math-7B) sample N 條 CoT

[2] 切 step (用 "\n\n" / "Step k:" / sentence-tokenizer)
    └─ 每條 trace → [s_1, s_2, ..., s_T]

[3] MCTS / rollout 估 step-level p(correct)
    └─ 每個 s_t 從該前綴 rollout K 次 → soft label r_t = k/K

[4] 訓 token-level / step-level classifier
    └─ Backbone: Llama-3.1-8B / Qwen2.5-7B
    └─ Head: linear(hidden_dim → 1) + sigmoid
    └─ Loss: BCE on step-end token,或 MSE on soft label

[5] 評估 ProcessBench / PRMBench / best-of-N MATH
```

切 step 沒有一致標準 — 行業常見以 `\n\n` 或 `"Step k:"` 為 delimiter,生成模型若沒這格式可先用 SFT 教它輸出。

---

## 6. 可執行 minimal example (~50 行)

以下用 `trl` + 自訂 reward head 訓 step-level PRM。假設資料已是 `{prompt, steps, labels}` 格式,labels 為每 step 的 soft probability。

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

MODEL = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(MODEL)
STEP_TOKEN = "<|step|>"               # 在每步結尾插入此 marker
tok.add_special_tokens({"additional_special_tokens": [STEP_TOKEN]})
STEP_ID = tok.convert_tokens_to_ids(STEP_TOKEN)

class PRM(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16)
        self.backbone.resize_token_embeddings(len(tok))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, step_mask, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(h).squeeze(-1)                # (B, L)
        step_logits = logits[step_mask.bool()]           # 只取 step-end token
        if labels is None: return step_logits.sigmoid()
        loss = nn.functional.binary_cross_entropy_with_logits(step_logits, labels.float())
        return loss, step_logits.sigmoid()

def collate(batch):
    text = [b["prompt"] + STEP_TOKEN.join(b["steps"]) + STEP_TOKEN for b in batch]
    enc  = tok(text, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    step_mask = enc.input_ids == STEP_ID
    labels = torch.tensor(sum([b["labels"] for b in batch], []))
    return {**enc, "step_mask": step_mask, "labels": labels}

ds = load_dataset("peiyi9979/Math-Shepherd", split="train")
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate)

model = PRM(MODEL).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step, batch in enumerate(dl):
    batch = {k: v.cuda() for k, v in batch.items()}
    loss, _ = model(**batch)
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 50 == 0: print(f"step {step} loss {loss.item():.4f}")
```

實務上會用 DeepSpeed ZeRO-3 + LoRA、加 gradient checkpointing,本範例純為展示骨架。

---

## 7. PRM 配合 GRPO:稠密 reward 拉推理品質

GRPO (DeepSeek) 預設用 verifiable outcome reward (RLVR),credit assignment 全靠 group-relative advantage。若引入 PRM,reward 變成:

$$r_t = \alpha \cdot r^{\text{outcome}} + (1 - \alpha) \cdot r^{\text{PRM}}_t$$

或更精細:把 PRM 分數轉成 **step-level advantage**,僅在 step boundary token 加 reward,中間 token 共享。

實務經驗:

- $\alpha = 0.3 \sim 0.5$ 是甜蜜點。完全靠 PRM 容易 hack,完全靠 outcome 又退回 GRPO 原始問題。
- PRM 必須先 **calibrate**:把 soft label 重新縮到 0-1 mean ~ 0.5,否則 reward scale 失衡。
- 在 long CoT (>4k tokens) 場景,PRM 的稠密性比 outcome 重要 10x — DeepSeek-R1 之後沒用 PRM 主因是 verifiable answer 已足夠;但若任務沒有 verifier,PRM 仍是首選。

---

## 8. 驗證式 reward:何時用 verifier 取代 PRM

**RLVR (Reinforcement Learning with Verifiable Rewards)** 的核心觀察:當任務本身可程式化驗證,根本不需要學 reward model。

| 任務 | Verifier | PRM 還需要嗎 |
|------|----------|-------------|
| 數學 numeric answer | SymPy `simplify(pred - gold) == 0` | 不太需要 (除非要 step-level 剪枝) |
| 競賽程式 | unit test pass rate | 不需要 |
| SQL | 對 DB 執行比對結果 | 不需要 |
| LeetCode | judge runtime | 不需要 |
| 自由 QA / 對話 | 沒有 deterministic verifier | **需要 PRM** |
| 醫療 / 法律推理 | 部分驗證 | 混合:verifier + PRM |

DeepSeek-R1-Zero 拋棄 PRM 全用 RLVR,正是因為 MATH/AIME/code 都有確定性 verifier。原則:**能用 verifier 就別訓 PRM** — 後者會引入額外 bias 與 hacking surface。

---

## 9. 生產坑

**(a) PRM 過擬合 training distribution**
PRM 在 MATH 訓的話,換到 AIME / 物理推理常退化 5-15 分。緩解:多領域 mix (VersaPRM 路線)、定期用 OOD set monitor。

**(b) Reward hacking**
模型學會輸出「PRM 喜歡的句式」 — 例如多寫 "Therefore," "Let me verify," 等 PRM 訓練資料常見的開頭。對策:
- PRM 訓練時加入 adversarial paraphrase。
- RL 階段加 KL penalty 與 reference model。
- 定期人工抽查 high-PRM-low-accuracy 樣本。

**(c) 跨領域泛化**
數學 PRM 拿去評 code 通常崩。建議 per-domain PRM 或用 generative PRM (R-PRM) 路線,因為 CoT-style 評估比 scalar head 抗 domain shift。

**(d) Step 切分不一致**
訓練資料用 `\n\n` 切,inference 時模型輸出沒 `\n\n` → PRM 算不到分數。解法:訓 policy model 時 SFT 強制 step format,或在 PRM tokenizer 端做 fuzzy step detection。

**(e) Soft label vs hard label**
$K$ rollout 太少 (K<8) 時 soft label 雜訊大,直接 binarize 反而穩。$K \geq 16$ 後 soft label 開始有意義。

**(f) PRM size 選擇**
經驗法則:PRM 約 policy model 的 0.5-1x size 即可。再大邊際效益遞減且 inference cost 爆炸 (rollout × PRM forward × N candidates)。

---

## 延伸閱讀

- Math-Shepherd: <https://arxiv.org/abs/2312.08935>
- R-PRM: <https://arxiv.org/abs/2503.21295>
- ReasonFlux-PRM: <https://openreview.net/forum?id=f3sZjkQbv2>
- VersaPRM: <https://arxiv.org/pdf/2502.06737>
- DeepSeek-R1 (對 PRM 取捨的討論): <https://arxiv.org/abs/2501.12948>
- OpenAI Lessons of Developing PRMs: <https://arxiv.org/pdf/2501.07301>
