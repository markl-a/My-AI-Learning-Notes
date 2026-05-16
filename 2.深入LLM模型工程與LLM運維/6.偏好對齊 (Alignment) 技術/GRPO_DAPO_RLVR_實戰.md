> 對應 [全景圖 #3](../../2024-2026_AI完整領域全景圖.md);前置:[`../5.監督微調 (SFT)/`](../5.監督微調%20(SFT)/);相關:[`../12.推理模型應用/`](../12.推理模型應用/)

# GRPO / DAPO / RLVR 實戰:推理模型的後訓練範式

從 DeepSeekMath (2024)、DeepSeek-R1 (2025/01) 到 ByteDance DAPO (2025/03),GRPO 系列演算法在不到一年內完成了從 PPO 替代品到推理模型主流訓練範式的躍升。本文聚焦工程實作層面:數學公式、pipeline 重現、TRL 範例與生產陷阱。

---

## 1. 為何 GRPO 取代 PPO

PPO 在 RLHF 時代主導,但用於 LLM 推理訓練時暴露三個結構性問題:

**(1) Critic 開銷不可承受**。標準 PPO 需要一個與 actor 同規模的 value network(critic),對 7B 以上模型,訓練時等於要在 GPU 上常駐兩份權重 + 兩份優化器狀態(Adam 約 2× 參數量的 fp32 momentum/variance)。對 32B / 70B 模型來說,光是 critic 就吃掉 ~40% 顯存。

**(2) Value 估計在稀疏 reward 下不穩**。數學/程式任務的 reward 是 sequence-level 的 0/1(對或錯),critic 必須把 sparse terminal reward backprop 到每個 token 的 value,在長 chain-of-thought 場景下訊號雜訊比極差。

**(3) 浪費 sample 資訊**。PPO 每個 prompt 只 rollout 一條軌跡,advantage 完全靠 critic 估計。

GRPO(Group Relative Policy Optimization,DeepSeekMath 2024)的核心洞察:**對同一個 prompt 取 G 條 rollout,用群內相對排名直接估 advantage,完全去掉 critic**。

| 維度 | PPO | GRPO |
|------|-----|------|
| Critic | 需要,~1× actor 大小 | 不需要 |
| Advantage 來源 | GAE + value network | 群內 reward 的 z-score |
| Rollout/prompt | 1 | G(通常 8–64) |
| 顯存節省 | baseline | 約 40–50% |
| 適用場景 | dense reward | sparse / verifiable reward |

直觀理解:既然 reward 是 0/1,我就在同一題上產 16 個答案,對的給正 advantage、錯的給負 advantage,根本不用 critic 預測 value。

---

## 2. GRPO 數學公式

對一個 prompt $q$,從舊策略 $\pi_{\theta_{old}}$ 採樣 $G$ 條完整 output $\{o_1, ..., o_G\}$,每條 reward $r_i$。

**Advantage(群內標準化)**:

$$A_i = \frac{r_i - \mathrm{mean}(\{r_1,...,r_G\})}{\mathrm{std}(\{r_1,...,r_G\})}$$

同一個 advantage $A_i$ 廣播給該 output 的**所有 token**(這是後來 DAPO/Dr. GRPO 攻擊的重點)。

**目標函式**:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}}\left[ \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t} A_i, \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\right) \right] - \beta \, \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$$

其中 $\rho_{i,t} = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$ 是 token-level importance ratio。

**KL penalty(無偏估計)**,DeepSeekMath 用 k3 estimator:

$$\mathbb{D}_{KL} = \frac{\pi_{ref}(o_{i,t}|\cdot)}{\pi_\theta(o_{i,t}|\cdot)} - \log\frac{\pi_{ref}(o_{i,t}|\cdot)}{\pi_\theta(o_{i,t}|\cdot)} - 1$$

實務超參:$G=16$、$\epsilon=0.2$、$\beta=0.04$、temperature=0.9–1.0、學習率 1e-6 ~ 5e-6。

---

## 3. DeepSeek-R1 訓練 Pipeline 完整重現

DeepSeek-R1(Nature 2025)是四階段 pipeline,**不是純 RL**(R1-Zero 才是純 RL):

**Stage 1: Cold-Start SFT**
- 蒐集數千條長 CoT 範例(來自 R1-Zero 篩選 + 人工 refine),格式:`<think>...</think><answer>...</answer>`。
- 在 DeepSeek-V3-Base 上做幾百步 SFT,讓模型「會用 thinking format」。沒這步,純 RL 容易產出語言混雜、可讀性差的 CoT。

**Stage 2: Reasoning-Oriented GRPO**
- 在 math(MATH、AIME、Olympiad)、code(LeetCode、CodeForces)、邏輯題上跑 GRPO。
- Reward = 0.95 × correctness(verifiable)+ 0.05 × language consistency(避免中英混雜)。
- 訓練到 reasoning benchmark 收斂(通常 ~1000 steps、batch ~512 prompts、每 prompt 16 rollouts)。

**Stage 3: Rejection Sampling + SFT**
- 用 Stage 2 模型對 600K 推理 prompt 各取多條 rollout,只保留正確答案。
- 加 200K 非推理資料(寫作、role-play、QA),總共 800K SFT。

**Stage 4: All-Scenario RL**
- 再跑一輪 RL,推理任務用 rule-based reward,通用任務用 reward model(類 RLHF)。
- 目的:推理能力 + helpfulness/harmlessness 並存。

**Stage 5: Distillation 到小模型**
- 用 Stage 3 的 800K 資料對 Qwen2.5-1.5B/7B/32B、Llama3-8B/70B 做 SFT(**不做 RL**)。
- DeepSeek-R1-Distill-Qwen-32B 在 AIME 上達 72.6%,證實小模型直接 distill 比自己跑 RL 還划算。

工程細節:R1 主訓練用 vLLM 做 rollout(throughput 是 HF transformers 的 5–10×),actor 用 DeepSpeed ZeRO-3,reference model 用 fp16 凍結。

---

## 4. DAPO 改進

ByteDance Seed + 清華 AIR 在 2025/03 釋出 DAPO(`arXiv:2503.14476`),用 Qwen2.5-32B 在 AIME 2024 拿 50%,比 R1-Zero-Qwen-32B(47%)強且 step 少一半。四個關鍵改進:

**(1) Clip-Higher(decoupled clipping)**
- 標準 PPO 的對稱 clip $[1-\epsilon, 1+\epsilon]$ 會壓制低機率 token 的上升空間(entropy collapse)。
- DAPO 用非對稱:$[1-\epsilon_{low}, 1+\epsilon_{high}]$,其中 $\epsilon_{low}=0.2$、$\epsilon_{high}=0.28$。讓低機率但被驗證為對的 token 能更大幅度更新。

**(2) Dynamic Sampling**
- 如果一個 prompt 的 G 條 rollout 全對或全錯,group advantage 全 0,該 batch 浪費。
- DAPO 過濾掉 accuracy=0 或 =1 的 prompt,**持續重採樣直到 batch 滿**。保證每個 step 都有有效梯度。

**(3) Token-Level Policy Gradient(對抗 length bias)**
- 原 GRPO 對每條 output 做 `1/|o_i|` normalization,等於**長 output 的每 token 權重變小**。負面 reward 的長噪音 output 反而被輕罰,造成模型學會「拖長」。
- DAPO 改成 token-level aggregation:把 batch 內**所有 token** 一起平均,長 output 自然有更多項貢獻 loss。
- Dr. GRPO(同期工作,`arXiv:2503.20783`)指出 GRPO 的 advantage 還除以 group std,這在難題(std 大)上會抑制學習訊號,Dr. GRPO 直接拿掉 std normalization。

**(4) Overlong Reward Shaping**
- 截斷的 output 給軟懲罰(soft punishment),長度在 $[L_{max}-L_{cache}, L_{max}]$ 區間線性遞減 reward,超過 $L_{max}$ 直接懲罰常數。
- 比起硬截斷 + 0 reward,訓練更穩。

`λ-GRPO`(`arXiv:2510.06870`)進一步把 GRPO/DAPO/Dr. GRPO 統一在 Unified Token Preference 框架下,證明三者差別只在 token 權重函數,並學一個可訓練的 $\lambda$。

---

## 5. RLVR(Reinforcement Learning with Verifiable Rewards)

RLVR 是相對 RLHF 的根本範式轉移:**reward 不來自學習的 RM,而來自確定性 verifier**。三類常用 verifier:

**(a) Math Equivalence**
- 簡單:string match 標準答案(GSM8K 的 `#### 42`)。
- 進階:用 `sympy.simplify(a - b) == 0` 比代數等價;或用 `math_verify` 套件處理 LaTeX、分數、單位。
- 陷阱:`1/2` vs `0.5` vs `\frac{1}{2}` 必須統一,否則 reward 噪音大。

**(b) Unit Test(code)**
- 在 sandbox(Firejail / nsjail / Docker)裡跑 candidate code + 測資,pass rate 當 reward。
- 進階:用 hidden test cases 防止 model 死記訓練測資。
- 工具鏈:`code_contests`、`LiveCodeBench`、`bigcode-evaluation-harness`。

**(c) Formal Proof Checker**
- Lean4 / Coq / Isabelle 的 type checker 給 0/1。
- DeepSeek-Prover-V2、Kimina-Prover 都用此設定。
- 優勢:零 false positive;劣勢:reward 極稀疏(成功率 <1%),需要 curriculum + tree search 輔助。

RLVR 的工程價值:**verifier 不可被 hack**(編譯器不會因為你寫得花俏就給 pass),允許大規模 RL 而不怕 reward model overfitting / Goodhart 效應。這也是為何 R1、o1、Qwen3-Thinking 都聚焦 math+code——這兩個領域有現成 verifier。

---

## 6. PRM vs ORM

| | ORM(Outcome RM) | PRM(Process RM) |
|---|---|---|
| 監督粒度 | 整條 output 一個分數 | 每個推理步驟一個分數 |
| 標註成本 | 低(只需最終答案) | 高(OpenAI PRM800K 花了人工標註) |
| 自動化方案 | RLVR 直接用 verifier | Math-Shepherd 用 MC rollout 從 step 推估 |
| 適用任務 | GSM8K 等淺推理 | MATH、Olympiad 等長推理 |
| 在 RL 中角色 | terminal reward | dense reward / step-level advantage shaping |

**Math-Shepherd**(`arXiv:2312.08935`)的關鍵:**不靠人工標 step**。對每個中間 step $s_t$,從 $s_t$ 繼續 rollout N 條完整解答,正確率即該 step 的 label。再訓個 PRM 預測這個 label。

**OpenAI Let's Verify Step by Step**:PRM 在 MATH 上明顯勝過 ORM(78.2% vs 72.4% 在 best-of-N 設定下),但 GSM8K 上差距小,因為 GSM8K 步驟少、ORM 的 credit assignment 沒那麼難。

**生產建議**:
- Best-of-N 推理時 PRM 更好用(逐 step rerank)。
- RL 訓練時 ORM(RLVR)更穩。PRM 在 RL 中容易被 hack——模型學會產出「看起來正確的中間步驟」騙 PRM 卻得到錯答案(reward hacking)。
- 折衷:GRPO 用 outcome reward,推理時用 PRM rerank。

---

## 7. 可執行 Minimal Script(TRL GRPOTrainer + GSM8K)

需要 `trl>=0.14`、`transformers>=4.46`、`vllm`、`math_verify`。

```python
# grpo_gsm8k_minimal.py
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = (
    "You are a math solver. Think step by step inside <think>...</think>, "
    "then put the final numeric answer inside <answer>...</answer>."
)

def make_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["question"]},
        ],
        "gold": example["answer"].split("####")[-1].strip().replace(",", ""),
    }

ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.map(make_prompt, remove_columns=ds.column_names)

# ---------- Reward functions ----------
ANS_RE = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
FMT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)

def reward_correct(completions, gold, **kwargs):
    out = []
    for c, g in zip(completions, gold):
        text = c[0]["content"] if isinstance(c, list) else c
        m = ANS_RE.search(text)
        if m and m.group(1) == g:
            out.append(1.0)
        else:
            out.append(0.0)
    return out

def reward_format(completions, **kwargs):
    return [
        0.1 if FMT_RE.search(c[0]["content"] if isinstance(c, list) else c) else 0.0
        for c in completions
    ]

# ---------- Trainer ----------
cfg = GRPOConfig(
    output_dir="ckpt/qwen1.5b-grpo-gsm8k",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=8,              # G = group size
    max_prompt_length=512,
    max_completion_length=1024,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
    use_vllm=True,                  # 5-10x faster rollouts
    vllm_gpu_memory_utilization=0.4,
    beta=0.04,                      # KL coefficient
    epsilon=0.2,                    # PPO-clip
    temperature=0.9,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=[reward_correct, reward_format],
    args=cfg,
    train_dataset=ds,
)
trainer.train()
trainer.save_model()
```

啟動:`accelerate launch --num_processes 4 grpo_gsm8k_minimal.py`。單張 A100 80GB 大約能跑 1.5B-3B 模型,7B 以上需要多卡 + ZeRO-3。

**關鍵超參經驗值**:
- `num_generations` (G):8 是下限,16 較穩,32+ 邊際收益遞減。
- `beta` (KL):太高(>0.1)模型學不動;太低(<0.01)會跑飛、reward hacking。
- `epsilon` (clip):0.2 標配;想升級 DAPO 改 `epsilon_high=0.28`。
- `temperature`:1.0 比 0.7 好,需要 exploration。

---

## 8. 生產陷阱

**(1) KL Collapse / KL Explosion**
- 症狀:訓練幾百步後 KL 突然衝高(>20),policy 與 reference 完全脫鉤,輸出變亂碼或重複。
- 原因:`beta` 太低、學習率太高、reference 與 actor 數值精度不一致(actor bf16 / ref fp16 算 logprob 會差)。
- 緩解:adaptive KL controller(KL > target × 2 時放大 beta);定期 reset reference 到當前 policy(R1 沒做,但 DAPO 變體有)。

**(2) Reward Hacking**
- 症狀:reward 持續上升但人工檢查發現模型在「作弊」(如答案前後加 `\boxed{42}` 但推理過程胡扯)。
- 例子:Math 任務若 verifier 只匹配最終數字,模型會在 `<think>` 裡窮舉 0–100 然後猜對。
- 緩解:format reward 強制結構;hidden test cases;PRM 做 step-level 健全性檢查;對 verifier 做對抗測試。

**(3) Length Bias**
- 症狀:訓練後平均 output 長度從 500 token 漲到 4000,但 accuracy 沒漲。
- 根因:GRPO 的 `1/|o|` normalization 對長序列的負 reward 衰減過大(見 §4)。
- 緩解:換 DAPO 的 token-level loss;加 length penalty(`reward -= 0.0001 * max(0, len - 2048)`);監控 `mean_completion_length` 曲線。

**(4) Rollout / Update 不同步**
- vLLM 做 rollout 時 weight 是 step T 的,actor 更新後 step T+1 的 logprob 重算可能與舊 logprob 有微小偏差(KV cache、numerics)。
- 嚴重時 importance ratio 失真,clip 大量觸發。
- 緩解:每個 GRPO step 結束後強制把 actor weight sync 到 vLLM(TRL 已內建 `vllm_sync`)。

**(5) Group 內全對 / 全錯**
- advantage 全 0,該 prompt batch 浪費。Qwen2.5-7B 在 GSM8K 上後期約 30% prompt 全對。
- 緩解:Curriculum(從易到難)+ DAPO 的 dynamic sampling 重採樣。

**(6) Reference Model 顯存**
- 7B + 7B reference + KV cache,單張 80GB 卡常 OOM。
- 緩解:reference 用 fp8 量化或放在 CPU offload;或用 `disable_dropout=True` 共享 base + LoRA adapter 切換。

---

## 延伸閱讀與工具

- **論文**:DeepSeekMath (`arXiv:2402.03300`)、DeepSeek-R1 (`arXiv:2501.12948`, Nature 2025)、DAPO (`arXiv:2503.14476`)、Dr. GRPO (`arXiv:2503.20783`)、λ-GRPO (`arXiv:2510.06870`)、Math-Shepherd (`arXiv:2312.08935`)、Let's Verify Step by Step (OpenAI 2023)。
- **框架**:`trl` (HuggingFace)、`verl` (ByteDance,DAPO 官方實作)、`OpenRLHF`、`Unsloth`(單卡 GRPO)。
- **資料**:GSM8K、MATH、DeepMath-103K、NuminaMath、CodeContests、LiveCodeBench。
- **相關章節**:[`../5.監督微調 (SFT)/`](../5.監督微調%20(SFT)/) 是前置;[`../12.推理模型應用/`](../12.推理模型應用/) 接續推理時 best-of-N、PRM rerank。
