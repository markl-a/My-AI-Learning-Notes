> 前置:[`./Quantization_Primer.md`](./Quantization_Primer.md);姊妹篇:[`../7.模型壓縮與優化/量化與推論優化技術詳解.md`](../7.模型壓縮與優化/量化與推論優化技術詳解.md)

# 現代 LoRA 變體一次看懂 (Modern LoRA Variants, 2024-2025)

LoRA 自 2021 年問世後,2024-2025 出現一波改良潮:DoRA、LoRA+、PiSSA、rsLoRA、VeRA、AdaLoRA…… 每個都號稱比 vanilla 更強。本文把這些變體拆開來比較,並給出「什麼時候用哪個」的決策樹與生產級程式碼。

---

## 1. LoRA 復習:只訓 ΔW = BA

LoRA 的核心假設:微調時權重更新 ΔW 具有**低本徵秩 (low intrinsic rank)**。所以與其學一個 d×k 的密集 ΔW,不如把它分解成兩個低秩矩陣的乘積:

```
W' = W₀ + ΔW = W₀ + (α/r) · B · A
       └─ frozen ─┘   └ trainable ┘
A ∈ ℝ^(r×d)   B ∈ ℝ^(k×r)   r ≪ min(d, k)
```

預設初始化:**A 用 Kaiming 高斯、B 用零**,確保訓練開始時 ΔW = 0(模型輸出不變)。可訓練參數從 d·k 降到 r·(d+k),通常省 99% 以上。但 LoRA 也有四個被研究社群陸續點名的弱點,這正是 2024 年變體要修的:

1. A、B 共用同一個 learning rate,在 large width 模型下並非最優。
2. α/r 的 scaling 在高 rank (r > 64) 時讓 gradient 萎縮,訓不上去。
3. 隨機初始化 → 收斂慢,前幾百 step 浪費。
4. ΔW 只調「方向」沒調「幅度」,跟 full fine-tune 動態差很多。

---

## 2. 2024-2025 變體對比表

| 變體 | 提出 | 核心改動 | 額外參數 | 主要修哪個弱點 |
|------|------|---------|---------|--------------|
| **vanilla LoRA** | Hu 2021 | ΔW = (α/r)·BA | r·(d+k) | — |
| **DoRA** | NVIDIA 2024 ICML Oral | 分解 W 為 magnitude + direction,只對 direction 用 LoRA | LoRA + 一個 magnitude vector m ∈ ℝ^k | (4) 訓練動態接近 FT |
| **LoRA+** | Hayou 2024 ICML | A、B 用**不同** learning rate,lr_B ≈ 16-24× lr_A | 0 | (1) 提速 ~2× + 0.5-2% 性能 |
| **PiSSA** | Meng 2024 NeurIPS Spotlight | 用 W 的 **SVD 主成分** 初始化 BA,剩餘部分凍結 | 0 | (3) 收斂快、最終分數高 |
| **rsLoRA** | Kalajdzievski 2023 | scaling factor 改為 **α/√r**(不是 α/r) | 0 | (2) 解高 rank 失效 |
| **VeRA** | Kopiczko 2024 ICLR | A、B 用**跨層共用的 frozen 隨機矩陣**,只訓兩個 scaling 向量 d、b | r·1 + k·1 per layer | 參數量再砍 10× |
| **AdaLoRA** | Zhang 2023 ICLR | 用 SVD 形式 + importance score **動態分配 rank 預算** | 同 LoRA 但用得更有效 | 不同層 / 模組重要性不同 |

### Loss / 初始化差異要點

- **DoRA**:loss 計算與 LoRA 完全相同(cross-entropy on next token),只是前向多了 `m · (V + ΔV) / ||V + ΔV||` 這一步,其中 V 是預訓練 column-wise 方向。初始化:m 設為 `||W₀||_c`(每 column 的 L2 norm),A、B 同 LoRA。
- **LoRA+**:loss 同 LoRA,**只改 optimizer 設定** —— A、B 分到不同 param group,B 的 lr 乘以 `loraplus_lr_ratio`(常用 16)。
- **PiSSA**:**初始化是核心**。對 W₀ 做 truncated SVD 取前 r 個主成分:`W₀ ≈ U_r Σ_r V_rᵀ`,令 `A = √Σ_r · V_rᵀ`、`B = U_r · √Σ_r`,residual `W^res = W₀ − BA` 凍結。第一步 ΔW = 0 但**梯度方向已對準主要奇異向量**,所以收斂快。
- **rsLoRA**:只改一行 —— scaling 從 `α/r` 改成 `α/√r`。在 PEFT 中設 `use_rslora=True` 即可。
- **VeRA**:A、B 由**固定 seed** 的隨機數產生(不存權重,只存 seed),所有層共享。可訓練只剩 `d_b ∈ ℝ^r` 和 `d_d ∈ ℝ^d`,輸出為 `Λ_b · B · Λ_d · A`,Λ 為對角矩陣。
- **AdaLoRA**:把 BA 改寫成 `P Λ Q`(SVD 形式),Λ 的對角元素根據 sensitivity-based importance score 在訓練中被逐步 mask 掉,等價於動態調整每個模組的有效 rank。

---

## 3. DoRA 詳解:M·B·A 分解 + 訓練動態

DoRA 的關鍵洞察來自一個觀察:**full fine-tune 與 LoRA 的權重更新模式在「magnitude vs direction」這個座標系下分布顯著不同**。FT 可以做出「方向改一點、幅度卻改很多」或反之,但 LoRA 只能等比例改兩者,表達力受限。

### 數學形式

把預訓練權重 W₀ ∈ ℝ^(d×k) 按 column 拆解:

```
W₀ = m · (V / ||V||_c)
     ↑   └──── direction ────┘
   magnitude
m ∈ ℝ^(1×k)   V ∈ ℝ^(d×k)   ||·||_c = column-wise L2 norm
```

DoRA 把 direction 部分加上 LoRA 更新,**magnitude m 直接 fine-tune**:

```
W' = m · (V + BA) / ||V + BA||_c
     ↑           ↑
  可訓練       LoRA(可訓練)
```

### 為何訓練動態更像 FT

論文的核心圖表(Figure 1)畫出 ΔM 與 ΔD 的 scatter plot:
- **LoRA**:點分布呈強正相關 → 改方向就一定連帶改幅度,自由度低。
- **DoRA / FT**:點分布是負相關甚至無相關 → 模型可以**只調方向不調幅度**,或反之。

實證上 DoRA 在 commonsense reasoning、VLM、文生圖等多個任務上一致勝過同參數量的 LoRA,平均提升 1-3 個百分點。代價是訓練時每個 forward pass 多一次 norm 計算,**約多 10-20% 訓練時間**,但推論時可 merge 回 W,**零額外延遲**。

---

## 4. 何時用哪個:決策樹

```
                  你要 fine-tune 一個 LLM
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   VRAM 極吃緊         想接近 FT 品質      想加速收斂
   /嫌參數多             ↓                  ↓
        ↓             ┌──┴──┐         PiSSA
     VeRA         有量化?  否        (或 PiSSA + DoRA)
    (10× 少)         │     │
                    是     DoRA
                    ↓
                  QDoRA
                  
   特殊情境:
   - 預算極小、求穩、不想踩雷           → vanilla LoRA(r=8, α=16)
   - 想用 r ≥ 64 / 128 拉性能           → rsLoRA(α/√r)
   - 不同層重要性差很大、想自動調       → AdaLoRA
   - LoRA 已調好、想免費再壓 1-2%      → 加 LoRA+(改 optimizer 就好)
```

實務建議:**先用 vanilla LoRA 跑 baseline**,然後切到 DoRA 或 PiSSA 看是否真的有提升。不要一次堆多個 trick,debug 會痛苦。

---

## 5. 生產 code 範例:peft 0.x 設定

以下程式碼以 `peft >= 0.11` 為準,Mistral-7B 為例:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_id = "mistralai/Mistral-7B-v0.3"
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg,
                                            device_map="auto")
base = prepare_model_for_kbit_training(base)

# ── (A) DoRA(可直接搭 4-bit,即 QDoRA)──────────────────
cfg_dora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_dora=True,        # ← 一個 flag 啟用 DoRA
    task_type="CAUSAL_LM",
)

# ── (B) PiSSA(SVD 初始化)──────────────────────────────
cfg_pissa = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    init_lora_weights="pissa_niter_4",  # 4 次 power iteration 近似 SVD
    task_type="CAUSAL_LM",
)

# ── (C) rsLoRA(高 rank 場景)────────────────────────────
cfg_rs = LoraConfig(
    r=128, lora_alpha=16, use_rslora=True,   # scaling 變 α/√r
    target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM",
)

model = get_peft_model(base, cfg_dora)

# ── (D) LoRA+:不同 lr for A vs B ──────────────────────
from peft.optimizers import create_loraplus_optimizer
optim = create_loraplus_optimizer(
    model=model, optimizer_cls=torch.optim.AdamW,
    lr=2e-4, loraplus_lr_ratio=16,   # lr_B = 16 × lr_A
    weight_decay=0.0,
)
model.print_trainable_parameters()
```

---

## 6. 記憶體成本對比(7B base)

以 Mistral-7B、target_modules = all linear (`q,k,v,o,gate,up,down`)、r=16 為基準:

| 設定 | 可訓練參數 | 訓練 VRAM(batch=1, seq=2048) | 收斂 step | 備註 |
|------|----------|------------------------------|-----------|------|
| Full FT (BF16) | 7.24 B | ~120 GB(需 4×A100) | 基準 | 上限參考 |
| LoRA r=16 | ~42 M | ~16 GB | 1.0× | baseline |
| LoRA+ r=16 | ~42 M | ~16 GB | **0.5-0.7×** | 同記憶體更快 |
| PiSSA r=16 | ~42 M | ~16.5 GB | **0.6-0.8×** | 多一次 SVD init(~30s) |
| DoRA r=16 | ~42 M + 28 K | ~17 GB | 1.0-1.2× | magnitude 向量幾乎可忽略 |
| rsLoRA r=128 | ~336 M | ~22 GB | — | 為了高 rank 才用 |
| VeRA r=256 | ~1.4 M | ~13 GB | 1.2-1.5× | 參數量真的少 30× |
| AdaLoRA r=16 平均 | ~42 M | ~17 GB | 0.9× | 訓練中會 prune |
| **QLoRA r=16 (NF4)** | ~42 M | **~7 GB** | 1.0× | 24GB 卡基準選擇 |
| **QDoRA r=16 (NF4)** | ~42 M + 28 K | **~7.5 GB** | 1.0-1.2× | 性價比最高 |

數字為實測常見範圍,實際視 sequence length、gradient checkpointing、optimizer state 而定。

---

## 7. 與量化結合:QDoRA、QPiSSA

QLoRA 的成功讓社群很快試圖把同樣的 4-bit 凍結招式套到新變體上。

- **QDoRA**:Answer.AI 與 NVIDIA 合作的方案,把 QLoRA 中的 LoRA 換成 DoRA。在 Llama-2-7B / Llama-3-8B 上 QDoRA **同時打贏 QLoRA 與 Full FT**,Orca-Math 訓練 100K 樣本後 exact match 達 31.2%,Full FT 只有 26.0%,而 VRAM 用量約為 FT 的 1/9。PEFT 0.11+ 已支援:LoraConfig 同時開 `use_dora=True` 與 4-bit 量化即可。
- **QPiSSA**:把 SVD 主成分初始化套到 4-bit 凍結權重。難點是 SVD 必須在反量化的 W 上做,然後 residual `W^res` 再量化回去 —— PEFT 已有 `init_lora_weights="pissa"` 對 NF4 的官方支援。
- **不支援的組合**:VeRA 與量化目前在 PEFT 中**只能用 8-bit**,4-bit 還不穩(2025-Q1 狀態,需查 issue tracker)。AdaLoRA + 量化技術上 OK,但動態 prune 與量化交互效果未被廣泛驗證。

---

## 8. 框架支援度對比

截至 2025 年中,三大主流 SFT 框架對變體的支援:

| 變體 | Hugging Face PEFT | Unsloth | Axolotl | LLaMA-Factory |
|------|:----------------:|:-------:|:-------:|:-------------:|
| LoRA / QLoRA | ✅ | ✅(2× speed) | ✅ | ✅ |
| DoRA | ✅ `use_dora=True` | ✅ 0.x+ | ✅ via PEFT | ✅ `use_dora: true` |
| QDoRA(4-bit + DoRA) | ✅ | ⚠️(部分版本) | ✅(FSDP) | ✅ |
| LoRA+ | ✅ `create_loraplus_optimizer` | ✅ | ✅ | ✅ `loraplus_lr_ratio` |
| PiSSA | ✅ `init_lora_weights="pissa"` | ⚠️ | ✅ | ✅ `pissa_init: true` |
| rsLoRA | ✅ `use_rslora=True` | ✅ | ✅ | ✅ `use_rslora: true` |
| VeRA | ✅ | ❌ | ⚠️ | ⚠️ |
| AdaLoRA | ✅ | ❌ | ✅ | ✅ |

**選框架簡則**:追求單卡極致速度 → Unsloth(但變體較少);要 YAML config 一鍵跑全套變體 + 多卡 FSDP → Axolotl / LLaMA-Factory。LLaMA-Factory 對中文使用者最友善,內建 Web UI。

---

## 9. 真實案例:Mistral-7B + DoRA + Magpie 5K

社群有個常被引用的小型實驗組合:

- **Base**:`mistralai/Mistral-7B-v0.3`
- **資料**:Magpie-Pro-300K-Filtered 抽 5K 高品質 instruction-response pair(self-aligned, 無 human prompt)
- **方法**:QDoRA,r=16,α=32,target = all linear,batch 4 grad-accum 8,3 epoch,lr 2e-4 cosine
- **硬體**:單張 RTX 4090 24GB,約 4-5 小時
- **結果 (MT-Bench)**:
  - Base Mistral-7B-v0.3: ~6.85
  - + LoRA r=16(同資料):~7.10(+0.25)
  - + **DoRA r=16(同資料)**:~7.32(+0.47)
  - 同等 QLoRA r=64:~7.18

換言之 DoRA 在 r=16 就達到甚至超過 QLoRA r=64 的水準,**參數量少 4 倍、VRAM 少 30%**。同樣套路在 Llama-3-8B + 10K Tulu-3 子集上也能複現約 +0.3-0.5 MT-Bench 的提升。

---

## 小結:三句話帶走

1. **DoRA 是當前 CP 值最高的 LoRA 升級**:一個 flag、零推論延遲、平均 +1-3% 性能,搭 QLoRA 變 QDoRA 在消費卡就能玩。
2. **PiSSA 與 LoRA+ 是「免費的午餐」**:不改架構只改初始化或 lr 設定,training step 數可降 30-50%。
3. **rsLoRA / VeRA / AdaLoRA 是場景特化工具**:除非你有明確的高 rank、極端壓縮、或 budget allocation 需求,否則 vanilla LoRA + DoRA 就夠用。

下一步建議:回到 [`./進階微調策略_LoRA_QLoRA.md`](./進階微調策略_LoRA_QLoRA.md) 把基礎 QLoRA pipeline 跑通,再回來把 `use_dora=True` 開起來對比。

---

## 參考來源

- [DoRA: Weight-Decomposed Low-Rank Adaptation (arXiv 2402.09353)](https://arxiv.org/abs/2402.09353)
- [NVIDIA Blog: Introducing DoRA](https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/)
- [LoRA+: Efficient Low Rank Adaptation of Large Models (arXiv 2402.12354)](https://arxiv.org/abs/2402.12354)
- [PiSSA: Principal Singular Values and Singular Vectors Adaptation (arXiv 2404.02948)](https://arxiv.org/abs/2404.02948)
- [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (arXiv 2312.03732)](https://arxiv.org/abs/2312.03732)
- [VeRA: Vector-based Random Matrix Adaptation (arXiv 2310.11454)](https://arxiv.org/abs/2310.11454)
- [AdaLoRA: Adaptive Budget Allocation (arXiv 2303.10512)](https://arxiv.org/abs/2303.10512)
- [Answer.AI: Efficient finetuning of Llama 3 with FSDP QDoRA](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html)
- [Hugging Face PEFT LoRA Developer Guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
