# DPO / SimPO / ORPO 對比實驗 — 偏好優化算法的演進與選型

> 對應 [全景圖 #3](../../2024-2026_AI完整領域全景圖.md);姊妹篇 [`./GRPO_DAPO_RLVR_實戰.md`](./GRPO_DAPO_RLVR_實戰.md);速查 [`./DPO家族公式速查.md`](./DPO家族公式速查.md)
>
> ⚡ **想立刻動手?**配套 Colab notebook:[`notebooks/Colab_DPO_Alignment_Mini_Demo.ipynb`](./notebooks/Colab_DPO_Alignment_Mini_Demo.ipynb)
> — Colab T4 跑通 Qwen2.5-0.5B + LoRA-DPO + Ultrafeedback 800 對、15 分鐘看見 reward margin 上升;可承接 [Notebook 1 (SFT)](../../5.監督微調%20(SFT)/hands_on_project/notebooks/Colab_LoRA_SFT_Mini_Demo.ipynb) 的 adapter。
> 第 10 節列 7 條 phantom-mesh 真實工程考量(reward hacking 偵測、KL collapse、incremental DPO 等)。

---

## 1. 四種偏好優化算法概覽

自 2023 年 DPO 問世以來,偏好對齊家族迅速擴張。下表先給出五種主流算法的核心 loss,後續章節展開推導與工程細節。所有公式中 $\pi_\theta$ 為待訓練 policy,$\pi_{\text{ref}}$ 為參考模型,$(x, y_w, y_l)$ 為 prompt 與 chosen / rejected 回應對。

**DPO (Direct Preference Optimization, 2023)**:把 RLHF 的 reward maximization 重寫成可閉式求解的對比 loss,以 reference model 為 KL 錨點。
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**IPO (Identity Preference Optimization, 2023)**:將 sigmoid 換成平方損失,緩解 DPO 在偏好標籤接近確定時的 overfitting。
$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w|x) / \pi_{\text{ref}}(y_w|x)}{\pi_\theta(y_l|x) / \pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

**KTO (Kahneman-Tversky Optimization, 2024)**:基於 prospect theory,只需 thumbs-up/down 一元標籤。
$$\mathcal{L}_{\text{KTO}} = \mathbb{E}\left[w(y) \cdot \left(1 - \sigma(\beta z(x,y))\right)\right],\quad z(x,y) = \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - \text{KL}_{\text{ref}}$$

**ORPO (Odds Ratio Preference Optimization, 2024)**:把 SFT 與偏好優化合併成單一階段,以對數機率比作為偏好項。
$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) - \lambda \log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

**SimPO (Simple Preference Optimization, 2024)**:移除 reference model,並對 log-prob 做長度歸一化。
$$\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

---

## 2. DPO 詳解:從 RLHF 推導、Reference Model、β 參數

DPO 的關鍵洞見是:RLHF 的最佳 policy 對於 KL-constrained reward maximization 有閉式解 $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$。把這個關係代回 Bradley-Terry preference model,可以把 reward $r$ 消掉,得到只關於 $\pi_\theta$ 的 loss。等於把「訓 reward model → PPO」兩階段塌縮成「直接對 policy 做 contrastive loss」。

**Reference model 的角色**有兩個:(1) 作為 KL 錨點,防止 policy 在訓練中飄太遠;(2) 提供 implicit reward 的 baseline,讓 $\log(\pi_\theta / \pi_{\text{ref}})$ 等價於 reward。實務上 reference model 通常就是 SFT checkpoint,在訓練全程凍結。代價是顯存 ×2 與 forward latency ×2。

**β (beta) 參數**控制 KL constraint 強度:β 越大,policy 越被綁在 reference 附近;β 越小,policy 可以走得更遠。常見值 0.1 - 0.5,Llama 系列實證上 0.1 是甜蜜點。β 過小會引發 **KL collapse**——policy 把所有機率質量塌到 chosen 序列上,失去 diversity。

---

## 3. SimPO 改進:無 Reference + 長度歸一化

SimPO (Princeton NLP, NeurIPS 2024) 對 DPO 動了兩刀:**砍掉 reference model**,並在每個 log-prob 上除以序列長度。理由很直接——DPO 的 implicit reward 是序列 log-prob,而 sequence log-prob 天生隨長度單調遞減,導致模型偏好「短而漂亮」或「長而冗」的退化解。SimPO 改用 **average log-prob** $\frac{1}{|y|}\log\pi_\theta(y|x)$,讓 reward 與「生成時 beam search / sampling 的實際 score」一致。

額外引入 reward margin $\gamma$,要求 chosen 與 rejected 的平均 log-prob 差距至少 $\gamma$,等價於 hinge-like behaviour,壓縮 verbosity bias。

**為何在 AlpacaEval 2 上 +6.4 pt**:SimPO 論文在 Llama-3-8B-Instruct 與 Mistral-7B-Instruct 上同時報告了 AlpacaEval 2 length-controlled win rate 比 DPO 高 6.4 點、Arena-Hard 高 7.5 點;Gemma-2-9B-it-SimPO 達到 72.4% LC win rate。同時因為不跑 reference forward,訓練 wall-clock 約快 20%、peak GPU memory 省 10%。

---

## 4. ORPO 改進:把 SFT Loss 與 Preference Loss 合併

ORPO (KAIST, EMNLP 2024) 的賣點是 **單階段訓練**——不需要先做 SFT 再做 DPO,而是把兩者合在同一個 loss 裡。其 preference 項使用 **odds ratio** $\text{odds}_\theta(y|x) = \pi_\theta(y|x) / (1 - \pi_\theta(y|x))$,直觀是「相對於不生成,模型生成這個序列的傾向」。

工程上的吸引力:傳統 pipeline 是 base → SFT → DPO 三段,每段都要切資料、調 LR、調 epoch。ORPO 把 chosen 樣本當 SFT 訓,同時用 odds ratio 把 rejected 樣本往下壓,一遍走完。在 Phi-2 (2.7B) / Llama-2 (7B) / Mistral (7B) 上,單跑 ORPO + UltraFeedback 可以超過 Zephyr-7B-β(它需要 SFT + DPO 兩階段),AlpacaEval 2.0 達 12.20%、MT-Bench 7.32。

**注意點**:ORPO 假設 chosen 樣本品質夠高(因為它直接拿去做 SFT)。若 preference dataset 的 chosen 本身有噪音,ORPO 會比 DPO 更受傷。

---

## 5. KTO:一元 Feedback,適合線上 User Signal

KTO (Contextual AI, 2024) 來自一個現實困境:**成對偏好標註太貴**。實際 production 系統能拿到的多半是 thumbs-up / thumbs-down、retry button、conversation length 這種一元訊號。KTO 把每個樣本獨立標成「desirable」或「undesirable」,不需要兩兩配對。

理論基礎是 Kahneman-Tversky prospect theory——人對 gain 與 loss 的敏感度不對稱,KTO 在 loss 裡用兩個權重 $\lambda_D, \lambda_U$ 模擬這種不對稱性。論文展示在 1B 到 30B 規模上,KTO 用一元訊號可以匹配甚至超過 DPO 的偏好訊號表現。

**典型場景**:聊天機器人 production 階段,把使用者 thumbs-up 視為 desirable、thumbs-down 或 regenerate 視為 undesirable,直接 fine-tune,不必額外請標註員配對。

---

## 6. 對比實驗框架

要公平比較這幾種演算法,實驗設計必須鎖住以下變因:

| 維度 | 統一設定 |
|---|---|
| Base model | Llama 3.1 8B Instruct |
| 偏好資料 | UltraFeedback (binarized) 或 HelpSteer2 |
| 一元資料(KTO 用) | 把 UltraFeedback 的 chosen 標 desirable、rejected 標 undesirable |
| Optimizer | AdamW, LR 5e-7, cosine schedule |
| Batch size | 128 (global), 1 epoch |
| β | DPO/SimPO/KTO 用 0.1;SimPO 額外設 γ=1.0;ORPO 用 λ=0.1 |
| Eval | Arena-Hard、MT-Bench、AlpacaEval 2 LC win rate、IFEval |
| 硬體 | 8×H100 80GB,bf16,FSDP |

每個演算法各跑 3 個 seed,計算 mean ± std。Eval 用同一個 judge(GPT-4o 或 Llama-3-70B-Instruct)以避免 judge bias 干擾。

---

## 7. 可執行 Code 範例 (trl)

```python
# pip install trl>=0.12 transformers>=4.45 datasets accelerate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig, ORPOTrainer, ORPOConfig, KTOTrainer, KTOConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16")
ref   = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16")  # DPO 用
ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

# --- DPO ---
dpo_cfg = DPOConfig(output_dir="out/dpo", beta=0.1, learning_rate=5e-7,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    num_train_epochs=1, bf16=True, max_length=2048, loss_type="sigmoid")
DPOTrainer(model, ref_model=ref, args=dpo_cfg, train_dataset=ds, processing_class=tok).train()

# --- SimPO (在 trl 是 DPOTrainer 的 loss_type) ---
simpo_cfg = DPOConfig(output_dir="out/simpo", beta=2.0, learning_rate=5e-7,
                     per_device_train_batch_size=2, gradient_accumulation_steps=8,
                     num_train_epochs=1, bf16=True, max_length=2048,
                     loss_type="simpo", simpo_gamma=1.0)  # 無 ref_model
DPOTrainer(model, ref_model=None, args=simpo_cfg, train_dataset=ds, processing_class=tok).train()

# --- ORPO (單階段,直接吃 base model,不需 SFT) ---
orpo_cfg = ORPOConfig(output_dir="out/orpo", beta=0.1, learning_rate=8e-6,
                      per_device_train_batch_size=2, gradient_accumulation_steps=8,
                      num_train_epochs=1, bf16=True, max_length=2048)
ORPOTrainer(model, args=orpo_cfg, train_dataset=ds, processing_class=tok).train()

# --- KTO (一元 feedback) ---
# 需把資料轉成 {"prompt", "completion", "label": bool}
kto_ds = ds.map(lambda r: [{"prompt": r["prompt"], "completion": r["chosen"][-1]["content"], "label": True},
                           {"prompt": r["prompt"], "completion": r["rejected"][-1]["content"], "label": False}],
                batched=False).flatten()
kto_cfg = KTOConfig(output_dir="out/kto", beta=0.1, desirable_weight=1.0, undesirable_weight=1.0,
                    learning_rate=5e-7, per_device_train_batch_size=2, num_train_epochs=1, bf16=True)
KTOTrainer(model, ref_model=ref, args=kto_cfg, train_dataset=kto_ds, processing_class=tok).train()
```

---

## 8. 2025 Controlled Study 警告

ICLR 2025 與 ACL 2025 多篇 controlled study(包括 CMU、Stanford CS224N、Harvard NLP 的對比實驗)指出一個尷尬事實:**大多數 DPO 變體在統一資料、統一 base、統一 eval 的條件下,並未在統計顯著性 (p<0.05) 上真正勝過 vanilla DPO**。許多論文宣稱的提升,實際上落在跨 seed 的標準差範圍內,或來自 hyperparameter 偏向作者方法的調參優勢。

實務含義:不要因為「新 paper 數字更高」就盲目換演算法。先把 DPO 在自己的資料上跑穩(包括 LR、β、epoch 數的網格搜尋),再考慮 SimPO / ORPO / KTO 解決你**特定**的痛點(verbosity、無 SFT 預算、無配對標註)。

---

## 9. 選型建議

| 情境 | 推薦算法 | 理由 |
|---|---|---|
| 通用對齊、有配對偏好資料、有 SFT checkpoint | **DPO** 或 **SimPO** | 成熟、社群支援足;SimPO 省顯存且 verbosity 較低 |
| 標註預算極少,只能拿到 thumbs-up/down | **KTO** | 一元訊號直接可用;適合 production 線上回收 |
| 想跳過 SFT 階段、單卡訓練、快速迭代 | **ORPO** | 單階段、無 ref model,但要求 chosen 品質高 |
| 強推理任務 (math / code) | 不用 DPO 家族,改用 [GRPO/RLVR](./GRPO_DAPO_RLVR_實戰.md) | 偏好訊號太弱,需 rule-based reward |

---

## 10. 常見坑

- **KL collapse**:β 過小或 epoch 太多時,policy 把 mass 全塌到 chosen。監控 `rewards/margins` 與 reference KL,後者若 > 30 通常已失控。
- **Verbosity bias**:DPO / IPO 的 reward 是 sequence log-prob,長序列更易拿高 reward。SimPO 的長度歸一化、或 length-controlled AlpacaEval 2 都是緩解方向。
- **Preference dataset noise**:UltraFeedback 等公開資料的 chosen/rejected 來自 GPT-4 judge,本身有 ~15-20% 噪音。IPO 的平方損失對噪音較魯棒;DPO 在高噪音資料上會 over-confident。
- **Reference model 漂移**:如果你的 SFT model 本身就不夠好,DPO 會繼承這些缺陷。建議先評估 SFT 品質再做 DPO。
- **Eval judge bias**:不要只用 AlpacaEval 1 或 MT-Bench 單一指標;務必同時看 length-controlled 版本與 Arena-Hard。

---

## 參考資料

- DPO 原始論文:Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023), arxiv 2305.18290
- SimPO:Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward" (NeurIPS 2024), arxiv 2405.14734
- ORPO:Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" (EMNLP 2024), arxiv 2403.07691
- KTO:Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2024), arxiv 2402.01306
- Controlled study:"Evaluating DPO and its Variants Across Multiple Tasks" (ACL SRW 2025)
- trl docs: <https://huggingface.co/docs/trl>
