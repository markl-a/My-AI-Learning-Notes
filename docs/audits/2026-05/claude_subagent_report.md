# AI 學習筆記 Repo 2026 Frontier 對齊度審計報告

**範圍**:`D:\Projects\My-AI-Learning-Notes`(主分支 `main`,2026-05-16)
**重點**:11-22 章 deep-dive、5 個 Colab notebook、CONCEPT_MAP / FRONTIER_TERMS_INDEX、兩輪自我審計
**對齊基準**:ICLR 2026 outstanding / NeurIPS 2025 best papers、各 frontier lab 2026 Q1-Q2 release、Epoch AI / METR 量化追蹤、Sierra τ²-bench、Apple PCC / NVIDIA Blackwell 等已落地的 2026 H1 事實。

---

## 一、整體結論:結構正確,但「鮮度 ≠ 真實度」

11-22 章已經涵蓋了 2024-2026 最重要的 12 條主敘事(Hardware、AI4Science、Robotics、Voice、Privacy、Provenance、Causal、GNN、Synthetic Data、Generative UI、Forecasting、Self-Improving),這個拓撲層級對個人學習者來說已經是繁中筆記裡 top-tier 的覆蓋面。但「結構正確」與「內容真實」是兩件事——本輪審計後我認為這份 repo 目前的**最大風險不是缺章節,而是內容過度 hype 化、被讀者一次性識破信用即破產**(這也呼應第一輪審計的致命問題 #1 / #3)。

下面分五個面向給觀察。

---

## 二、五面向審查

### 1. 內容鮮度差距:22 個方向尚未涵蓋的重要主題

22 章已經很完整,但對照 2026 H1 真實 AI 領域,**至少有四塊系統性缺漏**:

- **(a) Mechanistic Interpretability / AI Safety 子領域** —— MIT Tech Review 把 mechanistic interpretability 列為 2026 Breakthrough Technology,Anthropic 的 SAE / circuit tracing / attribution graph、DeepMind 的 pragmatic interpretability、ICLR 2026 機制可解釋性主題占比明顯上升。Repo 內 `2.深入LLM模型工程與LLM運維/10.進階話題/AI安全與對齊指南.md` 雖然有但散在,**沒有專章對「模型內部到底發生了什麼」做 deep-dive**。22 章寫 RSI,但 22 章假設讀者已經懂 alignment evaluation,這個假設目前不成立。
- **(b) Agent Evaluation / 長 horizon benchmark** —— 2025-2026 最熱的事實上是 Sierra τ²-bench / τ³-bench、SWE-Bench Verified、OSWorld、GAIA、METR HCAST、Berkeley Function Calling Leaderboard、`pass^k` 可靠度指標。Repo 內 `2.深入LLM模型工程與LLM運維/9.模型評估` 完全沒有覆蓋這層,21 章 forecasting 提到 METR 但不教如何跑 eval。**「Agent 工程師最高薪的能力就是設計能 hold-out 的 eval」,這條沒章節等於斷層**。
- **(c) Frontier RL / Self-Play / RLVR 訓練技術獨立章** —— GRPO、DAPO、RLVR、process reward model、DPO/IPO/KTO/SimPO、R1-Zero 風格 RL 在 `LLM_Core_Training_2024-2026.md §3` 只是一小節。NeurIPS 2025 best paper runner-up 直接質疑 RLVR 能否突破 base model 上限,這是 2026 frontier 真正在吵的命題,**值得獨立一章**。
- **(d) Energy / Power-Constrained AI** —— 11 章硬體有講電力但停在「核電 PPA」層級。Epoch 預測 frontier 訓練功率每年翻倍、Omdia 報告「frontier 模型成長放緩、small model 反而 scale up」、2026 已是「inference 占 lifecycle energy 63%」的反轉時點。這條經濟現實沒被整成決策框架。

另外比較邊緣但值得追蹤:**Spatial Computing (Apple Vision Pro / Meta Quest LLM)、Neurosymbolic AI 復興、On-Device Agent (Apple Intelligence、Snapdragon NPU agent)、AI-Native Database (Mooncake / Liquid)** —— 這四個 2026 H2 才會明朗,可以放 backlog。

### 2. Frontier 章節品質:hype 風險具體清單

我抽查 11-22 章後,**信任度風險高的具體段落**如下(這些就是兩輪自我審計沒逐一核實的):

| 位置 | 可疑陳述 | 風險 |
|---|---|---|
| 12 AI4Science §1 | 「RFdiffusion3 ... 傳聞中釋出,2026-05 待官方確認」 | 自己寫了「傳聞」還是寫進來,讀者掃過會誤認為事實 |
| 13 Robotics §1 | 「π0.6 (2025/11) 進一步強化 long-horizon」 | π0.6 公開資訊很少,版本號可能來自 agent 推測 |
| 13 §2 | 「Figure 03 已交付 350+」「BotQ 每小時 1 台」 | 具體數字未引用 |
| 14 Voice §10 | 「2026.03 ElevenLabs v3 GA」「Suno v5.5 8 分鐘」 | 寫死月份,實際 ElevenLabs v3 早於此 |
| 15 Privacy §3 | 「Apple 自研 AI server 晶片預計 2026H2 量產」 | 未發布,屬媒體推測 |
| 18 GNN §6 | UltraGCN「NDCG@20 +4% 至 +20%」 | 第一輪已抓到,新版改成「跨 dataset」但仍是孤立記憶,讀者無法驗證 |
| 20 Generative UI §5 | 「A2UI v0.9 / 2025/12」(已自加 ⚠️ disclaimer) | 已標 disclaimer 是模範,**其他章節該全面對齊這個寫法** |
| 21 Forecasting §1 | 「Gemini 3 Pro ECI 居首」「FrontierMath 38%」 | Epoch ECI 數字漂移快,寫死有風險 |
| 21 §4 | Anthropic「2026 中可能超越 OpenAI ARR」 | 第一輪已點名,本輪改為條件式但仍是「軟事實當硬事實」 |
| 22 Self-Improving §3 | Sakana v2 ICLR workshop 「6/7/6」 | 該事件存在但分數需 source-check |

**結構性的 hype 模式**:11-22 章每篇都有「2025-2026 重大進展」「2025-2026 重大里程碑」段落,**清一色把媒體報導 / vendor blog / agent 推測混為「事實清單」**。即使加了 References & Sources 免責聲明,實質上是「整節都需要被引用 source」卻只在末尾說「以官方為準」。這個免責句完全不解決問題,因為讀者讀正文時不會主動懷疑。

**修法建議**(不必補連結,但內容語氣要改):
1. 把「具體數字 / 具體月份 / 具體產品版本號」做三類標註:`[Confirmed]`(有公開 source)/ `[Reported]`(媒體報導未官方確認)/ `[Speculative]`(agent / 業內傳言)。
2. 21 章與 22 章兩個最受 hype 影響的章節,**全文掃一次把 ARR、ASL 級別、AGI 時程、Anthropic vs OpenAI 比較這些段落標 `[Reported]` 或改成條件句**。
3. 11、13、14 章把帶月份的「重大里程碑」清單拆成「已發布(可驗證)」與「規劃 / 媒體預告」兩塊。

### 3. 理論-實作銜接:5 個 notebook 之外的缺口

現有 5 份:`Colab_LoRA_SFT_Mini_Demo`、`Colab_DPO_Alignment_Mini_Demo`、`Colab_vLLM_Deploy_PrefixCache_Demo`、`Colab_MiniGraphRAG_Hands_On`、`Colab_LangGraph_Multi_Agent_Research_Demo`。這 5 條軸線覆蓋了「**訓練 → 對齊 → 部署 → RAG → Agent**」基本 LLM 工程鏈,優先級正確。

但對照 22 個 frontier 章節,**至少還缺以下可跑 notebook**(按 ROI 排序):

1. **GRPO / RLVR Mini Demo**(對應 22 章 + LLM_Core_Training §3)—— 用 TRL / verl + Qwen2.5-0.5B + 數學題自驗證,單張 T4 半小時跑完。**目前 repo 對 R1 路線完全沒有可執行物**,這是 2025-2026 最該補的洞,優先級高於下面任何一條。
2. **Agent Evaluation Notebook(τ²-bench 或 SWE-Bench Mini)** —— 真正讓讀者體驗「為什麼 agent 不可靠」,銜接面試準備章。
3. **Computer Use Agent Mini**(對應 13 章末段 + 20 章)—— Anthropic Computer Use API + Playwright,或 open-source 的 BrowserUse / Skyvern。教 action tokenization 直覺。
4. **Voice Agent end-to-end**(14 章)—— Pipecat 或 LiveKit Agents + Deepgram + Cartesia + Twilio。教 latency budget 與 turn-taking。
5. **AI4Science Hello-World**(12 章)—— ColabFold + Boltz-1 跑一個 PDB,或 Evo 2 / MACE-MP-0 在 Colab。
6. **Multi-tenant LoRA Serving**(對應部署章)—— LoRAX / Punica 在單張 GPU 上服務 10+ LoRA,Voice/Tabular agent 量產的關鍵。
7. **Causal Inference Mini**(17 章)—— DoWhy + EconML + Criteo Uplift,30 分鐘可跑完。
8. **Synthetic Data Pipeline**(19 章已有 `Magpie_distilabel_實作.md` 文字版,但**沒有 .ipynb**)。
9. **Mechanistic Interpretability Mini**(新主題)—— TransformerLens + SAE on GPT-2,看 induction head。

**建議**:GRPO + Agent Eval + Computer Use 三條是 2026 H2 必補,其他可分批。

### 4. 真實案例 vs 教科書:phantom-mesh 連結確實偏弱

Repo 自稱「整合 phantom-mesh 真實案例」,但實測 `案例` / `Case_0X` / `phantom-mesh` 出現位置主要在頂層導讀與面試章,**11-22 章 frontier deep-dive 與 phantom-mesh 的 PR / commit / repo 連結幾乎為零**。讀者讀到「Anthropic Computer Use 在 OSWorld 屠榜」時,無法跳到一個「我們在 phantom-mesh `agents/computer_use.py` 怎麼用」的具體連結。這個落差和第一輪審計 #3「核心模組是空殼」是同一個病根的不同切面:**文檔強、實作弱**。

可行的補法:
- 為 11-22 章每章加一個固定的「**phantom-mesh 對應實作**」尾段(可以是空 placeholder,但至少把「哪個目錄會放實作」寫死)。
- 在 20 章 Generative UI、14 章 Voice、13 章 Robotics、22 章 Self-Improving 這四個最容易動手的方向,**直接 link 到 phantom-mesh 的 issue / draft PR**,即使是「TODO」也比沒有好。

### 5. 2027 預測:本 repo 在 18 個月內最該補的軸線

從 ICLR 2026 / NeurIPS 2025 / 2026 Q1-Q2 frontier release / Epoch / METR / Anthropic RSP v3 綜合判斷,**2026 H2 到 2027 H1 的 frontier 主敘事會是這四條**:

1. **長 horizon Agent 可靠性(τ²-bench、`pass^k`、policy adherence、agentic memory)** —— ICLR 2026 oral 一半在做這個。
2. **Mechanistic Interpretability 工程化(SAE、attribution graph、circuit tracing 進 production)** —— Anthropic 已經喊出口號。
3. **能源 / 電力 / inference 經濟學(從 capex 轉 opex、small model + test-time compute 的混合 stack)** —— Omdia 的「frontier 放緩、small model scale up」是新訊號。
4. **AI for AI Engineering(AlphaEvolve 風格的 kernel / 編譯器 / 訓練 pipeline 自動優化)** —— DeepMind 公開 AlphaEvolve 在 Google 內部跑了 1 年、回收 0.7% 全球 compute,這條軸線會在 2026-2027 從 lab 蔓延到開源社群。

下面把這 4 條轉成具體可建檔的 spec。

---

## 3-5 個 2026-2027 必補主題(可直接放進 repo)

```markdown
# 2026-2027 必補主題清單

## 1. `23.Agent_Reliability_and_Evaluation/README.md`
**為什麼必補**:2026 ICLR oral 一半在做 long-horizon agent;τ²-bench / τ³-bench / `pass^k` 是 2026 H2 起企業評估 agent 的事實標準。Repo 目前 9.模型評估與 3.LLM應用工程/3.Agent 都沒覆蓋這層。
**章節大綱**:
1. Agent eval 為何難於 LLM eval(stochastic、stateful、tool-side noise)
2. 六大基準對位:GAIA / SWE-Bench Verified / OSWorld / τ²-bench / WebArena / METR HCAST
3. `pass^k` 與 reliability metric(取代 single-shot accuracy)
4. Policy Adherence、Tool Use Correctness、Plan Validity 三層
5. Self-play / Adversarial user simulator
6. 企業 internal eval harness 設計(golden trace、replay、shadow mode)
7. 對應 phantom-mesh `evals/` 目錄結構建議
8. **可跑 notebook**:`Colab_Tau2Bench_Mini_Demo.ipynb` 用 LangGraph agent + 簡化 retail domain

## 2. `24.Mechanistic_Interpretability/README.md`
**為什麼必補**:MIT Tech Review 2026 Breakthrough Technology;Anthropic 把 attribution graph 開源、ICLR 2026 機制可解釋性論文倍增;這條是 22 章 RSI / AI 安全的「真實技術底座」,沒有它,21、22 章就是純科幻。
**章節大綱**:
1. Why MI:從 black-box probing 到 circuit-level 因果
2. Sparse Autoencoder (SAE) 入門:dictionary learning、superposition、feature splitting
3. Activation Patching、Path Patching、Attribution Graph 三大工具
4. TransformerLens + nnsight + SAE Lens 工具棧
5. Anthropic / OpenAI / DeepMind 三家路線對比(bottom-up vs pragmatic)
6. 與 alignment / red-team / refusal 機制的銜接
7. **可跑 notebook**:`Colab_TransformerLens_SAE_GPT2.ipynb`(看 induction head + 1 個 SAE feature)
8. 與 17.Causal_ML 與 22.Self-Improving 的交叉引用

## 3. `25.Inference_Economics_and_Energy/README.md`(或併入 11 章作為新 §13-15)
**為什麼必補**:2026 H1 Omdia / Epoch / Brookings 一致訊號是「能源變天花板、inference 反超 training」。11 章硬體側已寫但停在硬體選型,沒寫「為什麼 small + test-time compute hybrid 會贏」、沒寫「electricity-aware scheduling」、沒寫「inference carbon accounting」。21 章經濟學寫了 ARR 但沒寫 GWh。
**章節大綱**:
1. 2026 反轉點:inference 占 lifecycle energy 63%
2. Frontier 訓練功率曲線(MW → GW,Epoch 數據)
3. Small model + test-time compute vs large model + single-shot(成本曲線交叉)
4. Power-aware scheduling、grid-aware datacenter、carbon-aware inference
5. Inference market structure:Together / Fireworks / Anyscale / Modal / Replicate / Cerebras Cloud
6. KV cache disaggregation、prefill-decode 分離、CXL pool(11 章 CXL_Disaggregated 的決策層)
7. 對中小型團隊的成本選型決策樹(self-host vs API vs serverless)

## 4. `26.AI_for_AI_Engineering/README.md`(自動化 AI 工程,擴展 22 章)
**為什麼必補**:22 章 Self-Improving 講的是「AI 寫論文 / RSI 風險」,屬研究敘事;但 AlphaEvolve 在 Google 內部跑 1 年、回收 0.7% compute、Gemini kernel +23%,**這是「AI 寫 AI infra」的工程敘事**,讀者更需要這個務實版本。
**章節大綱**:
1. AI for kernel:AlphaEvolve / Sakana CycleNet / KernelBench
2. AI for compiler:Triton / TVM 自動調優、PyTorch torch.compile 自學 schedule
3. AI for distributed training scheduler(自動 parallelism 切分)
4. AI for data curation(NeMo Data Designer、distilabel 自動化 pipeline 設計)
5. AI for hyperparameter / architecture search(在 LLM 時代如何重新激活 NAS / HPO)
6. AI for code review / regression bisect / flaky test triage
7. 為什麼這條軸線可能比 RSI 更先到產業:**有客觀 reward signal**

## 5. (選配)`27.Frontier_RLVR_and_Self_Play/README.md`
**為什麼必補**:GRPO / DAPO / RLVR / process reward model / R1-Zero 風格 RL 是 2025-2026 reasoning model 主流路線。Repo 內 `LLM_Core_Training_2024-2026.md §3` 帶過,但 NeurIPS 2025 best paper runner-up(RLVR 是否能突破 base model)是 frontier 真正在吵的命題,值得獨立。
**章節大綱**:
1. PPO → GRPO → DAPO 演進譜系
2. Verifiable Reward 設計:數學、程式、形式化證明、單元測試
3. Process Reward Model vs Outcome Reward Model
4. R1-Zero 路線:跳過 SFT 的可行性與失敗模式
5. RLVR 的邊界爭議(NeurIPS 2025 runner-up 那篇 paper 的論點與反駁)
6. 開源實作:TRL、verl、OpenRLHF、Levanter
7. **可跑 notebook**:`Colab_GRPO_Math_Mini_Demo.ipynb`(Qwen2.5-0.5B + GSM8K + verl,單 T4 半小時)
```

---

## 三、給作者的三點直接建議

1. **rebrand 「Frontier briefing」為「Frontier dossier with uncertainty markers」**。把 11-22 章每節的具體數字 / 月份 / 產品版本掛 `[Confirmed]` / `[Reported]` / `[Speculative]` 三色標。讀者一眼能分辨,信用立刻回升。
2. **23/24/25 三章先補,26/27 可以等**。Agent Eval、Mech Interp、Inference Economics 是 2026 H2 至 2027 H1 不會退潮的主敘事,先佔位即使內容只有 50%。
3. **每章末段加 phantom-mesh placeholder**。即使是「TODO: 對應實作位於 phantom-mesh `xxx/`」也比沒有好,把「文檔強實作弱」的結構性問題顯式化,後續 PR 才有對標。

---

**Sources:**
- [Announcing the ICLR 2026 Outstanding Papers](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)
- [Announcing the NeurIPS 2025 Best Paper Awards](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- [How much power will frontier AI training demand in 2030? — Epoch AI](https://epoch.ai/blog/power-demands-of-frontier-ai-training)
- [Omdia: Frontier AI model growth slows as small models scale up](https://omdia.tech.informa.com/pr/2026/apr/frontier-ai-model-growth-slows-as-small-models-scale-up-and-reshape-infrastructure-demand)
- [τ-Bench / τ²-Bench: Benchmarking AI agents for the real world — Sierra](https://sierra.ai/blog/benchmarking-ai-agents)
- [AI Model Release Timeline 2025-2026 — AI Flash Report](https://aiflashreport.com/model-releases.html)
- [Mechanistic Interpretability Explained: Circuits, SAE, Causal Tracing (2026)](https://medium.com/@adnanmasood/mechanistic-interpretability-explained-circuits-sparse-autoencoders-causal-tracing-and-ai-88ecc8d70b72)
- [Transformer Circuits Thread — Anthropic Interpretability](https://transformer-circuits.pub/)
