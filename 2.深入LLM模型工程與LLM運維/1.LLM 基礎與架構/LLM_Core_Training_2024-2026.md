# LLM 核心架構與訓練 深度分析 (2024-2026)

> 對應 [全景圖](../../2024-2026_AI完整領域全景圖.md) #3 + #4
> 從「單一 dense Transformer + RLHF」走向「MoE + 推理模型 + 混合架構 + 多階段後訓練」。

---

## 1. 架構演進:Dense → MoE → Hybrid → Diffusion

2024 仍以 Dense Transformer 為主流(Llama 3、Qwen2.5、Mistral 7B)。2024 Q4 起 Mixtral 8x7B/8x22B 證明 sparse MoE 在開源生態的可行性後,frontier 模型幾乎全面轉向 MoE。

**2025-2026 新突破**:
- **DeepSeek-V3 / V3.2**(2024/12):671B 總參數、37B 啟用,採 Multi-head Latent Attention (MLA) 與 DeepSeekMoE 架構,引入 **auxiliary-loss-free load balancing** 與 **Multi-Token Prediction (MTP)**,全程僅用 2.788M H800 GPU 小時
- **Llama 4 系列**(2025/04):Meta 首個 MoE 家族。Scout(17B active / 109B total / 16 experts / **10M token context** via iRoPE)、Maverick(17B active / 128 experts)、Behemoth(288B active / ~2T total)作為 codistillation teacher
- **Qwen3-MoE**(2025/05):Qwen3-235B-A22B 與 Qwen3-30B-A3B,128 experts / 8 activated,首次將 thinking / non-thinking 統一在單一 checkpoint
- **GPT-OSS**(2025/08):OpenAI 自 GPT-2 以來首次釋出開源權重,120B / 20B MoE,採 MXFP4 量化,120B 可單卡 80GB 跑、20B 可 16GB 邊緣部署
- **Hybrid (Mamba/SSM + Attention)**:Jamba (AI21)、Zamba (Zyphra)、**IBM Granite 4.0**(2025/11,Mamba-2 : Transformer = 9:1 + fine-grained MoE)— SSM 在企業推理場景顯著降低 KV cache 記憶體
- **Diffusion LLM**:Inception Labs 的 **Mercury Coder**(2025/06)達 1109 tokens/s,比同級 autoregressive 模型快 10x,LLaDA 學術路線首次商業驗證

**工程影響**:訓練 / 部署需懂 expert routing、load balance、MoE 分片(EP);推論需理解 active vs total param 的記憶體 / FLOPs 差異;hybrid 架構讓長上下文記憶體下降一個量級。

## 2. 推理模型 (Reasoning Models):Test-time Compute 革命

2024/09 OpenAI o1 開啟「長 CoT + RL on reasoning traces」典範。2025 全面進入 thinking 模式內建期。

**2025-2026 突破**:
- **OpenAI o3 / o4-mini**(2025/04):reasoning effort 三檔(low/medium/high),首次讓 reasoning 模型 agentic 呼叫 web、Python、image。o4-mini 在 AIME 2025 高 reasoning effort 配置下接近 99% pass(具體數字需以 OpenAI 官方 evaluation 配置為準)
- **DeepSeek-R1 / R1-Zero**(2025/01):以 V3-Base 為起點,**純 RL(GRPO)無需 SFT** 即誘發 reasoning(R1-Zero);R1 加入冷啟動 SFT + 多階段 RL,開源衝擊 frontier 閉源模型
- **Qwen-QwQ / Qwen3 thinking mode**:thinking budget 可由用戶在 chat template 中動態切換
- **Grok 4 / 4.1** (xAI)
- **Gemini 2.5 Pro / Gemini 3 Deep Think**:在 Humanity's Last Exam 從 37.5% 提升到 41%
- **Claude extended thinking** (4.5 / 4.7):可調 thinking budget

**技術路徑**:
1. RL on verifiable rewards (RLVR)
2. **PRM (Process Reward Model)** 對中間步驟打分
3. test-time compute scaling — 延長 CoT、majority voting、tree search

2026「reasoning」已從獨立模型線併入旗艦。

**工程影響**:API 設計需暴露 thinking budget / reasoning_effort;成本模型從 input/output token 變成 input/output/reasoning token 三段計費。

## 3. 後訓練 (Post-training) 範式:從 RLHF 到 GRPO/DAPO

經典三段 SFT → RM → PPO 已被解構為模組化 stack。

**2025-2026 突破**:
- **DPO** (2023) → **IPO / KTO / ORPO / SimPO**:DPO 拿掉 RM,IPO 修偏好過擬合,**KTO** 用 thumbs-up/down 一元 feedback,**ORPO** 把 SFT 與 preference 合併(無需 reference model),**SimPO** 拿掉 reference log-ratio(AlpacaEval 2 比 DPO 高 6.4 點)
- **GRPO** (DeepSeek, 2024):取消 critic / value network,用 group sampling 估 advantage,顯著降低記憶體;R1 系列、Qwen3 reasoning 標配
- **DAPO / RLOO / Dr. GRPO**:2025 一連串 GRPO 後繼,修正 length bias、token-level clipping
- **Constitutional AI / RLAIF**:Anthropic 路線在 Claude 4.x 系列繼續演進
- **Self-play / Self-improvement / Weak-to-strong**

**Caveat**:CMU/Stanford/Harvard/Princeton 2025 controlled study 指出大多數 DPO 變體在統計顯著性上未真正勝過 vanilla DPO。

**工程影響**:post-training pipeline 需設計 **SFT → preference (DPO/SimPO) → RLVR (GRPO)** 三層;reward signal 從人類偏好轉向 verifier(unit test、math checker、tool execution)。

## 4. 預訓練:Scaling Laws 演進與資料品質

Chinchilla(2022)的 20:1 token/param 比率已被打破——Llama 3 8B 訓了 ~1875:1,遠超 compute-optimal,為的是 **inference-optimal**。

**2025-2026**:
- **Over-training past Chinchilla**:成為 sub-10B 模型標配,但 over-trained 模型對量化更脆弱
- **Synthetic data 全流程化**:Phi-4(GPT-4o 生成的 high-quality synthetic)、DeepSeek-V3(reasoning chain 合成)、Nemotron 3 Super、Llama 4(Behemoth codistillation)
- **WSD (Warmup-Stable-Decay) schedule** 取代 cosine
- **Quality-aware scaling laws**:將 data quality 作為第三軸納入 loss prediction
- **資料牆**:人類網頁文字耗盡,轉向視訊轉錄、合成 + 真實混合、許可資料採購

**工程影響**:「訓更多 token」≠「更好」;data pipeline 的 dedup、quality filter、synthetic generation 比模型架構調整更影響最終效果。

## 5. 長上下文 (Long Context):從 128K 到 10M

2024 年 128K 是 frontier 基準。2025-2026 進入 1M-10M 區間。

**2025-2026 突破**:
- **RoPE 家族**:Position Interpolation → NTK-aware → **YaRN**(10x 少 token、2.5x 少步數)→ **LongRoPE**(2M+)→ **iRoPE**(Llama 4 Scout,交錯 RoPE 與 no-PE 層,從 256K 訓練外推到 10M)
- **Gemini 2.5 / 3 Pro**:原生 1M-2M context;**Grok 4 fast** 2M
- **Native Sparse Attention (NSA)** (DeepSeek, 2025/02):hierarchical compression + selective retention + sliding window 三路並行,FlashAttention-2 forward 9x / backward 6x 加速
- **MoBA (Mixture of Block Attention)** (Moonshot):MoE 思想套到 attention block routing
- **DeepSeek V3.2 Sparse Attention (DSA)**:較 NSA KV-cache 再降 50%
- **Ring / Striped Attention**:分散式長 context 訓練的事實標準
- **KV cache 壓縮**:H2O、SnapKV、StreamingLLM、PagedAttention 已是部署層必備

**工程影響**:1M+ context 改變 RAG 邊界——「全文塞進去」變得可行;但 needle-in-haystack 表現與標稱 context 仍有落差,須以「effective context」評估。

## 6. 模型壓縮:量化、蒸餾、剪枝

- **GPTQ**(layer-wise + inverse Hessian)、**AWQ**(activation-aware, W4A16):開源生態標配
- **GGUF**(llama.cpp / Ollama):CPU+GPU hybrid,Q2_K-Q8_0 細粒度
- **FP8 / FP4 / MXFP4**:H100/H200/B100 硬體加速;GPT-OSS 直接以 MXFP4 訓 MoE weights
- **SmoothQuant**:解決 activation outlier,達成 W8A8
- **NVFP4 / AutoRound**:NVIDIA Blackwell 世代 FP4 已成資料中心新預設
- **Codistillation**:Llama 4 Behemoth → Scout/Maverick 大規模教師蒸餾

**選型 = 部署目標**:GPU 推論(AWQ + vLLM)、邊緣(GGUF + llama.cpp)、DC 大規模(FP8/FP4 + TensorRT-LLM)。

## 7. 小模型趨勢:Edge LLM 民主化

sub-10B 模型在 2026 已超越 2024 年 GPT-4 在多數 benchmark:
- **Phi-4 / Phi-4-mini / Phi-4-multimodal** (Microsoft):14B 在 GSM8K 93.7%、MATH 73.5%
- **Gemma 3** (Google, 2025):128K context、140+ 語言、強 tool-use
- **Qwen3-0.6B/1.7B/4B/8B**:含 thinking mode,4B 已可做 agentic tool calling
- **Llama 3.2 1B/3B**:Meta 邊緣 / mobile 主力
- **SmolLM-3** (Hugging Face):3B 超越 Llama-3.2-3B / Qwen2.5-3B
- **GPT-OSS-20B**:16GB RAM 可跑,達 o3-mini 級

**工程影響**:on-device agent、tool-calling 在 1B-4B 等級已可商用;私有資料、低延遲、零雲端費用場景全面可行。

## 8. 2025-2026 必知里程碑

| 時間 | 模型 | 重要性 |
|---|---|---|
| 2024-12 | **DeepSeek-V3** | 671B MoE / MLA / MTP |
| 2025-01 | **DeepSeek-R1** | 純 RL + GRPO 誘發 reasoning |
| 2025-02 | **DeepSeek NSA** | 原生稀疏 attention,9x 加速 |
| 2025-04 | **Llama 4 (Scout/Maverick/Behemoth)** | Meta 首個 MoE、10M context、iRoPE |
| 2025-04 | **OpenAI o3 / o4-mini** | Reasoning agentic tool use |
| 2025-05 | **Qwen3 系列** | thinking/non-thinking 統一 |
| 2025-06 | **Mercury (Inception Labs)** | 商業級 diffusion LLM,1109 tok/s |
| 2025-08 | **GPT-OSS-120B/20B** | OpenAI 首次開源權重,MXFP4 MoE |
| 2025-11 | **IBM Granite 4.0** | Hybrid Mamba-2/Transformer + MoE |
| 2026-04 | **GPT-5.5 / Claude Opus 4.7 / Gemini 3.1 Pro** | reasoning 內建旗艦、2M+ context、GPQA Diamond 94%+ |

---

## 2026 LLM 工程師必修地圖

要在 2026 年勝任 LLM 工程師,至少需精通以下五層:

1. **架構層**:理解 dense / MoE / hybrid (Mamba+Attention) / diffusion 四條路線的 trade-off。具體:能解釋 MLA vs GQA、auxiliary-loss-free routing、iRoPE 為何能外推到 10M
2. **訓練層**:掌握 SFT → DPO/SimPO/ORPO → GRPO/DAPO 三段 pipeline;理解 RLVR 為何超越 RLHF;會用 PRM 與 verifier 設計獎勵
3. **推理層**:CoT、test-time compute scaling、thinking budget;能根據任務切換 fast vs deep think
4. **長上下文層**:RoPE/YaRN/LongRoPE/iRoPE 的差別;NSA/MoBA 等稀疏 attention;PagedAttention 與 KV cache 經濟學
5. **部署層**:量化選型(AWQ/GPTQ/GGUF/FP8/MXFP4)、推論引擎(vLLM、TensorRT-LLM、SGLang、llama.cpp)、edge vs cloud trade-off、cost-per-reasoning-token 預估

**核心心法**:2024 年的「更大就是更好」已死。2026 的競爭力在於 **post-training 工藝**(reward / data)、**test-time compute 編排** 與 **資料品質工程**——架構僅佔 20%,剩下 80% 在資料與訓練配方。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
