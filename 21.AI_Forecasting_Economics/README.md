# 21. AI 能力預測學 / AI 經濟學 / 勞動衝擊 (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #23
> 量化 AI 進步速度、預測 capability milestone、估算 R&D 自動化與就業衝擊的學科。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **Frontier 模型史與基本術語**(GPT-3 → GPT-5、Claude 1 → 4.7、Gemini、DeepSeek-V3/R1 — 對應 repo:[2.深入LLM模型工程與LLM運維/1.LLM 基礎與架構](../2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/README.md))
> 2. **Benchmark 概念**(MMLU、GPQA、SWE-bench、FrontierMath — 對應 repo:[2.深入LLM模型工程與LLM運維/9.模型評估 (Evaluation)](<../2.深入LLM模型工程與LLM運維/9.模型評估 (Evaluation)/>))
> 3. **基本對數刻度直覺**(scaling laws、compute = 6ND、log-linear plot — 若不熟,先看 [全景圖 #23](../2024-2026_AI完整領域全景圖.md))
>
> 本檔幾乎不講「如何訓練模型」,而是「如何 *測量* 模型進步速度」,工程實作讀者可選讀。
>
> **延伸 / 反向連結**:[AI安全與對齊指南](../2.深入LLM模型工程與LLM運維/10.進階話題/AI安全與對齊指南.md) | [22.Self_Improving_AI](../22.Self_Improving_AI/README.md)

---

## 1. AI 能力預測機構地圖

### Epoch AI — 量化派的標竿
**Epoch Capabilities Index (ECI)** 將 40+ 個 benchmark 拼接成單一「通用能力」標度。關鍵數據:ECI 在 2024/04 後出現 **90% 加速**,目前最佳模型能力提升約 **15.5 點/年**,遠超 2024 前的 8 點/年。**Frontier Model Database** 收錄 2800+ 個模型,量測訓練算力 2010-2024 年以 **4.1×/年 (90% CI: 3.7-4.6×)** 增長。Gemini 3 Pro 目前 ECI 居首(FrontierMath Tier 1-3 達 38%、GPQA Diamond 93%)。

**關鍵詞**:ECI、4.1x/year compute scaling、frontier database、1e26 FLOP 門檻 (2025-11 跨越)

### METR — 任務時長派
2025/03「Time Horizons」論文:以「人類專家完成同任務的中位數時間」測量 AI 能力。**HCAST + RE-Bench + SWAA** 套件覆蓋秒級到 8 小時級任務。原始發現:frontier 模型 time horizon **每 7 個月翻倍** (2019-2025)。

**2026/01 Time Horizon 1.1**:後 2023 時期 doubling time 修正為 **131 天**(~4.3 個月),比 v1 快 20%;長任務數據點(≥8 小時)從 14 倍增到 31。

**2026/02 Simpler Timelines Model**(Thomas Kwa):用 8 參數預測 99% AI R&D 自動化中位數 **2032 年末**,敏感度測試顯示 β=0.9 推遲到 2034。

### Metaculus / Manifold / Open Philanthropy
Metaculus「強 AGI 公開宣布」中位數 2024-2025 在 **2031/07** 附近,2026/04 一度推遲到 **2033/11**,但 2026 Q1 後更新的預測者全部往前壓縮——典型的「越接近越分歧」現象。

## 2. 重要預測文檔

### AI 2027 (2025-04, AI Futures Project)
Daniel Kokotajlo (前 OpenAI)、Eli Lifland、Thomas Larsen、Romeo Dean 撰寫,Scott Alexander 編輯。**逐月情境**:2025 內部 agent → 2026 coding agent 全面化 → **2027/03 超人類研究員** → 2027 年底完全自動化 OpenBrain R&D → 2028 superintelligence。Kokotajlo 2021 年「What 2026 Looks Like」追溯命中率極高。

### Situational Awareness (2024-06, Leopold Aschenbrenner)
165 頁,三個斷言:**AGI 約 2027**、**trillion-dollar cluster 必然到來**、美國需國家級動員。已被 Stargate 部分驗證。Aschenbrenner 創投 Situational Awareness LP($225M → $5.5B)。

### Anthropic Core Views / RSP v3 (2025-10)
取消預先定義 ASL-4/ASL-5,改採「**Capability Thresholds**」對映。Anthropic 已啟動 **ASL-3** 保護(model weight 防外洩、limited deploy)。

## 3. AGI 時程辯論光譜

- **Dario Amodei (Anthropic)**:2026-2027 達 "Nobel-laureate level",主張 coding+AI research feedback loop 加速
- **Sam Altman (OpenAI)**:已宣稱「正滑過 AGI 邊界邁向 superintelligence」
- **Demis Hassabis (DeepMind)**:5-10 年達真 AGI,認為「**還需 1-2 個 breakthrough**」
- **Yann LeCun (Meta)**:稱「general intelligence 是 BS」、力推 JEPA 架構替代 LLM
- **Geoffrey Hinton**:10-20% 滅絕風險、20 年內 AI 全面超人
- **Yoshua Bengio**:5 年內可達工程師等級,堅持「**多重未來、避免大斷言**」

**量化分歧軸**:
- 「現有 transformer + RL 是否足夠」(Amodei/Altman 是、Hassabis/LeCun 否)
- 「滅絕風險 P(doom)」(Hinton 10-20% vs LeCun ~0%)

## 4. AI Compute 經濟學

### 訓練成本曲線
- GPT-3 (2020): **~$4.6M**
- GPT-4 (2023): **$78-100M**
- GPT-4.5 / GPT-5:單次訓練估 **$500M**,總研發 ~$1B
- **Stargate ($500B, 10 GW)**:Abilene 已上線、Oracle 4.5 GW 擴張、5 新站宣布、Argentina $25B/500 MW、Michigan $7B 2026 動工

### Inference 已過 training
**RL 推理 heavy**(每題大量 rollouts);依 SemiAnalysis 等 2025 估算,hyperscaler 推理 capex 已逼近或開始超過訓練 capex(各家統計口徑不一);o1→o3 透過 **10× RL training compute** 把 AIME 從 83% 推到 96%+。

### 收入結構
- **OpenAI**:2025/07 ARR **$12B**,2025 年底 **$25B**;Fortune 報導 2028 營業虧損將達當年收入 ~3/4
- **Anthropic**:2025/07 ARR $4B → 年底 $19B → 2026 初 ~$30B run rate;依 Epoch 推估與多家媒體報導,Anthropic 年化營收增速快過 OpenAI,2026 中可能超越(雙方 GAAP 財報未公開)
- **Mary Meeker 2025 AI Trends**:推論成本兩年內下降 **99.7%**(GPT-3.5 >$10/M tok → DeepSeek-V3 <$1/M tok);AI 公司比 SaaS 達 $5M ARR **快 13 個月**

## 5. 勞動市場衝擊

### METR 開發者生產力 RCT (2025-07)
16 名資深 open-source 開發者、246 任務、平均 5 年該倉庫經驗。**允許 AI 時慢 19%,但自我感覺快 24%**——感知與真實差 ~40 個百分點。使用 Cursor Pro + Claude 3.5/3.7。**這篇是「AI 究竟提速還是減速資深工程師」辯論的最堅實證據**。

### Anthropic Economic Index (2025-2026)
- 2025/09 揭示地理分布不均
- 2025/11:**augmentation 52% 已反超 automation 45%**;前 10 任務佔 24% 流量
- 2026/01 Economic Primitives 釋出 5 個新原語
- 2026/03 Learning Curves:高 tenure 用戶嘗試更高價值任務、成功率較高

### McKinsey / Goldman Sachs
- Goldman Sachs:**全球 300M FTE-等量**任務可被自動化,美國 25% 工時暴露;GDP 提升 7%
- McKinsey:2030 美國 29.5% 工時可被自動化,75-375M 工人需轉型

## 6. AI 治理與晶片地緣政治

- **EU AI Act** 自 2024 起分階段生效,2026 通用 AI 模型義務生效
- 美國 2025/01 **AI Diffusion Rule** 三層架構(Tier 1 美+18 盟友、Tier 2 配額、Tier 3 禁運)
- 2025/04 Trump 禁 H20,三個月後反轉
- **2026/01/15** 正式採用更彈性的 H200/MI325X 等效門檻、對非美供應鏈先進 AI 晶片課 25% 關稅
- **GAIN AI Act** 進入立法討論
- Anthropic RSP v3、OpenAI Preparedness Framework v2 (2025/04) 同時演進

## 7. 失業 / 補貼 / 教育

Sam Altman 的 **OpenResearch UBI pilot** (2016-2019, 3000 人):每月 $1000 受贈者每週工時僅減 1.3 小時、未退出勞動市場——「UBI 不會養懶人」最強實證。

Musk、Altman、Suleyman 公開支持 AI-funded UBI;2025 多場美國 Guaranteed Income pilot 結項。教育系統回應緩慢——多數方案聚焦「prompt literacy / AI fluency」而非結構改革。

## 8. AI 公司商業模式演進

三種營收形態並存:
- **Tokens-as-a-Service**(OpenAI/Anthropic API,per-token,商品化,24 個月跌 99.7%)
- **Seat-based** subscription(ChatGPT $20、Claude $20、Cursor Pro $20)
- **Agent-as-a-Service / per-task**(Devin $500/月、Cursor 2026/02 過 $2B ARR)

Anthropic 主導 API/coding 高端;OpenAI 主導 consumer chat;Google 主導 multimodal+search。

## 9. 預測方法論本身

四種主流方法:
1. **Scaling Laws Extrapolation** (Kaplan 2020、Chinchilla 2022、**Farseer 2025** 大規模誤差降 4×)
2. **Benchmark Forecasting** (Epoch ECI、METR Time Horizons、AI 2027 用 SC→SAR→SIAR→ASI 階段)
3. **Expert Elicitation** (METR 2025/08 Pilot Study 訪 8 forecasting expert + 10 superforecaster)
4. **Bottom-up R&D Acceleration Modeling** (AI Futures Project takeoff model、METR Simpler Timelines)

test-time compute 重塑了 scaling law:o1/o3 的 log-linear inference scaling 加上 RL 訓練算力 → 新雙軸 scaling。

## 10. 立場光譜

| 軸位 | 代表 | 核心論點 | 可量化分歧 |
|---|---|---|---|
| Hard Doomer | Yudkowsky、Hinton(部分) | P(doom) > 50%, 需暫停 | 滅絕機率 |
| Soft Doomer | Bengio、Russell、Stuart | 治理可控但需強制 RSP | 失誤率閾值 |
| Cautious Insider | Amodei、Kokotajlo | RSP 可用、需 ASL-4 | AGI 中位數 2026-2028 |
| Pragmatic Boomer | Altman、Musk | 加速但需 UBI | AGI 已部分到來 |
| Architectural Skeptic | LeCun、Marcus | LLM 路徑錯誤 | 還需新範式 |
| Acceleration Right | Andreessen、e/acc | 零監管最優 | P(doom) ≈ 0 |

---

## 2026 AI Forecasting 工程師地圖

**A. 每週/每月必訂閱**:Epoch AI Blog + Substack「The Epoch AI Brief」、METR Blog/Notes、Anthropic Economic Index、SemiAnalysis Newsletter、Dwarkesh Podcast、Zvi Mowshowitz、AI Futures Project blog、Peter Wildeford forecasts、Stratechery、80,000 Hours Substack

**B. 量化資料源**:epoch.ai/data、METR HCAST/RE-Bench、Metaculus AI tournaments、Manifold AI markets、OpenLM ARC-AGI、SWE-bench-verified leaderboard

**C. 必讀核心文獻**:Kaplan 2020 scaling laws、Hoffmann 2022 Chinchilla、Farseer 2025、Aschenbrenner 2024 Situational Awareness、AI 2027、METR Time Horizons、Anthropic RSP v3、OpenAI Preparedness Framework v2、Goldman 300M jobs report、Mary Meeker 2025 AI Trends

**D. 量化技能棧**:Bayesian updating、Brier score 自評、log-linear regression on benchmarks、scaling law fitting (Python, scipy)、Monte Carlo simulation、time-horizon doubling 偵測

**E. 三個追蹤 KPI**:
1. **Frontier ECI 點/年**(目前 ~15.5)
2. **METR Time Horizon doubling**(目前 ~131 天)
3. **Anthropic+OpenAI 合計 ARR vs 合計年燒錢**(差距決定 bubble vs runway)

**F. 三個 contrarian thesis 試金石**:
1. inference scaling 是否會撞牆(LeCun 對)
2. compute 是否被電力卡死(Aschenbrenner 對的反面)
3. 自動化 AI R&D 是否真能 self-bootstrap(METR Simpler Timelines 的核心假設)

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
