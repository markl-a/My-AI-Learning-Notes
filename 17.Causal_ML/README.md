# 17. 因果機器學習 / Causal ML (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #11
> 「No causes in, no causes out」— Nancy Cartwright。在 LLM 把「相關性」榨到極限的時代,因果推論正從學術角落躍升為 AI 系統可靠性的核心軸線。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **監督式 ML 基礎(迴歸 / 分類 / cross-validation)**(對應 repo:[1.從AI到LLM基礎/3.ML_&_Data_Analysis](../1.從AI到LLM基礎/3.ML_&_Data_Analysis/README.md))
> 2. **能跑 LightGBM / XGBoost 等 boosted tree**(對應 repo:同上)
> 3. **基本機率與統計推論**(若 repo 內無,先看:[StatQuest 機率統計系列](https://statquest.org/))
>
> 對「相關性 ≠ 因果性」、Simpson's paradox、confounder 沒概念,請先讀 [全景圖 #11](../2024-2026_AI完整領域全景圖.md) 對應章節建立直覺。
>
> **延伸 / 反向連結**:[2.深入LLM模型工程與LLM運維/9.模型評估 (Evaluation)](<../2.深入LLM模型工程與LLM運維/9.模型評估 (Evaluation)/>) | [3.LLM應用工程/5.進階 RAG 與多元資料檢索](<../3.LLM應用工程/5.進階 RAG 與多元資料檢索/>)

---

## 1. 理論基礎:兩大學派的歷史張力與融合

### 1.1 Pearl 的 SCM / do-calculus

**Structural Causal Model (SCM)** 將因果系統表達為結構方程 + DAG,透過 **do-operator** 明確區分「觀察 P(Y|X)」與「介入 P(Y|do(X))」。do-calculus 三條規則提供完整代數識別工具。Pearl 的 **Ladder of Causation**(association → intervention → counterfactual)為 2024-2026 LLM 因果推理論文的評測框架。

### 1.2 Rubin 的 Potential Outcomes (PO / RCM)

以「同一單位的 Y(1) vs Y(0)」反事實對作為原語,**SUTVA + ignorability + positivity** 是三大金科玉律。經濟學、統計學、衛生政策幾乎完全在 PO 語言中工作。

**Pearl vs Rubin 共識**:二者形式上幾乎等價(在 ignorability ↔ d-separation 的橋樑下)。**識別**用 SCM 畫圖,**估計**用 PO 寫 estimand。PyWhy 已正式採取「graph + PO 雙語言」設計。

## 2. 四大主任務

### 2.1 Causal Identification(識別)
**backdoor criterion**(阻斷後門路徑)、**frontdoor criterion**、**IV (instrumental variable)**、**front-/back-door adjustment**。ID algorithm(Shpitser-Pearl)已內建於 DoWhy。

### 2.2 Causal Estimation(估計)
工具箱:**Matching、IPW、Outcome regression、Doubly Robust、Double ML、TMLE**。DR / DML 在 Neyman orthogonality 性質下成為產業預設。

- 高維 confounder → **DML (EconML LinearDML / CausalForestDML)**
- 異質效應 → **Causal Forest / X-learner**
- 違反 ignorability → **IV / RDD / DiD / Synthetic Control**

### 2.3 Causal Discovery(結構學習)
- **Constraint-based**: PC、FCI
- **Score-based**: GES、A*
- **Functional / SEM-based**: LiNGAM、DirectLiNGAM、ANM
- **Continuous optimization**: **NOTEARS** → DAG-GNN、GraN-DAG、**DECI**
- **GPU 加速**: AcceleratedLiNGAM

**注意**:NOTEARS 在「量綱尺度」下被證明不適合;社群轉向更穩健的 score-based + LLM-prior 混合。

### 2.4 Counterfactual Prediction(反事實預測)
Gumbel-max SCM、deep twin-network、CEVAE。應用於 OPE、credit attribution、醫療 what-if。

## 3. 與機器學習的融合

### 3.1 Double Machine Learning (DML)
Chernozhukov et al. 2018 的 DML 用 Neyman-orthogonal moment + cross-fitting,允許用 ML(LightGBM、NN)估 nuisance 但保持 √n-asymptotic normality。EconML 2025 (v0.16.0) 與 DoubleML R/Python 為標準實作。

### 3.2 Causal Forest / GRF
用隨機森林做「locally weighted」moment 估計,輸出 CATE 與 honest CI。適合中等維度、異質性顯著的場景(行銷、補貼)。

### 3.3 Meta-learners (S/T/X/DR/R-learner)
2025 大規模 Criteo Uplift v2.1(1398 萬樣本)實證 **S-Learner + LightGBM** 在 Qini 上勝 X-Learner / Causal Forest,挑戰「S 太簡單」的傳統認知。

### 3.4 Causal Representation Learning (CRL)
Bernhard Schölkopf 學派目標是「從低層像素 / token 中抽出高層因果變數」。2024 NeurIPS *From Causal to Concept-Based Representation Learning* 把 CRL 鬆綁為幾何概念恢復。Schölkopf 認為 CRL 是「**foundation model 的下一個 frontier**」。

### 3.5 Causal RL / OPE
Elias Bareinboim 的 Columbia CausalAI Lab 是大本營。2025 ICLR *Counterfactual Realizability and Decision-Making* 奠定理論。實務上 OPE + confounder-robust IPS / DR estimator 已是 Netflix、Spotify 推薦線上實驗的標準配置。

## 4. 工具棧(2026 業界 Stack)

### PyWhy 生態(Microsoft 主導)
| 工具 | 角色 | 狀態 |
|---|---|---|
| **DoWhy** | 四步 framework: model → identify → estimate → refute | v0.13 (Nov 2025) |
| **EconML** | HTE / DML / DRLearner / 政策學習 | v0.16.0 (Jul 2025) |
| **CausalML** (Uber) | Uplift modeling、Meta-learners、Tree-based | 活躍維護 |
| **causal-learn** (CMU) | PC、GES、LiNGAM 等 discovery | Python rewrite of Tetrad |

### 其他關鍵庫
- **CausalNex** (QuantumBlack/McKinsey):Bayesian Network + NOTEARS
- **Tigramite** (Jakob Runge):時序 PCMCI / PCMCI+
- **DoubleML**:R + Python 雙語言,學術正統 DML
- **CausalImpact** (Google):BSTS-based intervention analysis

**選型**:**DoWhy(識別) + EconML(估計) + causal-learn(發現) + 自家業務 DAG** 是 2026 最務實的 stack。

## 5. 應用場景

### A/B 測試 + 因果推論
Netflix「Round 2: A Survey of Causal Inference Applications at Netflix」(2024) 涵蓋 quasi-experiments、long-term metrics、heterogeneous effects。Airbnb **ACE (Artificial Counterfactual Estimation)** 用 ML 重建反事實。Booking.com 推 sequential testing + trigger analysis。

### 政策評估
DiD、Synthetic Control(Abadie 系列)、RDD 主導勞動經濟、教育改革評估。2024 後 **Synthetic DiD**、**Augmented SC** 為新熱點。

### 醫療 RWE
FDA 接受 Real-World Evidence 後,target trial emulation + TMLE 成為藥效再評估主力。CRL 滲透 single-cell 基因擾動分析。

### Marketing Uplift
標準四象限(Persuadables / Sure-things / Lost-causes / Sleeping-dogs)。Criteo Uplift v2.1 benchmark 是 2025 學術標桿。

### 推薦系統 debias
Selection / exposure / popularity bias 用 IPS、DR、Counterfactual VAE 處理。

## 6. 與 LLM 結合(2024-2026 最熱軸)

### LLM 能否做因果推論?

**Kıcıman, Ness, Sharma, Tan (arXiv 2305.00050)** benchmark:GPT-4 在 pairwise causal direction 上超越多數傳統 discovery,但作者強調 LLM 在做「**meta-causal**:從訓練語料記憶因果常識」,並非真正獨立推論。

**Jin et al. 2024** 的 **Corr2Cause** benchmark 顯示主流 LLM 在「從純相關矩陣推因果」近乎隨機,揭露 LLM 缺乏正式 causal calculus。

### Causal Chain-of-Thought 與 Causal-Aware Prompting
- **CDCR-SFT** 讓模型在回答前顯式構造 variable-level DAG
- **Structured Thinking Matters (2025)** 把 CoT 結構化為因果步驟
- **C2P** 將 do-calculus 規則灌進 prompt

### Causal RAG / GraphRAG
- **CausalRAG (ACL Findings 2025)** 把因果圖嵌入檢索向量空間
- **CDF-RAG** 用 PPO 強化學習對齊 retrieval 與 causal verifier

### 與 Hallucination 的關係
業界共識:幻覺源於 LLM 學「token 共現分布」而非「世界因果機制」。**CRL + do-calculus 提供「為什麼是這個答案」的機制式 grounding**,被 Bengio、Schölkopf 視為通往「System 2 reasoning」的關鍵路徑。

## 7. 時序因果

- **Granger causality**:統計學經典(預測力 ≠ 真正因果)
- **PCMCI / PCMCI+** (Jakob Runge):Tigramite,氣候/地球科學黃金標準
- **CausalRivers (ICLR 2025 Spotlight)**:東德 666 站 + 巴伐利亞 494 站河川流量,5 年資料、千級節點 ground-truth 圖 — **2025-2026 時序 causal discovery 最重要的 benchmark**

## 8. 重要書籍與課程

- **《The Book of Why》** (Pearl & Mackenzie, 2018)
- **《Causal Inference: The Mixtape》** (Cunningham):經濟學 DiD/IV/RDD
- **《Causal Inference: What If》** (Hernán & Robins):流行病學經典,免費 PDF
- **《Causal Inference and Discovery in Python》** (Aleksander Molak, 2024)
- **《Elements of Causal Inference》** (Peters, Janzing, Schölkopf):CRL 理論底
- **Brady Neal** 線上課:免費 YouTube 全套錄影
- **Matheus Facure《Causal Inference for the Brave and True》**:Python 開源書

## 9. 產業現狀

**市場**:Causal AI 2025 估 $60M-80B 不等(來源差異大);Precedence Research 看 2034 達 $1,127B、CAGR ~39%。需求方無爭議:**Microsoft、Uber、Netflix、Booking、Airbnb、Spotify、LinkedIn、TikTok 都有專屬 Causal Inference team(20-100 人量級)**。

**Vendor 化**:causaLens、Geminos、Causality Link 等專做 Causal AI SaaS;Dataiku DSS 14 把 Causal Prediction 內建到主流 ML 平台。

## 10. 未來方向(2026-2028)

1. **Foundation Models for Causal Inference**:把 SCM 結構作為 transformer 預訓練目標
2. **Causal RL Agent**:counterfactual rational agent
3. **CRL + Multimodal**:從 video + 動作學物理因果
4. **LLM-as-DAG-prior**:LLM 提供候選 DAG,resolve 等價類歧義
5. **Causal Benchmarks 規模化**:期待醫療、金融、推薦 in-the-wild 千節點 benchmark
6. **Causal Agentic AI**:Agent 做 counterfactual debrief、avoid confounded reward hacking

---

## 2026 Causal ML 工程師地圖

**6 個月入門路徑**:
1. Week 1-4:Brady Neal 線上課 + 《Book of Why》(建立直覺)
2. Week 5-10:Molak 書 + DoWhy 官方 tutorial(動手做 ATE / CATE)
3. Week 11-16:EconML DML / Causal Forest,跑 Criteo Uplift dataset
4. Week 17-20:causal-learn 跑 PC / DirectLiNGAM,讀 NOTEARS / DECI 論文
5. Week 21-24:選一個專業 vertical(時序 → Tigramite + CausalRivers;LLM → Corr2Cause + CausalRAG)
6. 持續:追 Bareinboim CausalAI Lab、Schölkopf MPI Tübingen、Athey Stanford、Chernozhukov MIT 的 arXiv

**範式總結**:相關性回答「**是什麼**」,因果性回答「**為什麼以及如果**」。深度學習擅長前者,工業決策需要後者。2026 的 ML 工程師若只會 fit/predict,職涯天花板就在「**監督學習的相關性陷阱**」;誰先把 SCM 與 PO 內化進產品迴路,誰就能在 LLM 時代供應「可解釋、可介入、可審計」的高價值 AI 系統。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
