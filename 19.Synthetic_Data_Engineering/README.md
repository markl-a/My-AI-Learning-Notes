# 19. Synthetic Data 與資料工程 (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #15
> 「2020 軍備競賽是 GPU,2023 是 Token 數量,2025-2026 是**資料品質與合成能力**。」

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **SFT(監督微調)流程**(對應 repo:[2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)](<../2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)/>))
> 2. **偏好對齊(DPO / RLHF / GRPO)概念**(對應 repo:[2.深入LLM模型工程與LLM運維/6.偏好對齊 (Alignment) 技術](<../2.深入LLM模型工程與LLM運維/6.偏好對齊 (Alignment) 技術/>))
> 3. **基本資料工程直覺**(deduplication、quality filter、scaling laws — 若不熟,先看 [全景圖 #15](../2024-2026_AI完整領域全景圖.md))
>
> 完全沒做過 fine-tuning 之前讀本檔,各節「distilabel pipeline / Magpie / RLAIF judge」等實作術語會跳得太快,建議先把 SFT 流程跑過一次。
>
> **延伸 / 反向連結**:[2.深入LLM模型工程與LLM運維/3.資料集準備與建立](../2.深入LLM模型工程與LLM運維/3.資料集準備與建立/) | [16.AI_Content_Authenticity](../16.AI_Content_Authenticity/README.md)(合成資料溯源 / 訓練資料簽章)

---

## 1. 為何 Synthetic Data 成為核心議題

Epoch AI 估算高品質英文文本將於 2026-2028 用盡 (Data Wall)。前沿模型在 15-20T tokens 後出現邊際遞減,而 GPT-5、Claude 4.7、Llama 4 的訓練量已逼近此線。同時三股壓力同步收緊:
- (a) NYT v. OpenAI、Getty v. Stability 等訴訟把版權數據變成負債
- (b) EU AI Act 2025/8/2 起強制「訓練內容公開摘要」
- (c) Cloudflare 預設封鎖 AI scraper、Reddit/X/StackOverflow 把資料貨幣化

**scaling laws 論述從「更多 token」轉向「更好 token」**——同一個 1T budget,合成/重寫資料可帶來 +7.1pp 與最高 7.7× 訓練加速。

## 2. 語言合成資料里程碑

四條代表路線:
- **Phi-3 / Phi-4 (Microsoft)** — 「Textbooks Are All You Need」延伸,以 GPT-4 生成課本級合成內容 + 嚴格過濾;Phi-4 (14B) 在 GPQA、MATH 上勝過數倍體量模型
- **DeepSeek-V3 / R1** — R1-Zero 純 RL 跑出 CoT,再用 rejection sampling 從 R1 蒸出冷啟 SFT 資料,AIME pass@1 從 15.6% → 71%,**證明模型可自舉生成自己的推理訓練集**
- **Nemotron-4 340B (NVIDIA, 2024/6)** — Instruct alignment data **98% 為合成**;2025 Nemotron-CC 重寫 2T tokens Common Crawl
- **Llama 4 Behemoth → Scout/Maverick (2025/4)** — Behemoth 作為 teacher,透過 codistillation loss 把知識壓進學生
- **OpenAI o-series** — 官方承認 o1/o3 reasoning 訓練使用大量合成 reasoning trace + Process Reward Model 篩選

## 3. 合成資料工具與框架

- **distilabel (Argilla / HF)** — OSS 事實標準。Step/Task DAG,可序列化為 YAML 並推 HF Hub,內建 Magpie、UltraFeedback、EvolInstruct、TextGrad、APIGen
- **Argilla** — annotation 平台,2024 被 HF 併購
- **Synthetic Data Vault (SDV)** — MIT 起家、Datacebo 商業化,主攻**表格/關聯式/時序**(CTGAN、PARSynthesizer、HMASynthesizer)
- **NVIDIA NeMo Curator + NeMo Data Designer** — GPU 加速的去重、品質分類器、PII 移除、domain mixing,可處理 PB 級語料
- **DataDreamer / Bonito / Genstruct / Augmentoolkit** — 從非結構化文件反向生成 instruction-response
- **Prime Intellect SYNTHETIC-1** — 分散式社群算力合成 14M verified reasoning traces

**選型**:小團隊起手 distilabel + Argilla;企業合規場景上 SDV;預訓練規模上 NeMo Curator。

## 4. 典型合成 pipeline 模式

六種範式已被驗證:
- **Self-Instruct (2022)** — seed task 自舉,Alpaca 起點
- **Evol-Instruct (WizardLM)** — 用「加約束 / 加深 / 具體化」算子迭代演化指令複雜度
- **UltraFeedback** — 多模型多評分維度生成偏好對,DPO 主力訓練集
- **Magpie (ICLR 2025)** — **不需要 seed**——只餵 chat template prefix,讓對齊模型自己吐出 user query,再正常產生 response。從 Llama-3-Instruct 生 4M 對話,精選 300K 即可超越 ShareGPT + UltraFeedback
- **PersonaHub (Tencent)** — 1B persona 描述 × 任務模板
- **Direct Prompting + Filter** — Phi 系列代表

**最佳實踐**:多樣性 (Magpie/Persona) 配合難度演化 (Evol) 配合偏好對 (UltraFeedback),三層疊合最有效。

## 5. 品質控管

「合成 ≠ 高品質」,過濾與評估佔合成成本一半以上:
- **Reward Model 過濾**:Nemotron-Reward、Skywork-Reward、ArmoRM
- **LLM-as-Judge**:GPT-4o/Claude/Gemini 給 1-10 分或 pairwise;留意 position bias、length bias、self-preference
- **語意去重**:embedding (E5、bge) + FAISS HNSW + threshold,SemDeDup 證明可丟 50% 仍提升表現
- **MinHash + LSH**:trillion-scale 標配,2025 LSHBloom 把記憶體再壓 6×,Milvus 2.6 內建 MinHash 索引
- **毒性 / PII**:Detoxify、Presidio、NeMo Guardrails
- **驗證式過濾**:程式碼可執行、數學答案可校對 → R1、SYNTHETIC-1 採用

## 6. 資料 Flywheel 概念

`deploy → user feedback → label → train → deploy` 變成現代 AI 公司護城河:
- **Scale AI** — 2025/6 Meta 49% 入股 ($14.3B) 後,Google/OpenAI/MSFT/xAI 集體出走
- **Surge AI** — bootstrap 到 $1B ARR、$25B 估值,主供 Anthropic、OpenAI 高端 RLHF
- **Mercor** — 從低雙位數百萬 ARR 12 個月衝到 $850M、估值 $10B,鎖定 PhD-level rubric
- **Hugging Face Spaces / Chatbot Arena** — user upvote 直接回流成偏好資料 (LMSYS-1M)
- **NVIDIA NeMo Data Flywheel Blueprint (2025)** — inference log → curate → retrain → A/B → redeploy

**心法**:從 day-1 設計 telemetry schema (prompt、response、user signal、tool call、latency),別等模型上線才補日誌。

## 7. 資料品質研究

開源語料賽進入精緻化階段:
- **FineWeb (15T) / FineWeb-Edu (1.3T)** — HF 用 Llama-3-70B 打教育性分數再篩,MMLU 33% → 37%
- **DCLM (Apple + UW)** — 240T token testbed + 53 評測;DCLM-baseline 7B 用 2.6T token 超過 Llama-3 8B (15T)
- **SmolLM Corpus / SmolLM2/3** — Cosmopedia + FineWeb-Edu + Python-Edu
- **Dolma (AI2)** — 3T token,完整公開過濾規則
- **BeyondWeb (DatologyAI, 2025)** — 把 rephrasing 推到 trillion scale

**心法**:domain-specific 預訓練先過 fastText/小 classifier 而非靠大模型 (成本差 100×)。

## 8. 資料治理 / 合規

2025/8/2 EU AI Act GPAI 條款生效,**訓練內容公開摘要**從建議變強制:
- **Template (2025/7/24)**:必須揭露模態 / 規模 / 來源類別 (public / licensed / scraped / user / synthetic)、處理措施、版權保留
- **Opt-out 機制**:robots.txt、ai.txt、TDM Reservation、C2PA TDM Assertion、Spawning Do Not Train、JPEG Trust v2、ISCC
- **C2PA / Content Credentials**:既標記生成內容也可承載 TDM opt-out
- **PII 偵測**:Presidio、AWS Comprehend、Lakera
- **訴訟**:NYT v. OpenAI 進入 discovery;Anthropic v. Bartz 和解 ($1.5B 圖書賠償)

## 9. 多模態合成資料

- **DataComp / DataComp-LM** — 12.8B image-text CommonPool;DataComp-1B = 1.4B filtered pairs
- **影像 caption 重寫** — CapsFusion、ShareCaptioner、Recap-DataComp-1B,ImageNet 提升 2%+
- **影片** — LAION video2dataset、Panda-70M、OpenVid-1M
- **音訊** — WavCaps、Auto-ACD
- **Tool-use / agent trace** — ToolBench、APIGen、Glaive-Function-Calling、xLAM、AgentInstruct;Anthropic / OpenAI 用沙盒環境 rollout 合成數百萬筆工具呼叫軌跡
- **3D / 機器人** — NVIDIA Cosmos、Genesis、ProcThor 生成 sim2real 軌跡

## 10. 「資料工程」職位演進

傳統 ETL/feature engineer 工作被分割成三角色:
- **Data Curator / Dataset Engineer** — 設計過濾規則、跑 ablation、寫 dataset card
- **Synthetic Data Designer** — 寫 prompt 模板、persona、judge rubric,設計 distilabel/NeMo pipeline。本質是「**用 prompt 寫資料集**」
- **Eval / RL Environment Engineer** — 設計 verifiable reward、unit-test-style 評估、agent sandbox

**消失中**:Spark/Airflow 排程、Hive 寫 SQL、純 dimensional modeling。
**新增**:HF datasets + Polars、Ray Data、vLLM/SGLang 批次推論、embedding store、LLM judge 校準、版權與隱私法規。

---

## 2026 資料工程師 (LLM 時代) 地圖

| 層 | 工具棧 | 關鍵能力 |
|---|---|---|
| **採集** | Common Crawl、自家 telemetry、合法授權 corpus | robots.txt/TDM 合規、C2PA 讀取 |
| **過濾 / 去重** | fastText、Llama-3-Edu-classifier、MinHash-LSH、LSHBloom、SemDeDup、Presidio | trillion-scale 處理、ablation 設計 |
| **合成生成** | distilabel、NeMo Data Designer、SDV、Magpie、PersonaHub | prompt 設計、persona library、judge rubric |
| **品質回饋** | Argilla、Prometheus-2、Skywork-Reward、Chatbot Arena 風格 pairwise | LLM-judge 校準、bias 控制 |
| **儲存 / 服務** | HF Datasets、Parquet + DuckDB、Polars、LanceDB、Milvus、Ray Data | 大規模 streaming、embedding index |
| **治理** | dataset card、EU Training Data Summary、C2PA、DP-SGD | 版權盡職、PII 稽核、lineage |
| **flywheel** | inference log → curate → eval → retrain | telemetry schema、A/B、回流節流 |

**一句話總結**:2026 的 AI 競爭力已不在「誰有更多 GPU」,而在「誰能以更低成本生成 + 過濾出**每一個 token 都帶資訊量**的資料」——這正把資料工程從幕後成本中心,推上模型品質的最前線。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
