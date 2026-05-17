# 15. 隱私運算與機密 AI / Privacy-Preserving AI (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #21
> 進入「規模化生產」階段:Apple PCC、NVIDIA Blackwell TEE-I/O、Google Gboard 把這層基礎建設變成標配。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **LLM 訓練流程(預訓練 / SFT / RLHF)**(對應 repo:[2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)](<../2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)/>))
> 2. **模型部署與推論基礎**(對應 repo:[2.深入LLM模型工程與LLM運維/8.模型部署與運維](../2.深入LLM模型工程與LLM運維/8.模型部署與運維/))
> 3. **基本密碼學概念**(對稱/非對稱加密、雜湊 — 若 repo 內無,先看:[Stanford CS 255 Lecture 1-3](https://crypto.stanford.edu/~dabo/courses/cs255_winter25/syllabus.html))
>
> 完全沒接觸過 FL/DP/TEE,先讀 [全景圖 #21](../2024-2026_AI完整領域全景圖.md) 對應章節。
>
> **延伸 / 反向連結**:[2.深入LLM模型工程與LLM運維/8.模型部署與運維](../2.深入LLM模型工程與LLM運維/8.模型部署與運維/) | [16.AI_Content_Authenticity](../16.AI_Content_Authenticity/README.md)

> **⚠️ 鮮度與可信度說明 / Freshness & Reliability**
> 本章涉及 **Apple PCC、NVIDIA Blackwell TEE-I/O、Intel TDX、AMD SEV-SNP、Google Confidential VMs 等 TEE 平台規格與時程,以及 Apple 自研 AI server 晶片量產推測、聯邦學習在 Gboard 規模、ε 等 DP 預算數字** 等具體陳述。這些資訊混合三類來源:
> 1. **NIST/IEEE 標準、Apple/NVIDIA 白皮書、學術 paper**(高信任度)
> 2. **媒體報導 / 供應鏈推測 / vendor blog**(屬「報導」非「事實」)
> 3. **AI agent 整理 + 我手動驗證**(可能有誤標,尤其 Apple 自研 AI server 晶片 2026 H2 量產、Blackwell TEE-I/O 細節)
>
> 任何要做合規 / 採購 / 架構決策的陳述(TEE 規格、認證等級、performance penalty、上市時程),**請以原廠 documentation 與相關監管文件為準**。本章對「TEE + FL + DP 三者進入規模化生產」這類 *結構性訊號* 較高信心,對「Apple 2026 H2 量產 X」這類 *時點推測* 較低信心。

---

## 1. 聯邦學習 (Federated Learning)

FedAvg 為基線,**FedProx** 解 client drift、**FedOpt** 處理非 IID;**Personalized FL** (Per-FedAvg、Ditto)、**Vertical FL** (SecureBoost、FATE) 對應企業跨機構場景。

**框架生態**:
- **Flower** (1.24, 2025/12) — 語言/框架中立,Python 3.13、heartbeat 機制、Helm 部署
- **NVIDIA FLARE** 2.7 (2025) — 醫療生產案例,內建 HE+DP
- **PySyft** (OpenMined) — data scientist friendly
- **OpenFL** (Intel) — 主打 TEE 整合
- **Apple PCC FL** — on-device + secure aggregation

**2025-2026 突破**:Google Gboard 部署 **Confidential Federated Analytics (CFA)**,兩天內偵測 3,600 個新印尼語詞彙。

**選型**:跨組織選 Flower (語言/框架中立) 或 FLARE (NVIDIA 硬體棧);僅 PyTorch 內部選 FLARE。永遠搭配 secure aggregation + DP。

## 2. 差分隱私 (Differential Privacy)

(ε,δ)-DP 與 Rényi DP (RDP accountant) 取代 moments accountant 成主流;**DP-SGD** 為訓練側標準。**Opacus** (Meta) 是 PyTorch 事實標準;Gboard 全線採 **DP-FTRL**,所有新神經 LM 必須 ρ-zCDP ρ∈(0.2, 2)。

**2025-2026 突破**:Google 發表 user-level DP fine-tuning of LLMs;**DP-ZeRO** (基於 DeepSpeed) 支援 >7B 參數 DP fine-tune,突破 Opacus DDP 無 sharding 的瓶頸;Anthropic Clio 揭示「DP for LLM 監控/觀察 pipelines」的新範式。

**實作建議**:LLM fine-tune 用 LoRA + DP-SGD (只對 adapter 加噪) 是 sweet spot;ε ≤ 8、δ ≤ 1/N²。避免 full-pretrain DP。

## 3. 機密運算 / TEE

CPU 側 **Intel TDX** (Google c3 GA、Azure GA) 與 **AMD SEV-SNP** (GCP N2D、Azure GA) 雙雙生產化;GPU 側 **NVIDIA H100 Confidential Computing** 是 2024 商用主力。

**2025-2026 突破**:
- **NVIDIA Blackwell (B200/GB200)** 業界首款 **TEE-I/O capable** GPU,NVLink inline 加密,**機密 mode 與明文模式效能幾乎無差**——挑戰「機密運算很慢」的舊印象(在特定 benchmark 與工作負載下)
- **Apple Private Cloud Compute** 2024 WWDC 公開白皮書,2026/02 升級 **M5 PCC servers** + **PCC Agent Worker** (專屬 iOS agent runtime);專屬 AI server 晶片預計 2026H2 量產
- **Google Confidential Space** 與 Azure Confidential VM 已成跨雲 attestation 基底

**實作建議**:LLM serving 從 Day 1 設計 attestation chain (host→GPU→workload)。

## 4. 同態加密 (FHE)

**CKKS** (近似實數) 主導 ML、**BFV/BGV** 主導整數工作流。庫:**Microsoft SEAL 4.2** (2025/09)、**OpenFHE 1.5.1** (2026/04)、**Zama Concrete ML v1.5+** (DataFrame API、neural net 2-3× 加速)、Lattigo (Go)。

**2025-2026 突破**:Zama 推出 **Zama Protocol**;TFHE-rs GPU benchmark 在 H100/B200 上把 bootstrapping 拉到亞毫秒級;FHE-friendly transformer (替換 softmax/GELU 為多項式) 開始可行。

**實作建議**:FHE 用於極端高敏感場景 (DNA、KYC)、ensemble 小模型或加密向量檢索;LLM 推理仍以 TEE 為主。

## 5. 安全多方計算 (MPC)

**MP-SPDZ** (data61) 為研究與基準的瑞士刀;**CrypTen** (Meta) 對接 PyTorch tensor + autograd。USENIX Security 2024/2025 揭露 SPDZ MAC check 在並發/多執行緒下的 key leakage,推動 patch 與形式化驗證;MD-ML 等新框架在三方 LAN 推理達到接近明文延遲。

**選型**:MPC 適合「3-5 方、半誠實、區域網」(銀行/保險聯合風控);跨地端高延遲場景優先 FL+TEE。

## 6. FL + DP + TEE 組合 stack

2026 的事實標準 = **FL (聚合層) + DP (噪聲注入) + TEE (server-side aggregator attestation)**。Google 的 **Confidential Federated Computations (CFC)** 是範式:client→encrypted upload→TEE aggregator (attested image)→DP noise→model update。

**實作建議**:設計時把「attestation evidence」當成必交付物。

## 7. 私有 LLM 推理

三條路線並存:
1. **on-device** (Apple Intelligence ~3B foundation model)
2. **PCC 範式** (敏感 prompt 上 attested TEE server)
3. **企業私有 endpoint** (Together AI Private, Helicone private, Azure OpenAI in CVM)

**Anthropic Confidential Inference whitepaper** (2025/10):基於 trusted VMs + GPU CC,定義 client↔server 雙向 attestation。Red Hat 推 confidential inference on OpenShift。

**實作建議**:敏感 prompt 路由策略 — 預設 PCC 級 TEE;auditing 留 hash-only log;client SDK 必驗 attestation quote。

## 8. 法規驅動

- **EU AI Act**:2024/08 生效,prohibited practices 2025/02 上路,**GPAI 義務 2025/08**,高風險系統 2026/08
- **GDPR Article 25 privacy-by-design** 把 FL/DP/PETs 列為合規預設手段
- **HIPAA、PCI DSS、中國個資法 (PIPL)** 與資料出境安全評估同步收緊

EU 監管機關正式承認 federated learning、synthetic data、DP、HE、MPC 為「資料最小化」可接受技術手段。

## 9. 產業採用案例

- **醫療**:NVFLARE 在乳癌/肺癌早篩跨醫院聯邦學習;歐洲 MELLODDY (藥廠)
- **金融**:跨銀行反洗錢 (AML) 用 MPC/FL
- **IoT/edge**:Android Private Compute Core、Apple on-device ML
- **消費級**:Google Gboard 全線 DP-FL+CFA (900+ 語言)、Apple Siri PCC 路由

## 10. 2025-2026 重大里程碑

- 2024/06 — Apple PCC 白皮書與 Apple Intelligence 發表
- 2024/08 — EU AI Act 生效
- 2024Q4 — NVIDIA Blackwell GB200 NVL72 出貨 (CC + TEE-I/O)
- 2025/03 — Google Confidential Federated Analytics 量產
- 2025/08 — EU GPAI 義務生效
- 2025/10 — Anthropic Confidential Inference 白皮書
- 2025/12 — Flower 1.24 釋出
- 2026/02 — Apple M5 PCC servers + PCC Agent Worker 架構
- 2026/04 — OpenFHE 1.5.1
- 2026/08 — EU AI Act 高風險條款 (擬延期)
- 2026H2 (預) — Apple 自研 AI server 晶片量產

---

## 2026 隱私運算工程師地圖

**4 層 stack**:

**Layer 0 — 硬體與 attestation**:NVIDIA H100/Blackwell CC mode、Intel TDX、AMD SEV-SNP、remote attestation (RA-TLS、Intel Trust Authority、Apple PCC transparency log)。

**Layer 1 — 密碼學原語**:RDP/zCDP 會算 ε;CKKS/BFV 能挑 scheme;SPDZ/ABY3 能挑 protocol。庫:Opacus, OpenFHE/SEAL, Concrete ML, MP-SPDZ, CrypTen。

**Layer 2 — 訓練/推理框架**:Flower 1.24 + NVFLARE 2.7 雙引擎;DP-LoRA + DP-ZeRO 處理 >7B LLM;PCC-style 推理棧 (attested TEE + RA-TLS + audit-only logging)。

**Layer 3 — 合規與治理**:EU AI Act 風險分類、GDPR Art.25 PETs 證據、HIPAA BAA、模型卡 + 隱私聲明 + attestation evidence 三件套交付。

**核心心法**:**FL 解資料移動、DP 解資料記憶、TEE 解資料可見性**,三者組合才能同時滿足「資料不出域 + 模型不背誦 + 雲端不偷看」這個 2026 的隱私三角。Apple PCC 與 NVIDIA Blackwell 已證明性能稅 (perf tax) <5%,藉口不再成立。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
