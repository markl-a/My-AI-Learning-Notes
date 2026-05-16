# 我的 AI 學習相關筆記

> 台灣繁體中文 | 從基礎數學到 LLM、Agent、RAG、Robotics、AI4Science 等 **22 個主題** 的完整 AI 工程師學習路徑筆記

[![GitHub stars](https://img.shields.io/github/stars/markl-a/My-AI-Learning-Notes?style=social)](https://github.com/markl-a/My-AI-Learning-Notes)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Last Update](https://img.shields.io/badge/last%20update-2026--05-green.svg)](./CHANGELOG.md)
[![Powered by phantom-mesh](https://img.shields.io/badge/powered%20by-phantom--mesh-purple)](https://github.com/markl-a/phantom-mesh)

---

## 🚦 三條入口路徑(按你的目標選)

| 目標 | 從這開始 | 估時 |
|---|---|---|
| 🗺️ **想懂 2024-2026 整片 AI 領域** | [2024-2026 AI 完整領域全景圖](./2024-2026_AI完整領域全景圖.md) + [CONCEPT_MAP.md](./CONCEPT_MAP.md) | 30 分鐘讀完 |
| ⚡ **想立刻動手跑 LLM** | [📒 5 份 Colab Notebook](#-五份-colab-notebook完整-llm-工程訓練曲線) | 30 分鐘跑通第一份 |
| 🎯 **準備 AI 工程師面試** | [主題 9 面試準備](./9.面試準備與職業發展/README.md) — 180+ 題 + 5 案例 | 6 週衝刺 |
| 🤝 **phantom-mesh contributor** | [全景圖 #13/14](./2024-2026_AI完整領域全景圖.md) + [Case_02 LLM Gateway](./9.面試準備與職業發展/2.系統設計案例/Case_02_LLM_Gateway_API_Platform.md) | 1 週 ramp-up |

> **新讀者**:強烈建議**先看全景圖**理解 22 個主題的彼此關係,再決定深入哪幾條。

---

## 🌐 phantom-mesh 生態定位

本知識庫是 [phantom-mesh](https://github.com/markl-a/phantom-mesh)(自託管多 agent AI runtime)生態系的**學習路徑 + 面試準備教材**。本 repo **不是 runtime 本身**,而是知識庫與面試材料庫。

**兩條軸線**:
1. **AI 工程師學習路徑** — 22 主題完整覆蓋(本 README 下半部目錄)
2. **AI / 資料科學面試準備** — 主題 9 整合 phantom-mesh 開發中遇到的真實問題(multi-agent 協調、provider fallback、streaming SSE 解析、cost tracking、跨平台 build chain),不是教科書泛題

---

## 📒 五份 Colab Notebook(完整 LLM 工程訓練曲線)

本 repo 2026-05 新增 **5 份可在 Colab 一鍵跑的 hands-on notebook**,串聯成一條完整訓練曲線:

```
   Notebook 1 (SFT-LoRA) ───→ 訓出 task-specific adapter
        ↓
   Notebook 5 (DPO 對齊) ───→ 用偏好對齊微調 adapter
        ↓
   Notebook 4 (vLLM 部署) ──→ 把 adapter 上 serve,實測 prefix cache
        ↓
   ┌──────────────────────┐
   │ 上層應用              │
   │ - Notebook 3 (GraphRAG) ── 加結構化檢索
   │ - Notebook 2 (LangGraph)  ── 加多 agent 編排
   └──────────────────────┘
```

| # | Notebook | 主題 | GPU | 時間 | 對應 case |
|---|---|---|---|---|---|
| 1 | [LoRA SFT mini demo](./2.深入LLM模型工程與LLM運維/5.監督微調%20(SFT)/hands_on_project/notebooks/Colab_LoRA_SFT_Mini_Demo.ipynb) | Qwen2.5-0.5B + alpaca | T4 | 30 min | 04 Q9 multi-tenant LoRA |
| 2 | [LangGraph 3-agent supervisor](./3.LLM應用工程/3.Agent/notebooks/Colab_LangGraph_Multi_Agent_Research_Demo.ipynb) | Planner / Researcher / Writer | CPU | 30 min | Case_04 Multi-Agent Research |
| 3 | [Mini GraphRAG 手刻](./3.LLM應用工程/5.進階%20RAG%20與多元資料檢索/notebooks/Colab_MiniGraphRAG_Hands_On.ipynb) | networkx + Louvain + LLM 抽取 | CPU | 15 min | Case_01 Enterprise RAG |
| 4 | [vLLM 部署 + prefix cache](./2.深入LLM模型工程與LLM運維/8.模型部署與運維/notebooks/Colab_vLLM_Deploy_PrefixCache_Demo.ipynb) | vLLM server + streaming + benchmark | T4 | 15 min | Case_02 LLM Gateway |
| 5 | [DPO 偏好對齊](./2.深入LLM模型工程與LLM運維/6.偏好對齊%20(Alignment)%20技術/notebooks/Colab_DPO_Alignment_Mini_Demo.ipynb) | LoRA-DPO + Ultrafeedback | T4 | 30 min | — |

**每份都含**:phantom-mesh 真實工程考量段、6-8 個擴展練習、對應 deep-dive md 入口。

---

## 📚 22 主題目錄(2024-2026 全景)

### Layer 1:基礎(1-3)

| # | 主題 | 重點 |
|---|---|---|
| 1 | [從AI到LLM基礎](./1.從AI到LLM基礎/README.md) | 數學 / Python / ML / DL 全棧;含 CV 全景、Multimodal Generation、Time Series FM 三份 2024-2026 frontier deep-dive |
| 2 | [深入LLM模型工程與LLM運維](./2.深入LLM模型工程與LLM運維/README.md) | LLM 核心、SFT、DPO/GRPO 對齊、量化、vLLM 部署;13 個子章節 |
| 3 | [LLM應用工程](./3.LLM應用工程/README.md) | RAG、Agent、MCP、多模態生成、安全;14 個子章節 |

### Layer 2:個人軌跡 / 補充材料(4-6)

| # | 主題 | 重點 |
|---|---|---|
| 4 | [相關的更新Blog](./4.相關的更新Blog/) | Claude Code 系列、Vibe Coding、AI 瀏覽器比較等個人連載 |
| 5 | [AI 研究前沿 2024-2025](./5.AI研究前沿_2024-2025/) | 50 篇關鍵論文索引 + Vibe Coding 學習指南 + 4 個實戰專案 |
| 6 | [DeepLearning.ai 短課程學習紀錄](./6.DeepLearning.ai短課程學習紀錄/) | Andrew Ng 系列 13 門課程筆記 |

### Layer 3:面試與職涯(9)

| # | 主題 | 重點 |
|---|---|---|
| 9 | [面試準備與職業發展](./9.面試準備與職業發展/README.md) | **180+ 題題庫 + 5 個系統設計案例 + 4 份職涯指南**;phantom-mesh 真實場景整合 |

### Layer 4:Frontier Deep-Dive(11-22,2026-05 新增)

| # | 主題 | 重點 |
|---|---|---|
| 11 | [AI Hardware / Compute Stack](./11.AI_Hardware_Compute/README.md) | GPU / TPU / HBM / 液冷 / Neuromorphic |
| 12 | [AI for Science (AI4Science)](./12.AI_For_Science/README.md) | AlphaFold 3 / Evo 2 / GNoME / GraphCast |
| 13 | [Robotics & Embodied AI](./13.Robotics_Embodied_AI/README.md) | VLA / 人形機器人 / 世界模型 |
| 14 | [Voice / Audio AI](./14.Voice_Audio_AI/README.md) | Whisper / ElevenLabs / Pipecat / Suno |
| 15 | [隱私運算 / 機密 AI](./15.Privacy_Confidential_AI/README.md) | FL / DP / TEE / Apple PCC |
| 16 | [AI 內容真實性](./16.AI_Content_Authenticity/README.md) | C2PA / SynthID / 浮水印 |
| 17 | [因果機器學習](./17.Causal_ML/README.md) | Pearl SCM / Rubin PO / DoWhy / EconML |
| 18 | [GNN / 圖學習](./18.GNN_Graph_Learning/README.md) | GNN / Graph Transformer / GraphRAG |
| 19 | [Synthetic Data 與資料工程](./19.Synthetic_Data_Engineering/README.md) | Magpie / distilabel / data flywheel |
| 20 | [Generative UI / AI-Native 介面](./20.Generative_UI/README.md) | v0 / Bolt / Lovable / A2UI |
| 21 | [AI 預測學 / 經濟學](./21.AI_Forecasting_Economics/README.md) | Epoch / METR / AI 2027 / RSP |
| 22 | [自動化 AI 研究 / Self-Improving](./22.Self_Improving_AI/README.md) | AI Scientist / AlphaEvolve / RSI |

---

## 🎯 五個系統設計案例(LLM System Design 深入版)

每個 case 模擬 45-60 分鐘 senior-level 白板面試,~3000 字完整推演。

| Case | 規模 | 核心考點 |
|---|---|---|
| [01 Enterprise RAG](./9.面試準備與職業發展/2.系統設計案例/Case_01_Enterprise_RAG_System.md) | 10M docs / 1000 QPS | hybrid search、rerank、GraphRAG、multi-tenancy |
| [02 LLM Gateway](./9.面試準備與職業發展/2.系統設計案例/Case_02_LLM_Gateway_API_Platform.md) | 50K RPS peak | smart routing、prompt+semantic cache、provider fallback |
| [03 Voice Agent](./9.面試準備與職業發展/2.系統設計案例/Case_03_Voice_Agent_Customer_Service.md) | 5K concurrent / p50 < 500ms | SIP、streaming、barge-in、HIPAA/PCI |
| [04 Multi-Agent Research](./9.面試準備與職業發展/2.系統設計案例/Case_04_Multi_Agent_Research_System.md) | 5K tasks/day / p90 < 30min | LangGraph、parallel fan-out、HITL、checkpoint |
| [05 Computer Use SaaS](./9.面試準備與職業發展/2.系統設計案例/Case_05_Computer_Use_SaaS.md) | 10K tasks / 1000 concurrent VM | sandbox、prompt injection 防禦、recording |

五案例串成 **「資料 → 路由 → 互動 → 編排 → 執行」五層完整地圖**(見 Case_05 結尾對照表)。

---

## 🛤 三條學習路徑(對應全景圖 Part 5)

### Path A — Generalist(6 個月廣度)

```
M1: LLM 核心 + Reasoning Models      → Notebook 1 (SFT-LoRA)
M2: RAG + GNN/GraphRAG              → Notebook 3 (Mini GraphRAG)
M3: Agent + Harness Engineering     → Notebook 2 (LangGraph)
M4: AI Coding + Generative UI       → Cursor / Claude Code 重度使用
M5: CV + Multimodal + Voice         → 主題 1.4 DL + 14.Voice
M6: Safety + Provenance + Synthetic → 主題 16 + 19
```

### Path B — LLM 工程師(6 個月深度)

```
M1-2: 三大模型技術報告(DeepSeek-V3 / Llama 4 / Qwen3)+ Notebook 1
M3:   GRPO/RLVR + Notebook 5 (DPO)
M4:   vLLM + EAGLE-3 部署 + Notebook 4
M5:   Magpie/distilabel + AI Scientist
M6:   Long Context + Hybrid Mamba
```

### Path C — 應用工程師(6 個月產品)

```
M1: RAG 基礎 + Magpie synthetic eval
M2: Agent + MCP + Notebook 2
M3: AI Coding + Generative UI
M4: Voice Agent + 多模態生成
M5: Safety + Privacy + Provenance
M6: 選一個垂直 vertical 做端到端 demo
```

更詳細見 [LEARNING_PATHS.md](./LEARNING_PATHS.md) 與 [全景圖 Part 5](./2024-2026_AI完整領域全景圖.md)。

---

## 🔑 核心索引文件

| 文件 | 用途 |
|---|---|
| [2024-2026_AI完整領域全景圖.md](./2024-2026_AI完整領域全景圖.md) | 22 主題俯瞰地圖 + 6 元主題 + 三條路徑 |
| [CONCEPT_MAP.md](./CONCEPT_MAP.md) | 核心概念 → canonical 檔案對照(關係圖式) |
| [FRONTIER_TERMS_INDEX.md](./FRONTIER_TERMS_INDEX.md) | 2026 frontier 熱詞索引(GLOSSARY 補充) |
| [GLOSSARY.md](./GLOSSARY.md) | 80+ 經典術語表(字母索引) |
| [LEARNING_PATHS.md](./LEARNING_PATHS.md) | 4 條詳細學習路徑 |
| [PREREQUISITES.md](./PREREQUISITES.md) | 各 Level 前置知識自測 |
| [QUICKSTART.md](./QUICKSTART.md) | 10 分鐘上手 |
| [SETUP_GUIDE.md](./SETUP_GUIDE.md) | 環境設定 |
| [CHANGELOG.md](./CHANGELOG.md) | 版本紀錄 |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | 貢獻指南 |

### 自我審計檔案(2026-05 完成兩輪)

- [專案爆破總診斷_2026-05.md](./專案爆破總診斷_2026-05.md) — 第一輪審計(10 個 agent 從不同角度挑刺)
- [專案爆破總診斷_第二輪_整合性_2026-05.md](./專案爆破總診斷_第二輪_整合性_2026-05.md) — 第二輪(整合性、跨章節)

> 兩輪都記載修補項與 ROI,作者依此整合本輪所有改動。**舊版 README(2215 行,2025-11)備份於 [docs/archive/README_2025-11_pre_rewrite.md](./docs/archive/README_2025-11_pre_rewrite.md)**。

---

## 💡 心法

1. **不要追完美廣度**。22 個主題不必每個都精通。選 Path A/B/C 其中一條,加 2-3 個 frontier 主題深耕。
2. **理論密度高、實作密度低是 repo 常態問題**。多用 5 份 notebook,跑通了再回頭讀 md 會有頓悟。
3. **phantom-mesh 視角是差異化**。同樣的 RAG / Agent 問題,從 multi-tenant runtime 視角看會多出 5 個工程約束(provider fallback、cost cap、streaming 解析、checkpoint resume、跨平台 build) — 這在面試中是真正的價值點。
4. **面試是溝通,不是知識量**。180+ 題庫不是要你背,而是讓你能在白板上 30 秒抓到問題核心後,用 Case_01-05 的範本展開。
5. **持續更新比一次完美重要**。本 repo 每季校對 frontier 章節;若你發現過時內容,歡迎 PR。

---

## 📝 貢獻、授權、聯絡

- **貢獻**:歡迎 PR,先看 [CONTRIBUTING.md](./CONTRIBUTING.md)
- **授權**:MIT — 學習用途請自由轉載、商業用途請註明出處
- **問題回報**:[GitHub Issues](https://github.com/markl-a/My-AI-Learning-Notes/issues)
- **生態**:[phantom-mesh runtime](https://github.com/markl-a/phantom-mesh)(本 repo 的主專案)

---

> **最後更新**:2026-05-16
>
> 本輪重大改動:新增 22 個 frontier deep-dive(11-22 章)、5 份 Colab notebook、5 個系統設計案例、4 大主題 README 整合、CONCEPT_MAP + FRONTIER_TERMS_INDEX、繁化簡中詞彙修正(~300 個 .md 檔)、跨章節 cross-link 補完。詳見 [CHANGELOG.md](./CHANGELOG.md) v1.2.0。
