# 9. 面試準備與職業發展

> **定位**:LLM/AI 工程師面試準備 **+ 職業發展** 全方位資源庫。
> 三個子目錄構成完整的「廣度題庫 → 深度案例 → 職涯規劃」三角。
>
> **與其他章節差異**:1-3 章是知識體系、11-22 是 frontier briefing,**本章是「把知識轉成 offer」的應用層**。
>
> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) Part 5 三條學習路徑的「最後一哩」。

---

## 📚 三大支柱(按準備順序)

### [1. LLM 面試題庫](./1.LLM面試題庫/)(廣度)

5 份題庫共 **180+ 題**,涵蓋從基礎到實戰場景:

| 題庫 | 題目數 | 重點 |
|---|---|---|
| [`01_基礎概念題`](./1.LLM面試題庫/01_基礎概念題.md) | 35 | Transformer / Attention / Tokenizer / Embedding / Fine-tuning 等核心概念 |
| [`02_架構設計題`](./1.LLM面試題庫/02_架構設計題.md) | 25 | 企業級 RAG、推論服務、訓練 pipeline 等架構決策 |
| [`03_程式碼實現題`](./1.LLM面試題庫/03_程式碼實現題.md) | 35 | 6 大類:Attention 實作、Tokenizer、訓練/微調、推論優化、RAG/Agent、系統小型 code |
| [`04_系統設計題`](./1.LLM面試題庫/04_系統設計題.md) | 12 | 大型系統設計速覽版(每題 200-500 字) |
| [`05_實戰場景題`](./1.LLM面試題庫/05_實戰場景題.md) | 40 | 7 大區:模型選型、RAG 生產、Agent 故障、推論延遲、訓練問題、安全合規、跨平台 |

**特色**:**11+ 題嵌入 phantom-mesh 真實開發場景**(multi-agent 協調、provider fallback、streaming SSE 解析、cost tracking、跨平台 build chain),不再是泛用教科書題。

### [2. 系統設計案例](./2.系統設計案例/)(深度)

5 個完整 case study,每個模擬 45-60 分鐘 senior-level 系統設計面試:

| Case | 規模 / 場景 | 核心考點 |
|---|---|---|
| [Case_01 Enterprise RAG](./2.系統設計案例/Case_01_Enterprise_RAG_System.md) | 10M docs / 100K users / 1000 QPS | hybrid search、rerank、GraphRAG、multi-tenancy |
| [Case_02 LLM Gateway](./2.系統設計案例/Case_02_LLM_Gateway_API_Platform.md) | 50K RPS peak / 860B tokens/day | smart routing、prompt+semantic cache、provider fallback |
| [Case_03 Voice Agent](./2.系統設計案例/Case_03_Voice_Agent_Customer_Service.md) | 5K concurrent / p50 < 500ms | SIP、STT/LLM/TTS pipeline、HIPAA/PCI |
| [Case_04 Multi-Agent Research](./2.系統設計案例/Case_04_Multi_Agent_Research_System.md) | 5K tasks/day / p90 < 30min | LangGraph supervisor、parallel fan-out、HITL |
| [Case_05 Computer Use SaaS](./2.系統設計案例/Case_05_Computer_Use_SaaS.md) | 10K tasks/day / 1000 concurrent VM | sandbox、credential vault、prompt injection 防禦 |

5 個 case 串成 **「資料 → 路由 → 互動 → 編排 → 執行」五層完整地圖**(見 Case_05 結尾對照表)。

### [3. 職業發展指南](./3.職業發展指南/)(規劃)

4 份指南涵蓋從求職到持續學習:

| 指南 | 主題 |
|---|---|
| [`01_AI_Engineer_vs_ML_Engineer_職涯路徑`](./3.職業發展指南/01_AI_Engineer_vs_ML_Engineer_職涯路徑.md) | 兩條路差異、新興職位、轉職路徑、矽谷+台灣薪資表 |
| [`02_履歷與作品集打造`](./3.職業發展指南/02_履歷與作品集打造.md) | 履歷格式、殺手項目、GitHub 經營、面試前清單 |
| [`03_行為面試_STAR_範例`](./3.職業發展指南/03_行為面試_STAR_範例.md) | **20 題 STAR 範例,5 題用 phantom-mesh 真實場景** |
| [`04_AI_工程師持續學習指南`](./3.職業發展指南/04_AI_工程師持續學習指南.md) | 每週/月/季/年節奏、與本 repo 22 主題地圖搭配 |

---

## 🎯 推薦準備時程

### 6 週衝刺路徑(已有 1-2 年 SWE / ML 經驗)

| 週 | 主題 |
|---|---|
| 1 | 通讀 `01_基礎概念題` + `02_架構設計題` 建立詞彙 |
| 2 | 動手 `03_程式碼實現題` 全部 35 題(白板手寫) |
| 3 | 讀 Case_01 RAG + 讀 Case_02 LLM Gateway,各蒙題重畫一次 |
| 4 | 讀 Case_03 Voice + Case_04 Multi-Agent + Case_05 Computer Use |
| 5 | 過 `05_實戰場景題` 40 題(每題講出 3 個 trade-off) |
| 6 | 行為面試 STAR 演練 + 履歷 + GitHub 整理 + Mock interview |

### 12 週紮實路徑(背景轉職 / 剛畢業)

前 6 週先補 [全景圖 Path C](../2024-2026_AI完整領域全景圖.md)(LLM 應用工程師),後 6 週進入上面 6 週衝刺。

### 3 個月深度準備(衝 frontier lab)

加做 [全景圖 Path B](../2024-2026_AI完整領域全景圖.md)(LLM 工程師)中的 R1-Zero 復現、vLLM + EAGLE-3 部署、Magpie pipeline,把 GitHub 補上 2-3 個能 demo 的 frontier 專案。

---

## 🧭 與 repo 其他章節的關聯

```
基礎知識(主題 1-3)→ 進階深度(11-22)→ 面試應用(本章)
                                          │
                              ┌───────────┼───────────┐
                              ↓           ↓           ↓
                          題庫(廣度)  案例(深度)  職涯(規劃)
```

每個面試題目都對應到 repo 內某個 deep-dive 章節:
- 推理引擎題 → [`../11.AI_Hardware_Compute/`](../11.AI_Hardware_Compute/README.md) + [`../2.深入LLM模型工程與LLM運維/8.模型部署與運維/`](../2.深入LLM模型工程與LLM運維/8.模型部署與運維/README.md)
- RAG 題 → [`../3.LLM應用工程/4.(RAG) 基礎/`](../3.LLM應用工程/4.(RAG)%20基礎/) + [`../3.LLM應用工程/5.進階 RAG 與多元資料檢索/`](../3.LLM應用工程/5.進階%20RAG%20與多元資料檢索/)
- Agent 題 → [`../3.LLM應用工程/3.Agent/`](../3.LLM應用工程/3.Agent/) + [`../22.Self_Improving_AI/`](../22.Self_Improving_AI/README.md)
- Voice 題 → [`../14.Voice_Audio_AI/`](../14.Voice_Audio_AI/README.md)
- 因果評估 / hallucination → [`../17.Causal_ML/`](../17.Causal_ML/README.md)
- 安全合規 → [`../15.Privacy_Confidential_AI/`](../15.Privacy_Confidential_AI/README.md) + [`../16.AI_Content_Authenticity/`](../16.AI_Content_Authenticity/README.md)
- AI 預測學 / RSP / 能力評估 → [`../21.AI_Forecasting_Economics/`](../21.AI_Forecasting_Economics/README.md)

---

## 🌐 phantom-mesh 角色定位

本章兌現主 README 對 **phantom-mesh 真實案例整合** 的承諾。涉及 phantom-mesh 開發實際遇到的問題:

| 領域 | 對應題目 / 案例 |
|---|---|
| **multi-agent 協調** | `02_架構設計題` Q1、`05_實戰場景題` Q17/Q20、Case_04 |
| **provider fallback** | `05_實戰場景題` Q1/Q19/Q35、Case_02 §5.7、Case_03 |
| **streaming SSE 解析** | `03_程式碼實現題` Q6/Q8、Case_02 §5.6、Case_03 |
| **cost tracking** | `02_架構設計題` Q2、`05_實戰場景題` Q1/Q36、Case_02 §5.5 |
| **跨平台 build chain** | `05_實戰場景題` Q40 |

**STAR 範例 5 題**(`03_行為面試_STAR_範例.md` Q6-Q10)直接用 phantom-mesh 真實開發場景作為「我做過的專案」素材,可直接套用到面試敘事。

---

## 💡 心法

1. **面試是溝通能力,不是知識量**。多用 "我會 X,因為 trade-off 是 Y vs Z,所以選 X" 的句型。
2. **題庫廣度 + 案例深度 + STAR 真實感** 三件套缺一不可。只刷題會被追問 "你做過嗎";只講案例會被質疑 "理論基礎";只談履歷會被問 "code 寫得怎樣"。
3. **phantom-mesh 是你的差異化**——多數候選人只有 LangChain demo,你有「真正在生產環境踩過 SSE 解析 / provider fallback / cost cap 三個雷」。
4. **準備兼顧** frontend(技術深度)與 backend(職涯故事)。面試官 60 分鐘看技術、5 分鐘看人,但**那 5 分鐘決定 offer 等級**。
5. **不要追完美**——選 3 個 case 深度準備(對應你想去的公司類型)勝過 5 個都半生熟。

---

## 🔗 快速連結

- 返回 [主 README](../README.md)
- [2024-2026 AI 完整領域全景圖](../2024-2026_AI完整領域全景圖.md)(廣度地圖)
- [CONCEPT_MAP.md](../CONCEPT_MAP.md)(概念與檔案對照)
- [FRONTIER_TERMS_INDEX.md](../FRONTIER_TERMS_INDEX.md)(2026 熱詞索引)
- [GLOSSARY.md](../GLOSSARY.md)(80+ 術語表)

**祝面試順利。**
