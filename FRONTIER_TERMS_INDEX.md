# Frontier Terms Index — 2026 前沿術語索引

> [GLOSSARY.md](./GLOSSARY.md) 是**基礎術語表**(已普及、教科書級的經典詞)。本檔是補充:**2024-2026 frontier 新詞**,GLOSSARY 尚未收錄但近兩年在 paper、技術部落格、業界討論中高頻出現的詞。
> 每條格式:定義(3-5 行) → 權威章節 → 全景圖定位。
> 詞普及後將逐步遷移到 GLOSSARY,本檔保留指標。

---

## A2A (Agent-to-Agent Protocol)
Google 主導、2025/Q4 捐贈 Linux Foundation 的 agent 間通訊協定;與 MCP 互補(MCP = agent ↔ tool,A2A = agent ↔ agent)。目標:跨組織 agent 能互相發現能力、協商分工、簽署任務。
→ 權威章節:待補(候選 [3.LLM應用工程/3.Agent/](./3.LLM應用工程/3.Agent/))
→ 全景圖定位:[#13 Agent 章節](./2024-2026_AI完整領域全景圖.md)

## A2UI (Agent-to-User Interface)
Google 2025/12 推出的跨框架 generative UI 標準 (v0.9)。讓 agent 不再只回文字而可以動態生成可互動 UI 元件 (slider/form/canvas),React/Vue/Svelte 皆可消費同一份協定。
→ 權威章節:[20.Generative_UI/README.md §5](./20.Generative_UI/README.md)
→ 全景圖定位:[#19 Generative UI](./2024-2026_AI完整領域全景圖.md)

## SGLang
LMSYS 出品的 LLM 推論引擎,核心兩大武器:RadixAttention (跨請求共享 KV-cache prefix 的基數樹) + Frontend DSL (Python 控制流可內嵌 LLM 呼叫)。2025-2026 為 reasoning model / agent workflow 首選引擎。
→ 權威章節:[2025_AI框架與工具生態.md](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/2025_AI框架與工具生態.md)
→ 全景圖定位:[#14 Harness Engineering](./2024-2026_AI完整領域全景圖.md)

## GRPO (Group Relative Policy Optimization)
DeepSeek 2024 提出,取消 critic / value network、用同 prompt 多次採樣的 group 估 advantage。顯著降低 RLHF 訓練成本,R1 系列、Qwen3 reasoning、Kimi K1.5 等 reasoning model 已標配。
→ 權威章節:[LLM_Core_Training_2024-2026.md §3](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md)(實作 script 待補)
→ 全景圖定位:[#4 推理模型](./2024-2026_AI完整領域全景圖.md)

## DAPO / RLVR (Reinforcement Learning from Verifiable Rewards)
DAPO 為 ByteDance Seed 2025 開源算法,在 GRPO 基礎上加 dynamic sampling 與 clip-higher;RLVR 則是更廣的範式:用「可自動驗證的訊號」(數學題答案、單元測試) 取代人類偏好標註。reasoning model 訓練主流路線。
→ 權威章節:[LLM_Core_Training_2024-2026.md §3](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md)
→ 全景圖定位:[#4 推理模型](./2024-2026_AI完整領域全景圖.md)

## EAGLE-3
2024-2025 起最快的投機解碼變體 (Speculative Decoding),用多 token feature 預測 + dynamic draft tree,vLLM / SGLang / TensorRT-LLM 都已整合。配合 chunked prefill 可達 2.5-6× 端到端加速。
→ 權威章節:[2.文字生成與解碼策略/](./2.深入LLM模型工程與LLM運維/2.文字生成與解碼策略/) §4.4(待補實作)

## SAM 3 / SAM 3.1 (Segment Anything 3)
Meta 2025/11,從 mask prompt 升級為 **Promptable Concept Segmentation** ── 給文字 (例如 "all yellow taxis") 或範例圖片就能找出影像中所有實例。影像分割從 SAM 1 的「點/框 prompt」進化到「概念 prompt」。
→ 權威章節:[CV_全景_2024-2026.md §1](./1.從AI到LLM基礎/4.DL/CV_全景_2024-2026.md)
→ 全景圖定位:[#5 CV 全景](./2024-2026_AI完整領域全景圖.md)

## Claude Code
Anthropic 2024 推出的終端 agent CLI,Sonnet 4.6 / Opus 4.7 後成為 SWE-Bench Verified 領跑工具 (75%+)。是 vibe coding 與企業 internal agentic workflow 的主力。
→ 權威章節:[Vibe_Coding_與_AIGC_生成式創作完整學習指南.md](./5.AI研究前沿_2024-2025/Vibe_Coding_與_AIGC_生成式創作完整學習指南.md)
→ 主要實戰筆記:[Day02,03 從頭熟練 Claude Code.md](./4.相關的更新Blog/Day02,03%20從頭熟練%20Claude%20Code.md) + [1015_熟練_Claude_Code.md](./4.相關的更新Blog/1015_熟練_Claude_Code.md)
→ 全景圖定位:[#17 AI Coding](./2024-2026_AI完整領域全景圖.md)

## Devin (Cognition Labs)
2024 首個 autonomous coding agent 商用化嘗試;2026/Q1 推出 Devin 2.2,加入 managed sub-agent (一個 Devin 可派遣多個子 Devin 平行處理 PR)。長 horizon 任務代表作。
→ 權威章節:待補
→ 全景圖定位:[#17 AI Coding](./2024-2026_AI完整領域全景圖.md)

## Sora 2 (OpenAI)
2025/Q4 影片生成旗艦,最長 15 秒 1080p clip,首次支援同步生成原生音訊與多 shot 一致性。對 Runway Gen-4 / Pika / Kling / Veo 3 形成競爭壓力。
→ 權威章節:[Multimodal_Generation_2024-2026.md §4](./1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md)
→ 全景圖定位:[#6 多模態生成](./2024-2026_AI完整領域全景圖.md)

## YOLO11 / YOLO26
Ultralytics 2024-2026 演進系列。YOLO11 (2024/09) 為 Anchor-free + C3k2 模組;**YOLO26 (2026/01) 原生 NMS-free,nano 模型 CPU 推論延遲降 43%**,首次官方支援端到端 batch=1 邊緣部署。
→ 權威章節:[CV_全景_2024-2026.md §1](./1.從AI到LLM基礎/4.DL/CV_全景_2024-2026.md)
→ 注意:repo 既有 [4.DL/04.Ultralytics](./1.從AI到LLM基礎/4.DL/04.Ultralytics/) 仍是 YOLOv8 範例,待更新到 YOLO11/26。

## pgvector
PostgreSQL extension,2026 向量庫五強之一 (vs. Pinecone / Weaviate / Milvus / Qdrant)。優勢:資料庫即向量庫,不必額外運維;0.7+ 後支援 HNSW + binary quantization,效能追上專用引擎。
→ 權威章節:待補(候選 [向量資料庫完整比較指南.md](./3.LLM應用工程/5.進階%20RAG%20與多元資料檢索/向量資料庫完整比較指南.md))
→ 全景圖定位:[#11 RAG / 向量資料庫](./2024-2026_AI完整領域全景圖.md)

## LangGraph
LangChain 系的 stateful agent framework,2025-2026 Agent 框架龍頭。核心抽象:graph of state nodes + checkpointer,原生支援 human-in-the-loop、time travel debugging。已被 LinkedIn / Uber / Replit 等大廠生產採用。
→ 權威章節:[3.LLM應用工程/3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md](./3.LLM應用工程/3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md)
→ 全景圖定位:[#13 Agent](./2024-2026_AI完整領域全景圖.md)

## Mamba / SSM (State Space Models)
Albert Gu 2023 推出,線性時間 sequence model,理論上長 context 不再 O(N²)。2024-2025 Hybrid (Jamba / Granite 4 / Zamba2) 將 Mamba block 與 Attention block 混搭,在企業推理場景進入生產;純 Mamba 在 fine-grained recall 仍不如 Transformer。
→ 權威章節:[LLM_Core_Training_2024-2026.md §1](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md)(獨立 deep-dive 待補)
→ 全景圖定位:[#3 Attention 變體](./2024-2026_AI完整領域全景圖.md)

---

## 維護說明

- **GLOSSARY.md** 維護穩定術語(已普及,教科書級)。
- **本檔 (FRONTIER_TERMS_INDEX.md)** 維護 frontier 詞(2024-2026 才浮現)。
- 詞普及(進入主流教材 / 連續 2 年論文穩定使用)後逐步遷移到 GLOSSARY。
- 新增條目原則:出現 ≥3 篇 frontier paper、或 ≥1 家 frontier lab 官方部落格採用,即可入選。
- 條目過時(技術被取代且 ≥6 個月無新 paper)則歸入「歷史術語」附錄(待建)。
