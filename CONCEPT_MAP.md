# Concept Map — 核心概念權威來源對照

> 本檔是 [GLOSSARY.md](./GLOSSARY.md) 的關係圖補充:GLOSSARY 是字母索引(每詞一行定義),CONCEPT_MAP 是主題關係圖(每概念對應到 canonical 檔案)。
> 讀者找某個概念的權威來源,先查本檔。對於 2024-2026 才浮現的 frontier 新詞,請查 [FRONTIER_TERMS_INDEX.md](./FRONTIER_TERMS_INDEX.md)。

---

## L1 硬體層

| 概念 | Canonical 檔案 | 次要參考 |
|---|---|---|
| GPU / HBM / NVLink | [11.AI_Hardware_Compute/README.md](./11.AI_Hardware_Compute/README.md) | [3.LLM應用工程/6.推論優化/](./3.LLM應用工程/6.推論優化/) |
| Neuromorphic Computing | [11.AI_Hardware_Compute/Neuromorphic_Computing.md](./11.AI_Hardware_Compute/Neuromorphic_Computing.md) | — |
| FP8 / FP4 數值格式 | [11.AI_Hardware_Compute/README.md](./11.AI_Hardware_Compute/README.md) | [量化與推論優化技術詳解.md](./2.深入LLM模型工程與LLM運維/7.模型壓縮與優化/量化與推論優化技術詳解.md) |

---

## L2 模型層

| 概念 | Canonical | 次要 |
|---|---|---|
| Transformer 基礎 | [DL Path 第 14 章 預訓練](./1.從AI到LLM基礎/4.DL/00.DL_Path/14_自然語言處理：預訓練/) | [4.DL/06.Paper_with_code/](./1.從AI到LLM基礎/4.DL/06.Paper_with_code/) |
| MoE (Mixture of Experts) | [Mixture_of_Experts_架構詳解.md](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/Mixture_of_Experts_架構詳解.md) | [LLM_Core_Training_2024-2026.md](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md) §1 |
| Attention 變體 (FlashAttn / Linear / SSM) | [注意力機制最新變體與優化.md](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/注意力機制最新變體與優化.md) | — |
| RoPE / 長 context | [10.1_位置嵌入與上下文延展.md](./2.深入LLM模型工程與LLM運維/10.進階話題/10.1_位置嵌入與上下文延展.md) | LLM_Core_Training §5 |
| Reasoning Models | [12.推理模型應用/](./2.深入LLM模型工程與LLM運維/12.推理模型應用/) + [推理模型_Reasoning_Models_深度解析.md](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/推理模型_Reasoning_Models_深度解析.md) | LLM_Core_Training §2 |
| CV 全景 | [CV_全景_2024-2026.md](./1.從AI到LLM基礎/4.DL/CV_全景_2024-2026.md) | — |
| Diffusion / 多模態生成 | [Multimodal_Generation_2024-2026.md](./1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md) | [3.LLM應用工程/10.多模態生成/](./3.LLM應用工程/10.多模態生成/) |
| VLA / Robotics | [13.Robotics_Embodied_AI/README.md](./13.Robotics_Embodied_AI/README.md) | CV 全景 §7 |
| 時序 / 表格 Foundation Model | [Time_Series_Tabular_FM_2024-2026.md](./1.從AI到LLM基礎/3.ML_&_Data_Analysis/Time_Series_Tabular_FM_2024-2026.md) | — |
| GNN | [18.GNN_Graph_Learning/README.md](./18.GNN_Graph_Learning/README.md) + [00_入門前置.md](./18.GNN_Graph_Learning/00_入門前置.md) | 12.AI_For_Science (等變網路) |
| AI4Science | [12.AI_For_Science/README.md](./12.AI_For_Science/README.md) + [00_入門前置.md](./12.AI_For_Science/00_入門前置.md) | 18.GNN |
| Causal ML | [17.Causal_ML/README.md](./17.Causal_ML/README.md) + [00_入門前置.md](./17.Causal_ML/00_入門前置.md) | — |
| Mamba / SSM | [LLM_Core_Training_2024-2026.md §1](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md) | FRONTIER_TERMS_INDEX |

---

## L3 訓練 / 對齊層

| 概念 | Canonical | 次要 |
|---|---|---|
| Pre-training | [4.模型預訓練與預訓練模型選擇/](./2.深入LLM模型工程與LLM運維/4.模型預訓練與預訓練模型選擇/) | LLM_Core_Training §4 |
| SFT (Supervised Fine-Tuning) | [5.監督微調 (SFT)/](./2.深入LLM模型工程與LLM運維/5.監督微調%20(SFT)/) | — |
| LoRA / QLoRA / PEFT | [進階微調策略_LoRA_QLoRA.md](./2.深入LLM模型工程與LLM運維/5.監督微調%20(SFT)/進階微調策略_LoRA_QLoRA.md) | [7.模型壓縮與優化/](./2.深入LLM模型工程與LLM運維/7.模型壓縮與優化/) (低秩分解) |
| RLHF / DPO 家族 | [RLHF與偏好對齊完整指南.md](./2.深入LLM模型工程與LLM運維/6.偏好對齊%20(Alignment)%20技術/RLHF與偏好對齊完整指南.md) | LLM_Core_Training §3 |
| GRPO / DAPO / RLVR | [LLM_Core_Training_2024-2026.md §3](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md) (目前無獨立 deep-dive,待補) | [FRONTIER_TERMS_INDEX](./FRONTIER_TERMS_INDEX.md) |
| 現代對齊方法 | [11.現代對齊方法2024-2025/](./2.深入LLM模型工程與LLM運維/11.現代對齊方法2024-2025/) | — |
| Synthetic Data | [19.Synthetic_Data_Engineering/](./19.Synthetic_Data_Engineering/README.md) | [3.資料集準備與建立/](./2.深入LLM模型工程與LLM運維/3.資料集準備與建立/) |
| Self-Improving | [22.Self_Improving_AI/README.md](./22.Self_Improving_AI/README.md) | 12.推理模型應用 |

---

## L4 推理 / 部署層

| 概念 | Canonical | 次要 |
|---|---|---|
| 量化 (GPTQ / AWQ / GGUF / FP8) | [量化與推論優化技術詳解.md](./2.深入LLM模型工程與LLM運維/7.模型壓縮與優化/量化與推論優化技術詳解.md) | 11.AI_Hardware (FP8/FP4 硬體) |
| vLLM / SGLang / TensorRT-LLM | [1.LLM 部署/](./3.LLM應用工程/1.LLM%20部署/) | [8.模型部署與運維/](./2.深入LLM模型工程與LLM運維/8.模型部署與運維/) |
| FlashAttention | [注意力機制最新變體與優化.md §1](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/注意力機制最新變體與優化.md) | — |
| Speculative Decoding / EAGLE-3 | [2.文字生成與解碼策略/](./2.深入LLM模型工程與LLM運維/2.文字生成與解碼策略/) | [FRONTIER_TERMS_INDEX](./FRONTIER_TERMS_INDEX.md) |
| KV-Cache 優化 | [3.LLM應用工程/6.推論優化/2.KV-Cache/](./3.LLM應用工程/6.推論優化/2.KV-Cache/) | — |
| 部署運維與成本優化 | [模型部署與運維實戰指南.md](./2.深入LLM模型工程與LLM運維/8.模型部署與運維/模型部署與運維實戰指南.md) | [成本優化與Token管理.md](./2.深入LLM模型工程與LLM運維/8.模型部署與運維/成本優化與Token管理.md) |

---

## L5 應用 / 介面層

| 概念 | Canonical | 次要 |
|---|---|---|
| RAG 基礎 | [4.(RAG) 基礎/](./3.LLM應用工程/4.(RAG)%20基礎/) | — |
| Advanced / Graph / Agentic RAG | [5.進階 RAG 與多元資料檢索/](./3.LLM應用工程/5.進階%20RAG%20與多元資料檢索/) | 18.GNN (GraphRAG 圖結構) |
| 向量資料庫 | [向量資料庫完整比較指南.md](./3.LLM應用工程/5.進階%20RAG%20與多元資料檢索/向量資料庫完整比較指南.md) | FRONTIER_TERMS_INDEX (pgvector) |
| Agent / ReAct / LangGraph | [3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md](./3.LLM應用工程/3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md) | [Agent框架選擇決策指南.md](./3.LLM應用工程/3.Agent/Agent框架選擇決策指南.md) |
| MCP (Model Context Protocol) | [11.MCP協議與工具調用/](./3.LLM應用工程/11.MCP協議與工具調用/) | 全景圖 #13 |
| Voice Agent | [14.Voice_Audio_AI/README.md](./14.Voice_Audio_AI/README.md) | [10.多模態生成/6.語音與音訊AI/](./3.LLM應用工程/10.多模態生成/6.語音與音訊AI/) |
| AI Coding / Vibe Coding | [Vibe_Coding_與_AIGC_生成式創作完整學習指南.md](./5.AI研究前沿_2024-2025/Vibe_Coding_與_AIGC_生成式創作完整學習指南.md) | [13.AI程式助手/](./2.深入LLM模型工程與LLM運維/13.AI程式助手/) |
| Generative UI | [20.Generative_UI/README.md](./20.Generative_UI/README.md) | — |
| 提示工程與結構化輸出 | [12.進階提示工程與結構化輸出/](./3.LLM應用工程/12.進階提示工程與結構化輸出/) | — |

---

## L6 治理 / 信任層

| 概念 | Canonical | 次要 |
|---|---|---|
| AI Safety / Eval / Alignment | [AI安全與對齊指南.md](./2.深入LLM模型工程與LLM運維/10.進階話題/AI安全與對齊指南.md) | LLM_Core_Training 部分節 |
| AI 倫理與法規 | [AI倫理與法規指南.md](./2.深入LLM模型工程與LLM運維/10.進階話題/AI倫理與法規指南.md) | 21.AI_Forecasting_Economics |
| C2PA / SynthID / Provenance | [16.AI_Content_Authenticity/README.md](./16.AI_Content_Authenticity/README.md) | — |
| FL / DP / TEE / Privacy | [15.Privacy_Confidential_AI/README.md](./15.Privacy_Confidential_AI/README.md) + [00_入門前置.md](./15.Privacy_Confidential_AI/00_入門前置.md) | — |
| AI Forecasting / Economics | [21.AI_Forecasting_Economics/README.md](./21.AI_Forecasting_Economics/README.md) + [00_入門前置.md](./21.AI_Forecasting_Economics/00_入門前置.md) | — |
| LLM 安全 (Prompt Injection / Jailbreak) | [8.LLM安全與防禦/](./3.LLM應用工程/8.LLM安全與防禦/) | AI 安全與對齊指南 |

---

## 元主題 (Meta-Themes,跨所有層)

| 主題 | 散落位置 |
|---|---|
| Test-time Compute Scaling | [全景圖 Part 3 M1](./2024-2026_AI完整領域全景圖.md) + 12.推理模型應用 |
| 合成資料 + 蒸餾飛輪 | [全景圖 M2](./2024-2026_AI完整領域全景圖.md) + 19.Synthetic_Data + 22.Self_Improving_AI |
| Multimodal 統一 | [全景圖 M3](./2024-2026_AI完整領域全景圖.md) + Multimodal_Generation_2024-2026 |
| Agent 化 | [全景圖 M4](./2024-2026_AI完整領域全景圖.md) + 13/14/20/22 各章 |
| 能效危機 | [全景圖 M5](./2024-2026_AI完整領域全景圖.md) + 11.AI_Hardware |
| Open vs Closed 地緣 | [全景圖 M6](./2024-2026_AI完整領域全景圖.md) |

---

## 使用說明

- **找概念權威來源**:看「Canonical」欄。
- **找相關章節**:看「次要」欄。
- **想看整片地圖**:看 [2024-2026_AI完整領域全景圖.md](./2024-2026_AI完整領域全景圖.md)。
- **想查字母順序定義**:看 [GLOSSARY.md](./GLOSSARY.md) (穩定基礎詞)。
- **想查 frontier 新詞**:看 [FRONTIER_TERMS_INDEX.md](./FRONTIER_TERMS_INDEX.md) (2024-2026 浮現的詞)。
- **學習路徑**:看 [LEARNING_PATHS.md](./LEARNING_PATHS.md)。
- **入門前置條件**:看 [PREREQUISITES.md](./PREREQUISITES.md) (整體) 或各 deep-dive 的 `00_入門前置.md`。
