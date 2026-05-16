# 6. 偏好對齊 (Alignment) 技術

> 章節定位:LLM 對齊技術全景 — 從經典 RLHF 到 2024-2026 的 DPO 家族、GRPO/DAPO 與 RLVR。
> 此章節是 SFT 完成後讓模型「貼近人類偏好」與「具備推理能力」的關鍵環節。

---

## 📌 章節地位

對齊 (Alignment) 是把已具備語言能力的 base model / SFT model,進一步調整到符合人類價值觀、減少有害輸出、並提升任務表現的階段。2024-2026 期間此領域變化極大,從 RLHF (PPO) 一統江山,演進到 DPO 家族百花齊放,再到 GRPO/DAPO/RLVR 驅動的「推理模型」革命 (o1、DeepSeek-R1)。

學習建議的先決條件:
- 已了解 SFT (見 `../5.監督微調 (SFT)/`)
- 熟悉 reward 概念與基本 RL 術語 (policy / value / advantage)
- 熟悉 KL divergence 概念

---

## 📚 涵蓋範圍

| 類別 | 方法 | 備註 |
|---|---|---|
| **經典 RL-based** | RLHF (PPO) | OpenAI / Anthropic 的原始路線 |
| **DPO 家族** | DPO、IPO、KTO、ORPO、SimPO、CPO | 直接優化偏好,無需 reward model |
| **群組相對 RL** | GRPO、DAPO、RLOO | DeepSeek-R1 / Qwen3 用於推理 |
| **可驗證獎勵** | RLVR (Rule-based Reward) | 數學、code 等有 ground truth 任務 |
| **過程獎勵** | PRM (Process Reward Model) | 對推理鏈每一步給分 |
| **約束類** | Constitutional AI、RLAIF | Anthropic / Google 路線 |
| **混合** | Iterative DPO、Online DPO | 結合 SFT + DPO 多輪迭代 |

---

## 🗂️ 本章文件清單

### 理論與綜述
| 文件 | 內容重點 |
|---|---|
| [`RLHF與偏好對齊完整指南.md`](./RLHF與偏好對齊完整指南.md) | RLHF / DPO / KTO / IPO 理論與實作範例(完整教學) |
| [`2025_LLM訓練與優化技術.md`](./2025_LLM訓練與優化技術.md) | 2025 訓練優化趨勢,含 GRPO / DAPO 入門 |
| [`DPO家族公式速查.md`](./DPO家族公式速查.md) | DPO / IPO / KTO / ORPO / SimPO loss formula 速查(從 11 章合併進來) |

### 2026-05 新增 deep-dive(實戰導向)
| 文件 | 內容重點 |
|---|---|
| [`DPO_SimPO_ORPO_對比實驗.md`](./DPO_SimPO_ORPO_對比實驗.md) | 5 種演算法統一資料集對比、trl code 60 行、選型決策、CMU 2025 controlled study 警告 |
| [`GRPO_DAPO_RLVR_實戰.md`](./GRPO_DAPO_RLVR_實戰.md) | DeepSeek-R1 五階段 pipeline、GRPO 數學推導、80 行 TRL GRPOTrainer + GSM8K 範例、6 類生產陷阱 |
| [`PRM_訓練實作.md`](./PRM_訓練實作.md) | PRM vs ORM、Math-Shepherd 自動標註、weak-to-strong、50 行 trl + custom reward head、與 GRPO 整合 |

---

## 🔗 與其他章節的關係

- **緊接前章**:[`../5.監督微調 (SFT)/`](../5.監督微調%20(SFT)/) — 對齊的起點通常是 SFT 後的 checkpoint
- **~~進階延伸 (已合併)~~**:原 [`../11.現代對齊方法2024-2025/`](../11.現代對齊方法2024-2025/) 內容已於 2026-05 併入本章(見 `DPO家族公式速查.md`)
- **應用銜接**:[`../12.推理模型應用/`](../12.推理模型應用/) — GRPO/RLVR 訓練出的推理模型如何使用
- **核心訓練流程**:[`../1.LLM 基礎與架構/LLM_Core_Training_2024-2026.md`](../1.LLM%20基礎與架構/LLM_Core_Training_2024-2026.md) Section 3 詳述 alignment 在完整 training pipeline 中的位置
- **領域全景**:[`../../2024-2026_AI完整領域全景圖.md`](../../2024-2026_AI完整領域全景圖.md) #20 對齊章節,提供宏觀視角

---

## 📈 建議學習順序

1. **理論基礎**:先讀 `RLHF與偏好對齊完整指南.md`,搞懂 RLHF 為什麼需要 RM、為什麼需要 KL penalty
2. **DPO 線**:理解為什麼 DPO 能繞過 reward model — 對 Bradley-Terry 模型作 closed-form 推導
3. **2024 後新方法**:依序看 ORPO (SFT+DPO 合一) → SimPO (length-normalized) → KTO (binary feedback)
4. **推理對齊**:跳到 GRPO/DAPO — 這是 o1 / DeepSeek-R1 / Qwen3-Reasoning 的關鍵
5. **實作**:目前範例 code 缺口較大,建議參考 [TRL](https://github.com/huggingface/trl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[verl](https://github.com/volcengine/verl) 等開源庫

---

## ⚠️ 已知缺口 (Known Gaps)

- ✅ GRPO/DAPO/RLVR 理論與 80 行 TRL script:**已補**(`GRPO_DAPO_RLVR_實戰.md`)
- ✅ DPO/SimPO/ORPO 對比實驗:**已補**(`DPO_SimPO_ORPO_對比實驗.md`)
- ✅ PRM 訓練流程 + Math-Shepherd:**已補**(`PRM_訓練實作.md`)
- ✅ 與 11 章合併:**已完成**(2026-05)
- ⏳ **仍缺**:可直接 run 的完整 `.ipynb`(本章的 md 仍以「教學 + script snippet」為主,差一份能在 Colab/Kaggle 一鍵跑通的端到端 notebook)
- ⏳ **仍缺**:Constitutional AI 完整流程 + AI feedback dataset 製作 hands-on
- ⏳ **仍缺**:reward hacking 真實案例集(對應全景圖 #4 警告)

歡迎透過 PR 補充上述缺口。

---

## 🎯 學習產出檢核

讀完本章後,你應該能回答:

- [ ] RLHF 中 reward model 為何容易 reward hacking?如何緩解?
- [ ] DPO 推導中 reward 是怎麼被「消掉」的?
- [ ] GRPO 相比 PPO 省了什麼計算?為什麼適合大模型?
- [ ] RLVR 為何只適用於數學/code 而非開放式對話?
- [ ] DAPO 在 GRPO 上做了哪些改進 (clip-higher、token-level loss)?
