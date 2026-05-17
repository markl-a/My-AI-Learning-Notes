# 三方審計報告 / 2026-05

由三個獨立 AI agent / CLI 平行掃描此 repo,從不同切入角度給出優化建議。

## 審計範圍

| 角度 | 工具 | Prompt | 原始報告 |
|---|---|---|---|
| **資訊架構 / 讀者體驗** | Gemini CLI(`-p --yolo`) | [gemini_audit_prompt.md](./gemini_audit_prompt.md) | [gemini_report.md](./gemini_report.md) |
| **工程實踐 / CI / 依賴 / 安全** | Codex CLI(`exec -s workspace-write`) | [codex_audit_prompt.md](./codex_audit_prompt.md) | [codex_report.md](./codex_report.md) |
| **2026 frontier 內容鮮度對齊** | Claude subagent | [claude_audit_prompt.md](./claude_audit_prompt.md) | [claude_subagent_report.md](./claude_subagent_report.md) |

## 合成優化建議

三家共識項目經 verify 後產出在 repo 根目錄:

- **[../../../三方審計優化建議_2026-05.md](../../../三方審計優化建議_2026-05.md)** — P0/P1/P2/P3 + 「agent 過時 / 誇大」分類

## 三句共識診斷

> Gemini(UX):「華麗的建案廣告,不是可入住的房子」
>
> Codex(工程):「CI 看似完整,實際全部不擋錯」
>
> Claude(內容):「結構正確,但鮮度 ≠ 真實度」

## 已執行的修法

見 [CHANGELOG.md `[1.3.0]`](../../../CHANGELOG.md) 與 [`[1.4.0]`](../../../CHANGELOG.md):
- 真實必修 3 件:CI security 真擋錯 / SFT API 升 SFTConfig / langchain 版本統一
- Dependabot 75 → 22
- CI 健康化(safety→pip-audit / 移除 build / notebook 改 validate)
- 11 個壞 notebook schema 修復
- 12 個 frontier 章節 + 6 個 demo 共 18 處 disclaimer

## 重要決策:agent 已過時 / 誇大項

verify 後**不動**的項目(避免依賴過時報告誤動):
- 主題 2 第 6/8 章 README「0 行空殼」→ 實際 93 + 136 行(早已補)
- `.DS_Store`「32 個 tracked」→ 實際 2 個
- 全景圖「0 出度黑洞」→ 實際已有 12 個連結
- 雙軌目錄重命名 → 對個人筆記 repo 為「炸藥級」改動,ROI 差
- eval/exec 改用 simpleeval → 教學脈絡保留,改加 README ⚠️ 警告
- 新增 23-27 章(Mech Interp / Agent Eval / Inference Econ / AI4AI / RLVR)→ Claude 自己也說「停止加新內容」,放 backlog
