# 13. AI 程式助手 (AI Coding Assistants)

> **定位**:把「AI 寫程式」這條軸線從**模型工程角度**展開——這些工具如何被訓練、如何 ground 在 repo、如何被 harness engineering 起來。
> **應用層視角**(怎麼用)請見 [`../../5.AI研究前沿_2024-2025/Vibe_Coding_與_AIGC_生成式創作完整學習指南.md`](../../5.AI研究前沿_2024-2025/Vibe_Coding_與_AIGC_生成式創作完整學習指南.md);
> **長篇實戰筆記**請見 [`../../4.相關的更新Blog/Day02,03 從頭熟練 Claude Code.md`](../../4.相關的更新Blog/);
> 對應 [全景圖 #17](../../2024-2026_AI完整領域全景圖.md)。

---

## 📂 本目錄內容

- [`AI程式助手深度指南.md`](./AI程式助手深度指南.md)(若存在)

---

## 🎯 核心技術範式

### 1. Repo-level Context

- **codebase indexing**(Cursor 用 OpenAI embedding + 自家 vector store)
- **code graph + AST**(tree-sitter 抽 top-level 簽名)
- **LSP hover / definition**(Aider 的 repo map)
- **Symbol graph**(Sourcegraph、Augment)

### 2. Tool Use 五件套

`read_file` / `edit_file` / `bash` / `grep` / `apply_patch` 是現代 coding agent 的最小工具集。

### 3. Multi-file Diff Edit

Aider 開創 unified diff / search-replace block 風格,顯著降低 token 成本與失敗率(對比讓 LLM 重寫整檔)。

### 4. Context Engineering

- `CLAUDE.md` / `AGENTS.md` 作為「持久 system prompt」(repo 級 priming)
- **Skills**(按需載入的 procedural memory)
- **Subagents**(獨立 context 防主 context 汙染)

### 5. 自我驗證

`run tests` / `type check` / `build` — Devin 2.2 直接給 agent 一個 desktop 點按鈕。

---

## 🛠 工具格局

### IDE 內嵌型
- **GitHub Copilot**(82% 滲透率)
- **Cursor**(50K+ 企業組織,$1B ARR / 2 年)
- **Windsurf (Codeium)**(Cascade、ZDR)
- **JetBrains AI、Tabnine、Continue**

### CLI / 終端型
- **Claude Code**(發布 6 個月達 $1B ARR、企業 coding 42% 市佔)
- **OpenAI Codex CLI、Aider、Gemini CLI、Sourcegraph Amp**

### Web / Cloud 型
- **v0 by Vercel**、**bolt.new**、**Lovable**、**Replit Agent**、**StackBlitz**

### Autonomous Agent
- **Devin (Cognition, 2.2)**、**OpenHands**(原 OpenDevin)、**SWE-agent**、**Cline**、**Roo Code**

---

## 📊 底層模型 + Coding Benchmark

主流 coding 模型(2026 Q1):
- **Claude Sonnet 5 / Opus 4.7 Mythos**(SWE-Bench Verified 92-94%)
- **GPT-5.5 / GPT-5.3 Codex**(85-89%)
- **Qwen3-Coder**、**DeepSeek-V3**、**Codestral 25**、**StarCoder2**(開源)

主流 Benchmark(2026):
- **SWE-Bench Pro**(替代飽和的 SWE-Bench Verified,頂尖 ~46%)
- **Aider Polyglot**(多語 diff edit)
- **Terminal-Bench 2.0**(CLI agent)
- **LiveCodeBench**(防汙染、持續更新)

---

## 🏗 Harness Engineering(本主題與 #14 重疊)

「Harness 是 LLM 外圍的 OS」— 同一個模型換 harness,SWE-Bench 可從 Top 30 衝 Top 5。

Anthropic 公開把 Claude Code 的 harness 封裝為 **[Claude Agent SDK](../../3.LLM應用工程/3.Agent/Claude_Agent_SDK_DeepAgents.md)**(2026/03 從 Claude Code SDK 改名):compaction、hooks、skills、subagents、filesystem-as-memory。

---

## 🚨 安全與風險

- **Slopsquatting**:20% LLM 推薦的 npm 套件不存在,43% 在 10 次 query 內穩定重現 → 攻擊者可預先註冊
- **Hallucinated API**:LLM 引用不存在的函式 / 錯誤 prop
- **Security debt**:CodeRabbit 報告 AI PR 安全問題比人類 PR 多 2.74×;Veracode 45% AI 程式碼有瑕疵
- **Code Review 負擔**:AI 把寫程式時間轉嫁到 review 時間;senior 工程師成為瓶頸

防禦工具:**Snyk DeepCode AI、Socket、Endor Labs、CodeRabbit、Greptile**。

---

## 🔗 延伸閱讀

- [`../../5.AI研究前沿_2024-2025/Vibe_Coding_與_AIGC_生成式創作完整學習指南.md`](../../5.AI研究前沿_2024-2025/Vibe_Coding_與_AIGC_生成式創作完整學習指南.md) — 應用層全景
- [`../../3.LLM應用工程/3.Agent/Claude_Agent_SDK_DeepAgents.md`](../../3.LLM應用工程/3.Agent/Claude_Agent_SDK_DeepAgents.md) — Claude Code 的 harness 開放化
- [`../../4.相關的更新Blog/Day02,03 從頭熟練 Claude Code.md`](../../4.相關的更新Blog/) — 個人實戰筆記
- [`../../FRONTIER_TERMS_INDEX.md`](../../FRONTIER_TERMS_INDEX.md) — Claude Code / Devin / Cursor / Vibe Coding 等熱詞索引
- [全景圖 #17](../../2024-2026_AI完整領域全景圖.md)

**核心心法**:2026 的工程師價值在於**把人類意圖編碼為 spec、context、skill**,讓 agent 群可被審計地產出可信程式碼——而不是「最會寫程式的人」。
