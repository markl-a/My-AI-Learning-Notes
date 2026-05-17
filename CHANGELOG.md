# 更新日誌 (Changelog)

本文件記錄專案的所有重要更新。格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
版本號遵循 [語義化版本](https://semver.org/lang/zh-TW/)。

## [未發布] - Unreleased

### 即將推出
- 英文版本核心文檔
- MkDocs 文檔網站上線
- 更多實戰專案範例

---

## [1.4.0] - 2026-05-17

第三輪收尾:把章節 hype 風險用 disclaimer 一次性顯化,並把審計過程的原始產物
歷史化進 `docs/audits/`,讓未來自己 / 協作者能回溯本輪決策。

### Added
- **`docs/audits/2026-05/` 永久審計記錄**
  - 三份原始報告:`gemini_report.md`、`codex_report.md`、`claude_subagent_report.md`
  - 三份 audit prompts(Gemini / Codex / Claude subagent)
  - `README.md` 索引 + 三句共識診斷 + 「agent 已過時 / 誇大」決策記錄
  - 從 `.tmp/audit_results/`(.gitignore)歷史化進 git
- **18 處章節 disclaimer**
  - **12 個 frontier deep-dive 章節(11-22)章首加「⚠️ 鮮度與可信度說明」**:把「具體 ARR / 版本 / 月份 / benchmark 數字」混合三類來源(官方 / 報導 / agent 整理)分離,提醒讀者決策級數字以原 source 為準
  - **6 個含 `eval`/`exec`/AST `safe_eval` 的 demo README 加「⚠️ 教學示範 — 生產勿用」橫幅**:9.1 RAG-Agent(順帶列 CORS / path traversal / auth gate 安全洞)、11 MCP、ReAct agent、CrewAI 範例、LangChain Agents 範例、basic_apis
- **Memory 系統**(在 `C:\Users\m4932\.claude\projects\.../memory/`)
  - `feedback_verify_before_act.md` — verify agent claims 再行動的工作節奏
  - `project_repo_positioning.md` — 此 repo 是個人筆記 + phantom-mesh demo,非 production library
  - `reference_audit_artifacts.md` — audit 檔位置 + Gemini/Codex CLI / gh API 慣例

### Changed
- `三方審計優化建議_2026-05.md` 內所有 `.tmp/...` 連結改指 `docs/audits/2026-05/`

---

## [1.3.0] - 2026-05-16

第二輪健全化:由 Gemini / Codex / Claude subagent 三方獨立審計觸發,
逐項 verify 後動真實必修 + dependabot 大清理 + CI 從「裝飾性綠」改為「有意義的綠」。

### Added
- `三方審計優化建議_2026-05.md`:三方獨立審計合成,共識項目歸為 P0/P1/P2/P3,並標出 agent 誇大/過時項

### Changed
- **CI security 從「擺設」變「真實擋錯」**(`.github/workflows/ci.yml`)
  - Safety(API 已過時)→ 改用 `pip-audit --strict` 一支
  - Bandit 仍標 `continue-on-error`(教學 repo dummy keys 會誤報)
  - status job 把 `security` 加入成功 / 失敗條件
  - 移除 build job(repo 非 PyPI 套件,根目錄多個 top-level dir 導致 setuptools 偵測失敗)
  - Notebook 測試從 `nbconvert --execute` 全 repo(常 hang ~50 分鐘)改為 `nbformat.validate`(秒級)
- **SFT 訓練腳本升 TRL 0.12+ API**(`2.深入LLM/5.SFT/hands_on_project/scripts/3_train_model.py`)
  - `TrainingArguments` → `SFTConfig`、`evaluation_strategy` → `eval_strategy`
  - `dataset_text_field` / `max_seq_length` 從 SFTTrainer kwargs 移到 SFTConfig
  - `tokenizer=` → `processing_class=`
- **統一 langchain / openai 版本**(7 個子專案 requirements.txt)
  - `langchain>=0.1.0` → `>=0.3.0`、`langchain-openai>=0.0.5` → `>=0.2.0`、`openai>=1.0-1.12` → `>=1.50.0`,對齊根 `requirements-llm.txt`

### Fixed
- **Dependabot:75 → 22 alerts**(清掉 1 critical + 30 high + 22 moderate)
  - next 14.0.3 → 14.2.35(+ eslint-config-next 同步)
  - python-multipart 0.0.6 → 0.0.27(3 個檔)
  - GitPython 3.1.40 → 3.1.45、lxml 4.9.3 → 6.1.0、black 23.11.0 → 24.10.0
  - requests 2.31.0 → 2.33.0(4 個 d2l setup.py + 1 個 docker)
  - python-dotenv 1.0.0 → 1.2.2(5 個檔)、pytest 7.4.3 → 8.4.0
  - Markdown 3.5.1 → 3.8.1、PyPDF2 3.0.1 → 3.9.0、postcss 8.4.31 → 8.5.10
  - 剩 22 alerts 全在 archive `web-ui` demo 的 next.js 需要 major upgrade,user 決定保留
- **11 個壞 notebook 修復**(被新 nbformat 驗證抓出)
  - `1.從AI到LLM基礎/4.DL/03.Pytorch/1.ResNet/ResNet.ipynb`:cell stream output 有非法 `metadata` 欄
  - `1.從AI到LLM基礎/3.ML_&_Data_Analysis/` 內 10 個檔:`metadata.kernelspec.name` 缺失、cell 缺 `id` / `outputs` / `execution_count`、stream output 缺 `name`
- D2L_Jupyter_Notebooks/tensorflow/* 內 ~30 個 0-byte 空檔(d2l 上游 archive)在 CI validation 跳過

### Known Issues
- 剩 22 個 dependabot alerts(web-ui demo 內 next.js 14.x 無 patch,需升 next 15.x major)
- 11-22 章的 hype 章節仍用具體數字 / 月份 / 版本,只在章首加 disclaimer(未逐段加 `[Confirmed]`/`[Reported]`/`[Speculative]` 標籤)

---

## [1.2.0] - 2026-05-16

### Added
- 22 個 2024-2026 frontier deep-dive 文件(11-22 新章節)
- `2024-2026_AI完整領域全景圖.md` 作為主控索引
- 主題 1/2/3 內補 5 個深度檔(CV、多模態生成、時序/表格、LLM 核心、AI Coding)
- `專案爆破總診斷_2026-05.md` 全 repo 審計報告

### Fixed
- 主題 2 第 6/8 章 README 從 0 行補成正常導覽
- `7.LLM應用部屬` → `7.LLM應用部署`(別字)
- 移除 deploy.yml 死碼、停 benchmark.yml schedule
- 清理被誤追蹤的 .DS_Store 與 .coverage
- 歸檔 7 份工作報告到 docs/archive/
- 修正「人工智慧到生成式AI 」中的 AI 幻覺(2025 條目)
- 繁化簡中詞彙批次替換為台灣慣用詞

### Known Issues
- 主題 2 第 6/11 章主題重複待合併
- `1.LLM 部署` 與 `7.LLM應用部署` 主題重複待合併
- `8.LLM安全與防禦` 與 `13.LLM安全最佳實踐` 主題重複待合併
- CI 仍紅,需重寫(改為 lychee link check + mkdocs build)

---

## [1.1.0] - 2025-12-15

### 新增 (Added)
- ✨ **MCP 協議與工具呼叫完整模組** (`3.LLM應用工程/11.MCP協議與工具呼叫/`)
  - Anthropic MCP SDK 使用指南
  - MCP Server 開發教程
  - 與 Function Calling 對比分析

- ✨ **進階提示工程與結構化輸出** (`3.LLM應用工程/12.進階提示工程與結構化輸出/`)
  - Prompt Engineering 2.0 技術
  - DSPy、Guidance 框架使用
  - ReAct、Tree of Thoughts 進階技術

- ✨ **現代 LLM 對齊方法 2024-2025** (`2.深入LLM模型工程與LLM運維/11.現代對齊方法2024-2025/`)
  - DPO、IPO、SimPO、KTO、ORPO 完整實作指南
  - TRL 庫使用教程
  - 方法選擇決策樹

- ✨ **推理模型應用指南** (`2.深入LLM模型工程與LLM運維/12.推理模型應用/`)
  - OpenAI o1/o3、DeepSeek-R1、Gemini 2.0 Flash Thinking
  - 性能基準與成本分析
  - 混合推理系統架構

- ✨ **LLM 安全最佳實踐** (`3.LLM應用工程/13.LLM安全最佳實踐/`)
  - OWASP LLM Top 10 防護
  - 提示注入防禦機制
  - PII 保護與審計日誌

- ✨ **AI 系統測試框架** (`tests/ai_systems/`)
  - RAG 檢索品質測試（NDCG、MRR、MAP）
  - pytest 測試用例完整範例

- ✨ **LLM 面試題庫** (`9.面試準備與職業發展/1.LLM面試題庫/`)
  - 100 道完整面試準備題目
  - 涵蓋架構、訓練、RAG、Agent、系統設計

- ✨ **術語表** (`GLOSSARY.md`)
  - 100+ AI/LLM 核心術語定義
  - 中英文對照

- ✨ **先修知識清單** (`PREREQUISITES.md`)
  - 自我評估檢查清單
  - 分級學習路徑建議
  - 補充資源推薦

- ✨ **開發環境支持**
  - GitHub Codespaces 配置 (`.devcontainer/`)
  - Binder 支持 (`runtime.txt`)
  - Makefile 統一命令界面

### 改進 (Changed)
- 📚 優化專案文檔結構
- 🔧 更新 CI/CD 工作流配置

### 修復 (Fixed)
- 🐛 修正部分程式碼範例中的語法錯誤

---

## [1.0.0] - 2025-12-14

### 新增 (Added)
- 🎉 **多 Agent 深度分析優化計劃** (`OPTIMIZATION_PLAN_2024-2025.md`)
  - 10 個 Agent 並行分析結果
  - P0/P1/P2 優先級任務規劃
  - 詳細時間表和 KPI 指標

### 改進 (Changed)
- 📚 完善專案品質與社區協作基礎設施

---

## [0.9.0] - 2025-12-10

### 新增 (Added)
- 📚 **AI 研究前沿 2024-2025** (`5.AI研究前沿_2024-2025/`)
  - 50+ 篇關鍵論文導讀
  - 最新技術趨勢分析
  - Sora、o1、GraphRAG 等前沿技術

- 📚 **品質保證框架** (`quality_assurance/`)
  - 內容審查模板
  - 程式碼驗證工具
  - 品質標準文檔

### 改進 (Changed)
- 🔧 增強 CI/CD 配置
- 📝 更新學習路徑文檔

---

## [0.8.0] - 2025-11-15

### 新增 (Added)
- 📚 **LLM 應用工程完整模組** (`3.LLM應用工程/`)
  - LLM 部署指南
  - LLM as API 使用
  - Agent 系統設計
  - RAG 基礎與進階
  - 推論優化技術

- 📚 **深入 LLM 模型工程** (`2.深入LLM模型工程與LLM運維/`)
  - LLM 基礎與架構
  - 文字生成與解碼策略
  - 預訓練技術
  - 監督微調 (SFT)
  - 模型壓縮與優化
  - 模型部署與運維

### 改進 (Changed)
- 📚 重組章節結構
- 🔧 添加更多實戰專案

---

## [0.5.0] - 2025-09-01

### 新增 (Added)
- 📚 **從 AI 到 LLM 基礎** (`1.從AI到LLM基礎/`)
  - 數學基礎 (Math for ML)
  - AI 簡介
  - 機器學習與資料分析
  - 深度學習完整路徑

- 📚 **DeepLearning.ai 短課程學習紀錄**

### 改進 (Changed)
- 📝 建立基礎文檔結構

---

## [0.1.0] - 2025-06-01

### 新增 (Added)
- 🎉 專案初始化
- 📚 建立基本目錄結構
- 📝 README.md 初版

---

## 版本說明

- **主版本號** (Major): 不相容的 API 修改或重大架構變更
- **次版本號** (Minor): 新增功能，向下相容
- **修訂號** (Patch): 問題修正，向下相容

### 標籤說明

- ✨ 新功能
- 📚 文檔更新
- 🔧 工具/配置改進
- 🐛 錯誤修復
- 🎉 重大里程碑
- ⚠️ 重大變更/破壞性更新
- 🗑️ 移除功能

---

## 貢獻

如果你發現任何問題或有改進建議，歡迎：
1. 提交 [Issue](https://github.com/markl-a/My-AI-Learning-Notes/issues)
2. 發起 [Pull Request](https://github.com/markl-a/My-AI-Learning-Notes/pulls)

感謝所有貢獻者的付出！
