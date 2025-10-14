Claude Flow 入門以及程式碼分析
# Claude Flow 入門路線
## 目錄
1. [學習路線概覽]
2. [第一階段]：入門基礎 
3. [第二階段]：核心功能 
4. [第三階段]：進階應用 
5. [第四階段]：專家級應用 
6. [實戰項目建議]
7. [常見問題與解答]
 - -
## 學習路線概覽
```
┌─────────────────────────────────────────────────────────────────┐
│ Claude Flow 技能樹 │
├─────────────────────────────────────────────────────────────────┤
│ │
│ [階段 1] [階段 2] [階段 3] [階段 4] │
│ 入門基礎 → 核心功能 → 進階應用 → 專家級 │
│ │
│ • 環境設置 • Swarm協調 • 蜂巢智能 • 企業級 │
│ • 基本命令 • SPARC方法論 • 神經網絡 • 架構設計 │
│ • 記憶系統 • Agent管理 • GitHub整合 • 性能優化 │
│ • CLI操作 • 並行執行 • MCP工具 • 雲端部署 │
│ │
└─────────────────────────────────────────────────────────────────┘
```
### 🎓 技能等級定義
| 等級 | 能力範圍 | 預計時間 | 專案複雜度 |
| - - - | - - - - -| - - - - -| - - - - - -|
| **🌱 入門** | 基本操作、單一任務 | 簡單腳本、單文件項目 |
| **🌿 初階** | 多代理協調、基礎開發 | 小型應用、功能模塊 |
| **🌳 進階** | 複雜協調、全棧開發 | 中型項目、完整應用 |
| **🌲 專家** | 企業架構、性能優化 | 大型系統、分散式應用 |
 - -
## 第一階段：入門
### 📚 學習目標
- ✅ 完成環境設置與配置
- ✅ 掌握基本CLI命令操作
- ✅ 理解記憶系統原理
- ✅ 完成第一個AI任務
 - -
### 🛠️ 1. 環境設置與安裝
#### 1.1 系統需求檢查
```bash
# 檢查Node.js版本 (需要 >= 20.0.0)
node - version
# 檢查npm版本 (需要 >= 9.0.0)
npm - version
# 檢查系統資源
# 最低需求: 2GB RAM, 2核CPU, 500MB硬碟
```
#### 1.2 安裝Claude Code (必要前置)
```bash
# 全域安裝Claude Code
npm install -g @anthropic-ai/claude-code
# (可選) 跳過權限檢查以加速設置
claude - dangerously-skip-permissions
# 驗證安裝
claude - version
```
#### 1.3 安裝Claude Flow
```bash
# 方法1: NPX執行 (推薦 - 總是使用最新版本)
npx claude-flow@alpha init - force
# 方法2: 全域安裝
npm install -g claude-flow@alpha
# 驗證安裝
claude-flow - version
# 應顯示: v2.7.0-alpha.10
# 查看幫助
npx claude-flow@alpha - help
```
#### 1.4 專案初始化
```bash
# 初始化新專案
npx claude-flow@alpha init - force - project-name "my-first-project"
# 檢查生成的文件結構
ls -la .claude/
# 應包含:
# - agents/ (代理定義)
# - commands/ (自定義命令)
# - memory.db (記憶資料庫)
# 驗證配置
npx claude-flow@alpha sparc modes
```
#### 📝 練習1: 環境驗證
```bash
# 完整驗證流程
npx claude-flow@alpha init - force
npx claude-flow@alpha - version
npx claude-flow@alpha sparc modes
# 預期輸出應包含:
# - 版本號: v2.7.0-alpha.10
# - 可用模式列表
# - 配置文件確認
```
 - -
### 🎯 2：基本命令與操作
#### 2.1 核心命令學習
```bash
# 1. 查看所有可用命令
npx claude-flow@alpha - help
# 2. 查看特定命令幫助
npx claude-flow@alpha swarm - help
npx claude-flow@alpha sparc - help
npx claude-flow@alpha memory - help
# 3. 查看可用的SPARC模式
npx claude-flow@alpha sparc modes
# 4. 查看MCP工具狀態
npx claude-flow@alpha mcp status
```
#### 2.2 第一個AI任務
```bash
# 簡單任務: 讓AI分析一個文件
npx claude-flow@alpha swarm "分析package.json並總結專案資訊" - claude
# 等待任務完成，觀察輸出
# 注意輸出中的:
# - 任務分解過程
# - Agent協調資訊
# - 最終結果摘要
```
#### 2.3 SPARC方法論初體驗
```bash
# 使用SPARC TDD工作流
npx claude-flow@alpha sparc tdd "實作一個簡單的計算機函數"
# SPARC會自動執行5個階段:
# S - Specification (需求分析)
# P - Pseudocode (偽代碼設計)
# A - Architecture (架構設計)
# R - Refinement (精煉優化)
# C - Code (程式碼實作)
```
#### 📝 練習2: 基本任務執行
```bash
# 練習任務1: 程式碼分析
npx claude-flow@alpha swarm "分析當前目錄結構並提供優化建議" - claude
# 練習任務2: 簡單開發
npx claude-flow@alpha sparc tdd "創建一個待辦事項API端點"
# 練習任務3: 文檔生成
npx claude-flow@alpha swarm "為專案生成README.md" - claude
```
 - -
### 🧠 3：記憶系統深入
#### 3.1 理解ReasoningBank記憶系統
Claude Flow使用**ReasoningBank**作為持久化記憶系統：
- **儲存位置**: `.swarm/memory.db` (SQLite資料庫)
- **查詢速度**: 2–3毫秒 (語義搜尋)
- **嵌入維度**: 1024維度 (hash-based) 或 1536維度 (OpenAI)
- **無需API金鑰**: 使用deterministic hash embeddings
#### 3.2 記憶操作命令
```bash
# 1. 儲存記憶
npx claude-flow@alpha memory store api_config "REST API配置說明" \
 - namespace backend - reasoningbank
# 2. 查詢記憶 (語義搜尋)
npx claude-flow@alpha memory query "API配置" \
 - namespace backend - reasoningbank
# 3. 列出所有記憶
npx claude-flow@alpha memory list \
 - namespace backend - reasoningbank
# 4. 查看記憶狀態
npx claude-flow@alpha memory status - reasoningbank
# 5. 刪除特定記憶
npx claude-flow@alpha memory delete api_config \
 - namespace backend - reasoningbank
```
#### 3.3 命名空間組織策略
```bash
# 按功能領域組織
npx claude-flow@alpha memory store auth_flow "認證流程" - namespace auth
npx claude-flow@alpha memory store user_schema "用戶資料結構" - namespace database
npx claude-flow@alpha memory store api_patterns "API設計模式" - namespace architecture
# 按專案階段組織
npx claude-flow@alpha memory store requirements "需求規格" - namespace planning
npx claude-flow@alpha memory store design_decisions "設計決策" - namespace design
npx claude-flow@alpha memory store test_results "測試結果" - namespace testing
```
#### 📝 練習3: 記憶系統實戰
```bash
# 場景: 構建一個知識庫
# 步驟1: 儲存專案知識
npx claude-flow@alpha memory store project_overview "電商平台開發專案" \
 - namespace project - reasoningbank
npx claude-flow@alpha memory store tech_stack "Node.js, React, PostgreSQL" \
 - namespace project - reasoningbank
# 步驟2: 查詢知識
npx claude-flow@alpha memory query "專案技術" \
 - namespace project - reasoningbank
# 步驟3: 檢查狀態
npx claude-flow@alpha memory status - reasoningbank
# 預期結果:
# ✅ Total memories: 2
# ✅ Query latency: 2–3ms
```
 - -
### 📊 4：基礎項目實戰
#### 4.1 項目: 簡單的任務管理系統
**目標**: 使用Claude Flow構建一個基礎的任務管理API
```bash
# 步驟1: 專案初始化
npx claude-flow@alpha init - force - project-name "task-manager"
cd task-manager
# 步驟2: 使用SPARC開發
npx claude-flow@alpha sparc tdd "創建任務管理REST API，包含CRUD操作"
# SPARC會自動:
# 1. 分析需求 (Specification)
# 2. 設計算法 (Pseudocode)
# 3. 規劃架構 (Architecture)
# 4. 測試驅動開發 (Refinement)
# 5. 實作程式碼 (Code)
# 步驟3: 儲存設計決策
npx claude-flow@alpha memory store api_design "RESTful API for task management" \
 - namespace task-manager - reasoningbank
```
#### 4.2 觀察與學習重點
```bash
# 1. 查看生成的檔案結構
tree . -L 2
# 2. 檢查測試檔案
cat tests/*.test.js
# 3. 查看API路由
cat src/routes/*.js
# 4. 回顧記憶內容
npx claude-flow@alpha memory list - namespace task-manager - reasoningbank
```
#### 📝 練習4: 擴展功能
```bash
# 基於已有基礎，添加新功能
npx claude-flow@alpha swarm "為任務管理系統添加用戶認證功能" - claude
# 使用記憶系統查詢現有設計
npx claude-flow@alpha memory query "API設計" - namespace task-manager - reasoningbank
# 儲存新的設計決策
npx claude-flow@alpha memory store auth_design "JWT based authentication" \
 - namespace task-manager - reasoningbank
```
 - -
### 第一階段檢核表
完成以下檢核項目後，即可進入第二階段：
- [ ] ✅ 成功安裝Claude Code和Claude Flow
- [ ] ✅ 理解基本CLI命令結構
- [ ] ✅ 能夠執行簡單的swarm任務
- [ ] ✅ 理解SPARC方法論的5個階段
- [ ] ✅ 掌握記憶系統的基本操作（儲存、查詢、列表）
- [ ] ✅ 完成至少一個簡單的實戰項目
- [ ] ✅ 能夠組織和檢索記憶內容
- [ ] ✅ 理解命名空間的使用場景
**預期能力**: 能夠獨立使用Claude Flow完成簡單的開發任務，理解基本的記憶系統操作。
 - -
## 第二階段：核心功能 
### 📚 學習目標
- ✅ 掌握Swarm協調機制
- ✅ 深入理解SPARC工作流
- ✅ 學會多代理協作
- ✅ 掌握並行執行策略
 - -
### 🐝 5：Swarm協調機制
#### 5.1 Swarm基礎概念
```bash
# Swarm的三種模式:
# 1. 快速任務執行 (推薦用於簡單任務)
npx claude-flow@alpha swarm "構建REST API與認證" - claude
# 2. 多代理協調 (中等複雜度任務)
npx claude-flow@alpha swarm init - topology mesh - max-agents 5
npx claude-flow@alpha swarm spawn researcher "分析API模式"
npx claude-flow@alpha swarm spawn coder "實作端點"
npx claude-flow@alpha swarm spawn tester "編寫測試"
npx claude-flow@alpha swarm status
# 3. 蜂巢思維 (複雜項目)
npx claude-flow@alpha hive-mind wizard
npx claude-flow@alpha hive-mind spawn "構建企業級系統" - claude
```
#### 5.2 拓撲結構深入
```bash
# 拓撲結構選擇指南:
# Mesh (網狀) - 平等協作
# 適用: 代碼審查、頭腦風暴、研究分析
npx claude-flow@alpha swarm init - topology mesh - max-agents 5
# Hierarchical (層級) - 皇后領導
# 適用: 大型項目、複雜架構、企業系統
npx claude-flow@alpha swarm init - topology hierarchical - max-agents 8
# Adaptive (自適應) - 動態調整
# 適用: 需求變化快、探索性項目
npx claude-flow@alpha swarm init - topology adaptive - max-agents 6
```
#### 5.3 Agent生命周期管理
```bash
# 完整的Agent生命周期:
# 1. 初始化swarm
npx claude-flow@alpha swarm init - topology mesh - max-agents 5
# 2. 生成agents
npx claude-flow@alpha swarm spawn researcher "研究微服務模式"
npx claude-flow@alpha swarm spawn architect "設計系統架構"
npx claude-flow@alpha swarm spawn coder "實作核心服務"
# 3. 監控狀態
npx claude-flow@alpha swarm status
# 4. 查看結果
npx claude-flow@alpha swarm results
# 5. 清理資源
npx claude-flow@alpha swarm cleanup
```
#### 📝 練習5: Swarm協調實戰
```bash
# 場景: 構建一個完整的用戶管理系統
# 步驟1: 初始化swarm
npx claude-flow@alpha swarm init - topology hierarchical - max-agents 6
# 步驟2: 派遣專業agents
npx claude-flow@alpha swarm spawn researcher "研究用戶管理最佳實踐"
npx claude-flow@alpha swarm spawn system-architect "設計用戶管理架構"
npx claude-flow@alpha swarm spawn backend-dev "實作用戶API"
npx claude-flow@alpha swarm spawn tester "編寫測試套件"
npx claude-flow@alpha swarm spawn reviewer "代碼審查"
# 步驟3: 監控進度
watch -n 5 'npx claude-flow@alpha swarm status'
# 步驟4: 儲存協調經驗
npx claude-flow@alpha memory store swarm_pattern "6-agent hierarchical for user management" \
 - namespace patterns - reasoningbank
```
 - -
### 🎨 6：SPARC深度應用
#### 6.1 SPARC完整工作流
```bash
# SPARC的5個階段詳解:
# 1. Specification (規格分析)
npx claude-flow@alpha sparc run spec-pseudocode "電商支付系統"
# 輸出: requirements.md, user-stories.md
# 2. Pseudocode (偽代碼設計)
# 自動在spec-pseudocode中完成
# 3. Architecture (架構設計)
npx claude-flow@alpha sparc run architect "支付系統架構設計"
# 輸出: architecture.md, component-diagram.md
# 4. Refinement (精煉 - TDD)
npx claude-flow@alpha sparc tdd "實作支付API"
# 輸出: tests/*.test.js, src/*.js
# 5. Code (程式碼整合)
npx claude-flow@alpha sparc run integration "整合支付模組"
# 輸出: integrated system
```
#### 6.2 批次處理與管道
```bash
# 並行執行多個SPARC階段
npx claude-flow@alpha sparc batch spec-pseudocode,architect,code "微服務架構"
# 完整開發管道
npx claude-flow@alpha sparc pipeline "電商平台完整系統"
# 自動執行: spec -> pseudo -> arch -> refine -> code
# 並發處理多個任務
npx claude-flow@alpha sparc concurrent architect "tasks.txt"
# tasks.txt內容:
# - 認證服務
# - 支付服務
# - 訂單服務
```
#### 6.3 SPARC與記憶整合
```bash
# 場景: 建立可重用的開發模式庫
# 執行SPARC並儲存模式
npx claude-flow@alpha sparc tdd "REST API with JWT authentication"
# 儲存架構決策
npx claude-flow@alpha memory store auth_architecture "JWT-based REST API architecture" \
 - namespace patterns - reasoningbank
# 儲存測試策略
npx claude-flow@alpha memory store test_strategy "Integration tests with Supertest" \
 - namespace patterns - reasoningbank
# 後續項目查詢參考
npx claude-flow@alpha memory query "認證架構" - namespace patterns - reasoningbank
```
#### 📝 練習6: SPARC完整流程
```bash
# 項目: 構建一個博客平台
# 階段1: 規格與設計
npx claude-flow@alpha sparc batch spec-pseudocode,architect "博客平台系統"
# 階段2: TDD開發
npx claude-flow@alpha sparc tdd "博客文章CRUD API"
# 階段3: 整合
npx claude-flow@alpha sparc run integration "整合博客系統"
# 儲存項目知識
npx claude-flow@alpha memory store blog_architecture "完整博客平台架構" \
 - namespace blog-project - reasoningbank
# 驗證輸出
ls -la src/ tests/ docs/
```
 - -
### ⚡ 7：並行執行與性能優化
#### 7.1 並行執行的黃金法則
**關鍵原則**: "1 MESSAGE = ALL RELATED OPERATIONS"
```bash
# ❌ 錯誤: 順序執行 (慢6倍)
npx claude-flow@alpha swarm spawn coder "feature A"
npx claude-flow@alpha swarm spawn tester "test A"
npx claude-flow@alpha swarm spawn reviewer "review A"
# ✅ 正確: 並行執行 (推薦)
npx claude-flow@alpha swarm - agents coder,tester,reviewer - parallel "完整功能開發"
```
#### 7.2 批次操作最佳實踐
```bash
# 場景: 全棧應用開發
# 單一命令並行派遣所有agents
npx claude-flow@alpha swarm \
 - agents backend-dev,coder,system-architect,tester,reviewer,api-docs \
 - topology hierarchical \
 - parallel \
"構建完整的電商API系統"
# Claude Flow會自動:
# 1. 並行初始化所有agents
# 2. 分配任務並協調
# 3. 合併結果
# 4. 生成完整輸出
```
#### 7.3 性能監控與分析
```bash
# 啟用性能分析
npx claude-flow@alpha hooks post-task - analyze-performance true
# 查看性能指標
npx claude-flow@alpha swarm status - metrics
# 導出性能報告
npx claude-flow@alpha hooks session-end - export-metrics true
# 預期性能指標:
# - Task completion rate: 96.3%
# - Speed improvement: 2.8–4.4x
# - Token reduction: 32.3%
# - SWE-Bench score: 84.8%
```
#### 📝 練習7: 性能優化實戰
```bash
# 任務: 比較順序 vs 並行執行
# 測試1: 順序執行 (記錄時間)
time npx claude-flow@alpha swarm spawn researcher "research"
time npx claude-flow@alpha swarm spawn coder "implement"
time npx claude-flow@alpha swarm spawn tester "test"
# 測試2: 並行執行 (記錄時間)
time npx claude-flow@alpha swarm \
 - agents researcher,coder,tester \
 - parallel \
"完整功能開發"
# 預期結果: 並行執行快2.8–4.4倍
```
 - -
### 🎯 第二階段檢核表
- [ ] ✅ 理解三種Swarm模式的使用場景
- [ ] ✅ 能夠選擇合適的拓撲結構
- [ ] ✅ 掌握完整的SPARC工作流程
- [ ] ✅ 能夠執行批次處理和管道操作
- [ ] ✅ 理解並行執行的重要性
- [ ] ✅ 能夠監控和分析性能指標
- [ ] ✅ 建立可重用的模式庫
**預期能力**: 能夠使用多代理協調完成中等複雜度的項目，理解性能優化策略。
 - -
## 第三階段：進階應用
### 📚 學習目標
- ✅ 掌握蜂巢智能系統
- ✅ 整合GitHub工作流
- ✅ 使用100+ MCP工具
- ✅ 實作複雜的企業級項目
 - -
### 🐝 8：蜂巢智能系統
#### 8.1 蜂巢思維架構
```bash
# Hive-Mind vs Swarm 比較:
# Swarm - 適合快速任務
# • 即時啟動
# • 任務範圍記憶
# • 臨時協調
# Hive-Mind - 適合複雜項目
# • 互動式設置
# • 項目級SQLite記憶
# • 持久化會話
# • 可恢復狀態
```
#### 8.2 蜂巢系統使用
```bash
# 1. 初始化蜂巢系統 (互動式向導)
npx claude-flow@alpha hive-mind wizard
# 向導會詢問:
# - 項目名稱
# - 項目範圍
# - 所需agents
# - 拓撲結構
# - 記憶策略
# 2. 派遣蜂巢任務
npx claude-flow@alpha hive-mind spawn "構建微服務平台" - claude
# 3. 檢查蜂巢狀態
npx claude-flow@alpha hive-mind status
# 輸出範例:
# Session: hive-xxxxx
# Status: Active
# Agents: 8
# Memory: 150 patterns
# Uptime: 2h 34m
# 4. 恢復之前的會話
npx claude-flow@alpha hive-mind resume session-xxxxx
# 5. 列出所有會話
npx claude-flow@alpha hive-mind list-sessions
```
#### 8.3 蜂巢記憶管理
```bash
# 蜂巢使用項目級記憶
# 查詢蜂巢記憶
npx claude-flow@alpha memory query "微服務模式" \
 - namespace hive-project - reasoningbank
# 查看記憶統計
npx claude-flow@alpha memory status - reasoningbank
# 輸出範例:
# Total memories: 150
# Namespaces: 5
# Storage: .swarm/memory.db
# Size: 12.5 MB
# Embeddings: 150
# Query latency: 2ms
# 導出蜂巢知識
npx claude-flow@alpha hive-mind export-knowledge - format json
```
#### 📝 練習8: 蜂巢系統實戰
```bash
# 項目: 構建一個企業級電商平台
# 步驟1: 初始化蜂巢
npx claude-flow@alpha hive-mind wizard
# 輸入:
# - 項目名: ecommerce-platform
# - 範圍: 全棧電商系統
# - Agents: 10
# - 拓撲: hierarchical
# 步驟2: 派遣開發任務
npx claude-flow@alpha hive-mind spawn \
"構建電商平台，包含用戶管理、產品目錄、購物車、支付、訂單管理" \
 - claude
# 步驟3: 監控進度
watch -n 10 'npx claude-flow@alpha hive-mind status'
# 步驟4: 暫停與恢復
npx claude-flow@alpha hive-mind pause
# … 休息一下 …
npx claude-flow@alpha hive-mind resume session-xxxxx
# 步驟5: 導出項目知識
npx claude-flow@alpha hive-mind export-knowledge \
 - format json \
 - output ecommerce-knowledge.json
```
 - -
### 🔧 9：MCP工具深度整合
#### 9.1 MCP伺服器設置
```bash
# 1. 添加Claude Flow MCP伺服器 (必需)
claude mcp add claude-flow npx claude-flow@alpha mcp start
# 2. 添加增強協調 (可選)
claude mcp add ruv-swarm npx ruv-swarm mcp start
# 3. 添加雲端功能 (可選 - 需要註冊)
claude mcp add flow-nexus npx flow-nexus@latest mcp start
# 驗證MCP設置
npx claude-flow@alpha mcp status
# 輸出範例:
# MCP Servers:
# ✅ claude-flow: Running (100 tools)
# ✅ ruv-swarm: Running (15 tools)
# ⚠️ flow-nexus: Not configured
```
#### 9.2 核心MCP工具使用
```bash
# 100 MCP工具分類:
# === 協調工具 ===
mcp__claude-flow__swarm_init # 初始化swarm
mcp__claude-flow__agent_spawn # 生成agent
mcp__claude-flow__task_orchestrate # 任務協調
# === 記憶工具 ===
mcp__claude-flow__memory_usage # 記憶操作
mcp__claude-flow__memory_search # 模式搜尋
# === 監控工具 ===
mcp__claude-flow__swarm_status # Swarm狀態
mcp__claude-flow__agent_list # Agent列表
mcp__claude-flow__agent_metrics # Agent指標
# === 神經網絡工具 ===
mcp__claude-flow__neural_status # 神經網絡狀態
mcp__claude-flow__neural_train # 訓練模式
mcp__claude-flow__neural_patterns # 模式分析
# === GitHub工具 ===
mcp__claude-flow__github_repo_analyze # 倉庫分析
mcp__claude-flow__github_pr_manage # PR管理
mcp__claude-flow__github_issue_track # Issue追蹤
# === 性能工具 ===
mcp__claude-flow__benchmark_run # 基準測試
mcp__claude-flow__performance_report # 性能報告
```
#### 9.3 MCP工具實戰範例
```bash
# 場景: 使用MCP工具進行全自動開發
# 1. 初始化開發環境
npx claude-flow@alpha mcp start
# 2. 在Claude Code中使用MCP工具
# (在Claude Code對話中)
"""
請使用MCP工具構建用戶認證系統:
1. mcp__claude-flow__swarm_init (拓撲: hierarchical)
2. mcp__claude-flow__agent_spawn (類型: backend-dev)
3. mcp__claude-flow__agent_spawn (類型: tester)
4. mcp__claude-flow__task_orchestrate (任務: 認證系統)
5. mcp__claude-flow__github_pr_manage (創建PR)
"""
# 3. 監控MCP任務狀態
npx claude-flow@alpha mcp task-status <task-id>
# 4. 獲取結果
npx claude-flow@alpha mcp task-results <task-id>
```
#### 📝 練習9: MCP工具整合
```bash
# 練習: 建立自動化測試管道
# 步驟1: 配置MCP
claude mcp add claude-flow npx claude-flow@alpha mcp start
# 步驟2: 在Claude Code中執行
# 使用MCP工具:
# - swarm_init (mesh拓撲)
# - agent_spawn (tester, reviewer)
# - task_orchestrate (測試任務)
# 步驟3: 查看結果
npx claude-flow@alpha mcp task-results <task-id>
# 步驟4: 儲存自動化模式
npx claude-flow@alpha memory store testing_pipeline "自動化測試管道配置" \
 - namespace automation - reasoningbank
```
 - -
### 🔗 10：GitHub深度整合
#### 10.1 GitHub整合設置
```bash
# 1. 配置GitHub整合
npx claude-flow@alpha github init
# 會詢問:
# - GitHub token
# - 默認倉庫
# - PR模板
# - Issue標籤
# 2. 配置6種GitHub模式
# - PR Manager: PR管理與審查
# - Code Review Swarm: 多agent代碼審查
# - Issue Tracker: Issue追蹤與分類
# - Release Manager: 發布管理
# - Workflow Automation: CI/CD自動化
# - Multi-repo Swarm: 跨倉庫協調
```
#### 10.2 GitHub工作流範例
```bash
# === 模式1: PR Manager ===
npx claude-flow@alpha github pr-manager
# 自動執行:
# 1. 獲取待處理PR
# 2. 多agent審查
# 3. 運行測試
# 4. 提供反饋
# 5. 自動合併 (可選)
# === 模式2: Code Review Swarm ===
npx claude-flow@alpha github code-review-swarm - pr 123
# 派遣agents:
# - code-analyzer: 代碼質量
# - security-manager: 安全審查
# - perf-analyzer: 性能分析
# - reviewer: 代碼風格
# === 模式3: Issue Tracker ===
npx claude-flow@alpha github issue-tracker
# 自動執行:
# 1. 分類issues
# 2. 分配優先級
# 3. 建議解決方案
# 4. 追蹤進度
# === 模式4: Release Manager ===
npx claude-flow@alpha github release-manager - version 2.1.0
# 自動執行:
# 1. 生成CHANGELOG
# 2. 創建release notes
# 3. 打tag
# 4. 發布到npm/GitHub
```
#### 10.3 GitHub + Swarm 進階模式
```bash
# 場景: 自動化整個開發流程
# 步驟1: 從Issue開始
npx claude-flow@alpha github issue-tracker
# 選擇issue #45: "添加用戶頭像功能"
# 步驟2: 派遣開發swarm
npx claude-flow@alpha swarm \
 - agents backend-dev,coder,tester \
 - parallel \
"實作issue #45的用戶頭像功能"
# 步驟3: 自動創建PR
npx claude-flow@alpha github create-pr \
 - title "feat: 添加用戶頭像功能" \
 - issue 45 \
 - auto-review
# 步驟4: Code Review Swarm自動審查
# (自動觸發)
# 步驟5: 合併後自動更新Issue
# (自動關閉issue #45)
```
#### 📝 練習10: GitHub自動化流程
```bash
# 練習: 建立完整的CI/CD自動化
# 步驟1: 初始化GitHub整合
npx claude-flow@alpha github init
# 步驟2: 配置workflow automation
npx claude-flow@alpha github workflow-automation setup
# 步驟3: 創建測試PR
# (手動在GitHub創建PR #999)
# 步驟4: 觸發自動化流程
npx claude-flow@alpha github pr-manager - pr 999
# 步驟5: 監控並觀察自動化
watch -n 10 'gh pr view 999'
# 預期結果:
# ✅ 自動代碼審查
# ✅ 自動運行測試
# ✅ 自動性能分析
# ✅ 自動合併 (如果全部通過)
```
 - -
### 🎯 第三階段檢核表
- [ ] ✅ 掌握蜂巢智能系統的使用
- [ ] ✅ 能夠管理持久化會話
- [ ] ✅ 理解100+ MCP工具的分類與使用
- [ ] ✅ 配置並使用GitHub整合的6種模式
- [ ] ✅ 建立端到端的自動化工作流
- [ ] ✅ 能夠處理企業級複雜項目
**預期能力**: 能夠使用蜂巢智能和GitHub整合構建複雜的企業級應用，掌握高級自動化策略。
 - -
## 第四階段：專家級應用
### 📚 學習目標
- ✅ 神經網絡自學習系統
- ✅ 企業架構設計
- ✅ 性能優化與擴展
- ✅ 雲端部署與運維
 - -
### 🧠 11：神經網絡與自學習
#### 11.1 SAFLA神經模組
```bash
# SAFLA = Self-Aware Feedback Loop Algorithm
# 1. 初始化神經模組
npx claude-flow@alpha neural init - force
# 生成文件:
# .claude/agents/neural/
# ├── safla-agent.md # Agent定義
# ├── neural-patterns.md # 學習模式
# └── training-config.yaml # 訓練配置
# 2. 檢查神經網絡狀態
npx claude-flow@alpha neural status
# 輸出範例:
# Neural Module: Active
# Patterns: 27
# Training: Continuous
# Last update: 2 min ago
```
#### 11.2 訓練與模式學習
```bash
# 場景: 讓系統學習最佳實踐
# 1. 執行任務並啟用學習
npx claude-flow@alpha swarm "構建REST API" \
 - neural-training \
 - claude
# 2. 任務完成後訓練模式
npx claude-flow@alpha neural train \
 - pattern "rest-api-development" \
 - source ".swarm/memory.db"
# 3. 查看學習到的模式
npx claude-flow@alpha neural patterns list
# 輸出範例:
# Pattern ID | Name | Success Rate | Uses
# - - - - - - - - - - - - - - - - - - - - -
# pattern-01 | REST API Dev | 96.3% | 15
# pattern-02 | TDD Workflow | 94.8% | 12
# pattern-03 | Code Review | 98.1% | 20
# 4. 應用學習到的模式
npx claude-flow@alpha swarm "構建新的API" \
 - use-pattern "rest-api-development" \
 - claude
```
#### 11.3 持續學習與改進
```bash
# 配置持續學習
# 1. 啟用自動訓練
npx claude-flow@alpha neural config \
 - auto-train true \
 - min-confidence 0.85
# 2. 設置反饋循環
npx claude-flow@alpha hooks post-task \
 - neural-feedback \
 - analyze-performance true
# 3. 定期評估模式
npx claude-flow@alpha neural evaluate \
 - patterns all \
 - export-report
# 4. 清理低效模式
npx claude-flow@alpha neural prune \
 - threshold 0.80 \
 - dry-run
```
#### 📝 練習11: 神經網絡實戰
```bash
# 練習: 建立自學習開發系統
# 步驟1: 初始化神經模組
npx claude-flow@alpha neural init - force
# 步驟2: 執行10個開發任務，啟用學習
for i in {1..10}; do
npx claude-flow@alpha swarm "開發任務 $i" \
 - neural-training \
 - claude
done
# 步驟3: 訓練模式
npx claude-flow@alpha neural train - auto-detect
# 步驟4: 評估學習效果
npx claude-flow@alpha neural patterns list - sort-by success-rate
# 步驟5: 使用最佳模式
best_pattern=$(npx claude-flow@alpha neural patterns list - best)
npx claude-flow@alpha swarm "新任務" \
 - use-pattern "$best_pattern" \
 - claude
# 預期結果:
# ✅ 任務完成時間減少20–30%
# ✅ 代碼質量提升
# ✅ 自動選擇最佳agents
```
 - -
### 🏗️ 12：企業架構設計
#### 12.1 系統架構代理
```bash
# 使用system-architect agent
# 1. 高層架構設計
npx claude-flow@alpha swarm \
 - agents system-architect \
"設計微服務電商平台架構，支持100萬用戶"
# 輸出:
# - architecture-overview.md
# - component-diagram.mermaid
# - deployment-diagram.mermaid
# - scaling-strategy.md
# 2. 詳細組件設計
npx claude-flow@alpha swarm \
 - agents system-architect,backend-dev \
"設計用戶服務微服務組件"
# 3. 數據庫架構
npx claude-flow@alpha swarm \
 - agents system-architect \
"設計支持10TB數據的數據庫架構"
```
#### 12.2 完整企業級項目
```bash
# 項目: 構建大型企業SaaS平台
# 階段1: 架構設計 (2週)
npx claude-flow@alpha hive-mind wizard
# 配置:
# - 項目: enterprise-saas
# - Agents: 12
# - 拓撲: hierarchical
npx claude-flow@alpha hive-mind spawn \
"設計企業SaaS平台，包含多租戶、SSO、API網關、微服務" \
 - claude
# 階段2: 核心服務開發 (3週)
npx claude-flow@alpha sparc pipeline \
"實作認證服務、用戶管理、租戶管理、API網關"
# 階段3: 測試與集成 (1週)
npx claude-flow@alpha swarm \
 - agents tester,production-validator,perf-analyzer \
 - parallel \
"完整測試與性能驗證"
# 階段4: 部署與監控 (1週)
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"設置CI/CD管道與監控系統"
```
#### 12.3 架構模式庫
```bash
# 建立可重用的架構模式
# 儲存微服務架構模式
npx claude-flow@alpha memory store microservices_pattern \
"完整微服務架構模式，包含API網關、服務發現、配置中心" \
 - namespace architecture - reasoningbank
# 儲存數據庫模式
npx claude-flow@alpha memory store database_pattern \
"PostgreSQL主從複製 + Redis緩存 + Elasticsearch搜索" \
 - namespace architecture - reasoningbank
# 儲存部署模式
npx claude-flow@alpha memory store deployment_pattern \
"Kubernetes + Docker + Helm + GitOps" \
 - namespace architecture - reasoningbank
# 查詢架構模式
npx claude-flow@alpha memory query "微服務" \
 - namespace architecture - reasoningbank
```
#### 📝 練習12: 企業架構實戰
```bash
# 練習: 設計可擴展的社交媒體平台
# 步驟1: 需求分析
npx claude-flow@alpha sparc run spec-pseudocode \
"社交媒體平台，支持1000萬用戶，每日10億請求"
# 步驟2: 架構設計
npx claude-flow@alpha swarm \
 - agents system-architect \
"設計高可用、高並發的社交媒體架構"
# 步驟3: 組件分解
npx claude-flow@alpha sparc batch architect,code \
"用戶服務,內容服務,推薦服務,消息服務,搜索服務"
# 步驟4: 儲存架構決策
npx claude-flow@alpha memory store social_architecture \
"完整社交媒體平台架構" \
 - namespace social-project - reasoningbank
# 步驟5: 生成架構文檔
npx claude-flow@alpha swarm \
 - agents api-docs \
"生成完整的架構文檔和API規格"
```
 - -
### ⚡ 13：性能優化與擴展
#### 13.1 性能分析工具
```bash
# 使用perf-analyzer和performance-benchmarker agents
# 1. 性能基準測試
npx claude-flow@alpha swarm \
 - agents performance-benchmarker \
"對API進行壓力測試，目標: 10000 req/s"
# 2. 瓶頸分析
npx claude-flow@alpha swarm \
 - agents perf-analyzer \
"分析系統瓶頸並提供優化建議"
# 3. 性能優化
npx claude-flow@alpha swarm \
 - agents perf-analyzer,coder,tester \
 - parallel \
"優化API性能，目標提升50%"
# 4. 性能監控
npx claude-flow@alpha hooks post-task \
 - analyze-performance true \
 - export-metrics
```
#### 13.2 擴展策略
```bash
# 水平擴展規劃
# 1. 分析擴展需求
npx claude-flow@alpha swarm \
 - agents system-architect,perf-analyzer \
"規劃支持10x流量的擴展策略"
# 2. 負載均衡設計
npx claude-flow@alpha swarm \
 - agents system-architect \
"設計多層負載均衡架構"
# 3. 數據庫分片
npx claude-flow@alpha swarm \
 - agents system-architect,backend-dev \
"設計數據庫分片策略，支持100TB數據"
# 4. 緩存策略
npx claude-flow@alpha swarm \
 - agents system-architect \
"設計多層緩存策略 (CDN + Redis + Application)"
```
#### 13.3 性能優化實戰
```bash
# 場景: 優化慢速API
# 步驟1: 性能基準
npx claude-flow@alpha swarm \
 - agents performance-benchmarker \
"測試/api/users端點性能"
# 結果:
# Current: 500 req/s, 200ms p95 latency
# Target: 5000 req/s, 50ms p95 latency
# 步驟2: 瓶頸分析
npx claude-flow@alpha swarm \
 - agents perf-analyzer \
"分析/api/users性能瓶頸"
# 發現問題:
# - N+1查詢問題
# - 無緩存
# - 無連接池
# 步驟3: 實作優化
npx claude-flow@alpha swarm \
 - agents coder,tester \
 - parallel \
"優化/api/users: 添加查詢優化、Redis緩存、連接池"
# 步驟4: 驗證結果
npx claude-flow@alpha swarm \
 - agents performance-benchmarker \
"重新測試/api/users性能"
# 結果:
# Optimized: 6000 req/s, 35ms p95 latency ✅
# Improvement: 12x throughput, 5.7x latency reduction
```
#### 📝 練習13: 性能優化專案
```bash
# 練習: 優化一個完整的Web應用
# 步驟1: 初始性能測試
npx claude-flow@alpha swarm \
 - agents performance-benchmarker \
"測試完整應用性能基準"
# 步驟2: 全面分析
npx claude-flow@alpha swarm \
 - agents perf-analyzer \
"分析應用所有瓶頸"
# 步驟3: 制定優化計劃
npx claude-flow@alpha sparc run spec-pseudocode \
"性能優化計劃，目標: 5x提升"
# 步驟4: 並行實作優化
npx claude-flow@alpha swarm \
 - agents coder,tester,reviewer \
 - topology mesh \
 - parallel \
"實作所有性能優化"
# 步驟5: 驗證與對比
npx claude-flow@alpha swarm \
 - agents performance-benchmarker \
"對比優化前後性能"
# 步驟6: 儲存優化模式
npx claude-flow@alpha memory store performance_optimization \
"完整性能優化模式與結果" \
 - namespace optimization - reasoningbank
```
 - -
### ☁️ 14：雲端部署與運維
#### 14.1 容器化與Kubernetes
```bash
# 使用cicd-engineer agent
# 1. Docker容器化
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建生產級Dockerfile和docker-compose.yml"
# 2. Kubernetes部署
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建完整的Kubernetes manifests (Deployment, Service, Ingress, ConfigMap)"
# 3. Helm Charts
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建Helm chart，支持多環境部署"
# 4. CI/CD管道
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建GitHub Actions workflow: test -> build -> deploy"
```
#### 14.2 監控與運維
```bash
# 1. 監控設置
npx claude-flow@alpha swarm \
 - agents cicd-engineer,system-architect \
"設置Prometheus + Grafana監控系統"
# 2. 日誌聚合
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"設置ELK Stack (Elasticsearch, Logstash, Kibana)"
# 3. 告警規則
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"配置告警規則: CPU, Memory, Error Rate, Latency"
# 4. 健康檢查
npx claude-flow@alpha swarm \
 - agents cicd-engineer,backend-dev \
"實作health check端點和readiness probe"
```
#### 14.3 完整DevOps流程
```bash
# 項目: 完整的雲端DevOps流程
# 階段1: 基礎設施即代碼
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"使用Terraform創建AWS基礎設施"
# 階段2: 容器化
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"Docker化所有服務"
# 階段3: Kubernetes部署
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"部署到EKS (AWS Kubernetes)"
# 階段4: CI/CD
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"設置完整的GitOps工作流"
# 階段5: 監控告警
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"配置全面的監控和告警系統"
# 階段6: 自動擴展
npx claude-flow@alpha swarm \
 - agents cicd-engineer,system-architect \
"配置HPA (Horizontal Pod Autoscaler)"
```
#### 📝 練習14: 完整DevOps實戰
```bash
# 練習: 從零搭建生產環境
# 步驟1: 項目準備
npx claude-flow@alpha hive-mind wizard
# 項目: production-deployment
# 步驟2: 容器化
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建多階段Dockerfile優化鏡像大小"
# 步驟3: 本地測試
docker-compose up -d
# 驗證服務正常運行
# 步驟4: Kubernetes配置
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建k8s manifests: staging和production環境"
# 步驟5: CI/CD管道
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"創建GitHub Actions: lint -> test -> build -> deploy"
# 步驟6: 監控設置
npx claude-flow@alpha swarm \
 - agents cicd-engineer \
"部署Prometheus + Grafana + Alertmanager"
# 步驟7: 驗證部署
# 運行完整的smoke tests
# 步驟8: 儲存DevOps模式
npx claude-flow@alpha memory store devops_pipeline \
"完整的DevOps部署流程與配置" \
 - namespace devops - reasoningbank
```
 - -
### 🎯 第四階段檢核表
- [ ] ✅ 掌握神經網絡自學習系統
- [ ] ✅ 能夠設計企業級架構
- [ ] ✅ 掌握性能分析與優化技巧
- [ ] ✅ 能夠進行10x性能提升
- [ ] ✅ 掌握容器化與Kubernetes
- [ ] ✅ 建立完整的CI/CD管道
- [ ] ✅ 配置生產級監控告警系統
- [ ] ✅ 實作自動擴展策略
**預期能力**: 能夠使用Claude Flow構建、優化和部署大型企業級系統，掌握完整的DevOps流程。
 - -
## 🚀 實戰項目建議
### 初級項目 (1–2週)
#### 項目1: 個人博客API
```bash
# 功能: CRUD API for 博客文章
npx claude-flow@alpha sparc tdd "博客API with CRUD操作"
# 學習重點:
# - SPARC工作流
# - 測試驅動開發
# - 記憶系統使用
```
#### 項目2: 待辦事項應用
```bash
# 功能: 完整的待辦事項管理
npx claude-flow@alpha swarm "全棧待辦應用: React + Node.js + MongoDB" - claude
# 學習重點:
# - Swarm協調
# - 前後端協作
# - 基本部署
```
 - -
### 中級項目 (3–4週)
#### 項目3: 電商平台
```bash
# 功能: 用戶、產品、購物車、訂單
npx claude-flow@alpha hive-mind spawn "完整電商平台" - claude
# 學習重點:
# - 蜂巢智能
# - 複雜業務邏輯
# - 支付整合
# - 性能優化
```
#### 項目4: 社交媒體API
```bash
# 功能: 用戶、帖子、評論、點贊、關注
npx claude-flow@alpha sparc pipeline "社交媒體API with 推薦算法"
# 學習重點:
# - 複雜數據模型
# - 推薦系統
# - 實時通知
# - 緩存策略
```
 - -
### 高級項目 (6–8週)
#### 項目5: 微服務平台
```bash
# 功能: 多個微服務 + API網關
npx claude-flow@alpha hive-mind spawn \
"微服務平台: Auth, Users, Products, Orders, Payments" \
 - claude
# 學習重點:
# - 微服務架構
# - 服務間通信
# - 分散式追蹤
# - Kubernetes部署
```
#### 項目6: SaaS多租戶平台
```bash
# 功能: 多租戶、SSO、計費、分析
npx claude-flow@alpha hive-mind spawn \
"企業SaaS平台with 多租戶架構" \
 - claude
# 學習重點:
# - 多租戶架構
# - SSO整合
# - 訂閱計費
# - 企業級安全
# - 完整DevOps
```
 - -
## ❓ 常見問題與解答
### Q1: 如何選擇Swarm vs Hive-Mind?
**答**:
| 場景 | 推薦 | 原因 |
| - - - | - - - | - - - |
| 簡單任務 (< 1小時) | Swarm | 即時啟動，無需配置 |
| 中型項目 (1–3天) | Swarm | 任務範圍記憶足夠 |
| 大型項目 (1週+) | Hive-Mind | 需要持久化記憶 |
| 需要暫停/恢復 | Hive-Mind | 支持會話管理 |
| 探索性開發 | Swarm | 快速迭代 |
| 企業級系統 | Hive-Mind | 項目級記憶和協調 |
 - -
### Q2: 記憶系統最佳實踐?
**答**:
```bash
# 1. 使用清晰的命名空間
 - namespace project/feature/component
# 2. 儲存決策，而非實作細節
# ✅ 好
npx claude-flow@alpha memory store auth_strategy "JWT tokens with 15min expiry" \
 - namespace auth - reasoningbank
# ❌ 壞
npx claude-flow@alpha memory store code "function login() { … }" \
 - namespace auth - reasoningbank
# 3. 定期查詢以強化記憶
npx claude-flow@alpha memory query "認證" - recent
# 4. 清理過時記憶
npx claude-flow@alpha memory cleanup - older-than 30days
```
 - -
### Q3: 如何優化性能?
**答**:
```bash
# 1. 始終使用並行執行
npx claude-flow@alpha swarm \
 - agents agent1,agent2,agent3 \
 - parallel \
"任務"
# 2. 限制agent數量 (3–6個最佳)
# ✅ 好: 5 agents
# ❌ 壞: 15 agents (協調開銷大)
# 3. 使用適當的拓撲
# 簡單任務: mesh
# 複雜項目: hierarchical
# 動態需求: adaptive
# 4. 啟用記憶緩存
npx claude-flow@alpha memory config - cache true
# 5. 使用神經網絡模式
npx claude-flow@alpha swarm \
 - use-pattern "best-practice" \
"任務"
```
 - -
### Q4: GitHub整合失敗怎麼辦?
**答**:
```bash
# 1. 檢查token權限
gh auth status
# 2. 重新配置
npx claude-flow@alpha github init - force
# 3. 驗證MCP設置
npx claude-flow@alpha mcp status
# 4. 檢查網絡連接
curl -I https://api.github.com
# 5. 查看詳細錯誤
npx claude-flow@alpha github pr-manager - verbose - debug
```
 - -
### Q5: 如何處理大型項目?
**答**:
```bash
# 策略: 分解 + 蜂巢 + 模式庫
# 1. 使用蜂巢智能
npx claude-flow@alpha hive-mind wizard
# 配置10+ agents
# 2. 分階段開發
# 階段1: 架構設計 (1週)
# 階段2: 核心功能 (2–3週)
# 階段3: 整合測試 (1週)
# 階段4: 優化部署 (1週)
# 3. 建立項目特定模式庫
npx claude-flow@alpha memory store project_patterns \
"項目開發模式與最佳實踐" \
 - namespace project - reasoningbank
# 4. 定期會話恢復
npx claude-flow@alpha hive-mind resume session-xxxxx
# 5. 導出知識庫
npx claude-flow@alpha hive-mind export-knowledge \
 - output project-knowledge.json
```
 - -
### Q6: 如何學習得更快?
**答**:
1. **每日實踐** (30–60分鐘)
```bash
# 每天完成一個小任務
npx claude-flow@alpha swarm "daily-task-$(date +%Y%m%d)" - claude
```
2. **建立個人模式庫**
```bash
# 記錄每次學到的模式
npx claude-flow@alpha memory store learning_$(date +%Y%m%d) \
"今天學到的…" \
 - namespace learning - reasoningbank
```
3. **參與社群**
- GitHub Issues討論
- Discord社群交流
- 分享項目經驗
4. **閱讀文檔**
- 每週閱讀一個高級主題
- 實作文檔中的範例
- 查看源碼理解實作
5. **逐步提升複雜度**
- 1: 簡單腳本
- 2: 小型應用
- 3: 中型項目
- 4: 大型系統
 - -
## 📊 技能評估表
使用此表評估你的Claude Flow技能等級:
### 🌱 入門級 
| 技能 | 熟練度 | 檢核 |
| - - - | - - - - | - - - |
| 安裝與配置 | ⭐⭐⭐⭐⭐ | [ ] |
| 基本CLI命令 | ⭐⭐⭐⭐⭐ | [ ] |
| 簡單Swarm任務 | ⭐⭐⭐⭐⭐ | [ ] |
| 記憶系統基礎 | ⭐⭐⭐⭐☆ | [ ] |
| SPARC TDD | ⭐⭐⭐☆☆ | [ ] |
**得分**: ___/5 ⭐ 需要 3+ ⭐ 進入下階段
 - -
### 🌿 初階級
| 技能 | 熟練度 | 檢核 |
| - - - | - - - - | - - - |
| Swarm拓撲選擇 | ⭐⭐⭐⭐⭐ | [ ] |
| 多Agent協調 | ⭐⭐⭐⭐☆ | [ ] |
| SPARC完整流程 | ⭐⭐⭐⭐⭐ | [ ] |
| 並行執行策略 | ⭐⭐⭐⭐☆ | [ ] |
| 記憶組織策略 | ⭐⭐⭐⭐☆ | [ ] |
**得分**: ___/5 ⭐ 需要 4+ ⭐ 進入下階段
 - -
### 🌳 進階級 
| 技能 | 熟練度 | 檢核 |
| - - - | - - - - | - - - |
| 蜂巢智能系統 | ⭐⭐⭐⭐⭐ | [ ] |
| MCP工具整合 | ⭐⭐⭐⭐☆ | [ ] |
| GitHub自動化 | ⭐⭐⭐⭐⭐ | [ ] |
| 複雜項目管理 | ⭐⭐⭐⭐☆ | [ ] |
| 性能優化基礎 | ⭐⭐⭐☆☆ | [ ] |
**得分**: ___/5 ⭐ 需要 4+ ⭐ 進入下階段
 - -
### 🌲 專家級 
| 技能 | 熟練度 | 檢核 |
| - - - | - - - - | - - - |
| 神經網絡自學習 | ⭐⭐⭐⭐☆ | [ ] |
| 企業架構設計 | ⭐⭐⭐⭐⭐ | [ ] |
| 性能深度優化 | ⭐⭐⭐⭐☆ | [ ] |
| Kubernetes部署 | ⭐⭐⭐⭐☆ | [ ] |
| 完整DevOps流程 | ⭐⭐⭐⭐⭐ | [ ] |
**得分**: ___/5 ⭐ 需要 4+ ⭐ 達到專家級
 - -
## 🎓 結業證書
當你完成所有階段並通過技能評估後，你可以自豪地聲稱:
```
🏆 Claude Flow 認證專家
━━━━━━━━━━━━━━━━━━━━━━━━━
姓名: __________________
完成日期: ______________
技能認證:
✅ 基礎操作與CLI命令
✅ Swarm協調與多Agent管理
✅ SPARC方法論深度應用
✅ 蜂巢智能系統運用
✅ GitHub完整整合
✅ MCP工具精通
✅ 神經網絡自學習
✅ 企業架構設計
✅ 性能優化專家
✅ 雲端部署運維
完成項目數: ___
累計開發時數: ___
━━━━━━━━━━━━━━━━━━━━━━━━━
Claude Flow v2.7.0-alpha.10
```
 - -
## 📚 延伸學習資源
### 官方文檔
- [README.md](../README.md) - 項目概覽
- [ARCHITECTURE.md](./architecture/ARCHITECTURE.md) - 系統架構
- [API_DOCUMENTATION.md](./api/API_DOCUMENTATION.md) - API參考
### 高級主題
- [NEURAL-MODULE.md](./integrations/reasoningbank/REASONING-AGENTS.md) - 神經網絡
- [GOAL-MODULE.md](./reference/GOAL-MODULE.md) - GOAP規劃
- [HIVE-MIND.md](./reference/HIVE-MIND.md) - 蜂巢智能
### 整合指南
- [GITHUB-INTEGRATION.md](./reference/GITHUB-INTEGRATION.md) - GitHub整合
- [MCP-SETUP-GUIDE.md](./setup/MCP-SETUP-GUIDE.md) - MCP設置
### 社群資源
- **GitHub**: [github.com/ruvnet/claude-flow](https://github.com/ruvnet/claude-flow)
- **Issues**: [問題回報與功能請求](https://github.com/ruvnet/claude-flow/issues)
- **Discord**: [Agentics Foundation社群](https://discord.com/invite/dfxmpwkG2D)
 - -
## 🎉 結語
恭喜你完成Claude Flow完整學習路線圖! 🎊
記住這些關鍵原則:
1. **並行執行是王道** - 始終在一個訊息中批次所有操作
2. **建立模式庫** - 記錄並重用成功的模式
3. **持續學習** - 使用神經網絡讓系統自我改進
4. **實踐為主** - 理論需要通過項目實踐鞏固
🔍 Claude Flow 程式碼流程深度分析
## 📖 目錄
1. [系統架構總覽](#系統架構總覽)
2. [核心啟動流程](#核心啟動流程)
3. [CLI命令處理系統](#cli命令處理系統)
4. [Swarm協調機制](#swarm協調機制)
5. [記憶管理系統](#記憶管理系統)
6. [MCP整合層](#mcp整合層)
7. [SPARC工作流](#sparc工作流)
8. [事件驅動架構](#事件驅動架構)
9. [資料流與狀態管理](#資料流與狀態管理)
10. [錯誤處理與恢復](#錯誤處理與恢復)
---
## 🏗️ 系統架構總覽
### 架構圖
```
┌─────────────────────────────────────────────────────────────────┐
│                      Claude Flow 系統架構                          │
└─────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────┐
│                        使用者介面層                                 │
├───────────────────────────────────────────────────────────────────┤
│  CLI Entry Point                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ bin/        │───>│ src/cli/     │───>│ cli-core.ts  │        │
│  │ claude-flow │    │ main.ts      │    │ (CLI類)      │        │
│  └─────────────┘    └──────────────┘    └──────────────┘        │
│                                                   │                │
│                                                   v                │
│                                          ┌──────────────┐         │
│                                          │ commands/    │         │
│                                          │ index.ts     │         │
│                                          └──────────────┘         │
└───────────────────────────────────────────────────────────────────┘
│
v
┌───────────────────────────────────────────────────────────────────┐
│                        命令路由層                                   │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ swarm.ts │  │ sparc.ts │  │ memory   │  │ init.ts  │         │
│  │          │  │          │  │ .ts      │  │          │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
└───────┼─────────────┼─────────────┼─────────────┼────────────────┘
│             │             │             │
v             v             v             v
┌───────────────────────────────────────────────────────────────────┐
│                        核心協調層                                   │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │ Orchestrator     │    │ SwarmCoordinator │                    │
│  │ orchestrator-    │    │ coordinator.ts   │                    │
│  │ fixed.ts         │    │                  │                    │
│  └────────┬─────────┘    └────────┬─────────┘                    │
│           │                       │                               │
│           v                       v                               │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │ Agent管理        │    │ Task分配         │                    │
│  │ - 生成Agents     │    │ - 任務分解       │                    │
│  │ - 生命週期管理   │    │ - 負載均衡       │                    │
│  │ - 狀態追蹤       │    │ - 優先級排序     │                    │
│  └──────────────────┘    └──────────────────┘                    │
└───────────────────────────────────────────────────────────────────┘
│
v
┌───────────────────────────────────────────────────────────────────┐
│                        基礎設施層                                   │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ EventBus     │  │ ConfigManager│  │ Logger       │           │
│  │ event-bus.ts │  │ config.ts    │  │ logger.ts    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                     │
│         v                 v                 v                     │
│  ┌─────────────────────────────────────────────────┐             │
│  │           事件驅動通信系統                        │             │
│  │  - agent:spawned    - task:completed            │             │
│  │  - swarm:started    - system:ready              │             │
│  └─────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────┘
│
v
┌───────────────────────────────────────────────────────────────────┐
│                        資料持久層                                   │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │ MemoryManager    │    │ Persistence      │                    │
│  │ manager.ts       │    │ json-persistence │                    │
│  └────────┬─────────┘    └────────┬─────────┘                    │
│           │                       │                               │
│           v                       v                               │
│  ┌─────────────────────────────────────────┐                     │
│  │ 儲存後端                                  │                     │
│  │ ┌──────────┐    ┌──────────────┐       │                     │
│  │ │ SQLite   │    │ ReasoningBank│       │                     │
│  │ │ Backend  │    │ (agentic-flow)│      │                     │
│  │ └──────────┘    └──────────────┘       │                     │
│  │ .swarm/memory.db                        │                     │
│  └─────────────────────────────────────────┘                     │
└───────────────────────────────────────────────────────────────────┘
│
v
┌───────────────────────────────────────────────────────────────────┐
│                        外部整合層                                   │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ MCP Server   │  │ GitHub       │  │ Neural       │           │
│  │ server.ts    │  │ Integration  │  │ Module       │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────────┘
```
### 模組職責表
| 層級 | 模組 | 檔案位置 | 核心職責 |
|------|------|---------|---------|
| **介面層** | CLI核心 | `src/cli/cli-core.ts` | 命令解析、參數處理、路由分發 |
| **介面層** | 命令系統 | `src/cli/commands/*.ts` | 各命令的具體實作 |
| **協調層** | Orchestrator | `src/core/orchestrator-fixed.ts` | Agent生命週期、任務分配 |
| **協調層** | SwarmCoordinator | `src/swarm/coordinator.ts` | Swarm拓撲、協調策略 |
| **基礎層** | EventBus | `src/core/event-bus.ts` | 事件發布/訂閱機制 |
| **基礎層** | ConfigManager | `src/core/config.ts` | 配置管理、安全性 |
| **資料層** | MemoryManager | `src/memory/manager.ts` | 記憶CRUD、索引、緩存 |
| **資料層** | Persistence | `src/core/json-persistence.ts` | SQLite資料持久化 |
| **整合層** | MCPServer | `src/mcp/server.ts` | MCP協議實作 |
---
## 🚀 核心啟動流程
### 1. 程式進入點
**檔案**: `bin/claude-flow.js` → `src/cli/main.ts`
```javascript
// src/cli/main.ts (第1-32行)
#!/usr/bin/env node
/**
* Claude-Flow CLI - Main entry point for Node.js
*/
import { CLI, VERSION } from './cli-core.js';
import { setupCommands } from './commands/index.js';
async function main() {
// 1. 創建CLI實例
const cli = new CLI('claude-flow', 'Advanced AI Agent Orchestration System');
// 2. 註冊所有命令
setupCommands(cli);
// 3. 執行CLI (解析process.argv)
await cli.run();
}
// 執行主函數
main().catch((error) => {
console.error('Fatal error:', error);
process.exit(1);
});
```
**流程步驟**:
```
┌──────────────────────────────────────────────────────────────┐
│ 步驟 1: 初始化CLI實例                                         │
├──────────────────────────────────────────────────────────────┤
│ new CLI('claude-flow', 'description')                        │
│   │                                                           │
│   ├─ 初始化命令映射 (commands: Map)                          │
│   ├─ 設置全局選項 (help, version, config, verbose)          │
│   └─ 準備命令註冊器                                          │
└──────────────────────────────────────────────────────────────┘
│
v
┌──────────────────────────────────────────────────────────────┐
│ 步驟 2: 註冊命令                                              │
├──────────────────────────────────────────────────────────────┤
│ setupCommands(cli)                                           │
│   │                                                           │
│   ├─ swarm命令     → swarmAction()                          │
│   ├─ sparc命令     → sparcAction()                          │
│   ├─ memory命令    → SimpleMemoryManager                    │
│   ├─ init命令      → 初始化專案                             │
│   ├─ neural命令    → 神經模組管理                           │
│   ├─ goal命令      → GOAP模組管理                           │
│   ├─ mcp命令       → MCP伺服器控制                          │
│   └─ 其他企業命令...                                         │
└──────────────────────────────────────────────────────────────┘
│
v
┌──────────────────────────────────────────────────────────────┐
│ 步驟 3: 解析並執行命令                                        │
├──────────────────────────────────────────────────────────────┤
│ cli.run(process.argv.slice(2))                              │
│   │                                                           │
│   ├─ parseArgs() - 解析命令行參數                            │
│   ├─ 檢查版本/幫助標誌                                        │
│   ├─ 從命令映射中查找命令                                     │
│   ├─ 載入配置文件 (可選)                                      │
│   └─ 執行命令的action()函數                                  │
└──────────────────────────────────────────────────────────────┘
```
---
### 2. CLI核心類實作
**檔案**: `src/cli/cli-core.ts` (第36-277行)
#### CLI類結構
```typescript
class CLI {
// === 核心屬性 ===
private commands: Map<string, Command> = new Map();  // 命令註冊表
private globalOptions: Option[] = [...];              // 全局選項定義
constructor(
private name: string,        // CLI名稱: 'claude-flow'
private description: string  // CLI描述
) {}
// === 核心方法 ===
// 1. 命令註冊
command(cmd: Command): this {
this.commands.set(cmd.name, cmd);
// 同時註冊別名
if (cmd.aliases) {
for (const alias of cmd.aliases) {
this.commands.set(alias, cmd);
}
}
return this;
}
// 2. 執行命令 (核心方法)
async run(args = process.argv.slice(2)): Promise<void> {
// 解析參數
const flags = this.parseArgs(args);
// 處理全局標誌
if (flags.version || flags.v) {
console.log(`${this.name} v${VERSION}`);
return;
}
// 提取命令名稱
const commandName = flags._[0]?.toString() || '';
// 顯示幫助
if (!commandName || flags.help || flags.h) {
this.showHelp();
return;
}
// 查找命令
const command = this.commands.get(commandName);
if (!command) {
console.error(`Unknown command: ${commandName}`);
process.exit(1);
}
// 構建命令上下文
const ctx: CommandContext = {
args: flags._.slice(1).map(String),  // 剩餘參數
flags: flags,                         // 所有標誌
config: await this.loadConfig(flags.config)  // 配置文件
};
// 執行命令action
try {
await command.action(ctx);
} catch (error) {
console.error(`Error executing command:`, error.message);
if (flags.verbose) {
console.error(error);
}
process.exit(1);
}
}
// 3. 參數解析器 (手動實作)
private parseArgs(args: string[]): Record<string, any> {
const result: Record<string, any> = { _: [] };
let i = 0;
while (i < args.length) {
const arg = args[i];
if (arg.startsWith('--')) {
// 長選項: --option value
const key = arg.slice(2);
if (i + 1 < args.length && !args[i + 1].startsWith('-')) {
result[key] = args[i + 1];
i += 2;
} else {
result[key] = true;  // 布爾標誌
i++;
}
} else if (arg.startsWith('-')) {
// 短選項: -o value
const key = arg.slice(1);
if (i + 1 < args.length && !args[i + 1].startsWith('-')) {
result[key] = args[i + 1];
i += 2;
} else {
result[key] = true;
i++;
}
} else {
// 位置參數
result._.push(arg);
i++;
}
}
return result;
}
// 4. 配置載入
private async loadConfig(configPath?: string): Promise<Record<string, unknown> | undefined> {
const configFile = configPath || 'claude-flow.config.json';
try {
const content = await fs.readFile(configFile, 'utf8');
return JSON.parse(content);
} catch {
return undefined;  // 配置文件可選
}
}
}
```
#### 參數解析範例
```bash
# 命令: npx claude-flow swarm "build API" --max-agents 5 --parallel
# 解析結果:
{
_: ['swarm', 'build API'],           # 位置參數
'max-agents': '5',                    # 長選項
parallel: true,                       # 布爾標誌
}
```
---
### 3. 命令註冊系統
**檔案**: `src/cli/commands/index.ts` (第56-90行)
```typescript
export function setupCommands(cli: CLI): void {
// === 神經模組命令 ===
cli.command({
name: 'neural',
description: 'Neural module management',
subcommands: [
{
name: 'init',
description: 'Initialize SAFLA neural module',
options: [
{ name: 'force', short: 'f', type: 'boolean' },
{ name: 'target', short: 't', type: 'string', defaultValue: '.claude/agents/neural' }
],
action: async (ctx: CommandContext) => {
const { initNeuralModule } = await import('../../scripts/init-neural.js');
await initNeuralModule({
force: ctx.flags.force as boolean,
targetDir: ctx.flags.target as string,
});
},
},
],
});
// === Goal模組命令 ===
cli.command({
name: 'goal',
description: 'Goal module management',
subcommands: [
{
name: 'init',
description: 'Initialize GOAP goal module',
options: [
{ name: 'force', type: 'boolean' },
{ name: 'target', type: 'string' }
],
action: async (ctx: CommandContext) => {
const { initGoalModule } = await import('../../scripts/init-goal.js');
await initGoalModule({ ... });
},
},
],
});
// === Swarm命令 (核心) ===
cli.command({
name: 'swarm',
description: 'Swarm orchestration',
action: swarmAction,  // 導入自 swarm.ts
options: [
{ name: 'strategy', type: 'string', description: 'Strategy type' },
{ name: 'max-agents', type: 'number', description: 'Max agents' },
{ name: 'parallel', type: 'boolean', description: 'Parallel execution' },
// ... 更多選項
]
});
// === SPARC命令 ===
cli.command({
name: 'sparc',
description: 'SPARC development methodology',
action: sparcAction,  // 導入自 sparc.ts
options: [
{ name: 'verbose', type: 'boolean' },
]
});
// === Memory命令 ===
cli.command({
name: 'memory',
description: 'Memory management',
action: async (ctx: CommandContext) => {
const manager = new SimpleMemoryManager(ctx.flags);
await manager.run(ctx.args);
}
});
// ... 其他命令 (init, mcp, config, status, monitor, session)
}
```
---
## 🐝 Swarm協調機制
### 1. Swarm命令入口
**檔案**: `src/cli/commands/swarm.ts` (第12-150行)
#### 執行流程
```typescript
export async function swarmAction(ctx: CommandContext) {
// === 步驟1: 參數解析 ===
const objective = ctx.args.join(' ').trim();
if (!objective) {
error('Usage: swarm <objective>');
return;
}
// 提取選項
const options = {
strategy: ctx.flags.strategy || 'auto',
maxAgents: ctx.flags.maxAgents || ctx.flags['max-agents'] || 5,
maxDepth: ctx.flags.maxDepth || ctx.flags['max-depth'] || 3,
research: ctx.flags.research || false,
parallel: ctx.flags.parallel || false,
memoryNamespace: ctx.flags.memoryNamespace || 'swarm',
timeout: ctx.flags.timeout || 60,
review: ctx.flags.review || false,
monitor: ctx.flags.monitor || false,
persistence: ctx.flags.persistence !== false,  // 預設true
distributed: ctx.flags.distributed || false,
};
// === 步驟2: 生成Swarm ID ===
const swarmId = generateId('swarm');  // 例如: swarm-1697542850123-a3f4e
// === 步驟3: Dry Run模式 (可選) ===
if (options.dryRun) {
warning('DRY RUN - Swarm Configuration:');
console.log(`Swarm ID: ${swarmId}`);
console.log(`Objective: ${objective}`);
console.log(`Strategy: ${options.strategy}`);
console.log(`Max Agents: ${options.maxAgents}`);
return;
}
// === 步驟4: 初始化Swarm協調器 ===
const coordinator = new SwarmCoordinator({
maxAgents: options.maxAgents,
maxConcurrentTasks: options.parallel ? options.maxAgents : 1,
taskTimeout: options.timeout * 60 * 1000,  // 轉換為毫秒
enableMonitoring: options.monitor,
enableWorkStealing: options.parallel,
enableCircuitBreaker: true,
memoryNamespace: options.memoryNamespace,
coordinationStrategy: options.distributed ? 'distributed' : 'centralized',
});
// === 步驟5: 初始化背景執行器 ===
const executor = new BackgroundExecutor({
maxConcurrentTasks: options.maxAgents,
defaultTimeout: options.timeout * 60 * 1000,
logPath: `./swarm-runs/${swarmId}/background-tasks`,
enablePersistence: options.persistence,
});
// === 步驟6: 初始化Swarm記憶 ===
const memoryManager = new SwarmMemoryManager({
namespace: options.memoryNamespace,
persistence: options.persistence,
});
// === 步驟7: 啟動協調器 ===
await coordinator.initialize();
success(`🐝 Swarm ${swarmId} initialized`);
console.log(`📋 Objective: ${objective}`);
console.log(`🎯 Strategy: ${options.strategy}`);
// === 步驟8: 執行任務分解與分配 ===
// (後續在coordinator內部處理)
}
```
---
### 2. SwarmCoordinator核心實作
**檔案**: `src/swarm/coordinator.ts` (第33-200行)
#### 類結構與狀態管理
```typescript
export class SwarmCoordinator extends EventEmitter implements SwarmEventEmitter {
// === 核心狀態 ===
private logger: Logger;
private config: SwarmConfig;
private swarmId: SwarmId;
// 狀態管理
private agents: Map<string, AgentState> = new Map();      // Agent狀態追蹤
private tasks: Map<string, TaskDefinition> = new Map();   // 任務定義
private objectives: Map<string, SwarmObjective> = new Map();  // 目標追蹤
// 執行狀態
private _isRunning: boolean = false;
private status: SwarmStatus = 'planning';  // planning | executing | paused | completed | failed
private startTime?: Date;
private endTime?: Date;
// 性能追蹤
private metrics: SwarmMetrics;
private events: SwarmEvent[] = [];
private lastHeartbeat: Date = new Date();
// 背景進程
private heartbeatTimer?: NodeJS.Timeout;
private monitoringTimer?: NodeJS.Timeout;
private cleanupTimer?: NodeJS.Timeout;
// 策略實例
private autoStrategy: AutoStrategy;
constructor(config: Partial<SwarmConfig> = {}) {
super();
// 配置Logger (根據設定調整日誌級別)
const logLevel = config.logging?.level || 'error';
this.logger = new Logger(
{ level: logLevel, format: 'text', destination: 'console' },
{ component: 'SwarmCoordinator' }
);
// 生成唯一Swarm ID
this.swarmId = this.generateSwarmId();
// 合併配置
this.config = this.mergeWithDefaults(config);
// 初始化指標
this.metrics = this.initializeMetrics();
// 初始化策略
this.autoStrategy = new AutoStrategy(config);
// 設置事件處理器
this.setupEventHandlers();
this.logger.info('SwarmCoordinator initialized', {
swarmId: this.swarmId.id,
mode: this.config.mode,
strategy: this.config.strategy,
});
}
}
```
#### 生命週期管理
```typescript
// === 初始化流程 ===
async initialize(): Promise<void> {
if (this._isRunning) {
throw new Error('Swarm coordinator already running');
}
this.logger.info('Initializing swarm coordinator...');
this.status = 'initializing';
try {
// 1. 驗證配置
const validation = await this.validateConfiguration();
if (!validation.valid) {
throw new Error(
`Configuration validation failed: ${validation.errors.map(e => e.message).join(', ')}`
);
}
// 2. 初始化子系統
await this.initializeSubsystems();
// 3. 啟動背景進程
this.startBackgroundProcesses();
// 4. 更新狀態
this._isRunning = true;
this.startTime = new Date();
this.status = 'executing';
// 5. 發送事件
this.emitSwarmEvent({
id: generateId('event'),
timestamp: new Date(),
type: 'swarm.started',
source: this.swarmId.id,
data: { swarmId: this.swarmId },
broadcast: true,
processed: false,
});
this.logger.info('Swarm coordinator initialized successfully');
} catch (error) {
this.status = 'failed';
this.logger.error('Failed to initialize swarm coordinator', { error });
throw error;
}
}
// === 關閉流程 ===
async shutdown(): Promise<void> {
if (!this._isRunning) {
return;
}
this.logger.info('Shutting down swarm coordinator...');
this.status = 'paused';
try {
// 1. 停止背景進程
this.stopBackgroundProcesses();
// 2. 優雅停止所有agents
await this.stopAllAgents();
// 3. 完成運行中的任務
await this.completeRunningTasks();
// 4. 儲存最終狀態
await this.saveState();
// 5. 更新狀態
this._isRunning = false;
this.endTime = new Date();
this.status = 'completed';
// 6. 發送完成事件
this.emitSwarmEvent({
id: generateId('event'),
timestamp: new Date(),
type: 'swarm.completed',
source: this.swarmId.id,
data: {
swarmId: this.swarmId,
metrics: this.metrics,
duration: this.endTime.getTime() - (this.startTime?.getTime() || 0),
},
broadcast: true,
processed: false,
});
this.logger.info('Swarm coordinator shut down successfully');
} catch (error) {
this.logger.error('Error during swarm coordinator shutdown', { error });
throw error;
}
}
// === 暫停/恢復 ===
async pause(): Promise<void> {
if (!this._isRunning || this.status === 'paused') {
return;
}
this.logger.info('Pausing swarm coordinator...');
this.status = 'paused';
// 暫停所有agents (實際實作略)
}
async resume(): Promise<void> {
if (!this._isRunning || this.status !== 'paused') {
return;
}
this.logger.info('Resuming swarm coordinator...');
this.status = 'executing';
// 恢復所有agents (實際實作略)
}
```
#### 背景進程
```typescript
private startBackgroundProcesses(): void {
// 1. Heartbeat (每10秒)
this.heartbeatTimer = setInterval(() => {
this.performHeartbeat();
}, 10000);
// 2. 監控 (每5秒)
if (this.config.enableMonitoring) {
this.monitoringTimer = setInterval(() => {
this.performMonitoring();
}, 5000);
}
// 3. 清理 (每60秒)
this.cleanupTimer = setInterval(() => {
this.performCleanup();
}, 60000);
}
private stopBackgroundProcesses(): void {
if (this.heartbeatTimer) {
clearInterval(this.heartbeatTimer);
}
if (this.monitoringTimer) {
clearInterval(this.monitoringTimer);
}
if (this.cleanupTimer) {
clearInterval(this.cleanupTimer);
}
}
private performHeartbeat(): void {
this.lastHeartbeat = new Date();
this.emitSwarmEvent({
id: generateId('event'),
timestamp: new Date(),
type: 'swarm.heartbeat',
source: this.swarmId.id,
data: {
agentCount: this.agents.size,
taskCount: this.tasks.size,
status: this.status,
},
broadcast: false,
processed: false,
});
}
```
---
### 3. Agent與Task管理
#### Agent狀態追蹤
```typescript
interface AgentState {
id: AgentId;
type: AgentType;
status: 'idle' | 'busy' | 'failed' | 'terminated';
currentTask?: TaskId;
completedTasks: TaskId[];
capabilities: string[];
performance: {
successRate: number;
avgCompletionTime: number;
totalTasks: number;
};
createdAt: Date;
lastActive: Date;
}
// Agent派遣
async spawnAgent(profile: AgentProfile): Promise<AgentId> {
const agentId = generateId('agent');
const state: AgentState = {
id: agentId,
type: profile.type,
status: 'idle',
currentTask: undefined,
completedTasks: [],
capabilities: profile.capabilities,
performance: {
successRate: 1.0,
avgCompletionTime: 0,
totalTasks: 0,
},
createdAt: new Date(),
lastActive: new Date(),
};
this.agents.set(agentId, state);
this.emitSwarmEvent({
type: 'agent.spawned',
data: { agentId, profile },
// ...
});
return agentId;
}
// Agent終止
async terminateAgent(agentId: AgentId): Promise<void> {
const agent = this.agents.get(agentId);
if (!agent) {
throw new Error(`Agent ${agentId} not found`);
}
// 重新分配當前任務
if (agent.currentTask) {
await this.reassignTask(agent.currentTask);
}
agent.status = 'terminated';
this.agents.delete(agentId);
this.emitSwarmEvent({
type: 'agent.terminated',
data: { agentId, reason: 'User requested' },
// ...
});
}
```
#### Task分配與執行
```typescript
interface TaskDefinition {
id: TaskId;
type: TaskType;
description: string;
priority: TaskPriority;  // 'critical' | 'high' | 'normal' | 'low'
status: TaskStatus;      // 'pending' | 'assigned' | 'running' | 'completed' | 'failed'
assignedAgent?: AgentId;
dependencies: TaskId[];
result?: any;
error?: string;
createdAt: Date;
startedAt?: Date;
completedAt?: Date;
}
// 任務提交
async submitTask(task: TaskDefinition): Promise<TaskId> {
const taskId = generateId('task');
task.id = taskId;
task.status = 'pending';
task.createdAt = new Date();
this.tasks.set(taskId, task);
// 立即嘗試分配
await this.assignTask(taskId);
this.emitSwarmEvent({
type: 'task.submitted',
data: { taskId, task },
// ...
});
return taskId;
}
// 智能任務分配
private async assignTask(taskId: TaskId): Promise<void> {
const task = this.tasks.get(taskId);
if (!task) return;
// 檢查依賴
const dependenciesMet = task.dependencies.every(depId => {
const dep = this.tasks.get(depId);
return dep && dep.status === 'completed';
});
if (!dependenciesMet) {
return;  // 等待依賴完成
}
// 查找最佳agent
const bestAgent = this.findBestAgent(task);
if (!bestAgent) {
return;  // 沒有可用agent
}
// 分配任務
task.assignedAgent = bestAgent.id;
task.status = 'assigned';
bestAgent.status = 'busy';
bestAgent.currentTask = taskId;
this.emitSwarmEvent({
type: 'task.assigned',
data: { taskId, agentId: bestAgent.id },
// ...
});
// 開始執行
await this.executeTask(taskId);
}
// 最佳Agent選擇算法
private findBestAgent(task: TaskDefinition): AgentState | undefined {
const candidates = Array.from(this.agents.values())
.filter(agent =>
agent.status === 'idle' &&
this.hasRequiredCapabilities(agent, task)
);
if (candidates.length === 0) {
return undefined;
}
// 根據性能評分排序
candidates.sort((a, b) => {
const scoreA = a.performance.successRate * (1 / (a.performance.avgCompletionTime || 1));
const scoreB = b.performance.successRate * (1 / (b.performance.avgCompletionTime || 1));
return scoreB - scoreA;  // 降序
});
return candidates[0];
}
```
---
## 💾 記憶管理系統
### 1. MemoryManager架構
**檔案**: `src/memory/manager.ts` (第47-200行)
#### 多層記憶架構
```
┌─────────────────────────────────────────────────────────┐
│                    記憶管理架構                           │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Layer 1: MemoryManager (管理層)                         │
├─────────────────────────────────────────────────────────┤
│  - createBank()      創建agent專用記憶庫                │
│  - store()           儲存記憶條目                        │
│  - retrieve()        檢索單個記憶                        │
│  - query()           複雜查詢                            │
│  - update()          更新記憶                            │
│  - delete()          刪除記憶                            │
└─────────────────────────────────────────────────────────┘
│
┌──────────────┼──────────────┐
│              │              │
v              v              v
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Layer 2: Cache  │ │ Layer 2: Index  │ │ Layer 2: Backend│
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ MemoryCache     │ │ MemoryIndexer   │ │ IMemoryBackend  │
│                 │ │                 │ │                 │
│ - LRU緩存       │ │ - 關鍵字索引    │ │ ┌─────────────┐ │
│ - 熱數據        │ │ - 全文搜尋      │ │ │SQLiteBackend│ │
│ - 快速訪問      │ │ - 時間索引      │ │ └─────────────┘ │
│                 │ │ - 類型索引      │ │ ┌─────────────┐ │
│ 訪問: O(1)      │ │                 │ │ │ Markdown    │ │
│ 容量: 配置限制  │ │ 訪問: O(log n)  │ │ │ Backend     │ │
│                 │ │                 │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│
v
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Storage (儲存層)                                │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐  ┌──────────────────────────┐  │
│ │ SQLite Database     │  │ ReasoningBank            │  │
│ │ .swarm/memory.db    │  │ (agentic-flow@1.5.13)   │  │
│ │                     │  │                          │  │
│ │ Tables:             │  │ - semantic_search()      │  │
│ │ - memories          │  │ - patterns               │  │
│ │ - embeddings        │  │ - embeddings             │  │
│ │ - indexes           │  │ - trajectories           │  │
│ │ - metadata          │  │ - links                  │  │
│ └─────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```
#### MemoryManager核心實作
```typescript
export class MemoryManager implements IMemoryManager {
private backend: IMemoryBackend;
private cache: MemoryCache;
private indexer: MemoryIndexer;
private banks = new Map<string, MemoryBank>();  // Agent記憶庫
private initialized = false;
private syncInterval?: number;
constructor(
private config: MemoryConfig,
private eventBus: IEventBus,
private logger: ILogger,
) {
// 1. 初始化後端 (SQLite or Markdown)
this.backend = this.createBackend();
// 2. 初始化緩存 (LRU)
this.cache = new MemoryCache(
this.config.cacheSizeMB * 1024 * 1024,  // 轉換為bytes
this.logger,
);
// 3. 初始化索引器
this.indexer = new MemoryIndexer(this.logger);
}
// === 初始化 ===
async initialize(): Promise<void> {
if (this.initialized) {
return;
}
this.logger.info('Initializing memory manager...');
try {
// 1. 初始化後端
await this.backend.initialize();
// 2. 建立索引 (從現有記憶)
const allEntries = await this.backend.getAllEntries();
await this.indexer.buildIndex(allEntries);
// 3. 啟動同步定時器
this.startSyncInterval();
this.initialized = true;
this.logger.info('Memory manager initialized');
} catch (error) {