# AI Agent 與 Agentic Workflows

完整的 AI Agent 學習資源,從理論到實踐,涵蓋主流框架和實戰案例。

---

## ⚡ 一鍵跑 demos(2026-05 新增)

📒 **[`notebooks/Colab_LangGraph_Multi_Agent_Research_Demo.ipynb`](./notebooks/Colab_LangGraph_Multi_Agent_Research_Demo.ipynb)** — **30 分鐘跑通 3-agent supervisor + handoff**

- **架構**:Supervisor → Planner / Researcher / Writer + Tavily web search tool
- **環境**:Colab CPU(不需 GPU,全 API)、需 OpenAI + Tavily 兩把 key
- **成本**:單次 demo run 約 $0.03-0.10
- **產出**:結構化 Markdown 研究報告(含引用)
- **整合**:第 14 節列 phantom-mesh 7 條真實工程考量(checkpoint persistence、parallel fan-out、cost tracking、provider fallback、streaming SSE、HITL、tool sandbox)
- **延伸**:8 個練習(加 Critic / 並行 / 多 tool / Fact-checker / HITL / MCP / Mem0 / Eval)

對應 deep-dive:[`./LangGraph_supervisor_handoff_實戰.md`](./LangGraph_supervisor_handoff_實戰.md);系統設計案例:[Case_04 Multi-Agent Research](../../9.面試準備與職業發展/2.系統設計案例/Case_04_Multi_Agent_Research_System.md)。

---

## 📚 學習資源

### 主要文檔

#### [AI_Agents_與_Agentic_Workflows_2024-2025.md](./AI_Agents_與_Agentic_Workflows_2024-2025.md)

**完整的 AI Agent 指南**，包含：

1. **AI Agent 核心概念** - 什麼是 Agent、核心組件、設計原則
2. **ReAct 模式與思維鏈** - Reasoning + Acting 模式詳解
3. **主流框架深度解析** - LangGraph、CrewAI、AutoGPT、AutoGen
4. **框架比較與選擇指南** - 決策樹、場景匹配、混合策略
5. **實戰案例** - AI 研究助手、客服自動化系統
6. **Agent 工具設計與整合** - MCP 協議、工具開發
7. **評估與監控** - 性能指標、成本追蹤、LangSmith
8. **最佳實踐與設計模式** - 設計原則、提示工程、安全性
9. **未來趨勢與展望** - 多模態、自主學習、Agent OS
10. **延伸閱讀** - 論文、文檔、開源項目

## 💻 實戰範例

### [examples/](./examples/)

**完整的可運行程式碼範例**，所有程式碼都經過測試並可直接執行：

#### 📁 範例結構

```
examples/
├── 01_react_agent/           ✅ ReAct 模式 Agent
│   ├── README.md            # 詳細說明文檔
│   └── react_agent_basic.py # 可運行的基礎範例
├── 02_langgraph_agent/       🚧 LangGraph 狀態機 Agent
├── 03_crewai_multi_agent/    🚧 CrewAI 多 Agent 協作
├── 04_autogen_conversational/🚧 AutoGen 對話式 Agent
├── 05_ai_research_assistant/ 🚧 完整實戰：AI 研究助手
├── utils/                    ✅ 共用工具模組
│   ├── agent_utils.py       # Agent 開發工具
│   ├── cost_tracker.py      # 成本追蹤器
│   ├── evaluator.py         # 性能評估器
│   ├── logger.py            # 日誌配置
│   └── prompt_templates.py  # 提示模板集合
├── requirements.txt          ✅ 依賴套件
├── .env.example             ✅ 環境變數模板
└── README.md                ✅ 詳細使用說明
```

圖例：✅ 已完成 | 🚧 進行中 | 📝 規劃中

#### 🚀 快速開始

```bash
# 1. 進入範例目錄
cd examples/

# 2. 安裝依賴（建議使用虛擬環境）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 配置環境變數（可選，某些範例需要）
cp .env.example .env
# 編輯 .env 文件，添加你的 API 金鑰

# 4. 運行範例
cd 01_react_agent/
python react_agent_basic.py  # 不需要 API 金鑰
```

#### 📖 各範例說明

##### 01. ReAct Agent ✅

展示 **Reasoning + Acting** 模式的核心實作：

- `react_agent_basic.py` - 使用模擬工具的基礎範例（可直接運行）
- 展示 Thought → Action → Observation 循環
- 不需要 API 金鑰

**學習目標**：
- 理解 ReAct 循環機制
- 掌握工具定義和呼叫
- 學習提示工程技巧

##### 02. LangGraph Agent 🚧

使用 **LangGraph** 構建狀態機 Agent（開發中）

**將包含**：
- 基礎狀態圖 Agent
- 條件路由和分支
- 人機協作範例
- 狀態持久化

##### 03. CrewAI 多 Agent 🚧

展示 **多 Agent 協作**完成複雜任務（開發中）

**將包含**：
- 研究團隊協作範例
- 角色分工和任務編排
- 階層式流程
- 自定義工具整合

##### 04. AutoGen 對話式 Agent 🚧

展示 **AutoGen** 的對話式 AI 系統（開發中）

**將包含**：
- 基礎對話 Agent
- 程式碼執行 Agent
- 多 Agent 群聊
- 教學助手範例

##### 05. AI 研究助手（完整項目）🚧

**生產級端到端項目**（開發中）

**功能**：
- 自動搜尋和收集技術資料
- 智能分析和總結
- 生成結構化技術報告
- 程式碼示例生成
- 成本追蹤和監控

## 🛠 工具與輔助模組

### utils/ 目錄

可重用的工具函數和類別：

- **agent_utils.py** - Agent 開發工具
  - `setup_environment()` - 環境設置
  - `get_llm()` - 獲取 LLM 實例
  - `create_tool_from_function()` - 建立工具
  - `safe_execute()` - 安全執行（帶重試）

- **cost_tracker.py** - API 成本追蹤
  - `CostTracker` 類別
  - 追蹤不同模型的使用量和成本
  - 生成詳細的成本報告

- **evaluator.py** - Agent 性能評估
  - `AgentEvaluator` 類別
  - 使用 LLM 評估 Agent 輸出品質
  - 生成評估報告

- **logger.py** - 統一日誌配置
  - `setup_logger()` - 配置 logger
  - 支援文件和控制台輸出

- **prompt_templates.py** - 提示模板集合
  - ReAct 提示模板
  - Agent 系統提示
  - Few-Shot 範例
  - 各種任務專用模板

## 📊 框架比較

| 框架 | 學習曲線 | 控制精度 | 多 Agent | 最適場景 |
|------|---------|---------|----------|---------|
| **LangGraph** | 中-高 | 極高 | 手動編排 | 複雜業務流程 |
| **CrewAI** | 低-中 | 中 | 原生支援 | 內容創作、團隊協作 |
| **AutoGPT** | 中 | 低 | 不支援 | 實驗性自主任務 |
| **AutoGen** | 中 | 高 | 原生支援 | 研究助手、程式碼執行 |

**選擇建議**：
- 需要精確控制流程 → **LangGraph**
- 快速構建多 Agent 系統 → **CrewAI**
- 人機對話式交互 → **AutoGen**
- 完全自主執行（慎用） → **AutoGPT**

## 🎯 學習路徑

### 初學者路徑

1. **閱讀主文檔** - 理解 AI Agent 核心概念
2. **運行基礎範例** - `examples/01_react_agent/`
3. **學習工具使用** - 查看 `utils/` 模組
4. **實作簡單 Agent** - 使用提供的模板

### 進階路徑

1. **深入框架** - LangGraph、CrewAI、AutoGen
2. **研究實戰案例** - 主文檔第 6 章
3. **學習最佳實踐** - 主文檔第 9 章
4. **構建完整項目** - 參考 `05_ai_research_assistant/`

### 專家路徑

1. **性能優化** - 成本控制、並行執行、快取
2. **安全性** - 輸入驗證、權限控制、審計
3. **生產部署** - 監控、評估、持續改進
4. **貢獻開源** - 分享你的實作和經驗

## 📖 補充資源

### 論文

- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Toolformer](https://arxiv.org/abs/2302.04761)

### 官方文檔

- [LangChain](https://python.langchain.com/docs/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)
- [AutoGen](https://microsoft.github.io/autogen/)

### 開源項目

- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [AutoGen GitHub](https://github.com/microsoft/autogen)

## 🤝 貢獻指南

歡迎貢獻！可以：

- 提交 Issue 報告問題
- 提交 PR 改進程式碼或文檔
- 分享你的 Agent 實作範例
- 改進現有範例

## ❓ 常見問題

**Q: 需要哪些先備知識？**
A: 基礎的 Python 編程和對 LLM 的基本了解。

**Q: 需要付費 API 嗎？**
A: 基礎範例可以用模擬工具運行。實際應用需要 OpenAI 或其他 LLM API。

**Q: 運行範例的成本如何？**
A: 基礎範例 < $0.10，複雜範例 $0.50-$2.00。建議先用 gpt-3.5-turbo 測試。

**Q: 如何選擇框架？**
A: 參考「框架比較」章節和主文檔的「框架比較與選擇指南」。

## 📧 聯繫與支持

- 問題或建議：提交 GitHub Issue
- 討論交流：查看項目 Discussions

---

**最後更新**: 2024/2025
**版本**: 2.0.0

**更新日誌**：
- ✅ 補充主文檔 6 個缺失章節
- ✅ 建立完整的 examples 資料夾結構
- ✅ 實作工具模組（utils/）
- ✅ 添加 ReAct Agent 可運行範例
- 🚧 其他框架範例開發中...

## 延伸閱讀
- [14.Voice_Audio_AI](../../14.Voice_Audio_AI/README.md) — Voice Agent 端到端
- [20.Generative_UI](../../20.Generative_UI/README.md) — Tool-rendered UI 模式
- [13.Robotics_Embodied_AI](../../13.Robotics_Embodied_AI/README.md) — VLA 同源(GUI Agent ↔ Robot Agent)
- [22.Self_Improving_AI](../../22.Self_Improving_AI/README.md) — Multi-Agent 自我改進
