# Claude Agent SDK 與 DeepAgents:Harness Engineering 實戰

> 對應 [全景圖 #13/#14](../../2024-2026_AI完整領域全景圖.md);搭配 [`./LangGraph_supervisor_handoff_實戰.md`](./LangGraph_supervisor_handoff_實戰.md)

2025 下半年開始,Agent 框架的主流共識從「框架抽象 (framework abstraction)」轉向「harness engineering」——也就是把模型放進一個帶 tool、memory、context 管理的執行環境中,讓模型自己決定下一步。本文介紹兩個此路線的代表:Anthropic 的 **Claude Agent SDK** 與 LangChain 的 **DeepAgents**。

---

## 1. Claude Agent SDK:Claude Code 開放給世界

2026 年 3 月,Anthropic 把原本的 **Claude Code SDK** 正式更名為 **Claude Agent SDK**。這不只是改名,而是宣告:過去支撐 Claude Code 這個終端 IDE 的整套 harness——包含工具系統、子 agent、skills、hooks、context 自動壓縮——現在以 Python 和 TypeScript 套件的形式開放給開發者,可以用來打造你自己的 agent。

Python 套件名 `claude-agent-sdk`(舊名 `claude_code_sdk` 已 deprecated),TypeScript 是 `@anthropic-ai/claude-agent-sdk`。Python 端的型別 `ClaudeCodeOptions` 也改名為 `ClaudeAgentOptions`。底層其實是一個內嵌的 Claude Code CLI binary,SDK 透過 stdio 與其溝通,讓你的 Python 程式直接得到檔案操作、終端命令、多步工作流串接的能力。

## 2. 核心特性

Claude Agent SDK 提供五個關鍵能力:

- **Compaction(自動壓縮)**:context window 接近上限時,舊的 tool output 會先被丟掉,接著對話會被自動摘要。你不必再手動處理 token 數。
- **Hooks(生命週期掛鉤)**:`PreToolUse`、`PostToolUse`、`UserPromptSubmit`、`Stop`、`SessionStart`、`PreCompact` 等事件可以插入 Python 函式,執行 lint、權限檢查、secret scan、policy gate。
- **Skills(技能系統)**:Markdown + 腳本檔案,模型自己決定何時載入。比 tool 更輕量、比 prompt 更具結構。
- **Subagent(子代理)**:每個 subagent 有獨立的 context window,執行完只回傳最終結果,不汙染主對話。
- **Filesystem-as-memory**:用真實檔案系統當持久記憶,而不是塞進 context 或外部 vector DB。`CLAUDE.md`、`./skills/`、`./.claude/` 都是這套設計的一部分。

## 3. 與 OpenAI Agents SDK / LangGraph 對比

| 維度 | Claude Agent SDK | OpenAI Agents SDK | LangGraph |
|---|---|---|---|
| 哲學 | Production-grade 單一 agent | Multi-agent handoff | 顯式狀態機 |
| 控制流 | 模型驅動 (model-led) | Handoff DSL | 圖節點 + reducer |
| State | Session/檔案系統 | Run object | `StateGraph` + checkpointer |
| 適用 | Coding、長期任務 | Customer routing | 複雜多步、HITL |
| 強項 | Skills、subagent context isolation | 簡潔的 handoff | `interrupt()` 人機協作 |

OpenAI 2026 年 4 月也補上了 harness(file ops、code exec、sandbox 整合 E2B/Modal/Daytona 等七家),但 multi-agent handoff 仍是其招牌。LangGraph 適合需要精細狀態管理與 human-in-the-loop checkpoint 的工作流。Claude Agent SDK 則最像「把 Claude Code 當函式庫用」。

## 4. 環境準備

```bash
pip install claude-agent-sdk
export ANTHROPIC_API_KEY=sk-ant-...
# 或 export CLAUDE_CODE_USE_BEDROCK=1 走 AWS Bedrock
```

需要 Python 3.10+,且底層會在第一次執行時下載 Claude Code CLI binary。Node 端只要 `npm i @anthropic-ai/claude-agent-sdk`。

## 5. 完整起手式:Code Refactor Agent

```python
import asyncio
from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, tool, create_sdk_mcp_server,
)

@tool("run_tests", "Run pytest in the workspace", {"path": str})
async def run_tests(args):
    import subprocess
    out = subprocess.run(
        ["pytest", args["path"], "-q"], capture_output=True, text=True
    )
    return {"content": [{"type": "text", "text": out.stdout + out.stderr}]}

mcp = create_sdk_mcp_server(
    name="refactor-tools", version="0.1.0", tools=[run_tests],
)

options = ClaudeAgentOptions(
    system_prompt=(
        "You are a senior Python refactoring assistant. "
        "Always: (1) read code, (2) propose plan, (3) edit, (4) run tests."
    ),
    mcp_servers={"refactor": mcp},
    allowed_tools=["Read", "Edit", "Bash", "mcp__refactor__run_tests"],
    permission_mode="acceptEdits",
    max_turns=20,
)

async def main():
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Refactor src/parser.py: extract pure functions, "
            "add type hints, then run tests."
        )
        async for msg in client.receive_response():
            print(msg)

asyncio.run(main())
```

短短 40 多行就拿到一個會讀檔、編輯、跑測試、自我驗證的 refactor agent。

## 6. Skills 系統

Skill 是 `~/.claude/skills/<name>/SKILL.md`(或專案的 `.claude/skills/`)。SKILL.md 的 frontmatter 描述何時觸發,內容描述步驟與引用的 script。**模型一開始只看到 skill 的名字與描述,需要時才把內容讀進來**——這個「漸進載入 (progressive disclosure)」是 skills 比 system prompt 更省 token 的核心理由。例如可以做:

- `code-review-skill/SKILL.md`:給定一個 diff,如何進行多語言 code review。
- `release-notes/SKILL.md`:從 git log 產生 changelog 的 SOP。
- `pgsql-perf/SKILL.md`:看到慢 SQL 時如何 EXPLAIN 分析。

Skills 是可攜的——同一個 skill 在 Claude Code CLI、Messages API、Agent SDK 三個地方都能用。

## 7. Subagent 機制

主 agent 透過 `Task` 工具呼叫一個 subagent,subagent 在自己的 context 中跑數十次 tool call、看大量檔案、做密集搜尋,最後只回一段「我找到的結論」給主 agent。這解決了一個老問題:**深度探索與長對話互相汙染**。你可以在 `.claude/agents/researcher.md` 定義 subagent 的 system prompt、可用 tool、模型(便宜的 Haiku 給 grep,Opus 給 reasoning)。

## 8. Hooks:把策略門檻寫進 harness

```python
async def block_secrets(input, tool_use_id, context):
    if "AWS_SECRET" in str(input.get("tool_input", {})):
        return {"behavior": "deny", "reason": "secret detected"}
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [{"matcher": "Bash", "hooks": [block_secrets]}],
        "PostToolUse": [{"matcher": "Edit",
                         "hooks": [run_linter_and_feedback]}],
        "Stop": [{"hooks": [save_session_summary]}],
    },
)
```

`PreToolUse` 可以否決 tool 呼叫;`PostToolUse` 可以把 lint 錯誤回灌給模型讓它自我修復;`Stop` 可以做 session summary 或 evals。Hooks 是 deterministic 的,跟 LLM 的隨機性互補。

## 9. DeepAgents:LangChain 的同路線答案

DeepAgents 是 LangChain 在 2024 末推出、2025 持續演進的 agent harness,蓋在 LangGraph 之上。它直接借鑑 Claude Code 的設計理念:**planner + executor + memory + sub-agent + 虛擬檔案系統**。

DeepAgents 內建三個中介層 (middleware):

- **`write_todos` middleware**:強制模型先列 todo list 才動手,大幅改善長任務的一致性。
- **Filesystem middleware**:虛擬檔案系統,backend 可插拔(in-memory、本地磁碟、LangGraph Store、Modal/Daytona sandbox)。
- **SubAgent middleware**:提供 `task` 工具,可以 spawn 帶獨立 context 的子 agent。

DeepAgents 與 LangGraph 完全相容,所以可以直接用 LangGraph 的 checkpointer、interrupt()、Memory Store。

## 10. DeepAgents 範例:研究 Agent

```python
import os
from deepagents import create_deep_agent, SubAgent
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(query: str, max_results: int = 5):
    """Search the web via Tavily and return result snippets."""
    return tavily.search(query, max_results=max_results)

research_sub = SubAgent(
    name="researcher",
    description="Deep-dive into one specific topic and return notes.",
    prompt=(
        "You research ONE sub-topic thoroughly. Use internet_search "
        "freely, save findings to /notes/<topic>.md, return a summary."
    ),
    tools=["internet_search"],
)

agent = create_deep_agent(
    tools=[internet_search],
    instructions=(
        "You are a research lead. Plan with write_todos, "
        "spawn `researcher` subagents per sub-topic in parallel, "
        "synthesize findings into /report.md, return that file."
    ),
    subagents=[research_sub],
)

result = agent.invoke({"messages": [
    {"role": "user",
     "content": "Compare Claude Agent SDK vs DeepAgents for 2026."}
]})
print(result["files"]["/report.md"])
```

執行時你會看到 agent 先寫 todo、spawn 多個 `researcher`(各自獨立 context)、寫檔到虛擬 FS、最後組合報告。

## 11. Harness Engineering 心法

這兩個框架的設計反映了 2025 年圈內共識:

- **Context curation**:不是把所有東西塞進 prompt,而是讓 agent 知道「有什麼可以讀」(skills、filesystem、subagent)再 on-demand 載入。
- **Tool ergonomics**:工具描述要寫給模型看;tool input schema 要簡單;返回值要結構化方便 chain。
- **Verification loop**:跑 test、跑 lint、跑 type check,把結果回灌給 agent,讓它自我修復——比一次大 prompt 更可靠。
- **Context isolation**:深度探索丟給 subagent,主 thread 保持乾淨。
- **Deterministic guardrails**:hooks / middleware 處理權限、secret、policy,不靠 LLM 自律。

## 12. 真實案例

- **Anthropic 內部:Claude Code 本身**就是用這套 harness 蓋的。它的 plan mode、`/compact`、`CLAUDE.md` memory、SubAgent 機制(Task tool)、Hooks(PreToolUse 攔截 dangerous bash)、Skills(Anthropic 內建 + 使用者自訂),全部都是 Agent SDK 開放出來的同一組原語。
- **DeepAgents 在學術研究工作流**:不少 lab 用 DeepAgents 做文獻 review——planner 拆主題、parallel subagents 抓 arxiv、filesystem 累積筆記、最後產生 LaTeX 草稿。比起單一 ReAct loop 在 50+ 步後就跑偏,DeepAgents 的 context isolation + plan-based execution 讓 200+ 步的任務仍可控。社群已有 `open_deep_research` 等模板可直接 fork。

兩個框架走的是同一條路:**模型即作業系統的核心,框架的工作是把 OS 周邊蓋好**。選哪個取決於生態——重 Anthropic / Claude Code 工作流的選 Claude Agent SDK,重 LangChain / LangGraph 既有資產的選 DeepAgents。兩者甚至可以混搭(在 DeepAgents 裡用 Claude 模型呼叫 MCP server)。Harness engineering 正在取代 framework engineering,成為 agent 開發的主戰場。

---

**參考來源:**

- [Migrate to Claude Agent SDK — Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/migration-guide)
- [Agent SDK overview — Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/overview)
- [claude-agent-sdk-python — GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [Deep Agents overview — Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/overview)
- [langchain-ai/deepagents — GitHub](https://github.com/langchain-ai/deepagents)
- [Comparison with Claude Agent SDK — LangChain Docs](https://docs.langchain.com/oss/python/deepagents/comparison)
- [2026 AI Agent Framework Showdown — QubitTool](https://qubittool.com/blog/ai-agent-framework-comparison-2026)
- [Inside the Claude Agents SDK — ML6 Blog](https://www.ml6.eu/en/blog/inside-the-claude-agents-sdk-lessons-from-the-ai-engineer-summit)
