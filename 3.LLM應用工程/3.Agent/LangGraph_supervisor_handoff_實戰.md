> 對應 [全景圖 #13](../../2024-2026_AI完整領域全景圖.md);搭配 [`../11.MCP協議與工具調用/`](../11.MCP協議與工具調用/);[`../11.MCP協議與工具調用/MCP_server_完整開發.md`](../11.MCP協議與工具調用/MCP_server_完整開發.md)(B4 即將生成)

# LangGraph Supervisor + Handoff 實戰:打造可控的多 Agent 系統

> 從 ReAct 單體到 supervisor 拓樸,從 in-memory state 到 production checkpoint,本篇是一份可貼上就跑的 LangGraph 0.4 全景指南。

---

## 1. 為什麼 2026 年 LangGraph 仍是 stateful agent 的王者

2024 的 agent 熱潮過後,大家發現「能 demo」跟「能上線」差了不只一個量級。CrewAI 靠 DSL 贏在原型,AutoGen 靠對話贏在研究情境,但要把 agent 放進需要 SLA、retry、人工審批、可重放的真實系統,**LangGraph 是 2026 年公認的標配**。

它的賣點不在「會呼叫 LLM」,而在四個底層能力堆出來的工程性:

1. **StateGraph**:把 agent flow 顯式建模成有向圖,節點和邊都是程式碼可審計的物件,告別「黑盒 prompt chain」。
2. **Checkpointer**:每一個 super-step 都會把整張圖的 state snapshot 落地,意味著流程可以中斷數天再恢復,可以 time-travel 回到任一步重跑。
3. **interrupt()**:原生的 human-in-the-loop pause,呼叫端拿到 `__interrupt__` payload,做完決策後用 `Command(resume=...)` 餵回去,完全不用自己寫 polling 與 message queue。
4. **LangSmith tracing**:每個 node、tool call、token usage、latency 都自動上報,debug 從「印 log」升級成「看時間軸」。

對手框架要做到上述任一項,通常得自己堆 Redis + state machine + websocket。LangGraph 把這些做進預設值,所以企業選型清單會把它放在最前面。

---

## 2. 核心抽象:5 個必須背下來的名詞

| 抽象 | 角色 | 你會寫的程式碼 |
|------|------|--------------|
| **State** | 一個 TypedDict / Pydantic model,描述整張圖的共享資料 | `class S(TypedDict): messages: Annotated[list, add_messages]` |
| **Node** | 一個純函式:`(state) -> partial_state`,接收狀態回傳要 merge 的部分 | `def researcher(state): ...` |
| **Edge** | 固定路徑:從 A 一定走到 B | `graph.add_edge("A", "B")` |
| **Conditional Edge** | 由函式決定下一個 node:`(state) -> "node_name"` | `graph.add_conditional_edges("supervisor", route_fn)` |
| **Checkpoint** | 每個 super-step 的完整 state 快照,綁在 `thread_id` 上 | `graph.compile(checkpointer=SqliteSaver(...))` |

關鍵理解:**State 的 reducer (`Annotated[list, add_messages]`)** 決定多個 node 同時更新時的合併規則,這也是 LangGraph 能跑平行 branch 的基礎。

---

## 3. 起手式:30 行寫一個 ReAct Agent

最小可運行的 LangGraph agent,內建 tool calling、loop、stop condition:

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查某個城市天氣"""
    return f"{city} 今天 22 度多雲"

@tool
def get_stock_price(ticker: str) -> float:
    """查股價"""
    return {"AAPL": 195.3, "NVDA": 920.1}.get(ticker, 0.0)

model = ChatAnthropic(model="claude-opus-4-7")
agent = create_react_agent(
    model=model,
    tools=[get_weather, get_stock_price],
    prompt="你是一個會用工具的助理,先用工具再回答。",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "台北天氣?AAPL 股價?"}]
})
print(result["messages"][-1].content)
```

`create_react_agent` 內部已是一張 StateGraph,有 `agent` 與 `tools` 兩個 node、一條 conditional edge 判斷該不該停。

---

## 4. Supervisor 拓樸:中心調度官 + 多個 Worker

當任務需要不同領域的 agent,純線性 ReAct 開始崩潰。Supervisor pattern 的精神是:**一個 LLM 專門做路由決策,worker 各自專精**。

```
                    ┌─────────────────┐
                    │   Supervisor    │ ← 決定誰下一個 / 結束
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌─────────┐    ┌─────────┐    ┌──────────┐
         │Researcher│    │  Coder  │    │ Reviewer │
         └─────────┘    └─────────┘    └──────────┘
```

`langgraph-supervisor` 套件把這個拓樸打包成一行 `create_supervisor(...)`,底層用 `create_handoff_tool` 包裝每個 worker 成 supervisor 看得到的「工具」。supervisor 呼叫 `transfer_to_researcher` 就等於把控制權連同 message history 轉交。

---

## 5. 完整可執行範例:3-agent Research + Code + Review 系統

```python
"""
3-Agent Supervisor System:
- Researcher: 用 Tavily 搜資料
- Coder: 用 Python REPL 跑數值/畫圖
- Reviewer: 批評 / 給修改建議
- Supervisor: 決定誰先做、誰下一個、何時結束
"""
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from langgraph.types import Command, interrupt

# ---------- 0. 模型與工具 ----------
llm = ChatAnthropic(model="claude-opus-4-7", temperature=0)

tavily = TavilySearchResults(max_results=5)
python_repl = PythonREPLTool()

@tool
def dangerous_shell(cmd: str) -> str:
    """執行 shell。高風險,需人類核可。"""
    decision = interrupt({
        "action": "shell_exec",
        "cmd": cmd,
        "prompt": "approve? yes / no / edit"
    })
    if decision.get("approved") is not True:
        return f"USER REJECTED: {decision.get('reason', '')}"
    import subprocess
    return subprocess.check_output(cmd, shell=True, text=True)

# ---------- 1. Worker Agents ----------
researcher = create_react_agent(
    model=llm,
    tools=[tavily],
    name="researcher",
    prompt=(
        "你是研究員。只負責搜尋與整理事實,"
        "絕對不要寫程式、不要評論,完成後回 supervisor。"
    ),
)

coder = create_react_agent(
    model=llm,
    tools=[python_repl, dangerous_shell],
    name="coder",
    prompt=(
        "你是工程師。基於 researcher 提供的資料,"
        "寫並執行 Python 程式碼產生結果。完成回 supervisor。"
    ),
)

reviewer = create_react_agent(
    model=llm,
    tools=[],
    name="reviewer",
    prompt=(
        "你是嚴格 reviewer。檢查 researcher 的事實是否正確、"
        "coder 的程式邏輯與結果是否合理。若有問題明確指出,"
        "並建議交回 researcher 或 coder 修正,否則回 supervisor 結案。"
    ),
)

# ---------- 2. Supervisor ----------
supervisor_prompt = """你是專案經理,你管理三位專家:
- researcher: 負責蒐集事實
- coder: 負責寫程式 / 跑分析
- reviewer: 負責檢查品質
依任務需求決定誰先做、誰下一個。當 reviewer 確認沒問題時,輸出 FINISH。
"""

app = create_supervisor(
    agents=[researcher, coder, reviewer],
    model=llm,
    prompt=supervisor_prompt,
    output_mode="last_message",     # 只把 worker 的最後一則訊息傳回主 thread
    handoff_tool_prefix="delegate_to",
)

# ---------- 3. 持久化 + 編譯 ----------
checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
graph = app.compile(checkpointer=checkpointer)

# ---------- 4. 執行 ----------
config = {"configurable": {"thread_id": "demo-001"}}

events = graph.stream(
    {"messages": [{
        "role": "user",
        "content": "找出 NVIDIA 過去 4 季 EPS,畫成柱狀圖,並評估趨勢。",
    }]},
    config=config,
    stream_mode="values",
)
for ev in events:
    msg = ev["messages"][-1]
    print(f"[{getattr(msg, 'name', 'user')}] {msg.content[:200]}")

# ---------- 5. 處理 interrupt(若 coder 嘗試呼叫 dangerous_shell) ----------
state = graph.get_state(config)
if state.next and "__interrupt__" in state.values:
    pending = state.values["__interrupt__"][0].value
    print("HITL pending:", pending)
    # 假設人類點了同意
    graph.invoke(Command(resume={"approved": True}), config=config)
```

執行流程:supervisor 先 `delegate_to_researcher` -> researcher 用 Tavily 搜 EPS -> 回 supervisor -> `delegate_to_coder` -> coder 跑 matplotlib -> 回 supervisor -> `delegate_to_reviewer` -> reviewer 通過 -> supervisor 輸出 FINISH。整個過程每一步都有 checkpoint。

---

## 6. Handoff 機制細節:它不是魔法,只是被包裝的 tool call

`create_handoff_tool(agent_name="coder")` 會生成一個普通的 LangChain tool。當 supervisor LLM 呼叫它,LangGraph 做三件事:

1. **包裝成 Command**:把 `goto=coder_node` 放進回傳值,告訴 graph 下一個跳轉目標。
2. **state 傳遞**:預設把當前所有 messages 帶過去(完整 history)。你可以改成只傳 summary,或抽出特定欄位減少 token。
3. **tool message 回填**:在 supervisor thread 留下「已交付給 coder」的 ToolMessage,讓 LLM 知道交接成功。

進階用法:
- `create_forward_message_tool`:supervisor 認為 worker 答案已經足夠,直接 forward 最後一則 message,省掉重新 paraphrase 的 token 浪費與失真。
- 自訂 handoff:你可以傳遞「壓縮過的 task brief」而非完整 history,避免每個 worker 都吃滿 context。

---

## 7. 記憶體層:從 SQLite 到 Mem0 / Zep

LangGraph 把記憶體拆成兩層:

- **Short-term (thread-scoped)**:checkpointer 處理。產線常見組合:
  - `MemorySaver()` 開發 demo
  - `SqliteSaver` 單機部署
  - `PostgresSaver` + `psycopg` async 適合多副本
- **Long-term (cross-thread)**:跨對話的使用者偏好、知識。LangGraph 提供 `BaseStore` 介面,可串:
  - **Mem0**:自帶 LLM-驅動的記憶萃取與合併,寫入 `add(messages, user_id)`,查時自動取 top-k 相關。
  - **Zep**:有 entity graph,適合需要事實一致性的長期記憶。
  - 自家 vector store + embedding。

實務上 short-term checkpointer 一定要接 PG / SQLite,long-term 看是否需要跨 session 個人化決定要不要加 Mem0/Zep。

---

## 8. Human-in-the-Loop:`interrupt()` 才是正解

過去 HITL 要靠輪詢資料庫旗標,現在一行 `interrupt(payload)` 就能在任意 node 內暫停。

```python
@tool
def send_email(to: str, body: str):
    decision = interrupt({"to": to, "preview": body[:300]})
    if not decision.get("approved"):
        return "cancelled"
    smtp.send(to, body)
    return "ok"
```

對應的 client 端:

```python
state = graph.get_state(config)
if state.values.get("__interrupt__"):
    payload = state.values["__interrupt__"][0].value
    user_input = ask_human_ui(payload)          # 你自己的 UI
    graph.invoke(Command(resume=user_input), config=config)
```

重點:**interrupt 的暫停是持久化的**。你可以關掉伺服器,三天後讀回 `thread_id`,從中斷點繼續。這比 OpenAI Assistants API 的 in-memory pause 強得多。

---

## 9. LangSmith 追蹤:時間軸 + Replay + Eval

開啟方式只要環境變數:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=ls__...
export LANGSMITH_PROJECT=supervisor-demo
```

LangSmith 對 LangGraph 是一級公民:
- **Tree view**:每個 node、tool call、LLM call 階層展開,可看到 prompt / completion 全文。
- **Replay**:點任一 run -> "Run from here",修改 prompt 或 model 後重跑,做 A/B。
- **Eval datasets**:把線上樣本一鍵存成 dataset,接 evaluator(LLM-as-judge / 規則)做迴歸測試。

production 偵錯 90% 的問題都靠這個面板解決,別自己用 print。

---

## 10. Production 部署:LangGraph Platform / LangServe / FastAPI

選擇順序:

1. **LangGraph Platform / Cloud**:LangChain 官方託管,內建 task queue、checkpoint、cron、HITL UI、可重放控制台。不想自管最快路徑。
2. **LangServe + FastAPI**:`add_routes(app, graph, path="/agent")` 自動產生 REST + streaming endpoint。記得:
   - checkpointer 改 `AsyncPostgresSaver`
   - LLM client 用 async
   - 用 WebSocket / SSE 推 streaming token 給前端
3. **純 FastAPI 自管**:給有強烈定制需求的團隊。注意:
   - 多 worker 進程需要共享 checkpoint store,本機檔案 SQLite 會壞
   - 部署多副本時,interrupt resume 必須路由到能讀到該 thread 的副本(或全 stateless 透過 PG)
   - Tracing 用 OpenTelemetry exporter 接內部 APM

production 最後一哩通常死在「沒做 checkpoint persistence」+「沒做 idempotency」這兩件事,務必先做。

---

## 11. 三個真實案例

### Case A. 企業研究助手(KPMG / 顧問業常見)
- Researcher 跑公開資料(Tavily / SEC EDGAR / 內部 SharePoint)
- Analyst worker 跑 SQL on DWH
- Writer worker 套公司格式產 docx
- Supervisor 判斷需要再補資料還是已可下筆
- HITL 在「對外發送」前 interrupt 給人類審稿

### Case B. 客服 escalation(SaaS / 電信)
- Triage agent 判斷意圖
- KB agent 從 RAG 取答案
- Refund agent 接 billing API,**金額 > $50 強制 interrupt 給 supervisor 人類審批**
- Supervisor agent 全程持有 conversation state,checkpoint 讓 session 中斷後重來不丟訊息

### Case C. AI Code Review pipeline
- Diff parser node 切 hunks
- Static analyzer worker(eslint / pylint / clippy)
- LLM reviewer worker 給 high-level 建議
- Security worker 跑 secret scan + CVE check
- Supervisor 整合所有 finding -> 發 GitHub review comment
- 高風險(如改 IAM)走 interrupt 給人類 sign-off

---

## 12. 框架對比簡表

| 維度 | **LangGraph** | CrewAI | AutoGen (AG2) | Claude Agent SDK |
|------|--------------|--------|---------------|-------------------|
| 核心抽象 | StateGraph + node/edge | Crew / Role / Task | GroupChat + 對話 | Tool chain + sub-agents |
| 學習曲線 | 中(要懂 state schema) | 低(role-based DSL) | 中(對話模式較多) | 低(opinionated API) |
| State 持久化 | 內建 checkpointer + time travel | 序列傳遞 task output | in-memory(需自接) | 透過 MCP server |
| HITL | 原生 `interrupt()` | 需自行串 | 需自行串 | Memory beta + 工具確認 |
| Observability | LangSmith 一級整合 | 企業版才有 | 需自接 | Anthropic Console |
| Model 中立 | 是 | 是 | 是 | 否(僅 Claude) |
| MCP 支援 | 透過 adapter | 透過 adapter | 透過 adapter | 原生一級 |
| Production 成熟度 | 高(2026 v0.4+) | 中 | 中(研究取向) | 高(企業擴張快) |
| 適合場景 | 可控、多步、需審計 | 快速原型、簡單流程 | 多 agent 辯論 / 研究 | Claude 為主、需 MCP |

選型口訣:**要可控 + 要審計 + 要長 session → LangGraph;要快 demo → CrewAI;只用 Claude 又重 MCP → Claude Agent SDK**。

---

## 延伸閱讀
- [Agent 框架選擇決策指南](./Agent框架選擇決策指南.md)
- [Agent 記憶系統完整指南](./Agent記憶系統完整指南.md)
- [`../11.MCP協議與工具調用/MCP_server_完整開發.md`](../11.MCP協議與工具調用/MCP_server_完整開發.md)(即將生成)
- 官方:[LangGraph Supervisor](https://reference.langchain.com/python/langgraph-supervisor)、[Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)、[Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
