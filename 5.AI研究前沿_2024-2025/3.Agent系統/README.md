# Agent系統 - 10篇關鍵論文

> 2024-2025年AI Agent的自主決策與工作流編排突破，包含LangGraph、CrewAI、AutoGen等關鍵技術及完整程式碼實現

> **⚠️ 教學示範 — `examples/` 內 demo 含 AST 沙箱 `safe_eval`,生產勿直接複用**
>
> `examples/02_crewai_multi_agent.py:344-355` 用 `ast` 模組白名單實作 `safe_eval` 給 CrewAI agent 當算式工具。雖比 `eval()` 安全,但 production 請改用 `simpleeval`/`numexpr` 套件或專門數學服務,並對任何 agent tool 輸入加 schema validate + timeout。

---

## 📋 論文與項目列表

| # | 項目/論文 | 機構 | 發布時間 | 程式碼 | 影響力 |
|---|-----------|------|----------|------|--------|
| 1 | LangGraph | LangChain | 2024 | [GitHub](https://github.com/langchain-ai/langgraph) | ⭐⭐⭐⭐⭐ |
| 2 | CrewAI | CrewAI Inc | 2024 | [GitHub](https://github.com/joaomdmoura/crewAI) | ⭐⭐⭐⭐⭐ |
| 3 | AutoGen v0.3 | Microsoft | 2024 | [GitHub](https://github.com/microsoft/autogen) | ⭐⭐⭐⭐⭐ |
| 4 | ReAct | Google Research | 2024 | [GitHub](https://github.com/ysymyth/ReAct) | ⭐⭐⭐⭐⭐ |
| 5 | Model Context Protocol | Anthropic | 2024.11 | [GitHub](https://github.com/anthropics) | ⭐⭐⭐⭐ |
| 6 | Reflexion | Northeastern U | 2024 | [GitHub](https://github.com/noahshinn024/reflexion) | ⭐⭐⭐⭐ |
| 7 | MetaGPT | DeepWisdom | 2024 | [GitHub](https://github.com/geekan/MetaGPT) | ⭐⭐⭐⭐ |
| 8 | AutoGPT | Significant Gravitas | 2024 | [GitHub](https://github.com/Significant-Gravitas/AutoGPT) | ⭐⭐⭐⭐ |
| 9 | AgentBench | Tsinghua U | 2024 | [GitHub](https://github.com/THUDM/AgentBench) | ⭐⭐⭐ |
| 10 | ToolFormer | Meta AI | 2024 | [Paper](https://arxiv.org/abs/2302.04761) | ⭐⭐⭐⭐ |

---

## 核心技術與程式碼實現

### 1. LangGraph - 可控Agent工作流

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# 定義狀態
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence, operator.add]

# 建立工具
search = DuckDuckGoSearchRun()

# 建立Agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [search])

# 運行
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for AI news 2024"}]
})
print(result["messages"][-1].content)
```

### 2. CrewAI - 多Agent協作

```python
from crewai import Agent, Task, Crew

# 定義Agents
researcher = Agent(
    role='Research Analyst',
    goal='Find and analyze information',
    backstory='Expert researcher',
    verbose=True
)

writer = Agent(
    role='Writer',
    goal='Create content',
    backstory='Professional writer',
    verbose=True
)

# 定義Tasks
research = Task(
    description='Research AI trends',
    agent=researcher
)

write = Task(
    description='Write article',
    agent=writer
)

# 執行
crew = Crew(agents=[researcher, writer], tasks=[research, write])
result = crew.kickoff()
```

### 3. AutoGen - 對話式Agent

```python
import autogen

config_list = [{"model": "gpt-4", "api_key": "YOUR_KEY"}]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    code_execution_config={"work_dir": "coding"}
)

user_proxy.initiate_chat(assistant, message="寫一個排序演算法")
```

---

**最後更新**: 2025-01-19
