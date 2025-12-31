# AI Agents 與 Agentic Workflows 2024-2025

## 目錄
1. [前言](#1-前言)
2. [AI Agent 核心概念](#2-ai-agent-核心概念)
   - 2.1 [什麼是 AI Agent](#21-什麼是-ai-agent)
   - 2.2 [Agent 的核心組件](#22-agent-的核心組件)
   - 2.3 [Agent vs 傳統 LLM 應用](#23-agent-vs-傳統-llm-應用)
   - 2.4 [Agentic Workflows 的演進](#24-agentic-workflows-的演進)
3. [ReAct 模式與思維鏈](#3-react-模式與思維鏈)
   - 3.1 [ReAct 原理](#31-react-原理)
   - 3.2 [Chain of Thought (CoT)](#32-chain-of-thought-cot)
   - 3.3 [ReAct 實作](#33-react-實作)
4. [主流 Agent 框架深度解析](#4-主流-agent-框架深度解析)
   - 4.1 [LangGraph](#41-langgraph)
   - 4.2 [CrewAI](#42-crewai)
   - 4.3 [AutoGPT](#43-autogpt)
   - 4.4 [AutoGen](#44-autogen)
5. [框架比較與選擇指南](#5-框架比較與選擇指南)
6. [實戰案例：多 Agent 協作系統](#6-實戰案例多-agent-協作系統)
7. [Agent 工具設計與整合](#7-agent-工具設計與整合)
8. [評估與監控](#8-評估與監控)
9. [最佳實踐與設計模式](#9-最佳實踐與設計模式)
10. [未來趨勢與展望](#10-未來趨勢與展望)
11. [延伸閱讀](#11-延伸閱讀)

---

## 1. 前言

### 1.1 為什麼需要 AI Agents？

傳統的 LLM 應用通常是**單次對話**模式：用戶輸入 → 模型生成 → 結束。這種模式有明顯的限制：

- **無法處理複雜任務**：需要多步驟推理的問題難以解決
- **缺乏工具使用能力**：無法調用外部 API、搜尋引擎、資料庫等
- **無記憶與學習**：每次對話獨立，無法累積經驗
- **不能自主行動**：需要人工介入每個步驟

**AI Agent** 的出現解決了這些問題，它具備：

1. **自主性 (Autonomy)**：能夠自主決策並執行行動
2. **工具使用 (Tool Use)**：調用各種外部工具和 API
3. **規劃能力 (Planning)**：將複雜任務分解為子任務
4. **記憶系統 (Memory)**：短期和長期記憶
5. **反思機制 (Reflection)**：從錯誤中學習並改進

### 1.2 2024-2025 年的 Agent 革命

2024 年是 **Agentic AI** 的爆發年：

**關鍵里程碑**：
- **2023.03**：GPT-4 發布，支援 Function Calling
- **2023.08**：AutoGPT 開源，引爆自主 Agent 熱潮
- **2023.11**：OpenAI Assistants API 發布
- **2024.01**：LangGraph 正式版發布
- **2024.03**：Claude 3 加強 Agent 能力
- **2024.05**：GPT-4o 進一步優化多模態 Agent
- **2024.09**：多 Agent 系統成為主流架構
- **2025.01**：企業級 Agent 平台成熟

**市場趨勢**：
- **企業採用率**：超過 60% 的企業正在探索 AI Agents
- **投資熱度**：Agent 基礎設施公司獲得大量融資
- **應用場景**：從客服、數據分析到軟體開發全面應用

### 1.3 本章學習目標

- 理解 AI Agent 的核心概念和設計原則
- 掌握 LangGraph、CrewAI、AutoGPT、AutoGen 四大框架
- 學會設計和實作多 Agent 協作系統
- 了解工具整合、評估監控和最佳實踐
- 把握未來 Agent 技術發展趨勢

---

## 2. AI Agent 核心概念

### 2.1 什麼是 AI Agent

**定義**：AI Agent 是一個能夠**感知環境、做出決策、執行行動**以達成目標的智能系統。

**經典 Agent 架構**：

```
┌─────────────────────────────────────┐
│           AI Agent                   │
│                                      │
│  ┌──────────┐      ┌──────────┐   │
│  │ Perceive │─────>│  Think   │   │
│  │ (感知)    │      │  (思考)   │   │
│  └──────────┘      └──────────┘   │
│       ▲                  │          │
│       │                  ▼          │
│  ┌──────────┐      ┌──────────┐   │
│  │Environment│<─────│   Act    │   │
│  │  (環境)    │      │  (行動)   │   │
│  └──────────┘      └──────────┘   │
└─────────────────────────────────────┘
```

**核心循環**（Agent Loop）：

```python
# 偽代碼
while not task_completed:
    # 1. 感知：獲取當前狀態
    observation = perceive(environment)

    # 2. 思考：使用 LLM 規劃下一步
    thought, action = llm.decide(observation, memory, goal)

    # 3. 行動：執行工具調用或生成回應
    result = execute(action)

    # 4. 更新：存儲到記憶
    memory.update(thought, action, result)

    # 5. 檢查：是否達成目標
    task_completed = check_goal(result)
```

### 2.2 Agent 的核心組件

#### 2.2.1 規劃模組 (Planning)

**單路徑規劃**：
- **思維鏈 (Chain of Thought, CoT)**：逐步推理
- **ReAct**：Reasoning + Acting 交替進行

**多路徑規劃**：
- **思維樹 (Tree of Thoughts, ToT)**：探索多個推理分支
- **圖規劃 (Graph Planning)**：複雜的狀態轉移圖

**實例**：
```python
# 任務：訂購披薩
# 規劃分解：
plan = [
    "查詢附近披薩店",
    "比較價格和評分",
    "選擇最佳店家",
    "確認菜單和配料",
    "提交訂單",
    "追蹤配送狀態"
]
```

#### 2.2.2 記憶系統 (Memory)

**短期記憶 (Working Memory)**：
- 當前對話上下文
- 臨時變數和中間結果
- 實作：In-context learning (上下文窗口)

**長期記憶 (Long-term Memory)**：
- 歷史對話記錄
- 學到的知識和經驗
- 實作：向量資料庫 (Vector DB)、關係資料庫

**記憶架構**：

| 記憶類型 | 存儲方式 | 檢索方式 | 容量 | 持久性 |
|---------|---------|---------|------|--------|
| 短期記憶 | 上下文窗口 | 全量載入 | 有限 (4K-128K tokens) | 會話級別 |
| 工作記憶 | 暫存變數 | 直接訪問 | 小 | 任務級別 |
| 語義記憶 | 向量資料庫 | 相似度搜尋 | 大 | 永久 |
| 情節記憶 | 時序資料庫 | 時間/事件查詢 | 中-大 | 永久 |

**實作範例**：
```python
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma

# 短期記憶
short_term_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 長期記憶（向量檢索）
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings
)
long_term_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)
```

#### 2.2.3 工具使用 (Tool Use)

**工具定義**：Agent 可調用的外部功能，包括：

- **搜尋工具**：Google Search, Wikipedia, Arxiv
- **計算工具**：Python REPL, WolframAlpha, Calculator
- **資料庫工具**：SQL Query, Vector DB Search
- **API 工具**：天氣 API, 地圖 API, 電商 API
- **文件工具**：讀寫檔案, PDF 解析
- **程式碼工具**：代碼執行器, Git 操作

**Function Calling**：

OpenAI Function Calling 格式：
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取指定城市的天氣資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名稱，例如：台北"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "溫度單位"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

**工具執行流程**：
```
User: 台北明天天氣如何？
  ↓
LLM: [決策] 需要調用 get_weather 工具
  ↓
Tool Call: get_weather(location="台北", date="明天")
  ↓
Tool Result: {"temp": 25, "condition": "晴天", ...}
  ↓
LLM: [整合結果] 根據天氣數據生成回應
  ↓
Response: 台北明天晴天，氣溫約 25 度...
```

#### 2.2.4 行動執行 (Action Execution)

**行動類型**：

1. **觀察型行動**：獲取信息但不改變狀態
   - 搜尋、查詢、讀取

2. **操作型行動**：改變環境狀態
   - 寫入、刪除、發送請求

3. **溝通型行動**：與人或其他 Agent 互動
   - 提問、回答、協商

**行動格式化**：
```python
from typing import Literal, Union
from pydantic import BaseModel

class Action(BaseModel):
    """行動定義"""
    type: Literal["tool_use", "response", "delegate"]
    name: str  # 工具名稱或行動名稱
    arguments: dict  # 參數

class Observation(BaseModel):
    """觀察結果"""
    success: bool
    result: Union[str, dict]
    error: str = None
```

### 2.3 Agent vs 傳統 LLM 應用

| 維度 | 傳統 LLM 應用 | AI Agent |
|------|--------------|----------|
| **交互模式** | 單次請求-回應 | 多輪自主循環 |
| **任務複雜度** | 簡單查詢、文本生成 | 複雜多步驟任務 |
| **工具使用** | 無或預定義 | 動態選擇和調用 |
| **規劃能力** | 無 | 任務分解和規劃 |
| **記憶** | 會話級別 | 持久化長期記憶 |
| **自主性** | 低（需要人工引導） | 高（自主決策執行） |
| **錯誤處理** | 直接失敗 | 嘗試修復和重試 |
| **適用場景** | 問答、摘要、翻譯 | 研究、分析、自動化 |

### 2.4 Agentic Workflows 的演進

**第一代：單一 Agent (2022-2023)**
- 簡單的工具調用
- 基於提示工程的決策
- 代表：ChatGPT Plugins

**第二代：ReAct Agent (2023)**
- 思考-行動循環
- 工具鏈調用
- 代表：LangChain Agents

**第三代：規劃型 Agent (2023-2024)**
- 任務分解和規劃
- 自我反思和修正
- 代表：AutoGPT, BabyAGI

**第四代：多 Agent 協作 (2024-2025)**
- Agent 間通信和協作
- 角色專業化
- 分佈式執行
- 代表：CrewAI, AutoGen

**第五代：自我進化 Agent (2025+)**
- 持續學習
- 知識積累
- 能力自我擴展

---

## 3. ReAct 模式與思維鏈

### 3.1 ReAct 原理

**ReAct** = **Reasoning** (推理) + **Acting** (行動)

**核心思想**：將推理軌跡和任務特定行動交織在一起，使 LLM 能夠動態推理、規劃並執行行動。

**論文**：ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)

**基本循環**：

```
Thought 1: 我需要找出台灣最高的山
Action 1: Search[台灣最高的山]
Observation 1: 玉山，海拔3,952公尺

Thought 2: 我找到了答案，玉山是台灣最高的山
Action 2: Finish[玉山，海拔3,952公尺]
```

**與 Chain of Thought 的區別**：

| 特性 | Chain of Thought | ReAct |
|------|-----------------|-------|
| **輸出** | 純推理文本 | 推理 + 行動 |
| **工具使用** | 無 | 有 |
| **環境交互** | 無 | 有 |
| **應用** | 數學推理、邏輯題 | 需要外部資訊的任務 |

### 3.2 Chain of Thought (CoT)

**思維鏈**讓 LLM 展示中間推理步驟，提升複雜推理能力。

**Zero-shot CoT**：
```
提示：Let's think step by step.

問題：一個正方形的周長是 24 公分，它的面積是多少？

回應：
讓我們一步一步思考：
1. 正方形周長 = 4 × 邊長
2. 24 = 4 × 邊長
3. 邊長 = 24 / 4 = 6 公分
4. 面積 = 邊長 × 邊長 = 6 × 6 = 36 平方公分

答案：36 平方公分
```

**Few-shot CoT**：
提供推理範例，引導模型模仿。

**Self-Consistency**：
生成多條推理路徑，採用投票機制選擇最終答案。

### 3.3 ReAct 實作

#### 3.3.1 基礎 ReAct Agent

```python
import openai
from typing import List, Dict, Any
import json

class ReActAgent:
    """基礎 ReAct Agent 實作"""

    def __init__(self, model: str = "gpt-4", max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations
        self.tools = {}

    def register_tool(self, name: str, func: callable, description: str):
        """註冊工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }

    def _create_prompt(self, question: str, scratchpad: str) -> str:
        """創建 ReAct 提示"""
        tool_descriptions = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

        prompt = f"""你是一個能使用工具的 AI 助手。你可以使用以下工具：

{tool_descriptions}

請使用以下格式回答問題：

Question: 輸入的問題
Thought 1: 你的推理過程
Action 1: 工具名稱[參數]
Observation 1: 工具返回的結果
... (重複 Thought/Action/Observation)
Thought N: 我現在知道最終答案了
Final Answer: 最終答案

開始！

Question: {question}
{scratchpad}"""
        return prompt

    def _parse_action(self, text: str) -> tuple:
        """解析行動"""
        # 簡化的解析邏輯
        if "Action" in text:
            action_line = [line for line in text.split("\n") if line.startswith("Action")][0]
            # 提取工具名稱和參數
            # 格式: Action N: ToolName[argument]
            action_part = action_line.split(":", 1)[1].strip()
            if "[" in action_part and "]" in action_part:
                tool_name = action_part.split("[")[0].strip()
                argument = action_part.split("[")[1].split("]")[0].strip()
                return tool_name, argument
        return None, None

    def run(self, question: str) -> str:
        """執行 ReAct 循環"""
        scratchpad = ""

        for i in range(self.max_iterations):
            # 創建提示
            prompt = self._create_prompt(question, scratchpad)

            # 調用 LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            output = response.choices[0].message.content
            scratchpad += output + "\n"

            # 檢查是否完成
            if "Final Answer:" in output:
                final_answer = output.split("Final Answer:")[1].strip()
                return final_answer

            # 解析並執行行動
            tool_name, argument = self._parse_action(output)
            if tool_name and tool_name in self.tools:
                result = self.tools[tool_name]["function"](argument)
                scratchpad += f"Observation {i+1}: {result}\n"
            else:
                scratchpad += f"Observation {i+1}: 工具不存在或解析失敗\n"

        return "達到最大迭代次數，未能完成任務"


# 使用範例
def search_tool(query: str) -> str:
    """模擬搜尋工具"""
    # 實際應該調用真實搜尋 API
    mock_results = {
        "台灣最高的山": "玉山，海拔 3,952 公尺，位於台灣中央山脈",
        "台灣人口": "約 2,300 萬人（2024年）"
    }
    return mock_results.get(query, "未找到相關資訊")

def calculator_tool(expression: str) -> str:
    """計算器工具（安全版本）"""
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }

    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](safe_eval(node.operand))
        raise ValueError("不支援的運算")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"計算錯誤: {str(e)}"


# 創建 Agent
agent = ReActAgent()
agent.register_tool("Search", search_tool, "搜尋網路資訊")
agent.register_tool("Calculator", calculator_tool, "執行數學計算")

# 執行查詢
question = "台灣最高的山是什麼？它的海拔是多少？"
answer = agent.run(question)
print(f"問題：{question}")
print(f"答案：{answer}")
```

#### 3.3.2 使用 LangChain 的 ReAct Agent

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import OpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_experimental.utilities import PythonREPL

# 初始化工具
search = GoogleSearchAPIWrapper()
python_repl = PythonREPL()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="搜尋網路資訊。當你需要查找最新資訊或不知道的事實時使用。"
    ),
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="Python shell。用於執行 Python 代碼。輸入應該是有效的 Python 命令。"
    )
]

# 初始化 LLM
llm = OpenAI(temperature=0, model="gpt-4")

# 創建 ReAct Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# 執行查詢
response = agent.run(
    "2023 年諾貝爾物理學獎得主是誰？他們的主要貢獻是什麼？"
)
print(response)
```

**執行流程輸出**：
```
> Entering new AgentExecutor chain...

Thought: 我需要搜尋 2023 年諾貝爾物理學獎的資訊
Action: Search
Action Input: "2023 Nobel Prize in Physics"

Observation: The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz and Anne L'Huillier for experimental methods that generate attosecond pulses of light for the study of electron dynamics in matter...

Thought: 我現在知道答案了
Final Answer: 2023 年諾貝爾物理學獎由 Pierre Agostini、Ferenc Krausz 與 Anne L'Huillier 共同獲得，肯定他們在產生阿秒雷射脈衝以研究物質中電子動態的實驗方法。

> Finished chain.
```

---

## 4. 主流 Agent 框架深度解析

### 4.1 LangGraph

**定位**：LangChain 團隊推出的**狀態機編排框架**，用於構建複雜的 Agent 工作流。

#### 4.1.1 核心概念

**圖結構 (Graph)**：
- **節點 (Node)**：執行單元（函數）
- **邊 (Edge)**：狀態轉移
- **條件邊 (Conditional Edge)**：根據狀態決定下一個節點

**狀態管理 (State)**：
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    """Agent 狀態定義"""
    messages: list  # 對話歷史
    next_action: str  # 下一步行動
    intermediate_steps: list  # 中間步驟
    final_answer: str  # 最終答案
```

**檢查點 (Checkpointing)**：
- 支援狀態持久化
- 實現時間旅行（回溯到之前狀態）
- 支援暫停和恢復

#### 4.1.2 架構圖

```
START
  │
  ▼
┌─────────┐
│  LLM    │ ─────> 決定下一步
│  Node   │
└─────────┘
  │
  ├───> [條件判斷]
  │
  ├──> Tools Node (執行工具)
  │         │
  │         ▼
  │    ┌─────────┐
  │    │ Tool 1  │
  │    │ Tool 2  │
  │    │ Tool 3  │
  │    └─────────┘
  │         │
  ▼         ▼
Should Continue?
  │
  ├─ YES ─> 返回 LLM Node
  │
  └─ NO ──> END
```

#### 4.1.3 完整實作範例

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Annotated, Sequence
import operator

# 定義工具
@tool
def get_weather(location: str) -> str:
    """獲取天氣資訊"""
    # 模擬 API 調用
    weather_data = {
        "台北": "晴天，25°C",
        "高雄": "多雲，28°C",
        "台中": "陰天，23°C"
    }
    return weather_data.get(location, "未知地區")

@tool
def search_web(query: str) -> str:
    """搜尋網路資訊"""
    # 模擬搜尋
    return f"關於 '{query}' 的搜尋結果..."

tools = [get_weather, search_web]
tool_executor = ToolExecutor(tools)

# 定義狀態
class AgentState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    intermediate_steps: Annotated[list, operator.add]

# 定義節點函數
def call_model(state: AgentState):
    """調用 LLM 決策"""
    messages = state["messages"]

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

def call_tools(state: AgentState):
    """執行工具調用"""
    messages = state["messages"]
    last_message = messages[-1]

    # 執行工具
    tool_calls = last_message.tool_calls
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_args
        )

        result = tool_executor.invoke(action)
        results.append({
            "tool_call_id": tool_call["id"],
            "output": result
        })

    return {"messages": results, "intermediate_steps": [(tool_call, result)]}

def should_continue(state: AgentState):
    """決定是否繼續循環"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果沒有工具調用，則結束
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# 構建圖
workflow = StateGraph(AgentState)

# 添加節點
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

# 設置入口
workflow.set_entry_point("agent")

# 添加條件邊
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# 從 tools 返回 agent
workflow.add_edge("tools", "agent")

# 編譯
app = workflow.compile()

# 使用
from langchain_core.messages import HumanMessage

inputs = {
    "messages": [HumanMessage(content="台北明天天氣如何？")],
    "intermediate_steps": []
}

result = app.invoke(inputs)
print(result["messages"][-1].content)
```

#### 4.1.4 進階功能：人機協作

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 創建帶檢查點的 Agent
memory = MemorySaver()
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=tools,
    checkpointer=memory
)

# 配置線程
config = {"configurable": {"thread_id": "user-123"}}

# 第一次交互
result = agent.invoke(
    {"messages": [("user", "幫我查台北天氣")]},
    config=config
)

# 可以中斷並在稍後繼續
# 第二次交互（使用相同 thread_id）
result = agent.invoke(
    {"messages": [("user", "那高雄呢？")]},
    config=config
)
```

#### 4.1.5 LangGraph 優缺點

**優點**：
- ✅ 完全控制：精確控制 Agent 流程
- ✅ 狀態管理：強大的狀態機制
- ✅ 可視化：支援圖可視化
- ✅ 檢查點：支援暫停/恢復
- ✅ 靈活性：適合複雜工作流

**缺點**：
- ❌ 學習曲線：概念較多，需要時間學習
- ❌ 代碼量：相比高階框架代碼較多
- ❌ 調試複雜：圖結構調試較困難

**適用場景**：
- 需要精確控制流程的應用
- 複雜的多步驟工作流
- 需要人機協作的系統
- 需要狀態持久化的應用

#### 4.1.6 2024-2025 版本更新重點

- **Durable Execution**：內建檢查點（Checkpointing）與恢復機制，可在長時間任務或失敗後從中斷點繼續執行。
- **Human-in-the-loop**：任意節點都能插入人工審核與改寫狀態，適合金融、法務等需要合規把關的流程。
- **Comprehensive Memory**：同時支援短期、長期記憶，官方文件提供工作記憶與持久存儲的設計參考。
- **LangSmith / LangGraph Studio**：透過官方平台可視化節點執行、檢查 state diff，並提供雲端部署選項。
- **生態整合**：維持與 LangChain、LangGraphJS 同步更新，可直接使用 LangChain 模組或轉換成前端工作流。

---

### 4.2 CrewAI

**定位**：專注於**多 Agent 協作**的高階框架，模擬真實團隊工作模式。

#### 4.2.1 核心概念

**Agent（成員）**：
- **Role（角色）**：Agent 的專業領域
- **Goal（目標）**：Agent 的工作目標
- **Backstory（背景）**：Agent 的經驗和特質

**Task（任務）**：
- **Description（描述）**：任務說明
- **Expected Output（預期輸出）**：期望結果
- **Agent（執行者）**：負責的 Agent

**Crew（團隊）**：
- **Agents**：團隊成員列表
- **Tasks**：任務列表
- **Process**：執行流程（順序/階層）

#### 4.2.2 工作流程

```
Crew (團隊)
  │
  ├─> Agent 1: Researcher (研究員)
  │     └─> Task: 收集資料
  │
  ├─> Agent 2: Writer (作家)
  │     └─> Task: 撰寫文章
  │
  └─> Agent 3: Editor (編輯)
        └─> Task: 審核修改
```

#### 4.2.3 完整實作範例

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 定義 Agent 1: 研究員
researcher = Agent(
    role='科技研究員',
    goal='深入研究 AI Agents 的最新發展',
    backstory="""你是一位經驗豐富的 AI 研究員，專精於追蹤和分析
    人工智慧領域的最新突破。你善於從學術論文、技術博客和行業報告中
    提取關鍵資訊。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool]  # 可以添加搜尋工具
)

# 定義 Agent 2: 技術作家
writer = Agent(
    role='技術作家',
    goal='將複雜的技術概念轉化為易懂的文章',
    backstory="""你是一位才華洋溢的技術作家，擅長將艱深的技術內容
    轉化為引人入勝的文章。你的文章既專業又易讀，深受讀者喜愛。""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 定義 Agent 3: 編輯
editor = Agent(
    role='內容編輯',
    goal='確保文章質量和準確性',
    backstory="""你是一位嚴謹的編輯，對細節要求極高。你負責審查文章
    的邏輯性、準確性和可讀性，確保最終輸出達到出版標準。""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# 定義任務 1: 研究
research_task = Task(
    description="""研究 2024-2025 年 AI Agents 的最新發展，包括：
    1. 主流框架（LangGraph, CrewAI, AutoGPT, AutoGen）
    2. 關鍵技術突破
    3. 實際應用案例
    4. 未來趨勢

    提供詳細的研究報告，包含具體數據和來源。""",
    expected_output="一份結構化的研究報告，包含至少 5 個關鍵發現",
    agent=researcher
)

# 定義任務 2: 撰寫
writing_task = Task(
    description="""基於研究報告，撰寫一篇關於 AI Agents 的技術文章。
    文章應該：
    1. 有吸引人的標題和引言
    2. 清晰的結構（引言、主體、結論）
    3. 包含具體例子和代碼示例
    4. 適合技術背景的讀者

    字數約 2000-3000 字。""",
    expected_output="一篇完整的技術文章，包含標題、章節和代碼範例",
    agent=writer,
    context=[research_task]  # 依賴研究任務的輸出
)

# 定義任務 3: 編輯
editing_task = Task(
    description="""審查並優化技術文章，確保：
    1. 技術準確性
    2. 邏輯連貫性
    3. 語言流暢性
    4. 格式一致性

    提供修改建議並生成最終版本。""",
    expected_output="經過審核和優化的最終文章",
    agent=editor,
    context=[writing_task]
)

# 創建團隊
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # 順序執行
    verbose=2
)

# 執行
result = crew.kickoff()

print("=" * 50)
print("最終輸出：")
print("=" * 50)
print(result)
```

#### 4.2.4 進階功能：階層式流程

```python
from crewai import Crew, Process

# 階層式流程：有一個 Manager Agent 分配任務
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.hierarchical,  # 階層式
    manager_llm=ChatOpenAI(model="gpt-4", temperature=0)
)

result = crew.kickoff()
```

**階層式 vs 順序式**：

| 特性 | 順序式 (Sequential) | 階層式 (Hierarchical) |
|------|-------------------|---------------------|
| **執行順序** | 固定順序 | Manager 動態分配 |
| **適用場景** | 明確的流程 | 複雜的協作任務 |
| **靈活性** | 低 | 高 |
| **開銷** | 低 | 高（需要 Manager） |

#### 4.2.5 自定義工具

```python
from crewai_tools import BaseTool

class CustomSearchTool(BaseTool):
    name: str = "Custom Search"
    description: str = "搜尋特定領域的技術資訊"

    def _run(self, query: str) -> str:
        # 實作搜尋邏輯
        # 這裡可以調用 Google API, Arxiv API 等
        return f"搜尋結果：{query}"

# 使用自定義工具
search_tool = CustomSearchTool()

researcher = Agent(
    role='研究員',
    goal='研究最新技術',
    backstory='...',
    tools=[search_tool],
    llm=llm
)
```

#### 4.2.6 CrewAI 優缺點

**優點**：
- ✅ 簡單易用：高階 API，快速上手
- ✅ 多 Agent 協作：天生支援團隊協作
- ✅ 角色明確：清晰的角色和責任劃分
- ✅ 生產就緒：內建錯誤處理和重試機制

**缺點**：
- ❌ 控制有限：無法精細控制流程
- ❌ 成本較高：多 Agent 意味著更多 API 調用
- ❌ 調試困難：多 Agent 交互調試複雜

**適用場景**：
- 需要模擬團隊協作的場景
- 內容創作（研究 → 撰寫 → 編輯）
- 數據分析（收集 → 分析 → 報告）
- 軟體開發（設計 → 編碼 → 測試）

---

### 4.3 AutoGPT

**定位**：首個廣為人知的**自主 AI Agent**，目標是完成用戶指定的目標。

#### 4.3.1 核心理念

**自主性**：
- 自動分解任務
- 自主決策下一步行動
- 持續執行直到達成目標

**長期記憶**：
- 使用向量資料庫存儲長期記憶
- 檢索相關歷史經驗

**自我反思**：
- 評估當前進展
- 修正錯誤方向

#### 4.3.2 工作流程

```
Goal: 研究並撰寫關於量子計算的報告

AutoGPT 自主執行：
  │
  ├─> Task 1: 搜尋量子計算基礎知識
  │     └─> Action: Google Search
  │
  ├─> Task 2: 閱讀並總結找到的文章
  │     └─> Action: Browse Website
  │
  ├─> Task 3: 搜尋最新研究論文
  │     └─> Action: Arxiv Search
  │
  ├─> Task 4: 整理資料並撰寫大綱
  │     └─> Action: Write File
  │
  ├─> Task 5: 撰寫報告內容
  │     └─> Action: Append File
  │
  └─> Task 6: 審查並完善報告
        └─> Action: Read + Edit File

Goal Achieved! ✓
```

#### 4.3.3 AutoGPT 架構

```python
# AutoGPT 的簡化版實作概念

class AutoGPT:
    def __init__(self, goal: str):
        self.goal = goal
        self.tasks = []
        self.memory = VectorMemory()

    def run(self):
        """主執行循環"""
        while not self.is_goal_achieved():
            # 1. 反思當前狀態
            current_state = self.reflect()

            # 2. 規劃下一步
            next_tasks = self.plan(current_state)

            # 3. 執行任務
            for task in next_tasks:
                result = self.execute_task(task)
                self.memory.store(task, result)

            # 4. 自我評估
            if self.should_adjust_plan():
                self.replan()

        return self.compile_results()

    def plan(self, state: dict) -> list:
        """使用 LLM 規劃任務"""
        prompt = f"""
        Goal: {self.goal}
        Current State: {state}
        Previous Tasks: {self.tasks}

        What are the next 3 tasks to accomplish the goal?
        """
        tasks = llm.generate(prompt)
        return parse_tasks(tasks)

    def execute_task(self, task: dict):
        """執行單個任務"""
        if task['type'] == 'search':
            return self.tools.search(task['query'])
        elif task['type'] == 'code':
            return self.tools.execute_code(task['code'])
        # ... 其他工具

    def reflect(self) -> dict:
        """反思當前進展"""
        prompt = f"""
        Goal: {self.goal}
        Completed Tasks: {self.tasks}

        Evaluate the progress and current state.
        """
        return llm.generate(prompt)

    def is_goal_achieved(self) -> bool:
        """檢查目標是否達成"""
        prompt = f"""
        Goal: {self.goal}
        Current Results: {self.compile_results()}

        Has the goal been achieved? Answer Yes or No.
        """
        response = llm.generate(prompt)
        return "yes" in response.lower()
```

#### 4.3.4 AutoGPT 的挑戰

**成本問題**：
- 無限循環可能導致巨額 API 費用
- 需要設置最大迭代次數和預算限制

**可靠性問題**：
- 容易陷入循環
- 可能偏離原始目標
- 決策品質不穩定

**實用性問題**：
- 對於簡單任務過於複雜
- 對於複雜任務又不夠可靠
- 需要大量監督

#### 4.3.5 AutoGPT 的演進

**AutoGPT v0.5+ (2024)**：
- 改進的規劃算法
- 更好的錯誤處理
- 支援插件系統
- Web UI 介面

**衍生項目**：
- **BabyAGI**：簡化版自主 Agent
- **AgentGPT**：Web 端部署版本
- **SuperAGI**：企業級自主 Agent 平台

---

### 4.4 AutoGen

**定位**：Microsoft 開發的**多 Agent 對話框架**，專注於 Agent 間的協作對話。

#### 4.4.1 核心概念

**對話式 AI**：
- Agent 通過對話協作
- 支援人-機對話和機-機對話
- 靈活的對話模式

**Agent 類型**：

1. **AssistantAgent**：使用 LLM 的助手
2. **UserProxyAgent**：代表用戶，可執行代碼
3. **GroupChatManager**：管理多 Agent 對話
4. **Custom Agent**：自定義 Agent

#### 4.4.2 基礎架構

```
┌─────────────┐         ┌─────────────┐
│  Assistant  │<──────>│ UserProxy   │
│   Agent     │  Chat   │   Agent     │
└─────────────┘         └─────────────┘
       │                       │
       │                       ├─> Execute Code
       │                       └─> Request Human Input
       │
       └─> Call LLM
```

#### 4.4.3 基礎實作範例

```python
import autogen

# 配置 LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120
}

# 創建 Assistant Agent
assistant = autogen.AssistantAgent(
    name="助理",
    system_message="""你是一個有幫助的 AI 助理。
    你可以幫助用戶解決問題、回答問題、執行任務。""",
    llm_config=llm_config
)

# 創建 User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="用戶代理",
    human_input_mode="NEVER",  # 不需要人工輸入
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False  # 設為 True 以使用 Docker
    }
)

# 啟動對話
user_proxy.initiate_chat(
    assistant,
    message="請幫我用 Python 計算斐波那契數列的前 10 項"
)
```

**執行流程**：
```
UserProxy: 請幫我用 Python 計算斐波那契數列的前 10 項
Assistant: 產生 Python 程式碼並呼叫執行
UserProxy: 執行程式並回報結果
```

---

## 7. Agent 工具設計與整合

### 7.1 Model Context Protocol（MCP）

2024 年底開始，OpenAI、Anthropic、Claude Desktop 等主流平台紛紛導入 **Model Context Protocol**，讓 Agent 能以標準化方式連接內部 API、資料庫或檔案系統。MCP 將「工具伺服器」與「模型客戶端」分離，具備以下優點：

- **標準訊息格式**：所有呼叫都透過 JSON-RPC + JSON Schema 定義的 `tools`、`resources`、`prompts` 進行交換。
- **跨客戶端互通**：同一個 MCP 伺服器可以同時被 Claude Desktop、OpenAI Assistants API、LangGraph 等客戶端使用。
- **安全控管**：伺服器端可限制指令、檔案路徑與參數，並提供審計紀錄。

以下示範如何使用官方 Python SDK（`pip install "mcp[cli]"`）建立一個可以被 Claude Desktop 或 LangGraph 使用的工具伺服器：

```python
"""server.py — 最小 MCP 伺服器示例"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""

    return a + b


@mcp.resource("weather://{city}")
def get_weather(city: str) -> str:
    """Return a static weather string for demo purposes."""

    data = {"taipei": "Cloudy 26°C", "taichung": "Sunny 24°C"}
    return data.get(city.lower(), "No data")


@mcp.prompt()
def greet_user(name: str, tone: str = "friendly") -> str:
    """Return a reusable prompt template."""

    styles = {
        "friendly": "Please write a warm greeting",
        "formal": "Please compose a formal greeting",
    }
    return f"{styles.get(tone, styles['friendly'])} for {name}."


if __name__ == "__main__":
    mcp.run()
```

開發時可以使用：

```bash
uv run mcp dev server.py        # 啟動 MCP Inspector，測試工具/資源
uv run mcp install server.py    # 安裝到 Claude Desktop 或其他客戶端
```

**整合建議**：

1. **權限設計**：限制工具可讀寫的路徑與 API 金鑰，必要時加入審計日誌。
2. **錯誤回傳**：確保所有工具都回傳結構化錯誤（HTTP 狀態碼、錯誤訊息、重試建議）。
3. **與 LangGraph / CrewAI 結合**：可以將 MCP 工具包裝成 LangChain `Tool`，並在 Agent workflow 中統一管理。
4. **版本管理**：使用 `pyproject.toml` 或 `uv` 管理 MCP 伺服器依賴，並以 CI 驗證工具行為。
5. **安全測試**：將 MCP 伺服器納入紅隊測試與 Prompt Injection 防護範圍，避免被惡意利用。

---

## 5. 框架比較與選擇指南

### 5.1 框架全面比較

| 維度 | LangGraph | CrewAI | AutoGPT | AutoGen |
|------|-----------|--------|---------|---------|
| **學習曲線** | 中-高 | 低-中 | 中 | 中 |
| **控制精度** | 極高 | 中 | 低 | 高 |
| **多 Agent** | 支援（需手動編排） | 原生支援 | 不支援 | 原生支援 |
| **狀態管理** | 極佳（StateGraph） | 基礎 | 中等 | 中等 |
| **工具整合** | 靈活 | 簡單 | 中等 | 靈活 |
| **人機協作** | 極佳（Checkpointing） | 一般 | 一般 | 極佳 |
| **可視化** | 支援（LangSmith） | 基礎 | 無 | 基礎 |
| **生產就緒** | 高 | 高 | 中 | 高 |
| **社群活躍度** | 極高 | 高 | 中 | 高 |
| **文檔質量** | 優秀 | 良好 | 一般 | 優秀 |
| **成本控制** | 精確 | 中等 | 低（容易失控） | 良好 |
| **適合規模** | 小-大型 | 小-中型 | 小型 | 小-中型 |

### 5.2 選擇決策樹

```
需要多 Agent 協作？
├─ 是
│  ├─ 需要精確控制流程？
│  │  ├─ 是 → LangGraph
│  │  └─ 否 → CrewAI 或 AutoGen
│  └─ 需要人機協作對話？
│     └─ 是 → AutoGen
│
└─ 否（單 Agent）
   ├─ 需要複雜狀態管理？
   │  └─ 是 → LangGraph
   └─ 需要完全自主執行？
      └─ 是 → AutoGPT（慎用）
```

### 5.3 典型應用場景匹配

#### 5.3.1 內容創作流程
**推薦**：CrewAI

**理由**：
- 天然的角色分工（研究員、作家、編輯）
- 順序式流程符合創作邏輯
- 簡單易用，快速上手

**範例**：
```python
# 研究員 → 作家 → 編輯 → 最終文章
crew = Crew(
    agents=[researcher, writer, editor],
    process=Process.sequential
)
```

#### 5.3.2 複雜業務流程自動化
**推薦**：LangGraph

**理由**：
- 需要精確控制每個步驟
- 複雜的條件分支
- 需要人機協作審批

**範例**：
```python
# 發票處理流程：OCR → 驗證 → 人工審核 → 入帳
workflow = StateGraph(InvoiceState)
workflow.add_conditional_edges("validate", should_human_review)
```

#### 5.3.3 研究助手
**推薦**：AutoGen

**理由**：
- 需要執行代碼分析數據
- 人機對話式交互
- 靈活的工具調用

**範例**：
```python
# 用戶提問 ↔ Assistant 分析 ↔ 執行代碼 ↔ 展示結果
assistant.initiate_chat(user_proxy, message=query)
```

#### 5.3.4 客服機器人
**推薦**：LangGraph

**理由**：
- 需要精確的對話流程控制
- 狀態管理（用戶資訊、訂單狀態）
- 支援中斷和恢復

### 5.4 混合使用策略

很多實際應用會結合多個框架：

```python
# 範例：使用 LangGraph 編排，CrewAI 執行子任務

from langgraph.graph import StateGraph
from crewai import Crew

class HybridAgentState(TypedDict):
    task: str
    research_result: str
    final_output: str

def research_node(state):
    """使用 CrewAI 執行研究任務"""
    crew = Crew(agents=[researcher], tasks=[research_task])
    result = crew.kickoff()
    return {"research_result": result}

def analysis_node(state):
    """使用 LangGraph 控制分析流程"""
    # 複雜的條件邏輯
    ...

workflow = StateGraph(HybridAgentState)
workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
```

---

## 6. 實戰案例：多 Agent 協作系統

### 6.1 案例一：AI 技術研究助手

#### 6.1.1 需求分析

**目標**：構建一個能自動研究 AI 技術主題並生成報告的系統。

**功能要求**：
1. 搜尋最新論文和文章
2. 總結關鍵發現
3. 生成結構化報告
4. 包含代碼示例

**Agent 設計**：
- **研究員 Agent**：搜尋和收集資料
- **分析師 Agent**：分析和總結
- **工程師 Agent**：撰寫代碼示例
- **編輯 Agent**：整合和優化報告

#### 6.1.2 使用 CrewAI 實作

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool, WebsiteSearchTool

# 初始化工具
search_tool = SerperDevTool()  # Google Search API
web_tool = WebsiteSearchTool()

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Agent 1: 研究員
researcher = Agent(
    role='AI 技術研究員',
    goal='深入研究指定的 AI 技術主題，找出最新發展和關鍵論文',
    backstory="""你是一位經驗豐富的 AI 研究員，擅長快速掌握新技術。
    你知道如何搜尋高質量的技術資源，包括 Arxiv、GitHub、技術博客等。""",
    tools=[search_tool, web_tool],
    verbose=True,
    llm=llm
)

# Agent 2: 分析師
analyst = Agent(
    role='技術分析師',
    goal='分析研究結果，提取核心概念和技術要點',
    backstory="""你是一位技術分析專家，能夠從大量資訊中提取關鍵洞察。
    你擅長識別技術趨勢、比較不同方法的優劣。""",
    verbose=True,
    llm=llm
)

# Agent 3: 工程師
engineer = Agent(
    role='軟體工程師',
    goal='撰寫清晰的代碼示例，展示技術實作',
    backstory="""你是一位經驗豐富的軟體工程師，擅長將技術概念轉化為可運行的代碼。
    你的代碼清晰、有註釋、遵循最佳實踐。""",
    verbose=True,
    llm=llm
)

# Agent 4: 編輯
editor = Agent(
    role='技術編輯',
    goal='整合所有內容，生成高質量的技術報告',
    backstory="""你是一位專業的技術編輯，能夠將複雜的技術內容組織成
    易讀的報告。你注重邏輯性、完整性和可讀性。""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# 定義任務
def create_research_tasks(topic: str):
    task1 = Task(
        description=f"""研究主題：{topic}

        請完成以下任務：
        1. 搜尋最新的論文和文章（過去 12 個月）
        2. 找出 3-5 個關鍵資源
        3. 識別主要的技術突破和趨勢
        4. 記錄重要的數據和統計

        輸出格式：
        - 資源列表（標題、來源、連結）
        - 關鍵發現（每個 2-3 句話）""",
        expected_output="結構化的研究報告，包含資源清單和關鍵發現",
        agent=researcher
    )

    task2 = Task(
        description=f"""基於研究員提供的資料，進行深入分析：

        1. 總結技術的核心概念
        2. 比較不同的實作方法
        3. 分析優勢和限制
        4. 識別最佳實踐
        5. 提出實際應用場景

        輸出應該清晰、結構化，適合技術讀者。""",
        expected_output="技術分析報告，包含核心概念、方法比較、最佳實踐",
        agent=analyst,
        context=[task1]
    )

    task3 = Task(
        description=f"""基於分析結果，撰寫代碼示例：

        1. 選擇 2-3 個關鍵概念
        2. 為每個概念撰寫完整的代碼示例
        3. 添加註釋和說明
        4. 確保代碼可以運行
        5. 包含 requirements（依賴套件）

        代碼應該：
        - 簡潔清晰
        - 遵循 PEP 8 風格
        - 包含錯誤處理
        - 有完整的文檔字符串""",
        expected_output="完整的代碼示例，包含註釋和使用說明",
        agent=engineer,
        context=[task2]
    )

    task4 = Task(
        description=f"""整合所有內容，生成最終報告：

        報告結構：
        1. 執行摘要（200 字）
        2. 技術背景和動機
        3. 核心概念解析
        4. 實作方法比較
        5. 代碼示例和說明
        6. 最佳實踐建議
        7. 延伸閱讀

        要求：
        - 邏輯清晰，結構完整
        - 技術準確
        - 適合中高級技術讀者
        - Markdown 格式""",
        expected_output="完整的技術報告（Markdown 格式）",
        agent=editor,
        context=[task1, task2, task3]
    )

    return [task1, task2, task3, task4]

# 創建並執行 Crew
def research_topic(topic: str):
    tasks = create_research_tasks(topic)

    crew = Crew(
        agents=[researcher, analyst, engineer, editor],
        tasks=tasks,
        process=Process.sequential,
        verbose=2
    )

    result = crew.kickoff()
    return result

# 使用範例
if __name__ == "__main__":
    topic = "LangGraph for Multi-Agent Systems"
    report = research_topic(topic)

    # 保存報告
    with open(f"report_{topic.replace(' ', '_')}.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✓ 報告已生成：report_{topic.replace(' ', '_')}.md")
```

### 6.2 案例二：客戶服務自動化系統

#### 6.2.1 需求分析

**場景**：電商客服系統，處理訂單查詢、退換貨、投訴等。

**要求**：
- 理解客戶意圖
- 查詢訂單資料庫
- 執行業務操作（退款、換貨）
- 需要人工審核（高金額）
- 生成服務記錄

#### 6.2.2 使用 LangGraph 實作

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Literal, Annotated
import operator

# 定義狀態
class CustomerServiceState(TypedDict):
    customer_id: str
    query: str
    intent: str  # query, refund, exchange, complaint
    order_info: dict
    action_required: str
    needs_human: bool
    response: str
    messages: Annotated[list, operator.add]

# 定義工具
@tool
def query_order(order_id: str) -> dict:
    """查詢訂單資訊"""
    # 模擬資料庫查詢
    mock_db = {
        "ORD001": {
            "order_id": "ORD001",
            "customer_id": "CUST123",
            "items": ["商品A", "商品B"],
            "total": 1500,
            "status": "已配送",
            "date": "2024-01-15"
        },
        "ORD002": {
            "order_id": "ORD002",
            "customer_id": "CUST123",
            "items": ["商品C"],
            "total": 3500,
            "status": "處理中",
            "date": "2024-01-20"
        }
    }
    return mock_db.get(order_id, {"error": "訂單不存在"})

@tool
def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """處理退款"""
    # 模擬退款處理
    if amount > 3000:
        return {
            "status": "pending_approval",
            "message": "高額退款需要人工審核",
            "amount": amount
        }
    return {
        "status": "success",
        "message": f"退款 ${amount} 已處理，預計 3-5 個工作天到帳",
        "amount": amount
    }

tools = [query_order, process_refund]

# 定義節點
def classify_intent(state: CustomerServiceState):
    """分類客戶意圖"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt = f"""分類以下客戶查詢的意圖：

客戶查詢：{state['query']}

可能的意圖：
- order_query: 查詢訂單狀態
- refund: 申請退款
- exchange: 申請換貨
- complaint: 投訴

只回答意圖類別（一個詞）。"""

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    return {"intent": intent}

def handle_query(state: CustomerServiceState):
    """處理查詢"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = state.get("messages", [])
    messages.append({
        "role": "user",
        "content": state["query"]
    })

    response = llm_with_tools.invoke(messages)

    # 如果有工具調用
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "query_order":
            order_info = query_order.invoke(tool_call["args"])
            return {
                "order_info": order_info,
                "messages": [{"role": "assistant", "content": str(order_info)}]
            }

    return {"messages": [{"role": "assistant", "content": response.content}]}

def handle_refund(state: CustomerServiceState):
    """處理退款"""
    order_info = state.get("order_info", {})

    if not order_info or "error" in order_info:
        return {
            "response": "請先提供有效的訂單編號",
            "needs_human": False
        }

    # 執行退款
    result = process_refund.invoke({
        "order_id": order_info["order_id"],
        "amount": order_info["total"],
        "reason": "客戶申請"
    })

    needs_human = result["status"] == "pending_approval"

    return {
        "response": result["message"],
        "needs_human": needs_human
    }

def human_review(state: CustomerServiceState):
    """人工審核"""
    print("\n" + "="*50)
    print("需要人工審核")
    print("="*50)
    print(f"客戶 ID: {state['customer_id']}")
    print(f"訂單資訊: {state['order_info']}")
    print(f"請求: {state['query']}")
    print("="*50)

    # 在實際應用中，這裡會暫停並等待人工輸入
    approval = input("批准？(y/n): ")

    if approval.lower() == 'y':
        return {"response": "您的退款申請已批准，預計 3-5 個工作天到帳"}
    else:
        return {"response": "抱歉，您的退款申請未通過審核"}

def generate_response(state: CustomerServiceState):
    """生成最終回應"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    context = f"""
    客戶意圖: {state['intent']}
    訂單資訊: {state.get('order_info', '無')}
    處理結果: {state.get('response', '處理中')}

    請生成一個友善、專業的客服回應。
    """

    response = llm.invoke(context)
    return {"response": response.content}

# 決策函數
def should_handle_refund(state: CustomerServiceState) -> Literal["refund", "query", "other"]:
    """決定下一步"""
    intent = state.get("intent", "")
    if "refund" in intent:
        return "refund"
    elif "query" in intent or "order" in intent:
        return "query"
    else:
        return "other"

def needs_approval(state: CustomerServiceState) -> Literal["human", "respond"]:
    """是否需要人工審核"""
    return "human" if state.get("needs_human", False) else "respond"

# 構建圖
workflow = StateGraph(CustomerServiceState)

# 添加節點
workflow.add_node("classify", classify_intent)
workflow.add_node("handle_query", handle_query)
workflow.add_node("handle_refund", handle_refund)
workflow.add_node("human_review", human_review)
workflow.add_node("respond", generate_response)

# 設置流程
workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    should_handle_refund,
    {
        "query": "handle_query",
        "refund": "handle_refund",
        "other": "respond"
    }
)

workflow.add_edge("handle_query", "respond")

workflow.add_conditional_edges(
    "handle_refund",
    needs_approval,
    {
        "human": "human_review",
        "respond": "respond"
    }
)

workflow.add_edge("human_review", "respond")
workflow.add_edge("respond", END)

# 編譯
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 使用範例
if __name__ == "__main__":
    # 測試案例 1: 查詢訂單
    config = {"configurable": {"thread_id": "customer-123"}}

    result = app.invoke({
        "customer_id": "CUST123",
        "query": "我想查詢訂單 ORD001 的狀態",
        "messages": []
    }, config=config)

    print("\n客服回應：")
    print(result["response"])

    # 測試案例 2: 申請退款（低金額）
    result = app.invoke({
        "customer_id": "CUST123",
        "query": "我要退款訂單 ORD001",
        "messages": []
    }, config=config)

    print("\n客服回應：")
    print(result["response"])

    # 測試案例 3: 申請退款（高金額，需審核）
    result = app.invoke({
        "customer_id": "CUST123",
        "query": "我要退款訂單 ORD002",
        "messages": []
    }, config=config)

    print("\n客服回應：")
    print(result["response"])
```

---

## 8. 評估與監控

### 8.1 Agent 性能評估指標

#### 8.1.1 任務成功率

```python
class AgentEvaluator:
    """Agent 評估器"""

    def __init__(self):
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.task_logs = []

    def evaluate_task(self, task_description: str,
                     expected_output: str,
                     actual_output: str) -> dict:
        """評估單個任務"""
        self.total_tasks += 1

        # 使用 LLM 評估輸出質量
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        eval_prompt = f"""評估 AI Agent 的輸出質量：

任務描述：{task_description}
期望輸出：{expected_output}
實際輸出：{actual_output}

請評估以下維度（0-10 分）：
1. 準確性：輸出是否正確
2. 完整性：是否包含所有必要資訊
3. 相關性：是否切題
4. 質量：語言和格式是否良好

以 JSON 格式輸出：
{{
    "accuracy": 分數,
    "completeness": 分數,
    "relevance": 分數,
    "quality": 分數,
    "overall": 平均分,
    "feedback": "改進建議"
}}"""

        result = llm.invoke(eval_prompt)

        # 解析結果
        import json
        try:
            scores = json.loads(result.content)
            success = scores["overall"] >= 7.0

            if success:
                self.successful_tasks += 1
            else:
                self.failed_tasks += 1

            log_entry = {
                "task": task_description,
                "scores": scores,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
            self.task_logs.append(log_entry)

            return scores
        except:
            return {"error": "評估失敗"}

    def get_success_rate(self) -> float:
        """獲取成功率"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def generate_report(self) -> str:
        """生成評估報告"""
        report = f"""
# Agent 評估報告

## 總體統計
- 總任務數：{self.total_tasks}
- 成功任務：{self.successful_tasks}
- 失敗任務：{self.failed_tasks}
- 成功率：{self.get_success_rate():.2%}

## 平均分數
"""
        if self.task_logs:
            avg_accuracy = sum(log["scores"].get("accuracy", 0) for log in self.task_logs) / len(self.task_logs)
            avg_completeness = sum(log["scores"].get("completeness", 0) for log in self.task_logs) / len(self.task_logs)
            avg_relevance = sum(log["scores"].get("relevance", 0) for log in self.task_logs) / len(self.task_logs)
            avg_quality = sum(log["scores"].get("quality", 0) for log in self.task_logs) / len(self.task_logs)

            report += f"""
- 準確性：{avg_accuracy:.2f}/10
- 完整性：{avg_completeness:.2f}/10
- 相關性：{avg_relevance:.2f}/10
- 質量：{avg_quality:.2f}/10
"""

        return report
```

#### 8.1.2 成本監控

```python
class CostTracker:
    """成本追蹤器"""

    # 2025 價格（美元）
    PRICING = {
        "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "claude-3-opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
        "claude-3.5-sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    }

    def __init__(self):
        self.usage_log = []
        self.total_cost = 0.0

    def log_usage(self, model: str, input_tokens: int, output_tokens: int):
        """記錄使用量"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        total_cost = input_cost + output_cost

        self.total_cost += total_cost

        self.usage_log.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost
        })

    def get_report(self) -> dict:
        """獲取成本報告"""
        return {
            "total_cost": self.total_cost,
            "total_calls": len(self.usage_log),
            "avg_cost_per_call": self.total_cost / len(self.usage_log) if self.usage_log else 0,
            "by_model": self._group_by_model()
        }

    def _group_by_model(self) -> dict:
        """按模型分組統計"""
        grouped = {}
        for log in self.usage_log:
            model = log["model"]
            if model not in grouped:
                grouped[model] = {
                    "calls": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0
                }
            grouped[model]["calls"] += 1
            grouped[model]["total_cost"] += log["cost"]
            grouped[model]["total_tokens"] += log["input_tokens"] + log["output_tokens"]

        return grouped
```

### 8.2 LangSmith 整合

```python
import os
from langsmith import Client
from langchain.callbacks import LangChainTracer

# 設置 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "agent-evaluation"

# 創建追蹤器
tracer = LangChainTracer(project_name="agent-evaluation")

# 在 Agent 中使用
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[tracer],  # 添加追蹤
    verbose=True
)

# 執行會自動記錄到 LangSmith
result = agent.run("查詢台北天氣")
```

---

## 9. 最佳實踐與設計模式

### 9.1 Agent 設計原則

#### 9.1.1 單一職責原則

每個 Agent 應該有清晰的單一職責：

```python
# ❌ 不好：一個 Agent 做太多事
universal_agent = Agent(
    role="萬能助手",
    goal="處理所有任務"
)

# ✅ 好：職責明確的 Agent
search_agent = Agent(
    role="搜尋專家",
    goal="搜尋和收集資訊"
)

analysis_agent = Agent(
    role="分析師",
    goal="分析數據並提取洞察"
)
```

#### 9.1.2 明確的輸入輸出

```python
from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    """明確的輸入格式"""
    query: str = Field(description="用戶查詢")
    context: dict = Field(default={}, description="上下文資訊")
    max_iterations: int = Field(default=5, description="最大迭代次數")

class AgentOutput(BaseModel):
    """明確的輸出格式"""
    result: str = Field(description="最終結果")
    confidence: float = Field(description="信心分數 0-1")
    sources: list[str] = Field(description="資訊來源")
    intermediate_steps: list[dict] = Field(description="中間步驟")
```

#### 9.1.3 錯誤處理和重試

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustAgent:
    """具備錯誤處理的 Agent"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def execute_with_retry(self, task: str) -> str:
        """帶重試的執行"""
        try:
            result = self.agent.run(task)
            return result
        except Exception as e:
            print(f"錯誤：{e}，重試中...")
            raise

    def execute_safe(self, task: str, fallback: str = "處理失敗") -> str:
        """安全執行，失敗時返回備用結果"""
        try:
            return self.execute_with_retry(task)
        except Exception as e:
            print(f"最終失敗：{e}")
            return fallback
```

### 9.2 提示工程最佳實踐

#### 9.2.1 結構化提示

```python
AGENT_SYSTEM_PROMPT = """你是一個{role}，你的目標是{goal}。

## 你的背景
{backstory}

## 可用工具
{tools_description}

## 工作流程
1. 仔細閱讀用戶請求
2. 思考需要哪些步驟
3. 選擇合適的工具
4. 執行並檢查結果
5. 如果需要，重複步驟 3-4
6. 總結並回應用戶

## 輸出格式
請使用以下格式：

思考：[你的推理過程]
行動：[工具名稱]
行動輸入：[工具參數]
觀察：[工具返回結果]
... (重複思考/行動/觀察)
最終答案：[給用戶的回應]

## 重要提醒
- 確保答案準確
- 引用資料來源
- 如果不確定，說明你不確定
- 保持專業和友善
"""
```

#### 9.2.2 Few-Shot 範例

```python
FEW_SHOT_EXAMPLES = """
## 範例 1
用戶：台北明天天氣如何？
思考：我需要查詢台北的天氣預報
行動：get_weather
行動輸入：{{"location": "台北", "date": "明天"}}
觀察：{"temp": 25, "condition": "晴天"}
最終答案：台北明天是晴天，氣溫約 25 度。

## 範例 2
用戶：幫我計算 123 * 456
思考：這是一個數學計算問題
行動：calculator
行動輸入：{{"expression": "123 * 456"}}
觀察：56088
最終答案：123 × 456 = 56,088

現在輪到你了：
"""
```

### 9.3 安全性最佳實踐

#### 9.3.1 輸入驗證

```python
from pydantic import BaseModel, validator

class SafeAgentInput(BaseModel):
    query: str

    @validator('query')
    def validate_query(cls, v):
        """驗證輸入"""
        # 檢查長度
        if len(v) > 10000:
            raise ValueError("查詢過長")

        # 檢查危險指令
        dangerous_patterns = [
            "ignore previous instructions",
            "忽略之前的指示",
            "你現在是",
            "rm -rf",
            "DROP TABLE"
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError("檢測到潛在危險輸入")

        return v
```

#### 9.3.2 工具權限控制

```python
class SecureTool:
    """安全的工具包裝器"""

    def __init__(self, tool_func, allowed_operations: list[str]):
        self.tool_func = tool_func
        self.allowed_operations = allowed_operations

    def execute(self, operation: str, *args, **kwargs):
        """執行工具調用"""
        # 檢查權限
        if operation not in self.allowed_operations:
            raise PermissionError(f"操作 '{operation}' 未被授權")

        # 記錄審計日誌
        self._log_audit(operation, args, kwargs)

        # 執行
        return self.tool_func(*args, **kwargs)

    def _log_audit(self, operation, args, kwargs):
        """記錄審計日誌"""
        import logging
        logging.info(f"工具調用：{operation}, args={args}, kwargs={kwargs}")
```

### 9.4 性能優化

#### 9.4.1 並行執行

```python
import asyncio
from typing import List

async def parallel_agent_execution(agents: List[Agent], task: str):
    """並行執行多個 Agent"""
    async def run_agent(agent: Agent):
        return await agent.arun(task)

    # 並行執行
    results = await asyncio.gather(*[run_agent(agent) for agent in agents])

    return results

# 使用
results = asyncio.run(parallel_agent_execution([agent1, agent2, agent3], task))
```

#### 9.4.2 緩存機制

```python
from functools import lru_cache
import hashlib

class CachedAgent:
    """帶緩存的 Agent"""

    def __init__(self, agent):
        self.agent = agent
        self.cache = {}

    def run(self, task: str) -> str:
        """執行任務（帶緩存）"""
        # 生成緩存鍵
        cache_key = hashlib.md5(task.encode()).hexdigest()

        # 檢查緩存
        if cache_key in self.cache:
            print("使用緩存結果")
            return self.cache[cache_key]

        # 執行
        result = self.agent.run(task)

        # 保存緩存
        self.cache[cache_key] = result

        return result
```

---

## 10. 未來趨勢與展望

### 10.1 技術發展方向

#### 10.1.1 多模態 Agent

未來的 Agent 將整合：
- **視覺**：圖像理解和生成
- **聽覺**：語音交互
- **行動**：機器人控制

```python
# 未來的多模態 Agent 概念
class MultimodalAgent:
    def process(self, inputs: dict):
        """
        inputs = {
            "text": "找出這張圖片中的問題",
            "image": image_data,
            "audio": audio_data
        }
        """
        # 整合多種模態
        vision_result = self.vision_model(inputs["image"])
        audio_result = self.audio_model(inputs["audio"])

        # 融合推理
        result = self.reasoning_model(
            text=inputs["text"],
            vision=vision_result,
            audio=audio_result
        )

        return result
```

#### 10.1.2 自主學習 Agent

Agent 將具備：
- **持續學習**：從交互中學習
- **知識積累**：構建個人知識庫
- **技能擴展**：自主學習新工具

#### 10.1.3 Agent 作業系統

類似 OS 的 Agent 平台：
- **資源管理**：管理計算和 API 配額
- **進程調度**：協調多個 Agent
- **權限控制**：安全和隱私保護

### 10.2 應用場景擴展

#### 10.2.1 科學研究

- 自動文獻綜述
- 實驗設計和執行
- 數據分析和可視化

#### 10.2.2 軟體開發

- 全自動代碼生成
- 智能測試和調試
- 架構設計和重構

#### 10.2.3 個人助理

- 日程管理和規劃
- 學習輔導
- 健康監測和建議

### 10.3 挑戰與機遇

#### 挑戰

1. **可靠性**：如何確保 Agent 穩定運行
2. **可解釋性**：如何理解 Agent 的決策
3. **安全性**：如何防止濫用和攻擊
4. **成本**：如何降低運行成本
5. **標準化**：如何建立統一標準

#### 機遇

1. **生產力革命**：大幅提升工作效率
2. **新職業**：Agent 訓練師、監督者
3. **行業轉型**：傳統行業自動化升級

---

## 11. 延伸閱讀

### 11.1 論文

- **ReAct**: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
- **Chain of Thought**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- **Toolformer**: Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- **AutoGPT**: "Auto-GPT: An Autonomous GPT-4 Experiment" (2023)

### 11.2 官方文檔

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)

### 11.3 開源項目

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGen](https://github.com/microsoft/autogen)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

### 11.4 學習資源

- [DeepLearning.AI: Building Applications with Vector Databases](https://www.deeplearning.ai/)
- [LangChain Academy](https://academy.langchain.com/)
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)

---

## 總結

AI Agents 正在重塑我們與 AI 的交互方式。從簡單的 ReAct 模式到複雜的多 Agent 系統，我們看到了巨大的發展潛力。

**關鍵要點**：

1. **選擇合適的框架**：根據需求選擇 LangGraph、CrewAI、AutoGen 等
2. **設計清晰的 Agent**：單一職責、明確輸入輸出
3. **注重安全性**：輸入驗證、權限控制、審計日誌
4. **持續評估**：監控性能、成本、成功率
5. **迭代改進**：從實踐中學習，不斷優化

**下一步行動**：

- 選擇一個框架開始實踐
- 構建簡單的 Agent 原型
- 逐步增加複雜性
- 分享你的經驗和學習

祝你在 AI Agent 的旅程中取得成功！🚀
