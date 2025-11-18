# AI Agents 範例

## 📚 什麼是 AI Agent？

AI Agent 是一種能夠：
1. **感知**環境（透過工具獲取資訊）
2. **推理**思考（使用 LLM 進行推理）
3. **行動**執行（調用工具完成任務）
4. **學習**改進（從結果中學習，調整策略）

的智能系統。

## 🎯 Agent 的核心元件

```
Agent = LLM (大腦) + Tools (工具) + Planning (規劃) + Memory (記憶)
```

### 1. LLM（大腦）
負責理解、推理和決策

### 2. Tools（工具）
Agent 可以調用的功能：
- 搜尋引擎
- 計算器
- 資料庫查詢
- API 呼叫
- 自訂函數

### 3. Planning（規劃）
決定要採取什麼行動的策略：
- **ReAct**: Reasoning + Acting（推理 + 行動）
- **Plan-and-Execute**: 先規劃，再執行
- **Tool-Calling**: 直接調用工具

### 4. Memory（記憶）
記住之前的對話和行動

## 📖 範例說明

### agent_demo.py

包含四個完整的 Agent 示範：

#### 1. ReAct Agent
```python
# ReAct = Reasoning（推理） + Acting（行動）
# Agent 會交替進行思考和行動

Question: 台北的天氣如何？
Thought: 我需要使用 Weather 工具來查詢天氣
Action: Weather
Action Input: 台北
Observation: 晴天，溫度 28°C
Thought: 我現在知道答案了
Final Answer: 台北今天是晴天，溫度 28°C
```

**適合場景：**
- 需要多步推理的複雜任務
- 需要看到 Agent 思考過程

**優點：**
- 思考過程透明
- 容易除錯
- 可以處理複雜邏輯

**缺點：**
- 速度較慢
- Token 消耗較多

#### 2. Tool-Calling Agent
```python
# 使用 OpenAI 的 Function Calling 功能
# 更快速、更精確的工具調用

問題: 計算 999 加 1
→ 直接調用 Calculator("999 + 1")
→ 返回結果: 1000
```

**適合場景：**
- 簡單直接的工具調用
- 需要快速回應

**優點：**
- 速度快
- Token 消耗少
- 工具調用更準確

**缺點：**
- 需要支援 Function Calling 的模型
- 複雜推理能力較弱

#### 3. Search Agent
```python
# 使用網路搜尋工具的 Agent
# 可以獲取最新資訊

工具:
- DuckDuckGo Search（網路搜尋）
- Wikipedia（百科知識）
- Calculator（計算）
```

**適合場景：**
- 需要最新資訊
- 需要查詢百科知識
- 研究和資訊蒐集

#### 4. Conversational Agent
```python
# 帶有記憶的對話式 Agent
# 能記住之前的對話內容

使用者: 台北今天天氣如何？
AI: 台北今天是晴天，溫度 28°C

使用者: 那邊的溫度是幾度？  # 參考前一個問題
AI: 台北的溫度是 28°C      # Agent 記得在談論台北
```

**適合場景：**
- 多輪對話
- 需要上下文理解
- 客服機器人

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd "3.LLM應用工程/1.LangchainDemos"
pip install -r requirements.txt
```

### 2. 設定環境變數

```bash
# 複製範例檔案
cp .env.example .env

# 編輯 .env，填入你的 API Key
OPENAI_API_KEY=your_key_here
```

### 3. 執行範例

```bash
cd "2.AI Agents範例"
python agent_demo.py
```

## 💡 使用範例

### 範例 1: 建立簡單的 Agent

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from utils import get_llm

# 1. 定義工具
def my_tool(input: str) -> str:
    return f"處理結果: {input}"

tools = [
    Tool(
        name="MyTool",
        func=my_tool,
        description="我的自訂工具"
    )
]

# 2. 建立 LLM
llm = get_llm()

# 3. 建立 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個助手"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. 建立 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 5. 使用 Agent
result = agent_executor.invoke({"input": "測試"})
print(result["output"])
```

### 範例 2: 自訂工具

```python
from langchain.tools import Tool

def search_database(query: str) -> str:
    """搜尋資料庫"""
    # 你的資料庫查詢邏輯
    results = database.search(query)
    return f"找到 {len(results)} 筆資料"

def send_email(content: str) -> str:
    """發送郵件"""
    # 你的郵件發送邏輯
    email_service.send(content)
    return "郵件已發送"

tools = [
    Tool(
        name="DatabaseSearch",
        func=search_database,
        description="在資料庫中搜尋資訊"
    ),
    Tool(
        name="SendEmail",
        func=send_email,
        description="發送電子郵件"
    ),
]
```

### 範例 3: 使用對話歷史

```python
from agent_demo import ConversationalAgent

agent = ConversationalAgent()

# 連續對話
agent.chat("你好")
agent.chat("台北天氣如何？")
agent.chat("那邊溫度幾度？")  # 會記得在討論台北

# 清除歷史，開始新對話
agent.clear_history()
```

## 🛠️ 可用的工具

### 內建工具

```python
# 計算器
from langchain.tools import Tool
calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="進行數學計算"
)

# 搜尋
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

# Wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()

# Python REPL（執行 Python 程式碼）
from langchain.tools import PythonREPLTool
python_repl = PythonREPLTool()
```

### 更多工具

LangChain 提供了 50+ 種內建工具：
- 檔案操作
- API 呼叫
- 資料庫查詢
- 網路爬蟲
- 程式碼執行
- ...

查看完整列表：https://python.langchain.com/docs/integrations/tools/

## 📊 Agent 類型比較

| Agent 類型 | 速度 | Token 消耗 | 複雜推理 | 透明度 | 適用場景 |
|-----------|------|-----------|---------|--------|---------|
| ReAct | 慢 | 高 | ★★★★★ | ★★★★★ | 複雜任務 |
| Tool-Calling | 快 | 低 | ★★★☆☆ | ★★★☆☆ | 簡單任務 |
| Plan-and-Execute | 中 | 中 | ★★★★☆ | ★★★★☆ | 多步驟任務 |

## 🎓 進階主題

### 1. 錯誤處理

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 處理解析錯誤
    max_iterations=5,             # 限制最大迭代次數
    early_stopping_method="generate",  # 提前停止策略
)
```

### 2. 自訂輸出解析器

```python
from langchain.agents import AgentOutputParser

class MyOutputParser(AgentOutputParser):
    def parse(self, text: str):
        # 自訂解析邏輯
        pass
```

### 3. 串流輸出

```python
for chunk in agent_executor.stream({"input": "問題"}):
    print(chunk)
```

## 🔍 常見問題

### Q1: Agent 一直重複同樣的動作怎麼辦？

設定最大迭代次數：
```python
AgentExecutor(max_iterations=5)
```

### Q2: 如何讓 Agent 使用特定工具？

在 prompt 中明確指示：
```python
prompt = """
如果需要計算，請使用 Calculator 工具。
如果需要搜尋，請使用 Search 工具。
"""
```

### Q3: 如何除錯 Agent？

開啟 verbose 模式：
```python
agent_executor = AgentExecutor(verbose=True)
```

或使用 LangSmith 追蹤：
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

### Q4: Agent 效能如何優化？

1. 使用更快的模型（如 gpt-3.5-turbo）
2. 減少工具數量
3. 使用 Tool-Calling 而非 ReAct
4. 設定合理的 max_iterations

## 🌟 最佳實踐

1. **工具描述要清楚**
   ```python
   # ❌ 不好
   description="計算"

   # ✅ 好
   description="進行數學計算。輸入應該是數學表達式，例如 '2+2' 或 '10*5'"
   ```

2. **限制 Agent 的行動範圍**
   ```python
   max_iterations=5  # 避免無限循環
   ```

3. **提供明確的指示**
   ```python
   system_prompt = "請用繁體中文回答。如果不確定，請誠實說不知道。"
   ```

4. **處理錯誤**
   ```python
   handle_parsing_errors=True
   ```

5. **使用適當的模型**
   - 簡單任務：gpt-3.5-turbo
   - 複雜推理：gpt-4

## 📚 參考資源

- [LangChain Agents 官方文件](https://python.langchain.com/docs/modules/agents/)
- [ReAct 論文](https://arxiv.org/abs/2210.03629)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

## 下一步

1. 嘗試修改範例程式，加入自己的工具
2. 建立一個實用的 Agent（例如：研究助手、程式碼助手）
3. 探索更多內建工具
4. 學習如何將 Agent 部署到生產環境

---

Happy Coding! 🚀
