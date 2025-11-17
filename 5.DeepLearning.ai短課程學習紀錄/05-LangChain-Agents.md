# Functions, Tools and Agents with LangChain

## 📋 課程概述

深入學習 OpenAI Function Calling 和 LangChain Agents，建立能夠使用工具的智慧代理。

### 課程目標
- 掌握 OpenAI Function Calling API
- 學習建立和使用 LangChain Tools
- 理解 Agents 的工作原理
- 實作多工具協作的智慧系統

### 課程時長
約 1 小時

## 🔧 OpenAI Function Calling

### 基本概念

Function Calling 讓 LLM 能夠識別何時需要呼叫外部函數，並生成正確的參數。

```python
from openai import OpenAI
import json

client = OpenAI()

# 定義函數規格
functions = [
    {
        "name": "get_weather",
        "description": "獲取指定城市的當前天氣資訊",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名稱，例如：台北、台中"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "溫度單位"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "獲取股票的當前價格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代碼，例如：2330.TW（台積電）"
                }
            },
            "required": ["symbol"]
        }
    }
]

# 呼叫 API
messages = [{"role": "user", "content": "台北現在天氣如何？"}]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto"  # 讓模型自動決定是否呼叫函數
)

response_message = response.choices[0].message

# 檢查是否需要呼叫函數
if response_message.function_call:
    function_name = response_message.function_call.name
    function_args = json.loads(response_message.function_call.arguments)

    print(f"需要呼叫函數：{function_name}")
    print(f"參數：{function_args}")

    # 實際呼叫函數
    if function_name == "get_weather":
        # 模擬天氣 API
        weather_info = {
            "location": function_args["location"],
            "temperature": 25,
            "condition": "晴天",
            "humidity": 65
        }

        # 將結果回傳給模型
        messages.append(response_message)
        messages.append({
            "role": "function",
            "name": function_name,
            "content": json.dumps(weather_info, ensure_ascii=False)
        })

        # 獲取最終回應
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        print(f"\n最終回應：{second_response.choices[0].message.content}")
```

### 完整範例：多函數呼叫

```python
import json
from typing import Dict, Any

# 定義實際的函數實作
def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """獲取天氣資訊（模擬）"""
    weather_data = {
        "台北": {"temp": 25, "condition": "晴天"},
        "台中": {"temp": 27, "condition": "多雲"},
        "高雄": {"temp": 29, "condition": "晴天"}
    }

    if location in weather_data:
        data = weather_data[location]
        if unit == "fahrenheit":
            data["temp"] = data["temp"] * 9/5 + 32
        return {
            "location": location,
            "temperature": data["temp"],
            "condition": data["condition"],
            "unit": unit
        }
    return {"error": "找不到該城市的天氣資訊"}

def get_stock_price(symbol: str) -> Dict[str, Any]:
    """獲取股票價格（模擬）"""
    stocks = {
        "2330.TW": {"name": "台積電", "price": 580, "change": +2.5},
        "2317.TW": {"name": "鴻海", "price": 105, "change": -1.2}
    }

    if symbol in stocks:
        return stocks[symbol]
    return {"error": "找不到該股票"}

def calculate(expression: str) -> Dict[str, Any]:
    """計算數學表達式"""
    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

# 可用函數映射
available_functions = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "calculate": calculate
}

# 函數定義
functions = [
    {
        "name": "get_weather",
        "description": "獲取指定城市的天氣",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名稱"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "查詢股票價格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代碼"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "calculate",
        "description": "計算數學表達式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "數學表達式"}
            },
            "required": ["expression"]
        }
    }
]

def run_conversation(user_message: str):
    """執行完整的對話流程"""
    messages = [{"role": "user", "content": user_message}]

    # 第一次 API 呼叫
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    response_message = response.choices[0].message

    # 檢查是否需要呼叫函數
    if response_message.function_call:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)

        print(f"🔧 呼叫函數：{function_name}")
        print(f"📝 參數：{function_args}")

        # 呼叫實際函數
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(f"✅ 函數回應：{function_response}\n")

        # 將結果加入對話
        messages.append(response_message)
        messages.append({
            "role": "function",
            "name": function_name,
            "content": json.dumps(function_response, ensure_ascii=False)
        })

        # 第二次 API 呼叫
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        return second_response.choices[0].message.content

    return response_message.content

# 測試
print("範例 1：天氣查詢")
print(run_conversation("台北今天天氣如何？"))

print("\n範例 2：股票查詢")
print(run_conversation("台積電的股價多少？"))

print("\n範例 3：計算")
print(run_conversation("幫我算 1234 * 5678"))
```

## 🛠️ LangChain Tools

### 建立自訂工具

```python
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type

# 方法 1：使用 Tool 類別
def search_wikipedia(query: str) -> str:
    """搜尋維基百科（模擬）"""
    return f"關於「{query}」的維基百科資訊..."

wikipedia_tool = Tool(
    name="維基百科搜尋",
    func=search_wikipedia,
    description="搜尋維基百科獲取知識。輸入：搜尋關鍵字"
)

# 方法 2：使用 StructuredTool（支援多參數）
def calculate_age(birth_year: int, current_year: int = 2025) -> str:
    """計算年齡"""
    age = current_year - birth_year
    return f"年齡：{age} 歲"

age_calculator = StructuredTool.from_function(
    func=calculate_age,
    name="年齡計算器",
    description="根據出生年份計算年齡"
)

# 方法 3：繼承 BaseTool（最靈活）
class CustomSearchInput(BaseModel):
    query: str = Field(description="搜尋關鍵字")
    location: str = Field(description="搜尋地區", default="台灣")

class CustomSearchTool(BaseTool):
    name = "自訂搜尋"
    description = "搜尋台灣地區的資訊"
    args_schema: Type[BaseModel] = CustomSearchInput

    def _run(self, query: str, location: str = "台灣") -> str:
        """執行工具"""
        return f"在「{location}」搜尋「{query}」的結果..."

    async def _arun(self, query: str, location: str = "台灣") -> str:
        """非同步執行"""
        return self._run(query, location)

# 測試工具
custom_tool = CustomSearchTool()
result = custom_tool.run({"query": "台北美食", "location": "台北"})
print(result)
```

### 使用預建工具

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Wikipedia 工具
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

result = wikipedia.run("台灣")
print(result[:200])
```

### 多工具組合

```python
from langchain.tools import Tool
import requests

# 工具 1：天氣查詢
def get_weather_info(city: str) -> str:
    """查詢天氣"""
    # 實際應該呼叫天氣 API
    return f"{city}的天氣：晴天，25°C"

# 工具 2：匯率查詢
def get_exchange_rate(currency: str) -> str:
    """查詢匯率"""
    rates = {
        "USD": 31.5,
        "JPY": 0.21,
        "EUR": 34.2
    }
    rate = rates.get(currency.upper(), "N/A")
    return f"1 {currency} = {rate} TWD"

# 工具 3：新聞搜尋
def search_news(topic: str) -> str:
    """搜尋新聞"""
    return f"關於「{topic}」的最新新聞..."

# 建立工具列表
tools = [
    Tool(
        name="天氣查詢",
        func=get_weather_info,
        description="查詢指定城市的天氣資訊。輸入：城市名稱"
    ),
    Tool(
        name="匯率查詢",
        func=get_exchange_rate,
        description="查詢外幣對台幣的匯率。輸入：貨幣代碼（USD、JPY、EUR）"
    ),
    Tool(
        name="新聞搜尋",
        func=search_news,
        description="搜尋最新新聞。輸入：搜尋主題"
    )
]
```

## 🤖 LangChain Agents

### 建立基本 Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 初始化模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 建立提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位樂於助人的AI助理。使用繁體中文回答。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 建立 agent
agent = create_openai_functions_agent(llm, tools, prompt)

# 建立 executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 執行
result = agent_executor.invoke({
    "input": "台北的天氣如何？另外美元對台幣的匯率是多少？"
})

print(result["output"])
```

### 帶有記憶的 Agent

```python
from langchain.memory import ConversationBufferMemory

# 建立記憶
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 更新提示模板
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一位樂於助人的AI助理。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 建立帶記憶的 agent
agent_with_memory = create_openai_functions_agent(
    llm, tools, prompt_with_history
)

agent_executor_with_memory = AgentExecutor(
    agent=agent_with_memory,
    tools=tools,
    memory=memory,
    verbose=True
)

# 多輪對話
print("對話 1：")
response1 = agent_executor_with_memory.invoke({
    "input": "台北的天氣如何？"
})
print(response1["output"])

print("\n對話 2：")
response2 = agent_executor_with_memory.invoke({
    "input": "那台中呢？"  # agent 會記得在問天氣
})
print(response2["output"])
```

## 💡 實戰專案：多功能助理

```python
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import json
from datetime import datetime

class MultiToolAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.tools = self.create_tools()
        self.agent_executor = self.create_agent()

    def create_tools(self):
        """建立所有工具"""

        # 工具 1：行事曆
        def add_event(event_details: str) -> str:
            """新增行事曆事件"""
            # 模擬新增事件
            return f"✅ 已新增事件：{event_details}"

        # 工具 2：提醒設定
        def set_reminder(reminder: str) -> str:
            """設定提醒"""
            return f"⏰ 已設定提醒：{reminder}"

        # 工具 3：計算機
        def calculate(expression: str) -> str:
            """執行計算"""
            try:
                result = eval(expression)
                return f"計算結果：{expression} = {result}"
            except:
                return "計算錯誤"

        # 工具 4：單位轉換
        def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
            """單位轉換"""
            conversions = {
                ("km", "mile"): 0.621371,
                ("kg", "lb"): 2.20462,
                ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32
            }

            key = (from_unit.lower(), to_unit.lower())
            if key in conversions:
                factor = conversions[key]
                if callable(factor):
                    result = factor(value)
                else:
                    result = value * factor
                return f"{value} {from_unit} = {result:.2f} {to_unit}"
            return "不支援此單位轉換"

        # 工具 5：資訊搜尋
        def search_info(query: str) -> str:
            """搜尋資訊"""
            return f"關於「{query}」的搜尋結果..."

        return [
            Tool(
                name="行事曆",
                func=add_event,
                description="新增行事曆事件。輸入：事件詳情"
            ),
            Tool(
                name="提醒",
                func=set_reminder,
                description="設定提醒事項。輸入：提醒內容"
            ),
            Tool(
                name="計算機",
                func=calculate,
                description="執行數學計算。輸入：數學表達式"
            ),
            Tool(
                name="單位轉換",
                func=convert_unit,
                description="轉換單位（km/mile, kg/lb, celsius/fahrenheit）。"
                           "輸入格式：值 原單位 目標單位（例如：100 km mile）"
            ),
            Tool(
                name="資訊搜尋",
                func=search_info,
                description="搜尋各類資訊。輸入：搜尋關鍵字"
            )
        ]

    def create_agent(self):
        """建立 agent"""
        memory = ConversationBufferWindowMemory(
            k=5,  # 保留最近 5 輪對話
            memory_key="chat_history",
            return_messages=True
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位全能的AI助理，可以幫助使用者：
            - 管理行事曆和提醒
            - 執行計算和單位轉換
            - 搜尋資訊

            請用繁體中文回答，並保持友善專業的態度。
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def chat(self, user_input: str):
        """對話"""
        result = self.agent_executor.invoke({"input": user_input})
        return result["output"]

    def interactive(self):
        """互動模式"""
        print("=" * 60)
        print("多功能助理已啟動！")
        print("我可以幫你：管理行事曆、設定提醒、計算、轉換單位、搜尋資訊")
        print("輸入 'quit' 結束對話")
        print("=" * 60)

        while True:
            user_input = input("\n👤 您：")

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再見！")
                break

            if not user_input.strip():
                continue

            response = self.chat(user_input)
            print(f"\n🤖 助理：{response}")

# 使用範例
if __name__ == "__main__":
    # assistant = MultiToolAssistant()
    # assistant.interactive()

    # 或單次對話
    # response = assistant.chat("幫我計算 1234 * 5678，然後轉換 100 公里是多少英里")
    # print(response)
    pass
```

## ✅ 最佳實踐

### 1. 工具設計原則
- 功能單一明確
- 提供清晰的描述
- 參數命名要直觀
- 錯誤處理要完善

### 2. Agent 優化
- 使用適當的溫度（0 for tools）
- 限制最大迭代次數
- 提供清晰的系統提示
- 實作錯誤恢復機制

### 3. 安全性考量
- 驗證工具輸入
- 限制工具權限
- 記錄所有操作
- 實作審核機制

## 📚 延伸學習

- **LangGraph**：建立更複雜的 agent 工作流程
- **Multi-Agent Systems**：多個 agent 協作
- **Custom Agent Types**：自訂 agent 類型

---

**課程連結**：[DeepLearning.ai - Functions, Tools and Agents](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)

**完成日期**：2025-01-17
