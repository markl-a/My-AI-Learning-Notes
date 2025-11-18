"""
LangChain Agent 範例
展示不同類型的 AI Agent 及其應用
"""

import os
import sys
from pathlib import Path

# 添加父目錄到路徑以導入 utils
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_environment, get_llm, setup_langsmith

from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    create_tool_calling_agent,
)
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper


# ============================================================================
# 自訂工具定義
# ============================================================================

def calculator(expression: str) -> str:
    """
    計算數學表達式

    Args:
        expression: 數學表達式，例如 "2 + 2"

    Returns:
        計算結果
    """
    try:
        # 安全地評估數學表達式
        # 注意：在生產環境中應使用更安全的方法
        result = eval(expression, {"__builtins__": {}}, {})
        return f"計算結果: {result}"
    except Exception as e:
        return f"計算錯誤: {str(e)}"


def get_current_weather(location: str) -> str:
    """
    取得指定地點的天氣（模擬）

    Args:
        location: 地點名稱

    Returns:
        天氣資訊
    """
    # 這裡是模擬的天氣資料
    # 實際應用中應該呼叫真實的天氣 API
    weather_data = {
        "台北": "晴天，溫度 28°C",
        "台中": "多雲，溫度 26°C",
        "高雄": "陰天，溫度 30°C",
        "台南": "晴天，溫度 29°C",
    }

    return weather_data.get(
        location,
        f"無法取得 {location} 的天氣資訊（模擬資料）"
    )


def get_word_length(word: str) -> str:
    """
    計算字串長度

    Args:
        word: 輸入字串

    Returns:
        字串長度
    """
    return f"'{word}' 的長度是 {len(word)} 個字元"


# ============================================================================
# ReAct Agent 範例
# ============================================================================

def demo_react_agent():
    """
    示範 ReAct Agent

    ReAct (Reasoning + Acting) 是一種讓 LLM 交替進行推理和行動的方法
    """
    print("=" * 80)
    print("示範 1: ReAct Agent")
    print("=" * 80)

    load_environment()
    setup_langsmith()

    # 建立工具
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="用於進行數學計算。輸入應該是數學表達式，例如 '2 + 2' 或 '10 * 5'"
        ),
        Tool(
            name="Weather",
            func=get_current_weather,
            description="取得指定地點的天氣資訊。輸入應該是地點名稱，例如 '台北' 或 '高雄'"
        ),
        Tool(
            name="WordLength",
            func=get_word_length,
            description="計算字串的長度。輸入應該是一個字串"
        ),
    ]

    # 建立 LLM
    llm = get_llm(model="gpt-3.5-turbo", temperature=0)

    # ReAct prompt 模板
    react_prompt = PromptTemplate.from_template("""
請回答以下問題。你有以下工具可以使用：

{tools}

使用以下格式：

Question: 需要回答的問題
Thought: 你應該思考要做什麼
Action: 要採取的行動，應該是 [{tool_names}] 其中之一
Action Input: 行動的輸入
Observation: 行動的結果
... (這個 Thought/Action/Action Input/Observation 可以重複 N 次)
Thought: 我現在知道最終答案了
Final Answer: 對原始問題的最終答案

開始！

Question: {input}
Thought: {agent_scratchpad}
""")

    # 建立 agent
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 測試問題
    questions = [
        "台北的天氣如何？",
        "計算 123 乘以 456 等於多少？",
        "'LangChain' 這個字有幾個字元？",
        "如果台北現在是 28 度，而高雄是 30 度，兩地溫差是多少？（需要先查詢天氣，然後計算）",
    ]

    for question in questions:
        print(f"\n{'=' * 80}")
        print(f"問題: {question}")
        print(f"{'=' * 80}")

        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n最終答案: {result['output']}")
        except Exception as e:
            print(f"\n錯誤: {str(e)}")


# ============================================================================
# Tool-Calling Agent 範例
# ============================================================================

def demo_tool_calling_agent():
    """
    示範 Tool-Calling Agent

    使用 OpenAI 的 function calling 功能
    """
    print("\n" + "=" * 80)
    print("示範 2: Tool-Calling Agent（使用 OpenAI Function Calling）")
    print("=" * 80)

    load_environment()

    # 建立工具
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="用於進行數學計算。輸入應該是數學表達式"
        ),
        Tool(
            name="Weather",
            func=get_current_weather,
            description="取得指定地點的天氣資訊"
        ),
    ]

    # 建立支援 function calling 的 LLM
    llm = get_llm(model="gpt-3.5-turbo", temperature=0)

    # Tool-calling prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個有用的助手。
請使用提供的工具來回答問題。
請用繁體中文回答。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 建立 agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 測試問題
    questions = [
        "高雄今天天氣如何？",
        "計算 999 加 1 等於多少？",
        "台中和台南的天氣分別如何？",
    ]

    for question in questions:
        print(f"\n{'=' * 80}")
        print(f"問題: {question}")
        print(f"{'=' * 80}")

        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n最終答案: {result['output']}")
        except Exception as e:
            print(f"\n錯誤: {str(e)}")


# ============================================================================
# 使用外部工具的 Agent 範例
# ============================================================================

def demo_agent_with_search():
    """
    示範使用搜尋工具的 Agent
    """
    print("\n" + "=" * 80)
    print("示範 3: 使用網路搜尋的 Agent")
    print("=" * 80)

    load_environment()

    # 建立搜尋工具
    search = DuckDuckGoSearchRun()

    # 建立 Wikipedia 工具
    wikipedia = WikipediaAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="用於搜尋網路上的最新資訊。輸入應該是搜尋查詢"
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="用於查詢 Wikipedia 上的資訊。輸入應該是要查詢的主題"
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="用於進行數學計算"
        ),
    ]

    # 建立 LLM
    llm = get_llm(model="gpt-3.5-turbo", temperature=0)

    # Tool-calling prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個研究助手。
請使用提供的工具來回答問題。
如果需要最新資訊，請使用 Search 工具。
如果需要百科知識，請使用 Wikipedia 工具。
請用繁體中文回答。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 建立 agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5  # 限制最大迭代次數
    )

    # 測試問題
    questions = [
        "LangChain 是什麼？",
        "最新的 GPT 模型是什麼？",
        "台灣的首都在哪裡？",
    ]

    for question in questions:
        print(f"\n{'=' * 80}")
        print(f"問題: {question}")
        print(f"{'=' * 80}")

        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n最終答案: {result['output']}")
        except Exception as e:
            print(f"\n錯誤: {str(e)}")


# ============================================================================
# 對話式 Agent 範例
# ============================================================================

class ConversationalAgent:
    """對話式 Agent，支援記憶功能"""

    def __init__(self):
        load_environment()

        self.tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="用於進行數學計算"
            ),
            Tool(
                name="Weather",
                func=get_current_weather,
                description="取得指定地點的天氣資訊"
            ),
        ]

        self.llm = get_llm(model="gpt-3.5-turbo", temperature=0)

        # 建立 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一個友善的助手。
請記住之前的對話內容，並在回答時考慮上下文。
使用提供的工具來回答問題。
請用繁體中文回答。"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 建立 agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

        self.chat_history = []

    def chat(self, message: str) -> str:
        """
        進行對話

        Args:
            message: 使用者訊息

        Returns:
            AI 回應
        """
        result = self.agent_executor.invoke({
            "input": message,
            "chat_history": self.chat_history
        })

        # 更新對話歷史
        self.chat_history.extend([
            HumanMessage(content=message),
            AIMessage(content=result["output"])
        ])

        return result["output"]

    def clear_history(self):
        """清除對話歷史"""
        self.chat_history = []
        print("✓ 對話歷史已清除")


def demo_conversational_agent():
    """示範對話式 Agent"""
    print("\n" + "=" * 80)
    print("示範 4: 對話式 Agent（包含記憶）")
    print("=" * 80)

    agent = ConversationalAgent()

    # 對話序列
    conversations = [
        "你好！",
        "台北今天的天氣如何？",
        "那邊的溫度是幾度？",  # 參考前一個問題
        "幫我計算 100 除以 4",
        "再乘以 3 呢？",  # 參考前一個計算
    ]

    for message in conversations:
        print(f"\n{'=' * 80}")
        print(f"使用者: {message}")
        print(f"{'=' * 80}")

        response = agent.chat(message)
        print(f"\nAI: {response}")


# ============================================================================
# 主程式
# ============================================================================

if __name__ == "__main__":
    try:
        # 執行所有示範
        demo_react_agent()
        demo_tool_calling_agent()
        demo_agent_with_search()
        demo_conversational_agent()

        print("\n" + "=" * 80)
        print("✓ 所有 Agent 示範執行完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
