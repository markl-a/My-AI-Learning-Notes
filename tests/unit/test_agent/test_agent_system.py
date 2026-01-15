"""
Agent 系統單元測試
測試 Agent 架構、工具調用、記憶機制等功能
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import json


# ============ 測試用的 Agent 類實現 ============

class ToolCallStatus(Enum):
    """工具調用狀態"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class Tool:
    """工具定義"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable = None

    def execute(self, **kwargs) -> Any:
        """執行工具"""
        if self.function:
            return self.function(**kwargs)
        return f"Mock result for {self.name}"


@dataclass
class ToolCall:
    """工具調用記錄"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    status: ToolCallStatus = ToolCallStatus.PENDING


@dataclass
class Message:
    """對話訊息"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class AgentMemory:
    """Agent 記憶"""
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 100

    def add_message(self, message: Message):
        """添加訊息"""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # 保留系統訊息，刪除最舊的對話
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            other_msgs = other_msgs[-(self.max_messages - len(system_msgs)):]
            self.messages = system_msgs + other_msgs

    def get_context(self, n: int = 10) -> List[Message]:
        """獲取最近 n 條訊息"""
        return self.messages[-n:]

    def clear(self):
        """清空記憶"""
        self.messages = []


class Agent:
    """基礎 Agent 類"""

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        tools: List[Tool] = None,
        max_iterations: int = 10
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = AgentMemory()
        self.max_iterations = max_iterations

        if system_prompt:
            self.memory.add_message(Message(role="system", content=system_prompt))

    def add_tool(self, tool: Tool):
        """添加工具"""
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """移除工具"""
        if tool_name in self.tools:
            del self.tools[tool_name]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        """執行工具調用"""
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)

        if tool_name not in self.tools:
            tool_call.status = ToolCallStatus.FAILED
            tool_call.result = f"Tool '{tool_name}' not found"
            return tool_call

        try:
            tool = self.tools[tool_name]
            result = tool.execute(**arguments)
            tool_call.result = result
            tool_call.status = ToolCallStatus.SUCCESS
        except Exception as e:
            tool_call.status = ToolCallStatus.FAILED
            tool_call.result = str(e)

        return tool_call

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """獲取可用工具列表（OpenAI 格式）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]


# ============ 測試類 ============

class TestTool:
    """Tool 類測試"""

    def test_tool_creation(self):
        """測試工具創建"""
        tool = Tool(
            name="calculator",
            description="進行數學計算",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        )

        assert tool.name == "calculator"
        assert tool.description == "進行數學計算"

    def test_tool_execution_mock(self):
        """測試工具執行（Mock）"""
        tool = Tool(
            name="test_tool",
            description="測試工具",
            parameters={}
        )

        result = tool.execute()
        assert "Mock result" in result

    def test_tool_execution_with_function(self):
        """測試工具執行（真實函數）"""
        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="加法",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                }
            },
            function=add
        )

        result = tool.execute(a=2, b=3)
        assert result == 5


class TestToolCall:
    """ToolCall 測試"""

    def test_tool_call_creation(self):
        """測試工具調用創建"""
        call = ToolCall(
            tool_name="search",
            arguments={"query": "test"}
        )

        assert call.tool_name == "search"
        assert call.status == ToolCallStatus.PENDING
        assert call.result is None

    def test_tool_call_status_update(self):
        """測試狀態更新"""
        call = ToolCall(tool_name="test", arguments={})
        call.status = ToolCallStatus.SUCCESS
        call.result = "完成"

        assert call.status == ToolCallStatus.SUCCESS
        assert call.result == "完成"


class TestMessage:
    """Message 測試"""

    def test_message_creation(self):
        """測試訊息創建"""
        msg = Message(role="user", content="你好")

        assert msg.role == "user"
        assert msg.content == "你好"
        assert msg.tool_calls == []

    def test_message_with_tool_calls(self):
        """測試帶工具調用的訊息"""
        tool_call = ToolCall(tool_name="search", arguments={"q": "AI"})
        msg = Message(
            role="assistant",
            content="讓我搜尋一下",
            tool_calls=[tool_call]
        )

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].tool_name == "search"


class TestAgentMemory:
    """AgentMemory 測試"""

    def test_memory_add_message(self):
        """測試添加訊息"""
        memory = AgentMemory()
        memory.add_message(Message(role="user", content="測試"))

        assert len(memory.messages) == 1

    def test_memory_max_messages(self):
        """測試最大訊息限制"""
        memory = AgentMemory(max_messages=5)

        for i in range(10):
            memory.add_message(Message(role="user", content=f"訊息 {i}"))

        assert len(memory.messages) <= 5

    def test_memory_preserve_system_message(self):
        """測試保留系統訊息"""
        memory = AgentMemory(max_messages=5)
        memory.add_message(Message(role="system", content="你是助手"))

        for i in range(10):
            memory.add_message(Message(role="user", content=f"訊息 {i}"))

        # 系統訊息應該被保留
        system_msgs = [m for m in memory.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "你是助手"

    def test_memory_get_context(self):
        """測試獲取上下文"""
        memory = AgentMemory()
        for i in range(20):
            memory.add_message(Message(role="user", content=f"訊息 {i}"))

        context = memory.get_context(n=5)
        assert len(context) == 5
        assert context[-1].content == "訊息 19"

    def test_memory_clear(self):
        """測試清空記憶"""
        memory = AgentMemory()
        memory.add_message(Message(role="user", content="測試"))
        memory.clear()

        assert len(memory.messages) == 0


class TestAgent:
    """Agent 類測試"""

    def test_agent_creation(self):
        """測試 Agent 創建"""
        agent = Agent(name="TestAgent", system_prompt="你是一個測試助手")

        assert agent.name == "TestAgent"
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "system"

    def test_agent_add_tool(self):
        """測試添加工具"""
        agent = Agent(name="TestAgent")
        tool = Tool(name="calculator", description="計算器", parameters={})

        agent.add_tool(tool)

        assert "calculator" in agent.tools

    def test_agent_remove_tool(self):
        """測試移除工具"""
        tool = Tool(name="calculator", description="計算器", parameters={})
        agent = Agent(name="TestAgent", tools=[tool])

        agent.remove_tool("calculator")

        assert "calculator" not in agent.tools

    def test_agent_execute_tool_success(self):
        """測試成功執行工具"""
        def multiply(x: int, y: int) -> int:
            return x * y

        tool = Tool(
            name="multiply",
            description="乘法",
            parameters={},
            function=multiply
        )
        agent = Agent(name="TestAgent", tools=[tool])

        result = agent.execute_tool("multiply", {"x": 3, "y": 4})

        assert result.status == ToolCallStatus.SUCCESS
        assert result.result == 12

    def test_agent_execute_tool_not_found(self):
        """測試執行不存在的工具"""
        agent = Agent(name="TestAgent")

        result = agent.execute_tool("nonexistent", {})

        assert result.status == ToolCallStatus.FAILED
        assert "not found" in result.result

    def test_agent_execute_tool_error(self):
        """測試工具執行錯誤"""
        def error_func():
            raise ValueError("測試錯誤")

        tool = Tool(name="error_tool", description="錯誤工具", parameters={}, function=error_func)
        agent = Agent(name="TestAgent", tools=[tool])

        result = agent.execute_tool("error_tool", {})

        assert result.status == ToolCallStatus.FAILED
        assert "測試錯誤" in result.result

    def test_agent_get_available_tools(self):
        """測試獲取可用工具列表"""
        tools = [
            Tool(name="tool1", description="工具1", parameters={"type": "object"}),
            Tool(name="tool2", description="工具2", parameters={"type": "object"})
        ]
        agent = Agent(name="TestAgent", tools=tools)

        available = agent.get_available_tools()

        assert len(available) == 2
        assert all(t["type"] == "function" for t in available)
        tool_names = [t["function"]["name"] for t in available]
        assert "tool1" in tool_names
        assert "tool2" in tool_names


class TestReActPattern:
    """ReAct 模式測試"""

    def test_thought_action_observation_cycle(self):
        """測試 Thought-Action-Observation 循環"""
        # 模擬 ReAct 步驟
        steps = []

        # Thought
        steps.append({"type": "thought", "content": "我需要搜尋最新的 AI 新聞"})

        # Action
        steps.append({"type": "action", "tool": "search", "input": "latest AI news"})

        # Observation
        steps.append({"type": "observation", "result": "找到 10 篇相關文章"})

        # Final Answer
        steps.append({"type": "answer", "content": "根據搜尋結果..."})

        assert len(steps) == 4
        assert steps[0]["type"] == "thought"
        assert steps[1]["type"] == "action"
        assert steps[2]["type"] == "observation"
        assert steps[3]["type"] == "answer"

    def test_max_iterations(self):
        """測試最大迭代次數限制"""
        agent = Agent(name="TestAgent", max_iterations=5)

        assert agent.max_iterations == 5


class TestFunctionCalling:
    """Function Calling 測試"""

    def test_function_schema_generation(self):
        """測試函數 Schema 生成"""
        schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "獲取天氣資訊",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名稱"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "location" in schema["function"]["parameters"]["properties"]

    def test_parse_function_call(self):
        """測試解析函數調用"""
        # 模擬 OpenAI 的 function_call 響應
        function_call = {
            "name": "get_weather",
            "arguments": '{"location": "Taipei", "unit": "celsius"}'
        }

        name = function_call["name"]
        args = json.loads(function_call["arguments"])

        assert name == "get_weather"
        assert args["location"] == "Taipei"
        assert args["unit"] == "celsius"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
