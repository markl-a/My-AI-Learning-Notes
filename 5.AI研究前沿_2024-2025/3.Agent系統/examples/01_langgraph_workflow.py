"""
LangGraph Agent 工作流
使用 LangGraph 構建可控的 Agent 工作流
支持狀態管理、條件路由、並行執行
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os


# 定義狀態
class AgentState(TypedDict):
    """Agent 狀態"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    iteration_count: int


class ResearchAgent:
    """研究型 Agent - 使用 LangGraph 編排工作流"""

    def __init__(self, model: str = "gpt-4"):
        """
        初始化 Agent

        Args:
            model: LLM 模型名稱
        """
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0
        )

        # 構建工作流圖
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """構建工作流圖"""
        # 創建狀態圖
        workflow = StateGraph(AgentState)

        # 添加節點
        workflow.add_node("planner", self.plan)
        workflow.add_node("researcher", self.research)
        workflow.add_node("analyzer", self.analyze)
        workflow.add_node("writer", self.write)

        # 設定入口點
        workflow.set_entry_point("planner")

        # 添加邊（定義執行流程）
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyzer")

        # 條件邊：根據分析結果決定下一步
        workflow.add_conditional_edges(
            "analyzer",
            self.should_continue,
            {
                "continue": "researcher",  # 繼續研究
                "write": "writer",          # 開始寫作
            }
        )

        workflow.add_edge("writer", END)

        # 編譯工作流
        return workflow.compile()

    def plan(self, state: AgentState) -> AgentState:
        """規劃步驟"""
        print("\n=== 規劃階段 ===")

        # 獲取用戶查詢
        user_query = state["messages"][-1].content

        # 使用 LLM 制定計劃
        plan_prompt = f"""你是一個研究助手。用戶的查詢是：

{user_query}

請制定一個研究計劃，列出需要調查的關鍵問題。"""

        response = self.llm.invoke([HumanMessage(content=plan_prompt)])

        print(f"研究計劃:\n{response.content}")

        # 更新狀態
        return {
            "messages": [AIMessage(content=f"計劃: {response.content}")],
            "next_step": "research",
            "iteration_count": 0
        }

    def research(self, state: AgentState) -> AgentState:
        """研究步驟"""
        print("\n=== 研究階段 ===")

        # 獲取當前計劃
        plan = state["messages"][-1].content

        # 模擬研究（實際應調用搜索工具、RAG等）
        research_prompt = f"""基於以下研究計劃，進行研究並收集信息：

{plan}

請提供研究發現。（這是第 {state['iteration_count'] + 1} 次研究）"""

        response = self.llm.invoke([HumanMessage(content=research_prompt)])

        print(f"研究發現:\n{response.content[:200]}...")

        # 更新狀態
        return {
            "messages": [AIMessage(content=f"研究發現: {response.content}")],
            "iteration_count": state["iteration_count"] + 1
        }

    def analyze(self, state: AgentState) -> AgentState:
        """分析步驟"""
        print("\n=== 分析階段 ===")

        # 獲取研究發現
        research_findings = state["messages"][-1].content

        # 分析是否需要更多研究
        analyze_prompt = f"""分析以下研究發現：

{research_findings}

判斷信息是否充分。如果信息充分，回答"充分"；如果需要更多研究，回答"需要更多研究"。"""

        response = self.llm.invoke([HumanMessage(content=analyze_prompt)])

        print(f"分析結果: {response.content}")

        # 更新狀態
        return {
            "messages": [AIMessage(content=f"分析: {response.content}")]
        }

    def write(self, state: AgentState) -> AgentState:
        """寫作步驟"""
        print("\n=== 寫作階段 ===")

        # 獲取所有研究發現
        all_research = "\n".join([
            msg.content for msg in state["messages"]
            if "研究發現" in msg.content
        ])

        # 撰寫最終報告
        write_prompt = f"""基於以下研究發現，撰寫一份完整的報告：

{all_research}

請提供結構化的報告。"""

        response = self.llm.invoke([HumanMessage(content=write_prompt)])

        print(f"最終報告:\n{response.content}")

        # 更新狀態
        return {
            "messages": [AIMessage(content=f"報告: {response.content}")]
        }

    def should_continue(self, state: AgentState) -> str:
        """決定是否繼續研究"""
        # 獲取分析結果
        analysis = state["messages"][-1].content

        # 檢查是否達到最大迭代次數
        if state["iteration_count"] >= 3:
            print("已達到最大研究次數，開始寫作")
            return "write"

        # 根據分析結果決定
        if "充分" in analysis or "足夠" in analysis:
            return "write"
        else:
            print("需要更多研究，繼續調查")
            return "continue"

    def run(self, query: str) -> str:
        """
        運行 Agent

        Args:
            query: 用戶查詢

        Returns:
            最終報告
        """
        print(f"\n{'='*60}")
        print(f"開始處理查詢: {query}")
        print(f"{'='*60}")

        # 初始化狀態
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next_step": "plan",
            "iteration_count": 0
        }

        # 執行工作流
        result = self.workflow.invoke(initial_state)

        # 返回最終報告
        final_message = result["messages"][-1].content
        return final_message


class SimpleAgent:
    """簡單的 LangGraph Agent 示例"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def create_simple_workflow(self) -> StateGraph:
        """創建簡單的工作流"""
        workflow = StateGraph(AgentState)

        # 添加節點
        workflow.add_node("greet", self.greet)
        workflow.add_node("process", self.process)
        workflow.add_node("respond", self.respond)

        # 設定流程
        workflow.set_entry_point("greet")
        workflow.add_edge("greet", "process")
        workflow.add_edge("process", "respond")
        workflow.add_edge("respond", END)

        return workflow.compile()

    def greet(self, state: AgentState) -> AgentState:
        """問候"""
        print("步驟 1: 問候")
        return {
            "messages": [AIMessage(content="你好！我收到了你的消息。")]
        }

    def process(self, state: AgentState) -> AgentState:
        """處理"""
        print("步驟 2: 處理")
        user_message = state["messages"][0].content
        response = self.llm.invoke([HumanMessage(content=f"總結：{user_message}")])
        return {
            "messages": [AIMessage(content=response.content)]
        }

    def respond(self, state: AgentState) -> AgentState:
        """回應"""
        print("步驟 3: 回應")
        return {
            "messages": [AIMessage(content="處理完成！")]
        }


def example_simple_workflow():
    """示例 1: 簡單工作流"""
    print("=== 示例 1: 簡單的 LangGraph 工作流 ===")

    agent = SimpleAgent()
    workflow = agent.create_simple_workflow()

    # 運行工作流
    result = workflow.invoke({
        "messages": [HumanMessage(content="介紹一下人工智能")],
        "next_step": "greet",
        "iteration_count": 0
    })

    print("\n工作流完成！")
    for msg in result["messages"]:
        print(f"- {msg.content}")


def example_research_agent():
    """示例 2: 研究型 Agent"""
    print("\n=== 示例 2: 研究型 Agent ===")

    # 注意：這個示例需要設定 OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("請設定 OPENAI_API_KEY 環境變數")
        return

    agent = ResearchAgent(model="gpt-4o-mini")  # 使用較便宜的模型

    # 運行查詢
    query = "解釋什麼是 Transformer 架構，以及它為什麼重要？"
    result = agent.run(query)

    print(f"\n最終結果:\n{result}")


def example_conditional_routing():
    """示例 3: 條件路由"""
    print("\n=== 示例 3: 條件路由 ===")

    class ConditionalState(TypedDict):
        value: int
        result: str

    def check_value(state: ConditionalState) -> ConditionalState:
        """檢查值"""
        print(f"檢查值: {state['value']}")
        return state

    def process_positive(state: ConditionalState) -> ConditionalState:
        """處理正數"""
        return {"result": f"{state['value']} 是正數"}

    def process_negative(state: ConditionalState) -> ConditionalState:
        """處理負數"""
        return {"result": f"{state['value']} 是負數"}

    def route_by_value(state: ConditionalState) -> str:
        """根據值路由"""
        return "positive" if state["value"] > 0 else "negative"

    # 構建工作流
    workflow = StateGraph(ConditionalState)
    workflow.add_node("check", check_value)
    workflow.add_node("positive", process_positive)
    workflow.add_node("negative", process_negative)

    workflow.set_entry_point("check")
    workflow.add_conditional_edges(
        "check",
        route_by_value,
        {
            "positive": "positive",
            "negative": "negative"
        }
    )
    workflow.add_edge("positive", END)
    workflow.add_edge("negative", END)

    app = workflow.compile()

    # 測試
    for value in [10, -5, 0]:
        result = app.invoke({"value": value, "result": ""})
        print(f"輸入: {value}, 結果: {result['result']}")


if __name__ == "__main__":
    print("LangGraph Agent 工作流示例")
    print("=" * 60)

    # 運行示例
    example_simple_workflow()

    # 運行其他示例（需要 API key）
    # example_research_agent()

    example_conditional_routing()

    print("\n所有示例完成！")
    print("\nLangGraph 優勢:")
    print("1. 可視化的工作流定義")
    print("2. 靈活的狀態管理")
    print("3. 支持條件路由和循環")
    print("4. 內建檢查點和持久化")
    print("5. 與 LangSmith 集成，便於調試")
