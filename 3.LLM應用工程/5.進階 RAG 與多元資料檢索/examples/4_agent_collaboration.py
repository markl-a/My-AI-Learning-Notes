"""
LLM + RAG + Agents 協作系統

這個模組展示如何將 LLM、RAG 和 Agent 結合起來，創建智能的問答和任務執行系統。

核心概念：
1. Agent 架構：ReAct (Reasoning + Acting) 模式
2. 工具整合：RAG、搜尋、計算等工具
3. 多 Agent 協作：專業化分工
4. 任務規劃與執行

使用場景：
- 複雜問答系統
- 研究助手
- 數據分析助手
- 自動化工作流程
"""

import os
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class TaskType(Enum):
    """任務類型"""
    INFORMATION_RETRIEVAL = "information_retrieval"
    DATA_ANALYSIS = "data_analysis"
    CALCULATION = "calculation"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"


@dataclass
class AgentAction:
    """Agent 動作記錄"""
    tool: str
    input: str
    output: str
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResult:
    """Agent 執行結果"""
    query: str
    answer: str
    actions: List[AgentAction]
    total_steps: int
    task_type: TaskType


class RAGTool(BaseTool):
    """RAG 檢索工具"""

    name: str = "rag_search"
    description: str = """
    使用這個工具來搜尋知識庫中的相關資訊。
    輸入應該是一個明確的問題或查詢。
    這個工具適合回答事實性問題和查找特定資訊。
    """

    vector_store: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        """執行 RAG 檢索"""
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            if not docs:
                return "未找到相關資訊。"

            # 格式化結果
            result = "找到以下相關資訊：\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"[資料 {i}]\n{doc.page_content}\n\n"

            return result
        except Exception as e:
            return f"檢索錯誤: {str(e)}"

    async def _arun(self, query: str) -> str:
        """異步執行（暫不實現）"""
        return self._run(query)


class CalculatorTool(BaseTool):
    """計算器工具"""

    name: str = "calculator"
    description: str = """
    使用這個工具進行數學計算。
    輸入應該是一個數學表達式，例如: "2 + 2" 或 "10 * 5 / 2"
    支援基本運算: +, -, *, /, **, ()
    """

    def _run(self, expression: str) -> str:
        """執行計算"""
        try:
            # 清理輸入
            expression = expression.strip()

            # 安全檢查：只允許數字和基本運算符
            allowed_chars = set("0123456789+-*/().() ")
            if not all(c in allowed_chars for c in expression):
                return "錯誤：表達式包含不允許的字符"

            # 計算
            result = eval(expression)
            return f"計算結果: {result}"

        except ZeroDivisionError:
            return "錯誤：除以零"
        except Exception as e:
            return f"計算錯誤: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """異步執行"""
        return self._run(expression)


class ReasoningTool(BaseTool):
    """推理工具"""

    name: str = "reasoning"
    description: str = """
    使用這個工具進行邏輯推理和分析。
    適合處理需要多步推理的複雜問題。
    輸入應該是需要推理的問題或假設。
    """

    llm: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, problem: str) -> str:
        """執行推理"""
        reasoning_prompt = ChatPromptTemplate.from_template("""
請對以下問題進行深入的邏輯推理和分析：

問題：{problem}

請：
1. 分解問題
2. 列出關鍵假設
3. 逐步推理
4. 得出結論

推理過程：
""")

        messages = reasoning_prompt.format_messages(problem=problem)
        response = self.llm.invoke(messages)
        return response.content

    async def _arun(self, problem: str) -> str:
        """異步執行"""
        return self._run(problem)


class DataAnalysisTool(BaseTool):
    """數據分析工具"""

    name: str = "data_analysis"
    description: str = """
    使用這個工具分析數據模式、趨勢和統計資訊。
    輸入應該是需要分析的數據或數據相關的問題。
    可以進行基本的統計分析和模式識別。
    """

    llm: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, data_query: str) -> str:
        """執行數據分析"""
        analysis_prompt = ChatPromptTemplate.from_template("""
作為數據分析師，請分析以下數據或問題：

{data_query}

請提供：
1. 數據摘要
2. 關鍵發現
3. 趨勢分析
4. 建議

分析結果：
""")

        messages = analysis_prompt.format_messages(data_query=data_query)
        response = self.llm.invoke(messages)
        return response.content

    async def _arun(self, data_query: str) -> str:
        """異步執行"""
        return self._run(data_query)


class RAGAgent:
    """
    RAG Agent - 整合 RAG 和 Agent 架構

    使用 ReAct (Reasoning + Acting) 模式：
    1. Thought: 思考下一步做什麼
    2. Action: 選擇並執行工具
    3. Observation: 觀察執行結果
    4. 重複直到得到最終答案
    """

    def __init__(
        self,
        vector_store_path: str = "./chroma_db_agent",
        llm_model: str = "gpt-3.5-turbo"
    ):
        """初始化 Agent"""
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.tools = []
        self.agent_executor = None

    def ingest_documents(self, documents: List[str]):
        """攝取文檔到知識庫"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = [Document(page_content=doc) for doc in documents]
        splits = text_splitter.split_documents(docs)

        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )

        print(f"✓ 知識庫創建完成: {len(splits)} 個文檔塊")

    def setup_tools(self):
        """設置 Agent 可用的工具"""
        if not self.vector_store:
            raise ValueError("請先使用 ingest_documents() 創建知識庫")

        self.tools = [
            RAGTool(vector_store=self.vector_store),
            CalculatorTool(),
            ReasoningTool(llm=self.llm),
            DataAnalysisTool(llm=self.llm)
        ]

        print(f"✓ 已設置 {len(self.tools)} 個工具")

    def create_agent(self):
        """創建 ReAct Agent"""
        # 創建 Agent 提示
        react_prompt = ChatPromptTemplate.from_template("""
你是一個智能助手，可以使用各種工具來回答問題和完成任務。

可用工具：
{tools}

工具描述：
{tool_names}

使用以下格式：

Question: 需要回答的問題
Thought: 思考應該做什麼
Action: 要使用的工具名稱，必須是 [{tool_names}] 之一
Action Input: 工具的輸入
Observation: 工具的輸出結果
... (可以重複 Thought/Action/Observation 多次)
Thought: 我現在知道最終答案了
Final Answer: 對原始問題的最終答案

開始！

Question: {input}
Thought: {agent_scratchpad}
""")

        # 創建 ReAct Agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )

        # 創建 Agent 執行器
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

        print("✓ Agent 創建完成")

    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        執行查詢

        Args:
            question: 使用者問題
            verbose: 是否輸出詳細過程

        Returns:
            執行結果
        """
        if not self.agent_executor:
            raise ValueError("請先使用 create_agent() 創建 Agent")

        try:
            result = self.agent_executor.invoke({"input": question})

            return {
                "question": question,
                "answer": result.get("output", "無法生成答案"),
                "success": True
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"執行失敗: {str(e)}",
                "success": False
            }


class MultiAgentSystem:
    """
    多 Agent 協作系統

    不同的 Agent 負責不同類型的任務：
    - Research Agent: 研究和資訊檢索
    - Analysis Agent: 數據分析
    - Synthesis Agent: 綜合和總結
    """

    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """初始化多 Agent 系統"""
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.agents = {}

    def create_research_agent(self, vector_store) -> RAGAgent:
        """創建研究 Agent"""
        agent = RAGAgent()
        agent.vector_store = vector_store
        agent.llm = self.llm
        agent.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        return agent

    def classify_task(self, query: str) -> TaskType:
        """
        分類任務類型

        Args:
            query: 使用者查詢

        Returns:
            任務類型
        """
        classify_prompt = ChatPromptTemplate.from_template("""
分析以下查詢並決定任務類型：

查詢：{query}

任務類型：
- information_retrieval: 需要檢索和查找資訊
- data_analysis: 需要分析數據或模式
- calculation: 需要數學計算
- reasoning: 需要邏輯推理
- synthesis: 需要綜合多個資訊來源

只輸出任務類型，不要解釋。

任務類型：
""")

        messages = classify_prompt.format_messages(query=query)
        response = self.llm.invoke(messages)

        try:
            task_type_str = response.content.strip().lower()
            return TaskType(task_type_str)
        except ValueError:
            return TaskType.INFORMATION_RETRIEVAL  # 默認

    def route_to_agent(self, query: str, task_type: TaskType) -> str:
        """
        路由到適當的 Agent

        Args:
            query: 查詢
            task_type: 任務類型

        Returns:
            執行結果
        """
        # 這裡簡化實現，實際應該有多個專門的 Agent
        if task_type == TaskType.REASONING:
            return self.execute_reasoning(query)
        elif task_type == TaskType.CALCULATION:
            return self.execute_calculation(query)
        else:
            return "需要相應的 Agent 實現"

    def execute_reasoning(self, query: str) -> str:
        """執行推理任務"""
        reasoning_prompt = ChatPromptTemplate.from_template("""
請對以下問題進行深入分析和推理：

{query}

請提供詳細的推理過程和結論。

回答：
""")

        messages = reasoning_prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        return response.content

    def execute_calculation(self, query: str) -> str:
        """執行計算任務"""
        calc_tool = CalculatorTool()

        # 提取數學表達式
        extract_prompt = ChatPromptTemplate.from_template("""
從以下查詢中提取數學表達式：

{query}

只輸出數學表達式，不要解釋。

表達式：
""")

        messages = extract_prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        expression = response.content.strip()

        # 執行計算
        result = calc_tool._run(expression)
        return result


def main():
    """示例程式"""
    print("=" * 80)
    print("LLM + RAG + Agents 協作系統示範")
    print("=" * 80)
    print()

    # 準備知識庫文檔
    documents = [
        """
        Python 是一種高級編程語言，由 Guido van Rossum 在 1991 年創建。
        Python 的設計哲學強調代碼的可讀性，使用空格縮進來定義代碼塊。
        Python 支援多種編程範式，包括物件導向、函數式和程序式編程。
        """,
        """
        機器學習是人工智能的一個分支，專注於讓計算機從數據中學習。
        常見的機器學習算法包括線性回歸、決策樹、隨機森林和神經網絡。
        機器學習可分為監督學習、非監督學習和強化學習三大類。
        """,
        """
        深度學習是機器學習的一個子領域，使用多層神經網絡來學習數據表示。
        卷積神經網絡（CNN）特別適合處理圖像數據。
        循環神經網絡（RNN）適合處理序列數據，如文本和時間序列。
        Transformer 架構在 2017 年提出後，徹底改變了自然語言處理領域。
        """,
        """
        RAG（檢索增強生成）結合了檢索系統和生成模型的優勢。
        RAG 系統先從知識庫檢索相關文檔，然後基於這些文檔生成答案。
        這種方法可以有效減少模型幻覺，提供更準確和可靠的回答。
        """,
        """
        Agent 是能夠感知環境、做出決策並採取行動的智能體。
        ReAct 模式結合了推理（Reasoning）和行動（Acting），讓 Agent 能夠
        透過思考-行動-觀察的循環來解決複雜問題。
        多 Agent 系統可以讓不同的 Agent 協作完成複雜任務。
        """
    ]

    # 測試 1: RAG Agent
    print("測試 1: RAG Agent")
    print("=" * 80)

    rag_agent = RAGAgent(vector_store_path="./chroma_agent_demo")
    print("\n1.1 攝取文檔...")
    rag_agent.ingest_documents(documents)

    print("\n1.2 設置工具...")
    rag_agent.setup_tools()

    print("\n1.3 創建 Agent...")
    rag_agent.create_agent()

    # 測試查詢
    test_queries = [
        "Python 是誰創建的？它的特點是什麼？",
        "機器學習有哪些主要類型？",
        "什麼是 RAG？它有什麼優勢？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n查詢 {i}: {query}")
        print("-" * 80)

        result = rag_agent.query(query, verbose=True)

        print(f"\n答案: {result['answer']}\n")

    # 測試 2: 多 Agent 系統
    print("\n" + "=" * 80)
    print("測試 2: 多 Agent 系統")
    print("=" * 80)

    multi_agent = MultiAgentSystem()

    test_cases = [
        ("什麼是深度學習？", "information_retrieval"),
        ("計算 15 * 23 + 45", "calculation"),
        ("為什麼 RAG 能減少模型幻覺？", "reasoning")
    ]

    for query, expected_type in test_cases:
        print(f"\n查詢: {query}")
        print("-" * 40)

        # 分類任務
        task_type = multi_agent.classify_task(query)
        print(f"任務類型: {task_type.value}")
        print(f"預期類型: {expected_type}")

        # 執行任務
        result = multi_agent.route_to_agent(query, task_type)
        print(f"結果: {result[:200]}...")
        print()

    print("=" * 80)
    print("示範完成！")
    print("=" * 80)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤: 請設置 OPENAI_API_KEY 環境變數")
    else:
        main()
