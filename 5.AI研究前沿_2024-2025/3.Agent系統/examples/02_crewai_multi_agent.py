"""
CrewAI 多Agent協作系統
實現多個AI Agent的協同工作
適合複雜任務的分工與協作
"""

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from typing import List, Optional
import os


class ResearchCrew:
    """研究團隊 - 使用 CrewAI 協調多個 Agent"""

    def __init__(self, model: str = "gpt-4"):
        """
        初始化研究團隊

        Args:
            model: 使用的 LLM 模型
        """
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        print(f"初始化 CrewAI 研究團隊")
        print(f"使用模型: {model}")

    def create_research_crew(self) -> Crew:
        """創建研究團隊"""

        # 1. 定義 Agents
        researcher = Agent(
            role='資深研究員',
            goal='進行深入的研究，收集準確的信息',
            backstory="""你是一位經驗豐富的研究員，擅長從各種來源
            收集和驗證信息。你總是追求準確性和全面性。""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        analyst = Agent(
            role='數據分析師',
            goal='分析研究數據，提供深入見解',
            backstory="""你是一位專業的數據分析師，能夠從複雜的數據中
            提取有價值的洞察。你擅長發現模式和趨勢。""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        writer = Agent(
            role='技術撰稿人',
            goal='將研究成果撰寫成清晰易懂的報告',
            backstory="""你是一位優秀的技術撰稿人，能夠將複雜的技術
            內容轉化為清晰、引人入勝的文章。""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # 2. 定義 Tasks
        research_task = Task(
            description="""研究主題：{topic}

            任務要求：
            1. 收集關於該主題的最新信息
            2. 驗證信息的準確性
            3. 整理關鍵發現
            4. 提供可靠的來源

            輸出格式：詳細的研究筆記，包含關鍵發現和來源。""",
            agent=researcher,
            expected_output="詳細的研究報告，包含關鍵發現和參考來源"
        )

        analysis_task = Task(
            description="""分析研究員提供的數據。

            任務要求：
            1. 識別主要趨勢和模式
            2. 分析優勢和挑戰
            3. 提供數據驅動的見解
            4. 提出建議

            輸出格式：結構化的分析報告。""",
            agent=analyst,
            expected_output="數據分析報告，包含趨勢、見解和建議"
        )

        writing_task = Task(
            description="""撰寫最終報告。

            任務要求：
            1. 整合研究和分析結果
            2. 使用清晰、專業的語言
            3. 結構化組織內容
            4. 包含執行摘要

            輸出格式：完整的研究報告，適合向管理層展示。""",
            agent=writer,
            expected_output="完整的專業報告，包含執行摘要和詳細分析"
        )

        # 3. 創建 Crew
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential,  # 順序執行
            verbose=True
        )

        return crew

    def create_content_team(self) -> Crew:
        """創建內容創作團隊"""

        # Agent: 內容策劃
        planner = Agent(
            role='內容策劃師',
            goal='規劃引人入勝的內容策略',
            backstory="""你是一位創意內容策劃師，擅長理解受眾需求
            並設計吸引人的內容計劃。""",
            verbose=True,
            llm=self.llm
        )

        # Agent: 內容撰寫
        writer = Agent(
            role='內容撰寫者',
            goal='創作高質量、吸引人的內容',
            backstory="""你是一位才華橫溢的撰稿人，能夠創作出
            既有信息量又引人入勝的內容。""",
            verbose=True,
            llm=self.llm
        )

        # Agent: 編輯審查
        editor = Agent(
            role='編輯',
            goal='確保內容質量和一致性',
            backstory="""你是一位嚴謹的編輯，對細節有敏銳的洞察力，
            能夠提升內容的質量和可讀性。""",
            verbose=True,
            llm=self.llm
        )

        # Tasks
        planning_task = Task(
            description="""為主題 {topic} 制定內容計劃。

            包括：
            1. 目標受眾分析
            2. 關鍵信息點
            3. 內容結構建議
            4. 預期效果""",
            agent=planner,
            expected_output="詳細的內容策劃方案"
        )

        writing_task = Task(
            description="""根據內容計劃撰寫文章。

            要求：
            1. 遵循計劃的結構
            2. 保持專業但易讀的風格
            3. 包含具體例子
            4. 長度約 1000 字""",
            agent=writer,
            expected_output="完整的文章草稿"
        )

        editing_task = Task(
            description="""審查和改進文章。

            檢查：
            1. 語法和拼寫
            2. 邏輯連貫性
            3. 事實準確性
            4. 整體可讀性

            提供改進建議並產生最終版本。""",
            agent=editor,
            expected_output="審查後的最終文章"
        )

        crew = Crew(
            agents=[planner, writer, editor],
            tasks=[planning_task, writing_task, editing_task],
            process=Process.sequential,
            verbose=True
        )

        return crew

    def create_product_team(self) -> Crew:
        """創建產品開發團隊"""

        # Product Manager
        pm = Agent(
            role='產品經理',
            goal='定義產品需求和優先級',
            backstory="""你是經驗豐富的產品經理，擅長理解用戶需求
            並將其轉化為清晰的產品規格。""",
            verbose=True,
            allow_delegation=True,  # 允許委派任務
            llm=self.llm
        )

        # Engineer
        engineer = Agent(
            role='軟體工程師',
            goal='設計技術方案並評估可行性',
            backstory="""你是資深軟體工程師，擅長系統設計和
            技術架構決策。""",
            verbose=True,
            llm=self.llm
        )

        # QA
        qa = Agent(
            role='質量保證工程師',
            goal='確保產品質量和可靠性',
            backstory="""你是細心的QA工程師，擅長發現潛在問題
            並提供測試策略。""",
            verbose=True,
            llm=self.llm
        )

        # Tasks
        requirements_task = Task(
            description="""定義產品需求：{topic}

            輸出：
            1. 用戶故事
            2. 功能需求列表
            3. 優先級排序
            4. 成功指標""",
            agent=pm,
            expected_output="產品需求文檔（PRD）"
        )

        technical_design_task = Task(
            description="""設計技術方案。

            包括：
            1. 系統架構
            2. 技術棧選擇
            3. 數據模型
            4. API 設計
            5. 風險評估""",
            agent=engineer,
            expected_output="技術設計文檔"
        )

        qa_plan_task = Task(
            description="""制定測試計劃。

            包括：
            1. 測試策略
            2. 測試用例
            3. 質量指標
            4. 風險緩解措施""",
            agent=qa,
            expected_output="測試計劃文檔"
        )

        crew = Crew(
            agents=[pm, engineer, qa],
            tasks=[requirements_task, technical_design_task, qa_plan_task],
            process=Process.sequential,
            verbose=True
        )

        return crew


def example_research_crew():
    """示例 1: 研究團隊"""
    print("=== 示例 1: CrewAI 研究團隊 ===\n")

    research_crew = ResearchCrew(model="gpt-3.5-turbo")
    crew = research_crew.create_research_crew()

    # 執行研究任務
    result = crew.kickoff(inputs={
        "topic": "2024年大型語言模型的最新發展"
    })

    print("\n研究報告:")
    print(result)


def example_content_team():
    """示例 2: 內容創作團隊"""
    print("=== 示例 2: 內容創作團隊 ===\n")

    content_crew = ResearchCrew(model="gpt-3.5-turbo")
    crew = content_crew.create_content_team()

    result = crew.kickoff(inputs={
        "topic": "如何使用 AI 提升工作效率"
    })

    print("\n創作的內容:")
    print(result)


def example_product_team():
    """示例 3: 產品開發團隊"""
    print("=== 示例 3: 產品開發團隊 ===\n")

    product_crew = ResearchCrew(model="gpt-3.5-turbo")
    crew = product_crew.create_product_team()

    result = crew.kickoff(inputs={
        "topic": "AI 驅動的客戶服務聊天機器人"
    })

    print("\n產品規劃:")
    print(result)


def example_with_tools():
    """示例 4: 使用工具的 Agent"""
    print("=== 示例 4: 帶工具的 Agent ===\n")

    # 定義工具
    def search_web(query: str) -> str:
        """搜索網絡（模擬）"""
        return f"搜索結果：{query} 的相關信息..."

    def calculate(expression: str) -> str:
        """計算器"""
        try:
            result = eval(expression)
            return f"計算結果: {result}"
        except:
            return "計算錯誤"

    search_tool = Tool(
        name="網絡搜索",
        func=search_web,
        description="在網絡上搜索信息"
    )

    calc_tool = Tool(
        name="計算器",
        func=calculate,
        description="執行數學計算"
    )

    # 創建帶工具的 Agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    research_agent = Agent(
        role='研究助手',
        goal='回答問題並提供準確信息',
        backstory='你是一個helpful的研究助手，可以使用工具查找和計算信息。',
        tools=[search_tool, calc_tool],
        verbose=True,
        llm=llm
    )

    task = Task(
        description="回答：世界上最高的建築是什麼？它的高度是多少？",
        agent=research_agent,
        expected_output="建築名稱和準確高度"
    )

    crew = Crew(
        agents=[research_agent],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    print(f"\n結果: {result}")


if __name__ == "__main__":
    print("CrewAI 多Agent協作系統示例")
    print("=" * 60)
    print()

    # 檢查環境變數
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 請設置 OPENAI_API_KEY 環境變數")
        print("export OPENAI_API_KEY='your-api-key'\n")

    try:
        # 運行示例（需要 API key）
        example_research_crew()
        # example_content_team()
        # example_product_team()
        # example_with_tools()

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n注意:")
        print("1. 需要安裝 CrewAI: pip install crewai")
        print("2. 需要設置 OPENAI_API_KEY")
        print("3. 確保網絡連接正常")

    print("\nCrewAI 特性:")
    print("✓ 角色導向的 Agent 設計")
    print("✓ 靈活的任務編排")
    print("✓ 支持順序和並行執行")
    print("✓ Agent 間可以委派任務")
    print("✓ 內建記憶和上下文管理")
    print("✓ 支持自定義工具")

    print("\n適用場景:")
    print("1. 複雜研究任務")
    print("2. 內容創作流程")
    print("3. 產品開發協作")
    print("4. 數據分析項目")
    print("5. 客戶服務自動化")

    print("\n最佳實踐:")
    print("1. 清晰定義每個 Agent 的角色和目標")
    print("2. 合理設計任務流程")
    print("3. 使用詳細的 backstory 增強 Agent 能力")
    print("4. 必要時啟用 allow_delegation")
    print("5. 監控執行過程（verbose=True）")
