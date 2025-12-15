"""
AutoGen 對話式 AI Agent
支持多Agent對話、代碼執行、工具使用
適合複雜的協作任務
"""

import autogen
from typing import List, Dict, Optional, Callable
import os


class AutoGenSystem:
    """AutoGen 系統封裝"""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        """
        初始化 AutoGen 系統

        Args:
            model: LLM 模型
            api_key: OpenAI API Key
        """
        self.config_list = [
            {
                "model": model,
                "api_key": api_key or os.getenv("OPENAI_API_KEY")
            }
        ]

        self.llm_config = {
            "config_list": self.config_list,
            "temperature": 0.7,
            "timeout": 120
        }

        print(f"AutoGen 系統初始化完成")
        print(f"模型: {model}")

    def create_assistant_user_pair(
        self,
        assistant_system_message: str = "你是一個helpful的AI助手。"
    ):
        """
        創建助手-用戶對

        Args:
            assistant_system_message: 助手的系統消息

        Returns:
            (assistant, user_proxy) 元組
        """
        # 創建助手 Agent
        assistant = autogen.AssistantAgent(
            name="assistant",
            system_message=assistant_system_message,
            llm_config=self.llm_config
        )

        # 創建用戶代理
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",  # 不需要人工輸入
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False  # 設為 True 使用 Docker（更安全）
            }
        )

        return assistant, user_proxy

    def create_code_assistant(self):
        """創建代碼助手"""
        assistant = autogen.AssistantAgent(
            name="code_assistant",
            system_message="""你是一個專業的程序員助手。
當需要編寫代碼時，請提供完整、可執行的Python代碼。
代碼應該包含必要的註釋和錯誤處理。
完成任務後，請回覆 TERMINATE。""",
            llm_config=self.llm_config
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False
            }
        )

        return assistant, user_proxy

    def create_group_chat(
        self,
        agents: List[autogen.Agent],
        max_round: int = 10
    ):
        """
        創建群組聊天

        Args:
            agents: Agent 列表
            max_round: 最大輪數

        Returns:
            GroupChatManager
        """
        groupchat = autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )

        return manager

    def create_research_team(self):
        """創建研究團隊"""

        # 研究員
        researcher = autogen.AssistantAgent(
            name="researcher",
            system_message="""你是一位資深研究員。
你的職責是：
1. 搜集相關信息
2. 分析數據和趨勢
3. 提供研究發現
當完成研究後，請說明 "研究完成"。""",
            llm_config=self.llm_config
        )

        # 批評者
        critic = autogen.AssistantAgent(
            name="critic",
            system_message="""你是一位嚴謹的批評者。
你的職責是：
1. 審查研究發現
2. 指出潛在問題
3. 提供建設性建議
如果研究充分，回覆 "批准"。""",
            llm_config=self.llm_config
        )

        # 撰稿人
        writer = autogen.AssistantAgent(
            name="writer",
            system_message="""你是一位專業撰稿人。
你的職責是：
1. 整合研究結果
2. 撰寫清晰的報告
3. 確保邏輯連貫
完成後回覆 TERMINATE。""",
            llm_config=self.llm_config
        )

        # 用戶代理
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False
        )

        return [user_proxy, researcher, critic, writer]


def example_basic_conversation():
    """示例 1: 基本對話"""
    print("=== 示例 1: 基本對話 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")
    assistant, user_proxy = system.create_assistant_user_pair()

    # 開始對話
    user_proxy.initiate_chat(
        assistant,
        message="解釋一下什麼是遞迴，並給一個Python例子。"
    )


def example_code_execution():
    """示例 2: 代碼執行"""
    print("\n=== 示例 2: 代碼生成與執行 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")
    assistant, user_proxy = system.create_code_assistant()

    # 請求生成並執行代碼
    user_proxy.initiate_chat(
        assistant,
        message="""寫一個Python函數來計算斐波那契數列的第n項。
然後測試 n=10 的情況。
執行代碼並顯示結果。"""
    )


def example_math_problem():
    """示例 3: 數學問題求解"""
    print("\n=== 示例 3: 數學問題求解 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")
    assistant, user_proxy = system.create_code_assistant()

    user_proxy.initiate_chat(
        assistant,
        message="""解決這個數學問題：
有一個班級40名學生，其中60%是女生。
如果再加入5名男生，男生占比是多少？

請寫Python代碼計算並驗證答案。"""
    )


def example_data_analysis():
    """示例 4: 數據分析"""
    print("\n=== 示例 4: 數據分析 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")
    assistant, user_proxy = system.create_code_assistant()

    user_proxy.initiate_chat(
        assistant,
        message="""創建一個簡單的數據分析示例：
1. 生成隨機數據（100個點）
2. 計算均值、中位數、標準差
3. 使用matplotlib繪製直方圖
4. 保存圖表為 analysis.png

執行代碼並報告結果。"""
    )


def example_group_chat():
    """示例 5: 群組對話"""
    print("\n=== 示例 5: 多Agent群組對話 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")
    agents = system.create_research_team()
    manager = system.create_group_chat(agents, max_round=12)

    # 開始群組對話
    user = agents[0]  # user_proxy
    user.initiate_chat(
        manager,
        message="研究主題：大型語言模型在2024年的主要進展。請團隊協作完成這個研究。"
    )


def example_two_agent_debate():
    """示例 6: 雙Agent辯論"""
    print("\n=== 示例 6: Agent辯論 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")

    # 正方
    pro_agent = autogen.AssistantAgent(
        name="pro_debater",
        system_message="""你是辯論的正方。
你支持：AI將會創造更多工作機會。
請提供有力的論據和例證。""",
        llm_config=system.llm_config
    )

    # 反方
    con_agent = autogen.AssistantAgent(
        name="con_debater",
        system_message="""你是辯論的反方。
你認為：AI將會取代大量工作。
請提供有力的論據和例證。
在3輪辯論後，請回覆 TERMINATE。""",
        llm_config=system.llm_config
    )

    # 開始辯論
    pro_agent.initiate_chat(
        con_agent,
        message="我認為AI技術將會創造更多新的工作機會，理由如下：...",
        max_turns=6  # 限制輪數
    )


def example_custom_function():
    """示例 7: 自定義函數調用"""
    print("\n=== 示例 7: 自定義函數 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")

    # 定義自定義函數
    def get_stock_price(symbol: str) -> str:
        """獲取股票價格（模擬）"""
        prices = {
            "AAPL": "$180.50",
            "GOOGL": "$140.25",
            "MSFT": "$375.80"
        }
        return prices.get(symbol, "未找到")

    # 註冊函數
    autogen.register_function(
        get_stock_price,
        caller=system.create_assistant_user_pair()[1],  # user_proxy
        executor=system.create_assistant_user_pair()[1],
        name="get_stock_price",
        description="獲取股票價格"
    )

    assistant, user_proxy = system.create_assistant_user_pair(
        assistant_system_message="""你是一個股票分析助手。
你可以使用 get_stock_price 函數查詢股票價格。"""
    )

    user_proxy.initiate_chat(
        assistant,
        message="幫我查詢 AAPL 和 GOOGL 的股價，並告訴我哪個更高。"
    )


def example_iterative_refinement():
    """示例 8: 迭代改進"""
    print("\n=== 示例 8: 迭代改進代碼 ===\n")

    system = AutoGenSystem(model="gpt-4o-mini")

    # 代碼撰寫者
    coder = autogen.AssistantAgent(
        name="coder",
        system_message="你是一個Python程序員，專注於編寫代碼。",
        llm_config=system.llm_config
    )

    # 代碼審查者
    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="""你是一個代碼審查專家。
檢查代碼的：
1. 正確性
2. 效率
3. 可讀性
4. 錯誤處理
提供改進建議。如果代碼足夠好，回覆 APPROVE。""",
        llm_config=system.llm_config
    )

    # 用戶
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "APPROVE" in x.get("content", ""),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )

    # 創建群組
    groupchat = autogen.GroupChat(
        agents=[user_proxy, coder, reviewer],
        messages=[],
        max_round=10
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=system.llm_config
    )

    user_proxy.initiate_chat(
        manager,
        message="寫一個函數來檢查字符串是否是回文。請審查者檢查並提供反饋。"
    )


if __name__ == "__main__":
    print("AutoGen 對話式 AI Agent 示例")
    print("=" * 60)
    print()

    # 檢查環境變數
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 請設置 OPENAI_API_KEY 環境變數")
        print("export OPENAI_API_KEY='your-api-key'\n")

    try:
        # 運行示例（需要 API key）
        example_basic_conversation()
        # example_code_execution()
        # example_math_problem()
        # example_group_chat()
        # example_two_agent_debate()

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n注意:")
        print("1. 需要安裝 AutoGen: pip install pyautogen")
        print("2. 需要設置 OPENAI_API_KEY")
        print("3. 代碼執行功能需要安全的環境")
        print("4. 建議在生產環境使用 Docker")

    print("\nAutoGen 特性:")
    print("✓ 對話式 Agent 交互")
    print("✓ 自動代碼生成與執行")
    print("✓ 支持多Agent協作")
    print("✓ 靈活的終止條件")
    print("✓ 內建錯誤恢復")
    print("✓ 支持人機協作")

    print("\n適用場景:")
    print("1. 代碼生成與調試")
    print("2. 數據分析任務")
    print("3. 複雜問題求解")
    print("4. 多輪對話應用")
    print("5. Agent 協作研究")

    print("\n安全建議:")
    print("1. 生產環境使用 Docker 執行代碼")
    print("2. 限制代碼執行權限")
    print("3. 監控 API 使用量")
    print("4. 設置合理的終止條件")
    print("5. 驗證生成的代碼")

    print("\n與其他框架對比:")
    print("- LangGraph: 更適合複雜工作流")
    print("- CrewAI: 更適合角色扮演協作")
    print("- AutoGen: 更適合對話和代碼任務")
