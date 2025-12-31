"""
基礎 ReAct Agent 範例（使用模擬工具）

這個範例展示 ReAct (Reasoning + Acting) 模式的核心概念。
使用模擬的 LLM 和工具，不需要 API 金鑰即可運行。
"""

from typing import Dict, Tuple, Optional
import re


class MockLLM:
    """模擬的 LLM，用於演示"""

    def __init__(self):
        self.knowledge_base = {
            "台灣最高的山": "玉山是台灣最高的山，海拔3,952公尺",
            "台灣人口": "台灣人口約2,300萬人（2024年）",
            "Python 用途": "Python 是一種高階程式語言，廣泛用於 Web 開發、數據分析、AI 等領域",
        }

    def generate(self, prompt: str) -> str:
        """
        模擬 LLM 生成回應

        在實際應用中，這裡會調用 OpenAI API
        """
        # 簡化的模擬邏輯
        if "思考下一步" in prompt or "Thought" in prompt:
            if "台灣最高的山" in prompt:
                return "Thought: 我需要搜尋台灣最高的山的資訊\nAction: Search[台灣最高的山]"
            elif "玉山" in prompt and "海拔" in prompt:
                return "Thought: 我已經找到答案了\nFinal Answer: 玉山，海拔3,952公尺"

        return "Thought: 我需要更多資訊\nAction: Search[未知]"


class ReActAgent:
    """
    基礎 ReAct Agent 實作

    展示 Reasoning + Acting 的核心循環
    """

    def __init__(self, max_iterations: int = 10):
        """
        初始化 Agent

        Args:
            max_iterations: 最大迭代次數（防止無限循環）
        """
        self.llm = MockLLM()
        self.max_iterations = max_iterations
        self.tools = self._init_tools()

    def _init_tools(self) -> Dict:
        """初始化工具"""
        return {
            "Search": self._search_tool,
            "Calculator": self._calculator_tool,
        }

    def _search_tool(self, query: str) -> str:
        """
        模擬搜尋工具

        Args:
            query: 搜尋查詢

        Returns:
            搜尋結果
        """
        # 模擬搜尋結果
        mock_results = {
            "台灣最高的山": "玉山，海拔 3,952 公尺，位於台灣中央山脈",
            "台灣人口": "約 2,300 萬人（2024年）",
            "Python 用途": "Python 是一種高階程式語言，用於 Web 開發、數據分析、AI 等"
        }

        return mock_results.get(query, f"找不到關於 '{query}' 的資訊")

    def _calculator_tool(self, expression: str) -> str:
        """
        計算器工具

        Args:
            expression: 數學表達式

        Returns:
            計算結果
        """
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

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析 LLM 輸出的行動

        Args:
            text: LLM 的輸出文本

        Returns:
            (工具名稱, 參數) 或 (None, None)
        """
        # 匹配格式: Action: ToolName[argument]
        pattern = r'Action:\s*(\w+)\[([^\]]+)\]'
        match = re.search(pattern, text)

        if match:
            tool_name = match.group(1)
            argument = match.group(2)
            return tool_name, argument

        return None, None

    def run(self, question: str) -> str:
        """
        執行 ReAct 循環

        Args:
            question: 用戶問題

        Returns:
            最終答案
        """
        print(f"\n問題：{question}\n")
        print("執行 ReAct 循環...\n")

        scratchpad = []  # 存儲推理歷史

        for i in range(self.max_iterations):
            # 構建提示
            prompt = self._create_prompt(question, scratchpad)

            # 調用 LLM
            response = self.llm.generate(prompt)
            scratchpad.append(response)

            print(f"步驟 {i+1}:")
            print(response)
            print()

            # 檢查是否完成
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[1].strip()
                return final_answer

            # 解析並執行行動
            tool_name, argument = self._parse_action(response)

            if tool_name and tool_name in self.tools:
                # 執行工具
                result = self.tools[tool_name](argument)
                observation = f"Observation: {result}"
                scratchpad.append(observation)
                print(observation)
                print()
            else:
                print(f"警告：無法解析行動或工具不存在")

        return "達到最大迭代次數，未能完成任務"

    def _create_prompt(self, question: str, scratchpad: list) -> str:
        """創建 ReAct 提示"""
        tools_desc = """
可用工具：
- Search[query]: 搜尋網路資訊
- Calculator[expression]: 執行數學計算
"""

        history = "\n".join(scratchpad) if scratchpad else ""

        prompt = f"""{tools_desc}

請使用以下格式回答問題：

Thought: 你的推理過程
Action: 工具名稱[參數]
Observation: 工具返回的結果
... (重複 Thought/Action/Observation)
Thought: 我現在知道最終答案了
Final Answer: 最終答案

問題: {question}

{history}

思考下一步："""

        return prompt


def main():
    """主函數"""
    # 創建 Agent
    agent = ReActAgent(max_iterations=5)

    # 測試問題
    questions = [
        "台灣最高的山是什麼？它的海拔是多少？",
        "計算 123 * 456",
        "台灣有多少人口？"
    ]

    for question in questions:
        print("=" * 60)
        answer = agent.run(question)
        print("=" * 60)
        print(f"答案：{answer}\n")


if __name__ == "__main__":
    main()
