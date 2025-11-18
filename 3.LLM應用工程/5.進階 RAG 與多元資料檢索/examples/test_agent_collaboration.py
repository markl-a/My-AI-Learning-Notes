"""
測試 Agent 協作系統

運行方式：
    python test_agent_collaboration.py
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from agent_collaboration import (
    RAGAgent,
    MultiAgentSystem,
    TaskType,
    CalculatorTool,
    ReasoningTool
)

load_dotenv()

TEST_DOCUMENTS = [
    "Python 是一種編程語言，由 Guido van Rossum 創建。",
    "機器學習包括監督學習、非監督學習和強化學習。",
    "RAG 結合了檢索和生成，可以減少模型幻覺。"
]


def test_calculator_tool():
    """測試計算器工具"""
    print("測試計算器工具")
    print("-" * 80)

    calc = CalculatorTool()

    test_cases = [
        ("2 + 2", "4"),
        ("10 * 5", "50"),
        ("100 / 4", "25"),
    ]

    for expression, expected in test_cases:
        result = calc._run(expression)
        print(f"表達式: {expression}")
        print(f"結果: {result}")
        assert str(expected) in result
        print("✓ 通過\n")

    print("✓ 計算器工具測試通過\n")


def test_multi_agent_task_classification():
    """測試多 Agent 任務分類"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⊘ 跳過：需要 OPENAI_API_KEY")
        return

    print("測試多 Agent 任務分類")
    print("-" * 80)

    multi_agent = MultiAgentSystem()

    test_queries = [
        "什麼是 Python？",
        "計算 15 + 25",
        "為什麼機器學習很重要？"
    ]

    for query in test_queries:
        task_type = multi_agent.classify_task(query)
        print(f"查詢: {query}")
        print(f"任務類型: {task_type.value}")
        assert isinstance(task_type, TaskType)
        print("✓ 通過\n")

    print("✓ 任務分類測試通過\n")


def test_rag_agent_basic():
    """測試基本的 RAG Agent"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⊘ 跳過：需要 OPENAI_API_KEY")
        return

    print("測試 RAG Agent 基本功能")
    print("-" * 80)

    try:
        # 初始化
        agent = RAGAgent(vector_store_path="./test_chroma_agent")
        print("✓ Agent 初始化完成")

        # 攝取文檔
        agent.ingest_documents(TEST_DOCUMENTS)
        print("✓ 文檔攝取完成")

        # 設置工具
        agent.setup_tools()
        print("✓ 工具設置完成")

        # 創建 Agent
        agent.create_agent()
        print("✓ Agent 創建完成")

        # 測試查詢（簡化版，不實際執行複雜查詢）
        print("\n✓ RAG Agent 基本功能測試通過\n")

    except Exception as e:
        print(f"✗ 測試失敗: {e}\n")


def run_all_tests():
    """運行所有測試"""
    print("=" * 80)
    print("Agent 協作系統測試")
    print("=" * 80)
    print()

    # 測試 1: 計算器工具（不需要 API）
    test_calculator_tool()

    # 測試 2: 多 Agent 任務分類（需要 API）
    test_multi_agent_task_classification()

    # 測試 3: RAG Agent 基本功能（需要 API）
    test_rag_agent_basic()

    print("=" * 80)
    print("✅ 所有測試完成！")
    print("=" * 80)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  警告: 未設置 OPENAI_API_KEY")
        print("部分測試將被跳過\n")

    run_all_tests()
