"""
基礎 MCP 伺服器範例
展示如何使用 Python SDK 創建一個簡單的 MCP 伺服器

安裝依賴:
    pip install mcp

運行方式:
    python 01_basic_mcp_server.py
"""

import asyncio
from typing import Any
from datetime import datetime

# MCP SDK 導入
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK 未安裝，此範例展示概念結構")


# ============ 工具定義 ============

def get_current_time() -> str:
    """獲取當前時間"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    安全的數學計算器
    僅支持基礎運算
    """
    # 安全檢查：只允許數字和基本運算符
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "錯誤：表達式包含不允許的字符"

    try:
        # 使用 eval 前進行嚴格驗證
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"計算錯誤: {str(e)}"


def search_notes(query: str, limit: int = 5) -> list:
    """
    搜尋筆記（模擬實現）
    實際應用中應連接真實的搜尋後端
    """
    # 模擬搜尋結果
    mock_notes = [
        {"id": 1, "title": "Transformer 架構解析", "content": "Self-attention 機制..."},
        {"id": 2, "title": "RAG 系統設計", "content": "檢索增強生成..."},
        {"id": 3, "title": "LLM 微調技巧", "content": "LoRA, QLoRA..."},
        {"id": 4, "title": "Prompt Engineering", "content": "Few-shot, CoT..."},
        {"id": 5, "title": "Agent 架構", "content": "ReAct, Tool Use..."},
    ]

    # 簡單的關鍵字匹配
    results = [
        note for note in mock_notes
        if query.lower() in note["title"].lower() or query.lower() in note["content"].lower()
    ]

    return results[:limit]


# ============ MCP 伺服器實現 ============

if MCP_AVAILABLE:
    # 創建 MCP 伺服器實例
    server = Server("ai-learning-notes-server")

    # 定義可用工具
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出所有可用工具"""
        return [
            Tool(
                name="get_time",
                description="獲取當前系統時間",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="calculate",
                description="執行數學計算，支持加減乘除和括號",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "數學表達式，例如: 2+3*4"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            Tool(
                name="search_notes",
                description="搜尋 AI 學習筆記",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜尋關鍵字"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回結果數量上限",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    # 處理工具調用
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """執行工具調用"""
        if name == "get_time":
            result = get_current_time()
        elif name == "calculate":
            result = calculate(arguments.get("expression", ""))
        elif name == "search_notes":
            notes = search_notes(
                arguments.get("query", ""),
                arguments.get("limit", 5)
            )
            result = "\n".join([f"- {n['title']}: {n['content']}" for n in notes])
            if not result:
                result = "未找到匹配的筆記"
        else:
            result = f"未知工具: {name}"

        return [TextContent(type="text", text=result)]

    # 定義資源
    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """列出可用資源"""
        return [
            Resource(
                uri="notes://learning-path",
                name="學習路線圖",
                description="AI 學習路線完整指南",
                mimeType="text/markdown"
            ),
            Resource(
                uri="notes://glossary",
                name="術語表",
                description="AI/ML 常用術語解釋",
                mimeType="text/markdown"
            )
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """讀取資源內容"""
        if uri == "notes://learning-path":
            return """
# AI 學習路線圖

## 階段一：基礎
- Python 程式設計
- 數學基礎（線性代數、微積分、概率統計）

## 階段二：機器學習
- Scikit-learn
- 特徵工程
- 模型評估

## 階段三：深度學習
- PyTorch / TensorFlow
- CNN, RNN, Transformer

## 階段四：LLM 應用
- Prompt Engineering
- RAG 系統
- Agent 開發
"""
        elif uri == "notes://glossary":
            return """
# AI/ML 術語表

- **LLM**: Large Language Model，大型語言模型
- **RAG**: Retrieval-Augmented Generation，檢索增強生成
- **LoRA**: Low-Rank Adaptation，低秩適應微調
- **MCP**: Model Context Protocol，模型上下文協議
- **Agent**: 能夠自主使用工具完成任務的 AI 系統
"""
        else:
            return f"資源不存在: {uri}"


# ============ 主程序 ============

async def main():
    """主函數"""
    if not MCP_AVAILABLE:
        print("\n" + "="*50)
        print("MCP SDK 概念演示")
        print("="*50)

        print("\n工具列表:")
        print("1. get_time - 獲取當前時間")
        print("2. calculate - 數學計算")
        print("3. search_notes - 搜尋筆記")

        print("\n範例調用:")
        print(f"get_time() = {get_current_time()}")
        print(f"calculate('2+3*4') = {calculate('2+3*4')}")
        print(f"search_notes('RAG') = {search_notes('RAG')}")

        print("\n要實際運行 MCP 伺服器，請安裝:")
        print("  pip install mcp")
        return

    # 啟動 MCP 伺服器（透過 stdio）
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
