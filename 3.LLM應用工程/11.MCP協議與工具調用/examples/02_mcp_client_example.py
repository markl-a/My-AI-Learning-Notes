"""
MCP 客戶端範例
展示如何連接和使用 MCP 伺服器

這個範例展示了如何:
1. 連接到 MCP 伺服器
2. 列出可用工具
3. 調用工具
4. 處理響應
"""

import asyncio
import json
from typing import Any, Optional
from dataclasses import dataclass, field


# ============ MCP 客戶端模擬實現 ============

@dataclass
class MCPTool:
    """MCP 工具定義"""
    name: str
    description: str
    input_schema: dict


@dataclass
class MCPResource:
    """MCP 資源定義"""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"


@dataclass
class MCPClient:
    """
    MCP 客戶端模擬
    實際使用時應該使用 mcp.client 模組
    """
    server_name: str
    tools: list = field(default_factory=list)
    resources: list = field(default_factory=list)
    connected: bool = False

    async def connect(self) -> bool:
        """連接到 MCP 伺服器"""
        print(f"正在連接到 MCP 伺服器: {self.server_name}")
        await asyncio.sleep(0.1)  # 模擬連接延遲
        self.connected = True
        print("✅ 連接成功")
        return True

    async def disconnect(self) -> None:
        """斷開連接"""
        self.connected = False
        print("已斷開連接")

    async def list_tools(self) -> list[MCPTool]:
        """列出可用工具"""
        if not self.connected:
            raise ConnectionError("未連接到伺服器")

        # 模擬從伺服器獲取工具列表
        self.tools = [
            MCPTool(
                name="get_weather",
                description="獲取指定城市的天氣資訊",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名稱"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            ),
            MCPTool(
                name="search_web",
                description="搜尋網路資訊",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="read_file",
                description="讀取檔案內容",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            )
        ]
        return self.tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        調用工具

        Args:
            name: 工具名稱
            arguments: 工具參數

        Returns:
            工具執行結果
        """
        if not self.connected:
            raise ConnectionError("未連接到伺服器")

        print(f"📞 調用工具: {name}")
        print(f"   參數: {json.dumps(arguments, ensure_ascii=False)}")

        # 模擬工具執行
        await asyncio.sleep(0.2)

        # 模擬響應
        if name == "get_weather":
            result = {
                "status": "success",
                "data": {
                    "city": arguments.get("city", "Unknown"),
                    "temperature": 25,
                    "unit": arguments.get("unit", "celsius"),
                    "condition": "晴天",
                    "humidity": 60
                }
            }
        elif name == "search_web":
            result = {
                "status": "success",
                "data": {
                    "results": [
                        {"title": f"搜尋結果 1 - {arguments.get('query')}", "url": "https://example.com/1"},
                        {"title": f"搜尋結果 2 - {arguments.get('query')}", "url": "https://example.com/2"},
                    ],
                    "total": 100
                }
            }
        elif name == "read_file":
            result = {
                "status": "success",
                "data": {
                    "content": f"模擬檔案內容: {arguments.get('path')}",
                    "size": 1024
                }
            }
        else:
            result = {
                "status": "error",
                "error": f"未知工具: {name}"
            }

        print(f"   結果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return result

    async def list_resources(self) -> list[MCPResource]:
        """列出可用資源"""
        if not self.connected:
            raise ConnectionError("未連接到伺服器")

        self.resources = [
            MCPResource(
                uri="file:///docs/readme.md",
                name="README",
                description="專案說明文件",
                mime_type="text/markdown"
            ),
            MCPResource(
                uri="db://users",
                name="用戶資料庫",
                description="用戶資料表",
                mime_type="application/json"
            )
        ]
        return self.resources

    async def read_resource(self, uri: str) -> str:
        """讀取資源內容"""
        if not self.connected:
            raise ConnectionError("未連接到伺服器")

        print(f"📖 讀取資源: {uri}")
        return f"資源內容 ({uri}): 這是模擬的資源內容..."


# ============ 使用範例 ============

async def example_basic_usage():
    """基礎使用範例"""
    print("\n" + "="*60)
    print("MCP 客戶端基礎使用範例")
    print("="*60)

    client = MCPClient(server_name="example-server")

    try:
        # 連接到伺服器
        await client.connect()

        # 列出工具
        print("\n📋 可用工具:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # 調用工具
        print("\n🔧 工具調用示範:")

        # 獲取天氣
        weather = await client.call_tool("get_weather", {"city": "Taipei", "unit": "celsius"})

        # 搜尋網頁
        search = await client.call_tool("search_web", {"query": "MCP protocol", "limit": 5})

        # 列出資源
        print("\n📚 可用資源:")
        resources = await client.list_resources()
        for resource in resources:
            print(f"  - {resource.name} ({resource.uri})")

    finally:
        await client.disconnect()


async def example_with_llm_integration():
    """
    與 LLM 整合的範例
    展示如何將 MCP 工具轉換為 LLM 可用的格式
    """
    print("\n" + "="*60)
    print("MCP + LLM 整合範例")
    print("="*60)

    client = MCPClient(server_name="llm-integration-server")
    await client.connect()

    try:
        # 獲取工具並轉換為 OpenAI 格式
        tools = await client.list_tools()

        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })

        print("\n🤖 轉換為 OpenAI 工具格式:")
        print(json.dumps(openai_tools, ensure_ascii=False, indent=2))

        # 模擬 LLM 決定使用工具
        print("\n💭 模擬 LLM 決策過程:")
        print("   用戶: '台北今天天氣如何？'")
        print("   LLM: 我需要使用 get_weather 工具...")

        # 執行工具調用
        result = await client.call_tool("get_weather", {"city": "Taipei"})

        # 模擬 LLM 整合結果
        print("\n🎯 LLM 最終回覆:")
        if result["status"] == "success":
            data = result["data"]
            print(f"   台北今天天氣{data['condition']}，"
                  f"氣溫 {data['temperature']}°C，"
                  f"濕度 {data['humidity']}%。")

    finally:
        await client.disconnect()


async def example_error_handling():
    """錯誤處理範例"""
    print("\n" + "="*60)
    print("MCP 錯誤處理範例")
    print("="*60)

    client = MCPClient(server_name="error-demo-server")

    # 嘗試在未連接時調用工具
    print("\n❌ 測試未連接錯誤:")
    try:
        await client.call_tool("test", {})
    except ConnectionError as e:
        print(f"   捕獲到錯誤: {e}")

    await client.connect()

    # 調用不存在的工具
    print("\n❌ 測試工具不存在錯誤:")
    result = await client.call_tool("nonexistent_tool", {})
    if result["status"] == "error":
        print(f"   錯誤訊息: {result['error']}")

    await client.disconnect()


# ============ 主程序 ============

async def main():
    """運行所有範例"""
    await example_basic_usage()
    await example_with_llm_integration()
    await example_error_handling()

    print("\n" + "="*60)
    print("✅ 所有範例執行完成")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
