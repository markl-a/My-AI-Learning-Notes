# Model Context Protocol (MCP) 協議與工具調用

> **最後更新**: 2025-12-14
> **狀態**: 2024-2025年AI工具調用的新標準

---

## 📋 目錄

1. [MCP概述](#1-mcp概述)
2. [核心架構](#2-核心架構)
3. [與傳統Function Calling的對比](#3-與傳統function-calling的對比)
4. [MCP SDK使用指南](#4-mcp-sdk使用指南)
5. [自訂MCP伺服器開發](#5-自訂mcp伺服器開發)
6. [企業級整合方案](#6-企業級整合方案)
7. [最佳實踐](#7-最佳實踐)
8. [實戰案例](#8-實戰案例)

---

## 1. MCP概述

### 1.1 什麼是MCP？

**Model Context Protocol (MCP)** 是由Anthropic於2024年11月推出的開放標準協議，旨在統一AI應用與外部工具、數據源之間的通信方式。

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP 生態系統架構                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌─────────┐     MCP協議      ┌─────────────────┐        │
│    │  LLM    │◄─────────────────►│   MCP Server    │        │
│    │ (Host)  │                   │                 │        │
│    └─────────┘                   └────────┬────────┘        │
│         │                                 │                 │
│         │                    ┌────────────┼────────────┐    │
│         │                    │            │            │    │
│         ▼                    ▼            ▼            ▼    │
│    ┌─────────┐          ┌────────┐  ┌────────┐  ┌────────┐ │
│    │  User   │          │ Files  │  │  APIs  │  │Database│ │
│    │Interface│          └────────┘  └────────┘  └────────┘ │
│    └─────────┘                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 MCP的核心價值

| 特性 | 傳統方式 | MCP方式 |
|------|---------|---------|
| **標準化** | 每個API各有不同格式 | 統一的JSON-RPC協議 |
| **可發現性** | 需手動配置工具清單 | 動態工具發現和註冊 |
| **安全性** | 分散的權限管理 | 統一的權限和審計 |
| **可擴展性** | N×M的集成複雜度 | N+M的線性複雜度 |
| **生態系統** | 碎片化 | 700+ MCP伺服器可復用 |

### 1.3 主要支持的AI系統

- ✅ **Claude Desktop** - 原生支持
- ✅ **Claude Code** - 完整MCP整合
- ✅ **Cursor IDE** - 內建支持
- ✅ **Windsurf** - 支持MCP
- ⏳ **OpenAI** - 計劃支持中
- ⏳ **Google Gemini** - 評估中

---

## 2. 核心架構

### 2.1 MCP組件模型

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP 協議層次                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 4: 應用層                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tools (工具) | Resources (資源) | Prompts (提示)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  Layer 3: 能力層                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Capabilities | Permissions | Sampling               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  Layer 2: 傳輸層                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  stdio | HTTP+SSE | WebSocket                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  Layer 1: 序列化層                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  JSON-RPC 2.0                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心概念

#### Tools (工具)
LLM可以調用的函數，執行特定操作：

```python
# MCP工具定義示例
{
    "name": "search_documents",
    "description": "搜索文檔庫中的相關內容",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查詢"
            },
            "limit": {
                "type": "integer",
                "description": "返回結果數量",
                "default": 10
            }
        },
        "required": ["query"]
    }
}
```

#### Resources (資源)
提供給LLM讀取的數據源：

```python
# MCP資源定義示例
{
    "uri": "file:///documents/report.pdf",
    "name": "Annual Report 2024",
    "mimeType": "application/pdf",
    "description": "公司年度報告"
}
```

#### Prompts (提示模板)
預定義的提示詞模板：

```python
# MCP提示模板示例
{
    "name": "code_review",
    "description": "代碼審查提示模板",
    "arguments": [
        {
            "name": "code",
            "description": "要審查的代碼",
            "required": True
        },
        {
            "name": "language",
            "description": "編程語言",
            "required": False
        }
    ]
}
```

---

## 3. 與傳統Function Calling的對比

### 3.1 OpenAI Function Calling

```python
# OpenAI傳統方式
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取天氣信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```

### 3.2 MCP方式

```python
# MCP方式 - 更簡潔、更標準化
from mcp import Server, Tool

server = Server("weather-server")

@server.tool()
async def get_weather(location: str) -> str:
    """獲取天氣信息"""
    # 實現邏輯
    return f"{location}的天氣: 晴天, 25°C"

# 自動生成schema，自動處理序列化
```

### 3.3 對比總結

| 維度 | Function Calling | MCP |
|------|-----------------|-----|
| **定義方式** | 手動JSON Schema | 裝飾器自動推斷 |
| **傳輸協議** | HTTP REST | JSON-RPC (多傳輸) |
| **工具發現** | 靜態配置 | 動態發現 |
| **狀態管理** | 無狀態 | 支持會話狀態 |
| **權限控制** | 應用層實現 | 協議原生支持 |
| **生態系統** | 各廠商獨立 | 統一開放標準 |
| **調試體驗** | 依賴廠商 | MCP Inspector |

---

## 4. MCP SDK使用指南

### 4.1 安裝

```bash
# Python SDK
pip install mcp

# 或使用 uv (推薦)
uv add mcp

# TypeScript SDK
npm install @modelcontextprotocol/sdk
```

### 4.2 Python客戶端使用

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 連接MCP伺服器
    server_params = StdioServerParameters(
        command="python",
        args=["my_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化連接
            await session.initialize()

            # 列出可用工具
            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")

            # 調用工具
            result = await session.call_tool(
                "search_documents",
                arguments={"query": "機器學習", "limit": 5}
            )
            print(f"搜索結果: {result.content}")

            # 讀取資源
            resources = await session.list_resources()
            for resource in resources.resources:
                content = await session.read_resource(resource.uri)
                print(f"資源 {resource.name}: {content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.3 與LangChain整合

```python
from langchain_mcp import MCPToolkit
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI

# 創建MCP工具包
mcp_toolkit = MCPToolkit(
    servers=[
        {"command": "python", "args": ["file_server.py"]},
        {"command": "npx", "args": ["@mcp/weather-server"]}
    ]
)

# 獲取所有MCP工具
tools = mcp_toolkit.get_tools()

# 創建Agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools)

# 執行
result = agent.invoke({"input": "查詢北京天氣並搜索相關旅遊文章"})
```

---

## 5. 自訂MCP伺服器開發

### 5.1 基礎伺服器結構

```python
# my_mcp_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource
import asyncio

# 創建伺服器實例
server = Server("my-custom-server")

# 定義工具
@server.tool()
async def calculate(expression: str) -> str:
    """
    計算數學表達式

    Args:
        expression: 要計算的數學表達式 (例如: "2 + 2")

    Returns:
        計算結果
    """
    import ast
    import operator

    # 安全的運算符映射
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
        else:
            raise ValueError(f"不支援的運算: {type(node)}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return f"結果: {result}"
    except Exception as e:
        return f"計算錯誤: {str(e)}"

@server.tool()
async def search_knowledge_base(
    query: str,
    category: str = "all",
    limit: int = 10
) -> list[dict]:
    """
    搜索知識庫

    Args:
        query: 搜索查詢
        category: 分類過濾 (all, tech, business, science)
        limit: 返回結果數量

    Returns:
        搜索結果列表
    """
    # 實現搜索邏輯
    results = [
        {"title": "機器學習入門", "score": 0.95},
        {"title": "深度學習實戰", "score": 0.87}
    ]
    return results[:limit]

# 定義資源
@server.resource("config://app-settings")
async def get_app_settings() -> str:
    """應用程式配置"""
    return """
    {
        "version": "1.0.0",
        "features": ["search", "calculate", "summarize"]
    }
    """

@server.resource("file://{path}")
async def read_file(path: str) -> str:
    """讀取文件內容"""
    with open(path, 'r') as f:
        return f.read()

# 定義提示模板
@server.prompt()
async def summarize_template(text: str, style: str = "concise") -> str:
    """
    文本摘要提示模板

    Args:
        text: 要摘要的文本
        style: 摘要風格 (concise, detailed, bullet_points)
    """
    templates = {
        "concise": f"請用一句話總結以下內容:\n\n{text}",
        "detailed": f"請詳細總結以下內容，包括主要觀點和支持論據:\n\n{text}",
        "bullet_points": f"請用要點形式總結以下內容:\n\n{text}"
    }
    return templates.get(style, templates["concise"])

# 主函數
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 進階功能: 狀態管理

```python
from mcp.server import Server
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SessionState:
    user_id: str
    context: Dict[str, Any]
    history: list

class StatefulServer:
    def __init__(self):
        self.server = Server("stateful-server")
        self.sessions: Dict[str, SessionState] = {}
        self._setup_tools()

    def _setup_tools(self):
        @self.server.tool()
        async def start_session(user_id: str) -> str:
            """開始新會話"""
            self.sessions[user_id] = SessionState(
                user_id=user_id,
                context={},
                history=[]
            )
            return f"會話已創建: {user_id}"

        @self.server.tool()
        async def add_to_context(user_id: str, key: str, value: str) -> str:
            """添加上下文信息"""
            if user_id not in self.sessions:
                return "會話不存在"
            self.sessions[user_id].context[key] = value
            return f"已添加 {key} 到上下文"

        @self.server.tool()
        async def get_context(user_id: str) -> dict:
            """獲取當前上下文"""
            if user_id not in self.sessions:
                return {"error": "會話不存在"}
            return self.sessions[user_id].context
```

### 5.3 進階功能: 權限控制

```python
from mcp.server import Server
from mcp.types import Permission
from functools import wraps

class SecureServer:
    def __init__(self):
        self.server = Server("secure-server")
        self.permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "guest": ["read"]
        }

    def require_permission(self, permission: str):
        """權限裝飾器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 從上下文獲取用戶角色
                role = kwargs.get("role", "guest")
                if permission not in self.permissions.get(role, []):
                    raise PermissionError(f"需要 {permission} 權限")
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def setup_tools(self):
        @self.server.tool()
        @self.require_permission("read")
        async def read_data(path: str, role: str = "guest") -> str:
            """讀取數據 (需要read權限)"""
            return f"讀取: {path}"

        @self.server.tool()
        @self.require_permission("write")
        async def write_data(path: str, content: str, role: str = "guest") -> str:
            """寫入數據 (需要write權限)"""
            return f"寫入到: {path}"

        @self.server.tool()
        @self.require_permission("admin")
        async def admin_action(action: str, role: str = "guest") -> str:
            """管理操作 (需要admin權限)"""
            return f"執行管理操作: {action}"
```

---

## 6. 企業級整合方案

### 6.1 Claude Desktop配置

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
// %APPDATA%\Claude\claude_desktop_config.json (Windows)

{
  "mcpServers": {
    "knowledge-base": {
      "command": "python",
      "args": ["/path/to/knowledge_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://...",
        "API_KEY": "your-api-key"
      }
    },
    "file-system": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://..."
      }
    }
  }
}
```

### 6.2 生產環境部署架構

```
┌─────────────────────────────────────────────────────────────────┐
│                    企業MCP部署架構                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API Gateway                           │   │
│  │              (認證、限流、日誌)                           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────┼───────────────────────────────┐   │
│  │                    MCP Router                            │   │
│  │              (服務發現、負載均衡)                         │   │
│  └──────┬──────────────────┼──────────────────┬────────────┘   │
│         │                  │                  │                 │
│  ┌──────▼──────┐   ┌───────▼───────┐  ┌──────▼──────┐         │
│  │ Knowledge   │   │  File System  │  │  Database   │         │
│  │ MCP Server  │   │  MCP Server   │  │ MCP Server  │         │
│  └──────┬──────┘   └───────┬───────┘  └──────┬──────┘         │
│         │                  │                  │                 │
│  ┌──────▼──────┐   ┌───────▼───────┐  ┌──────▼──────┐         │
│  │   Vector    │   │   S3/Blob     │  │  PostgreSQL │         │
│  │   Database  │   │   Storage     │  │   + Redis   │         │
│  └─────────────┘   └───────────────┘  └─────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  監控 & 日誌                              │   │
│  │         Prometheus | Grafana | ELK Stack                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Docker部署示例

```dockerfile
# Dockerfile.mcp-server
FROM python:3.11-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製代碼
COPY . .

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mcp; print('healthy')" || exit 1

# 非root用戶
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 啟動
CMD ["python", "server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-knowledge:
    build:
      context: ./knowledge-server
      dockerfile: Dockerfile.mcp-server
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data:ro
    networks:
      - mcp-network
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  mcp-filesystem:
    image: mcp/server-filesystem:latest
    volumes:
      - ./documents:/documents:ro
    networks:
      - mcp-network

  mcp-router:
    build: ./mcp-router
    ports:
      - "8080:8080"
    depends_on:
      - mcp-knowledge
      - mcp-filesystem
    networks:
      - mcp-network

networks:
  mcp-network:
    driver: bridge
```

---

## 7. 最佳實踐

### 7.1 工具設計原則

```python
# ✅ 好的工具設計
@server.tool()
async def search_documents(
    query: str,
    filters: dict = None,
    limit: int = 10,
    offset: int = 0
) -> dict:
    """
    搜索文檔庫中的相關內容

    搜索支持全文檢索和向量相似度匹配，可以通過filters
    參數進行精確過濾。

    Args:
        query: 搜索查詢，支持自然語言
        filters: 過濾條件，格式如 {"category": "tech", "date_after": "2024-01-01"}
        limit: 返回結果數量，最大100
        offset: 分頁偏移量

    Returns:
        包含搜索結果的字典:
        {
            "results": [...],
            "total": 100,
            "has_more": True
        }

    Examples:
        >>> await search_documents("機器學習入門", limit=5)
        >>> await search_documents("Python", filters={"category": "tutorial"})
    """
    # 實現...

# ❌ 不好的工具設計
@server.tool()
async def search(q: str) -> list:
    """搜索"""
    # 缺少詳細描述、參數說明、返回值說明
    pass
```

### 7.2 錯誤處理

```python
from mcp.types import McpError, ErrorCode

@server.tool()
async def process_file(path: str) -> str:
    """處理文件"""
    try:
        # 檢查文件存在
        if not os.path.exists(path):
            raise McpError(
                ErrorCode.InvalidParams,
                f"文件不存在: {path}"
            )

        # 檢查文件大小
        size = os.path.getsize(path)
        if size > 10 * 1024 * 1024:  # 10MB
            raise McpError(
                ErrorCode.InvalidParams,
                f"文件過大: {size} bytes (最大10MB)"
            )

        # 處理文件
        with open(path, 'r') as f:
            content = f.read()

        return content

    except PermissionError:
        raise McpError(
            ErrorCode.InvalidRequest,
            f"無權限讀取文件: {path}"
        )
    except Exception as e:
        raise McpError(
            ErrorCode.InternalError,
            f"處理文件時發生錯誤: {str(e)}"
        )
```

### 7.3 性能優化

```python
import asyncio
from functools import lru_cache
from typing import List

class OptimizedServer:
    def __init__(self):
        self.server = Server("optimized-server")
        self._cache = {}
        self._setup_tools()

    def _setup_tools(self):
        # 使用緩存
        @self.server.tool()
        async def cached_search(query: str) -> list:
            """帶緩存的搜索"""
            cache_key = f"search:{query}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            result = await self._do_search(query)
            self._cache[cache_key] = result
            return result

        # 批量處理
        @self.server.tool()
        async def batch_process(items: List[str]) -> List[dict]:
            """批量處理多個項目"""
            # 並行處理
            tasks = [self._process_item(item) for item in items]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return [
                {"item": item, "result": r, "error": None}
                if not isinstance(r, Exception)
                else {"item": item, "result": None, "error": str(r)}
                for item, r in zip(items, results)
            ]

        # 流式響應
        @self.server.tool()
        async def stream_large_result(query: str):
            """流式返回大結果集"""
            async for chunk in self._stream_search(query):
                yield chunk
```

---

## 8. 實戰案例

### 8.1 RAG知識庫MCP伺服器

完整實現請參見: [examples/rag_mcp_server.py](./examples/rag_mcp_server.py)

### 8.2 數據庫查詢MCP伺服器

完整實現請參見: [examples/database_mcp_server.py](./examples/database_mcp_server.py)

### 8.3 API整合MCP伺服器

完整實現請參見: [examples/api_integration_server.py](./examples/api_integration_server.py)

---

## 📚 參考資源

- [MCP官方文檔](https://modelcontextprotocol.io/)
- [MCP GitHub倉庫](https://github.com/modelcontextprotocol)
- [MCP伺服器目錄](https://github.com/modelcontextprotocol/servers)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

## 🔗 相關章節

- [Agent工具設計與整合](../3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md#7-agent工具設計與整合)
- [Function Calling詳解](../12.進階提示工程與結構化輸出/function_calling_guide.md)
- [LLM安全與防禦](../8.LLM安全與防禦/README.md)
