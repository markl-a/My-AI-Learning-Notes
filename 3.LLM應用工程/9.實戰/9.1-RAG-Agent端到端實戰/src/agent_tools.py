"""Agent 工具集"""
import logging
from typing import Any, Dict, List, Optional
import json
import re

logger = logging.getLogger(__name__)


class BaseTool:
    """工具基類"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs) -> str:
        """執行工具"""
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        """獲取工具的 schema（用於 LLM 理解）"""
        return {
            "name": self.name,
            "description": self.description
        }


class RAGSearchTool(BaseTool):
    """RAG 文檔檢索工具"""

    def __init__(self, vector_store_manager):
        super().__init__(
            name="rag_search",
            description="在文檔庫中搜索相關信息。適用於查詢已知的文檔內容、事實性問題等。"
        )
        self.vector_store = vector_store_manager

    def run(self, query: str, top_k: int = 5) -> str:
        """執行 RAG 搜索

        Args:
            query: 查詢文本
            top_k: 返回結果數量

        Returns:
            格式化的搜索結果
        """
        try:
            results = self.vector_store.similarity_search(query, top_k=top_k)

            if not results:
                return "未找到相關文檔。"

            # 格式化結果
            formatted_results = []
            for i, (doc, metadata, score) in enumerate(results, 1):
                source = metadata.get('source', 'Unknown')
                formatted_results.append(
                    f"[文檔 {i}] (來源: {source}, 相關度: {score:.2f})\n{doc}\n"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return f"搜索失敗: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的查詢文本"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的結果數量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


class CalculatorTool(BaseTool):
    """計算器工具"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="執行數學計算。支持基本運算、科學計算等。輸入數學表達式，返回計算結果。"
        )

    def run(self, expression: str) -> str:
        """執行計算

        Args:
            expression: 數學表達式

        Returns:
            計算結果
        """
        try:
            # 安全的數學表達式評估
            # 只允許數字、運算符和常見函數
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow
            }

            # 添加數學函數
            try:
                import math
                allowed_names.update({
                    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                    "tan": math.tan, "log": math.log, "exp": math.exp,
                    "pi": math.pi, "e": math.e
                })
            except ImportError:
                pass

            # 清理表達式
            expression = expression.strip()

            # 檢查是否包含不安全的內容
            if re.search(r'[^0-9+\-*/().,\s\w]', expression):
                return "錯誤：表達式包含不允許的字符"

            # 評估表達式
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return f"計算結果: {result}"

        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return f"計算錯誤: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要計算的數學表達式，例如: '2 + 2', 'sqrt(16)', 'sin(pi/2)'"
                    }
                },
                "required": ["expression"]
            }
        }


class WebSearchTool(BaseTool):
    """網路搜索工具（模擬）"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search",
            description="在互聯網上搜索最新信息。適用於實時數據、新聞、無法在文檔庫中找到的信息。"
        )
        self.api_key = api_key

    def run(self, query: str, num_results: int = 5) -> str:
        """執行網路搜索

        Args:
            query: 搜索查詢
            num_results: 結果數量

        Returns:
            搜索結果
        """
        # 注意：這是一個模擬實現
        # 在生產環境中，你需要集成真實的搜索 API（如 Serper, Google Custom Search 等）

        if not self.api_key:
            return "網路搜索功能未配置。請設置 API Key。"

        try:
            # 模擬搜索結果
            logger.info(f"Performing web search for: {query}")

            # 這裡應該調用真實的搜索 API
            # 示例：使用 Serper API
            """
            import requests
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            response = requests.post(url, headers=headers, data=payload)
            results = response.json()
            """

            # 模擬返回
            return f"[模擬] 網路搜索 '{query}' 的結果:\n1. 相關結果1\n2. 相關結果2\n3. 相關結果3\n\n注意：這是模擬結果。要啟用真實搜索，請配置搜索 API。"

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"搜索失敗: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的查詢文本"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回的結果數量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


class CodeInterpreterTool(BaseTool):
    """代碼解釋器工具"""

    def __init__(self):
        super().__init__(
            name="code_interpreter",
            description="執行 Python 代碼片段。可用於數據處理、繪圖、複雜計算等。注意：僅支持安全的代碼執行。"
        )

    def run(self, code: str) -> str:
        """執行 Python 代碼

        Args:
            code: Python 代碼

        Returns:
            執行結果
        """
        try:
            # 注意：這是一個簡化的實現
            # 在生產環境中，應該使用沙箱環境（如 Docker）來執行代碼

            # 檢查危險操作
            dangerous_keywords = ['import os', 'import sys', 'exec', 'eval', '__import__', 'open']
            for keyword in dangerous_keywords:
                if keyword in code:
                    return f"錯誤：代碼包含不允許的操作: {keyword}"

            # 創建受限的執行環境
            allowed_imports = {
                'math': __import__('math'),
                'statistics': __import__('statistics'),
                'json': __import__('json'),
            }

            # 嘗試導入數據科學庫
            try:
                allowed_imports['numpy'] = __import__('numpy')
                allowed_imports['np'] = __import__('numpy')
            except ImportError:
                pass

            # 捕獲輸出
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                # 執行代碼
                exec(code, allowed_imports)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            if not output:
                output = "代碼執行成功，無輸出。"

            return f"執行結果:\n{output}"

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"執行錯誤: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要執行的 Python 代碼"
                    }
                },
                "required": ["code"]
            }
        }


class ToolRegistry:
    """工具註冊表"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """註冊工具

        Args:
            tool: 工具實例
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """獲取工具

        Args:
            name: 工具名稱

        Returns:
            工具實例或 None
        """
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """獲取所有工具"""
        return list(self.tools.values())

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """獲取所有工具的 schema"""
        return [tool.get_schema() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """執行工具

        Args:
            tool_name: 工具名稱
            **kwargs: 工具參數

        Returns:
            執行結果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return f"錯誤：工具 '{tool_name}' 不存在"

        try:
            return tool.run(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"工具執行失敗: {str(e)}"
