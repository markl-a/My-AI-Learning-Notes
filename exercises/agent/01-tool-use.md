# 練習 3.1：工具定義與呼叫

**難度**: ⭐⭐ 基礎
**預計時間**: 1 小時
**前置知識**: Python、API 呼叫、JSON

## 學習目標

完成本練習後，你將能夠：

- [ ] 定義符合規範的工具 schema
- [ ] 實現工具呼叫邏輯
- [ ] 處理工具呼叫錯誤
- [ ] 理解 Function Calling 工作流程

## 背景知識

工具呼叫 (Tool Use / Function Calling) 讓 LLM 能夠與外部系統互動。

基本流程：
1. 定義工具 schema（名稱、描述、參數）
2. 發送用戶請求 + 工具定義給 LLM
3. LLM 決定是否呼叫工具，返回呼叫參數
4. 執行工具，將結果返回給 LLM
5. LLM 基於結果生成最終回答

## 練習任務

### 任務 1：定義工具 Schema

為以下功能定義工具 schema：

```python
# 1. 天氣查詢工具
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "獲取指定城市的天氣資訊",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO: 定義參數
                "city": {
                    "type": "string",
                    "description": "城市名稱，如：台北、東京"
                },
                # 添加更多參數...
            },
            "required": ["city"]
        }
    }
}

# 2. 計算器工具
calculator_tool = {
    # TODO: 完成定義
}

# 3. 網頁搜索工具
search_tool = {
    # TODO: 完成定義
}
```

### 任務 2：實現工具執行器

```python
from typing import Any, Dict, Callable

class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """註冊工具"""
        self.tools[name] = func

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        執行工具呼叫

        Args:
            tool_name: 工具名稱
            arguments: 呼叫參數

        Returns:
            工具執行結果（字串格式）
        """
        # TODO: 實現執行邏輯
        # 1. 檢查工具是否存在
        # 2. 驗證參數
        # 3. 執行並處理錯誤
        pass

# 實現具體工具
def get_weather(city: str, unit: str = "celsius") -> str:
    """模擬天氣查詢"""
    # TODO: 實現（可使用模擬資料）
    weather_data = {
        "台北": {"temp": 28, "condition": "晴"},
        "東京": {"temp": 22, "condition": "多雲"},
    }
    return str(weather_data.get(city, {"error": "城市不存在"}))

def calculate(expression: str) -> str:
    """安全的數學計算"""
    # TODO: 實現安全的表達式計算
    # 提示：使用 ast.literal_eval 或限制允許的操作
    pass

# 註冊工具
executor = ToolExecutor()
executor.register("get_weather", get_weather)
executor.register("calculate", calculate)
```

### 任務 3：完整的 Tool Use 流程

```python
from openai import OpenAI

client = OpenAI()

def chat_with_tools(user_message: str, tools: list, executor: ToolExecutor) -> str:
    """
    支持工具呼叫的對話函數

    Args:
        user_message: 用戶消息
        tools: 工具定義列表
        executor: 工具執行器

    Returns:
        最終回答
    """
    messages = [{"role": "user", "content": user_message}]

    # 第一次呼叫：獲取工具呼叫意圖
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # TODO: 處理工具呼叫
    # 1. 檢查是否有 tool_calls
    # 2. 如果有，執行每個工具呼叫
    # 3. 將結果添加到 messages
    # 4. 再次呼叫 LLM 獲取最終回答

    return "最終回答"

# 測試
result = chat_with_tools(
    "台北現在的天氣怎麼樣？",
    tools=[weather_tool],
    executor=executor
)
print(result)
```

### 任務 4：錯誤處理

實現健壯的錯誤處理：

```python
class ToolError(Exception):
    """工具執行錯誤"""
    pass

def safe_tool_execution(executor: ToolExecutor, tool_name: str, arguments: dict) -> str:
    """
    安全的工具執行封裝

    處理以下情況：
    1. 工具不存在
    2. 參數缺失/類型錯誤
    3. 執行超時
    4. 運行時錯誤
    """
    # TODO: 實現錯誤處理邏輯
    pass
```

## 驗證方法

- [ ] 工具 schema 符合 OpenAI Function Calling 規範
- [ ] `ToolExecutor` 能正確執行註冊的工具
- [ ] 完整流程能處理「台北天氣」和「計算 25*4」等請求
- [ ] 錯誤處理能優雅地處理各種異常情況

## 延伸思考

1. **並行工具呼叫**：如何處理 LLM 同時呼叫多個工具？
2. **工具依賴**：如果工具 B 依賴工具 A 的結果怎麼辦？
3. **安全考量**：如何防止工具被濫用？

## 下一步

完成本練習後，繼續學習：
- [練習 3.2：ReAct 模式實作](./02-react.md)
