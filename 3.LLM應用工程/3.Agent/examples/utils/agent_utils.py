"""
Agent 相關工具函數

提供 Agent 開發常用的輔助函數。
"""

import os
from typing import Callable, Any, Optional, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)


def setup_environment() -> Dict[str, str]:
    """
    設置環境變數並載入 API 金鑰

    Returns:
        包含已載入環境變數的字典
    """
    # 載入 .env 文件
    load_dotenv()

    # 檢查必要的環境變數
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.warning(f"缺少環境變數: {', '.join(missing_vars)}")
        logger.warning("請確保 .env 文件包含所有必要的 API 金鑰")

    # 返回所有環境變數
    env_vars = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "serper_api_key": os.getenv("SERPER_API_KEY", ""),
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "langchain_api_key": os.getenv("LANGCHAIN_API_KEY", ""),
    }

    return env_vars


def get_llm(
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    **kwargs
) -> ChatOpenAI:
    """
    獲取配置好的 LLM 實例

    Args:
        model: 模型名稱
        temperature: 溫度參數 (0-1)
        max_tokens: 最大 token 數
        streaming: 是否啟用流式輸出
        **kwargs: 其他 LLM 參數

    Returns:
        配置好的 ChatOpenAI 實例
    """
    params = {
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
        **kwargs
    }

    if max_tokens:
        params["max_tokens"] = max_tokens

    return ChatOpenAI(**params)


def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    return_direct: bool = False
) -> Tool:
    """
    從普通函數創建 LangChain Tool

    Args:
        func: 要包裝的函數
        name: 工具名稱（默認使用函數名）
        description: 工具描述（默認使用函數文檔字符串）
        return_direct: 是否直接返回工具結果

    Returns:
        LangChain Tool 實例

    Example:
        >>> def search_wikipedia(query: str) -> str:
        ...     '''搜尋維基百科'''
        ...     return f"維基百科結果：{query}"
        >>>
        >>> tool = create_tool_from_function(search_wikipedia)
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or "沒有描述"

    return Tool(
        name=tool_name,
        func=func,
        description=tool_description,
        return_direct=return_direct
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    **kwargs
) -> Any:
    """
    安全執行函數，帶重試和錯誤處理

    Args:
        func: 要執行的函數
        *args: 位置參數
        fallback_value: 失敗時的備用值
        **kwargs: 關鍵字參數

    Returns:
        函數執行結果或備用值

    Example:
        >>> result = safe_execute(risky_api_call, param1="value")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"執行失敗：{func.__name__}, 錯誤：{str(e)}")
        if fallback_value is not None:
            logger.info(f"返回備用值：{fallback_value}")
            return fallback_value
        raise


def format_tool_output(output: Any, max_length: int = 1000) -> str:
    """
    格式化工具輸出，限制長度

    Args:
        output: 原始輸出
        max_length: 最大長度

    Returns:
        格式化後的字符串
    """
    output_str = str(output)

    if len(output_str) > max_length:
        return output_str[:max_length] + f"\n... (輸出過長，已截斷。總長度：{len(output_str)})"

    return output_str


def validate_agent_input(query: str, max_length: int = 10000) -> tuple[bool, str]:
    """
    驗證 Agent 輸入的安全性

    Args:
        query: 用戶輸入
        max_length: 最大長度

    Returns:
        (是否有效, 錯誤訊息)
    """
    # 檢查長度
    if len(query) > max_length:
        return False, f"輸入過長（{len(query)} > {max_length}）"

    # 檢查危險模式
    dangerous_patterns = [
        "ignore previous instructions",
        "忽略之前的指示",
        "你現在是",
        "forget everything",
        "disregard",
    ]

    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            return False, f"檢測到潛在危險輸入：{pattern}"

    return True, ""


def parse_agent_response(response: str) -> Dict[str, str]:
    """
    解析 Agent 的結構化回應

    Args:
        response: Agent 回應文本

    Returns:
        解析後的字典

    Example:
        >>> text = "思考：我需要搜尋\\n行動：search\\n最終答案：結果"
        >>> parsed = parse_agent_response(text)
        >>> print(parsed["thought"])
        我需要搜尋
    """
    result = {
        "thought": "",
        "action": "",
        "action_input": "",
        "observation": "",
        "final_answer": ""
    }

    lines = response.split("\n")
    current_key = None

    for line in lines:
        line = line.strip()

        # 檢測各種標記
        if line.startswith("思考：") or line.startswith("Thought:"):
            current_key = "thought"
            result[current_key] = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("行動：") or line.startswith("Action:"):
            current_key = "action"
            result[current_key] = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("行動輸入：") or line.startswith("Action Input:"):
            current_key = "action_input"
            result[current_key] = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("觀察：") or line.startswith("Observation:"):
            current_key = "observation"
            result[current_key] = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("最終答案：") or line.startswith("Final Answer:"):
            current_key = "final_answer"
            result[current_key] = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif current_key and line:
            # 繼續前一個 key 的內容
            result[current_key] += "\n" + line

    return result
