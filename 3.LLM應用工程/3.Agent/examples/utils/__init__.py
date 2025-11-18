"""
AI Agent 實戰範例 - 共用工具模組

提供可重用的工具函數、評估器、成本追蹤等功能。
"""

from .agent_utils import (
    setup_environment,
    get_llm,
    create_tool_from_function,
    safe_execute,
)

from .cost_tracker import CostTracker

from .evaluator import AgentEvaluator

from .logger import setup_logger, get_logger

from .prompt_templates import (
    REACT_PROMPT_TEMPLATE,
    AGENT_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
)

__all__ = [
    # Agent 工具
    "setup_environment",
    "get_llm",
    "create_tool_from_function",
    "safe_execute",

    # 追蹤和評估
    "CostTracker",
    "AgentEvaluator",

    # 日誌
    "setup_logger",
    "get_logger",

    # 提示模板
    "REACT_PROMPT_TEMPLATE",
    "AGENT_SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
]
