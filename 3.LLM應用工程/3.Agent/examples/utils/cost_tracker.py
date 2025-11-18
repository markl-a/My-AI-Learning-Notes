"""
成本追蹤器

追蹤 LLM API 調用的成本。
"""

from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class CostTracker:
    """
    LLM API 成本追蹤器

    追蹤不同模型的 token 使用量和成本。
    """

    # 2024-2025 年價格（美元/1K tokens）
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    }

    def __init__(self, session_name: Optional[str] = None):
        """
        初始化成本追蹤器

        Args:
            session_name: 會話名稱（用於區分不同的實驗）
        """
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.usage_log: List[Dict] = []
        self.total_cost = 0.0

    def log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        記錄單次 API 調用

        Args:
            model: 模型名稱
            input_tokens: 輸入 token 數
            output_tokens: 輸出 token 數
            metadata: 額外的元數據

        Returns:
            包含成本資訊的字典
        """
        # 獲取價格
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})

        # 計算成本（價格是每 1K tokens）
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        self.total_cost += total_cost

        # 創建日誌條目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "metadata": metadata or {}
        }

        self.usage_log.append(log_entry)

        logger.info(
            f"API 調用：{model} | "
            f"Tokens: {input_tokens}+{output_tokens}={input_tokens + output_tokens} | "
            f"成本: ${total_cost:.6f}"
        )

        return log_entry

    def get_summary(self) -> Dict:
        """
        獲取成本摘要

        Returns:
            包含統計資訊的字典
        """
        if not self.usage_log:
            return {
                "total_cost": 0.0,
                "total_calls": 0,
                "total_tokens": 0,
                "avg_cost_per_call": 0.0,
                "by_model": {}
            }

        total_tokens = sum(log["total_tokens"] for log in self.usage_log)

        summary = {
            "session_name": self.session_name,
            "total_cost": self.total_cost,
            "total_calls": len(self.usage_log),
            "total_tokens": total_tokens,
            "avg_cost_per_call": self.total_cost / len(self.usage_log) if self.usage_log else 0,
            "avg_tokens_per_call": total_tokens / len(self.usage_log) if self.usage_log else 0,
            "by_model": self._group_by_model()
        }

        return summary

    def _group_by_model(self) -> Dict:
        """按模型分組統計"""
        grouped = {}

        for log in self.usage_log:
            model = log["model"]

            if model not in grouped:
                grouped[model] = {
                    "calls": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }

            grouped[model]["calls"] += 1
            grouped[model]["total_cost"] += log["total_cost"]
            grouped[model]["total_tokens"] += log["total_tokens"]
            grouped[model]["input_tokens"] += log["input_tokens"]
            grouped[model]["output_tokens"] += log["output_tokens"]

        return grouped

    def print_summary(self):
        """打印成本摘要"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print(f"成本追蹤摘要 - {summary['session_name']}")
        print("=" * 60)
        print(f"總成本: ${summary['total_cost']:.4f}")
        print(f"總調用次數: {summary['total_calls']}")
        print(f"總 Token 數: {summary['total_tokens']:,}")
        print(f"平均每次調用成本: ${summary['avg_cost_per_call']:.4f}")
        print(f"平均每次調用 Tokens: {summary['avg_tokens_per_call']:.0f}")

        if summary['by_model']:
            print("\n按模型統計:")
            print("-" * 60)
            for model, stats in summary['by_model'].items():
                print(f"\n{model}:")
                print(f"  調用次數: {stats['calls']}")
                print(f"  總成本: ${stats['total_cost']:.4f}")
                print(f"  總 Tokens: {stats['total_tokens']:,}")
                print(f"  輸入 Tokens: {stats['input_tokens']:,}")
                print(f"  輸出 Tokens: {stats['output_tokens']:,}")

        print("=" * 60 + "\n")

    def save_to_file(self, filepath: str):
        """
        保存追蹤數據到文件

        Args:
            filepath: 文件路徑
        """
        summary = self.get_summary()
        data = {
            "summary": summary,
            "logs": self.usage_log
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"成本追蹤數據已保存到：{filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CostTracker':
        """
        從文件載入追蹤數據

        Args:
            filepath: 文件路徑

        Returns:
            CostTracker 實例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tracker = cls(session_name=data["summary"]["session_name"])
        tracker.usage_log = data["logs"]
        tracker.total_cost = data["summary"]["total_cost"]

        return tracker

    def reset(self):
        """重置追蹤器"""
        self.usage_log = []
        self.total_cost = 0.0
        logger.info("成本追蹤器已重置")


# 全局追蹤器實例
_global_tracker = None


def get_global_tracker() -> CostTracker:
    """獲取全局追蹤器實例"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker():
    """重置全局追蹤器"""
    global _global_tracker
    _global_tracker = None
