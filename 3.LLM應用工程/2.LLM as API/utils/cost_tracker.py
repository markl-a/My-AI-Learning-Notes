"""
成本追踪工具
追踪和分析 LLM API 的使用成本
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import csv


# 價格表（美元 per 1M tokens）- 2025年1月
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
}


@dataclass
class UsageRecord:
    """使用記錄"""
    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None


class CostTracker:
    """成本追踪器"""

    def __init__(self, log_file: str = "costs.json"):
        self.log_file = log_file
        self.records: List[UsageRecord] = []
        self._load_records()

    def _load_records(self):
        """載入歷史記錄"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.records = [
                    UsageRecord(
                        timestamp=datetime.fromisoformat(r['timestamp']),
                        **{k: v for k, v in r.items() if k != 'timestamp'}
                    )
                    for r in data
                ]
        except FileNotFoundError:
            self.records = []

    def _save_records(self):
        """保存記錄"""
        with open(self.log_file, 'w') as f:
            json.dump(
                [{**asdict(r), 'timestamp': r.timestamp.isoformat()} for r in self.records],
                f,
                indent=2
            )

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Dict[str, float]:
        """計算成本"""
        if model not in PRICING:
            raise ValueError(f"未知的模型: {model}")

        pricing = PRICING[model]
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }

    def track(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> UsageRecord:
        """追踪一次使用"""
        costs = self.calculate_cost(model, prompt_tokens, completion_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost=costs["input_cost"],
            output_cost=costs["output_cost"],
            total_cost=costs["total_cost"],
            request_id=request_id,
            user_id=user_id,
            tags=tags or []
        )

        self.records.append(record)
        self._save_records()

        return record

    def get_total_cost(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> float:
        """獲取總成本"""
        filtered = self._filter_records(provider, model, user_id, start_date, end_date)
        return sum(r.total_cost for r in filtered)

    def get_total_tokens(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """獲取總 token 數"""
        filtered = self._filter_records(provider, model, user_id, start_date, end_date)

        return {
            "prompt_tokens": sum(r.prompt_tokens for r in filtered),
            "completion_tokens": sum(r.completion_tokens for r in filtered),
            "total_tokens": sum(r.total_tokens for r in filtered)
        }

    def get_statistics(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """獲取統計資訊"""
        filtered = self._filter_records(provider, model, user_id, start_date, end_date)

        if not filtered:
            return {
                "request_count": 0,
                "total_cost": 0,
                "average_cost": 0,
                "total_tokens": 0,
                "average_tokens": 0
            }

        total_cost = sum(r.total_cost for r in filtered)
        total_tokens = sum(r.total_tokens for r in filtered)

        return {
            "request_count": len(filtered),
            "total_cost": total_cost,
            "average_cost": total_cost / len(filtered),
            "min_cost": min(r.total_cost for r in filtered),
            "max_cost": max(r.total_cost for r in filtered),
            "total_tokens": total_tokens,
            "average_tokens": total_tokens / len(filtered),
            "by_provider": self._group_by(filtered, "provider"),
            "by_model": self._group_by(filtered, "model")
        }

    def get_cost_by_provider(self) -> Dict[str, float]:
        """按提供商統計成本"""
        result = {}
        for record in self.records:
            result[record.provider] = result.get(record.provider, 0) + record.total_cost
        return result

    def get_cost_by_model(self) -> Dict[str, float]:
        """按模型統計成本"""
        result = {}
        for record in self.records:
            result[record.model] = result.get(record.model, 0) + record.total_cost
        return result

    def export_to_csv(self, output_file: str):
        """導出為 CSV"""
        with open(output_file, 'w', newline='') as f:
            if self.records:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'timestamp', 'provider', 'model',
                        'prompt_tokens', 'completion_tokens', 'total_tokens',
                        'input_cost', 'output_cost', 'total_cost',
                        'request_id', 'user_id', 'tags'
                    ]
                )
                writer.writeheader()

                for record in self.records:
                    row = asdict(record)
                    row['timestamp'] = record.timestamp.isoformat()
                    row['tags'] = ','.join(record.tags) if record.tags else ''
                    writer.writerow(row)

    def _filter_records(
        self,
        provider: Optional[str],
        model: Optional[str],
        user_id: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[UsageRecord]:
        """過濾記錄"""
        filtered = self.records

        if provider:
            filtered = [r for r in filtered if r.provider == provider]

        if model:
            filtered = [r for r in filtered if r.model == model]

        if user_id:
            filtered = [r for r in filtered if r.user_id == user_id]

        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]

        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]

        return filtered

    def _group_by(self, records: List[UsageRecord], field: str) -> Dict:
        """按字段分組統計"""
        groups = {}

        for record in records:
            key = getattr(record, field)
            if key not in groups:
                groups[key] = {
                    "count": 0,
                    "total_cost": 0,
                    "total_tokens": 0
                }

            groups[key]["count"] += 1
            groups[key]["total_cost"] += record.total_cost
            groups[key]["total_tokens"] += record.total_tokens

        return groups


def print_statistics(tracker: CostTracker):
    """打印統計資訊"""
    stats = tracker.get_statistics()

    print("=" * 60)
    print("LLM API 使用統計")
    print("=" * 60)

    print(f"\n總請求數: {stats['request_count']}")
    print(f"總成本: ${stats['total_cost']:.4f}")
    print(f"平均成本: ${stats['average_cost']:.6f}")
    print(f"最小成本: ${stats['min_cost']:.6f}")
    print(f"最大成本: ${stats['max_cost']:.6f}")

    print(f"\n總 Tokens: {stats['total_tokens']:,}")
    print(f"平均 Tokens: {stats['average_tokens']:.0f}")

    print("\n按提供商:")
    for provider, data in stats['by_provider'].items():
        print(f"  {provider}:")
        print(f"    請求數: {data['count']}")
        print(f"    總成本: ${data['total_cost']:.4f}")
        print(f"    總 Tokens: {data['total_tokens']:,}")

    print("\n按模型:")
    for model, data in stats['by_model'].items():
        print(f"  {model}:")
        print(f"    請求數: {data['count']}")
        print(f"    總成本: ${data['total_cost']:.4f}")
        print(f"    總 Tokens: {data['total_tokens']:,}")


# 使用示例
if __name__ == "__main__":
    # 創建追踪器
    tracker = CostTracker("logs/costs.json")

    # 追踪使用
    tracker.track(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=200,
        user_id="user123",
        tags=["chat", "production"]
    )

    tracker.track(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        prompt_tokens=500,
        completion_tokens=800,
        user_id="user456",
        tags=["code_review"]
    )

    # 顯示統計
    print_statistics(tracker)

    # 導出 CSV
    tracker.export_to_csv("logs/costs_export.csv")
    print("\n✅ 已導出到 logs/costs_export.csv")
