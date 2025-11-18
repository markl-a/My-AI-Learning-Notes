"""
性能監控工具
監控 LLM API 的性能指標，包括延遲、吞吐量、錯誤率等
"""

import time
import statistics
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json


@dataclass
class PerformanceMetric:
    """性能指標"""
    timestamp: datetime
    provider: str
    model: str
    operation: str  # chat, completion, embedding
    latency: float  # 秒
    tokens_per_second: Optional[float]
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class PerformanceMonitor:
    """性能監控器"""

    def __init__(self, log_file: str = "logs/performance.json"):
        self.log_file = log_file
        self.metrics: List[PerformanceMetric] = []
        self._load_metrics()

    def _load_metrics(self):
        """載入歷史指標"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.metrics = [
                    PerformanceMetric(
                        timestamp=datetime.fromisoformat(m['timestamp']),
                        **{k: v for k, v in m.items() if k != 'timestamp'}
                    )
                    for m in data
                ]
        except FileNotFoundError:
            self.metrics = []

    def _save_metrics(self):
        """保存指標"""
        with open(self.log_file, 'w') as f:
            json.dump(
                [{**asdict(m), 'timestamp': m.timestamp.isoformat()} for m in self.metrics],
                f,
                indent=2
            )

    def record(
        self,
        provider: str,
        model: str,
        operation: str,
        latency: float,
        tokens: Optional[int] = None,
        success: bool = True,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """記錄一次請求的性能指標"""
        tokens_per_second = tokens / latency if tokens and latency > 0 else None

        metric = PerformanceMetric(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            operation=operation,
            latency=latency,
            tokens_per_second=tokens_per_second,
            success=success,
            error_type=error_type,
            error_message=error_message
        )

        self.metrics.append(metric)
        self._save_metrics()

    def get_statistics(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        operation: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> Dict:
        """獲取統計資訊"""
        filtered = self._filter_metrics(provider, model, operation, time_range)

        if not filtered:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "error_rate": 0,
                "avg_latency": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0
            }

        latencies = [m.latency for m in filtered]
        successful = [m for m in filtered if m.success]
        failed = [m for m in filtered if not m.success]

        sorted_latencies = sorted(latencies)

        return {
            "total_requests": len(filtered),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "error_rate": len(failed) / len(filtered) * 100,
            "avg_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p50_latency": sorted_latencies[int(len(sorted_latencies) * 0.50)],
            "p95_latency": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99_latency": sorted_latencies[int(len(sorted_latencies) * 0.99)],
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "avg_tokens_per_second": self._calculate_avg_tps(filtered)
        }

    def get_error_breakdown(
        self,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, int]:
        """獲取錯誤類型統計"""
        filtered = self._filter_metrics(time_range=time_range)
        failed = [m for m in filtered if not m.success]

        breakdown = {}
        for metric in failed:
            error_type = metric.error_type or "Unknown"
            breakdown[error_type] = breakdown.get(error_type, 0) + 1

        return breakdown

    def get_latency_trend(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        interval: timedelta = timedelta(hours=1)
    ) -> List[Dict]:
        """獲取延遲趨勢"""
        filtered = self._filter_metrics(provider, model)

        if not filtered:
            return []

        # 按時間間隔分組
        trends = []
        start_time = min(m.timestamp for m in filtered)
        end_time = max(m.timestamp for m in filtered)

        current_time = start_time
        while current_time <= end_time:
            next_time = current_time + interval

            interval_metrics = [
                m for m in filtered
                if current_time <= m.timestamp < next_time
            ]

            if interval_metrics:
                latencies = [m.latency for m in interval_metrics]
                trends.append({
                    "timestamp": current_time.isoformat(),
                    "count": len(interval_metrics),
                    "avg_latency": statistics.mean(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                })

            current_time = next_time

        return trends

    def _filter_metrics(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        operation: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> List[PerformanceMetric]:
        """過濾指標"""
        filtered = self.metrics

        if provider:
            filtered = [m for m in filtered if m.provider == provider]

        if model:
            filtered = [m for m in filtered if m.model == model]

        if operation:
            filtered = [m for m in filtered if m.operation == operation]

        if time_range:
            cutoff = datetime.now() - time_range
            filtered = [m for m in filtered if m.timestamp >= cutoff]

        return filtered

    def _calculate_avg_tps(self, metrics: List[PerformanceMetric]) -> Optional[float]:
        """計算平均 tokens/second"""
        tps_values = [m.tokens_per_second for m in metrics if m.tokens_per_second is not None]
        return statistics.mean(tps_values) if tps_values else None


def print_performance_report(monitor: PerformanceMonitor):
    """打印性能報告"""
    print("=" * 60)
    print("LLM API 性能報告")
    print("=" * 60)

    # 整體統計
    overall = monitor.get_statistics()
    print("\n【整體統計】")
    print(f"總請求數: {overall['total_requests']}")
    print(f"成功請求: {overall['successful_requests']}")
    print(f"失敗請求: {overall['failed_requests']}")
    print(f"錯誤率: {overall['error_rate']:.2f}%")

    print("\n【延遲統計（秒）】")
    print(f"平均延遲: {overall['avg_latency']:.3f}")
    print(f"中位數: {overall['median_latency']:.3f}")
    print(f"P95: {overall['p95_latency']:.3f}")
    print(f"P99: {overall['p99_latency']:.3f}")
    print(f"最小: {overall['min_latency']:.3f}")
    print(f"最大: {overall['max_latency']:.3f}")

    if overall['avg_tokens_per_second']:
        print(f"\n平均吞吐量: {overall['avg_tokens_per_second']:.2f} tokens/秒")

    # 錯誤統計
    errors = monitor.get_error_breakdown(timedelta(days=7))
    if errors:
        print("\n【錯誤類型統計（最近7天）】")
        for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")

    # 按提供商統計
    print("\n【按提供商統計】")
    for provider in set(m.provider for m in monitor.metrics):
        stats = monitor.get_statistics(provider=provider)
        print(f"\n{provider}:")
        print(f"  請求數: {stats['total_requests']}")
        print(f"  平均延遲: {stats['avg_latency']:.3f}s")
        print(f"  錯誤率: {stats['error_rate']:.2f}%")


# 使用示例
if __name__ == "__main__":
    # 創建監控器
    monitor = PerformanceMonitor("logs/performance.json")

    # 模擬記錄一些指標
    monitor.record(
        provider="openai",
        model="gpt-4o-mini",
        operation="chat",
        latency=1.234,
        tokens=300,
        success=True
    )

    monitor.record(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        operation="chat",
        latency=2.567,
        tokens=500,
        success=True
    )

    monitor.record(
        provider="openai",
        model="gpt-4o",
        operation="chat",
        latency=3.456,
        tokens=400,
        success=False,
        error_type="RateLimitError",
        error_message="Rate limit exceeded"
    )

    # 顯示報告
    print_performance_report(monitor)

    # 顯示最近24小時的趨勢
    print("\n\n【延遲趨勢（最近24小時）】")
    trends = monitor.get_latency_trend(
        interval=timedelta(hours=1)
    )

    for trend in trends[:5]:  # 只顯示前5個
        print(f"\n時間: {trend['timestamp']}")
        print(f"  請求數: {trend['count']}")
        print(f"  平均延遲: {trend['avg_latency']:.3f}s")
