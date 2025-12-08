"""
成本追蹤器單元測試

測試 CostTracker 類的各項功能。
"""

import pytest
import json
import tempfile
import os

# 添加路徑以導入模組
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
UTILS_PATH = os.path.join(
    PROJECT_ROOT,
    '3.LLM應用工程', '3.Agent', 'examples', 'utils'
)
sys.path.insert(0, UTILS_PATH)

from cost_tracker import CostTracker, get_global_tracker, reset_global_tracker


class TestCostTracker:
    """CostTracker 測試類"""

    def setup_method(self):
        """每個測試前重置"""
        reset_global_tracker()

    def test_initialization(self):
        """測試初始化"""
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert len(tracker.usage_log) == 0
        assert tracker.session_name.startswith("session_")

    def test_initialization_with_name(self):
        """測試帶名稱的初始化"""
        tracker = CostTracker(session_name="test_session")
        assert tracker.session_name == "test_session"

    def test_log_usage_gpt4(self):
        """測試記錄 GPT-4 使用量"""
        tracker = CostTracker()

        result = tracker.log_usage(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500
        )

        # GPT-4 價格: input=$0.03/1K, output=$0.06/1K
        expected_input_cost = (1000 / 1000) * 0.03  # $0.03
        expected_output_cost = (500 / 1000) * 0.06  # $0.03
        expected_total = expected_input_cost + expected_output_cost  # $0.06

        assert result["model"] == "gpt-4"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert abs(result["input_cost"] - expected_input_cost) < 0.0001
        assert abs(result["output_cost"] - expected_output_cost) < 0.0001
        assert abs(result["total_cost"] - expected_total) < 0.0001
        assert abs(tracker.total_cost - expected_total) < 0.0001

    def test_log_usage_gpt35_turbo(self):
        """測試記錄 GPT-3.5-Turbo 使用量"""
        tracker = CostTracker()

        result = tracker.log_usage(
            model="gpt-3.5-turbo",
            input_tokens=2000,
            output_tokens=1000
        )

        # GPT-3.5-Turbo 價格: input=$0.0005/1K, output=$0.0015/1K
        expected_input_cost = (2000 / 1000) * 0.0005  # $0.001
        expected_output_cost = (1000 / 1000) * 0.0015  # $0.0015
        expected_total = expected_input_cost + expected_output_cost  # $0.0025

        assert abs(result["total_cost"] - expected_total) < 0.0001

    def test_log_usage_claude(self):
        """測試記錄 Claude 使用量"""
        tracker = CostTracker()

        result = tracker.log_usage(
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=1000
        )

        # Claude-3-Sonnet 價格: input=$0.003/1K, output=$0.015/1K
        expected_input_cost = (1000 / 1000) * 0.003  # $0.003
        expected_output_cost = (1000 / 1000) * 0.015  # $0.015
        expected_total = expected_input_cost + expected_output_cost  # $0.018

        assert abs(result["total_cost"] - expected_total) < 0.0001

    def test_log_usage_unknown_model(self):
        """測試記錄未知模型（應返回 0 成本）"""
        tracker = CostTracker()

        result = tracker.log_usage(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=1000
        )

        assert result["total_cost"] == 0.0
        assert tracker.total_cost == 0.0

    def test_log_usage_with_metadata(self):
        """測試帶元數據的記錄"""
        tracker = CostTracker()

        metadata = {"task": "summarization", "user": "test_user"}
        result = tracker.log_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            metadata=metadata
        )

        assert result["metadata"] == metadata

    def test_multiple_logs_accumulate(self):
        """測試多次記錄累積成本"""
        tracker = CostTracker()

        tracker.log_usage("gpt-4", 1000, 500)
        tracker.log_usage("gpt-4", 1000, 500)

        assert len(tracker.usage_log) == 2
        # 每次 $0.06，共 $0.12
        assert abs(tracker.total_cost - 0.12) < 0.0001

    def test_get_summary_empty(self):
        """測試空追蹤器的摘要"""
        tracker = CostTracker()
        summary = tracker.get_summary()

        assert summary["total_cost"] == 0.0
        assert summary["total_calls"] == 0
        assert summary["total_tokens"] == 0
        assert summary["by_model"] == {}

    def test_get_summary_with_data(self):
        """測試有數據時的摘要"""
        tracker = CostTracker(session_name="test")

        tracker.log_usage("gpt-4", 1000, 500)
        tracker.log_usage("gpt-3.5-turbo", 2000, 1000)

        summary = tracker.get_summary()

        assert summary["session_name"] == "test"
        assert summary["total_calls"] == 2
        assert summary["total_tokens"] == 4500  # 1500 + 3000
        assert "gpt-4" in summary["by_model"]
        assert "gpt-3.5-turbo" in summary["by_model"]
        assert summary["by_model"]["gpt-4"]["calls"] == 1
        assert summary["by_model"]["gpt-3.5-turbo"]["calls"] == 1

    def test_group_by_model(self):
        """測試按模型分組統計"""
        tracker = CostTracker()

        tracker.log_usage("gpt-4", 1000, 500)
        tracker.log_usage("gpt-4", 2000, 1000)
        tracker.log_usage("gpt-3.5-turbo", 1000, 500)

        grouped = tracker._group_by_model()

        assert grouped["gpt-4"]["calls"] == 2
        assert grouped["gpt-4"]["input_tokens"] == 3000
        assert grouped["gpt-4"]["output_tokens"] == 1500
        assert grouped["gpt-3.5-turbo"]["calls"] == 1

    def test_reset(self):
        """測試重置功能"""
        tracker = CostTracker()

        tracker.log_usage("gpt-4", 1000, 500)
        assert tracker.total_cost > 0
        assert len(tracker.usage_log) > 0

        tracker.reset()

        assert tracker.total_cost == 0.0
        assert len(tracker.usage_log) == 0

    def test_save_and_load(self):
        """測試保存和載入"""
        tracker = CostTracker(session_name="save_test")

        tracker.log_usage("gpt-4", 1000, 500, {"task": "test"})
        tracker.log_usage("gpt-3.5-turbo", 2000, 1000)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            tracker.save_to_file(filepath)

            # 驗證文件存在
            assert os.path.exists(filepath)

            # 載入並驗證
            loaded_tracker = CostTracker.load_from_file(filepath)

            assert loaded_tracker.session_name == "save_test"
            assert abs(loaded_tracker.total_cost - tracker.total_cost) < 0.0001
            assert len(loaded_tracker.usage_log) == 2
        finally:
            os.unlink(filepath)

    def test_global_tracker(self):
        """測試全局追蹤器"""
        reset_global_tracker()

        tracker1 = get_global_tracker()
        tracker2 = get_global_tracker()

        assert tracker1 is tracker2

        tracker1.log_usage("gpt-4", 100, 50)

        assert tracker2.total_cost > 0

    def test_pricing_accuracy(self):
        """測試定價準確性"""
        tracker = CostTracker()

        # 測試所有已定義的模型
        for model, pricing in CostTracker.PRICING.items():
            result = tracker.log_usage(model, 1000, 1000)

            expected_cost = pricing["input"] + pricing["output"]
            assert abs(result["total_cost"] - expected_cost) < 0.0001, \
                f"Model {model} pricing mismatch"

            tracker.reset()


class TestCostTrackerEdgeCases:
    """邊界情況測試"""

    def test_zero_tokens(self):
        """測試零 token 的情況"""
        tracker = CostTracker()
        result = tracker.log_usage("gpt-4", 0, 0)

        assert result["total_cost"] == 0.0
        assert result["total_tokens"] == 0

    def test_large_token_count(self):
        """測試大量 token"""
        tracker = CostTracker()
        result = tracker.log_usage("gpt-4", 100000, 50000)

        # GPT-4: 100K * $0.03 + 50K * $0.06 = $3 + $3 = $6
        expected_cost = (100000 / 1000) * 0.03 + (50000 / 1000) * 0.06
        assert abs(result["total_cost"] - expected_cost) < 0.01

    def test_timestamp_in_log(self):
        """測試日誌包含時間戳"""
        tracker = CostTracker()
        result = tracker.log_usage("gpt-4", 100, 50)

        assert "timestamp" in result
        # 驗證 ISO 格式
        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])
