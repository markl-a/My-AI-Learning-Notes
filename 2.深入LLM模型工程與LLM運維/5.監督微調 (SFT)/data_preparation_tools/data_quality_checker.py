"""
數據質量檢查工具
檢查訓練數據的質量問題，包括重複、格式錯誤、長度異常等
"""

import json
import re
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import statistics
from dataclasses import dataclass
import hashlib


@dataclass
class QualityReport:
    """質量報告"""
    total_examples: int
    duplicates: List[Tuple[int, int]]  # 重複樣本的索引對
    format_errors: List[Dict]
    length_anomalies: List[Dict]
    empty_fields: List[Dict]
    quality_score: float
    recommendations: List[str]


class DataQualityChecker:
    """數據質量檢查器"""

    def __init__(self, data_file: str):
        """
        初始化檢查器

        Args:
            data_file: JSON 數據文件路徑
        """
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """載入數據"""
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_all(self) -> QualityReport:
        """執行所有質量檢查"""
        print(f"檢查數據文件: {self.data_file}")
        print(f"總樣本數: {len(self.data)}")
        print("-" * 60)

        duplicates = self.check_duplicates()
        format_errors = self.check_format()
        length_anomalies = self.check_length_distribution()
        empty_fields = self.check_empty_fields()

        # 計算質量分數
        quality_score = self._calculate_quality_score(
            duplicates, format_errors, length_anomalies, empty_fields
        )

        # 生成建議
        recommendations = self._generate_recommendations(
            duplicates, format_errors, length_anomalies, empty_fields
        )

        return QualityReport(
            total_examples=len(self.data),
            duplicates=duplicates,
            format_errors=format_errors,
            length_anomalies=length_anomalies,
            empty_fields=empty_fields,
            quality_score=quality_score,
            recommendations=recommendations
        )

    def check_duplicates(self) -> List[Tuple[int, int]]:
        """檢查重複樣本"""
        print("\n檢查重複樣本...")

        duplicates = []
        seen = {}

        for i, example in enumerate(self.data):
            # 創建樣本的哈希值
            content = f"{example.get('instruction', '')}|{example.get('input', '')}|{example.get('output', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash in seen:
                duplicates.append((seen[content_hash], i))
                print(f"  發現重複: 樣本 {seen[content_hash]} 和 {i}")
            else:
                seen[content_hash] = i

        print(f"發現 {len(duplicates)} 對重複樣本")
        return duplicates

    def check_format(self) -> List[Dict]:
        """檢查格式錯誤"""
        print("\n檢查格式...")

        errors = []
        required_fields = ["instruction", "output"]

        for i, example in enumerate(self.data):
            # 檢查必需字段
            for field in required_fields:
                if field not in example:
                    errors.append({
                        "index": i,
                        "type": "missing_field",
                        "field": field,
                        "message": f"缺少必需字段: {field}"
                    })

            # 檢查字段類型
            for field in ["instruction", "input", "output"]:
                if field in example and not isinstance(example[field], str):
                    errors.append({
                        "index": i,
                        "type": "invalid_type",
                        "field": field,
                        "message": f"字段 {field} 不是字符串類型"
                    })

        print(f"發現 {len(errors)} 個格式錯誤")
        for error in errors[:5]:  # 只顯示前 5 個
            print(f"  {error}")

        return errors

    def check_length_distribution(self) -> List[Dict]:
        """檢查長度分佈異常"""
        print("\n檢查長度分佈...")

        anomalies = []

        # 收集長度統計
        instruction_lengths = []
        output_lengths = []

        for example in self.data:
            instruction_lengths.append(len(example.get("instruction", "")))
            output_lengths.append(len(example.get("output", "")))

        # 計算統計量
        inst_mean = statistics.mean(instruction_lengths)
        inst_std = statistics.stdev(instruction_lengths) if len(instruction_lengths) > 1 else 0
        out_mean = statistics.mean(output_lengths)
        out_std = statistics.stdev(output_lengths) if len(output_lengths) > 1 else 0

        print(f"指令長度: 平均={inst_mean:.1f}, 標準差={inst_std:.1f}")
        print(f"輸出長度: 平均={out_mean:.1f}, 標準差={out_std:.1f}")

        # 檢查異常值（超過 3 個標準差）
        for i, example in enumerate(self.data):
            inst_len = len(example.get("instruction", ""))
            out_len = len(example.get("output", ""))

            if inst_std > 0 and abs(inst_len - inst_mean) > 3 * inst_std:
                anomalies.append({
                    "index": i,
                    "field": "instruction",
                    "length": inst_len,
                    "mean": inst_mean,
                    "message": f"指令長度異常: {inst_len} (平均: {inst_mean:.1f})"
                })

            if out_std > 0 and abs(out_len - out_mean) > 3 * out_std:
                anomalies.append({
                    "index": i,
                    "field": "output",
                    "length": out_len,
                    "mean": out_mean,
                    "message": f"輸出長度異常: {out_len} (平均: {out_mean:.1f})"
                })

        print(f"發現 {len(anomalies)} 個長度異常")
        return anomalies

    def check_empty_fields(self) -> List[Dict]:
        """檢查空字段"""
        print("\n檢查空字段...")

        empty_issues = []

        for i, example in enumerate(self.data):
            # 檢查指令是否為空
            if not example.get("instruction", "").strip():
                empty_issues.append({
                    "index": i,
                    "field": "instruction",
                    "message": "指令為空"
                })

            # 檢查輸出是否為空
            if not example.get("output", "").strip():
                empty_issues.append({
                    "index": i,
                    "field": "output",
                    "message": "輸出為空"
                })

        print(f"發現 {len(empty_issues)} 個空字段")
        return empty_issues

    def analyze_diversity(self) -> Dict:
        """分析數據多樣性"""
        print("\n分析數據多樣性...")

        # 指令開頭詞統計
        instruction_starts = []
        for example in self.data:
            instruction = example.get("instruction", "")
            if instruction:
                # 提取前 2 個詞
                words = instruction.split()[:2]
                instruction_starts.append(" ".join(words))

        start_counter = Counter(instruction_starts)
        print(f"\n最常見的指令開頭 (前 10):")
        for start, count in start_counter.most_common(10):
            print(f"  {start}: {count} 次 ({count/len(self.data)*100:.1f}%)")

        # 輸出長度分佈
        output_lengths = [len(ex.get("output", "")) for ex in self.data]
        print(f"\n輸出長度分佈:")
        print(f"  最小: {min(output_lengths)}")
        print(f"  最大: {max(output_lengths)}")
        print(f"  平均: {statistics.mean(output_lengths):.1f}")
        print(f"  中位數: {statistics.median(output_lengths):.1f}")

        return {
            "instruction_starts": dict(start_counter),
            "output_length_stats": {
                "min": min(output_lengths),
                "max": max(output_lengths),
                "mean": statistics.mean(output_lengths),
                "median": statistics.median(output_lengths)
            }
        }

    def _calculate_quality_score(
        self,
        duplicates: List,
        format_errors: List,
        length_anomalies: List,
        empty_fields: List
    ) -> float:
        """計算質量分數 (0-100)"""
        if len(self.data) == 0:
            return 0.0

        # 扣分項
        deductions = 0

        # 重複樣本：每對扣 2 分
        deductions += len(duplicates) * 2

        # 格式錯誤：每個扣 5 分
        deductions += len(format_errors) * 5

        # 空字段：每個扣 3 分
        deductions += len(empty_fields) * 3

        # 長度異常：每個扣 1 分
        deductions += len(length_anomalies) * 1

        # 計算分數
        score = max(0, 100 - deductions)
        return score

    def _generate_recommendations(
        self,
        duplicates: List,
        format_errors: List,
        length_anomalies: List,
        empty_fields: List
    ) -> List[str]:
        """生成改進建議"""
        recommendations = []

        if duplicates:
            recommendations.append(
                f"發現 {len(duplicates)} 對重複樣本，建議移除以提高數據多樣性"
            )

        if format_errors:
            recommendations.append(
                f"發現 {len(format_errors)} 個格式錯誤，建議修復後再訓練"
            )

        if empty_fields:
            recommendations.append(
                f"發現 {len(empty_fields)} 個空字段，建議填充或移除這些樣本"
            )

        if length_anomalies:
            recommendations.append(
                f"發現 {len(length_anomalies)} 個長度異常，建議檢查是否為數據錯誤"
            )

        # 樣本數量建議
        if len(self.data) < 100:
            recommendations.append(
                "樣本數量較少 (<100)，建議增加更多訓練數據"
            )
        elif len(self.data) < 500:
            recommendations.append(
                "樣本數量適中 (100-500)，對於簡單任務可能足夠"
            )

        if not recommendations:
            recommendations.append("數據質量良好，可以開始訓練！")

        return recommendations


def print_report(report: QualityReport):
    """打印質量報告"""
    print("\n" + "=" * 60)
    print("數據質量報告")
    print("=" * 60)

    print(f"\n總樣本數: {report.total_examples}")
    print(f"質量分數: {report.quality_score:.1f}/100")

    print(f"\n問題統計:")
    print(f"  重複樣本: {len(report.duplicates)} 對")
    print(f"  格式錯誤: {len(report.format_errors)} 個")
    print(f"  長度異常: {len(report.length_anomalies)} 個")
    print(f"  空字段: {len(report.empty_fields)} 個")

    print(f"\n改進建議:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 60)


def main():
    """示例使用"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python data_quality_checker.py <data_file.json>")
        sys.exit(1)

    data_file = sys.argv[1]

    checker = DataQualityChecker(data_file)
    report = checker.check_all()

    # 分析多樣性
    diversity = checker.analyze_diversity()

    # 打印報告
    print_report(report)


if __name__ == "__main__":
    main()
