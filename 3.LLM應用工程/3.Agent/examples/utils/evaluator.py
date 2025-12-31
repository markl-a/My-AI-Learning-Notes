"""
Agent 評估器

評估 Agent 的性能和輸出質量。
"""

from datetime import datetime
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
import json
import logging

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Agent 性能評估器

    使用 LLM 評估 Agent 的輸出質量。
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        初始化評估器

        Args:
            model: 用於評估的 LLM 模型
        """
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.evaluations: List[Dict] = []

    def evaluate_task(
        self,
        task_description: str,
        expected_output: str,
        actual_output: str,
        evaluation_criteria: Optional[List[str]] = None
    ) -> Dict:
        """
        評估單個任務

        Args:
            task_description: 任務描述
            expected_output: 期望輸出
            actual_output: 實際輸出
            evaluation_criteria: 評估標準列表

        Returns:
            評估結果字典
        """
        # 默認評估標準
        if evaluation_criteria is None:
            evaluation_criteria = [
                "準確性 (Accuracy): 輸出是否正確",
                "完整性 (Completeness): 是否包含所有必要資訊",
                "相關性 (Relevance): 是否切題",
                "質量 (Quality): 語言和格式是否良好"
            ]

        criteria_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(evaluation_criteria)])

        eval_prompt = f"""評估 AI Agent 的輸出質量。

任務描述：
{task_description}

期望輸出：
{expected_output}

實際輸出：
{actual_output}

評估標準（每項 0-10 分）：
{criteria_str}

請以 JSON 格式輸出評估結果：
{{
    "accuracy": <分數>,
    "completeness": <分數>,
    "relevance": <分數>,
    "quality": <分數>,
    "overall": <平均分>,
    "success": <true/false>,
    "feedback": "<改進建議>",
    "strengths": ["<優點1>", "<優點2>"],
    "weaknesses": ["<缺點1>", "<缺點2>"]
}}

只返回 JSON，不要包含其他內容。"""

        try:
            response = self.llm.invoke(eval_prompt)
            result_text = response.content.strip()

            # 嘗試解析 JSON
            # 移除可能的 markdown 代碼塊標記
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            scores = json.loads(result_text)

            # 確保包含所有必要欄位
            if "overall" not in scores:
                score_values = [
                    scores.get("accuracy", 0),
                    scores.get("completeness", 0),
                    scores.get("relevance", 0),
                    scores.get("quality", 0)
                ]
                scores["overall"] = sum(score_values) / len(score_values)

            if "success" not in scores:
                scores["success"] = scores["overall"] >= 7.0

            # 添加元數據
            evaluation = {
                "timestamp": datetime.now().isoformat(),
                "task_description": task_description,
                "scores": scores,
                "success": scores["success"]
            }

            self.evaluations.append(evaluation)

            logger.info(
                f"任務評估完成：{scores['overall']:.1f}/10 | "
                f"成功: {scores['success']}"
            )

            return scores

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失敗：{e}")
            logger.error(f"LLM 回應：{result_text}")

            # 返回預設評分
            return {
                "error": "評估失敗",
                "overall": 0,
                "success": False,
                "feedback": "評估過程中發生錯誤"
            }

        except Exception as e:
            logger.error(f"評估失敗：{e}")
            return {
                "error": str(e),
                "overall": 0,
                "success": False,
                "feedback": "評估過程中發生錯誤"
            }

    def get_statistics(self) -> Dict:
        """
        獲取統計資訊

        Returns:
            統計字典
        """
        if not self.evaluations:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 0.0,
                "average_scores": {}
            }

        successful_tasks = sum(1 for e in self.evaluations if e["success"])
        failed_tasks = len(self.evaluations) - successful_tasks

        # 計算平均分數
        score_keys = ["accuracy", "completeness", "relevance", "quality", "overall"]
        avg_scores = {}

        for key in score_keys:
            scores = [
                e["scores"].get(key, 0)
                for e in self.evaluations
                if isinstance(e["scores"].get(key), (int, float))
            ]
            avg_scores[key] = sum(scores) / len(scores) if scores else 0

        return {
            "total_tasks": len(self.evaluations),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / len(self.evaluations) if self.evaluations else 0,
            "average_scores": avg_scores
        }

    def generate_report(self) -> str:
        """
        生成評估報告

        Returns:
            Markdown 格式的報告
        """
        stats = self.get_statistics()

        report = f"""# Agent 評估報告

## 總體統計

- **總任務數**: {stats['total_tasks']}
- **成功任務**: {stats['successful_tasks']}
- **失敗任務**: {stats['failed_tasks']}
- **成功率**: {stats['success_rate']:.2%}

## 平均分數

"""

        if stats['average_scores']:
            for key, score in stats['average_scores'].items():
                report += f"- **{key.capitalize()}**: {score:.2f}/10\n"

        report += "\n## 詳細評估記錄\n\n"

        for i, evaluation in enumerate(self.evaluations, 1):
            scores = evaluation["scores"]
            report += f"### 任務 {i}\n\n"
            report += f"- **時間**: {evaluation['timestamp']}\n"
            report += f"- **任務**: {evaluation['task_description'][:100]}...\n"
            report += f"- **總分**: {scores.get('overall', 0):.1f}/10\n"
            report += f"- **狀態**: {'✅ 成功' if evaluation['success'] else '❌ 失敗'}\n"

            if "feedback" in scores:
                report += f"- **反饋**: {scores['feedback']}\n"

            if "strengths" in scores and scores['strengths']:
                report += f"- **優點**: {', '.join(scores['strengths'])}\n"

            if "weaknesses" in scores and scores['weaknesses']:
                report += f"- **缺點**: {', '.join(scores['weaknesses'])}\n"

            report += "\n"

        return report

    def print_report(self):
        """打印評估報告"""
        report = self.generate_report()
        print(report)

    def save_report(self, filepath: str):
        """
        保存評估報告到文件

        Args:
            filepath: 文件路徑
        """
        report = self.generate_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"評估報告已保存到：{filepath}")

    def save_data(self, filepath: str):
        """
        保存評估數據（JSON 格式）

        Args:
            filepath: 文件路徑
        """
        data = {
            "statistics": self.get_statistics(),
            "evaluations": self.evaluations
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"評估數據已保存到：{filepath}")
