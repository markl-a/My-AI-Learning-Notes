"""
Agent 系統性能基準測試
測試不同 Agent 框架的性能和效率
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any
import json
from datetime import datetime
import os


class AgentBenchmark:
    """Agent 基準測試類"""

    def __init__(self):
        """初始化基準測試"""
        self.results = []

    async def benchmark_reaction_time(
        self,
        agent_type: str,
        tasks: List[str],
        num_runs: int = 5
    ) -> Dict[str, float]:
        """
        測試 Agent 反應時間

        Args:
            agent_type: Agent 類型（如 'langgraph', 'crewai', 'autogen'）
            tasks: 任務列表
            num_runs: 運行次數

        Returns:
            反應時間統計
        """
        print(f"\n⏱️  測試 {agent_type} Agent 反應時間...")

        reaction_times = []

        for i in range(num_runs):
            for task in tasks:
                start_time = time.time()

                # 模擬 Agent 執行
                # 實際測試中應調用真實的 Agent
                await self._simulate_agent_execution(agent_type, task)

                end_time = time.time()
                reaction_time = (end_time - start_time) * 1000  # ms

                reaction_times.append(reaction_time)
                print(f"  運行 {i+1}, 任務: {task[:30]}... -> {reaction_time:.2f}ms")

        stats = {
            "mean": statistics.mean(reaction_times),
            "median": statistics.median(reaction_times),
            "min": min(reaction_times),
            "max": max(reaction_times),
            "stdev": statistics.stdev(reaction_times) if len(reaction_times) > 1 else 0
        }

        return stats

    async def _simulate_agent_execution(self, agent_type: str, task: str):
        """模擬 Agent 執行（實際測試中替換為真實調用）"""
        # 模擬不同 Agent 的執行時間
        delays = {
            "langgraph": 0.5,
            "crewai": 0.8,
            "autogen": 1.0,
            "react": 0.6
        }
        await asyncio.sleep(delays.get(agent_type, 0.5))

    async def benchmark_task_completion(
        self,
        agent_type: str,
        complex_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        測試任務完成能力

        Args:
            agent_type: Agent 類型
            complex_tasks: 複雜任務列表

        Returns:
            任務完成統計
        """
        print(f"\n✅ 測試 {agent_type} Agent 任務完成能力...")

        completion_stats = {
            "total_tasks": len(complex_tasks),
            "completed": 0,
            "failed": 0,
            "partial": 0,
            "avg_steps": 0,
            "avg_time": 0
        }

        all_steps = []
        all_times = []

        for i, task in enumerate(complex_tasks):
            print(f"  任務 {i+1}/{len(complex_tasks)}: {task['description'][:50]}...")

            start_time = time.time()

            # 模擬任務執行
            result = await self._execute_complex_task(agent_type, task)

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000

            # 評估結果
            if result["status"] == "completed":
                completion_stats["completed"] += 1
            elif result["status"] == "failed":
                completion_stats["failed"] += 1
            else:
                completion_stats["partial"] += 1

            all_steps.append(result["steps"])
            all_times.append(execution_time)

            print(f"    狀態: {result['status']}, "
                  f"步驟: {result['steps']}, "
                  f"時間: {execution_time:.2f}ms")

        completion_stats["avg_steps"] = statistics.mean(all_steps)
        completion_stats["avg_time"] = statistics.mean(all_times)
        completion_stats["success_rate"] = completion_stats["completed"] / completion_stats["total_tasks"]

        return completion_stats

    async def _execute_complex_task(
        self,
        agent_type: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行複雜任務（模擬）"""
        # 模擬不同複雜度的任務
        complexity = task.get("complexity", "medium")

        delays = {
            "simple": 1.0,
            "medium": 2.0,
            "complex": 3.0
        }

        await asyncio.sleep(delays.get(complexity, 2.0))

        # 模擬結果
        import random
        success_rate = 0.85

        return {
            "status": "completed" if random.random() < success_rate else "failed",
            "steps": random.randint(3, 10),
            "output": "Task completed successfully"
        }

    async def benchmark_multi_agent_coordination(
        self,
        agent_type: str,
        num_agents: int,
        coordination_tasks: List[Dict]
    ) -> Dict[str, Any]:
        """
        測試多 Agent 協調性能

        Args:
            agent_type: Agent 類型
            num_agents: Agent 數量
            coordination_tasks: 協調任務列表

        Returns:
            協調性能統計
        """
        print(f"\n🤝 測試 {agent_type} 多 Agent 協調 ({num_agents} agents)...")

        coordination_stats = {
            "num_agents": num_agents,
            "total_tasks": len(coordination_tasks),
            "coordination_overhead": 0,
            "avg_coordination_time": 0,
            "efficiency": 0
        }

        coordination_times = []

        for i, task in enumerate(coordination_tasks):
            print(f"  協調任務 {i+1}/{len(coordination_tasks)}...")

            start_time = time.time()

            # 模擬多 Agent 協調
            result = await self._simulate_multi_agent(agent_type, num_agents, task)

            end_time = time.time()
            coordination_time = (end_time - start_time) * 1000

            coordination_times.append(coordination_time)

            print(f"    協調時間: {coordination_time:.2f}ms, "
                  f"消息數: {result['messages']}")

        coordination_stats["avg_coordination_time"] = statistics.mean(coordination_times)
        coordination_stats["coordination_overhead"] = coordination_stats["avg_coordination_time"] / num_agents
        coordination_stats["efficiency"] = 1.0 / coordination_stats["coordination_overhead"]

        return coordination_stats

    async def _simulate_multi_agent(
        self,
        agent_type: str,
        num_agents: int,
        task: Dict
    ) -> Dict[str, Any]:
        """模擬多 Agent 協調"""
        # 協調開銷隨 Agent 數量增加
        base_delay = 0.5
        coordination_delay = base_delay * (num_agents / 2)

        await asyncio.sleep(coordination_delay)

        return {
            "messages": num_agents * 3,  # 每個 Agent 平均 3 條消息
            "status": "completed"
        }

    async def benchmark_tool_usage(
        self,
        agent_type: str,
        tool_tasks: List[Dict]
    ) -> Dict[str, Any]:
        """
        測試工具使用效率

        Args:
            agent_type: Agent 類型
            tool_tasks: 需要使用工具的任務列表

        Returns:
            工具使用統計
        """
        print(f"\n🔧 測試 {agent_type} Agent 工具使用效率...")

        tool_stats = {
            "total_tasks": len(tool_tasks),
            "tool_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_call_time": 0
        }

        call_times = []

        for i, task in enumerate(tool_tasks):
            print(f"  工具任務 {i+1}/{len(tool_tasks)}...")

            start_time = time.time()

            # 模擬工具調用
            result = await self._simulate_tool_usage(agent_type, task)

            end_time = time.time()
            call_time = (end_time - start_time) * 1000

            tool_stats["tool_calls"] += result["num_calls"]
            tool_stats["successful_calls"] += result["successful"]
            tool_stats["failed_calls"] += result["failed"]
            call_times.append(call_time)

            print(f"    工具調用: {result['num_calls']}, "
                  f"成功: {result['successful']}, "
                  f"時間: {call_time:.2f}ms")

        tool_stats["avg_call_time"] = statistics.mean(call_times)
        tool_stats["success_rate"] = tool_stats["successful_calls"] / tool_stats["tool_calls"] if tool_stats["tool_calls"] > 0 else 0

        return tool_stats

    async def _simulate_tool_usage(
        self,
        agent_type: str,
        task: Dict
    ) -> Dict[str, int]:
        """模擬工具使用"""
        import random

        num_tools = task.get("required_tools", 3)
        await asyncio.sleep(0.3 * num_tools)

        successful = int(num_tools * 0.9)  # 90% 成功率

        return {
            "num_calls": num_tools,
            "successful": successful,
            "failed": num_tools - successful
        }

    async def run_full_benchmark(
        self,
        agent_types: List[str]
    ):
        """
        運行完整的 Agent 基準測試

        Args:
            agent_types: 要測試的 Agent 類型列表
        """
        print("=" * 70)
        print("🤖 Agent 系統性能基準測試")
        print("=" * 70)

        # 定義測試任務
        simple_tasks = [
            "Summarize this text",
            "Translate to Chinese",
            "Generate a list of ideas"
        ]

        complex_tasks = [
            {"description": "Research and write a report", "complexity": "complex"},
            {"description": "Analyze data and create visualization", "complexity": "complex"},
            {"description": "Code review and refactoring", "complexity": "medium"},
            {"description": "Plan a project timeline", "complexity": "medium"},
            {"description": "Answer customer query", "complexity": "simple"}
        ]

        coordination_tasks = [
            {"type": "collaborative_writing", "requires": ["research", "analysis", "writing"]},
            {"type": "code_development", "requires": ["design", "coding", "testing"]},
            {"type": "data_pipeline", "requires": ["extraction", "transformation", "loading"]}
        ]

        tool_tasks = [
            {"required_tools": 3, "type": "web_search"},
            {"required_tools": 5, "type": "data_analysis"},
            {"required_tools": 2, "type": "file_operations"}
        ]

        all_results = {}

        for agent_type in agent_types:
            print(f"\n{'='*70}")
            print(f"📌 測試 Agent: {agent_type}")
            print(f"{'='*70}")

            agent_results = {}

            # 1. 反應時間測試
            reaction_stats = await self.benchmark_reaction_time(
                agent_type=agent_type,
                tasks=simple_tasks,
                num_runs=3
            )
            agent_results["reaction_time"] = reaction_stats

            # 2. 任務完成測試
            completion_stats = await self.benchmark_task_completion(
                agent_type=agent_type,
                complex_tasks=complex_tasks
            )
            agent_results["task_completion"] = completion_stats

            # 3. 多 Agent 協調測試
            coordination_stats = await self.benchmark_multi_agent_coordination(
                agent_type=agent_type,
                num_agents=3,
                coordination_tasks=coordination_tasks
            )
            agent_results["coordination"] = coordination_stats

            # 4. 工具使用測試
            tool_stats = await self.benchmark_tool_usage(
                agent_type=agent_type,
                tool_tasks=tool_tasks
            )
            agent_results["tool_usage"] = tool_stats

            all_results[agent_type] = agent_results

            await asyncio.sleep(1)

        # 保存和打印結果
        self.save_results(all_results)
        self.print_summary(all_results)

    def save_results(self, results: Dict):
        """保存結果"""
        os.makedirs("results", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/agent_benchmark_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 結果已保存到: {filename}")

    def print_summary(self, results: Dict):
        """打印結果摘要"""
        print("\n" + "=" * 70)
        print("📊 Agent 基準測試摘要")
        print("=" * 70)

        for agent_type, metrics in results.items():
            print(f"\n🔹 {agent_type}")
            print("-" * 70)

            if "reaction_time" in metrics:
                rt = metrics["reaction_time"]
                print(f"  反應時間:")
                print(f"    平均: {rt['mean']:.2f}ms")
                print(f"    中位數: {rt['median']:.2f}ms")

            if "task_completion" in metrics:
                tc = metrics["task_completion"]
                print(f"  任務完成:")
                print(f"    成功率: {tc['success_rate']:.2%}")
                print(f"    平均步驟: {tc['avg_steps']:.1f}")
                print(f"    平均時間: {tc['avg_time']:.2f}ms")

            if "coordination" in metrics:
                coord = metrics["coordination"]
                print(f"  多 Agent 協調:")
                print(f"    協調時間: {coord['avg_coordination_time']:.2f}ms")
                print(f"    協調開銷: {coord['coordination_overhead']:.2f}ms/agent")

            if "tool_usage" in metrics:
                tool = metrics["tool_usage"]
                print(f"  工具使用:")
                print(f"    成功率: {tool['success_rate']:.2%}")
                print(f"    平均調用時間: {tool['avg_call_time']:.2f}ms")


async def main():
    """主函數"""
    # 要測試的 Agent 類型
    agent_types = [
        "langgraph",
        "crewai",
        "autogen",
        "react"
    ]

    benchmark = AgentBenchmark()
    await benchmark.run_full_benchmark(agent_types=agent_types)


if __name__ == "__main__":
    asyncio.run(main())
