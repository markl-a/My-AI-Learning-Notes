# Prompt 版本管理與工程化

## 概述

隨著 LLM 應用的複雜度增加，Prompt 管理變得至關重要。本指南涵蓋 Prompt 版本控制、A/B 測試、自動評估和生產環境管理的最佳實踐。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prompt 工程化流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │  開發    │──▶│  測試    │──▶│  評估    │──▶│  部署    │   │
│  │ Prompt   │   │ Prompt   │   │ Prompt   │   │ Prompt   │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│       │              │              │              │          │
│       ▼              ▼              ▼              ▼          │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              版本控制系統 (Git/DB)                    │     │
│  └──────────────────────────────────────────────────────┘     │
│                          │                                     │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │           監控與回饋收集                              │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prompt 版本控制

### 基礎版本管理系統

```python
"""
Prompt 版本管理系統
"""
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3
from enum import Enum


class PromptStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    APPROVED = "approved"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class PromptVersion:
    """Prompt 版本"""
    id: str
    name: str
    version: str
    template: str
    variables: List[str]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    status: PromptStatus = PromptStatus.DRAFT
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """生成唯一 ID"""
        content = f"{self.name}:{self.version}:{self.template}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def render(self, **kwargs) -> str:
        """渲染 Prompt"""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{{{var}}}}}", str(kwargs[var]))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """從字典建立"""
        data["status"] = PromptStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class PromptRegistry:
    """Prompt 註冊表"""

    def __init__(self, db_path: str = "prompts.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化資料庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, version)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_name
            ON prompts(name)
        """)

        conn.commit()
        conn.close()

    def register(self, prompt: PromptVersion) -> str:
        """註冊 Prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO prompts (id, name, version, data) VALUES (?, ?, ?, ?)",
                (prompt.id, prompt.name, prompt.version, json.dumps(prompt.to_dict()))
            )
            conn.commit()
            return prompt.id
        except sqlite3.IntegrityError:
            # 版本已存在，更新
            prompt.updated_at = datetime.now()
            cursor.execute(
                "UPDATE prompts SET data = ? WHERE id = ?",
                (json.dumps(prompt.to_dict()), prompt.id)
            )
            conn.commit()
            return prompt.id
        finally:
            conn.close()

    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptVersion]:
        """獲取 Prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if version:
            cursor.execute(
                "SELECT data FROM prompts WHERE name = ? AND version = ?",
                (name, version)
            )
        else:
            # 獲取最新版本
            cursor.execute(
                "SELECT data FROM prompts WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,)
            )

        row = cursor.fetchone()
        conn.close()

        if row:
            return PromptVersion.from_dict(json.loads(row[0]))
        return None

    def get_by_id(self, prompt_id: str) -> Optional[PromptVersion]:
        """根據 ID 獲取 Prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT data FROM prompts WHERE id = ?", (prompt_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return PromptVersion.from_dict(json.loads(row[0]))
        return None

    def list_versions(self, name: str) -> List[PromptVersion]:
        """列出所有版本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT data FROM prompts WHERE name = ? ORDER BY created_at DESC",
            (name,)
        )

        rows = cursor.fetchall()
        conn.close()

        return [PromptVersion.from_dict(json.loads(row[0])) for row in rows]

    def get_production(self, name: str) -> Optional[PromptVersion]:
        """獲取生產版本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT data FROM prompts WHERE name = ?",
            (name,)
        )

        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            prompt = PromptVersion.from_dict(json.loads(row[0]))
            if prompt.status == PromptStatus.PRODUCTION:
                return prompt

        return None

    def record_metric(
        self,
        prompt_id: str,
        metric_name: str,
        metric_value: float
    ):
        """記錄指標"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO prompt_metrics (prompt_id, metric_name, metric_value) VALUES (?, ?, ?)",
            (prompt_id, metric_name, metric_value)
        )

        conn.commit()
        conn.close()

    def get_metrics(
        self,
        prompt_id: str,
        metric_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """獲取指標"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if metric_name:
            cursor.execute(
                """SELECT metric_name, metric_value, recorded_at
                   FROM prompt_metrics
                   WHERE prompt_id = ? AND metric_name = ?
                   ORDER BY recorded_at DESC""",
                (prompt_id, metric_name)
            )
        else:
            cursor.execute(
                """SELECT metric_name, metric_value, recorded_at
                   FROM prompt_metrics
                   WHERE prompt_id = ?
                   ORDER BY recorded_at DESC""",
                (prompt_id,)
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {"name": row[0], "value": row[1], "recorded_at": row[2]}
            for row in rows
        ]
```

### Git 整合

```python
"""
Prompt Git 版本控制
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import subprocess


@dataclass
class PromptFile:
    """Prompt 文件格式"""
    name: str
    version: str
    model: str
    template: str
    variables: List[str]
    temperature: float = 0.7
    max_tokens: int = 1000
    description: str = ""
    tags: List[str] = None
    examples: List[Dict] = None

    def to_yaml(self) -> str:
        """轉換為 YAML"""
        data = {
            "name": self.name,
            "version": self.version,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "description": self.description,
            "tags": self.tags or [],
            "variables": self.variables,
            "template": self.template,
            "examples": self.examples or []
        }
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, content: str) -> "PromptFile":
        """從 YAML 建立"""
        data = yaml.safe_load(content)
        return cls(**data)


class GitPromptManager:
    """Git 整合的 Prompt 管理器"""

    def __init__(self, repo_path: str = "./prompts"):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)

    def save_prompt(self, prompt: PromptFile, commit: bool = True) -> str:
        """保存 Prompt 到文件"""
        # 建立目錄結構: prompts/{name}/{version}.yaml
        prompt_dir = self.repo_path / prompt.name
        prompt_dir.mkdir(parents=True, exist_ok=True)

        file_path = prompt_dir / f"{prompt.version}.yaml"
        file_path.write_text(prompt.to_yaml(), encoding="utf-8")

        if commit:
            self._git_commit(
                file_path,
                f"Update prompt {prompt.name} to version {prompt.version}"
            )

        return str(file_path)

    def load_prompt(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[PromptFile]:
        """載入 Prompt"""
        prompt_dir = self.repo_path / name

        if not prompt_dir.exists():
            return None

        if version:
            file_path = prompt_dir / f"{version}.yaml"
        else:
            # 獲取最新版本
            files = sorted(prompt_dir.glob("*.yaml"), reverse=True)
            if not files:
                return None
            file_path = files[0]

        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")
        return PromptFile.from_yaml(content)

    def list_prompts(self) -> Dict[str, List[str]]:
        """列出所有 Prompts"""
        result = {}

        for prompt_dir in self.repo_path.iterdir():
            if prompt_dir.is_dir():
                versions = [
                    f.stem for f in prompt_dir.glob("*.yaml")
                ]
                result[prompt_dir.name] = sorted(versions, reverse=True)

        return result

    def diff_versions(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> str:
        """比較兩個版本"""
        file1 = self.repo_path / name / f"{version1}.yaml"
        file2 = self.repo_path / name / f"{version2}.yaml"

        if not file1.exists() or not file2.exists():
            raise FileNotFoundError("One or both versions not found")

        result = subprocess.run(
            ["diff", "-u", str(file1), str(file2)],
            capture_output=True,
            text=True
        )

        return result.stdout

    def get_history(self, name: str, version: str) -> List[Dict]:
        """獲取版本歷史"""
        file_path = self.repo_path / name / f"{version}.yaml"

        result = subprocess.run(
            ["git", "log", "--oneline", "--", str(file_path)],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        commits = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(" ", 1)
                commits.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else ""
                })

        return commits

    def _git_commit(self, file_path: Path, message: str):
        """Git 提交"""
        subprocess.run(
            ["git", "add", str(file_path)],
            cwd=self.repo_path
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo_path
        )


# Prompt YAML 格式範例
PROMPT_YAML_EXAMPLE = """
name: customer_support
version: "2.1.0"
model: gpt-4o
temperature: 0.7
max_tokens: 1000
description: 客戶支援對話 Prompt

tags:
  - customer-service
  - chat
  - production

variables:
  - customer_name
  - issue_type
  - order_id

template: |
  你是一位專業的客戶服務代表。

  客戶資訊：
  - 姓名：{{customer_name}}
  - 問題類型：{{issue_type}}
  - 訂單編號：{{order_id}}

  請以友善、專業的態度回應客戶的問題。
  如果需要更多資訊，請禮貌地詢問。

examples:
  - input:
      customer_name: "王小明"
      issue_type: "退款"
      order_id: "ORD-12345"
    expected_behavior: "詢問退款原因並提供退款流程說明"

  - input:
      customer_name: "李小華"
      issue_type: "物流查詢"
      order_id: "ORD-67890"
    expected_behavior: "提供物流追蹤資訊或查詢方式"
"""
```

## A/B 測試框架

```python
"""
Prompt A/B 測試框架
"""
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class ABTestVariant:
    """A/B 測試變體"""
    name: str
    prompt_id: str
    weight: float = 1.0
    is_control: bool = False


@dataclass
class ABTestResult:
    """測試結果"""
    variant_name: str
    prompt_id: str
    user_id: str
    response: str
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ABTest:
    """A/B 測試配置"""
    id: str
    name: str
    variants: List[ABTestVariant]
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool = True
    min_sample_size: int = 100
    confidence_level: float = 0.95


class ABTestManager:
    """A/B 測試管理器"""

    def __init__(self, registry: "PromptRegistry"):
        self.registry = registry
        self.tests: Dict[str, ABTest] = {}
        self.results: Dict[str, List[ABTestResult]] = defaultdict(list)

    def create_test(
        self,
        name: str,
        control_prompt_id: str,
        variant_prompt_ids: List[str],
        weights: Optional[List[float]] = None
    ) -> ABTest:
        """建立 A/B 測試"""
        test_id = hashlib.sha256(
            f"{name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        variants = [
            ABTestVariant(
                name="control",
                prompt_id=control_prompt_id,
                weight=weights[0] if weights else 1.0,
                is_control=True
            )
        ]

        for i, prompt_id in enumerate(variant_prompt_ids):
            variants.append(ABTestVariant(
                name=f"variant_{i+1}",
                prompt_id=prompt_id,
                weight=weights[i+1] if weights else 1.0
            ))

        test = ABTest(
            id=test_id,
            name=name,
            variants=variants,
            start_time=datetime.now()
        )

        self.tests[test_id] = test
        return test

    def get_variant(
        self,
        test_id: str,
        user_id: str
    ) -> ABTestVariant:
        """為用戶分配變體（確定性分配）"""
        test = self.tests.get(test_id)
        if not test or not test.is_active:
            raise ValueError(f"Test {test_id} not found or inactive")

        # 使用用戶 ID 進行確定性分配
        hash_value = int(hashlib.sha256(
            f"{test_id}:{user_id}".encode()
        ).hexdigest(), 16)

        total_weight = sum(v.weight for v in test.variants)
        threshold = (hash_value % 10000) / 10000.0

        cumulative = 0
        for variant in test.variants:
            cumulative += variant.weight / total_weight
            if threshold < cumulative:
                return variant

        return test.variants[-1]

    def record_result(
        self,
        test_id: str,
        result: ABTestResult
    ):
        """記錄測試結果"""
        self.results[test_id].append(result)

    def analyze_test(
        self,
        test_id: str,
        metric_name: str = "satisfaction"
    ) -> Dict[str, Any]:
        """分析測試結果"""
        test = self.tests.get(test_id)
        results = self.results.get(test_id, [])

        if not test or not results:
            return {"error": "No data available"}

        # 按變體分組
        variant_results = defaultdict(list)
        for result in results:
            if metric_name in result.metrics:
                variant_results[result.variant_name].append(
                    result.metrics[metric_name]
                )

        analysis = {
            "test_id": test_id,
            "test_name": test.name,
            "total_samples": len(results),
            "variants": {}
        }

        control_data = None

        for variant in test.variants:
            data = variant_results.get(variant.name, [])

            if not data:
                continue

            stats = {
                "sample_size": len(data),
                "mean": statistics.mean(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data)
            }

            if variant.is_control:
                control_data = data
                stats["is_control"] = True
            elif control_data:
                # 計算相對提升
                lift = (stats["mean"] - statistics.mean(control_data)) / statistics.mean(control_data) * 100
                stats["lift"] = lift

                # 簡單的統計顯著性檢驗
                stats["is_significant"] = self._check_significance(
                    control_data, data, test.confidence_level
                )

            analysis["variants"][variant.name] = stats

        return analysis

    def _check_significance(
        self,
        control: List[float],
        variant: List[float],
        confidence: float
    ) -> bool:
        """檢查統計顯著性（簡化的 t 檢驗）"""
        if len(control) < 30 or len(variant) < 30:
            return False

        from scipy import stats
        t_stat, p_value = stats.ttest_ind(control, variant)
        return p_value < (1 - confidence)

    def get_winner(
        self,
        test_id: str,
        metric_name: str = "satisfaction"
    ) -> Optional[str]:
        """獲取獲勝變體"""
        analysis = self.analyze_test(test_id, metric_name)

        if "error" in analysis:
            return None

        best_variant = None
        best_mean = -float("inf")

        for name, stats in analysis["variants"].items():
            if stats.get("is_significant", False) or stats.get("is_control"):
                if stats["mean"] > best_mean:
                    best_mean = stats["mean"]
                    best_variant = name

        return best_variant


class MultiArmedBandit:
    """多臂老虎機（動態流量分配）"""

    def __init__(
        self,
        variants: List[str],
        exploration_rate: float = 0.1
    ):
        self.variants = variants
        self.exploration_rate = exploration_rate
        self.rewards: Dict[str, List[float]] = {v: [] for v in variants}
        self.counts: Dict[str, int] = {v: 0 for v in variants}

    def select_variant(self) -> str:
        """選擇變體（epsilon-greedy）"""
        if random.random() < self.exploration_rate:
            # 探索：隨機選擇
            return random.choice(self.variants)

        # 利用：選擇最佳
        best_variant = None
        best_mean = -float("inf")

        for variant in self.variants:
            if self.rewards[variant]:
                mean_reward = statistics.mean(self.rewards[variant])
                if mean_reward > best_mean:
                    best_mean = mean_reward
                    best_variant = variant
            else:
                # 未嘗試過的變體優先
                return variant

        return best_variant or self.variants[0]

    def update(self, variant: str, reward: float):
        """更新獎勵"""
        self.rewards[variant].append(reward)
        self.counts[variant] += 1

    def get_stats(self) -> Dict[str, Dict]:
        """獲取統計"""
        stats = {}
        for variant in self.variants:
            rewards = self.rewards[variant]
            stats[variant] = {
                "count": self.counts[variant],
                "mean_reward": statistics.mean(rewards) if rewards else 0,
                "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0
            }
        return stats


class ThompsonSampling:
    """Thompson Sampling（貝葉斯優化）"""

    def __init__(self, variants: List[str]):
        self.variants = variants
        # Beta 分布參數 (成功次數, 失敗次數)
        self.alpha: Dict[str, float] = {v: 1.0 for v in variants}
        self.beta: Dict[str, float] = {v: 1.0 for v in variants}

    def select_variant(self) -> str:
        """根據 Thompson Sampling 選擇變體"""
        samples = {}
        for variant in self.variants:
            # 從 Beta 分布採樣
            sample = random.betavariate(
                self.alpha[variant],
                self.beta[variant]
            )
            samples[variant] = sample

        return max(samples, key=samples.get)

    def update(self, variant: str, success: bool):
        """更新分布"""
        if success:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1

    def get_probabilities(self) -> Dict[str, float]:
        """獲取各變體的預期成功率"""
        return {
            v: self.alpha[v] / (self.alpha[v] + self.beta[v])
            for v in self.variants
        }
```

## 自動評估系統

```python
"""
Prompt 自動評估系統
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import re
from openai import OpenAI


@dataclass
class EvaluationResult:
    """評估結果"""
    prompt_id: str
    evaluator_name: str
    score: float
    details: Dict[str, Any]
    passed: bool
    feedback: str = ""


class PromptEvaluator(ABC):
    """Prompt 評估器基類"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        pass


class LengthEvaluator(PromptEvaluator):
    """長度評估器"""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 2000,
        prompt_id: str = ""
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.prompt_id = prompt_id

    @property
    def name(self) -> str:
        return "length"

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        length = len(response)
        in_range = self.min_length <= length <= self.max_length

        return EvaluationResult(
            prompt_id=self.prompt_id,
            evaluator_name=self.name,
            score=1.0 if in_range else 0.0,
            details={
                "length": length,
                "min": self.min_length,
                "max": self.max_length
            },
            passed=in_range,
            feedback=f"Response length: {length}" if in_range else f"Response length {length} out of range [{self.min_length}, {self.max_length}]"
        )


class FormatEvaluator(PromptEvaluator):
    """格式評估器"""

    def __init__(
        self,
        required_patterns: List[str] = None,
        forbidden_patterns: List[str] = None,
        prompt_id: str = ""
    ):
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []
        self.prompt_id = prompt_id

    @property
    def name(self) -> str:
        return "format"

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        issues = []
        score = 1.0

        # 檢查必需的模式
        for pattern in self.required_patterns:
            if not re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Missing required pattern: {pattern}")
                score -= 0.2

        # 檢查禁止的模式
        for pattern in self.forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Found forbidden pattern: {pattern}")
                score -= 0.3

        score = max(0, score)

        return EvaluationResult(
            prompt_id=self.prompt_id,
            evaluator_name=self.name,
            score=score,
            details={"issues": issues},
            passed=len(issues) == 0,
            feedback="; ".join(issues) if issues else "Format check passed"
        )


class LLMEvaluator(PromptEvaluator):
    """LLM 評估器"""

    def __init__(
        self,
        criteria: List[str],
        client: OpenAI = None,
        model: str = "gpt-4o-mini",
        prompt_id: str = ""
    ):
        self.criteria = criteria
        self.client = client or OpenAI()
        self.model = model
        self.prompt_id = prompt_id

    @property
    def name(self) -> str:
        return "llm_judge"

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        criteria_text = "\n".join(f"- {c}" for c in self.criteria)

        evaluation_prompt = f"""
請評估以下 AI 回應的品質。

原始 Prompt:
{prompt}

AI 回應:
{response}

{f"預期回應: {expected}" if expected else ""}

評估標準:
{criteria_text}

請為每個標準評分（1-5分），並提供整體評分和改進建議。

回應格式（JSON）:
{{
    "criteria_scores": {{"標準名稱": 分數}},
    "overall_score": 整體分數（1-5）,
    "feedback": "改進建議",
    "passed": true/false（是否達到合格標準，整體分數>=3.5為合格）
}}
"""

        response_obj = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            response_format={"type": "json_object"}
        )

        import json
        result = json.loads(response_obj.choices[0].message.content)

        return EvaluationResult(
            prompt_id=self.prompt_id,
            evaluator_name=self.name,
            score=result["overall_score"] / 5.0,
            details={
                "criteria_scores": result["criteria_scores"],
                "raw_score": result["overall_score"]
            },
            passed=result["passed"],
            feedback=result["feedback"]
        )


class SemanticSimilarityEvaluator(PromptEvaluator):
    """語義相似度評估器"""

    def __init__(
        self,
        threshold: float = 0.8,
        prompt_id: str = ""
    ):
        self.threshold = threshold
        self.prompt_id = prompt_id

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        if not expected:
            return EvaluationResult(
                prompt_id=self.prompt_id,
                evaluator_name=self.name,
                score=1.0,
                details={"message": "No expected response provided"},
                passed=True
            )

        # 計算語義相似度
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([response, expected])

        from numpy import dot
        from numpy.linalg import norm

        similarity = dot(embeddings[0], embeddings[1]) / (
            norm(embeddings[0]) * norm(embeddings[1])
        )

        return EvaluationResult(
            prompt_id=self.prompt_id,
            evaluator_name=self.name,
            score=float(similarity),
            details={
                "similarity": float(similarity),
                "threshold": self.threshold
            },
            passed=similarity >= self.threshold,
            feedback=f"Semantic similarity: {similarity:.2%}"
        )


class EvaluationPipeline:
    """評估管線"""

    def __init__(self, evaluators: List[PromptEvaluator] = None):
        self.evaluators = evaluators or []

    def add_evaluator(self, evaluator: PromptEvaluator):
        """添加評估器"""
        self.evaluators.append(evaluator)

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, EvaluationResult]:
        """執行所有評估"""
        results = {}

        for evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(prompt, response, expected, context)
                results[evaluator.name] = result
            except Exception as e:
                results[evaluator.name] = EvaluationResult(
                    prompt_id="",
                    evaluator_name=evaluator.name,
                    score=0.0,
                    details={"error": str(e)},
                    passed=False,
                    feedback=f"Evaluation failed: {e}"
                )

        return results

    def get_overall_score(
        self,
        results: Dict[str, EvaluationResult],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """計算加權總分"""
        if not results:
            return 0.0

        if weights:
            total_weight = sum(weights.get(name, 1.0) for name in results)
            weighted_sum = sum(
                results[name].score * weights.get(name, 1.0)
                for name in results
            )
            return weighted_sum / total_weight

        return sum(r.score for r in results.values()) / len(results)

    def is_passing(
        self,
        results: Dict[str, EvaluationResult],
        require_all: bool = True
    ) -> bool:
        """檢查是否通過"""
        if require_all:
            return all(r.passed for r in results.values())
        return any(r.passed for r in results.values())
```

## 生產環境管理

```python
"""
Prompt 生產環境管理
"""
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import redis
import json


class PromptCache:
    """Prompt 快取"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600
    ):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl

    def _cache_key(self, prompt_name: str, version: str) -> str:
        """生成快取鍵"""
        return f"prompt:{prompt_name}:{version}"

    def get(
        self,
        prompt_name: str,
        version: str = "production"
    ) -> Optional[Dict]:
        """獲取快取的 Prompt"""
        key = self._cache_key(prompt_name, version)
        data = self.redis.get(key)

        if data:
            return json.loads(data)
        return None

    def set(
        self,
        prompt_name: str,
        version: str,
        prompt_data: Dict
    ):
        """設置快取"""
        key = self._cache_key(prompt_name, version)
        self.redis.setex(key, self.ttl, json.dumps(prompt_data))

    def invalidate(self, prompt_name: str, version: str = None):
        """使快取失效"""
        if version:
            key = self._cache_key(prompt_name, version)
            self.redis.delete(key)
        else:
            # 使所有版本失效
            pattern = f"prompt:{prompt_name}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)


class PromptRouter:
    """Prompt 路由器"""

    def __init__(
        self,
        registry: "PromptRegistry",
        cache: Optional[PromptCache] = None
    ):
        self.registry = registry
        self.cache = cache
        self.rollouts: Dict[str, Dict] = {}

    def configure_rollout(
        self,
        prompt_name: str,
        versions: Dict[str, float]
    ):
        """配置版本分流

        Args:
            prompt_name: Prompt 名稱
            versions: 版本到流量百分比的映射，如 {"v1.0": 0.9, "v2.0": 0.1}
        """
        total = sum(versions.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Rollout percentages must sum to 1.0, got {total}")

        self.rollouts[prompt_name] = versions

    def get_prompt(
        self,
        prompt_name: str,
        user_id: Optional[str] = None
    ) -> Optional["PromptVersion"]:
        """獲取 Prompt（考慮分流）"""
        rollout = self.rollouts.get(prompt_name)

        if not rollout:
            # 沒有分流配置，返回生產版本
            return self.registry.get_production(prompt_name)

        # 確定性分流
        if user_id:
            hash_value = int(hashlib.sha256(
                f"{prompt_name}:{user_id}".encode()
            ).hexdigest(), 16)
            bucket = (hash_value % 100) / 100.0
        else:
            import random
            bucket = random.random()

        cumulative = 0
        for version, percentage in rollout.items():
            cumulative += percentage
            if bucket < cumulative:
                # 嘗試從快取獲取
                if self.cache:
                    cached = self.cache.get(prompt_name, version)
                    if cached:
                        return PromptVersion.from_dict(cached)

                # 從註冊表獲取
                prompt = self.registry.get(prompt_name, version)

                # 更新快取
                if prompt and self.cache:
                    self.cache.set(prompt_name, version, prompt.to_dict())

                return prompt

        return self.registry.get_production(prompt_name)


class PromptMonitor:
    """Prompt 監控"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    def record_usage(
        self,
        prompt_name: str,
        version: str,
        latency_ms: float,
        tokens_used: int,
        success: bool
    ):
        """記錄使用情況"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H")
        key = f"prompt_metrics:{prompt_name}:{version}:{timestamp}"

        pipe = self.redis.pipeline()
        pipe.hincrby(key, "count", 1)
        pipe.hincrbyfloat(key, "total_latency", latency_ms)
        pipe.hincrby(key, "total_tokens", tokens_used)
        pipe.hincrby(key, "success" if success else "failure", 1)
        pipe.expire(key, 86400 * 7)  # 保留 7 天
        pipe.execute()

    def get_metrics(
        self,
        prompt_name: str,
        version: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """獲取指標"""
        metrics = {
            "total_count": 0,
            "total_latency": 0,
            "total_tokens": 0,
            "success_count": 0,
            "failure_count": 0
        }

        now = datetime.now()
        for i in range(hours):
            timestamp = (now - timedelta(hours=i)).strftime("%Y-%m-%d-%H")
            key = f"prompt_metrics:{prompt_name}:{version}:{timestamp}"

            data = self.redis.hgetall(key)
            if data:
                metrics["total_count"] += int(data.get(b"count", 0))
                metrics["total_latency"] += float(data.get(b"total_latency", 0))
                metrics["total_tokens"] += int(data.get(b"total_tokens", 0))
                metrics["success_count"] += int(data.get(b"success", 0))
                metrics["failure_count"] += int(data.get(b"failure", 0))

        if metrics["total_count"] > 0:
            metrics["avg_latency"] = metrics["total_latency"] / metrics["total_count"]
            metrics["avg_tokens"] = metrics["total_tokens"] / metrics["total_count"]
            metrics["success_rate"] = metrics["success_count"] / metrics["total_count"]
        else:
            metrics["avg_latency"] = 0
            metrics["avg_tokens"] = 0
            metrics["success_rate"] = 0

        return metrics

    def get_comparison(
        self,
        prompt_name: str,
        versions: List[str],
        hours: int = 24
    ) -> Dict[str, Dict]:
        """版本比較"""
        comparison = {}
        for version in versions:
            comparison[version] = self.get_metrics(prompt_name, version, hours)
        return comparison


class PromptRollbackManager:
    """Prompt 回滾管理"""

    def __init__(
        self,
        registry: "PromptRegistry",
        router: PromptRouter,
        monitor: PromptMonitor
    ):
        self.registry = registry
        self.router = router
        self.monitor = monitor
        self.rollback_history: List[Dict] = []

    def check_health(
        self,
        prompt_name: str,
        version: str,
        success_threshold: float = 0.95,
        latency_threshold: float = 5000
    ) -> Dict[str, Any]:
        """健康檢查"""
        metrics = self.monitor.get_metrics(prompt_name, version, hours=1)

        health = {
            "healthy": True,
            "issues": []
        }

        if metrics["total_count"] < 10:
            health["issues"].append("Insufficient data")
            return health

        if metrics["success_rate"] < success_threshold:
            health["healthy"] = False
            health["issues"].append(
                f"Success rate {metrics['success_rate']:.2%} below threshold {success_threshold:.2%}"
            )

        if metrics["avg_latency"] > latency_threshold:
            health["healthy"] = False
            health["issues"].append(
                f"Latency {metrics['avg_latency']:.0f}ms above threshold {latency_threshold}ms"
            )

        return health

    def auto_rollback(
        self,
        prompt_name: str,
        current_version: str
    ) -> Optional[str]:
        """自動回滾"""
        health = self.check_health(prompt_name, current_version)

        if not health["healthy"]:
            # 找到上一個健康版本
            versions = self.registry.list_versions(prompt_name)
            current_idx = next(
                (i for i, v in enumerate(versions) if v.version == current_version),
                -1
            )

            if current_idx > 0:
                previous_version = versions[current_idx + 1]

                # 執行回滾
                self.router.configure_rollout(prompt_name, {
                    previous_version.version: 1.0
                })

                self.rollback_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt_name": prompt_name,
                    "from_version": current_version,
                    "to_version": previous_version.version,
                    "reason": health["issues"]
                })

                return previous_version.version

        return None
```

## CLI 工具

```python
"""
Prompt 管理 CLI 工具
"""
import click
from rich.console import Console
from rich.table import Table


console = Console()


@click.group()
def cli():
    """Prompt 管理工具"""
    pass


@cli.command()
@click.argument("name")
@click.option("--version", "-v", default=None, help="版本號")
def get(name: str, version: str):
    """獲取 Prompt"""
    from prompt_manager import PromptRegistry

    registry = PromptRegistry()
    prompt = registry.get(name, version)

    if prompt:
        console.print(f"[bold green]Prompt: {prompt.name}[/]")
        console.print(f"Version: {prompt.version}")
        console.print(f"Status: {prompt.status.value}")
        console.print(f"Model: {prompt.model}")
        console.print("\n[bold]Template:[/]")
        console.print(prompt.template)
    else:
        console.print(f"[red]Prompt '{name}' not found[/]")


@cli.command()
@click.argument("name")
def list_versions(name: str):
    """列出所有版本"""
    from prompt_manager import PromptRegistry

    registry = PromptRegistry()
    versions = registry.list_versions(name)

    table = Table(title=f"Versions of {name}")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Model")

    for v in versions:
        table.add_row(
            v.version,
            v.status.value,
            v.created_at.strftime("%Y-%m-%d %H:%M"),
            v.model
        )

    console.print(table)


@cli.command()
@click.argument("name")
@click.argument("version")
@click.option("--status", "-s", type=click.Choice(["draft", "testing", "approved", "production", "deprecated"]))
def set_status(name: str, version: str, status: str):
    """設置 Prompt 狀態"""
    from prompt_manager import PromptRegistry, PromptStatus

    registry = PromptRegistry()
    prompt = registry.get(name, version)

    if prompt:
        prompt.status = PromptStatus(status)
        registry.register(prompt)
        console.print(f"[green]Updated {name} v{version} to {status}[/]")
    else:
        console.print(f"[red]Prompt not found[/]")


@cli.command()
@click.argument("name")
@click.option("--hours", "-h", default=24, help="查看多少小時的資料")
def metrics(name: str, hours: int):
    """查看 Prompt 指標"""
    from prompt_manager import PromptMonitor, PromptRegistry

    registry = PromptRegistry()
    monitor = PromptMonitor()

    versions = registry.list_versions(name)[:5]  # 最近 5 個版本

    table = Table(title=f"Metrics for {name} (last {hours}h)")
    table.add_column("Version")
    table.add_column("Requests")
    table.add_column("Avg Latency")
    table.add_column("Success Rate")
    table.add_column("Avg Tokens")

    for v in versions:
        m = monitor.get_metrics(name, v.version, hours)
        table.add_row(
            v.version,
            str(m["total_count"]),
            f"{m['avg_latency']:.0f}ms",
            f"{m['success_rate']:.1%}",
            f"{m['avg_tokens']:.0f}"
        )

    console.print(table)


@cli.command()
@click.argument("file_path")
def import_yaml(file_path: str):
    """從 YAML 導入 Prompt"""
    from prompt_manager import GitPromptManager, PromptFile

    with open(file_path) as f:
        content = f.read()

    prompt = PromptFile.from_yaml(content)
    manager = GitPromptManager()
    path = manager.save_prompt(prompt)

    console.print(f"[green]Imported {prompt.name} v{prompt.version}[/]")
    console.print(f"Saved to: {path}")


if __name__ == "__main__":
    cli()
```

## 最佳實踐

```yaml
# Prompt 工程化最佳實踐

版本控制:
  命名規範:
    - 使用語義化版本: major.minor.patch
    - major: 重大邏輯變更
    - minor: 功能增強
    - patch: 小修改/錯誤修復

  文件結構:
    prompts/
    ├── customer_support/
    │   ├── 1.0.0.yaml
    │   ├── 1.1.0.yaml
    │   └── 2.0.0.yaml
    ├── code_review/
    │   └── 1.0.0.yaml
    └── README.md

  變更追蹤:
    - 每次變更都需要描述原因
    - 記錄 A/B 測試結果
    - 保留歷史版本

測試策略:
  單元測試:
    - 格式驗證
    - 長度檢查
    - 關鍵詞包含

  整合測試:
    - 端到端回應測試
    - 與實際 LLM 交互
    - 邊界情況測試

  回歸測試:
    - 每次版本更新前運行
    - 確保不影響現有功能
    - 自動化測試套件

部署流程:
  環境區分:
    - development: 開發測試
    - staging: 預發布驗證
    - production: 生產環境

  金絲雀發布:
    1. 新版本 5% 流量
    2. 監控 1 小時
    3. 逐步增加至 100%
    4. 準備回滾方案

  自動回滾:
    - 設置成功率閾值
    - 設置延遲閾值
    - 自動監控和回滾

監控指標:
  關鍵指標:
    - 請求量
    - 成功率
    - 平均延遲
    - Token 使用量
    - 用戶滿意度

  警報設置:
    - 成功率 < 95%
    - 延遲 > 5s
    - 錯誤激增

  儀表板:
    - 實時監控
    - 版本比較
    - 趨勢分析
```

## 相關資源

- [LangSmith](https://smith.langchain.com/) - LangChain 官方 Prompt 管理平台
- [Weights & Biases Prompts](https://wandb.ai/site/prompts) - W&B Prompt 追蹤
- [Promptfoo](https://promptfoo.dev/) - 開源 Prompt 測試工具
- [Helicone](https://helicone.ai/) - LLM 可觀測性平台
