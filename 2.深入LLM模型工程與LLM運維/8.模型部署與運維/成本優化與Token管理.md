# 成本優化與 Token 管理 (Cost Optimization and Token Management)

## 概述

在生產環境中，AI 成本可能快速增長。有效的成本管理和 Token 優化是維持可持續 AI 應用的關鍵。

## 成本結構分析

```
┌─────────────────────────────────────────────────────────────┐
│                    AI 應用成本結構                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  API 成本 (通常 60-80%)                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 輸入 Token        輸出 Token        模型選擇          │  │
│  │ $0.15-15/1M      $0.60-60/1M      GPT-4 vs Mini     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  基礎設施成本 (15-25%)                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 向量資料庫        運算資源          儲存              │  │
│  │ Pinecone/        GPU/CPU          S3/GCS            │  │
│  │ Weaviate         instances        storage           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  維運成本 (5-15%)                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 監控              日誌             人力維護           │  │
│  │ Datadog/         CloudWatch/      DevOps            │  │
│  │ Prometheus       ELK              team              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. Token 管理與計算

### Token 計數器

```python
import tiktoken
from typing import Union, List
from functools import lru_cache

class TokenCounter:
    """Token 計數器"""

    # 主流模型的 token 編碼器
    ENCODERS = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "claude": "cl100k_base",  # 近似值
    }

    def __init__(self, model: str = "gpt-4o"):
        encoding_name = self.ENCODERS.get(model, "cl100k_base")
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.model = model

    def count(self, text: str) -> int:
        """計算 token 數量"""
        return len(self.encoder.encode(text))

    def count_messages(self, messages: List[dict]) -> int:
        """計算對話訊息的 token 數量"""
        total = 0

        for message in messages:
            # 每個訊息有基礎 token 開銷
            total += 4  # <|im_start|>, role, \n, <|im_end|>
            total += self.count(message.get("role", ""))
            total += self.count(message.get("content", ""))

        total += 2  # 對話結尾

        return total

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = None
    ) -> dict:
        """估算成本"""
        model = model or self.model

        # 2024-2025 定價（美元）
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "claude-3-opus": {"input": 15.00, "output": 75.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        model_pricing = pricing.get(model, pricing["gpt-4o"])

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": model
        }

    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
        truncate_from: str = "end"
    ) -> str:
        """截斷文本到指定 token 數"""
        tokens = self.encoder.encode(text)

        if len(tokens) <= max_tokens:
            return text

        if truncate_from == "start":
            truncated = tokens[-max_tokens:]
        else:
            truncated = tokens[:max_tokens]

        return self.encoder.decode(truncated)

# 使用範例
counter = TokenCounter("gpt-4o")

text = "這是一段測試文字，用於計算 token 數量。"
tokens = counter.count(text)
print(f"Token 數: {tokens}")

# 估算成本
cost = counter.estimate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="gpt-4o-mini"
)
print(f"預估成本: ${cost['total_cost']:.4f}")
```

### 成本監控系統

```python
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class UsageRecord:
    """使用記錄"""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    request_type: str
    metadata: dict = field(default_factory=dict)

class CostMonitor:
    """成本監控器"""

    def __init__(
        self,
        budget_daily: float = 100.0,
        budget_monthly: float = 3000.0,
        storage_path: str = "./cost_data"
    ):
        self.budget_daily = budget_daily
        self.budget_monthly = budget_monthly
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.records: List[UsageRecord] = []
        self.token_counter = TokenCounter()

        self._load_records()

    def _load_records(self):
        """載入歷史記錄"""
        records_file = self.storage_path / "records.json"
        if records_file.exists():
            with open(records_file, 'r') as f:
                data = json.load(f)
                self.records = [
                    UsageRecord(
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        **{k: v for k, v in r.items() if k != "timestamp"}
                    )
                    for r in data
                ]

    def _save_records(self):
        """儲存記錄"""
        records_file = self.storage_path / "records.json"
        data = [
            {
                **r.__dict__,
                "timestamp": r.timestamp.isoformat()
            }
            for r in self.records[-10000:]  # 只保留最近 10000 筆
        ]
        with open(records_file, 'w') as f:
            json.dump(data, f)

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str = "chat",
        metadata: Optional[dict] = None
    ):
        """記錄使用量"""
        cost_info = self.token_counter.estimate_cost(
            input_tokens, output_tokens, model
        )

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost_info["total_cost"],
            request_type=request_type,
            metadata=metadata or {}
        )

        self.records.append(record)
        self._save_records()

        # 檢查預算
        self._check_budget_alerts()

        return record

    def get_daily_usage(
        self,
        date: Optional[datetime] = None
    ) -> dict:
        """取得每日使用量"""
        date = date or datetime.now()
        start = datetime(date.year, date.month, date.day)
        end = start + timedelta(days=1)

        daily_records = [
            r for r in self.records
            if start <= r.timestamp < end
        ]

        return self._aggregate_records(daily_records)

    def get_monthly_usage(
        self,
        year: int = None,
        month: int = None
    ) -> dict:
        """取得每月使用量"""
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        monthly_records = [
            r for r in self.records
            if r.timestamp.year == year and r.timestamp.month == month
        ]

        return self._aggregate_records(monthly_records)

    def _aggregate_records(
        self,
        records: List[UsageRecord]
    ) -> dict:
        """彙總記錄"""
        if not records:
            return {
                "total_cost": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "request_count": 0,
                "by_model": {},
                "by_type": {}
            }

        by_model = {}
        by_type = {}

        for r in records:
            # 按模型
            if r.model not in by_model:
                by_model[r.model] = {"cost": 0, "requests": 0}
            by_model[r.model]["cost"] += r.cost
            by_model[r.model]["requests"] += 1

            # 按類型
            if r.request_type not in by_type:
                by_type[r.request_type] = {"cost": 0, "requests": 0}
            by_type[r.request_type]["cost"] += r.cost
            by_type[r.request_type]["requests"] += 1

        return {
            "total_cost": sum(r.cost for r in records),
            "total_input_tokens": sum(r.input_tokens for r in records),
            "total_output_tokens": sum(r.output_tokens for r in records),
            "request_count": len(records),
            "by_model": by_model,
            "by_type": by_type
        }

    def _check_budget_alerts(self):
        """檢查預算警報"""
        daily = self.get_daily_usage()
        monthly = self.get_monthly_usage()

        alerts = []

        if daily["total_cost"] > self.budget_daily * 0.8:
            alerts.append(f"每日預算使用 {daily['total_cost']/self.budget_daily*100:.1f}%")

        if monthly["total_cost"] > self.budget_monthly * 0.8:
            alerts.append(f"每月預算使用 {monthly['total_cost']/self.budget_monthly*100:.1f}%")

        for alert in alerts:
            print(f"⚠️ 預算警告: {alert}")

        return alerts

    def get_optimization_suggestions(self) -> List[str]:
        """取得優化建議"""
        monthly = self.get_monthly_usage()
        suggestions = []

        # 分析模型使用
        for model, stats in monthly.get("by_model", {}).items():
            if "gpt-4o" in model and stats["requests"] > 100:
                suggestions.append(
                    f"考慮將部分 {model} 請求降級到 gpt-4o-mini，"
                    f"可節省約 {stats['cost'] * 0.9:.2f} 美元"
                )

        # 分析請求類型
        if monthly.get("by_type", {}).get("embedding", {}).get("requests", 0) > 1000:
            suggestions.append(
                "考慮實作嵌入快取，減少重複的 embedding 請求"
            )

        return suggestions

# 使用範例
monitor = CostMonitor(budget_daily=50, budget_monthly=1000)

# 記錄使用
monitor.record_usage(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    request_type="chat"
)

# 查看使用量
daily = monitor.get_daily_usage()
print(f"今日花費: ${daily['total_cost']:.2f}")

# 取得優化建議
suggestions = monitor.get_optimization_suggestions()
```

## 2. 快取策略

### Prompt 快取

```python
import hashlib
import json
from typing import Optional, Any
from datetime import datetime, timedelta
import redis

class PromptCache:
    """Prompt 快取系統"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,  # 1 小時
        max_cache_size: int = 10000
    ):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size

        # 統計
        self.hits = 0
        self.misses = 0

    def _generate_key(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """生成快取鍵"""
        # 只有 temperature=0 才能安全快取
        if temperature > 0:
            return None

        content = json.dumps({
            "prompt": prompt,
            "model": model,
            "kwargs": kwargs
        }, sort_keys=True)

        return f"prompt_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        **kwargs
    ) -> Optional[str]:
        """取得快取"""
        key = self._generate_key(prompt, model, temperature, **kwargs)

        if not key:
            return None

        cached = self.redis.get(key)

        if cached:
            self.hits += 1
            return json.loads(cached)["response"]
        else:
            self.misses += 1
            return None

    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        temperature: float = 0.0,
        ttl: Optional[int] = None,
        **kwargs
    ):
        """設定快取"""
        key = self._generate_key(prompt, model, temperature, **kwargs)

        if not key:
            return

        data = {
            "response": response,
            "cached_at": datetime.now().isoformat(),
            "model": model
        }

        self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(data)
        )

    def get_stats(self) -> dict:
        """取得統計"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": self.redis.dbsize()
        }

    def clear(self):
        """清除快取"""
        keys = self.redis.keys("prompt_cache:*")
        if keys:
            self.redis.delete(*keys)

# 使用範例
cache = PromptCache()

# 檢查快取
cached_response = cache.get(
    prompt="什麼是機器學習？",
    model="gpt-4o-mini"
)

if cached_response:
    print(f"快取命中: {cached_response}")
else:
    # 調用 API
    response = "機器學習是..."  # 實際 API 調用

    # 儲存快取
    cache.set(
        prompt="什麼是機器學習？",
        model="gpt-4o-mini",
        response=response
    )
```

### 語義快取

```python
from openai import OpenAI
import numpy as np
from typing import Optional, Tuple
import chromadb

class SemanticCache:
    """語義快取 - 根據語義相似度匹配"""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        persist_dir: str = "./semantic_cache"
    ):
        self.client = OpenAI()
        self.similarity_threshold = similarity_threshold

        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name="semantic_cache",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> list[float]:
        """取得文字嵌入"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float]
    ) -> float:
        """計算餘弦相似度"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(
        self,
        query: str,
        model: str
    ) -> Tuple[Optional[str], float]:
        """語義搜尋快取"""
        query_embedding = self._get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where={"model": model}
        )

        if not results['ids'][0]:
            return None, 0.0

        # 計算相似度
        similarity = 1 - results['distances'][0][0]

        if similarity >= self.similarity_threshold:
            response = results['metadatas'][0][0]['response']
            return response, similarity

        return None, similarity

    def set(
        self,
        query: str,
        response: str,
        model: str
    ):
        """儲存到語義快取"""
        query_embedding = self._get_embedding(query)

        self.collection.add(
            ids=[hashlib.md5(query.encode()).hexdigest()],
            embeddings=[query_embedding],
            documents=[query],
            metadatas=[{
                "response": response,
                "model": model,
                "cached_at": datetime.now().isoformat()
            }]
        )

# 使用範例
semantic_cache = SemanticCache(similarity_threshold=0.92)

# 查詢快取
query = "解釋一下什麼是深度學習"
cached, similarity = semantic_cache.get(query, "gpt-4o")

if cached:
    print(f"語義快取命中 (相似度: {similarity:.2f})")
    print(cached)
else:
    # 即使問法不同，語義相似的查詢也能命中
    # 例如 "深度學習是什麼？" 和 "請解釋深度學習" 可能命中同一快取
    pass
```

## 3. 模型路由與降級

### 智能模型選擇

```python
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class ModelConfig:
    name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_context: int
    capabilities: list[str]

class ModelRouter:
    """智能模型路由器"""

    MODELS = {
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            cost_per_1k_input=2.50,
            cost_per_1k_output=10.00,
            max_context=128000,
            capabilities=["reasoning", "coding", "analysis", "creative"]
        ),
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            max_context=128000,
            capabilities=["general", "coding", "translation"]
        ),
        "claude-3-haiku": ModelConfig(
            name="claude-3-haiku",
            cost_per_1k_input=0.25,
            cost_per_1k_output=1.25,
            max_context=200000,
            capabilities=["general", "fast", "long_context"]
        )
    }

    def __init__(self):
        self.complexity_classifier = self._default_complexity_classifier

    def _default_complexity_classifier(
        self,
        prompt: str,
        **kwargs
    ) -> TaskComplexity:
        """預設複雜度分類器"""
        # 簡單啟發式規則
        prompt_lower = prompt.lower()

        # 複雜任務指標
        complex_indicators = [
            "分析", "推理", "解釋為什麼", "比較", "評估",
            "analyze", "reason", "explain why", "compare", "evaluate"
        ]

        # 簡單任務指標
        simple_indicators = [
            "翻譯", "總結", "列出", "格式化",
            "translate", "summarize", "list", "format"
        ]

        if any(ind in prompt_lower for ind in complex_indicators):
            return TaskComplexity.COMPLEX

        if any(ind in prompt_lower for ind in simple_indicators):
            return TaskComplexity.SIMPLE

        # 根據長度判斷
        if len(prompt) > 2000:
            return TaskComplexity.COMPLEX
        elif len(prompt) < 200:
            return TaskComplexity.SIMPLE

        return TaskComplexity.MEDIUM

    def route(
        self,
        prompt: str,
        required_capabilities: Optional[list[str]] = None,
        max_cost_per_request: Optional[float] = None,
        prefer_speed: bool = False
    ) -> str:
        """路由到最佳模型"""
        complexity = self.complexity_classifier(prompt)

        # 根據複雜度選擇候選模型
        if complexity == TaskComplexity.SIMPLE:
            candidates = ["gpt-4o-mini", "claude-3-haiku"]
        elif complexity == TaskComplexity.MEDIUM:
            candidates = ["gpt-4o-mini", "gpt-4o"]
        else:
            candidates = ["gpt-4o", "claude-sonnet-4-20250514"]

        # 過濾具備所需能力的模型
        if required_capabilities:
            candidates = [
                m for m in candidates
                if all(
                    cap in self.MODELS[m].capabilities
                    for cap in required_capabilities
                )
            ]

        # 成本過濾
        if max_cost_per_request:
            # 估算成本（假設 1000 token 輸入，500 輸出）
            candidates = [
                m for m in candidates
                if (self.MODELS[m].cost_per_1k_input +
                    self.MODELS[m].cost_per_1k_output * 0.5) < max_cost_per_request
            ]

        # 速度優先
        if prefer_speed:
            if "claude-3-haiku" in candidates:
                return "claude-3-haiku"
            if "gpt-4o-mini" in candidates:
                return "gpt-4o-mini"

        # 返回第一個候選（成本最優）
        return candidates[0] if candidates else "gpt-4o-mini"

    def estimate_savings(
        self,
        prompts: list[str],
        default_model: str = "gpt-4o"
    ) -> dict:
        """估算使用路由的節省"""
        default_cost = 0
        routed_cost = 0

        for prompt in prompts:
            # 預設成本
            default_config = self.MODELS[default_model]
            default_cost += (
                default_config.cost_per_1k_input +
                default_config.cost_per_1k_output * 0.5
            )

            # 路由成本
            routed_model = self.route(prompt)
            routed_config = self.MODELS[routed_model]
            routed_cost += (
                routed_config.cost_per_1k_input +
                routed_config.cost_per_1k_output * 0.5
            )

        savings = default_cost - routed_cost
        savings_pct = (savings / default_cost) * 100 if default_cost > 0 else 0

        return {
            "default_cost": default_cost,
            "routed_cost": routed_cost,
            "savings": savings,
            "savings_percentage": savings_pct
        }

# 使用範例
router = ModelRouter()

# 簡單查詢 -> 使用便宜模型
model = router.route("幫我翻譯這句話成英文")
print(f"簡單任務使用: {model}")  # gpt-4o-mini

# 複雜查詢 -> 使用強力模型
model = router.route("分析這段程式碼的時間複雜度並解釋優化策略")
print(f"複雜任務使用: {model}")  # gpt-4o

# 估算節省
prompts = [
    "翻譯這段文字",
    "分析市場趨勢",
    "總結這篇文章",
    "解釋量子計算原理"
]
savings = router.estimate_savings(prompts)
print(f"預估節省: {savings['savings_percentage']:.1f}%")
```

### 自動降級策略

```python
from typing import Callable, Optional
import time
from functools import wraps

class ModelFallback:
    """模型降級策略"""

    def __init__(
        self,
        primary_model: str = "gpt-4o",
        fallback_chain: list[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.primary_model = primary_model
        self.fallback_chain = fallback_chain or [
            "gpt-4o-mini",
            "claude-3-haiku"
        ]
        self.max_retries = max_retries
        self.timeout = timeout

        # 統計
        self.primary_calls = 0
        self.fallback_calls = 0
        self.failures = 0

    def with_fallback(
        self,
        call_func: Callable,
        *args,
        **kwargs
    ):
        """帶降級的調用"""
        models_to_try = [self.primary_model] + self.fallback_chain

        last_error = None

        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    # 更新模型參數
                    kwargs["model"] = model

                    result = call_func(*args, **kwargs)

                    # 統計
                    if model == self.primary_model:
                        self.primary_calls += 1
                    else:
                        self.fallback_calls += 1

                    return result

                except Exception as e:
                    last_error = e

                    # 判斷是否可重試
                    if self._is_retryable(e):
                        time.sleep(2 ** attempt)  # 指數退避
                        continue
                    else:
                        break  # 嘗試下一個模型

        self.failures += 1
        raise last_error

    def _is_retryable(self, error: Exception) -> bool:
        """判斷錯誤是否可重試"""
        retryable_errors = [
            "rate_limit",
            "timeout",
            "server_error",
            "503",
            "429"
        ]
        error_str = str(error).lower()
        return any(e in error_str for e in retryable_errors)

    def get_stats(self) -> dict:
        """取得統計"""
        total = self.primary_calls + self.fallback_calls
        fallback_rate = self.fallback_calls / total if total > 0 else 0

        return {
            "primary_calls": self.primary_calls,
            "fallback_calls": self.fallback_calls,
            "failures": self.failures,
            "fallback_rate": fallback_rate
        }

# 使用裝飾器
def with_model_fallback(
    primary_model: str = "gpt-4o",
    fallback_chain: list[str] = None
):
    """模型降級裝飾器"""
    fallback = ModelFallback(primary_model, fallback_chain)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return fallback.with_fallback(func, *args, **kwargs)
        wrapper.get_stats = fallback.get_stats
        return wrapper

    return decorator

# 使用範例
@with_model_fallback(primary_model="gpt-4o")
def call_llm(prompt: str, model: str = "gpt-4o"):
    # 實際的 API 調用
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# 調用時自動處理降級
result = call_llm("你好")
```

## 4. 批次處理優化

### 批次請求處理

```python
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from dataclasses import dataclass
import time

@dataclass
class BatchRequest:
    """批次請求"""
    id: str
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 500

@dataclass
class BatchResult:
    """批次結果"""
    id: str
    response: str
    tokens_used: int
    cost: float
    duration: float

class BatchProcessor:
    """批次處理器"""

    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limit_rpm: int = 500
    ):
        self.client = AsyncOpenAI()
        self.max_concurrent = max_concurrent
        self.rate_limit_rpm = rate_limit_rpm
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 速率限制
        self.request_times: List[float] = []

    async def _rate_limit(self):
        """速率限制"""
        now = time.time()

        # 清理舊記錄
        self.request_times = [
            t for t in self.request_times
            if now - t < 60
        ]

        # 如果達到限制，等待
        if len(self.request_times) >= self.rate_limit_rpm:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.request_times.append(time.time())

    async def _process_single(
        self,
        request: BatchRequest
    ) -> BatchResult:
        """處理單個請求"""
        async with self.semaphore:
            await self._rate_limit()

            start_time = time.time()

            response = await self.client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens
            )

            duration = time.time() - start_time

            # 計算成本
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._calculate_cost(
                request.model, input_tokens, output_tokens
            )

            return BatchResult(
                id=request.id,
                response=response.choices[0].message.content,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                duration=duration
            )

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """計算成本"""
        pricing = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
        }
        input_rate, output_rate = pricing.get(model, (1.0, 2.0))

        return (
            (input_tokens / 1000) * input_rate +
            (output_tokens / 1000) * output_rate
        )

    async def process_batch(
        self,
        requests: List[BatchRequest]
    ) -> List[BatchResult]:
        """處理批次請求"""
        tasks = [
            self._process_single(req)
            for req in requests
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 處理錯誤
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(BatchResult(
                    id=requests[i].id,
                    response=f"Error: {str(result)}",
                    tokens_used=0,
                    cost=0,
                    duration=0
                ))
            else:
                processed_results.append(result)

        return processed_results

    def process_batch_sync(
        self,
        requests: List[BatchRequest]
    ) -> List[BatchResult]:
        """同步處理批次"""
        return asyncio.run(self.process_batch(requests))

# 使用範例
processor = BatchProcessor(max_concurrent=5)

requests = [
    BatchRequest(id=f"req_{i}", prompt=f"問題 {i}")
    for i in range(10)
]

results = processor.process_batch_sync(requests)

total_cost = sum(r.cost for r in results)
total_time = max(r.duration for r in results)
print(f"批次處理完成: {len(results)} 請求, 成本 ${total_cost:.4f}, 耗時 {total_time:.2f}s")
```

## 5. 預算控制與警報

### 預算管理系統

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable
from enum import Enum

class BudgetAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    DOWNGRADE = "downgrade"

@dataclass
class BudgetPolicy:
    """預算策略"""
    daily_limit: float
    monthly_limit: float
    warning_threshold: float = 0.8
    block_threshold: float = 1.0
    downgrade_threshold: float = 0.9

class BudgetGuard:
    """預算守衛"""

    def __init__(
        self,
        policy: BudgetPolicy,
        cost_tracker: CostMonitor,
        alert_callback: Optional[Callable] = None
    ):
        self.policy = policy
        self.tracker = cost_tracker
        self.alert_callback = alert_callback or self._default_alert

    def _default_alert(self, message: str, severity: str):
        """預設警報"""
        print(f"[{severity.upper()}] {message}")

    def check_budget(self) -> BudgetAction:
        """檢查預算狀態"""
        daily = self.tracker.get_daily_usage()
        monthly = self.tracker.get_monthly_usage()

        daily_usage = daily["total_cost"] / self.policy.daily_limit
        monthly_usage = monthly["total_cost"] / self.policy.monthly_limit

        max_usage = max(daily_usage, monthly_usage)

        if max_usage >= self.policy.block_threshold:
            self.alert_callback(
                f"預算已超限！日: {daily_usage:.0%}, 月: {monthly_usage:.0%}",
                "critical"
            )
            return BudgetAction.BLOCK

        if max_usage >= self.policy.downgrade_threshold:
            self.alert_callback(
                f"預算接近上限，自動降級模型",
                "warning"
            )
            return BudgetAction.DOWNGRADE

        if max_usage >= self.policy.warning_threshold:
            self.alert_callback(
                f"預算使用較高：日 {daily_usage:.0%}, 月 {monthly_usage:.0%}",
                "warning"
            )
            return BudgetAction.WARN

        return BudgetAction.ALLOW

    def guard_request(
        self,
        estimated_cost: float,
        model: str
    ) -> tuple[BudgetAction, str]:
        """守衛請求"""
        action = self.check_budget()

        if action == BudgetAction.BLOCK:
            return action, None

        if action == BudgetAction.DOWNGRADE:
            # 自動降級
            downgrade_map = {
                "gpt-4o": "gpt-4o-mini",
                "claude-sonnet-4-20250514": "claude-3-haiku",
            }
            model = downgrade_map.get(model, model)

        return action, model

# 使用範例
policy = BudgetPolicy(
    daily_limit=50.0,
    monthly_limit=1000.0,
    warning_threshold=0.7
)

tracker = CostMonitor()
guard = BudgetGuard(policy, tracker)

# 在每次請求前檢查
action, model = guard.guard_request(estimated_cost=0.01, model="gpt-4o")

if action == BudgetAction.BLOCK:
    print("請求被阻止：超出預算")
elif action == BudgetAction.DOWNGRADE:
    print(f"自動降級到: {model}")
else:
    print(f"使用模型: {model}")
```

## 成本優化檢查清單

```markdown
## 成本優化檢查清單

### Token 優化
- [ ] 使用 token 計數器監控使用量
- [ ] 截斷過長的輸入
- [ ] 使用系統提示精簡化
- [ ] 移除冗餘的上下文

### 快取策略
- [ ] 實作 prompt 快取
- [ ] 考慮語義快取
- [ ] 設定合適的 TTL
- [ ] 監控快取命中率

### 模型選擇
- [ ] 使用智能路由
- [ ] 根據任務複雜度選擇模型
- [ ] 實作降級策略

### 批次處理
- [ ] 合併相似請求
- [ ] 使用異步處理
- [ ] 優化並發數

### 預算管理
- [ ] 設定每日/每月預算
- [ ] 配置警報閾值
- [ ] 實作自動降級
- [ ] 定期審查成本報告
```

## 延伸閱讀

- [OpenAI API Pricing](https://openai.com/pricing)
- [Anthropic Claude Pricing](https://anthropic.com/pricing)
- [Token Optimization Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [LLM Cost Optimization](https://www.latent.space/p/cost-optimization)
