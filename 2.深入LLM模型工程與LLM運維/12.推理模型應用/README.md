# 推理模型 (Reasoning Models) 應用指南

> **最後更新**: 2025-12-14
> **涵蓋模型**: OpenAI o1/o3, DeepSeek-R1, Gemini 2.0 Flash Thinking
>
> 📚 補充歷史視角請見 [`../1.LLM 基礎與架構/推理模型_Reasoning_Models_深度解析.md`](../1.LLM%20基礎與架構/推理模型_Reasoning_Models_深度解析.md)(2025-01 版本)

---

## 📋 目錄

1. [推理模型概述](#1-推理模型概述)
2. [主要推理模型對比](#2-主要推理模型對比)
3. [使用場景與最佳實踐](#3-使用場景與最佳實踐)
4. [成本效益分析](#4-成本效益分析)
5. [實戰程式碼示例](#5-實戰程式碼示例)
6. [與傳統模型的協同使用](#6-與傳統模型的協同使用)

---

## 1. 推理模型概述

### 1.1 什麼是推理模型？

推理模型是一類專門設計來進行**多步推理**的大型語言模型。與傳統LLM不同，推理模型會在生成最終答案前，先進行內部的"思考"過程。

```
┌─────────────────────────────────────────────────────────────┐
│              傳統LLM vs 推理模型                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  傳統LLM (如GPT-4):                                         │
│  ┌────────┐                    ┌────────┐                  │
│  │ 輸入   │ ─────────────────► │ 輸出   │                  │
│  │ Prompt │     直接生成       │ Answer │                  │
│  └────────┘                    └────────┘                  │
│                                                             │
│  推理模型 (如o1):                                           │
│  ┌────────┐    ┌─────────────────┐    ┌────────┐          │
│  │ 輸入   │ ─► │ 思考鏈          │ ─► │ 輸出   │          │
│  │ Prompt │    │ (Chain of       │    │ Answer │          │
│  └────────┘    │  Thought)       │    └────────┘          │
│                │ • 分解問題       │                         │
│                │ • 嘗試多種方法   │                         │
│                │ • 驗證答案       │                         │
│                └─────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心特點

| 特點 | 描述 |
|------|------|
| **內部推理** | 生成答案前先進行內部思考 |
| **多步分解** | 自動將複雜問題分解為子問題 |
| **自我修正** | 能夠識別和修正自己的錯誤 |
| **延遲生成** | 回答速度較慢，但質量更高 |
| **計算密集** | 消耗更多的Token和計算資源 |

---

## 2. 主要推理模型對比

### 2.1 OpenAI o系列

| 模型 | 發布時間 | 推理能力 | 速度 | 成本 |
|------|---------|---------|------|------|
| o1-preview | 2024-09 | ⭐⭐⭐⭐ | 慢 | $15/1M input |
| o1 | 2024-12 | ⭐⭐⭐⭐⭐ | 中 | $15/1M input |
| o1-mini | 2024-09 | ⭐⭐⭐ | 快 | $3/1M input |
| o3 | 2025-01 | ⭐⭐⭐⭐⭐+ | 中 | TBD |
| o3-mini | 2025-01 | ⭐⭐⭐⭐ | 快 | TBD |

### 2.2 DeepSeek-R1

```python
# DeepSeek-R1特點
deepseek_r1 = {
    "發布時間": "2025-01-20",
    "開源狀態": "完全開源 (MIT)",
    "模型規模": "671B (MoE)",
    "推理能力": "接近o1水平",
    "成本優勢": "比o1便宜90%+",
    "蒸餾版本": [
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-70B"
    ]
}
```

### 2.3 性能對比

```
┌─────────────────────────────────────────────────────────────┐
│                   推理能力基準測試                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AIME 2024 (數學競賽):                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ o3 (high)      ████████████████████████████ 96.7%  │    │
│  │ DeepSeek-R1    ███████████████████████████ 79.8%   │    │
│  │ o1             ███████████████████████ 74.4%       │    │
│  │ o1-mini        █████████████████ 60.0%             │    │
│  │ Claude 3.5     ████████ 16.0%                      │    │
│  │ GPT-4o         ████ 9.3%                           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  SWE-bench Verified (程式碼):                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ o3              ████████████████████████████ 71.7% │    │
│  │ DeepSeek-R1     ████████████████████████ 49.2%     │    │
│  │ o1              ███████████████████████ 48.9%      │    │
│  │ Claude 3.5      ██████████████████████ 50.8%       │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  GPQA Diamond (科學推理):                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ o3 (high)      ████████████████████████████ 87.7%  │    │
│  │ o1             ████████████████████████ 78.3%      │    │
│  │ DeepSeek-R1    ███████████████████████ 71.5%       │    │
│  │ Claude 3.5     ██████████████████ 65.0%            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 使用場景與最佳實踐

### 3.1 適合推理模型的場景

✅ **高度推薦**:
- 複雜數學問題和證明
- 多步驟邏輯推理
- 程式碼調試和複雜演算法設計
- 科學研究問題
- 策略規劃和決策分析

⚠️ **可以使用**:
- 複雜的文檔分析
- 多約束條件的優化問題
- 需要精確性的任務

❌ **不推薦**:
- 簡單問答和聊天
- 創意寫作（推理模型較生硬）
- 實時交互（延遲高）
- 成本敏感的大量請求

### 3.2 提示詞最佳實踐

```python
# ✅ 好的推理模型提示詞
good_prompt = """
問題：一個球從10米高的地方自由落下，每次彈起的高度是落下高度的3/4。
      求球在停止彈跳前經過的總路程。

請一步步分析這個問題，包括：
1. 識別問題類型
2. 列出已知條件
3. 建立數學模型
4. 求解
5. 驗證答案的合理性
"""

# ❌ 不好的提示詞
bad_prompt = "球從10米高落下，每次彈起3/4高度，求總路程"

# ✅ 程式碼任務的好提示詞
code_prompt = """
任務：實現一個高效的LRU快取。

要求：
1. 支持get(key)和put(key, value)操作
2. 兩種操作的時間複雜度都是O(1)
3. 當快取滿時，移除最近最少使用的項目
4. 快取容量在初始化時指定

請提供：
1. 設計思路
2. 資料結構選擇的原因
3. 完整的Python實現
4. 時間和空間複雜度分析
5. 測試用例
"""
```

### 3.3 o1系列特殊注意事項

```python
from openai import OpenAI

client = OpenAI()

# o1系列的限制：
# 1. 不支持system message（會自動忽略或報錯）
# 2. 不支持temperature參數
# 3. 不支持streaming
# 4. 不支持function calling（截至2024-12）

# ✅ 正確用法
response = client.chat.completions.create(
    model="o1",
    messages=[
        # 注意：沒有system message！
        {
            "role": "user",
            "content": """你是一個數學專家。請解決以下問題：

            證明：對於任意正整數n，n³-n總是能被6整除。

            請提供完整的數學證明。"""
        }
    ],
    # 注意：沒有temperature！
    max_completion_tokens=8000  # o1使用這個參數而非max_tokens
)

# ✅ 獲取推論token消耗
usage = response.usage
print(f"輸入Tokens: {usage.prompt_tokens}")
print(f"輸出Tokens: {usage.completion_tokens}")
print(f"推理Tokens: {usage.completion_tokens_details.reasoning_tokens}")
```

---

## 4. 成本效益分析

### 4.1 價格對比

| 模型 | 輸入價格 | 輸出價格 | 推理特點 |
|------|---------|---------|---------|
| **GPT-4o** | $2.50/1M | $10/1M | 無內部推理 |
| **o1-mini** | $3/1M | $12/1M | 輕量推理 |
| **o1** | $15/1M | $60/1M | 完整推理 |
| **o3-mini** | TBD | TBD | 中等推理 |
| **DeepSeek-R1 API** | $0.55/1M | $2.19/1M | 開源，可本地 |

### 4.2 成本計算器

```python
def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0
) -> dict:
    """計算API呼叫成本"""

    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
        "o1": {"input": 15.0, "output": 60.0},
        "deepseek-r1": {"input": 0.55, "output": 2.19},
    }

    if model not in prices:
        raise ValueError(f"Unknown model: {model}")

    price = prices[model]

    # 推論tokens計入輸出
    total_output = output_tokens + reasoning_tokens

    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (total_output / 1_000_000) * price["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "input_cost": f"${input_cost:.4f}",
        "output_cost": f"${output_cost:.4f}",
        "total_cost": f"${total_cost:.4f}"
    }

# 示例：複雜數學問題
result = calculate_cost(
    model="o1",
    input_tokens=500,
    output_tokens=2000,
    reasoning_tokens=10000  # o1會使用大量推論tokens
)
print(result)
# {'model': 'o1', 'total_cost': '$0.7275'}

# 對比：使用DeepSeek-R1
result_deepseek = calculate_cost(
    model="deepseek-r1",
    input_tokens=500,
    output_tokens=2000,
    reasoning_tokens=10000
)
print(result_deepseek)
# {'model': 'deepseek-r1', 'total_cost': '$0.0266'}
```

### 4.3 何時使用推理模型的決策框架

```python
def should_use_reasoning_model(task_info: dict) -> dict:
    """
    決定是否使用推理模型

    Args:
        task_info: {
            "complexity": "low/medium/high/very_high",
            "accuracy_requirement": "low/medium/high/critical",
            "latency_tolerance": "low/medium/high",  # 可接受的延遲
            "budget_sensitivity": "low/medium/high",
            "task_type": "math/code/reasoning/creative/qa"
        }
    """

    scores = {
        "complexity": {"low": 0, "medium": 1, "high": 2, "very_high": 3},
        "accuracy_requirement": {"low": 0, "medium": 1, "high": 2, "critical": 3},
        "latency_tolerance": {"low": 0, "medium": 1, "high": 2},
        "budget_sensitivity": {"high": 0, "medium": 1, "low": 2}
    }

    task_type_bonus = {
        "math": 2,
        "code": 1.5,
        "reasoning": 2,
        "creative": -1,
        "qa": 0
    }

    score = 0
    score += scores["complexity"][task_info["complexity"]]
    score += scores["accuracy_requirement"][task_info["accuracy_requirement"]]
    score += scores["latency_tolerance"][task_info["latency_tolerance"]]
    score += scores["budget_sensitivity"][task_info["budget_sensitivity"]]
    score += task_type_bonus.get(task_info["task_type"], 0)

    if score >= 8:
        return {
            "recommendation": "o1",
            "reason": "高複雜度、高準確性要求，推薦使用完整推理模型",
            "score": score
        }
    elif score >= 5:
        return {
            "recommendation": "o1-mini or DeepSeek-R1",
            "reason": "中等複雜度，可以使用輕量推理模型以平衡成本",
            "score": score
        }
    else:
        return {
            "recommendation": "GPT-4o or Claude",
            "reason": "任務複雜度不高，使用標準模型更具成本效益",
            "score": score
        }

# 使用示例
task = {
    "complexity": "very_high",
    "accuracy_requirement": "critical",
    "latency_tolerance": "high",
    "budget_sensitivity": "low",
    "task_type": "math"
}
print(should_use_reasoning_model(task))
```

---

## 5. 實戰程式碼示例

### 5.1 數學問題求解

```python
from openai import OpenAI

client = OpenAI()

def solve_math_problem(problem: str) -> dict:
    """使用o1解決數學問題"""

    response = client.chat.completions.create(
        model="o1",
        messages=[
            {
                "role": "user",
                "content": f"""請解決以下數學問題，並提供詳細的解題過程：

{problem}

要求：
1. 清晰地列出每個步驟
2. 解釋每步的數學原理
3. 驗證最終答案
"""
            }
        ],
        max_completion_tokens=8000
    )

    return {
        "problem": problem,
        "solution": response.choices[0].message.content,
        "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
        "total_tokens": response.usage.total_tokens
    }

# 示例
result = solve_math_problem("""
求解方程組：
x² + y² = 25
x + y = 7
""")
print(result["solution"])
```

### 5.2 複雜程式碼調試

```python
def debug_code_with_reasoning(
    code: str,
    error_message: str,
    expected_behavior: str
) -> dict:
    """使用推理模型調試程式碼"""

    prompt = f"""
我有以下程式碼出現問題，請幫我分析並修復。

## 程式碼
```python
{code}
```

## 錯誤資訊
```
{error_message}
```

## 預期行為
{expected_behavior}

請：
1. 分析錯誤的根本原因
2. 解釋為什麼會出現這個問題
3. 提供修復後的程式碼
4. 解釋修改的內容和原因
5. 建議如何避免類似問題
"""

    response = client.chat.completions.create(
        model="o1",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000
    )

    return {
        "analysis": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens
    }

# 示例
buggy_code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)
"""

result = debug_code_with_reasoning(
    code=buggy_code,
    error_message="Input [3, 1, 3, 2, 3] returns [1, 2, 3] instead of [1, 2, 3, 3, 3]",
    expected_behavior="快速排序應該正確處理重複元素"
)
```

### 5.3 使用DeepSeek-R1

```python
from openai import OpenAI

# DeepSeek使用OpenAI兼容的API
deepseek_client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com"
)

def solve_with_deepseek_r1(problem: str) -> dict:
    """使用DeepSeek-R1解決問題"""

    response = deepseek_client.chat.completions.create(
        model="deepseek-reasoner",  # DeepSeek-R1的模型名稱
        messages=[
            {
                "role": "user",
                "content": problem
            }
        ],
        max_tokens=8000
    )

    # DeepSeek-R1會返回思考過程
    message = response.choices[0].message

    return {
        "answer": message.content,
        "reasoning": message.reasoning_content,  # 思考過程
        "usage": response.usage
    }

# 本地部署DeepSeek-R1蒸餾版
def setup_local_deepseek_r1():
    """使用Ollama本地部署DeepSeek-R1蒸餾版"""

    # 安裝命令
    commands = [
        "ollama pull deepseek-r1:7b",  # 7B蒸餾版
        "ollama pull deepseek-r1:14b", # 14B蒸餾版
        "ollama pull deepseek-r1:32b", # 32B蒸餾版
    ]

    # Python呼叫
    from ollama import Client

    client = Client()

    response = client.chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "user", "content": "解釋什麼是動態規劃"}
        ]
    )

    return response
```

---

## 6. 與傳統模型的協同使用

### 6.1 分層策略

```python
class HybridReasoningSystem:
    """混合推理系統：根據任務複雜度選擇模型"""

    def __init__(self):
        self.client = OpenAI()

        self.models = {
            "fast": "gpt-4o-mini",      # 簡單任務
            "standard": "gpt-4o",        # 中等任務
            "reasoning": "o1-mini",      # 需要推理的任務
            "deep_reasoning": "o1"       # 複雜推理任務
        }

    def classify_task(self, task: str) -> str:
        """使用快速模型判斷任務複雜度"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """分析任務複雜度，回答一個詞：
                    - simple: 簡單問答、翻譯、摘要
                    - medium: 需要分析但邏輯直接的任務
                    - complex: 多步推理、數學問題、程式碼調試
                    - very_complex: 複雜數學證明、演算法設計、策略規劃"""
                },
                {"role": "user", "content": f"任務: {task}"}
            ],
            max_tokens=10
        )

        complexity = response.choices[0].message.content.strip().lower()

        mapping = {
            "simple": "fast",
            "medium": "standard",
            "complex": "reasoning",
            "very_complex": "deep_reasoning"
        }

        return mapping.get(complexity, "standard")

    def process(self, task: str) -> dict:
        """處理任務"""

        # 1. 分類任務
        model_tier = self.classify_task(task)
        model = self.models[model_tier]

        # 2. 根據模型類型呼叫
        if model in ["o1", "o1-mini"]:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task}],
                max_completion_tokens=8000
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一個有幫助的助手。"},
                    {"role": "user", "content": task}
                ],
                max_tokens=4000
            )

        return {
            "model_used": model,
            "model_tier": model_tier,
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }

# 使用
system = HybridReasoningSystem()

# 簡單任務 -> gpt-4o-mini
result1 = system.process("今天天氣怎麼樣？")

# 複雜任務 -> o1
result2 = system.process("證明：任何大於2的偶數都可以表示為兩個質數之和")
```

### 6.2 推理-執行分離模式

```python
class ReasonExecuteSeparation:
    """
    推理和執行分離的架構：
    - 使用推理模型生成計劃
    - 使用快速模型執行計劃中的子任務
    """

    def __init__(self):
        self.client = OpenAI()

    def generate_plan(self, task: str) -> list:
        """使用o1生成執行計劃"""

        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[{
                "role": "user",
                "content": f"""為以下任務生成詳細的執行計劃：

{task}

請以JSON陣列格式輸出計劃步驟，每個步驟包含：
- step_id: 步驟編號
- description: 步驟描述
- dependencies: 依賴的步驟ID列表
- complexity: 複雜度 (low/medium/high)

只輸出JSON，不要其他內容。"""
            }],
            max_completion_tokens=4000
        )

        import json
        plan = json.loads(response.choices[0].message.content)
        return plan

    def execute_step(self, step: dict, context: str) -> str:
        """使用快速模型執行單個步驟"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini" if step["complexity"] == "low" else "gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是一個任務執行助手。根據上下文完成指定的子任務。"
                },
                {
                    "role": "user",
                    "content": f"""上下文：
{context}

當前任務：{step['description']}

請完成這個任務並提供結果。"""
                }
            ],
            max_tokens=2000
        )

        return response.choices[0].message.content

    def run(self, task: str) -> dict:
        """完整執行流程"""

        # 1. 生成計劃
        plan = self.generate_plan(task)

        # 2. 按順序執行
        results = []
        context = f"原始任務: {task}\n\n已完成的步驟:\n"

        for step in plan:
            result = self.execute_step(step, context)
            results.append({
                "step": step,
                "result": result
            })
            context += f"\n步驟 {step['step_id']}: {result}\n"

        return {
            "plan": plan,
            "results": results,
            "final_context": context
        }

# 使用
system = ReasonExecuteSeparation()
result = system.run("撰寫一份關於人工智能在醫療領域應用的研究報告")
```

---

## 📚 參考資源

- [OpenAI o1 Guide](https://platform.openai.com/docs/guides/reasoning)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)

---

## 🔗 相關章節

- [LLM最佳實踐指南](../LLM最佳實踐指南.md)
- [模型評估](../9.模型評估/README.md)
- [推論優化](../../3.LLM應用工程/6.推論優化/README.md)
