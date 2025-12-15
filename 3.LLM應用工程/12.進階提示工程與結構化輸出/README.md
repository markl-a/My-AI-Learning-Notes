# 進階提示工程與結構化輸出 (Prompt Engineering 2.0)

> **最後更新**: 2025-12-14
> **狀態**: 2024-2025年提示工程最新實踐

---

## 📋 目錄

1. [從Prompt 1.0到2.0](#1-從prompt-10到20)
2. [結構化輸出](#2-結構化輸出)
3. [Function Calling深度指南](#3-function-calling深度指南)
4. [提示優化框架](#4-提示優化框架)
5. [Chain-of-Thought進階](#5-chain-of-thought進階)
6. [多模態提示工程](#6-多模態提示工程)
7. [提示安全與防禦](#7-提示安全與防禦)
8. [自動化提示優化](#8-自動化提示優化)

---

## 1. 從Prompt 1.0到2.0

### 1.1 演進對比

| 特性 | Prompt 1.0 | Prompt 2.0 |
|------|-----------|-----------|
| **輸出格式** | 自由文本 | 結構化JSON/Schema |
| **可靠性** | 依賴模型理解 | Schema強制約束 |
| **工具調用** | 模擬/解析 | 原生Function Calling |
| **推理方式** | 單步回答 | CoT/ToT多步推理 |
| **優化方法** | 人工調整 | DSPy自動優化 |
| **評估指標** | 主觀評價 | 量化指標 |

### 1.2 2024-2025核心趨勢

```
┌─────────────────────────────────────────────────────────────┐
│                 Prompt Engineering 2.0 技術棧               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Structured  │  │   Chain of  │  │   Tool      │        │
│  │   Output    │  │   Thought   │  │   Use       │        │
│  │ (JSON/XML)  │  │  (CoT/ToT)  │  │ (MCP/FC)    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│                   ┌──────▼──────┐                          │
│                   │    DSPy     │                          │
│                   │  Framework  │                          │
│                   └──────┬──────┘                          │
│                          │                                 │
│         ┌────────────────┼────────────────┐                │
│         │                │                │                │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│  │   Auto      │  │   Prompt    │  │   Eval &    │        │
│  │   Prompting │  │   Caching   │  │   Metrics   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 結構化輸出

### 2.1 JSON Schema強制輸出

#### OpenAI方式 (response_format)

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

client = OpenAI()

# 定義輸出結構
class ProductReview(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float
    key_points: List[str]
    suggested_improvements: Optional[List[str]] = None

# 使用structured output
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "分析產品評論並輸出結構化結果"},
        {"role": "user", "content": "這個產品很好用，但價格太貴了，希望能便宜一點"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_review",
            "strict": True,
            "schema": ProductReview.model_json_schema()
        }
    }
)

# 解析結果
result = ProductReview.model_validate_json(response.choices[0].message.content)
print(f"情感: {result.sentiment}")
print(f"信心度: {result.confidence}")
print(f"要點: {result.key_points}")
```

#### Anthropic Claude方式

```python
from anthropic import Anthropic
import json

client = Anthropic()

# 使用XML標籤強制結構
system_prompt = """
你是一個產品評論分析助手。請嚴格按照以下JSON格式輸出:
{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "key_points": ["要點1", "要點2"],
    "suggested_improvements": ["建議1"] // 可選
}
"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=system_prompt,
    messages=[
        {"role": "user", "content": "分析: 這個產品很好用，但價格太貴了"}
    ]
)

# Claude會返回JSON格式
result = json.loads(response.content[0].text)
```

### 2.2 複雜嵌套結構

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SubTask(BaseModel):
    title: str = Field(description="子任務標題")
    estimated_hours: float = Field(ge=0, description="預估時數")
    dependencies: List[str] = Field(default=[], description="依賴的其他任務ID")

class Task(BaseModel):
    id: str = Field(description="唯一任務ID")
    title: str = Field(description="任務標題")
    description: str = Field(description="詳細描述")
    priority: Priority = Field(description="優先級")
    subtasks: List[SubTask] = Field(default=[], description="子任務列表")
    assignee: Optional[str] = Field(default=None, description="負責人")

class ProjectPlan(BaseModel):
    project_name: str
    total_estimated_hours: float
    tasks: List[Task]
    risks: List[str] = Field(default=[], description="潛在風險")

    class Config:
        json_schema_extra = {
            "examples": [{
                "project_name": "網站重構",
                "total_estimated_hours": 120,
                "tasks": [
                    {
                        "id": "T001",
                        "title": "需求分析",
                        "description": "收集和分析需求",
                        "priority": "high",
                        "subtasks": [],
                        "assignee": "張三"
                    }
                ],
                "risks": ["時間緊迫", "技術複雜度高"]
            }]
        }

# 使用
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是專案規劃助手，請生成詳細的專案計劃"},
        {"role": "user", "content": "規劃一個電商網站開發專案，包含用戶系統、商品管理、訂單系統"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "project_plan",
            "strict": True,
            "schema": ProjectPlan.model_json_schema()
        }
    }
)
```

### 2.3 XML結構化輸出

```python
# XML格式適合層次化內容
system_prompt = """
請使用以下XML格式輸出分析結果:

<analysis>
    <summary>簡短摘要</summary>
    <sections>
        <section id="1">
            <title>章節標題</title>
            <content>章節內容</content>
            <key_findings>
                <finding>發現1</finding>
                <finding>發現2</finding>
            </key_findings>
        </section>
    </sections>
    <recommendations>
        <recommendation priority="high">建議1</recommendation>
        <recommendation priority="medium">建議2</recommendation>
    </recommendations>
</analysis>
"""

# 解析XML
import xml.etree.ElementTree as ET

def parse_analysis(xml_string: str) -> dict:
    root = ET.fromstring(xml_string)
    return {
        "summary": root.find("summary").text,
        "sections": [
            {
                "id": section.get("id"),
                "title": section.find("title").text,
                "content": section.find("content").text,
                "findings": [f.text for f in section.findall("key_findings/finding")]
            }
            for section in root.findall("sections/section")
        ],
        "recommendations": [
            {"priority": rec.get("priority"), "text": rec.text}
            for rec in root.findall("recommendations/recommendation")
        ]
    }
```

---

## 3. Function Calling深度指南

### 3.1 OpenAI Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

# 定義工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索產品目錄",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索關鍵詞"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "home"],
                        "description": "產品分類"
                    },
                    "price_range": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"}
                        },
                        "description": "價格範圍"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "rating", "newest"],
                        "default": "rating"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "獲取產品詳細信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "產品ID"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "添加產品到購物車",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1, "default": 1}
                },
                "required": ["product_id"]
            }
        }
    }
]

# 工具實現
def search_products(query: str, category: str = None, price_range: dict = None, sort_by: str = "rating"):
    # 實際實現搜索邏輯
    return {"products": [{"id": "P001", "name": "示例產品", "price": 99.99}]}

def get_product_details(product_id: str):
    return {"id": product_id, "name": "產品名稱", "description": "詳細描述", "price": 99.99}

def add_to_cart(product_id: str, quantity: int = 1):
    return {"success": True, "cart_total": quantity}

# 工具映射
tool_functions = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "add_to_cart": add_to_cart
}

# 對話循環
def chat_with_tools(user_message: str, conversation_history: list):
    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # 如果需要調用工具
    if assistant_message.tool_calls:
        conversation_history.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # 調用對應函數
            function_response = tool_functions[function_name](**function_args)

            # 添加工具結果
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_response, ensure_ascii=False)
            })

        # 獲取最終回覆
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history
        )

        return final_response.choices[0].message.content

    return assistant_message.content
```

### 3.2 並行工具調用

```python
import asyncio
from typing import List, Dict, Any

async def execute_tool_calls_parallel(tool_calls: List) -> List[Dict[str, Any]]:
    """並行執行多個工具調用"""

    async def execute_single(tool_call):
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # 異步執行工具
        if function_name in async_tool_functions:
            result = await async_tool_functions[function_name](**function_args)
        else:
            # 同步工具包裝為異步
            result = await asyncio.to_thread(
                tool_functions[function_name],
                **function_args
            )

        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False)
        }

    # 並行執行所有工具調用
    results = await asyncio.gather(*[execute_single(tc) for tc in tool_calls])
    return results
```

### 3.3 工具選擇策略

```python
# 強制使用特定工具
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "search_products"}}
)

# 禁止使用工具
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="none"
)

# 必須使用工具（至少一個）
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="required"
)
```

---

## 4. 提示優化框架

### 4.1 DSPy框架

```python
import dspy

# 配置LLM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 定義Signature
class SentimentAnalysis(dspy.Signature):
    """分析文本情感"""
    text: str = dspy.InputField(desc="要分析的文本")
    sentiment: str = dspy.OutputField(desc="情感: positive/negative/neutral")
    confidence: float = dspy.OutputField(desc="置信度 0-1")

# 使用Predictor
predictor = dspy.Predict(SentimentAnalysis)
result = predictor(text="這個產品太棒了！")
print(f"情感: {result.sentiment}, 置信度: {result.confidence}")

# Chain of Thought
class ReasonedSentiment(dspy.Signature):
    """分析文本情感並給出推理過程"""
    text: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="分析推理過程")
    sentiment: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

cot_predictor = dspy.ChainOfThought(ReasonedSentiment)
result = cot_predictor(text="產品質量不錯，但客服態度很差")

# 自動優化
from dspy.teleprompt import BootstrapFewShot

# 準備訓練數據
trainset = [
    dspy.Example(text="太好了！", sentiment="positive", confidence=0.95),
    dspy.Example(text="很失望", sentiment="negative", confidence=0.9),
    # ...更多示例
]

# 優化
optimizer = BootstrapFewShot(metric=lambda pred, gold: pred.sentiment == gold.sentiment)
optimized_predictor = optimizer.compile(predictor, trainset=trainset)
```

### 4.2 Guidance框架

```python
from guidance import models, gen, select

# 載入模型
gpt4 = models.OpenAI("gpt-4o")

# 結構化生成
@guidance
def product_analysis(lm, product_description):
    lm += f"""
    分析以下產品描述:
    {product_description}

    分析結果:
    - 產品類別: {select(['電子產品', '服裝', '食品', '家居'], name='category')}
    - 目標用戶: {gen('target_audience', max_tokens=50, stop='\\n')}
    - 主要賣點:
      1. {gen('selling_point_1', max_tokens=30, stop='\\n')}
      2. {gen('selling_point_2', max_tokens=30, stop='\\n')}
      3. {gen('selling_point_3', max_tokens=30, stop='\\n')}
    - 價格定位: {select(['高端', '中端', '平價'], name='price_tier')}
    - 推薦評分: {gen('rating', regex='[1-5]')}/5
    """
    return lm

result = gpt4 + product_analysis("Apple iPhone 15 Pro Max 256GB")
print(f"類別: {result['category']}")
print(f"評分: {result['rating']}")
```

### 4.3 LMQL查詢語言

```python
import lmql

@lmql.query
def classify_intent(user_input):
    '''lmql
    argmax
        "用戶輸入: {user_input}\n"
        "意圖分類:\n"
        "- 類別: [CATEGORY]"
        "- 置信度: [CONFIDENCE]"
    from
        "openai/gpt-4o"
    where
        CATEGORY in ["查詢", "購買", "投訴", "建議", "其他"]
        and CONFIDENCE in ["高", "中", "低"]
    '''

result = classify_intent("我想退貨")
```

---

## 5. Chain-of-Thought進階

### 5.1 標準CoT

```python
cot_prompt = """
請一步步思考來解決這個問題:

問題: {question}

讓我們一步步來:
1. 首先，我需要理解問題...
2. 然後，分析關鍵信息...
3. 接著，應用相關知識...
4. 最後，得出結論...

答案:
"""
```

### 5.2 Self-Consistency (自我一致性)

```python
import collections

def self_consistency_cot(question: str, num_samples: int = 5) -> str:
    """
    通過多次採樣和投票提高CoT可靠性
    """
    answers = []

    for _ in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請一步步思考並給出答案"},
                {"role": "user", "content": question}
            ],
            temperature=0.7  # 增加隨機性獲得多樣答案
        )

        # 提取最終答案
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)

    # 投票選出最常見答案
    answer_counts = collections.Counter(answers)
    most_common = answer_counts.most_common(1)[0][0]

    return most_common
```

### 5.3 Tree of Thoughts (ToT)

```python
from typing import List, Tuple

class TreeOfThoughts:
    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def generate_thoughts(self, state: str, k: int = 3) -> List[str]:
        """生成k個可能的思考方向"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "生成解決問題的可能思路"},
                {"role": "user", "content": f"""
                    當前狀態: {state}

                    請生成{k}個不同的思考方向來推進問題解決:
                    1.
                    2.
                    3.
                """}
            ]
        )
        return self._parse_thoughts(response.choices[0].message.content)

    def evaluate_thought(self, state: str, thought: str) -> float:
        """評估思路的質量 (0-1)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"""
                    問題狀態: {state}
                    思路: {thought}

                    評估這個思路解決問題的潛力 (0-10分):
                """}
            ]
        )
        score = self._extract_score(response.choices[0].message.content)
        return score / 10

    def solve(self, problem: str, max_depth: int = 3, beam_width: int = 2) -> str:
        """
        使用BFS+剪枝的ToT解決問題
        """
        # 初始狀態
        states = [(problem, [])]  # (當前狀態, 思考路徑)

        for depth in range(max_depth):
            candidates = []

            for state, path in states:
                # 生成新思路
                thoughts = self.generate_thoughts(state)

                for thought in thoughts:
                    # 評估思路
                    score = self.evaluate_thought(state, thought)
                    new_state = f"{state}\n思考{depth+1}: {thought}"
                    candidates.append((new_state, path + [thought], score))

            # 保留最優的beam_width個
            candidates.sort(key=lambda x: x[2], reverse=True)
            states = [(s, p) for s, p, _ in candidates[:beam_width]]

        # 返回最佳路徑的最終答案
        best_state, best_path = states[0]
        return self._generate_final_answer(best_state)
```

### 5.4 ReAct (Reasoning + Acting)

```python
class ReActAgent:
    def __init__(self, tools: dict):
        self.tools = tools
        self.max_iterations = 10

    def run(self, question: str) -> str:
        """ReAct循環"""
        prompt = f"""
        回答以下問題，使用Thought/Action/Observation格式:

        問題: {question}

        可用工具: {list(self.tools.keys())}

        格式:
        Thought: 我需要思考...
        Action: tool_name(arg1, arg2)
        Observation: [工具返回結果]
        ... (重複直到得到答案)
        Thought: 我現在知道答案了
        Final Answer: 最終答案
        """

        conversation = [{"role": "user", "content": prompt}]

        for i in range(self.max_iterations):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation,
                stop=["Observation:"]
            )

            assistant_message = response.choices[0].message.content

            # 檢查是否有最終答案
            if "Final Answer:" in assistant_message:
                return assistant_message.split("Final Answer:")[-1].strip()

            # 解析並執行Action
            action = self._parse_action(assistant_message)
            if action:
                tool_name, args = action
                observation = self.tools[tool_name](*args)

                conversation.append({"role": "assistant", "content": assistant_message})
                conversation.append({"role": "user", "content": f"Observation: {observation}"})

        return "無法在限定步驟內找到答案"
```

---

## 6. 多模態提示工程

### 6.1 視覺提示 (Vision Prompting)

```python
import base64

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 圖像分析
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                    分析這張圖片並提供:
                    1. 圖片內容描述
                    2. 主要物體檢測
                    3. 場景分類
                    4. 情感/氛圍分析

                    請以JSON格式輸出。
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image('image.jpg')}",
                        "detail": "high"  # low/high/auto
                    }
                }
            ]
        }
    ],
    response_format={"type": "json_object"}
)
```

### 6.2 多圖像對比分析

```python
def compare_images(images: List[str], comparison_prompt: str) -> dict:
    """對比多張圖片"""
    content = [{"type": "text", "text": comparison_prompt}]

    for i, img_path in enumerate(images):
        content.append({
            "type": "text",
            "text": f"圖片 {i+1}:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
            }
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}]
    )

    return response.choices[0].message.content

# 使用
result = compare_images(
    ["product_v1.jpg", "product_v2.jpg"],
    "對比這兩個產品設計，分析差異和改進點"
)
```

### 6.3 視覺CoT (Visual Chain of Thought)

```python
visual_cot_prompt = """
請按照以下步驟分析這張圖片:

步驟1 - 整體觀察:
- 描述圖片的整體場景
- 識別主要元素

步驟2 - 細節分析:
- 觀察每個主要元素的特徵
- 注意顏色、形狀、位置關係

步驟3 - 推理:
- 基於觀察推斷場景的含義
- 分析可能的上下文

步驟4 - 結論:
- 總結圖片的主題
- 給出相關建議或見解
"""
```

---

## 7. 提示安全與防禦

### 7.1 提示注入防禦

```python
import re
from typing import Tuple

class PromptGuard:
    # 危險模式
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+instructions?",
        r"disregard\s+(previous|all|above)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now\s+a?",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(if|a)",
        r"roleplay\s+as",
        r"jailbreak",
        r"DAN\s*mode",
        r"\[system\]",
        r"<\|im_start\|>",
    ]

    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.INJECTION_PATTERNS
        ]

    def check_input(self, user_input: str) -> Tuple[bool, str]:
        """
        檢查用戶輸入是否包含注入嘗試
        Returns: (is_safe, reason)
        """
        for pattern in self.compiled_patterns:
            if pattern.search(user_input):
                return False, f"檢測到可疑模式: {pattern.pattern}"

        # 檢查特殊字符比例
        special_chars = sum(1 for c in user_input if not c.isalnum() and not c.isspace())
        if special_chars / len(user_input) > 0.3:
            return False, "特殊字符比例過高"

        return True, "通過安全檢查"

    def sanitize_input(self, user_input: str) -> str:
        """清理用戶輸入"""
        # 移除可能的控制字符
        cleaned = ''.join(c for c in user_input if c.isprintable() or c in '\n\t')

        # 轉義可能的指令分隔符
        cleaned = cleaned.replace("```", "'''")
        cleaned = cleaned.replace("<|", "< |")
        cleaned = cleaned.replace("|>", "| >")

        return cleaned

# 使用
guard = PromptGuard()

def safe_chat(user_input: str) -> str:
    is_safe, reason = guard.check_input(user_input)
    if not is_safe:
        return f"輸入被拒絕: {reason}"

    sanitized = guard.sanitize_input(user_input)
    # 繼續處理...
```

### 7.2 輸出驗證

```python
class OutputValidator:
    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
            "phone": r'\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }

    def check_pii(self, output: str) -> List[str]:
        """檢查輸出是否包含PII"""
        found_pii = []
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, output):
                found_pii.append(pii_type)
        return found_pii

    def redact_pii(self, output: str) -> str:
        """遮蔽輸出中的PII"""
        redacted = output
        for pii_type, pattern in self.pii_patterns.items():
            redacted = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", redacted)
        return redacted

    def validate_json_output(self, output: str, schema: dict) -> Tuple[bool, str]:
        """驗證JSON輸出是否符合schema"""
        from jsonschema import validate, ValidationError

        try:
            data = json.loads(output)
            validate(instance=data, schema=schema)
            return True, "驗證通過"
        except json.JSONDecodeError as e:
            return False, f"JSON解析錯誤: {e}"
        except ValidationError as e:
            return False, f"Schema驗證錯誤: {e.message}"
```

---

## 8. 自動化提示優化

### 8.1 提示評估框架

```python
from dataclasses import dataclass
from typing import List, Callable
import numpy as np

@dataclass
class EvalResult:
    score: float
    details: dict

class PromptEvaluator:
    def __init__(self, metrics: List[Callable]):
        self.metrics = metrics

    def evaluate(
        self,
        prompt: str,
        test_cases: List[dict],
        model: str = "gpt-4o"
    ) -> EvalResult:
        """評估提示詞在測試用例上的表現"""
        scores = []

        for test in test_cases:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": test["input"]}
                ]
            )

            output = response.choices[0].message.content

            # 計算各指標
            case_scores = {}
            for metric in self.metrics:
                case_scores[metric.__name__] = metric(
                    output,
                    test.get("expected"),
                    test["input"]
                )
            scores.append(case_scores)

        # 聚合結果
        avg_scores = {}
        for metric_name in scores[0].keys():
            avg_scores[metric_name] = np.mean([s[metric_name] for s in scores])

        overall_score = np.mean(list(avg_scores.values()))

        return EvalResult(score=overall_score, details=avg_scores)

# 定義評估指標
def relevance_score(output: str, expected: str, input_text: str) -> float:
    """相關性評分"""
    # 使用嵌入相似度
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    emb_output = model.encode(output)
    emb_expected = model.encode(expected)

    similarity = np.dot(emb_output, emb_expected) / (
        np.linalg.norm(emb_output) * np.linalg.norm(emb_expected)
    )
    return float(similarity)

def format_compliance(output: str, expected: str, input_text: str) -> float:
    """格式符合度"""
    try:
        json.loads(output)
        return 1.0
    except:
        return 0.0
```

### 8.2 自動提示優化器

```python
class PromptOptimizer:
    def __init__(self, evaluator: PromptEvaluator):
        self.evaluator = evaluator

    def optimize(
        self,
        initial_prompt: str,
        test_cases: List[dict],
        iterations: int = 10
    ) -> str:
        """迭代優化提示詞"""
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = self.evaluator.evaluate(current_prompt, test_cases).score

        for i in range(iterations):
            # 生成變體
            variants = self._generate_variants(current_prompt)

            for variant in variants:
                result = self.evaluator.evaluate(variant, test_cases)

                if result.score > best_score:
                    best_score = result.score
                    best_prompt = variant
                    print(f"迭代 {i+1}: 發現更好的提示詞 (分數: {best_score:.4f})")

            current_prompt = best_prompt

        return best_prompt

    def _generate_variants(self, prompt: str) -> List[str]:
        """生成提示詞變體"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是提示詞優化專家"},
                {"role": "user", "content": f"""
                    原始提示詞:
                    {prompt}

                    請生成5個改進版本，每個版本嘗試不同的優化策略:
                    1. 更清晰的指令
                    2. 添加示例
                    3. 結構化格式
                    4. 添加約束
                    5. 簡化表達

                    以JSON數組格式輸出。
                """}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("variants", [])
```

---

## 📚 參考資源

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Claude Prompt Library](https://docs.anthropic.com/claude/prompt-library)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Guidance GitHub](https://github.com/guidance-ai/guidance)
- [LMQL Documentation](https://lmql.ai/)

---

## 🔗 相關章節

- [MCP協議與工具調用](../11.MCP協議與工具調用/README.md)
- [Agent工具設計](../3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md)
- [LLM安全與防禦](../8.LLM安全與防禦/README.md)
