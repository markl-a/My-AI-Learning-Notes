# AI Agents in LangGraph

## 📋 課程概述

深入學習 LangGraph 框架，建立複雜的狀態機式 AI 代理。

### 課程目標
- 掌握 LangGraph 核心概念
- 學習建立有狀態的工作流程
- 實作多步驟推理代理
- 理解循環和條件分支

### 課程時長
約 1 小時

## 🎯 LangGraph 核心概念

LangGraph 讓你能夠建立包含循環和條件邏輯的複雜 AI 工作流程。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# 定義狀態
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_step: str
    result: str

# 建立圖
workflow = StateGraph(AgentState)

# 添加節點（步驟）
workflow.add_node("step1", step1_function)
workflow.add_node("step2", step2_function)

# 添加邊（流程）
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", END)

# 設定入口
workflow.set_entry_point("step1")

# 編譯
app = workflow.compile()
```

## 💡 實戰應用

建立自我修正的研究代理，能夠驗證和改進自己的輸出。

---

**課程連結**：[DeepLearning.ai - AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)

**完成日期**：2025-01-17
