# Multi AI Agent Systems

## 📋 課程概述

學習建立多代理協作系統，讓多個 AI 代理共同完成複雜任務。

### 課程目標
- 理解多代理系統架構
- 學習代理間通訊機制
- 實作團隊協作型 AI 系統
- 掌握任務分配和協調策略

### 課程時長
約 1 小時

## 🎯 多代理系統架構

```python
from crewai import Agent, Task, Crew

# 定義多個專業代理
researcher = Agent(
    role='研究員',
    goal='收集和分析資訊',
    backstory='你是一位經驗豐富的研究員'
)

writer = Agent(
    role='作家',
    goal='撰寫高品質內容',
    backstory='你是一位專業的技術寫作者'
)

reviewer = Agent(
    role='審核員',
    goal='確保內容品質',
    backstory='你是一位嚴謹的內容審核專家'
)

# 建立團隊
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    verbose=True
)
```

## 💡 實戰範例

建立內容創作團隊：研究、撰寫、審核三個代理協作完成文章創作。

---

**課程連結**：[DeepLearning.ai - Multi AI Agent Systems](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)

**完成日期**：2025-01-17
