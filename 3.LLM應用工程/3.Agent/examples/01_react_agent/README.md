# ReAct Agent 範例

> **⚠️ 教學示範 — AST 沙箱 `safe_eval` 也僅供教學**
>
> `react_agent_basic.py:101-112` 的 calculator tool 用 `ast` 模組自實作 `safe_eval`(白名單運算符)當算式求值器,**比 `eval()` 安全但仍非生產級**。Production tool calling 請改用 `simpleeval` / `numexpr` 等專門套件,或乾脆把算式委派給可驗證的數學服務。

ReAct (Reasoning + Acting) 是一種讓 LLM 交替進行推理和行動的模式。

## 📖 ReAct 模式簡介

ReAct 的核心循環：

```
Thought (思考) → Action (行動) → Observation (觀察) → Thought → ...
```

與傳統 CoT (Chain of Thought) 的區別：
- **CoT**: 純推理，無法呼叫外部工具
- **ReAct**: 推理 + 工具呼叫，可以獲取外部資訊

## 📁 文件說明

- `react_agent_basic.py` - 基礎 ReAct Agent（使用模擬工具，可直接運行）
- `react_with_langchain.py` - 使用 LangChain 的 ReAct Agent
- `react_custom_prompt.py` - 自定義提示的 ReAct Agent
- `tools.py` - 工具定義模組

## 🚀 快速開始

### 1. 基礎範例（不需要 API 金鑰）

```bash
python react_agent_basic.py
```

這個範例使用模擬的 LLM 和工具，可以直接運行。

### 2. 使用真實 LLM

```bash
# 設置 API 金鑰
export OPENAI_API_KEY=your_key_here

# 運行
python react_with_langchain.py
```

## 💡 核心概念

### ReAct 循環

```python
while not task_completed:
    # 1. 思考：分析當前狀態
    thought = llm.generate(f"當前狀態: {state}, 思考下一步")

    # 2. 決定行動
    action, action_input = parse_action(thought)

    # 3. 執行工具
    observation = execute_tool(action, action_input)

    # 4. 更新狀態
    state = update_state(thought, action, observation)

    # 5. 檢查是否完成
    task_completed = check_completion(state)
```

### 提示工程

ReAct 的關鍵是好的提示格式：

```
思考 1: 我需要找出台灣的首都
行動 1: Search[台灣首都]
觀察 1: 台北是台灣的首都

思考 2: 我現在知道答案了
最終答案: 台北
```

## 🛠 工具設計

好的工具應該：

1. **單一職責**: 每個工具只做一件事
2. **清晰描述**: LLM 需要知道工具的功能
3. **結構化輸入/輸出**: 便於解析
4. **錯誤處理**: 返回有意義的錯誤訊息

範例：

```python
def search_tool(query: str) -> str:
    """
    搜尋網路資訊

    Args:
        query: 搜尋關鍵詞

    Returns:
        搜尋結果摘要
    """
    # 實作搜尋邏輯
    ...
```

## 🎯 學習目標

完成這些範例後，你應該能夠：

- ✅ 理解 ReAct 的思考-行動-觀察循環
- ✅ 設計和實作自定義工具
- ✅ 編寫有效的 ReAct 提示
- ✅ 處理 Agent 的錯誤和邊界情況
- ✅ 評估 Agent 的性能

## 📊 範例輸出

執行 `react_agent_basic.py` 的輸出示例：

```
問題：台灣最高的山是什麼？它的海拔是多少？

執行 ReAct 循環...

步驟 1:
思考: 我需要搜尋台灣最高的山
行動: Search[台灣最高的山]
觀察: 玉山是台灣最高的山，海拔3,952公尺

步驟 2:
思考: 我已經找到了答案
最終答案: 玉山，海拔3,952公尺

答案：玉山，海拔3,952公尺
```

## 🔍 進階主題

### 1. 錯誤恢復

如何處理工具呼叫失敗：

```python
try:
    result = tool.execute(params)
except ToolError as e:
    # 返回錯誤給 LLM，讓它決定下一步
    observation = f"錯誤：{str(e)}。請嘗試其他方法。"
```

### 2. 成本優化

- 限制最大迭代次數
- 使用較便宜的模型（gpt-3.5-turbo）
- 快取常見查詢結果

### 3. 性能提升

- 並行執行獨立的工具呼叫
- 使用向量資料庫加速檢索
- 實作工具結果快取

## ❓ 常見問題

**Q: ReAct 容易陷入循環嗎？**
A: 是的。解決方法：
- 設置最大迭代次數
- 檢測重複的思考-行動模式
- 提供明確的終止條件

**Q: 如何選擇使用哪個工具？**
A: LLM 根據工具描述和當前狀態決定。關鍵是：
- 工具描述要清晰
- 提供 Few-Shot 範例
- 使用強大的模型（gpt-4）

**Q: ReAct vs Function Calling 有什麼區別？**
A:
- **ReAct**: 顯式的推理過程，可解釋性強
- **Function Calling**: 黑盒決策，更高效

兩者可以結合使用。

## 📚 延伸閱讀

- [ReAct 論文](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct 文檔](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [提示工程指南](https://www.promptingguide.ai/)

## 🤝 貢獻

發現問題或有改進建議？歡迎提交 Issue 或 PR！
