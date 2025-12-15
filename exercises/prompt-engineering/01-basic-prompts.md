# 練習 1.1：基礎 Prompt 設計

**難度**: ⭐ 入門
**預計時間**: 30 分鐘
**前置知識**: 基本 Python、API 調用概念

## 學習目標

完成本練習後，你將能夠：

- [ ] 理解 Prompt 的基本結構
- [ ] 編寫清晰明確的指令
- [ ] 使用角色設定改善輸出
- [ ] 比較不同 Prompt 的效果差異

## 背景知識

Prompt Engineering 的核心在於：**清晰、具體、有上下文**。

一個好的 Prompt 通常包含：

1. **角色設定** (Role)：定義 AI 的身份
2. **任務描述** (Task)：明確要做什麼
3. **上下文** (Context)：提供必要背景
4. **輸出格式** (Format)：期望的回應格式
5. **範例** (Examples)：可選的示例

## 練習任務

### 任務 1：改善模糊 Prompt

以下是一個模糊的 Prompt，請改寫使其更加明確：

**原始 Prompt:**
```
幫我寫一封郵件
```

**你的改進版本:**
```
請幫我撰寫一封給客戶的回覆郵件：

背景：
- 客戶詢問產品退貨政策
- 我們的政策是 30 天無理由退貨
- 需要提供退貨流程

要求：
- 語氣專業友善
- 長度控制在 150 字以內
- 包含退貨步驟編號列表
```

### 任務 2：角色設定實驗

使用相同問題，比較不同角色設定的效果：

```python
import openai

question = "如何學習程式設計？"

# Prompt 1: 無角色設定
prompt1 = question

# Prompt 2: 教師角色
prompt2 = f"""你是一位有 10 年教學經驗的程式設計講師。
請回答學生的問題：{question}
回答應該簡潔易懂，適合初學者。"""

# Prompt 3: 業界專家角色
prompt3 = f"""你是 Google 資深軟體工程師。
基於你的實戰經驗，回答：{question}
著重於業界實用技能。"""

# 執行並比較結果
# TODO: 完成代碼
```

**你的任務：**

1. 完成上面的代碼
2. 記錄三種角色的回答差異
3. 分析哪種角色最適合什麼場景

### 任務 3：輸出格式控制

要求 AI 以特定格式輸出：

```python
# 練習：讓 AI 輸出 JSON 格式

prompt = """分析以下電影評論的情感：

評論：「這部電影劇情緊湊，演員演技精湛，但結局有些倉促。整體來說值得一看。」

請以 JSON 格式輸出，包含：
- sentiment: 正面/負面/中性
- score: 1-10 的評分
- keywords: 關鍵詞列表
- summary: 一句話總結

只輸出 JSON，不要其他說明。
"""

# TODO: 執行並驗證輸出是否為有效 JSON
```

### 任務 4：迭代優化

從一個簡單 Prompt 開始，逐步優化：

**第一版：**
```
總結這篇文章
```

**第二版（加入長度限制）：**
```
用 3 句話總結這篇文章
```

**第三版（加入格式要求）：**
```
用 3 個要點總結這篇文章，每個要點一句話
```

**你的任務：** 繼續優化到第五版，每次添加一個改進。

## 驗證方法

完成以下自我檢查：

- [ ] 任務 1：改進版 Prompt 包含明確的背景、要求和格式
- [ ] 任務 2：成功運行代碼並記錄三種輸出的差異
- [ ] 任務 3：AI 輸出可被 `json.loads()` 解析
- [ ] 任務 4：第五版 Prompt 明顯優於第一版

## 參考解答

<details>
<summary>點擊查看參考解答</summary>

### 任務 2 參考代碼

```python
from openai import OpenAI

client = OpenAI()

def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# 執行三個 prompt
results = {
    "無角色": get_response(prompt1),
    "教師": get_response(prompt2),
    "工程師": get_response(prompt3)
}

for role, response in results.items():
    print(f"\n=== {role} ===")
    print(response)
```

### 任務 4 第五版示例

```
請閱讀以下文章，並提供結構化摘要：

[文章內容]

要求：
1. 用 3 個要點總結核心內容
2. 每個要點控制在 20 字以內
3. 使用「→」符號表示因果關係
4. 最後一行給出適合的標籤（最多3個）

格式：
• 要點1
• 要點2
• 要點3
標籤：#標籤1 #標籤2 #標籤3
```

</details>

## 延伸思考

1. **溫度參數影響**：嘗試不同的 `temperature` 值（0.0, 0.5, 1.0），觀察輸出的變化
2. **Prompt 注入風險**：思考如何防止用戶輸入破壞你的 Prompt 結構
3. **多語言考量**：同一個 Prompt 在中英文下的效果差異

## 下一步

完成本練習後，繼續學習：
- [練習 1.2：Few-shot 學習實作](./02-few-shot.md)
- [Prompt Engineering 完整指南](../../3.LLM應用工程/3.提示工程學/README.md)
