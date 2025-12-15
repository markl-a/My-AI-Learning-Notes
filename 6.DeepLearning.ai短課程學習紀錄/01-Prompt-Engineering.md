# ChatGPT Prompt Engineering for Developers

## 📋 課程概述

這門課程由 Andrew Ng 和 OpenAI 的 Isa Fulford 共同教授，專注於教導開發者如何使用大型語言模型（LLM）建立應用程式。

### 課程目標
- 掌握提示工程（Prompt Engineering）的核心原則
- 學習如何撰寫有效的提示詞
- 了解 LLM 在實際應用中的最佳實踐
- 建立實用的 AI 應用程式

### 適合對象
- Python 開發者（具備基礎程式設計能力）
- 想要整合 LLM 到應用程式的工程師
- AI/ML 產品經理和研究人員

### 課程時長
約 1 小時

## 🎯 核心概念

### 兩大核心原則

#### 1. 撰寫清晰明確的指令（Write Clear and Specific Instructions）

**原則說明**：提示詞應該清晰表達你想要模型做什麼，提供足夠的上下文資訊。

**最佳實踐**：

##### a) 使用分隔符號（Delimiters）

使用分隔符號可以清楚標示輸入的不同部分，避免提示注入攻擊。

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0  # 控制隨機性，0 表示最確定的輸出
    )
    return response.choices[0].message.content

# 使用三個引號作為分隔符號
text = """
你應該透過提供盡可能清晰和具體的指令來表達你希望模型執行的任務。\
這將引導模型朝向期望的輸出，並降低收到無關或不正確回應的可能性。\
不要將撰寫清晰的提示與撰寫簡短的提示混為一談。\
在許多情況下，較長的提示為模型提供了更多的清晰度和上下文，\
這實際上可以產生更詳細和相關的輸出。
"""

prompt = f"""
將由三個反引號分隔的文字總結成一句話。
```{text}```
"""

response = get_completion(prompt)
print(response)
```

**常用分隔符號**：
- 三個反引號：` ``` `
- 三個引號：`"""`
- 三個破折號：`---`
- XML 標籤：`<tag></tag>`
- 角括號：`< >`

##### b) 要求結構化輸出

請求 JSON、HTML 或其他結構化格式的輸出，便於程式處理。

```python
prompt = """
生成三本虛構書籍的清單，包含書名、作者和類別。
以 JSON 格式提供，包含以下鍵值：book_id, title, author, genre。
"""

response = get_completion(prompt)
print(response)
```

**輸出範例**：
```json
[
  {
    "book_id": 1,
    "title": "時間迴廊的祕密",
    "author": "林靜雯",
    "genre": "科幻小說"
  },
  {
    "book_id": 2,
    "title": "古城夜譚",
    "author": "陳明哲",
    "genre": "推理懸疑"
  },
  {
    "book_id": 3,
    "title": "茶道心語",
    "author": "王美琴",
    "genre": "生活散文"
  }
]
```

##### c) 要求模型檢查條件是否滿足

讓模型在執行任務前先檢查假設條件。

```python
text_1 = """
泡一杯茶很簡單！首先，你需要把水煮開。\
在進行過程中，拿一個杯子並放入茶包。\
一旦水夠熱了，就把它倒在茶包上。\
讓它靜置一會兒，讓茶葉浸泡。幾分鐘後，\
取出茶包。如果你喜歡，可以加一些糖或牛奶調味。\
就這樣，你可以享受一杯美味的茶了！
"""

prompt = f"""
你將獲得由三個引號分隔的文字。
如果它包含一系列的指令，\
請按照以下格式重寫這些指令：

步驟 1 - ...
步驟 2 - ...
...
步驟 N - ...

如果文字不包含一系列的指令，\
則簡單地寫「未提供步驟」。

\"\"\"{text_1}\"\"\"
"""

response = get_completion(prompt)
print("完成後的文字 1:")
print(response)
```

**輸出**：
```
完成後的文字 1:
步驟 1 - 把水煮開
步驟 2 - 拿一個杯子並放入茶包
步驟 3 - 將熱水倒在茶包上
步驟 4 - 讓茶葉浸泡幾分鐘
步驟 5 - 取出茶包
步驟 6 - 根據喜好加糖或牛奶
步驟 7 - 享受你的茶
```

##### d) 少樣本提示（Few-shot Prompting）

提供成功執行任務的範例，然後要求模型執行任務。

```python
prompt = """
你的任務是以一致的風格回答問題。

<孩子>: 教我何謂耐心。

<祖父>: 挖掘最深峽谷的河流源自一處不起眼的泉源；\
最宏偉的交響樂源自單一音符；\
最精緻的織錦始於一條孤獨的線。

<孩子>: 教我何謂韌性。
"""

response = get_completion(prompt)
print(response)
```

#### 2. 給模型時間「思考」（Give the Model Time to Think）

**原則說明**：如果模型匆忙得出不正確的結論，應該重新設計查詢，在模型提供最終答案之前請求一連串相關的推理。

##### a) 指定完成任務所需的步驟

```python
text = """
在一個迷人的村莊裡，兄妹傑克和吉兒出發去從山頂的井裡打水。\
他們一邊唱著歡樂的歌，一邊往上爬，\
然而不幸降臨——傑克被一塊石頭絆倒，從山上滾下來，\
吉兒緊隨其後。雖然受了點傷，\
他們仍然回到了溫馨的家中擁抱。\
儘管發生了意外，他們的冒險精神依然不減，\
他們繼續愉快地探索。
"""

prompt = f"""
執行以下操作：
1 - 用一句話總結以下由三個反引號分隔的文字。
2 - 將摘要翻譯成英文。
3 - 在英文摘要中列出每個名字。
4 - 輸出包含以下鍵值的 JSON 物件：english_summary, num_names。

請用換行符號分隔你的答案。

文字：
```{text}```
"""

response = get_completion(prompt)
print("提示的完成：")
print(response)
```

**要求指定格式的輸出**：

```python
prompt = f"""
你的任務是執行以下操作：
1 - 用一句話總結以下由 <> 分隔的文字。
2 - 將摘要翻譯成英文。
3 - 在英文摘要中列出每個名字。
4 - 輸出包含以下鍵值的 JSON 物件：
   english_summary, num_names。

使用以下格式：
文字：<要總結的文字>
摘要：<摘要>
翻譯：<摘要的翻譯>
名字：<英文摘要中的名字列表>
輸出 JSON：<包含 english_summary 和 num_names 的 JSON>

文字：<{text}>
"""

response = get_completion(prompt)
print("\n提示 2 的完成：")
print(response)
```

##### b) 指導模型在匆忙得出結論之前找出自己的解決方案

```python
prompt = """
判斷學生的解答是否正確。

問題：
我正在建造一個太陽能發電裝置，需要幫忙計算財務。
- 土地成本為每平方英尺 100 美元
- 我可以以每平方英尺 250 美元購買太陽能板
- 我協商了一份維護合約，每年固定費用為 100,000 美元，\
  另加每平方英尺 10 美元
作為平方英尺數量的函數，第一年營運的總成本是多少？

學生的解答：
設 x 為裝置的大小（平方英尺）。
成本：
1. 土地成本：100x
2. 太陽能板成本：250x
3. 維護成本：100,000 + 100x
總成本：100x + 250x + 100,000 + 100x = 450x + 100,000
"""

response = get_completion(prompt)
print(response)
```

**注意**：模型同意學生的答案，但實際上是錯的！維護成本應該是 `100,000 + 10x`，而非 `100,000 + 100x`。

**改進的提示**：

```python
prompt = """
你的任務是判斷學生的解答是否正確。
要解決這個問題，請執行以下步驟：
- 首先，自己解決問題。
- 然後將你的解答與學生的解答進行比較，\
  並評估學生的解答是否正確。
在你自己解決問題之前，不要決定學生的解答是否正確。

使用以下格式：
問題：
'''
問題文字
'''
學生的解答：
'''
學生的解答
'''
實際解答：
'''
解決問題的步驟和你的解答
'''
學生的解答與實際解答是否相同：
'''
是或否
'''
學生的成績：
'''
正確或不正確
'''

問題：
'''
我正在建造一個太陽能發電裝置，需要幫忙計算財務。
- 土地成本為每平方英尺 100 美元
- 我可以以每平方英尺 250 美元購買太陽能板
- 我協商了一份維護合約，每年固定費用為 100,000 美元，\
  另加每平方英尺 10 美元
作為平方英尺數量的函數，第一年營運的總成本是多少？
'''
學生的解答：
'''
設 x 為裝置的大小（平方英尺）。
成本：
1. 土地成本：100x
2. 太陽能板成本：250x
3. 維護成本：100,000 + 100x
總成本：100x + 250x + 100,000 + 100x = 450x + 100,000
'''
實際解答：
"""

response = get_completion(prompt)
print(response)
```

## 💡 實際應用案例

### 1. 文本摘要（Summarizing）

```python
prod_review = """
我為女兒的生日買了這隻熊貓娃娃，她很喜歡，並且到哪都帶著它。\
它很柔軟、超級可愛，而且臉看起來很友善。\
不過相對於我付的價格來說有點小。\
我想可能還有其他更大的選擇，價格是一樣的。\
它比預期早一天到貨，所以我在送給女兒之前有機會自己玩了一下。
"""

prompt = f"""
你的任務是從電商網站的產品評論中生成簡短摘要。

請將下方三個反引號分隔的評論摘要成最多 30 個字。

評論：```{prod_review}```
"""

response = get_completion(prompt)
print(response)
```

**針對特定部門的摘要**：

```python
prompt = f"""
你的任務是從電商網站的產品評論中生成簡短摘要，\
以便向運輸部門提供反饋。

請將下方三個反引號分隔的評論摘要成最多 30 個字，\
並專注於提及產品運輸和交付的任何方面。

評論：```{prod_review}```
"""

response = get_completion(prompt)
print(response)
```

### 2. 推論（Inferring）

#### 情感分析

```python
lamp_review = """
我需要一盞漂亮的臥室燈，這盞燈有額外的儲物空間，\
價格也不算太高。收到得很快——在兩天內就到了。\
在運輸過程中，我們的燈繩斷了，公司很樂意寄來一個新的。\
幾天內也到貨了。組裝很容易。\
然後我發現有一個零件遺失了，所以我聯繫了他們的客服，\
他們很快就給我寄來了遺失的零件！\
對我來說，Lumina 是一家關心客戶和產品的好公司。
"""

prompt = f"""
以下產品評論的情感是什麼？\
評論由三個反引號分隔。

評論文字：'''{lamp_review}'''
"""

response = get_completion(prompt)
print(response)
```

**以單一詞彙表達情感**：

```python
prompt = f"""
以下產品評論的情感是什麼？\
評論由三個反引號分隔。

用一個詞回答，「正面」或「負面」。

評論文字：'''{lamp_review}'''
"""

response = get_completion(prompt)
print(response)
```

#### 識別情緒類型

```python
prompt = f"""
識別以下評論作者表達的情緒類型列表。\
列表中不要超過五個項目。將你的答案格式化為\
以逗號分隔的小寫單詞列表。

評論文字：'''{lamp_review}'''
"""

response = get_completion(prompt)
print(response)
```

#### 提取產品和公司名稱

```python
prompt = f"""
從評論文字中識別以下項目：
- 評論者購買的產品
- 製造該產品的公司

評論由三個反引號分隔。\
將你的回應格式化為 JSON 物件，\
鍵值為「產品」和「品牌」。

評論文字：'''{lamp_review}'''
"""

response = get_completion(prompt)
print(response)
```

### 3. 文本轉換（Transforming）

#### 翻譯

```python
prompt = f"""
將以下英文文字翻譯成繁體中文：
```Hi, I would like to order a blender```
"""

response = get_completion(prompt)
print(response)
```

**識別語言**：

```python
prompt = f"""
告訴我以下文字是什麼語言：
```Combien coûte le lampadaire?```
"""

response = get_completion(prompt)
print(response)
```

**多語言翻譯**：

```python
prompt = f"""
將以下文字翻譯成繁體中文和韓文：
```I want to order a basketball```
"""

response = get_completion(prompt)
print(response)
```

#### 語氣轉換

```python
prompt = f"""
將以下文字從俚語翻譯成正式的商業信函：
'老兄，這是 Joe，看看這個規格書。'
"""

response = get_completion(prompt)
print(response)
```

#### 格式轉換

```python
data_json = { "restaurant employees" :[
    {"name":"林小明", "email":"xiaoming.lin@example.com"},
    {"name":"陳美華", "email":"meihua.chen@example.com"},
    {"name":"王大偉", "email":"dawei.wang@example.com"}
]}

prompt = f"""
將以下 Python 字典從 JSON 轉換為 HTML 表格，\
保留表格標題和欄位名稱：{data_json}
"""

response = get_completion(prompt)
print(response)
```

#### 拼寫和文法檢查

```python
text = [
  "The girl with the black and white puppies have a ball.",  # 文法錯誤
  "Yolanda has her notebook.", # 正確
  "Its going to be a long day. Does the car need it's oil changed?",  # its/it's
  "Their goes my freedom. There going to bring they're suitcases.",  # there/their/they're
]

for t in text:
    prompt = f"""校對並更正以下文字，\
    並重寫更正後的版本。如果你沒有找到\
    任何錯誤，就說「未發現錯誤」。\
    不要在文字周圍使用任何標點符號：
    ```{t}```"""

    response = get_completion(prompt)
    print(response)
```

### 4. 文本擴展（Expanding）

```python
# 給定客戶評論和情感
sentiment = "負面"

review = """
他們在十一月份仍然有季節性的銷售價格，\
這還算不錯。但大約一週後，我在自己的\
臥室裡查看同樣的系統時，發現同樣的系統\
在同一網站上的價格下降了約 $50 左右。\
它還配備了額外的物品，雖然我想不起來是什麼了。\
客戶服務很好。我聯繫了他們，他們給了我差價。
"""

prompt = f"""
你是一位客戶服務 AI 助理。
你的任務是向尊貴的客戶發送電子郵件回覆。
給定由 ``` 分隔的客戶電子郵件，\
生成一個回覆以感謝客戶的評論。
如果情感是正面或中性，感謝他們的評論。
如果情感是負面，道歉並建議他們可以聯繫客戶服務。
確保使用評論中的具體細節。
用簡潔和專業的語氣撰寫。
將電子郵件簽名為『AI 客戶代理』。
客戶評論：```{review}```
評論情感：{sentiment}
"""

response = get_completion(prompt)
print(response)
```

## ⚠️ 模型限制

### 幻覺（Hallucinations）

LLM 有時會生成聽起來合理但實際上不真實的陳述。

```python
prompt = """
告訴我關於 AeroGlide UltraSlim Smart 牙刷\
由 Boie 公司生產的資訊
"""

response = get_completion(prompt)
print(response)
```

**注意**：Boie 是一家真實的公司，但產品名稱是虛構的！模型可能會編造不存在的產品詳細資訊。

### 減少幻覺的策略

1. **要求模型先找到相關引文，然後根據這些引文回答問題**
2. **要求模型追溯答案到原始文檔**
3. **使用 RAG（檢索增強生成）技術**

```python
prompt = """
告訴我關於 AeroGlide UltraSlim Smart 牙刷\
由 Boie 公司生產的資訊。

如果你不知道答案，請說「我不知道」，\
而不是編造資訊。
"""

response = get_completion(prompt)
print(response)
```

## 🛠️ 實務技巧

### Temperature 參數調整

```python
# Temperature = 0：更確定、更一致的輸出（適合生產環境）
response = get_completion(prompt, temperature=0)

# Temperature = 0.7：更有創意、更多樣化的輸出（適合創意任務）
response = get_completion(prompt, temperature=0.7)

# Temperature = 1.0：最大隨機性（適合頭腦風暴）
response = get_completion(prompt, temperature=1.0)
```

### 迭代式提示開發

提示工程是一個迭代過程：

1. **第一版提示**：寫一個初始提示
2. **測試結果**：檢查輸出
3. **分析問題**：找出不符合預期的地方
4. **改進提示**：調整指令、增加範例、改變格式
5. **重複**：持續迭代直到滿意

### 完整應用範例：產品評論分析系統

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ReviewAnalyzer:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def get_completion(self, prompt, temperature=0):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def analyze_sentiment(self, review):
        """分析評論情感"""
        prompt = f"""
        分析以下產品評論的情感。
        用一個詞回答：「正面」、「負面」或「中性」。

        評論：'''{review}'''
        """
        return self.get_completion(prompt)

    def extract_topics(self, review):
        """提取評論中的主題"""
        prompt = f"""
        識別以下評論中提到的主題。
        將答案格式化為 JSON 陣列。
        可能的主題包括：價格、品質、運輸、客服、功能。

        評論：'''{review}'''
        """
        return self.get_completion(prompt)

    def generate_response(self, review, sentiment):
        """生成客服回覆"""
        prompt = f"""
        你是一位專業的客戶服務代表。
        根據以下評論和情感分析，生成一封簡短的回覆郵件。

        評論：'''{review}'''
        情感：{sentiment}

        要求：
        - 感謝客戶的反饋
        - 如果是負面評論，表示歉意並提供解決方案
        - 如果是正面評論，表達感謝
        - 保持專業和友善的語氣
        - 簽名為「客戶服務團隊」
        """
        return self.get_completion(prompt)

    def full_analysis(self, review):
        """完整分析流程"""
        print("=" * 50)
        print("原始評論：")
        print(review)
        print("\n" + "=" * 50)

        # 情感分析
        sentiment = self.analyze_sentiment(review)
        print(f"\n情感分析：{sentiment}")

        # 主題提取
        topics = self.extract_topics(review)
        print(f"\n主題分析：\n{topics}")

        # 生成回覆
        response = self.generate_response(review, sentiment)
        print(f"\n建議回覆：\n{response}")
        print("=" * 50)

        return {
            "sentiment": sentiment,
            "topics": topics,
            "response": response
        }

# 使用範例
if __name__ == "__main__":
    analyzer = ReviewAnalyzer()

    review = """
    這個產品真的超出我的預期！品質非常好，
    價格也很合理。運輸速度很快，包裝也很完善。
    唯一的小缺點是說明書有點簡略，
    但整體來說非常滿意。會推薦給朋友！
    """

    result = analyzer.full_analysis(review)
```

## 📚 延伸學習

### 進階主題
1. **Chain-of-Thought Prompting**：引導模型逐步推理
2. **System Messages**：使用系統訊息設定模型行為
3. **Function Calling**：讓模型呼叫外部函數
4. **Fine-tuning**：針對特定任務微調模型

### 推薦資源
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide by DAIR.AI](https://www.promptingguide.ai/)
- [LangChain Documentation](https://python.langchain.com/)

## ✅ 重點回顧

1. **兩大核心原則**：
   - 撰寫清晰明確的指令
   - 給模型時間思考

2. **關鍵技巧**：
   - 使用分隔符號
   - 要求結構化輸出
   - 提供範例（Few-shot）
   - 指定步驟
   - 讓模型檢查假設

3. **常見應用**：
   - 摘要
   - 推論（情感分析、主題提取）
   - 轉換（翻譯、格式轉換）
   - 擴展（內容生成）

4. **注意事項**：
   - 注意幻覺問題
   - 使用適當的 temperature
   - 迭代改進提示
   - 添加驗證機制

---

**課程連結**：[DeepLearning.ai - ChatGPT Prompt Engineering](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

**完成日期**：2025-01-17
