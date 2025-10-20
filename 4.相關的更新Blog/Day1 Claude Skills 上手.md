**Claude Agent Skills 從入門到應用的學習指南**。

-----

### **Claude Agent Skills 學習指南：從入門到上手**

#### **前言：什麼是 Agent Skills？為什麼要學？**

想像一下，您是一位大廚，而 Claude 是您的得力助手。您每天都要教他如何處理食材、用什麼火侯、如何擺盤。這很費時，而且每次都可能有些微差異。

**Agent Skills** 就如同您為這位助手寫下的一套**標準作業流程 (SOP) 食譜**。您只需將複雜的工作流程、個人的專業知識或團隊的統一規範，打包成一個個「技能包 (Skill)」。當需要執行特定任務時，直接告訴 Claude 使用哪個「技能」，他就能精準、高效、穩定地完成工作。

**學習 Agent Skills 的核心價值：**

  * **個人化與效率：** 將您獨特的重複性工作流程自動化，大幅提升效率。
  * **知識傳承與協作：** 將專家知識或團隊規範打包，方便團隊成員共享，確保輸出品質一致。
  * **釋放 AI 全部潛力：** 突破簡單的問答，讓 Claude 能夠執行更複雜、多步驟的任務，成為真正的「代理人 (Agent)」。

-----

### **第一章：基礎核心概念**

在動手之前，先理解 Skills 是如何運作的。

#### **1. Skills 的核心架構**

  * **虛擬機環境 (Virtual Machine)：** Claude 會為您的 Skill 提供一個安全的沙盒環境。在這個環境裡，他不僅能讀寫檔案，還能執行程式碼，目前支援 **Bash、Python 和 Node.js**。這代表您的 Skill 不僅僅是文字指令，還可以包含強大的腳本。
  * **技能目錄 (Skill Directory)：** 每個 Skill 都是一個獨立的資料夾，裡面包含了所有必要的檔案。
  * **打包與部署 (Packaging & Deployment)：** 您需要將整個技能目錄壓縮成一個 `.zip` 檔案，然後上傳到 Claude 的設定中啟用。

#### **2. Skill 的靈魂：`skill.md` 檔案**

這是每個 Skill 中**最核心、必須存在**的檔案，它定義了 Skill 的一切。它由兩個主要部分組成：

  * **元數據 (Metadata)：**
      * 位於檔案的最頂部，用 `---` 包圍。
      * **`name` (名稱):** (必填) Skill 的唯一識別名稱。
      * **`description` (描述):** (必填) 用簡潔的語言描述這個 Skill 的功能。Claude 會根據這個描述來判斷何時該使用您的 Skill。
      * **`version` (版本):** (選填) 方便您管理 Skill 的更新迭代。
      * **`dependencies` (依賴):** (選填) 如果您的 Skill 依賴其他 Skill，可以在這裡聲明。
  * **指令 (Instructions)：**
      * 位於元數據下方。
      * 這是您給 Claude 的詳細步驟說明，告訴他「如何」完成這個 Skill 所定義的任務。
      * **關鍵點：** Claude 並不會一開始就加載所有指令。他會先讀取所有已啟用 Skill 的 `metadata`，當您的需求觸發了某個 Skill 的 `description` 時，他才會去加載對應的 `instructions` 來執行。

-----

### **第二章：手把手，創建您的第一個 Skill**

我們以影片中的「個人化寫作風格」為例，一步步創建一個實用的 Skill。

#### **步驟一：定義目標 (The Goal)**

  * **我想解決什麼問題？** 我希望 Claude 根據網路文章重寫內容時，能符合我的個人寫作風格，避免生硬的「AI 感」。
  * **具體需求是什麼？**
    1.  避免過度使用項目符號 (bullet points) 和列表。
    2.  文章結構要流暢，段落間有自然的過渡。
    3.  使用正確的中文標點符號。
    4.  根據情境（如技術分享 vs. 個人筆記）調整語氣。
    5.  正確轉換人稱（例如將原文的 "I" 轉述為 "作者發現..."）。

#### **步驟二：與 Claude 對話，共同撰寫指令**

這一步是影片的精華所在。您不需要自己從零開始寫指令，而是**讓 Claude 成為您開發 Skill 的夥伴**。

1.  **提供背景知識：** 將官方的 Agent Skills 開發文檔連結或內容丟給 Claude，讓他先「學習」如何開發 Skill。
2.  **描述您的目標：** 用自然語言告訴 Claude 您想創建一個什麼樣的 Skill，以及您的具體需求（參考步驟一）。
3.  **回答 Claude 的提問：** Claude 會像一個需求分析師一樣，向您提問以釐清細節。例如：「您偏好的文章結構是什麼樣的？」、「您希望在轉述時如何處理人稱？」您回答得越詳細，最終生成的指令就越精準。
4.  **讓 Claude 生成 `skill.md`：** 經過幾輪的溝通，Claude 就能完全理解您的意圖，並為您生成一份高品質的 `skill.md` 檔案內容。

#### **步驟三：打包 Skill**

這是最容易出錯的一步，請務必遵循正確的結構。

1.  創建一個資料夾，例如 `personal_writing_style`。
2.  將 Claude 生成的 `skill.md` 檔案放入這個資料夾。
3.  **將 `personal_writing_style` 這個資料夾本身** 壓縮成 `.zip` 檔案。

**錯誤的結構：** 直接壓縮 `skill.md` 檔案。
**正確的結構：** 壓縮包解壓縮後，應該是一個**資料夾**，而不是一堆散落的檔案。

```
personal_writing_style.zip
└── personal_writing_style/
    └── skill.md
```

#### **步驟四：上傳與測試**

1.  **上傳：** 進入 Claude 桌面應用或網頁版的設定 (Settings) -\> 功能 (Capabilities)，上傳您的 `.zip` 技能包並啟用它。
2.  **測試：**
      * **對照組：** 先不使用 Skill，向 Claude 提出原始需求（例如：「幫我根據這篇文章寫一篇技術部落格」），觀察其原始輸出。
      * **實驗組：** 提出同樣的需求，但明確指示使用您的 Skill（例如：「請使用 `personal_writing_style` 這個技能，幫我根據這篇文章寫一篇技術部落格」）。
3.  **驗證：** 比較兩個版本的輸出，驗證 Skill 是否達到了您預期的效果。

-----

### **第三章：進階技巧，讓您的 Skill 更強大**

當您掌握了基礎後，可以嘗試以下進階功能。

#### **1. 整合腳本 (Python/Node.js)**

如果您的任務需要數據處理、API 調用或複雜的邏輯運算，可以在 Skill 中加入腳本。

  * **結構：** 在您的技能目錄中創建一個 `scripts` 資料夾，將 `.py` 或 `.js` 檔案放入其中。
  * **調用：** 在 `skill.md` 的指令中，使用 Bash 命令來執行您的腳本，例如：
    ````markdown
    To analyze the data, run the following script and report the results:

    ```bash
    python scripts/analyze_data.py --input /path/to/data.csv
    ````
    ```
    
    ```

#### **2. 拆分複雜指令**

如果您的指令非常長，可以將其拆分成多個 `.md` 檔案，以保持主 `skill.md` 的清晰。

  * **結構：** 在技能目錄中創建額外的 Markdown 檔案，例如 `advanced_forms.md`。
  * **引用：** 在 `skill.md` 中，引導 Claude 去閱讀其他檔案，例如：
    ```markdown
    For basic tasks, follow the steps below. For advanced form filling, please refer to the instructions in `advanced_forms.md`.
    ```

#### **3. 編寫高品質指令的最佳實踐**

  * **清晰明確：** 使用簡單、直接的語言。避免模棱兩可的詞彙。
  * **提供範例 (Few-shot Prompting)：** 在指令中給出「好的範例」和「壞的範例」，讓 Claude 更清楚您的標準。
  * **設定角色：** 在指令開頭可以為 Claude 設定一個角色，例如：「你現在是一位資深的技術文章作者...」。
  * **結構化指令：** 使用標題、列表和程式碼區塊來組織您的指令，使其易於閱讀和理解。

-----
這裡提供三個由淺入深的 Claude Agent Skills 範例，包含完整的程式碼和檔案結構，您可以直接複製使用。

-----

### **範例一：會議記錄整理大師 (純文字處理)**

這個 Skill 不需要寫任何外部腳本，純粹依靠 `skill.md` 的強大指令，將雜亂的會議筆記整理成專業、標準化的格式。

#### **🎯 用途**

將一段貼上來的會議逐字稿或零散筆記，自動轉換為包含決議事項 (Action Items)、重點摘要的結構化會議記錄。

#### **📁 檔案結構**

```
meeting_summarizer.zip
└── meeting_summarizer/
    └── skill.md
```

#### **📄 `skill.md` 內容**

```markdown
---
name: meeting_summarizer
version: 1.0
description: 將雜亂的會議逐字稿或筆記整理成標準化的會議記錄格式。
---
你現在是一位專業的會議記錄員。你的任務是將使用者提供的文本整理成一份清晰、專業的會議記錄。請嚴格遵循以下格式輸出：

### 會議主題：[根據內容自動生成主題]
* **日期：** [填寫今天的日期，格式為 YYYY-MM-DD]
* **與會人員：** [從文本中識別並列出所有與會者，如果無法識別則填寫「未提供」]

---

#### 📌 決議事項 (Action Items)
* **[AI-01]** [具體任務描述] - **負責人：** @[姓名] - **預計完成日期：** [YYYY-MM-DD]
* **[AI-02]** [具體任務描述] - **負責人：** @[姓名] - **預計完成日期：** [YYYY-MM-DD]
* *(如果沒有決議事項，請在此處註明「本次會議無具體決議事項」)*

---

#### 📝 重點摘要
1.  **[重點一]**: [用一到兩句話總結第一個討論重點]
2.  **[重點二]**: [用一到兩句話總結第二個討論重點]
3.  **[其他重要資訊]**: [任何其他值得記錄的關鍵討論或資訊]

---

**指導原則：**
- **決議事項** 必須包含明確的任務、唯一的負責人和完成日期。如果文本中沒有提到日期，請標示為「待定」。
- **重點摘要** 應高度概括，避免口語化，專注於結論和關鍵資訊。
- 保持整體格式的簡潔與專業。
```

#### **🚀 如何使用**

在對話框中輸入：

> 請使用 `meeting_summarizer` 技能，幫我整理這份會議筆記：「好的各位，今天我們來同步一下 Q4 的專案進度。小王，上次說的那個 UI 優化，麻煩你這週五前完成。另外，Amy，資料庫的壓力測試報告下週一要出來喔。我們討論了三個方案，最後決定採用方案B，因為成本最低。大概就是這樣。」

-----

### **範例二：Python 註解產生器 (整合 Python 腳本)**

這個 Skill 展示了如何整合 Python 腳本來執行程式碼分析，然後讓 Claude 根據分析結果生成高品質的註解。

#### **🎯 用途**

自動為一個 Python 檔案中的所有函式，生成符合 Google 風格的 Docstring 註解。

#### **📁 檔案結構**

```
docstring_generator.zip
└── docstring_generator/
    ├── skill.md
    └── scripts/
        └── analyze_code.py
```

#### **🐍 `scripts/analyze_code.py` 內容**

*這個腳本使用 Python 內建的 `ast` 模組來安全地解析程式碼結構。*

```python
# scripts/analyze_code.py
import ast
import sys
import json

def analyze_functions(filepath):
    """
    分析指定的 Python 檔案，提取所有函式的結構資訊。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as source:
            tree = ast.parse(source.read())
    except FileNotFoundError:
        print(json.dumps({"error": f"檔案未找到: {filepath}"}))
        return

    functions_data = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "returns": ast.unparse(node.returns) if node.returns else None
            }
            functions_data.append(func_info)
    
    # 以 JSON 格式輸出，方便 Claude 解析
    print(json.dumps(functions_data))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_functions(sys.argv[1])
    else:
        print(json.dumps({"error": "請提供檔案路徑作為參數"}))
```

#### **📄 `skill.md` 內容**

````markdown
---
name: python_docstring_generator
version: 1.0
description: 分析 Python 程式碼檔案，並為其中的函式自動生成 Google 風格的 docstring 註解。
---
你是一個 Python 程式碼文件專家。你的任務是為使用者提供的 Python 檔案中的每個函式生成 Google 風格的 docstring。

**執行步驟：**
1.  使用者會提供一個本地的 Python 檔案路徑。
2.  執行 `scripts/analyze_code.py` 腳本來分析該檔案的結構。命令如下：
    ```bash
    python scripts/analyze_code.py [使用者提供的檔案路徑]
    ```
3.  腳本會輸出一串 JSON，其中包含了所有函式的名稱、參數和回傳型別。
4.  根據這份 JSON 提供的資訊，為**每一個**函式生成一個完整的、符合 Google 風格的 docstring。

**Google Docstring 模板：**
```python
"""[對函式功能的簡潔描述].

Args:
    [參數1_名稱] ([參數1_型別]): [參數1的描述].
    [參數2_名稱] ([參數2_型別]): [參數2的描述].

Returns:
    [回傳值型別]: [對回傳值的描述].
"""
````

  - 如果函式沒有參數，請省略 `Args:` 區塊。
  - 如果函式沒有回傳值 (`returns` 為 None)，請省略 `Returns:` 區塊。
  - 請將生成的 docstring 整合回原始函式定義中，並以完整的程式碼區塊形式呈現給使用者。

<!-- end list -->

```

#### **🚀 如何使用**
*您需要先在本地創建一個 Python 檔案，例如 `/Users/user/dev/my_app.py`*

在對話框中輸入：
> 請使用 `python_docstring_generator` 技能，幫我為這個檔案 `/Users/user/dev/my_app.py` 裡的所有函式加上註解。

---

### **範例三：每日技術新聞簡報 (呼叫外部資源)**

這個 Skill 透過 Python 腳本抓取 RSS feed，實現了與外部網路資源的互動，讓 Claude 能提供即時資訊。

#### **🎯 用途**
從指定的技術新聞 RSS 來源抓取最新文章，並整理成一份簡潔的每日簡報。

#### **⚠️ 事前準備**
這個腳本需要安裝 `feedparser` 函式庫。請在您的終端機中執行：
`pip install feedparser`

#### **📁 檔案結構**
```

tech\_news\_briefing.zip
└── tech\_news\_briefing/
├── skill.md
└── scripts/
└── fetch\_news.py

````

#### **🐍 `scripts/fetch_news.py` 內容**
```python
# scripts/fetch_news.py
import feedparser
import json

# 定義要抓取的技術新聞 RSS Feed 來源
RSS_FEEDS = {
    "Hacker News": "https://news.ycombinator.com/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml"
}

def get_latest_news(limit=3):
    """
    從多個來源獲取最新的技術新聞。
    """
    all_news = {}
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        # 取最新的 limit 篇文章
        articles = [{"title": entry.title, "link": entry.link} for entry in feed.entries[:limit]]
        all_news[source] = articles
    
    # 以 JSON 格式輸出
    print(json.dumps(all_news, indent=2))

if __name__ == "__main__":
    get_latest_news()
````

#### **📄 `skill.md` 內容**

````markdown
---
name: tech_news_briefing
version: 1.0
description: 從指定的技術新聞來源 (RSS) 獲取最新文章，並生成一份簡潔的每日簡報。
---
你是一位資深的科技新聞編輯。你的任務是為使用者準備一份今日的技術新聞簡報。

**執行步驟：**
1.  首先，執行 `scripts/fetch_news.py` 腳本來獲取最新的新聞列表。命令如下：
    ```bash
    python scripts/fetch_news.py
    ```
2.  腳本會輸出一份包含新聞來源、標題和連結的 JSON 資料。
3.  請將這份 JSON 資料整理成一份易於閱讀的 Markdown 格式簡報。

**簡報格式要求：**
- 以一個友善的問候開頭，例如「早安！這是您今天的技術新聞簡報」。
- 按照新聞來源（例如 "Hacker News", "TechCrunch"）進行分組。
- 每個來源下列出其最新的文章標題，並附上原始連結。
- 在簡報的結尾，加上一句總結或祝福的話。

**輸出範例：**
> ### ☕ 早安！這是您今天的技術新聞簡報
>
> **來自 Hacker News:**
> * [文章標題一](文章連結一)
> * [文章標題二](文章連結二)
>
> **來自 TechCrunch:**
> * [文章標題三](文章連結三)
> * [文章標題四](文章連結四)
>
> 希望這些資訊對您有幫助，祝您有美好的一天！
````

#### **🚀 如何使用**

在對話框中輸入：

> 早安，請用 `tech_news_briefing` 技能為我準備今天的技術新聞簡報。
