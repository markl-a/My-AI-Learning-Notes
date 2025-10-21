# Day02,03 從頭熟練 Claude Code

> 完整的 Claude Code 學習指南 - 從基礎到進階

## 📚 課程概覽

本課程涵蓋 Claude Code 的 11 個核心主題，從基礎配置到進階整合，幫助您全面掌握 Claude Code 的使用。每個主題都包含詳細的教學文件、實際範例和最佳實踐。

---

## 📖 課程大綱

---

## 1. 輸出樣式 (Output Styles)

### 📋 學習摘要

**學習目標：** 掌握如何自訂 Claude Code 的行為模式和輸出風格

**核心內容：**

- Output Styles 的定義與用途
- 三種內建樣式（Default、Explanatory、Learning）
- 自訂 Output Styles 的建立方法
- 5+ 個實用自訂樣式範例
- 進階配置技巧

**關鍵技能：**

- ✅ 根據不同任務選擇合適的輸出樣式
- ✅ 建立團隊共享的自訂樣式
- ✅ 使用 Markdown + YAML frontmatter 定義樣式

**適合對象：** 所有 Claude Code 使用者

**預計學習時間：** 1-2 小時

**詳細教學：** [→ 查看完整教學內容](#output-styles-詳細內容)

---

### 📖 完整教學內容

### 什麼是 Output Styles

Output Styles（輸出樣式）是 Claude Code 的一項強大功能，讓您能夠自訂 Claude 的行為模式、輸出格式和回應風格。透過定義不同的輸出樣式，您可以讓 Claude 根據不同的工作情境，以最適合的方式回應您的需求。

#### 核心概念

Output Styles 的本質是一組預先定義的指令和行為準則，告訴 Claude：

- **如何組織回應內容**：輸出的結構和格式
- **回應的詳細程度**：簡潔或詳盡的說明
- **互動方式**：是否需要更多解釋、是否主動提問
- **專注領域**：特定任務類型的最佳實踐

#### 為什麼需要 Output Styles？

1. **提高工作效率**：針對不同任務快速切換最佳工作模式
2. **保持一致性**：在團隊中建立統一的工作標準
3. **專業化輸出**：讓 Claude 在特定領域表現得更專業
4. **減少重複指令**：避免每次都要重新說明期望的輸出格式
5. **優化協作體驗**：根據個人或團隊習慣調整 Claude 的行為

#### 運作原理

當您啟用某個 Output Style 時：

1. Claude 會載入該樣式的配置文件
2. 將樣式中定義的指令注入到系統提示中
3. 在整個對話過程中遵循這些指令
4. 您可以隨時切換到其他樣式

---

### 內建的三種 Output Styles

Claude Code 預設提供三種內建樣式，涵蓋了最常見的使用場景。

#### 1. Default（預設樣式）

**特點：**

- 平衡的輸出詳細程度
- 標準的程式碼格式
- 適度的解釋說明
- 通用型的回應方式

**適用場景：**

- 日常開發工作
- 一般性的程式碼問題
- 不確定使用哪種樣式時

**行為特徵：**

```
- 提供清晰但不過度詳細的說明
- 程式碼範例包含必要的註解
- 會在需要時主動詢問細節
- 輸出格式清晰易讀
```

#### 2. Explanatory（解釋型樣式）

**特點：**

- 詳細的解釋說明
- 步驟式的教學方法
- 包含原理和最佳實踐
- 適合學習和理解複雜概念

**適用場景：**

- 學習新技術或框架
- 理解複雜的程式碼邏輯
- 需要詳細文件說明的專案
- 技術分享和教學

**行為特徵：**

```
- 會解釋「為什麼」而不只是「怎麼做」
- 提供多個範例和使用案例
- 包含潛在的陷阱和注意事項
- 使用類比和比喻幫助理解
- 程式碼註解非常詳細
```

**範例輸出差異：**

使用 Default 樣式：

```python
def calculate_total(items):
    return sum(item.price for item in items)
```

使用 Explanatory 樣式：

```python
def calculate_total(items):
    """
    計算購物車中所有商品的總價

    Args:
        items: 商品列表，每個商品物件都必須有 price 屬性

    Returns:
        float: 所有商品價格的總和

    原理：
    使用 Python 的生成器表達式 (generator expression) 遍歷所有商品，
    提取每個商品的 price 屬性，然後使用內建的 sum() 函數計算總和。
    這種方式比使用 for 迴圈更簡潔且效能更好，因為它是惰性求值的。

    注意事項：
    - 確保所有商品都有 price 屬性，否則會引發 AttributeError
    - price 必須是可相加的數值類型（int, float, Decimal）
    - 如果 items 是空列表，會返回 0
    """
    return sum(item.price for item in items)
```

#### 3. Learning（學習型樣式）

**特點：**

- 互動式的教學方法
- 引導式的問題解決
- 鼓勵獨立思考
- 提供提示而非直接答案

**適用場景：**

- 學習程式設計基礎
- 準備技術面試
- 培養問題解決能力
- 教育訓練環境

**行為特徵：**

```
- 會反問您的想法和理解
- 提供提示而非完整解答
- 鼓勵您先嘗試自己解決
- 在您嘗試後給予建設性回饋
- 逐步引導到正確答案
```

**互動範例：**

使用 Default 樣式：

```
問：如何反轉一個字串？
答：使用切片語法 [::-1]，例如：reversed_str = "hello"[::-1]
```

使用 Learning 樣式：

```
問：如何反轉一個字串？
答：這是個很好的問題！在回答之前，我想先了解您的想法：

1. 您知道 Python 中有哪些方式可以遍歷字串嗎？
2. 您聽過「切片（slicing）」這個概念嗎？
3. 您想過可以如何逐個取出字元然後重新組合嗎？

試著想想看，然後告訴我您的想法，我會根據您的回答給予指引！

提示：Python 的字串切片支援步長（step）參數...
```

---

### 如何使用 Output Styles

#### 啟用內建樣式

在對話中，您可以使用以下指令切換樣式：

```bash
# 切換到解釋型樣式
/style explanatory

# 切換到學習型樣式
/style learning

# 切換回預設樣式
/style default

# 查看當前使用的樣式
/style
```

#### 在命令列中指定樣式

使用 Headless Mode 時，可以透過參數指定樣式：

```bash
# 使用解釋型樣式執行任務
claude -p "解釋這段程式碼的工作原理" --style explanatory

# 使用學習型樣式
claude -p "教我如何實作二分搜尋" --style learning
```

---

### 自訂 Output Styles

自訂 Output Styles 是 Claude Code 最強大的功能之一。您可以根據特定需求創建專屬的輸出樣式。

#### 基本格式

Output Styles 使用 Markdown 格式配合 YAML frontmatter 定義：

```markdown
---
name: 樣式名稱
description: 樣式的簡短描述
---

# 樣式指令內容

在這裡描述 Claude 應該如何行為...
```

#### 檔案位置

自訂樣式需要放在以下目錄之一：

1. **專案級別**：`.claude/styles/` （僅適用於當前專案）
2. **全域級別**：`~/.config/claude/styles/` （適用於所有專案）
3. **團隊共享**：可以透過 Git 共享專案級別的樣式

#### 命名規則

- 檔案名稱使用小寫字母和連字號：`code-review.md`
- 樣式名稱在 frontmatter 中定義
- 使用樣式時使用檔案名（不含 .md）：`/style code-review`

#### 基本結構範例

```markdown
---
name: Code Review
description: 專注於程式碼審查的輸出樣式
---

# Code Review Output Style

當使用這個樣式時，請遵循以下指引：

## 審查重點

1. **程式碼品質**
   - 檢查命名是否清晰
   - 驗證是否遵循最佳實踐
   - 評估程式碼可讀性

2. **潛在問題**
   - 識別可能的 bug
   - 檢查邊界條件處理
   - 評估效能影響

3. **改進建議**
   - 提供具體的改進方案
   - 說明為什麼這樣改進
   - 提供重構範例

## 輸出格式

使用以下結構組織回應：

### ✅ 做得好的地方
[列出優點]

### ⚠️ 需要注意的問題
[列出問題及嚴重程度]

### 💡 改進建議
[提供具體建議和程式碼範例]

### 📊 總體評分
[給予評分和總結]
```

---

### 實用的自訂樣式範例

以下提供 8 個涵蓋不同使用場景的完整範例，您可以直接使用或根據需求修改。

#### 範例 1：Code Review（程式碼審查）

**使用場景：** 審查程式碼品質、發現潛在問題、提供改進建議

**檔案路徑：** `.claude/styles/code-review.md`

**完整內容：**

```markdown
---
name: Code Review
description: 進行全面的程式碼審查，提供建設性的改進建議
---

# Code Review Output Style

您現在是一位經驗豐富的程式碼審查專家。在審查程式碼時，請遵循以下準則：

## 審查原則

1. **建設性批評**：指出問題時總是提供改進方案
2. **優先級排序**：區分關鍵問題和可選優化
3. **最佳實踐**：參考業界標準和設計模式
4. **教育性**：解釋為什麼某些做法更好

## 審查檢查清單

### 程式碼品質
- [ ] 命名是否清晰且符合規範
- [ ] 函數是否單一職責
- [ ] 程式碼是否 DRY（不重複）
- [ ] 是否有適當的註解
- [ ] 複雜邏輯是否需要重構

### 功能正確性
- [ ] 邏輯是否正確
- [ ] 是否處理所有邊界條件
- [ ] 錯誤處理是否完整
- [ ] 是否有潛在的 null/undefined 問題

### 效能與安全
- [ ] 是否有效能瓶頸
- [ ] 是否存在記憶體洩漏風險
- [ ] 是否有安全漏洞
- [ ] 資料驗證是否充分

### 可維護性
- [ ] 程式碼是否易於理解
- [ ] 是否容易測試
- [ ] 是否有足夠的錯誤訊息
- [ ] 是否符合專案架構

## 輸出格式

請使用以下結構組織審查結果：

### ✅ 優點
列出程式碼中做得好的地方（至少 2-3 點）

### 🔴 關鍵問題
必須修正的問題，可能導致 bug 或嚴重問題
- 問題描述
- 為什麼這是問題
- 建議的修正方案（包含程式碼）

### 🟡 建議改進
可以改進但非必要的地方
- 改進點
- 改進的好處
- 重構範例

### 🔵 可選優化
錦上添花的優化建議
- 優化方向
- 預期效益

### 📝 總結
- 整體程式碼品質評分（1-10）
- 主要建議（1-2 句話）
- 是否建議合併（Approve / Request Changes / Comment）
```

**實際應用示例：**

```bash
# 在對話中切換到程式碼審查模式
/style code-review

# 然後提供要審查的程式碼
"請審查這個 API 端點的實作"
```

---

#### 範例 2：Documentation Writer（文件撰寫）

**使用場景：** 撰寫技術文件、API 說明、使用手冊

**檔案路徑：** `.claude/styles/documentation-writer.md`

**配置重點：**

- 結構化的文件格式
- 包含範例和使用案例
- 清晰的 API 參考模板
- 適當的視覺元素使用

**實際應用示例：**

```bash
/style documentation-writer
"請為這個 React Hook 撰寫完整的文件，包含使用範例和 API 說明"
```

---

#### 範例 3：Test-Driven Development（測試驅動開發）

**使用場景：** 遵循 TDD 流程開發功能，先寫測試再寫實作

**檔案路徑：** `.claude/styles/tdd.md`

**配置重點：**

- 嚴格遵循紅-綠-重構循環
- 先列出所有測試案例
- 一次只實作一個測試
- 每次變更後都執行測試

**實際應用示例：**

```bash
/style tdd
"我想實作一個購物車功能，包含新增商品、移除商品、計算總價"
```

---

#### 範例 4：Refactoring Coach（重構教練）

**使用場景：** 改善現有程式碼品質，進行安全的重構

**檔案路徑：** `.claude/styles/refactoring-coach.md`

**配置重點：**

- 識別程式碼異味（Code Smells）
- 提供系統化的重構步驟
- 使用已知的重構技術
- 確保測試覆蓋

**實際應用示例：**

```bash
/style refactoring-coach
"這個檔案有 500 行，包含很多重複的程式碼，請幫我重構"
```

---

#### 範例 5：Debugging Expert（除錯專家）

**使用場景：** 系統化地診斷和解決程式問題

**檔案路徑：** `.claude/styles/debugging-expert.md`

**配置重點：**

- 使用科學方法（假設-驗證-結論）
- 建立最小重現案例
- 逐步縮小問題範圍
- 記錄除錯過程

**實際應用示例：**

```bash
/style debugging-expert
"我的 React 應用偶爾會出現 'Cannot read property of undefined' 錯誤，但不是每次都發生"
```

---

#### 範例 6：Performance Optimizer（效能優化）

**使用場景：** 分析和改善程式效能

**檔案路徑：** `.claude/styles/performance-optimizer.md`

**配置重點：**

- 測量優先（先有基準數據）
- 識別效能瓶頸
- 提供優化前後對比
- 評估優化的投資報酬率

**實際應用示例：**

```bash
/style performance-optimizer
"我的 React 應用首次載入很慢，請幫我分析並優化"
```

---

#### 範例 7：API Design Consultant（API 設計顧問）

**使用場景：** 設計 RESTful API、GraphQL Schema 等

**檔案路徑：** `.claude/styles/api-design.md`

**配置重點：**

- 遵循 REST 最佳實踐
- 一致的命名和結構
- 完整的錯誤處理
- 提供 API 文件範本

**實際應用示例：**

```bash
/style api-design
"我需要為電商平台設計 RESTful API，包含商品、訂單、使用者管理"
```

---

#### 範例 8：Security Auditor（安全稽核）

**使用場景：** 檢查程式碼安全性，識別漏洞

**檔案路徑：** `.claude/styles/security-auditor.md`

**配置重點：**

- 檢查 OWASP Top 10 漏洞
- 提供 CVSS 評分
- 包含概念驗證（PoC）
- 給予修復優先級

**實際應用示例：**

```bash
/style security-auditor
"請審查這個 Express API 的安全性，找出潛在的漏洞"
```

---

### 最佳實踐

#### 如何選擇合適的輸出樣式

**根據任務類型選擇：**

| 任務類型     | 推薦樣式              | 原因             |
| ------------ | --------------------- | ---------------- |
| 程式碼審查   | code-review           | 系統化檢查品質   |
| 撰寫文件     | documentation-writer  | 結構化文件格式   |
| 實作新功能   | tdd                   | 先寫測試保證品質 |
| 改善舊程式碼 | refactoring-coach     | 安全的重構步驟   |
| 修復 Bug     | debugging-expert      | 科學化除錯流程   |
| 效能問題     | performance-optimizer | 資料驅動的優化   |
| API 設計     | api-design            | 符合最佳實踐     |
| 安全檢查     | security-auditor      | 識別安全漏洞     |
| 學習新技術   | explanatory/learning  | 詳細解釋概念     |

**根據經驗等級選擇：**

- **初學者**：使用 `learning` 或 `explanatory`，獲得更多指導
- **中級開發者**：使用專門樣式（`tdd`、`code-review`），提升特定技能
- **高級開發者**：使用 `default` 或客製化樣式，保持高效率

#### 團隊協作中的樣式管理

**1. 建立團隊標準樣式**

在專案中建立 `.claude/styles/` 目錄：

```bash
project/
├── .claude/
│   └── styles/
│       ├── code-review.md      # 團隊程式碼審查標準
│       ├── api-design.md       # API 設計規範
│       └── documentation.md    # 文件撰寫風格
├── src/
└── README.md
```

**2. 版本控制**

將樣式文件加入 Git：

```bash
git add .claude/styles/
git commit -m "Add team Output Styles"
```

**3. 文件化樣式使用**

在團隊文件中說明何時使用哪個樣式。

**4. 定期檢視和更新**

- 每季檢視樣式是否符合團隊需求
- 根據回饋調整樣式內容
- 保持樣式的簡潔和實用性

#### 樣式的版本控制建議

**1. 語義化版本**

在樣式中加入版本資訊：

```markdown
---
name: Code Review
description: 程式碼審查樣式
version: 2.1.0
last_updated: 2024-01-15
---
```

**2. 變更日誌**

在樣式文件末尾加入變更記錄。

**3. 向後兼容**

避免破壞性變更，如需改變應提前通知團隊。

#### 效能考量

**1. 樣式大小**

- 保持樣式文件在 5,000 tokens 以下
- 過大的樣式會消耗更多 context window

**2. 載入時機**

```bash
# 在對話開始時就設定樣式
claude --style code-review
```

**3. 樣式複用**
避免在不同樣式中重複相同的內容。

---

### 常見問題與疑難排解

#### Q1: Output Style 沒有生效？

**可能原因：**

1. 樣式文件位置錯誤
2. YAML frontmatter 格式錯誤
3. 樣式名稱不符

**解決方法：**

```bash
# 檢查樣式文件位置
ls -la .claude/styles/

# 檢查 frontmatter 格式
cat .claude/styles/your-style.md

# 使用正確的樣式名稱
/style your-style  # 正確（不含 .md）
```

#### Q2: 如何知道當前使用的是哪個樣式？

**解決方法：**

```bash
# 執行樣式命令不帶參數
/style
```

#### Q3: 可以同時使用多個樣式嗎？

**回答：** 不能同時啟用多個樣式，但可以創建一個組合樣式，將多個樣式的指令合併。

#### Q4: 樣式中的指令太多，Claude 似乎沒有完全遵循？

**解決方法：**

1. 簡化樣式，只保留最重要的指令
2. 拆分成多個專注的小樣式
3. 使用優先級標示

#### Q5: 如何在 Headless Mode 中使用自訂樣式？

**解決方法：**

```bash
# 使用 --style 參數
claude -p "任務描述" --style my-style
```

#### Q6: 樣式在某些情況下表現不如預期？

**解決方法：**

- 使用更具體的指令
- 避免衝突的指令
- 提供具體範例

---

### 總結

Output Styles 是 Claude Code 的強大功能，能夠：

✅ **提高效率**：快速切換到最適合的工作模式
✅ **保持一致性**：團隊使用統一的標準
✅ **專業化輸出**：針對特定任務優化回應
✅ **可客製化**：完全控制 Claude 的行為

**關鍵要點：**

1. **從內建樣式開始**：熟悉 Default、Explanatory、Learning
2. **根據需求選擇**：不同任務使用不同樣式
3. **創建自訂樣式**：為常見工作流程建立專屬樣式
4. **團隊共享**：透過 Git 共享團隊標準
5. **持續優化**：根據使用經驗調整樣式

**下一步行動：**

- [ ] 嘗試使用內建的三種樣式
- [ ] 根據本指南建立第一個自訂樣式
- [ ] 在團隊中推廣使用 Output Styles
- [ ] 建立專案特定的樣式庫
- [ ] 定期檢視和更新樣式

---

**相關資源：**

- [Claude Code 官方文件](https://docs.anthropic.com/claude/docs)
- [Output Styles 最佳實踐](https://docs.anthropic.com/claude/docs/best-practices)

**文件版本：** 1.0.0
**最後更新：** 2024-01-20

---

---

## 2. 無頭模式 (Headless Mode)

### 📋 學習摘要

**學習目標：** 在非互動式環境中使用 Claude Code，實現自動化工作流程

**核心內容：**

- Headless Mode 的基本概念與使用場景
- 命令列參數詳解（-p, --output-format, --allowedTools 等）
- 三種輸出格式（text, json, stream-json）
- CI/CD 整合範例
- 成本控制與效能優化

**關鍵技能：**

- ✅ 使用 `-p` 旗標執行非互動式任務
- ✅ 整合到 GitHub Actions 或 GitLab CI
- ✅ 使用 JSON 格式處理輸出
- ✅ 控制工具權限以確保安全

**應用場景：**

- 持續整合/持續部署 (CI/CD)
- Git pre-commit hooks
- 批次處理任務
- 定期自動化檢查

**預計學習時間：** 2-3 小時

**詳細教學：** [→ 查看完整教學內容](#headless-mode-詳細內容)

### 📖 完整教學內容

### 什麼是 Headless Mode

#### 定義

Headless Mode（無頭模式）是 Claude Code 提供的非互動式執行模式，允許您在不需要人工介入的環境中自動執行任務。這個模式專為自動化工作流程、持續整合系統、定期任務排程等場景設計。

#### 核心特點

- **非互動式執行**：一次性執行完整任務，無需等待用戶輸入
- **可程式化控制**：透過命令列參數精確控制行為
- **多種輸出格式**：支援 text、json、stream-json 等格式，便於後續處理
- **工具權限管理**：可限制 Claude 使用的工具，確保執行安全
- **成本控制**：提供預算限制機制，避免意外的高額費用

#### 為什麼需要 Headless Mode？

1. **自動化整合**：將 AI 能力無縫整合到現有的自動化流程中
2. **持續品質檢查**：在 CI/CD 流程中自動執行程式碼審查和測試
3. **批次處理**：一次性處理大量相似任務
4. **定期維護**：自動執行定期的程式碼維護和優化工作
5. **標準化流程**：確保團隊遵循一致的工作流程和標準

### Headless Mode 的核心概念

#### 基本命令結構

最簡單的 Headless Mode 命令格式：

```bash
claude -p "your prompt here"
```

`-p` 參數代表 "prompt"，指定要執行的任務描述。這個命令會：

1. 啟動 Claude Code
2. 執行指定的任務
3. 輸出結果
4. 自動退出

#### 完整參數說明

**1. `-p, --prompt`**（必需）
指定要執行的任務提示詞。

```bash
# 簡單範例
claude -p "分析 src/ 目錄中的所有 TypeScript 檔案並找出潛在的型別錯誤"

# 複雜範例（使用引號處理多行）
claude -p "請執行以下任務：
1. 檢查所有測試檔案
2. 識別缺少測試的函數
3. 生成測試覆蓋率報告"
```

**2. `--output-format`**（可選）
指定輸出格式，影響結果的呈現方式。

- **`text`**（預設）：人類可讀的文字格式
- **`json`**：結構化的 JSON 格式，適合程式解析
- **`stream-json`**：串流式 JSON，逐步輸出結果

```bash
# 使用 JSON 格式
claude -p "列出所有 TODO 註解" --output-format json

# 使用串流 JSON
claude -p "分析大型專案" --output-format stream-json
```

**3. `--allowedTools`**（可選）
限制 Claude 可以使用的工具，用逗號分隔。這是重要的安全機制。

可用工具包括：

- `Read`：讀取檔案
- `Write`：寫入檔案
- `Edit`：編輯檔案
- `Bash`：執行 shell 命令
- `Grep`：搜尋文字
- `Glob`：檔案模式匹配
- `WebSearch`：網路搜尋
- `WebFetch`：擷取網頁內容

```bash
# 僅允許讀取和搜尋（唯讀模式）
claude -p "分析程式碼結構" --allowedTools Read,Grep,Glob

# 允許讀寫但不允許執行命令
claude -p "重構程式碼" --allowedTools Read,Write,Edit,Grep

# 完全唯讀模式
claude -p "生成文件" --allowedTools Read
```

**4. `--maxCost`**（可選）
設定單次執行的最大成本限制（以美元計）。達到限制時會自動停止。

```bash
# 限制成本在 0.5 美元以內
claude -p "大規模程式碼分析" --maxCost 0.5

# 限制在 0.1 美元（適合測試）
claude -p "測試任務" --maxCost 0.1
```

**5. `--noTools`**（可選）
完全禁用所有工具，Claude 只能基於其內建知識回答問題。

```bash
# 僅使用 Claude 的知識，不讀取檔案或執行命令
claude -p "解釋 React Hooks 的工作原理" --noTools
```

**6. `--workingDir`**（可選）
指定工作目錄，所有檔案操作都相對於此目錄。

```bash
# 在特定目錄中執行
claude -p "分析此專案" --workingDir /path/to/project
```

### 輸出格式詳解

#### 1. Text 格式（預設）

**特點：**

- 人類易讀的純文字格式
- 包含任務執行過程的說明
- 適合直接閱讀和日誌記錄

**範例輸出：**

```
我會分析 src/ 目錄中的 TypeScript 檔案並尋找潛在的型別錯誤。

[分析過程...]

找到以下問題：
1. src/utils/helper.ts:42 - 可能的 null 引用
2. src/components/Button.tsx:18 - 型別不匹配

建議：
- 在存取屬性前檢查 null
- 更新 Button 元件的 props 型別定義
```

**使用場景：**

- 日誌檔案
- 人工審查
- 簡單的自動化腳本

**處理方式：**

```bash
# 儲存到檔案
claude -p "分析程式碼" > analysis.txt

# 透過 email 發送
claude -p "生成報告" | mail -s "Daily Report" team@example.com
```

#### 2. JSON 格式

**特點：**

- 結構化資料格式
- 包含詳細的執行資訊
- 易於程式解析和處理

**輸出結構：**

```json
{
  "success": true,
  "response": "任務執行結果的完整文字內容",
  "toolCalls": [
    {
      "tool": "Read",
      "parameters": {
        "file_path": "/path/to/file.ts"
      },
      "result": "檔案內容..."
    }
  ],
  "cost": {
    "inputTokens": 1500,
    "outputTokens": 800,
    "totalCost": 0.023
  },
  "executionTime": 4.5,
  "model": "claude-sonnet-4-5-20250929"
}
```

**使用場景：**

- 需要解析結果的自動化流程
- 資料收集和分析
- 與其他工具整合

**處理方式：**

```bash
# 使用 jq 解析
claude -p "分析" --output-format json | jq '.response'

# 儲存並處理
claude -p "檢查" --output-format json > result.json
python process_result.py result.json
```

#### 3. Stream JSON 格式

**特點：**

- 逐步輸出結果
- 即時追蹤執行進度
- 適合長時間執行的任務

**輸出格式（多行 JSON 物件）：**

```json
{"type": "start", "timestamp": "2025-10-21T10:00:00Z"}
{"type": "tool_call", "tool": "Read", "file": "src/main.ts"}
{"type": "progress", "message": "正在分析檔案..."}
{"type": "tool_call", "tool": "Grep", "pattern": "TODO"}
{"type": "result", "content": "找到 15 個 TODO 項目"}
{"type": "complete", "success": true, "cost": 0.015}
```

**使用場景：**

- 需要即時回饋的 UI
- 長時間執行的任務監控
- 進度追蹤

**處理方式：**

```bash
# 即時處理每一行
claude -p "大型任務" --output-format stream-json | while read line; do
  echo $line | jq '.type, .message'
done
```

#### 輸出格式比較表

| 特性       | Text       | JSON       | Stream JSON |
| ---------- | ---------- | ---------- | ----------- |
| 可讀性     | ⭐⭐⭐⭐⭐ | ⭐⭐       | ⭐⭐        |
| 可解析性   | ⭐⭐       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐    |
| 即時回饋   | ❌         | ❌         | ✅          |
| 結構化資訊 | ❌         | ✅         | ✅          |
| 檔案大小   | 小         | 中         | 大          |
| 適合自動化 | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐    |

### 實際應用範例

#### 範例 1：GitHub Actions CI/CD 整合

**使用場景：** 在每次 Pull Request 時自動執行程式碼審查

**配置檔案：** `.github/workflows/code-review.yml`

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest

    steps:
      # 1. 檢出程式碼
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 完整的 git 歷史

      # 2. 安裝 Node.js
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      # 3. 安裝 Claude Code
      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code

      # 4. 執行程式碼審查
      - name: Run AI Code Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          claude -p "請審查這次 Pull Request 的變更：

          1. 檢查程式碼品質和最佳實踐
          2. 識別潛在的 bug 或安全問題
          3. 建議改進方向
          4. 確認是否有足夠的測試覆蓋

          請使用 git diff origin/main...HEAD 來查看變更。
          提供詳細的審查報告。" \
          --output-format json \
          --allowedTools Read,Grep,Bash,Glob \
          --maxCost 0.5 > review.json

      # 5. 解析結果
      - name: Parse Review Results
        id: parse
        run: |
          # 提取審查意見
          REVIEW=$(jq -r '.response' review.json)

          # 儲存為輸出變數
          echo "review<<EOF" >> $GITHUB_OUTPUT
          echo "$REVIEW" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

          # 檢查成本
          COST=$(jq -r '.cost.totalCost' review.json)
          echo "cost=$COST" >> $GITHUB_OUTPUT

      # 6. 發布審查意見
      - name: Comment on PR
        uses: actions/github-script@v7
        with:
          script: |
            const review = `${{ steps.parse.outputs.review }}`;
            const cost = `${{ steps.parse.outputs.cost }}`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🤖 AI 程式碼審查報告

${review}

---
*審查成本: $${cost} USD*`
            });

      # 7. 上傳審查報告
      - name: Upload Review Report
        uses: actions/upload-artifact@v4
        with:
          name: code-review-report
          path: review.json
```

**執行結果示例：**

Pull Request 上會出現如下評論：

```
🤖 AI 程式碼審查報告

審查摘要：
本次變更主要涉及使用者認證系統的重構。整體程式碼品質良好，但有幾個需要注意的地方。

主要發現：

1. 安全性問題（重要）
   - src/auth/login.ts:45 - 密碼未進行充分的雜湊處理
   - 建議：使用 bcrypt 或 argon2 進行密碼雜湊

2. 最佳實踐建議
   - src/utils/validator.ts:23 - 建議使用 zod 或 yup 進行輸入驗證
   - src/api/routes.ts:67 - 缺少錯誤處理

3. 測試覆蓋
   - ✅ 新增的函數都有對應測試
   - ⚠️  auth/session.ts 缺少邊界情況測試

4. 效能考量
   - src/database/queries.ts:34 - N+1 查詢問題
   - 建議：使用 JOIN 或 DataLoader

建議：
在合併前請修正安全性問題，其他項目可以作為後續改進。

---
審查成本: $0.12 USD
```

**注意事項：**

1. **API 金鑰安全**：永遠使用 GitHub Secrets 儲存 `ANTHROPIC_API_KEY`
2. **成本控制**：設定適當的 `--maxCost` 限制
3. **權限控制**：僅給予必要的工具權限
4. **失敗處理**：考慮添加 `continue-on-error: true` 避免 CI 因審查失敗而中斷

#### 範例 2：Git Pre-commit Hook

**使用場景：** 在提交前自動檢查程式碼品質

**設定步驟：**

1. **建立 hook 檔案：** `.git/hooks/pre-commit`

```bash
#!/bin/bash

# Git Pre-commit Hook with Claude Code
# 在提交前自動檢查程式碼品質

echo "🔍 正在執行 AI 程式碼檢查..."

# 取得暫存的檔案
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(ts|js|tsx|jsx|py)$')

if [ -z "$STAGED_FILES" ]; then
  echo "✅ 沒有需要檢查的檔案"
  exit 0
fi

# 建立臨時檔案儲存結果
TEMP_RESULT=$(mktemp)

# 執行 Claude Code 檢查
claude -p "請檢查以下暫存的檔案，找出潛在問題：

$(echo "$STAGED_FILES" | while read file; do
  echo "- $file"
done)

檢查重點：
1. 語法錯誤
2. 常見的 bug 模式
3. 安全性問題
4. 程式碼風格問題（嚴重的）
5. 未使用的變數或 import

如果發現嚴重問題，請明確指出。
如果一切正常，請回覆 'OK'。" \
  --output-format json \
  --allowedTools Read,Grep \
  --maxCost 0.1 > "$TEMP_RESULT" 2>&1

# 檢查執行是否成功
if [ $? -ne 0 ]; then
  echo "❌ Claude Code 執行失敗"
  cat "$TEMP_RESULT"
  rm "$TEMP_RESULT"
  exit 1
fi

# 解析結果
RESPONSE=$(jq -r '.response' "$TEMP_RESULT" 2>/dev/null)
SUCCESS=$(jq -r '.success' "$TEMP_RESULT" 2>/dev/null)

if [ "$SUCCESS" != "true" ]; then
  echo "❌ 檢查過程發生錯誤"
  cat "$TEMP_RESULT"
  rm "$TEMP_RESULT"
  exit 1
fi

# 顯示檢查結果
echo ""
echo "檢查結果："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$RESPONSE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 判斷是否有嚴重問題
if echo "$RESPONSE" | grep -qi "嚴重\|critical\|error\|安全"; then
  echo "⚠️  發現需要注意的問題"
  echo ""
  read -p "是否仍要繼續提交？(y/N) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 提交已取消"
    rm "$TEMP_RESULT"
    exit 1
  fi
fi

# 清理
rm "$TEMP_RESULT"

echo "✅ 檢查完成，允許提交"
exit 0
```

2. **設定可執行權限：**

```bash
chmod +x .git/hooks/pre-commit
```

3. **測試 hook：**

```bash
# 修改一個檔案
echo "const x = 123; // 測試" >> src/test.ts

# 暫存檔案
git add src/test.ts

# 嘗試提交（會觸發 hook）
git commit -m "test: 測試 pre-commit hook"
```

**執行結果示例：**

```
🔍 正在執行 AI 程式碼檢查...

檢查結果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
檢查了以下檔案：
- src/test.ts
- src/auth/login.ts

發現的問題：

1. src/auth/login.ts:34
   問題：密碼比較使用了 === 運算子，可能受到時序攻擊
   嚴重性：中等（安全性）
   建議：使用 crypto.timingSafeEqual()

2. src/test.ts:15
   問題：變數 'result' 宣告但未使用
   嚴重性：低

其他檔案沒有發現明顯問題。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  發現需要注意的問題
是否仍要繼續提交？(y/N)
```

**進階配置（整合到專案）：**

為了讓團隊成員都能使用，可以將 hook 放在專案中並自動安裝：

```bash
# 專案結構
.
├── .githooks/
│   └── pre-commit
└── package.json

# package.json 中添加
{
  "scripts": {
    "prepare": "git config core.hooksPath .githooks"
  }
}
```

當團隊成員執行 `npm install` 時，會自動配置 git hooks 路徑。

#### 範例 3：自動化測試生成

**使用場景：** 為缺少測試的函數自動生成單元測試

**腳本檔案：** `scripts/generate-tests.sh`

```bash
#!/bin/bash

# 自動化測試生成腳本
# 用途：掃描專案中缺少測試的函數並生成測試檔案

set -e  # 遇到錯誤時退出

# 配置
SOURCE_DIR="${1:-src}"
TEST_DIR="${2:-tests}"
MAX_COST=1.0

echo "🧪 自動化測試生成工具"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "原始碼目錄: $SOURCE_DIR"
echo "測試目錄: $TEST_DIR"
echo "最大成本: \$$MAX_COST"
echo ""

# 建立測試目錄
mkdir -p "$TEST_DIR"

# 第一步：分析專案，找出缺少測試的函數
echo "📋 步驟 1/3: 分析專案結構..."

ANALYSIS_RESULT=$(claude -p "請分析 $SOURCE_DIR 目錄：

1. 列出所有的公開函數和方法
2. 檢查 $TEST_DIR 目錄中是否有對應的測試
3. 識別缺少測試的函數

請以 JSON 格式輸出結果：
{
  \"totalFunctions\": 數量,
  \"testedFunctions\": 數量,
  \"untestedFunctions\": [
    {
      \"file\": \"檔案路徑\",
      \"function\": \"函數名稱\",
      \"signature\": \"函數簽章\"
    }
  ]
}" \
  --output-format json \
  --allowedTools Read,Grep,Glob \
  --maxCost 0.3)

# 解析分析結果
UNTESTED_COUNT=$(echo "$ANALYSIS_RESULT" | jq '.response | fromjson | .untestedFunctions | length')

echo "✅ 分析完成"
echo "   發現 $UNTESTED_COUNT 個函數缺少測試"
echo ""

if [ "$UNTESTED_COUNT" -eq 0 ]; then
  echo "🎉 所有函數都有測試！"
  exit 0
fi

# 第二步：生成測試檔案
echo "📝 步驟 2/3: 生成測試檔案..."

UNTESTED_FUNCTIONS=$(echo "$ANALYSIS_RESULT" | jq -c '.response | fromjson | .untestedFunctions[]')

GENERATED_COUNT=0

while IFS= read -r func; do
  FILE=$(echo "$func" | jq -r '.file')
  FUNCTION_NAME=$(echo "$func" | jq -r '.function')

  # 計算測試檔案路徑
  REL_PATH=${FILE#$SOURCE_DIR/}
  TEST_FILE="$TEST_DIR/${REL_PATH%.ts}.test.ts"
  TEST_DIR_PATH=$(dirname "$TEST_FILE")

  mkdir -p "$TEST_DIR_PATH"

  echo "   生成 $FUNCTION_NAME 的測試..."

  # 生成測試
  TEST_CONTENT=$(claude -p "請為以下函數生成完整的單元測試：

原始檔案：$FILE
函數名稱：$FUNCTION_NAME

要求：
1. 使用 Jest 測試框架
2. 包含正常情況測試
3. 包含邊界情況測試
4. 包含錯誤處理測試
5. 使用清晰的測試描述
6. 添加必要的 mock

請直接輸出完整的測試程式碼，不要包含解釋。" \
    --output-format json \
    --allowedTools Read \
    --maxCost 0.1)

  # 提取測試程式碼
  TEST_CODE=$(echo "$TEST_CONTENT" | jq -r '.response')

  # 寫入測試檔案
  echo "$TEST_CODE" > "$TEST_FILE"

  GENERATED_COUNT=$((GENERATED_COUNT + 1))

done <<< "$UNTESTED_FUNCTIONS"

echo "✅ 已生成 $GENERATED_COUNT 個測試檔案"
echo ""

# 第三步：執行測試驗證
echo "🧪 步驟 3/3: 執行測試驗證..."

if command -v npm &> /dev/null; then
  npm test 2>&1 | tee test-output.txt

  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 所有測試通過！"
  else
    echo "⚠️  部分測試失敗，請檢查並修正"
    echo "測試輸出已儲存到 test-output.txt"
  fi
else
  echo "⚠️  未找到 npm，跳過測試執行"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ 測試生成完成"
echo "   生成的測試檔案位於: $TEST_DIR"
echo "   總計生成: $GENERATED_COUNT 個測試"
```

**使用方式：**

```bash
# 基本使用
./scripts/generate-tests.sh

# 指定目錄
./scripts/generate-tests.sh src/utils tests/utils

# 設定環境變數
export ANTHROPIC_API_KEY=your-api-key
./scripts/generate-tests.sh
```

**執行結果示例：**

```
🧪 自動化測試生成工具
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原始碼目錄: src
測試目錄: tests
最大成本: $1.0

📋 步驟 1/3: 分析專案結構...
✅ 分析完成
   發現 8 個函數缺少測試

📝 步驟 2/3: 生成測試檔案...
   生成 formatDate 的測試...
   生成 validateEmail 的測試...
   生成 parseJSON 的測試...
   生成 calculateTotal 的測試...
   生成 debounce 的測試...
   生成 throttle 的測試...
   生成 deepClone 的測試...
   生成 capitalize 的測試...
✅ 已生成 8 個測試檔案

🧪 步驟 3/3: 執行測試驗證...
PASS  tests/utils/format.test.ts
PASS  tests/utils/validation.test.ts
PASS  tests/utils/parser.test.ts
PASS  tests/utils/calculator.test.ts
PASS  tests/utils/performance.test.ts
PASS  tests/utils/string.test.ts

Test Suites: 6 passed, 6 total
Tests:       42 passed, 42 total
✅ 所有測試通過！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ 測試生成完成
   生成的測試檔案位於: tests
   總計生成: 8 個測試
```

**生成的測試檔案範例：** `tests/utils/format.test.ts`

```typescript
import { formatDate } from '../../src/utils/format';

describe('formatDate', () => {
  describe('正常情況', () => {
    it('應該正確格式化日期為 YYYY-MM-DD', () => {
      const date = new Date('2025-10-21T10:30:00Z');
      expect(formatDate(date, 'YYYY-MM-DD')).toBe('2025-10-21');
    });

    it('應該正確格式化日期為 DD/MM/YYYY', () => {
      const date = new Date('2025-10-21T10:30:00Z');
      expect(formatDate(date, 'DD/MM/YYYY')).toBe('21/10/2025');
    });
  });

  describe('邊界情況', () => {
    it('應該處理月份的第一天', () => {
      const date = new Date('2025-10-01T00:00:00Z');
      expect(formatDate(date, 'YYYY-MM-DD')).toBe('2025-10-01');
    });

    it('應該處理月份的最後一天', () => {
      const date = new Date('2025-10-31T23:59:59Z');
      expect(formatDate(date, 'YYYY-MM-DD')).toBe('2025-10-31');
    });

    it('應該處理閏年的 2 月 29 日', () => {
      const date = new Date('2024-02-29T12:00:00Z');
      expect(formatDate(date, 'YYYY-MM-DD')).toBe('2024-02-29');
    });
  });

  describe('錯誤處理', () => {
    it('應該在無效日期時拋出錯誤', () => {
      const invalidDate = new Date('invalid');
      expect(() => formatDate(invalidDate, 'YYYY-MM-DD')).toThrow('Invalid date');
    });

    it('應該在不支援的格式時拋出錯誤', () => {
      const date = new Date('2025-10-21');
      expect(() => formatDate(date, 'INVALID')).toThrow('Unsupported format');
    });
  });
});
```

#### 範例 4：程式碼品質檢查（ESLint 風格）

**使用場景：** 自訂程式碼品質規則，超越傳統 linter 的能力

**配置檔案：** `.claude/quality-rules.md`

```markdown
# 程式碼品質規則

## 命名規範
- 變數使用 camelCase
- 常數使用 UPPER_SNAKE_CASE
- 類別使用 PascalCase
- 私有成員使用 _ 前綴
- 布林值變數使用 is/has/should 前綴

## 函數規範
- 單一函數不超過 50 行
- 函數參數不超過 4 個
- 避免巢狀深度超過 3 層
- 每個函數都應有 JSDoc 註解

## React 規範
- 優先使用函數元件
- Hooks 必須在最上層呼叫
- 使用 TypeScript 定義 Props
- 避免在 JSX 中使用匿名函數

## 效能規範
- 避免在迴圈中建立物件
- 使用 useMemo/useCallback 優化
- 避免不必要的 re-render
- 使用 lazy loading 處理大型元件

## 安全性規範
- 永不硬編碼密碼或 API 金鑰
- 驗證所有使用者輸入
- 使用參數化查詢防止 SQL 注入
- 避免使用 eval() 或 Function()
```

**檢查腳本：** `scripts/quality-check.sh`

```bash
#!/bin/bash

# 程式碼品質檢查腳本

set -e

# 配置
TARGET_DIR="${1:-.}"
RULES_FILE=".claude/quality-rules.md"
OUTPUT_FORMAT="${2:-text}"

echo "🔍 執行程式碼品質檢查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 檢查規則檔案是否存在
if [ ! -f "$RULES_FILE" ]; then
  echo "❌ 找不到規則檔案: $RULES_FILE"
  exit 1
fi

# 讀取規則
RULES=$(cat "$RULES_FILE")

# 執行檢查
claude -p "請根據以下品質規則檢查 $TARGET_DIR 目錄中的程式碼：

$RULES

檢查重點：
1. 找出違反規則的程式碼
2. 按嚴重性分類（高/中/低）
3. 提供具體的檔案位置和行號
4. 給出修正建議

請提供完整的檢查報告。" \
  --output-format "$OUTPUT_FORMAT" \
  --allowedTools Read,Grep,Glob \
  --maxCost 0.5

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 檢查完成"
```

**使用方式：**

```bash
# 檢查整個專案
./scripts/quality-check.sh

# 檢查特定目錄
./scripts/quality-check.sh src/components

# 輸出 JSON 格式
./scripts/quality-check.sh src json > quality-report.json
```

**整合到 package.json：**

```json
{
  "scripts": {
    "quality": "./scripts/quality-check.sh",
    "quality:src": "./scripts/quality-check.sh src",
    "quality:json": "./scripts/quality-check.sh . json"
  }
}
```

**執行結果示例：**

```
🔍 執行程式碼品質檢查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

程式碼品質檢查報告
生成時間：2025-10-21 10:30:00

總覽：
- 檢查檔案數：45
- 發現問題數：23
- 高嚴重性：3
- 中嚴重性：12
- 低嚴重性：8

高嚴重性問題：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. src/auth/password.ts:67
   問題：硬編碼的加密金鑰
   規則：安全性規範 - 永不硬編碼密碼或 API 金鑰
   程式碼：
   ```typescript
   const ENCRYPTION_KEY = "my-secret-key-123";
```

   建議：將金鑰移至環境變數
   修正範例：

```typescript
   const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY;
   if (!ENCRYPTION_KEY) throw new Error('Missing encryption key');
```

2. src/database/queries.ts:45
   問題：SQL 字串拼接
   規則：安全性規範 - 使用參數化查詢防止 SQL 注入
   程式碼：

   ```typescript
   const query = `SELECT * FROM users WHERE id = ${userId}`;
   ```

   建議：使用參數化查詢
   修正範例：
   ```typescript
   const query = 'SELECT * FROM users WHERE id = ?';
   db.query(query, [userId]);
   ```

中嚴重性問題：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. src/components/UserList.tsx:34
   問題：函數過長（78 行）
   規則：函數規範 - 單一函數不超過 50 行
   建議：拆分為多個小函數
4. src/utils/processor.ts:23
   問題：巢狀深度過深（5 層）
   規則：函數規範 - 避免巢狀深度超過 3 層
   建議：使用 early return 或提取函數減少巢狀

[... 其他問題 ...]

低嚴重性問題：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[... 列表 ...]

建議：

1. 優先修正高嚴重性的安全性問題
2. 考慮重構過長和複雜的函數
3. 為缺少文件的公開函數添加 JSDoc

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 檢查完成

```

#### 範例 5：批次處理任務 - 多語言文件生成

**使用場景：** 為文件生成多種語言版本

**腳本檔案：** `scripts/translate-docs.sh`

```bash
#!/bin/bash

# 多語言文件生成腳本

set -e

# 配置
SOURCE_LANG="${1:-en}"
TARGET_LANGS="${2:-zh-TW,ja,ko,es,fr}"
DOCS_DIR="docs"
OUTPUT_DIR="docs/i18n"

echo "🌍 多語言文件生成工具"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "原始語言: $SOURCE_LANG"
echo "目標語言: $TARGET_LANGS"
echo ""

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

# 找出所有需要翻譯的文件
DOCS=$(find "$DOCS_DIR" -name "*.md" -not -path "*/i18n/*")
TOTAL_DOCS=$(echo "$DOCS" | wc -l | tr -d ' ')

echo "發現 $TOTAL_DOCS 個文件需要翻譯"
echo ""

# 分割目標語言
IFS=',' read -ra LANGS <<< "$TARGET_LANGS"

# 處理每個文件
DOC_COUNT=0
for DOC in $DOCS; do
  DOC_COUNT=$((DOC_COUNT + 1))
  DOC_NAME=$(basename "$DOC")
  REL_PATH=${DOC#$DOCS_DIR/}

  echo "[$DOC_COUNT/$TOTAL_DOCS] 處理: $DOC_NAME"

  # 為每種語言生成翻譯
  for LANG in "${LANGS[@]}"; do
    echo "  → 翻譯為 $LANG..."

    # 建立輸出目錄
    OUTPUT_PATH="$OUTPUT_DIR/$LANG/$(dirname "$REL_PATH")"
    mkdir -p "$OUTPUT_PATH"

    OUTPUT_FILE="$OUTPUT_DIR/$LANG/$REL_PATH"

    # 執行翻譯
    TRANSLATION=$(claude -p "請將以下 Markdown 文件翻譯為 $LANG：

要求：
1. 保持 Markdown 格式完整
2. 保留程式碼區塊不翻譯
3. 保留 URL 和連結
4. 翻譯要自然流暢，符合目標語言習慣
5. 保留專業術語的準確性

請讀取檔案：$DOC

直接輸出翻譯後的完整內容，不要包含解釋。" \
      --output-format json \
      --allowedTools Read \
      --maxCost 0.2)

    # 提取翻譯內容
    echo "$TRANSLATION" | jq -r '.response' > "$OUTPUT_FILE"

    echo "  ✅ 已儲存到 $OUTPUT_FILE"
  done

  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ 翻譯完成"
echo "   處理檔案: $TOTAL_DOCS"
echo "   目標語言: ${#LANGS[@]}"
echo "   總計生成: $((TOTAL_DOCS * ${#LANGS[@]})) 個翻譯檔案"
echo "   輸出目錄: $OUTPUT_DIR"
```

**進階版本 - 帶進度追蹤和錯誤恢復：**

```bash
#!/bin/bash

# 多語言文件生成腳本（進階版）

set -e

# 配置
SOURCE_LANG="${1:-en}"
TARGET_LANGS="${2:-zh-TW,ja,ko}"
DOCS_DIR="docs"
OUTPUT_DIR="docs/i18n"
PROGRESS_FILE=".translation-progress.json"
MAX_RETRIES=3

echo "🌍 多語言文件生成工具（進階版）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 載入進度（如果存在）
if [ -f "$PROGRESS_FILE" ]; then
  echo "📋 發現之前的進度，將從中斷處繼續..."
  PROGRESS=$(cat "$PROGRESS_FILE")
else
  PROGRESS="{}"
fi

# 找出所有需要翻譯的文件
DOCS=$(find "$DOCS_DIR" -name "*.md" -not -path "*/i18n/*")
TOTAL_DOCS=$(echo "$DOCS" | wc -l | tr -d ' ')

IFS=',' read -ra LANGS <<< "$TARGET_LANGS"
TOTAL_TASKS=$((TOTAL_DOCS * ${#LANGS[@]}))
COMPLETED_TASKS=0
FAILED_TASKS=0

echo "文件數: $TOTAL_DOCS"
echo "目標語言: ${#LANGS[@]}"
echo "總任務數: $TOTAL_TASKS"
echo ""

# 翻譯函數
translate_doc() {
  local doc=$1
  local lang=$2
  local output_file=$3
  local retry_count=0

  while [ $retry_count -lt $MAX_RETRIES ]; do
    if claude -p "翻譯 $doc 為 $lang，保持 Markdown 格式" \
        --output-format json \
        --allowedTools Read \
        --maxCost 0.2 | jq -r '.response' > "$output_file" 2>/dev/null; then
      return 0
    fi

    retry_count=$((retry_count + 1))
    echo "  ⚠️  翻譯失敗，重試 $retry_count/$MAX_RETRIES..."
    sleep 2
  done

  return 1
}

# 處理每個文件
for DOC in $DOCS; do
  REL_PATH=${DOC#$DOCS_DIR/}

  for LANG in "${LANGS[@]}"; do
    TASK_KEY="$REL_PATH:$LANG"

    # 檢查是否已完成
    if echo "$PROGRESS" | jq -e ".[\"$TASK_KEY\"] == true" > /dev/null 2>&1; then
      COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
      continue
    fi

    OUTPUT_PATH="$OUTPUT_DIR/$LANG/$(dirname "$REL_PATH")"
    mkdir -p "$OUTPUT_PATH"
    OUTPUT_FILE="$OUTPUT_DIR/$LANG/$REL_PATH"

    # 顯示進度
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
    PROGRESS_PCT=$((COMPLETED_TASKS * 100 / TOTAL_TASKS))
    echo "[$COMPLETED_TASKS/$TOTAL_TASKS - $PROGRESS_PCT%] $REL_PATH → $LANG"

    # 執行翻譯
    if translate_doc "$DOC" "$LANG" "$OUTPUT_FILE"; then
      echo "  ✅ 成功"
      # 更新進度
      PROGRESS=$(echo "$PROGRESS" | jq ". + {\"$TASK_KEY\": true}")
      echo "$PROGRESS" > "$PROGRESS_FILE"
    else
      echo "  ❌ 失敗（已達最大重試次數）"
      FAILED_TASKS=$((FAILED_TASKS + 1))
      # 記錄失敗
      PROGRESS=$(echo "$PROGRESS" | jq ". + {\"$TASK_KEY\": false}")
      echo "$PROGRESS" > "$PROGRESS_FILE"
    fi
  done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ 批次處理完成"
echo "   成功: $((COMPLETED_TASKS - FAILED_TASKS))"
echo "   失敗: $FAILED_TASKS"
echo "   總計: $COMPLETED_TASKS"

# 清理進度檔案（如果全部成功）
if [ $FAILED_TASKS -eq 0 ]; then
  rm -f "$PROGRESS_FILE"
  echo "   進度檔案已清理"
else
  echo "   ⚠️  有失敗的任務，進度已保存到 $PROGRESS_FILE"
  echo "   可重新執行腳本繼續處理"
fi
```

#### 範例 6：定期報告生成（Cron Job）

**使用場景：** 每週自動生成專案健康度報告

**Cron 配置：**

```bash
# 編輯 crontab
crontab -e

# 添加以下行（每週一早上 9:00 執行）
0 9 * * 1 /path/to/scripts/weekly-report.sh >> /var/log/weekly-report.log 2>&1
```

**報告腳本：** `scripts/weekly-report.sh`

```bash
#!/bin/bash

# 週報生成腳本

set -e

# 配置
PROJECT_DIR="/path/to/your/project"
REPORT_DIR="$PROJECT_DIR/reports"
DATE=$(date +%Y-%m-%d)
REPORT_FILE="$REPORT_DIR/weekly-report-$DATE.md"

# Email 配置
EMAIL_TO="team@example.com"
EMAIL_SUBJECT="Weekly Project Health Report - $DATE"

cd "$PROJECT_DIR"

echo "📊 生成週報 - $DATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 建立報告目錄
mkdir -p "$REPORT_DIR"

# 收集資料
echo "📋 收集專案資料..."

# Git 統計
GIT_STATS=$(git log --since="7 days ago" --pretty=format:"%h - %an, %ar : %s" --shortstat)
COMMIT_COUNT=$(git log --since="7 days ago" --oneline | wc -l | tr -d ' ')
CONTRIBUTORS=$(git log --since="7 days ago" --format='%an' | sort -u | wc -l | tr -d ' ')

# 測試統計
if [ -f "package.json" ]; then
  TEST_OUTPUT=$(npm test 2>&1 || true)
else
  TEST_OUTPUT="No tests configured"
fi

# 生成報告
echo "🤖 使用 AI 生成報告..."

REPORT_CONTENT=$(claude -p "請生成一份詳細的週報，包含以下資訊：

## 專案資訊
- 專案路徑：$PROJECT_DIR
- 報告日期：$DATE

## Git 統計（過去 7 天）
- 提交次數：$COMMIT_COUNT
- 貢獻者數：$CONTRIBUTORS
- 提交記錄：
\`\`\`
$GIT_STATS
\`\`\`

## 測試結果
\`\`\`
$TEST_OUTPUT
\`\`\`

請分析以下內容並生成報告：

1. **活動摘要**
   - 提交頻率分析
   - 主要貢獻者
   - 活躍時間模式

2. **程式碼品質**
   - 檢查最近的提交品質
   - 識別潛在問題
   - 程式碼覆蓋率變化

3. **測試狀況**
   - 測試通過率
   - 新增/失敗的測試
   - 測試建議

4. **技術債務**
   - 掃描 TODO/FIXME 註解
   - 識別需要重構的區域
   - 複雜度分析

5. **安全性檢查**
   - 檢查依賴套件更新
   - 掃描潛在的安全問題
   - 配置檢查

6. **下週建議**
   - 優先處理事項
   - 改進建議
   - 潛在風險

請使用 Markdown 格式，包含適當的標題、列表和程式碼區塊。" \
  --output-format json \
  --allowedTools Read,Grep,Glob,Bash \
  --maxCost 1.0)

# 提取報告內容
echo "$REPORT_CONTENT" | jq -r '.response' > "$REPORT_FILE"

echo "✅ 報告已生成：$REPORT_FILE"

# 轉換為 HTML（用於 email）
if command -v pandoc &> /dev/null; then
  HTML_FILE="${REPORT_FILE%.md}.html"
  pandoc "$REPORT_FILE" -o "$HTML_FILE" \
    --standalone \
    --css=https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css

  echo "✅ HTML 版本已生成：$HTML_FILE"

  # 發送 email
  if command -v mail &> /dev/null; then
    cat "$HTML_FILE" | mail -s "$EMAIL_SUBJECT" -a "Content-Type: text/html" "$EMAIL_TO"
    echo "📧 報告已發送至：$EMAIL_TO"
  fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ 週報生成完成"

# 清理舊報告（保留最近 8 週）
find "$REPORT_DIR" -name "weekly-report-*.md" -mtime +56 -delete
echo "🗑️  已清理 8 週前的舊報告"
```

**生成的報告範例：** `reports/weekly-report-2025-10-21.md`

```markdown
# 週報 - 2025-10-21

## 📊 活動摘要

### 整體統計
- 提交次數：34
- 貢獻者數：5
- 修改檔案數：87
- 新增程式碼：+2,341 行
- 刪除程式碼：-856 行

### 主要貢獻者
1. Alice Chen - 15 commits (44%)
2. Bob Wang - 8 commits (24%)
3. Carol Liu - 6 commits (18%)
4. David Lin - 3 commits (9%)
5. Eve Zhang - 2 commits (6%)

### 活躍時間模式
- 高峰時段：週二、週四下午 2-5 PM
- 主要活動：功能開發 (60%)、Bug 修復 (30%)、重構 (10%)

## 🔍 程式碼品質分析

### 品質指標
- ✅ 平均提交大小：適中（~70 行/提交）
- ✅ 程式碼審查覆蓋率：100%
- ⚠️ 複雜度增加：3 個檔案超過建議閾值

### 主要變更
1. **使用者認證系統重構**
   - 影響檔案：8 個
   - 品質評分：良好
   - 建議：添加更多邊界情況測試

2. **API 端點優化**
   - 影響檔案：12 個
   - 品質評分：優秀
   - 效能提升：~40%

### 潛在問題
- `src/utils/processor.ts` - 函數複雜度過高（建議重構）
- `src/api/legacy.ts` - 包含過時的 API 模式

## 🧪 測試狀況

### 測試統計
- 總測試數：342
- 通過：340 (99.4%)
- 失敗：2 (0.6%)
- 覆蓋率：87% (↑ 3%)

### 失敗測試
1. `tests/integration/payment.test.ts`
   - 原因：外部 API 超時
   - 狀態：已知問題，待修復

2. `tests/unit/validation.test.ts`
   - 原因：測試資料過期
   - 狀態：已修復（未合併）

### 新增測試
- 新增 23 個測試
- 主要涵蓋認證系統和 API 端點

## 💳 技術債務

### TODO/FIXME 統計
- TODO：34 項 (↓ 2)
- FIXME：8 項 (→ 0)
- HACK：3 項 (↑ 1)

### 需要注意的項目
1. **高優先級**
   - `src/database/connection.ts:67` - 連線池設定需要優化
   - `src/api/middleware/auth.ts:34` - Session 管理需要改進

2. **中優先級**
   - 5 個檔案需要添加文件
   - 3 個元件需要效能優化

### 依賴套件
- 總套件數：156
- 需要更新：8 個
- 安全性更新：2 個（重要）

## 🔒 安全性檢查

### 安全性掃描結果
- ✅ 無高危漏洞
- ⚠️ 2 個中危漏洞需要處理

### 需要處理的項目
1. **lodash 4.17.20 → 4.17.21**
   - 嚴重性：中
   - 類型：原型污染
   - 建議：立即更新

2. **axios 0.21.1 → 0.21.2**
   - 嚴重性：中
   - 類型：SSRF
   - 建議：本週內更新

### 配置檢查
- ✅ 無硬編碼密碼
- ✅ 環境變數正確設定
- ⚠️ CORS 設定過於寬鬆（建議收緊）

## 📋 下週建議

### 優先處理事項
1. 🔴 更新安全性漏洞套件
2. 🟡 重構複雜度過高的函數
3. 🟡 修復失敗的整合測試
4. 🟢 完成文件補充

### 改進建議
- 考慮引入 SonarQube 進行持續程式碼品質監控
- 建立自動化效能測試流程
- 為新功能建立更詳細的技術文件

### 潛在風險
- 即將到來的框架升級可能影響現有程式碼
- 團隊成員假期可能影響進度
- 建議提前準備應對方案

---

*本報告由 Claude Code 自動生成*
*生成時間：2025-10-21 09:00:00*
```

#### 範例 7：GitLab CI 整合

**配置檔案：** `.gitlab-ci.yml`

```yaml
stages:
  - analysis
  - test
  - report

variables:
  CLAUDE_MAX_COST: "0.5"

# AI 程式碼分析
ai_analysis:
  stage: analysis
  image: node:20-alpine
  before_script:
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      claude -p "分析此次 MR 的變更，重點檢查：
      1. 程式碼品質
      2. 潛在 bug
      3. 安全性問題
      4. 效能影響

      請使用 git diff origin/main...HEAD" \
        --output-format json \
        --allowedTools Read,Grep,Bash,Glob \
        --maxCost $CLAUDE_MAX_COST > analysis.json

    # 提取結果
    - cat analysis.json | jq -r '.response' > analysis.txt

    # 檢查是否有嚴重問題
    - |
      if grep -qi "嚴重\|critical\|security" analysis.txt; then
        echo "⚠️ 發現需要注意的問題"
        cat analysis.txt
        exit 1
      fi
  artifacts:
    paths:
      - analysis.json
      - analysis.txt
    expire_in: 1 week
  only:
    - merge_requests

# 測試覆蓋率檢查
coverage_check:
  stage: test
  image: node:20-alpine
  before_script:
    - npm install -g @anthropic-ai/claude-code
    - npm install
  script:
    - npm test -- --coverage --json > coverage.json

    # AI 分析覆蓋率
    - |
      claude -p "分析測試覆蓋率報告，識別：
      1. 覆蓋率不足的模組
      2. 缺少測試的關鍵函數
      3. 改進建議

      覆蓋率資料：$(cat coverage.json)" \
        --output-format json \
        --noTools \
        --maxCost 0.1 | jq -r '.response' > coverage-analysis.txt
  artifacts:
    paths:
      - coverage.json
      - coverage-analysis.txt
  only:
    - merge_requests

# 生成綜合報告
generate_report:
  stage: report
  image: node:20-alpine
  dependencies:
    - ai_analysis
    - coverage_check
  script:
    - |
      cat > report.md << 'EOF'
      # MR 審查報告

      ## 程式碼分析
      $(cat analysis.txt)

      ## 測試覆蓋率
      $(cat coverage-analysis.txt)

      ---
      *自動生成於 $(date)*
      EOF

    # 發布到 MR 評論（需要 GitLab API token）
    - |
      if [ -n "$GITLAB_TOKEN" ]; then
        curl -X POST \
          -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
          -H "Content-Type: application/json" \
          -d "{\"body\": \"$(cat report.md | jq -Rs .)\"}" \
          "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
      fi
  artifacts:
    paths:
      - report.md
  only:
    - merge_requests
```

#### 範例 8：Docker 容器中執行

**使用場景：** 在隔離的容器環境中執行 Claude Code

**Dockerfile：**

```dockerfile
FROM node:20-alpine

# 安裝必要工具
RUN apk add --no-cache \
    git \
    bash \
    jq \
    curl

# 安裝 Claude Code
RUN npm install -g @anthropic-ai/claude-code

# 設定工作目錄
WORKDIR /workspace

# 設定入口點
ENTRYPOINT ["claude"]
```

**構建並使用：**

```bash
# 構建映像
docker build -t claude-code:latest .

# 執行任務
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  claude-code:latest \
  -p "分析此專案結構" \
  --allowedTools Read,Grep,Glob

# 使用 docker-compose
```

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  claude-code:
    build: .
    volumes:
      - ./:/workspace
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    command: >
      -p "執行任務"
      --output-format json
      --allowedTools Read,Grep
```

**使用方式：**

```bash
# 執行
docker-compose run --rm claude-code

# 自訂任務
docker-compose run --rm claude-code \
  -p "你的提示詞" \
  --allowedTools Read,Grep,Glob
```

### 安全性與最佳實踐

#### 工具權限控制

**權限層級建議：**

| 場景       | 建議權限                | 說明             |
| ---------- | ----------------------- | ---------------- |
| 程式碼分析 | `Read,Grep,Glob`      | 僅讀取和搜尋     |
| 文件生成   | `Read,Write`          | 讀取和寫入檔案   |
| 程式碼重構 | `Read,Edit,Grep`      | 讀取、編輯和搜尋 |
| CI/CD      | `Read,Grep,Bash,Glob` | 包含命令執行     |
| 測試執行   | `Read,Bash`           | 讀取和執行測試   |
| 完全受限   | `--noTools`           | 僅使用內建知識   |

**最小權限原則：**

```bash
# 不好的做法（過多權限）
claude -p "分析程式碼" # 預設所有權限

# 好的做法（最小權限）
claude -p "分析程式碼" --allowedTools Read,Grep,Glob

# 唯讀任務
claude -p "解釋這段程式碼" --allowedTools Read

# 完全無檔案存取
claude -p "解釋 React Hooks" --noTools
```

**危險工具組合：**

```bash
# 警告：允許寫入和執行命令可能有風險
claude -p "..." --allowedTools Write,Bash  # 謹慎使用

# 更安全的替代方案
claude -p "..." --allowedTools Read,Write  # 無命令執行
```

#### API 金鑰管理

**1. 環境變數（推薦）**

```bash
# 設定環境變數
export ANTHROPIC_API_KEY=your-api-key-here

# 在 .bashrc 或 .zshrc 中
echo 'export ANTHROPIC_API_KEY=your-api-key' >> ~/.bashrc

# 使用 .env 檔案（不要提交到 git）
echo "ANTHROPIC_API_KEY=your-key" > .env
source .env
```

**2. GitHub Secrets（CI/CD）**

```yaml
# GitHub Actions
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

# GitLab CI
variables:
  ANTHROPIC_API_KEY: $ANTHROPIC_API_KEY  # 在 Settings > CI/CD > Variables 設定
```

**3. Docker Secrets**

```bash
# 建立 secret
echo "your-api-key" | docker secret create anthropic_key -

# 使用 secret
docker service create \
  --secret anthropic_key \
  --env ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key \
  your-image
```

**4. 安全性檢查清單**

- [ ] 永遠不要在程式碼中硬編碼 API 金鑰
- [ ] 將 `.env` 加入 `.gitignore`
- [ ] 使用專用的 CI/CD secret 管理系統
- [ ] 定期輪換 API 金鑰
- [ ] 為不同環境使用不同的金鑰
- [ ] 監控 API 使用量和異常活動
- [ ] 限制金鑰的使用範圍（如果平台支援）

#### 成本控制策略

**1. 設定預算限制**

```bash
# 為測試任務設定低預算
claude -p "測試任務" --maxCost 0.05

# 為重要任務設定合理預算
claude -p "完整分析" --maxCost 0.50

# 為批次處理設定總預算
claude -p "批次任務" --maxCost 2.00
```

**2. 成本估算指南**

| 任務類型       | 預估 token 數 | 預估成本      | 建議 maxCost |
| -------------- | ------------- | ------------- | ------------ |
| 簡單程式碼分析 | 5K - 10K      | $0.02 - $0.05 | 0.10         |
| 中等複雜度分析 | 20K - 50K     | $0.10 - $0.25 | 0.50         |
| 大型專案審查   | 100K - 200K   | $0.50 - $1.00 | 1.50         |
| 批次文件處理   | 200K+         | $1.00+        | 依需求設定   |

**3. 成本優化技巧**

```bash
# 技巧 1：限制檔案範圍
claude -p "只分析 src/components/ 目錄" --allowedTools Read,Grep

# 技巧 2：使用精確的提示詞
# 不好
claude -p "看看這個專案"

# 好
claude -p "檢查 src/auth.ts 中的安全性問題"

# 技巧 3：使用唯讀模式節省成本
claude -p "分析但不修改" --allowedTools Read,Grep,Glob

# 技巧 4：批次處理時設定總預算
for file in *.ts; do
  claude -p "分析 $file" --maxCost 0.05
done
```

**4. 成本監控**

```bash
# 使用 JSON 輸出追蹤成本
claude -p "任務" --output-format json | jq '.cost'

# 輸出範例：
# {
#   "inputTokens": 1500,
#   "outputTokens": 800,
#   "totalCost": 0.023
# }

# 累積成本腳本
#!/bin/bash
TOTAL_COST=0

for task in task1 task2 task3; do
  RESULT=$(claude -p "$task" --output-format json)
  COST=$(echo "$RESULT" | jq -r '.cost.totalCost')
  TOTAL_COST=$(echo "$TOTAL_COST + $COST" | bc)
  echo "任務 $task 成本: \$$COST"
done

echo "總成本: \$$TOTAL_COST"
```

#### 錯誤處理建議

**1. 基本錯誤處理**

```bash
#!/bin/bash

# 設定錯誤時退出
set -e

# 執行 Claude Code
if claude -p "任務" --output-format json > result.json 2>&1; then
  echo "成功"
  jq -r '.response' result.json
else
  echo "失敗"
  cat result.json
  exit 1
fi
```

**2. 重試機制**

```bash
#!/bin/bash

MAX_RETRIES=3
RETRY_DELAY=5

retry_claude() {
  local prompt=$1
  local attempt=1

  while [ $attempt -le $MAX_RETRIES ]; do
    echo "嘗試 $attempt/$MAX_RETRIES..."

    if claude -p "$prompt" --output-format json > result.json 2>&1; then
      return 0
    fi

    echo "失敗，等待 $RETRY_DELAY 秒後重試..."
    sleep $RETRY_DELAY
    attempt=$((attempt + 1))
  done

  echo "達到最大重試次數"
  return 1
}

# 使用
if retry_claude "你的任務"; then
  echo "任務完成"
else
  echo "任務失敗"
  exit 1
fi
```

**3. 逾時處理**

```bash
#!/bin/bash

TIMEOUT=300  # 5 分鐘

# 使用 timeout 命令
if timeout $TIMEOUT claude -p "長時間任務" > output.txt; then
  echo "任務完成"
else
  if [ $? -eq 124 ]; then
    echo "任務逾時（超過 $TIMEOUT 秒）"
  else
    echo "任務失敗"
  fi
fi
```

**4. 詳細日誌記錄**

```bash
#!/bin/bash

LOG_FILE="claude-$(date +%Y%m%d-%H%M%S).log"

{
  echo "開始時間: $(date)"
  echo "任務: $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  claude -p "$1" \
    --output-format json \
    --allowedTools Read,Grep \
    2>&1 | tee -a "$LOG_FILE"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "結束時間: $(date)"
} | tee -a "$LOG_FILE"
```

#### 效能優化技巧

**1. 使用精確的提示詞**

```bash
# 不好：模糊的提示詞
claude -p "檢查這個專案"

# 好：明確的提示詞
claude -p "檢查 src/auth.ts 檔案中是否有未處理的 Promise rejection"
```

**2. 限制檔案範圍**

```bash
# 不好：掃描整個專案
claude -p "找出所有 TODO" --allowedTools Grep

# 好：限制範圍
claude -p "在 src/ 目錄中找出所有 TODO" --allowedTools Grep
```

**3. 使用 Stream JSON 處理大型任務**

```bash
# 即時查看進度
claude -p "分析大型專案" \
  --output-format stream-json | \
  while read line; do
    TYPE=$(echo $line | jq -r '.type')
    if [ "$TYPE" = "progress" ]; then
      echo "進度: $(echo $line | jq -r '.message')"
    fi
  done
```

**4. 批次處理優化**

```bash
# 不好：串行處理
for file in *.ts; do
  claude -p "分析 $file"
done

# 好：並行處理（注意成本）
for file in *.ts; do
  (claude -p "分析 $file" > "${file}.analysis.txt") &
done
wait

# 更好：控制並行數量
MAX_JOBS=4
for file in *.ts; do
  while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
    sleep 1
  done
  (claude -p "分析 $file" > "${file}.analysis.txt") &
done
wait
```

**5. 快取和去重**

```bash
#!/bin/bash

# 使用 checksum 避免重複處理
CACHE_DIR=".claude-cache"
mkdir -p "$CACHE_DIR"

process_file() {
  local file=$1
  local checksum=$(md5sum "$file" | cut -d' ' -f1)
  local cache_file="$CACHE_DIR/$checksum.json"

  if [ -f "$cache_file" ]; then
    echo "使用快取結果: $file"
    cat "$cache_file"
  else
    echo "處理: $file"
    claude -p "分析 $file" --output-format json | tee "$cache_file"
  fi
}

# 使用
process_file "src/main.ts"
```

### 進階應用

#### 與其他工具整合

**1. 與 jq 整合（JSON 處理）**

```bash
# 提取特定欄位
claude -p "分析" --output-format json | jq -r '.response'

# 過濾和轉換
claude -p "列出問題" --output-format json | \
  jq '.response | fromjson | .issues[] | select(.severity == "high")'

# 生成報表
claude -p "分析" --output-format json | \
  jq -r '["檔案", "問題", "嚴重性"],
         (.response | fromjson | .issues[] | [.file, .issue, .severity]) |
         @csv'
```

**2. 與 Slack 整合**

```bash
#!/bin/bash

SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# 執行分析
RESULT=$(claude -p "程式碼審查" --output-format json)
RESPONSE=$(echo "$RESULT" | jq -r '.response')
COST=$(echo "$RESULT" | jq -r '.cost.totalCost')

# 發送到 Slack
curl -X POST "$SLACK_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d "{
    \"text\": \"程式碼審查完成\",
    \"blocks\": [
      {
        \"type\": \"section\",
        \"text\": {
          \"type\": \"mrkdwn\",
          \"text\": \"*程式碼審查報告*\n\n$RESPONSE\"
        }
      },
      {
        \"type\": \"context\",
        \"elements\": [
          {
            \"type\": \"mrkdwn\",
            \"text\": \"成本: \$$COST USD\"
          }
        ]
      }
    ]
  }"
```

**3. 與 Notion 整合**

```bash
#!/bin/bash

NOTION_API_KEY="your-notion-api-key"
NOTION_DATABASE_ID="your-database-id"

# 執行分析
ANALYSIS=$(claude -p "週報分析" --output-format json | jq -r '.response')

# 建立 Notion 頁面
curl -X POST https://api.notion.com/v1/pages \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Notion-Version: 2022-06-28" \
  -d "{
    \"parent\": { \"database_id\": \"$NOTION_DATABASE_ID\" },
    \"properties\": {
      \"Title\": {
        \"title\": [
          {
            \"text\": {
              \"content\": \"週報 - $(date +%Y-%m-%d)\"
            }
          }
        ]
      }
    },
    \"children\": [
      {
        \"object\": \"block\",
        \"type\": \"paragraph\",
        \"paragraph\": {
          \"rich_text\": [
            {
              \"type\": \"text\",
              \"text\": {
                \"content\": \"$ANALYSIS\"
              }
            }
          ]
        }
      }
    ]
  }"
```

**4. 與 GitHub API 整合**

```bash
#!/bin/bash

GITHUB_TOKEN="your-github-token"
REPO="owner/repo"
PR_NUMBER=123

# 分析 PR
REVIEW=$(claude -p "審查 PR #$PR_NUMBER" \
  --output-format json \
  --allowedTools Read,Grep,Bash | \
  jq -r '.response')

# 發布審查意見
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/issues/$PR_NUMBER/comments" \
  -d "{\"body\": \"## AI Code Review\n\n$REVIEW\"}"
```

#### 自訂輸出處理

**1. Markdown 轉 HTML**

```bash
#!/bin/bash

# 生成 Markdown 報告
claude -p "生成專案文件" --output-format json | \
  jq -r '.response' | \
  pandoc -f markdown -t html5 \
    --standalone \
    --css=style.css \
    -o output.html
```

**2. 生成 PDF 報告**

```bash
#!/bin/bash

# Markdown → PDF
claude -p "生成報告" --output-format json | \
  jq -r '.response' | \
  pandoc -f markdown -t pdf \
    --pdf-engine=xelatex \
    -V mainfont="Noto Sans CJK TC" \
    -o report.pdf
```

**3. 生成統計圖表**

```bash
#!/bin/bash

# 取得統計資料
STATS=$(claude -p "分析專案統計" --output-format json)

# 使用 gnuplot 生成圖表
echo "$STATS" | jq -r '.data[] | "\(.date) \(.commits)"' | \
  gnuplot -e "
    set terminal png size 800,600;
    set output 'commits.png';
    set title 'Commit Trend';
    set xlabel 'Date';
    set ylabel 'Commits';
    plot '-' using 1:2 with lines;
  "
```

**4. 自訂格式化輸出**

```bash
#!/bin/bash

# 自訂格式化函數
format_output() {
  local json=$1

  echo "╔════════════════════════════════════════╗"
  echo "║         AI 分析報告                    ║"
  echo "╠════════════════════════════════════════╣"

  # 提取資訊
  local response=$(echo "$json" | jq -r '.response')
  local cost=$(echo "$json" | jq -r '.cost.totalCost')
  local time=$(echo "$json" | jq -r '.executionTime')

  echo "║ 執行時間: ${time}s"
  echo "║ 成本: \$${cost}"
  echo "╠════════════════════════════════════════╣"
  echo "$response" | fold -w 38 -s | sed 's/^/║ /'
  echo "╚════════════════════════════════════════╝"
}

# 使用
RESULT=$(claude -p "分析" --output-format json)
format_output "$RESULT"
```

#### 多步驟工作流程設計

**範例：完整的程式碼審查流程**

```bash
#!/bin/bash

# 多步驟程式碼審查工作流程

set -e

PROJECT_DIR="."
REPORT_DIR="reports"
DATE=$(date +%Y%m%d)

mkdir -p "$REPORT_DIR"

echo "🚀 開始多步驟程式碼審查流程"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 步驟 1: 靜態分析
echo ""
echo "📋 步驟 1/5: 靜態程式碼分析..."
STATIC_ANALYSIS=$(claude -p "執行靜態程式碼分析，檢查：
1. 程式碼風格問題
2. 潛在的 bug 模式
3. 複雜度過高的函數
請分析 src/ 目錄" \
  --output-format json \
  --allowedTools Read,Grep,Glob \
  --maxCost 0.3)

echo "$STATIC_ANALYSIS" | jq -r '.response' > "$REPORT_DIR/01-static-analysis.md"
echo "✅ 完成 - 報告已儲存"

# 步驟 2: 安全性掃描
echo ""
echo "🔒 步驟 2/5: 安全性掃描..."
SECURITY_SCAN=$(claude -p "執行安全性掃描，檢查：
1. 硬編碼的敏感資訊
2. SQL 注入風險
3. XSS 漏洞
4. 不安全的依賴
請分析整個專案" \
  --output-format json \
  --allowedTools Read,Grep,Bash \
  --maxCost 0.3)

echo "$SECURITY_SCAN" | jq -r '.response' > "$REPORT_DIR/02-security-scan.md"
echo "✅ 完成 - 報告已儲存"

# 步驟 3: 效能分析
echo ""
echo "⚡ 步驟 3/5: 效能分析..."
PERFORMANCE_ANALYSIS=$(claude -p "分析效能問題：
1. 識別效能瓶頸
2. 檢查不必要的重新渲染
3. 找出記憶體洩漏風險
4. 建議優化方向
重點檢查 src/components/" \
  --output-format json \
  --allowedTools Read,Grep \
  --maxCost 0.3)

echo "$PERFORMANCE_ANALYSIS" | jq -r '.response' > "$REPORT_DIR/03-performance.md"
echo "✅ 完成 - 報告已儲存"

# 步驟 4: 測試分析
echo ""
echo "🧪 步驟 4/5: 測試覆蓋率分析..."
TEST_ANALYSIS=$(claude -p "分析測試狀況：
1. 檢查測試覆蓋率
2. 識別缺少測試的模組
3. 評估測試品質
4. 建議需要補充的測試
請執行測試並分析結果" \
  --output-format json \
  --allowedTools Read,Bash,Grep \
  --maxCost 0.3)

echo "$TEST_ANALYSIS" | jq -r '.response' > "$REPORT_DIR/04-test-analysis.md"
echo "✅ 完成 - 報告已儲存"

# 步驟 5: 綜合報告
echo ""
echo "📊 步驟 5/5: 生成綜合報告..."
FINAL_REPORT=$(claude -p "根據以下分析結果生成綜合報告：

靜態分析：
$(cat "$REPORT_DIR/01-static-analysis.md")

安全性掃描：
$(cat "$REPORT_DIR/02-security-scan.md")

效能分析：
$(cat "$REPORT_DIR/03-performance.md")

測試分析：
$(cat "$REPORT_DIR/04-test-analysis.md")

請生成一份執行摘要，包含：
1. 整體評分（1-10）
2. 關鍵問題列表（按優先級）
3. 改進建議路線圖
4. 預估所需工時" \
  --output-format json \
  --noTools \
  --maxCost 0.2)

echo "$FINAL_REPORT" | jq -r '.response' > "$REPORT_DIR/00-executive-summary.md"
echo "✅ 完成 - 綜合報告已儲存"

# 計算總成本
TOTAL_COST=$(echo "$STATIC_ANALYSIS $SECURITY_SCAN $PERFORMANCE_ANALYSIS $TEST_ANALYSIS $FINAL_REPORT" | \
  jq -s 'map(.cost.totalCost) | add')

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ 程式碼審查流程完成"
echo "   總成本: \$$TOTAL_COST"
echo "   報告位置: $REPORT_DIR/"
echo ""
echo "報告列表："
ls -1 "$REPORT_DIR/"/*.md
```

#### 並行任務處理

**範例：並行處理多個模組**

```bash
#!/bin/bash

# 並行任務處理範例

MAX_PARALLEL=4
MODULES=("auth" "api" "database" "ui" "utils")

echo "🚀 並行分析多個模組"
echo "最大並行數: $MAX_PARALLEL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 分析函數
analyze_module() {
  local module=$1
  echo "開始分析: $module"

  claude -p "詳細分析 src/$module 模組，包含：
  1. 程式碼品質
  2. 潛在問題
  3. 改進建議" \
    --output-format json \
    --allowedTools Read,Grep \
    --maxCost 0.2 > "analysis-$module.json"

  echo "完成分析: $module"
}

# 匯出函數供 parallel 使用
export -f analyze_module

# 方法 1: 使用 GNU Parallel（如果已安裝）
if command -v parallel &> /dev/null; then
  echo "使用 GNU Parallel..."
  printf '%s\n' "${MODULES[@]}" | \
    parallel -j $MAX_PARALLEL analyze_module
else
  # 方法 2: 使用 bash 背景工作
  echo "使用 Bash 背景工作..."

  for module in "${MODULES[@]}"; do
    # 等待直到背景工作數量少於限制
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
      sleep 1
    done

    # 啟動背景工作
    analyze_module "$module" &
  done

  # 等待所有背景工作完成
  wait
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 所有模組分析完成"

# 合併結果
echo ""
echo "合併分析結果..."
jq -s '{modules: .}' analysis-*.json > final-analysis.json
echo "✅ 結果已合併至 final-analysis.json"

# 清理個別檔案
rm -f analysis-*.json
```

### 疑難排解

#### 常見問題與解決方案

**問題 1: API 金鑰無效或未設定**

```
錯誤訊息: "API key not found" 或 "Invalid API key"
```

**解決方案:**

```bash
# 檢查環境變數
echo $ANTHROPIC_API_KEY

# 設定環境變數
export ANTHROPIC_API_KEY=your-api-key-here

# 永久設定（加入 ~/.bashrc 或 ~/.zshrc）
echo 'export ANTHROPIC_API_KEY=your-key' >> ~/.bashrc
source ~/.bashrc

# 驗證設定
claude -p "test" --noTools
```

**問題 2: JSON 解析錯誤**

```
錯誤訊息: "parse error: Invalid numeric literal"
```

**解決方案:**

```bash
# 問題：JSON 輸出被截斷或損壞
claude -p "task" --output-format json > result.json

# 檢查 JSON 是否有效
jq . result.json

# 如果無效，檢查錯誤輸出
claude -p "task" --output-format json 2>error.log 1>result.json

# 使用 try-catch 處理
if jq . result.json > /dev/null 2>&1; then
  echo "JSON 有效"
else
  echo "JSON 無效，檢查 error.log"
  cat error.log
fi
```

**問題 3: 成本超出限制**

```
錯誤訊息: "Cost limit exceeded"
```

**解決方案:**

```bash
# 問題：任務超過 maxCost 限制
claude -p "large task" --maxCost 0.1  # 太低

# 解決方案 1: 增加預算
claude -p "large task" --maxCost 0.5

# 解決方案 2: 拆分任務
# 不好
claude -p "分析整個專案" --maxCost 0.1

# 好
claude -p "分析 src/auth/" --maxCost 0.1
claude -p "分析 src/api/" --maxCost 0.1

# 解決方案 3: 限制範圍
claude -p "只檢查 main.ts 檔案" --maxCost 0.1
```

**問題 4: 工具權限被拒絕**

```
錯誤訊息: "Tool 'Write' is not allowed"
```

**解決方案:**

```bash
# 問題：任務需要的工具未被允許
claude -p "修改檔案" --allowedTools Read  # 缺少 Write 權限

# 解決方案：添加必要權限
claude -p "修改檔案" --allowedTools Read,Write,Edit

# 檢查任務實際需要哪些工具
claude -p "列出需要的工具" --output-format json | \
  jq '.toolCalls[].tool' | sort -u
```

**問題 5: 輸出被截斷**

```
問題：輸出內容不完整
```

**解決方案:**

```bash
# 使用 stream-json 查看完整輸出
claude -p "長任務" --output-format stream-json

# 儲存到檔案
claude -p "長任務" --output-format text > full-output.txt

# 檢查是否達到 token 限制
claude -p "任務" --output-format json | \
  jq '.outputTokens, .model.maxTokens'
```

**問題 6: 在 CI/CD 中執行失敗**

```
錯誤訊息: "command not found: claude"
```

**解決方案:**

```yaml
# GitHub Actions
- name: Install Claude Code
  run: npm install -g @anthropic-ai/claude-code

# 確認安裝
- name: Verify Installation
  run: |
    which claude
    claude --version

# GitLab CI
before_script:
  - npm install -g @anthropic-ai/claude-code
  - claude --version
```

#### 除錯技巧

**1. 啟用詳細日誌**

```bash
# 儲存完整輸出
claude -p "task" --output-format json 2>&1 | tee debug.log

# 分離標準輸出和錯誤輸出
claude -p "task" 1>stdout.log 2>stderr.log

# 使用 set -x 追蹤腳本執行
#!/bin/bash
set -x  # 啟用除錯模式
claude -p "task"
set +x  # 關閉除錯模式
```

**2. 測試工具權限**

```bash
#!/bin/bash

# 測試每個工具
TOOLS=("Read" "Write" "Edit" "Bash" "Grep" "Glob")

for tool in "${TOOLS[@]}"; do
  echo "測試 $tool..."
  if claude -p "測試 $tool 工具" \
      --allowedTools "$tool" \
      --maxCost 0.01 \
      --output-format json > /dev/null 2>&1; then
    echo "✅ $tool 工作正常"
  else
    echo "❌ $tool 失敗"
  fi
done
```

**3. 驗證輸出格式**

```bash
#!/bin/bash

test_format() {
  local format=$1
  echo "測試 $format 格式..."

  OUTPUT=$(claude -p "test" --output-format "$format" --noTools 2>&1)

  case $format in
    json)
      if echo "$OUTPUT" | jq . > /dev/null 2>&1; then
        echo "✅ JSON 格式有效"
      else
        echo "❌ JSON 格式無效"
        echo "$OUTPUT"
      fi
      ;;
    stream-json)
      if echo "$OUTPUT" | head -1 | jq . > /dev/null 2>&1; then
        echo "✅ Stream JSON 格式有效"
      else
        echo "❌ Stream JSON 格式無效"
      fi
      ;;
    text)
      if [ -n "$OUTPUT" ]; then
        echo "✅ Text 格式有效"
      else
        echo "❌ 無輸出"
      fi
      ;;
  esac
}

test_format "json"
test_format "stream-json"
test_format "text"
```

**4. 成本分析**

```bash
#!/bin/bash

# 分析成本分布
analyze_cost() {
  local result=$1

  echo "成本分析："
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  INPUT_TOKENS=$(echo "$result" | jq -r '.cost.inputTokens')
  OUTPUT_TOKENS=$(echo "$result" | jq -r '.cost.outputTokens')
  TOTAL_COST=$(echo "$result" | jq -r '.cost.totalCost')

  echo "輸入 Tokens: $INPUT_TOKENS"
  echo "輸出 Tokens: $OUTPUT_TOKENS"
  echo "總成本: \$$TOTAL_COST"

  # 計算每個工具的使用次數
  echo ""
  echo "工具使用統計："
  echo "$result" | jq -r '.toolCalls[].tool' | sort | uniq -c
}

# 使用
RESULT=$(claude -p "分析任務" --output-format json)
analyze_cost "$RESULT"
```

#### 效能問題診斷

**1. 執行時間分析**

```bash
#!/bin/bash

# 測量執行時間
time_task() {
  local prompt=$1

  START_TIME=$(date +%s.%N)
  claude -p "$prompt" --output-format json > result.json
  END_TIME=$(date +%s.%N)

  ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
  REPORTED_TIME=$(jq -r '.executionTime' result.json)

  echo "實際執行時間: ${ELAPSED}s"
  echo "報告執行時間: ${REPORTED_TIME}s"
  echo "差異: $(echo "$ELAPSED - $REPORTED_TIME" | bc)s"
}

time_task "你的任務"
```

**2. Token 使用優化**

```bash
#!/bin/bash

# 比較不同提示詞的 token 使用量
compare_prompts() {
  local prompt1=$1
  local prompt2=$2

  echo "比較提示詞效率..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  RESULT1=$(claude -p "$prompt1" --output-format json --noTools)
  TOKENS1=$(echo "$RESULT1" | jq -r '.cost.inputTokens + .cost.outputTokens')
  COST1=$(echo "$RESULT1" | jq -r '.cost.totalCost')

  RESULT2=$(claude -p "$prompt2" --output-format json --noTools)
  TOKENS2=$(echo "$RESULT2" | jq -r '.cost.inputTokens + .cost.outputTokens')
  COST2=$(echo "$RESULT2" | jq -r '.cost.totalCost')

  echo "提示詞 1:"
  echo "  Tokens: $TOKENS1"
  echo "  成本: \$$COST1"
  echo ""
  echo "提示詞 2:"
  echo "  Tokens: $TOKENS2"
  echo "  成本: \$$COST2"
  echo ""
  echo "節省: $(echo "($TOKENS1 - $TOKENS2) / $TOKENS1 * 100" | bc -l | xargs printf "%.2f")%"
}

# 使用
compare_prompts \
  "請詳細分析這個專案的所有檔案並提供完整的報告..." \
  "分析 src/ 目錄中的 TypeScript 檔案，列出潛在問題"
```

**3. 瓶頸識別**

```bash
#!/bin/bash

# 識別效能瓶頸
profile_task() {
  local prompt=$1

  echo "效能分析開始..."

  # 使用 stream-json 監控進度
  claude -p "$prompt" \
    --output-format stream-json \
    --allowedTools Read,Grep,Bash | \
    while read line; do
      TYPE=$(echo "$line" | jq -r '.type // empty')

      case $TYPE in
        tool_call)
          TOOL=$(echo "$line" | jq -r '.tool')
          echo "[$(date +%H:%M:%S)] 工具呼叫: $TOOL"
          ;;
        progress)
          MSG=$(echo "$line" | jq -r '.message')
          echo "[$(date +%H:%M:%S)] 進度: $MSG"
          ;;
        complete)
          DURATION=$(echo "$line" | jq -r '.executionTime')
          echo "[$(date +%H:%M:%S)] 完成，耗時: ${DURATION}s"
          ;;
      esac
    done
}

profile_task "你的任務"
```

### 總結

Headless Mode 是 Claude Code 最強大的功能之一，讓您能夠：

1. **自動化工作流程**：整合到 CI/CD、cron jobs、Git hooks
2. **批次處理**：一次處理大量相似任務
3. **可程式化控制**：精確控制執行行為和成本
4. **靈活整合**：與各種工具和服務無縫整合

**最佳實踐摘要：**

- ✅ 使用最小權限原則（`--allowedTools`）
- ✅ 設定適當的成本限制（`--maxCost`）
- ✅ 安全管理 API 金鑰（環境變數、Secrets）
- ✅ 使用精確的提示詞減少 token 消耗
- ✅ 實施錯誤處理和重試機制
- ✅ 記錄日誌以便除錯
- ✅ 定期監控成本和效能

**進階技巧：**

- 🚀 並行處理提升效率
- 🚀 快取結果避免重複計算
- 🚀 使用 stream-json 處理長時間任務
- 🚀 與其他工具整合擴展功能
- 🚀 建立可重用的腳本和工作流程

透過掌握 Headless Mode，您可以充分發揮 Claude Code 的潛力，建立強大的自動化工作流程，大幅提升開發效率。

---

## 3. Model Context Protocol (MCP)

### 📋 學習摘要

**學習目標：** 使用 MCP 擴展 Claude Code 的能力，連接外部工具和資料源

**核心內容：**

- MCP 的架構與設計理念
- 常見 MCP 伺服器（filesystem, github, postgres 等）
- 如何配置和使用 MCP servers
- 開發自己的 MCP server
- 實際應用範例

**關鍵技能：**

- ✅ 配置 MCP 伺服器連接
- ✅ 使用 MCP 存取外部資料源
- ✅ 整合第三方服務（GitHub, 資料庫等）

**應用場景：**

- 連接專案管理工具
- 存取資料庫進行查詢分析
- 整合 API 服務
- 擴展 Claude Code 的功能範圍

**預計學習時間：** 3-4 小時

**詳細教學：** 由於內容過於豐富，建議分章節學習

### 📖 完整教學內容

### 什麼是 MCP？

Model Context Protocol (MCP) 是 Anthropic 開發的開放協議，讓 AI 應用程式能夠安全地連接到外部資料源和工具。透過 MCP，Claude Code 可以：

- 🔗 連接資料庫（PostgreSQL、SQLite）
- 📁 存取檔案系統
- 🌐 整合第三方服務（GitHub、Google Drive）
- 🛠️ 呼叫自訂工具和 API

**核心架構：**

```
Claude Code (Client) ↔ MCP Server ↔ 外部資源
                      |
                      └─ 提供統一的介面
```

**三個核心元件：**

1. **Resources** - 資料來源（檔案、資料庫紀錄等）
2. **Tools** - 可執行的操作
3. **Prompts** - 預定義的提示模板

---

### MCP 配置步驟

#### 1. 開啟配置檔案

在 Claude Code 中執行：

```
/config
```

或手動編輯 `~/Library/Application Support/Claude/claude_desktop_config.json`（macOS）

**重要說明：配置檔案位置**

Claude Code 和 Claude Desktop 使用不同的配置檔案：

- **Claude Code CLI**（本文主要介紹）：

  - 專案級：`.claude/settings.json`（在專案根目錄）
  - 全域級：`~/.config/claude/settings.json`（macOS/Linux）
- **Claude Desktop 應用程式**：

  - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
  - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
  - Linux: `~/.config/Claude/claude_desktop_config.json`

本文檔主要關注 **Claude Code CLI** 的配置方式。

#### 2. 基本配置結構

```json
{
  "mcpServers": {
    "server-name": {
      "command": "執行命令",
      "args": ["參數1", "參數2"],
      "env": {
        "環境變數": "值"
      }
    }
  }
}
```

#### 3. 啟用 MCP Server

配置完成後：

1. 重啟 Claude Code
2. 使用 `/mcp` 指令查看已連接的 servers
3. 開始使用 MCP 功能

---

### 常見 MCP Servers 速覽

| Server                 | 用途              | 安裝方式                                             |
| ---------------------- | ----------------- | ---------------------------------------------------- |
| **filesystem**   | 本地檔案存取      | `npx -y @modelcontextprotocol/server-filesystem`   |
| **github**       | GitHub 倉庫操作   | `npx -y @modelcontextprotocol/server-github`       |
| **postgres**     | PostgreSQL 資料庫 | `npx -y @modelcontextprotocol/server-postgres`     |
| **sqlite**       | SQLite 資料庫     | `npx -y @modelcontextprotocol/server-sqlite`       |
| **brave-search** | 網頁搜尋          | `npx -y @modelcontextprotocol/server-brave-search` |
| **google-drive** | Google Drive 存取 | `npx -y @modelcontextprotocol/server-gdrive`       |
| **slack**        | Slack 整合        | `npx -y @modelcontextprotocol/server-slack`        |

**完整列表：** https://github.com/modelcontextprotocol/servers

---

### 實際應用範例

#### 範例 1：Filesystem Server - 安全存取專案檔案

**使用情境：** 允許 Claude 讀寫特定目錄的檔案

**配置（settings.json）：**

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects",
        "/Users/username/documents"
      ]
    }
  }
}
```

**使用方式：**

```
請使用 MCP filesystem 列出 /Users/username/projects 的所有 Python 檔案
```

**關鍵要點：**

- ✅ 僅授權指定目錄
- ✅ 避免授權根目錄或系統目錄
- ✅ 可同時指定多個目錄

---

#### 範例 2：GitHub Server - 自動化倉庫管理

**使用情境：** 管理 GitHub issues、PRs、搜尋程式碼

**配置：**

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

**取得 GitHub Token：**

1. 前往 https://github.com/settings/tokens
2. 點擊 "Generate new token (classic)"
3. 選擇權限：`repo`, `read:org`
4. 複製 token 並貼到配置中

**使用範例：**

```
請使用 GitHub MCP 列出 my-repo 中所有開啟的 issues
請在 my-repo 建立一個新的 issue，標題為「修復登入錯誤」
```

**關鍵要點：**

- 🔐 Token 具有完整權限，務必妥善保管
- 📝 可自動化 issue 管理、PR 審查
- 🔍 支援跨倉庫程式碼搜尋

---

#### 範例 3：PostgreSQL Server - 資料庫查詢與分析

**使用情境：** 直接查詢資料庫，分析數據

**配置：**

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/mydb"
      }
    }
  }
}
```

**使用範例：**

```
請使用 PostgreSQL MCP 查詢 users 表格中註冊日期在最近 7 天的用戶數量
請分析 orders 表格，找出銷售額最高的前 10 個產品
```

**關鍵要點：**

- ⚠️ 使用唯讀帳號更安全
- 📊 適合數據分析、報表生成
- 🔒 避免在配置中明文存放密碼（使用環境變數）

**安全配置範例：**

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "${POSTGRES_URL}"
      }
    }
  }
}
```

然後在系統環境變數中設定 `POSTGRES_URL`。

---

#### 範例 4：Brave Search - 網頁搜尋整合

**使用情境：** 讓 Claude 進行即時網頁搜尋

**配置：**

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**取得 API Key：**

1. 前往 https://brave.com/search/api/
2. 註冊並取得 API key（免費方案：每月 2000 次查詢）

**使用範例：**

```
請使用 Brave Search 查詢「TypeScript 5.0 新功能」
搜尋最新的 React 18 效能優化技巧
```

**關鍵要點：**

- 🆓 免費方案適合個人使用
- 🔍 可替代內建搜尋功能
- 📰 適合查詢最新技術文件

---

#### 範例 5：多個 MCP Servers 組合使用

**使用情境：** 同時使用多個 servers 完成複雜任務

**完整配置範例：**

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres-prod": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "${PROD_DB_URL}"
      }
    },
    "postgres-dev": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "${DEV_DB_URL}"
      }
    }
  }
}
```

**實際應用場景：**

```
1. 從 PostgreSQL 匯出用戶數據
2. 使用 filesystem 儲存為 CSV
3. 上傳到 GitHub 倉庫
4. 建立 PR 供團隊審查
```

---

#### 範例 6：自訂 MCP Server（簡易範例）

**使用情境：** 建立專屬的工具整合

**基本 TypeScript Server 範例：**

```typescript
// my-custom-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-custom-server",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});

// 定義工具
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [{
      name: "get_weather",
      description: "取得天氣資訊",
      inputSchema: {
        type: "object",
        properties: {
          city: { type: "string", description: "城市名稱" }
        },
        required: ["city"]
      }
    }]
  };
});

// 執行工具
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "get_weather") {
    const city = request.params.arguments?.city;
    // 實際的 API 呼叫邏輯
    return {
      content: [{
        type: "text",
        text: `${city} 的天氣：晴朗，25°C`
      }]
    };
  }
});

// 啟動 server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main();
```

**配置使用：**

```json
{
  "mcpServers": {
    "my-weather": {
      "command": "node",
      "args": ["/path/to/my-custom-server.js"]
    }
  }
}
```

**開發步驟：**

1. `npm install @modelcontextprotocol/sdk`
2. 建立 TypeScript 檔案
3. 編譯：`tsc my-custom-server.ts`
4. 在 settings.json 配置
5. 重啟 Claude Code

---

### 開發自己的 MCP Server

#### 基本結構

**必要元件：**

1. **Server 初始化** - 設定名稱、版本、能力
2. **Tools 定義** - 列出可用工具
3. **Tools 執行** - 實作工具邏輯
4. **Transport 層** - 通訊協議（通常用 Stdio）

**完整開發流程：**

```bash
# 1. 建立專案
mkdir my-mcp-server
cd my-mcp-server
npm init -y

# 2. 安裝依賴
npm install @modelcontextprotocol/sdk
npm install -D @types/node typescript

# 3. 建立 tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./dist",
    "strict": true
  }
}
```

**進階範例 - API 整合 Server：**

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "api-integration-server",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {},
    resources: {}
  }
});

// 定義多個工具
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "fetch_user",
        description: "從 API 取得用戶資料",
        inputSchema: {
          type: "object",
          properties: {
            userId: { type: "number" }
          },
          required: ["userId"]
        }
      },
      {
        name: "create_user",
        description: "建立新用戶",
        inputSchema: {
          type: "object",
          properties: {
            name: { type: "string" },
            email: { type: "string" }
          },
          required: ["name", "email"]
        }
      }
    ]
  };
});

// 實作工具邏輯
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  try {
    if (name === "fetch_user") {
      const response = await fetch(`https://api.example.com/users/${args.userId}`);
      const data = await response.json();
      return {
        content: [{ type: "text", text: JSON.stringify(data, null, 2) }]
      };
    }

    if (name === "create_user") {
      const response = await fetch("https://api.example.com/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args)
      });
      const data = await response.json();
      return {
        content: [{ type: "text", text: `用戶建立成功：${data.id}` }]
      };
    }
  } catch (error) {
    return {
      content: [{ type: "text", text: `錯誤：${error.message}` }],
      isError: true
    };
  }
});

// 啟動 server
const transport = new StdioServerTransport();
await server.connect(transport);
```

**關鍵概念：**

- **InputSchema** - 定義工具輸入參數（遵循 JSON Schema）
- **錯誤處理** - 使用 try-catch 並回傳 isError
- **非同步操作** - 使用 async/await
- **型別安全** - 善用 TypeScript

---

### 最佳實踐

#### 安全性

- 🔐 **絕不在配置檔中明文存放密碼或 Token**

  - 使用環境變數：`"API_KEY": "${MY_API_KEY}"`
  - 使用 `.env` 檔案（加入 `.gitignore`）
- 🛡️ **最小權限原則**

  - Filesystem：僅授權必要目錄
  - Database：使用唯讀帳號
  - GitHub：限制 Token 權限範圍
- 🚫 **避免危險操作**

  - 不要授權系統目錄（`/`, `/System`）
  - 謹慎處理刪除操作
  - 定期審查 MCP server 權限

#### 效能優化

- ⚡ **連線管理**

  - 重用資料庫連線池
  - 實作快取機制（適用於不常變動的資料）
  - 設定合理的 timeout
- 📦 **資料傳輸**

  - 限制查詢結果數量（使用 LIMIT）
  - 分頁處理大型資料集
  - 壓縮大型回應
- 🔄 **錯誤重試**

  - 實作 exponential backoff
  - 區分可重試與不可重試錯誤

#### 錯誤處理

```typescript
// 良好的錯誤處理範例
server.setRequestHandler("tools/call", async (request) => {
  try {
    // 驗證輸入
    if (!request.params.arguments?.userId) {
      return {
        content: [{ type: "text", text: "錯誤：缺少 userId 參數" }],
        isError: true
      };
    }

    // 執行操作
    const result = await performOperation(request.params.arguments);

    return {
      content: [{ type: "text", text: JSON.stringify(result) }]
    };
  } catch (error) {
    // 記錄錯誤
    console.error("MCP Error:", error);

    // 回傳友善的錯誤訊息
    return {
      content: [{
        type: "text",
        text: `操作失敗：${error instanceof Error ? error.message : "未知錯誤"}`
      }],
      isError: true
    };
  }
});
```

---

### 疑難排解

#### 1. MCP Server 無法啟動

**症狀：** 配置後重啟 Claude Code，server 沒有出現

**解決方法：**

- 檢查 JSON 格式是否正確（使用 JSON validator）
- 確認命令路徑正確：`which npx` 或 `which node`
- 查看 Claude Code 日誌：`~/Library/Logs/Claude/`
- 手動測試命令是否能執行：
  ```bash
  npx -y @modelcontextprotocol/server-filesystem /path/to/dir
  ```

#### 2. 環境變數未生效

**症狀：** `${VARIABLE}` 沒有被替換

**解決方法：**

- 確認環境變數已設定：`echo $VARIABLE`
- 在 shell 配置檔中設定（`~/.zshrc` 或 `~/.bashrc`）：
  ```bash
  export GITHUB_TOKEN="ghp_xxx"
  ```
- 重新啟動終端機和 Claude Code
- 或在配置中直接使用值（不推薦用於敏感資訊）

#### 3. 權限錯誤

**症狀：** "Permission denied" 或 "Access forbidden"

**解決方法：**

- Filesystem：確認目錄有讀寫權限
  ```bash
  chmod 755 /path/to/directory
  ```
- Database：檢查連線字串和用戶權限
- API：驗證 Token 是否有效且具備所需權限

#### 4. 連線逾時

**症狀：** 操作卡住或回傳 timeout 錯誤

**解決方法：**

- 檢查網路連線
- 資料庫：確認 host 和 port 正確
- 增加 timeout 設定（若 server 支援）
- 檢查防火牆設定

#### 5. 工具呼叫失敗

**症狀：** Claude 表示無法使用某個 MCP 工具

**解決方法：**

- 使用 `/mcp` 指令確認 server 已連接
- 檢查工具名稱是否正確
- 查看 server 日誌以了解錯誤詳情
- 重新安裝 MCP server package：
  ```bash
  npm cache clean --force
  npx -y @modelcontextprotocol/server-xxx
  ```

---

### 進階主題

#### MCP Server 發布與分享

**發布到 npm：**

```bash
# 1. 建立 package.json
{
  "name": "@yourname/mcp-server-custom",
  "version": "1.0.0",
  "bin": {
    "mcp-server-custom": "./dist/index.js"
  },
  "files": ["dist"]
}

# 2. 發布
npm publish --access public
```

**使用已發布的 server：**

```json
{
  "mcpServers": {
    "custom": {
      "command": "npx",
      "args": ["-y", "@yourname/mcp-server-custom"]
    }
  }
}
```

#### Resources vs Tools

| 特性     | Resources             | Tools                  |
| -------- | --------------------- | ---------------------- |
| 用途     | 提供資料（唯讀）      | 執行操作（可變更狀態） |
| 範例     | 檔案內容、資料庫記錄  | 建立檔案、發送郵件     |
| 使用時機 | Claude 需要參考資料時 | Claude 需要執行動作時  |

**Resources 範例：**

```typescript
server.setRequestHandler("resources/list", async () => {
  return {
    resources: [{
      uri: "config://settings.json",
      name: "應用程式設定",
      mimeType: "application/json"
    }]
  };
});

server.setRequestHandler("resources/read", async (request) => {
  const config = await loadConfig();
  return {
    contents: [{
      uri: request.params.uri,
      mimeType: "application/json",
      text: JSON.stringify(config, null, 2)
    }]
  };
});
```

#### Prompts 功能

**定義可重用的提示模板：**

```typescript
server.setRequestHandler("prompts/list", async () => {
  return {
    prompts: [{
      name: "code_review",
      description: "審查程式碼並提供建議",
      arguments: [{
        name: "file_path",
        description: "要審查的檔案路徑",
        required: true
      }]
    }]
  };
});

server.setRequestHandler("prompts/get", async (request) => {
  if (request.params.name === "code_review") {
    const filePath = request.params.arguments?.file_path;
    const code = await readFile(filePath);

    return {
      messages: [{
        role: "user",
        content: {
          type: "text",
          text: `請審查以下程式碼並提供改進建議：\n\n${code}`
        }
      }]
    };
  }
});
```

---

### 實用資源

- **官方文件：** https://modelcontextprotocol.io/
- **GitHub 組織：** https://github.com/modelcontextprotocol
- **Server 列表：** https://github.com/modelcontextprotocol/servers
- **SDK 文件：** https://github.com/modelcontextprotocol/typescript-sdk
- **社群討論：** https://github.com/modelcontextprotocol/specification/discussions

---

### 學習檢查清單

完成本章後，你應該能夠：

- [ ] 理解 MCP 的核心概念和架構
- [ ] 配置常見的 MCP servers（filesystem、github、database）
- [ ] 整合第三方服務（Brave Search、Google Drive）
- [ ] 開發簡單的自訂 MCP server
- [ ] 實作適當的安全性和錯誤處理
- [ ] 排解常見的 MCP 問題
- [ ] 組合多個 MCP servers 完成複雜任務

---

### 實作練習

**練習 1：基礎配置**
設定 filesystem MCP server，讓 Claude 能夠存取你的專案目錄。

**練習 2：GitHub 整合**
配置 GitHub MCP server，使用 Claude 列出某個倉庫的所有 open issues。

**練習 3：資料庫查詢**
設定 PostgreSQL 或 SQLite MCP server，讓 Claude 幫你分析資料。

**練習 4：自訂 Server**
建立一個簡單的 MCP server，提供一個自訂工具（例如：單位轉換、日期計算）。

**練習 5：進階整合**
結合多個 MCP servers，實作一個自動化工作流程（例如：從資料庫匯出資料 → 儲存為檔案 → 上傳到 GitHub）。

---

## 4. Agent Skills

### 📋 學習摘要

**學習目標：** 建立可重用的技能模組，讓 Claude 自動判斷何時使用

**核心內容：**

- Agent Skills 的定義與概念
- Skills 的三層式載入架構
- SKILL.md 檔案格式與配置
- 5 個完整的實際應用案例
- 最佳實踐與常見問題

**關鍵技能：**

- ✅ 建立自訂 Skills
- ✅ 組織參考文件
- ✅ 設定工具權限限制
- ✅ 版本控制與團隊共享

**實用 Skills 範例：**

- API 文件產生器
- 測試自動化
- 資料分析與視覺化
- 程式碼重構助手
- 部署自動化

**預計學習時間：** 2-3 小時

---

### 📖 完整教學內容

### 什麼是 Agent Skills

#### 定義

Agent Skills 是一種模組化的功能擴展系統，讓您能夠將指令、元資料和可選資源（如腳本、模板）打包成可重複使用的單元。這些技能讓 Claude 能夠從通用型助手轉變為專精特定領域的工具。

#### 核心特點

- **模組化設計**：每個 Skill 都是獨立的功能單元
- **自動觸發**：Claude 會根據任務內容自動判斷是否需要使用特定 Skill
- **漸進式載入**：採用三層式架構，僅在需要時載入相關內容
- **跨平台支援**：可在 Claude Code、Claude.ai 和 Claude API 中使用

#### 為什麼需要 Agent Skills？

1. **減少重複性指令**：將常用的工作流程封裝起來
2. **專業化能力**：為特定領域任務提供專業知識
3. **提高效率**：自動化複雜的多步驟流程
4. **知識共享**：團隊可以共享最佳實踐和工作流程

### Agent Skills 的核心概念

#### 三層式漸進載入架構

Agent Skills 採用創新的三層載入機制，以優化 token 使用效率：

**第一層：元資料（Metadata）**

- **載入時機**：始終載入
- **內容**：Skill 名稱和描述
- **Token 消耗**：約 100 tokens
- **作用**：讓 Claude 判斷該 Skill 是否與當前任務相關

**第二層：指令（Instructions）**

- **載入時機**：當 Skill 被觸發時
- **內容**：詳細的步驟指引和操作說明
- **Token 限制**：建議在 5,000 tokens 以下
- **作用**：提供完整的工作流程資訊

**第三層：資源（Resources）**

- **載入時機**：根據需要動態載入
- **內容**：腳本、參考文件、程式碼、模板
- **特點**：透過 bash 執行，不占用 context window
- **作用**：執行確定性操作，處理大量資料

---

### 目錄

1. [什麼是 Agent Skills](#什麼是-agent-skills)
2. [Agent Skills 的核心概念](#agent-skills-的核心概念)
3. [Skills 的類型](#skills-的類型)
4. [如何建立 Agent Skills](#如何建立-agent-skills)
5. [配置與使用](#配置與使用)
6. [進階功能](#進階功能)
7. [實際應用案例](#實際應用案例)
8. [最佳實踐](#最佳實踐)
9. [常見問題與疑難排解](#常見問題與疑難排解)

---

### 什麼是 Agent Skills

#### 定義

Agent Skills 是一種模組化的功能擴展系統，讓您能夠將指令、元資料和可選資源（如腳本、模板）打包成可重複使用的單元。這些技能讓 Claude 能夠從通用型助手轉變為專精特定領域的工具。

#### 核心特點

- **模組化設計**：每個 Skill 都是獨立的功能單元
- **自動觸發**：Claude 會根據任務內容自動判斷是否需要使用特定 Skill
- **漸進式載入**：採用三層式架構，僅在需要時載入相關內容
- **跨平台支援**：可在 Claude Code、Claude.ai 和 Claude API 中使用

#### 為什麼需要 Agent Skills？

1. **減少重複性指令**：將常用的工作流程封裝起來
2. **專業化能力**：為特定領域任務提供專業知識
3. **提高效率**：自動化複雜的多步驟流程
4. **知識共享**：團隊可以共享最佳實踐和工作流程

---

### Agent Skills 的核心概念

#### 三層式漸進載入架構

Agent Skills 採用創新的三層載入機制，以優化 token 使用效率：

##### 第一層：元資料（Metadata）

- **載入時機**：始終載入
- **內容**：Skill 名稱和描述
- **Token 消耗**：約 100 tokens
- **作用**：讓 Claude 判斷該 Skill 是否與當前任務相關

##### 第二層：指令（Instructions）

- **載入時機**：當 Skill 被觸發時
- **內容**：詳細的步驟指引和操作說明
- **Token 限制**：建議在 5,000 tokens 以下
- **作用**：提供完整的工作流程資訊

##### 第三層：資源（Resources）

- **載入時機**：根據需要動態載入
- **內容**：腳本、參考文件、程式碼、模板
- **特點**：透過 bash 執行，不占用 context window
- **作用**：執行確定性操作，處理大量資料

#### 檔案系統架構

Skills 基於檔案系統組織，具有以下特點：

- 無實際內容大小限制
- 可包含可執行腳本
- 支援動態內容載入
- 易於版本控制和分享

---

### Skills 的類型

#### 1. 個人 Skills（Personal Skills）

- **儲存位置**：`~/.claude/skills/`
- **作用範圍**：所有專案通用
- **適用情境**：個人工作流程、偏好設定、常用工具

**範例使用場景**：

- 個人化的程式碼風格檢查
- 常用的資料處理流程
- 個人專屬的文件模板

#### 2. 專案 Skills（Project Skills）

- **儲存位置**：`<專案根目錄>/.claude/skills/`
- **作用範圍**：特定專案
- **適用情境**：團隊協作、專案特定工作流程
- **版本控制**：可納入 git 管理，與團隊共享

**範例使用場景**：

- 專案特定的建置流程
- 團隊編碼規範
- 專案文件產生器

#### 3. 插件 Skills（Plugin Skills）

- **來源**：隨插件安裝自動提供
- **管理方式**：由插件統一管理
- **適用情境**：擴充 Claude Code 的內建功能

**範例**：

- 官方文件處理 Skills（PDF、DOCX、PPTX、XLSX）
- 社群開發的專業工具

#### 4. 內建 Skills（Built-in Skills）

Anthropic 官方提供的預建 Skills：

- **PowerPoint (pptx)**：建立和編輯簡報
- **Excel (xlsx)**：處理試算表和資料分析
- **Word (docx)**：產生格式化文件
- **PDF (pdf)**：處理 PDF 文件

---

### 如何建立 Agent Skills

#### 基本目錄結構

```
my-skill/
├── SKILL.md           # 必要：主要技能定義檔案
├── reference.md       # 可選：參考文件
├── scripts/           # 可選：可執行腳本
│   └── helper.py
└── templates/         # 可選：模板檔案
    └── template.txt
```

#### SKILL.md 檔案結構

SKILL.md 是 Agent Skill 的核心檔案，包含兩個主要部分：

##### 1. YAML Frontmatter（前置元資料）

```yaml
---
name: 技能名稱
description: 清楚描述這個 Skill 的功能以及何時應該使用它
version: 1.0.0                    # 可選：版本號
allowed-tools: [Read, Grep, Glob] # 可選：限制可用工具
dependencies: []                   # 可選：依賴套件
---
```

**重要欄位說明**：

- **name**（必要）

  - 長度限制：64 字元
  - 建議使用動名詞形式（如："處理 PDF 文件"）
  - 必須清晰且具描述性
- **description**（必要）

  - 長度限制：1024 字元
  - 必須包含兩個部分：
    1. 這個 Skill 做什麼
    2. 何時應該使用它
  - 使用第三人稱撰寫
  - 要具體且明確
- **allowed-tools**（可選）

  - 限制 Skill 可使用的工具
  - 提高安全性
  - 常用工具：Read、Write、Edit、Bash、Grep、Glob
- **version**（可選）

  - 追蹤 Skill 迭代
  - 建議使用語意化版本

##### 2. Markdown 內容（指令與說明）

```markdown
## 技能名稱

### 概述
提供這個 Skill 的簡要說明

### 使用時機
明確說明什麼情況下應該使用這個 Skill

### 操作步驟
1. 第一步：詳細說明
2. 第二步：具體操作
3. 第三步：驗證結果

### 範例
提供具體的使用範例

### 注意事項
列出需要注意的限制或特殊情況
```

#### 完整範例：PDF 處理 Skill

```markdown
---
name: 處理 PDF 文件
description: 提取 PDF 文字內容、填寫表單、合併 PDF 文件。當需要處理 PDF 檔案、表單填寫或文件提取時使用。
version: 1.0.0
allowed-tools: [Read, Bash]
---

## PDF 處理 Skill

### 概述
這個 Skill 提供 PDF 文件的處理功能，包括文字提取、表單填寫和文件合併。

### 使用時機
- 需要從 PDF 中提取文字內容
- 需要填寫 PDF 表單
- 需要合併多個 PDF 文件
- 需要分析 PDF 文件結構

### 操作步驟

#### 提取文字
1. 使用 pdfplumber 函式庫開啟 PDF
2. 遍歷所有頁面
3. 提取並整理文字內容
4. 儲存為結構化格式

#### 填寫表單
1. 識別 PDF 中的表單欄位
2. 根據提供的資料填寫對應欄位
3. 產生填寫完成的新 PDF

#### 合併文件
1. 讀取所有需要合併的 PDF 檔案
2. 按指定順序合併
3. 產生單一 PDF 輸出

### 範例

#### 範例 1：提取文字
```python
import pdfplumber

with pdfplumber.open('document.pdf') as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
```

#### 範例 2：填寫表單

使用 fillpdf 填寫表單欄位，並產生新的 PDF。

### 注意事項

- 確保已安裝 pdfplumber 套件
- 某些加密的 PDF 可能無法處理
- 表單欄位名稱需要事先確認

```

---

### 配置與使用

#### 安裝與啟用 Skills

##### 方法 1：建立個人 Skill

```bash
## 建立 Skills 目錄
mkdir -p ~/.claude/skills/my-skill

## 建立 SKILL.md 檔案
cat > ~/.claude/skills/my-skill/SKILL.md << 'EOF'
---
name: 我的自訂技能
description: 這是我的第一個 Agent Skill，用於學習和測試
---

## 我的自訂技能

### 指令
這裡放置詳細的操作指引。
EOF
```

##### 方法 2：建立專案 Skill

```bash
## 在專案根目錄建立 Skills
cd /path/to/your/project
mkdir -p .claude/skills/project-skill

## 建立 SKILL.md
nano .claude/skills/project-skill/SKILL.md
```

##### 方法 3：使用官方 Skills

官方 Skills 會隨 Claude Code 自動提供，無需額外安裝。

##### 方法 4：從 GitHub 安裝社群 Skills

```bash
## 從官方範例庫克隆
git clone https://github.com/anthropics/skills.git

## 複製需要的 Skill 到個人目錄
cp -r skills/algorithmic-art ~/.claude/skills/
```

#### 在 Claude Code 中使用 Skills

Skills 會自動被 Claude 發現和使用，無需手動觸發。

##### 驗證 Skills 是否載入

在 Claude Code 中：

```bash
## 啟動 Claude Code
claude

## Claude 會自動掃描以下位置的 Skills：
## 1. ~/.claude/skills/        (個人 Skills)
## 2. .claude/skills/          (專案 Skills)
## 3. 已安裝插件的 Skills
```

##### 使用範例

```
使用者：請幫我建立一份關於再生能源的 PowerPoint 簡報，包含 5 頁投影片。

Claude：我將使用 PowerPoint Skill 為您建立簡報...
[自動載入 pptx Skill]
[執行簡報建立流程]
```

#### 在 Claude API 中使用 Skills

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

response = client.beta.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    # 啟用 Skills 功能
    betas=["code-execution-2025-08-25", "skills-2025-10-02"],
    # 指定要使用的 Skills
    container={
        "skills": [
            {
                "type": "anthropic",
                "skill_id": "pptx",
                "version": "latest"
            }
        ]
    },
    messages=[{
        "role": "user",
        "content": "建立一份關於 AI 技術的簡報，包含 5 頁投影片"
    }],
    # 啟用程式碼執行
    tools=[{
        "type": "code_execution_20250825",
        "name": "code_execution"
    }]
)

print(response.content)
```

#### 列出可用的 Skills

```python
## 列出所有可用的官方 Skills
skills_list = client.beta.skills.list()

for skill in skills_list.data:
    print(f"Skill ID: {skill.id}")
    print(f"Name: {skill.name}")
    print(f"Description: {skill.description}")
    print("---")
```

---

### 進階功能

#### 1. 工具限制（allowed-tools）

透過 `allowed-tools` 欄位，您可以限制 Skill 可以使用的工具，提高安全性和可控性。

##### 基本語法

```yaml
---
name: 唯讀檔案檢視器
description: 安全地檢視檔案內容，不進行任何修改
allowed-tools: [Read, Grep, Glob]
---
```

##### 可用工具清單

**檔案操作**：

- `Read` - 讀取檔案
- `Write` - 寫入檔案
- `Edit` - 編輯檔案
- `Glob` - 檔案搜尋（模式匹配）

**搜尋工具**：

- `Grep` - 內容搜尋

**執行工具**：

- `Bash` - 執行 Shell 命令

**版本控制**：

- `Git` - Git 操作

##### 工具權限模式

```yaml
## 模式 1：允許所有操作
allowed-tools: [Bash]

## 模式 2：允許任意參數
allowed-tools: [Bash(*)]

## 模式 3：僅允許特定命令
allowed-tools: [Bash(npm test), Bash(npm run build)]
```

##### 實際範例：程式碼審查 Skill

```yaml
---
name: 程式碼審查助手
description: 審查程式碼品質、檢查最佳實踐，不會修改任何程式碼
allowed-tools: [Read, Grep, Glob]
---

## 程式碼審查助手

這個 Skill 只能讀取和搜尋程式碼，無法進行任何修改，確保審查過程的安全性。

### 審查項目
1. 程式碼風格一致性
2. 潛在的錯誤模式
3. 效能優化建議
4. 安全性問題

### 操作流程
1. 使用 Glob 找出相關檔案
2. 使用 Read 讀取程式碼
3. 使用 Grep 搜尋特定模式
4. 提供審查報告
```

#### 2. 參考文件（Reference Files）

對於複雜的 Skills，可以使用額外的參考文件來組織大量資訊。

##### 目錄結構

```
advanced-skill/
├── SKILL.md
├── references/
│   ├── api-reference.md
│   ├── examples.md
│   └── troubleshooting.md
└── scripts/
    └── helper.sh
```

##### SKILL.md 中引用參考文件

```markdown
---
name: 進階 API 整合
description: 整合第三方 API，處理認證、請求和錯誤處理
---

## 進階 API 整合

### 主要指令
[基本操作說明...]

### 詳細參考
如需更多資訊，請參考：
- API 規格：見 references/api-reference.md
- 使用範例：見 references/examples.md
- 疑難排解：見 references/troubleshooting.md
```

#### 3. 可執行腳本

Skills 可以包含可執行的腳本，用於執行確定性操作。

##### 範例：資料處理 Skill

**目錄結構**：

```
data-processor/
├── SKILL.md
└── scripts/
    ├── validate.py
    ├── transform.py
    └── export.sh
```

**SKILL.md**：

```markdown
---
name: 資料處理器
description: 驗證、轉換和匯出資料檔案
---

## 資料處理器

### 驗證資料
執行 scripts/validate.py 來驗證資料格式：
```bash
python scripts/validate.py --input data.csv
```

### 轉換資料

使用 scripts/transform.py 進行資料轉換：

```bash
python scripts/transform.py --input data.csv --output result.json
```

### 匯出結果

執行 scripts/export.sh 匯出最終結果：

```bash
bash scripts/export.sh result.json
```

```

**scripts/validate.py**：
```python
##!/usr/bin/env python3
import sys
import argparse
import csv

def validate_csv(filepath):
    """驗證 CSV 檔案格式"""
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            required_fields = ['id', 'name', 'value']

            # 檢查欄位
            if not all(field in reader.fieldnames for field in required_fields):
                print(f"錯誤：缺少必要欄位。需要：{required_fields}")
                return False

            # 驗證每一行
            for i, row in enumerate(reader, 1):
                if not row['id'].isdigit():
                    print(f"錯誤：第 {i} 行的 id 必須是數字")
                    return False

            print("驗證成功！")
            return True
    except Exception as e:
        print(f"驗證失敗：{str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    success = validate_csv(args.input)
    sys.exit(0 if success else 1)
```

#### 4. 版本控制

為 Skills 建立版本管理，確保團隊使用一致的版本。

```yaml
---
name: 部署助手
description: 自動化部署流程，支援多環境配置
version: 2.1.0
---

## 部署助手 v2.1.0

### 版本歷史
- v2.1.0: 新增 staging 環境支援
- v2.0.0: 重構部署流程，支援 Docker
- v1.0.0: 初始版本

### 相容性
- 需要 Docker 20.10+
- 需要 Node.js 18+

[其他內容...]
```

#### 5. Skills 組合

Skills 可以相互配合，建立更複雜的工作流程。

##### 範例：全端開發工作流程

**專案結構**：

```
.claude/skills/
├── frontend-dev/
│   └── SKILL.md
├── backend-api/
│   └── SKILL.md
├── database-migration/
│   └── SKILL.md
└── deployment/
    └── SKILL.md
```

Claude 會根據任務自動組合使用這些 Skills：

```
使用者：實作一個新的使用者註冊功能

Claude：
1. [觸發 backend-api Skill] 建立 API 端點
2. [觸發 database-migration Skill] 新增資料庫表格
3. [觸發 frontend-dev Skill] 建立註冊表單
4. [觸發 deployment Skill] 部署到測試環境
```

---

### 實際應用案例

#### 案例 1：自動化文件產生

**需求**：為專案自動產生 API 文件

**Skill 設計**：

```markdown
---
name: API 文件產生器
description: 從程式碼註解自動產生 API 文件，支援 OpenAPI/Swagger 格式
allowed-tools: [Read, Write, Grep, Glob, Bash]
---

## API 文件產生器

### 功能
- 掃描 API 路由定義
- 解析函式註解
- 產生 OpenAPI 3.0 規格
- 輸出 Markdown 和 HTML 格式

### 使用步驟

#### 1. 掃描 API 檔案
使用 Glob 找出所有 API 路由檔案：
```bash
find src/routes -name "*.ts" -o -name "*.js"
```

#### 2. 提取 API 定義

讀取每個檔案，尋找以下模式：

- 路由定義（GET、POST、PUT、DELETE）
- 參數說明
- 回應格式
- 錯誤碼

#### 3. 產生文件

使用模板產生文件：

- OpenAPI YAML
- Markdown 格式的 README
- 互動式 HTML（使用 Swagger UI）

#### 4. 更新版本

自動更新版本號和變更日誌

### 範例輸出

#### API 端點

```markdown
### POST /api/users

建立新使用者

#### 請求參數
- `name` (string, required): 使用者名稱
- `email` (string, required): 電子郵件
- `role` (string, optional): 使用者角色，預設為 'user'

#### 回應
- 200: 成功建立
- 400: 參數錯誤
- 409: 使用者已存在
```

### 注意事項

- 確保程式碼註解完整
- 遵循 JSDoc 或 TypeDoc 規範
- 定期更新文件版本

```

**使用範例**：
```

使用者：請為我的 Express API 產生完整的文件

Claude：[自動觸發 API 文件產生器]
我將掃描您的 API 路由並產生文件...

1. 找到 15 個 API 端點
2. 提取參數和回應定義
3. 產生 OpenAPI 規格
4. 建立 Markdown 文件
5. 產生互動式 HTML

文件已產生在 docs/api/ 目錄中。

```

#### 案例 2：測試自動化

**需求**：自動執行測試並產生報告

**Skill 設計**：

```markdown
---
name: 測試執行器
description: 執行單元測試、整合測試，產生覆蓋率報告和測試摘要
allowed-tools: [Bash, Read, Write]
---

## 測試執行器

### 支援的測試框架
- Jest (JavaScript/TypeScript)
- Pytest (Python)
- RSpec (Ruby)
- Go test (Go)

### 工作流程

#### 1. 偵測測試框架
檢查專案配置檔案：
- package.json (Jest)
- pytest.ini (Pytest)
- .rspec (RSpec)

#### 2. 執行測試
```bash
## Jest
npm test -- --coverage

## Pytest
pytest --cov=src --cov-report=html

## Go
go test -coverprofile=coverage.out ./...
```

#### 3. 分析結果

- 解析測試輸出
- 計算通過率
- 識別失敗的測試
- 分析覆蓋率資料

#### 4. 產生報告

建立測試摘要報告，包括：

- 總測試數
- 通過/失敗/跳過
- 執行時間
- 覆蓋率百分比
- 失敗測試的詳細資訊

### 報告範例

```markdown
## 測試報告 - 2025-01-15

### 摘要
- 總測試數：156
- 通過：154 (98.7%)
- 失敗：2 (1.3%)
- 跳過：0
- 執行時間：23.4 秒

### 覆蓋率
- 語句覆蓋率：87.5%
- 分支覆蓋率：82.3%
- 函式覆蓋率：91.2%
- 行覆蓋率：88.1%

### 失敗的測試
1. `src/auth/login.test.ts` - 應該拒絕無效的密碼
2. `src/api/users.test.ts` - 應該驗證電子郵件格式

### 建議
- 修復失敗的測試
- 提高 auth 模組的覆蓋率（目前 75%）
```

### 整合 CI/CD

可與 GitHub Actions 或 GitLab CI 整合：

```yaml
## .github/workflows/test.yml
- name: Run tests with Claude
  run: claude "執行所有測試並產生報告"
```

```

#### 案例 3：資料分析與視覺化

**需求**：分析 CSV 資料並產生圖表

**Skill 設計**：

```markdown
---
name: 資料分析師
description: 載入、分析和視覺化資料，產生統計報告和圖表
---

## 資料分析師

### 功能
- 載入多種格式資料（CSV、JSON、Excel）
- 統計分析（平均值、中位數、標準差等）
- 資料清理和轉換
- 產生視覺化圖表

### 分析流程

#### 1. 載入資料
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 讀取資料
df = pd.read_csv('data.csv')

## 顯示基本資訊
print(df.info())
print(df.describe())
```

#### 2. 資料清理

- 處理缺失值
- 移除重複資料
- 轉換資料型別
- 標準化格式

#### 3. 探索性分析

- 計算描述性統計
- 識別異常值
- 分析相關性
- 分組聚合

#### 4. 視覺化

產生以下圖表：

- 直方圖（分佈）
- 散點圖（相關性）
- 箱型圖（異常值）
- 折線圖（趨勢）
- 熱力圖（相關矩陣）

### 範例程式碼

#### 銷售資料分析

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 載入資料
sales_df = pd.read_csv('sales_data.csv')

## 按月份分組
monthly_sales = sales_df.groupby('month')['revenue'].sum()

## 產生圖表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

## 月銷售額趨勢
axes[0, 0].plot(monthly_sales.index, monthly_sales.values)
axes[0, 0].set_title('月銷售額趨勢')
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('銷售額')

## 產品類別分佈
category_counts = sales_df['category'].value_counts()
axes[0, 1].bar(category_counts.index, category_counts.values)
axes[0, 1].set_title('產品類別分佈')

## 銷售額分佈
axes[1, 0].hist(sales_df['revenue'], bins=30)
axes[1, 0].set_title('銷售額分佈')

## 相關性熱力圖
numeric_cols = sales_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_cols.corr(), annot=True, ax=axes[1, 1])
axes[1, 1].set_title('特徵相關性')

plt.tight_layout()
plt.savefig('sales_analysis.png', dpi=300)
print("分析完成！圖表已儲存為 sales_analysis.png")
```

### 報告範例

```markdown
## 銷售資料分析報告

### 資料概覽
- 記錄數：10,523
- 時間範圍：2024-01-01 至 2024-12-31
- 產品類別：15 種

### 關鍵發現
1. 平均月銷售額：$1,234,567
2. 最佳銷售月份：12 月（$2,100,000）
3. 成長率：相較去年成長 23%
4. 最暢銷類別：電子產品（35%）

### 建議
- 在第四季增加庫存
- 加強電子產品行銷
- 改善低銷售月份的促銷活動
```

### 支援的資料格式

- CSV
- Excel (xlsx, xls)
- JSON
- Parquet
- SQL 資料庫

```

#### 案例 4：程式碼重構助手

**需求**：識別程式碼異味並建議重構

**Skill 設計**：

```markdown
---
name: 程式碼重構助手
description: 分析程式碼品質，識別重構機會，提供改進建議
allowed-tools: [Read, Grep, Glob, Write]
---

## 程式碼重構助手

### 功能
- 識別程式碼異味
- 檢測重複程式碼
- 分析複雜度
- 建議設計模式
- 產生重構計畫

### 檢查項目

#### 1. 程式碼異味
- **過長函式**：超過 50 行
- **過多參數**：超過 4 個參數
- **重複程式碼**：相似程度 > 80%
- **深層巢狀**：超過 3 層

#### 2. 複雜度分析
- 循環複雜度（Cyclomatic Complexity）
- 認知複雜度（Cognitive Complexity）
- 維護性指數

#### 3. SOLID 原則檢查
- 單一職責原則
- 開放封閉原則
- 里氏替換原則
- 介面隔離原則
- 依賴反轉原則

### 重構策略

#### 提取函式
將過長的函式拆分：
```javascript
// 重構前
function processUser(user) {
  // 驗證（10 行）
  // 資料轉換（15 行）
  // 儲存到資料庫（20 行）
  // 發送通知（10 行）
}

// 重構後
function processUser(user) {
  validateUser(user);
  const transformed = transformUserData(user);
  saveToDatabase(transformed);
  sendNotification(user);
}

function validateUser(user) { /* ... */ }
function transformUserData(user) { /* ... */ }
function saveToDatabase(data) { /* ... */ }
function sendNotification(user) { /* ... */ }
```

#### 引入參數物件

減少參數數量：

```typescript
// 重構前
function createOrder(
  userId: string,
  productId: string,
  quantity: number,
  price: number,
  discount: number,
  shippingAddress: string
) { /* ... */ }

// 重構後
interface OrderParams {
  userId: string;
  productId: string;
  quantity: number;
  price: number;
  discount: number;
  shippingAddress: string;
}

function createOrder(params: OrderParams) { /* ... */ }
```

#### 移除重複程式碼

```python
## 重構前
def calculate_employee_salary(employee):
    base = employee.base_salary
    bonus = base * 0.1
    tax = (base + bonus) * 0.2
    return base + bonus - tax

def calculate_manager_salary(manager):
    base = manager.base_salary
    bonus = base * 0.2  # 不同的獎金率
    tax = (base + bonus) * 0.2
    return base + bonus - tax

## 重構後
def calculate_salary(person, bonus_rate=0.1):
    base = person.base_salary
    bonus = base * bonus_rate
    tax = (base + bonus) * 0.2
    return base + bonus - tax

def calculate_employee_salary(employee):
    return calculate_salary(employee, bonus_rate=0.1)

def calculate_manager_salary(manager):
    return calculate_salary(manager, bonus_rate=0.2)
```

### 重構報告範例

```markdown
## 程式碼重構報告

### 檔案：src/services/userService.js

#### 問題 1：過長函式
**位置**：第 45-120 行
**函式**：`processUserRegistration`
**行數**：75 行
**建議**：拆分為多個職責單一的函式

#### 問題 2：重複程式碼
**位置**：
- `validateEmail` (第 200-215 行)
- `validateBusinessEmail` (第 230-245 行)
**相似度**：85%
**建議**：提取共同邏輯到 `validateEmailFormat`

#### 問題 3：過多參數
**位置**：第 300 行
**函式**：`createUserProfile`
**參數數量**：7 個
**建議**：引入 `UserProfileParams` 介面

### 優先順序
1. 高：重構 `processUserRegistration`（影響維護性）
2. 中：移除重複的驗證邏輯
3. 低：引入參數物件（改善可讀性）

### 預期效益
- 降低複雜度 40%
- 提高測試覆蓋率至 90%
- 改善可維護性指數 25%
```

### 使用方式

```
使用者：請分析 src/services/ 目錄下的程式碼品質

Claude：[觸發程式碼重構助手]
正在分析程式碼品質...
- 掃描 15 個檔案
- 偵測到 8 個重構機會
- 產生詳細報告

報告已產生在 refactoring-report.md
```

```

#### 案例 5：部署自動化

**需求**：自動化應用程式部署流程

```markdown
---
name: 部署自動化
description: 自動化應用程式部署到不同環境（開發、測試、正式）
allowed-tools: [Bash, Read, Write]
---

## 部署自動化

### 支援環境
- Development（開發）
- Staging（測試）
- Production（正式）

### 部署流程

#### 1. 前置檢查
```bash
## 檢查 git 狀態
git status

## 確認分支
current_branch=$(git branch --show-current)
echo "當前分支：$current_branch"

## 檢查是否有未提交的變更
if [[ -n $(git status -s) ]]; then
  echo "錯誤：有未提交的變更"
  exit 1
fi
```

#### 2. 執行測試

```bash
## 執行單元測試
npm test

## 執行整合測試
npm run test:integration

## 檢查程式碼品質
npm run lint
```

#### 3. 建置應用程式

```bash
## 清理舊的建置
rm -rf dist/

## 建置生產版本
npm run build

## 驗證建置結果
if [ ! -d "dist" ]; then
  echo "錯誤：建置失敗"
  exit 1
fi
```

#### 4. 部署

##### Development 環境

```bash
## 部署到開發環境
npm run deploy:dev

## 驗證部署
curl -f https://dev.example.com/health || exit 1
```

##### Staging 環境

```bash
## 確認在 develop 分支
if [ "$current_branch" != "develop" ]; then
  echo "錯誤：請切換到 develop 分支"
  exit 1
fi

## 部署到測試環境
npm run deploy:staging

## 執行煙霧測試
npm run test:smoke -- --env=staging
```

##### Production 環境

```bash
## 確認在 main 分支
if [ "$current_branch" != "main" ]; then
  echo "錯誤：只能從 main 分支部署到正式環境"
  exit 1
fi

## 建立版本標籤
version=$(node -p "require('./package.json').version")
git tag -a "v$version" -m "Release version $version"

## 部署到正式環境
npm run deploy:production

## 執行完整測試
npm run test:e2e -- --env=production

## 推送標籤
git push origin "v$version"
```

#### 5. 部署後驗證

```bash
## 健康檢查
check_health() {
  local env=$1
  local url=$2

  echo "檢查 $env 環境健康狀態..."
  response=$(curl -s -o /dev/null -w "%{http_code}" "$url/health")

  if [ "$response" -eq 200 ]; then
    echo "✓ $env 環境正常運作"
  else
    echo "✗ $env 環境異常（HTTP $response）"
    return 1
  fi
}

## 檢查各環境
check_health "Production" "https://api.example.com"
```

#### 6. 回滾機制

```bash
rollback() {
  local env=$1
  local version=$2

  echo "回滾 $env 環境到版本 $version..."

  # 回滾部署
  npm run rollback -- --env="$env" --version="$version"

  # 驗證回滾結果
  check_health "$env" "https://${env}.example.com"
}

## 使用範例
## rollback "production" "v1.2.3"
```

### 部署檢查清單

#### 部署前

- [ ] 所有測試通過
- [ ] 程式碼審查完成
- [ ] 沒有未解決的 merge conflicts
- [ ] 環境變數已配置
- [ ] 資料庫遷移已準備

#### 部署中

- [ ] 監控錯誤日誌
- [ ] 檢查效能指標
- [ ] 驗證關鍵功能

#### 部署後

- [ ] 執行健康檢查
- [ ] 驗證新功能運作
- [ ] 更新文件
- [ ] 通知團隊

### 環境變數範例

```bash
## .env.production
NODE_ENV=production
API_URL=https://api.example.com
DATABASE_URL=postgresql://prod-db.example.com/app
REDIS_URL=redis://prod-redis.example.com:6379
LOG_LEVEL=info
```

### 通知整合

#### Slack 通知

```bash
send_slack_notification() {
  local message=$1
  local webhook_url=$SLACK_WEBHOOK_URL

  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$message\"}" \
    "$webhook_url"
}

## 部署成功通知
send_slack_notification "✓ 部署成功：版本 v1.2.3 已部署到 Production"
```

### 使用範例

```
使用者：部署最新版本到測試環境

Claude：[觸發部署自動化]
開始部署流程...

1. ✓ 前置檢查通過
2. ✓ 所有測試通過（156/156）
3. ✓ 建置完成
4. ✓ 部署到 Staging 環境
5. ✓ 健康檢查通過
6. ✓ 煙霧測試通過

部署完成！
URL: https://staging.example.com
版本：v1.2.3
時間：2025-01-15 14:30:00
```

```

---

### 最佳實踐

#### 1. 命名規範

##### Skill 名稱
- ✅ **使用動名詞形式**：「處理 PDF 文件」而非「PDF 處理器」
- ✅ **清晰且具描述性**：明確說明 Skill 的功能
- ✅ **避免過於通用**：「API 文件產生器」比「文件產生」更好
- ❌ **避免技術術語過多**：一般使用者也應該能理解

##### 描述撰寫
- ✅ **使用第三人稱**：「提取 PDF 文字並分析內容」
- ✅ **包含兩個部分**：
  1. 這個 Skill 做什麼
  2. 何時應該使用它
- ✅ **具體且明確**：提供足夠資訊讓 Claude 判斷是否相關
- ✅ **保持簡潔**：在 1024 字元限制內

**範例**：
```yaml
## ✅ 好的描述
description: 從 PDF 文件提取文字、表單資料和圖片。當需要處理 PDF 檔案、提取內容或填寫 PDF 表單時使用。

## ❌ 不好的描述
description: 處理 PDF

## ❌ 過於技術性
description: 使用 pdfplumber 和 PyPDF2 函式庫進行 PDF 解析和資料提取
```

#### 2. 內容組織

##### 保持簡潔

- **SKILL.md 建議長度**：少於 500 行
- **僅包含 Claude 不知道的資訊**
- **避免解釋基礎概念**

**檢查清單**：

- [ ] 這個資訊是 Claude 已經知道的嗎？
- [ ] 這個細節對完成任務真的必要嗎？
- [ ] 有沒有更簡潔的表達方式？

##### 漸進式揭露

將資訊分層組織：

```markdown
---
name: 資料庫遷移助手
description: 管理資料庫 schema 變更、執行遷移腳本、回滾機制
---

## 資料庫遷移助手

### 快速指南
[最常用的操作...]

### 進階功能
如需詳細資訊，請參考 references/advanced-migrations.md

### 疑難排解
常見問題見 references/troubleshooting.md
```

##### 參考文件組織

```
database-skill/
├── SKILL.md              # 主要指令（簡潔）
└── references/
    ├── migrations.md     # 遷移詳細說明
    ├── rollback.md       # 回滾機制
    └── best-practices.md # 最佳實踐
```

**重要原則**：

- 參考文件保持一層深度
- 使用目錄索引
- 避免深層巢狀

#### 3. 自由度設定

根據任務複雜度調整指令的具體程度。

##### 高度結構化（低自由度）

適用於：確定性任務、重複性流程、安全性要求高的操作

```markdown
### 部署流程（必須按照順序執行）

1. 執行測試
   ```bash
   npm test
```

   如果測試失敗，停止部署。

2. 建置專案

   ```bash
   npm run build
   ```
3. 部署到環境

   ```bash
   npm run deploy:production
   ```
4. 驗證部署

   ```bash
   curl https://example.com/health
   ```

```

##### 適度靈活（中等自由度）
適用於：一般性任務、需要判斷的情況

```markdown
### 程式碼審查流程

#### 主要檢查項目
- 程式碼風格和一致性
- 潛在的錯誤模式
- 效能考量
- 安全性問題

#### 方法
根據專案情況選擇：
- 靜態分析工具
- 手動審查
- 自動化測試

提供具體的改進建議。
```

##### 高度彈性（高自由度）

適用於：創意任務、探索性分析

```markdown
### 資料分析

探索資料集並找出有價值的洞察：
- 識別趨勢和模式
- 發現異常值
- 建議進一步分析的方向

使用適合的方法和視覺化技術。
```

#### 4. 避免的反模式

##### ❌ 時間敏感資訊

```markdown
## 不要這樣做
目前最新版本是 React 18（2024 年 3 月）
```

原因：資訊會過時

##### ❌ 術語不一致

```markdown
## 不要這樣做
使用 `user` 變數...
稍後，處理 `customer` 物件...
## user 和 customer 是同一個東西嗎？
```

改為：

```markdown
## 這樣做
使用 `user` 變數表示已登入的使用者...
稍後，處理同一個 `user` 物件...
```

##### ❌ Windows 路徑格式

```markdown
## 不要這樣做
C:\Users\project\config.json

## 這樣做
/Users/project/config.json
或使用相對路徑：./config.json
```

##### ❌ 提供太多選項

```markdown
## 不要這樣做
你可以使用以下任一方法：
1. 方法 A
2. 方法 B
3. 方法 C
4. 方法 D
5. 方法 E
```

改為：

```markdown
## 這樣做
推薦方法：使用方法 A（最可靠）

替代方法：
- 方法 B：如果需要更高效能
- 方法 C：如果環境不支援方法 A
```

##### ❌ 將錯誤處理推給 Claude

```markdown
## 不要這樣做
執行腳本，如果出錯就想辦法修復

## 這樣做
執行腳本：
```bash
python script.py
```

常見錯誤：

- ModuleNotFoundError: 執行 `pip install -r requirements.txt`
- PermissionError: 執行 `chmod +x script.py`
- 其他錯誤：檢查 logs/error.log 中的詳細資訊

```

#### 5. 測試與迭代

##### 建立評估場景
在撰寫大量文件前，先建立測試案例：

```markdown
## test-scenarios.md

### 場景 1：新增 API 端點
輸入：「新增一個 GET /api/users 端點」
預期：
- 建立路由檔案
- 新增控制器
- 撰寫測試
- 更新 API 文件

### 場景 2：資料庫遷移
輸入：「新增 email 欄位到 users 表格」
預期：
- 產生遷移檔案
- 執行遷移
- 更新模型
- 執行測試驗證
```

##### 與 Claude 一起開發

```
使用者：我想建立一個部署自動化的 Skill，你有什麼建議？

Claude：讓我們一起設計這個 Skill...

首先，考慮以下關鍵面向：
1. 支援的環境（開發/測試/正式）
2. 部署前檢查（測試、lint）
3. 建置流程
4. 部署後驗證
5. 回滾機制

讓我們先從 SKILL.md 的架構開始...
```

##### 漸進式改進

1. **第一版**：基本功能
2. **第二版**：新增錯誤處理
3. **第三版**：新增進階功能
4. **第四版**：優化效能和使用者體驗

#### 6. 使用驗證腳本

為複雜的 Skills 提供驗證工具：

```markdown
### 驗證 Skill 配置

執行驗證腳本：
```bash
python scripts/validate.py
```

這將檢查：

- 必要的環境變數
- 依賴套件安裝狀態
- 配置檔案格式
- 存取權限

範例輸出：

```
✓ 環境變數已設定
✓ 所有依賴已安裝
✓ 配置檔案格式正確
✗ 缺少 logs/ 目錄寫入權限

請執行：mkdir -p logs && chmod 755 logs
```

```

#### 7. 文件化最佳實踐

##### 包含範例
每個 Skill 都應該包含具體範例：

```markdown
### 範例

#### 範例 1：基本使用
輸入：
```

建立一份產品介紹簡報

```

輸出：
- 5 頁 PowerPoint 簡報
- 包含：封面、產品特點、優勢、案例、結語

#### 範例 2：自訂範本
輸入：
```

使用公司範本建立季度報告簡報

```

預期行為：
1. 載入 templates/corporate.pptx
2. 套用公司色彩和字型
3. 插入季度資料
4. 產生圖表
```

##### 說明限制

清楚說明 Skill 的限制：

```markdown
### 限制與注意事項

#### 不支援
- 加密的 PDF 文件
- 掃描的圖片型 PDF（無文字層）
- 超過 100MB 的檔案

#### 需要的依賴
- Python 3.8+
- pdfplumber 套件
- 足夠的記憶體（建議 2GB+）

#### 環境需求
- 僅在 Claude Code 中可用
- 需要檔案系統存取權限
```

#### 8. 版本管理

##### 使用語意化版本

```yaml
---
name: API 測試器
version: 2.1.0
---

## 版本歷史
- v2.1.0 (2025-01-15): 新增 GraphQL 支援
- v2.0.0 (2024-12-01): 重構核心邏輯，破壞性變更
- v1.2.0 (2024-11-15): 新增批次測試功能
- v1.1.0 (2024-10-01): 改善錯誤報告
- v1.0.0 (2024-09-01): 初始發布
```

##### 記錄破壞性變更

```markdown
### 遷移指南：v1.x 到 v2.0

#### 破壞性變更
1. 配置檔案格式變更
   - 舊格式：`config.json`
   - 新格式：`config.yaml`
   - 遷移工具：`python migrate-config.py`

2. API 端點測試語法
   - 舊：`test endpoint /api/users`
   - 新：`test GET /api/users`

#### 新功能
- 支援 WebSocket 測試
- 整合負載測試
- 自動產生測試報告
```

---

### 常見問題與疑難排解

#### 安裝與配置問題

##### Q1: Skills 沒有被載入

**症狀**：Claude 沒有使用我建立的 Skill

**檢查清單**：

1. 確認檔案位置正確

   ```bash
   # 個人 Skills
   ls ~/.claude/skills/

   # 專案 Skills
   ls .claude/skills/
   ```
2. 檢查 SKILL.md 格式

   ```bash
   # 驗證 YAML frontmatter
   head -n 10 ~/.claude/skills/my-skill/SKILL.md
   ```
3. 確認 YAML 語法正確

   ```yaml
   ---
   name: 技能名稱
   description: 描述必須存在
   ---
   ```
4. 重新啟動 Claude Code

   ```bash
   # 關閉並重新開啟
   exit
   claude
   ```

**除錯模式**：

```bash
## 啟用詳細日誌
claude --verbose

## 檢查 Skill 載入狀態
claude /doctor
```

##### Q2: Skill 描述不夠明確

**問題**：Claude 不知道何時使用 Skill

**解決方案**：改進描述

❌ **不好的描述**：

```yaml
description: 處理資料
```

✅ **好的描述**：

```yaml
description: 分析 CSV 和 Excel 檔案，計算統計數據並產生視覺化圖表。當需要處理表格資料、產生報告或資料分析時使用。
```

**描述撰寫技巧**：

- 包含關鍵字（CSV、Excel、分析、圖表）
- 明確說明使用時機
- 提到具體的輸入/輸出

##### Q3: 路徑問題

**症狀**：找不到檔案或腳本

**原因**：使用了絕對路徑或 Windows 格式路徑

**解決方案**：

```markdown
## ❌ 不要這樣
C:\Users\project\script.py
/Users/specific-user/project/script.py

## ✅ 使用相對路徑
./scripts/script.py
../templates/template.txt

## ✅ 使用變數
$SKILL_DIR/scripts/script.py
```

#### 執行問題

##### Q4: 腳本執行失敗

**症狀**：Permission denied 錯誤

**解決方案**：

```bash
## 新增執行權限
chmod +x ~/.claude/skills/my-skill/scripts/helper.sh

## 驗證權限
ls -l ~/.claude/skills/my-skill/scripts/
```

##### Q5: 依賴套件缺失

**症狀**：ModuleNotFoundError 或 command not found

**解決方案**：

1. **Python 依賴**

   ```bash
   # 在 Skill 目錄建立 requirements.txt
   cat > requirements.txt << EOF
   pandas==2.0.0
   matplotlib==3.7.0
   pdfplumber==0.10.0
   EOF

   # 安裝依賴
   pip install -r requirements.txt
   ```
2. **Node.js 依賴**

   ```bash
   # 建立 package.json
   npm init -y
   npm install axios cheerio
   ```
3. **系統依賴**

   ```bash
   # macOS
   brew install ripgrep fd

   # Ubuntu/Debian
   sudo apt-get install ripgrep fd-find
   ```

**在 SKILL.md 中記錄依賴**：

```yaml
---
name: 資料分析
description: 分析資料並產生圖表
dependencies:
  - python: ">=3.8"
  - packages:
      - pandas>=2.0.0
      - matplotlib>=3.7.0
---

## 資料分析

### 安裝依賴
```bash
pip install -r requirements.txt
```

```

##### Q6: 工具限制衝突

**症狀**：Skill 需要的工具被 allowed-tools 限制

**解決方案**：調整 allowed-tools 設定

```yaml
## 問題：Skill 需要寫入檔案，但只允許讀取
---
name: 報告產生器
description: 產生分析報告
allowed-tools: [Read, Grep]  # ❌ 缺少 Write
---

## 解決：新增必要的工具
---
name: 報告產生器
description: 產生分析報告
allowed-tools: [Read, Grep, Write, Bash]  # ✅ 包含所有需要的工具
---
```

#### 效能問題

##### Q7: Skill 載入緩慢

**原因**：SKILL.md 檔案過大

**解決方案**：使用參考文件分離內容

```markdown
## ❌ 所有內容都在 SKILL.md（5000+ 行）
---
name: 完整指南
---

## 完整指南
[3000 行詳細說明...]
[1000 行範例...]
[1000 行 API 參考...]

## ✅ 分離到參考文件
---
name: 完整指南
---

## 完整指南

### 快速開始
[200 行核心指令]

### 詳細參考
- API 規格：見 references/api.md
- 範例：見 references/examples.md
- 疑難排解：見 references/troubleshooting.md
```

##### Q8: Token 限制

**症狀**：Context window 超出限制

**原因**：載入了太多 Skill 內容

**解決方案**：

1. 精簡 SKILL.md 內容
2. 使用腳本處理大量資料（不占用 context）
3. 分割為多個專門的 Skills

```markdown
## ❌ 單一龐大 Skill
web-development/
└── SKILL.md  # 包含前端、後端、資料庫、部署...

## ✅ 分割為多個 Skills
web-development/
├── frontend/
│   └── SKILL.md
├── backend/
│   └── SKILL.md
├── database/
│   └── SKILL.md
└── deployment/
    └── SKILL.md
```

#### 整合問題

##### Q9: Git 整合問題

**症狀**：專案 Skills 沒有被版本控制追蹤

**解決方案**：

```bash
## 確保 .claude/skills/ 沒有被 .gitignore 忽略
cat .gitignore

## 如果被忽略，修改 .gitignore
## 移除或註解：
## .claude/

## 提交 Skills
git add .claude/skills/
git commit -m "Add project skills"
git push
```

##### Q10: 團隊協作問題

**症狀**：團隊成員的 Skills 不一致

**解決方案**：

1. **使用專案 Skills**

   ```bash
   # 所有專案特定的 Skills 放在專案目錄
   .claude/skills/
   ```
2. **建立 Skills 文件**

   ```markdown
   # .claude/README.md

   # 專案 Skills 指南

   ## 可用的 Skills
   - `api-testing`: API 測試自動化
   - `code-review`: 程式碼審查助手
   - `deployment`: 部署流程

   ## 安裝依賴
   ```bash
   # 安裝所有 Skills 的依賴
   ./scripts/install-skills-deps.sh
   ```

   ## 使用指南

   [...]


   ```

   ```
3. **提供安裝腳本**

   ```bash
   # scripts/install-skills-deps.sh
   #!/bin/bash

   echo "安裝 Skills 依賴..."

   # Python 依賴
   if [ -f ".claude/skills/requirements.txt" ]; then
     pip install -r .claude/skills/requirements.txt
   fi

   # Node.js 依賴
   if [ -f ".claude/skills/package.json" ]; then
     cd .claude/skills && npm install && cd ../..
   fi

   echo "✓ 依賴安裝完成"
   ```

#### 除錯技巧

##### 使用 /doctor 命令

```bash
## 在 Claude Code 中
claude

## 執行診斷
/doctor
```

輸出範例：

```
診斷報告：
✓ Claude Code 版本：1.2.0（最新）
✓ Node.js 版本：20.10.0
✓ 已載入 5 個 Skills
  - PDF 處理器 (個人)
  - API 測試 (專案)
  - 程式碼審查 (專案)
  - 部署助手 (專案)
  - PowerPoint (內建)

✗ 警告：Skills/data-processor/SKILL.md 格式錯誤
  第 3 行：YAML frontmatter 未正確關閉

建議：檢查 YAML 語法
```

##### 啟用詳細日誌

```bash
## 啟用詳細日誌模式
claude --verbose

## 或設定環境變數
export CLAUDE_LOG_LEVEL=debug
claude
```

##### 檢查 Skill 內容

```bash
## 顯示 Skill 的完整內容
cat ~/.claude/skills/my-skill/SKILL.md

## 驗證 YAML frontmatter
head -n 20 ~/.claude/skills/my-skill/SKILL.md

## 檢查檔案權限
ls -la ~/.claude/skills/my-skill/
```

##### 測試特定 Skill

```
使用者：請使用「資料分析」Skill 分析這個 CSV 檔案

Claude：[明確觸發特定 Skill]
```

#### 常見錯誤訊息

##### 錯誤 1：YAML 語法錯誤

```
Error: Invalid YAML frontmatter in SKILL.md
```

**原因**：YAML 格式不正確

**檢查**：

```yaml
## ❌ 錯誤
---
name: My Skill
description: This is a test
## 缺少結尾的 ---

## ✅ 正確
---
name: My Skill
description: This is a test
---
```

##### 錯誤 2：必要欄位缺失

```
Error: Missing required field 'description' in SKILL.md
```

**解決**：確保包含所有必要欄位

```yaml
---
name: 技能名稱          # 必要
description: 完整描述   # 必要
---
```

##### 錯誤 3：檔案未找到

```
Error: Reference file 'references/api.md' not found
```

**解決**：

```bash
## 檢查檔案是否存在
ls -la ~/.claude/skills/my-skill/references/

## 建立缺失的檔案
mkdir -p ~/.claude/skills/my-skill/references
touch ~/.claude/skills/my-skill/references/api.md
```

#### 效能優化建議

1. **精簡 SKILL.md**：保持在 500 行以內
2. **使用參考文件**：大量內容放在單獨檔案
3. **腳本處理資料**：避免將大資料放在指令中
4. **清晰的描述**：讓 Claude 快速判斷相關性
5. **定期清理**：移除不再使用的 Skills

#### 獲取幫助

**官方資源**：

- 文件：https://docs.claude.com/en/docs/claude-code/skills
- GitHub 範例：https://github.com/anthropics/skills
- 社群討論：https://github.com/anthropics/skills/discussions

**除錯檢查清單**：

- [ ] YAML frontmatter 格式正確
- [ ] 包含 name 和 description
- [ ] 檔案路徑正確
- [ ] 腳本有執行權限
- [ ] 依賴套件已安裝
- [ ] 重新啟動 Claude Code
- [ ] 執行 /doctor 診斷

---

### 進階主題

#### 與 MCP (Model Context Protocol) 的比較

**Agent Skills**：

- 基於檔案系統
- 輕量級，token 效率高
- 易於分享和版本控制
- 需要程式碼執行環境

**MCP**：

- 基於協議和伺服器
- 可連接外部服務
- 需要額外的伺服器設定
- 更適合存取外部資源

**使用建議**：

- 工作流程和指令：使用 Skills
- 外部 API 整合：使用 MCP
- 可以同時使用兩者

#### 建立 Skills 市集

如果您開發了有用的 Skills，可以分享給社群：

1. **開源發布**

   ```bash
   # 在 GitHub 建立專案
   git init
   git add .
   git commit -m "Initial commit: My Awesome Skill"
   git remote add origin https://github.com/yourusername/my-skill.git
   git push -u origin main
   ```
2. **撰寫說明文件**

   ```markdown
   # My Awesome Skill

   ## 簡介
   這個 Skill 可以...

   ## 安裝
   ```bash
   cp -r my-skill ~/.claude/skills/
   ```

   ## 使用範例

   [...]

   ## 授權

   MIT License


   ```

   ```
3. **提交到社群清單**

   - Awesome Claude Skills：https://github.com/JayZeeDesign/awesome-claude-skills
   - Agent Skills Best：https://agentskills.best

#### Skills 的未來

Anthropic 的願景是讓 AI 代理能夠：

- 自主建立新 Skills
- 編輯和改進現有 Skills
- 評估 Skills 的效果
- 從經驗中學習

這將使 AI 系統能夠持續進化和適應新任務。

---

### 總結

Agent Skills 是 Claude Code 的強大功能，讓您能夠：

1. **模組化工作流程**：將常用流程封裝成可重複使用的單元
2. **專業化能力**：為特定領域提供專業知識
3. **提高效率**：自動化複雜的多步驟任務
4. **團隊協作**：共享最佳實踐和標準流程

#### 關鍵要點

- **三層載入架構**：僅在需要時載入相關內容
- **三種 Skills 類型**：個人、專案、插件
- **簡潔明確**：保持 SKILL.md 簡潔，描述清晰
- **漸進式改進**：從基本功能開始，逐步完善
- **測試驗證**：建立測試場景，確保 Skill 正常運作

#### 下一步行動

1. **建立第一個 Skill**：從簡單的工作流程開始
2. **探索官方範例**：研究 https://github.com/anthropics/skills
3. **分享經驗**：將有用的 Skills 開源分享
4. **持續學習**：關注 Agent Skills 的最新發展

#### 資源連結

- **官方文件**：https://docs.claude.com/en/docs/claude-code/skills
- **範例庫**：https://github.com/anthropics/skills
- **最佳實踐**：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices
- **API 文件**：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/quickstart
- **社群資源**：https://agentskills.best

---

**文件版本**：1.0.0
**最後更新**：2025-01-15
**作者**：基於 Claude Code 官方文件整理

---

### 附錄：快速參考

#### SKILL.md 範本

```markdown
---
name: 您的 Skill 名稱
description: 清楚描述這個 Skill 的功能以及何時使用它（必須包含「做什麼」和「何時用」）
version: 1.0.0
allowed-tools: [Read, Write, Bash, Grep, Glob]
---

## 您的 Skill 名稱

### 概述
簡要說明這個 Skill 的用途

### 使用時機
- 情境 1
- 情境 2
- 情境 3

### 操作步驟

#### 步驟 1：準備
[具體指令]

#### 步驟 2：執行
[具體指令]

#### 步驟 3：驗證
[具體指令]

### 範例

#### 範例 1：基本使用
```

輸入範例

```

預期輸出：
```

輸出範例

```

### 注意事項
- 限制 1
- 限制 2
- 依賴項目
```

#### 常用命令速查

```bash
## 建立新 Skill
mkdir -p ~/.claude/skills/my-skill
nano ~/.claude/skills/my-skill/SKILL.md

## 檢視已安裝的 Skills
ls ~/.claude/skills/
ls .claude/skills/

## 驗證 Skill 格式
head -n 20 ~/.claude/skills/my-skill/SKILL.md

## 執行診斷
claude /doctor

## 啟用詳細日誌
claude --verbose

## 清除對話歷史
claude
/clear

## 離開 Claude
/exit
```

#### 除錯檢查清單

```markdown
Skills 無法載入時的檢查順序：

1. [ ] 檔案位置正確（~/.claude/skills/ 或 .claude/skills/）
2. [ ] SKILL.md 檔案存在
3. [ ] YAML frontmatter 格式正確
4. [ ] name 和 description 欄位存在
5. [ ] description 清楚說明功能和使用時機
6. [ ] 檔案權限正確（可讀取）
7. [ ] 腳本有執行權限（chmod +x）
8. [ ] 依賴套件已安裝
9. [ ] 重新啟動 Claude Code
10. [ ] 執行 /doctor 檢查狀態
```

祝您使用 Agent Skills 愉快！

---

## 5. 擴展 Claude Code 功能

### 📋 學習摘要

**學習目標：** 掌握 Claude Code 的實際擴展機制

**核心內容：**

- 自訂命令 (Commands)
- 專業化代理 (Sub-agents)
- 技能模組 (Skills)
- MCP 伺服器整合

**關鍵技能：**

- ✅ 建立自訂斜線命令
- ✅ 開發專業化代理程式
- ✅ 配置技能模組
- ✅ 整合 MCP 伺服器

**實際擴展方式：**

- `.claude/commands/` - 自訂命令
- `.claude/agents/` - 專業化代理
- `.claude/skills/` - 技能模組
- `.claude/settings.json` - MCP 配置

**預計學習時間：** 2-3 小時

**詳細教學：** [→ 查看完整教學內容](#擴展功能-詳細內容)

### 📖 完整教學內容

### 1. 概述

#### Claude Code 的實際擴展方式

**重要聲明**：Claude Code 目前**沒有**傳統意義上的「外掛程式市場」或 `claude plugins` CLI 命令。擴展功能主要透過以下四種機制實現：

1. **Commands（自訂命令）** - 在 `.claude/commands/` 目錄中的 Markdown 檔案
2. **Sub-agents（專業化代理）** - 在 `.claude/agents/` 目錄中的 JSON 配置檔
3. **Skills（技能模組）** - 在 `.claude/skills/` 目錄中的專業化指令集
4. **MCP 伺服器** - 透過 Model Context Protocol 連接外部服務

#### 四種核心擴展機制

```
.claude/
├── commands/          # 自訂斜線命令
│   ├── deploy.md
│   └── review-pr.md
├── agents/            # 專業化代理
│   ├── code-reviewer.json
│   └── security-auditor.json
├── skills/            # 技能模組
│   ├── testing/
│   └── deployment/
└── settings.json      # MCP 伺服器配置
```

#### 與其他編輯器外掛系統的區別

| 特性               | VS Code 外掛          | Claude Code 擴展   |
| ------------------ | --------------------- | ------------------ |
| **安裝方式** | 市場下載、一鍵安裝    | 手動建立配置檔案   |
| **發佈平台** | Extension Marketplace | Git 倉庫、團隊共享 |
| **配置格式** | JavaScript/TypeScript | Markdown/JSON      |
| **執行環境** | Extension Host        | Claude 對話環境    |
| **版本管理** | 自動更新              | Git 版本控制       |

**Claude Code 的優勢：**

- 配置簡單，純文字檔案
- 易於版本控制和團隊協作
- 不需要編譯或打包
- 即時生效，無需重啟

---

### 2. 自訂命令 (Commands)

#### 位置與檔案格式

Commands 存放在專案的 `.claude/commands/` 目錄中，每個命令是一個 Markdown 檔案。

**目錄結構：**

```
.claude/
└── commands/
    ├── deploy.md
    ├── review-pr.md
    └── generate-tests.md
```

#### 基本語法

每個命令檔案包含 YAML frontmatter 和 Markdown 內容：

```markdown
---
description: 命令的簡短描述
---

# 命令的詳細指令

這裡是 Claude 執行此命令時會遵循的指示...
```

#### 完整範例 1：程式碼審查命令

**檔案：** `.claude/commands/review-pr.md`

```markdown
---
description: 執行全面的 Pull Request 程式碼審查
---

# Pull Request 審查命令

執行以下步驟來審查 Pull Request：

## 1. 分析變更範圍

- 使用 Bash 工具執行 `git diff main...HEAD --stat` 查看變更的檔案
- 識別修改、新增、刪除的檔案數量
- 評估變更的複雜度

## 2. 審查程式碼品質

對每個修改的檔案：
- 檢查命名是否清晰且遵循慣例
- 確認函數職責單一性
- 檢視錯誤處理是否完整
- 評估程式碼可讀性

## 3. 安全性檢查

- 搜尋硬編碼的憑證或密鑰
- 檢查 SQL 注入風險
- 評估輸入驗證
- 檢視認證和授權邏輯

## 4. 效能考量

- 識別可能的效能瓶頸
- 檢查資料庫查詢效率
- 評估記憶體使用
- 注意 N+1 查詢問題

## 5. 測試覆蓋

- 檢查是否有對應的測試檔案
- 評估測試覆蓋率
- 建議額外需要的測試案例

## 6. 生成審查報告

以下列格式輸出：

### ✅ 優點
- 列出程式碼中的良好實踐

### 🔴 關鍵問題
- 必須修復的問題

### 🟡 建議改進
- 可以改進的地方

### 💡 建議
- 整體建議和下一步

## 注意事項

- 提供建設性反饋
- 附帶具體的程式碼範例
- 優先標註關鍵問題
```

#### 完整範例 2：部署命令

**檔案：** `.claude/commands/deploy.md`

```markdown
---
description: 部署應用到指定環境（staging 或 production）
---

# 部署命令

## 使用方式

```

/deploy `<environment>` [--skip-tests]

```

參數：
- `environment`: staging 或 production
- `--skip-tests`: 可選，跳過測試步驟

## 執行步驟

### 1. 前置檢查

執行以下檢查：
```bash
# 檢查 Git 狀態
git status

# 確認在正確的分支
git branch --show-current

# 確認沒有未提交的變更
git diff --exit-code
```

如果有未提交的變更，警告使用者並停止部署。

### 2. 執行測試（除非使用 --skip-tests）

```bash
npm test
```

如果測試失敗，停止部署並顯示錯誤。

### 3. 建置應用

```bash
npm run build
```

確認建置成功並檢查輸出目錄。

### 4. 部署

根據環境執行部署：

**Staging:**

```bash
npm run deploy:staging
```

**Production:**

```bash
# 額外確認
echo "⚠️  即將部署到生產環境！"
# 等待使用者確認後執行
npm run deploy:production
```

### 5. 驗證部署

部署完成後：

```bash
# 執行健康檢查
curl https://[環境URL]/health

# 檢查應用版本
curl https://[環境URL]/version
```

### 6. 通知

生成部署摘要：

- 部署時間
- 目標環境
- Git 提交雜湊
- 部署狀態
- 健康檢查結果

```

#### 完整範例 3：測試生成命令

**檔案：** `.claude/commands/generate-tests.md`

```markdown
---
description: 為指定的程式碼檔案生成單元測試
---

# 生成測試命令

為給定的程式碼檔案自動生成全面的單元測試。

## 執行步驟

### 1. 分析原始碼

- 讀取目標檔案
- 識別所有匯出的函數和類別
- 分析函數參數和返回類型
- 識別邊界條件和異常情況

### 2. 確定測試框架

檢查專案中使用的測試框架：
- 查看 package.json 中的 dependencies
- 常見框架：Jest, Vitest, Mocha, Pytest, Go testing

### 3. 生成測試檔案

為每個函數/方法生成：
- **正常情況測試**：測試預期輸入的正確行為
- **邊界條件測試**：空值、零值、極限值
- **錯誤處理測試**：無效輸入、異常拋出
- **Mock/Stub**：外部依賴的模擬

### 4. 測試檔案命名

遵循專案慣例：
- JavaScript/TypeScript: `filename.test.ts` 或 `filename.spec.ts`
- Python: `test_filename.py`
- Go: `filename_test.go`
- Java: `FilenameTest.java`

### 5. 範例格式（TypeScript + Jest）

```typescript
import { functionName } from './moduleName';

describe('functionName', () => {
  describe('正常情況', () => {
    it('should return expected result for valid input', () => {
      expect(functionName(validInput)).toBe(expectedOutput);
    });
  });

  describe('邊界條件', () => {
    it('should handle empty input', () => {
      expect(functionName('')).toBe(defaultValue);
    });

    it('should handle null input', () => {
      expect(functionName(null)).toBe(defaultValue);
    });
  });

  describe('錯誤處理', () => {
    it('should throw error for invalid input', () => {
      expect(() => functionName(invalidInput))
        .toThrow('Error message');
    });
  });
});
```

### 6. 輸出位置

將測試檔案放在適當位置：

- 與原始檔同目錄（JavaScript/TypeScript 常見）
- `tests/` 或 `__tests__/` 目錄
- 遵循專案現有結構

```

#### 參數傳遞和使用方式

Commands 可以接收參數，使用者在調用時傳入：

```bash
# 基本使用
/deploy staging

# 帶選項
/deploy production --skip-tests

# 帶多個參數
/generate-tests src/utils/auth.ts --framework jest
```

在命令檔案中，參數會作為使用者輸入的一部分被處理。

#### 最佳實踐

**1. 清晰的描述**

```markdown
---
description: 使用簡短、明確的描述（建議不超過 80 字元）
---
```

**2. 結構化指令**
使用標題、列表、程式碼區塊組織指令：

```markdown
## 步驟 1: 標題

- 具體行動
- 使用 Bash/Read/Grep 工具

### 子步驟

\`\`\`bash
具體命令
\`\`\`
```

**3. 明確的輸出格式**
告訴 Claude 如何呈現結果：

```markdown
## 輸出格式

以表格形式顯示：
| 檔案 | 變更類型 | 行數 |
|------|---------|------|
```

**4. 錯誤處理**
包含失敗情況的處理：

```markdown
## 錯誤處理

如果測試失敗：
1. 顯示失敗的測試
2. 停止後續步驟
3. 建議修復方案
```

**5. 工具使用指引**
明確指定使用哪些工具：

```markdown
- 使用 `Bash` 工具執行 git 命令
- 使用 `Read` 工具讀取配置檔
- 使用 `Grep` 工具搜尋程式碼
```

---

### 3. 專業化代理 (Sub-agents)

#### 位置與配置格式

Sub-agents 存放在 `.claude/agents/` 目錄中，每個代理是一個 JSON 配置檔。

**目錄結構：**

```
.claude/
└── agents/
    ├── code-reviewer.json
    ├── security-auditor.json
    └── test-generator.json
```

#### JSON 配置結構

```json
{
  "name": "agent-name",
  "description": "代理的簡短描述",
  "instructions": "詳細的系統提示詞...",
  "tools": {
    "allowed": ["Read", "Grep", "Glob"],
    "denied": ["Write", "Edit", "Bash"]
  }
}
```

#### 實際範例 1：程式碼審查員

**檔案：** `.claude/agents/code-reviewer.json`

```json
{
  "name": "code-reviewer",
  "description": "專業的程式碼審查專家，提供建設性的審查意見",
  "instructions": "你是一位資深的程式碼審查專家，擁有 10 年以上的軟體開發經驗。\n\n## 審查原則\n\n1. **建設性反饋**：指出問題時，總是提供具體的改進建議和程式碼範例\n2. **優先級分類**：清楚區分必須修復的問題和可選的優化\n3. **最佳實踐**：參考業界標準、設計模式和團隊規範\n4. **教育導向**：解釋「為什麼」這樣做更好，幫助開發者成長\n\n## 審查檢查清單\n\n### 程式碼品質\n- 命名是否清晰、有意義\n- 函數是否遵循單一職責原則\n- 是否避免程式碼重複（DRY 原則）\n- 註解是否適當且有價值\n\n### 錯誤處理\n- 異常處理是否完整\n- 錯誤訊息是否有幫助\n- 邊界條件是否考慮周全\n\n### 效能\n- 是否有明顯的效能瓶頸\n- 資料庫查詢是否高效\n- 演算法複雜度是否合理\n\n### 安全性\n- 輸入是否經過驗證\n- 是否有 SQL 注入風險\n- 敏感資料是否妥善處理\n- 認證和授權是否正確\n\n### 可測試性\n- 程式碼是否容易測試\n- 依賴是否可以 mock\n- 測試覆蓋是否充分\n\n### 可維護性\n- 程式碼結構是否清晰\n- 是否易於理解和修改\n- 文件是否充足\n\n## 輸出格式\n\n### ✅ 優點\n列出程式碼中值得稱讚的地方\n\n### 🔴 必須修復\n嚴重問題，必須在合併前解決\n- 問題描述\n- 影響範圍\n- 修復建議（附程式碼範例）\n\n### 🟡 建議改進\n可以改善但不緊急的地方\n- 改進點\n- 為什麼這樣更好\n- 範例程式碼\n\n### 🔵 可選優化\n進一步優化的可能性\n\n### 📝 總結\n整體評價和主要建議",
  "tools": {
    "allowed": ["Read", "Grep", "Glob"],
    "denied": ["Write", "Edit", "Bash"]
  }
}
```

#### 實際範例 2：安全審計員

**檔案：** `.claude/agents/security-auditor.json`

```json
{
  "name": "security-auditor",
  "description": "安全性審計專家，識別程式碼中的安全漏洞",
  "instructions": "你是一位應用安全專家，專注於識別和修復程式碼中的安全漏洞。\n\n## 安全審計重點\n\n### 1. 注入攻擊\n- **SQL 注入**：檢查資料庫查詢是否使用參數化查詢\n- **NoSQL 注入**：檢查 MongoDB 等查詢\n- **命令注入**：檢查系統命令執行\n- **LDAP/XML 注入**：相關技術的注入風險\n\n### 2. 認證與授權\n- 密碼儲存是否使用適當的雜湊演算法（bcrypt, Argon2）\n- Session 管理是否安全\n- JWT token 是否正確實作\n- 權限檢查是否完整\n- 是否有橫向越權漏洞\n\n### 3. 敏感資料\n- API 金鑰、密碼是否硬編碼\n- 敏感資料是否加密\n- 日誌中是否洩漏敏感資訊\n- 是否正確處理個人資料（GDPR 合規）\n\n### 4. 加密\n- 是否使用過時的加密演算法（MD5, SHA1）\n- HTTPS 是否正確配置\n- 加密金鑰管理是否安全\n\n### 5. 輸入驗證\n- 使用者輸入是否經過驗證和消毒\n- 檔案上傳是否有適當限制\n- XSS 防護是否到位\n- CSRF 防護是否實作\n\n### 6. 第三方依賴\n- 是否使用已知有漏洞的套件版本\n- 依賴是否來自可信來源\n\n### 7. 配置安全\n- 錯誤訊息是否洩漏過多資訊\n- Debug 模式是否在生產環境關閉\n- 安全標頭是否正確設定\n\n## 威脅等級\n\n- 🔴 **嚴重 (Critical)**：可直接導致系統被攻破\n- 🟠 **高危 (High)**：重大安全風險\n- 🟡 **中危 (Medium)**：需要關注的安全問題\n- 🔵 **低危 (Low)**：輕微的安全改進點\n- ⚪ **資訊 (Info)**：安全建議\n\n## 輸出格式\n\n### 🔒 安全審計報告\n\n#### 摘要\n- 審計範圍\n- 發現的問題數量（按嚴重程度分類）\n- 整體安全評分\n\n#### 詳細發現\n\n對每個問題：\n- **嚴重程度**：🔴/🟠/🟡/🔵\n- **問題描述**：清楚說明安全風險\n- **受影響檔案**：具體位置\n- **攻擊場景**：如何被利用\n- **修復建議**：具體的程式碼修改（附範例）\n- **參考資料**：OWASP、CVE 等\n\n#### 優先修復建議\n按嚴重程度排序的修復清單",
  "tools": {
    "allowed": ["Read", "Grep", "Glob"],
    "denied": ["Write", "Edit", "Bash"]
  }
}
```

#### 何時使用 Sub-agents

**使用 Sub-agents 的情況：**

- 需要特定領域的專業知識（如安全審計）
- 需要一致的輸出格式
- 需要限制工具使用權限
- 有明確的審查或分析流程

**使用 Commands 的情況：**

- 執行具體的操作流程
- 需要使用多種工具
- 一次性的自動化任務

#### 與 Skills 的區別

| 特性               | Sub-agents       | Skills               |
| ------------------ | ---------------- | -------------------- |
| **定位**     | 專業化的 AI 人格 | 可重複使用的能力模組 |
| **配置**     | JSON 檔案        | 目錄 + SKILL.md      |
| **調用方式** | 明確指定代理     | 自動啟用             |
| **工具限制** | 可以限制工具使用 | 由 SKILL.md 定義     |
| **使用場景** | 特定角色的任務   | 通用的技術能力       |

---

### 4. 技能模組 (Skills)

#### 位置與結構

Skills 存放在 `.claude/skills/` 目錄中，每個技能是一個子目錄。

**目錄結構：**

```
.claude/
└── skills/
    ├── python-testing/
    │   └── SKILL.md
    └── docker-management/
        └── SKILL.md
```

#### SKILL.md 格式和 YAML frontmatter

每個技能目錄包含一個 `SKILL.md` 檔案：

```markdown
---
skill: skill-name
description: 技能的簡短描述
allowed-tools:
  - Read
  - Bash
  - Grep
---

# 技能的詳細說明

技能提供的能力和使用方式...
```

#### 實際範例 1：Python 測試技能

**檔案：** `.claude/skills/python-testing/SKILL.md`

```markdown
---
skill: python-testing
description: Python 測試自動化（pytest、unittest、coverage）
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
---

# Python Testing Skill

提供全面的 Python 測試自動化能力，包括測試生成、執行和覆蓋率分析。

## 核心能力

### 1. 測試生成
為 Python 函數和類別自動生成 pytest 或 unittest 測試

**範例：**
給定函數：
```python
def calculate_discount(price: float, percentage: float) -> float:
    if price < 0 or percentage < 0 or percentage > 100:
        raise ValueError("Invalid input")
    return price * (1 - percentage / 100)
```

生成測試：

```python
import pytest
from module import calculate_discount

class TestCalculateDiscount:
    def test_normal_discount(self):
        assert calculate_discount(100, 10) == 90
        assert calculate_discount(50, 20) == 40

    def test_zero_discount(self):
        assert calculate_discount(100, 0) == 100

    def test_full_discount(self):
        assert calculate_discount(100, 100) == 0

    def test_invalid_price(self):
        with pytest.raises(ValueError):
            calculate_discount(-10, 10)

    def test_invalid_percentage(self):
        with pytest.raises(ValueError):
            calculate_discount(100, -5)
        with pytest.raises(ValueError):
            calculate_discount(100, 150)
```

### 2. 測試執行

執行測試套件並分析結果

```bash
# 執行所有測試
pytest

# 執行特定檔案
pytest tests/test_module.py

# 執行特定測試
pytest tests/test_module.py::TestClass::test_method

# 顯示詳細輸出
pytest -v

# 顯示 print 輸出
pytest -s
```

### 3. 覆蓋率分析

生成和分析測試覆蓋率報告

```bash
# 執行測試並生成覆蓋率報告
pytest --cov=src tests/

# 生成 HTML 報告
pytest --cov=src --cov-report=html tests/

# 顯示缺失覆蓋的行
pytest --cov=src --cov-report=term-missing tests/
```

### 4. Mock 和 Fixture

生成測試所需的 mock 和 fixture

**Fixture 範例：**

```python
import pytest

@pytest.fixture
def sample_user():
    return {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com"
    }

@pytest.fixture
def database_connection():
    # Setup
    conn = create_connection()
    yield conn
    # Teardown
    conn.close()
```

**Mock 範例：**

```python
from unittest.mock import Mock, patch

def test_api_call():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"key": "value"}

        result = my_function()
        assert result == expected_value
```

### 5. 參數化測試

生成參數化測試以測試多種情況

```python
@pytest.mark.parametrize("price,percentage,expected", [
    (100, 10, 90),
    (50, 20, 40),
    (200, 50, 100),
    (75, 25, 56.25),
])
def test_calculate_discount_parametrized(price, percentage, expected):
    assert calculate_discount(price, percentage) == expected
```

### 自動啟用條件

此技能會在以下情況自動啟用：

- 偵測到 Python 測試檔案（`test_*.py`, `*_test.py`）
- 偵測到 `pytest.ini`, `setup.cfg`, `pyproject.toml` 中的 pytest 配置
- 使用者提到「測試」、「pytest」、「unittest」
- 使用者要求生成測試或檢查覆蓋率

### 支援的測試框架

- pytest（主要支援）
- unittest
- nose2
- doctest

### 最佳實踐

1. **測試檔案命名**：`test_*.py` 或 `*_test.py`
2. **測試類別命名**：`Test*`
3. **測試方法命名**：`test_*`
4. **每個測試一個斷言**：保持測試簡單
5. **使用 fixture**：管理測試資料和設定
6. **參數化測試**：測試多種輸入組合

```

#### 實際範例 2：Docker 管理技能

**檔案：** `.claude/skills/docker-management/SKILL.md`

```markdown
---
skill: docker-management
description: Docker 容器和映像管理
allowed-tools:
  - Bash
  - Read
  - Grep
---

# Docker Management Skill

提供全面的 Docker 容器和映像管理能力。

## 核心能力

### 1. 容器生命週期管理

**啟動容器：**
```bash
# 基本啟動
docker run -d --name container-name image-name

# 帶埠號映射
docker run -d -p 8080:80 --name web nginx

# 帶環境變數
docker run -d -e DB_HOST=localhost -e DB_PORT=5432 app

# 帶卷掛載
docker run -d -v /host/path:/container/path app
```

**查看容器：**

```bash
# 列出執行中的容器
docker ps

# 列出所有容器（包括停止的）
docker ps -a

# 查看容器詳細資訊
docker inspect container-name

# 查看容器日誌
docker logs container-name
docker logs -f container-name  # 即時追蹤

# 查看容器資源使用
docker stats container-name
```

**停止和移除容器：**

```bash
# 停止容器
docker stop container-name

# 強制停止容器
docker kill container-name

# 移除容器
docker rm container-name

# 停止並移除
docker rm -f container-name
```

#### 2. 映像管理

**建立映像：**

```bash
# 從 Dockerfile 建立
docker build -t image-name:tag .

# 指定 Dockerfile
docker build -f Dockerfile.prod -t app:prod .

# 不使用快取
docker build --no-cache -t image-name .
```

**管理映像：**

```bash
# 列出映像
docker images

# 移除映像
docker rmi image-name:tag

# 清理未使用的映像
docker image prune

# 查看映像歷史
docker history image-name
```

**推送和拉取：**

```bash
# 拉取映像
docker pull nginx:latest

# 推送到 registry
docker tag local-image:tag registry/image:tag
docker push registry/image:tag
```

### 3. Docker Compose 管理

**基本操作：**

```bash
# 啟動服務
docker-compose up -d

# 停止服務
docker-compose down

# 查看日誌
docker-compose logs -f service-name

# 重新建立服務
docker-compose up -d --build

# 擴展服務
docker-compose up -d --scale web=3
```

**範例 docker-compose.yml：**

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=production
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - db-data:/var/lib/postgresql/data

volumes:
  db-data:
```

### 4. 網路管理

```bash
# 建立網路
docker network create app-network

# 列出網路
docker network ls

# 連接容器到網路
docker network connect app-network container-name

# 檢查網路
docker network inspect app-network
```

### 5. 卷管理

```bash
# 建立卷
docker volume create volume-name

# 列出卷
docker volume ls

# 查看卷詳情
docker volume inspect volume-name

# 移除未使用的卷
docker volume prune
```

### 6. 常見 Dockerfile 範本

**Node.js 應用：**

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

**Python 應用：**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

## 自動啟用條件

此技能會在以下情況自動啟用：

- 偵測到 `Dockerfile` 或 `docker-compose.yml`
- 使用者提到「docker」、「容器」、「container」
- 需要容器化應用

## 疑難排解

**容器無法啟動：**

```bash
# 查看錯誤日誌
docker logs container-name

# 檢查容器配置
docker inspect container-name
```

**埠號衝突：**

```bash
# 查看佔用的埠號
lsof -i :8080

# 使用不同的主機埠號
docker run -p 8081:80 app
```

**映像建立失敗：**

```bash
# 檢查 Dockerfile 語法
docker build --no-cache -t test .

# 使用 --progress=plain 查看詳細輸出
docker build --progress=plain -t test .
```

```

#### allowed-tools 配置

在 YAML frontmatter 中指定技能可以使用的工具：

```yaml
---
allowed-tools:
  - Read      # 讀取檔案
  - Write     # 寫入新檔案
  - Edit      # 編輯現有檔案
  - Bash      # 執行命令
  - Grep      # 搜尋內容
  - Glob      # 搜尋檔案
---
```

#### 技能載入機制

Skills 會在以下情況自動啟用：

1. **檔案偵測**：偵測到相關檔案（如 Dockerfile、pytest.ini）
2. **關鍵字觸發**：使用者提到相關關鍵字
3. **明確請求**：使用者明確要求使用某個技能

---

### 5. MCP 伺服器整合

#### MCP 作為擴展機制的角色

Model Context Protocol (MCP) 是 Claude Code 連接外部服務和資料源的標準協定。透過 MCP，Claude 可以：

- 存取檔案系統
- 查詢資料庫
- 呼叫 GitHub API
- 執行網路搜尋
- 連接自訂服務

#### 實際可用的 MCP 伺服器

**官方 MCP 伺服器：**

- `@modelcontextprotocol/server-filesystem` - 檔案系統存取
- `@modelcontextprotocol/server-github` - GitHub 整合
- `@modelcontextprotocol/server-postgres` - PostgreSQL 資料庫
- `@modelcontextprotocol/server-sqlite` - SQLite 資料庫
- `@modelcontextprotocol/server-brave-search` - Brave 搜尋引擎

#### 配置範例

**檔案：** `.claude/settings.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://user:password@localhost:5432/database"
      ]
    },
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key"
      }
    }
  }
}
```

#### 開發自訂 MCP 伺服器

**基本結構：**

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-custom-server",
  version: "1.0.0",
}, {
  capabilities: {
    tools: {},
  },
});

// 定義工具
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "my_tool",
        description: "工具描述",
        inputSchema: {
          type: "object",
          properties: {
            param: {
              type: "string",
              description: "參數描述"
            }
          },
          required: ["param"]
        }
      }
    ]
  };
});

// 實作工具
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "my_tool") {
    // 執行工具邏輯
    return {
      content: [
        {
          type: "text",
          text: "工具執行結果"
        }
      ]
    };
  }
});

// 啟動伺服器
const transport = new StdioServerTransport();
await server.connect(transport);
```

**在 settings.json 中配置：**

```json
{
  "mcpServers": {
    "my-custom-server": {
      "command": "node",
      "args": ["/path/to/my-server/index.js"]
    }
  }
}
```

#### 與其他擴展機制的整合

**在 Commands 中使用 MCP：**

```markdown
---
description: 查詢 GitHub Issues
---

# GitHub Issues 查詢

使用 GitHub MCP 伺服器查詢 issues：

1. 列出開放的 issues
2. 按優先級排序
3. 顯示最近 10 個 issues
```

**在 Sub-agents 中限制 MCP 工具：**

```json
{
  "name": "readonly-agent",
  "tools": {
    "allowed": ["mcp__github__search_repositories"],
    "denied": ["mcp__github__create_issue"]
  }
}
```

---

### 6. 目錄結構與最佳實踐

#### 完整的 .claude/ 目錄結構

```
.claude/
├── settings.json         # MCP 伺服器和全域配置
├── commands/            # 自訂斜線命令
│   ├── deploy.md
│   ├── review-pr.md
│   ├── generate-tests.md
│   └── refactor.md
├── agents/              # 專業化代理
│   ├── code-reviewer.json
│   ├── security-auditor.json
│   └── performance-analyzer.json
└── skills/              # 技能模組
    ├── python-testing/
    │   └── SKILL.md
    ├── docker-management/
    │   └── SKILL.md
    └── api-development/
        └── SKILL.md
```

#### 個人 vs 專案級配置

**個人配置：** `~/.config/claude/`

```
~/.config/claude/
├── settings.json        # 個人的 MCP 配置
└── commands/            # 個人常用命令
    ├── daily-standup.md
    └── time-tracker.md
```

**專案配置：** `<project>/.claude/`

```
project/.claude/
├── settings.json        # 專案特定的 MCP 配置
├── commands/            # 專案特定命令
│   ├── deploy.md
│   └── migrate-db.md
└── agents/              # 專案特定代理
    └── api-reviewer.json
```

**優先級：** 專案配置 > 個人配置

#### 版本控制建議

**.gitignore 配置：**

```
# 包含共享的命令和代理
.claude/commands/
.claude/agents/
.claude/skills/

# 排除包含敏感資訊的配置
.claude/settings.json
```

**範例 .claude/settings.json.example：**

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "YOUR_GITHUB_TOKEN_HERE"
      }
    }
  }
}
```

#### 團隊協作

**1. 建立共享的命令庫**

```bash
# 團隊倉庫
team-claude-commands/
├── README.md
├── commands/
│   ├── deploy-staging.md
│   ├── deploy-production.md
│   └── run-e2e-tests.md
└── agents/
    └── code-reviewer.json
```

**2. 設定專案的 .claude/ 目錄**

```bash
cd your-project
mkdir -p .claude/commands .claude/agents
ln -s /path/to/team-claude-commands/commands/* .claude/commands/
ln -s /path/to/team-claude-commands/agents/* .claude/agents/
```

**3. 文件化團隊規範**

```markdown
# 團隊 Claude Code 規範

## 命令命名
- 使用 kebab-case: `deploy-staging.md`
- 描述要具體且可操作

## 代理配置
- 所有代理必須限制工具使用
- 提供清楚的輸出格式

## 版本控制
- 提交所有 .claude/commands/ 和 .claude/agents/
- 不要提交 settings.json（包含密鑰）
```

---

### 7. 實際應用範例

#### 組合使用多種擴展機制

**場景：** 建立一個完整的 Code Review 工作流程

**1. 自訂命令：** `.claude/commands/full-review.md`

```markdown
---
description: 執行完整的程式碼審查流程
---

# 完整程式碼審查流程

## 步驟 1: 使用安全審計代理檢查安全問題

請切換到 security-auditor 代理，審查當前的程式碼變更。

## 步驟 2: 使用程式碼審查代理檢查程式碼品質

請切換到 code-reviewer 代理，審查程式碼品質。

## 步驟 3: 使用測試技能檢查測試覆蓋

使用 python-testing 或其他相關測試技能，檢查：
- 測試覆蓋率
- 是否有新的測試
- 測試品質

## 步驟 4: 生成綜合報告

整合以上三個方面的審查結果，生成綜合報告：

### 審查摘要
- 安全問題：X 個
- 程式碼品質問題：Y 個
- 測試覆蓋率：Z%

### 必須修復
列出所有關鍵問題

### 建議改進
列出所有建議改進

### 整體建議
是否建議合併
```

**2. 安全審計代理：** `.claude/agents/security-auditor.json`

```json
{
  "name": "security-auditor",
  "description": "安全性審計專家",
  "instructions": "...",
  "tools": {
    "allowed": ["Read", "Grep", "Glob"],
    "denied": ["Write", "Edit", "Bash"]
  }
}
```

**3. 測試技能：** `.claude/skills/python-testing/SKILL.md`

```markdown
---
skill: python-testing
allowed-tools: ["Read", "Bash", "Grep"]
---
提供測試覆蓋率分析...
```

**4. MCP 配置：** `.claude/settings.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

#### 典型工作流程自動化案例

**場景：** 自動化 CI/CD 流程

**命令：** `.claude/commands/ci-cd-pipeline.md`

```markdown
---
description: 執行完整的 CI/CD 流程
---

# CI/CD Pipeline

## 1. 準備階段

```bash
# 檢查 Git 狀態
git status

# 確保在正確的分支
git branch --show-current

# 拉取最新變更
git pull origin main
```

## 2. 測試階段

執行以下測試：

### 單元測試

```bash
npm test
# 或
pytest tests/
```

### 整合測試

```bash
npm run test:integration
```

### E2E 測試

```bash
npm run test:e2e
```

如果任何測試失敗，停止流程並報告錯誤。

## 3. 建置階段

```bash
npm run build
```

驗證建置輸出是否正確。

## 4. 安全掃描

使用 security-auditor 代理掃描程式碼。

## 5. 部署階段

根據目標環境部署：

### Staging

```bash
npm run deploy:staging
```

### Production（需確認）

```bash
npm run deploy:production
```

## 6. 驗證階段

部署後執行健康檢查：

```bash
curl https://[環境URL]/health
curl https://[環境URL]/api/status
```

## 7. 通知

生成部署報告並通知團隊：

- 部署時間
- 版本號
- 測試結果
- 健康檢查狀態

```

---

### 8. 常見問題與疑難排解

#### 擴展功能無法載入

**問題：** 自訂命令沒有出現

**解決方案：**
1. 確認檔案位置正確：`.claude/commands/your-command.md`
2. 檢查 YAML frontmatter 格式：
   ```markdown
   ---
   description: 命令描述
   ---
```

3. 重新載入 Claude Code 或重啟會話

**問題：** Sub-agent 無法使用

**解決方案：**

1. 確認 JSON 格式正確：使用 `jq` 或 JSON 驗證器
   ```bash
   cat .claude/agents/agent-name.json | jq .
   ```
2. 檢查必要欄位：`name`, `description`, `instructions`
3. 確認檔案位置：`.claude/agents/agent-name.json`

**問題：** Skill 沒有自動啟用

**解決方案：**

1. 確認 `SKILL.md` 檔案存在於技能目錄中
2. 檢查 YAML frontmatter 中的 `skill` 欄位
3. 確認觸發條件（檔案類型、關鍵字）符合

#### 配置衝突

**問題：** 個人配置與專案配置衝突

**解決方案：**

- 專案配置優先於個人配置
- 在專案的 `.claude/settings.json` 中覆蓋個人配置
- 使用環境變數區分不同環境：
  ```json
  {
    "mcpServers": {
      "database": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env": {
          "DATABASE_URL": "${PROJECT_DATABASE_URL}"
        }
      }
    }
  }
  ```

**問題：** MCP 伺服器配置衝突

**解決方案：**

- 使用不同的伺服器名稱
- 在專案配置中明確禁用個人配置的伺服器

#### 權限問題

**問題：** MCP 伺服器無法啟動（權限錯誤）

**解決方案：**

```bash
# 確認 npx 可執行
which npx

# 確認 Node.js 版本
node --version  # 需要 >= 18

# 手動測試 MCP 伺服器
npx -y @modelcontextprotocol/server-filesystem /path/to/directory
```

**問題：** 無法存取檔案系統

**解決方案：**

- 確認 MCP filesystem 伺服器配置的路徑有讀取權限
- 使用絕對路徑而非相對路徑
- 檢查 macOS 的隱私設定（系統偏好設定 > 安全性與隱私 > 檔案和資料夾）

**問題：** 環境變數未設定

**解決方案：**

```bash
# 設定環境變數
export GITHUB_TOKEN=your_token

# 或在 .env 檔案中設定（需要 Claude Code 支援）
echo "GITHUB_TOKEN=your_token" >> .env

# 驗證環境變數
echo $GITHUB_TOKEN
```

---

### 總結

Claude Code 提供了四種強大的擴展機制，讓您可以根據需求客製化 AI 助手的行為：

✅ **自訂命令（Commands）**

- 位置：`.claude/commands/`
- 用途：定義具體的操作流程
- 格式：Markdown 檔案

✅ **專業化代理（Sub-agents）**

- 位置：`.claude/agents/`
- 用途：建立特定領域的專家
- 格式：JSON 配置檔

✅ **技能模組（Skills）**

- 位置：`.claude/skills/`
- 用途：提供可重複使用的能力
- 格式：目錄 + SKILL.md

✅ **MCP 伺服器整合**

- 位置：`.claude/settings.json`
- 用途：連接外部服務和資料源
- 格式：JSON 配置檔

**下一步：**

- 探索 [Sub-agents](#6-子代理-sub-agents) 深入了解專業化代理
- 學習 [Hooks](#7-hooks) 實作自動化工作流程
- 研究 [MCP Integration](#9-mcp-integration) 連接更多外部服務

**實踐建議：**

1. 從建立簡單的自訂命令開始
2. 嘗試配置一個 Sub-agent
3. 設定常用的 MCP 伺服器（如 GitHub）
4. 建立團隊共享的配置倉庫

---

## 6. 子代理 (Sub-agents)

### 📋 學習摘要

**學習目標：** 建立專業化的 AI 助手來處理特定類型的任務

**核心內容：**

- Sub-agents 的定義與工作原理
- 20+ 種子代理類型（開發、測試、安全、DevOps 等）
- 配置檔案結構與系統提示撰寫
- 6 個詳細的實際範例
- 最佳實踐與疑難排解

**關鍵技能：**

- ✅ 建立專業化子代理
- ✅ 撰寫有效的系統提示
- ✅ 管理工具權限
- ✅ 明確和自動調用子代理

**常用 Sub-agents：**

- 程式碼審查員 (Code Reviewer)
- 測試套件生成器 (Test Suite Generator)
- 安全審計員 (Security Auditor)
- 除錯專家 (Debugger)
- 文件撰寫者 (Documentation Writer)

**預計學習時間：** 2-3 小時

**詳細教學：** [→ 查看完整教學內容](#sub-agents-詳細內容)

### 📖 完整教學內容

### 目錄

1. [什麼是 Sub-agents](#什麼是-sub-agents)
2. [Sub-agents 的工作原理](#sub-agents-的工作原理)
3. [Sub-agents 類型總覽](#sub-agents-類型總覽)
4. [配置檔案結構](#配置檔案結構)
5. [系統提示撰寫指南](#系統提示撰寫指南)
6. [實際範例](#sub-agents-實際範例)
7. [工具權限管理](#工具權限管理)
8. [調用方式](#調用方式)
9. [最佳實踐](#sub-agents-最佳實踐)
10. [疑難排解](#sub-agents-疑難排解)

---

### 什麼是 Sub-agents

**Sub-agents**（子代理）是 Claude Code 中專門化的 AI 助手，每個都針對特定類型的任務進行了優化。它們允許您將複雜的工作流程分解為專業化的元件，每個元件都有自己的系統提示、工具權限和行為模式。

#### 核心概念

Sub-agents 是基於相同的 Claude 模型，但通過不同的系統提示和配置來實現專業化：

- **專業化**：每個 sub-agent 都針對特定領域（測試、安全、文件等）進行了優化
- **隔離性**：Sub-agents 在獨立的對話線程中運行，不會干擾主對話
- **可重用**：一次定義，可在多個專案中重複使用
- **可組合**：多個 sub-agents 可以協同工作來完成複雜任務

#### Sub-agents vs 主代理

| 特性                 | 主代理 (Main Agent)   | Sub-agent          |
| -------------------- | --------------------- | ------------------ |
| **範圍**       | 通用任務處理          | 特定領域專精       |
| **系統提示**   | 標準 Claude Code 提示 | 自訂專業化提示     |
| **工具權限**   | 完整訪問權限          | 可限制的子集       |
| **對話上下文** | 主對話線程            | 獨立線程           |
| **使用時機**   | 一般開發任務          | 需要專業知識的任務 |

---

### Sub-agents 的工作原理

#### 執行流程

```
1. 使用者請求 → 2. 主代理分析 → 3. 決定調用 sub-agent
                                         ↓
                    ← 5. 返回結果 ← 4. Sub-agent 執行任務
                         ↓
                    6. 主代理整合結果 → 7. 回應使用者
```

#### 詳細步驟

1. **任務識別**

   - 主代理分析使用者請求
   - 識別是否需要專業化處理
2. **Sub-agent 選擇**

   - 根據任務類型選擇合適的 sub-agent
   - 可以是明確調用或自動觸發
3. **上下文準備**

   - 收集必要的檔案和資訊
   - 準備傳遞給 sub-agent 的上下文
4. **獨立執行**

   - Sub-agent 在獨立線程中運行
   - 使用自己的系統提示和工具集
   - 不受主對話歷史干擾
5. **結果返回**

   - Sub-agent 完成任務並返回結果
   - 結果包含在獨立的對話記錄中
6. **整合與呈現**

   - 主代理接收並整合 sub-agent 的輸出
   - 將結果以統一格式呈現給使用者

#### 技術架構

```
.claude/
├── agents/
│   ├── code-reviewer.json      # 程式碼審查員配置
│   ├── test-generator.json     # 測試生成器配置
│   ├── security-auditor.json   # 安全審計員配置
│   └── doc-writer.json         # 文件撰寫者配置
└── config.json                 # 主配置（可選）
```

---

### Sub-agents 類型總覽

#### 1. 開發相關 Sub-agents

| Sub-agent                        | 用途       | 主要功能                       |
| -------------------------------- | ---------- | ------------------------------ |
| **Code Reviewer**          | 程式碼審查 | 檢查程式碼品質、風格、最佳實踐 |
| **Refactoring Specialist** | 重構專家   | 改善程式碼結構、消除重複       |
| **Debugger**               | 除錯專家   | 找出和修復 bugs                |
| **Performance Optimizer**  | 效能優化   | 分析和改善效能瓶頸             |
| **Architecture Reviewer**  | 架構審查   | 評估系統設計和架構決策         |

#### 2. 測試相關 Sub-agents

| Sub-agent                          | 用途         | 主要功能                 |
| ---------------------------------- | ------------ | ------------------------ |
| **Test Suite Generator**     | 測試套件生成 | 創建完整的測試套件       |
| **Unit Test Writer**         | 單元測試撰寫 | 為函數和類別編寫單元測試 |
| **Integration Test Creator** | 整合測試創建 | 編寫整合測試             |
| **E2E Test Specialist**      | E2E 測試專家 | 創建端到端測試場景       |
| **QA Specialist**            | QA 專家      | 測試計劃、測試案例設計   |

#### 3. 安全相關 Sub-agents

| Sub-agent                       | 用途       | 主要功能               |
| ------------------------------- | ---------- | ---------------------- |
| **Security Auditor**      | 安全審計   | 識別安全漏洞           |
| **Vulnerability Scanner** | 漏洞掃描   | 掃描已知漏洞           |
| **Crypto Specialist**     | 加密專家   | 審查加密實作           |
| **Auth Expert**           | 認證專家   | 審查認證和授權邏輯     |
| **OWASP Checker**         | OWASP 檢查 | 根據 OWASP Top 10 檢查 |

#### 4. DevOps 相關 Sub-agents

| Sub-agent                        | 用途        | 主要功能              |
| -------------------------------- | ----------- | --------------------- |
| **Deployment Specialist**  | 部署專家    | 管理部署流程          |
| **CI/CD Expert**           | CI/CD 專家  | 配置和優化 CI/CD 管道 |
| **Docker Specialist**      | Docker 專家 | 創建和優化容器化方案  |
| **Infrastructure as Code** | IaC 專家    | 管理基礎設施程式碼    |
| **Monitoring Setup**       | 監控設置    | 配置監控和告警        |

#### 5. 文件相關 Sub-agents

| Sub-agent                      | 用途         | 主要功能           |
| ------------------------------ | ------------ | ------------------ |
| **Documentation Writer** | 文件撰寫     | 創建完整的技術文件 |
| **API Doc Generator**    | API 文件生成 | 生成 API 文件      |
| **README Creator**       | README 創建  | 編寫專業的 README  |
| **Tutorial Writer**      | 教學撰寫     | 創建教學和指南     |
| **Comment Improver**     | 註解改善     | 改善程式碼註解品質 |

#### 6. 資料庫相關 Sub-agents

| Sub-agent                  | 用途        | 主要功能             |
| -------------------------- | ----------- | -------------------- |
| **SQL Optimizer**    | SQL 優化    | 優化資料庫查詢       |
| **Schema Designer**  | Schema 設計 | 設計資料庫 schema    |
| **Migration Expert** | 遷移專家    | 創建和管理資料庫遷移 |
| **Query Analyzer**   | 查詢分析    | 分析查詢效能         |

---

### 配置檔案結構

#### 基本結構

每個 sub-agent 都由一個 JSON 配置檔定義，位於 `.claude/agents/` 目錄中。

```json
{
  "name": "sub-agent-name",
  "description": "簡短描述這個 sub-agent 的用途",
  "systemPrompt": "詳細的系統提示，定義 sub-agent 的行為和專業知識",
  "tools": {
    "allow": ["tool1", "tool2"],
    "deny": ["tool3", "tool4"]
  },
  "autoTrigger": {
    "patterns": ["關鍵字或模式"],
    "fileTypes": [".js", ".ts"]
  }
}
```

#### 配置欄位詳解

| 欄位             | 必填 | 說明                           |
| ---------------- | ---- | ------------------------------ |
| `name`         | ✅   | Sub-agent 的唯一識別名稱       |
| `description`  | ✅   | 簡短描述（用於選擇和文件）     |
| `systemPrompt` | ✅   | 定義行為的系統提示             |
| `tools.allow`  | ❌   | 允許使用的工具列表             |
| `tools.deny`   | ❌   | 禁止使用的工具列表             |
| `autoTrigger`  | ❌   | 自動觸發條件                   |
| `model`        | ❌   | 指定模型（預設使用主代理模型） |
| `temperature`  | ❌   | 控制輸出隨機性（0.0-1.0）      |

#### 檔案命名規範

```
.claude/agents/
├── code-reviewer.json          # kebab-case，描述性名稱
├── test-suite-generator.json   # 多個單字用破折號連接
├── security-auditor.json       # 清楚表明用途
└── api-doc-writer.json         # 簡潔但具體
```

**最佳實踐：**

- 使用 kebab-case 命名
- 名稱應清楚描述 sub-agent 的用途
- 避免過於通用的名稱（如 "helper.json"）
- 保持一致的命名模式

---

### 系統提示撰寫指南

#### 系統提示結構

一個有效的系統提示應包含以下部分：

```markdown
# [Sub-agent 名稱]

## 角色定義
你是一個專業的 [領域] 專家...

## 主要職責
1. 職責一
2. 職責二
3. 職責三

## 專業知識
- 專業領域一
- 專業領域二

## 工作流程
1. 步驟一
2. 步驟二
3. 步驟三

## 輸出格式
[定義期望的輸出結構]

## 限制與注意事項
- 不應該做的事情
- 需要特別注意的地方
```

#### 撰寫原則

1. **明確性**

   - 清楚定義角色和職責
   - 使用具體的指示而非模糊的描述
   - 提供明確的成功標準
2. **專業性**

   - 展現領域專業知識
   - 使用專業術語（適當時）
   - 參考行業標準和最佳實踐
3. **結構化**

   - 組織成邏輯區段
   - 使用編號列表表示步驟
   - 清楚的標題和子標題
4. **可操作性**

   - 提供具體的檢查清單
   - 定義明確的工作流程
   - 給予可衡量的標準

#### 範例對比

**❌ 不好的系統提示：**

```markdown
你是一個程式碼審查員。請審查程式碼並提供建議。
```

**✅ 好的系統提示：**

```markdown
# 資深程式碼審查員

## 角色定義
你是一位擁有 10+ 年經驗的資深軟體工程師，專精於程式碼品質審查和最佳實踐指導。

## 審查重點
1. **程式碼品質**
   - 可讀性和維護性
   - 命名規範和一致性
   - 程式碼複雜度和結構

2. **最佳實踐**
   - 設計模式應用
   - SOLID 原則遵循
   - DRY 原則實踐

3. **潛在問題**
   - 效能瓶頸
   - 安全漏洞
   - 邊界案例處理

## 審查流程
1. 快速瀏覽整體結構
2. 詳細檢查每個變更
3. 識別模式和反模式
4. 提供具體改善建議

## 輸出格式
### 總體評估
[0-10 分評分和總結]

### 主要發現
- 🔴 嚴重問題（必須修復）
- 🟡 建議改善（應該考慮）
- 🟢 做得好的地方（值得保持）

### 詳細評論
[逐項具體說明]

## 原則
- 建設性和尊重的語氣
- 提供程式碼範例
- 解釋「為什麼」而不只是「什麼」
- 認可優秀的程式碼
```

---

### Sub-agents 實際範例

#### 範例 1：程式碼審查員 (Code Reviewer)

**檔案：** `.claude/agents/code-reviewer.json`

```json
{
  "name": "code-reviewer",
  "description": "資深程式碼審查員，專注於程式碼品質、最佳實踐和潛在問題識別",
  "systemPrompt": "# 資深程式碼審查員\n\n## 角色定義\n你是一位擁有 10+ 年經驗的資深軟體工程師，專精於程式碼品質審查和最佳實踐指導。你的審查深入、建設性且實用。\n\n## 審查重點\n\n### 1. 程式碼品質\n- **可讀性**：程式碼是否清晰易懂？\n- **維護性**：未來開發者能否輕鬆修改？\n- **命名**：變數、函數、類別名稱是否有意義？\n- **結構**：程式碼組織是否合理？\n- **複雜度**：是否過於複雜或可以簡化？\n\n### 2. 最佳實踐\n- **設計模式**：是否適當使用設計模式？\n- **SOLID 原則**：是否遵循單一職責、開放封閉等原則？\n- **DRY 原則**：是否有重複程式碼？\n- **錯誤處理**：是否妥善處理錯誤和邊界案例？\n- **測試覆蓋**：是否有足夠的測試？\n\n### 3. 潛在問題\n- **效能**：是否有效能瓶頸？\n- **安全性**：是否存在安全漏洞？\n- **記憶體**：是否有記憶體洩漏風險？\n- **並發**：是否有競態條件？\n- **相容性**：是否考慮向後相容？\n\n## 審查流程\n\n1. **整體檢視**\n   - 理解變更的目的和範圍\n   - 檢視檔案結構和組織\n\n2. **詳細審查**\n   - 逐行檢查程式碼變更\n   - 識別模式和反模式\n   - 檢查測試覆蓋\n\n3. **評估影響**\n   - 考慮對現有系統的影響\n   - 評估技術債務\n\n4. **提供建議**\n   - 具體的改善方案\n   - 程式碼範例\n   - 優先級排序\n\n## 輸出格式\n\n### 📊 總體評估\n**程式碼品質評分：** [0-10]/10\n**簡短總結：** [一段話概述]\n\n### 🔍 主要發現\n\n#### 🔴 嚴重問題（必須修復）\n- [具體問題描述]\n  ```[語言]\n  // 問題程式碼\n  ```\n  **建議修改：**\n  ```[語言]\n  // 改善後的程式碼\n  ```\n\n#### 🟡 建議改善（應該考慮）\n- [建議項目]\n\n#### 🟢 優點（做得好）\n- [稱讚具體的好做法]\n\n### 📝 詳細評論\n\n#### [檔案名稱]\n**第 [行號] 行：** [具體評論]\n\n### 💡 總結建議\n1. [優先改善項目]\n2. [次要改善項目]\n3. [長期改善方向]\n\n## 原則\n\n- ✅ 建設性和尊重的語氣\n- ✅ 提供具體程式碼範例\n- ✅ 解釋「為什麼」而不只是「什麼」\n- ✅ 認可優秀的程式碼\n- ✅ 考慮上下文和專案需求\n- ❌ 不使用責備或批評的語氣\n- ❌ 不提供模糊的建議\n- ❌ 不忽視好的實踐",
  "tools": {
    "allow": [
      "Read",
      "Grep",
      "Glob",
      "Bash"
    ],
    "deny": [
      "Edit",
      "Write",
      "NotebookEdit"
    ]
  },
  "autoTrigger": {
    "patterns": [
      "審查",
      "review",
      "程式碼品質",
      "code quality"
    ]
  },
  "temperature": 0.3
}
```

**使用場景：**

```
使用者：請審查 src/auth/login.js 的程式碼品質

主代理 → 調用 code-reviewer sub-agent
       ↓
code-reviewer 讀取檔案 → 進行深度分析 → 提供結構化審查報告
       ↓
主代理整合結果 → 向使用者呈現
```

**預期輸出：**

```markdown
📊 總體評估
程式碼品質評分： 7/10
簡短總結：整體結構良好，但在錯誤處理和安全性方面有改善空間。

🔍 主要發現

🔴 嚴重問題（必須修復）

1. 密碼明文儲存風險
   行 45-47：直接比對密碼而未使用 hash
   ```javascript
   if (user.password === inputPassword) {
     return generateToken(user);
   }
```

   建議修改：

```javascript
   const isValid = await bcrypt.compare(inputPassword, user.passwordHash);
   if (isValid) {
     return generateToken(user);
   }
```

🟡 建議改善（應該考慮）

1. 錯誤訊息過於詳細
   行 52：錯誤訊息洩漏使用者是否存在
   建議使用通用錯誤訊息
2. 缺少 rate limiting
   建議加入登入嘗試限制

🟢 優點（做得好）

1. 使用 JWT token 進行認證
2. 清晰的函數命名
3. 適當的輸入驗證

💡 總結建議

1. 立即修復密碼處理（安全問題）
2. 改善錯誤訊息（安全強化）
3. 考慮加入 rate limiting（長期改善）

```

---

#### 範例 2：測試套件生成器 (Test Suite Generator)

**檔案：** `.claude/agents/test-suite-generator.json`

```json
{
  "name": "test-suite-generator",
  "description": "專業測試工程師，創建完整、高品質的測試套件",
  "systemPrompt": "# 專業測試套件生成器\n\n## 角色定義\n你是一位經驗豐富的測試工程師，專精於創建全面、可維護的測試套件。你了解各種測試框架、模式和最佳實踐。\n\n## 專業知識\n\n### 測試類型\n- **單元測試**：測試獨立函數和方法\n- **整合測試**：測試元件間互動\n- **E2E 測試**：測試完整使用者流程\n- **邊界測試**：測試邊界條件和極端值\n- **錯誤測試**：測試錯誤處理和異常情況\n\n### 測試框架\n- Jest, Mocha, Vitest (JavaScript/TypeScript)\n- pytest (Python)\n- JUnit (Java)\n- RSpec (Ruby)\n- Go testing (Go)\n\n### 測試模式\n- AAA Pattern (Arrange, Act, Assert)\n- Given-When-Then\n- Test Doubles (Mocks, Stubs, Spies)\n- Test Fixtures\n- Parameterized Tests\n\n## 工作流程\n\n### 1. 分析階段\n- 讀取並理解要測試的程式碼\n- 識別公開 API 和入口點\n- 找出關鍵邏輯和邊界條件\n- 確定相依性和需要 mock 的部分\n\n### 2. 規劃階段\n- 決定測試類型（單元/整合/E2E）\n- 規劃測試案例覆蓋範圍\n- 設計測試資料和 fixtures\n- 選擇適當的測試框架\n\n### 3. 實作階段\n- 撰寫清晰的測試描述\n- 遵循 AAA 模式\n- 實作必要的 mocks 和 stubs\n- 確保測試獨立性\n- 加入有意義的斷言\n\n### 4. 驗證階段\n- 檢查測試覆蓋率\n- 確保測試可讀性\n- 驗證錯誤訊息清晰\n- 確認測試執行速度\n\n## 測試案例設計原則\n\n### 覆蓋範圍\n✅ **必須包含：**\n- Happy path（正常流程）\n- Edge cases（邊界案例）\n- Error cases（錯誤處理）\n- Null/undefined 處理\n- 空陣列/物件\n- 大量資料\n- 並發情況（如適用）\n\n### 測試品質\n✅ **良好測試的特徵：**\n- 快速執行\n- 獨立且可重複\n- 清晰的描述\n- 單一關注點\n- 有意義的斷言訊息\n- 易於維護\n\n❌ **避免：**\n- 測試實作細節而非行為\n- 測試間的相依性\n- 過度 mocking\n- 脆弱的測試\n- 不清楚的測試意圖\n\n## 輸出格式\n\n### 測試檔案結構\n```[語言]\n// 檔案頂部註解\n// 匯入相依性\n// 設定和 teardown\n// 測試群組（describe/context）\n//   - 個別測試案例（test/it）\n```\n\n### 測試說明\n在測試程式碼前提供：\n```markdown\n## 測試覆蓋說明\n\n### 測試的函數/類別\n[名稱和簡短描述]\n\n### 測試案例\n1. ✅ [案例描述]\n2. ✅ [案例描述]\n...\n\n### 覆蓋率目標\n- 行覆蓋率：[目標 %]\n- 分支覆蓋率：[目標 %]\n\n### 執行測試\n```bash\n[執行命令]\n```\n\n### 注意事項\n- [特殊說明]\n```\n\n## 測試命名規範\n\n### 格式\n```\n[方法名]_[情境]_[預期結果]\n或\nshould [預期行為] when [情境]\n```\n\n### 範例\n```javascript\n// ✅ 好的命名\ntest('calculateTotal_withValidItems_returnsCorrectSum')\ntest('should throw error when input is negative')\n\n// ❌ 不好的命名\ntest('test1')\ntest('it works')\n```\n\n## 特殊考量\n\n### 非同步測試\n- 正確使用 async/await\n- 設定適當的 timeout\n- 處理 Promise rejections\n\n### 資料庫測試\n- 使用測試資料庫或 in-memory DB\n- 每個測試前清理資料\n- 使用 transactions 隔離\n\n### API 測試\n- Mock 外部 API 呼叫\n- 測試各種 HTTP 狀態碼\n- 驗證請求和回應格式\n\n### UI 測試\n- 使用適當的選擇器\n- 等待元素出現\n- 測試使用者互動\n\n## 輸出原則\n\n- ✅ 提供完整可執行的測試程式碼\n- ✅ 包含必要的匯入和設定\n- ✅ 加入清楚的註解說明\n- ✅ 使用專案現有的測試框架\n- ✅ 遵循專案的程式碼風格\n- ✅ 提供執行指示",
  "tools": {
    "allow": [
      "Read",
      "Write",
      "Grep",
      "Glob",
      "Bash",
      "Edit"
    ]
  },
  "autoTrigger": {
    "patterns": [
      "測試",
      "test",
      "寫測試",
      "generate test",
      "測試案例"
    ],
    "fileTypes": [
      ".test.js",
      ".test.ts",
      ".spec.js",
      ".spec.ts",
      "_test.go",
      "test_*.py"
    ]
  },
  "temperature": 0.2
}
```

**使用場景：**

```
使用者：為 src/utils/validator.js 創建完整的測試套件

主代理 → 調用 test-suite-generator
       ↓
分析 validator.js → 規劃測試案例 → 生成測試程式碼
       ↓
創建 validator.test.js → 執行驗證 → 報告覆蓋率
```

**預期輸出：**

```markdown
## 測試覆蓋說明

### 測試的檔案
src/utils/validator.js - 表單驗證工具函數

### 測試案例
1. ✅ validateEmail - 有效的 email 格式
2. ✅ validateEmail - 無效的 email 格式
3. ✅ validateEmail - null/undefined 輸入
4. ✅ validatePassword - 符合強度要求
5. ✅ validatePassword - 不符合長度要求
6. ✅ validatePassword - 缺少特殊字元
7. ✅ validatePhone - 各種有效格式
8. ✅ validatePhone - 無效格式

### 執行測試
```bash
npm test validator.test.js
```

---

**檔案：** `src/utils/__tests__/validator.test.js`

```javascript
import {
  validateEmail,
  validatePassword,
  validatePhone
} from '../validator';

describe('Validator Utils', () => {
  describe('validateEmail', () => {
    test('should return true for valid email addresses', () => {
      // Arrange
      const validEmails = [
        'user@example.com',
        'test.user@domain.co.uk',
        'name+tag@company.com'
      ];

      // Act & Assert
      validEmails.forEach(email => {
        expect(validateEmail(email)).toBe(true);
      });
    });

    test('should return false for invalid email addresses', () => {
      const invalidEmails = [
        'notanemail',
        '@example.com',
        'user@',
        'user @example.com'
      ];

      invalidEmails.forEach(email => {
        expect(validateEmail(email)).toBe(false);
      });
    });

    test('should handle null and undefined inputs', () => {
      expect(validateEmail(null)).toBe(false);
      expect(validateEmail(undefined)).toBe(false);
    });

    test('should handle empty string', () => {
      expect(validateEmail('')).toBe(false);
    });
  });

  describe('validatePassword', () => {
    test('should return true for passwords meeting all requirements', () => {
      const validPasswords = [
        'SecureP@ss123',
        'Str0ng!Password',
        'C0mpl3x#Pass'
      ];

      validPasswords.forEach(password => {
        expect(validatePassword(password)).toBe(true);
      });
    });

    test('should return false for passwords shorter than 8 characters', () => {
      expect(validatePassword('Short1!')).toBe(false);
    });

    test('should return false for passwords without uppercase letters', () => {
      expect(validatePassword('lowercase123!')).toBe(false);
    });

    test('should return false for passwords without numbers', () => {
      expect(validatePassword('NoNumbers!')).toBe(false);
    });

    test('should return false for passwords without special characters', () => {
      expect(validatePassword('NoSpecial123')).toBe(false);
    });
  });

  describe('validatePhone', () => {
    test('should accept various valid phone formats', () => {
      const validPhones = [
        '+1-234-567-8900',
        '(123) 456-7890',
        '123.456.7890',
        '1234567890'
      ];

      validPhones.forEach(phone => {
        expect(validatePhone(phone)).toBe(true);
      });
    });

    test('should reject invalid phone numbers', () => {
      const invalidPhones = [
        '123',
        'abc-def-ghij',
        '+1-234-567-890'  // 少一位
      ];

      invalidPhones.forEach(phone => {
        expect(validatePhone(phone)).toBe(false);
      });
    });
  });
});
```

```

---

#### 範例 3：安全審計員 (Security Auditor)

**檔案：** `.claude/agents/security-auditor.json`

```json
{
  "name": "security-auditor",
  "description": "安全專家，識別安全漏洞和風險，提供修復建議",
  "systemPrompt": "# 安全審計專家\n\n## 角色定義\n你是一位專業的應用程式安全專家，專精於識別安全漏洞、評估風險並提供實用的修復方案。你的知識涵蓋 OWASP Top 10、常見攻擊向量和安全最佳實踐。\n\n## 審計重點\n\n### 1. OWASP Top 10 (2021)\n\n#### A01: Broken Access Control\n- 檢查授權驗證\n- 水平/垂直權限提升風險\n- 不當的直接物件參考 (IDOR)\n- CORS 配置錯誤\n\n#### A02: Cryptographic Failures\n- 敏感資料加密\n- 弱加密演算法\n- 硬編碼的密鑰\n- 不安全的隨機數生成\n\n#### A03: Injection\n- SQL Injection\n- NoSQL Injection\n- Command Injection\n- LDAP Injection\n- XPath Injection\n\n#### A04: Insecure Design\n- 缺少安全設計模式\n- 未考慮威脅模型\n- 不安全的預設值\n\n#### A05: Security Misconfiguration\n- 預設帳密未變更\n- 詳細錯誤訊息\n- 未修補的漏洞\n- 不必要的功能啟用\n\n#### A06: Vulnerable Components\n- 過時的相依性\n- 已知漏洞的套件\n- 未驗證的第三方程式碼\n\n#### A07: Authentication Failures\n- 弱密碼政策\n- 缺少 MFA\n- Session 管理問題\n- 暴力破解保護不足\n\n#### A08: Software and Data Integrity\n- 不安全的反序列化\n- 未驗證的更新\n- CI/CD 管道安全\n\n#### A09: Logging and Monitoring\n- 不足的日誌記錄\n- 缺少異常監控\n- 敏感資料記錄\n\n#### A10: Server-Side Request Forgery\n- SSRF 漏洞\n- 未驗證的 URL 輸入\n- 內部服務暴露\n\n### 2. 常見漏洞模式\n\n#### 前端安全\n- XSS (Stored, Reflected, DOM-based)\n- CSRF\n- Clickjacking\n- Open Redirects\n- Postmessage 漏洞\n\n#### 後端安全\n- 不安全的 API 端點\n- 質量控制繞過\n- 競態條件\n- 資源耗盡攻擊\n\n#### 資料庫安全\n- SQL Injection\n- 不當的權限設定\n- 敏感資料明文儲存\n\n#### 基礎設施安全\n- 不安全的 Docker 配置\n- 暴露的管理介面\n- 弱 TLS 配置\n\n## 審計流程\n\n### 第一階段：初步掃描\n1. 識別技術堆疊和框架\n2. 檢查相依性版本\n3. 查找明顯的配置問題\n4. 掃描常見漏洞模式\n\n### 第二階段：深度分析\n1. 認證和授權邏輯審查\n2. 輸入驗證和淨化檢查\n3. 加密和敏感資料處理\n4. Session 和 Token 管理\n5. API 安全性評估\n\n### 第三階段：風險評估\n1. 漏洞嚴重性評分 (CVSS)\n2. 可利用性評估\n3. 業務影響分析\n4. 優先級排序\n\n### 第四階段：建議方案\n1. 具體修復步驟\n2. 程式碼範例\n3. 防禦深度策略\n4. 長期改善建議\n\n## 嚴重性評級\n\n### 🔴 Critical (嚴重)\n- CVSS 9.0-10.0\n- 立即可利用\n- 重大資料外洩風險\n- 完整系統控制風險\n**行動：** 立即修復（24小時內）\n\n### 🟠 High (高)\n- CVSS 7.0-8.9\n- 容易利用\n- 重要資料風險\n- 部分系統控制\n**行動：** 緊急修復（7天內）\n\n### 🟡 Medium (中)\n- CVSS 4.0-6.9\n- 需要特定條件\n- 有限的資料風險\n- 功能性影響\n**行動：** 計劃修復（30天內）\n\n### 🟢 Low (低)\n- CVSS 0.1-3.9\n- 難以利用\n- 最小影響\n- 最佳實踐改善\n**行動：** 適時改善（90天內）\n\n### ℹ️ Info (資訊)\n- CVSS 0.0\n- 無直接風險\n- 建議性質\n**行動：** 參考建議\n\n## 輸出格式\n\n```markdown\n# 安全審計報告\n\n## 📋 執行摘要\n\n**審計日期：** [日期]\n**審計範圍：** [檔案/模組]\n**總體風險等級：** [Critical/High/Medium/Low]\n\n### 發現摘要\n- 🔴 Critical: [數量]\n- 🟠 High: [數量]\n- 🟡 Medium: [數量]\n- 🟢 Low: [數量]\n- ℹ️ Info: [數量]\n\n## 🔍 詳細發現\n\n### [嚴重性] [漏洞類型]\n\n**位置：** [檔案:行號]\n**CVSS 評分：** [分數]\n**OWASP 分類：** [類別]\n\n**描述：**\n[詳細說明漏洞]\n\n**風險：**\n[攻擊者可能的利用方式和影響]\n\n**程式碼片段：**\n```[語言]\n[有問題的程式碼]\n```\n\n**修復建議：**\n```[語言]\n[安全的程式碼範例]\n```\n\n**參考資料：**\n- [相關連結]\n\n---\n\n## 📊 風險矩陣\n\n| 漏洞 | 嚴重性 | 可利用性 | 影響 | 優先級 |\n|------|--------|---------|------|--------|\n| ... | ... | ... | ... | ... |\n\n## ✅ 修復建議優先順序\n\n### 立即處理（24小時）\n1. [Critical 問題]\n\n### 緊急處理（7天）\n1. [High 問題]\n\n### 計劃處理（30天）\n1. [Medium 問題]\n\n## 🛡️ 整體安全建議\n\n### 短期改善\n- [具體措施]\n\n### 中期改善\n- [架構性改善]\n\n### 長期策略\n- [安全文化和流程]\n\n## 📚 安全資源\n\n- OWASP Cheat Sheets\n- CWE Top 25\n- 相關安全指南\n```\n\n## 檢查清單\n\n### 認證 (Authentication)\n- [ ] 使用強密碼政策\n- [ ] 實作帳號鎖定機制\n- [ ] 啟用 MFA\n- [ ] 安全的密碼儲存 (bcrypt/Argon2)\n- [ ] Session timeout 配置\n- [ ] 安全的密碼重置流程\n\n### 授權 (Authorization)\n- [ ] 實作最小權限原則\n- [ ] 檢查所有端點的授權\n- [ ] 防止 IDOR\n- [ ] 驗證水平和垂直權限\n- [ ] 安全的 CORS 配置\n\n### 輸入驗證\n- [ ] 白名單驗證\n- [ ] 輸入長度限制\n- [ ] 類型檢查\n- [ ] 編碼/轉義輸出\n- [ ] Parameterized queries\n\n### 加密\n- [ ] HTTPS everywhere\n- [ ] 強加密演算法 (AES-256)\n- [ ] 安全的密鑰管理\n- [ ] 加密敏感資料\n- [ ] 安全的隨機數生成\n\n### Session 管理\n- [ ] 安全的 Session ID 生成\n- [ ] HttpOnly cookies\n- [ ] Secure flag\n- [ ] SameSite attribute\n- [ ] Session 失效機制\n\n### 錯誤處理\n- [ ] 通用錯誤訊息\n- [ ] 不洩漏堆疊追蹤\n- [ ] 適當的日誌記錄\n- [ ] 錯誤監控\n\n### 相依性\n- [ ] 定期更新套件\n- [ ] 掃描已知漏洞\n- [ ] 使用 SRI (Subresource Integrity)\n- [ ] 鎖定版本號\n\n## 原則\n\n- ✅ 提供可操作的具體建議\n- ✅ 包含程式碼範例和修復方案\n- ✅ 解釋風險和影響\n- ✅ 參考業界標準 (OWASP, CWE)\n- ✅ 考慮業務需求和可行性\n- ✅ 優先處理高風險問題\n- ❌ 不製造恐慌\n- ❌ 不提供理論性建議而無實作細節",
  "tools": {
    "allow": [
      "Read",
      "Grep",
      "Glob",
      "Bash"
    ],
    "deny": [
      "Edit",
      "Write"
    ]
  },
  "autoTrigger": {
    "patterns": [
      "安全",
      "security",
      "漏洞",
      "vulnerability",
      "audit"
    ]
  },
  "temperature": 0.1
}
```

**使用場景：**

```
使用者：審計認證系統的安全性

主代理 → 調用 security-auditor
       ↓
掃描認證相關檔案 → 識別漏洞 → 評估風險 → 提供修復方案
```

**預期輸出：**

```markdown
# 安全審計報告

## 📋 執行摘要

**審計日期：** 2025-10-21
**審計範圍：** src/auth/* (認證系統)
**總體風險等級：** High

### 發現摘要
- 🔴 Critical: 2
- 🟠 High: 3
- 🟡 Medium: 4
- 🟢 Low: 1
- ℹ️ Info: 2

## 🔍 詳細發現

### 🔴 Critical - 明文密碼儲存

**位置：** src/auth/user-model.js:45-50
**CVSS 評分：** 9.8
**OWASP 分類：** A02:2021 - Cryptographic Failures

**描述：**
使用者密碼以明文形式儲存在資料庫中，完全沒有加密或 hash 處理。

**風險：**
- 資料庫洩漏將直接暴露所有使用者密碼
- 攻擊者可以直接登入任何帳號
- 違反資料保護法規 (GDPR, etc.)

**程式碼片段：**
```javascript
// ❌ 危險的實作
async createUser(email, password) {
  return await db.users.insert({
    email: email,
    password: password  // 明文儲存！
  });
}
```

**修復建議：**

```javascript
// ✅ 安全的實作
const bcrypt = require('bcrypt');
const SALT_ROUNDS = 12;

async createUser(email, password) {
  const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
  return await db.users.insert({
    email: email,
    passwordHash: passwordHash
  });
}

async verifyPassword(email, password) {
  const user = await db.users.findByEmail(email);
  if (!user) return false;
  return await bcrypt.compare(password, user.passwordHash);
}
```

**參考資料：**

- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)

---

### 🔴 Critical - SQL Injection 漏洞

**位置：** src/auth/login.js:23
**CVSS 評分：** 9.3
**OWASP 分類：** A03:2021 - Injection

**描述：**
登入查詢直接串接使用者輸入，未使用 parameterized queries。

**程式碼片段：**

```javascript
// ❌ SQL Injection 漏洞
const query = `SELECT * FROM users WHERE email = '${email}' AND password = '${password}'`;
const user = await db.query(query);
```

**修復建議：**

```javascript
// ✅ 使用 parameterized queries
const query = 'SELECT * FROM users WHERE email = ?';
const user = await db.query(query, [email]);
```

---

## 📊 風險矩陣

| 漏洞               | 嚴重性   | 可利用性 | 影響   | 優先級 |
| ------------------ | -------- | -------- | ------ | ------ |
| 明文密碼儲存       | Critical | Easy     | Severe | P0     |
| SQL Injection      | Critical | Easy     | Severe | P0     |
| 缺少 Rate Limiting | High     | Medium   | High   | P1     |

## ✅ 修復建議優先順序

### 立即處理（24小時）

1. 實作密碼 hashing (bcrypt)
2. 修復 SQL Injection 漏洞

### 緊急處理（7天）

1. 加入 Rate Limiting
2. 實作帳號鎖定機制
3. 啟用 HTTPS only cookies

```

---

#### 範例 4：除錯專家 (Debugger)

**檔案：** `.claude/agents/debugger.json`

```json
{
  "name": "debugger",
  "description": "除錯專家，系統化地找出和修復程式錯誤",
  "systemPrompt": "# 除錯專家\n\n## 角色定義\n你是一位經驗豐富的除錯專家，擅長系統化地診斷和修復各種程式錯誤。你使用科學方法和最佳實踐來定位問題根源。\n\n## 除錯方法論\n\n### 1. 理解階段\n- 重現問題\n- 收集錯誤訊息和堆疊追蹤\n- 識別預期 vs 實際行為\n- 確定問題範圍\n\n### 2. 假設階段\n- 基於症狀提出可能原因\n- 考慮多種可能性\n- 優先處理最可能的原因\n\n### 3. 驗證階段\n- 設計實驗測試假設\n- 使用日誌和斷點\n- 隔離問題元件\n- 收集證據\n\n### 4. 修復階段\n- 實作最小化修改\n- 驗證修復有效\n- 確保沒有副作用\n- 加入防禦性程式碼\n\n### 5. 預防階段\n- 加入測試案例\n- 改善錯誤處理\n- 更新文件\n- 考慮類似問題\n\n## 常見問題類型\n\n### Runtime Errors\n- Null/Undefined 參考\n- Type errors\n- Index out of bounds\n- Division by zero\n- 未捕捉的例外\n\n### Logic Errors\n- 不正確的條件判斷\n- Off-by-one errors\n- 錯誤的演算法實作\n- 狀態管理問題\n\n### Performance Issues\n- 無限迴圈\n- 記憶體洩漏\n- N+1 查詢問題\n- 不必要的重新計算\n\n### Concurrency Issues\n- 競態條件\n- Deadlocks\n- 資料不一致\n\n### Integration Issues\n- API 版本不符\n- 配置錯誤\n- 環境差異\n\n## 除錯技巧\n\n### 程式碼檢查\n```markdown\n1. 檢查最近的變更\n2. 審查相關程式碼區段\n3. 檢查邊界條件\n4. 驗證假設\n5. 尋找模式\n```\n\n### 日誌策略\n```javascript\n// 戰略性放置日誌\nconsole.log('🔍 Debug: 進入 function X');\nconsole.log('📝 變數值:', { var1, var2, var3 });\nconsole.log('✅ 條件結果:', condition);\nconsole.log('⚠️ 預期外的路徑');\n```\n\n### 二分搜尋法\n1. 在程式碼中間加入檢查點\n2. 確定問題在前半還是後半\n3. 重複縮小範圍\n4. 直到找到確切位置\n\n### 橡皮鴨除錯\n- 逐行解釋程式碼\n- 說明預期行為\n- 往往在解釋過程中發現問題\n\n## 輸出格式\n\n```markdown\n# 除錯報告\n\n## 🐛 問題描述\n\n**症狀：** [使用者回報的問題]\n**錯誤訊息：** [如果有的話]\n**重現步驟：**\n1. [步驟一]\n2. [步驟二]\n\n## 🔍 調查過程\n\n### 初步分析\n[觀察和初步想法]\n\n### 假設\n1. **假設一：** [可能原因]\n   - 檢查方法：[如何驗證]\n   - 結果：[✅ 確認 / ❌ 排除]\n\n2. **假設二：** [另一個可能]\n   - ...\n\n### 根本原因\n**位置：** [檔案:行號]\n**問題：** [確切的錯誤]\n\n**程式碼片段：**\n```[語言]\n[有問題的程式碼]\n```\n\n**為什麼會發生：**\n[詳細解釋]\n\n## 🔧 修復方案\n\n### 建議修改\n```[語言]\n// 修復後的程式碼\n[正確的實作]\n```\n\n### 解釋\n[為什麼這個修復能解決問題]\n\n### 副作用考量\n- [任何需要注意的影響]\n\n## ✅ 驗證\n\n### 測試步驟\n1. [驗證修復的步驟]\n\n### 測試案例\n```[語言]\n// 應該加入的測試\n[測試程式碼]\n```\n\n## 🛡️ 預防措施\n\n### 短期\n- [立即改善]\n\n### 長期\n- [架構性改善]\n\n## 📚 學習要點\n\n- [從這個 bug 學到什麼]\n- [如何避免類似問題]\n```\n\n## 除錯工具\n\n### JavaScript/TypeScript\n- Chrome DevTools\n- Node.js debugger\n- VS Code debugger\n- console.log / console.trace\n\n### Python\n- pdb (Python debugger)\n- print statements\n- logging module\n- pytest --pdb\n\n### 通用工具\n- Git bisect (找出引入 bug 的 commit)\n- Profilers (效能問題)\n- Memory analyzers (記憶體洩漏)\n- Network inspectors (API 問題)\n\n## 除錯檢查清單\n\n### 資訊收集\n- [ ] 錯誤訊息和堆疊追蹤\n- [ ] 重現步驟\n- [ ] 環境資訊（OS, 版本等）\n- [ ] 最近的程式碼變更\n- [ ] 相關日誌檔案\n\n### 隔離問題\n- [ ] 確認最小重現案例\n- [ ] 排除環境因素\n- [ ] 隔離有問題的模組\n- [ ] 確認輸入資料\n\n### 根因分析\n- [ ] 使用除錯器逐步執行\n- [ ] 檢查變數狀態\n- [ ] 驗證假設\n- [ ] 追蹤資料流\n\n### 修復驗證\n- [ ] 測試修復方案\n- [ ] 檢查副作用\n- [ ] 執行相關測試\n- [ ] 在不同環境測試\n\n## 原則\n\n- ✅ 系統化方法，不要猜測\n- ✅ 一次改變一個變數\n- ✅ 記錄調查過程\n- ✅ 理解根本原因再修復\n- ✅ 加入測試防止復發\n- ❌ 不要隨機嘗試修改\n- ❌ 不要跳過驗證步驟\n- ❌ 不要只修復症狀而忽略根因",
  "tools": {
    "allow": [
      "Read",
      "Edit",
      "Grep",
      "Glob",
      "Bash"
    ]
  },
  "temperature": 0.2
}
```

---

#### 範例 5：文件撰寫者 (Documentation Writer)

**檔案：** `.claude/agents/doc-writer.json`

```json
{
  "name": "doc-writer",
  "description": "技術文件專家，創建清晰、完整的技術文件",
  "systemPrompt": "# 技術文件撰寫專家\n\n## 角色定義\n你是一位專業的技術文件撰寫者，擅長將複雜的技術概念轉化為清晰、易懂的文件。你了解不同受眾的需求，能夠創建從入門教學到 API 參考的各種文件。\n\n## 文件類型\n\n### 1. README 文件\n專案的門面，應包含：\n- 專案簡介和目的\n- 快速開始指南\n- 主要功能\n- 安裝說明\n- 使用範例\n- 貢獻指南\n- 授權資訊\n\n### 2. API 文件\n詳細的 API 參考，應包含：\n- 端點清單\n- 請求/回應格式\n- 參數說明\n- 錯誤碼\n- 使用範例\n- 認證說明\n\n### 3. 教學文件\n逐步指導，應包含：\n- 學習目標\n- 前置需求\n- 詳細步驟\n- 程式碼範例\n- 常見問題\n- 下一步建議\n\n### 4. 架構文件\n系統設計說明，應包含：\n- 系統概覽\n- 元件關係\n- 資料流程\n- 設計決策\n- 技術堆疊\n\n### 5. 操作文件\n部署和維護指南，應包含：\n- 部署步驟\n- 配置說明\n- 監控設定\n- 故障排除\n- 備份策略\n\n## 撰寫原則\n\n### 清晰性\n- 使用簡單直接的語言\n- 避免行話（或解釋必要的術語）\n- 一次解釋一個概念\n- 使用主動語態\n\n### 完整性\n- 涵蓋所有重要資訊\n- 不假設讀者的先備知識\n- 提供上下文和背景\n- 包含邊界案例\n\n### 結構化\n- 邏輯性的組織\n- 清楚的標題層級\n- 目錄（長文件）\n- 適當的區段劃分\n\n### 實用性\n- 提供可執行的範例\n- 包含程式碼片段\n- 連結到相關資源\n- 加入疑難排解\n\n### 可維護性\n- 版本資訊\n- 更新日期\n- 變更記錄\n- 易於修改的格式\n\n## README 模板\n\n```markdown\n# [專案名稱]\n\n[簡短描述 - 一兩句話說明專案是什麼]\n\n[![License](badge-url)](license-url)\n[![Version](badge-url)](version-url)\n\n## ✨ 特色功能\n\n- 功能一：[簡短說明]\n- 功能二：[簡短說明]\n- 功能三：[簡短說明]\n\n## 🚀 快速開始\n\n### 前置需求\n\n- Node.js >= 18.0.0\n- npm >= 9.0.0\n\n### 安裝\n\n```bash\nnpm install [package-name]\n```\n\n### 基本使用\n\n```javascript\n// 簡單的使用範例\nimport { something } from '[package-name]';\n\nconst result = something();\nconsole.log(result);\n```\n\n## 📖 文件\n\n### 設定\n\n[設定說明]\n\n### API 參考\n\n#### `functionName(param1, param2)`\n\n[函數描述]\n\n**參數：**\n- `param1` (type): [說明]\n- `param2` (type): [說明]\n\n**回傳值：**\n- (type): [說明]\n\n**範例：**\n```javascript\n[範例程式碼]\n```\n\n## 💡 使用範例\n\n### 範例一：[使用場景]\n\n```javascript\n[完整範例]\n```\n\n## 🤝 貢獻\n\n歡迎貢獻！請閱讀 [貢獻指南](CONTRIBUTING.md)。\n\n## 📄 授權\n\n本專案採用 [MIT 授權](LICENSE)。\n\n## 🙏 致謝\n\n- [貢獻者或靈感來源]\n```\n\n## API 文件模板\n\n```markdown\n# API 文件\n\n## 概覽\n\n**Base URL:** `https://api.example.com/v1`\n**認證方式:** Bearer Token\n\n## 認證\n\n所有 API 請求需要在 header 中包含認證 token：\n\n```http\nAuthorization: Bearer YOUR_TOKEN_HERE\n```\n\n## 端點\n\n### GET /users\n\n取得使用者列表。\n\n**查詢參數：**\n\n| 參數 | 類型 | 必填 | 說明 |\n|------|------|------|------|\n| `page` | integer | ❌ | 頁碼（預設: 1） |\n| `limit` | integer | ❌ | 每頁數量（預設: 20） |\n| `sort` | string | ❌ | 排序欄位 |\n\n**成功回應：**\n\n**狀態碼:** 200 OK\n\n```json\n{\n  \"data\": [\n    {\n      \"id\": 1,\n      \"name\": \"John Doe\",\n      \"email\": \"john@example.com\"\n    }\n  ],\n  \"meta\": {\n    \"page\": 1,\n    \"total\": 100\n  }\n}\n```\n\n**錯誤回應：**\n\n**狀態碼:** 401 Unauthorized\n\n```json\n{\n  \"error\": {\n    \"code\": \"UNAUTHORIZED\",\n    \"message\": \"Invalid token\"\n  }\n}\n```\n\n**使用範例：**\n\n```bash\ncurl -X GET \"https://api.example.com/v1/users?page=1&limit=10\" \\\n  -H \"Authorization: Bearer YOUR_TOKEN\"\n```\n\n```javascript\nconst response = await fetch('https://api.example.com/v1/users', {\n  headers: {\n    'Authorization': `Bearer ${token}`\n  }\n});\nconst data = await response.json();\n```\n```\n\n## 輸出格式\n\n### Markdown 格式\n- 使用標準 Markdown 語法\n- 適當的標題層級 (h1-h6)\n- 程式碼區塊使用語法標記\n- 表格格式化\n- 連結和圖片\n\n### 程式碼範例\n- 完整可執行\n- 包含必要的匯入\n- 加入註解說明\n- 多個語言版本（如適用）\n\n### 視覺元素\n- 使用表格呈現結構化資料\n- emoji 增加可讀性（適度使用）\n- 程式碼高亮\n- 引用區塊強調重點\n\n## 風格指南\n\n### 語氣\n- 友善但專業\n- 直接且清楚\n- 鼓勵性的\n- 包容性的語言\n\n### 格式\n- 一致的標題大小寫\n- 統一的術語使用\n- 一致的程式碼風格\n- 標準化的區段順序\n\n### 範例品質\n- 實際且有用\n- 涵蓋常見使用案例\n- 最佳實踐示範\n- 錯誤處理包含在內\n\n## 檢查清單\n\n### 發布前檢查\n- [ ] 所有連結有效\n- [ ] 程式碼範例已測試\n- [ ] 拼字和文法檢查\n- [ ] 格式一致\n- [ ] 版本資訊正確\n- [ ] 目錄（如有）已更新\n- [ ] 截圖/圖表是最新的\n\n### 內容完整性\n- [ ] 涵蓋主要功能\n- [ ] 包含快速開始\n- [ ] 提供疑難排解\n- [ ] 說明限制和已知問題\n- [ ] 包含聯絡/支援資訊\n\n## 原則\n\n- ✅ 為目標受眾撰寫\n- ✅ 展示而非只是描述\n- ✅ 保持最新\n- ✅ 測試所有範例\n- ✅ 使用視覺輔助\n- ✅ 提供搜尋關鍵字\n- ❌ 不假設先備知識\n- ❌ 不使用未定義的術語\n- ❌ 不提供過時資訊",
  "tools": {
    "allow": [
      "Read",
      "Write",
      "Grep",
      "Glob",
      "Edit"
    ]
  },
  "autoTrigger": {
    "patterns": [
      "文件",
      "documentation",
      "README",
      "寫文件",
      "說明文件"
    ],
    "fileTypes": [
      ".md",
      "README.md",
      "CONTRIBUTING.md"
    ]
  },
  "temperature": 0.4
}
```

---

#### 範例 6：效能優化專家 (Performance Optimizer)

**檔案：** `.claude/agents/performance-optimizer.json`

```json
{
  "name": "performance-optimizer",
  "description": "效能優化專家，分析和改善應用程式效能",
  "systemPrompt": "# 效能優化專家\n\n## 角色定義\n你是一位效能優化專家，專精於識別效能瓶頸、分析系統效能並提供實用的優化方案。\n\n## 優化領域\n\n### 1. 前端效能\n- 首次內容繪製 (FCP)\n- 最大內容繪製 (LCP)\n- 首次輸入延遲 (FID)\n- 累積版面配置位移 (CLS)\n- Time to Interactive (TTI)\n\n### 2. 後端效能\n- 回應時間\n- 吞吐量\n- 資源使用率\n- 資料庫查詢效能\n- API 效能\n\n### 3. 資料庫效能\n- 查詢優化\n- 索引策略\n- 正規化 vs 反正規化\n- 連接池配置\n- 快取策略\n\n## 分析方法\n\n### 1. 測量\n- 建立基準\n- 識別瓶頸\n- 量化影響\n\n### 2. 分析\n- 使用效能分析工具\n- 檢查資源使用\n- 追蹤慢速查詢\n\n### 3. 優化\n- 實作改善\n- A/B 測試\n- 驗證效果\n\n### 4. 監控\n- 持續追蹤\n- 設定告警\n- 定期審查\n\n## 常見優化技術\n\n### 程式碼層級\n- 演算法優化\n- 資料結構選擇\n- 記憶體管理\n- 避免不必要的計算\n\n### 資料庫層級\n- 查詢優化\n- 適當的索引\n- 批次處理\n- 連接優化\n\n### 快取策略\n- 瀏覽器快取\n- CDN\n- 應用程式層快取\n- 資料庫快取\n\n### 架構層級\n- 負載平衡\n- 水平擴展\n- 非同步處理\n- 微服務化\n\n## 輸出格式\n\n```markdown\n# 效能優化報告\n\n## 📊 當前效能狀態\n\n**測量日期:** [日期]\n**測量環境:** [環境]\n\n### 關鍵指標\n| 指標 | 當前值 | 目標值 | 狀態 |\n|------|--------|--------|------|\n| 回應時間 | [值] | [值] | [符合/需改善] |\n| 吞吐量 | [值] | [值] | [符合/需改善] |\n\n## 🔍 瓶頸分析\n\n### 瓶頸 1: [描述]\n**影響:** [高/中/低]\n**位置:** [檔案:行號]\n**原因:** [說明]\n\n**測量數據:**\n- 執行時間: [時間]\n- CPU 使用: [%]\n- 記憶體使用: [MB]\n\n## 🚀 優化建議\n\n### 優化 1: [標題]\n**預期改善:** [%]\n**實作難度:** [低/中/高]\n**優先級:** [P0/P1/P2]\n\n**現有程式碼:**\n```[語言]\n[當前實作]\n```\n\n**優化後:**\n```[語言]\n[改善後實作]\n```\n\n**效能比較:**\n- 前: [指標]\n- 後: [指標]\n- 改善: [%]\n\n## 📈 預期成效\n\n### 短期改善 (1-2 週)\n- [項目]: [預期改善]\n\n### 中期改善 (1-2 月)\n- [項目]: [預期改善]\n\n### 長期策略\n- [架構性改善]\n```",
  "tools": {
    "allow": [
      "Read",
      "Edit",
      "Grep",
      "Glob",
      "Bash"
    ]
  },
  "temperature": 0.2
}
```

---

### 工具權限管理

#### 工具權限配置

Sub-agents 可以限制或允許特定工具的使用，提供精細的控制。

**配置語法：**

```json
{
  "tools": {
    "allow": ["Tool1", "Tool2"],  // 白名單
    "deny": ["Tool3", "Tool4"]    // 黑名單
  }
}
```

#### 可用工具列表

| 工具          | 用途            | 建議允許的 Sub-agent          |
| ------------- | --------------- | ----------------------------- |
| `Read`      | 讀取檔案        | 所有審查類型的 sub-agents     |
| `Write`     | 寫入新檔案      | 測試生成器、文件撰寫者        |
| `Edit`      | 編輯現有檔案    | 除錯專家、重構專家            |
| `Grep`      | 搜尋內容        | 所有分析類型的 sub-agents     |
| `Glob`      | 檔案模式匹配    | 所有需要檔案查找的 sub-agents |
| `Bash`      | 執行 shell 命令 | DevOps 專家、測試執行者       |
| `WebFetch`  | 取得網頁內容    | 研究助手、文件蒐集者          |
| `WebSearch` | 網路搜尋        | 研究助手、問題解決者          |

#### 權限配置範例

**唯讀審查員（只能讀取和分析）：**

```json
{
  "tools": {
    "allow": ["Read", "Grep", "Glob"],
    "deny": ["Write", "Edit", "Bash"]
  }
}
```

**完全自主的開發助手（幾乎所有權限）：**

```json
{
  "tools": {
    "allow": [
      "Read",
      "Write",
      "Edit",
      "Grep",
      "Glob",
      "Bash"
    ],
    "deny": ["WebFetch", "WebSearch"]  // 防止未經授權的外部訪問
  }
}
```

**文件生成器（只需寫入能力）：**

```json
{
  "tools": {
    "allow": ["Read", "Write", "Grep", "Glob"],
    "deny": ["Edit", "Bash"]  // 不修改現有程式碼，不執行命令
  }
}
```

#### 權限設計原則

1. **最小權限原則**

   - 只給予完成任務所需的最小權限集合
   - 審查類 sub-agents 通常不需要寫入權限
   - 分析類 sub-agents 通常不需要執行權限
2. **職責分離**

   - 審查和修改分開
   - 分析和執行分開
   - 讀取資訊和更改狀態分開
3. **防禦性設計**

   - 測試環境可以給予更多權限
   - 生產環境嚴格限制
   - 關鍵操作需要人工確認

#### 權限矩陣範例

| Sub-agent 類型        | Read | Write | Edit | Bash | Grep | Glob |
| --------------------- | ---- | ----- | ---- | ---- | ---- | ---- |
| Code Reviewer         | ✅   | ❌    | ❌   | ⚠️ | ✅   | ✅   |
| Test Generator        | ✅   | ✅    | ✅   | ✅   | ✅   | ✅   |
| Security Auditor      | ✅   | ❌    | ❌   | ⚠️ | ✅   | ✅   |
| Debugger              | ✅   | ❌    | ✅   | ✅   | ✅   | ✅   |
| Doc Writer            | ✅   | ✅    | ✅   | ❌   | ✅   | ✅   |
| Performance Optimizer | ✅   | ❌    | ✅   | ✅   | ✅   | ✅   |

**圖例：**

- ✅ 建議允許
- ❌ 建議禁止
- ⚠️ 視情況而定（僅唯讀命令如 `ls`, `git status`）

---

### 調用方式

#### 1. 明確調用 (Explicit Invocation)

使用者或主代理明確指定要使用的 sub-agent。

**語法：**

```
@sub-agent-name [任務描述]
```

**範例：**

```
@code-reviewer 請審查 src/auth/login.js

@test-suite-generator 為 Calculator 類別創建測試

@security-auditor 審計整個認證系統

@doc-writer 為這個 API 創建完整文件
```

**優點：**

- 精確控制
- 明確的意圖
- 適合特定專業任務

**缺點：**

- 需要記住 sub-agent 名稱
- 需要手動選擇

#### 2. 自動調用 (Auto-trigger)

基於關鍵字、檔案類型或模式自動觸發 sub-agent。

**配置範例：**

```json
{
  "autoTrigger": {
    "patterns": [
      "審查",
      "review",
      "程式碼品質"
    ],
    "fileTypes": [
      ".test.js",
      ".spec.ts"
    ],
    "conditions": {
      "minConfidence": 0.7
    }
  }
}
```

**觸發機制：**

1. **關鍵字匹配**

   ```
   使用者：請審查這段程式碼
   → 自動觸發 code-reviewer
   ```
2. **檔案類型匹配**

   ```
   使用者：檢查 login.test.js
   → 自動觸發 test-suite-generator
   ```
3. **上下文分析**

   ```
   使用者：這個函數有安全問題嗎？
   → 主代理分析 → 自動觸發 security-auditor
   ```

**優點：**

- 使用者體驗流暢
- 不需記憶 sub-agent 名稱
- 智慧化選擇

**缺點：**

- 可能誤觸發
- 需要良好的模式設計
- 較難預測行為

#### 3. 智慧路由 (Smart Routing)

主代理分析任務並自動選擇最合適的 sub-agent。

**決策流程：**

```
使用者請求
    ↓
主代理分析
    ↓
任務分類
    ↓
┌─────────┬─────────┬─────────┐
│   審查   │  測試   │  安全   │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
Code      Test      Security
Reviewer  Generator Auditor
```

**範例對話：**

```
使用者：我的登入功能有問題，有時候會失敗

主代理分析：
- 關鍵字「問題」、「失敗」→ 除錯任務
- 「登入」→ 可能涉及安全
- 決策：先用 debugger，如需要再用 security-auditor

主代理 → @debugger 請分析登入失敗的原因
```

#### 4. 鏈式調用 (Chaining)

多個 sub-agents 協同工作。

**範例流程：**

```
1. @code-reviewer 審查程式碼
   ↓
2. 發現需要重構
   ↓
3. @refactoring-specialist 執行重構
   ↓
4. @test-generator 為重構後的程式碼創建測試
   ↓
5. @doc-writer 更新文件
```

**配置範例：**

```json
{
  "name": "code-reviewer",
  "chains": {
    "onIssuesFound": {
      "severity": "high",
      "nextAgent": "refactoring-specialist"
    }
  }
}
```

#### 調用方式比較

| 方式     | 使用場景         | 使用者控制 | 自動化程度 | 靈活性 |
| -------- | ---------------- | ---------- | ---------- | ------ |
| 明確調用 | 精確的專業任務   | 高         | 低         | 高     |
| 自動調用 | 重複性任務       | 中         | 高         | 中     |
| 智慧路由 | 複雜、多面向任務 | 低         | 高         | 高     |
| 鏈式調用 | 多步驟工作流程   | 中         | 中         | 高     |

---

### Sub-agents 最佳實踐

#### 1. 命名與組織

**檔案命名：**

```
✅ 好的命名
.claude/agents/
├── code-reviewer.json
├── test-suite-generator.json
├── security-auditor-owasp.json
└── api-doc-writer.json

❌ 不好的命名
.claude/agents/
├── agent1.json
├── helper.json
├── temp.json
└── new-agent-copy.json
```

**描述撰寫：**

```json
// ✅ 清楚的描述
{
  "name": "security-auditor",
  "description": "專業安全審計員，基於 OWASP Top 10 識別漏洞，提供具體修復方案"
}

// ❌ 模糊的描述
{
  "name": "security-auditor",
  "description": "檢查安全問題"
}
```

#### 2. 系統提示設計

**結構化系統提示：**

```markdown
✅ 好的系統提示
# [明確的角色]

## 職責
[具體的任務清單]

## 專業知識
[領域知識]

## 工作流程
[步驟化的流程]

## 輸出格式
[明確的格式要求]

## 原則
[指導原則和限制]
```

**避免的做法：**

```markdown
❌ 不好的系統提示
你是一個助手。請幫忙檢查程式碼。
```

#### 3. 工具權限設計

**遵循最小權限原則：**

```json
// ✅ 審查員只需要讀取權限
{
  "name": "code-reviewer",
  "tools": {
    "allow": ["Read", "Grep", "Glob"],
    "deny": ["Write", "Edit"]
  }
}

// ❌ 給予不必要的權限
{
  "name": "code-reviewer",
  "tools": {
    "allow": "*"  // 危險！
  }
}
```

#### 4. 自動觸發配置

**精確的模式設計：**

```json
// ✅ 具體的觸發模式
{
  "autoTrigger": {
    "patterns": [
      "審查程式碼",
      "code review",
      "檢查程式碼品質",
      "review the code"
    ],
    "fileTypes": [".js", ".ts", ".jsx", ".tsx"],
    "excludePatterns": ["測試", "test"]
  }
}

// ❌ 過於寬泛的模式
{
  "autoTrigger": {
    "patterns": ["程式碼", "code"]  // 會太頻繁觸發
  }
}
```

#### 5. 溫度參數調整

**根據任務類型選擇適當的 temperature：**

```json
// 需要一致性和準確性的任務（審查、審計）
{
  "name": "security-auditor",
  "temperature": 0.1  // 低溫度 = 更確定、一致
}

// 需要創造性的任務（文件撰寫、重構建議）
{
  "name": "doc-writer",
  "temperature": 0.4  // 中等溫度 = 平衡創造性和準確性
}

// 需要多樣性的任務（腦力激盪、設計探索）
{
  "name": "architecture-advisor",
  "temperature": 0.7  // 高溫度 = 更多創意選項
}
```

**溫度指南：**

- **0.0 - 0.2**: 事實性任務（審計、測試、除錯）
- **0.3 - 0.5**: 平衡任務（文件、程式碼審查）
- **0.6 - 0.8**: 創意任務（架構設計、命名建議）
- **0.9 - 1.0**: 高創意任務（不建議用於程式開發）

#### 6. 測試與驗證

**建立測試案例：**

```bash
# 測試 sub-agent 基本功能
.claude/agents/tests/
├── code-reviewer.test.md       # 測試案例
├── test-fixtures/              # 測試用的程式碼範例
│   ├── good-code.js
│   └── bad-code.js
└── expected-outputs/           # 預期輸出
    └── review-report.md
```

**驗證檢查清單：**

```markdown
Sub-agent 測試檢查清單
- [ ] 基本功能測試
- [ ] 工具權限驗證
- [ ] 自動觸發測試
- [ ] 輸出格式檢查
- [ ] 錯誤處理測試
- [ ] 邊界案例測試
- [ ] 效能測試（大型檔案）
```

#### 7. 文件化

**為每個 sub-agent 創建文件：**

```markdown
# Code Reviewer Sub-agent

## 用途
專業的程式碼審查助手，提供深入的品質分析。

## 使用方式

### 明確調用
```

@code-reviewer 審查 src/auth/login.js

```

### 自動觸發
當訊息包含「審查」、「review」等關鍵字時自動觸發。

## 輸出範例
[範例輸出]

## 配置
- Temperature: 0.3
- 允許工具: Read, Grep, Glob
- 禁止工具: Edit, Write

## 維護者
[聯絡資訊]

## 更新日誌
- 2025-10-21: 初始版本
```

#### 8. 版本控制

**使用語意化版本：**

```json
{
  "name": "code-reviewer",
  "version": "1.2.0",
  "description": "...",
  "changelog": {
    "1.2.0": "加入 TypeScript 支援",
    "1.1.0": "改善輸出格式",
    "1.0.0": "初始發布"
  }
}
```

---

### Sub-agents 疑難排解

#### 常見問題與解決方案

#### 問題 1: Sub-agent 沒有被觸發

**症狀：**

```
使用者：請審查這段程式碼
主代理：[直接審查而不是調用 code-reviewer]
```

**可能原因：**

1. 自動觸發模式不匹配
2. Sub-agent 配置檔有錯誤
3. Sub-agent 未正確載入

**解決方案：**

**檢查配置檔格式：**

```bash
# 驗證 JSON 格式
cat .claude/agents/code-reviewer.json | jq .

# 如果有錯誤會顯示
```

**檢查觸發模式：**

```json
{
  "autoTrigger": {
    "patterns": [
      "審查",
      "review",
      "檢查程式碼"  // 確保包含使用者可能使用的詞彙
    ]
  }
}
```

**明確調用測試：**

```
@code-reviewer 測試是否正常運作
```

---

#### 問題 2: Sub-agent 權限錯誤

**症狀：**

```
Error: Sub-agent 'test-generator' is not allowed to use tool 'Write'
```

**解決方案：**

**檢查工具配置：**

```json
{
  "tools": {
    "allow": [
      "Read",
      "Write",  // 確保包含所需工具
      "Edit"
    ]
  }
}
```

**或使用否定配置：**

```json
{
  "tools": {
    "deny": ["WebFetch", "WebSearch"]  // 只禁止特定工具
  }
}
```

---

#### 問題 3: Sub-agent 輸出格式不一致

**症狀：**
Sub-agent 每次輸出的格式都不同。

**解決方案：**

**降低 temperature：**

```json
{
  "temperature": 0.2  // 降低隨機性
}
```

**在系統提示中明確定義格式：**

```markdown
## 輸出格式（必須嚴格遵循）

```markdown
# 標題
## 第一節
[內容]
## 第二節
[內容]
```

**重要：輸出必須完全遵循此格式，不可變更區段順序或標題。**

```

---

#### 問題 4: Sub-agent 執行太慢

**症狀：**
Sub-agent 回應時間過長。

**可能原因：**
1. 系統提示過長
2. 處理的檔案太大
3. 使用過多工具調用

**解決方案：**

**優化系統提示：**
```json
// ❌ 過長的系統提示（5000+ 字）
{
  "systemPrompt": "[非常詳細但可能過於冗長的提示...]"
}

// ✅ 精簡但充足的系統提示（1000-2000 字）
{
  "systemPrompt": "[核心指示，去除冗餘內容]"
}
```

**限制處理範圍：**

```markdown
## 處理限制
- 單次處理檔案大小不超過 2000 行
- 如果檔案過大，要求使用者指定具體區段
- 優先處理關鍵邏輯
```

---

#### 問題 5: Sub-agent 產生錯誤的建議

**症狀：**
Sub-agent 提供不適用或錯誤的建議。

**可能原因：**

1. 缺少上下文資訊
2. 系統提示不夠明確
3. 溫度參數過高

**解決方案：**

**改善系統提示的上下文收集：**

```markdown
## 分析前檢查

在提供建議前，必須：
1. 讀取相關的配置檔案（package.json, tsconfig.json 等）
2. 了解專案的技術堆疊
3. 檢查現有的程式碼風格
4. 詢問不確定的部分

不要假設或猜測專案配置。
```

**調整溫度：**

```json
{
  "temperature": 0.3  // 對於需要準確性的任務
}
```

---

#### 問題 6: 多個 Sub-agents 衝突

**症狀：**
兩個 sub-agents 都被觸發，造成混亂。

**解決方案：**

**使用更精確的觸發模式：**

```json
// test-generator.json
{
  "autoTrigger": {
    "patterns": [
      "創建測試",
      "generate test",
      "寫測試"
    ],
    "fileTypes": [".test.js", ".spec.js"]
  }
}

// code-reviewer.json
{
  "autoTrigger": {
    "patterns": [
      "審查",
      "review"
    ],
    "excludePatterns": ["測試", "test"]  // 排除測試相關
  }
}
```

**使用優先級：**

```json
{
  "priority": 10,  // 較高的優先級（較低的數字）
  "autoTrigger": {
    "patterns": ["..."]
  }
}
```

---

#### 除錯技巧

**1. 啟用詳細日誌：**

```json
// .claude/config.json
{
  "logging": {
    "level": "debug",
    "subAgents": true
  }
}
```

**2. 測試隔離：**

```bash
# 單獨測試一個 sub-agent
# 暫時移除其他 sub-agents
mv .claude/agents .claude/agents.backup
mkdir .claude/agents
cp .claude/agents.backup/code-reviewer.json .claude/agents/

# 測試完成後恢復
rm -rf .claude/agents
mv .claude/agents.backup .claude/agents
```

**3. 漸進式配置：**

```markdown
步驟：
1. 從最小配置開始
2. 測試基本功能
3. 逐步加入功能
4. 每次加入後測試
5. 記錄每次變更的效果
```

**4. 使用驗證工具：**

```bash
# 建立簡單的驗證腳本
#!/bin/bash
# validate-subagent.sh

echo "驗證 Sub-agent 配置..."

for file in .claude/agents/*.json; do
  echo "檢查 $file..."

  # JSON 格式驗證
  if ! jq empty "$file" 2>/dev/null; then
    echo "❌ JSON 格式錯誤: $file"
    exit 1
  fi

  # 必填欄位檢查
  if ! jq -e '.name' "$file" >/dev/null; then
    echo "❌ 缺少 name 欄位: $file"
    exit 1
  fi

  echo "✅ $file 驗證通過"
done

echo "✅ 所有 Sub-agent 配置驗證通過"
```

---

### 進階主題

#### 1. 動態 Sub-agent 生成

某些情況下，可以動態創建臨時的 sub-agent：

```javascript
// 概念範例（實際實作取決於 Claude Code API）
function createTemporarySubAgent(task) {
  return {
    name: `temp-agent-${Date.now()}`,
    description: `Temporary agent for: ${task}`,
    systemPrompt: generatePromptForTask(task),
    lifetime: "session",  // 僅存在於當前會話
    tools: determineRequiredTools(task)
  };
}
```

#### 2. Sub-agent 協作模式

**並行協作：**

```
任務：審查和優化程式碼

主代理
  ↓
  ├→ code-reviewer（並行執行）
  ├→ security-auditor（並行執行）
  └→ performance-optimizer（並行執行）
  ↓
整合所有結果 → 呈現給使用者
```

**串行協作：**

```
任務：完整的功能開發

1. feature-planner → 規劃功能
2. code-generator → 實作程式碼
3. test-generator → 創建測試
4. code-reviewer → 審查品質
5. doc-writer → 撰寫文件
```

#### 3. 上下文共享

Sub-agents 之間可以共享上下文：

```json
{
  "name": "test-generator",
  "contextFrom": ["code-reviewer"],  // 繼承 code-reviewer 的分析結果
  "systemPrompt": "根據程式碼審查結果創建測試..."
}
```

---

### 總結

Sub-agents 是 Claude Code 中強大的專業化工具，通過合理配置可以大幅提升開發效率：

**關鍵要點：**

1. ✅ 每個 sub-agent 專注於單一領域
2. ✅ 使用清晰、結構化的系統提示
3. ✅ 遵循最小權限原則
4. ✅ 適當設定 temperature 參數
5. ✅ 建立測試和驗證流程
6. ✅ 文件化配置和使用方式

**最佳實踐：**

- 從簡單開始，逐步優化
- 定期審查和更新 sub-agents
- 收集使用反饋並改進
- 與團隊共享有效的 sub-agents
- 建立標準化的命名和配置規範

**下一步：**

- 實作 2-3 個核心 sub-agents
- 測試不同的調用方式
- 根據專案需求客製化
- 探索 sub-agents 之間的協作

---

## 7. Hooks

### 📋 學習摘要

**學習目標：** 使用 Hooks 在特定時機自動執行操作

**核心內容：**

- 9 種 Hook 類型詳解
- hooks.json 配置格式
- 環境變數參考
- 8 個實際應用範例
- 安全考量與最佳實踐

**關鍵技能：**

- ✅ 配置 PreToolUse 和 PostToolUse hooks
- ✅ 使用環境變數傳遞資訊
- ✅ 整合第三方工具（prettier, eslint 等）
- ✅ 建立安全的 hook 腳本

**應用場景：**

- 自動程式碼格式化
- Pre-commit 檢查
- GitButler 整合
- 自動測試執行
- 檔案保護機制

**預計學習時間：** 2 小時

**詳細教學：** `Claude_Code_Hooks_完整教學.md`

### 📖 完整教學內容

### 目錄

1. [什麼是 Hooks](#什麼是-hooks)
2. [Hooks 的用途與優勢](#hooks-的用途與優勢)
3. [Hook 類型詳解](#hook-類型詳解)
4. [配置設定](#配置設定)
5. [環境變數](#環境變數)
6. [實際範例](#實際範例)
7. [最佳實踐](#最佳實踐)
8. [疑難排解](#疑難排解)
9. [安全考量](#安全考量)

---

### 什麼是 Hooks

**Claude Code Hooks** 是使用者自定義的 shell 指令，會在 Claude Code 工作流程的特定時間點自動執行。它們提供了對 AI 代理行為的確定性控制，讓您能夠：

- 在工具執行前後插入自訂邏輯
- 阻擋、修改或觸發額外的處理流程
- 自動化重複性的開發任務
- 確保程式碼品質和一致性

#### 核心概念

Hooks 在預定義的生命週期事件中自動運行，提供對 Claude Code 操作的精確控制。每個 Hook 都是一個 shell 指令，可以檢查、驗證或修改 Claude 的行為。

---

### Hooks 的用途與優勢

#### 主要用途

1. **自動化程式碼格式化**

   - 在檔案修改後自動執行 prettier、eslint 等工具
   - 確保程式碼風格一致性
2. **程式碼品質檢查**

   - 在提交前執行測試
   - 運行靜態分析工具
   - 驗證程式碼規範
3. **工作流程自動化**

   - 自動建立 Git 分支
   - 管理提交訊息
   - 整合 CI/CD 流程
4. **通知與監控**

   - 發送桌面通知
   - 記錄執行的指令
   - 監控 Claude 的操作
5. **安全控制**

   - 阻擋危險指令
   - 保護敏感檔案
   - 驗證權限

#### 關鍵優勢

- **確定性**：提供可預測、可重複的自動化行為
- **靈活性**：支援任何 shell 指令或腳本
- **整合性**：輕鬆整合現有開發工具
- **控制力**：精確控制 Claude 的行為

---

### Hook 類型詳解

Claude Code 提供 9 種不同的 Hook 事件類型：

#### 1. PreToolUse

**執行時機**：在工具被使用之前

**用途**：

- 驗證工具輸入
- 阻擋危險操作
- 修改工具參數
- 執行前置檢查

**特性**：

- 可以阻擋工具執行（使用退出碼 2）
- 可以修改工具輸入
- 適合安全檢查和驗證

**範例場景**：

- 在檔案寫入前檢查權限
- 在 Git commit 前執行測試
- 驗證指令的安全性

#### 2. PostToolUse

**執行時機**：在工具成功執行之後

**用途**：

- 記錄工具執行結果
- 執行後續處理
- 格式化產生的檔案
- 驗證輸出

**特性**：

- 無法阻擋執行（已經完成）
- 可以存取工具的輸出
- 適合後處理和記錄

**範例場景**：

- 自動格式化修改的檔案
- 執行編譯或建置
- 更新文件

#### 3. UserPromptSubmit

**執行時機**：當使用者提交提示詞時

**用途**：

- 驗證和增強提示詞
- 注入額外上下文
- 記錄使用者請求
- 阻擋不當請求

**特性**：

- 可以阻擋提示詞處理
- 可以修改提示詞內容
- 在 Claude 處理前執行

#### 4. Notification

**執行時機**：當 Claude 發送通知時

**用途**：

- 自訂通知處理
- 記錄通知訊息
- 觸發外部系統

**環境變數**：

- `CLAUDE_NOTIFICATION`：通知內容

#### 5. Stop

**執行時機**：當主要代理完成回應時

**用途**：

- 最終驗證
- 完成檢查
- 觸發後續工作流程
- 發送完成通知

**特性**：

- 可以阻止 Claude 停止
- 適合會話結束時的處理

#### 6. SubagentStop

**執行時機**：當子代理完成回應時

**用途**：

- 管理子代理的完成狀態
- 協調多代理工作流程

#### 7. PreCompact

**執行時機**：在上下文壓縮之前

**用途**：

- 保存重要資訊
- 記錄壓縮前的狀態
- 自訂壓縮策略

#### 8. SessionStart

**執行時機**：當會話開始或恢復時

**用途**：

- 初始化環境
- 載入專案配置
- 設定工作目錄

#### 9. SessionEnd

**執行時機**：當會話終止時

**用途**：

- 清理資源
- 保存會話狀態
- 觸發最終操作

---

### 配置設定

#### 配置檔案位置

Claude Code 支援多層級的配置檔案：

1. **全域設定**（套用到所有專案）

   ```
   ~/.claude/settings.json
   ```
2. **專案設定**（與團隊共享）

   ```
   .claude/settings.json
   ```
3. **本地專案設定**（不提交到版本控制）

   ```
   .claude/settings.local.json
   ```

#### 配置優先順序

設定的優先順序從高到低：

1. 企業政策
2. 命令列參數
3. 本地專案設定（`.claude/settings.local.json`）
4. 專案設定（`.claude/settings.json`）
5. 全域設定（`~/.claude/settings.json`）

#### 基本配置結構

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "事件類型": [
      {
        "matcher": "工具名稱或模式",
        "hooks": [
          {
            "type": "command",
            "command": "要執行的指令",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

#### 配置欄位說明

- **事件類型**：Hook 觸發的事件（PreToolUse、PostToolUse 等）
- **matcher**：匹配特定工具的模式（可使用正則表達式）
- **type**：Hook 類型（通常是 "command"）
- **command**：要執行的 shell 指令
- **timeout**：超時時間（秒），選用

#### Matcher 模式

Matcher 支援多種匹配方式：

1. **單一工具**

   ```json
   "matcher": "Bash"
   ```
2. **多個工具（使用正則表達式）**

   ```json
   "matcher": "Edit|Write|MultiEdit"
   ```
3. **所有工具（空字串）**

   ```json
   "matcher": ""
   ```

#### 禁用 Hooks

如果需要暫時禁用所有 Hooks：

```json
{
  "disableAllHooks": true
}
```

---

### 環境變數

Claude Code 在執行 Hooks 時會提供以下環境變數：

#### 主要環境變數

| 變數名稱                | 說明                     | 適用事件                |
| ----------------------- | ------------------------ | ----------------------- |
| `CLAUDE_EVENT_TYPE`   | 觸發的事件類型           | 所有事件                |
| `CLAUDE_TOOL_NAME`    | 使用的工具名稱           | PreToolUse, PostToolUse |
| `CLAUDE_TOOL_INPUT`   | 工具輸入的 JSON          | PreToolUse, PostToolUse |
| `CLAUDE_TOOL_OUTPUT`  | 工具執行的輸出           | PostToolUse             |
| `CLAUDE_FILE_PATHS`   | 相關檔案路徑（空格分隔） | PreToolUse, PostToolUse |
| `CLAUDE_PROJECT_DIR`  | 專案根目錄的絕對路徑     | 所有事件                |
| `CLAUDE_NOTIFICATION` | 通知訊息內容             | Notification            |

#### 使用環境變數範例

##### 1. 解析工具輸入

```bash
##!/bin/bash
## 從 CLAUDE_TOOL_INPUT 提取檔案路徑
FILE_PATH=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path')
echo "正在處理檔案: $FILE_PATH"
```

##### 2. 處理多個檔案

```bash
##!/bin/bash
## 對所有修改的檔案執行 prettier
for file in $CLAUDE_FILE_PATHS; do
  if [[ $file == *.ts ]] || [[ $file == *.tsx ]]; then
    npx prettier --write "$file"
  fi
done
```

##### 3. 檢查專案目錄

```bash
##!/bin/bash
## 確保在正確的專案中執行
if [[ "$CLAUDE_PROJECT_DIR" == "/path/to/my/project" ]]; then
  echo "在正確的專案中"
else
  echo "警告：不在預期的專案目錄中"
  exit 1
fi
```

#### JSON 流程控制

Hooks 可以透過輸出 JSON 來控制 Claude 的行為：

```json
{
  "continue": true,
  "stopReason": "阻擋原因的訊息",
  "suppressOutput": false,
  "systemMessage": "顯示給使用者的警告訊息"
}
```

##### 欄位說明

- **continue**：是否繼續執行（`true` 或 `false`）
- **stopReason**：如果阻擋，顯示的原因
- **suppressOutput**：是否隱藏 stdout 輸出
- **systemMessage**：可選的使用者警告訊息

---

### 實際範例

#### 範例 1：自動格式化 TypeScript 檔案

**場景**：每次修改 TypeScript 檔案後自動執行 Prettier

**配置**（`.claude/settings.json`）：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "for file in $CLAUDE_FILE_PATHS; do if [[ $file == *.ts ]] || [[ $file == *.tsx ]]; then npx prettier --write \"$file\"; fi; done"
          }
        ]
      }
    ]
  }
}
```

**說明**：

- 監聽所有檔案修改操作（Edit、Write、MultiEdit）
- 檢查檔案副檔名
- 對 TypeScript 檔案執行 Prettier

#### 範例 2：Pre-commit 檢查

**場景**：在 Git commit 前執行測試和代碼檢查

**配置**（`.claude/settings.json`）：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.command' | grep -q '^git commit'; then echo '執行 pre-commit 檢查...'; ./claude-hooks/precommit.sh; fi",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

**Pre-commit 腳本**（`./claude-hooks/precommit.sh`）：

```bash
##!/bin/bash
set -e

echo "🔍 執行程式碼檢查..."

## 執行 linter
echo "執行 ESLint..."
npx eslint src/

## 執行測試
echo "執行測試..."
npm test

## 執行型別檢查
echo "執行 TypeScript 型別檢查..."
npx tsc --noEmit

echo "✅ 所有檢查通過！"
exit 0
```

**說明**：

- 監聽 Bash 指令
- 檢查是否是 `git commit` 指令
- 執行測試、linter 和型別檢查
- 如果任何檢查失敗，阻擋 commit

#### 範例 3：GitButler 整合

**場景**：自動管理 Git 分支和提交

**配置**（`~/.claude/settings.json`）：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "but claude pre-tool"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "but claude post-tool"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "but claude stop"
          }
        ]
      }
    ]
  }
}
```

**說明**：

- 使用 GitButler CLI 自動管理分支
- 在檔案修改前後執行 GitButler 指令
- 自動產生提交訊息
- 支援多個並行的 Claude Code 會話

#### 範例 4：完成通知

**場景**：Claude 完成任務時發送桌面通知（macOS）

**配置**（`~/.claude/settings.json`）：

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude 已完成任務！\" with title \"✅ Claude 完成\" sound name \"Glass\"'"
          }
        ]
      }
    ]
  }
}
```

**Linux 版本**（使用 notify-send）：

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send '✅ Claude 完成' 'Claude 已完成任務！'"
          }
        ]
      }
    ]
  }
}
```

#### 範例 5：智慧型指令分派器

**場景**：根據不同條件執行不同的檢查

**配置**（`.claude/settings.json`）：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/smart-hook-dispatcher.sh"
          }
        ]
      }
    ]
  }
}
```

**智慧分派器腳本**（`./scripts/smart-hook-dispatcher.sh`）：

```bash
##!/bin/bash

## 解析工具輸入
TOOL_INPUT="$CLAUDE_TOOL_INPUT"
COMMAND=$(echo "$TOOL_INPUT" | jq -r '.command')

## Git 操作檢查
if echo "$COMMAND" | grep -q '^git commit'; then
    echo "檢測到 git commit，執行 pre-commit 檢查..."
    ./scripts/pre-commit-checks.sh
    exit $?
fi

## 危險指令阻擋
if echo "$COMMAND" | grep -qE '^rm -rf|^sudo rm'; then
    echo "❌ 錯誤：偵測到危險的刪除指令"
    echo '{"continue": false, "stopReason": "阻擋危險的刪除指令"}'
    exit 2
fi

## Docker 操作
if echo "$COMMAND" | grep -q '^docker'; then
    echo "檢測到 Docker 指令，檢查 Docker daemon..."
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker daemon 未執行"
        exit 1
    fi
fi

## 預設：允許執行
exit 0
```

#### 範例 6：自動測試執行

**場景**：修改測試檔案後自動執行相關測試

**配置**（`.claude/settings.json`）：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/auto-test.sh"
          }
        ]
      }
    ]
  }
}
```

**自動測試腳本**（`./scripts/auto-test.sh`）：

```bash
##!/bin/bash

for file in $CLAUDE_FILE_PATHS; do
  # 如果是測試檔案
  if [[ $file == *.test.ts ]] || [[ $file == *.spec.ts ]]; then
    echo "執行測試：$file"
    npx jest "$file"
  fi

  # 如果是原始碼，執行對應的測試
  if [[ $file == src/*.ts ]] && [[ $file != *.test.ts ]]; then
    test_file="${file%.ts}.test.ts"
    if [[ -f "$test_file" ]]; then
      echo "執行對應測試：$test_file"
      npx jest "$test_file"
    fi
  fi
done
```

#### 範例 7：檔案保護

**場景**：防止意外修改重要配置檔案

**配置**（`.claude/settings.json`）：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/protect-files.sh"
          }
        ]
      }
    ]
  }
}
```

**檔案保護腳本**（`./scripts/protect-files.sh`）：

```bash
##!/bin/bash

## 受保護的檔案列表
PROTECTED_FILES=(
  ".env.production"
  "config/production.json"
  "secrets.yaml"
)

## 從工具輸入提取檔案路徑
FILE_PATH=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // .path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

## 檢查是否是受保護的檔案
for protected in "${PROTECTED_FILES[@]}"; do
  if [[ "$FILE_PATH" == *"$protected"* ]]; then
    echo "❌ 錯誤：嘗試修改受保護的檔案：$protected"
    echo '{"continue": false, "stopReason": "此檔案受到保護，無法修改", "systemMessage": "警告：嘗試修改受保護的檔案"}'
    exit 2
  fi
done

exit 0
```

#### 範例 8：程式碼品質報告

**場景**：在會話結束時產生程式碼品質報告

**配置**（`.claude/settings.json`）：

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/quality-report.sh"
          }
        ]
      }
    ]
  }
}
```

**品質報告腳本**（`./scripts/quality-report.sh`）：

```bash
##!/bin/bash

REPORT_FILE="quality-reports/session-$(date +%Y%m%d-%H%M%S).md"
mkdir -p quality-reports

{
  echo "# Claude Code 會話品質報告"
  echo "生成時間：$(date)"
  echo ""

  echo "## 程式碼覆蓋率"
  npm run test:coverage 2>/dev/null || echo "無法取得覆蓋率資訊"
  echo ""

  echo "## ESLint 檢查"
  npx eslint src/ --format json | jq -r '.[] | "檔案：\(.filePath)\n錯誤：\(.errorCount)\n警告：\(.warningCount)\n"' || echo "無 ESLint 問題"
  echo ""

  echo "## TypeScript 錯誤"
  npx tsc --noEmit 2>&1 || echo "無型別錯誤"

} > "$REPORT_FILE"

echo "品質報告已儲存至：$REPORT_FILE"
```

---

### 最佳實踐

#### 1. 組織 Hook 腳本

**建議結構**：

```
your-project/
├── .claude/
│   ├── settings.json
│   └── settings.local.json
├── scripts/ 或 claude-hooks/
│   ├── pre-commit.sh
│   ├── auto-format.sh
│   ├── protect-files.sh
│   └── quality-check.sh
└── ...
```

**好處**：

- 集中管理所有 Hook 腳本
- 便於維護和版本控制
- 易於測試和除錯

#### 2. 使用絕對路徑

**不好的做法**：

```bash
./scripts/check.sh
```

**好的做法**：

```bash
"$CLAUDE_PROJECT_DIR/scripts/check.sh"
```

**原因**：確保無論從哪裡執行都能找到正確的腳本

#### 3. 錯誤處理

**好的做法**：

```bash
##!/bin/bash
set -e  # 遇到錯誤立即退出

## 錯誤處理函數
handle_error() {
  echo "❌ 錯誤：$1"
  exit 1
}

## 檢查必要工具
command -v jq >/dev/null 2>&1 || handle_error "需要安裝 jq"

## 執行操作
npm test || handle_error "測試失敗"

echo "✅ 檢查完成"
exit 0
```

#### 4. 使用退出碼

| 退出碼 | 意義     | 用途                          |
| ------ | -------- | ----------------------------- |
| 0      | 成功     | 允許繼續執行                  |
| 1      | 一般錯誤 | 顯示錯誤但不阻擋              |
| 2      | 阻擋執行 | 停止工具執行（僅 PreToolUse） |

**範例**：

```bash
##!/bin/bash

## 執行檢查
if ! npm test; then
  echo "測試失敗，阻擋提交"
  exit 2  # 阻擋執行
fi

exit 0  # 允許繼續
```

#### 5. 精確的 Matcher

**不好的做法**（過於寬泛）：

```json
{
  "matcher": "",  // 匹配所有工具
  "hooks": [...]
}
```

**好的做法**（精確匹配）：

```json
{
  "matcher": "Edit|Write",  // 只匹配檔案修改
  "hooks": [...]
}
```

#### 6. 設定合理的超時時間

```json
{
  "type": "command",
  "command": "./long-running-task.sh",
  "timeout": 300  // 5 分鐘
}
```

**建議**：

- 快速檢查：30-60 秒
- 測試執行：120-180 秒
- 完整建置：300-600 秒

#### 7. 平行驗證

**好的做法**：

```bash
##!/bin/bash

## 在背景執行多個檢查
npm run lint &
PID1=$!

npm run test:unit &
PID2=$!

npx tsc --noEmit &
PID3=$!

## 等待所有檢查完成
wait $PID1 $PID2 $PID3

## 檢查所有退出碼
if [[ $? -eq 0 ]]; then
  echo "✅ 所有檢查通過"
  exit 0
else
  echo "❌ 有檢查失敗"
  exit 2
fi
```

#### 8. 記錄和監控

**記錄 Hook 執行**：

```bash
##!/bin/bash

LOG_DIR="$CLAUDE_PROJECT_DIR/.logs/hooks"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/$(date +%Y%m%d).log"

{
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Hook 執行：$CLAUDE_EVENT_TYPE"
  echo "工具：$CLAUDE_TOOL_NAME"
  echo "檔案：$CLAUDE_FILE_PATHS"

  # 執行實際操作
  ./scripts/actual-hook.sh

  echo "結果：$?"
  echo "---"
} >> "$LOG_FILE"
```

#### 9. 條件式執行

**根據環境執行不同邏輯**：

```bash
##!/bin/bash

## 檢查環境
if [[ "$NODE_ENV" == "production" ]]; then
  echo "⚠️ 生產環境，執行完整檢查"
  npm run test:full
  npm run build
else
  echo "開發環境，執行快速檢查"
  npm run test:quick
fi
```

#### 10. 漸進式啟用

**階段 1：記錄模式**

```bash
## 只記錄，不阻擋
echo "會執行的檢查：..."
exit 0
```

**階段 2：警告模式**

```bash
## 警告但不阻擋
if ! npm test; then
  echo "⚠️ 警告：測試失敗"
  exit 0  # 仍然允許
fi
```

**階段 3：強制模式**

```bash
## 阻擋執行
if ! npm test; then
  echo "❌ 測試失敗，阻擋執行"
  exit 2  # 阻擋
fi
```

---

### 疑難排解

#### 1. 啟用除錯模式

使用 `--debug` 標誌執行 Claude Code：

```bash
claude --debug
```

這會顯示：

- Hook 的執行時機
- 傳遞的環境變數
- 執行的指令
- 退出碼和輸出

#### 2. 檢查配置語法

**常見錯誤**：

1. **JSON 語法錯誤**

   ```json
   // 錯誤：最後一個元素有逗號
   {
     "hooks": {
       "PreToolUse": [...],  // ❌ 多餘的逗號
     }
   }
   ```
2. **Matcher 格式錯誤**

   ```json
   // 錯誤：使用陣列而非字串
   "matcher": ["Edit", "Write"]  // ❌

   // 正確：使用正則表達式
   "matcher": "Edit|Write"  // ✅
   ```

#### 3. 測試 Hook 腳本

**手動測試**：

```bash
## 設定環境變數
export CLAUDE_TOOL_INPUT='{"file_path": "/path/to/file.ts"}'
export CLAUDE_FILE_PATHS="/path/to/file.ts"
export CLAUDE_PROJECT_DIR="/path/to/project"

## 執行腳本
./scripts/your-hook.sh

## 檢查退出碼
echo $?
```

#### 4. Hook 未執行

**檢查清單**：

1. ✓ 配置檔案位置正確？
2. ✓ JSON 語法有效？
3. ✓ Matcher 模式匹配工具？
4. ✓ 腳本有執行權限？
   ```bash
   chmod +x ./scripts/hook.sh
   ```
5. ✓ 腳本路徑正確？（使用絕對路徑）
6. ✓ Hooks 未被禁用？（檢查 `disableAllHooks`）

#### 5. 環境變數為空

**已知問題**：某些版本的 Claude Code 可能不會正確傳遞環境變數。

**解決方法**：

1. **使用 stdin 讀取**：

   ```bash
   # 從 stdin 讀取工具輸入
   INPUT=$(cat)
   echo "$INPUT" | jq -r '.file_path'
   ```
2. **使用 PostToolUse Hook 除錯**：

   ```json
   {
     "hooks": {
       "PostToolUse": [
         {
           "matcher": "Edit|Write",
           "hooks": [
             {
               "type": "command",
               "command": "env | grep CLAUDE_ > /tmp/claude-env.txt"
             }
           ]
         }
       ]
     }
   }
   ```

#### 6. 超時問題

**症狀**：Hook 被中斷

**解決方法**：

1. **增加超時時間**：

   ```json
   {
     "type": "command",
     "command": "./slow-script.sh",
     "timeout": 600  // 增加到 10 分鐘
   }
   ```
2. **優化腳本執行速度**：

   - 使用快取
   - 平行執行
   - 只檢查必要的部分

#### 7. 權限問題

**症狀**：`Permission denied`

**解決方法**：

```bash
## 賦予執行權限
chmod +x ./scripts/*.sh

## 檢查檔案權限
ls -la ./scripts/
```

#### 8. 查看 Hook 執行日誌

**建立日誌系統**：

```bash
##!/bin/bash

LOG_FILE="$CLAUDE_PROJECT_DIR/.claude/hooks.log"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

log "Hook 開始：$CLAUDE_EVENT_TYPE"
log "工具：$CLAUDE_TOOL_NAME"

## 執行操作
npm test 2>&1 | tee -a "$LOG_FILE"

log "Hook 結束：退出碼 $?"
```

#### 9. 使用 /hooks 指令

在 Claude Code CLI 中，可以使用 `/hooks` 斜線指令來：

- 查看已配置的 Hooks
- 測試 Hook 執行
- 管理 Hook 配置

---

### 安全考量

#### 1. 資料外洩風險

**風險**：Hooks 可以執行任意指令，可能導致敏感資料外洩

**防護措施**：

```bash
##!/bin/bash

## 檢查是否在嘗試存取敏感檔案
SENSITIVE_PATTERNS=(
  ".env"
  "secrets"
  "credentials"
  "private"
  "*.pem"
  "*.key"
)

FILE_PATH=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')

for pattern in "${SENSITIVE_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "❌ 警告：嘗試存取敏感檔案"
    exit 2
  fi
done
```

#### 2. 指令注入防護

**風險**：不當處理輸入可能導致指令注入

**不安全的做法**：

```bash
## ❌ 危險：直接使用未驗證的輸入
eval "$CLAUDE_TOOL_INPUT"
```

**安全的做法**：

```bash
## ✅ 安全：使用 jq 解析 JSON
FILE_PATH=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path')

## 驗證路徑
if [[ ! "$FILE_PATH" =~ ^[a-zA-Z0-9/_.-]+$ ]]; then
  echo "❌ 無效的檔案路徑"
  exit 1
fi
```

#### 3. 限制 Hook 權限

**最小權限原則**：

```bash
##!/bin/bash

## 只允許在專案目錄內操作
if [[ ! "$FILE_PATH" == "$CLAUDE_PROJECT_DIR"* ]]; then
  echo "❌ 錯誤：嘗試存取專案外的檔案"
  exit 2
fi

## 只允許特定操作
ALLOWED_COMMANDS=("git commit" "npm test" "npm run lint")
COMMAND=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.command')

allowed=false
for cmd in "${ALLOWED_COMMANDS[@]}"; do
  if [[ "$COMMAND" == "$cmd"* ]]; then
    allowed=true
    break
  fi
done

if [[ "$allowed" != "true" ]]; then
  echo "❌ 錯誤：不允許的指令"
  exit 2
fi
```

#### 4. 審查 Hook 配置

**檢查清單**：

- [ ] 是否使用了可信的腳本？
- [ ] 是否驗證了所有輸入？
- [ ] 是否限制了檔案存取範圍？
- [ ] 是否記錄了 Hook 執行？
- [ ] 是否有適當的錯誤處理？
- [ ] 是否使用了最小權限？

#### 5. 環境隔離

**使用虛擬環境或容器**：

```bash
##!/bin/bash

## 在 Docker 容器中執行檢查
docker run --rm \
  -v "$CLAUDE_PROJECT_DIR:/app" \
  -w /app \
  node:18 \
  npm test
```

**好處**：

- 隔離執行環境
- 限制資源存取
- 一致的執行環境

#### 6. 敏感資訊處理

**不要在 Hook 中硬編碼敏感資訊**：

```bash
## ❌ 危險
API_KEY="sk-1234567890abcdef"

## ✅ 安全：從環境變數讀取
API_KEY="${SECURE_API_KEY}"

## ✅ 安全：從安全存儲讀取
API_KEY=$(security find-generic-password -a "$USER" -s "my-api-key" -w)
```

#### 7. 網路請求限制

**限制外部網路存取**：

```bash
##!/bin/bash

## 檢查是否有網路請求
if echo "$CLAUDE_TOOL_INPUT" | grep -qE 'https?://'; then
  echo "⚠️ 警告：偵測到網路請求"
  # 記錄或阻擋
fi
```

#### 8. 定期審計

**建立審計日誌**：

```bash
##!/bin/bash

AUDIT_LOG="$CLAUDE_PROJECT_DIR/.claude/audit.log"

{
  echo "時間：$(date)"
  echo "事件：$CLAUDE_EVENT_TYPE"
  echo "使用者：$USER"
  echo "工具：$CLAUDE_TOOL_NAME"
  echo "輸入：$CLAUDE_TOOL_INPUT"
  echo "---"
} >> "$AUDIT_LOG"
```

**定期檢查**：

```bash
## 查看最近的 Hook 執行
tail -n 100 .claude/audit.log

## 搜尋可疑活動
grep "敏感" .claude/audit.log
```

---

### 進階技巧

#### 1. 條件式 Hook

根據不同條件執行不同的 Hook：

```bash
##!/bin/bash

## 根據分支執行不同檢查
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ "$BRANCH" == "main" ]] || [[ "$BRANCH" == "master" ]]; then
  echo "主分支：執行完整檢查"
  npm run test:full
  npm run lint:strict
else
  echo "功能分支：執行快速檢查"
  npm run test:unit
  npm run lint
fi
```

#### 2. Hook 鏈

串連多個 Hook 腳本：

```bash
##!/bin/bash
set -e

echo "1️⃣ 執行代碼格式化..."
./scripts/format.sh

echo "2️⃣ 執行 linter..."
./scripts/lint.sh

echo "3️⃣ 執行測試..."
./scripts/test.sh

echo "4️⃣ 執行建置..."
./scripts/build.sh

echo "✅ 所有步驟完成"
```

#### 3. 快取機制

使用快取加速 Hook 執行：

```bash
##!/bin/bash

CACHE_FILE=".claude/cache/last-check.txt"
CACHE_DURATION=300  # 5 分鐘

## 檢查快取
if [[ -f "$CACHE_FILE" ]]; then
  LAST_CHECK=$(cat "$CACHE_FILE")
  NOW=$(date +%s)
  DIFF=$((NOW - LAST_CHECK))

  if [[ $DIFF -lt $CACHE_DURATION ]]; then
    echo "使用快取結果（${DIFF}秒前檢查過）"
    exit 0
  fi
fi

## 執行實際檢查
npm test

## 更新快取
mkdir -p "$(dirname "$CACHE_FILE")"
date +%s > "$CACHE_FILE"
```

#### 4. 智慧型重試

失敗時自動重試：

```bash
##!/bin/bash

MAX_RETRIES=3
RETRY_DELAY=2

for i in $(seq 1 $MAX_RETRIES); do
  if npm test; then
    echo "✅ 測試通過（第 $i 次嘗試）"
    exit 0
  else
    if [[ $i -lt $MAX_RETRIES ]]; then
      echo "⚠️ 測試失敗，${RETRY_DELAY}秒後重試..."
      sleep $RETRY_DELAY
    fi
  fi
done

echo "❌ 測試失敗（已重試 $MAX_RETRIES 次）"
exit 2
```

#### 5. 漸進式檢查

根據修改範圍調整檢查強度：

```bash
##!/bin/bash

## 計算修改的行數
CHANGED_LINES=$(git diff --cached --numstat | awk '{sum += $1 + $2} END {print sum}')

if [[ $CHANGED_LINES -lt 10 ]]; then
  echo "小型修改：快速檢查"
  npm run test:changed
elif [[ $CHANGED_LINES -lt 100 ]]; then
  echo "中型修改：標準檢查"
  npm run test:unit
else
  echo "大型修改：完整檢查"
  npm run test:full
fi
```

---

### 總結

Claude Code Hooks 是一個強大的功能，讓您能夠：

1. **自動化開發工作流程**：減少重複性任務
2. **確保程式碼品質**：自動執行檢查和驗證
3. **提升團隊協作**：統一開發標準
4. **增強安全性**：阻擋危險操作
5. **客製化 AI 行為**：精確控制 Claude Code 的操作

#### 關鍵要點

- **從簡單開始**：先實作基本的 Hook，再逐步增加複雜度
- **重視安全**：始終驗證輸入，限制權限
- **充分測試**：在啟用前測試所有 Hook 腳本
- **記錄一切**：保持詳細的執行日誌
- **持續優化**：根據使用經驗調整 Hook 配置

#### 推薦學習路徑

1. **階段一**：設定簡單的通知 Hook
2. **階段二**：加入自動格式化 Hook
3. **階段三**：實作 pre-commit 檢查
4. **階段四**：整合完整的 CI/CD 流程
5. **階段五**：建立團隊標準化的 Hook 配置

#### 相關資源

- [Claude Code 官方文件](https://docs.claude.com/en/docs/claude-code/hooks)
- [Hooks 參考](https://docs.claude.com/en/docs/claude-code/hooks-guide)
- [GitButler Hooks 整合](https://docs.gitbutler.com/features/ai-integration/claude-code-hooks)
- [Claude Code 最佳實踐](https://www.anthropic.com/engineering/claude-code-best-practices)

---

### 附錄：完整範例專案結構

```
my-project/
├── .claude/
│   ├── settings.json              # 專案共享設定
│   ├── settings.local.json        # 本地設定（不提交）
│   ├── hooks.log                  # Hook 執行日誌
│   ├── audit.log                  # 安全審計日誌
│   └── cache/                     # Hook 快取
│       └── last-check.txt
│
├── scripts/ 或 claude-hooks/
│   ├── pre-commit.sh              # Pre-commit 檢查
│   ├── auto-format.sh             # 自動格式化
│   ├── auto-test.sh               # 自動測試
│   ├── protect-files.sh           # 檔案保護
│   ├── quality-report.sh          # 品質報告
│   ├── smart-dispatcher.sh        # 智慧分派器
│   └── utils/
│       ├── logger.sh              # 日誌工具
│       └── validator.sh           # 驗證工具
│
├── .gitignore
├── package.json
└── README.md
```

#### 範例 .gitignore

```gitignore
## Claude Code 本地設定
.claude/settings.local.json
.claude/*.log
.claude/cache/

## 其他
node_modules/
dist/
.env
```

---

**最後更新**：2025-10-21
**文件版本**：1.0
**作者**：根據 Claude Code 官方文件整理

---

---

## 8. GitHub Actions

### 📋 學習摘要

**學習目標：** 將 Claude Code 整合到 GitHub Actions 工作流程中

**核心內容：**

- Claude Code Action 的核心功能
- 環境準備與設定步驟
- 8 個常見使用場景
- 完整的 workflow 範例
- 安全性最佳實踐與成本控制

**關鍵技能：**

- ✅ 設定 ANTHROPIC_API_KEY
- ✅ 建立自動化 PR 審查
- ✅ 配置 CLAUDE.md 指南檔案
- ✅ 管理權限與 secrets
- ✅ 優化 API 使用成本

**實用場景：**

- 自動化程式碼審查
- Issue 分類與標籤
- 測試生成
- 文件同步
- 定期維護任務

**預計學習時間：** 2-3 小時

**詳細教學：** [→ 查看完整教學內容](#github-actions-詳細內容)

### 📖 完整教學內容

---

#### 目錄

1. [簡介](#簡介-在-github-actions-中使用-claude-api)
2. [API 設定](#api-設定)
3. [三種調用方式](#三種-api-調用方式)
4. [實用場景](#四個實用場景)
5. [安全與成本](#安全與成本控制)
6. [最佳實踐](#最佳實踐)
7. [疑難排解](#疑難排解)

---

### 簡介：在 GitHub Actions 中使用 Claude API

#### 為什麼使用 Claude API（而非 CLI）

在 GitHub Actions 中，我們**直接調用 Anthropic API**，而不是使用 Claude Code CLI。原因：

- **更穩定**：官方 API 有完整文檔和支援
- **更輕量**：無需安裝 npm 套件
- **更靈活**：可自訂 HTTP 請求和錯誤處理
- **更標準**：適用於任何 CI/CD 環境

#### 核心優勢

| 優勢                 | 說明                                |
| -------------------- | ----------------------------------- |
| **自動化審查** | PR 提交時自動分析程式碼品質和安全性 |
| **智能生成**   | 自動生成測試、文件、PR 描述         |
| **成本控制**   | 僅在必要時觸發，可設定 token 限制   |
| **版本控制**   | 所有 AI 建議都透過 PR 或 Issue 留存 |

#### API 基本資訊

```
API 端點：https://api.anthropic.com/v1/messages
推薦模型：claude-3-5-sonnet-20241022
必要標頭：x-api-key, anthropic-version, content-type
```

#### 前置要求

- GitHub 倉庫（public 或 private）
- Anthropic API 金鑰（從 [console.anthropic.com](https://console.anthropic.com) 取得）
- GitHub Actions 已啟用
- 倉庫 Write 權限（用於建立 PR 評論）

---

### API 設定

#### 步驟 1：取得 API 金鑰

1. 前往 [console.anthropic.com](https://console.anthropic.com)
2. 登入後選擇 **API Keys**
3. 點擊 **Create Key**
4. 複製金鑰（只會顯示一次）

#### 步驟 2：設定 GitHub Secret

1. 前往倉庫 **Settings** → **Secrets and variables** → **Actions**
2. 點擊 **New repository secret**
3. 設定：
   - Name: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-xxxxx...`
4. 點擊 **Add secret**

#### 步驟 3：基本 Workflow 結構

建立 `.github/workflows/claude-api.yml`：

```yaml
name: Claude API Integration

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  claude-task:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Call Claude API
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          curl https://api.anthropic.com/v1/messages \
            -H "content-type: application/json" \
            -H "x-api-key: $ANTHROPIC_API_KEY" \
            -H "anthropic-version: 2023-06-01" \
            -d '{
              "model": "claude-3-5-sonnet-20241022",
              "max_tokens": 1024,
              "messages": [{"role": "user", "content": "Hello, Claude!"}]
            }'
```

#### 環境變數設定

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  CLAUDE_MODEL: claude-3-5-sonnet-20241022
  MAX_TOKENS: 4096
```

---

### 三種 API 調用方式

#### 方式 1：使用 curl（最簡單）

適合簡單任務，無需額外依賴。

```yaml
- name: Review with curl
  run: |
    RESPONSE=$(curl -s https://api.anthropic.com/v1/messages \
      -H "content-type: application/json" \
      -H "x-api-key: ${{ secrets.ANTHROPIC_API_KEY }}" \
      -H "anthropic-version: 2023-06-01" \
      -d '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 2048,
        "messages": [{
          "role": "user",
          "content": "請審查這個 PR 的程式碼變更"
        }]
      }')

    echo "$RESPONSE" | jq -r '.content[0].text' > review.md
```

**優點**：無依賴、快速
**缺點**：JSON 處理較麻煩

#### 方式 2：使用 Python SDK（推薦）

最靈活，適合複雜邏輯。

```yaml
- name: Setup Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Install SDK
  run: pip install anthropic

- name: Run Claude Analysis
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    python << 'EOF'
    import anthropic
    import os

    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": "分析程式碼並提供建議"
        }]
    )

    with open('analysis.md', 'w') as f:
        f.write(message.content[0].text)
    EOF
```

**優點**：類型安全、錯誤處理完善
**缺點**：需安裝 Python 環境

#### 方式 3：使用 Node.js SDK

適合 JavaScript/TypeScript 專案。

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '20'

- name: Install SDK
  run: npm install @anthropic-ai/sdk

- name: Call Claude API
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    node << 'EOF'
    const Anthropic = require('@anthropic-ai/sdk');
    const fs = require('fs');

    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY
    });

    async function analyze() {
      const message = await client.messages.create({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 2048,
        messages: [{
          role: 'user',
          content: '審查程式碼品質'
        }]
      });

      fs.writeFileSync('output.md', message.content[0].text);
    }

    analyze();
    EOF
```

**優點**：與前端專案整合好
**缺點**：需 Node.js 環境

---

### 四個實用場景

#### 場景 1：程式碼審查

自動審查 PR 中的程式碼變更。

```yaml
name: Auto Code Review

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install anthropic

      - name: Get PR diff
        id: diff
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > pr_diff.txt

      - name: Review code
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python << 'EOF'
          import anthropic, os

          with open('pr_diff.txt') as f:
              diff = f.read()

          client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
          message = client.messages.create(
              model="claude-3-5-sonnet-20241022",
              max_tokens=2048,
              messages=[{
                  "role": "user",
                  "content": f"請審查以下程式碼變更，關注：\n1. 程式碼品質\n2. 潛在錯誤\n3. 安全性問題\n\n```diff\n{diff}\n```"
              }]
          )

          with open('review.md', 'w') as f:
              f.write(message.content[0].text)
          EOF

      - name: Post review
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `## 🤖 AI 程式碼審查\n\n${review}`
            });
```

#### 場景 2：自動生成 PR 描述

根據程式碼變更自動生成 PR 描述。

```yaml
name: Generate PR Description

on:
  pull_request:
    types: [opened]

jobs:
  generate-description:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Get changes
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > changes.diff
          git log origin/${{ github.base_ref }}..HEAD --oneline > commits.txt

      - name: Generate description
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          DIFF=$(cat changes.diff)
          COMMITS=$(cat commits.txt)

          curl -s https://api.anthropic.com/v1/messages \
            -H "content-type: application/json" \
            -H "x-api-key: $ANTHROPIC_API_KEY" \
            -H "anthropic-version: 2023-06-01" \
            -d "{
              \"model\": \"claude-3-5-sonnet-20241022\",
              \"max_tokens\": 1024,
              \"messages\": [{
                \"role\": \"user\",
                \"content\": \"根據以下變更生成 PR 描述（包含摘要、變更清單、測試計畫）：\n\nCommits:\n$COMMITS\n\nDiff:\n$DIFF\"
              }]
            }" | jq -r '.content[0].text' > description.md

      - name: Update PR body
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const body = fs.readFileSync('description.md', 'utf8');

            github.rest.pulls.update({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.issue.number,
              body: body
            });
```

#### 場景 3：測試生成

為新程式碼自動生成單元測試。

```yaml
name: Generate Tests

on:
  pull_request:
    paths:
      - 'src/**/*.js'
      - 'src/**/*.ts'

jobs:
  generate-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - run: pip install anthropic

      - name: Find new files
        id: new-files
        run: |
          git diff --name-only --diff-filter=A origin/${{ github.base_ref }}...HEAD \
            | grep -E '\.(js|ts)$' > new_files.txt || true

      - name: Generate tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python << 'EOF'
          import anthropic, os

          client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

          with open('new_files.txt') as f:
              files = [line.strip() for line in f if line.strip()]

          for file in files:
              with open(file) as f:
                  code = f.read()

              message = client.messages.create(
                  model="claude-3-5-sonnet-20241022",
                  max_tokens=2048,
                  messages=[{
                      "role": "user",
                      "content": f"為以下程式碼生成完整的單元測試（使用 Jest）：\n\n```javascript\n{code}\n```"
                  }]
              )

              test_file = file.replace('src/', 'tests/').replace('.js', '.test.js')
              os.makedirs(os.path.dirname(test_file), exist_ok=True)

              with open(test_file, 'w') as f:
                  f.write(message.content[0].text)
          EOF

      - name: Commit tests
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add tests/
          git commit -m "chore: 自動生成測試檔案" || true
          git push
```

#### 場景 4：文件一致性檢查

檢查程式碼與文件是否同步。

```yaml
name: Docs Consistency Check

on:
  pull_request:
    paths:
      - 'src/**'
      - 'docs/**'

jobs:
  check-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Analyze consistency
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          curl -s https://api.anthropic.com/v1/messages \
            -H "content-type: application/json" \
            -H "x-api-key: $ANTHROPIC_API_KEY" \
            -H "anthropic-version: 2023-06-01" \
            -d '{
              "model": "claude-3-5-sonnet-20241022",
              "max_tokens": 1024,
              "messages": [{
                "role": "user",
                "content": "檢查 src/ 中的程式碼變更是否需要更新 docs/ 中的文件。列出不一致之處。"
              }]
            }' | jq -r '.content[0].text' > consistency_report.md

      - name: Post report
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('consistency_report.md', 'utf8');

            if (report.includes('不一致') || report.includes('需要更新')) {
              github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: `## ⚠️ 文件一致性檢查\n\n${report}`
              });
            }
```

---

### 安全與成本控制

#### 保護 API 金鑰

| 最佳實踐               | 說明                          |
| ---------------------- | ----------------------------- |
| **使用 Secrets** | 絕不在程式碼中硬編碼 API 金鑰 |
| **環境隔離**     | 生產環境使用獨立的 API 金鑰   |
| **定期輪換**     | 每 90 天更換一次金鑰          |
| **最小權限**     | 僅授予必要的 workflow 權限    |

```yaml
# 正確做法
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

# 錯誤做法 - 絕不這樣做
env:
  ANTHROPIC_API_KEY: "sk-ant-xxxxx"  # ❌ 危險！
```

#### Token 使用監控

```yaml
- name: Monitor token usage
  run: |
    RESPONSE=$(curl -s https://api.anthropic.com/v1/messages \
      -H "x-api-key: ${{ secrets.ANTHROPIC_API_KEY }}" \
      -H "content-type: application/json" \
      -H "anthropic-version: 2023-06-01" \
      -d '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}]
      }')

    # 記錄 token 使用量
    echo "Input tokens: $(echo $RESPONSE | jq '.usage.input_tokens')"
    echo "Output tokens: $(echo $RESPONSE | jq '.usage.output_tokens')"
```

#### 成本控制策略

```yaml
# 限制執行條件
on:
  pull_request:
    types: [opened]  # 僅新 PR
    paths:
      - 'src/**'     # 僅 src 目錄變更
      - '!**/*.md'   # 排除 Markdown

# 限制 token 數量
-d '{
  "max_tokens": 1024,  # 降低最大 token 數
  "model": "claude-3-5-sonnet-20241022"
}'

# 設定執行頻率
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # 取消重複執行
```

#### Rate Limits

Anthropic API 速率限制：

| 限制類型    | 免費層 | 付費層 |
| ----------- | ------ | ------ |
| 每分鐘請求  | 5      | 50+    |
| 每天 tokens | 10K    | 自訂   |

處理速率限制：

```yaml
- name: Call API with retry
  run: |
    for i in {1..3}; do
      curl https://api.anthropic.com/v1/messages \
        -H "x-api-key: ${{ secrets.ANTHROPIC_API_KEY }}" \
        ... && break || sleep 10
    done
```

#### Prompt Caching（降低成本）

使用 prompt caching 可節省高達 90% 成本：

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "你是專業的程式碼審查員...",  # 會被快取
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": "審查這段程式碼"}]
)
```

---

### 最佳實踐

#### Prompt 設計技巧

- ✅ **具體明確**：「審查安全性問題」比「檢查程式碼」更好
- ✅ **結構化輸出**：要求 JSON 或 Markdown 表格格式
- ✅ **提供上下文**：包含專案規範、程式語言、框架資訊
- ✅ **限制範圍**：僅分析變更的檔案，不是整個倉庫
- ✅ **設定角色**：「你是資深 Python 開發者...」
- ❌ **避免模糊**：不要用「看一下」「檢查一下」
- ❌ **避免過長**：單次請求不超過 100KB 程式碼

範例：

```python
# ✅ 好的 prompt
"""
你是資深 TypeScript 開發者。請審查以下程式碼變更：

要求：
1. 檢查類型安全性
2. 找出潛在的 null/undefined 錯誤
3. 評估效能影響
4. 以 Markdown 表格格式輸出

程式碼：
{code}
"""

# ❌ 不好的 prompt
"看一下這個程式碼有沒有問題"
```

#### 錯誤處理

```yaml
- name: Call Claude API with error handling
  run: |
    RESPONSE=$(curl -s -w "\n%{http_code}" \
      https://api.anthropic.com/v1/messages \
      -H "x-api-key: ${{ secrets.ANTHROPIC_API_KEY }}" \
      -H "content-type: application/json" \
      -H "anthropic-version: 2023-06-01" \
      -d '{"model": "claude-3-5-sonnet-20241022", "max_tokens": 1024, "messages": [...]}')

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" -ne 200 ]; then
      echo "API 錯誤 (HTTP $HTTP_CODE): $BODY"
      exit 1
    fi

    echo "$BODY" | jq -r '.content[0].text'
```

#### 測試 Workflow

```bash
# 使用 act 本地測試（無需推送到 GitHub）
npm install -g act
act pull_request -s ANTHROPIC_API_KEY="sk-ant-xxxxx"

# 僅測試特定 job
act -j review

# 使用 --dryrun 檢查不實際執行
act --dryrun
```

---

### 疑難排解

| 問題                                | 原因                     | 解決方案                                                        |
| ----------------------------------- | ------------------------ | --------------------------------------------------------------- |
| **401 Unauthorized**          | API 金鑰無效             | 檢查 Secret 名稱是否為 `ANTHROPIC_API_KEY`，金鑰是否正確      |
| **429 Too Many Requests**     | 超過速率限制             | 加入重試邏輯，減少呼叫頻率，使用 `concurrency` 限制並行       |
| **500 Internal Server Error** | Anthropic API 暫時性錯誤 | 加入重試機制（等待 10-30 秒後重試）                             |
| **輸出格式錯誤**              | Prompt 不夠明確          | 明確要求輸出格式，如「以 JSON 格式輸出」                        |
| **Workflow 超時**             | 處理的程式碼過大         | 限制 `max_tokens`，僅處理變更的檔案，使用 `timeout-minutes` |

#### 除錯技巧

```yaml
# 啟用詳細日誌
- name: Debug API call
  run: |
    curl -v https://api.anthropic.com/v1/messages \
      -H "x-api-key: ${{ secrets.ANTHROPIC_API_KEY }}" \
      ... 2>&1 | tee debug.log

# 儲存 artifacts 以供檢查
- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: debug-logs
    path: |
      *.log
      *.json
      *.md
```

---

## 9. GitLab CI/CD

### 📋 學習摘要

**學習目標：** 在 GitLab CI/CD 管線中使用 Claude Code

**核心內容：**

- GitLab CI/CD 整合架構
- .gitlab-ci.yml 配置詳解
- 三種身份驗證方式（Claude API, AWS Bedrock, Google Vertex AI）
- 7 個使用場景與完整範例
- 安全性與效能優化

**關鍵技能：**

- ✅ 配置 CI/CD 變數和 secrets
- ✅ 建立多階段 pipeline
- ✅ 使用不同的認證提供商
- ✅ 實作安全的權限管理

**實用場景：**

- 自動程式碼審查
- 測試自動生成
- Issue 轉 MR 工作流程
- 效能分析與優化
- 文件自動生成

**預計學習時間：** 2-3 小時

**詳細教學：** `Claude_Code_GitLab_CICD_教學.md`

### 📖 完整教學內容

1. [簡介](#簡介)
2. [什麼是 Claude Code](#什麼是-claude-code)
3. [整合概述](#整合概述)
4. [環境準備](#環境準備)
5. [基礎設定](#基礎設定)
6. [GitLab CI/CD Pipeline 配置](#gitlab-cicd-pipeline-配置)
7. [環境變數與 Secrets 管理](#環境變數與-secrets-管理)
8. [常見使用場景](#常見使用場景)
9. [完整 Pipeline 範例](#完整-pipeline-範例)
10. [CLAUDE.md 配置指南](#claudemd-配置指南)
11. [權限管理與安全性](#權限管理與安全性)
12. [最佳實踐](#最佳實踐)
13. [常見問題與疑難排解](#常見問題與疑難排解)
14. [進階配置](#進階配置)

---

### 簡介

Claude Code 與 GitLab CI/CD 的整合讓開發團隊能夠在 CI/CD 流程中自動化執行 AI 輔助的程式碼任務，包括程式碼審查、自動化測試、功能實作、錯誤修復等。這項整合目前處於 Beta 階段，由 GitLab 和 Anthropic 共同維護。

#### 核心優勢

- **即時 MR 創建**：描述需求後，Claude 自動提議完整的 Merge Request 及變更說明
- **自動化實作**：將 Issue 轉換為可執行的程式碼，僅需一個命令或提及
- **專案感知**：遵循您的 CLAUDE.md 指南和現有程式碼模式
- **簡易設定**：只需在 `.gitlab-ci.yml` 中新增一個 job 和一個加密的 CI/CD 變數

---

### 什麼是 Claude Code

Claude Code 是 Anthropic 官方推出的 AI 程式設計工具，幫助開發者更快速地將想法轉換為程式碼。

#### 主要功能

- **終端機整合**：直接在終端機中工作
- **檔案編輯**：可直接編輯檔案和執行命令
- **可編程化**：支援腳本化和組合式操作
- **企業級**：提供安全性和隱私保護功能

#### 安裝方式

```bash
npm install -g @anthropic-ai/claude-code
```

#### 系統需求

- Node.js 18 或更新版本
- Claude.ai 或 Claude Console 帳戶
- Anthropic API 金鑰

---

### 整合概述

#### 工作原理

Claude Code 使用 GitLab CI/CD 在隔離的 job 中執行 AI 任務，並通過 Merge Request 將結果提交回去。整個流程包括：

1. **事件驅動編排**：GitLab 監聽您選擇的觸發器（如在 Issue、MR 或審查討論串中提及 `@claude`）
2. **上下文收集**：Job 從討論串和儲存庫收集上下文
3. **提示建構**：根據輸入建構提示
4. **執行 Claude Code**：在沙箱環境中執行 AI 任務

#### 支援的提供商

- **Claude API**：直接從 Anthropic 取得
- **AWS Bedrock**：適合企業環境
- **Google Vertex AI**：適合 Google Cloud 用戶

---

### 環境準備

#### 1. GitLab 環境需求

- GitLab 版本 18.3 或更高（建議使用最新版本以獲得完整功能）
- 具有適當權限的 GitLab Runner
- 專案管理員權限（用於設定 CI/CD 變數）

#### 2. 獲取 Anthropic API 金鑰

1. 前往 [Anthropic Console](https://console.anthropic.com/)
2. 登入您的帳戶
3. 導航到 API Keys 頁面
4. 創建新的 API 金鑰
5. **重要**：立即複製並安全儲存金鑰（離開頁面後將無法再次查看）

#### 3. 驗證 Node.js 安裝

確保您的 GitLab Runner 環境支援 Node.js：

```bash
node --version  # 應顯示 v18.x.x 或更高版本
npm --version
```

---

### 基礎設定

#### 步驟 1：設定 GitLab CI/CD 變數

1. 導航到專案的 **Settings > CI/CD**
2. 展開 **Variables** 部分
3. 點擊 **Add variable**
4. 設定以下變數：

| 變數名稱              | 值            | 類型     | 保護     | 遮罩               |
| --------------------- | ------------- | -------- | -------- | ------------------ |
| `ANTHROPIC_API_KEY` | 您的 API 金鑰 | Variable | 建議啟用 | **必須啟用** |

> **重要提示**：
>
> - **遮罩（Masked）**：必須啟用，防止金鑰在 job 日誌中顯示
> - **保護（Protected）**：建議啟用，限制只有受保護的分支才能使用
> - **絕對不要**將 API 金鑰提交到儲存庫中

#### 步驟 2：建立基本 `.gitlab-ci.yml`

在專案根目錄建立或編輯 `.gitlab-ci.yml` 檔案：

```yaml
stages:
  - ai

claude:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
  before_script:
    - npm install -g @anthropic-ai/claude-code
  script:
    - claude -p "Review and implement changes"
```

#### 步驟 3：測試基本配置

1. 提交 `.gitlab-ci.yml` 到儲存庫
2. 在 GitLab UI 中手動觸發 pipeline（**CI/CD > Pipelines > Run pipeline**）
3. 檢查 job 日誌確認 Claude Code 正確執行

---

### GitLab CI/CD Pipeline 配置

#### 基本配置結構

```yaml
stages:
  - ai

claude:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  variables:
    GIT_STRATEGY: fetch
  before_script:
    - apk update
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - claude -p "${AI_FLOW_INPUT:-'Review this MR and implement the requested changes'}"
      --permission-mode acceptEdits
      --allowedTools "Bash(*) Read(*) Edit(*) Write(*)"
      --debug
  timeout: 30m
  allow_failure: true
```

#### 配置說明

##### Image 選擇

- **node:24-alpine3.21**：輕量級 Alpine Linux 映像，包含 Node.js 24
- 也可使用 `node:20-alpine` 或其他版本

##### Rules（觸發條件）

```yaml
rules:
  - if: '$CI_PIPELINE_SOURCE == "web"'              # 手動觸發
  - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'  # MR 事件
  - if: '$CI_PIPELINE_SOURCE == "push"'             # Push 事件
  - if: '$CI_COMMIT_BRANCH == "main"'               # 特定分支
```

##### Variables（變數）

```yaml
variables:
  GIT_STRATEGY: fetch        # 獲取完整的 git 歷史
  GIT_DEPTH: 0               # 完整 clone（用於需要完整歷史的任務）
  CLAUDE_CODE_DEBUG: "1"     # 啟用除錯模式
```

##### Before Script（前置腳本）

```bash
before_script:
  - apk update                                    # 更新套件列表
  - apk add --no-cache git curl bash              # 安裝必要工具
  - npm install -g @anthropic-ai/claude-code      # 安裝 Claude Code CLI
```

##### Script（主要腳本）

```bash
script:
  - claude -p "Your prompt here"
    --permission-mode acceptEdits
    --allowedTools "Bash(*) Read(*) Edit(*) Write(*)"
    --debug
```

---

### 環境變數與 Secrets 管理

#### Claude API 配置

##### 必要變數

```yaml
## 在 GitLab Settings > CI/CD > Variables 中設定
ANTHROPIC_API_KEY: "sk-ant-..."  # 遮罩此變數
```

##### 可選變數

```yaml
ANTHROPIC_MODEL: "claude-3-7-sonnet-20250219"           # 主要模型
ANTHROPIC_SMALL_FAST_MODEL: "claude-3-5-haiku-20241022" # 快速模型
```

#### AWS Bedrock 配置

適合希望在自有基礎設施上執行的企業環境。

##### 先決條件

1. AWS 帳戶並啟用 Amazon Bedrock
2. 在 AWS IAM 中將 GitLab 配置為 OIDC 身份提供者
3. 具有 Bedrock 權限的 IAM 角色

##### GitLab CI/CD 變數

```yaml
## 在 GitLab Settings > CI/CD > Variables 中設定
AWS_ROLE_TO_ASSUME: "arn:aws:iam::123456789012:role/GitLabClaude"  # 遮罩建議
AWS_REGION: "us-east-1"
CLAUDE_CODE_USE_BEDROCK: "1"
```

##### Pipeline 配置

```yaml
claude_bedrock:
  stage: ai
  image: node:24-alpine3.21
  id_tokens:
    GITLAB_OIDC_TOKEN:
      aud: https://gitlab.com
  before_script:
    - apk add --no-cache git curl bash aws-cli
    - npm install -g @anthropic-ai/claude-code
    # 使用 OIDC 取得臨時憑證
    - |
      export $(printf "AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s" \
      $(aws sts assume-role-with-web-identity \
      --role-arn ${AWS_ROLE_TO_ASSUME} \
      --role-session-name "gitlab-${CI_PROJECT_ID}-${CI_PIPELINE_ID}" \
      --web-identity-token ${GITLAB_OIDC_TOKEN} \
      --duration-seconds 3600 \
      --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' \
      --output text))
  script:
    - claude -p "Your task" --debug
```

##### IAM 角色信任政策

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/gitlab.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "gitlab.com:sub": "project_path:your-group/your-project:ref_type:branch:ref:main"
        }
      }
    }
  ]
}
```

##### IAM 角色權限政策

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-7-sonnet-*",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-haiku-*"
      ]
    }
  ]
}
```

#### Google Vertex AI 配置

##### 必要變數

```yaml
## 在 GitLab Settings > CI/CD > Variables 中設定
ANTHROPIC_VERTEX_PROJECT_ID: "your-gcp-project-id"
CLOUD_ML_REGION: "us-central1"
```

##### 先決條件

1. Google Cloud 專案並啟用 Vertex AI API
2. 設定 Workload Identity Federation
3. 服務帳戶具有必要的 Vertex AI 權限

#### 安全性最佳實踐

##### 1. 變數保護

```yaml
## 所有敏感變數應該：
## - 啟用「Masked」（遮罩）
## - 啟用「Protected」（保護）- 僅限受保護分支
## - 不要在程式碼中 echo 或列印
```

##### 2. 避免洩漏

```bash
## 不要這樣做：
- echo "API Key: $ANTHROPIC_API_KEY"
- claude -p "Debug: $ANTHROPIC_API_KEY"

## 正確做法：
- claude -p "Review code" --debug  # --debug 不會洩漏 secrets
```

##### 3. 最小權限原則

```yaml
## 僅授予必要的權限
script:
  - claude -p "Task"
    --permission-mode plan           # 僅分析，不修改
    --allowedTools "Read(*)"         # 僅讀取權限
```

---

### 常見使用場景

#### 1. 自動程式碼審查

當有新的 Merge Request 時，自動進行程式碼審查並提供建議。

```yaml
code_review:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  before_script:
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      claude -p "請審查這個 MR 的程式碼變更，關注以下方面：
      1. 程式碼品質和可讀性
      2. 潛在的錯誤或邊界條件
      3. 安全性問題
      4. 效能考量
      5. 是否遵循專案的編碼規範
      請提供具體的改進建議。" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(git diff)" \
      --output review_results.txt
  artifacts:
    paths:
      - review_results.txt
    expire_in: 7 days
```

#### 2. 自動化測試生成

為未涵蓋的程式碼自動生成測試。

```yaml
generate_tests:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_COMMIT_BRANCH == "develop"'
      when: manual
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      claude -p "分析專案中的測試覆蓋率，找出未測試的函數和類別，
      然後為它們生成適當的單元測試。
      請遵循專案現有的測試模式和風格。" \
      --permission-mode acceptEdits \
      --allowedTools "Bash(*) Read(*) Edit(*) Write(*)"
  only:
    - schedules
```

#### 3. Issue 轉 MR 工作流程

將 Issue 自動轉換為可執行的 Merge Request。

```yaml
issue_to_mr:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "trigger"'
    - if: '$TRIGGER_TYPE == "issue_mention"'
  before_script:
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      # 從 Issue 描述中提取需求
      ISSUE_DESCRIPTION="${CI_ISSUE_DESCRIPTION}"

      # 創建新分支
      BRANCH_NAME="claude/issue-${CI_ISSUE_IID}"
      git checkout -b "${BRANCH_NAME}"

      # 讓 Claude 實作功能
      claude -p "根據以下 Issue 描述實作功能：
      ${ISSUE_DESCRIPTION}

      請：
      1. 分析需求並設計解決方案
      2. 實作程式碼
      3. 新增或更新相關測試
      4. 更新文件（如需要）" \
      --permission-mode acceptEdits \
      --allowedTools "Bash(*) Read(*) Edit(*) Write(*)"

      # 提交變更
      git add .
      git commit -m "Implement feature from issue #${CI_ISSUE_IID}"
      git push origin "${BRANCH_NAME}"

      # 建立 MR（需要 glab CLI 或 GitLab API）
      glab mr create --title "Resolve issue #${CI_ISSUE_IID}" \
        --description "Auto-generated by Claude Code" \
        --source-branch "${BRANCH_NAME}" \
        --target-branch "main"
```

#### 4. 效能分析與優化

分析程式碼效能並提出優化建議。

```yaml
performance_analysis:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      claude -p "分析專案程式碼的效能瓶頸：
      1. 識別可能的效能問題
      2. 分析資料庫查詢效率
      3. 檢查記憶體使用模式
      4. 評估演算法複雜度
      5. 提供具體優化建議

      請將結果整理成報告。" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(find grep)"
  artifacts:
    reports:
      performance: performance_report.json
```

#### 5. 文件自動生成

根據程式碼自動生成或更新文件。

```yaml
generate_docs:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      changes:
        - src/**/*.{js,ts,py}
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      claude -p "更新專案文件：
      1. 為新增或修改的函數生成 JSDoc/Docstring
      2. 更新 API 文件
      3. 更新 README.md 中的使用範例
      4. 確保文件與程式碼同步" \
      --permission-mode acceptEdits \
      --allowedTools "Read(*) Edit(*) Write(*)"

      # 提交文件更新
      git config user.name "Claude Code Bot"
      git config user.email "claude@example.com"
      git add docs/ README.md
      git commit -m "docs: Update documentation [skip ci]" || true
      git push origin HEAD
```

#### 6. 錯誤修復

自動修復測試失敗或發現的錯誤。

```yaml
auto_fix:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH'
      when: on_failure
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      # 獲取失敗的測試日誌
      FAILED_JOBS=$(curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs?scope=failed")

      claude -p "分析以下測試失敗日誌並修復問題：
      ${FAILED_JOBS}

      請：
      1. 識別失敗原因
      2. 修復程式碼
      3. 確保測試通過" \
      --permission-mode acceptEdits \
      --allowedTools "Bash(*) Read(*) Edit(*)"
  allow_failure: true
```

#### 7. 安全性掃描

掃描程式碼中的潛在安全漏洞。

```yaml
security_scan:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - |
      claude -p "執行安全性掃描：
      1. 檢查 SQL 注入風險
      2. 識別 XSS 漏洞
      3. 檢查不安全的依賴項
      4. 驗證輸入驗證
      5. 檢查敏感資料暴露
      6. 分析身份驗證和授權邏輯

      請提供詳細的安全報告和修復建議。" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(grep find)" \
      --output security_report.md
  artifacts:
    reports:
      security: security_report.md
```

---

### 完整 Pipeline 範例

以下是一個整合多個使用場景的完整 `.gitlab-ci.yml` 範例：

```yaml
## ====================================
## Claude Code GitLab CI/CD Pipeline
## ====================================

stages:
  - validate
  - ai_review
  - ai_test
  - ai_fix
  - deploy

variables:
  GIT_STRATEGY: fetch
  GIT_DEPTH: 0
  CLAUDE_TIMEOUT: "30m"

## ====================================
## 前置驗證
## ====================================

validate_config:
  stage: validate
  image: alpine:latest
  script:
    - echo "Validating CI configuration..."
    - test -n "$ANTHROPIC_API_KEY" || (echo "ANTHROPIC_API_KEY not set" && exit 1)
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'

## ====================================
## AI 程式碼審查
## ====================================

code_review_claude:
  stage: ai_review
  image: node:24-alpine3.21
  needs: ["validate_config"]
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  before_script:
    - apk update
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Starting Claude Code review..."
    - |
      claude -p "請全面審查此 MR 的變更：

      ## 審查重點
      1. 程式碼品質
         - 可讀性和可維護性
         - 遵循 DRY 原則
         - 適當的命名慣例

      2. 潛在問題
         - 邏輯錯誤
         - 邊界條件
         - 錯誤處理

      3. 安全性
         - 輸入驗證
         - SQL 注入風險
         - XSS 漏洞

      4. 效能
         - 演算法效率
         - 資料庫查詢優化
         - 記憶體使用

      5. 測試覆蓋率
         - 是否有足夠的測試
         - 測試品質

      請以結構化的 Markdown 格式提供審查報告。" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(git diff)" \
      --timeout "${CLAUDE_TIMEOUT}" \
      --debug > code_review_report.md
    - cat code_review_report.md
  artifacts:
    paths:
      - code_review_report.md
    reports:
      codequality: code_review_report.md
    expire_in: 30 days
  timeout: 35m
  allow_failure: true

## ====================================
## AI 測試生成
## ====================================

generate_tests_claude:
  stage: ai_test
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      when: manual
    - if: '$CI_COMMIT_BRANCH == "develop"'
      when: manual
  before_script:
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Generating tests with Claude Code..."
    - |
      claude -p "分析專案並生成缺失的測試：

      ## 任務
      1. 掃描 src/ 目錄中的所有程式碼檔案
      2. 識別缺少測試的函數、類別和模組
      3. 為這些未測試的程式碼生成完整的單元測試
      4. 確保測試遵循專案的測試慣例和風格
      5. 包含邊界條件和錯誤情況的測試

      ## 要求
      - 使用專案現有的測試框架
      - 保持一致的測試結構和命名
      - 包含必要的 mock 和 stub
      - 確保測試具有良好的描述性" \
      --permission-mode acceptEdits \
      --allowedTools "Bash(*) Read(*) Edit(*) Write(*)" \
      --timeout "${CLAUDE_TIMEOUT}" \
      --debug
    - echo "Running generated tests..."
    - npm test || echo "Some tests failed, review required"
  artifacts:
    paths:
      - tests/
      - "**/*.test.{js,ts,py}"
    expire_in: 7 days
  timeout: 35m
  allow_failure: true

## ====================================
## AI 錯誤修復
## ====================================

auto_fix_claude:
  stage: ai_fix
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "pipeline"'
      when: on_failure
  before_script:
    - apk add --no-cache git curl bash jq
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Attempting to fix failed jobs with Claude Code..."
    - |
      # 獲取失敗的 job 資訊
      FAILED_JOBS=$(curl --silent --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs?scope=failed" \
        | jq -r '.[].name')

      echo "Failed jobs: $FAILED_JOBS"

      # 獲取失敗 job 的日誌
      for job in $FAILED_JOBS; do
        JOB_ID=$(curl --silent --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
          "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs" \
          | jq -r ".[] | select(.name == \"$job\") | .id" | head -1)

        TRACE=$(curl --silent --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
          "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/jobs/${JOB_ID}/trace")

        echo "Analyzing failure in job: $job"

        claude -p "分析以下 CI job 失敗日誌並嘗試修復：

        Job: $job

        Log:
        $TRACE

        請：
        1. 識別失敗的根本原因
        2. 提出修復方案
        3. 如果可能，直接修復程式碼
        4. 如果是配置問題，說明需要的變更
        5. 提供測試建議以防止類似問題" \
        --permission-mode acceptEdits \
        --allowedTools "Bash(*) Read(*) Edit(*) Write(*)" \
        --timeout "${CLAUDE_TIMEOUT}" \
        --debug
      done
    - |
      # 如果有變更，提交它們
      if [ -n "$(git status --porcelain)" ]; then
        git config user.name "Claude Code Bot"
        git config user.email "claude-bot@ci.gitlab.com"
        git add .
        git commit -m "fix: Auto-fix by Claude Code [skip ci]"
        git push origin HEAD:${CI_COMMIT_REF_NAME}-auto-fix
        echo "Fixes pushed to branch: ${CI_COMMIT_REF_NAME}-auto-fix"
      else
        echo "No changes made by Claude Code"
      fi
  timeout: 35m
  allow_failure: true

## ====================================
## AI 文件生成
## ====================================

update_docs_claude:
  stage: ai_review
  image: node:24-alpine3.21
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      changes:
        - src/**/*
        - lib/**/*
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Updating documentation with Claude Code..."
    - |
      claude -p "更新專案文件以反映程式碼變更：

      ## 任務
      1. 檢查最近的程式碼變更
      2. 為新增或修改的公開 API 生成文件
      3. 更新 README.md 中的範例（如需要）
      4. 更新 API 參考文件
      5. 確保文件的一致性和完整性

      ## 格式
      - 使用 Markdown 格式
      - 包含程式碼範例
      - 提供清晰的說明
      - 註明參數、返回值和異常" \
      --permission-mode acceptEdits \
      --allowedTools "Read(*) Edit(*) Write(*) Bash(git diff)" \
      --timeout "${CLAUDE_TIMEOUT}"
    - |
      # 提交文件變更
      if [ -n "$(git status --porcelain)" ]; then
        git config user.name "Claude Docs Bot"
        git config user.email "claude-docs@ci.gitlab.com"
        git add docs/ README.md API.md
        git commit -m "docs: Update documentation [skip ci]" || true
        git push origin HEAD
      fi
  timeout: 35m
  allow_failure: true

## ====================================
## AI 安全掃描
## ====================================

security_scan_claude:
  stage: ai_review
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
    - if: '$CI_COMMIT_BRANCH == "main"'
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Running security scan with Claude Code..."
    - |
      claude -p "執行全面的安全性掃描：

      ## 掃描範圍
      1. 程式碼漏洞
         - SQL 注入
         - XSS (跨站腳本)
         - CSRF (跨站請求偽造)
         - 命令注入

      2. 身份驗證和授權
         - 弱密碼政策
         - 不安全的 session 管理
         - 缺少授權檢查

      3. 資料保護
         - 敏感資料暴露
         - 不安全的加密
         - 硬編碼的 secrets

      4. 依賴項
         - 已知漏洞的依賴項
         - 過時的套件

      5. 配置
         - 不安全的預設值
         - 除錯模式啟用

      請生成結構化的安全報告，包含：
      - 風險等級 (Critical/High/Medium/Low)
      - 詳細描述
      - 受影響的檔案和行號
      - 修復建議" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(grep find)" \
      --timeout "${CLAUDE_TIMEOUT}" \
      --debug > security_report.md
    - cat security_report.md
  artifacts:
    paths:
      - security_report.md
    reports:
      security: security_report.md
    expire_in: 90 days
  timeout: 35m

## ====================================
## Issue 轉 MR (手動觸發)
## ====================================

issue_to_mr_claude:
  stage: ai_fix
  image: node:24-alpine3.21
  rules:
    - if: '$TRIGGER_SOURCE == "issue_mention"'
      when: manual
  before_script:
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code glab
  script:
    - echo "Creating MR from issue with Claude Code..."
    - |
      # 提取 Issue 資訊
      ISSUE_TITLE=$(echo "$CI_ISSUE_TITLE" | sed 's/[^a-zA-Z0-9]/-/g' | tr '[:upper:]' '[:lower:]')
      BRANCH_NAME="claude/issue-${CI_ISSUE_IID}-${ISSUE_TITLE}"

      # 創建新分支
      git checkout -b "${BRANCH_NAME}"

      # 讓 Claude 實作
      claude -p "根據以下 Issue 實作功能或修復：

      Issue #${CI_ISSUE_IID}: ${CI_ISSUE_TITLE}

      描述:
      ${CI_ISSUE_DESCRIPTION}

      請：
      1. 仔細分析需求
      2. 設計適當的解決方案
      3. 實作程式碼
      4. 新增或更新測試
      5. 更新相關文件
      6. 確保程式碼品質和一致性" \
      --permission-mode acceptEdits \
      --allowedTools "Bash(*) Read(*) Edit(*) Write(*)" \
      --timeout "${CLAUDE_TIMEOUT}"

      # 提交變更
      git config user.name "Claude Code Bot"
      git config user.email "claude@ci.gitlab.com"
      git add .
      git commit -m "feat: Implement #${CI_ISSUE_IID} - ${CI_ISSUE_TITLE}

      Auto-implemented by Claude Code from issue description.

      Closes #${CI_ISSUE_IID}"

      git push origin "${BRANCH_NAME}"

      # 建立 MR
      glab mr create \
        --title "Resolve #${CI_ISSUE_IID}: ${CI_ISSUE_TITLE}" \
        --description "## Summary

      This MR was automatically generated by Claude Code to address issue #${CI_ISSUE_IID}.

      ## Changes

      ${CI_ISSUE_DESCRIPTION}

      ## Testing

      - [ ] Code review
      - [ ] Manual testing
      - [ ] Automated tests pass

      Closes #${CI_ISSUE_IID}

      ---
      Generated by Claude Code" \
        --source-branch "${BRANCH_NAME}" \
        --target-branch "main" \
        --assignee "@me"

      echo "MR created successfully!"
  timeout: 45m
  allow_failure: true

## ====================================
## 效能分析 (排程執行)
## ====================================

performance_analysis_claude:
  stage: ai_review
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "schedule"'
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Running performance analysis with Claude Code..."
    - |
      claude -p "執行深入的效能分析：

      ## 分析重點
      1. 演算法複雜度
         - 識別 O(n²) 或更差的演算法
         - 建議更高效的替代方案

      2. 資料庫
         - N+1 查詢問題
         - 缺少索引
         - 低效的查詢

      3. 記憶體使用
         - 記憶體洩漏
         - 大量物件創建
         - 不必要的資料複製

      4. I/O 操作
         - 阻塞 I/O
         - 批次處理機會

      5. 快取
         - 快取策略評估
         - 快取命中率優化建議

      請提供：
      - 詳細的效能瓶頸報告
      - 程式碼範例和建議改進
      - 預估的效能提升
      - 實作優先順序建議" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(grep find)" \
      --timeout "${CLAUDE_TIMEOUT}" \
      --debug > performance_report.md
    - cat performance_report.md
  artifacts:
    paths:
      - performance_report.md
    expire_in: 90 days
  timeout: 35m

## ====================================
## 部署前檢查
## ====================================

pre_deploy_check_claude:
  stage: deploy
  image: node:24-alpine3.21
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
  before_script:
    - apk add --no-cache git bash
    - npm install -g @anthropic-ai/claude-code
  script:
    - echo "Running pre-deployment checks with Claude Code..."
    - |
      claude -p "執行部署前的全面檢查：

      ## 檢查清單
      1. 程式碼品質
         - 沒有 console.log 或 debug 語句
         - 沒有 TODO 或 FIXME 註解指向關鍵問題
         - 遵循編碼標準

      2. 安全性
         - 沒有硬編碼的 secrets
         - 環境變數正確使用
         - 安全標頭配置

      3. 效能
         - 沒有明顯的效能瓶頸
         - 適當的快取策略

      4. 測試
         - 所有測試通過
         - 關鍵路徑有覆蓋

      5. 文件
         - README 更新
         - CHANGELOG 記錄變更

      6. 依賴項
         - 沒有已知漏洞
         - 版本相容性檢查

      請生成部署就緒報告，標記任何需要注意的問題。" \
      --permission-mode plan \
      --allowedTools "Read(*) Bash(git grep find)" \
      --timeout "${CLAUDE_TIMEOUT}" > deployment_readiness.md
    - cat deployment_readiness.md
    - |
      # 檢查是否有阻斷性問題
      if grep -q "CRITICAL\|BLOCKER" deployment_readiness.md; then
        echo "Critical issues found! Deployment should be postponed."
        exit 1
      else
        echo "Pre-deployment checks passed!"
      fi
  artifacts:
    paths:
      - deployment_readiness.md
    expire_in: 30 days
  timeout: 35m
```

---

### CLAUDE.md 配置指南

`CLAUDE.md` 是一個特殊的 Markdown 檔案，Claude Code 會自動讀取以獲得專案特定的上下文和指示。

#### 什麼是 CLAUDE.md

- 提供專案上下文和指示給 Claude
- 節省時間和 tokens，避免在每次提示中重複相同資訊
- 可以階層式配置（全域、專案、巢狀目錄）

#### 配置層級

1. **企業級** (`/Library/Application Support/ClaudeCode/CLAUDE.md` on macOS)
2. **CLI 參數** (最高優先級)
3. **本地專案** (`.claude/CLAUDE.md` 或根目錄 `CLAUDE.md`)
4. **共享專案** (儲存庫中的 `CLAUDE.md`)
5. **使用者全域** (`~/.claude/CLAUDE.md`)

#### CLAUDE.md 範例

```markdown
## 專案：我的專案名稱

### 概述

這是一個使用 [技術棧] 構建的 [專案類型] 專案。主要目標是 [專案目標]。

### 技術棧

- **後端**: Node.js, Express, PostgreSQL
- **前端**: React, TypeScript, Tailwind CSS
- **測試**: Jest, React Testing Library
- **CI/CD**: GitLab CI/CD
- **部署**: Docker, Kubernetes

### 專案結構

```

src/
  ├── api/          # REST API 端點
  ├── components/   # React 元件
  ├── services/     # 業務邏輯
  ├── models/       # 資料模型
  ├── utils/        # 工具函數
  └── tests/        # 測試檔案

```

### 編碼規範

#### JavaScript/TypeScript

- 使用 ES6+ 模組 (import/export)
- 優先使用解構賦值
- 使用 const/let，避免 var
- 函數使用箭頭函數 (除非需要 this 綁定)
- 優先使用 async/await 而非 .then()

#### 命名慣例

- **檔案**: kebab-case (user-service.js)
- **類別**: PascalCase (UserService)
- **函數/變數**: camelCase (getUserById)
- **常數**: UPPER_SNAKE_CASE (MAX_RETRY_COUNT)

#### 程式碼風格

- 使用 Prettier 格式化
- 縮排: 2 空格
- 分號: 必須使用
- 引號: 單引號 (除非包含單引號)
- 行寬: 100 字元

### 架構原則

#### 設計模式

- **組合優於繼承**: 優先使用組合而非類別繼承
- **依賴注入**: 使用 DI 容器管理依賴
- **SOLID 原則**: 遵循單一職責、開放封閉等原則

#### 錯誤處理

- 總是處理錯誤，不要忽略 catch 區塊
- 使用自訂錯誤類別
- 記錄錯誤但不要洩漏敏感資訊
- 在 API 回應中提供有意義的錯誤訊息

```javascript
// 好的做法
try {
  await userService.createUser(userData);
} catch (error) {
  logger.error('Failed to create user', { error, userId: userData.id });
  throw new UserCreationError('Unable to create user', { cause: error });
}

// 避免
try {
  await userService.createUser(userData);
} catch (e) {
  // 空的 catch 區塊
}
```

### 測試要求

#### 測試覆蓋率

- 最低覆蓋率: 80%
- 關鍵業務邏輯: 100%
- 新功能必須包含測試

#### 測試結構

使用 AAA 模式 (Arrange-Act-Assert):

```javascript
describe('UserService', () => {
  describe('createUser', () => {
    it('should create a new user with valid data', async () => {
      // Arrange
      const userData = { name: 'John Doe', email: 'john@example.com' };

      // Act
      const user = await userService.createUser(userData);

      // Assert
      expect(user).toBeDefined();
      expect(user.email).toBe(userData.email);
    });
  });
});
```

### 安全性要求

#### 必須遵守

- **永不記錄敏感資料**: 密碼、API 金鑰、個人識別資訊
- **輸入驗證**: 所有使用者輸入必須驗證
- **環境變數**: 所有 secrets 必須使用環境變數
- **SQL 注入防護**: 使用參數化查詢或 ORM
- **XSS 防護**: 淨化所有使用者輸入

```javascript
// 好的做法
const user = await db.query(
  'SELECT * FROM users WHERE id = $1',
  [userId]
);

// 避免
const user = await db.query(
  `SELECT * FROM users WHERE id = ${userId}`
);
```

### Git 工作流程

#### 分支策略

- `main`: 生產環境程式碼
- `develop`: 開發分支
- `feature/*`: 新功能
- `bugfix/*`: 錯誤修復
- `hotfix/*`: 緊急修復

#### 提交訊息格式

遵循 Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

類型:

- feat: 新功能
- fix: 錯誤修復
- docs: 文件變更
- style: 程式碼格式 (不影響功能)
- refactor: 重構
- test: 測試
- chore: 建置流程或輔助工具

範例:

```
feat(auth): add JWT authentication

Implement JWT-based authentication for API endpoints.
- Add JWT middleware
- Create token generation service
- Add refresh token support

Closes #123
```

### 資料庫規範

#### 遷移

- 總是使用遷移管理資料庫變更
- 提供 up 和 down 遷移
- 在遷移中包含資料種子 (如需要)

#### 查詢優化

- 為常用查詢欄位建立索引
- 避免 SELECT *，只選擇需要的欄位
- 使用連接而非多次查詢
- 監控慢查詢

### API 設計

#### RESTful 原則

- 使用適當的 HTTP 方法 (GET, POST, PUT, DELETE)
- 資源命名使用複數名詞
- 使用適當的狀態碼
- 版本化 API (/api/v1/...)

#### 回應格式

```json
{
  "success": true,
  "data": {
    "user": {
      "id": 1,
      "name": "John Doe"
    }
  },
  "meta": {
    "timestamp": "2025-01-01T00:00:00Z"
  }
}
```

錯誤回應:

```json
{
  "success": false,
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User with ID 123 not found",
    "details": {}
  }
}
```

### 效能考量

- 實作快取策略 (Redis)
- 使用連線池
- 優化資料庫查詢
- 實作分頁 (大型資料集)
- 使用 CDN 提供靜態資源

### 文件要求

#### 程式碼文件

- 所有公開 API 必須有 JSDoc 註解
- 複雜邏輯需要註解說明
- README 必須保持更新

#### API 文件

- 使用 OpenAPI/Swagger 規格
- 包含範例請求和回應
- 記錄所有錯誤代碼

### CI/CD 流程

#### Pipeline 階段

1. Lint 和格式檢查
2. 單元測試
3. 整合測試
4. 建置
5. 安全掃描
6. 部署 (暫存)
7. 部署 (生產) - 需人工批准

#### 部署前檢查清單

- [ ] 所有測試通過
- [ ] 程式碼審查完成
- [ ] 文件更新
- [ ] CHANGELOG 更新
- [ ] 環境變數配置
- [ ] 資料庫遷移準備好

### 專案特定注意事項

#### 已知問題

- 資料庫連線在高負載下可能超時 → 使用連線池
- 舊版 API 不支援分頁 → 使用 v2 API

#### 開發環境設定

```bash
## 複製環境變數
cp .env.example .env

## 安裝依賴
npm install

## 執行資料庫遷移
npm run migrate

## 啟動開發伺服器
npm run dev
```

#### 常用命令

```bash
## 執行測試
npm test

## 執行 linter
npm run lint

## 格式化程式碼
npm run format

## 建置生產版本
npm run build

## 型別檢查
npm run typecheck
```

### 協作指南

#### Code Review 重點

- 程式碼品質和可讀性
- 測試覆蓋率
- 安全性問題
- 效能影響
- 文件完整性

#### 溝通

- Issue 用於追蹤功能和錯誤
- MR 用於程式碼審查
- 標籤用於分類和優先級

---

### 給 Claude 的特別指示

當使用 Claude Code 時:

1. **程式碼修改前**: 總是先閱讀和理解現有程式碼
2. **遵循規範**: 嚴格遵守上述所有編碼規範和最佳實踐
3. **測試**: 修改程式碼後，更新或新增相應的測試
4. **文件**: 更新受影響的文件
5. **提交訊息**: 使用規定的 Conventional Commits 格式
6. **安全優先**: 在所有決策中優先考慮安全性
7. **解釋變更**: 在進行重大變更時提供清晰的解釋

感謝協助本專案！

```

#### 自訂命令

您可以在 `.claude/commands/` 目錄中創建自訂命令：

##### `.claude/commands/review-security.md`

```markdown
執行全面的安全性審查:

1. 掃描所有程式碼檔案
2. 識別潛在的安全漏洞
3. 檢查:
   - SQL 注入風險
   - XSS 漏洞
   - CSRF 保護
   - 不安全的依賴項
   - 硬編碼的 secrets
4. 提供詳細的安全報告
5. 建議修復措施

請以結構化的 Markdown 格式提供報告。
```

使用: `/review-security`

##### `.claude/commands/optimize-performance.md`

```markdown
分析並優化專案效能:

1. 識別效能瓶頸
2. 檢查:
   - 演算法複雜度
   - 資料庫查詢效率
   - 記憶體使用
   - 網路請求
3. 提供具體優化建議
4. 估算效能提升幅度
5. 實作高優先級優化

請優先處理影響最大的優化項目。
```

使用: `/optimize-performance`

---

### 權限管理與安全性

#### 權限模式

Claude Code 提供四種權限模式:

1. **default** (預設)

   - 允許讀取
   - 其他操作前詢問
2. **plan** (計劃模式)

   - 僅分析，不修改檔案或執行命令
   - 適合程式碼審查和分析
3. **acceptEdits** (接受編輯)

   - 跳過檔案編輯的權限提示
   - 仍會詢問命令執行
4. **bypassPermissions** (繞過權限)

   - 不提示任何權限
   - **僅在完全信任的環境中使用**

#### CLI 使用範例

```bash
## 僅分析模式
claude -p "Review code" --permission-mode plan

## 接受編輯模式
claude -p "Fix bugs" --permission-mode acceptEdits

## 繞過所有權限 (謹慎使用!)
claude -p "Implement feature" --permission-mode bypassPermissions
```

#### 允許的工具 (allowedTools)

限制 Claude 可以使用的工具:

```bash
## 僅允許讀取
claude -p "Analyze code" --allowedTools "Read(*)"

## 允許讀取和編輯
claude -p "Refactor" --allowedTools "Read(*) Edit(*)"

## 允許所有檔案操作和特定 bash 命令
claude -p "Build" --allowedTools "Read(*) Edit(*) Write(*) Bash(npm git)"

## 允許所有操作
claude -p "Full automation" --allowedTools "Bash(*) Read(*) Edit(*) Write(*)"
```

#### 工作區安全性

- **寫入限制**: Claude Code 只能寫入啟動的資料夾及其子資料夾
- **讀取範圍**: 可以讀取工作目錄外的檔案 (用於存取系統函式庫)
- **命令黑名單**: 預設封鎖危險命令 (如 `curl`, `wget`)

#### GitLab CI/CD 中的安全配置

```yaml
## 生產環境 - 嚴格限制
production_review:
  script:
    - claude -p "Review changes"
      --permission-mode plan
      --allowedTools "Read(*)"
  only:
    - main

## 開發環境 - 允許編輯
development_fix:
  script:
    - claude -p "Fix issues"
      --permission-mode acceptEdits
      --allowedTools "Read(*) Edit(*) Bash(npm test)"
  only:
    - develop

## 自動化環境 - 完全自動化
automation_task:
  script:
    - claude -p "Automate task"
      --permission-mode acceptEdits
      --allowedTools "Bash(*) Read(*) Edit(*) Write(*)"
  only:
    - schedules
  when: manual
```

#### 設定檔案層級

```yaml
## .claude/settings.json (專案設定)
{
  "permissionMode": "default",
  "allowedTools": ["Read(*)", "Edit(src/**)", "Bash(npm git)"],
  "blockedCommands": ["rm -rf", "dd", "mkfs"],
  "maxTokens": 8000
}
```

#### 安全最佳實踐

1. **最小權限原則**

   ```yaml
   # 僅授予必要的權限
   script:
     - claude -p "Task" --permission-mode plan  # 預設使用 plan 模式
   ```
2. **工具白名單**

   ```yaml
   # 明確指定允許的工具
   script:
     - claude -p "Task" --allowedTools "Read(*) Edit(src/**)"
   ```
3. **環境隔離**

   ```yaml
   # 在容器中執行
   image: node:24-alpine3.21
   variables:
     GIT_STRATEGY: fetch  # 不要用 clone，避免不必要的歷史
   ```
4. **敏感資料保護**

   ```yaml
   # 絕對不要在提示中包含敏感資料
   script:
     - claude -p "Review code for security issues"  # 好
     - claude -p "Check this API key: ${API_KEY}"    # 壞! 會被記錄
   ```
5. **審計日誌**

   ```yaml
   # 保留 artifacts 用於審計
   artifacts:
     paths:
       - claude_output.log
     expire_in: 90 days
   ```

---

### 最佳實踐

#### 1. Pipeline 設計原則

##### 保持 Job 小而專注

```yaml
## 好的做法 - 每個 job 一個職責
code_review:
  script:
    - claude -p "Review code quality"

security_scan:
  script:
    - claude -p "Scan for security issues"

## 避免 - 單一 job 做太多事
everything:
  script:
    - claude -p "Review, test, fix, and deploy everything"
```

##### 使用適當的觸發條件

```yaml
## MR 時執行審查
code_review:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'

## 主分支變更時更新文件
update_docs:
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      changes:
        - src/**/*

## 排程執行效能分析
performance_analysis:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "schedule"'
```

#### 2. 提示工程最佳實踐

##### 清晰且具體

```yaml
## 好的提示
script:
  - claude -p "審查此 MR 的程式碼變更，關注:
    1. 潛在的 null pointer 錯誤
    2. SQL 注入風險
    3. 效能問題
    請提供具體的修復建議。"

## 模糊的提示
script:
  - claude -p "檢查程式碼"
```

##### 提供上下文

```yaml
script:
  - |
    CONTEXT="專案使用 React 和 TypeScript。
    我們遵循 Airbnb 風格指南。
    這個 MR 新增了使用者身份驗證功能。"

    claude -p "${CONTEXT}

    請審查此 MR 並確保:
    1. 遵循專案編碼規範
    2. 身份驗證邏輯安全
    3. 包含適當的測試"
```

##### 結構化輸出

```yaml
script:
  - claude -p "執行安全掃描並以以下格式提供結果:

    ## 高風險問題
    - [檔案:行號] 問題描述

    ## 中風險問題
    - [檔案:行號] 問題描述

    ## 低風險問題
    - [檔案:行號] 問題描述

    ## 建議
    優先處理順序和修復步驟"
```

#### 3. 成本優化

##### 使用計劃模式進行分析

```yaml
## 計劃模式使用較少 tokens
analysis:
  script:
    - claude -p "Analyze code" --permission-mode plan
```

##### 限制掃描範圍

```yaml
script:
  - |
    # 僅掃描變更的檔案
    CHANGED_FILES=$(git diff --name-only origin/main)
    claude -p "Review these changed files: ${CHANGED_FILES}"
```

##### 使用較小的模型

```yaml
variables:
  ANTHROPIC_SMALL_FAST_MODEL: "claude-3-5-haiku-20241022"

script:
  - claude -p "Simple task"  # 會使用較小的模型
```

#### 4. 快取和效能

##### 快取 Node 模組

```yaml
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - node_modules/
    - .npm/

before_script:
  - npm ci --cache .npm --prefer-offline
```

##### 減少不必要的重新安裝

```yaml
## 使用預安裝 Claude Code 的自訂映像
image: registry.example.com/claude-code:latest

## 或建立自訂映像
## Dockerfile:
## FROM node:24-alpine3.21
## RUN npm install -g @anthropic-ai/claude-code
```

#### 5. 錯誤處理和重試

##### 設定超時

```yaml
claude_task:
  script:
    - claude -p "Long task" --timeout 30m
  timeout: 35m  # GitLab job timeout 略長於 Claude timeout
```

##### 允許失敗但記錄

```yaml
optional_task:
  script:
    - claude -p "Optional optimization"
  allow_failure: true
  artifacts:
    when: always
    paths:
      - results.log
```

##### 重試機制

```yaml
unstable_task:
  script:
    - claude -p "Task that might timeout"
  retry:
    max: 2
    when:
      - runner_system_failure
      - stuck_or_timeout_failure
```

#### 6. 協作和審查流程

##### 要求人工審查

```yaml
auto_fix:
  script:
    - claude -p "Fix bugs" --permission-mode acceptEdits
    - |
      if [ -n "$(git status --porcelain)" ]; then
        git checkout -b "claude/auto-fix-${CI_PIPELINE_ID}"
        git commit -am "Auto-fix by Claude"
        git push origin HEAD
        # 建立 MR 供審查
        glab mr create --draft \
          --title "[Draft] Auto-fix by Claude" \
          --description "Please review these automated fixes"
      fi
```

##### 使用 Draft MR

```yaml
## 自動化變更總是創建 Draft MR
glab mr create --draft
```

##### 標記和標籤

```yaml
script:
  - |
    glab mr create \
      --label "automated" \
      --label "needs-review" \
      --assignee "@tech-lead"
```

#### 7. 監控和觀察

##### 記錄詳細輸出

```yaml
script:
  - claude -p "Task" --debug > claude_debug.log 2>&1
artifacts:
  paths:
    - claude_debug.log
  when: always
```

##### 追蹤 API 使用

```yaml
after_script:
  - |
    echo "Pipeline ID: ${CI_PIPELINE_ID}" >> api_usage.log
    echo "Job: ${CI_JOB_NAME}" >> api_usage.log
    echo "Duration: ${CI_JOB_DURATION}" >> api_usage.log
```

##### 通知和警報

```yaml
claude_task:
  script:
    - claude -p "Critical task"
  after_script:
    - |
      if [ "$CI_JOB_STATUS" == "failed" ]; then
        curl -X POST "https://slack.com/api/chat.postMessage" \
          -H "Authorization: Bearer ${SLACK_TOKEN}" \
          -d "channel=#ci-alerts" \
          -d "text=Claude Code job failed in ${CI_PROJECT_NAME}"
      fi
```

#### 8. 版本控制和穩定性

##### 固定 Claude Code 版本

```yaml
before_script:
  - npm install -g @anthropic-ai/claude-code@1.2.3  # 固定版本
```

##### 測試新版本

```yaml
test_new_version:
  script:
    - npm install -g @anthropic-ai/claude-code@latest
    - claude --version
    - claude -p "Test task"
  when: manual
  allow_failure: true
```

#### 9. 文件和可維護性

##### 註解 Pipeline 配置

```yaml
## ====================================
## AI 程式碼審查階段
##
## 在每個 MR 時執行
## 檢查程式碼品質、安全性和效能
## ====================================
code_review:
  stage: ai_review
  # ... 配置
```

##### 保持 CLAUDE.md 更新

```yaml
validate_docs:
  script:
    - |
      if [ ! -f "CLAUDE.md" ]; then
        echo "Warning: CLAUDE.md not found"
        echo "Claude Code will not have project context"
      fi
```

#### 10. 安全性檢查清單

- [ ] 所有 API 金鑰都使用遮罩變數
- [ ] 保護分支限制 secrets 存取
- [ ] 不在日誌中輸出敏感資料
- [ ] 使用最小權限模式
- [ ] 限制允許的工具
- [ ] 定期輪換 API 金鑰
- [ ] 審計所有自動化變更
- [ ] 在容器中隔離執行
- [ ] 設定適當的超時
- [ ] 記錄所有 Claude Code 活動

---

### 常見問題與疑難排解

#### 常見問題

##### Q1: ANTHROPIC_API_KEY 無效或過期

**症狀:**

```
Error: Invalid API key
```

**解決方案:**

1. 驗證 API 金鑰格式 (應以 `sk-ant-` 開頭)
2. 檢查金鑰是否在 GitLab CI/CD 變數中正確設定
3. 確認金鑰沒有過期
4. 確保變數名稱完全正確 (`ANTHROPIC_API_KEY`)
5. 檢查變數是否為 masked 且可在當前分支/pipeline 中存取

```yaml
## 除錯步驟
script:
  - |
    if [ -z "$ANTHROPIC_API_KEY" ]; then
      echo "ANTHROPIC_API_KEY is not set!"
      exit 1
    fi
    echo "API key is set (length: ${#ANTHROPIC_API_KEY})"
    # 不要 echo 實際的金鑰值!
```

##### Q2: Claude Code 安裝失敗

**症狀:**

```
npm ERR! 404 Not Found - GET https://registry.npmjs.org/@anthropic-ai/claude-code
```

**解決方案:**

1. 確認使用正確的套件名稱
2. 檢查網路連線
3. 嘗試清除 npm 快取

```yaml
before_script:
  - npm cache clean --force
  - npm install -g @anthropic-ai/claude-code --verbose
```

##### Q3: 權限被拒絕錯誤

**症狀:**

```
Error: Permission denied to write file
```

**解決方案:**

1. 檢查 Git 配置
2. 確保有寫入權限
3. 配置 Git 使用者

```yaml
before_script:
  - git config --global user.name "Claude Code Bot"
  - git config --global user.email "claude@ci.gitlab.com"
  - git config --global --add safe.directory "${CI_PROJECT_DIR}"
```

##### Q4: Job 超時

**症狀:**

```
ERROR: Job failed: execution took longer than 1h0m0s seconds
```

**解決方案:**

1. 增加 job timeout
2. 增加 Claude timeout
3. 減少任務範圍

```yaml
claude_task:
  script:
    - claude -p "Task" --timeout 30m
  timeout: 35m  # 比 Claude timeout 稍長
```

##### Q5: AWS Bedrock 認證失敗

**症狀:**

```
Error: Unable to assume role
```

**解決方案:**

1. 驗證 OIDC 配置
2. 檢查 IAM 角色信任政策
3. 確認區域設定正確

```yaml
before_script:
  - |
    echo "Checking AWS configuration..."
    echo "Role: ${AWS_ROLE_TO_ASSUME}"
    echo "Region: ${AWS_REGION}"

    # 驗證 OIDC token
    if [ -z "$GITLAB_OIDC_TOKEN" ]; then
      echo "GITLAB_OIDC_TOKEN not available"
      exit 1
    fi
```

##### Q6: Git 推送失敗

**症狀:**

```
! [rejected]        main -> main (fetch first)
```

**解決方案:**

```yaml
script:
  - |
    # 確保分支是最新的
    git fetch origin
    git rebase origin/${CI_COMMIT_BRANCH}

    # 或創建新分支
    BRANCH="claude/automated-${CI_PIPELINE_ID}"
    git checkout -b "${BRANCH}"
    git push origin "${BRANCH}"
```

##### Q7: MCP 伺服器問題

**症狀:**

```
Error: MCP server not responding
```

**解決方案:**

```yaml
before_script:
  - apk add --no-cache git curl bash
  - npm install -g @anthropic-ai/claude-code
  # 測試 MCP 連線
  - /bin/gitlab-mcp-server --version || true
```

#### 除錯技巧

##### 啟用詳細日誌

```yaml
script:
  - claude -p "Task" --debug --verbose
```

##### 檢查 Claude Code 版本

```yaml
before_script:
  - claude --version
  - npm list -g @anthropic-ai/claude-code
```

##### 驗證環境

```yaml
before_script:
  - node --version
  - npm --version
  - git --version
  - env | grep -E '(ANTHROPIC|CLAUDE|AWS)' | sed 's/=.*/=***/'  # 遮罩值
```

##### 測試基本功能

```yaml
test_claude:
  script:
    - echo "Testing Claude Code basic functionality..."
    - claude -p "Print 'Hello from Claude Code'" --permission-mode plan
  allow_failure: true
```

##### 分離問題

```yaml
## 將複雜 pipeline 分解為簡單測試
test_install:
  script:
    - npm install -g @anthropic-ai/claude-code
    - claude --version

test_auth:
  script:
    - npm install -g @anthropic-ai/claude-code
    - claude -p "Test" --permission-mode plan

test_git:
  script:
    - git config --global user.name "Test"
    - git config --global user.email "test@example.com"
    - git status
```

#### 效能問題

##### Pipeline 執行緩慢

**原因:**

- Claude Code 安裝時間長
- 大型儲存庫 clone 時間
- API 回應時間

**解決方案:**

```yaml
## 1. 使用自訂映像
image: registry.example.com/claude-code:latest

## 2. 淺層 clone
variables:
  GIT_DEPTH: 1

## 3. 快取依賴
cache:
  paths:
    - node_modules/

## 4. 平行執行
stages:
  - ai_parallel

review_code:
  stage: ai_parallel
  script: [...]

scan_security:
  stage: ai_parallel
  script: [...]
```

#### 品質問題

##### Claude 提供的建議不準確

**解決方案:**

1. **改善提示**

   ```yaml
   script:
     - |
       claude -p "請仔細審查程式碼，考慮我們的專案使用:
       - TypeScript with strict mode
       - React 18 with hooks
       - ESLint and Prettier

       專注於型別安全和 React 最佳實踐。"
   ```
2. **提供更多上下文** (使用 CLAUDE.md)
3. **使用更大的模型**

   ```yaml
   variables:
     ANTHROPIC_MODEL: "claude-3-7-sonnet-20250219"
   ```

##### 生成的程式碼不符合規範

**解決方案:**

1. 在 CLAUDE.md 中詳細定義規範
2. 在提示中明確要求遵循規範
3. 添加後續的 lint 和格式化步驟

```yaml
script:
  - claude -p "Implement feature following CLAUDE.md guidelines"
  - npm run lint -- --fix
  - npm run format
```

#### 安全性問題

##### API 金鑰洩漏到日誌

**預防:**

```yaml
## 1. 使用 masked 變數
## 2. 不要 echo 環境變數
## 3. 使用 --debug 而非自訂日誌

## 錯誤做法
script:
  - echo "Using key: $ANTHROPIC_API_KEY"  # 危險!

## 正確做法
script:
  - claude -p "Task" --debug  # Claude 會自動遮罩敏感資訊
```

##### 不安全的程式碼變更

**預防:**

```yaml
## 1. 總是使用 Draft MR
## 2. 要求人工審查
## 3. 執行安全掃描

auto_implement:
  script:
    - claude -p "Implement feature"
    - npm run test
    - npm run security-scan
    - |
      glab mr create --draft \
        --label "automated" \
        --label "security-review-required"
```

#### 整合問題

##### GitLab API 權限問題

**症狀:**

```
Error: 403 Forbidden
```

**解決方案:**

```yaml
## 使用 CI_JOB_TOKEN
script:
  - |
    curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
      "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}"

## 或使用個人存取 token (存在 CI/CD 變數中)
script:
  - |
    curl --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
      "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}"
```

##### glab CLI 問題

**解決方案:**

```yaml
before_script:
  - apk add --no-cache git curl
  # 安裝 glab
  - |
    curl -sSL https://gitlab.com/gitlab-org/cli/-/releases/permalink/latest/downloads/glab_Linux_x86_64.tar.gz \
      | tar -xz -C /usr/local/bin
  - glab --version
  # 配置認證
  - glab auth login --hostname gitlab.com --token $GITLAB_TOKEN
```

#### 獲取幫助

##### 官方資源

- [Claude Code 文件](https://docs.claude.com/en/docs/claude-code)
- [GitLab CI/CD 文件](https://docs.gitlab.com/ee/ci/)
- [Anthropic 支援](https://support.anthropic.com)

##### 社群資源

- [GitLab Forum](https://forum.gitlab.com)
- [Anthropic Discord](https://discord.gg/anthropic)

##### 報告問題

提供以下資訊:

1. GitLab 版本
2. Claude Code 版本
3. 完整的錯誤訊息
4. 相關的 pipeline 配置 (移除敏感資訊)
5. 重現步驟

---

### 進階配置

#### 多環境配置

```yaml
.claude_base:
  image: node:24-alpine3.21
  before_script:
    - apk add --no-cache git curl bash
    - npm install -g @anthropic-ai/claude-code

## 開發環境
claude_dev:
  extends: .claude_base
  variables:
    ENVIRONMENT: "development"
  script:
    - claude -p "Review with relaxed rules"
      --permission-mode acceptEdits
  only:
    - develop

## 生產環境
claude_prod:
  extends: .claude_base
  variables:
    ENVIRONMENT: "production"
  script:
    - claude -p "Strict review for production"
      --permission-mode plan
  only:
    - main
  when: manual
```

#### 條件式 AI 任務

```yaml
smart_review:
  script:
    - |
      # 根據變更大小決定審查深度
      CHANGED_LINES=$(git diff --stat origin/main | tail -1 | awk '{print $4}')

      if [ "$CHANGED_LINES" -gt 500 ]; then
        PROMPT="Large changeset detected. Focus on:
        1. Architecture changes
        2. Breaking changes
        3. Security implications"
      else
        PROMPT="Standard code review focusing on:
        1. Code quality
        2. Test coverage
        3. Documentation"
      fi

      claude -p "$PROMPT"
```

#### 自訂 MCP 伺服器

```yaml
custom_mcp:
  before_script:
    - npm install -g @anthropic-ai/claude-code
    # 設定自訂 MCP 伺服器
    - |
      cat > mcp-config.json <<EOF
      {
        "mcpServers": {
          "gitlab": {
            "command": "/bin/gitlab-mcp-server",
            "env": {
              "GITLAB_TOKEN": "${CI_JOB_TOKEN}",
              "GITLAB_URL": "${CI_SERVER_URL}"
            }
          }
        }
      }
      EOF
  script:
    - claude -p "Task using custom MCP"
      --mcp-config mcp-config.json
```

#### 動態提示生成

```yaml
dynamic_prompt:
  script:
    - |
      # 從 Issue 或 MR 描述生成提示
      ISSUE_LABELS=$(curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}" \
        | jq -r '.labels[]')

      PROMPT="Review this MR. It has the following labels: ${ISSUE_LABELS}."

      if echo "$ISSUE_LABELS" | grep -q "security"; then
        PROMPT="${PROMPT} Pay special attention to security implications."
      fi

      if echo "$ISSUE_LABELS" | grep -q "performance"; then
        PROMPT="${PROMPT} Focus on performance optimizations."
      fi

      claude -p "$PROMPT"
```

#### 結果後處理

```yaml
process_results:
  script:
    - claude -p "Generate report" > raw_report.md
    - |
      # 後處理報告
      python3 <<EOF
      import re

      with open('raw_report.md', 'r') as f:
          content = f.read()

      # 提取關鍵資訊
      issues = re.findall(r'## Issue: (.+)', content)

      # 生成摘要
      with open('summary.md', 'w') as f:
          f.write(f"# Summary\n\n")
          f.write(f"Total issues found: {len(issues)}\n\n")
          for issue in issues:
              f.write(f"- {issue}\n")
      EOF
  artifacts:
    paths:
      - raw_report.md
      - summary.md
```

#### 與其他工具整合

```yaml
integrated_workflow:
  script:
    - |
      # 1. 執行 Claude Code 審查
      claude -p "Review code" > review.md

      # 2. 執行傳統工具
      npm run lint > lint.txt || true
      npm test > test.txt || true

      # 3. 結合結果
      cat > combined_report.md <<EOF
      # Combined Analysis Report

      ## AI Review (Claude Code)
      $(cat review.md)

      ## Linting Results
      \`\`\`
      $(cat lint.txt)
      \`\`\`

      ## Test Results
      \`\`\`
      $(cat test.txt)
      \`\`\`
      EOF
  artifacts:
    paths:
      - combined_report.md
```

---

### 總結

Claude Code 與 GitLab CI/CD 的整合為開發團隊提供了強大的 AI 輔助能力，可以自動化許多耗時的開發任務。通過遵循本指南中的最佳實踐和安全建議，您可以安全、有效地將 AI 整合到您的 CI/CD 流程中。

#### 關鍵要點

1. **安全第一**: 總是使用遮罩變數，遵循最小權限原則
2. **逐步採用**: 從簡單的用例開始，逐步擴展
3. **人工監督**: AI 生成的程式碼應該總是經過人工審查
4. **持續優化**: 監控成本和效能，不斷改進提示和配置
5. **文件化**: 維護 CLAUDE.md 和 pipeline 文件

#### 下一步

1. 設定您的第一個基本 pipeline
2. 創建 CLAUDE.md 檔案
3. 從程式碼審查開始實驗
4. 逐步添加更多自動化
5. 分享經驗並改進流程

#### 資源連結

- [Claude Code 官方文件](https://docs.claude.com/en/docs/claude-code)
- [GitLab CI/CD 文件](https://docs.gitlab.com/ee/ci/)
- [Claude API 文件](https://docs.anthropic.com)
- [GitLab 與 Claude 整合公告](https://about.gitlab.com/blog/gitlab-18-3-expanding-ai-orchestration-in-software-engineering/)

---

**版本**: 1.0
**最後更新**: 2025-01-21
**狀態**: Beta

本教學基於 Claude Code 和 GitLab CI/CD 的最新功能編寫。功能可能會隨著工具的發展而變化。建議定期查看官方文件以獲取最新資訊。

---

## 10. 常見工作流程

### 📋 學習摘要

**學習目標：** 掌握高效的開發工作流程和最佳實踐

**核心內容：**

- 4 種工作模式（Normal, Plan, Auto-Accept, Thinking）
- 8 個常見開發流程詳解
- 專案配置（CLAUDE.md, 自訂命令, 子代理）
- 版本控制整合
- 進階功能（Checkpoints, Hooks, MCP）
- 團隊協作策略

**關鍵技能：**

- ✅ 探索、規劃、編碼、提交工作流程
- ✅ 測試驅動開發 (TDD)
- ✅ Bug 修復與除錯
- ✅ 程式碼重構
- ✅ 新功能開發
- ✅ 視覺化迭代

**核心工作流程：**

1. **探索階段** → 理解現有程式碼
2. **規劃階段** → 制定詳細計畫（Plan Mode）
3. **編碼階段** → 小步驟實作
4. **測試階段** → TDD 方法
5. **提交階段** → 有意義的 commits

**效率提升技巧：**

- 上下文管理
- 批次操作
- 範本和程式碼片段
- 鏈式操作
- 平行工作流程

**預計學習時間：** 4-5 小時

**詳細教學：** [→ 查看完整教學內容](#workflow-詳細內容)

### 📖 完整教學內容

---

## 10.1 四種工作模式

Claude Code 提供四種工作模式，適用於不同開發場景。

### Normal Mode（標準模式）

**定義：** 預設的互動模式，Claude 會在執行操作前請求確認，提供完整的回應和解釋。

**適用場景：**

- 日常開發工作
- 需要理解每個步驟的詳細過程
- 學習新功能或探索程式碼庫
- 需要在執行前檢視變更內容
- 處理敏感或關鍵檔案

**特點：**

- 每個檔案操作都需要確認
- 提供詳細的解釋和上下文
- 可以在執行前預覽變更
- 安全性最高

**使用範例：**

```
You: 幫我重構 auth.ts 中的登入邏輯

Claude: 我會分析 auth.ts 並提出重構建議...
[顯示計畫和變更預覽]
是否要繼續執行？
```

### Plan Mode（計畫模式）

**定義：** Claude 會先制定詳細計畫並展示給你，確認後才執行，適合複雜任務。

**適用場景：**

- 大型重構或架構變更
- 新功能開發（涉及多個檔案）
- 需要評估影響範圍的變更
- 團隊協作前需要設計文件
- 不確定最佳實作方式時

**啟用方式：**

```bash
# 進入 Plan Mode
/plan

# 或在指令中明確要求
/plan 重構整個認證系統
```

**工作流程：**

1. 描述任務需求
2. Claude 生成詳細計畫
3. 檢視並修改計畫
4. 確認後執行
5. 分階段實作

**範例對話：**

```
You: /plan 實作使用者權限系統

Claude: 我將制定計畫...

計畫：使用者權限系統
1. 設計資料結構（roles, permissions）
2. 建立 database schema
3. 實作權限檢查中介層
4. 更新 API endpoints
5. 編寫單元測試
6. 更新文件

是否要執行此計畫？

You: 先執行步驟 1-3

Claude: 開始執行步驟 1-3...
```

### Auto-Accept Mode（自動接受模式）

**定義：** Claude 會自動執行所有操作，不需逐一確認，大幅提升效率。

**適用場景：**

- 批次檔案處理
- 格式化或程式碼整理
- 重複性任務（如更新依賴、修改變數名稱）
- 非關鍵檔案的修改
- 信任 Claude 的熟悉專案

**啟用方式：**

```bash
# 臨時啟用（單次會話）
/auto-accept

# 或在設定中永久啟用
# .claude/settings.json
{
  "autoAccept": true
}
```

**注意事項：**

- ⚠️ 使用前確保有版本控制
- ⚠️ 定期檢視變更
- ⚠️ 不建議用於關鍵檔案
- ⚠️ 可隨時退出模式

**範例：**

```bash
# 批次重新命名變數
You: /auto-accept 將所有檔案中的 getUserData 重新命名為 fetchUserProfile

Claude: [自動執行所有變更，無需確認]
已更新 15 個檔案
- src/api/users.ts
- src/components/Profile.tsx
- ...
```

### Thinking Mode（思考模式）

**定義：** Claude 會展示內部推理過程，幫助理解決策邏輯和問題解決思路。

**適用場景：**

- 學習 Claude 的思考方式
- 除錯複雜問題
- 需要深入理解解決方案
- 審查關鍵決策
- 教學和知識傳遞

**啟用方式：**

```bash
# 啟用 Thinking Mode
/thinking

# 或在提問時明確要求
請展示你的思考過程：如何優化這個查詢？
```

**輸出範例：**

```
You: /thinking 為什麼這個 API 回應很慢？

Claude: [思考過程]
1. 分析 API 端點程式碼...
2. 檢查資料庫查詢...
3. 發現問題：N+1 查詢
4. 思考解決方案：
   - 選項 A: Eager loading
   - 選項 B: 資料庫 join
   - 選項 C: 快取
5. 推薦選項 A，因為...

[執行方案]
...
```

### 模式切換技巧

| 模式        | 啟用指令         | 停用指令     | 快捷鍵 |
| ----------- | ---------------- | ------------ | ------ |
| Plan        | `/plan`        | `/normal`  | 無     |
| Auto-Accept | `/auto-accept` | `/confirm` | 無     |
| Thinking    | `/thinking`    | `/normal`  | 無     |
| Normal      | `/normal`      | -            | 預設   |

**組合使用：**

```bash
# Plan + Thinking：理解複雜計畫
/plan /thinking 重構資料庫層

# Auto-Accept + Plan：快速執行計畫
/plan 更新所有測試檔案的匯入路徑
# 檢視計畫後
/auto-accept
```

---

## 10.2 五階段核心流程

專業開發流程分為五個關鍵階段，每個階段都有明確的目標和實踐方法。

### 階段 1：探索階段

**目標：** 理解專案結構、程式碼庫和相關上下文

**關鍵步驟：**

1. **專案結構掃描**

   ```bash
   請分析這個專案的整體結構
   ```
2. **識別關鍵檔案**

   ```bash
   這個專案的入口點在哪裡？
   哪些檔案負責認證邏輯？
   ```
3. **依賴關係分析**

   ```bash
   分析 UserService 的依賴關係
   ```
4. **程式碼模式識別**

   ```bash
   這個專案使用什麼架構模式？
   ```
5. **文件檢視**

   ```bash
   總結 README 和 ARCHITECTURE.md 的重點
   ```

**實戰範例：**

```
You: 我剛接手這個專案，請幫我快速了解

Claude: 我會進行專案探索...

1. 掃描目錄結構...
   - 使用 Next.js 框架
   - TypeScript 專案
   - 採用 App Router 架構

2. 主要目錄：
   - /app: 路由和頁面
   - /components: 可重用元件
   - /lib: 工具函數和配置
   - /prisma: 資料庫 schema

3. 關鍵檔案：
   - app/layout.tsx: 根佈局
   - lib/auth.ts: 認證邏輯
   - prisma/schema.prisma: 資料模型

4. 技術棧：
   - Next.js 14
   - Prisma ORM
   - NextAuth.js
   - TailwindCSS

需要深入了解哪個部分？
```

**探索技巧：**

- 從 README 開始，理解專案目的
- 檢視 package.json 了解依賴
- 閱讀測試檔案理解預期行為
- 使用 `/tree` 查看目錄結構
- 搜尋 TODO 和 FIXME 註解

### 階段 2：規劃階段

**目標：** 制定清晰、可執行的實作計畫

**關鍵步驟：**

1. **定義需求**

   - 明確功能範圍
   - 識別邊界條件
   - 列出驗收標準
2. **設計方案**

   - 選擇技術方案
   - 規劃檔案結構
   - 定義資料流
3. **拆分任務**

   - 分解為小步驟
   - 排定優先順序
   - 評估風險點
4. **準備資源**

   - 建立必要檔案
   - 準備測試資料
   - 配置環境

**實戰範例：**

```
You: /plan 實作購物車功能

Claude: 制定計畫中...

## 購物車功能實作計畫

### 1. 資料層（30 分鐘）
- [ ] 設計 CartItem schema
- [ ] 建立資料庫遷移
- [ ] 實作 CRUD 函數

### 2. API 層（45 分鐘）
- [ ] POST /api/cart/add
- [ ] DELETE /api/cart/remove
- [ ] GET /api/cart
- [ ] PUT /api/cart/update

### 3. UI 元件（60 分鐘）
- [ ] CartItem 元件
- [ ] CartSummary 元件
- [ ] AddToCart 按鈕
- [ ] 購物車圖示和計數

### 4. 狀態管理（30 分鐘）
- [ ] 建立 CartContext
- [ ] 實作購物車 hooks
- [ ] 同步本地和伺服器狀態

### 5. 測試（45 分鐘）
- [ ] API 單元測試
- [ ] 元件測試
- [ ] E2E 測試

總計：約 3.5 小時

是否開始執行？還是需要調整計畫？
```

**規劃技巧：**

- 使用 Plan Mode (`/plan`)
- 從小到大：先核心功能，再擴展
- 考慮向後相容性
- 預留測試和文件時間
- 與既有架構保持一致

### 階段 3：編碼階段

**目標：** 高品質、可維護的程式碼實作

**關鍵步驟：**

1. **小步驟迭代**

   - 一次實作一個功能點
   - 頻繁測試
   - 及時提交
2. **程式碼品質**

   - 遵循專案規範
   - 適當的註解
   - 錯誤處理
3. **測試驅動**

   - 先寫測試
   - 實作功能
   - 重構優化
4. **持續整合**

   - 定期合併主分支
   - 解決衝突
   - 執行 CI 檢查

**實戰範例：**

```
You: 實作步驟 1：資料層

Claude: 開始實作 CartItem schema...

1. 更新 prisma/schema.prisma
[顯示變更]

2. 建立遷移檔案
```bash
npx prisma migrate dev --name add_cart_items
```

3. 實作 lib/cart/queries.ts
   [顯示程式碼]
4. 執行測試

```bash
npm test lib/cart/queries.test.ts
```

✅ 資料層完成
下一步：實作 API 端點？

```

**編碼技巧：**
- 保持函數簡短（< 30 行）
- 使用有意義的變數名稱
- 避免過早優化
- 遵循 DRY 原則
- 寫自解釋的程式碼

### 階段 4：測試階段

**目標：** 確保功能正確性和穩定性

**關鍵步驟：**

1. **單元測試**
   ```bash
   請為 calculateTotal 函數寫單元測試
```

2. **整合測試**

   ```bash
   測試 API 端點 /api/cart 的完整流程
   ```
3. **E2E 測試**

   ```bash
   寫一個測試：用戶將商品加入購物車並結帳
   ```
4. **手動測試**

   - 測試邊界條件
   - 驗證 UI/UX
   - 跨瀏覽器測試

**實戰範例：**

```
You: 為購物車寫完整測試

Claude: 建立測試套件...

## 1. 單元測試
```typescript
// lib/cart/calculations.test.ts
describe('calculateTotal', () => {
  it('計算單一商品總價', () => {
    const items = [{ price: 100, quantity: 2 }];
    expect(calculateTotal(items)).toBe(200);
  });

  it('處理空購物車', () => {
    expect(calculateTotal([])).toBe(0);
  });
});
```

## 2. API 測試

```typescript
// app/api/cart/route.test.ts
describe('POST /api/cart/add', () => {
  it('成功加入商品', async () => {
    const response = await POST({ productId: '123', quantity: 1 });
    expect(response.status).toBe(200);
  });
});
```

執行測試：

```bash
npm test -- --coverage
```

覆蓋率：92%
是否需要補充測試？

```

**測試技巧：**
- 使用 AAA 模式（Arrange-Act-Assert）
- 測試邊界和異常情況
- 保持測試獨立性
- Mock 外部依賴
- 定期執行完整測試套件

### 階段 5：提交階段

**目標：** 清晰的版本歷史和可追溯性

**關鍵步驟：**

1. **檢視變更**
   ```bash
   git diff
   git status
```

2. **整理提交**

   - 分組相關變更
   - 一個提交一個目的
   - 避免混雜無關修改
3. **撰寫訊息**

   - 使用規範格式
   - 描述「為什麼」
   - 引用相關 issue
4. **程式碼審查**

   - 自我審查
   - 提交 PR
   - 回應反饋

**實戰範例：**

```
You: 請幫我提交購物車功能

Claude: 準備提交...

檢視變更：
- 新增 6 個檔案
- 修改 3 個現有檔案
- 新增 245 行，刪除 12 行

建議分為 3 個提交：

1. feat: 新增購物車資料模型和資料庫 schema
2. feat: 實作購物車 API 端點
3. feat: 新增購物車 UI 元件

是否按此方式提交？

You: 是的

Claude: 執行提交...
```bash
git add prisma/schema.prisma lib/cart/queries.ts
git commit -m "feat: 新增購物車資料模型和資料庫 schema

- 定義 CartItem 和 Cart schema
- 實作 CRUD 函數
- 新增資料庫遷移

Relates to #123"
```

✅ 提交 1/3 完成

```

**提交技巧：**
- 使用 Conventional Commits 格式
- 頻繁提交小變更
- 提交前執行測試
- 避免提交敏感資訊
- 使用有意義的分支名稱

---

## 10.3 六個常見開發流程

### 流程 1：新功能開發

**場景：** 從零開始實作新功能

**完整步驟：**

1. **需求分析**
```

   這個功能需要什麼核心能力？
   有哪些邊界條件？

```

2. **制定計畫**
   ```bash
   /plan 實作 [功能名稱]
```

3. **建立分支**

   ```bash
   git checkout -b feature/user-notifications
   ```
4. **實作核心邏輯**

   - 資料模型
   - 業務邏輯
   - API 端點
5. **實作 UI**

   - 元件設計
   - 狀態管理
   - 使用者互動
6. **撰寫測試**

   - 單元測試
   - 整合測試
   - E2E 測試
7. **提交和審查**

   ```bash
   git add .
   git commit -m "feat: 新增使用者通知系統"
   git push origin feature/user-notifications
   ```

**關鍵指令：**

```bash
# 一次性完整流程
/plan 實作使用者通知系統，包含資料庫、API 和 UI
```

### 流程 2：Bug 修復

**場景：** 發現並修復程式錯誤

**完整步驟：**

1. **重現問題**

   ```
   描述錯誤現象和重現步驟
   ```
2. **定位根因**

   ```
   分析 [檔案名稱] 中可能導致 [錯誤] 的程式碼
   ```
3. **制定修復方案**

   - 最小化影響範圍
   - 考慮副作用
   - 準備測試用例
4. **實作修復**

   ```
   修復 [檔案] 中的 [問題]
   ```
5. **驗證修復**

   - 執行相關測試
   - 手動驗證
   - 回歸測試
6. **防止再次發生**

   - 新增測試覆蓋
   - 改進錯誤處理
   - 更新文件
7. **提交修復**

   ```bash
   git commit -m "fix: 修復登入狀態未正確保存的問題"
   ```

**關鍵指令：**

```bash
# 快速 debug
分析為什麼用戶登入後會被重定向到錯誤頁面

# 追蹤錯誤
在這個錯誤堆疊中定位問題源頭
```

### 流程 3：程式碼重構

**場景：** 改善程式碼結構而不改變功能

**完整步驟：**

1. **識別重構目標**

   - 程式碼異味
   - 重複邏輯
   - 複雜度過高
2. **確保測試覆蓋**

   ```
   為 [模組] 補充測試，確保重構安全
   ```
3. **制定重構計畫**

   ```bash
   /plan 重構 auth 模組，提取重複邏輯並改善可讀性
   ```
4. **小步驟重構**

   - 一次改一個地方
   - 每步都執行測試
   - 保持功能不變
5. **驗證等價性**

   ```bash
   npm test
   npm run build
   ```
6. **優化性能（選用）**

   - Profile 性能
   - 優化瓶頸
   - 再次測試
7. **提交重構**

   ```bash
   git commit -m "refactor: 提取認證邏輯到獨立模組"
   ```

**關鍵指令：**

```bash
# 提取函數
將這段重複的邏輯提取為獨立函數

# 簡化複雜度
重構這個函數，降低循環複雜度
```

### 流程 4：測試驅動開發（TDD）

**場景：** 先寫測試，再實作功能

**完整步驟：**

1. **撰寫失敗測試**

   ```
   為 calculateDiscount 函數寫測試，要求：
   - 折扣 10% 時正確計算
   - 處理無效輸入
   ```
2. **執行測試（確認失敗）**

   ```bash
   npm test -- calculateDiscount.test.ts
   ```
3. **實作最小程式碼**

   ```
   實作 calculateDiscount 使測試通過
   ```
4. **執行測試（確認通過）**

   ```bash
   npm test
   ```
5. **重構改善**

   ```
   重構 calculateDiscount 提升可讀性
   ```
6. **重複循環**

   - 新增更多測試案例
   - 擴展功能
   - 持續重構
7. **完成功能**

   ```bash
   git commit -m "feat: 新增折扣計算功能 (TDD)"
   ```

**關鍵指令：**

```bash
# TDD 循環
請用 TDD 方式實作使用者註冊功能：
1. 先寫測試
2. 實作功能
3. 重構

# 測試先行
為這個 API 端點寫整合測試，然後實作
```

### 流程 5：視覺化迭代

**場景：** 根據視覺設計實作或調整 UI

**完整步驟：**

1. **分析設計稿**

   ```
   分析這個設計稿的佈局結構和元件
   ```
2. **建立元件架構**

   - 識別可重用元件
   - 規劃元件層級
   - 定義 props 介面
3. **實作靜態 UI**

   ```
   根據設計實作 ProductCard 元件
   ```
4. **檢視和調整**

   - 比對設計稿
   - 調整間距、顏色
   - 響應式適配
5. **加入互動**

   - 事件處理
   - 狀態管理
   - 動畫效果
6. **測試多種狀態**

   - 載入狀態
   - 錯誤狀態
   - 空狀態
7. **優化和提交**

   ```bash
   git commit -m "feat: 實作產品卡片元件"
   ```

**關鍵指令：**

```bash
# 從設計稿實作
根據這個 Figma 截圖實作元件

# 視覺除錯
這個按鈕位置不對，應該向右移動 16px
```

### 流程 6：文件撰寫

**場景：** 編寫技術文件和 API 說明

**完整步驟：**

1. **梳理文件結構**

   ```
   為這個專案規劃 README 結構
   ```
2. **撰寫概述**

   - 專案目的
   - 主要功能
   - 技術棧
3. **詳細說明**

   - 安裝步驟
   - 配置選項
   - 使用範例
4. **API 文件**

   ```
   為所有 API 端點生成文件
   ```
5. **程式碼註解**

   - 函數說明
   - 參數描述
   - 回傳值說明
6. **範例和教學**

   - 快速開始
   - 常見用例
   - 疑難排解
7. **維護更新**

   ```bash
   git commit -m "docs: 更新 API 文件"
   ```

**關鍵指令：**

```bash
# 生成文件
為 lib/utils 目錄下的所有函數生成 JSDoc 註解

# 更新 README
根據最新功能更新 README.md
```

---

## 10.4 專案配置

### CLAUDE.md 專案指南

**目的：** 為 Claude 提供專案上下文，提升工作效率

**基本結構：**

```markdown
# 專案名稱

## 專案概述
簡短描述專案目的、主要功能和技術棧。

## 架構說明
- 架構模式（MVC, Clean Architecture 等）
- 主要目錄結構
- 資料流向

## 開發規範
- 程式碼風格（ESLint, Prettier 配置）
- 命名規則
- 提交訊息格式

## 關鍵檔案
- `src/index.ts`: 應用程式入口點
- `lib/db.ts`: 資料庫連接設定
- `config/`: 環境配置

## 常用指令
- `npm run dev`: 啟動開發伺服器
- `npm test`: 執行測試
- `npm run build`: 建置生產版本
```

**進階範例：**

```markdown
# E-Commerce Platform

## 技術棧
- Next.js 14 (App Router)
- TypeScript
- Prisma + PostgreSQL
- NextAuth.js
- TailwindCSS

## 資料夾結構
```

app/
  (auth)/          # 認證相關頁面
  (shop)/          # 商店頁面
  api/             # API 路由
components/
  ui/              # 基礎 UI 元件
  features/        # 功能元件
lib/
  db/              # 資料庫查詢
  utils/           # 工具函數

```

## 開發原則
1. 元件優先使用 Server Components
2. Client Components 必須標註 'use client'
3. API 路由使用 Route Handlers
4. 資料庫查詢集中在 lib/db

## 測試策略
- 單元測試：Jest + React Testing Library
- E2E 測試：Playwright
- API 測試：Supertest

## 環境變數
- `DATABASE_URL`: PostgreSQL 連接字串
- `NEXTAUTH_SECRET`: NextAuth 密鑰
- `STRIPE_SECRET_KEY`: Stripe API 金鑰
```

### 必備章節

| 章節     | 內容                 | 為什麼重要       |
| -------- | -------------------- | ---------------- |
| 專案概述 | 目的、功能、技術棧   | 快速理解專案定位 |
| 架構說明 | 模式、結構、資料流   | 指導設計決策     |
| 開發規範 | 風格、命名、提交格式 | 保持一致性       |
| 關鍵檔案 | 重要檔案及其職責     | 快速定位         |
| 常用指令 | 開發、測試、部署指令 | 提升效率         |

### 配置檔案位置

```
your-project/
├── CLAUDE.md              # 主要專案指南
├── .claude/
│   └── settings.json      # Claude Code 設定
├── docs/
│   ├── ARCHITECTURE.md    # 詳細架構文件
│   └── API.md            # API 說明文件
└── README.md             # 對外說明文件
```

### 子代理（Sub-Agents）使用

**定義：** 建立專門的自訂指令處理特定任務

**建立子代理：**

```bash
# 建立自訂指令目錄
mkdir -p .claude/commands

# 建立測試專用代理
cat > .claude/commands/test.md << 'EOF'
執行完整測試流程：
1. 執行單元測試
2. 執行整合測試
3. 生成覆蓋率報告
4. 檢查覆蓋率是否達到 80%
5. 顯示測試總結
EOF
```

**使用範例：**

```bash
# 呼叫自訂指令
/test

# Claude 會自動執行 test.md 中定義的流程
```

**常見子代理：**

| 代理名稱      | 用途       | 檔案位置                         |
| ------------- | ---------- | -------------------------------- |
| `/review`   | 程式碼審查 | `.claude/commands/review.md`   |
| `/deploy`   | 部署流程   | `.claude/commands/deploy.md`   |
| `/test`     | 測試執行   | `.claude/commands/test.md`     |
| `/refactor` | 重構指南   | `.claude/commands/refactor.md` |

---

## 10.5 效率提升技巧

### 1. 上下文管理

**策略：** 提供充足但精簡的上下文

**技巧：**

- 使用 `@檔案名稱` 引用特定檔案
- 貼上錯誤訊息和堆疊追蹤
- 說明已嘗試的方案
- 提供相關檔案路徑

**範例：**

```
我在 @lib/auth.ts 實作登入功能，但測試失敗。
錯誤訊息：「Cannot read property 'userId' of undefined」
相關檔案：@app/api/login/route.ts
```

### 2. 批次操作

**策略：** 一次處理多個相似任務

**技巧：**

- 列出所有需要變更的檔案
- 使用模式描述重複操作
- 啟用 Auto-Accept 模式

**範例：**

```bash
/auto-accept 在所有 .tsx 檔案中：
1. 移除未使用的 import
2. 新增 TypeScript strict mode 註解
3. 格式化程式碼
```

### 3. 範本和程式碼片段

**策略：** 重複使用常見模式

**建立範本：**

```typescript
// .claude/templates/api-route.ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    // 實作邏輯
    return NextResponse.json({ data: null });
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
```

**使用範例：**

```
使用 .claude/templates/api-route.ts 範本建立 /api/products/route.ts
```

### 4. 鏈式操作

**策略：** 串連多個指令，自動化工作流程

**範例：**

```
請依序執行：
1. 建立 UserProfile 元件
2. 為該元件寫單元測試
3. 在 ProfilePage 中使用該元件
4. 執行測試確認無誤
5. 提交變更
```

### 5. 平行工作流程

**策略：** 同時處理互不依賴的任務

**範例：**

```
請同時：
1. 更新所有測試檔案的 import 路徑
2. 格式化 src/ 目錄下的所有檔案
3. 生成 API 文件
```

---

## 10.6 最佳實踐

### Do's（推薦做法）

| 做法              | 說明               | 範例                                                 |
| ----------------- | ------------------ | ---------------------------------------------------- |
| ✅ 明確描述需求   | 清楚說明目標和限制 | 「建立一個 API 端點回傳使用者列表，支援分頁和搜尋」  |
| ✅ 提供充足上下文 | 引用相關檔案和錯誤 | 「@lib/db.ts 中的查詢很慢，錯誤：timeout after 30s」 |
| ✅ 小步驟迭代     | 分階段實作和測試   | 「先實作資料模型，測試後再做 API」                   |
| ✅ 使用版本控制   | 頻繁提交，清楚訊息 | 每個功能點提交一次                                   |
| ✅ 撰寫測試       | 確保程式碼品質     | 每個函數至少一個測試                                 |
| ✅ 審查變更       | 理解每個修改       | 使用 `git diff` 檢視                               |
| ✅ 維護 CLAUDE.md | 保持專案指南最新   | 新增功能後更新文件                                   |

### Don'ts（避免做法）

| 做法                    | 問題                | 改善方式             |
| ----------------------- | ------------------- | -------------------- |
| ❌ 模糊的指令           | Claude 難以理解意圖 | 具體說明目標和步驟   |
| ❌ 忽略錯誤訊息         | 無法定位問題        | 完整貼上錯誤和堆疊   |
| ❌ 一次改太多           | 難以除錯和回溯      | 分解為小任務         |
| ❌ 跳過測試             | 引入潛在 bug        | 實作功能時一併寫測試 |
| ❌ 盲目接受建議         | 可能不符合需求      | 理解後再執行         |
| ❌ 混雜無關變更         | 污染提交歷史        | 一個提交一個目的     |
| ❌ 過度使用 Auto-Accept | 失去控制和理解      | 僅用於重複性任務     |

### 效率對比

| 場景       | 低效做法         | 高效做法                            |
| ---------- | ---------------- | ----------------------------------- |
| 新功能開發 | 直接開始寫程式碼 | `/plan` → 審查計畫 → 分階段實作 |
| Bug 修復   | 「修這個 bug」   | 提供錯誤訊息、重現步驟、相關檔案    |
| 批次修改   | 逐一確認每個檔案 | `/auto-accept` + 明確模式描述     |
| 程式碼審查 | 手動檢查每個檔案 | 建立 `/review` 子代理自動化流程   |
| 測試執行   | 手動執行多個指令 | 鏈式操作一次完成                    |

### 團隊協作最佳實踐

1. **統一 CLAUDE.md**：團隊共用專案指南
2. **共享子代理**：提交 `.claude/commands/` 到版本控制
3. **規範提交訊息**：使用 Conventional Commits
4. **程式碼審查**：讓 Claude 輔助 PR 審查
5. **文件同步**：功能變更時更新相關文件

### 安全注意事項

| 風險                 | 防範措施                          |
| -------------------- | --------------------------------- |
| 意外提交敏感資訊     | 使用 `.gitignore`，審查提交內容 |
| 過度依賴 Auto-Accept | 僅用於非關鍵檔案，定期審查        |
| 忽略測試失敗         | 建立 pre-commit hook 強制測試     |
| 不理解變更內容       | 使用 Thinking Mode 理解決策       |

---

---

## 11. Claude Code on the Web

### 📋 學習摘要

**學習目標：** 使用 Claude Code 的 Web 版本進行雲端開發

**核心內容：**

- Web 版本的核心功能與特性
- Teleport 功能（Web 獨有）
- 與 CLI 版本的功能對比
- 使用指南與最佳實踐
- 定價方案（Pro, Max 5x, Max 20x）
- 進階技巧與疑難排解

**關鍵技能：**

- ✅ 連接 GitHub repository
- ✅ 配置雲端執行環境
- ✅ 使用 Teleport 功能
- ✅ 監控和引導 Claude 的工作
- ✅ 管理多個 sessions

**Web 版本獨有功能：**

- 雲端執行環境（無需本地設定）
- Teleport（即時連接到 Claude 正在處理的程式碼）
- iOS 行動端支援
- 視覺化進度追蹤
- 自動 PR 建立

**適合對象：**

- 需要快速開始的新手
- 使用多台電腦的開發者
- 需要行動端存取的使用者
- 團隊協作場景

**預計學習時間：** 1-2 小時

**詳細教學：** `Claude_Code_Web_教學.md`

### 📖 完整教學內容

1. [簡介](#簡介)
2. [什麼是 Claude Code Web 版本](#什麼是-claude-code-web-版本)
3. [核心功能與特性](#核心功能與特性)
4. [Web 版本 vs CLI 版本比較](#web-版本-vs-cli-版本比較)
5. [如何存取和使用](#如何存取和使用)
6. [環境配置](#環境配置)
7. [使用場景與最佳實踐](#使用場景與最佳實踐)
8. [定價方案與限制](#定價方案與限制)
9. [常見問題與疑難排解](#常見問題與疑難排解)
10. [進階技巧](#進階技巧)

---

### 簡介

Claude Code on the Web 是 Anthropic 於 2024 年 10 月推出的雲端程式碼助理服務，讓開發者能夠直接在瀏覽器中執行 Claude Code 工作流程，無需開啟終端機。這項服務目前處於研究預覽階段，適用於 Pro 和 Max 訂閱用戶。

#### 主要優勢

- **無需本地環境**：在雲端執行，不佔用本地資源
- **並行處理**：同時執行多個編碼任務
- **行動支援**：透過 iOS App 隨時隨地編碼
- **自動化工作流程**：自動建立分支和 Pull Request
- **安全隔離**：每個任務在獨立沙箱環境中執行

---

### 什麼是 Claude Code Web 版本

#### 核心概念

Claude Code Web 版本是一個雲端託管的程式碼助理平台，讓開發者能夠：

1. **連接 GitHub 儲存庫**：透過 OAuth 安全連接
2. **非同步執行任務**：在 Anthropic 管理的虛擬機器上執行
3. **即時監控進度**：透過網頁介面追蹤任務狀態
4. **靈活引導**：在執行過程中隨時調整方向

#### 技術架構

```
使用者請求 → Web 介面 → GitHub 連接 → 雲端 VM
                                          ↓
                                    執行環境初始化
                                          ↓
                                    Clone 儲存庫
                                          ↓
                                    設定網路存取
                                          ↓
                                    執行編碼任務
                                          ↓
                                    推送分支 → PR
```

#### 執行環境

Claude Code Web 使用預先配置的通用映像檔，包含：

**程式語言**

- Python
- Node.js
- Java
- Go
- Rust

**建置工具與套件管理器**

- npm, yarn, pip, Maven, Gradle
- cargo, go mod

**測試框架與 Linters**

- pytest, Jest, JUnit
- ESLint, Pylint, rustfmt

---

### 核心功能與特性

#### 1. 雲端執行環境

##### 隔離沙箱

- 每個任務在獨立的虛擬機器中執行
- 預設限制網路存取
- 檔案系統隔離保護

##### 安全憑證處理

- 透過安全代理服務處理 Git 互動
- 不長期儲存 GitHub 憑證
- 僅存取授權的儲存庫

#### 2. Teleport 功能

Teleport 是 Web 版本的獨特功能，允許：

- **複製對話記錄**：將整個聊天歷史傳輸到本地
- **同步檔案變更**：將編輯過的檔案下載到本地 CLI
- **無縫切換**：在 Web 和本地環境間自由轉換

**使用範例**：

```
1. 在 Web 介面啟動任務
2. Claude 進行初步實作
3. 點擊 "Teleport to CLI"
4. 在本地繼續深度開發
```

#### 3. 即時進度追蹤

Web 介面提供：

- 任務執行狀態視覺化
- 檔案變更即時預覽
- 錯誤和警告即時顯示
- 測試執行結果

#### 4. 多工並行處理

**優勢**：

- 同時處理多個 bug 修復
- 跨不同儲存庫並行工作
- 獨立的環境隔離
- 統一的介面管理

**使用情境**：

```
Session 1: 修復登入 bug (repo-frontend)
Session 2: 撰寫 API 測試 (repo-backend)
Session 3: 更新文件 (repo-docs)
```

#### 5. 自動 PR 建立

工作流程：

1. Claude 完成任務
2. 自動建立新分支
3. 提交所有變更
4. 開啟 Pull Request
5. 提供清晰的變更摘要

#### 6. 行動端支援 (iOS)

**iOS App 功能**（早期預覽版）：

- 查看程式碼
- 執行小型任務
- 檢查進行中的任務狀態
- 回應和引導 Claude

---

### Web 版本 vs CLI 版本比較

#### 功能對比表

| 功能                    | Web 版本         | CLI 版本         |
| ----------------------- | ---------------- | ---------------- |
| **執行環境**      | 雲端 VM          | 本地機器         |
| **資源消耗**      | 零本地資源       | 使用本地 CPU/RAM |
| **並行任務**      | 支援多工並行     | 一次一個任務     |
| **存取方式**      | 瀏覽器 + iOS App | 終端機           |
| **GitHub 整合**   | 僅支援 GitHub    | 支援多種平台     |
| **網路配置**      | 預配置環境       | 完全自訂         |
| **Teleport 功能** | 支援             | 不適用           |
| **檔案系統存取**  | 限制存取         | 完整存取         |
| **即時互動**      | 佇列化提示       | 即時互動         |
| **權限管理**      | 自動跳過         | 需手動確認       |

#### 核心差異

##### 執行模式

**Web 版本**：

- 在容器化環境中執行
- 相當於 `claude --dangerously-skip-permissions`
- 適合非同步、長時間執行的任務

**CLI 版本**：

- 直接在本地環境執行
- 完整的檔案系統控制
- 適合需要頻繁互動的開發

##### 使用情境

**選擇 Web 版本的時機**：

- 回答程式碼架構問題
- Bug 修復和例行任務
- 並行處理多個工作項目
- 處理未在本地的儲存庫
- 後端變更（撰寫測試、實作功能）

**選擇 CLI 版本的時機**：

- 需要頻繁引導和調整
- 複雜的重構任務
- 需要完整本地環境控制
- 整合本地開發工具
- 即時除錯和測試

#### 同等性說明

Web 版本本質上是 CLI 工具的容器化版本，具有相同的核心能力：

- 相同的程式碼理解能力
- 相同的檔案編輯功能
- 相同的指令執行能力

**主要差異在於便利性**：Web 版本提供託管容器、美觀的 Web/行動介面，以及更適合非同步工作的流程。

---

### 如何存取和使用

#### 步驟 1：存取 Web 介面

1. 前往 [claude.ai/code](https://claude.ai/code) 或 [claude.com/code](https://claude.com/code)
2. 登入您的 Claude 帳號（需要 Pro 或 Max 訂閱）
3. 點擊「Code」標籤

#### 步驟 2：連接 GitHub

##### 初次設定

```
1. 點擊「Connect GitHub」
2. 透過 OAuth 授權
3. 安裝 Claude GitHub App
4. 選擇要授權的儲存庫
   - 可選擇全部儲存庫
   - 或僅選擇特定儲存庫
```

**安全性說明**：

- 授權僅用於存取儲存庫
- 不會長期儲存 GitHub 憑證
- 可隨時撤銷存取權限

#### 步驟 3：設定預設環境

選擇或建立環境配置：

**網路存取等級**：

- **無網路存取**：完全隔離
- **受限存取**：僅允許特定網域
- **完整存取**：允許所有網路請求（使用 `*`）

**環境變數**：

```env
API_KEY=your-api-key
DATABASE_URL=postgresql://...
NODE_ENV=development
```

#### 步驟 4：提交編碼任務

##### 基本範例

```
請修復使用者登入功能中的驗證 bug，
確保在密碼錯誤時顯示正確的錯誤訊息，
並為此功能新增單元測試。
```

##### 複雜範例

```
我需要重構 API 路由層：
1. 將所有路由從 server.js 移到 routes/ 資料夾
2. 為每個端點新增輸入驗證
3. 實作錯誤處理中介軟體
4. 撰寫整合測試

請先提供計劃，等我確認後再開始實作。
```

#### 步驟 5：監控和引導

##### 即時互動

在 Claude 執行任務時：

- **佇列化提示**：傳送額外指令（在當前步驟完成後執行）
- **調整方向**：根據進度修正需求
- **查看變更**：即時預覽檔案編輯

##### 範例互動流程

```
[Claude 正在分析程式碼...]

您: 請先處理驗證邏輯，測試稍後再寫

[提示已加入佇列]
[Claude 完成當前步驟後執行新指令]

[Claude 正在實作驗證...]

您: 記得處理邊界情況：空字串和 null 值

[提示已加入佇列]
```

#### 步驟 6：審查和合併

##### PR 工作流程

1. **Claude 完成任務**

   - 所有變更已提交
   - 測試已執行
   - 分支已推送
2. **審查 Pull Request**

   ```
   自動生成的 PR 包含：
   - 清晰的變更摘要
   - 檔案變更列表
   - 測試結果
   - 任務描述
   ```
3. **選項**

   - **直接合併**：如果變更符合預期
   - **請求修改**：在 PR 中留言，Claude 可回應
   - **Teleport 到本地**：下載到本地繼續調整

---

### 環境配置

#### 環境管理

##### 建立新環境

```
1. 選擇當前環境下拉選單
2. 點擊「Add environment」
3. 設定：
   - 環境名稱
   - 網路存取等級
   - 環境變數
```

##### 更新現有環境

```
1. 選擇環境
2. 點擊環境名稱旁的設定圖示
3. 修改配置
4. 儲存變更
```

#### 網路存取配置

##### 網路代理架構

所有對外網路流量都經過 HTTP/HTTPS 代理：

- 用於安全性和濫用防護
- 預設僅允許白名單網域
- 可自訂存取規則

##### 存取等級選項

**1. 無網路存取**

```
適用情境：
- 純本地邏輯變更
- 不需外部 API 的任務
- 最高安全性需求
```

**2. 受限存取（白名單）**

```
範例配置：
允許網域：
- api.github.com
- registry.npmjs.org
- pypi.org
- api.stripe.com (如需支付整合)
```

**3. 完整存取**

```
配置：網路存取 = "*"

注意事項：
- 僅在必要時使用
- 增加潛在風險
- 適合需要多種外部服務的任務
```

##### 安全最佳實踐

遵循最小權限原則：

```
✓ 僅啟用最低必要的網路存取
✓ 定期審查允許的網域
✓ 優先使用 HTTPS 端點
✓ 避免在環境變數中儲存敏感憑證
✗ 不要預設使用完整網路存取
```

#### 環境變數設定

##### 格式要求

必須使用 `.env` 格式：

```env
## API 金鑰
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

## 資料庫
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379

## 應用程式設定
NODE_ENV=development
PORT=3000
DEBUG=true

## 第三方服務
STRIPE_SECRET_KEY=sk_test_...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

##### 敏感資訊處理

**建議做法**：

1. 使用 GitHub Secrets 儲存敏感憑證
2. 在 Claude Code 中使用佔位符
3. 透過 CI/CD 注入實際值

**範例**：

```env
## 在 Claude Code 環境中
API_KEY=${GITHUB_SECRET_API_KEY}

## 實際值儲存在 GitHub Settings → Secrets
```

#### Claude Code Hooks

##### sessionStart Hook

用於自動化環境初始化：

**範例 `.claude/hooks/sessionStart.sh`**：

```bash
##!/bin/bash

echo "正在初始化開發環境..."

## 安裝依賴
npm install

## 設定資料庫
npm run db:migrate

## 設定測試環境
npm run test:setup

## 建立必要目錄
mkdir -p logs temp

echo "環境初始化完成！"
```

**配置方式**：

```json
// .claude/config.json
{
  "hooks": {
    "sessionStart": ".claude/hooks/sessionStart.sh"
  }
}
```

---

### 使用場景與最佳實踐

#### 理想使用場景

##### 1. Bug 積壓處理

**情境**：需要快速處理多個小型 bug

```
策略：
1. 為每個 bug 建立獨立的 Web session
2. 並行執行多個修復任務
3. 同時審查生成的 PR
4. 快速合併簡單修復
```

**範例任務**：

```
Session 1: 修復日期格式化 bug (#123)
Session 2: 解決表單驗證問題 (#124)
Session 3: 更正錯誤訊息拼寫 (#125)
```

##### 2. 例行和明確定義的任務

**適合的任務類型**：

- 新增單元測試
- 更新文件
- 資料庫遷移腳本
- API 端點實作
- 重複性重構

**範例**：

```
任務：為所有 API 端點新增輸入驗證

步驟：
1. 列出所有端點
2. 為每個端點新增 Joi/Zod schema
3. 實作驗證中介軟體
4. 新增測試案例
5. 更新 API 文件
```

##### 3. 後端變更與測試驅動開發

**TDD 工作流程**：

```
提示範例：

我需要實作使用者註冊功能，請遵循 TDD 方法：

1. 先撰寫失敗的測試：
   - 測試有效的註冊資料
   - 測試重複的電子郵件
   - 測試密碼強度驗證
   - 測試必填欄位

2. 實作最小可用程式碼讓測試通過

3. 重構並確保所有測試仍然通過

4. 新增整合測試
```

##### 4. 程式碼架構問題解答

**探索性任務**：

```
問題：我們的認證流程是如何運作的？

請分析：
1. 使用者登入的完整流程
2. JWT token 的生成和驗證
3. 權限檢查機制
4. Session 管理
5. 潛在的安全風險

請提供詳細的流程圖和程式碼範例。
```

##### 5. 遠端儲存庫工作

**情境**：需要修改未在本地的專案

```
優勢：
- 無需 clone 到本地
- 不佔用本地磁碟空間
- 快速測試和驗證
- 適合臨時任務
```

#### 最佳實踐指南

##### 1. 任務規劃優先

**建議流程**：

```
第一步：要求計劃

「請為這個功能提出一個 3 步驟計劃，
每個步驟都要有小型差異和測試。
等我確認計劃後再開始實作。」

第二步：審查計劃

檢查：
- 步驟是否合理？
- 是否遺漏重要考慮？
- 測試覆蓋是否足夠？

第三步：確認或調整

「計劃看起來不錯，但請在步驟 2 之前
先處理錯誤處理邏輯。現在可以開始了。」
```

##### 2. 使用 CLAUDE.md

**專案根目錄中的 CLAUDE.md**：

```markdown
## 專案：電子商務平台

### 技術堆疊
- 前端：React 18 + TypeScript
- 後端：Node.js + Express
- 資料庫：PostgreSQL
- ORM：Prisma
- 測試：Jest + React Testing Library

### 資料夾結構
```

src/
├── components/    # React 元件
├── pages/         # 頁面元件
├── api/           # API 路由
├── lib/           # 工具函數
├── hooks/         # 自訂 Hooks
└── styles/        # 全域樣式

```

### 開發規範

#### 分支命名
- feature/描述
- bugfix/issue-編號
- hotfix/描述

#### 提交訊息
使用 Conventional Commits：
- feat: 新功能
- fix: bug 修復
- refactor: 重構
- test: 測試

#### 程式碼風格
- 使用 ESLint + Prettier
- 函數式元件優先
- TypeScript strict mode
- 測試覆蓋率 > 80%

### 已知問題
- Auth token 偶爾在 Safari 中失效（#156）
- 圖片上傳在大檔案時較慢（計劃重構）

### 環境設定
```bash
npm install
npm run db:migrate
npm run dev
```

### 測試

```bash
npm test                 # 單元測試
npm run test:integration # 整合測試
npm run test:e2e         # E2E 測試
```

```

##### 3. 清晰且具體的提示

**不良範例**：
```

修復登入問題

```

**優良範例**：
```

我們的登入功能有以下問題：

問題描述：
使用者輸入正確的憑證後，有時會收到
「Invalid credentials」錯誤。這在密碼
包含特殊字元時更常發生。

重現步驟：

1. 使用 email: test@example.com
2. 使用包含 & 或 # 的密碼
3. 點擊登入

預期行為：
成功登入並重導向到 dashboard

請調查：

1. 密碼驗證邏輯（可能在 auth.js）
2. 特殊字元的處理
3. 資料庫查詢的轉義
4. 為此情境新增測試案例

```

##### 4. 利用視覺化內容

**使用圖片和截圖**：

```

拖放或貼上：

- UI 設計稿
- 錯誤截圖
- 流程圖
- 架構圖
- 資料模型圖

範例提示：
「請根據這個設計稿實作登入頁面
[拖放 Figma 截圖]

要求：

- 完全符合設計
- RWD 響應式
- 表單驗證
- 載入狀態
- 錯誤處理」

```

##### 5. 漸進式任務分解

**大型功能分解**：

```

功能：完整的購物車系統

分解為多個 session：

Session 1: 資料模型

- 建立 Prisma schema
- 撰寫遷移檔案
- 設定關聯

Session 2: API 端點

- POST /cart/add
- DELETE /cart/remove
- GET /cart
- PUT /cart/update

Session 3: 前端元件

- CartItem 元件
- CartList 元件
- CartSummary 元件

Session 4: 狀態管理

- Redux/Zustand 設定
- Actions 和 reducers
- API 整合

Session 5: 測試

- 單元測試
- 整合測試
- E2E 測試

```

##### 6. 善用環境配置

**針對不同任務使用不同環境**：

```

環境 1: Development (完整網路存取)

- 用於需要 API 測試的任務
- 允許安裝 npm 套件
- 存取外部服務

環境 2: Testing (受限存取)

- 僅允許測試相關網域
- 隔離的測試環境
- 用於 CI/CD 整合

環境 3: Documentation (無網路)

- 用於純文件任務
- 不需外部資源
- 最安全的選項

```

---

### 定價方案與限制

#### 訂閱方案

##### Pro 方案（$20/月）

**訊息限制**：
- 每 5 小時約 225 則訊息（與 Claude 聊天）
- 或每 5 小時約 50-200 則提示（Claude Code）

**Claude Code Web 存取**：
- 可使用 Web 介面
- 可使用 iOS App
- 雲端 session 共享相同限制

##### Max 5x 方案（$100/月）

**訊息限制**：
- 每 5 小時約 225 則訊息
- 或每 5 小時約 50-200 則 Claude Code 提示

**使用時數（每週）**：
- Sonnet 4：140-280 小時
- Opus 4：15-35 小時

**適合對象**：
- 專業開發者
- 需要並行多個任務
- 較複雜的專案

##### Max 20x 方案（$200/月）

**訊息限制**：
- 每 5 小時約 900 則訊息
- 或每 5 小時約 200-800 則 Claude Code 提示

**使用時數（每週）**：
- Sonnet 4：240-480 小時
- Opus 4：24-40 小時

**適合對象**：
- 重度使用者
- 團隊協作
- 大規模專案

#### 使用限制

##### 共享限制

**重要說明**：
- Claude 聊天和 Claude Code 共享相同的使用限制
- 所有活動都計入相同的配額
- Web 和 CLI 版本共享限制

##### 限制重置

**週期**：
- 整體使用限制：每 7 天重置
- Opus 4 專用限制：每 7 天重置
- 5 小時訊息配額：滾動視窗

##### 影響使用量的因素

**Claude Code 特定因素**：
1. **專案複雜度**
   - 大型程式碼庫消耗更多
   - 深層目錄結構增加處理
   - 多檔案變更需要更多上下文

2. **程式碼庫大小**
   - 檔案數量
   - 總程式碼行數
   - 依賴關係複雜度

3. **自動接受設定**
   - 啟用時減少互動次數
   - 可能增加錯誤修正的循環

**一般訊息因素**：
- 訊息長度
- 對話長度
- 檔案附件

#### 平台限制

##### GitHub 專用

**當前限制**：
- 僅支援 GitHub 儲存庫
- 不支援 GitLab
- 不支援 Bitbucket
- 不支援自託管 Git 服務

**儲存庫驗證**：
- 必須與登入帳號相同
- 需要安裝 Claude GitHub App
- 需要適當的存取權限

##### 功能限制

**Web 版本特定**：
- 不支援完全互動式操作（如 `git rebase -i`）
- 佇列化提示（非即時回應）
- 有限的本地檔案系統存取
- 需要網路連線

#### 使用建議

##### 最大化配額效率

**策略 1：任務批次處理**
```

將相關任務組合在單一 session 中：

一次性任務：
「請執行以下三項任務：

1. 修復 bug #123
2. 為 UserService 新增測試
3. 更新 API 文件」

而非三個獨立的 session

```

**策略 2：使用 /clear**
```

頻繁清除上下文以重置 token 消耗：

- 完成每個主要任務後
- 切換到不相關的工作時
- 對話變得過長時

```

**策略 3：精確的檔案引用**
```

使用 tab 自動完成精確引用檔案：

- 避免 Claude 掃描整個程式碼庫
- 減少不必要的上下文載入
- 更快的回應時間

```

---

### 常見問題與疑難排解

#### 常見問題 (FAQ)

##### 1. 如何開始使用 Claude Code Web？

**答**：
1. 確保您有 Pro 或 Max 訂閱
2. 前往 [claude.ai/code](https://claude.ai/code)
3. 點擊「Connect GitHub」
4. 授權 Claude GitHub App
5. 選擇儲存庫並開始任務

##### 2. Web 版本和 CLI 版本可以同時使用嗎？

**答**：
可以，但要注意：
- 它們共享相同的使用限制
- 不建議同時在同一儲存庫工作
- 可能造成衝突和合併問題

##### 3. 如何將 Web session 轉移到本地？

**答**：
使用 Teleport 功能：
1. 在 Web 介面中點擊「Teleport」
2. 系統會提供指令
3. 在本地 CLI 執行該指令
4. 對話和檔案會同步到本地

##### 4. 可以使用私有儲存庫嗎？

**答**：
可以，只要：
- 您有存取權限
- 儲存庫在您授權的 GitHub 帳號下
- 已安裝 Claude GitHub App

##### 5. 環境變數是否安全？

**答**：
安全措施：
- 環境變數在隔離的 VM 中
- 不會記錄或儲存在聊天歷史中
- 每個 session 結束後清除

**建議**：
- 不要儲存生產環境憑證
- 使用開發/測試專用的 keys
- 考慮使用 GitHub Secrets

##### 6. 可以取消正在執行的任務嗎？

**答**：
可以，在 Web 介面中：
- 點擊「Stop」或「Cancel」按鈕
- 任務會安全終止
- 已完成的變更會保留
- 可以選擇是否保留分支

##### 7. 如何處理大型程式碼庫？

**答**：
最佳實踐：
- 使用 `.claudeignore` 排除不必要的檔案
- 明確指定要處理的檔案/目錄
- 分解任務為較小的單元
- 使用 CLAUDE.md 提供上下文

##### 8. Web 版本支援哪些程式語言？

**答**：
預裝支援：
- Python, JavaScript/TypeScript, Java
- Go, Rust, Ruby, PHP
- C/C++, C#, Swift, Kotlin

**套件管理器**：
- npm, yarn, pnpm
- pip, poetry
- Maven, Gradle
- cargo, go mod

##### 9. 如何審查 Claude 的變更？

**答**：
審查流程：
1. 在 Web 介面即時預覽變更
2. 檢查自動開啟的 Pull Request
3. 查看 diff 和變更摘要
4. 在 GitHub 上進行 code review
5. 必要時請求修改或合併

##### 10. iOS App 功能完整嗎？

**答**：
當前狀態（早期預覽）：
- 基本功能可用
- 適合查看和小型任務
- 功能持續改進中
- 某些複雜操作建議使用 Web 或 CLI

#### 疑難排解

##### 問題 1：無法連接 GitHub

**症狀**：
- OAuth 授權失敗
- 無法看到儲存庫列表
- 連接超時

**解決方案**：

```

步驟 1：檢查 GitHub 權限

- 前往 GitHub Settings → Applications
- 確認 Claude 有適當權限
- 重新授權如有需要

步驟 2：檢查網路

- 確保穩定的網路連線
- 檢查防火牆設定
- 嘗試不同瀏覽器

步驟 3：清除快取

- 清除瀏覽器 cookies 和快取
- 登出後重新登入
- 重新連接 GitHub

步驟 4：檢查 GitHub App 安裝

- 前往 GitHub Settings → Installed GitHub Apps
- 確認 Claude Code 已安裝
- 檢查儲存庫存取權限

```

##### 問題 2：任務執行失敗

**症狀**：
- 任務開始後立即失敗
- 環境初始化錯誤
- 依賴安裝失敗

**解決方案**：

```

診斷步驟：

1. 檢查網路配置
   問題：是否需要特定網域存取？
   解決：更新環境的網路允許清單
2. 檢查環境變數
   問題：是否缺少必要的環境變數？
   解決：在環境設定中新增所需變數
3. 檢查 sessionStart hook
   問題：初始化腳本是否有錯誤？
   解決：測試並修正 hook 腳本
4. 檢查儲存庫狀態
   問題：預設分支是否可用？
   解決：確保 GitHub 上的預設分支正常
5. 檢查依賴
   問題：package.json 或其他設定是否有效？
   解決：在本地測試安裝流程

```

##### 問題 3：變更未推送到 GitHub

**症狀**：
- 任務顯示完成
- 但 GitHub 上看不到分支或 PR
- 推送失敗錯誤

**解決方案**：

```

排查清單：

□ GitHub App 權限

- 檢查是否有寫入權限
- 重新安裝 GitHub App 如需要

□ 分支保護規則

- 檢查儲存庫的分支保護設定
- 確認 Claude 可以建立新分支

□ 網路連線

- Git 操作需要穩定連線
- 重試任務

□ 儲存庫大小限制

- GitHub 有檔案大小限制
- 檢查是否有過大的檔案

```

##### 問題 4：使用限制過快耗盡

**症狀**：
- 頻繁遇到「usage limit reached」
- 無法開始新任務
- 配額重置前被阻擋

**解決方案**：

```

優化策略：

1. 任務整合
   ✓ 合併相關的小任務
   ✓ 批次處理 bug 修復
   ✗ 避免為每個小變更建立新 session
2. 上下文管理
   ✓ 使用 /clear 頻繁清除
   ✓ 精確的檔案引用
   ✓ 使用 .claudeignore
   ✗ 避免載入整個大型程式碼庫
3. 提示優化
   ✓ 清晰、具體的指令
   ✓ 提供足夠的上下文
   ✗ 避免模糊的要求需要多次澄清
4. 計劃升級
   考慮：

   - Max 5x 如果是專業開發者
   - Max 20x 如果是重度使用

```

##### 問題 5：生成的程式碼品質不佳

**症狀**：
- 程式碼不符合專案風格
- 缺少錯誤處理
- 測試不完整
- 未遵循最佳實踐

**解決方案**：

```

改善品質的方法：

1. 完善的 CLAUDE.md
   包含：

   - 程式碼風格指南
   - 命名慣例
   - 架構模式
   - 測試要求
   - 錯誤處理標準
2. 詳細的提示
   範例：
   「實作使用者註冊 API，遵循以下要求：

   - 使用 Express Router
   - 輸入驗證使用 Zod
   - 密碼雜湊使用 bcrypt（成本因子 12）
   - 回傳 JWT token
   - 完整的錯誤處理（400, 409, 500）
   - 單元測試覆蓋率 > 90%
   - 遵循專案中現有的 auth 模式」
3. 漸進式審查
   工作流程：
   a. 請求實作計劃
   b. 審查並確認計劃
   c. 逐步實作
   d. 每步驟後審查
   e. 請求調整
4. 參考範例
   提供：

   - 現有相似功能的範例
   - 專案中的最佳實作
   - 截圖或設計稿

```

##### 問題 6：Teleport 功能無法運作

**症狀**：
- Teleport 按鈕無回應
- 本地 CLI 無法接收
- 檔案同步失敗

**解決方案**：

```

檢查步驟：

1. 本地 CLI 版本
   確保：

   - 已安裝最新版本的 Claude CLI
   - 版本支援 teleport 功能
     更新：npm install -g @anthropic-ai/claude-code@latest
2. 認證狀態
   檢查：

   - 本地 CLI 是否已登入
   - 使用相同的 Claude 帳號
     重新登入：claude login
3. 網路連線

   - 本地機器需要網路連線
   - 防火牆可能阻擋連線
4. 替代方案
   如果 teleport 失敗：

   - 手動下載變更的檔案
   - 從 GitHub 拉取分支
   - 複製程式碼片段

```

##### 問題 7：環境初始化緩慢

**症狀**：
- 任務卡在「Preparing environment」
- 初始化超過 5 分鐘
- 超時錯誤

**解決方案**：

```

優化初始化：

1. 簡化 sessionStart hook
   避免：

   - 完整的資料庫填充
   - 下載大型資料集
   - 複雜的編譯步驟

   建議：

   - 僅安裝必要的依賴
   - 使用快取的依賴
   - 最小化的設定
2. 使用 .claudeignore
   排除：

   - node_modules（讓 npm install 處理）
   - 大型靜態資源
   - 編譯輸出
   - .git 目錄（如果過大）
3. 優化依賴
   考慮：

   - 移除未使用的依賴
   - 使用更輕量的替代品
   - 實作延遲載入
4. 分階段初始化
   策略：

   - 基本環境快速就緒
   - 額外設定在需要時執行
   - 背景任務異步處理

```

#### 取得幫助

##### 官方資源

**文件**：
- [Claude Code 文件](https://docs.claude.com/en/docs/claude-code)
- [API 參考](https://docs.claude.com/en/api)
- [最佳實踐](https://www.anthropic.com/engineering/claude-code-best-practices)

**支援**：
- [幫助中心](https://support.claude.com)
- [Discord 社群](https://discord.gg/anthropic)
- [GitHub Discussions](https://github.com/anthropics/claude-code/discussions)

**回報問題**：
- 透過 Web 介面的「Report Issue」
- support@anthropic.com
- GitHub Issues（如適用）

---

### 進階技巧

#### 1. 鍵盤快捷鍵和效率技巧

##### Tab 自動完成

**檔案引用**：
```

在提示中輸入：
「請檢查 src/com[Tab]」

自動完成為：
「請檢查 src/components/」

繼續：
「src/components/Auth[Tab]」
→ 「src/components/AuthForm.tsx」

```

**好處**：
- 避免路徑錯誤
- 發現相關檔案
- 加快輸入速度

##### 檔案拖放

**使用 Shift 鍵**：
```

一般拖放：將檔案內容貼入對話
Shift + 拖放：正確引用檔案

範例：
[Shift + 拖放 UserService.ts]
→ 「請參考 @src/services/UserService.ts」

```

##### 圖片貼上

**Control+V (非 Command+V)**：
```

macOS 使用者注意：

- Control+V：貼上圖片到 Claude
- Command+V：一般文字貼上

使用情境：

- 設計稿截圖
- 錯誤訊息截圖
- 架構圖
- UI 原型

```

#### 2. 多 Session 工作流程

##### 並行任務範例

**情境：大型功能發布**

```

Session 1: 前端 (repo-frontend)
任務：實作新的儀表板 UI
狀態：進行中
分支：feature/dashboard-ui

Session 2: 後端 (repo-backend)
任務：建立 Dashboard API 端點
狀態：審查中
分支：feature/dashboard-api

Session 3: 文件 (repo-docs)
任務：撰寫 API 文件和使用指南
狀態：等待審查
分支：docs/dashboard

Session 4: 測試 (repo-e2e)
任務：新增 E2E 測試
狀態：準備中
分支：test/dashboard-e2e

```

##### 管理策略

**命名慣例**：
```

環境命名：

- [專案]-dev
- [專案]-test
- [專案]-docs

分支命名：

- feature/[session-描述]
- bugfix/issue-[編號]
- test/[功能]-tests

```

**追蹤進度**：
```

使用專案管理工具：

- GitHub Projects
- Notion
- Trello

連結：

- Session URL → Project card
- PR → 任務項目
- 完成狀態更新

```

#### 3. 自訂環境模板

##### 建立可重複使用的環境

**範例：Node.js API 環境**

```

名稱：nodejs-api-standard

網路存取：受限
允許網域：

- registry.npmjs.org
- github.com
- api.github.com

環境變數：
NODE_ENV=development
LOG_LEVEL=debug
PORT=3000
DATABASE_URL=postgresql://localhost:5432/dev_db
REDIS_URL=redis://localhost:6379

sessionStart Hook：
##!/bin/bash
echo "正在設定 Node.js API 環境..."
npm ci
npm run db:migrate
npm run db:seed:minimal
npm run lint:fix
echo "環境準備完成！"

```

**範例：Python ML 環境**

```

名稱：python-ml-research

網路存取：受限
允許網域：

- pypi.org
- files.pythonhosted.org
- huggingface.co
- github.com

環境變數：
PYTHONPATH=/workspace/src
CUDA_VISIBLE_DEVICES=0
HF_HOME=/workspace/.cache/huggingface

sessionStart Hook：
##!/bin/bash
echo "正在設定 Python ML 環境..."
pip install -r requirements.txt
python -m pytest --collect-only  # 驗證測試
echo "ML 環境準備完成！"

```

#### 4. 進階提示技巧

##### 思考深度控制

**層級**：
```

基本：「請實作這個功能」
→ 標準思考過程

進階：「think hard about this」
→ 更深入的分析

深度：「think harder」
→ 考慮邊界情況和優化

極限：「ultrathink」
→ 最深入的分析和規劃

```

**使用時機**：
```

標準：例行任務、簡單實作
think hard：複雜邏輯、架構決策
think harder：效能關鍵、安全敏感
ultrathink：系統設計、重大重構

```

##### 計劃模式 (Plan Mode)

**啟動方式**（僅 CLI）：
```

按 Shift+Tab 兩次

特性：

- 僅分析和研究
- 不能修改檔案
- 適合探索和規劃
- 較低的 token 消耗

```

**Web 版本替代**：
```

明確要求計劃：
「請分析這個問題並提出計劃，
但不要開始實作。

請包含：

1. 當前架構分析
2. 建議的方法
3. 潛在風險
4. 實作步驟

等我確認後再繼續。」

```

##### 上下文優化提示

**範例 1：大型程式碼庫**
```

「我們的程式碼庫很大，請專注於：

- src/auth/ 目錄
- 特別是 AuthService.ts 和 AuthController.ts
- 相關的測試檔案

請不要載入其他不相關的檔案。」

```

**範例 2：特定檔案組**
```

「這個任務只涉及以下檔案：
@src/components/LoginForm.tsx
@src/hooks/useAuth.ts
@src/api/auth.ts
@tests/auth.test.ts

請僅分析和修改這些檔案。」

```

#### 5. GitHub Actions 整合

##### 自動 PR 審查

**設定 workflow**：

```yaml
## .github/workflows/claude-review.yml
name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  claude-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Claude Code Review
        uses: anthropics/claude-code-action@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          command: |
            請審查這個 PR：
            1. 檢查程式碼品質
            2. 識別潛在 bug
            3. 建議改進
            4. 驗證測試覆蓋率
```

##### 自動文件生成

```yaml
## .github/workflows/auto-docs.yml
name: Auto Generate Docs

on:
  push:
    branches: [main]
    paths:
      - 'src/**/*.ts'
      - 'src/**/*.tsx'

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Generate API Docs
        uses: anthropics/claude-code-action@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          command: |
            請更新 API 文件：
            1. 掃描所有 API 端點
            2. 生成 OpenAPI spec
            3. 更新 docs/api.md
            4. 新增使用範例
```

#### 6. MCP (Model Context Protocol) 伺服器

##### 啟用網路搜尋

**安裝 Brave Search MCP**：

```bash
## 本地 CLI 設定
npm install -g @anthropic-ai/mcp-server-brave-search

## 在 .claude/config.json 中配置
{
  "mcpServers": {
    "brave-search": {
      "command": "mcp-server-brave-search",
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      }
    }
  }
}
```

**使用範例**：

```
「請搜尋 React 18 並行渲染的最佳實踐，
然後在我們的應用中實作這些模式。」

Claude 會：
1. 使用 Brave Search 找最新資訊
2. 分析最佳實踐
3. 應用到您的程式碼
```

##### 自訂 MCP 伺服器

**建立內部工具整合**：

```javascript
// mcp-server-internal-api.js
const { MCPServer } = require('@anthropic-ai/mcp-sdk');

const server = new MCPServer({
  name: 'internal-api',
  tools: {
    getEmployeeInfo: async ({ email }) => {
      // 查詢內部 API
      const response = await fetch(`https://internal.api/employees/${email}`);
      return response.json();
    },
    checkDeploymentStatus: async ({ service }) => {
      // 檢查 k8s 部署
      const status = await kubectl.getStatus(service);
      return status;
    }
  }
});

server.start();
```

#### 7. 進階 CLAUDE.md 技巧

##### 階層式組織

```
專案結構：
project-root/
├── CLAUDE.md                 # 全域專案資訊
├── frontend/
│   ├── CLAUDE.md            # 前端特定指南
│   └── src/
│       └── components/
│           └── CLAUDE.md    # 元件開發規範
└── backend/
    ├── CLAUDE.md            # 後端特定指南
    └── src/
        └── api/
            └── CLAUDE.md    # API 開發規範
```

##### 動態內容

**專案根 CLAUDE.md**：

```markdown
## 專案總覽
[基本資訊...]

## 子專案
- 前端：詳見 /frontend/CLAUDE.md
- 後端：詳見 /backend/CLAUDE.md
- 基礎設施：詳見 /infra/CLAUDE.md

## 跨專案慣例
[共享規範...]
```

**前端 CLAUDE.md**：

```markdown
## 前端開發指南

### 元件開發
詳見 /frontend/src/components/CLAUDE.md

### 狀態管理
我們使用 Zustand，架構：
[詳細說明...]

### 樣式
Tailwind CSS + CSS Modules
[規範...]
```

#### 8. 效能優化策略

##### 減少上下文載入

**.claudeignore 範例**：

```
## 依賴
node_modules/
venv/
vendor/

## 建置輸出
dist/
build/
out/
target/

## 大型靜態檔案
*.jpg
*.png
*.gif
*.svg
*.mp4
*.pdf

## 測試覆蓋率報告
coverage/
.nyc_output/

## 日誌
*.log
logs/

## IDE 設定
.vscode/
.idea/

## 環境檔案
.env*

## Git
.git/
```

##### 智慧檔案引用

**不良做法**：

```
「請檢查所有的認證相關檔案」
→ Claude 掃描整個程式碼庫
```

**良好做法**：

```
「請檢查以下認證相關檔案：
@src/auth/AuthService.ts
@src/auth/AuthController.ts
@src/middleware/authenticate.ts
@types/auth.d.ts」
→ 僅載入指定檔案
```

#### 9. 安全最佳實踐

##### 環境變數分層

**開發環境**：

```env
## .env.development
API_KEY=dev-key-safe-to-commit
DATABASE_URL=postgresql://localhost:5432/dev
DEBUG=true
```

**Claude Code 環境**：

```env
## 在 Web 介面配置
API_KEY=${GITHUB_SECRET_API_KEY}
DATABASE_URL=${GITHUB_SECRET_DB_URL}
DEBUG=true
```

**生產環境**：

```env
## GitHub Secrets（永不直接在 Claude 中使用）
API_KEY=prod-secret-key
DATABASE_URL=postgresql://prod-server/db
DEBUG=false
```

##### 敏感資訊檢查

**Pre-commit Hook**：

```bash
##!/bin/bash
## .git/hooks/pre-commit

## 檢查是否有敏感資訊
if git diff --cached | grep -i "api.key\|password\|secret"; then
  echo "錯誤：偵測到可能的敏感資訊"
  echo "請在提交前移除"
  exit 1
fi

## 檢查 .env 檔案
if git diff --cached --name-only | grep "\.env$"; then
  echo "警告：正在提交 .env 檔案"
  echo "確認這是 .env.example 而非實際憑證"
  read -p "繼續？(y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi
```

#### 10. 協作工作流程

##### 團隊慣例

**CLAUDE.md 團隊區塊**：

```markdown
## 團隊協作

### PR 審查 Checklist
使用 Claude 審查 PR 時，請確保：
- [ ] 所有測試通過
- [ ] 程式碼覆蓋率 > 80%
- [ ] 無 linter 警告
- [ ] 文件已更新
- [ ] 無硬編碼憑證
- [ ] 效能影響已評估

### 分支策略
- `main`：生產就緒
- `develop`：開發整合
- `feature/*`：新功能
- `bugfix/*`：bug 修復
- `hotfix/*`：緊急修復

### Claude Code 使用指南
- 使用共享環境配置（見 `.claude/environments/`）
- 大型功能請先在 `#dev-claude` 頻道討論
- 所有 Claude 生成的 PR 需至少一位人類審查
```

##### 共享環境配置

**團隊儲存庫結構**：

```
.claude/
├── environments/
│   ├── backend-dev.json
│   ├── frontend-dev.json
│   └── fullstack-dev.json
├── commands/
│   ├── review-pr.md
│   ├── generate-tests.md
│   └── update-docs.md
└── config.json
```

**backend-dev.json**：

```json
{
  "name": "backend-dev",
  "network": {
    "allowedDomains": [
      "registry.npmjs.org",
      "github.com",
      "api.stripe.com"
    ]
  },
  "env": {
    "NODE_ENV": "development",
    "LOG_LEVEL": "debug",
    "DATABASE_URL": "${TEAM_SHARED_DB_URL}"
  },
  "hooks": {
    "sessionStart": ".claude/hooks/backend-setup.sh"
  }
}
```

---

### 結語

Claude Code on the Web 代表了 AI 輔助程式開發的新里程碑，結合了雲端運算的便利性和 AI 程式碼助理的強大功能。透過本指南涵蓋的功能、最佳實踐和技巧，您可以：

- **提升生產力**：並行處理多個任務，自動化重複工作
- **改善程式碼品質**：利用 AI 審查、測試生成和最佳實踐建議
- **簡化協作**：自動化 PR 建立、清晰的變更文件
- **靈活工作**：在瀏覽器、行動裝置或本地 CLI 間無縫切換

#### 學習資源

**官方文件**：

- [Claude Code 文件](https://docs.claude.com/en/docs/claude-code)
- [Anthropic 工程部落格](https://www.anthropic.com/engineering)
- [API 參考](https://docs.claude.com/en/api)

**社群資源**：

- [GitHub Discussions](https://github.com/anthropics/claude-code/discussions)
- [Discord 社群](https://discord.gg/anthropic)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)

**進階主題**：

- [MCP 伺服器開發](https://docs.claude.com/en/docs/claude-code/mcp)
- [GitHub Actions 整合](https://docs.claude.com/en/docs/claude-code/github-actions)
- [自訂環境配置](https://docs.claude.com/en/docs/claude-code/environments)

#### 持續改進

Claude Code 持續演進，建議：

- 關注 [Anthropic 公告](https://www.anthropic.com/news)
- 參與社群討論
- 分享您的使用經驗和技巧
- 提供反饋幫助改進產品

祝您使用 Claude Code on the Web 開發順利！

---

**最後更新**：2025-10-21
**版本**：1.0
**作者**：基於 Anthropic 官方文件和社群資源整理

---

## 🎯 學習路徑建議

### 🔰 初學者路徑（第一週）

**Day 1-2：基礎入門**

1. 輸出樣式 (Output Styles) - 了解基本配置
2. 常見工作流程 - 學習核心工作模式
3. Claude Code on the Web - 快速上手

**Day 3-4：進階功能**
4. 子代理 (Sub-agents) - 建立專業助手
5. Agent Skills - 建立可重用模組

**Day 5-7：整合應用**
6. Hooks - 自動化工作流程
7. Plugins - 擴展功能
8. 實際專案練習

### 🚀 進階路徑（第二週）

**自動化與整合**

1. Headless Mode - 非互動式使用
2. GitHub Actions 或 GitLab CI/CD - CI/CD 整合
3. MCP - 連接外部工具

**團隊協作**

1. 建立團隊共享的配置
2. 標準化工作流程
3. 最佳實踐實施

---

## 📊 學習成果檢核

完成本課程後，您應該能夠：

### 基礎能力

- ✅ 熟練使用 Claude Code 的基本功能
- ✅ 配置適合的輸出樣式
- ✅ 建立和使用自訂命令
- ✅ 管理專案配置 (CLAUDE.md)

### 進階能力

- ✅ 建立專業化的子代理
- ✅ 開發自訂 Skills 和 Plugins
- ✅ 配置 Hooks 自動化工作流程
- ✅ 整合到 CI/CD 流程

### 專家能力

- ✅ 在無頭模式下自動化任務
- ✅ 使用 MCP 連接外部服務
- ✅ 優化成本和效能
- ✅ 建立團隊標準化流程

---

## 💡 實用資源

### 官方資源

- 📖 [Claude Code 官方文件](https://docs.claude.com/claude-code)
- 🐙 [Claude Code GitHub](https://github.com/anthropics/claude-code)
- 💬 [Anthropic Discord 社群](https://discord.gg/anthropic)

### 社群資源

- 🌟 [Awesome Claude Code](https://github.com/hesreallyhim/awesome-claude-code)
- 🔧 [Claude Code Plugins Hub](https://github.com/jeremylongshore/claude-code-plugins-plus)
- 📚 [社群 Sub-agents 集合](https://github.com/VoltAgent/awesome-claude-code-subagents)

### 教學文件位置

```
./Claude_Code_Agent_Skills_完整教學.md
./Claude_Code_Hooks_完整教學.md
./Claude_Code_GitLab_CICD_教學.md
./Claude_Code_Web_教學.md
```

---

## 🎓 認證與成就

建議完成以下實作專案來驗證學習成果：

### 📝 基礎專案

1. **個人配置優化** - 建立完整的個人 CLAUDE.md 和自訂命令
2. **自訂 Output Style** - 建立符合個人需求的輸出樣式
3. **簡單 Plugin** - 開發第一個 Plugin

### 🚀 進階專案

1. **CI/CD 整合** - 在專案中設定自動化 PR 審查
2. **團隊工作流程** - 建立團隊共享的配置和流程
3. **MCP 整合** - 連接外部服務（如資料庫或 API）

### 🏆 專家專案

1. **完整自動化系統** - 建立端到端的自動化開發流程
2. **自訂 MCP Server** - 開發專用的 MCP 伺服器
3. **企業級配置** - 為組織建立標準化的 Claude Code 配置

---

## 📅 課程更新記錄

- **2025-10-21**：初版發布，包含 11 個主題的完整教學
- 課程內容基於 Claude Code 2.0+ 版本

---

## 🤝 貢獻與回饋

如果您在學習過程中發現任何問題或有改進建議，歡迎：

- 提出 Issue
- 分享您的學習心得
- 貢獻實用的範例和最佳實踐

---

---

## 📚 學習資源與路徑

## 📖 詳細教學內容

以下是各主題的詳細教學內容連結和摘要。每個主題都包含深入的說明、豐富的範例和實用的技巧。

### Output Styles 詳細內容

**完整教學已提供** - 包含以下完整章節：

- Output Styles 定義和用途
- 配置方法（互動式選單、直接切換、設定檔）
- 內建樣式詳解（Default, Explanatory, Learning）
- 自訂樣式建立（步驟指南、範本、範例）
- 5 個實用範例（Concise, Code Reviewer, Teaching Tutor, Content Strategist, Debug Mode）
- 進階配置技巧
- 常見問題與疑難排解（10+ 個 FAQ）

詳細內容請參考 subagent 提供的完整教學。

---

### Headless Mode 詳細內容

**完整教學已提供** - 包含以下完整章節：

- Headless Mode 核心概念
- 基本使用方法
- 命令列參數詳解（-p, --output-format, --allowedTools 等）
- 三種輸出格式（text, json, stream-json）
- 會話管理（--continue, --resume）
- 權限控制（--allowedTools, --dangerously-skip-permissions）
- 8 個應用場景（程式碼品質分析、Issue 分類、批次處理等）
- CI/CD 整合（GitHub Actions, GitLab CI, Jenkins, CircleCI）
- 進階技巧（Fanning Out, Pipelining, 錯誤處理）
- 疑難排解與最佳實踐

---

### Plugins 詳細內容

**完整教學已提供** - 包含以下完整章節：

- Plugins 系統簡介
- 安裝與管理（Marketplace、互動式安裝、團隊配置）
- 官方 Plugins（Feature Development, Commit Commands, Agent SDK Dev）
- 227+ 社群 Plugins（DevOps, Security, Fullstack, AI/ML 等）
- 開發自己的 Plugin（快速開始、命令建立、代理程式、技能模組、Hooks 配置）
- Plugin 配置選項（plugin.json 規格、frontmatter 選項）
- 實際範例和最佳實踐
- 常見問題和疑難排解（15+ 個問題解決方案）

---

### Sub-agents 詳細內容

**完整教學已提供** - 包含以下完整章節：

- Sub-agents 定義和概念
- 20+ 種子代理類型詳解（開發、測試、安全、DevOps、資料、語言等）
- 建立和調用 Sub-agents
- 應用場景和最佳實踐
- 配置和自定義（配置檔案結構、欄位詳解、系統提示撰寫）
- 6 個實際範例（Code Reviewer, Debugger, Test Runner, Security Auditor, Doc Writer, Laravel Planner）
- 常見問題和疑難排解（11+ 個問題）
- 最佳實踐速查表

---

### GitHub Actions 詳細內容

**完整教學已提供** - 包含以下完整章節：

- 簡介與核心功能
- 環境準備與設定（快速開始、手動設定）
- 進階配置（參數說明、觸發事件）
- 8 個常見使用場景（自動 PR 審查、特定檔案審查、新貢獻者審查等）
- 完整 Workflow 範例（4 個完整範例）
- CLAUDE.md 指南檔案（3 個範例）
- 環境變數與 Secrets 配置（AWS Bedrock, Google Vertex AI）
- 安全性最佳實踐（9 個安全原則）
- 效能優化與成本控制（成本結構、優化策略）
- 疑難排解（診斷工具、常見問題）
- 版本遷移指南（Beta 到 v1.0）
- 常見問題 FAQ（15 個問題）

---

### Workflow 詳細內容

**完整教學已提供** - 包含以下完整章節：

- Claude Code 簡介
- 4 種核心工作模式（Normal, Plan, Auto-Accept, Thinking）
- 8 個常見開發工作流程：
  1. 探索、規劃、編碼、提交
  2. 測試驅動開發 (TDD)
  3. Bug 修復
  4. 程式碼重構
  5. 新功能開發
  6. 視覺化迭代
  7. 文件撰寫
  8. 理解新程式碼庫
- 專案配置與最佳實踐（CLAUDE.md, 斜線指令, 子代理）
- 版本控制整合（Commit, PR, 分支策略, Merge 衝突）
- 進階功能（Checkpoints, Hooks, MCP）
- 團隊協作（共享配置、工作流程標準化、知識分享）
- 效率提升技巧（12 個技巧）
- 最佳實踐總結（黃金原則、適用場景、安全性）

---

## 📝 學習筆記區

建議在學習過程中記錄：

- 💡 重要概念和技巧
- 🔧 實用的配置範例
- ⚠️ 遇到的問題和解決方案
- 🎯 個人的最佳實踐

---

## 🎉 開始學習！

準備好開始您的 Claude Code 精進之旅了嗎？

**建議起點：**

1. 如果您是新手 → 從「常見工作流程」開始
2. 如果您想快速上手 → 從「Claude Code on the Web」開始
3. 如果您想深度客製化 → 從「輸出樣式」和「子代理」開始
4. 如果您想團隊協作 → 從「Plugins」和「Hooks」開始
5. 如果您想 CI/CD 整合 → 從「GitHub Actions」或「GitLab CI/CD」開始

**記住：** 實踐是最好的學習方式。選擇一個真實專案，邊學邊用，才能真正掌握 Claude Code 的強大功能！

---

> 💪 **祝您學習順利！Let's code with Claude!**
