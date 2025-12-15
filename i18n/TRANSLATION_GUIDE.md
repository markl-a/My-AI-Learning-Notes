# 翻譯指南 | Translation Guide

本文件提供翻譯 AI Learning Notes 的詳細指南。

## 翻譯原則

### 1. 準確性優先

技術文檔的翻譯首要原則是準確性。不確定的術語請：

- 查閱 `glossary/terms.json`
- 參考原始論文或官方文檔
- 在 PR 中提出討論

### 2. 保持格式一致

```markdown
# 原文格式
## 標題二
- 列表項目
- `代碼片段`

# 翻譯後保持相同格式
## Heading 2
- List item
- `code snippet`
```

### 3. 程式碼區塊處理

```python
# 原文：中文註釋
def hello():
    """這是函數說明"""
    print("你好")  # 輸出問候

# 翻譯：保留代碼，只翻譯註釋
def hello():
    """This is function description"""
    print("你好")  # Output greeting (keep original output)
```

### 4. 專有名詞處理

| 類型 | 處理方式 | 範例 |
|------|----------|------|
| 技術術語 | 使用標準翻譯 | Transformer → Transformer |
| 產品名稱 | 保持原文 | OpenAI, Claude, GPT-4 |
| 概念名詞 | 翻譯+原文 | 微調 (Fine-tuning) |

## 檔案命名規範

```
i18n/
├── en/                    # 英文
│   ├── README.md
│   ├── QUICKSTART.md
│   └── chapters/
│       └── 1-fundamentals/
├── ja/                    # 日文
│   └── ...
└── ko/                    # 韓文（未來）
    └── ...
```

## 翻譯流程

### Step 1: 認領任務

1. 查看 [i18n Issues](https://github.com/markl-a/My-AI-Learning-Notes/labels/i18n)
2. 在 Issue 中留言認領
3. 等待分配確認

### Step 2: 創建分支

```bash
git checkout -b i18n/en-chapter-1
```

### Step 3: 翻譯文件

使用以下模板開始翻譯：

```markdown
---
original: path/to/original.md
translator: your-github-username
reviewers: []
status: in-progress
last_updated: 2025-01-15
---

# Translated Title

[翻譯內容]
```

### Step 4: 自我檢查

使用以下清單確認品質：

- [ ] 所有術語符合 glossary 標準
- [ ] Markdown 格式正確
- [ ] 連結可正常訪問
- [ ] 程式碼區塊未被破壞
- [ ] 無明顯機器翻譯痕跡
- [ ] 語句通順，符合目標語言習慣

### Step 5: 提交 PR

```bash
git add i18n/en/
git commit -m "i18n(en): translate chapter 1 fundamentals"
git push origin i18n/en-chapter-1
```

PR 標題格式：`i18n(lang): description`

## 常見問題

### Q: 遇到不確定的術語怎麼辦？

A: 在 PR 中標註 `[REVIEW NEEDED]`，等待審核者確認。

### Q: 原文有錯誤怎麼辦？

A: 先完成翻譯，另開 Issue 報告原文問題。

### Q: 如何處理文化相關內容？

A: 可適當本地化，但需在 PR 說明中註明修改原因。

### Q: 圖片中的文字需要翻譯嗎？

A: 目前不需要，但歡迎貢獻本地化圖片。

## 審核流程

1. **初審**：格式檢查、連結驗證
2. **技術審核**：術語準確性、技術正確性
3. **語言審核**：語句流暢度、本地化品質
4. **最終確認**：合併到主分支

## 貢獻者名單

感謝所有翻譯貢獻者！完整名單見 [CONTRIBUTORS.md](../CONTRIBUTORS.md)

---

有問題？歡迎在 [Discussions](https://github.com/markl-a/My-AI-Learning-Notes/discussions) 提問！
