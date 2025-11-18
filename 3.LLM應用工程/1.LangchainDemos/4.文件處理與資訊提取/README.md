# 文件處理與資訊提取

## 📚 功能

1. **文件載入** - 支援 PDF、Word、TXT、網頁等多種格式
2. **文件總結** - 自動提取重點並生成摘要
3. **資訊提取** - 從文件中提取特定資訊（人名、日期、金額等）
4. **文件問答** - 基於文件內容回答問題
5. **文件翻譯** - 翻譯文件內容

## 🚀 快速開始

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()

# 載入文件
documents = processor.load_document("example.pdf")

# 總結
summary = processor.summarize(documents[0].page_content)
print(summary)

# 提取資訊
names = processor.extract_info(text, "人名")

# 問答
answer = processor.answer_questions(documents, "這份文件在講什麼？")
```

## 💡 使用場景

- **研究輔助**: 快速閱讀和理解論文
- **合約審查**: 提取合約中的關鍵條款
- **會議記錄**: 自動整理會議重點
- **新聞分析**: 批量處理新聞文章
- **多語言文件**: 翻譯和理解外文文件

## 📖 詳細範例

見 `document_processor.py` 中的示範程式。

執行：
```bash
python document_processor.py
```
