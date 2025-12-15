# 國際化 (i18n) 指南

本目錄包含專案的國際化相關資源和翻譯指南。

## 目錄結構

```
i18n/
├── README.md              # 本文件
├── en/                    # 英文翻譯
│   └── README.md          # 英文版首頁
├── ja/                    # 日文翻譯（計劃中）
├── glossary/              # 多語言術語對照
│   └── terms.json         # 術語資料庫
└── TRANSLATION_GUIDE.md   # 翻譯指南
```

## 翻譯優先級

### P0 - 核心文檔（優先翻譯）

1. `README.md` - 專案首頁
2. `QUICKSTART.md` - 快速入門
3. `LEARNING_PATHS.md` - 學習路徑
4. `GLOSSARY.md` - 術語表
5. `CONTRIBUTING.md` - 貢獻指南

### P1 - 基礎章節

1. `1.從AI到LLM基礎/README.md`
2. `3.LLM應用工程/README.md`
3. 各章節的核心概念介紹

### P2 - 進階內容

- 詳細教程
- 實戰項目
- 研究前沿

## 如何貢獻翻譯

### 1. 選擇要翻譯的文件

查看 [翻譯進度追蹤](https://github.com/markl-a/My-AI-Learning-Notes/issues?q=label%3Ai18n) 找到需要翻譯的文件。

### 2. 翻譯規範

- 保持原文的 Markdown 格式
- 程式碼區塊保持原樣，只翻譯註釋
- 使用 `i18n/glossary/terms.json` 中的標準術語
- 保持技術準確性

### 3. 提交翻譯

```bash
# 創建翻譯分支
git checkout -b i18n/en-readme

# 翻譯完成後提交
git add i18n/en/
git commit -m "i18n(en): translate README.md"
git push origin i18n/en-readme

# 創建 Pull Request
```

## 術語一致性

翻譯時請參考 `glossary/terms.json` 確保術語一致：

| 中文 | English | 說明 |
|------|---------|------|
| 大型語言模型 | Large Language Model (LLM) | 保持縮寫 |
| 提示工程 | Prompt Engineering | 專有名詞 |
| 檢索增強生成 | Retrieval-Augmented Generation (RAG) | 保持縮寫 |
| 微調 | Fine-tuning | 技術術語 |
| 對齊 | Alignment | AI 安全術語 |
| 智能體/代理 | Agent | 根據上下文選擇 |

## 翻譯品質檢查清單

- [ ] 技術術語準確
- [ ] 格式與原文一致
- [ ] 連結可正常訪問
- [ ] 程式碼範例可運行
- [ ] 無機器翻譯痕跡
- [ ] 語句通順自然

## 聯繫方式

如有翻譯相關問題，請：

1. 開啟 [Issue](https://github.com/markl-a/My-AI-Learning-Notes/issues/new?labels=i18n)
2. 在 PR 中討論
3. 查閱現有的翻譯討論

感謝您對國際化的貢獻！
