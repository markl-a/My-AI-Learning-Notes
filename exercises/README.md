# 實戰練習題庫

本目錄包含各章節的實戰練習題，幫助你鞏固學習成果。

## 目錄

| 難度 | 圖示 | 說明 |
|------|------|------|
| 入門 | ⭐ | 適合初學者 |
| 基礎 | ⭐⭐ | 需要基本概念 |
| 進階 | ⭐⭐⭐ | 需要實作經驗 |
| 挑戰 | ⭐⭐⭐⭐ | 綜合性項目 |

## 練習列表

### 1. Prompt Engineering 練習

| # | 練習名稱 | 難度 | 預計時間 |
|---|---------|------|----------|
| 1.1 | [基礎 Prompt 設計](./prompt-engineering/01-basic-prompts.md) | ⭐ | 30 分鐘 |
| 1.2 | [Few-shot 學習實作](./prompt-engineering/02-few-shot.md) | ⭐⭐ | 45 分鐘 |
| 1.3 | [Chain-of-Thought 推理](./prompt-engineering/03-cot.md) | ⭐⭐ | 1 小時 |
| 1.4 | [結構化輸出控制](./prompt-engineering/04-structured-output.md) | ⭐⭐⭐ | 1.5 小時 |

### 2. RAG 系統練習

| # | 練習名稱 | 難度 | 預計時間 |
|---|---------|------|----------|
| 2.1 | [文檔切分策略](./rag/01-chunking.md) | ⭐⭐ | 1 小時 |
| 2.2 | [向量檢索優化](./rag/02-retrieval.md) | ⭐⭐⭐ | 2 小時 |
| 2.3 | [混合搜索實作](./rag/03-hybrid-search.md) | ⭐⭐⭐ | 2 小時 |
| 2.4 | [端到端 RAG 系統](./rag/04-e2e-rag.md) | ⭐⭐⭐⭐ | 4 小時 |

### 3. Agent 開發練習

| # | 練習名稱 | 難度 | 預計時間 |
|---|---------|------|----------|
| 3.1 | [工具定義與呼叫](./agent/01-tool-use.md) | ⭐⭐ | 1 小時 |
| 3.2 | [ReAct 模式實作](./agent/02-react.md) | ⭐⭐⭐ | 2 小時 |
| 3.3 | [多 Agent 協作](./agent/03-multi-agent.md) | ⭐⭐⭐⭐ | 3 小時 |

### 4. 模型微調練習

| # | 練習名稱 | 難度 | 預計時間 |
|---|---------|------|----------|
| 4.1 | [LoRA 微調入門](./fine-tuning/01-lora-basics.md) | ⭐⭐⭐ | 2 小時 |
| 4.2 | [資料準備與清洗](./fine-tuning/02-data-prep.md) | ⭐⭐ | 1.5 小時 |
| 4.3 | [評估與迭代](./fine-tuning/03-evaluation.md) | ⭐⭐⭐ | 2 小時 |

## 如何使用

### 1. 選擇練習

根據你的學習階段選擇適合的練習：

- **剛開始學習**：從 ⭐ 難度開始
- **有一定基礎**：挑戰 ⭐⭐⭐ 難度
- **追求深入理解**：完成 ⭐⭐⭐⭐ 項目

### 2. 動手實作

每個練習都包含：

- 📋 **學習目標**：明確要達成的目標
- 📚 **前置知識**：需要先了解的內容
- 🔧 **實作步驟**：詳細的操作指引
- ✅ **驗證方法**：如何確認完成
- 💡 **延伸思考**：進一步探索的方向

### 3. 提交成果

完成練習後，歡迎：

- 在 Discussions 分享心得
- 提交 PR 改進練習內容
- 開 Issue 提問或建議

## 練習環境設置

### 基礎環境

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置 API 金鑰（如需要）
export OPENAI_API_KEY="your-key"
```

### 使用 Codespaces

點擊下方按鈕在雲端環境中練習：

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/markl-a/My-AI-Learning-Notes)

### 使用 Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/markl-a/My-AI-Learning-Notes/main)

## 提交你的解答

我們鼓勵學習者提交自己的解答：

1. Fork 此 repo
2. 在 `exercises/submissions/your-github-username/` 下建立你的解答
3. 提交 PR

優秀解答會被收錄到示例中！

## 貢獻新練習

歡迎貢獻新的練習題！請參考 [練習模板](./TEMPLATE.md) 和 [貢獻指南](../CONTRIBUTING.md)。
