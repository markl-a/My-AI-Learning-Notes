# 📊 LLM 評估速查表

快速參考指南，幫助你在評估 LLM 時做出正確選擇。

---

## 🎯 評估指標快速選擇

### 按任務類型選擇

| 任務類型 | 推薦指標 | 備選指標 | 不推薦 |
|---------|---------|---------|--------|
| **機器翻譯** | BLEU, COMET | METEOR, chrF | Perplexity |
| **文本摘要** | ROUGE-L | BERTScore | BLEU |
| **問答系統** | Exact Match, F1 | BERTScore | ROUGE |
| **對話系統** | 人類評估, Elo | LLM-as-Judge | BLEU |
| **代碼生成** | Pass@k | 功能測試 | BLEU |
| **創意寫作** | 人類評估 | LLM-as-Judge | 自動化指標 |
| **分類任務** | Accuracy, F1 | Precision, Recall | Perplexity |
| **語言建模** | Perplexity | - | BLEU |

### 按評估目標選擇

| 評估目標 | 方法 | 工具/基準 |
|---------|------|----------|
| **通用能力** | 標準基準 | MMLU, HellaSwag, ARC |
| **推理能力** | 邏輯/數學任務 | GSM8K, BIG-Bench |
| **知識廣度** | 多學科測試 | MMLU, TriviaQA |
| **真實性** | 事實檢查 | TruthfulQA |
| **安全性** | 紅隊測試 | 自定義安全測試集 |
| **偏見檢測** | 公平性測試 | BBQ, HELM Bias |
| **實用性** | 人類評估 | A/B Testing, CSAT |

---

## 📋 評估基準對比表

### 通用基準

| 基準 | 任務數 | 難度 | 更新頻率 | 數據洩漏風險 | 適用模型 |
|------|--------|------|---------|-------------|---------|
| **MMLU** | 57 | ★★★★☆ | 靜態 | 中 | 通用 LLM |
| **BIG-Bench** | 200+ | ★★★★★ | 靜態 | 中 | 通用 LLM |
| **HellaSwag** | 1 | ★★★☆☆ | 靜態 | 高 | 通用 LLM |
| **TruthfulQA** | 1 | ★★★★★ | 靜態 | 低 | 通用 LLM |
| **MT-Bench** | 80 | ★★★★☆ | 靜態 | 低 | 對話模型 |
| **AlpacaEval** | 805 | ★★★☆☆ | 靜態 | 低 | 指令模型 |
| **LiveBench** | 多個 | ★★★★☆ | **每月更新** | **極低** | 通用 LLM |

### 任務特定基準

| 基準 | 任務 | 樣本數 | 評估指標 | 典型分數範圍 |
|------|------|--------|---------|-------------|
| **SQuAD 2.0** | 閱讀理解 | 150K | EM, F1 | 60-90% |
| **HumanEval** | 代碼生成 | 164 | Pass@k | 20-70% |
| **GSM8K** | 數學推理 | 8.5K | Accuracy | 10-90% |
| **CNN/DM** | 摘要 | 300K | ROUGE | R-L: 35-45 |
| **WMT** | 翻譯 | 數百萬 | BLEU | 20-40 |
| **C-Eval** | 中文知識 | 14K | Accuracy | 40-80% |

---

## 🛠️ 評估工具快速命令

### LM Evaluation Harness

```bash
# 基本評估
lm_eval --model hf \
  --model_args pretrained=MODEL_NAME \
  --tasks TASK_NAME \
  --batch_size 8

# 常用任務組合
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc \
  --num_fewshot 5 \
  --device cuda:0 \
  --output_path results/

# 列出所有任務
lm_eval --tasks list
```

### OpenAI Evals

```bash
# 運行評估
oaieval gpt-3.5-turbo EVAL_NAME

# 查看可用評估
oaieval list

# 自定義評估
oaieval MODEL_NAME my-custom-eval \
  --record_path ./results/
```

### 使用 Python API

```python
# LM Eval Harness
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=MODEL_NAME",
    tasks=["mmlu", "hellaswag"],
    num_fewshot=5,
    batch_size=8
)

# LangChain
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("embedding_distance")
result = evaluator.evaluate_strings(
    prediction="Paris",
    reference="Paris is the capital"
)

# LLM-as-Judge
from openai import OpenAI

def llm_judge(question, answer):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"評估答案質量（1-10）\n問題：{question}\n答案：{answer}"
        }]
    )
    return response.choices[0].message.content
```

---

## 📊 評估流程決策樹

```
開始評估
│
├─ 有明確的參考答案？
│  ├─ 是 → 使用自動化指標（BLEU, EM, F1）
│  └─ 否 ↓
│
├─ 任務是否開放式？
│  ├─ 是 → 使用 LLM-as-Judge 或人類評估
│  └─ 否 → 使用任務特定基準
│
├─ 預算和時間如何？
│  ├─ 充足 → 多層評估（自動化 + LLM + 人類）
│  ├─ 中等 → LLM-as-Judge + 抽樣人類評估
│  └─ 緊張 → 自動化指標
│
└─ 評估目的？
   ├─ 研究發表 → 使用標準基準 + 統計檢驗
   ├─ 生產部署 → A/B 測試 + 持續監控
   └─ 快速迭代 → 自動化指標 + 小規模人類評估
```

---

## 💰 評估成本估算

### 時間成本

| 評估方法 | 1000 樣本評估時間 | 設置時間 |
|---------|------------------|---------|
| **自動化指標** | 10-30 分鐘 | 1 小時 |
| **LLM-as-Judge** | 2-4 小時 | 2 小時 |
| **眾包評估** | 2-3 天 | 4-8 小時 |
| **專家評估** | 1-2 週 | 1 週 |

### 金錢成本（1000 樣本）

| 評估方法 | 成本範圍（USD） | 備註 |
|---------|----------------|------|
| **自動化指標** | $0-10 | 計算資源 |
| **GPT-4 Judge** | $20-50 | API 費用 |
| **MTurk 眾包** | $500-1500 | $0.50-1.50/樣本 |
| **專家評估** | $5000-15000 | $5-15/樣本 |

---

## ⚠️ 常見陷阱檢查清單

### 數據問題

- [ ] 測試集與訓練集完全分離
- [ ] 測試集已去重
- [ ] 沒有時間洩漏（測試數據來自未來）
- [ ] 測試集分布與實際應用一致

### 評估設計問題

- [ ] 指標與業務目標對齊
- [ ] 使用了多個評估維度
- [ ] 計算了統計顯著性
- [ ] 設置了合理的基準線

### 實施問題

- [ ] 評估代碼經過驗證
- [ ] 記錄了所有超參數
- [ ] 多次運行確認穩定性
- [ ] 保存了詳細的評估日誌

### 解釋問題

- [ ] 沒有過度解讀小差異
- [ ] 考慮了評估的局限性
- [ ] 進行了錯誤案例分析
- [ ] 報告了完整結果（不只是最好的）

---

## 🎯 快速評估方案推薦

### 場景 1：通用對話助手

```
第一層：自動化基準
- MMLU（知識）
- HellaSwag（常識）
- TruthfulQA（真實性）

第二層：LLM-as-Judge
- MT-Bench（多輪對話）
- AlpacaEval（指令遵循）

第三層：人類評估
- 100 個真實用戶對話抽樣
- 5 維度評分（相關、準確、有用、安全、流暢）
```

### 場景 2：代碼助手

```
第一層：功能測試
- HumanEval（Python）
- MBPP（基礎編程）

第二層：實際使用測試
- 20 個真實編程任務
- 衡量：正確性、效率、可讀性

第三層：開發者反饋
- A/B 測試
- 用戶滿意度調查
```

### 場景 3：領域特化（如醫療）

```
第一層：領域基準
- MedQA（醫學知識）
- PubMedQA（文獻理解）

第二層：案例測試
- 100 個真實臨床案例
- 專家評分

第三層：安全性評估
- 紅隊測試（有害建議）
- 幻覺檢測
```

---

## 📈 性能基準參考

### 主流模型典型分數（2024）

| 模型 | MMLU | HellaSwag | GSM8K | HumanEval | MT-Bench |
|------|------|-----------|-------|-----------|----------|
| **GPT-4** | 86% | 95% | 92% | 67% | 9.0/10 |
| **Claude 3 Opus** | 86% | 95% | 95% | 84% | 9.0/10 |
| **Gemini Pro** | 79% | 88% | 86% | 67% | 8.1/10 |
| **Llama-3-70B** | 79% | 87% | 76% | 62% | 8.0/10 |
| **GPT-3.5** | 70% | 85% | 57% | 48% | 7.9/10 |

### 分數解讀指南

**MMLU（0-100%）：**
- 90%+：頂尖水平
- 80-90%：優秀
- 70-80%：良好
- 60-70%：尚可
- <60%：需要改進

**HumanEval Pass@1（0-100%）：**
- 70%+：卓越
- 50-70%：優秀
- 30-50%：良好
- 10-30%：基礎
- <10%：較弱

**MT-Bench（1-10）：**
- 9-10：頂尖
- 8-9：優秀
- 7-8：良好
- 6-7：尚可
- <6：需改進

---

## 🔗 快速鏈接

### 工具與框架
- [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenAI Evals](https://github.com/openai/evals)
- [HELM](https://crfm.stanford.edu/helm/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation)

### 排行榜
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Chatbot Arena](https://chat.lmsys.org/)
- [Big Code Models](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
- [C-Eval](https://cevalbenchmark.com/)

### 數據集
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Papers with Code](https://paperswithcode.com/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

---

## 📝 評估報告模板

```markdown
# 模型評估報告

## 基本信息
- 模型名稱：[MODEL_NAME]
- 評估日期：[DATE]
- 評估者：[NAME]
- 評估目的：[PURPOSE]

## 評估設置
- 測試集大小：[N] 樣本
- 評估任務：[TASKS]
- 評估工具：[TOOLS]
- 超參數：[PARAMS]

## 評估結果

### 自動化指標
| 任務 | 指標 | 分數 | 基準對比 |
|------|------|------|---------|
| MMLU | Accuracy | X% | ±Y% vs baseline |
| ... | ... | ... | ... |

### LLM-as-Judge 評估
- 平均分數：X/10
- 主要優勢：[STRENGTHS]
- 主要弱點：[WEAKNESSES]

### 人類評估
- 樣本數：N
- 平均滿意度：X/5
- 關鍵發現：[FINDINGS]

## 錯誤分析
- 錯誤類型分布：[CHART]
- 典型錯誤案例：[EXAMPLES]
- 改進建議：[RECOMMENDATIONS]

## 結論與建議
- 總體評估：[SUMMARY]
- 是否推薦部署：[YES/NO]
- 下一步行動：[ACTIONS]
```

---

## 🎓 學習路徑建議

### 新手（0-3 個月）
1. 理解傳統指標（BLEU, ROUGE）
2. 運行一次 LM Eval Harness
3. 嘗試 LLM-as-Judge

### 進階（3-6 個月）
1. 設計自定義評估任務
2. 實施 A/B 測試
3. 建立持續評估流程

### 專家（6+ 個月）
1. 開發領域特化評估體系
2. 貢獻開源評估項目
3. 發表評估方法論文

---

**💡 提示**：評估是迭代過程，隨著理解加深不斷改進評估方法。

**🔄 更新**：本速查表定期更新，請關注最新版本。

**📞 反饋**：發現錯誤或有改進建議？歡迎提交 Issue！
