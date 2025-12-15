# Evaluating and Debugging Generative AI

## 📋 課程概述

學習評估和除錯生成式 AI 應用的方法和工具。

### 課程目標
- 掌握 LLM 評估指標
- 學習常見問題診斷
- 建立評估管道
- 實作監控系統

### 課程時長
約 1 小時

## 🎯 評估指標

### 1. 自動評估

```python
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

# ROUGE 分數
rouge = Rouge()
scores = rouge.get_scores(prediction, reference)

# BLEU 分數
bleu_score = sentence_bleu([reference], prediction)
```

### 2. LLM 作為評審

```python
def llm_evaluate(answer, reference):
    prompt = f"""
    評估以下答案與參考答案的相似度（0-10分）：

    參考答案：{reference}
    實際答案：{answer}

    分數：
    """
    # 使用 LLM 評分
    return score
```

## 💡 監控和除錯

建立完整的評估和監控系統。

---

**課程連結**：[DeepLearning.ai - Evaluating AI](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)

**完成日期**：2025-01-17
