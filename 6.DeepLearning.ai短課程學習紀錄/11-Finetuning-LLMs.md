# Finetuning Large Language Models

## 📋 課程概述

深入學習大型語言模型微調技術，包括 LoRA、QLoRA 等高效方法。

### 課程目標
- 理解 LLM 微調原理
- 學習資料準備和處理
- 掌握 LoRA/QLoRA 技術
- 實作領域專屬模型

### 課程時長
約 1 小時

## 🎯 微調基礎

### 什麼時候需要微調？

1. 特定領域知識
2. 特殊輸出格式
3. 行為模式調整
4. 降低推理成本

### LoRA 技術

Low-Rank Adaptation - 高效微調方法

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入基礎模型
model = AutoModelForCausalLM.from_pretrained("model_name")
tokenizer = AutoTokenizer.from_pretrained("model_name")

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 應用 LoRA
model = get_peft_model(model, lora_config)
```

## 💡 完整微調流程

資料準備 → 模型配置 → 訓練 → 評估 → 部署

---

**課程連結**：[DeepLearning.ai - Finetuning LLMs](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)

**完成日期**：2025-01-17
