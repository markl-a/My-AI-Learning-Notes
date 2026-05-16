# SFT 從入門到熟練完整學習路徑

本文檔提供了一個結構化的學習路徑，幫助你從零開始掌握監督微調 (Supervised Fine-Tuning)，並成長為這個領域的專家。

## 目錄

1. [學習路徑概覽](#學習路徑概覽)
2. [階段一：基礎入門（1-2週）](#階段一基礎入門)
3. [階段二：實踐應用（2-3週）](#階段二實踐應用)
4. [階段三：進階技術（3-4週）](#階段三進階技術)
5. [階段四：生產部署（2-3週）](#階段四生產部署)
6. [階段五：專家級別（持續學習）](#階段五專家級別)
7. [學習資源](#學習資源)
8. [故障排除指南](#故障排除指南)
9. [最佳實踐清單](#最佳實踐清單)

---

## 學習路徑概覽

```
階段一：基礎入門 (1-2週)
    ↓
理解 SFT 概念、預訓練vs微調、基本資料格式

階段二：實踐應用 (2-3週)
    ↓
動手實踐、使用開源工具、完成第一個項目

階段三：進階技術 (3-4週)
    ↓
PEFT方法、多任務學習、持續學習

階段四：生產部署 (2-3週)
    ↓
部署、監控、優化、A/B測試

階段五：專家級別 (持續)
    ↓
前沿研究、論文復現、開源貢獻
```

**總時間估計**：10-14 週達到熟練水平

**先決條件**：
- Python 編程基礎
- 機器學習基礎概念
- 基本的 NLP 知識（有幫助但不必需）

---

## 階段一：基礎入門

**目標**：理解 SFT 的基本概念和原理

**時間**：1-2 週

### 第 1 天：理解什麼是 SFT

**學習內容**：
1. 閱讀主 README 的前兩章
   - [SFT 概念](./README.md#51-sft-概念)
   - [全參數微調 vs. PEFT 方法](./README.md#52-全參數微調-vs-peft-方法)

2. 理解關鍵概念：
   - 預訓練 vs 微調的區別
   - Next Token Prediction
   - 指令微調的目標

**動手練習**：
```python
# 理解 next token prediction
text = "The cat sat on the"
# 模型預測：mat (概率最高)

# 理解 instruction tuning
instruction = "翻譯成英文"
input_text = "你好"
expected_output = "Hello"
```

**檢查點**：
- [ ] 能用自己的話解釋什麼是 SFT
- [ ] 理解為什麼需要微調
- [ ] 知道 instruction tuning 的基本格式

### 第 2-3 天：資料格式和準備

**學習內容**：
1. 學習常見資料格式
   - Alpaca 格式
   - ShareGPT 格式
   - OpenAI 格式

2. 閱讀 [資料準備工具文檔](./data_preparation_tools/README.md)

**動手練習**：
```bash
# 安裝工具
cd data_preparation_tools
pip install -r requirements.txt

# 運行資料品質檢查
python data_quality_checker.py sample_data.json

# 練習格式轉換
python data_formatter.py
```

**檢查點**：
- [ ] 能夠準備標準格式的訓練資料
- [ ] 知道如何檢查資料品質
- [ ] 理解不同格式的優缺點

### 第 4-5 天：第一次微調體驗

**學習內容**：
1. 跟隨[實戰項目快速開始](./hands_on_project/README.md#快速開始)
2. 使用小模型（GPT-2）進行實驗

**動手練習**：
```bash
# 使用示例資料訓練
cd hands_on_project

# 生成小規模資料
python scripts/1_generate_data.py --num_examples 100

# 訓練模型（在 CPU 上也能運行）
python scripts/3_train_model.py \
    --model_name gpt2 \
    --train_data data/sample/example_data.json \
    --num_epochs 1 \
    --batch_size 2
```

**檢查點**：
- [ ] 成功運行第一次訓練
- [ ] 理解訓練日誌的含義
- [ ] 能夠保存和載入模型

### 第 6-7 天：評估和測試

**學習內容**：
1. 學習評估指標
   - 困惑度 (Perplexity)
   - 任務特定指標

2. 測試微調後的模型

**動手練習**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入模型
model = AutoModelForCausalLM.from_pretrained("./models/customer_service_bot")
tokenizer = AutoTokenizer.from_pretrained("./models/customer_service_bot")

# 測試
prompt = "### 指令:\n客戶詢問退貨流程\n\n### 回答:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

**檢查點**：
- [ ] 知道如何評估模型性能
- [ ] 能夠進行簡單的推理測試
- [ ] 理解模型輸出的質量

### 第 1-2 週總結項目

**項目**：微調一個簡單的問答模型

**要求**：
1. 收集或生成 100 個問答對
2. 使用 GPT-2 或類似小模型
3. 訓練 2-3 個 epochs
4. 評估和測試
5. 撰寫簡單的項目報告

**評估標準**：
- 資料品質分數 > 80
- 模型能生成合理的回答
- 完成完整的訓練-評估流程

---

## 階段二：實踐應用

**目標**：掌握實際的微調技能和工具

**時間**：2-3 週

### 第 1 週：PEFT 方法實踐

**學習內容**：
1. 深入學習 LoRA 和 QLoRA
   - 閱讀 [README 中的 LoRA 部分](./README.md#1-lora-low-rank-adaptation)
   - 閱讀原始論文（可選）

2. 實踐 QLoRA 訓練

**動手練習**：
```bash
# 使用 QLoRA 訓練更大的模型
python scripts/3_train_model.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --use_qlora \
    --train_data data/processed/train.json \
    --lora_r 64 \
    --lora_alpha 16
```

**任務**：
- [ ] 對比全參數微調和 LoRA 的內存使用
- [ ] 實驗不同的 LoRA 秩 (r=8, 16, 32, 64)
- [ ] 記錄性能和資源使用的權衡

### 第 2 週：資料工程

**學習內容**：
1. 使用 AI 輔助生成資料
2. 資料品質控制
3. 資料增強技術

**動手練習**：
```python
# 使用 AI 生成高品質資料
from data_preparation_tools.ai_assisted_data_generator import AIDataGenerator

generator = AIDataGenerator(provider="anthropic")

# 生成 1000 個樣本
data = generator.generate_examples_from_topic(
    topic="你的領域",
    num_examples=1000,
    example_types=["類型1", "類型2", "類型3"]
)

# 質量檢查
from data_preparation_tools.data_quality_checker import DataQualityChecker

checker = DataQualityChecker("your_data.json")
report = checker.check_all()

# 根據報告改進資料
```

**任務**：
- [ ] 生成高品質的訓練資料（質量分數 > 85）
- [ ] 實踐資料增強技術
- [ ] 建立資料版本控制

### 第 3 週：端到端項目

**學習內容**：
1. 完成[實戰項目](./hands_on_project/)的所有步驟
2. 從資料準備到模型部署

**項目要求**：
選擇一個實際場景（例如）：
- 客服機器人
- 程式碼生成助手
- 文字摘要工具
- 翻譯模型

**完整流程**：
```bash
# 1. 資料生成
python scripts/1_generate_data.py --num_examples 2000

# 2. 資料準備
python scripts/2_prepare_data.py --val_ratio 0.1

# 3. 訓練
python scripts/3_train_model.py --use_qlora --num_epochs 3

# 4. 評估
python scripts/4_evaluate_model.py

# 5. 部署
python scripts/5_deploy_model.py --port 8000
```

**檢查點**：
- [ ] 完成一個完整的端到端項目
- [ ] 模型在測試集上表現良好
- [ ] 能夠部署和使用模型

---

## 階段三：進階技術

**目標**：掌握高級微調技術

**時間**：3-4 週

### 第 1-2 週：多任務學習

**學習內容**：
1. 閱讀[多任務學習文檔](./advanced_topics/multi_task_learning.md)
2. 理解任務混合和採樣策略

**動手練習**：
```python
from advanced_topics.multi_task_learning import MultiTaskDataset, train_multi_task_model

# 準備多個任務的資料
task_datasets = {
    "qa": qa_data,
    "summarization": summary_data,
    "translation": translation_data
}

# 使用溫度採樣訓練
model = train_multi_task_model(
    model_name="gpt2",
    task_datasets=task_datasets,
    temperature=0.7
)
```

**項目**：
- [ ] 訓練一個處理 3+ 任務的模型
- [ ] 對比單任務和多任務的性能
- [ ] 分析任務間的遷移效果

### 第 3 週：持續學習

**學習內容**：
1. 閱讀[持續學習文檔](./advanced_topics/continual_learning.md)
2. 理解災難性遺忘及其解決方案

**動手練習**：
```python
from advanced_topics.continual_learning import ComprehensiveContinualLearner

# 初始化持續學習器
learner = ComprehensiveContinualLearner(
    model_name="gpt2",
    strategies=["replay", "ewc", "lwf"]
)

# 依次訓練多個任務
learner.train_task(task_a_data, "task_a")
learner.train_task(task_b_data, "task_b")
learner.train_task(task_c_data, "task_c")

# 評估所有任務
learner._evaluate_all_tasks()
```

**項目**：
- [ ] 實現一個持續學習系統
- [ ] 對比不同防遺忘策略的效果
- [ ] 測量遺忘度和向後遷移

### 第 4 週：高級優化

**學習內容**：
1. 混合精度訓練
2. 梯度累積和檢查點
3. 分布式訓練
4. 模型量化和蒸餾

**動手練習**：
```python
# 使用混合精度訓練
training_args = TrainingArguments(
    ...
    fp16=True,  # 或 bf16=True
    gradient_checkpointing=True,
    gradient_accumulation_steps=8
)

# DeepSpeed 配置
deepspeed_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2
    }
}
```

**任務**：
- [ ] 實驗不同的優化策略
- [ ] 對比訓練速度和內存使用
- [ ] 達到最佳的效率-性能平衡

---

## 階段四：生產部署

**目標**：將模型部署到生產環境

**時間**：2-3 週

### 第 1 週：模型優化和服務化

**學習內容**：
1. 模型量化
2. 推論優化
3. API 開發

**動手練習**：
```python
# 模型量化
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your_model",
    load_in_8bit=True,
    device_map="auto"
)

# FastAPI 服務
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
async def generate(request: GenerateRequest):
    # 生成邏輯
    return response
```

**任務**：
- [ ] 優化推論速度
- [ ] 實現批量推理
- [ ] 建立 REST API

### 第 2 週：監控和維護

**學習內容**：
1. 日誌記錄
2. 性能監控
3. 錯誤處理
4. 用戶反饋收集

**動手練習**：
```python
import logging
from prometheus_client import Counter, Histogram

# 日誌
logger = logging.getLogger(__name__)

# 指標
request_count = Counter('requests_total', 'Total requests')
latency = Histogram('request_latency_seconds', 'Request latency')

@app.post("/generate")
async def generate(request):
    request_count.inc()
    with latency.time():
        response = model.generate(...)
    return response
```

**任務**：
- [ ] 設置日誌系統
- [ ] 實現監控儀表板
- [ ] 建立告警機制

### 第 3 週：持續改進

**學習內容**：
1. A/B 測試
2. 模型版本管理
3. 自動重訓
4. 資料飛輪

**動手練習**：
```python
# A/B 測試
import random

def route_model(user_id):
    if hash(user_id) % 100 < 10:  # 10% 流量
        return model_v2
    else:
        return model_v1

# 定期重訓
def scheduled_retrain():
    new_data = collect_production_data()
    updated_model = train(old_model, new_data)
    deploy(updated_model)
```

**任務**：
- [ ] 實施 A/B 測試
- [ ] 建立自動化重訓管道
- [ ] 從生產資料中學習

---

## 階段五：專家級別

**目標**：成為領域專家，推動前沿發展

**時間**：持續學習

### 研究前沿

**學習內容**：
1. 閱讀最新論文
2. 複現 SOTA 方法
3. 參與學術討論

**推薦論文列表**：
- LoRA: Low-Rank Adaptation of Large Language Models
- QLoRA: Efficient Finetuning of Quantized LLMs
- FLAN: Finetuned Language Models are Zero-Shot Learners
- Constitutional AI: Harmlessness from AI Feedback

**任務**：
- [ ] 每月閱讀 3-5 篇論文
- [ ] 複現至少 1 個方法
- [ ] 撰寫論文筆記或博客

### 開源貢獻

**參與方式**：
1. 貢獻到 Hugging Face Transformers
2. 建立有用的工具和腳本
3. 分享最佳實踐

**任務**：
- [ ] 提交至少 1 個 PR 到開源項目
- [ ] 發布自己的微調工具或模型
- [ ] 幫助社區解決問題

### 實戰經驗

**高級項目**：
1. 大規模多任務模型
2. 領域特定的 LLM
3. 多語言模型
4. 多模態模型

---

## 學習資源

### 官方文檔

1. **Hugging Face**
   - [Transformers Documentation](https://huggingface.co/docs/transformers)
   - [PEFT Library](https://github.com/huggingface/peft)
   - [TRL Library](https://github.com/huggingface/trl)

2. **論文**
   - [LoRA](https://arxiv.org/abs/2106.09685)
   - [QLoRA](https://arxiv.org/abs/2305.14314)
   - [InstructGPT](https://arxiv.org/abs/2203.02155)

### 在線課程

1. **DeepLearning.AI**
   - Finetuning Large Language Models
   - LLMOps

2. **Fast.ai**
   - Practical Deep Learning for Coders

### 社區資源

1. **論壇**
   - Hugging Face Forums
   - r/MachineLearning
   - Stack Overflow

2. **Discord/Slack**
   - Hugging Face Discord
   - LLM Enthusiasts

### 博客和文章

1. Sebastian Ruder - [An Overview of Multi-Task Learning](https://ruder.io/multi-task/)
2. Lilian Weng - [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
3. Jay Alammar - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

## 故障排除指南

### 訓練問題

#### 問題 1: CUDA Out of Memory

**症狀**：
```
RuntimeError: CUDA out of memory
```

**解決方案**：
```python
# 1. 減小 batch size
training_args.per_device_train_batch_size = 1

# 2. 使用梯度累積
training_args.gradient_accumulation_steps = 8

# 3. 使用梯度檢查點
training_args.gradient_checkpointing = True

# 4. 使用 QLoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# 5. 減小序列長度
max_seq_length = 256  # 從 512 減小到 256
```

#### 問題 2: 訓練損失不下降

**症狀**：
- 損失保持在高位
- 或損失變為 NaN

**診斷**：
```python
# 檢查資料
print(train_data[0])  # 確保資料正確

# 檢查學習率
print(f"Learning rate: {training_args.learning_rate}")

# 檢查梯度
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.grad}")
```

**解決方案**：
```python
# 1. 降低學習率
learning_rate = 1e-5  # 從 2e-4 降低

# 2. 增加 warmup
warmup_ratio = 0.1

# 3. 檢查資料品質
from data_quality_checker import DataQualityChecker
checker = DataQualityChecker("train.json")
report = checker.check_all()

# 4. 使用梯度裁剪
max_grad_norm = 1.0
```

#### 問題 3: 過擬合

**症狀**：
- 訓練損失很低，驗證損失很高
- 模型在訓練集上表現好，測試集上表現差

**解決方案**：
```python
# 1. 增加資料量
# 使用資料增強或生成更多資料

# 2. 早停
training_args.load_best_model_at_end = True
training_args.metric_for_best_model = "eval_loss"

# 3. 正則化
lora_config.lora_dropout = 0.1  # 增加 dropout
training_args.weight_decay = 0.01

# 4. 減少訓練 epochs
num_train_epochs = 2  # 從 5 減少到 2
```

### 推理問題

#### 問題 4: 生成品質差

**症狀**：
- 生成重複內容
- 生成無關內容
- 生成格式不正確

**解決方案**：
```python
# 1. 調整生成參數
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,      # 增加多樣性
    top_p=0.9,           # nucleus sampling
    top_k=50,            # top-k sampling
    do_sample=True,      # 啟用採樣
    repetition_penalty=1.2,  # 懲罰重複
    no_repeat_ngram_size=3   # 避免重複 n-gram
)

# 2. 改進提示詞格式
prompt = """### 指令:
{instruction}

### 回答:
"""  # 確保格式與訓練時一致

# 3. 使用約束解碼
from transformers import LogitsProcessor

class CustomLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # 自定義邏輯
        return scores
```

#### 問題 5: 推論速度慢

**症狀**：
- 單個請求需要很長時間
- 吞吐量低

**解決方案**：
```python
# 1. 使用量化
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)

# 2. 批量推理
def batch_generate(prompts, batch_size=8):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(outputs))
    return results

# 3. 使用 KV cache
outputs = model.generate(
    **inputs,
    use_cache=True  # 預設就是 True
)

# 4. 減小最大生成長度
max_new_tokens = 128  # 從 512 減小
```

### 資料問題

#### 問題 6: 資料品質差

**症狀**：
- 資料品質檢查分數低
- 模型學不到有用的模式

**解決方案**：
```python
# 1. 使用質量檢查工具
from data_quality_checker import DataQualityChecker

checker = DataQualityChecker("data.json")
report = checker.check_all()

# 2. 移除重複
duplicates = report.duplicates
# 移除重複樣本

# 3. 過濾低品質樣本
def filter_quality(data):
    filtered = []
    for item in data:
        # 檢查長度
        if len(item["output"]) < 10:
            continue
        # 檢查內容
        if "error" in item["output"].lower():
            continue
        filtered.append(item)
    return filtered

# 4. 使用 AI 輔助改進
from ai_assisted_data_generator import AIDataGenerator

generator = AIDataGenerator()
improved = generator.generate_variations(low_quality_example)
```

---

## 最佳實踐清單

### 資料準備

- [ ] 資料品質分數 > 80
- [ ] 訓練集至少 1000 個樣本（簡單任務）或 10000+ 個樣本（複雜任務）
- [ ] 驗證集佔 10-20%
- [ ] 資料格式統一且正確
- [ ] 移除重複樣本
- [ ] 平衡不同類別/任務的資料
- [ ] 人工抽查至少 10% 的資料

### 訓練配置

- [ ] 使用合適的基座模型
- [ ] QLoRA/LoRA 用於大模型 (>1B 參數)
- [ ] 學習率：全參數 1e-5~5e-5，LoRA 1e-4~5e-4
- [ ] Warmup 步數：總步數的 3-10%
- [ ] 梯度裁剪：max_grad_norm = 1.0
- [ ] 混合精度訓練（fp16/bf16）
- [ ] 梯度檢查點（大模型）
- [ ] 保存最佳模型（基於驗證集）

### LoRA 超參數

- [ ] 秩 (r)：8-64（越大越接近全參數）
- [ ] alpha：r 的 1-2 倍
- [ ] dropout：0.05-0.1
- [ ] target_modules：包含注意力層（q,k,v,o）
- [ ] 對於重要任務，考慮包含 FFN 層

### 評估

- [ ] 在驗證集上評估困惑度
- [ ] 計算任務特定指標
- [ ] 人工評估（至少 50 個樣本）
- [ ] 使用 AI 評估器（GPT-4/Claude）
- [ ] 對比基座模型的性能
- [ ] 測試邊界情況
- [ ] 檢查有害輸出

### 部署

- [ ] 模型量化（8-bit 或 4-bit）
- [ ] 實現請求快取
- [ ] 批量推理
- [ ] 設置超時
- [ ] 日誌記錄
- [ ] 監控指標（延遲、吞吐量、錯誤率）
- [ ] 實現降級方案
- [ ] A/B 測試新模型

### 安全性

- [ ] 過濾敏感輸入
- [ ] 檢測並阻止有害輸出
- [ ] 實現頻率限制
- [ ] 不記錄敏感資訊
- [ ] 定期安全審計
- [ ] 用戶反饋機制

### 持續改進

- [ ] 收集生產資料
- [ ] 分析失敗案例
- [ ] 定期重訓（每週/每月）
- [ ] 跟踪性能指標趨勢
- [ ] 用戶滿意度調查
- [ ] 保持資料版本控制
- [ ] 保持模型版本控制

---

## 學習檢查點

### 基礎級別（1-2 週後）

- [ ] 理解 SFT 的基本概念
- [ ] 能夠準備訓練資料
- [ ] 完成第一次模型訓練
- [ ] 能夠評估模型性能

### 中級（1-2 月後）

- [ ] 熟練使用 LoRA/QLoRA
- [ ] 能夠處理資料品質問題
- [ ] 完成端到端項目
- [ ] 能夠部署簡單的服務

### 高級（3-4 月後）

- [ ] 掌握多任務學習
- [ ] 理解持續學習
- [ ] 能夠優化訓練和推理
- [ ] 能夠解決複雜問題

### 專家級（6+ 月後）

- [ ] 閱讀和複現論文
- [ ] 貢獻開源項目
- [ ] 指導他人
- [ ] 推動技術創新

---

## 下一步行動

根據你的當前水平，選擇對應的起點：

**完全新手** → 從階段一第 1 天開始
**有基礎** → 從階段二實踐應用開始
**有經驗** → 從階段三進階技術開始
**需要部署** → 直接到階段四生產部署

**記住**：
- 學習是循序漸進的過程
- 動手實踐比閱讀更重要
- 遇到問題是正常的，參考故障排除指南
- 加入社區，與他人交流學習

祝你學習順利！🚀
