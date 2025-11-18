# 端到端 SFT 實戰項目：電商客服機器人

這是一個完整的端到端監督微調項目，從數據準備到模型部署的完整流程。

## 項目概述

**目標**：訓練一個能夠處理電商客服對話的 LLM 模型

**任務類型**：
- 退換貨諮詢
- 物流查詢
- 產品問題
- 優惠活動
- 投訴處理

**技術棧**：
- 模型：LLaMA-2-7B / GPT-2 (示例)
- 微調方法：QLoRA
- 框架：Transformers, PEFT, TRL
- 評估：自定義指標 + GPT-4 評估

## 項目結構

```
hands_on_project/
├── README.md                 # 本文件
├── data/                     # 數據目錄
│   ├── raw/                  # 原始數據
│   ├── processed/            # 處理後的數據
│   └── sample/               # 示例數據
├── scripts/                  # 腳本目錄
│   ├── 1_generate_data.py    # 數據生成
│   ├── 2_prepare_data.py     # 數據準備
│   ├── 3_train_model.py      # 模型訓練
│   ├── 4_evaluate_model.py   # 模型評估
│   └── 5_deploy_model.py     # 模型部署
├── configs/                  # 配置文件
│   ├── data_config.yaml      # 數據配置
│   └── train_config.yaml     # 訓練配置
├── notebooks/                # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_evaluation_demo.ipynb
├── models/                   # 保存的模型
├── logs/                     # 訓練日誌
└── requirements.txt          # 依賴項
```

## 快速開始

### 1. 環境設置

```bash
# 克隆項目（如果適用）
cd hands_on_project

# 安裝依賴
pip install -r requirements.txt

# 設置 API 密鑰（用於 AI 輔助數據生成）
export ANTHROPIC_API_KEY="your-api-key"
# 或
export OPENAI_API_KEY="your-api-key"
```

### 2. 數據準備

#### 選項 A: 使用 AI 生成數據（推薦用於學習）

```bash
# 生成訓練數據
python scripts/1_generate_data.py \
    --num_examples 1000 \
    --output_dir data/raw \
    --topics "退換貨,物流,產品,優惠,投訴"

# 處理數據
python scripts/2_prepare_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --val_ratio 0.1
```

#### 選項 B: 使用自己的數據

```bash
# 準備數據為標準格式
python scripts/2_prepare_data.py \
    --input_file your_data.json \
    --output_dir data/processed \
    --val_ratio 0.1
```

### 3. 訓練模型

```bash
# 使用 QLoRA 訓練（推薦）
python scripts/3_train_model.py \
    --model_name "gpt2" \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --output_dir models/customer_service_bot \
    --use_qlora \
    --num_epochs 3 \
    --batch_size 4

# 使用完整配置文件
python scripts/3_train_model.py --config configs/train_config.yaml
```

### 4. 評估模型

```bash
# 評估模型性能
python scripts/4_evaluate_model.py \
    --model_dir models/customer_service_bot \
    --test_data data/processed/val.json \
    --output_dir logs/evaluation

# 使用 AI 輔助評估
python scripts/4_evaluate_model.py \
    --model_dir models/customer_service_bot \
    --test_data data/processed/val.json \
    --use_ai_judge \
    --ai_judge_model claude-3-5-sonnet-20241022
```

### 5. 部署模型

```bash
# 啟動推理服務器
python scripts/5_deploy_model.py \
    --model_dir models/customer_service_bot \
    --port 8000

# 測試 API
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "我想退貨，請問流程是什麼？"}'
```

## 詳細步驟

### 步驟 1: 數據生成

使用 AI 輔助生成高質量的客服對話數據：

```python
from ai_assisted_data_generator import AIDataGenerator

generator = AIDataGenerator(provider="anthropic")

# 生成退換貨相關數據
refund_data = generator.generate_examples_from_topic(
    topic="電商退換貨客服對話",
    num_examples=200,
    example_types=["退貨流程", "換貨申請", "退款查詢", "退貨原因"]
)

# 生成物流相關數據
shipping_data = generator.generate_examples_from_topic(
    topic="電商物流客服對話",
    num_examples=200,
    example_types=["物流查詢", "配送問題", "延遲投訴", "地址修改"]
)

# ... 其他類型
```

**數據格式**：

```json
{
  "instruction": "客戶詢問如何退貨",
  "input": "我買的商品不滿意，想要退貨",
  "output": "您好！我們支持7天無理由退貨。請提供您的訂單號，我幫您處理退貨申請。退貨流程如下：\n1. 提交退貨申請\n2. 等待審核（1-2工作日）\n3. 寄回商品\n4. 審核通過後退款（3-5工作日）\n\n請問您的訂單號是多少？",
  "task_type": "refund"
}
```

### 步驟 2: 數據質量控制

```python
from data_quality_checker import DataQualityChecker

# 檢查數據質量
checker = DataQualityChecker("data/raw/all_data.json")
report = checker.check_all()

# 根據報告改進數據
if report.quality_score < 80:
    print("數據質量需要改進！")
    # 移除重複、填充空值等
```

### 步驟 3: 模型訓練

**使用 QLoRA 進行高效訓練**：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # 或其他模型
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 的注意力層
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 訓練
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    max_seq_length=512,
    args=training_args
)

trainer.train()
```

**訓練監控**：

使用 Weights & Biases 或 TensorBoard 監控訓練：

```python
import wandb

wandb.init(project="customer-service-bot", name="qlora-gpt2")

training_args = TrainingArguments(
    output_dir="./models/customer_service_bot",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    report_to="wandb",  # 啟用 wandb
    bf16=True
)
```

### 步驟 4: 模型評估

#### 4.1 自動評估指標

```python
from evaluate_model import ModelEvaluator

evaluator = ModelEvaluator(model, tokenizer)

# 計算困惑度
perplexity = evaluator.compute_perplexity(test_data)
print(f"困惑度: {perplexity:.2f}")

# 計算任務特定指標
metrics = evaluator.evaluate_response_quality(test_data)
print(f"平均響應長度: {metrics['avg_length']}")
print(f"詞彙多樣性: {metrics['diversity']}")
```

#### 4.2 使用 AI 作為評估器

```python
from ai_judge import AIJudge

judge = AIJudge(model_name="claude-3-5-sonnet-20241022")

# 評估回答質量
scores = judge.evaluate_responses(
    model=model,
    test_cases=test_data,
    criteria=["準確性", "有用性", "專業性", "友好度"]
)

print(f"平均分數: {sum(scores) / len(scores):.2f}")
```

#### 4.3 人工評估

創建評估界面進行人工評估：

```python
# 使用 Gradio 創建評估界面
import gradio as gr

def evaluate_response(question, model_response):
    # 人工評分
    return {
        "score": score,
        "comments": comments
    }

demo = gr.Interface(
    fn=evaluate_response,
    inputs=[
        gr.Textbox(label="問題"),
        gr.Textbox(label="模型回答")
    ],
    outputs=[
        gr.Number(label="評分 (1-5)"),
        gr.Textbox(label="評論")
    ]
)

demo.launch()
```

### 步驟 5: 模型部署

#### 5.1 FastAPI 服務器

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    response: str
    confidence: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 生成回答
    response = generate_response(
        model,
        tokenizer,
        request.message,
        request.conversation_history
    )

    return ChatResponse(
        response=response["text"],
        confidence=response["confidence"]
    )

# 運行: uvicorn api:app --host 0.0.0.0 --port 8000
```

#### 5.2 Gradio 演示界面

```python
import gradio as gr

def chat_interface(message, history):
    response = generate_response(model, tokenizer, message, history)
    history.append((message, response))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="輸入消息")
    clear = gr.Button("清除")

    msg.submit(chat_interface, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)
```

## 進階功能

### 1. 多輪對話支持

```python
class ConversationManager:
    """管理多輪對話上下文"""

    def __init__(self, max_history=5):
        self.max_history = max_history
        self.conversations = {}

    def add_message(self, session_id, role, message):
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            "role": role,
            "content": message
        })

        # 保持歷史在限制內
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history*2:]

    def get_context(self, session_id):
        return self.conversations.get(session_id, [])
```

### 2. 意圖識別

```python
class IntentClassifier:
    """識別用戶意圖"""

    intents = {
        "refund": ["退貨", "退款", "不想要"],
        "shipping": ["物流", "配送", "到哪裡了"],
        "product": ["產品", "功能", "如何使用"],
        "complaint": ["投訴", "不滿意", "質量問題"]
    }

    def classify(self, message):
        # 簡單的關鍵詞匹配（實際應使用分類模型）
        for intent, keywords in self.intents.items():
            if any(kw in message for kw in keywords):
                return intent
        return "general"
```

### 3. 自動路由

```python
class ResponseRouter:
    """根據意圖路由到不同的處理器"""

    def __init__(self):
        self.handlers = {
            "refund": self.handle_refund,
            "shipping": self.handle_shipping,
            # ...
        }

    def route(self, intent, message):
        handler = self.handlers.get(intent, self.handle_general)
        return handler(message)

    def handle_refund(self, message):
        # 專門處理退款問題
        return generate_response(model, tokenizer, message, system_prompt="你是退換貨專員...")
```

## 性能優化

### 1. 批量推理

```python
def batch_inference(model, tokenizer, messages, batch_size=8):
    """批量處理請求"""
    results = []

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model.generate(**inputs)

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(responses)

    return results
```

### 2. 模型量化

```python
# 使用 int8 量化加速推理
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "models/customer_service_bot",
    load_in_8bit=True,
    device_map="auto"
)
```

### 3. 緩存策略

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_response(message: str) -> str:
    """緩存常見問題的回答"""
    return generate_response(model, tokenizer, message)
```

## 監控和維護

### 1. 日誌記錄

```python
import logging

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 記錄每次對話
logging.info(f"用戶: {user_message}")
logging.info(f"機器人: {bot_response}")
logging.info(f"意圖: {intent}")
```

### 2. 性能監控

```python
import time

def monitor_latency(func):
    """監控響應延遲"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start

        logging.info(f"延遲: {latency:.2f}秒")

        if latency > 2.0:
            logging.warning(f"響應慢: {latency:.2f}秒")

        return result
    return wrapper
```

### 3. 反饋收集

```python
class FeedbackCollector:
    """收集用戶反饋用於持續改進"""

    def __init__(self):
        self.feedback_db = []

    def collect_feedback(self, session_id, message, response, rating):
        self.feedback_db.append({
            "session_id": session_id,
            "message": message,
            "response": response,
            "rating": rating,
            "timestamp": time.time()
        })

    def get_low_rated_samples(self, threshold=3):
        """獲取評分低的樣本用於改進"""
        return [
            f for f in self.feedback_db
            if f["rating"] < threshold
        ]
```

## 持續改進

### 1. 收集真實數據

```python
# 從生產環境收集數據
def collect_production_data():
    """收集真實用戶對話"""
    # 匿名化
    # 質量篩選
    # 標註
    # 添加到訓練集
```

### 2. 定期重訓

```python
# 每週或每月重新訓練
def scheduled_retraining():
    # 合併舊數據和新數據
    new_data = collect_production_data()
    all_data = old_training_data + new_data

    # 使用持續學習方法訓練
    from continual_learning import ContinualLearner

    learner = ContinualLearner(model)
    learner.train_on_new_task(new_data, "production_data")
```

## 常見問題

### Q1: 訓練很慢怎麼辦？

**解決方案**：
- 使用 QLoRA 減少內存需求
- 減小 batch size，增加 gradient accumulation
- 使用混合精度訓練 (fp16/bf16)
- 考慮使用多 GPU

### Q2: 模型回答質量不高？

**解決方案**：
- 增加訓練數據量
- 提高數據質量（人工審核）
- 調整訓練超參數
- 嘗試更大的基座模型
- 添加更多示例到提示詞

### Q3: 如何處理模型的幻覺問題？

**解決方案**：
- 在訓練數據中明確標註不確定的情況
- 使用檢索增強生成 (RAG)
- 添加信心分數，低信心時提示人工介入
- 定期更新知識庫

## 下一步

完成這個項目後，可以：

1. **擴展功能**：
   - 添加多語言支持
   - 集成知識庫 (RAG)
   - 添加情感分析

2. **優化性能**：
   - 模型蒸餾
   - 量化優化
   - 分佈式部署

3. **生產部署**：
   - 容器化 (Docker)
   - 負載均衡
   - 監控告警

## 參考資源

- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face SFT Guide](https://huggingface.co/docs/trl/sft_trainer)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 許可證

MIT License
