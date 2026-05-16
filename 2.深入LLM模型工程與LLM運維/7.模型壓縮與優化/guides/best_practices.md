# 最佳實踐指南

## 通用原則

### 1. 始終從基準開始

```python
# ✅ 正確流程
# 1. 建立 FP16 基準
model_fp16 = load_model(dtype=torch.float16)
baseline_ppl = evaluate_perplexity(model_fp16)
baseline_speed = benchmark_speed(model_fp16)

# 2. 應用優化
model_quantized = quantize(model_fp16)

# 3. 對比評估
quantized_ppl = evaluate_perplexity(model_quantized)
quantized_speed = benchmark_speed(model_quantized)

print(f"困惑度變化: {(quantized_ppl/baseline_ppl - 1)*100:.2f}%")
print(f"速度提升: {quantized_speed/baseline_speed:.2f}x")
```

### 2. 漸進式優化

**遵循優化階梯**：
```
FP32 (訓練精度)
  ↓
FP16 (混合精度訓練)
  ↓
INT8 (保守量化)
  ↓
INT4 (激進量化)
  ↓
混合精度 (精細調優)
```

### 3. 資料驅動決策

**記錄所有實驗**：
```python
import wandb

# 初始化追蹤
wandb.init(project="model-compression")

# 記錄配置和結果
wandb.config.update({
    "quantization": "gptq",
    "bits": 4,
    "group_size": 128,
})

wandb.log({
    "perplexity": ppl,
    "inference_speed": speed,
    "model_size_mb": size,
})
```

---

## 量化最佳實踐

### 1. 選擇合適的方法

**決策樹**：
```
需要微調？
├─ Yes → QLoRA (bitsandbytes)
└─ No → 僅推理？
    ├─ GPU → GPTQ/AWQ
    └─ CPU → llama.cpp (GGUF)
```

### 2. 校準資料很關鍵

```python
# ❌ 錯誤：使用隨機或不相關資料
calibration_data = random_data()

# ✅ 正確：使用目標領域資料
calibration_data = load_dataset("your_domain", split="train[:1000]")

# ✅ 更好：使用代表性樣本
calibration_data = stratified_sample(dataset, n=1000)
```

### 3. 驗證量化質量

```python
def validate_quantization(original_model, quantized_model, test_data):
    """完整的量化驗證流程"""

    results = {}

    # 1. 困惑度
    results['ppl_original'] = evaluate_ppl(original_model, test_data)
    results['ppl_quantized'] = evaluate_ppl(quantized_model, test_data)
    results['ppl_degradation'] = (
        (results['ppl_quantized'] / results['ppl_original'] - 1) * 100
    )

    # 2. 下游任務
    for task in ['classification', 'qa', 'generation']:
        results[f'{task}_original'] = evaluate_task(original_model, task)
        results[f'{task}_quantized'] = evaluate_task(quantized_model, task)

    # 3. 推論速度
    results['speed_original'] = benchmark(original_model)
    results['speed_quantized'] = benchmark(quantized_model)
    results['speedup'] = results['speed_quantized'] / results['speed_original']

    # 4. 模型大小
    results['size_original'] = get_model_size(original_model)
    results['size_quantized'] = get_model_size(quantized_model)
    results['compression_ratio'] = results['size_original'] / results['size_quantized']

    return results
```

---

## LoRA 最佳實踐

### 1. 超參數選擇

**推薦起點**：
```python
lora_config = LoraConfig(
    r=8,                  # 秩：從小開始
    lora_alpha=16,        # alpha = 2 * r
    target_modules=[      # 目標：至少 Q, V
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,    # dropout：輕微正則化
    bias="none",          # 偏置：通常不訓練
    task_type="CAUSAL_LM"
)
```

**調優策略**：
```python
# 1. 如果欠擬合（loss 不下降）
# → 增加 r (8 → 16 → 32)
# → 添加更多 target_modules

# 2. 如果過擬合（train loss 低但 val loss 高）
# → 減小 r
# → 增加 lora_dropout
# → 減少訓練步數
```

### 2. 訓練配置

```python
training_args = TrainingArguments(
    output_dir="./output",

    # 批次和累積
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效批次 = 16

    # 學習率（LoRA 可用更高）
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    # 訓練步數
    num_train_epochs=3,
    max_steps=-1,  # 或指定固定步數

    # 評估
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,

    # 優化
    optim="paged_adamw_32bit",  # QLoRA 專用
    fp16=False,
    bf16=True,  # 如果支持

    # 日誌
    logging_steps=10,
    report_to="wandb",
)
```

### 3. 保存和部署

```python
# ✅ 推薦：僅保存 LoRA 適配器
model.save_pretrained("./lora-adapter")  # ~4MB

# 部署時載入
base_model = AutoModelForCausalLM.from_pretrained("base")
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# 推論優化：合併權重
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")  # 完整模型
```

---

## 部署最佳實踐

### 1. 推論優化

**使用專用推論引擎**：
```python
# ❌ 避免：直接使用 Transformers 推理（生產環境）
from transformers import pipeline
pipe = pipeline("text-generation", model="model")

# ✅ 推薦：使用優化的推論引擎
from vllm import LLM

llm = LLM(
    model="model",
    quantization="awq",
    tensor_parallel_size=1,
)
```

**配置最優參數**：
```python
# vLLM 推理配置
llm = LLM(
    model="model_path",

    # 量化
    quantization="gptq",  # 或 "awq"

    # 並行
    tensor_parallel_size=1,  # GPU 數量

    # 批次配置
    max_num_batched_tokens=4096,
    max_num_seqs=256,

    # KV cache
    gpu_memory_utilization=0.9,

    # LoRA（如需要）
    enable_lora=True,
    max_loras=4,
)
```

### 2. 監控和日誌

```python
import time
import logging
from prometheus_client import Counter, Histogram, Gauge

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus 指標
requests_total = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
active_requests = Gauge('active_requests', 'Active requests')
tokens_generated = Counter('tokens_generated_total', 'Total tokens generated')

@app.post("/generate")
async def generate(request: GenerateRequest):
    """生成端點with監控"""
    requests_total.inc()
    active_requests.inc()

    start_time = time.time()
    try:
        output = llm.generate(request.prompt)
        tokens_generated.inc(len(output))

        logger.info(
            f"Generated {len(output)} tokens in {time.time() - start_time:.2f}s"
        )

        return {"text": output}

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

    finally:
        active_requests.dec()
        request_duration.observe(time.time() - start_time)
```

### 3. 錯誤處理

```python
from fastapi import HTTPException
import asyncio

class LLMService:
    """帶錯誤處理的 LLM 服務"""

    def __init__(self, model_path, **kwargs):
        self.llm = LLM(model=model_path, **kwargs)
        self.max_retries = 3
        self.timeout = 30.0

    async def generate_with_retry(self, prompt, **kwargs):
        """帶重試的生成"""
        for attempt in range(self.max_retries):
            try:
                # 設置超時
                result = await asyncio.wait_for(
                    self._generate(prompt, **kwargs),
                    timeout=self.timeout
                )
                return result

            except asyncio.TimeoutError:
                logger.warning(f"Generation timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise HTTPException(status_code=504, detail="Generation timeout")

            except Exception as e:
                logger.error(f"Generation error: {str(e)} (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise HTTPException(status_code=500, detail=str(e))

                await asyncio.sleep(2 ** attempt)  # 指數退避

    async def _generate(self, prompt, **kwargs):
        """實際生成邏輯"""
        return self.llm.generate(prompt, **kwargs)
```

---

## 性能優化清單

### 訓練優化
- [ ] 使用混合精度訓練 (FP16/BF16)
- [ ] 啟用梯度檢查點
- [ ] 使用梯度累積模擬大批次
- [ ] 優化資料加載（多進程、預取）
- [ ] 使用高效優化器（paged_adamw）
- [ ] 啟用 torch.compile（PyTorch 2.0+）

### 推論優化
- [ ] 使用量化（INT8/INT4）
- [ ] 使用專用推論引擎（vLLM/TRT）
- [ ] 啟用 KV cache
- [ ] 批次推理
- [ ] 使用合適的採樣策略
- [ ] 優化生成參數（max_tokens, temperature）

### 部署優化
- [ ] 負載均衡
- [ ] 自動擴展
- [ ] 監控和告警
- [ ] 錯誤處理和重試
- [ ] 快取常見查詢
- [ ] API 速率限制

---

## 安全和隱私

### 1. 模型安全

```python
# 1. 輸入驗證
def validate_input(text: str) -> str:
    """驗證和清理輸入"""
    # 長度限制
    if len(text) > 10000:
        raise ValueError("Input too long")

    # 內容過濾（根據需求）
    # ...

    return text

# 2. 輸出過濾
def filter_output(text: str) -> str:
    """過濾敏感輸出"""
    # 實現內容審核邏輯
    # ...

    return text
```

### 2. 資料隱私

```python
# 1. 不記錄敏感資料
logger.info(f"Request from user: {hash(user_id)}")  # ✅ 散列
logger.info(f"Request from user: {user_id}")        # ❌ 明文

# 2. 定期清理日誌
import os
from datetime import datetime, timedelta

def cleanup_old_logs(log_dir, days=7):
    """刪除舊日誌"""
    cutoff = datetime.now() - timedelta(days=days)

    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)
        if os.path.getmtime(filepath) < cutoff.timestamp():
            os.remove(filepath)
```

---

## 成本優化

### 1. 雲端部署

```python
# 使用 Spot 實例（AWS/GCP）
# - 成本降低 70-90%
# - 需要容錯機制

# Kubernetes 自動擴展配置
"""yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 1          # 低流量時最小實例
  maxReplicas: 10         # 高流量時最大實例
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
```

### 2. 批次處理

```python
class BatchProcessor:
    """批次處理以提高吞吐量"""

    def __init__(self, llm, max_batch_size=32, wait_time=0.1):
        self.llm = llm
        self.max_batch_size = max_batch_size
        self.wait_time = wait_time
        self.queue = []

    async def process(self, prompt):
        """添加到批次隊列"""
        future = asyncio.Future()
        self.queue.append((prompt, future))

        # 等待批次填滿或超時
        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()
        else:
            asyncio.create_task(self._wait_and_process())

        return await future

    async def _wait_and_process(self):
        """等待後處理批次"""
        await asyncio.sleep(self.wait_time)
        await self._process_batch()

    async def _process_batch(self):
        """處理當前批次"""
        if not self.queue:
            return

        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        prompts = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        # 批次生成
        results = self.llm.generate(prompts)

        # 返回結果
        for future, result in zip(futures, results):
            future.set_result(result)
```

---

## 總結

**金律**：
1. **測量 → 優化 → 驗證**：永遠基於資料決策
2. **漸進式優化**：不要一次性應用所有優化
3. **記錄一切**：實驗、配置、結果
4. **自動化**：測試、部署、監控
5. **安全第一**：隱私、輸入驗證、錯誤處理

**避免的陷阱**：
- ❌ 過早優化
- ❌ 忽視基準測試
- ❌ 在生產環境中使用未經驗證的量化
- ❌ 忽視監控和日誌
- ❌ 犧牲可維護性追求性能

**持續改進**：
- 關注最新研究和工具
- 參與社群討論
- 分享經驗和最佳實踐
- 定期審查和更新部署
