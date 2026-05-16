# LLM 開發與部署最佳實踐指南

> 從開發到生產的完整實踐經驗總結

---

## 📋 目錄

- [模型選擇](#模型選擇)
- [訓練最佳實踐](#訓練最佳實踐)
- [微調策略](#微調策略)
- [推論優化](#推論優化)
- [提示工程](#提示工程)
- [生產部署](#生產部署)
- [常見陷阱與解決方案](#常見陷阱與解決方案)

---

## 模型選擇

### 按任務類型選擇

```
問答/對話     → Claude 3.5, GPT-4o, Gemini 2.5
程式碼生成     → Claude 3.5 Sonnet, GPT-4o, DeepSeek-Coder
數學推理     → o1, DeepSeek-R1, Claude 3.5
長文檔理解   → Gemini 1.5 Pro (2M ctx), Claude 4 (1M ctx)
多模態      → GPT-4o, Gemini Pro Vision
本地部署    → LLaMA 3, Mistral, Qwen
```

### 按資源選擇

| GPU 配置 | 推薦模型 (FP16) | 推薦模型 (INT4) |
|---------|---------------|---------------|
| 4GB (RTX 3060) | 1B-3B | 7B |
| 8GB (RTX 3070) | 3B-7B | 13B |
| 12GB (RTX 3080) | 7B | 13B-30B |
| 24GB (RTX 4090) | 13B | 30B-70B |
| 40GB (A100) | 30B-70B | 70B-180B |
| 80GB (A100) | 70B | 180B+ |

### 選擇檢查清單

- ✅ 任務是否需要最新知識?→ 考慮 RAG 而非更大模型
- ✅ 是否需要多語言支持?→ LLaMA, Qwen, Gemini
- ✅ 預算限制?→ 開源模型 + 本地部署
- ✅ 延遲要求?→ 考慮模型大小 vs 質量權衡

---

## 訓練最佳實踐

### 預訓練 (Pre-training)

**資料準備:**
```python
# ✅ 好的做法
- 多樣化資料源 (書籍, 網頁, 程式碼, 論文)
- 高品質過濾 (去重, 有害內容過濾, 語言檢測)
- 適當的資料混合比例

# ❌ 避免
- 單一資料源
- 低品質/重複資料
- 忽略資料預處理
```

**訓練配置:**
```python
# 推薦超參數 (7B 模型)
config = {
    'learning_rate': 3e-4,
    'batch_size': 512,  # 全局批次大小
    'warmup_steps': 2000,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'lr_schedule': 'cosine',
    'optimizer': 'AdamW',
    'betas': (0.9, 0.95),
}

# ✅ 使用混合精度
use_amp = True

# ✅ 梯度檢查點節省內存
use_gradient_checkpointing = True

# ✅ 梯度累積模擬大批次
gradient_accumulation_steps = 8
```

**監控指標:**
- 訓練損失 (應該穩定下降)
- 驗證損失 (監控過擬合)
- 學習率曲線
- GPU 利用率 (目標 >90%)
- 梯度範數 (檢測不穩定)

---

## 微調策略

### 何時使用何種方法

```
少量資料 (<1K 樣本)    → 提示工程 + Few-shot
中量資料 (1K-10K)      → LoRA 微調
大量資料 (>10K)        → 完整微調
改善對話質量          → RLHF 或 DPO
領域適配              → 持續預訓練 + 微調
```

### LoRA 微調最佳實踐

```python
from peft import LoraConfig, get_peft_model

# ✅ 推薦配置
lora_config = LoraConfig(
    r=8,                    # 秩: 8-32 通常足夠
    lora_alpha=32,          # 通常是 r 的 2-4 倍
    target_modules=[        # 目標模塊
        "q_proj",
        "v_proj",
        "k_proj",           # 可選,增加表達能力
        "o_proj",           # 可選
    ],
    lora_dropout=0.05,      # 防止過擬合
    bias="none",            # 通常不訓練 bias
    task_type="CAUSAL_LM"
)

# ❌ 避免
- r 太大 (>64) → 失去 LoRA 的效率優勢
- r 太小 (<4) → 表達能力不足
- target_modules 太少 → 性能受限
```

### SFT 資料品質檢查

```python
# ✅ 高品質 SFT 資料特徵
good_example = {
    "instruction": "解釋機器學習中的過擬合",  # 清晰具體
    "input": "",
    "output": "過擬合是指模型在訓練資料上表現很好..."  # 詳細準確
}

# ❌ 避免
bad_examples = [
    {"instruction": "解釋過擬合", "output": "就是太擬合了"},  # 太簡短
    {"instruction": "寫程式碼", "output": "def foo():pass"},  # 太模糊
]

# 資料品質檢查清單
- ✅ 指令清晰明確
- ✅ 回答詳細準確
- ✅ 格式一致
- ✅ 無有害/偏見內容
- ✅ 多樣性充足
```

---

## 推論優化

### 優化優先級 (從高到低)

1. **使用專業推理框架** (最大提升)
   ```bash
   # vLLM (推薦)
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-2-7b-hf \
       --tensor-parallel-size 2

   # TensorRT-LLM (NVIDIA GPU)
   # SGLang (RAG/Agent 場景)
   ```

2. **量化** (2-4x 加速, 內存減半)
   ```python
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig

   # 4-bit 量化
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True
   )

   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       quantization_config=bnb_config,
       device_map="auto"
   )
   ```

3. **Flash Attention** (5-9x 訓練加速)
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       attn_implementation="flash_attention_2",
       torch_dtype=torch.float16
   )
   ```

4. **KV 快取** (必須啟用)
   ```python
   # ✅ 正確使用
   past_key_values = None
   for _ in range(max_tokens):
       outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
       past_key_values = outputs.past_key_values
       # 只處理新 token
       input_ids = outputs.logits[:, -1:].argmax(dim=-1)
   ```

5. **批處理** (提升吞吐量)
   ```python
   # ✅ 批次推理
   prompts = ["prompt1", "prompt2", "prompt3"]
   inputs = tokenizer(prompts, padding=True, return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=100)
   ```

### 延遲優化技巧

```python
# 減少首token延遲 (TTFT)
- 使用較小模型
- 減少提示長度
- 預填充 KV 快取

# 提升吞吐量
- 增大批次大小
- 使用 Continuous Batching (vLLM)
- Paged Attention

# 減少總延遲
- 投機解碼 (Speculative Decoding)
- 早停策略
- 最優的 top-p/top-k 設置
```

---

## 提示工程

### 提示結構模板

```python
# ✅ 結構化提示
prompt_template = """<|system|>
你是一個專業的{domain}專家。請提供準確、詳細的回答。
</|system|>

<|user|>
{instruction}

{input}
</|user|>

<|assistant|>
"""

# 示例
final_prompt = prompt_template.format(
    domain="Python 編程",
    instruction="解釋以下程式碼的作用",
    input="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
)
```

### Few-shot 示例設計

```python
# ✅ 有效的 Few-shot
few_shot_prompt = """
任務: 將句子分類為正面或負面情感

示例 1:
輸入: 這部電影太精彩了!
輸出: 正面

示例 2:
輸入: 服務態度很差
輸出: 負面

示例 3:
輸入: 還可以,沒什麼特別的
輸出: 中性

現在分類:
輸入: {new_input}
輸出:
"""

# 關鍵要點:
- ✅ 示例與任務高度相關
- ✅ 涵蓋不同情況
- ✅ 格式一致
- ✅ 3-5 個示例通常最佳
```

### 提示優化技巧

1. **明確性勝於簡潔**
   ```python
   # ❌ 模糊
   "總結文章"

   # ✅ 明確
   "用 3-5 個要點總結這篇文章的核心論點"
   ```

2. **指定輸出格式**
   ```python
   # ✅ 結構化輸出
   "請以 JSON 格式返回結果,包含以下欄位: {name, age, occupation}"
   ```

3. **思維鏈提示 (CoT)**
   ```python
   # ✅ 激發推理
   "讓我們一步步思考這個問題"
   "首先分析...然後考慮...最後得出結論"
   ```

---

## 生產部署

### 部署檢查清單

**模型準備:**
- ✅ 量化模型以減少內存
- ✅ 測試模型在各種輸入下的表現
- ✅ 設置合理的生成參數 (max_tokens, temperature)
- ✅ 準備後備方案 (模型故障時)

**基礎設施:**
- ✅ GPU/CPU 資源規劃
- ✅ 負載均衡配置
- ✅ 自動擴縮容
- ✅ 監控和告警設置

**安全與合規:**
- ✅ 輸入驗證和清理
- ✅ 輸出過濾 (有害內容檢測)
- ✅ 速率限制
- ✅ 日誌記錄 (GDPR 合規)

### Docker 部署示例

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN pip install vllm transformers

# 下載模型
RUN huggingface-cli download meta-llama/Llama-2-7b-hf

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "meta-llama/Llama-2-7b-hf", \
     "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  llm-server:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### 監控指標

```python
# 關鍵指標
metrics_to_monitor = {
    # 性能指標
    'requests_per_second': 'QPS',
    'tokens_per_second': '吞吐量',
    'time_to_first_token': 'TTFT',
    'total_latency': '總延遲',

    # 資源指標
    'gpu_utilization': 'GPU 利用率',
    'gpu_memory_used': 'GPU 內存',
    'cpu_utilization': 'CPU 利用率',

    # 品質指標
    'error_rate': '錯誤率',
    'timeout_rate': '超時率',
}
```

---

## 常見陷阱與解決方案

### 1. 內存不足 (OOM)

**症狀:** CUDA out of memory

**解決方案:**
```python
# 方案 1: 減小批次大小
batch_size = 1  # 或更小

# 方案 2: 梯度累積
accumulation_steps = 8
effective_batch_size = batch_size * accumulation_steps

# 方案 3: 梯度檢查點
model.gradient_checkpointing_enable()

# 方案 4: 量化
load_in_8bit=True  # 或 load_in_4bit=True

# 方案 5: DeepSpeed ZeRO
from deepspeed import zero
# 啟用 ZeRO-3
```

### 2. 訓練不穩定

**症狀:** Loss 爆炸或不收斂

**解決方案:**
```python
# ✅ 降低學習率
learning_rate = 1e-5  # 而非 1e-4

# ✅ 增加 warmup
warmup_ratio = 0.1  # 10% 的步數用於 warmup

# ✅ 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# ✅ 使用更穩定的優化器
optimizer = torch.optim.AdamW(params, lr=lr, eps=1e-8)
```

### 3. 推論速度慢

**問題診斷:**
```python
import time

# 測量各部分耗時
start = time.time()
tokens = tokenizer(text)
tokenize_time = time.time() - start

start = time.time()
outputs = model.generate(**tokens)
generate_time = time.time() - start

start = time.time()
result = tokenizer.decode(outputs[0])
decode_time = time.time() - start

print(f"Tokenize: {tokenize_time:.3f}s")
print(f"Generate: {generate_time:.3f}s")
print(f"Decode: {decode_time:.3f}s")
```

**優化:**
- 如果 tokenize 慢 → 批處理, 快取
- 如果 generate 慢 → 使用推理框架, 量化, Flash Attention
- 如果 decode 慢 → 批處理解碼

### 4. 模型輸出品質差

**檢查清單:**
```python
# 1. 檢查訓練資料品質
- 是否有足夠的高品質樣本?
- 資料是否平衡?
- 是否有噪聲/錯誤標籤?

# 2. 檢查超參數
- 學習率是否合適?
- 訓練步數是否足夠?
- 是否過早停止?

# 3. 檢查推理參數
temperature = 0.7    # 太高 → 隨機, 太低 → 重複
top_p = 0.9          # 調整採樣範圍
max_new_tokens = 512  # 是否太短?

# 4. 嘗試更好的提示
- 添加 Few-shot 示例
- 更明確的指令
- 思維鏈提示
```

### 5. 模型過擬合

**症狀:** 訓練 loss 低,驗證 loss 高

**解決方案:**
```python
# 1. 增加資料
- 資料增強
- 收集更多樣本

# 2. 正則化
- LoRA dropout: 0.1
- Weight decay: 0.01
- Early stopping

# 3. 減少模型復雜度
- 減小 LoRA rank
- 減少訓練 epochs

# 4. 驗證集選擇
- 確保驗證集代表真實分布
```

---

## 性能優化速查

### 訓練加速

| 方法 | 加速比 | 內存影響 | 實施難度 |
|------|--------|---------|---------|
| Flash Attention 2 | 5-9x | 無影響 | 簡單 |
| 混合精度 (FP16) | 2-3x | 減半 | 簡單 |
| 梯度檢查點 | 0.8x (慢) | -50% | 簡單 |
| DeepSpeed ZeRO-3 | 1.5-2x | -70% | 中等 |
| 梯度累積 | 無影響 | 無影響 | 簡單 |

### 推理加速

| 方法 | 加速比 | 質量影響 | 實施難度 |
|------|--------|---------|---------|
| vLLM | 3-5x | 無 | 簡單 |
| INT8 量化 | 2-3x | 微小 | 簡單 |
| INT4 量化 | 3-4x | 小 | 簡單 |
| Flash Attention | 2-3x | 無 | 簡單 |
| KV 快取 | 10x+ | 無 | 必須 |
| 投機解碼 | 2-3x | 無 | 中等 |

---

## 總結

### 黃金法則

1. **從小做起**: 先用小模型驗證流程
2. **測量優先**: 優化前先基準測試
3. **漸進改進**: 一次改變一個變量
4. **文檔記錄**: 記錄所有實驗和配置
5. **自動化**: 自動化重複任務

### 推薦工作流

```
1. 定義任務 → 選擇基礎模型
       ↓
2. 準備資料 → 質量檢查
       ↓
3. 快速原型 → Few-shot / 提示工程
       ↓
4. 微調 (如需要) → LoRA / Full Fine-tuning
       ↓
5. 評估 → 多個測試集
       ↓
6. 優化 → 量化 / 推理框架
       ↓
7. 部署 → 監控 + 迭代
```

---

**最後更新**: 2025-01

**貢獻**: 歡迎提交 PR 添加更多最佳實踐!
