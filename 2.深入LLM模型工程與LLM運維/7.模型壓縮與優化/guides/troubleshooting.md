# 常見問題與故障排除

## 目錄
1. [顯存問題](#1-顯存問題)
2. [量化問題](#2-量化問題)
3. [訓練問題](#3-訓練問題)
4. [推理問題](#4-推理問題)
5. [部署問題](#5-部署問題)
6. [性能問題](#6-性能問題)

---

## 1. 顯存問題

### Q1.1: CUDA Out of Memory (OOM)

**問題**：
```
RuntimeError: CUDA out of memory. Tried to allocate XX GB
```

**可能原因**：
1. 批次大小過大
2. 序列長度過長
3. 模型過大
4. KV cache 佔用過多
5. 梯度累積設置不當

**解決方案**：

**方法 1：減小批次大小**
```python
# ❌ 錯誤
training_args = TrainingArguments(
    per_device_train_batch_size=8,
)

# ✅ 正確
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 減小
    gradient_accumulation_steps=8,  # 使用梯度累積
)
```

**方法 2：使用量化**
```python
# 使用 4-bit 量化
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
```

**方法 3：減小序列長度**
```python
tokenizer(
    text,
    max_length=512,  # 從 2048 減小到 512
    truncation=True
)
```

**方法 4：啟用梯度檢查點**
```python
model.gradient_checkpointing_enable()
```

**方法 5：清理快取**
```python
import torch
torch.cuda.empty_cache()
```

### Q1.2: 訓練時顯存持續增長

**問題**：顯存隨著訓練步數逐漸增加。

**原因**：
- Python 物件未正確釋放
- 計算圖未正確分離
- 日誌記錄保存過多張量

**解決方案**：

**方法 1：分離張量**
```python
# ❌ 錯誤
loss = model(inputs).loss
losses.append(loss)  # 保留計算圖

# ✅ 正確
loss = model(inputs).loss
losses.append(loss.item())  # 僅保存數值
```

**方法 2：手動釋放**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 定期清理
        if step % 100 == 0:
            torch.cuda.empty_cache()
```

### Q1.3: 推理時顯存不足

**問題**：推理時 OOM，但訓練正常。

**原因**：
- 批次大小過大
- KV cache 累積
- 生成序列過長

**解決方案**：

```python
# 方法 1：減小批次
outputs = model.generate(
    input_ids,
    max_new_tokens=100,  # 限制生成長度
    num_beams=1,         # 禁用 beam search
    do_sample=True,      # 使用採樣而非束搜索
)

# 方法 2：流式生成
from transformers import TextIteratorStreamer

streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(
    inputs=input_ids,
    streamer=streamer,
    max_new_tokens=100,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(new_text, end="")
```

---

## 2. 量化問題

### Q2.1: 量化後精度大幅下降

**問題**：量化後困惑度增加 > 5%。

**原因**：
1. 量化方法不適合
2. 校準資料不足或不合適
3. 敏感層被量化

**解決方案**：

**方法 1：使用更高精度**
```python
# INT4 → INT8
quantize_config = BaseQuantizeConfig(
    bits=8,  # 提高精度
)
```

**方法 2：改進校準資料**
```python
# ❌ 錯誤：使用不相關資料
calibration_data = load_dataset("random_text")

# ✅ 正確：使用目標領域資料
calibration_data = load_dataset("your_domain_data")
```

**方法 3：使用混合精度**
```python
# 保留敏感層為 FP16
from optimum.onnxruntime import ORTQuantizer

quantizer = ORTQuantizer.from_pretrained(model)
quantization_config = AutoQuantizationConfig.arm64(
    is_static=False,
    nodes_to_exclude=[  # 不量化的層
        "/transformer/ln_f/Add",
        "/lm_head/MatMul",
    ]
)
```

### Q2.2: GPTQ 量化過程卡住

**問題**：量化過程在某一層卡住不動。

**原因**：
- 層太大
- Hessian 計算數值不穩定
- 校準批次大小不當

**解決方案**：

```python
# 方法 1：調整參數
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,  # 增加阻尼
    desc_act=False,     # 禁用 desc_act
)

# 方法 2：減小批次
quantization_config = {
    "batch_size": 1,  # 從 128 減小到 1
}
```

### Q2.3: bitsandbytes 報錯

**問題**：
```
CUDA Setup failed despite GPU being available
```

**原因**：
- CUDA 版本不兼容
- bitsandbytes 安裝不正確

**解決方案**：

```bash
# 卸載並重新安裝
pip uninstall bitsandbytes -y

# 根據 CUDA 版本安裝
# CUDA 11.8
pip install bitsandbytes

# CUDA 12.1+
pip install bitsandbytes --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui

# 驗證
python -c "import bitsandbytes as bnb; print(bnb.cuda_setup.main())"
```

---

## 3. 訓練問題

### Q3.1: LoRA 訓練不收斂

**問題**：Loss 不下降或震盪。

**原因**：
1. 學習率不當
2. LoRA 秩太小
3. 資料問題

**解決方案**：

**方法 1：調整學習率**
```python
# ❌ 太高
training_args = TrainingArguments(learning_rate=3e-4)

# ✅ 合適
training_args = TrainingArguments(
    learning_rate=2e-4,
    warmup_ratio=0.03,    # 添加預熱
    lr_scheduler_type="cosine"
)
```

**方法 2：增加 LoRA 秩**
```python
# ❌ 太小
lora_config = LoraConfig(r=4)

# ✅ 增加
lora_config = LoraConfig(
    r=16,  # 從 4 增加到 16
    lora_alpha=32
)
```

**方法 3：添加正則化**
```python
lora_config = LoraConfig(
    r=16,
    lora_dropout=0.05,  # 添加 dropout
)
```

### Q3.2: 梯度爆炸/消失

**問題**：
```
RuntimeError: Gradients are NaN
```

**解決方案**：

```python
# 1. 啟用梯度裁剪
training_args = TrainingArguments(
    max_grad_norm=1.0,  # 梯度裁剪
)

# 2. 使用混合精度
training_args = TrainingArguments(
    fp16=True,          # 或 bf16=True
)

# 3. 降低學習率
training_args = TrainingArguments(
    learning_rate=1e-4,  # 降低
)

# 4. 檢查資料
def check_data(batch):
    for k, v in batch.items():
        if torch.isnan(v).any():
            print(f"NaN detected in {k}")
        if torch.isinf(v).any():
            print(f"Inf detected in {k}")
```

### Q3.3: QLoRA 訓練速度慢

**問題**：QLoRA 訓練比預期慢很多。

**原因**：
- 批次大小太小
- 未使用高效優化器
- 資料加載瓶頸

**解決方案**：

```python
# 1. 增加批次大小
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # 增加
    gradient_accumulation_steps=4,
)

# 2. 使用高效優化器
training_args = TrainingArguments(
    optim="paged_adamw_32bit",  # 分頁優化器
)

# 3. 優化資料加載
training_args = TrainingArguments(
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# 4. 啟用編譯（PyTorch 2.0+）
model = torch.compile(model)
```

---

## 4. 推理問題

### Q4.1: 推論速度慢

**問題**：推論速度遠低於預期。

**診斷**：

```python
import time

def benchmark_inference(model, tokenizer, prompt, num_runs=10):
    """基準測試推論速度"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # 預熱
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=10)

    # 測試
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=100)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    print(f"平均時間: {sum(times)/len(times):.3f}s")
    print(f"平均速度: {100 / (sum(times)/len(times)):.2f} tokens/s")
```

**解決方案**：

**方法 1：使用 vLLM**
```python
# ❌ 慢：使用 Transformers
outputs = model.generate(...)

# ✅ 快：使用 vLLM
from vllm import LLM

llm = LLM(model="model_path", quantization="gptq")
outputs = llm.generate(prompts)
```

**方法 2：啟用 KV cache**
```python
outputs = model.generate(
    input_ids,
    use_cache=True,  # 確保啟用
)
```

**方法 3：優化採樣**
```python
# ❌ 慢：使用 beam search
outputs = model.generate(num_beams=4)

# ✅ 快：使用採樣
outputs = model.generate(
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
)
```

### Q4.2: 生成品質差

**問題**：量化後生成重複、不連貫或錯誤。

**診斷**：

```python
# 檢查困惑度
from datasets import load_dataset

def evaluate_ppl(model, tokenizer):
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    total_loss = 0
    total_tokens = 0

    for example in test_data[:100]:
        inputs = tokenizer(example["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item() * inputs.input_ids.size(1)
            total_tokens += inputs.input_ids.size(1)

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Perplexity: {ppl:.2f}")
```

**解決方案**：

**方法 1：調整生成參數**
```python
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,      # 降低 (從 1.0)
    top_p=0.9,            # 添加
    top_k=50,             # 添加
    repetition_penalty=1.1,  # 添加
)
```

**方法 2：使用更高精度量化**
```python
# INT4 → INT8
quantize_config = BaseQuantizeConfig(bits=8)
```

**方法 3：混合精度**
```python
# 保留輸出層為 FP16
```

---

## 5. 部署問題

### Q5.1: vLLM 啟動失敗

**問題**：
```
ValueError: Cannot find model weights in the specified path
```

**解決方案**：

```bash
# 1. 檢查模型路徑
ls -la /path/to/model/

# 2. 下載模型
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./llama-2-7b

# 3. 啟動 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./llama-2-7b \
    --served-model-name llama-2-7b

# 4. 測試
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-2-7b",
        "prompt": "Hello",
        "max_tokens": 50
    }'
```

### Q5.2: TensorRT-LLM 編譯失敗

**問題**：
```
[TensorRT-LLM] ERROR: Could not build engine
```

**解決方案**：

```bash
# 1. 檢查 CUDA 版本
nvidia-smi

# 2. 檢查 TensorRT 版本
python -c "import tensorrt; print(tensorrt.__version__)"

# 3. 重新構建
python build.py \
    --model_dir ./llama-2-7b \
    --output_dir ./engine \
    --dtype float16 \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --log_level verbose  # 添加詳細日誌
```

### Q5.3: llama.cpp 在 Mac 上性能差

**問題**：在 M1/M2 Mac 上推論速度慢。

**解決方案**：

```bash
# 1. 重新編譯啟用 Metal
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make clean
LLAMA_METAL=1 make  # 啟用 Metal

# 2. 運行時使用 Metal
./main -m model.gguf -ngl 999 -p "Hello"
# -ngl 999: 將所有層卸載到 GPU
```

---

## 6. 性能問題

### Q6.1: GPU 利用率低

**問題**：GPU 利用率 < 50%。

**診斷**：

```bash
# 監控 GPU
nvidia-smi dmon -s u

# 使用 nvtop
nvtop

# PyTorch 性能分析
python -m torch.utils.bottleneck your_script.py
```

**原因**：
1. 批次大小太小
2. 資料加載瓶頸
3. CPU 預處理慢

**解決方案**：

```python
# 1. 增加批次大小
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # 增加
)

# 2. 優化資料加載
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # 多進程
    pin_memory=True,    # 固定內存
    prefetch_factor=2,  # 預取
)

# 3. 使用編譯模式
model = torch.compile(model, mode="max-autotune")
```

### Q6.2: 吞吐量低

**問題**：API 服務 QPS < 10。

**解決方案**：

**使用 vLLM 的 continuous batching**：
```python
from vllm import LLM

llm = LLM(
    model="model_path",
    max_num_batched_tokens=4096,  # 增加
    max_num_seqs=256,            # 並發序列數
)
```

**多實例部署**：
```yaml
# docker-compose.yml
services:
  vllm-1:
    image: vllm/vllm-openai
    deploy:
      replicas: 4  # 4個實例
```

**啟用並行**：
```python
llm = LLM(
    model="model_path",
    tensor_parallel_size=4,  # 4-way 張量並行
)
```

---

## 快速診斷檢查清單

### 顯存問題
- [ ] 減小批次大小
- [ ] 使用梯度累積
- [ ] 啟用量化
- [ ] 減小序列長度
- [ ] 啟用梯度檢查點

### 精度問題
- [ ] 檢查校準資料
- [ ] 使用更高精度量化
- [ ] 識別並保留敏感層
- [ ] 評估困惑度變化

### 速度問題
- [ ] 使用專用推論引擎（vLLM）
- [ ] 啟用量化
- [ ] 優化批次大小
- [ ] 使用合適的採樣方法
- [ ] 檢查 GPU 利用率

### 訓練問題
- [ ] 調整學習率
- [ ] 增加 LoRA 秩
- [ ] 啟用梯度裁剪
- [ ] 檢查資料品質
- [ ] 使用預熱策略

---

## 獲取幫助

**報告 Bug**：
1. 提供完整錯誤堆棧
2. 環境資訊（CUDA、PyTorch 版本等）
3. 最小可復現示例
4. 已嘗試的解決方案

**社群資源**：
- Hugging Face Forums
- GitHub Issues
- Discord/Slack 社群
- Stack Overflow

**調試技巧**：
```python
# 啟用調試日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 啟用 CUDA 同步錯誤檢查
CUDA_LAUNCH_BLOCKING=1 python your_script.py
```
