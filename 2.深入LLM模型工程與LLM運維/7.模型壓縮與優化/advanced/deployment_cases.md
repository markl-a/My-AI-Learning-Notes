# 實際部署案例 (Real-World Deployment Cases)

## 目錄
1. [案例概覽](#1-案例概覽)
2. [案例 1：企業級問答系統](#2-案例-1企業級問答系統)
3. [案例 2：移動端應用](#3-案例-2移動端應用)
4. [案例 3：高吞吐量API服務](#4-案例-3高吞吐量api服務)
5. [案例 4：邊緣設備推理](#5-案例-4邊緣設備推理)
6. [案例 5：本地私有部署](#6-案例-5本地私有部署)
7. [性能與成本對比](#7-性能與成本對比)

---

## 1. 案例概覽

### 1.1 典型部署場景

| 場景 | 模型大小 | 硬體 | 量化方法 | 主要挑戰 |
|------|---------|------|----------|---------|
| 企業 QA | 7B-13B | GPU 伺服器 | INT8/INT4 | 延遲 < 1s |
| 移動應用 | 1B-3B | 移動設備 | INT4/INT2 | 內存 < 4GB |
| API 服務 | 7B-70B | 多 GPU | INT8 + LoRA | 高吞吐量 |
| 邊緣設備 | 1B-7B | CPU/NPU | INT8/GGUF | 功耗 < 10W |
| 本地部署 | 7B-30B | 消費級 GPU | GGUF Q4/Q5 | 隱私保護 |

---

## 2. 案例 1：企業級問答系統

### 2.1 需求分析

**業務需求**：
- 內部知識庫問答（10萬篇文檔）
- 支持 1000 並發用戶
- 響應時間 < 500ms (P95)
- 準確率 > 90%

**技術限制**：
- 預算：4張 A100 GPU (40GB)
- 不能使用外部 API（資料隱私）
- 需要支持持續學習

### 2.2 解決方案

**模型選擇**：LLaMA-2-13B + LoRA

**量化策略**：
```
基座模型：GPTQ 4-bit
LoRA 適配器：FP16（僅 ~20MB）
總顯存：~9GB per GPU
```

**架構設計**：
```
┌──────────────────────────────────────┐
│  Load Balancer (nginx)               │
└──────────┬───────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐    ┌───▼───┐
│ GPU 1 │    │ GPU 2 │  vLLM (GPTQ)
│ 13B   │    │ 13B   │  Batch=32
└───┬───┘    └───┬───┘  Continuous batching
    │             │
┌───▼───┐    ┌───▼───┐
│ GPU 3 │    │ GPU 4 │  LoRA 適配器
│ 13B   │    │ 13B   │  動態加載
└───────┘    └───────┘
```

**實作步驟**：

```python
# 1. 準備量化模型
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

# 載入並量化
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantize_config=quantize_config
)

# 量化（使用企業文檔作為校準資料）
model.quantize(calibration_dataset)

# 保存
model.save_quantized("./llama-2-13b-gptq-4bit")

# 2. 訓練 LoRA 適配器（針對企業知識）
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# 在內部資料上微調
model = get_peft_model(quantized_model, lora_config)
# ... 訓練過程 ...
model.save_pretrained("./lora-adapters/enterprise-qa")

# 3. 使用 vLLM 部署
from vllm import LLM, SamplingParams

llm = LLM(
    model="./llama-2-13b-gptq-4bit",
    quantization="gptq",
    tensor_parallel_size=1,
    max_num_batched_tokens=4096,
    enable_lora=True,
)

# 載入 LoRA
llm.load_lora_adapter(
    lora_name="enterprise-qa",
    lora_path="./lora-adapters/enterprise-qa"
)

# 推理
prompts = [...]  # 批次問題
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, sampling_params, lora_name="enterprise-qa")
```

**API 服務**：
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    context: str = ""

@app.post("/api/qa")
async def qa_endpoint(query: Query):
    prompt = f"""基於以下上下文回答問題：

上下文：{query.context}

問題：{query.question}

回答："""

    outputs = llm.generate([prompt], sampling_params, lora_name="enterprise-qa")

    return {
        "answer": outputs[0].outputs[0].text,
        "confidence": calculate_confidence(outputs[0]),
    }

# 啟動服務
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2.3 性能結果

**性能指標**：
```
吞吐量：~120 queries/second (4 GPU)
延遲 (P50)：280ms
延遲 (P95)：450ms
延遲 (P99)：680ms
準確率：92.3%
顯存使用：8.5 GB per GPU
```

**成本分析**：
```
硬體成本：$40,000 (4x A100)
運行成本：~$8/hour (雲端) 或 $0 (自建)
vs. GPT-4 API：~$5,000/month (估計)
投資回收期：~8 個月
```

---

## 3. 案例 2：移動端應用

### 3.1 需求分析

**業務需求**：
- 離線語音助手（iOS/Android）
- 設備內推理（隱私）
- 應用大小 < 500MB
- 推論延遲 < 200ms
- 功耗合理（不發燙）

**技術限制**：
- RAM < 4GB
- 無專用 GPU（僅 CPU/NPU）
- 電池續航考量

### 3.2 解決方案

**模型選擇**：Phi-2 (2.7B) / Gemma-2B

**量化策略**：
```
方法：llama.cpp (GGUF)
格式：Q4_K_M（4-bit）
大小：~1.5GB
```

**移動端優化**：

```python
# 1. 轉換為 GGUF 格式
# 使用 llama.cpp 工具鏈
"""
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 編譯
make

# 轉換模型
python convert.py /path/to/phi-2 \
    --outfile phi-2-f16.gguf \
    --outtype f16

# 量化為 Q4_K_M
./quantize phi-2-f16.gguf phi-2-Q4_K_M.gguf Q4_K_M
"""

# 2. iOS 集成（Swift）
"""swift
import llama_cpp

class LLMEngine {
    private var context: OpaquePointer?

    init(modelPath: String) {
        // 載入模型
        let params = llama_context_default_params()
        params.n_ctx = 2048
        params.n_threads = 4  // iPhone 性能核心數
        params.use_mlock = true

        context = llama_init_from_file(modelPath, params)
    }

    func generate(prompt: String, maxTokens: Int = 100) -> String {
        // Tokenize
        let tokens = tokenize(prompt)

        // 推理
        var output = ""
        for _ in 0..<maxTokens {
            let nextToken = llama_sample_token(context, ...)
            if nextToken == eos_token { break }

            output += decode(nextToken)
            llama_eval(context, [nextToken], ...)
        }

        return output
    }
}

// 使用
let engine = LLMEngine(modelPath: Bundle.main.path(forResource: "phi-2-Q4_K_M", ofType: "gguf")!)
let response = engine.generate(prompt: "What is the weather like?")
"""

# 3. Android 集成（Kotlin + JNI）
"""kotlin
class LLMEngine(private val context: Context) {
    private var nativeHandle: Long = 0

    init {
        System.loadLibrary("llama-android")

        val modelPath = extractModel(context)
        nativeHandle = nativeInit(modelPath, 4) // 4 threads
    }

    fun generate(prompt: String, maxTokens: Int = 100): String {
        return nativeGenerate(nativeHandle, prompt, maxTokens)
    }

    private external fun nativeInit(modelPath: String, nThreads: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, maxTokens: Int): String

    fun cleanup() {
        nativeCleanup(nativeHandle)
    }
}

// 使用
val engine = LLMEngine(applicationContext)
val response = engine.generate("Tell me a joke")
"""
```

**性能優化**：

```cpp
// C++ 優化（llama.cpp 後端）
#include "llama.h"

// 啟用 ARM NEON 加速
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// 量化推論優化
void optimized_inference() {
    llama_context_params params = llama_context_default_params();

    // 移動端優化參數
    params.n_ctx = 1024;           // 較小上下文
    params.n_batch = 128;          // 批次大小
    params.n_threads = 4;          // 性能核心
    params.f16_kv = true;          // KV cache 使用 FP16
    params.use_mmap = true;        // 內存映射（減少 RAM）
    params.use_mlock = false;      // 移動端不鎖定內存

    // ... 推理邏輯 ...
}
```

### 3.3 應用集成

**應用結構**：
```
MyApp/
├── Models/
│   ├── phi-2-Q4_K_M.gguf         (1.5 GB)
│   └── tokenizer.json
├── llama.cpp/
│   ├── llama-android.so          (Android)
│   └── libllama.dylib             (iOS)
├── UI/
│   ├── ChatView
│   └── SettingsView
└── Core/
    ├── LLMEngine
    ├── PromptManager
    └── ResponseCache
```

**功能實現**：
```python
# 智能快取策略
class ResponseCache:
    """快取常見查詢以減少推理"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, prompt):
        # 模糊匹配
        for cached_prompt, response in self.cache.items():
            if similarity(prompt, cached_prompt) > 0.95:
                return response
        return None

    def set(self, prompt, response):
        if len(self.cache) >= self.max_size:
            # LRU 淘汰
            self.cache.pop(next(iter(self.cache)))
        self.cache[prompt] = response

# 批次處理
class BatchProcessor:
    """批次處理以提高吞吐量"""

    def __init__(self, engine, batch_size=4):
        self.engine = engine
        self.batch_size = batch_size
        self.queue = []

    async def process(self, prompt):
        self.queue.append(prompt)

        if len(self.queue) >= self.batch_size:
            return await self._flush()
        else:
            # 等待或超時
            await asyncio.sleep(0.05)
            return await self._flush()

    async def _flush(self):
        if not self.queue:
            return []

        prompts = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]

        return self.engine.generate_batch(prompts)
```

### 3.4 性能結果

**性能指標（iPhone 14 Pro）**：
```
首次載入：2.3s
推論延遲：150-180ms (per token)
RAM 使用：~2.1 GB
電池消耗：~5% per hour (持續使用)
應用大小：420 MB
```

**Android（Snapdragon 8 Gen 2）**：
```
首次載入：2.8s
推論延遲：180-220ms (per token)
RAM 使用：~2.3 GB
應用大小：450 MB
```

---

## 4. 案例 3：高吞吐量API服務

### 3.1 需求分析

**業務需求**：
- SaaS 文字生成 API
- 支持 10,000+ QPS
- 多租戶（每個租戶自定義模型）
- 成本可控

**技術限制**：
- 雲端部署（AWS/GCP/Azure）
- 延遲 < 2s (P99)
- 可擴展架構

### 3.2 解決方案

**架構設計**：
```
                    ┌─────────────┐
                    │  API Gateway │
                    │  (Kong/APIG) │
                    └──────┬──────┘
                           │
                   ┌───────┴────────┐
                   │  Load Balancer │
                   │   (AWS ALB)    │
                   └───────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ vLLM    │       │ vLLM    │       │ vLLM    │
   │ Pod 1   │       │ Pod 2   │       │ Pod N   │
   │ 8xA100  │       │ 8xA100  │       │ 8xA100  │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   LoRA      │
                    │   Registry  │
                    │  (S3/GCS)   │
                    └─────────────┘
```

**核心技術**：

```python
# 1. vLLM 服務配置
# vllm_config.yaml
"""yaml
model: meta-llama/Llama-2-70b-hf
tensor_parallel_size: 8
quantization: awq  # AWQ INT4
max_num_batched_tokens: 8192
max_num_seqs: 256
enable_lora: true
max_loras: 32  # 支持 32 個並發 LoRA
max_lora_rank: 64
"""

# 2. 動態 LoRA 加載
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import redis

class MultiTenantLLMService:
    """多租戶 LLM 服務"""

    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Llama-2-70b-hf",
            quantization="awq",
            tensor_parallel_size=8,
            enable_lora=True,
            max_loras=32,
        )

        self.redis = redis.Redis(host='localhost', port=6379)
        self.lora_cache = {}

    async def generate(self, tenant_id: str, prompt: str):
        """為特定租戶生成文字"""

        # 獲取租戶的 LoRA 路徑
        lora_path = await self.get_tenant_lora(tenant_id)

        # 建立 LoRA 請求
        lora_request = LoRARequest(
            lora_name=f"tenant_{tenant_id}",
            lora_int_id=hash(tenant_id) % 32,  # 分配 LoRA slot
            lora_local_path=lora_path
        ) if lora_path else None

        # 生成
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512
        )

        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request
        )

        return outputs[0].outputs[0].text

    async def get_tenant_lora(self, tenant_id: str):
        """獲取租戶 LoRA（帶快取）"""

        # 檢查快取
        if tenant_id in self.lora_cache:
            return self.lora_cache[tenant_id]

        # 從 Redis 獲取
        lora_path = self.redis.get(f"lora:{tenant_id}")

        if lora_path:
            lora_path = lora_path.decode('utf-8')
            self.lora_cache[tenant_id] = lora_path

        return lora_path

# 3. FastAPI 服務
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()
service = MultiTenantLLMService()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/v1/generate")
async def generate_endpoint(
    request: GenerateRequest,
    x_tenant_id: str = Header(...),
    x_api_key: str = Header(...)
):
    # 驗證 API key
    if not await validate_api_key(x_api_key, x_tenant_id):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 生成
    text = await service.generate(x_tenant_id, request.prompt)

    return {
        "text": text,
        "model": "llama-2-70b-awq",
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(text.split()),
        }
    }

# 4. 自動擴展配置（Kubernetes）
"""yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 8
          requests:
            nvidia.com/gpu: 8
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-70b-hf"
        - name: QUANTIZATION
          value: "awq"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_queue_size
      target:
        type: AverageValue
        averageValue: "100"
"""
```

### 3.3 性能優化

**Continuous Batching**：
```python
# vLLM 自動進行 continuous batching
# 配置批次參數以最大化吞吐量

# 監控指標
import prometheus_client

# 批次大小分佈
batch_size_histogram = prometheus_client.Histogram(
    'vllm_batch_size',
    'Batch size distribution',
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256]
)

# 隊列深度
queue_depth = prometheus_client.Gauge(
    'vllm_queue_depth',
    'Number of pending requests'
)

# 吞吐量
throughput = prometheus_client.Counter(
    'vllm_tokens_generated_total',
    'Total tokens generated'
)
```

### 3.4 性能結果

**性能指標（8x A100 per pod）**：
```
吞吐量：~15,000 tokens/second per pod
並發請求：~250 requests
延遲 (P50)：450ms
延遲 (P95)：1.2s
延遲 (P99)：1.8s
GPU 利用率：~85%
```

**成本分析（AWS）**：
```
實例類型：p4d.24xlarge (8x A100)
成本：$32.77/hour

處理能力：~50M tokens/day per instance
成本效率：~$0.016 per 1K tokens

vs. OpenAI GPT-3.5：
  - OpenAI：$0.002 per 1K tokens（輸出）
  - 但需要考慮：
    * 資料隱私
    * 可定制性（LoRA）
    * 服務穩定性
```

---

## 5. 案例 4：邊緣設備推理

### 3.1 需求分析

**業務需求**：
- 智能攝像頭（圖像描述）
- 工業檢測設備
- 功耗 < 10W
- 成本 < $500

**技術限制**：
- ARM CPU（如 Raspberry Pi 4/5）
- 或專用 NPU（如 Coral TPU）
- 無網絡連接（離線）

### 5.2 解決方案

**硬體選擇**：
```
選項 1：Raspberry Pi 5 (8GB)
  - CPU: Cortex-A76 × 4
  - RAM: 8GB
  - 成本: ~$80

選項 2：NVIDIA Jetson Orin Nano
  - GPU: 1024-core NVIDIA Ampere
  - RAM: 8GB
  - 成本: ~$499

選項 3：Google Coral Dev Board
  - TPU: Edge TPU
  - RAM: 4GB
  - 成本: ~$150
```

**模型選擇**：TinyLlama (1.1B) / Phi-2 (2.7B)

**實作（Raspberry Pi + llama.cpp）**：

```bash
# 1. 編譯 llama.cpp（ARM NEON 優化）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_NO_ACCELERATE=1 LLAMA_NO_METAL=1

# 2. 轉換和量化模型
python convert.py ~/models/TinyLlama-1.1B --outtype f16
./quantize TinyLlama-1.1B-f16.gguf TinyLlama-1.1B-Q4_0.gguf Q4_0

# 3. 測試推理
./main -m TinyLlama-1.1B-Q4_0.gguf \
       -p "Describe this image:" \
       -n 128 \
       -t 4 \  # 4 線程
       --mlock
```

**Python 集成**：
```python
from llama_cpp import Llama

class EdgeLLM:
    """邊緣設備 LLM 推論引擎"""

    def __init__(self, model_path, n_ctx=512, n_threads=4):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=128,
            use_mlock=True,  # 防止交換
            n_gpu_layers=0,  # CPU only
        )

    def generate(self, prompt, max_tokens=100):
        """生成文字"""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["</s>"],
            echo=False
        )

        return output['choices'][0]['text']

# 圖像描述應用
import cv2
from PIL import Image

class ImageCaptioner:
    """圖像描述生成器"""

    def __init__(self):
        self.llm = EdgeLLM("TinyLlama-1.1B-Q4_0.gguf")
        # 可以集成 CLIP 等視覺模型

    def caption(self, image_path):
        """生成圖像描述"""

        # 1. 提取圖像特徵（簡化示例）
        image = Image.open(image_path)
        # 實際應用中應使用 CLIP 等模型提取特徵

        # 2. 構建提示
        prompt = f"""Describe the following image:
[Image features would be embedded here]

Description:"""

        # 3. 生成描述
        caption = self.llm.generate(prompt, max_tokens=50)

        return caption.strip()

# 使用
captioner = ImageCaptioner()
description = captioner.caption("photo.jpg")
print(description)
```

### 5.3 性能結果

**Raspberry Pi 5 性能**：
```
模型：TinyLlama-1.1B-Q4_0
推論速度：~8 tokens/second
首 token 延遲：~500ms
RAM 使用：~800MB
功耗：~5W
成本：$80
```

**Jetson Orin Nano 性能**：
```
模型：Phi-2-Q4_K_M
推論速度：~35 tokens/second
首 token 延遲：~150ms
RAM 使用：~2.5GB
功耗：~8W
成本：$499
```

---

## 6. 案例 5：本地私有部署

### 6.1 需求分析

**業務需求**：
- 個人/小團隊使用
- 完全隱私（本地運行）
- 支持多種任務（寫作、編程、翻譯）
- 易於使用（GUI）

**技術限制**：
- 消費級硬體（RTX 4090 或 Mac Studio）
- 無技術背景用戶

### 6.2 解決方案

**方案 A：Windows/Linux + Oobabooga Text Generation WebUI**

```bash
# 1. 安裝 Text Generation WebUI
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui

# 2. 一鍵啟動腳本
./start_linux.sh  # Linux
./start_windows.bat  # Windows
./start_macos.sh  # macOS

# 3. 下載模型（自動 4-bit 量化）
# 在 Web UI 中：
# Model -> Download -> 輸入 "meta-llama/Llama-2-13b-chat-hf"
# Load in 4-bit: True
# Compute dtype: bfloat16
```

**方案 B：macOS + LM Studio**

```bash
# 1. 下載 LM Studio
# https://lmstudio.ai/

# 2. 在 UI 中搜索並下載模型
# 搜索 "llama-2-13b" -> 選擇 Q4_K_M 格式

# 3. 本地 API 伺服器
# Settings -> Local Server -> Start Server
# 兼容 OpenAI API 格式
```

**方案 C：自定義部署（高級用戶）**

```python
# docker-compose.yml
"""yaml
version: '3.8'

services:
  llm-server:
    image: ghcr.io/huggingface/text-generation-inference:latest
    ports:
      - "8080:80"
    volumes:
      - ./models:/data
    environment:
      - MODEL_ID=TheBloke/Llama-2-13B-chat-GPTQ
      - QUANTIZE=gptq
      - NUM_SHARD=1
      - MAX_INPUT_LENGTH=4096
      - MAX_TOTAL_TOKENS=8192
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_API_BASE_URL=http://llm-server:80
    depends_on:
      - llm-server
"""

# 啟動
# docker-compose up -d
```

**集成到工作流**：

```python
# VS Code 擴展配置
# settings.json
"""json
{
    "llm-copilot.apiEndpoint": "http://localhost:8080/v1",
    "llm-copilot.model": "llama-2-13b-chat",
    "llm-copilot.temperature": 0.2,
    "llm-copilot.maxTokens": 500
}
"""

# Obsidian 插件配置
"""json
{
    "endpoint": "http://localhost:8080/v1/chat/completions",
    "model": "llama-2-13b-chat",
    "systemPrompt": "You are a helpful writing assistant."
}
"""
```

### 6.3 性能結果

**RTX 4090 (24GB)**：
```
模型：Llama-2-13B-GPTQ-4bit
推論速度：~60 tokens/second
並發用戶：1-2
顯存使用：~9GB
成本：$1,600 (GPU)
```

**Mac Studio (M2 Ultra, 192GB)**：
```
模型：Llama-2-13B-Q5_K_M (GGUF)
推論速度：~25 tokens/second
並發用戶：1
RAM 使用：~15GB
成本：$6,000+
```

---

## 7. 性能與成本對比

### 7.1 綜合對比表

| 場景 | 硬體 | 模型 | 量化 | 吞吐量 | 延遲 | 成本 |
|------|------|------|------|--------|------|------|
| 企業 QA | 4x A100 | LLaMA-13B | GPTQ-4bit | 120 QPS | 280ms | $40k |
| 移動端 | iPhone 14 | Phi-2 | Q4_K_M | ~6 tok/s | 180ms | $1k |
| API 服務 | 8x A100 | LLaMA-70B | AWQ-4bit | 15k tok/s | 450ms | $33/hr |
| 邊緣設備 | RPi 5 | TinyLlama | Q4_0 | 8 tok/s | 500ms | $80 |
| 本地部署 | RTX 4090 | LLaMA-13B | GPTQ-4bit | 60 tok/s | 100ms | $1.6k |

### 7.2 成本效益分析

**雲端 vs. 自建**：
```
場景：API 服務（1M requests/day）

雲端（AWS p4d.24xlarge）：
  - 成本：$32.77/hour × 24 = $786/day
  - 優勢：彈性擴展、免維護
  - 適合：流量波動大

自建（本地資料中心）：
  - 成本：$40k (初始) + $200/month (電力)
  - ROI：~2 個月
  - 優勢：長期成本低、資料隱私
  - 適合：穩定負載、隱私敏感
```

### 7.3 選擇建議

**決策樹**：
```
需要極低延遲 (< 100ms) ?
├─ Yes → 使用更小模型 (1B-7B) + 激進量化
└─ No
    └─ 資料敏感 ?
        ├─ Yes → 本地部署
        └─ No
            └─ 高 QPS (> 1000) ?
                ├─ Yes → 雲端 + vLLM
                └─ No → 本地部署 + GGUF
```

---

## 總結

### 關鍵要點

1. **因地制宜選擇方案**：
   - 沒有一刀切的解決方案
   - 根據需求、預算、技術能力選擇

2. **量化是關鍵**：
   - 4-bit 量化是當前最佳平衡點
   - 配合合適工具（GPTQ/AWQ/GGUF）

3. **硬體利用**：
   - vLLM 適合 GPU 伺服器
   - llama.cpp 適合 CPU/邊緣設備
   - TensorRT-LLM 適合 NVIDIA 生態

4. **成本考量**：
   - 雲端：靈活但長期成本高
   - 自建：初期投資大但 ROI 快

5. **用戶體驗**：
   - 延遲是關鍵指標
   - 穩定性比峰值性能更重要

### 延伸資源

**部署工具**：
- vLLM: https://github.com/vllm-project/vllm
- Text Generation WebUI: https://github.com/oobabooga/text-generation-webui
- LM Studio: https://lmstudio.ai/
- Ollama: https://ollama.ai/

**硬體指南**：
- GPU 選擇: https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
- 邊緣設備對比: https://developer.nvidia.com/embedded/develop/hardware
