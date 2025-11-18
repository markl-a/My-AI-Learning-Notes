# 模型部署與運維

## 目錄
1. [部署環境選擇](#1-部署環境選擇)
   - 1.1 [GPU vs CPU](#11-gpu-vs-cpu)
   - 1.2 [雲端 vs 本機](#12-雲端-vs-本機)
   - 1.3 [硬體需求估算](#13-硬體需求估算)
2. [模型服務化與 API 化](#2-模型服務化與-api-化)
   - 2.1 [推理框架選擇](#21-推理框架選擇)
   - 2.2 [Hugging Face Inference Endpoints](#22-hugging-face-inference-endpoints)
   - 2.3 [vLLM 加速](#23-vllm-加速)
   - 2.4 [TensorRT-LLM](#24-tensorrt-llm)
3. [系統架構與維護](#3-系統架構與維護)
   - 3.1 [負載均衡](#31-負載均衡)
   - 3.2 [模型版本管理](#32-模型版本管理)
   - 3.3 [監控與記錄](#33-監控與記錄)
   - 3.4 [故障恢復](#34-故障恢復)
4. [資料安全與隱私](#4-資料安全與隱私)
5. [成本優化](#5-成本優化)
6. [實作範例](#6-實作範例)
7. [參考資源](#7-參考資源)

---

## 1. 部署環境選擇

### 1.1 GPU vs CPU

#### GPU 部署

**優勢**：
- 推理速度快（10-100x）
- 支援批次處理
- 適合高吞吐量場景

**劣勢**：
- 成本高
- 耗電大
- 需要專業維護

**適用場景**：
- 線上服務（低延遲需求）
- 大規模批次推理
- 7B+ 參數模型

#### CPU 部署

**優勢**：
- 成本低
- 可擴展性好（橫向擴展）
- 維護簡單

**劣勢**：
- 推理速度慢
- 不適合大模型

**適用場景**：
- 小型模型（< 3B）
- 低頻請求
- 邊緣設備
- 本地部署

#### 性能對比

| 模型 | 硬體 | 推理速度 (tokens/s) | 成本 |
|------|------|---------------------|------|
| 7B | A100 (80GB) | 100-150 | 高 |
| 7B | 4x CPU (64核) | 10-20 | 中 |
| 7B | llama.cpp (CPU) | 20-40 | 低 |
| 13B | A100 (80GB) | 60-90 | 高 |
| 70B | 4x A100 (80GB) | 20-30 | 極高 |

### 1.2 雲端 vs 本機

#### 雲端部署

**優勢**：
- 彈性擴展
- 無需硬體維護
- 高可用性
- 全球分佈

**劣勢**：
- 持續成本
- 資料隱私風險
- 網路延遲

**主要提供商**：

| 提供商 | GPU 類型 | 特點 | 價格範圍 |
|--------|---------|------|---------|
| AWS (SageMaker) | A100, H100 | 成熟、穩定 | $$$ |
| GCP (Vertex AI) | A100, TPU | AI 優化 | $$$ |
| Azure (ML Studio) | A100, V100 | 企業整合 | $$$ |
| Lambda Labs | A100, H100 | 專注 GPU | $$ |
| RunPod | A100, H100 | 便宜、靈活 | $ |
| Hugging Face | A100 | 易用 | $$ |

#### 本機部署

**優勢**：
- 完全控制
- 無持續成本
- 資料隱私
- 無網路依賴

**劣勢**：
- 初期投資大
- 維護成本高
- 擴展困難

**適用場景**：
- 高度敏感資料
- 長期大量使用
- 離線環境

### 1.3 硬體需求估算

#### 顯存需求計算

**基本公式**：
```
顯存 (GB) = 模型參數量 × 精度 × (1 + 開銷)

其中：
- FP32: 4 bytes
- FP16/BF16: 2 bytes
- INT8: 1 byte
- INT4: 0.5 bytes
- 開銷: 通常 1.2 (20% 額外開銷)
```

**範例計算**：

```python
def estimate_gpu_memory(params_b, precision="fp16", overhead=1.2):
    """
    估算 GPU 顯存需求

    Args:
        params_b: 參數量（十億）
        precision: 精度（fp32, fp16, int8, int4）
        overhead: 開銷係數

    Returns:
        所需顯存（GB）
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    params = params_b * 1e9
    bytes_needed = params * bytes_per_param[precision]
    gb_needed = bytes_needed / 1e9 * overhead

    return gb_needed

# 範例
print(f"LLaMA-7B (FP16): {estimate_gpu_memory(7, 'fp16'):.1f} GB")
print(f"LLaMA-7B (INT8): {estimate_gpu_memory(7, 'int8'):.1f} GB")
print(f"LLaMA-7B (INT4): {estimate_gpu_memory(7, 'int4'):.1f} GB")
print(f"LLaMA-70B (FP16): {estimate_gpu_memory(70, 'fp16'):.1f} GB")
```

輸出：
```
LLaMA-7B (FP16): 16.8 GB
LLaMA-7B (INT8): 8.4 GB
LLaMA-7B (INT4): 4.2 GB
LLaMA-70B (FP16): 168.0 GB
```

#### 推薦配置

| 模型規模 | 推薦 GPU | 精度 | 說明 |
|---------|---------|------|------|
| < 3B | RTX 3060 (12GB) | FP16 | 入門級 |
| 7B | RTX 4090 (24GB) | FP16 | 消費級最佳 |
| 7B | A10 (24GB) | FP16 | 雲端經濟 |
| 13B | A100 (40GB) | FP16 | 專業級 |
| 13B | 2x RTX 4090 | INT8 | 消費級多卡 |
| 70B | 4x A100 (80GB) | FP16 | 大型模型 |
| 70B | 8x A100 (40GB) | INT8 | 經濟方案 |

---

## 2. 模型服務化與 API 化

### 2.1 推理框架選擇

#### 主流框架對比

| 框架 | 優勢 | 劣勢 | 適用場景 |
|------|------|------|---------|
| **vLLM** | 極快、PagedAttention | 功能較少 | 高吞吐量生產 |
| **TGI** (Text Generation Inference) | 易用、功能全 | 速度中等 | Hugging Face 生態 |
| **TensorRT-LLM** | 極致優化 | 複雜、NVIDIA only | NVIDIA GPU 極限性能 |
| **llama.cpp** | CPU 優化 | GPU 支援有限 | 本地/邊緣部署 |
| **FastChat** | 多模型支援 | 速度一般 | 研究、快速原型 |
| **Ray Serve** | 分散式、可擴展 | 複雜度高 | 大規模分散式 |

### 2.2 Hugging Face Inference Endpoints

**Hugging Face Inference Endpoints** 提供託管的模型推理服務，無需管理基礎設施。

#### 創建 Endpoint

```python
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    name="my-llama-endpoint",
    repository="meta-llama/Llama-2-7b-chat-hf",
    framework="pytorch",
    task="text-generation",
    accelerator="gpu",
    instance_type="g4dn.xlarge",  # AWS 實例類型
    instance_size="medium",
    region="us-east-1",
    vendor="aws",
    min_replica=1,
    max_replica=3,
    type="protected",
)

# 等待部署完成
endpoint.wait()

print(f"Endpoint URL: {endpoint.url}")
```

#### 使用 Endpoint

```python
from huggingface_hub import InferenceClient

client = InferenceClient(endpoint.url, token="your_token")

# 生成文字
response = client.text_generation(
    "What is machine learning?",
    max_new_tokens=100,
    temperature=0.7,
)

print(response)
```

**優勢**：
- 全託管（無需維護）
- 自動擴展
- 按使用計費
- 簡單易用

**成本**：
- 約 $0.60-$1.50/小時（取決於實例類型）

### 2.3 vLLM 加速

**vLLM** 是高性能 LLM 推理引擎，使用 PagedAttention 技術大幅提升吞吐量。

#### 核心特性

1. **PagedAttention**：
   - 將 KV cache 分頁儲存
   - 減少記憶體碎片
   - 提升批次處理效率

2. **Continuous Batching**：
   - 動態批次處理
   - 最大化 GPU 利用率

3. **高效記憶體管理**：
   - 減少記憶體浪費
   - 支援更大批次

#### 安裝

```bash
pip install vllm
```

#### 基本使用

```python
from vllm import LLM, SamplingParams

# 載入模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # GPU 數量
    dtype="float16",
)

# 設定採樣參數
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 批次推理
prompts = [
    "What is AI?",
    "Explain quantum computing.",
    "How does a neural network work?"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
```

#### OpenAI 相容 API 服務器

```bash
# 啟動 API 服務器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

**客戶端調用**：

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"  # vLLM 不需要 API key

# 使用 OpenAI API 格式
completion = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "What is deep learning?"}
    ],
    temperature=0.7,
    max_tokens=200,
)

print(completion.choices[0].message.content)
```

**性能優勢**：
- 比 HuggingFace 快 5-15x
- 吞吐量提升 10-20x
- 記憶體效率提升 2-3x

### 2.4 TensorRT-LLM

**TensorRT-LLM** 是 NVIDIA 的 LLM 推理優化框架，提供極致性能。

#### 核心優化

1. **Kernel Fusion**：融合運算核心
2. **INT4/INT8 量化**：極限壓縮
3. **In-flight Batching**：動態批次
4. **Multi-GPU Tensor並行**：多卡加速

#### 安裝

```bash
# 需要 NVIDIA GPU 和 CUDA
pip install tensorrt_llm
```

#### 模型轉換

```bash
# 將 HuggingFace 模型轉換為 TensorRT 引擎
python convert_checkpoint.py \
    --model_dir ./llama-2-7b-hf \
    --output_dir ./llama-2-7b-trt \
    --dtype float16

# 建構 TensorRT 引擎
trtllm-build \
    --checkpoint_dir ./llama-2-7b-trt \
    --output_dir ./llama-2-7b-engine \
    --gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 512
```

#### 推理

```python
from tensorrt_llm import LLM

# 載入 TensorRT 引擎
llm = LLM(model_dir="./llama-2-7b-engine")

# 推理
prompts = ["What is AI?"]
outputs = llm.generate(prompts, max_new_tokens=100)

print(outputs[0]["generation"])
```

**性能**：
- 比 PyTorch 快 3-8x
- 極低延遲（< 10ms TTFT）
- 最高吞吐量

**劣勢**：
- 設置複雜
- 僅支援 NVIDIA GPU
- 模型轉換耗時

---

## 3. 系統架構與維護

### 3.1 負載均衡

#### 架構設計

```
                        ┌─────────────┐
                        │  Load       │
        Client ────────>│  Balancer   │
                        │  (Nginx)    │
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
          ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
          │  Instance │  │  Instance │  │  Instance │
          │     1     │  │     2     │  │     3     │
          │  (GPU 1)  │  │  (GPU 2)  │  │  (GPU 3)  │
          └───────────┘  └───────────┘  └───────────┘
```

#### Nginx 配置

```nginx
upstream llm_backend {
    least_conn;  # 最少連接負載均衡

    server 192.168.1.101:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.102:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.103:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;

    # 超時設定（LLM 推理可能較慢）
    proxy_connect_timeout 60s;
    proxy_send_timeout 120s;
    proxy_read_timeout 120s;

    location /v1/ {
        proxy_pass http://llm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 限流（每秒 10 請求）
        limit_req zone=api_limit burst=20 nodelay;
    }
}

# 限流配置
http {
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
}
```

#### 使用 Ray Serve 進行分散式部署

```python
from ray import serve
import ray

# 初始化 Ray
ray.init()

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=3,  # 3 個副本
    max_concurrent_queries=10,
)
class LLMDeployment:
    def __init__(self):
        from vllm import LLM
        self.model = LLM("meta-llama/Llama-2-7b-chat-hf")

    async def __call__(self, request):
        prompt = await request.json()["prompt"]
        outputs = self.model.generate([prompt])
        return {"text": outputs[0].outputs[0].text}

# 部署
serve.run(LLMDeployment.bind(), route_prefix="/generate")
```

### 3.2 模型版本管理

#### 藍綠部署

```python
"""
藍綠部署：維護兩個版本，無縫切換

藍色環境：當前生產版本
綠色環境：新版本（測試）
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

# 模型版本配置
ACTIVE_VERSION = "blue"  # 或 "green"

models = {
    "blue": load_model("llama-7b-v1"),   # 當前版本
    "green": load_model("llama-7b-v2"),  # 新版本
}

@app.route("/generate", methods=["POST"])
def generate():
    version = ACTIVE_VERSION
    model = models[version]

    prompt = request.json["prompt"]
    output = model.generate(prompt)

    return jsonify({
        "text": output,
        "version": version
    })

@app.route("/switch_version", methods=["POST"])
def switch_version():
    """切換版本"""
    global ACTIVE_VERSION
    ACTIVE_VERSION = "green" if ACTIVE_VERSION == "blue" else "blue"
    return jsonify({"active_version": ACTIVE_VERSION})
```

#### Canary 部署

```python
"""
金絲雀部署：逐步導流到新版本

例如：95% 流量到舊版本，5% 到新版本
"""

import random

CANARY_PERCENTAGE = 5  # 5% 流量到新版本

@app.route("/generate", methods=["POST"])
def generate():
    # 根據百分比選擇版本
    if random.randint(1, 100) <= CANARY_PERCENTAGE:
        version = "green"  # 新版本
    else:
        version = "blue"   # 舊版本

    model = models[version]
    prompt = request.json["prompt"]
    output = model.generate(prompt)

    return jsonify({
        "text": output,
        "version": version
    })
```

#### 使用 MLflow 進行模型管理

```python
import mlflow
import mlflow.pytorch

# 記錄模型
with mlflow.start_run():
    # 訓練/微調模型
    model = train_model()

    # 記錄參數
    mlflow.log_params({
        "learning_rate": 2e-5,
        "epochs": 3,
        "model_name": "llama-7b-chat"
    })

    # 記錄指標
    mlflow.log_metrics({
        "eval_loss": 0.45,
        "perplexity": 12.3
    })

    # 保存模型
    mlflow.pytorch.log_model(model, "model")

# 載入模型
model_uri = "runs:/<run_id>/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
```

### 3.3 監控與記錄

#### 關鍵指標

**系統指標**：
- GPU 使用率
- GPU 記憶體使用
- CPU 使用率
- 網路流量

**業務指標**：
- 請求數量（QPS）
- 延遲（P50, P95, P99）
- 錯誤率
- 吞吐量（tokens/s）

#### Prometheus + Grafana 監控

**安裝 Prometheus 客戶端**：
```bash
pip install prometheus-client
```

**暴露指標**：
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定義指標
request_count = Counter('llm_requests_total', 'Total requests')
request_duration = Histogram('llm_request_duration_seconds', 'Request duration')
gpu_memory = Gauge('llm_gpu_memory_used_bytes', 'GPU memory used')
tokens_generated = Counter('llm_tokens_generated_total', 'Total tokens generated')

def generate_text(prompt):
    request_count.inc()  # 請求計數 +1

    start_time = time.time()

    # 推理
    output = model.generate(prompt)

    # 記錄延遲
    duration = time.time() - start_time
    request_duration.observe(duration)

    # 記錄生成的 token 數
    tokens_generated.inc(len(output.split()))

    # 記錄 GPU 記憶體
    import torch
    gpu_memory.set(torch.cuda.memory_allocated())

    return output

# 啟動 Prometheus 服務器（端口 8001）
start_http_server(8001)
```

**Prometheus 配置** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm_service'
    static_configs:
      - targets: ['localhost:8001']
```

#### 結構化日誌

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """結構化 JSON 日誌"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # 添加自定義字段
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "latency"):
            log_data["latency"] = record.latency

        return json.dumps(log_data)

# 設置日誌
logger = logging.getLogger("llm_service")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 使用
def generate_with_logging(prompt, user_id):
    start_time = time.time()

    try:
        output = model.generate(prompt)
        latency = time.time() - start_time

        # 記錄成功
        logger.info("Generation successful",
                    extra={"user_id": user_id, "latency": latency})

        return output

    except Exception as e:
        logger.error(f"Generation failed: {e}",
                     extra={"user_id": user_id})
        raise
```

### 3.4 故障恢復

#### 健康檢查

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """健康檢查端點"""
    try:
        # 檢查模型是否可用
        test_output = model.generate("test")

        # 檢查 GPU
        import torch
        if not torch.cuda.is_available():
            return jsonify({"status": "unhealthy", "reason": "GPU not available"}), 503

        return jsonify({"status": "healthy"}), 200

    except Exception as e:
        return jsonify({"status": "unhealthy", "reason": str(e)}), 503

@app.route("/ready", methods=["GET"])
def readiness_check():
    """就緒檢查"""
    if model_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "not ready"}), 503
```

#### 自動重啟機制

```python
import sys
import time
from functools import wraps

def auto_restart_on_oom(max_retries=3):
    """OOM 時自動重啟"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        retries += 1
                        logger.warning(f"OOM detected, retry {retries}/{max_retries}")

                        # 清理 GPU 記憶體
                        import torch
                        torch.cuda.empty_cache()

                        # 重新載入模型
                        reload_model()

                        time.sleep(5)
                    else:
                        raise

            logger.error("Max retries reached, restarting service")
            sys.exit(1)  # 讓容器管理器重啟

        return wrapper
    return decorator

@auto_restart_on_oom(max_retries=3)
def generate(prompt):
    return model.generate(prompt)
```

#### Docker 健康檢查

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# ... 安裝依賴和模型 ...

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "serve.py"]
```

---

## 4. 資料安全與隱私

### 4.1 資料加密

#### 傳輸加密 (TLS/HTTPS)

```python
from flask import Flask

app = Flask(__name__)

if __name__ == "__main__":
    # 使用 HTTPS
    app.run(
        host="0.0.0.0",
        port=443,
        ssl_context=("cert.pem", "key.pem")
    )
```

#### 資料遮罩

```python
import re

def mask_sensitive_data(text):
    """遮罩敏感資料"""
    # 遮罩電子郵件
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '***@***.***', text)

    # 遮罩電話號碼
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '***-***-****', text)

    # 遮罩信用卡號
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                  '****-****-****-****', text)

    return text

# 使用
prompt = "My email is john@example.com and phone is 123-456-7890"
masked = mask_sensitive_data(prompt)
logger.info(f"Prompt: {masked}")  # 不記錄敏感資料
```

### 4.2 訪問控制

#### API Key 驗證

```python
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

VALID_API_KEYS = {
    "key_user1": {"user_id": "user1", "tier": "premium"},
    "key_user2": {"user_id": "user2", "tier": "free"},
}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")

        if not api_key or api_key not in VALID_API_KEYS:
            return jsonify({"error": "Invalid API key"}), 401

        # 將使用者資訊添加到請求
        request.user_info = VALID_API_KEYS[api_key]

        return f(*args, **kwargs)

    return decorated_function

@app.route("/generate", methods=["POST"])
@require_api_key
def generate():
    user_info = request.user_info
    prompt = request.json["prompt"]

    # 根據層級限制
    if user_info["tier"] == "free" and len(prompt) > 500:
        return jsonify({"error": "Prompt too long for free tier"}), 403

    output = model.generate(prompt)
    return jsonify({"text": output})
```

#### 速率限制

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route("/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate():
    # ...
    pass
```

### 4.3 模型安全

#### 輸入過濾

```python
def is_safe_prompt(prompt):
    """檢查提示是否安全"""
    # 黑名單關鍵字
    blacklist = [
        "jailbreak", "ignore previous instructions",
        "DAN mode", "evil mode"
    ]

    prompt_lower = prompt.lower()
    for keyword in blacklist:
        if keyword in prompt_lower:
            return False

    # 長度限制
    if len(prompt) > 4000:
        return False

    return True

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]

    if not is_safe_prompt(prompt):
        return jsonify({"error": "Unsafe prompt detected"}), 400

    output = model.generate(prompt)
    return jsonify({"text": output})
```

#### 輸出過濾

```python
from transformers import pipeline

# 載入內容審核模型
moderator = pipeline("text-classification",
                     model="OpenAssistant/reward-model-deberta-v3-large-v2")

def is_safe_output(text):
    """檢查輸出是否安全"""
    result = moderator(text)

    # 如果包含不當內容
    if result[0]["label"] == "UNSAFE":
        return False

    return True

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    output = model.generate(prompt)

    # 檢查輸出
    if not is_safe_output(output):
        logger.warning(f"Unsafe output detected for prompt: {prompt}")
        return jsonify({"error": "Generated content violates policy"}), 400

    return jsonify({"text": output})
```

---

## 5. 成本優化

### 5.1 成本分析

#### GPU 成本比較

| 雲端提供商 | GPU 類型 | 每小時成本 | 每月成本 (730h) |
|-----------|---------|-----------|----------------|
| AWS (p4d) | A100 (40GB) | $4.00 | $2,920 |
| AWS (g5)  | A10G (24GB) | $1.00 | $730 |
| GCP | A100 (40GB) | $3.67 | $2,679 |
| Lambda Labs | A100 (40GB) | $1.10 | $803 |
| RunPod | A100 (80GB) | $1.89 | $1,380 |

#### 成本估算工具

```python
def estimate_monthly_cost(
    requests_per_day,
    avg_tokens_per_request,
    tokens_per_second,
    gpu_hourly_cost,
):
    """
    估算每月成本

    Args:
        requests_per_day: 每天請求數
        avg_tokens_per_request: 平均每請求 token 數
        tokens_per_second: GPU 生成速度 (tokens/s)
        gpu_hourly_cost: GPU 每小時成本
    """
    # 每天總 tokens
    daily_tokens = requests_per_day * avg_tokens_per_request

    # 每天所需秒數
    daily_seconds = daily_tokens / tokens_per_second

    # 每天所需小時數
    daily_hours = daily_seconds / 3600

    # 每月成本（假設 30 天）
    monthly_cost = daily_hours * 30 * gpu_hourly_cost

    return {
        "monthly_cost": monthly_cost,
        "daily_hours": daily_hours,
        "utilization": (daily_hours / 24) * 100
    }

# 範例
result = estimate_monthly_cost(
    requests_per_day=10000,
    avg_tokens_per_request=200,
    tokens_per_second=50,  # A100 on LLaMA-7B
    gpu_hourly_cost=1.10
)

print(f"每月成本: ${result['monthly_cost']:.2f}")
print(f"每天需要: {result['daily_hours']:.2f} 小時")
print(f"GPU 使用率: {result['utilization']:.1f}%")
```

### 5.2 優化策略

#### 1. 批次處理

```python
from collections import deque
import asyncio

class BatchProcessor:
    """批次處理器，累積請求後批次推理"""

    def __init__(self, model, batch_size=8, wait_time=0.1):
        self.model = model
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.queue = deque()
        self.processing = False

    async def add_request(self, prompt):
        """添加請求到佇列"""
        future = asyncio.Future()
        self.queue.append((prompt, future))

        # 觸發批次處理
        if not self.processing:
            asyncio.create_task(self.process_batch())

        return await future

    async def process_batch(self):
        """批次處理"""
        self.processing = True
        await asyncio.sleep(self.wait_time)  # 等待累積請求

        if not self.queue:
            self.processing = False
            return

        # 取出一批請求
        batch = []
        futures = []
        for _ in range(min(self.batch_size, len(self.queue))):
            prompt, future = self.queue.popleft()
            batch.append(prompt)
            futures.append(future)

        # 批次推理
        outputs = self.model.generate(batch)

        # 返回結果
        for future, output in zip(futures, outputs):
            future.set_result(output)

        self.processing = False

        # 如果還有請求，繼續處理
        if self.queue:
            asyncio.create_task(self.process_batch())

# 使用
processor = BatchProcessor(model, batch_size=8)

@app.route("/generate", methods=["POST"])
async def generate():
    prompt = request.json["prompt"]
    output = await processor.add_request(prompt)
    return jsonify({"text": output})
```

#### 2. 模型緩存

```python
from functools import lru_cache

class CachedLLM:
    """帶緩存的 LLM"""

    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def generate(self, prompt, **kwargs):
        # 創建緩存鍵
        cache_key = (prompt, frozenset(kwargs.items()))

        if cache_key in self.cache:
            self.hits += 1
            logger.info(f"Cache hit! Hit rate: {self.hit_rate():.2%}")
            return self.cache[cache_key]

        # 緩存未命中，實際推理
        self.misses += 1
        output = self.model.generate(prompt, **kwargs)

        # 保存到緩存
        self.cache[cache_key] = output

        return output

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
```

#### 3. Spot/Preemptible 實例

```python
"""
使用 Spot 實例可節省 60-90% 成本，但可能被中斷

策略：
1. 使用 Spot 實例作為主力
2. 保留少量 On-Demand 實例作為備份
3. 實作自動故障轉移
"""

# AWS Spot 請求
import boto3

ec2 = boto3.client('ec2')

response = ec2.request_spot_instances(
    InstanceCount=1,
    Type='persistent',
    LaunchSpecification={
        'ImageId': 'ami-xxxxxxxxx',
        'InstanceType': 'g4dn.xlarge',
        'KeyName': 'my-key',
    }
)
```

---

## 6. 實作範例

### 6.1 完整 FastAPI 部署

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
import uvicorn
from vllm import LLM, SamplingParams
import time

# ============================================
# 模型定義
# ============================================

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    text: str
    tokens: int
    latency: float
    model_version: str

# ============================================
# 初始化
# ============================================

app = FastAPI(title="LLM API", version="1.0.0")

# 載入模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    dtype="float16"
)

MODEL_VERSION = "llama-2-7b-v1.0"

# API Key 驗證
VALID_API_KEYS = {"test_key_123": "user1"}

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return VALID_API_KEYS[x_api_key]

# ============================================
# 端點
# ============================================

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {"status": "healthy", "model": MODEL_VERSION}

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    user_id: str = Depends(verify_api_key)
):
    """生成文字"""
    try:
        start_time = time.time()

        # 設定採樣參數
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )

        # 生成
        outputs = llm.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        latency = time.time() - start_time

        return GenerateResponse(
            text=generated_text,
            tokens=len(generated_text.split()),
            latency=latency,
            model_version=MODEL_VERSION
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 啟動
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # vLLM 使用單 worker
    )
```

**Docker 部署**：

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安裝 Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製代碼
COPY serve.py /app/serve.py
WORKDIR /app

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1

# 啟動
CMD ["python3", "serve.py"]
```

**docker-compose.yml**：

```yaml
version: '3.8'

services:
  llm-api:
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
    restart: unless-stopped
```

### 6.2 Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm
        image: your-registry/llm-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1  # 1 GPU
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-chat-hf"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 7. 參考資源

### 框架與工具

- **vLLM**: https://github.com/vllm-project/vllm
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
- **Text Generation Inference**: https://github.com/huggingface/text-generation-inference
- **Ray Serve**: https://docs.ray.io/en/latest/serve/index.html
- **MLflow**: https://mlflow.org/
- **Prometheus**: https://prometheus.io/
- **Grafana**: https://grafana.com/

### 文檔

- **Hugging Face Deployment Guide**: https://huggingface.co/docs/inference-endpoints/index
- **vLLM Documentation**: https://vllm.readthedocs.io/
- **NVIDIA TensorRT**: https://developer.nvidia.com/tensorrt
- **Kubernetes**: https://kubernetes.io/docs/

### 論文

- **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM, 2023)
- **Fast Transformer Decoding: One Write-Head is All You Need** (MQA, 2019)

---

## 總結

### 核心要點

1. **選擇合適的部署環境**
   - 高性能：GPU 雲端
   - 成本敏感：CPU 或 Spot 實例
   - 隱私需求：本地部署

2. **使用優化推理框架**
   - vLLM：高吞吐量
   - TensorRT-LLM：極致性能
   - llama.cpp：CPU/邊緣

3. **實施完善的監控**
   - 系統指標：GPU、記憶體
   - 業務指標：QPS、延遲
   - 日誌：結構化、可搜索

4. **確保安全性**
   - 資料加密（HTTPS）
   - 訪問控制（API Key）
   - 內容審核（輸入/輸出過濾）

5. **優化成本**
   - 批次處理
   - 緩存
   - Spot 實例
   - 適當的量化

### 最佳實踐

1. **從小規模開始**：先單機部署，驗證後再擴展
2. **漸進式發布**：使用藍綠或金絲雀部署
3. **監控為先**：部署前建立完善監控
4. **自動化一切**：CI/CD、自動擴展、自動恢復
5. **定期備份**：模型、配置、資料
