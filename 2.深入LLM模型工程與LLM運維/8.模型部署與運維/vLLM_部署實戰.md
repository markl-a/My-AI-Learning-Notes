# vLLM 部署實戰

> 對應 [全景圖 #14](../../2024-2026_AI完整領域全景圖.md);相關:[`../../3.LLM應用工程/6.推論優化/`](../../3.LLM應用工程/6.推論優化/);[`../../11.AI_Hardware_Compute/`](../../11.AI_Hardware_Compute/)
>
> ⚡ **想立刻動手?**配套 Colab notebook:[`notebooks/Colab_vLLM_Deploy_PrefixCache_Demo.ipynb`](./notebooks/Colab_vLLM_Deploy_PrefixCache_Demo.ipynb)
> — 在 Colab T4 GPU 跑通 vLLM server、實測 prefix cache 加速、continuous batching throughput,~15 分鐘跑完(含 vLLM 啟動 2-4 min)。
> 本檔是「概念深度」,notebook 是「親手跑一遍」。

---

## 1. vLLM 是什麼:為什麼不要再用 `transformers.generate()` 跑生產

vLLM 是 UC Berkeley Sky Computing Lab 開源的高吞吐 LLM 推論引擎,2023 年首發,到 2026 年已經是事實上的開源服務端標準。和 HuggingFace `transformers` 內建的 `model.generate()` 比,差距不是「快一點」,而是「能不能拿來做服務」的差別。

| 維度 | HF transformers | vLLM |
|---|---|---|
| KV cache 管理 | 連續 tensor,需預留 max_len | **PagedAttention**,按 block 分頁,幾乎零碎片 |
| Batching | 靜態 batch,要等齊整批 | **Continuous batching**,token-level 動態插入請求 |
| 共享前綴 | 每次重算 | **Automatic prefix caching (APC)**,KV block reuse |
| 多卡 | 要自己包 `accelerate` / `deepspeed` | 內建 tensor parallel / pipeline parallel |
| OpenAI API | 沒有 | 內建 `/v1/chat/completions` 相容端點 |
| Throughput (Llama-3.1-8B, H100) | ~600 tok/s | **5000-12000 tok/s**(視 batch) |

三個核心招式:

- **PagedAttention**:把 KV cache 切成固定大小的 block(預設 16 token),用 OS 虛擬記憶體那套來管,參考第 7 節。
- **Continuous batching**:每個 decode step 都可以加入新請求、踢掉完成的,不像傳統 batch 要等最慢那個結束。
- **Prefix caching**:相同 system prompt 的 KV block hash 過,直接 reuse,RAG 與 agent 場景常常省 30-70% prefill。

---

## 2. 環境準備:版本是最大的雷

vLLM 對 CUDA / torch / transformers 版本綁很死,跨小版本壞掉是常態。2026 年 5 月的推薦組合:

```bash
# 系統需求:Linux x86_64,CUDA 12.4+,Python 3.10-3.12
nvidia-smi   # 確認 driver >= 550 (對應 CUDA 12.4)
nvcc --version

# 推薦用獨立 venv 或 conda env,千萬不要混到別的專案
python -m venv ~/.venv/vllm
source ~/.venv/vllm/bin/activate

# vLLM 會自動拉 torch 對應版本,不要自己先裝 torch
pip install -U pip setuptools wheel
pip install vllm==0.9.1            # 或 latest stable
pip install "transformers>=4.45"   # tokenizer / chat template 用

# 驗證
python -c "import vllm; print(vllm.__version__)"
vllm --help
```

H100 / H200 / B100 走預設;A100 / L40S 也支援;消費卡(4090 / 5090)能跑但 NVLink 缺席,多卡效率打折。AMD ROCm、AWS Trainium、Intel Gaudi、Google TPU 都有 backend,但本文聚焦 NVIDIA。

`HF_TOKEN` 記得 export,不然 Llama / Gemma 這些 gated model 拉不下來:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
export HF_HOME=/data/hf_cache   # 不要塞到 home,模型動輒 70GB
```

---

## 3. 單 GPU 起服務:30 秒上線

最常見的場景,Llama-3.1-8B 跑一張 H100 / A100-80G / L40S:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --enable-prefix-caching
```

OpenAI API 相容端點直接打:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "用一句話解釋 PagedAttention"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

Python 端用 `openai` SDK 直接接(把 `base_url` 換掉即可,程式碼幾乎不動):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",   # vLLM 預設不驗證,正式環境用 --api-key 開啟
)

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一個資深 SRE,回答精簡。"},
        {"role": "user",   "content": "Kubernetes liveness 跟 readiness 差在哪?"},
    ],
    stream=True,
)
for chunk in resp:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

要 batch 跑離線任務不開 server,用 `LLM` 物件就好:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          gpu_memory_utilization=0.9,
          enable_prefix_caching=True)
params = SamplingParams(temperature=0.7, max_tokens=256)
prompts = ["寫一首關於 GPU 的俳句", "解釋什麼是 continuous batching"]
for out in llm.generate(prompts, params):
    print(out.outputs[0].text)
```

---

## 4. 多 GPU tensor parallelism:70B 跑 4 張卡

70B fp16 大約要 140GB,單張 H100-80G 塞不下。靠 `--tensor-parallel-size`(別名 `-tp`)做 column / row 切分:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  -tp 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --port 8000
```

注意點:

- **`tp` 數要能整除 attention head 數**:Llama-3 有 64 head,合法值 1/2/4/8/16/32/64。
- **NVLink 才香**:tp=4 跨 PCIe 會被 all-reduce 拖死,實測 H100 SXM(NVLink)比 PCIe 版快 1.8-2.5×。
- **跨機器**:加上 `--pipeline-parallel-size`(PP)做 stage 切,通常 tp 撐滿單機,PP 再跨機。Ray cluster 啟動:

```bash
# Head node
ray start --head --port=6379

# Worker node
ray start --address='<head_ip>:6379'

# Serve(tp=8 × pp=2 = 16 GPU,跨兩台 8×H100)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  -tp 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 32768
```

---

## 5. 量化模型部署:GGUF / AWQ / GPTQ / FP8 怎麼選

vLLM 對量化的支援不是一視同仁:

| 格式 | vLLM 支援度 | 用途 | 速度 | 精度損失 |
|---|---|---|---|---|
| **AWQ (W4A16)** | 一級公民,優化最深 | 通用 4-bit | 快,context 越長越突出 | 小 |
| **GPTQ (W4A16)** | 支援,稍慢於 AWQ | 通用 4-bit,舊生態 | 中 | 小 |
| **FP8 (W8A8)** | H100/B100 原生,推薦 | 高 batch 場景 | 最快 | 極小 |
| **GGUF** | 實驗性,效能差 | 不推薦,留給 llama.cpp | 慢 | - |
| **BitsAndBytes (NF4)** | 支援但僅 offline | 開發測試 | 慢 | 中 |

實戰範例,跑 Llama-3.1-70B AWQ(單張 H100-80G 就夠):

```bash
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --port 8000
```

FP8 + H100 是 2026 年生產的甜蜜點:

```bash
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --quantization fp8 -tp 2 --max-model-len 16384
```

GGUF 真的要跑,加 `--quantization gguf`,但建議直接用 llama.cpp / ollama,別折磨自己。

---

## 6. Prefix caching 詳解:相同 system prompt 別再算第二次

`--enable-prefix-caching` 一開,vLLM 會把每個 KV block(16 token)做 hash,後續請求若前綴 hash 命中,直接 reuse 那段 KV,跳過 prefill。

效果驚人的場景:

- **長 system prompt 的 chatbot**:5000-token 的角色設定 + few-shot,每次只算新使用者輸入。
- **RAG**:檢索結果排在前面、問題在後面時,命中率不高;改成「問題在前、context 在後」會打散,得依場景權衡。建議把固定模板 / instruction 集中在最前段。
- **Agent loop**:同一個 trajectory 反覆呼叫,前面所有 turn 都能 cache。

打開 metrics 可以看 hit rate:

```python
# 從 /metrics 端點抓
import requests, re
text = requests.get("http://localhost:8000/metrics").text
for line in text.splitlines():
    if "prefix_cache" in line or "gpu_cache_usage" in line:
        print(line)
# 看 vllm:gpu_prefix_cache_hits / vllm:gpu_prefix_cache_queries 算命中率
```

實測:RAG 服務同一個 system prompt(2k tokens),命中率穩定 95%+,TTFT 從 380ms 降到 90ms。

---

## 7. PagedAttention 原理:把 KV cache 當作虛擬記憶體

傳統做法:每條序列預留一塊連續 KV tensor,大小 = `max_seq_len × num_layers × num_heads × head_dim × 2`。問題:

- 短序列浪費(80% 都填不滿)
- 長序列被 max_len 截斷
- batch 中各序列長度不一,要 padding,記憶體碎片爆炸

PagedAttention 借用 OS 分頁的概念:

- **Physical block**:GPU 上實際的 KV tensor,固定 16 token 一塊。
- **Logical block**:每個請求看到的「連續」KV,實際透過 block table 對應到 physical block。
- **Block table**:`request_id -> [phys_block_0, phys_block_1, ...]`,每生成 16 token 就配一塊新的。
- **Copy-on-write**:beam search / parallel sampling 共享 prefix block,需要分歧時才複製。

結果:碎片率從 60-80% 降到 < 4%,同樣的 GPU 記憶體能裝 2-4× 的 concurrent request。

```python
# 看 block table(debug 用)
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", block_size=16)
# vllm 內部 scheduler 會 print 出 block usage
# 也可以從 /metrics 看 vllm:num_gpu_blocks_used
```

---

## 8. EAGLE-3 speculative decoding:2.5-6× 加速

Speculative decoding 的核心思路:用一個小的「draft model」一次猜 k 個 token,再讓 target model 一次 forward 驗證 k 個。命中率高的話,單次 forward 出多 token,等效加速 ≈ 接受率 × k。

EAGLE-3 是 2025 年的 SOTA draft 架構,2026 vLLM 0.9.x 正式整合,Llama-3.3-70B 實測 2.5-3× 加速,小 batch 場景可以衝到 5-6×。

啟動方式(70B + EAGLE-3 draft,4 張 H100):

```bash
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --seed 42 \
  -tp 4 \
  --max-model-len 16384 \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 3,
    "method": "eagle3",
    "draft_tensor_parallel_size": 1
  }'
```

`num_speculative_tokens`(k)設 3-5 最佳,太大反而拖慢。觀察接受率:

```bash
curl -s http://localhost:8000/metrics | grep spec_decode
# vllm:spec_decode_num_accepted_tokens_total
# vllm:spec_decode_num_draft_tokens_total
# 接受率 = accepted / draft,健康值 0.6-0.85
```

什麼時候 *不要* 開:大 batch(>32)。spec decode 在低 batch / latency-bound 場景受益最大,大 batch 已經 throughput-bound,spec 的 overhead 反而扣分。

---

## 9. Production 監控:Prometheus + Grafana

vLLM 預設在 `/metrics` 吐 Prometheus 格式。重點 metric:

| Metric | 看什麼 |
|---|---|
| `vllm:num_requests_running` | 當下並行請求數 |
| `vllm:num_requests_waiting` | queue 深度,持續 > 0 表示要擴容 |
| `vllm:gpu_cache_usage_perc` | KV cache 用量,> 90% 接近 OOM |
| `vllm:time_to_first_token_seconds` | TTFT,SLO 核心指標 |
| `vllm:time_per_output_token_seconds` | TPOT,串流順暢度 |
| `vllm:e2e_request_latency_seconds` | 端到端延遲 |
| `vllm:gpu_prefix_cache_hit_rate` | prefix cache 命中 |
| `vllm:spec_decode_acceptance_rate` | spec decode 接受率 |

Prometheus scrape 設定:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    scrape_interval: 5s
    static_configs:
      - targets: ['vllm-host:8000']
```

Grafana dashboard 直接 import 社群版 ID `19711`(vLLM Inference Dashboard),自己關心的話另畫一張 SLO 板:p50/p95/p99 TTFT + TPOT + queue depth + KV usage,告警設 p95 TTFT > 1s 或 queue depth > 50。

Docker Compose 一鍵起完整 stack:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.9.1
    runtime: nvidia
    environment:
      - HF_TOKEN=${HF_TOKEN}
    command: >
      --model meta-llama/Llama-3.1-8B-Instruct
      --enable-prefix-caching
      --gpu-memory-utilization 0.9
    ports: ["8000:8000"]
  prometheus:
    image: prom/prometheus
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
```

---

## 10. 三個真實 case study

### 10.1 RAG 服務(2k QPS 客服知識庫)

- 模型:Llama-3.1-8B-Instruct AWQ,單張 L40S
- system prompt 2.5k tokens(產品說明 + 回答規範),user query ~50 tokens,retrieved context ~3k tokens
- **關鍵**:把 system prompt 排在最前,retrieved context 放中間,user query 最後 → prefix cache 命中 92%
- 開 `--enable-chunked-prefill --max-num-batched-tokens 8192`,長 context 不會卡 decode
- 結果:p95 TTFT 180ms,單卡 ~1200 tok/s output throughput

### 10.2 客服 chatbot(多輪對話)

- 模型:Llama-3.3-70B FP8,2× H100
- 多輪歷史 + 工具呼叫定義(tools schema 約 1.8k tokens)
- 開 EAGLE-3 spec decode(k=4),小 batch latency 從 32 tok/s/req 拉到 95 tok/s/req
- session affinity:相同 user 路由到同一個 replica,prefix cache 命中提升到 80%+
- 配合 `--max-num-seqs 64` 限制單卡並發,避免長對話互相搶 KV

### 10.3 Code agent backend(Cursor-like)

- 模型:Qwen2.5-Coder-32B-Instruct AWQ,單張 H100
- 大量 file context(平均 10k+ tokens),輸出短(~300 tokens)
- **prefill bound**,開 `--enable-chunked-prefill` 拆 prefill 避免阻塞其他請求
- `--max-model-len 65536` 留長 context 空間,KV 預留要算清楚
- structured output:用 `guided_json` 強制輸出 JSON schema,避免 retry

```python
# Code agent 端 structured output
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    messages=[...],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "edits": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["file", "edits"]
        }
    }
)
```

---

## 11. 常見坑

1. **CUDA OOM**:`--gpu-memory-utilization` 預設 0.9,過高會搶到 driver / cuBLAS workspace 爆掉。同時 `--max-model-len` 拉太高吃滿 KV。先算:
   ```
   KV per token = 2 × num_layers × num_heads × head_dim × dtype_bytes
   # Llama-3.1-8B fp16: 2 × 32 × 8 × 128 × 2 ≈ 131KB / token
   # max_seq_len=8192 × concurrent=128 ⇒ ~134GB KV(這就 OOM 了)
   ```
   先降 `--max-num-seqs` 或 `--max-model-len`。

2. **Long prompt 卡住整個服務**:沒開 chunked prefill 時,一個 32k prompt 會把 decode 停掉幾秒。**永遠開 `--enable-chunked-prefill`**,2026 版本已預設 on,確認一下。

3. **Tokenizer 不匹配**:模型 repo 有自定 chat template 但 vLLM 抓錯,結果亂回。用 `--chat-template /path/to/template.jinja` 顯式指定,或在 request 帶 `chat_template_kwargs`。

4. **量化模型載不進去**:版本不對。AWQ checkpoint 要對 vLLM 版本,新版 vLLM 棄用舊 quant_method,認 repo 內 `config.json` 的 `quantization_config.quant_method`。

5. **多卡啟動慢 / hang**:NCCL 環境變數沒設,跨 NUMA 通訊出問題。加 `NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1`(無 IB)或反之。

6. **Streaming 中斷**:client 端 timeout 設太短;nginx 前面要關 buffering(`proxy_buffering off`)。

7. **第一個請求超慢**:CUDA graph capture + torch.compile warmup,屬正常。生產建議啟動後跑幾個 warmup request 再上線。

---

## 12. vs SGLang / TensorRT-LLM:什麼時候不該用 vLLM

| 引擎 | 強項 | 弱點 | 選它的時機 |
|---|---|---|---|
| **vLLM** | 生態最大,模型支援廣,熱啟動快(~60s),硬體覆蓋多(NV/AMD/TPU/Trainium/Gaudi) | 純極限 throughput 略輸 | 需要快速迭代、多模型、跨硬體、社群第一 |
| **SGLang** | RadixAttention 對共享前綴特強,H100 上比 vLLM 高 ~29% throughput;結構化生成快 | 模型新增稍慢,生態較小 | RAG / agent / 大量重複 prefix 場景;xAI、Cursor、LinkedIn 都用它 |
| **TensorRT-LLM** | NVIDIA 原生極限優化,throughput / latency 領先 10-30% | 編譯 28 分鐘起跳,只支援 NV,模型改動就要重 build | 單一穩定模型、長期 production、極致 cost-per-token |

實務建議:**MVP 用 vLLM,prefix-heavy 工作負載評估 SGLang,當你能精算到「換 TensorRT-LLM 一年省 X 萬 GPU 小時」再去吃編譯成本**。三個引擎並非互斥,大公司常見組合是 vLLM 跑長尾 + TRT-LLM 跑主力流量。

---

### 延伸閱讀

- 官方文件:https://docs.vllm.ai/
- PagedAttention 論文:Kwon et al., SOSP 2023
- EAGLE-3:https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/
- Production guide(2026):https://www.sitepoint.com/vllm-production-deployment-guide-2026/
- 對比 benchmark:Spheron / LeetLLM 2026 H100 benchmark
