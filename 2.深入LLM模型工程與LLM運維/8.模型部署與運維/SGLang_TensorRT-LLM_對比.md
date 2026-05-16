> 對應 [全景圖 #14];搭配閱讀:[`./vLLM_部署實戰.md`](./vLLM_部署實戰.md)

# SGLang vs TensorRT-LLM vs vLLM:三大推理引擎深度對比

## 1. 三大推理引擎定位

2024–2026 年開源 LLM serving 棧已收斂為三強鼎立,各自佔據不同生態位:

- **vLLM**:UC Berkeley 起家、PyTorch 原生的「通用最大公約數」。PagedAttention 是它的招牌特色,model coverage 最廣(支援幾乎所有 HF 上的開源權重)、文件最齊、上手最快、無需任何編譯步驟。是大多數團隊的首選起點,適合多模型快速切換、研究與 PoC、以及不確定流量模式的中小規模生產。
- **SGLang**:LMSYS / xAI / DeepSeek 等實戰團隊主導,主打「**結構化生成 + RadixAttention 共用前綴**」。它是 DeepSeek-V3、Llama 4、Qwen3 等 MoE 模型在開源側的事實標準執行引擎,在 Agent / RAG / 多輪對話這類「前綴大量重複」的場景上有壓倒性優勢,並且 Frontend 提供 Python DSL 可以直接寫 LLM 程式而不是字串拼接。
- **TensorRT-LLM**:NVIDIA 官方欽定的「極致性能」引擎。核心思路是把模型在離線時 **編譯成靜態 engine**(類似 ONNX → TRT 的流程),用 FP8/FP4、custom CUDA kernels、in-flight batching 壓榨硬體最後 10–30% 的性能。代價是 build time 長、debug 困難、跨硬體完全綁定 NVIDIA。是固定模型、固定 SLA、規模夠大可以攤平編譯成本時的最佳選擇。

> 三者都已被收進 SemiAnalysis 的 **InferenceMAX / InferenceX** 持續基準計畫,每天滾動跑分,意味著差距會在週級別波動,任何單一 benchmark 都該看日期。

## 2. SGLang 核心特性

**RadixAttention** 是 SGLang 最具辨識度的設計。它把所有正在使用的 KV cache 組織成一棵 **radix tree(基數樹)**:相同 token 前綴只佔用一份 KV、後續分支共用父節點。對 Agent / RAG / few-shot prompting 這種「system prompt + retrieved docs 相同、user query 不同」的工作負載,實測 cache hit rate 可達 60–90%,prefill 成本接近零。對話多輪場景同一個 session 的歷史也會自動共用,不需要手動 prefix caching API。

**Frontend DSL** 則讓 SGLang 不只是個 server,而是個 **LLM programming language**。使用者用 `@sgl.function` 寫的 Python 函數會被編譯成請求圖,執行時自動進行:批次合併、共用前綴偵測、JSON / regex 約束解碼、`fork()` / `gen()` 並行分支。對於需要「先生成思考、再生成答案、再 self-critique」這種多階段 pipeline,DSL 比 client 端手動串 chat completions API 快數倍。

**結構化輸出** 在 token level 完成:SGLang 用 xgrammar / outlines 整合的 FSM 在 logits 層 mask 掉違規 token,確保 JSON Schema、正則、grammar 100% 合規,且幾乎沒有延遲懲罰。

## 3. TensorRT-LLM 核心特性

TensorRT-LLM 與其說是 server,不如說是 **編譯器 + runtime**。流程是:HF checkpoint → `convert_checkpoint.py` 轉成 TRT-LLM 中間格式 → `trtllm-build` 編譯成 `.engine` 二進位 → Triton / `trtllm-serve` 載入。編譯期會做的關鍵優化包括:

- **kernel fusion**:把 LayerNorm + QKV projection + RoPE 等融成單一 CUDA kernel,減少 memory roundtrip
- **FP8 / FP4 量化**:Hopper FP8、Blackwell FP4 native tensor core,搭配 NVIDIA Modelopt 校準,在精度幾乎不掉的前提下吞吐量提升 1.6–2.5×
- **in-flight batching (IFB)**:NVIDIA 自家名詞,等價於 vLLM 的 continuous batching,但實作層級下沉到 C++ runtime
- **plugin 化的 MHA**:FlashAttention、FlashInfer、xqa 等可互換
- **multi-LoRA、speculative decoding (EAGLE-2/3、Medusa、Lookahead)** 都已併入主線

代價是:engine 與 GPU SKU、batch size 範圍、序列長度上限**綁死**,換 GPU、換 max_batch_size 就要重編,build 一個 70B 的 engine 在單機上往往要 20–40 分鐘。

## 4. Benchmark 數字(2025 Q4 – 2026 Q1)

以下數字綜合自 SemiAnalysis InferenceMAX v1、LMSYS / SGLang 官方 blog、Spheron 與 Modal 的第三方測試,**caveat 列在表後**:

| 場景 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Llama 3 70B / H100×8 / chat 1k→1k / throughput | 基準 1.00× | 1.03–1.05× | 1.15–1.30× |
| Llama 3 70B / H100×8 / TTFT p50 @ 10 conc | 120 ms | 112 ms | 105 ms |
| **DeepSeek-V3 / H200×8 / decode tok/s** | 1.00× | **~3.1×**(MLA + FlashMLA) | 1.4–1.8× |
| DeepSeek-R1 / GB200 NVL72 / decode | n/a | **4× vs H100** | ~3.8× vs H100 |
| Llama 4 Maverick (MoE) / B200×8 | 1.00× | 1.2–1.4× | 1.3–1.5× |

**Caveat 必讀:**
1. SGLang 對 DeepSeek 的 3× 領先**僅在 MLA 模型**成立,Llama 系密集模型差距僅 3–5%
2. TensorRT-LLM 的領先是「編譯後峰值」,沒算 10–40 分鐘 build time 與 engine 大小
3. RadixAttention 的紅利在「prompts 全部 unique」的合成 benchmark 上會歸零
4. InferenceMAX 每天跑、版本差一週數字就不同;**請以查詢當下 inferencex.semianalysis.com 為準**
5. B200/GB200 數字大量受惠於 FP8/FP4 與 NVLink Switch,不可外推至 H100

## 5. 選型決策樹

```
                需要部署 LLM 推理服務
                        |
        ┌───────────────┼────────────────┐
   多模型快速迭代       Agent / RAG       固定模型 + 極致性能
   (研究 / PoC /        前綴重複多        + 規模 > 10 GPU 月
    新模型隔週換)      (system prompt    + SLA 嚴格
        |              + retrieved docs)         |
       vLLM                |                TensorRT-LLM
   ✓ HF 直接 load        SGLang            ✓ FP8/FP4 native
   ✓ 文件最全          ✓ RadixAttention    ✓ in-flight batching
   ✓ 無編譯           ✓ Frontend DSL      ✗ build 20–40 min
                       ✓ DeepSeek/MoE 最快  ✗ 換 GPU 要重編
                        |
            跨硬體 / 非 NVIDIA?
                        |
            ┌───────────┼────────────┐
        AMD MI300X     AWS           Google TPU
            |        Trainium 2          |
        vLLM-ROCm   NeuronX-vLLM      JetStream
        SGLang-ROCm  (官方 fork)     / vLLM-TPU
        (官方支援)
```

**單句版本:**「研究用 vLLM、Agent / DeepSeek 用 SGLang、上線壓榨用 TensorRT-LLM、跨硬體再做 fork 選擇。」

## 6. SGLang Frontend DSL 範例:RAG + Agent flow

```python
import sglang as sgl

@sgl.function
def rag_agent(s, question: str, docs: list[str]):
    # 1) 共享 system prompt — RadixAttention 自動 cache
    s += sgl.system("You are a careful research assistant. "
                    "Always cite document IDs.")

    # 2) 共享 retrieved docs 區塊(byte-identical 才能命中 cache)
    s += sgl.user("Context:\n" + "\n".join(
        f"[{i}] {d}" for i, d in enumerate(docs)))

    # 3) 只有問題在尾端變動 — 前面全部走 prefix cache
    s += sgl.user(f"Question: {question}")

    # 4) 多階段:先思考、再答、再 self-check (fork 並行)
    s += sgl.assistant("Let me think step by step.\n")
    s += sgl.gen("reasoning", max_tokens=256, stop="\n\n")

    forks = s.fork(2)
    forks[0] += sgl.gen("draft", max_tokens=300)
    forks[1] += sgl.gen("critique",
                       max_tokens=150,
                       regex=r"(LGTM|NEEDS_FIX:.+)")
    forks.join()

    # 5) 結構化最終輸出
    s += sgl.gen("final_json",
                 max_tokens=400,
                 json_schema='{"answer": "string", "cited_ids": [int]}')

# 啟動
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = rag_agent.run(question="What is RadixAttention?",
                      docs=retrieved_docs)
print(state["final_json"])
```

關鍵紅利:同一批 `docs` 跑 1000 個不同 `question`,prefill 只算一次。

## 7. TensorRT-LLM Build 流程(Llama 3 70B FP8)

```bash
# 0. 環境(假設 H100/H200,CUDA 12.4+)
docker run --gpus all -it --rm \
  -v $PWD:/workspace nvcr.io/nvidia/tensorrt-llm/release:0.18.0 bash

# 1. 拉 HF 權重
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct \
  --local-dir /workspace/hf/llama3-70b

# 2. FP8 量化 + 轉 TRT-LLM checkpoint (用 Modelopt 校準)
cd /workspace/TensorRT-LLM/examples/quantization
python quantize.py \
  --model_dir /workspace/hf/llama3-70b \
  --output_dir /workspace/ckpt/llama3-70b-fp8 \
  --dtype float16 \
  --qformat fp8 \
  --kv_cache_dtype fp8 \
  --calib_size 512 \
  --tp_size 8

# 3. 編譯 engine(這一步 25–40 分鐘)
trtllm-build \
  --checkpoint_dir /workspace/ckpt/llama3-70b-fp8 \
  --output_dir /workspace/engines/llama3-70b-fp8-tp8 \
  --gemm_plugin fp8 \
  --use_fp8_context_fmha enable \
  --max_batch_size 64 \
  --max_input_len 8192 \
  --max_seq_len 16384 \
  --max_num_tokens 16384 \
  --workers 8

# 4. 啟動 OpenAI-compatible server
trtllm-serve \
  /workspace/engines/llama3-70b-fp8-tp8 \
  --tokenizer /workspace/hf/llama3-70b \
  --host 0.0.0.0 --port 8000 \
  --tp_size 8 \
  --max_batch_size 64

# 5. 壓測
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-70b","messages":[{"role":"user","content":"hi"}]}'
```

換 batch_size、換 seq_len 上限、換 GPU 數量任一個 → 第 3 步重跑。

## 8. Production Trade-off 矩陣

| 維度 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Cold start(新模型上線) | 分鐘級(HF load) | 分鐘級 | **30–60 分鐘**(build) |
| Build / compile time | 無 | 無 | 高 |
| Peak throughput | 中 | 中–高(MoE 最強) | **最高** |
| Debug 友善度 | 高(Python stack) | 中(部分 C++) | **低**(engine 黑盒) |
| Model coverage | **最廣** | 廣(主流 + MoE 強) | 中(官方支援清單) |
| 結構化輸出 | xgrammar 整合中 | **token-level 原生** | guided decoding 需配置 |
| 跨硬體 | NV / AMD / TPU / Trainium | NV / AMD | **僅 NVIDIA** |
| 社群活躍度 (GitHub stars 2026Q1) | ~45k | ~17k | ~12k |
| 商業支援 | Red Hat / Neural Magic | LMSYS, xAI | **NVIDIA 官方 + NIM** |
| 適合規模 | 小–中–大 | 中–大 | 大 |

## 9. 2026 趨勢

1. **Disaggregated Prefill / Decode 全面普及**:三家都已支援把 prefill(算力密集)與 decode(記憶體頻寬密集)分到不同 worker pool。SGLang + Mooncake / NIXL 在 GB200 NVL72 已實測 **decode throughput +2.7×**;vLLM v0.7 後內建,TensorRT-LLM 從 0.17 開始支援。預期 2026 下半年成為大規模部署的預設架構。
2. **EAGLE-3 / P-EAGLE 三家整合到齊**:EAGLE-3 已是 vLLM、SGLang、TensorRT-LLM **生產級預設**選項,平均 2–3× 解碼加速;P-EAGLE(parallel draft)2025 Q4 進 vLLM 0.16,SGLang 整合中,B200 上對 GPT-OSS 20B 再加 1.05–1.69×。
3. **CXL 記憶體層級 KV offload**:CXL 3.0 把 KV cache 從 HBM 卸載到 DDR / persistent memory,延遲增加可控(數百 ns),但每 GPU 可服務的 context tokens 量級提升 10×。SGLang 的 HiRadixCache、vLLM 的 LMCache 都在往這個方向走,2026 下半年隨 Granite Rapids-AP / Turin 平台落地。
4. **MoE 專用最佳化**:DeepSeek-V3 / Llama 4 / Qwen3-MoE 推動 EP (expert parallelism) 與 DeepEP all-to-all kernel,SGLang 是目前開源側落地最快的,TensorRT-LLM 透過 NIM 跟進。
5. **FP4 與 microscaling**:Blackwell native FP4 在 TensorRT-LLM 已可用,SGLang / vLLM 透過 NVFP4 量化追上,精度損失在大模型上可控制在 <1%。

---

**TL;DR:** 從 vLLM 開始,流量穩定且前綴重複多就切 SGLang,規模大到值得花一個工程師月做 engine pipeline 就上 TensorRT-LLM。三者都在快速演進,**任何選型決策的有效期建議不超過一季**,建議 CI 內固定跑 InferenceMAX 子集追蹤回歸。

### 參考來源
- [SemiAnalysis InferenceMAX / InferenceX](https://inferencex.semianalysis.com/)
- [LMSYS: SGLang + NVIDIA on InferenceMAX & GB200](https://www.lmsys.org/blog/2025-10-14-sa-inference-max/)
- [Spheron: vLLM vs TensorRT-LLM vs SGLang H100 Benchmarks (2026)](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Particula: SGLang vs vLLM in 2026](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [NVIDIA: TensorRT-LLM Build Workflow](https://nvidia.github.io/TensorRT-LLM/architecture/workflow.html)
- [NVIDIA: trtllm-build CLI](https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-build.html)
- [LMSYS: Fast and Expressive LLM Inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [AWS: P-EAGLE Parallel Speculative Decoding in vLLM](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)
- [Jarvis Labs: Disaggregated Prefill-Decode Architecture](https://jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode)
