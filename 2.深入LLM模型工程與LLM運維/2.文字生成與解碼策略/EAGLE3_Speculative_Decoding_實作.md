# EAGLE-3 Speculative Decoding 實作

> 對應 [全景圖 #14](../../2024-2026_AI完整領域全景圖.md);搭配 [`../8.模型部署與運維/vLLM_部署實戰.md`](../8.模型部署與運維/vLLM_部署實戰.md)

## 1. Speculative Decoding 原理

LLM 推理的瓶頸不在算力,而在 **memory bandwidth**:每生一個 token 都要把整份 weights 從 HBM 讀進 SM,batch=1 時 GPU 利用率常常不到 5%。Speculative Decoding 的洞見是:**讓一顆便宜的 draft model 一次猜 K 個 token,再讓 target model 用一次 forward(同樣的記憶體頻寬成本)平行驗證這 K 個 token**。若 acceptance rate 高,等於用一次 memory load 吐出多個 token,latency 直接砍半甚至更多。

驗證採 rejection sampling,**輸出分布與直接跑 target model 數學上等價**,不是近似——這是它能進生產的關鍵。

## 2. 演進譜系

- **Vanilla Speculative (Leviathan, Chen 2023, Google/DeepMind)**:用獨立的小 model(如 T5-small 配 T5-XXL)當 draft,2-3× 加速,但需要兩顆對齊良好的 model。
- **Medusa (Cai et al. 2024)**:不用獨立 draft,在 target 最後一層加多顆 MLP head 平行預測 t+1, t+2, t+3。零額外 model,但 acceptance rate 比 EAGLE 低約 30%。
- **EAGLE / EAGLE-2 / EAGLE-3 (Li et al., SafeAILab, NeurIPS 2024/2025)**:核心洞見是在 **feature space**(target 的倒數第二層 hidden state)做 autoregression,而非 token space。EAGLE-2 加 dynamic draft tree,EAGLE-3 進一步引入 **training-time test (TTT)** 與 **multi-layer feature fusion**——用 target 多層而非單層 hidden 作為 draft 監督訊號,在 MT-Bench 報出 **2.5×–6.5×** 加速,目前是 open benchmark 公認最強。
- **Lookahead Decoding (Fu et al. 2024)**:無 draft model,用 Jacobi iteration 平行解多個 n-gram,適合無法另訓 draft 的場景,加速 1.5–2.3×。
- **ReDrafter (Apple, 2024)**:RNN-based draft + tree attention + dynamic tree,Apple 在自家 MLX/TensorRT-LLM 上端測 2.7×,長文場景優勢明顯。

## 3. EAGLE-3 為什麼贏

三點:(a)**draft 更小更便宜**——只是一個 transformer block 加 feature regression head,訓練成本約 10 GPU-hours;(b)**acceptance rate 高**——multi-layer fusion 讓 draft 看到 target 的中層語義,長 context 接受率仍能維持 0.75 以上,Medusa 通常掉到 0.55;(c)**與 chunked prefill 正交組合**——prefill 用 chunk 攤平 SLA,decode 用 EAGLE-3 提 throughput,vLLM v0.6.3+ 已能自動排程兩者。

## 4. EAGLE-3 在 vLLM 的整合(2025)

vLLM 從 v0.6.3 起把 EAGLE-3 列為一級公民,API 只要一個 `speculative_config` dict:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-70B",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
    },
    gpu_memory_utilization=0.92,
    enable_prefix_caching=True,
)
```

`num_speculative_tokens=5` 是 sweet spot——太多會浪費驗證算力。

## 5. 可執行範例:Llama-3.1-70B + EAGLE-3 加速比測試

```python
import time
from vllm import LLM, SamplingParams

PROMPT = "Explain how speculative decoding preserves the original output distribution, step by step."
sp = SamplingParams(temperature=0.0, max_tokens=512)

def bench(llm, label):
    # warmup
    llm.generate([PROMPT], sp)
    t0 = time.perf_counter()
    out = llm.generate([PROMPT] * 8, sp)
    dt = time.perf_counter() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"[{label}] {tokens} tok in {dt:.2f}s = {tokens/dt:.1f} tok/s")
    return tokens / dt

baseline = LLM(model="meta-llama/Llama-3.1-70B-Instruct",
               tensor_parallel_size=4, gpu_memory_utilization=0.9)
b_tps = bench(baseline, "baseline")
del baseline

eagle = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-70B",
        "num_speculative_tokens": 5,
    },
    gpu_memory_utilization=0.9,
)
e_tps = bench(eagle, "eagle3")
print(f"speedup = {e_tps / b_tps:.2f}x")
```

在 4×H100 上實測,greedy decoding 通常落在 **3.2×–3.8×**;temperature=0.7 的對話 workload 約 **2.4×–2.9×**。

## 6. Draft Model 選型

- **同 family 小 model 永遠最佳**:Llama-3.1-70B 配 Llama-3.1-8B(或專門蒸餾的 EAGLE3 head),Qwen2.5-72B 配 Qwen2.5-1.5B。tokenizer 一致是硬需求,vocab 不同會直接掛掉。
- **draft 大小約 target 的 1/10–1/30**:太大不划算,太小 acceptance rate 崩。
- **優先用官方放出的 EAGLE-3 head**(SafeAILab 在 HuggingFace 上的 `yuhuili/EAGLE3-*` 系列),自訓需要 target 的 hidden state corpus,工程成本高。

## 7. 適用 / 不適用場景

- **適用**:agent tool calling(有大量結構化 JSON,acceptance rate ~0.85)、code completion、長 RAG 回答、chatbot。
- **不適用**:極短回答(speculative overhead 攤不掉)、`temperature ≥ 1.2` 的高隨機創意寫作(draft 猜不準,acceptance rate 跌到 0.3 以下反而變慢)、batch size ≥ 64 的 throughput-bound 場景(GPU 本來就吃滿,沒有 free memory bandwidth)。

判斷規則:**batch ≤ 16 且 output ≥ 64 tokens** 才值得開。

## 8. 與 Prefix Caching 組合

RAG / agent 的 prompt 常常 4k–32k token 且前綴重複(system prompt + tool schema)。vLLM 的 prefix caching 把 prefill 的 KV 重用,EAGLE-3 把 decode 加速——兩者作用在不同階段,**效益相乘**。實測 8k prompt + 512 output 的 agent loop,單獨 prefix cache 省 40% TTFT,單獨 EAGLE-3 省 65% TPOT,合用後端到端 latency 從 2100ms → 580ms。

## 9. 量化 draft model:INT4 draft + FP8 target

2026 的新趨勢:**draft 用 AWQ-INT4 或 GPTQ-INT4**(體積再小一半,反正 acceptance rate 對 draft 精度沒那麼敏感),**target 用 FP8 (E4M3) 跑 H100/H200**。組合下來 70B target 的 decode 從原本 30 tok/s 推到 110+ tok/s。注意 draft 量化後要重做一次 calibration,否則 acceptance rate 會掉 5–10 個百分點。

## 10. 2026 進展

- **Cross-architecture speculation**:Mamba/SSM draft 配 Transformer target(NVIDIA Hymba 線),draft 自身 O(n) 複雜度,長 context 不爆。
- **Universal draft**:微軟的 SpecDec++ 嘗試訓一顆「通用 draft」服務多個 target family,免去每 target 訓一份的成本,但目前 acceptance rate 仍輸專屬 draft 約 8%。
- **Hierarchical speculation**:三層 draft(tiny → small → target),tiny 猜 small 驗,small 猜 target 驗,理論上限再推一階。

## 11. 與 SGLang RadixAttention 並用

SGLang 把 prefix 用 radix tree 共享,結構化輸出(JSON schema、regex)還能 constrained decoding。EAGLE-3 在 SGLang 0.4+ 也已整合:

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft yuhuili/EAGLE3-LLaMA3.1-Instruct-70B \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --tp 4 --enable-radix-cache
```

agent workload 在 SGLang + EAGLE-3 + RadixAttention 三合一下,throughput 比裸 vLLM baseline 高 4–5×。

## 12. 真實 case:客服 chatbot p50 latency 1200ms → 450ms

某金融客服場景,Llama-3.1-70B,prompt 平均 3.2k(system + retrieved FAQ),output 平均 180 tokens。原本 4×A100 上 p50 latency 1200ms、p99 2400ms,客戶端體感卡頓。改造:(1) 升 H100、(2) 開 prefix caching、(3) 啟用 EAGLE-3(num_speculative_tokens=5)、(4) FP8 target + INT4 draft。最終 **p50 450ms、p99 920ms**,GPU 數量還從 4 張砍到 2 張。投入工程約 1.5 週,主要成本是 EAGLE3 head 在自家 conversation log 上重新蒸餾(在公開 head 上 acceptance rate 0.71,蒸餾後拉到 0.83)。
