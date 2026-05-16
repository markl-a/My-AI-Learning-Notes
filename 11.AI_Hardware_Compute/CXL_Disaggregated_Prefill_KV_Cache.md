# CXL 與 Disaggregated Prefill / KV Cache 記憶體層級化

> 對應 [全景圖 #1/#14](../2024-2026_AI完整領域全景圖.md);[`./README.md`](./README.md) §6 記憶體;[`../2.深入LLM模型工程與LLM運維/8.模型部署與運維/vLLM_部署實戰.md`](../2.深入LLM模型工程與LLM運維/8.模型部署與運維/vLLM_部署實戰.md)

---

## 1. 問題:Prefill 與 Decode 在同一 GPU 互相搶資源

LLM 推理本質上是兩個性格完全相反的階段被硬塞在同一張 GPU 上:

- **Prefill(處理輸入 prompt)**:一次性把整段 prompt 餵進去算 attention,**compute-bound**。GPU 算力 (FLOPs) 拉滿,HBM 頻寬反而沒吃滿。長 prompt(RAG、長 system prompt、64K context)的 TTFT(Time To First Token)主要被 prefill 拖住。
- **Decode(逐 token 生成)**:每生成一個 token 就要把全部 KV cache 從 HBM 讀一遍做 attention,batch 內每個 sequence 步調不同,**memory-bound**。HBM 頻寬被讀爆,GPU 算力其實閒著。

把兩者塞同一張 GPU 會發生:**prefill 一進來就「打斷」正在跑的 decode batch**(因為要搶 SM 算力),造成 decode 的 ITL(inter-token latency)抖動;反之 decode batch 太大時又拖累 prefill 的 TTFT。這是經典的 head-of-line blocking,在高併發、長 context 場景下被放大。2024 年的 DistServe 論文與 Mooncake(Moonshot Kimi 的 serving 系統)分別從學術與工業界證明:**把兩階段拆開,SLO 達成率與每 GPU 吞吐量可以同時提升**。

---

## 2. Disaggregated Prefill / Decode 架構

核心思想很單純:**開兩個 GPU pool**——

- **Prefill pool**:對 compute 敏感,可用較少張高算力卡(H100/H200/B200),tensor parallel 切得細,專心把 prompt 算完。
- **Decode pool**:對 HBM 頻寬與容量敏感,可用較多張卡組成大 batch,專心吐 token。

兩 pool 之間靠**前端 router/scheduler**(類似 vLLM 的 `disagg_proxy` 或 Mooncake Conductor)分派請求:request 進來先到 prefill worker 算完拿到 KV cache,把 KV cache 透過高速網路傳給某台 decode worker,後續所有 token 都從那台 decode worker 吐出來。

---

## 3. KV Cache 傳輸成本:傳輸 vs 重算的 trade-off

KV cache 一點都不小。70B 模型、128K context、batch 32,KV cache 可以輕鬆破 150GB(已經超過單張 H100 的 80GB HBM)。Disaggregated 架構最大的成本就在這條傳輸線:

- **HBM → HBM via NVLink**:同機櫃內 NVSwitch fabric(NVL72 內),900GB/s 起跳,延遲微秒級。
- **跨節點 via InfiniBand (NDR 400Gb / XDR 800Gb)**:RDMA GPUDirect,50-100GB/s 量級,延遲十微秒到百微秒。
- **跨節點 via RoCE / Ethernet 400G**:類似頻寬但延遲略高,成本較低。

如果傳輸時間 > 在 decode pool 重新 prefill 的時間,disaggregation 就不划算。所以業界的判斷準則是:**長 prompt + 高併發**才值得拆;短 prompt(<2K token)直接 colocated 反而更快。

---

## 4. vLLM Disaggregated Prefill(2025 GA,Mooncake-inspired)

vLLM 從 v0.6 (2024 Q4) 引入 experimental disaggregated prefill,到 2025 年底透過整合 **Mooncake Transfer Engine** 作為 KV Connector,正式在 vLLM v1 進入可生產使用階段(12/19/2025)。架構:

- **KVConnector 抽象層**:讓不同的傳輸後端(Mooncake、NIXL、自研 RDMA)可插拔。
- **Mooncake Transfer Engine**:統一處理 DRAM ↔ DRAM、DRAM ↔ GPU VRAM、DRAM ↔ 遠端 NVMe 的高速傳輸,內建 RDMA、striping、parallel I/O。
- **整合對象**:vLLM、SGLang(EPD = Encode-Prefill-Decode 三段拆分,2025/12 上線)、LMCache、vLLM-Ascend(NPU)。

---

## 5. CXL 3.x / 4.0 記憶體池化:延遲容忍場景的 KV cache 後段班

GPU HBM 容量永遠不夠用,CXL(Compute Express Link)允許把 DDR5 模組外掛到伺服器,甚至跨機櫃共享一池記憶體。時程上:

- **CXL 2.0**(2022):memory pooling 雛形。
- **CXL 3.1**(2023 規格,2025 商用):fabric switching、多主機共享同一塊記憶體區。
- **CXL 4.0**(2025/11 發佈):基於 PCIe 7.0,128 GT/s,bundled ports 可達 1.5 TB/s。

對 LLM 推理的意義:**把 KV cache 的「冷層」放到 CXL DDR**——剛算完的 prefix、要被換掉的 session、可預期會再被命中的熱門 system prompt,從 HBM 卸到 CXL pool,需要時再拉回來。學術界 2025-2026 的 TraCT、Beluga、CXL-SpecKV 等系統已證實:相較 200G RDMA,CXL pool 可達 3.8x 加速;相較 100G RDMA 達 6.5x。實際大規模商用部署排在 2026-2027。

---

## 6. LMCache:KV cache 多級儲存與跨 session 共享

[LMCache](https://lmcache.ai) 是專門做 KV cache 管理的 open-source 中介層,2025/5 與 Mooncake 整合,目標是讓 KV cache 像 CPU cache 一樣有 L1/L2/L3 階層:

```
HBM (GPU)     →  最熱,當前 batch 的 active KV
 ↓
DDR (CPU)     →  最近換出去的 session、熱門 prefix
 ↓
NVMe SSD      →  常用 system prompt、RAG document KV
 ↓
物件儲存 S3   →  冷資料、跨叢集備份
```

關鍵能力:**跨 session、跨節點共享 KV**。同一段企業 system prompt(可能 4K-16K token)被 1000 個用戶呼叫,只要算一次 prefill,KV 存在 LMCache 池裡,後續所有 worker 命中就直接 load。SGLang HiCache(2025/9)走的是同樣方向。

---

## 7. Prefix Cache 進階:跨節點熱門 prompt prefix 共享

單機 prefix cache(vLLM 的 `--enable-prefix-caching`)只解決同一 worker 內的命中,真實企業場景需要跨節點共享。Mooncake Store / LMCache 提供的 **distributed prefix cache** 機制:

- 對 prompt 做 hash(通常以 block 為單位,16/32 token)。
- 全域 metadata service(etcd / Redis)記錄 hash → 哪個節點持有 KV。
- 命中時透過 RDMA 把對應 block 拉回 local。

對「企業 chat 都帶相同 system prompt」「agent 重複使用工具描述」這類場景,TTFT 可以從秒級降到百毫秒級。

---

## 8. 架構圖

```
                                          ┌─────────────────────────┐
   Client ──► API Gateway ──► Router ────►│ Prefill Pool            │
                                  │       │  GPU0  GPU1  GPU2  ...  │
                                  │       │  (compute-bound, H100)  │
                                  │       └────────┬────────────────┘
                                  │                │ KV cache transfer
                                  │                │ NVLink / IB RDMA
                                  │                ▼
                                  │       ┌─────────────────────────┐
                                  └──────►│ Decode Pool             │
                                          │  GPU0  GPU1  ...  GPUN  │
                                          │  (memory-bound, H200)   │
                                          └────────┬────────────────┘
                                                   │
                          ┌────────────────────────┴────────────────────┐
                          │   LMCache / Mooncake Store (KV hierarchy)  │
                          │   HBM ◄─► CPU DDR ◄─► CXL pool ◄─► NVMe   │
                          │                  ▲                          │
                          │                  └──► S3 (cold tier)        │
                          └──────────────────────────────────────────────┘
```

---

## 9. 可執行範例

### 9.1 vLLM 啟動 disaggregated 模式(Mooncake connector)

**Prefill worker**:

```bash
VLLM_USE_V1=1 vllm serve deepseek-ai/DeepSeek-V3 \
  --port 8100 \
  --tensor-parallel-size 4 \
  --kv-transfer-config \
    '{"kv_connector":"MooncakeConnectorV1",
      "kv_role":"kv_producer",
      "kv_rank":0,
      "kv_parallel_size":2}'
```

**Decode worker**:

```bash
VLLM_USE_V1=1 vllm serve deepseek-ai/DeepSeek-V3 \
  --port 8200 \
  --tensor-parallel-size 8 \
  --kv-transfer-config \
    '{"kv_connector":"MooncakeConnectorV1",
      "kv_role":"kv_consumer",
      "kv_rank":1,
      "kv_parallel_size":2}'
```

**Proxy(分派 request)**:

```bash
python -m vllm.entrypoints.disagg_proxy \
  --prefill-endpoints http://prefill-1:8100 \
  --decode-endpoints  http://decode-1:8200 http://decode-2:8200
```

### 9.2 啟用 LMCache(跨 session prefix 共享)

```bash
export LMCACHE_CONFIG_FILE=/etc/lmcache.yaml
export LMCACHE_USE_EXPERIMENTAL=True

vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --enable-prefix-caching
```

`lmcache.yaml` 範例:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 100   # GB,CPU DDR 給 100GB
remote_url: "mooncakestore://controller:50051/"
remote_serde: "naive"
```

---

## 10. 適用場景判斷

| 場景 | 是否值得拆 P/D + 用 KV pool |
| --- | --- |
| RAG with 8K-32K context、要求 TTFT < 1s | 強烈建議 |
| 企業 chatbot,固定 system prompt 4K+,QPS > 50 | 必上 prefix cache,選配 P/D |
| Agent 框架,反覆送相同 tool description | 必上 cross-session prefix cache |
| 短 prompt (<1K)、高 QPS、輸出短 | 不建議,colocated 更快 |
| 單機、流量小、context 短 | 不要拆,徒增延遲與運維複雜度 |

---

## 11. 真實 Case 與 Caveat

- **DeepSeek 自家部署 R1(SemiAnalysis 報導)**:傳聞中 DeepSeek 自己跑 R1 採用 P/D 分離 + 大規模 EP(Expert Parallelism)。**Caveat**:確切叢集配置從未官方公開,以 SemiAnalysis / Hao AI Lab / LMSYS 等第三方拆解為主;具體數字會因 batch size、輸入輸出比例變動很大。
- **LMSYS 在 96 張 H100 上跑 DeepSeek-R1**:3 節點(24 GPU)做 prefill、9 節點(72 GPU)做 decode,達 52.3k input TPS / 22.3k output TPS per node(2025/5 公開數據)。
- **NVIDIA GB200 NVL72 + SGLang**:單卡 26k input / 13k output TPS,InferenceMAX(SemiAnalysis 主導)benchmark 預設用 SGLang 跑 DeepSeek。
- **Mooncake @ Kimi(Moonshot AI)**:工業界第一個公開承認用 KVCache-centric disaggregated 架構服務真實流量(數百萬 DAU)的案例,2024 USENIX FAST'25 論文。

---

## 12. 2026 趨勢

1. **Disaggregated P/D 進入主流預設**:vLLM、SGLang、TensorRT-LLM、Dynamo 都把 P/D 拆分變成 first-class 部署模式,而非 experimental flag。
2. **CXL 3.1 / 4.0 大規模商用**:hyperscaler 開始在 AI rack 標配 CXL memory pool,KV cache 從「全部塞 HBM」走向「分層儲存」。
3. **EPD 三段拆分**:多模態 encoder 也被單獨拆出來(SGLang 2025/12 已上線),GPU pool 從 2 池變 3 池。
4. **KV cache 變成跨叢集資產**:像 CDN 一樣有全域 prefix cache fleet,熱門 system prompt 預先 warm 到所有 region。
5. **硬體共設計**:CXL controller 內建 attention 運算單元(near-data processing),把 KV cache 的 attention 計算 offload 到 CXL 模組,減少把 KV 拉回 GPU 的次數(2025/11 的 Beluga、Scalable PNM 等論文已是這個方向)。

---

**延伸閱讀**

- [Mooncake 論文(USENIX FAST'25)](https://www.usenix.org/conference/fast25/presentation/qin)
- [vLLM Disaggregated Prefilling 官方文件](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [LMCache x Mooncake 整合公告](https://blog.lmcache.ai/en/2025/05/08/lmcache-x-mooncake-unite-to-pioneer-kvcache-centric-llm-serving-system/)
- [Disaggregated Inference: 18 Months Later (Hao AI Lab)](https://haoailab.com/blogs/distserve-retro/)
- [TraCT: CXL Shared Memory KV Cache(arXiv 2512.18194)](https://arxiv.org/pdf/2512.18194)
- [Beluga: CXL-Based KVCache Architecture(arXiv 2511.20172)](https://arxiv.org/html/2511.20172v2)
