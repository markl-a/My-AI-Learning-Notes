# Case 02 — Multi-Tenant LLM API Gateway(類 LiteLLM / OpenRouter / Portkey)

> **題目類型**:多租戶 LLM API Gateway 系統設計(類 LiteLLM / OpenRouter / Portkey)
> **參考真實系統**:LiteLLM (OSS 龍頭)、OpenRouter ($1B+ ARR)、Portkey、Cloudflare AI Gateway、Helicone
> **同類題庫速覽**:[`../1.LLM面試題庫/04_系統設計題.md` Q2](../1.LLM面試題庫/04_系統設計題.md)
> **姊妹案例**:[Case_01 Enterprise RAG](./Case_01_Enterprise_RAG_System.md);[Case_03 Voice Agent](./Case_03_Voice_Agent_Customer_Service.md)(待補)

---

## 題目

> **Design a multi-tenant LLM API Gateway:統一介面接 10+ providers(OpenAI / Anthropic / Google / DeepSeek / 自托管 vLLM…),服務 100K 開發者、peak 50K RPS、p99 gateway overhead < 100ms、99.95% SLA、主 provider 故障 5 秒內 fallback,內建 smart routing、prompt cache、semantic cache、cost tracking、rate limit。**

時間預算:60 分鐘白板 + 10 分鐘 Q&A。
聽眾預期:Staff / Senior Staff Engineer 級。

---

## 1. Clarification(5 分鐘,候選人主動提問)

LLM Gateway 表面像「反向代理 + 計費系統」,但魔鬼藏在 streaming、tool calling、cost 一致性、provider 差異標準化這幾顆地雷。先把需求釘死再畫圖。

候選人應該主動問的 **10 個問題**:

1. **部署型態**:是 SaaS(我們托管,賺 markup 或訂閱費)還是 self-hosted OSS(像 LiteLLM proxy)?還是兩者都要?
2. **BYOK 模式**:是否需要支援 Bring-Your-Own-Key(用戶帶 OpenAI / Anthropic key,我們只收 gateway 費)?還是 unified billing(我們墊付 provider 費,用戶月結)?
3. **地域**:單區、多區、全球邊緣節點?中國境內的模型(DeepSeek、Qwen、Doubao)需要在岸路由嗎?
4. **多語**:gateway 不關心語言(只是 byte 轉發),但 semantic cache 跟 PII filter 需要多語支援嗎?
5. **串流必須性**:SSE streaming 是 P0 還是 P1?bidirectional streaming(client 取消 → upstream 中止)要嗎?
6. **結構化輸出**:OpenAI JSON mode、tool calling、Anthropic tools、Gemini function calling — 是只 pass-through 還是要做 schema 翻譯,讓用戶寫一套就跑遍所有 provider?
7. **Fine-tuned model 上架**:用戶可以上傳自己 fine-tune 的模型(LoRA adapter 或 full checkpoint)讓 gateway 路由嗎?
8. **安全層**:PII 自動偵測 / redaction、prompt injection 防護、output moderation — 哪些是預設 ON、哪些 opt-in?
9. **SLA 與退款**:99.95% 對應月停機 21.9 min,超過如何退款?provider 端故障算我們的還是免責?
10. **計費粒度**:per-token pass-through + markup %、per-call flat fee、月訂閱 quota,還是三種都要?是否要 model evaluation / Arena 雙盲評分功能?

**假設用的答案**(以下 design 基於這些):
- SaaS 為主、OSS proxy 為輔(雙模);BYOK + unified 都支援
- 3 region(us-east、eu-west、ap-southeast)+ 中國節點獨立部署
- SSE streaming 是 P0;bidirectional cancel 必須
- Tool calling **要做 schema 翻譯**(用戶寫 OpenAI 格式,內部翻譯到 Anthropic / Gemini)
- PII redaction、prompt injection scan 預設 ON(可關)
- 計費三種都支援,基礎是 per-token + markup,企業客戶用月訂閱

---

## 2. Requirements

**Functional**
- Unified OpenAI-compatible API(`/v1/chat/completions`、`/v1/embeddings`、`/v1/audio/*`),pass-through 10+ providers
- Smart routing:quality-first / cost-first / latency-first 三種 policy,可由用戶 header 指定或按 model alias 自動選
- Prompt cache(exact hash)+ semantic cache(embedding 相似度)雙層
- Cost tracking:per-user / per-project / per-model 三層彙整,即時 + 月結
- Auth & rate limit:API key、JWT、OAuth;token bucket per user / project
- Tool calling schema 轉接(OpenAI ↔ Anthropic ↔ Gemini)
- Streaming SSE bidirectional proxy(含 cancel、reconnect)
- Observability:request trace、cost dashboard、latency 分位、error rate 分 provider
- Fine-tuned model 上架(用戶提供 endpoint URL 或上傳 LoRA → 路由到自託管 vLLM)

**Non-functional**
- **Scale**:100K 開發者、50K RPS peak、平均 5K RPS
- **Latency**:p99 gateway overhead < 100 ms(**不含** upstream LLM 推理時間)
- **Availability**:99.95% per region(月停機 ≤ 21.9 min)
- **Failover**:主 provider 故障 → 5 秒內切到備援 provider
- **Cost overhead**:gateway 自身成本 < 5% of pass-through revenue
- **安全**:TLS only、API key 加密存(KMS envelope)、PII 不落地(可選)

---

## 3. Capacity Estimation

```
QPS:           peak 50K、平均 5K
Daily calls:   5K × 86400 = 4.3 億 calls/day
Tokens/call:   平均 input 1500、output 500,共 2000 tokens
Daily tokens:  4.3 億 × 2K = 860B tokens/day(!)
Cache hit:     目標 prompt cache 20% + semantic cache 15% = 35%
              → 實際打 provider:50K × (1-0.35) = ~32.5K RPS peak
                                 5K × 0.65    = ~3.25K RPS 平均

Storage:
  Trace log:   每 call ~3 KB(headers + meta,不存 body)= 1.3 TB/day
              壓縮 + 30 天保留 → ~10 TB/month(熱)+ 冷存 90 天 S3
  Prompt cache:exact hash → key 8B value 平均 4KB,熱集合 ~100 GB(Redis cluster)
  Semantic cache:embedding(1024 dim int8 = 1KB)+ response(平均 4KB)
              熱集合 5M 條 → ~25 GB(Qdrant / Milvus)
  Cost ledger: per-call 一筆,壓縮 + 列存(ClickHouse)~5 TB/month

GPU(自託管 vLLM tier,給 fallback 與 fine-tuned 用):
              假設 5% 流量走自託管 = 2.5K RPS peak
              Llama 3.3 70B + AWQ on H100 ≈ 60 tok/s decode、5 並發
              → 約 200 張 H100 起跳(僅備援 tier)
              成本上更傾向「自託管只跑 fine-tune + 7B/13B」,大模型仍走 API

頻寬:
              SSE streaming 平均 500 output tokens × 4 bytes = 2 KB/call
              50K RPS × 4 KB(含 in+out)= 200 MB/s peak in/out
              → 多 region 邊緣節點扛,單區 < 100 MB/s

連接數:
              SSE 平均連接 8 秒(streaming 期間)
              50K RPS × 8s = 400K 同時連接 peak
              → 單 instance 撐 50K 長連接(Go + epoll / Rust + tokio),需要 8+ instance per region
```

**關鍵發現**:Gateway 自身不燒 GPU、燒的是**連接數 + log 寫入頻寬 + cache 命中率**。架構必須圍繞「streaming 連接管理」「cache 命中率最大化」「cost log 寫入不塞爆 OLAP」做 trade-off。

---

## 4. High-Level Architecture

```
                                    ┌─────────────────────────────────┐
   Client (SDK / curl) ───TLS────▶  │  Edge L7 LB (Cloudflare / Envoy)│
                                    │  TLS termination, geo routing   │
                                    └────────────┬────────────────────┘
                                                 │
                                    ┌────────────▼────────────────────┐
                                    │  API Gateway Layer              │
                                    │  - AuthN/Z (API key / JWT)      │
                                    │  - Schema validation            │
                                    │  - Rate limit (token bucket)    │
                                    │  - PII / injection scan         │
                                    └────────────┬────────────────────┘
                                                 │
                            ┌────────────────────┼────────────────────┐
                            │                    │                    │
                  ┌─────────▼────────┐  ┌────────▼─────────┐ ┌────────▼─────────┐
                  │ Cache Layer      │  │ Smart Router     │ │ Async Logger     │
                  │ - Exact (Redis)  │  │ - Policy engine  │ │ (Kafka)          │
                  │ - Semantic (Qd)  │  │ - RL bandit      │ │ → ClickHouse     │
                  │  HIT → 直接回    │  │ - Health check   │ │ → Cost ledger    │
                  └────────┬─────────┘  └────────┬─────────┘ └──────────────────┘
                           │ MISS              │ chosen provider
                           └─────────┬─────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │  Provider Adapter Layer │
                          │  (plugin per provider)  │
                          │  - Request translate    │
                          │  - Tool schema convert  │
                          │  - Streaming normalize  │
                          └──┬───────┬───────┬──────┘
                             │       │       │
              ┌──────────────▼─┐  ┌──▼────┐ ┌▼─────────┐  ... 10+ providers
              │ OpenAI Adapter │  │Anthrop│ │ Gemini   │
              │                │  │ic Adp │ │ Adapter  │
              └──────┬─────────┘  └──┬────┘ └──┬───────┘
                     │               │         │
              ┌──────▼───────────────▼─────────▼──────┐
              │  Streaming Proxy (SSE bidirectional)  │
              │  - Cancel propagation                 │
              │  - Chunk normalize (OpenAI delta fmt) │
              │  - Reconnect + idempotency key        │
              └──────────────────┬────────────────────┘
                                 │
                          ┌──────▼──────┐
                          │ Client(SSE) │
                          └─────────────┘

   Self-Hosted Tier:  vLLM cluster(H100)用於 fine-tuned model + fallback
   Circuit Breaker:   per (provider, region, model) tuple,3 連續 5xx → open 30s
   Multi-region:      us-east / eu-west / ap-southeast / cn-shanghai 獨立 stack
```

---

## 5. Deep Dive

### 5.1 Unified Schema 設計

選 **OpenAI-compatible 為 base**(生態最廣,SDK 最多),擴展 provider-specific 欄位走 `extra_body` 或前綴命名空間。

```jsonc
{
  "model": "claude-3.7-sonnet",       // alias,router 解析到實際 provider+model
  "messages": [...],
  "stream": true,
  "tools": [...],                      // OpenAI tools 格式為 base
  "extra_body": {                      // provider-specific
    "anthropic": { "cache_control": "ephemeral" },
    "gemini":    { "safety_settings": [...] }
  },
  "x-gateway": {                       // gateway 自有
    "routing_policy": "cost-first",    // quality | cost | latency
    "fallback_models": ["gpt-4o-mini", "deepseek-v3"],
    "cache_mode": "semantic",
    "user_id": "u_123",
    "project_id": "p_456"
  }
}
```

**踩雷點**:Anthropic 的 `system` 是 top-level、OpenAI 是 messages[0] role=system、Gemini 是 `systemInstruction`。Adapter 必須吃下這層差異,**對用戶完全透明**。

### 5.2 Smart Router

三種 routing policy,實作為 policy chain:

```
Request → [Hard constraints: max_cost, max_latency, region] (filter)
       → [Policy scorer]
            ├─ quality-first:讀近 7 天該 model 在該類 prompt 的 win-rate(Arena ELO)
            ├─ cost-first:  min($/1K tokens × estimated tokens)
            └─ latency-first:min(p50 latency of healthy providers)
       → [Tie-breaker: RL bandit] 探索新興模型(ε-greedy ε=0.05)
       → [Health check filter] 過濾 circuit-open 的 provider
       → 選定 provider
```

**Quality 模式核心是 routing table** — 後台離線跑 Arena 與 benchmark(SWE-Bench、τ-Bench、MT-Bench),按 query type(code / chat / reasoning / vision)維護 leaderboard,routing decision 直接查表。Bandit 層處理新 model 上線時的冷啟問題。

### 5.3 Prompt Cache(Exact)

實作分兩層:
1. **Gateway 端 exact cache**:`hash(model + messages + tools + temperature 等 deterministic params)` → response。Redis 集群,TTL 預設 1 小時、用戶可調。**溫度 > 0 的 request 預設不 cache**(除非用戶顯式 opt-in)。
2. **Provider prompt cache pass-through**:Anthropic 的 `cache_control: ephemeral`、OpenAI 的 automatic prompt caching、Gemini 的 context caching — gateway 自動加 cache marker(system prompt + tool defs 永遠標 cache),把這層 cost saving 帶給用戶。

**命中率實測**(production 數據參考):exact 命中率 15–25%,Anthropic prefix cache 命中率 60–80%(因為 system prompt 大量重複)。**這兩層可疊加省 40–60% input cost**。

### 5.4 Semantic Cache

針對「語意相同、字面不同」query(FAQ 類、客服 bot)效果驚人:

```
新 query → embed(small model, e.g. Voyage-3-lite or bge-small)
        → Qdrant ANN 查 top-1
        → cosine > 0.95(threshold 可調)→ 命中,回傳舊 response
                                       (附 header X-Cache: SEMANTIC,讓用戶知道)
```

**Trade-off**:
- 命中率:可達 30–50%(特定場景),但 threshold 設太低會回錯答案
- 風險:推理類、code 生成類 query 絕對不能用(同樣問題不同細節答案天差地別)
- 解法:**per-project opt-in**、**per-model-tier 預設**(GPT-4o-mini 開、o1 永遠關)

### 5.5 Cost Tracker

三層彙整,寫入路徑用 Kafka → ClickHouse:

```
請求結束 → emit event {
            user_id, project_id, model, provider,
            input_tokens, output_tokens, cached_tokens,
            cost_usd (= input × in_rate + output × out_rate - cache_discount),
            latency_ms, timestamp
          } → Kafka(gateway 不阻塞主流程)
       → Consumer 1:即時 Redis counter(per-user spending,給 rate limit 用)
       → Consumer 2:ClickHouse(per-user/project/model/day 物化視圖)
       → Consumer 3:Billing service(月底結算)
```

**精度問題**:provider 回傳 usage 不一定即時(streaming 末尾才給),且某些 provider 對 cached tokens 計價不同。需要 per-provider **計費規則表**,定期校準(每月對 invoice)。

> phantom-mesh cost tracking 模組做的是 **per-agent attribution**(同個 user 跑多個 agent,每個 agent 該分多少成本) — 對應 gateway 這層的 `agent_id` 維度擴展,把 cost ledger 從 3 層拉到 4 層(user / project / agent / model)。

### 5.6 Streaming SSE Proxy

最容易出 bug、最影響體驗的環節。要處理:

1. **Bidirectional cancel**:client 關連線 → gateway 偵測 → 立刻 close upstream socket → upstream 停止計費(避免用戶被收一個他根本沒看的回答)。Go 用 `req.Context().Done()`、Rust 用 tokio cancellation token。
2. **Chunk 格式正規化**:OpenAI 是 `data: {"choices":[{"delta":{"content":"..."}}]}`,Anthropic 是 `event: content_block_delta` + 多種 event type,Gemini 是 newline-delimited JSON。Adapter 把全部翻譯成 **OpenAI delta 格式** 給 client。
3. **Reconnect & idempotency**:client 帶 `X-Idempotency-Key`,gateway 在 Redis 記錄(key, 上次 chunk seq),斷線重連可從 last seq 繼續(對 deterministic / temperature=0 才安全)。
4. **Backpressure**:client 慢 → 不能拖死 upstream connection(會被 provider 計費)。Gateway 有 buffer + 超時策略,client 太慢就主動 cancel。

> phantom-mesh 中對應的 **SSE 解析器**(`stream_parser`)就是處理這層多 provider 格式差異 — 同樣的核心問題:把 5 家 streaming 格式 normalize 成統一事件流,讓上層 agent loop 可以無視 provider 差異。

### 5.7 Provider Fallback(5 秒切換)

```
Primary call → 超過 (TTFT_p95 × 2) 還沒收到 first token → 標記 slow
            → 連續 3 次 5xx 或 timeout → circuit breaker open(30s)
            → 自動切到 fallback list 第二個 provider(同等級 model)
            → 期間 health checker 每 5s probe primary,recovered 後 half-open
```

**關鍵設計**:
- Fallback list **預先映射好** equivalence class(GPT-4o ↔ Claude 3.5 Sonnet ↔ DeepSeek V3 ↔ Gemini 2.0 Flash 屬於同 tier)。
- **5 秒 SLA 怎麼達成**:不能等 60s timeout — 用 hedged request(發送後 2 秒沒回 first token,並行發第二個 provider,誰先回用誰,另一個 cancel)。代價是 hedged 期間 2x cost,但只發生在 < 5% 流量。
- **Idempotency**:fallback 之前的 partial stream 要 buffer,切換後從頭重發(stream 中段切 provider 等同回答跳針,用戶體驗極差)。所以 streaming 模式下 fallback **必須在 first token 之前**完成,first token 之後 fail 就只能讓 client 重試。

> phantom-mesh 的 **provider fallback 邏輯**(`provider_router.rs`)實作了同樣的 hedged request + equivalence class,差別在 phantom-mesh 還整合了 cost-aware fallback(主 provider 故障時優先選同 tier 但更便宜的)。

### 5.8 Rate Limit

Token bucket per (user, model_tier),Redis 集群實作:

```
key = ratelimit:{user_id}:{tier}
value = { tokens, last_refill_ms }
LUA script atomically: refill → check → decrement
```

三層限流:
1. **RPS limit**:防 abuse,默認 100 RPS / user
2. **TPM limit**(tokens per minute):對齊 provider 端限制,默認 200K TPM(企業版可拉高)
3. **Spending limit**:per-day / per-month USD cap,超過直接 429

**Burst allowance**:bucket 容量 = 限額 × 2(允許 2x 短時 burst),refill 速率 = 限額 / 60s。

### 5.9 Tool Calling Schema 轉接

用戶寫 OpenAI tools 格式,gateway 翻譯到目標 provider:

```
OpenAI:    { tools: [{ type: "function", function: { name, description, parameters: <JSON Schema> } }] }
Anthropic: { tools: [{ name, description, input_schema: <JSON Schema> }] }
Gemini:    { tools: [{ functionDeclarations: [{ name, description, parameters: <OpenAPI-ish> }] }] }
```

**踩雷點**:
- JSON Schema 在 Gemini 用的是 OpenAPI 3.0 subset(不支援 `oneOf`、`anyOf` 某些 case)→ adapter 要 sanitize
- Anthropic 的 tool_use response 是 content block,OpenAI 是 message.tool_calls — streaming 期間 chunk 形狀完全不同,要在 SSE proxy 做轉換
- 多輪 tool calling 的 history 也得反向翻譯(`tool_use` → `tool_calls`、`tool_result` → role:tool message)

### 5.10 Multi-Region & Edge

- 邊緣 LB 用 Cloudflare / AWS Global Accelerator,**用戶就近接 edge gateway**
- Auth & rate limit 在 edge 完成(用 Redis Global Replica,read 5ms)
- 真正 provider 路由考慮**模型地理位置**:Anthropic / OpenAI 走美西、Gemini 全球、DeepSeek / 通義 走亞太 / 中國節點
- 中國節點獨立部署、獨立合規(資料不出境),共用 control plane(藍綠管控)

---

## 6. Bottlenecks 與 Mitigation

| Bottleneck | 症狀 | Mitigation |
|---|---|---|
| Cache 命中率不夠 | LLM cost 沒壓下來 | 加 semantic cache + Anthropic prefix cache pass-through + 共用 system prompt 模板庫 |
| Streaming 連接數爆炸 | 單 instance 撐不住 400K 長連接 | Go/Rust 寫的專屬 SSE proxy(epoll/io_uring);考慮 gRPC streaming 取代 SSE(內部)+ NATS pub/sub 解耦 |
| Cost log 寫入塞爆 OLAP | ClickHouse 寫入 lag → 即時 quota 失準 | Kafka 緩衝 + ClickHouse 寫批次 + Redis counter 作即時近似值,OLAP 只做歷史查詢 |
| 新 provider 接入慢 | 每家 API 差異大、4 週工 | Plugin 架構 + 通用 conformance test suite(50+ test cases 跑遍所有 provider)+ 用戶可上傳「自訂 adapter」(類 LiteLLM 做法) |
| Provider rate limit 跨用戶溢出 | 某用戶把 OpenAI key quota 用爆,影響其他用戶 | Per-provider account pool + tenant-aware fair share(weighted round-robin) |
| Tool calling 翻譯 bug | schema 翻錯導致 silent 失敗 | 上線前對齊測試集(100+ tool 定義跑所有 provider),持續 fuzz |
| Hedged request 雙倍成本 | 5 秒 SLA 副作用 | Hedge ratio 動態調(provider 健康時降到 1%,故障時拉到 100%) |

---

## 7. Trade-offs(明確表態,別騎牆)

| 決定 | 選 A | 選 B | 我的選擇 |
|---|---|---|---|
| Quality vs Cost vs Latency | 三者都優化(不可能) | 讓用戶選 policy | **三選二 + 用戶可指定**:預設 cost-first,熱門 model alias 走 quality-first |
| BYOK vs Unified billing | BYOK(我們不墊錢、用戶複雜) | Unified(我們墊錢、收 markup) | **雙模**:免費 / 開發者 tier 預設 BYOK,企業 tier 預設 unified(月結 PO) |
| Edge gateway 一致性 vs 延遲 | 強一致(rate limit 全球同步,延遲 +20ms) | 最終一致(各 region 獨立,可能超用 5%) | **最終一致**:rate limit 容忍 5% 超用、cost 限額用 Redis Global 強同步 |
| Semantic cache 預設 on/off | 預設 on(省錢、有風險) | 預設 off(安全、命中率低) | **預設 off,per-project opt-in + per-tier 推薦**(chat tier 開、code/reasoning tier 永遠關) |
| Self-host vLLM tier | 全部 model 都自託管(成本可控) | 全部走 API(輕資產) | **80/20**:80% 流量走 API、20% 流量(fine-tuned + 內部敏感 + 7B/13B fallback)自託管 |
| Tool schema 翻譯 vs 強制單一格式 | 翻譯(用戶寫一套就跑全部,工程量大) | 不翻譯(用戶要寫多套) | **翻譯**:這是 gateway 的核心賣點之一,值得投入 |

---

## 8. Extension 題(面試官可能追問)

1. **Confidential inference(用戶不想讓 provider 看到 prompt)**:接 confidential compute(Azure Confidential GPU、NVIDIA H100 TEE、AWS Nitro Enclaves),只路由到支援 TEE 的自託管 vLLM,gateway 自己也跑在 enclave 內(避免 gateway 看到明文)。trade-off:throughput 降 15–25%、可選 model 大幅縮減,但對金融 / 醫療 / 法律 tenant 是剛需。
2. **Fine-tuned model 上架**:用戶上傳 LoRA adapter(safetensors)+ base model 規格 → gateway 後台跑驗證(load test + safety eval) → 通過後 mount 到 vLLM cluster(`--enable-lora`)+ 註冊 model alias。冷啟動用 lazy load(首次調用載入 adapter,~3 秒);熱模型常駐。計費按 GPU-second + base model 費。
3. **Arena-style 雙盲評測**:加 `/v1/arena` endpoint,後台同時打 2 個 random model,用戶投票哪個好 → 累積 ELO 分 → 反哺 routing table 的 quality 維度。對標 LMSYS Chatbot Arena、OpenRouter 的 leaderboard。
4. **Agent eval 接入**:跑 SWE-Bench、τ-Bench、AgentBench、MMAU 做 model 品質排行,結果寫入 routing table。可作為 enterprise tier 賣點(「我們的 routing 是基於 1000 個真實 agent task 評測選出來的」)。
5. **Edge inference**:小模型(< 3B)放邊緣節點(Cloudflare Workers AI、Vercel Edge)做超低延遲 routing pre-check / classification → 把「該用大 model 還是小 model」這個決策本身做到 < 20ms p99。

---

## 結語(白板下台前 30 秒)

> 「總結:這套 Gateway 用 unified OpenAI schema 統一 10+ provider,smart router(quality/cost/latency 三 policy + RL bandit)做選型,雙層 cache(exact + semantic + provider prefix cache pass-through)壓 40–60% input cost,streaming proxy 做 bidirectional cancel + chunk 正規化,fallback 用 circuit breaker + hedged request 達成 5 秒切換,cost tracker 走 Kafka + ClickHouse 三層彙整。最大風險是 streaming 連接數爆炸與 tool calling 翻譯 bug,我會用 Rust 改寫 SSE proxy + 100+ tool conformance test 持續守門。下一步我會做的兩件事是:(1) 上 confidential compute tier 拿企業客戶、(2) 加 Arena + agent eval 反哺 quality routing,把 routing table 從 benchmark-driven 變 production-driven。」

---

> 後續案例 Case_03 ~ Case_05 待補:
> - Case 03:Real-time Voice Agent(streaming STT/LLM/TTS、barge-in)
> - Case 04:Multi-Agent Research Platform(orchestration、long horizon、self-critique)
> - Case 05:Computer-Use SaaS(sandbox、安全、replay)

返回:[`./README.md`](./README.md) | [`../1.LLM面試題庫/04_系統設計題.md`](../1.LLM面試題庫/04_系統設計題.md)
