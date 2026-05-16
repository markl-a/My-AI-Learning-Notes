# Case 01 — Enterprise RAG System(多租戶 + GraphRAG)

> 速覽版題目請見 [`../1.LLM面試題庫/04_系統設計題.md`](../1.LLM面試題庫/04_系統設計題.md);
> 整體目錄定位請見 [`./README.md`](./README.md);
> 全景圖請見 [`../../11.全景圖_LLM_AI_應用工程地圖.md`](../../11.全景圖_LLM_AI_應用工程地圖.md)。

---

## 題目

> **Design a multi-tenant enterprise RAG system handling 10M docs, 100K users, 1000 QPS peak, p99 < 2s, with GraphRAG capability.**

時間預算:60 分鐘白板 + 10 分鐘 Q&A。
聽眾預期:Staff / Senior Staff Engineer 級。

---

## 1. Clarification(5 分鐘,候選人主動提問)

面試官給的題目永遠是模糊的。先收斂需求 — 這一步分數比畫圖還高。

候選人應該主動問的 **10 個問題**:

1. **資料型態**:純文字?還是含 PDF、簡報、表格、影像、影片?是否需要 OCR / 視覺理解(ColPali、Nougat)?
2. **多語言**:單語(英)、雙語、還是 50+ 語言?跨語檢索是否需要?
3. **Model 政策**:可以用第三方 API(OpenAI、Anthropic),還是必須私有部署(on-prem、VPC、air-gapped)?
4. **資料合規**:GDPR / HIPAA / SOC2 / FedRAMP?資料是否能離開客戶 region?
5. **租戶隔離強度**:邏輯隔離(namespace)即可,還是要 physical isolation(每租戶獨立 vector cluster)?
6. **更新頻率**:文件是 mostly read-only,還是高頻寫入(streaming logs、tickets、emails)?freshness SLO 多少?
7. **Query 類型分布**:factoid lookup vs summary vs multi-hop reasoning vs global sensemaking?(後者才需要 GraphRAG)
8. **互動模式**:single-turn QA、multi-turn chat、agentic(會自動下子查詢)?
9. **成本預算**:$ / query 上限?$ / month 上限?
10. **既有系統**:是否已經有 Elasticsearch、Snowflake、SharePoint?要不要 incremental 串接而非 rebuild?

**假設用的答案**(以下 design 基於這些):
- 多模態(text + PDF + 投影片視覺),雙語(英 + 中)
- 可用第三方 API,但 sensitive tenant 走 private VPC
- 邏輯隔離(namespace)+ per-tenant KMS 加密
- Freshness:文件變更後 ≤ 5 分鐘可被檢索到
- Query 混合:80% factoid + 15% summary + 5% global(GraphRAG 觸發)

---

## 2. Requirements

**Functional**
- 多租戶文件 ingestion(支援 PDF/PPT/DOCX/HTML/Markdown,含視覺內容)
- 自然語言 query → grounded answer + citation
- Hybrid retrieval(keyword + semantic)+ rerank
- GraphRAG 模式:當 query 觸發 global sensemaking 時啟動(如「彙整所有 Q3 客戶抱怨主題」)
- 多輪對話(session memory,7 天保留)
- Per-tenant 用量 / 成本 dashboard

**Non-functional**
- **Scale**:10M docs、100K MAU、1000 QPS peak、10K QPS surge
- **Latency**:p50 < 800 ms、p99 < 2 s(E2E,不含 streaming first token)
- **Availability**:99.9% per region,multi-region active-active
- **Cost**:< $0.008 / query(blended,含 retrieval + LLM)
- **Security**:每 tenant 獨立 KMS、PII redaction、audit log(SOC2)
- **Freshness**:doc update → searchable 在 5 min 內

---

## 3. Capacity Estimation

```
Documents:      10M docs × 平均 20 chunks/doc        = 200M chunks
Embedding dim:  1024 (Voyage-3-large)                 = 4 KB/chunk(float32)→ 1 KB(int8)
Vector storage: 200M × 1 KB                           = 200 GB(int8 量化後)
                + HNSW index overhead (~1.5x)         = ~300 GB
Metadata + BM25 索引:                                  ~150 GB
Total vector tier:                                     ~450 GB(可單一 shard 放,但要 replica → x3)

QPS:            1000 peak, 10K surge
Tokens/query:   入:1500(prompt+context),出:300       = 1800 tokens
Daily tokens:   1000 QPS × 86400 s × 1800 tokens × 30% duty cycle
              ≈ 47B tokens/day
LLM cost(GPT-4o-mini blended):
              入 $0.15/M、出 $0.60/M
              ≈ 47B × (1500×0.15 + 300×0.60)/1800/1M
              ≈ ~$8K/day → ~$240K/month

Embedding throughput:
              新增 doc:假設 100K docs/day × 20 chunks = 2M chunks/day
              ≈ 23 chunks/sec → 1 個 Voyage batch endpoint 足夠
GPU(若自託管 reranker):
              BGE-reranker-v2-m3 @ A10G:~500 pairs/sec
              1000 QPS × top 50 candidates = 50K pairs/sec
              → 需要 ~100 個 A10G(或改用 Cohere Rerank API)
Cache:        Redis cluster ~200 GB(熱資料 + semantic cache)
```

**關鍵發現**:LLM cost 是第一大開銷($240K/month),其次是 rerank GPU。架構必須圍繞「降低 LLM 呼叫量」與「rerank 外包 vs 自託管」做 trade-off。

---

## 4. High-Level Architecture

```
                                    ┌─────────────────────────────────┐
                                    │   API Gateway (Kong/Envoy)      │
                                    │   AuthN/Z, rate-limit, tenant id│
                                    └────────────┬────────────────────┘
                                                 │
        ┌────────────────────────────────────────┼────────────────────────────────────────┐
        │                                        │                                        │
┌───────▼────────┐                ┌──────────────▼─────────────┐                ┌─────────▼──────────┐
│ Ingestion Path │                │   Query / Generation Path  │                │  Control Plane     │
│ (async, K8s    │                │   (sync, low-latency)      │                │  - Tenant admin    │
│  jobs)         │                │                            │                │  - Quota / billing │
└───────┬────────┘                │  ┌──────────────────────┐  │                │  - Eval pipeline   │
        │                         │  │ Query Rewriter       │  │                │  - Langfuse        │
   ┌────▼────┐                    │  │ (intent + HyDE)      │  │                └────────────────────┘
   │ Object  │                    │  └─────────┬────────────┘  │
   │ Storage │  ┌─────────────┐   │            │               │
   │ (S3/GCS)│─▶│ Doc Parser  │   │  ┌─────────▼────────────┐  │
   └─────────┘  │ - Unstruct  │   │  │ Router               │  │
                │ - ColPali   │   │  │ (factoid/summary/    │  │
                │   (vision)  │   │  │  GraphRAG?)          │  │
                └──────┬──────┘   │  └─┬──────────────┬─────┘  │
                       │          │    │              │        │
                ┌──────▼──────┐   │ ┌──▼───┐    ┌─────▼─────┐ │
                │ Chunker +   │   │ │Hybrid│    │ GraphRAG  │ │
                │ Metadata    │   │ │ Retr │    │  Layer    │ │
                │ Enricher    │   │ │      │    │ (Neo4j +  │ │
                └──────┬──────┘   │ │BM25+ │    │  community│ │
                       │          │ │dense │    │  summary) │ │
                ┌──────▼──────┐   │ │ +RRF │    └─────┬─────┘ │
                │ Embedding   │   │ └──┬───┘          │       │
                │ Workers     │   │    │              │       │
                │ (Voyage v3) │   │ ┌──▼─────────────▼────┐  │
                └──┬───────┬──┘   │ │ Reranker            │  │
                   │       │      │ │ (Cohere Rerank 3.5) │  │
              ┌────▼──┐ ┌──▼───┐  │ └──────┬──────────────┘  │
              │Qdrant │ │Open- │  │        │                 │
              │Cluster│ │Search│◀─┼────────┘                 │
              │(dense)│ │(BM25)│  │ ┌──────▼──────────────┐  │
              └───┬───┘ └──┬───┘  │ │ Context Builder     │  │
                  │        │      │ │ (dedupe, citation)  │  │
                  │        │      │ └──────┬──────────────┘  │
                  └────┬───┘      │        │                 │
                       │          │ ┌──────▼──────────────┐  │
                ┌──────▼──────┐   │ │ LLM Router          │  │
                │ GraphRAG    │   │ │ (mini→sonnet escalation)│
                │ Builder     │   │ └──┬─────────────┬────┘  │
                │ (offline)   │   │    │             │       │
                └──────┬──────┘   │ ┌──▼──┐    ┌─────▼─────┐ │
                       │          │ │GPT- │    │ Claude    │ │
                ┌──────▼──────┐   │ │4o-  │    │ 3.7       │ │
                │ Neo4j /     │   │ │mini │    │ Sonnet    │ │
                │ Memgraph    │   │ └──┬──┘    └─────┬─────┘ │
                └─────────────┘   │    │             │       │
                                  │ ┌──▼─────────────▼────┐  │
                                  │ │ Streaming +         │  │
                                  │ │ Guardrails + Cite   │  │
                                  │ └──────┬──────────────┘  │
                                  │        ▼ SSE             │
                                  └─────────────────────────┘

  Caches:  L1 prompt-cache (Anthropic/OpenAI) │ L2 semantic-cache (Redis + embedding sim)
           L3 KV-cache reuse (vLLM if self-host) │ L4 retrieval-result cache (1-min TTL)
```

---

## 5. Deep Dive

### 5.1 Ingestion

**Pipeline**:S3 event → SQS → K8s job(Argo Workflow)→ output 寫回 S3 metadata + 推入 embedding queue。

- **Doc parser**:`unstructured.io` 處理 80% 結構化文件;PDF/PPT/掃描件走 **ColPali / ColQwen2** 視覺路徑,避免 OCR 失真。複雜表格用 `Camelot` + LLM 校正。
- **Chunking**:語意切塊(`recursive` + `semantic` 雙策略),每塊 256–512 tokens,overlap 50。表格、圖、code block 不切斷。
- **Metadata enrichment**:每 chunk 附 `{tenant_id, doc_id, doc_type, lang, mtime, section_path, page_no, author, acl_tags}`。ACL 從 SharePoint/Drive 同步,實作 row-level security。
- **Idempotency**:用 `(tenant_id, doc_id, version_hash)` 當 primary key,re-ingest 時 upsert。

**踩雷點**:大檔(>500 MB PDF)會把 worker memory 撐爆 → 用 streaming parser + 分頁子任務。

### 5.2 Embedding

**選型 trade-off**:

| Model | Dim | Latency | $ / 1M tokens | MTEB | 適用 |
|---|---|---|---|---|---|
| OpenAI text-embed-3-large | 3072 | 中 | $0.13 | 64.6 | baseline |
| Voyage-3-large | 1024 | 低 | $0.18 | 65.0 | **本案首選** |
| Cohere Embed v4 | 1536 | 中 | $0.12 | 64.5 | 多模態好 |
| BGE-M3(自託管) | 1024 | 低(GPU) | ~$0.05(攤提) | 63.8 | 大量 ingest |

選 **Voyage-3-large**:int8 量化後 1 KB/vector,儲存與 retrieval 都最便宜;若日後 ingest 量再翻倍(>10M chunks/day)則切換 BGE-M3 自託管。

**Batch embedding**:ingest 走 batch API(50% off),query-time 走 sync API。背壓控制:Kafka + worker pool,限速 token-bucket。

### 5.3 Vector Store 選型

| 系統 | 規模 | 多租戶 | Filter 效能 | 運維 |
|---|---|---|---|---|
| pgvector | <10M | ✓(schema/RLS) | 一般 | 低(已有 PG) |
| Milvus | 100M+ | ✓(collection) | 好 | 中 |
| Qdrant | 100M+ | ✓(payload index) | **最好** | 中 |
| Pinecone | 100M+ | ✓(namespace) | 好 | 最低(SaaS) |
| Weaviate | 50M+ | ✓(class) | 一般 | 中 |

選 **Qdrant cluster**:200M vectors、payload filter(tenant_id + ACL tags)在 Qdrant 上 benchmark 最快;支援 scalar quantization、HNSW + filterable index。
拓撲:6 shards × 2 replicas,across 3 AZ,write quorum=2、read quorum=1。

### 5.4 Hybrid Retrieval(BM25 + Dense + RRF)

```
query → [BM25 top-50 from OpenSearch]  ┐
      → [Dense top-50 from Qdrant]      ├─→ RRF fusion(k=60)→ top-100
      → [Sparse SPLADE top-50, optional]┘
```

- **RRF (Reciprocal Rank Fusion)** 比加權和穩定,不需調 α/β。
- BM25 對「精確型 query」(產品代號、人名)recall 顯著高過 dense。
- Dense 對「意圖型 query」與多語跨語檢索表現更好。
- **Metadata pre-filter** 必須在向量檢索之前生效(否則 recall 會塌)— Qdrant 用 payload index 支援。

### 5.5 Rerank

**Cohere Rerank 3.5**(API,$2 / 1K searches)→ 用於 production;
**BGE-reranker-v2-m3**(自託管,A10G)→ 用於 dev / 高 sensitivity tenant。

從 top-100 → top-10。實測 NDCG@10 提升 ~12 個百分點。
**Latency budget**:rerank 100 docs ≈ 150–250 ms(這是 p99 最容易爆的環節,要 budget 嚴格管理)。

### 5.6 GraphRAG Layer

GraphRAG **不是預設啟用**(成本太高),由 router 判斷:
- 偵測 trigger keyword:「彙整」「主題」「整體看」「過去一年」+ token span > 10。
- 或 retrieval 結果熵(top-10 cosine 距離方差)過低 → 暗示 query 是 global sensemaking。

**離線 indexing**(Microsoft GraphRAG 風格):
1. 每 chunk 抽 entity + relationship(LLM-based,GPT-4o-mini)。
2. 構圖,跑 Leiden community detection。
3. 每 community 生成 summary(LLM)+ 向量化。
4. 寫入 Neo4j + 額外 community 向量 collection。

**Query-time**:
- Local mode → entity neighborhood retrieval(類似 Knowledge Graph QA)。
- Global mode → map-reduce over community summaries。

**成本告警**:GraphRAG indexing 每 1M chunks 約 $3K–$8K(LLM extraction);只開啟給願付高階方案的 tenant。

### 5.7 LLM Generation(分層 router)

```
80% queries → GPT-4o-mini  ($0.15/$0.60 per M tokens)
15% queries → Claude 3.5 Haiku 或 Gemini 2.0 Flash(備援 + cost diversification)
 5% queries → Claude 3.7 Sonnet / GPT-4.1  (router 偵測 hard:multi-hop、code、long context)
```

Hard query 判定:
- Query length > 100 tokens
- Context > 50K tokens
- Router classifier(小 BERT)信心分數 > 0.7
- Tenant SLA tier = "premium"

**Provider failover**:circuit breaker(Hystrix-style),3 連續 5xx 自動 cut 30 秒,跨 provider 路由。

### 5.8 Caching(4 層)

| 層 | 內容 | TTL | 命中率(實測) | 省的錢 |
|---|---|---|---|---|
| L1 prompt cache | system prompt + tool defs | 5 min(API 端) | ~70% | 50% input cost |
| L2 semantic cache | 整 query 的 embedding,相似度 > 0.95 命中 | 1 hr | ~15% | 100% LLM cost |
| L3 KV-cache reuse | per-session 前綴 | session 期間 | n/a(self-host only) | 30–40% latency |
| L4 retrieval cache | (query_hash, tenant) → top-K | 60 s | ~25% | retrieval cost |

L1 + L2 + L4 預期降 LLM 成本 **35–45%**,是把 cost 從 $240K/m 壓到 $150K/m 的關鍵。

### 5.9 Multi-Tenancy

- **Identity**:JWT(`tenant_id` + `user_id` + `acl_tags[]`),gateway 解析後寫入 request header。
- **Vector**:Qdrant 用 `payload.tenant_id` 做 mandatory filter;index 上有 tenant 維度的 partitioning。
- **Quota**:Redis token bucket per tenant(QPS、tokens/min、storage GB)。
- **Cost attribution**:每 request 從 Langfuse 抽 cost(input/output/embed/rerank),寫入 ClickHouse,小時級彙整給 billing。
- **Noisy neighbor**:premium tenant 走獨立 LLM API key + 獨立 worker pool(K8s priorityClass)。

### 5.10 Observability

- **Tracing**:OpenTelemetry,traceId 從 gateway 一路串到 LLM provider response header。
- **LLM 專屬**:Langfuse(prompt version、cost、token usage、eval score)。
- **Metrics**:Prometheus + Grafana(p50/p99 per stage、cache hit、retrieval recall)。
- **Online eval**:1% traffic shadow 跑 RAGAS(faithfulness、answer relevancy),低於閾值觸發 alert。
- **Cost dashboard**:per-tenant、per-model、per-day,異常 spike (>3σ) PagerDuty。

---

## 6. Bottlenecks 與 Mitigation

| Bottleneck | 症狀 | Mitigation |
|---|---|---|
| Embedding write QPS | 大規模 backfill 卡住 ingest | 切到 batch endpoint(50% off,12h SLA);或自託管 BGE-M3 |
| Rerank latency p99 爆 | 1000 QPS 下 rerank 排隊 | 降 top-K 從 100→50;或預算內換 Cohere(API 端 latency 穩定) |
| GraphRAG indexing 成本 | LLM extraction 一次 burn 數萬美金 | 改用 GPT-4o-mini 抽 entity;只對 high-tier tenant 啟用;增量索引 |
| Cold-start tenant | 新 tenant 前 100 query 很慢 | 預熱:ingest 完跑一輪 synthetic query 暖 cache + 預載 HNSW |
| LLM provider 故障 | 單一 region down | 雙 provider 雙 region failover,circuit breaker |
| Hot doc(熱門知識庫文件) | 反覆檢索同 chunk | L4 retrieval cache + L1 prompt cache 雙保險 |

---

## 7. Trade-offs(明確表態,別騎牆)

| 決定 | 選 A | 選 B | 我的選擇 |
|---|---|---|---|
| Latency vs Cost | self-host LLM(穩定 latency,高固定成本) | API(low fix,latency 抖) | **API + 嚴格 SLO 監控**:量未到自託管甜蜜點 |
| Recall vs Precision | 大 top-K + 強 rerank(慢) | 小 top-K(快但漏) | **top-100→rerank→top-10**:用 rerank 兼顧 |
| Freshness vs Cost | 即時 streaming index(貴) | batch 每 5 min(夠用) | **5 min batch**:符合 SLO,成本 1/10 |
| 多模態 ColPali vs OCR | ColPali(精準,GPU 貴) | OCR + text(便宜但失真) | **ColPali for PDF/PPT,OCR fallback**:差異化賣點 |
| GraphRAG default on vs opt-in | always on(體驗一致) | opt-in by tier(省錢) | **opt-in**:成本太敏感,且 5% query 才需要 |

---

## 8. Extension 題(面試官可能追問)

1. **加上 Voice Agent**:在前面接 STT(Deepgram Nova-3)+ TTS(ElevenLabs / Cartesia),retrieval 路徑同;但 latency budget 要重新拆,p99 目標縮到 800 ms,prompt 變短(<400 tokens),須加上 barge-in 與 partial response 流式回放 → 見 Case 03(待補)。
2. **加上 Causal-aware retrieval**:在 chunk metadata 抽 causal triples(`cause → effect`),query rewriter 偵測「為什麼/導致」類問題後,retrieval 同時撈相關 causal subgraph;rerank 階段加 causal coherence 分數。Trade-off:多 ~100 ms latency,但 multi-hop 答對率 +18%。
3. **加上 Agentic 子查詢**:router 偵測 complex query 後啟動 plan-and-execute(LangGraph),自動分解成 3–5 個 sub-query,各跑一次 retrieval,最後 synthesizer 合併。成本翻 3–5x,但 win-rate 在 hard benchmark +25%。
4. **跨 region multi-master 寫**:目前是 region-active-active 但 tenant pin to home region。若要做跨 region 同步寫,得引入 vector CRDT(實驗中)或退而求其次:每 region 各自 ingest + 跨 region async replication(eventual consistency,5 min)。
5. **完全 on-prem(air-gapped)版本**:Voyage 換 BGE-M3、Cohere Rerank 換 BGE-reranker、GPT-4o-mini 換 Llama 3.3 70B / Qwen2.5 72B(vLLM)。成本固定化、latency 可控,但 quality 下降 ~8–12 pp,需要更多 prompt tuning 與 fine-tune。

---

## 結語(白板下台前 30 秒)

> 「總結:這套系統用 hybrid retrieval + rerank 拿 base recall,用 4 層 cache 把 LLM cost 壓 40%,GraphRAG opt-in 給 global query,multi-tenant 用 Qdrant payload + per-tenant quota 隔離。最大風險是 LLM cost spike 與 rerank latency,我會用 cost guardrail(per-tenant cap)+ shadow eval 持續監控。下一步我會做的兩件事是:(1) 上 vLLM 自託管 7B 模型接 60% factoid 流量、(2) 把 GraphRAG indexing 從 GPT-4o-mini 換成自微調的 Qwen 抽取器,index cost 再降一半。」

---

> 後續案例 Case_02 ~ Case_05 待補:
> - Case 02:LLM Gateway / Model Router(routing + cost optimization + failover)
> - Case 03:Real-time Voice Agent(streaming STT/LLM/TTS、barge-in)
> - Case 04:Multi-Agent Research Platform(orchestration、long horizon、self-critique)
> - Case 05:Computer-Use SaaS(sandbox、安全、replay)

返回:[`./README.md`](./README.md) | [`../1.LLM面試題庫/04_系統設計題.md`](../1.LLM面試題庫/04_系統設計題.md)
