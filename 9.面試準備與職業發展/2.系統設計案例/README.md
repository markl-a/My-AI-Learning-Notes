# 2. 系統設計案例 — LLM System Design Interview 深入解析

> **定位**:本目錄是 **LLM/AI 系統設計面試** 的「深入長文版」案例庫。
> 每個 case 模擬一場 45–60 分鐘 senior-level 系統設計白板面試的完整推演。
>
> **與題庫的差異**:
> - [`1.LLM面試題庫/04_系統設計題.md`](../1.LLM面試題庫/04_系統設計題.md) → **速覽版**(每題 200–500 字,適合快速複習、刷題)
> - 本目錄(`2.系統設計案例/`) → **深入版**(每 case 2500–3000 字,含 capacity estimation、ASCII 架構圖、deep-dive、trade-off 表)
>
> 兩者互補:刷題庫先建立廣度,讀本目錄再建立深度。
>
> 也建議搭配根目錄全景圖 [`../../README.md`](../../README.md) 與 [`../../2024-2026_AI完整領域全景圖.md`](../../2024-2026_AI完整領域全景圖.md) 對照各子系統定位。

---

## 案例索引

| # | 案例 | 規模 / 場景 | 核心考點 | 狀態 |
|---|---|---|---|---|
| 01 | [Enterprise RAG System](./Case_01_Enterprise_RAG_System.md) | 10M docs / 100K users / 1000 QPS / p99 < 2s,多租戶 + GraphRAG | retrieval pipeline、hybrid search、rerank、multi-tenancy | ✅ 完成 |
| 02 | [LLM Gateway / API Platform](./Case_02_LLM_Gateway_API_Platform.md) | 50K RPS peak / 860B tokens/day / p99 gateway < 100ms,跨 provider routing + smart cache | API gateway、smart routing、prompt + semantic cache、provider fallback、cost attribution | ✅ 完成 |
| 03 | [Voice Agent 客服系統](./Case_03_Voice_Agent_Customer_Service.md) | 5K concurrent / p50 E2E < 500ms,SIP + STT/LLM/TTS pipeline + HIPAA/PCI | streaming、VAD/endpointing、barge-in、PII/DTMF mask、cascaded vs S2S | ✅ 完成 |
| 04 | [Multi-Agent Research System](./Case_04_Multi_Agent_Research_System.md) | 5K tasks/day / 500 concurrent / p90 < 30min,planner/researcher/critic/writer 多 agent | LangGraph supervisor、parallel fan-out、citation tracking、HITL、checkpoint resume | ✅ 完成 |
| 05 | [Computer Use SaaS](./Case_05_Computer_Use_SaaS.md) | 10K tasks/day / 1000 concurrent VM,瀏覽器 + desktop 自動化、SOC2 合規 | sandbox、vision-action loop、credential vault、prompt injection 防禦、recording/replay | ✅ 完成 |

---

## 系統設計面試套路(5 階段框架)

無論題目是 RAG、Agent 還是 Voice,白板上一律走這 5 步:

```
1. Clarification     (3-5 min)  → 把模糊題目收斂成具體需求
2. Capacity Estimate (3-5 min)  → 算 QPS、storage、cost、GPU
3. High-Level Design (10 min)   → 畫出 component diagram,標清 data flow
4. Deep Dive         (20-25 min)→ 面試官指定一兩個元件深挖
5. Trade-off & Scale (5-10 min) → 明確 bottleneck、failure mode、未來擴展
```

**反模式**:跳過 1、2 直接畫圖,或把所有元件畫得一樣詳細 — 面試官會認為你抓不到重點。

---

## 通用 SLI/SLO 與 Metrics 表(LLM 系統)

| 維度 | 指標 | 典型 SLO(B2B SaaS) | 量測工具 |
|---|---|---|---|
| Latency | p50 / p95 / p99 TTFT、E2E | TTFT p99 < 500 ms / E2E p99 < 2 s | OpenTelemetry、Langfuse |
| Throughput | QPS、tokens/sec | 因業務而異 | Prometheus |
| Quality | retrieval recall@K、faithfulness、groundedness | recall@10 ≥ 0.85 | RAGAS、TruLens、custom evals |
| Cost | $ / 1K tokens、$ / query、$ / DAU | <$0.01 / query(B2B) | Langfuse cost tracking |
| Availability | uptime、success rate | 99.9% (3 nines) / 99.95% (premium) | Pingdom、health checks |
| Safety | jailbreak rate、PII leak rate | <0.1% | Guardrails AI、Lakera |

> **面試小技巧**:capacity estimation 階段要主動報出 p99 < X、QPS = Y、cost target = $Z,展示「以 SLO 為目標反推架構」的思維。

---

## 推薦準備資源

### 系統設計基本功(非 LLM 專屬)
- **Alex Xu — System Design Interview Vol. 1 & 2**:必讀,建立 component 詞彙與 capacity estimation 直覺。
- **Designing Data-Intensive Applications (Martin Kleppmann)**:CAP、replication、consistency 模型,深聊 distributed system 必備。
- **ByteByteGo Newsletter / YouTube**:每週一個系統設計案例,動畫好懂。

### LLM / ML System 專屬
- **Chip Huyen — Designing Machine Learning Systems**:ML pipeline、feature store、monitoring,適合 LLM 之外的 ML 元件。
- **Chip Huyen — AI Engineering (2024)**:LLM 應用層工程,RAG、agent、eval 全覆蓋。
- **Anthropic / OpenAI Cookbook**:官方範例就是面試官心中的「reference architecture」。
- **LangChain / LlamaIndex production docs**:caching、streaming、observability 實戰。

### 面試刷題
- **Pramp / interviewing.io**:LLM/ML 主題 mock。
- **Hello Interview — ML & AI System Design track**:有結構化模板。

---

## 學習路徑建議

1. 先讀完 [`1.LLM面試題庫/`](../1.LLM面試題庫/) 全部 4 份題庫(廣度)。
2. 讀 Case 01(本目錄)完整一遍,理解「為什麼這樣設計」(深度)。
3. 找一面白板,蒙著 Case 01 題目自己重畫一次,對照差距。
4. 對 Case 02 ~ 05 重複步驟 2–3 — 五個 case 串成「資料 → 路由 → 互動 → 編排 → 執行」五層完整地圖。
5. 五案例完整一覽請見 [Case_05 結尾的對照表](./Case_05_Computer_Use_SaaS.md#五案例完整一覽)。

最後別忘了:**面試的是溝通能力,不是知識量**。多用「我會 X,因為 trade-off 是 Y vs Z,所以選 X」的句型。

---

返回:[`../README.md`](../README.md) | [全景圖](../../11.全景圖_LLM_AI_應用工程地圖.md)
