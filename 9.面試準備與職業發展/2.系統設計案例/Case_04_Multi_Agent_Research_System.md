# Case 04 — Multi-Agent Research System(類 OpenAI Deep Research / Google AI co-scientist / Perplexity Deep Research)

> **題目類型**:Multi-Agent Research / Deep Research 系統設計
> **參考真實系統**:OpenAI Deep Research、Google AI co-scientist、Perplexity Deep Research、Sakana AI Scientist v2、You.com Research、Anthropic Claude with computer use
> **同類題庫速覽**:[`../1.LLM面試題庫/04_系統設計題.md` Q5](../1.LLM面試題庫/04_系統設計題.md)
> **姊妹案例**:[Case_01 RAG](./Case_01_Enterprise_RAG_System.md);[Case_02 LLM Gateway](./Case_02_LLM_Gateway_API_Platform.md);[Case_03 Voice Agent](./Case_03_Voice_Agent_Customer_Service.md);[Case_05 Computer Use SaaS](./Case_05_Computer_Use_SaaS.md)(待補)
> **延伸 deep-dive**:[`22.Self_Improving_AI/`](../../22.Self_Improving_AI/README.md);[`3.LLM應用工程/3.Agent/`](../../3.LLM應用工程/3.Agent/)

---

## 題目

> **Design a Multi-Agent Research System:用戶輸入研究問題,系統在 3–30 分鐘內輸出結構化報告(含引用、圖表、可下載)。100K 註冊用戶、daily 5K 主動研究任務、peak 500 concurrent;每任務平均消耗 50–500K tokens、20–200 web fetches、5–50 次 LLM 呼叫;p90 完成時間 < 10 分鐘(短任務)/ < 30 分鐘(深度任務);報告必須有引用、章節結構、可重現(同問題 ±10% 相似);多 agent 拓樸 planner / researcher × N / critic / writer / fact-checker;支援 HITL 中途介入;cost 透明(per-task budget 上限觸達警告)。**

時間預算:60 分鐘白板 + 10 分鐘 Q&A。
聽眾預期:Staff / Senior Staff Engineer 級,熟悉 agent orchestration、LangGraph、long-running workflow 者佳。

---

## 1. Clarification(5 分鐘,候選人主動提問)

Deep Research 系統最容易在 clarification 階段就翻車,因為「研究」二字看起來像通用問題,實際每個 vertical(法律 / 醫療 / 投資 / 學術 / 競品)對檢索源、引用品質、報告格式的要求完全不同。先把場景釘死,後面 budget 才有得算。

候選人應該主動問的 **10 個問題**:

1. **主要 use case**:學術文獻綜述、競品分析、投資 due diligence、醫療文獻、法律研究 — 五者對 source 權威性、引用嚴謹度、報告長度差一個量級。
2. **檢索源範圍**:公開 web 即可,還是要進階檢索(arXiv、PubMed、SEC EDGAR、Bloomberg、Westlaw)?後者牽涉 API 授權、cost、合規。
3. **是否要存取付費 API**(Westlaw、Bloomberg Terminal、Crunchbase Pro)?如要,per-task cost 上限至少 $10+。
4. **報告長度上限**:1K 字 brief、5K 字 standard、20K 字 in-depth?Context window 與 token 預算直接決定。
5. **引用品質要求**:每個 fact 必須連回同儕審查文獻(WP3 級),還是 blog / 論壇也接受?fact-checker 嚴格度與成本成正比。
6. **是否要圖表生成**:matplotlib 跑 code exec 生成資料圖、mermaid 流程圖、還是真實截圖(從來源網頁截)?三者技術棧差很大。
7. **HITL 介入點**:用戶能否在 planner 完成後改 sub-question?能否在 final report 前改方向?能否要求重做某段?
8. **多語**:用戶輸入語 vs 報告輸出語 vs 來源語三者是否解耦?中文使用者問日文新聞要英文報告,如何處理?
9. **fact-check pass 嚴格度**:每 claim 都回查 source(慢但準)、抽樣 20% 回查(快)、還是只標 confidence(最快)?
10. **計費模型**:per-task 一口價、月訂閱、token-based、tier-based?budget cap 觸達是 hard stop 還是 soft warning?

**假設用的答案**(以下 design 基於這些):
- Use case 混合:40% 競品 / 投資分析、30% 學術文獻、20% 一般知識探索、10% 法律 / 醫療(走獨立合規 stack)
- 公開 web + arXiv + Semantic Scholar + SEC EDGAR(免費 / 低成本 API);Bloomberg / Westlaw 為 enterprise add-on
- 報告長度:short(< 2K 字)、standard(2K–8K 字)、deep(8K–20K 字)三檔
- 引用必須有 URL + 段落 snippet,品質分級(peer-reviewed / news / blog / forum)讓用戶選 threshold
- 圖表:mermaid 流程圖 default、code exec 跑 matplotlib 給數據圖、來源截圖為高階 tier
- HITL:planner 完成後 user 可修改 sub-question(預設 timeout 60s 自動繼續),final report 前可要求改方向
- 多語:輸入 / 輸出語可分離,內部統一以英文 reasoning(query 翻譯後檢索)
- Fact-check:standard tier 抽樣 30%、deep tier 全量、自訂可關
- 計費:tier-based(Free 5 任務 / 月、Pro $20/月 100 任務、Enterprise per-task billing),per-task budget hard cap

---

## 2. Requirements

**Functional**
- Query understanding(把模糊問題拆成可執行的 research plan)
- Planning(DAG-based sub-question 分解 + 依賴關係 + 優先序)
- Parallel research(N 個 researcher agent 並行跑 sub-question)
- Tool layer(web search、scholar search、PDF parse、code exec、structured DB query)
- Synthesis(critic + writer 整合 sub-research → 結構化報告)
- Citation tracking(每 fact → source URL + snippet hash,自動生成 IEEE / APA / Chicago)
- Fact-check pass(對 hallucination 與矛盾偵測)
- HITL(planner 後、writer 前兩個 checkpoint;隨時可暫停 / 改方向)
- Checkpoint resume(任意 step 失敗或被中斷後可從上次斷點繼續)
- Export(PDF / Markdown / DOCX / 結構化 JSON)
- Cost transparency(任務開始前估算、執行中 streaming 更新、超預算自動降級或停止)

**Non-functional**
- **Scale**:100K MAU、5K daily tasks、500 peak concurrent、~150K LLM API calls/day
- **Latency**:p90 < 10 min(short)、p90 < 30 min(deep);任務啟動 < 5s(planner 開跑)
- **Availability**:99.5% task completion rate;單一 LLM provider 故障不影響整體
- **Cost**:per-task blended < $2(short)/ < $5(standard)/ < $15(deep);hard cap by tier
- **Reproducibility**:同一 query 兩次跑、報告相似度 > 90%(BERTScore)
- **Citation quality**:standard tier ≥ 95% claims 有可驗 URL、deep tier ≥ 98%
- **Security**:用戶 query 與報告 per-tenant 加密、不混訓練;企業 tier 走獨立 VPC

---

## 3. Capacity Estimation

```
任務量:
  Daily tasks:        5K
  Peak concurrent:    500
  Avg duration:       8 min(short 4 min / standard 12 min / deep 25 min 加權)
  Task throughput:    5000 / 86400s × peak_factor(3x)≈ 0.17 task/s avg, peak ~0.5/s

LLM 消耗:
  Per task avg:       200K tokens(50K prompt cache hit + 100K reasoning + 50K tool output)
  Per task LLM calls: ~20 次(planner 1-2 + researchers 5-15 + critic 3-5 + writer 1-3 + fact-checker 2-5)
  Daily tokens:       5K × 200K = 1B tokens/day
  Peak RPM:           500 concurrent × 5 LLM calls/min avg = 2500 RPM
  Cost(blended):     planner / critic / fact-checker 用 o3 / Sonnet thinking(貴)
                     researcher 主流量用 GPT-4o / Sonnet(中)
                     writer / dedup 用 GPT-4o-mini / Haiku(便宜)
                     blended ≈ $1.5 / 200K tokens = $1.5 / task LLM cost
                     Daily: 5K × $1.5 = $7.5K/day = ~$225K/month LLM 單項

Web fetch:
  Per task:           80 fetches avg(20–200 range)
  Daily HTTP:         5K × 80 = 400K fetches/day
  Peak:               ~30 fetches/s(含並發 burst)
  Source mix:         60% generic web、20% scholar、10% arXiv、10% SEC / news API
  Search API cost:    Brave $0.005/query × 100K = $500/day
                     Tavily $0.008/query × 50K = $400/day
                     Serper / direct fetch 剩餘
                     Total search cost ~$1K/day = $30K/month

Code execution:
  Per task with chart: ~30% tasks need code exec = 1500 tasks/day
  Per task code calls: 2-5 calls(matplotlib / pandas / data fetch)
  E2B / Modal sandbox: ~$0.001/sec × 30s avg × 5K = $150/day

PDF parsing:
  Per task PDF count:  10 avg(arXiv / 投資報告 / 法律文件)
  ColPali / 自託管:   GPU $0.5/hr × 4 hr = $2/day(攤提)
  Reducto / API:       $0.003/page × 10 pages × 5K = $150/day

Storage:
  Trace + state:       ~50GB/day(LangGraph checkpoint + tool output)
                       → 30 天熱(S3 STANDARD)+ ClickHouse 索引 = $30/month
  Long-term archive:   報告 + final state ~5GB/day × 1 年 = ~1.8TB = $40/month
  Vector cache(intermediate research embeddings): ~20GB,Redis = $50/month

每任務成本拆分(目標 standard tier < $2):
  LLM(blended reasoning + writing):  $1.50
  Web search API:                     $0.20
  PDF parse / code exec / scholar:    $0.10
  Storage + trace:                    $0.05
  Infra overhead(K8s、queue、obs):    $0.15
  合計:                                ~$2.00 ✓ (標準任務)
  Deep tier × 3 倍 token 與 fetch:     ~$6-15(走 enterprise billing)
```

**關鍵發現**:跟 RAG / Voice Agent 不同,Multi-Agent Research 的成本主軸是 **長任務 × 多 agent × 高 reasoning token**;一個任務內部 LLM 呼叫次數從 5 到 50 不等,**單任務內 cost 變異極大**(短任務 $0.5、深任務 $15),所以 budget cap 與 tier 對齊比平均成本控制更重要。架構必須圍繞「per-task budget hard cap」「checkpoint 中斷可恢復」「reasoning model 分層使用」做 trade-off。

---

## 4. High-Level Architecture

```
                                ┌───────────────────────────────────────┐
   User (Web / SDK)  ──HTTPS─▶ │  Task API Gateway                      │
                                │  - AuthN/Z, tenant id, tier check     │
                                │  - Budget pre-check, task quota       │
                                │  - WebSocket upgrade (HITL + stream)  │
                                └────────────┬──────────────────────────┘
                                             │ POST /research
                                             ▼
                          ┌────────────────────────────────────────────┐
                          │  Task Orchestrator(LangGraph supervisor)  │
                          │  - DAG state machine                       │
                          │  - Checkpoint to Postgres + S3 every step  │
                          │  - Budget tracker (hard cap by tier)       │
                          │  - HITL interrupt() hooks                  │
                          └────────────┬───────────────────────────────┘
                                       │
        ┌──────────────────────────────┼─────────────────────────────────┐
        │                              │                                 │
┌───────▼────────┐         ┌───────────▼────────────┐         ┌──────────▼─────────┐
│ Planner Agent  │         │ Researcher Pool        │         │ Critic Agent       │
│ - o3 / Sonnet  │         │ - LangGraph subgraph   │         │ - Sonnet thinking  │
│   thinking     │         │ - Parallel × N (5-15)  │         │ - Quality scorer   │
│ - Decompose →  │         │ - Each: query → tool → │         │ - 矛盾偵測          │
│   sub-question │         │   evidence → summary   │         │ - Re-research      │
│   DAG          │         └────────┬───────────────┘         │   trigger           │
└────────────────┘                  │                          └──────────┬──────────┘
                                    │                                     │
                                    ▼                                     │
                       ┌──────────────────────────┐                       │
                       │ Tool Layer (unified)     │                       │
                       │ - Web search (Brave/     │                       │
                       │   Tavily/Serper)         │                       │
                       │ - Scholar (Semantic      │                       │
                       │   Scholar / arXiv)       │                       │
                       │ - PDF parse (ColPali /   │                       │
                       │   Reducto)               │                       │
                       │ - Code exec (E2B / Modal │                       │
                       │   sandbox)               │                       │
                       │ - Structured DB (SEC,    │                       │
                       │   Crunchbase)            │                       │
                       └──────┬───────────────────┘                       │
                              │                                           │
                              ▼                                           ▼
                       ┌──────────────────────┐         ┌──────────────────────────┐
                       │ Evidence Store       │◀────────│ Writer Agent             │
                       │ - Postgres (claims + │         │ - GPT-4o / Sonnet        │
                       │   citations)         │────────▶│ - Mermaid / table render │
                       │ - S3 (raw HTML/PDF)  │         │ - Citation auto-format   │
                       │ - Snippet hash dedup │         └────────────┬─────────────┘
                       └──────────────────────┘                      │
                                ▲                                    ▼
                                │                       ┌─────────────────────────┐
                                │                       │ Fact-Checker Agent      │
                                └───────────────────────│ - Verify每claim回source │
                                                        │ - Mark verified/        │
                                                        │   unverified/contradict │
                                                        └────────────┬────────────┘
                                                                     │
                                                                     ▼
                                                        ┌─────────────────────────┐
                                                        │ Export Service          │
                                                        │ - PDF (WeasyPrint)      │
                                                        │ - Markdown / DOCX       │
                                                        │ - JSON (machine-read)   │
                                                        └─────────────────────────┘

  Async / Side channels:
     HITL WebSocket  ←→ Orchestrator(interrupt 點 push 等用戶決策)
     Cost stream    ──▶ Frontend(每 step 更新累計 cost)
     Langfuse trace ◀── 每個 agent step 自動 emit
     Replay store   ◀── Postgres + S3 checkpoint(任意 step 可 resume)

  Multi-region:  us-east-1 主、eu-west-1 給 EU 合規 tenant
                  LLM provider 各 region 各自 fallback chain
```

---

## 5. Deep Dive

### 5.1 Planner Agent(把模糊問題拆成可執行 DAG)

Planner 是整個系統的「智商上限」 — 拆得好 5 分鐘出好報告,拆得爛 30 分鐘出垃圾。

**Model 選型**:**o3 / Claude 4.7 extended thinking** 為主,GPT-4o reasoning 為備援。Reasoning model 比 chat model 在 plan 拆解上 win-rate 高 ~25pp(內部 benchmark 50 題)。

**Prompt 結構**:
```
<role>你是 senior research analyst,把用戶 query 拆成 5-15 個 sub-question DAG</role>
<input>{user_query, tier_context, prior_research(若 follow-up)}</input>
<output_schema>
  {
    "sub_questions": [{"id", "question", "priority", "depends_on", "expected_source_type"}],
    "estimated_minutes", "estimated_cost_usd", "report_outline": [...]
  }
</output_schema>
<constraints>
  - sub_question 必須可獨立檢索回答
  - depends_on 用於序列化(B 需要 A 結果才能查)
  - expected_source_type ∈ {scholar, news, sec, blog, mixed}
</constraints>
```

**DAG 結構**:大部分 sub-question 是並行的,少量是依賴的(例如「A 公司的競品」必須先答完 A 公司是什麼,才能查競品)。Planner 輸出 `depends_on` 邊,Orchestrator 用 LangGraph 的 Send / Map-Reduce 把無依賴 node 並行 fan-out。

**HITL 第一檢查點**:Planner 輸出後 push 給 user(WebSocket),user 60s 內可:(a) 加 / 改 / 刪 sub-question、(b) 直接 approve、(c) 要求 planner 重做。預設 timeout auto-approve(避免長尾用戶卡 pipeline)。

**踩雷點**:Planner 容易拆出「太籠統」(只有 3 個大方向)或「太細碎」(20+ 個 sub-question 並行燒錢)。對策:加 cost-aware prompt(「請考慮預算 $X、預期 Y 個 sub-question 為宜」)+ 後處理校正(post-validator agent 檢查 sub-question 數量在 [5, 15],超出強制 merge)。

### 5.2 Researcher Pool(並行 fan-out)

每個 sub-question 對應一個 researcher subgraph(LangGraph 的 `Send(researcher_node, state)`)。Researcher 內部是 ReAct loop:

```
researcher_loop:
  for iter in 1..max_iter (default 5):
    1. 根據 sub_question + prior_evidence,決定 next tool call
    2. 呼叫 tool (web search / scholar / PDF parse)
    3. 從結果抽 evidence snippet(structured: {claim, source_url, snippet, confidence})
    4. 判斷是否已足夠回答 sub_question(LLM self-check)
       - 是 → 寫 sub_research_result 並結束
       - 否 → 修正 query,下一輪
  return sub_research_result with N evidence items
```

**並行控制**:LangGraph supervisor 用 `Send` 把 N 個 researcher 同時啟動,K8s pod 預留 capacity(500 concurrent task × avg 8 researchers = 4000 並行 subgraph,但每個 researcher 是 io-bound、單 pod 可跑 50 個 coroutine → ~80 個 pod 撐峰值)。

**Model 選型**:預設 **Claude Sonnet 4 / GPT-4o**(reasoning + tool use 平衡);啟發式偵測「這 sub-question 是 lookup 類」就降級到 **Haiku / GPT-4o-mini**(省 70% cost);偵測「需要多步推理」就升級到 reasoning model。

**Sub-research 結果格式**:
```json
{
  "sub_question_id": "q3",
  "evidence": [
    {"claim": "Tesla Q3 2024 revenue was $25.18B", "source": "https://sec.gov/...", "snippet": "...$25,182 million...", "confidence": 0.95, "type": "scholar"}
  ],
  "summary": "300 字摘要",
  "open_questions": ["未能回答的子問題,可能 trigger re-research"]
}
```

### 5.3 Tool Layer(統一介面)

把所有外部能力收斂到一個 tool registry,每個 tool 有統一 schema、cost metering、retry policy。

| Tool | Provider | Cost / call | Latency p50 | 適用 |
|---|---|---|---|---|
| `web_search` | Brave(主)/ Tavily(備)/ Serper(再備) | $0.005 / $0.008 / $0.003 | 300–800ms | 通用 |
| `scholar_search` | Semantic Scholar API(免費 100/5min)+ arXiv API | $0(rate-limited) | 500–1500ms | 學術 |
| `pdf_parse` | ColPali 自託管(GPU)+ Reducto API fallback | $0.001 / $0.003/page | 2–10s | PDF / 投影片 |
| `code_exec` | E2B sandbox(主)/ Modal(備) | $0.001/sec | 100ms 啟動 + N | matplotlib / 計算 |
| `structured_db` | SEC EDGAR、Crunchbase、Bloomberg(企業) | varies | 200ms–2s | 投資 / 競品 |
| `fetch_url` | 自家爬蟲池(rotating IP + Playwright headless) | $0.0005 | 1–3s | 反爬 / JS-heavy |

**統一介面**:phantom-mesh tool registry 用 `Tool.invoke(name, args, budget_ctx)` 包一層 — 每次呼叫自動寫入 cost ledger、emit Langfuse span、超 budget 自動拒絕(return graceful error 給 agent,讓 agent 改策略)。

**踩雷點**:
- **Cloudflare / reCAPTCHA 擋爬**:Brave / Tavily 主要返回的是 snippet 不是全文;需要全文時走 `fetch_url` 用 Playwright + rotating residential IP(供應商 BrightData / Smartproxy),命中率仍只 ~70%。
- **Search API rate limit**:Brave 1 QPS / Pro, Tavily 1000 RPM。500 concurrent task × 5 search call/min = 2500 RPM,單一 provider 撐不住,必須 **多家輪換 + 自家爬蟲池兜底**。
- **Semantic Scholar 限速**:100 req / 5min / unauthenticated;申請 API key 拉到 1 req/s,但仍是大瓶頸 → 加 query cache(相同 query 在 1h 內走 Redis)。

### 5.4 Critic Agent(品質守門)

每個 researcher 完成 sub-research 後跑 critic,輸出 quality score(0–1)與行動建議:

```
critic_input: {sub_question, sub_research_result, peer_results}
critic_output: {
  depth_score,        // 內容深度 0-1
  citation_score,     // 引用品質 0-1
  contradiction_flags,// 與其他 sub-research 矛盾
  action: "accept" | "re-research" | "reject_and_skip"
}
```

**矛盾偵測**:critic 看 cross-sub-question 的 claim 是否衝突(例如 sub_q1 說 Tesla 2024 Q3 營收 25.18B、sub_q2 卻說 24.5B)。觸發 fact-check 加重 + 兩邊都標 `contradicted`。

**Re-research budget cap**:每個 sub-question 最多 re-research 2 次(避免 critic 太嚴格陷入 loop)。第 3 次仍不通過 → 接受次優結果並在報告中標 `[low_confidence]`。

**Model**:**Claude Sonnet thinking** 或 **o3-mini**,critic 不需要 reasoning 全開,但 tool use 與 structured output 要強。

### 5.5 Writer Agent(整合 → 結構化報告)

Writer 接收所有 sub-research result + planner 的 report_outline,輸出最終報告。

**Prompt 結構**:
```
<outline>{planner 提供的章節大綱}</outline>
<evidence>{所有 verified evidence,citation_id 已預分配}</evidence>
<style>
  - 章節結構:Executive Summary → 各 sub-topic → Open Questions → References
  - 每個 fact 必須帶 [^cite_id] 引用
  - 適合處用 mermaid 流程圖、表格
  - 語言:{report_lang},學術風格 / 商業簡報風格依 tier
</style>
<constraints>
  - 報告長度:{target_word_count} ± 10%
  - 引用密度:每 100 字至少 2 個 citation
  - 不得添加 evidence 中沒有的 fact
</constraints>
```

**Citation auto-format**:writer 用 `[^c1]` 內聯引用,後處理用 Postgres 的 evidence 表查 source metadata,自動生成 IEEE / APA / Chicago 三種格式的 References 章節,讓 user 選 export style。

**圖表生成**:
- **Mermaid**:writer 直接吐 mermaid 代碼,前端 render(零額外成本)
- **資料圖表**:writer 識別到適合可視化的數據 → tool call code_exec → 跑 matplotlib → 存 S3 → 報告插 image markdown
- **截圖引用**(高 tier):pdf_parse 已存原始頁面截圖、writer 引用時插入

**HITL 第二檢查點**:writer 完成 draft 後 push 給 user(WebSocket stream),user 可:(a) approve 直接 export、(b) 要求重寫某章節(該章節重跑 researcher + writer)、(c) 改方向(回 planner 重新拆解)。

### 5.6 Fact-Checker Agent(對 hallucination)

最後一道防線,對每個 "X is Y" 句型做 source 回查。

**處理流程**:
1. 從 writer draft 抽出所有 atomic claim(LLM 抽,類似 SAFE / FactScore)
2. 每個 claim 對應 evidence id → 查 Postgres 原始 snippet
3. LLM judge:"Does the snippet support the claim?" → `verified | unverified | contradicted`
4. 未通過的 claim:writer 改寫加 hedge(「根據 X,可能...」)或標 `[unverified]` warning
5. Contradicted 的 claim:強制移除或加 `[來源衝突]` 註記

**Tier 策略**:
- Short tier:抽樣 20% claim 回查
- Standard:30%
- Deep:100% 全量
- Enterprise medical / legal:100% + 第二輪 cross-verify

**Model**:**GPT-4o-mini / Haiku**,fact-check 是 cheap classification 任務,不需大模型。每 task 約 50 claim × $0.0001 = $0.005,可忽略。

**踩雷點**:LLM judge 自己也會 hallucinate(claim 與 snippet 語義差一點就誤判)。對策:**double check**(同一 claim 用兩個不同 model judge,disagreement 走人工或保守標 unverified)。

### 5.7 Memory & State(LangGraph + 雙層 checkpoint)

**State schema(Pydantic)**:
```python
class ResearchState(BaseModel):
    task_id: str
    user_query: str
    tier: Literal["short", "standard", "deep"]
    budget_cap_usd: float
    cost_accumulated: float
    plan: Optional[ResearchPlan]
    sub_research_results: Dict[str, SubResearchResult]
    draft: Optional[Report]
    fact_check_results: Dict[str, ClaimCheck]
    hitl_pending: Optional[HITLRequest]
    status: Literal["planning", "researching", "critiquing", "writing", "fact_checking", "done", "paused", "failed"]
```

**Checkpoint 雙層**:
- **Hot tier**:Postgres(LangGraph PostgresSaver)— 每 node transition 寫一次,支援 fast resume(< 1s 從上次中斷處繼續)
- **Cold tier**:S3(完成的 task 歸檔,30 天後 GLACIER)— 用於 audit / replay / training data

**Resume 場景**:
- LLM provider 5xx → 從上一個成功 checkpoint 自動 resume(orchestrator 級重試)
- 用戶 close browser → task 繼續跑,完成後 email 通知
- 用戶 30 分鐘後回來 → load state,繼續看進度 / HITL

**Replay**:Postgres + S3 完整 state 序列,可以「重跑同一 task 對比不同 model」做 eval。

### 5.8 HITL(Human-in-the-loop)

兩個強制 interrupt 點 + 隨時暫停:

1. **Planner 後 interrupt**:`interrupt({"type": "review_plan", "plan": ..., "estimated_cost": ...})` → user UI 顯示 plan + 估算,user 可改 / approve / reject;預設 60s timeout auto-approve。
2. **Writer 後 interrupt**:draft 顯示給 user,可 approve / 改章節 / 改方向;預設 5 分鐘 timeout auto-approve。
3. **Anytime pause**:user 點 pause → orchestrator 收到 signal,當前 in-flight LLM call 完成後停在下一個 checkpoint,state 凍結;user 可繼續或丟棄。

**WebSocket 協議**:
```
server → client:
  {"type": "progress", "stage": "researching", "sub_progress": {q1: "done", q2: "running"}, "cost_so_far": 0.45}
  {"type": "hitl_request", "kind": "review_plan", "payload": {...}, "timeout_seconds": 60}
  {"type": "final_report", "report_id": "...", "download_urls": {...}}
client → server:
  {"type": "hitl_response", "request_id": "...", "action": "approve" | "modify" | "reject", "payload": {...}}
  {"type": "pause"} / {"type": "resume"} / {"type": "cancel"}
```

**LangGraph interrupt() 用法**:在 planner 後 / writer 前的 node 用 `interrupt(payload)`,Orchestrator 把 thread 掛起、寫 checkpoint、發 WebSocket;收到 response 後 `Command(resume=response)` 喚醒。

### 5.9 Cost Control(per-task budget hard cap)

**Tier 對應預算**:
| Tier | Token cap | $ cap | Re-research max |
|---|---|---|---|
| Free | 50K tokens | $0.50 | 1 |
| Pro | 500K tokens | $5 | 2 |
| Enterprise | 自訂(default 2M) | $20 default | 3 |

**Tracking 機制**:每次 LLM call 與 tool call 完成後寫 `cost_ledger` 表,Orchestrator 在每個 node transition 前讀 ledger,超 80% budget → 觸發降級;超 100% → hard stop。

**降級階梯**(soft cap 觸達):
1. 第一級:把後續 LLM call 從 reasoning(o3 / Sonnet thinking)降到 chat(GPT-4o / Sonnet)
2. 第二級:再降到 cheap(GPT-4o-mini / Haiku)
3. 第三級:限制剩餘 sub-question 數量(切剩 50%)
4. Hard stop:用目前 evidence 強制 writer 出短版報告 + 標 `[budget_capped]`

**Frontend 顯示**:WebSocket stream 即時 push cost(每 step 更新),user UI 顯示 cost bar + 預警(80% 黃、100% 紅)。

### 5.10 Citation Tracking(嚴格,可重現)

每個 evidence 進 Evidence Store 時自動算 hash:
```sql
CREATE TABLE evidence (
  evidence_id UUID PRIMARY KEY,
  task_id UUID,
  source_url TEXT,
  snippet TEXT,
  snippet_hash CHAR(64),  -- SHA256(normalized snippet)
  source_type TEXT,        -- scholar / news / sec / blog
  retrieved_at TIMESTAMP,
  quality_score FLOAT
);
CREATE INDEX idx_snippet_hash ON evidence(snippet_hash);
```

**Dedup**:同一 task 內 snippet_hash 重複的 evidence 合併;跨 task 同 snippet 共用 hash(節省 storage)。

**Export 時 citation format**:後處理 service 把 `[^c1]` 替換成 user 選定的 style(IEEE / APA / Chicago),References 章節按引用順序排。DOI / arXiv ID 自動 enrich(若 URL 含 arxiv.org / doi.org)。

**可重現性**:同 query 兩次跑 — planner 用 temperature=0、研究階段同樣 fetch URL(若 web 結果未變,evidence 集合應該重疊 > 85%)。Eval 用 BERTScore 比對兩次報告相似度。

### 5.11 Observability + Eval

**Langfuse trace**:每個 agent step 一個 span(planner、researcher × N、critic、writer、fact-checker),span 含 cost、token、latency、model。Trace tree 完整保留,debug 與 cost analysis 都靠它。

**Quality metrics**(per task):
- Citation count、citation density(per 100 words)
- Verified rate(fact-checker pass 比例)
- Report word count
- Sub-question count vs estimated
- User feedback(thumbs up / down + free-text)

**Internal benchmark**:50 個 ground-truth 報告(人類專家寫的),每次 model 升級或 prompt 改動跑全量 eval:
- **LLM-as-judge**:Claude Opus 比對 generated vs ground truth,輸出 0-10 分(coverage / accuracy / structure / citation_quality 四維)
- **人工抽檢**:每月 50 task 抽 5 個給 senior analyst rate,calibrate LLM judge
- **Reproducibility**:同題目重跑 5 次取 pairwise BERTScore,< 0.85 alert

---

## 6. Bottlenecks 與 Mitigation

| Bottleneck | 症狀 | Mitigation |
|---|---|---|
| Web search rate limit(Brave / Tavily 撞牆) | 多 sub-question 並行時 search 排隊、整個 task 卡住 | 多家輪換(Brave + Tavily + Serper + 自家爬蟲池),per-task 內 search budget 配額;熱門 query 1h Redis cache |
| Cloudflare / 反爬擋住目標網站 | fetch_url 失敗、evidence 不全 | rotating residential IP(BrightData)+ Playwright stealth;失敗 retry 3 次後接受 partial、critic 標 `[source_limited]` |
| LLM cost 超預算 | 用戶報告寫一半就 hard stop | 階梯降級(o3 → Sonnet → GPT-4o → mini),writer 強制壓縮 + 標註;預估階段就提示用戶升級 tier |
| Long task 中途 LLM 5xx | 25 分鐘 task 跑 20 分鐘掛了 | 每 sub-question 完成 + 每 agent step 寫 PostgresSaver checkpoint;orchestrator 級自動 resume(retry budget 3 次) |
| Parallel sub-research 結果矛盾 | 不同 sub-question 對同事實給出不同答案 | critic 強制 consistency check,觸發 fact-checker 加重 + 兩邊都標 `[contradicted]`;writer 必須交代矛盾 |
| Planner 拆得太碎(20+ sub-question) | task cost 失控 | post-planner validator 強制 [5, 15] 區間,超出觸發 merge prompt;cost-aware prompt 告知 planner 預算 |
| HITL 用戶不回應 | task 永久 pending | 預設 timeout(planner 60s、writer 5min)auto-approve;email 提醒 + 24h 後 hard cancel |
| Fact-checker 自己 hallucinate(誤判 verified) | 報告有錯但被標 verified | double-check(兩個 model judge),disagreement 走保守路徑;eval set 持續追蹤 fact-checker FPR/FNR |
| Code exec sandbox 爆衝 | 用戶 query 觸發 matplotlib 跑無限迴圈 | E2B / Modal 設 30s timeout + memory cap;同一 task code call 最多 5 次 |
| Replay 來看不一致(同 query 兩次差很大) | reproducibility 打不到 90% | temperature=0 + seed pin、search result snapshot(同 query 1h 內 cache 結果)、planner 多次決定取 majority |

---

## 7. Trade-offs(明確表態,別騎牆)

| 決定 | 選 A | 選 B | 我的選擇 |
|---|---|---|---|
| Reasoning model(o3 / Sonnet thinking)全用 vs 分層 | 全用(品質高、慢且貴) | 分層(planner / critic 用、researcher 用 chat) | **分層**:planner + critic + fact-checker 用 reasoning(thinking budget 5K-10K);researcher 主流量用 chat,只在啟發式判斷「需多步推理」時升級 |
| Parallelism 深(15 並行 researcher) vs 淺(5 並行) | 深(快但貴) | 淺(慢但便宜) | **依 tier**:Free / Pro tier 限 8 並行,Enterprise / Deep 走 15;Planner 拆解時就 cost-aware |
| Hallucination 防護:fact-checker 全量 vs 抽樣 | 全量(慢、貴但準) | 抽樣(快、便宜但漏) | **tier-based**:Standard 30% 抽樣 + critical claim 全量(透過 LLM 抽哪些是 "重要 fact");Deep / 醫療 / 法律 100% |
| HITL 強制 vs 全自動 | 強制 HITL(品質好、慢) | 全自動(快、用戶懶) | **預設兩個 interrupt 點 + auto-approve timeout**:planner 60s、writer 5min;用戶 idle 時 task 不卡住,但 active 用戶能介入 |
| 來源權威 vs 廣度 | 只收 peer-reviewed(嚴謹但少) | 收所有(全面但雜) | **混合 + 分級標籤**:全部都收,但 evidence 表標 source_type,user 可設「只看 peer + sec」filter,critic 也據此給 quality_score |
| Multi-region vs 單 region | Multi(合規 + 低延遲) | 單 region(運維簡單) | **us-east 主 + eu-west 給 EU tenant**:tenant home region pinning;LLM provider 各 region 各自 fallback chain |
| 自家爬蟲池 vs 純 search API | 自家(全文取得、cost 可控) | 純 API(輕資產、合規簡單) | **search API 主流量 + 自家爬蟲兜底**:80% 流量靠 Brave/Tavily snippet 夠用,20% 需要全文才走 Playwright pool |

---

## 8. Extension 題(面試官可能追問)

1. **加入 confidence score(每段「我有多確定」)**:writer 在輸出時 per-section 標 confidence(由 critic 的 depth_score + fact-checker 的 verified_ratio 加權),前端用色塊顯示(綠 > 0.8、黃 0.5–0.8、紅 < 0.5)。Trade-off:用戶看到紅色段會質疑系統價值;UX 上要搭配「想知道更多?點此擴展研究」(觸發 re-research),把警告轉成 upsell。

2. **支援多模態 input(用戶丟 PDF / 圖片當研究材料)**:Task API 接受 file upload → 上傳到 S3 → 預處理 pipeline(ColPali 抽 vision embedding + LLM 抽 structured summary)→ 把 file_id 注入 planner state。Researcher 在需要時 tool call `read_user_upload(file_id)`。挑戰:大 PDF(500 頁年報)cost 控制 — 加 page-level retrieval,只把相關頁送 writer。對應 [Case_01 ColPali](./Case_01_Enterprise_RAG_System.md#52-embedding) 那套技術 stack 複用。

3. **加入 collaboration(多人共同編輯 + LLM 助手)**:每個 task 變 shared workspace,WebSocket multi-user(Yjs / Liveblocks 做 CRDT);user A 改 sub-question,user B 即時看到;LLM 助手作為「第 N+1 個 collaborator」,在 chat panel 接 user 指令(「再多查 X」「重寫第 3 章」)。技術重點:state 從 single-user 改 multi-user CRDT、HITL interrupt 要對應到具體 user。商業上是企業客戶最買單的 feature(研究團隊協作)。

4. **加入「持續 monitor」(用戶問題每週重跑找新證據)**:訂閱型 — user 把任務標 `monitor: weekly`,Scheduler(K8s CronJob)每週重跑同 plan,diff 新 evidence(snippet_hash 與上次比對),若 ≥ N 個新證據 → 生成 delta 報告 email 給 user。挑戰:long-running cost 控制(每週重跑都燒錢)→ 增量策略:planner 只重跑「時效性高的 sub-question」(news / sec filing),evergreen sub-question 沿用上次 evidence。對應 perplexity Spaces / 對標 Bloomberg 的 monitoring 產品。

5. **報告自動轉成 slide deck(Gamma / Tome 那種)**:writer 額外輸出 outline 給 slide-writer agent,後者按章節生成 5-15 張 slide(每張一個 key point + supporting evidence + visual)。整合 reveal.js / python-pptx 渲染成 PPTX。商業上是 consulting tenant 的 killer feature(研究員寫完報告還要做 slide deck,直接省 2 小時)。

---

## phantom-mesh 在本系統的角色(回應 Case_01-03 一脈)

- **Multi-Agent 協調(supervisor pattern + handoff token)**:phantom-mesh 的 `agent_supervisor` 模組把 planner → researcher × N → critic → writer → fact-checker 的 handoff 包成統一 token-based 協議,每個 agent 收到 token、處理完用 `handoff_to(next_agent, state)` 傳遞,Orchestrator 不需要硬編碼 graph。對應 LangGraph supervisor + Send pattern,但抽象一層讓 agent 拓樸可組裝。
- **Tool Layer 統一 registry**:phantom-mesh `tool_registry` 提供 web search / scholar / pdf_parse / code_exec / structured_db 同一 `Tool.invoke()` 介面,內建 cost metering、retry policy、provider fallback(Brave → Tavily → Serper)。複用 [Case_01](./Case_01_Enterprise_RAG_System.md) 的 retrieval pattern、複用 [Case_02 Gateway](./Case_02_LLM_Gateway_API_Platform.md) 的 provider fallback 邏輯。
- **Cost Tracker per-task multi-agent attribution**:phantom-mesh `cost_attribution` 模組擴展自 [Case_02](./Case_02_LLM_Gateway_API_Platform.md) 的 per-request ledger,變成 per-task × per-agent × per-step 三維 ledger;每個 agent 自己的 cost、tool call cost、reasoning token 全部歸到 task,前端 cost bar 即時 push;超 budget 自動降級或 hard stop。對應 Langfuse cost dashboard 的 backend。

---

## 結語(白板下台前 30 秒)

> 「總結:這套 Multi-Agent Research 系統用 LangGraph supervisor 編 planner / researcher × N / critic / writer / fact-checker 五階段 DAG,planner 用 o3 / Sonnet thinking 拆 sub-question、researcher 並行 fan-out 跑統一 tool layer(Brave / Tavily / Scholar / E2B / pdf_parse),critic 質量守門 + 觸發 re-research,writer 整合 + auto-citation,fact-checker 對 hallucination 跑 verified/unverified 標籤。Per-task budget hard cap 與 tier 對齊(Free $0.50、Pro $5、Enterprise $20),超預算階梯降級到 mini model。LangGraph PostgresSaver + S3 雙層 checkpoint 讓 30 分鐘 long task 任意 step 可 resume;HITL 在 planner 後與 writer 前兩個 interrupt 點,WebSocket 推 cost 與 progress。phantom-mesh 在 agent supervisor + tool registry + cost attribution 三處複用。最大風險是 long task 中途失敗與 cost 不可控,我用 checkpoint 雙層 + budget hard cap + 階梯降級三招守。下一步兩件事:(1) 上 confidence score UI 把品質透明化、(2) 加 monitoring 訂閱型 product 拉企業 retention。」

---

### 面試官最會追問的 3 個 follow-up

1. **「Planner 拆出 12 個 sub-question 並行跑,有 2 個失敗、3 個 critic 評低分要 re-research,你怎麼決定何時停手出報告?」** — 答:三層停止條件取 AND。(a) 預算:cost_accumulated > 90% budget_cap 強制收尾;(b) 時間:elapsed > deep_tier 的 SLA 30min 軟上限,軟超後 writer 直接用現有 evidence 出;(c) 品質:覆蓋率(已完成 sub-question / planner 拆出總數) ≥ 70% 視為可出報告,< 70% 但已撞時間 / 預算上限就出短版 + 標 `[partial_research]`。永遠不能讓 user 等到「不知道何時結束」 — 任何 task 30 分鐘鐵停,沒答完就誠實出短版。

2. **「Reproducibility 要求同題兩次 ±10%,但 web search 結果每天都在變,怎麼做?」** — 誠實答:真正可重現的是 **plan 與 reasoning 路徑**,不是 evidence 集合。對策:(a) Planner / critic / writer 的 LLM call 全部 temperature=0 + seed pin;(b) Search result 在 task 內 cache(同 query 1h 內走 Redis,確保並行 researcher 看到同 snapshot);(c) 跨 task 的 reproducibility 用 **同 evidence set 重跑** mode(replay 模式,給 audit / eval 用);(d) Eval metric 不要苛求 evidence 一致,而是看 conclusion 一致(BERTScore on report summary 段);承認真實 web 變動下做不到 100% byte 一致,90% 語意一致已是業界天花板。

3. **「multi-agent 通訊靠 LangGraph state passing,500 concurrent task × 8 agent × 10 step,Postgres checkpoint 寫入會不會爆?」** — 答:會,所以做兩層拆。(a) **熱 state 不寫 Postgres**:LangGraph SqliteSaver in-memory 跑 in-flight task,只在 milestone(planner 完成、每個 sub_research 完成、writer 完成)寫一次 Postgres;(b) **Postgres 用 Citus / Aurora 水平擴展**:per-task 一個 partition key,500 concurrent 寫入分散到 32 shards,寫入壓力可控;(c) Trace span 全量寫 ClickHouse(列存,寫入扛億級 events/day),Langfuse backend 也跟著切;(d) Hot path latency budget:checkpoint write < 50ms,超過就丟到 async queue 後寫(接受 resume 時可能損失最後一步,重新跑該 step 即可)。粗估:500 concurrent × 平均 10 milestone × 8min task = 10/s checkpoint write per task × 500 = 5K writes/s peak,Aurora 32-shard 撐得住。

---

> 後續案例 Case_05 待補:
> - Case 05:Computer-Use SaaS(sandbox、安全、replay)

返回:[`./README.md`](./README.md) | [`../1.LLM面試題庫/04_系統設計題.md`](../1.LLM面試題庫/04_系統設計題.md)
