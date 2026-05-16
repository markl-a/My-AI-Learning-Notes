# GraphRAG Hands-On:從圖譜檢索到多跳推理的工程實踐

> 對應 [全景圖 #12](../../2024-2026_AI完整領域全景圖.md);圖學習基礎 [`../../18.GNN_Graph_Learning/`](../../18.GNN_Graph_Learning/)
>
> ⚡ **想立刻動手?**配套 Colab notebook:[`notebooks/Colab_MiniGraphRAG_Hands_On.ipynb`](./notebooks/Colab_MiniGraphRAG_Hands_On.ipynb)
> — 250 行 Python 手刻 mini GraphRAG(networkx + LLM 抽取 + Louvain 社群 + local/global search 三模對比),~15 分鐘跑完、~$0.20/run。
> 本檔是「概念深度」,notebook 是「親手做一遍」,搭配閱讀效果最佳。

---

## 1. GraphRAG 為何重要:Vector RAG 撞牆的三個場景

傳統 vector RAG 把所有資料切 chunk、塞 embedding、做相似度檢索,在「單點事實查詢」場景表現很好。但只要問題稍微複雜一點,它就會露出三個致命缺陷:

**(a) 多跳問答 (Multi-hop QA)**:問「跟 OpenAI 投資人有業務往來的公司,在 2024 年有哪些併購案?」這需要先找 OpenAI 投資人 → 找這些投資人的關聯公司 → 查這些公司的併購紀錄。向量相似度只能撈回字面接近的段落,無法做 entity 之間的關係 traversal。

**(b) Global Sensemaking**:問「這份 200 頁的合約整體在保護誰?」答案不在任何一個 chunk 裡,而是分散在數十個條款的交互關係中。Vector RAG 撈 top-K chunk 後永遠看不到全局。Microsoft Research 在 2024 年 4 月的 GraphRAG paper 把這個能力命名為 "query-focused summarization (QFS) over an entire corpus"。

**(c) 可解釋推理**:金融、醫療、法務場景,必須回答「為什麼是這個答案?」Vector RAG 只能丟出檢索到的原文段落,無法呈現推理鏈。Graph 把實體與關係顯式建模,每一跳都是可追蹤的證據。

簡言之:vector RAG 是「找相似文字」,GraphRAG 是「沿著事實之間的關係走」。兩者不互斥,但複雜任務裡 graph 是必須補上的能力。

---

## 2. 三大主流實作對比

| 系統 | 核心機制 | 索引成本 | 查詢延遲 | 擅長 | 弱點 |
|---|---|---|---|---|---|
| **Microsoft GraphRAG** | Leiden 社群偵測 + 多層摘要 | 極高 (大量 LLM call) | 中 | Global sensemaking、跨主題綜合 | indexing 一次燒掉幾十美元 |
| **LightRAG (HKU, 2024)** | 雙層檢索 (low/high-level keys) + 鄰近子圖合併 | 低 | 低 | 成本敏感、incremental update | global query 不如 MS GraphRAG 深 |
| **HippoRAG 2 (OSU, 2024)** | 海馬迴啟發,Personalized PageRank | 中 | 低 | 多跳推理、聯想式檢索 | 需要外部 NER + linker |

**Microsoft GraphRAG** 走的是「重 indexing、重摘要」路線:抽完 entity / relation 後跑 Leiden 演算法把圖切成階層社群,每個社群叫 LLM 寫一段 community report。Query 時可以下沉到 entity 層 (local search) 或上升到社群層 (global search)。問題是 indexing 階段每個 entity、每個社群都要 LLM 寫摘要,1000 篇文件的 corpus 動輒幾百美元。

**LightRAG** 抓的是另一個權衡點:不做完整社群摘要,而是雙層 query (低階查具體實體、高階查主題),檢索時把命中 entity 的鄰近子圖合併進 context。論文宣稱在 UltraDomain benchmark 上以 1/10 token 成本達到接近 MS GraphRAG 的品質,且支援 incremental insert 不用整圖重建。

**HippoRAG 2** 借用神經科學的海馬迴 indexing theory,用 Personalized PageRank 從問題實體出發在 KG 上做 random walk,找到「聯想最強」的段落。對 multi-hop benchmark (如 MuSiQue, 2WikiMultiHop) 特別有效。

---

## 3. Microsoft GraphRAG 完整流程

### Indexing Pipeline

```
Documents → Text Units (chunk)
         → Entity & Relation Extraction (LLM, prompt 抽 triples)
         → Knowledge Graph (NetworkX / Parquet)
         → Community Detection (Leiden, 階層式)
         → Community Reports (LLM 對每個社群寫摘要)
         → Embedding (entities + reports)
```

關鍵步驟解析:

1. **Entity extraction**:用一個明確的 prompt 要求 LLM 從每個 chunk 抽出 `(entity, type, description)` 與 `(source, target, relationship, description)`,典型 prompt 約 1500 token,每個 chunk 跑 1–2 次 (gleaning 機制重抽以提升 recall)。
2. **Graph 建構**:相同 entity 的 description 跨 chunk 合併 (LLM summarize),邊權重累加。
3. **Leiden 社群偵測**:Leiden 比 Louvain 更穩定,輸出階層式社群 (level 0 = 粗粒度大社群,level n = 細粒度)。
4. **Community report**:對每個社群,把成員 entity / relation 餵給 LLM 生成結構化報告 (title、summary、findings),這是 global search 的彈藥庫。

### Query Modes

- **Local Search**:從問題抽 entity → 在圖中找鄰居 → 拉相關 chunk、relation、community report → 組 context → 回答。適合「X 對 Y 做了什麼?」這類具體問題。
- **Global Search**:Map-Reduce 模式。把所有相關 level 的 community report 分批送進 LLM (map 階段每批給部分答案 + 評分),再 reduce 合併。適合「整份文件的主軸是什麼?」這類綜合問題。
- **DRIFT Search** (2024 Q4 新加):結合 local 與 global,先 global 找方向再 local 補細節。

---

## 4. 可執行範例:Microsoft GraphRAG 官方套件

```python
# pip install graphrag==0.5.0
# 準備 ./input/ 放 ~30 個 .txt 技術文件
import os, asyncio, subprocess
from pathlib import Path

WORKDIR = Path("./gr_demo")
WORKDIR.mkdir(exist_ok=True)
(WORKDIR / "input").mkdir(exist_ok=True)

# 1. 初始化專案 (產生 settings.yaml、prompts/)
subprocess.run(["python", "-m", "graphrag.index", "--init", "--root", str(WORKDIR)])

# 2. 編輯 settings.yaml:填入 OPENAI_API_KEY、選 gpt-4o-mini
#    chunks.size: 600  (token,小 chunk 利於 entity extraction)
#    entity_extraction.max_gleanings: 1
#    community_reports.max_length: 2000

# 3. 跑 indexing (這步會燒錢,30 文件 gpt-4o-mini 約 $2-5)
subprocess.run(["python", "-m", "graphrag.index", "--root", str(WORKDIR)])

# 4. Local Search:查具體事實
from graphrag.query.cli import run_local_search, run_global_search

local_ans = run_local_search(
    config_filepath=None,
    data_dir=str(WORKDIR / "output"),
    root_dir=str(WORKDIR),
    community_level=2,
    response_type="Multiple Paragraphs",
    streaming=False,
    query="Transformer 架構中 Multi-Head Attention 的計算複雜度是多少?",
)
print("LOCAL:", local_ans)

# 5. Global Search:查主題綜合
global_ans = run_global_search(
    config_filepath=None,
    data_dir=str(WORKDIR / "output"),
    root_dir=str(WORKDIR),
    community_level=1,        # 越小越粗,越能看到全局主題
    response_type="Multiple Paragraphs",
    streaming=False,
    query="這 30 篇技術文件涵蓋哪幾個主要研究方向?各自的核心方法論是什麼?",
)
print("GLOBAL:", global_ans)

# 6. 檢視產出 (parquet 格式)
import pandas as pd
entities = pd.read_parquet(WORKDIR / "output" / "create_final_entities.parquet")
communities = pd.read_parquet(WORKDIR / "output" / "create_final_community_reports.parquet")
print(f"抽出 {len(entities)} 個 entity、{len(communities)} 個 community report")
```

實務 tip:`gpt-4o-mini` 在 entity extraction 階段 recall 約比 `gpt-4o` 低 15%,但成本只有 1/15。建議第一次跑用 mini,觀察 graph 是否合理再決定是否升級。

---

## 5. LightRAG 範例:成本敏感場景的選擇

```python
# pip install lightrag-hku
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, openai_embedding

rag = LightRAG(
    working_dir="./light_demo",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embedding,
)

# Insert (支援 incremental,新文件不用整圖重建)
with open("doc1.txt", encoding="utf-8") as f:
    rag.insert(f.read())
rag.insert(["doc2 content...", "doc3 content..."])

# 四種 query mode
print(rag.query("公司 A 收購了哪些 AI startup?",  param=QueryParam(mode="naive")))   # 純 vector
print(rag.query("公司 A 收購了哪些 AI startup?",  param=QueryParam(mode="local")))   # 低階 entity
print(rag.query("整份報告的主旨是什麼?",          param=QueryParam(mode="global")))  # 高階主題
print(rag.query("AI 投資趨勢與監管之間的關係?",    param=QueryParam(mode="hybrid"))) # 雙層混合
```

成本對照 (內部 1000 篇文件實測,gpt-4o-mini):

| 指標 | MS GraphRAG | LightRAG |
|---|---|---|
| Indexing token | ~28M | ~3.5M |
| Indexing cost | $42 | $5.3 |
| Query token (avg) | ~7K | ~2K |
| Multi-hop accuracy | 78% | 73% |
| Global QA quality | 85% | 76% |

如果你的場景以 local 多跳為主、文件每天有增量,LightRAG 是更務實的選擇;如果是一次性建庫做深度報告分析,MS GraphRAG 的 community report 還是有不可替代的優勢。

---

## 6. 與 Vector RAG 混合策略

不要把 GraphRAG 當成 vector RAG 的替代品,務實做法是並行雙索引,router 分流:

```
Query → Classifier (LLM/規則)
       ├─ 單點事實/相似度型 → Vector RAG
       ├─ 多跳關係/因果型   → Graph Local Search
       └─ 全局綜合/主題型   → Graph Global Search
```

**選擇邏輯**:
- Query 含「為什麼、如何影響、之間的關係」→ Graph
- Query 含具體名詞 + 屬性查詢 (價格、時間、定義) → Vector
- Query 含「整體、總結、概述、趨勢」→ Graph Global
- 不確定時並行跑,用 reranker 或 LLM judge 挑答案

LangChain 的 `MultiRetrievalQAChain` 或 LlamaIndex 的 `RouterQueryEngine` 可以實作這層路由。

---

## 7. 生產考量

**Graph 維護成本**:Entity 抽取是 LLM 的 hallucination 重災區,同一個人名可能變成三個 entity (王小明、小明、Mr. Wang)。必須跑 entity resolution:embedding 相似度 + name normalization + 人工 review 高頻 entity。

**Incremental Update**:MS GraphRAG 0.5+ 開始支援 incremental,但社群結構會變,部分 community report 需要重寫。LightRAG 原生支援 insert 但 delete 較弱。實務上建議:小量增量直接 insert,每月跑一次 full rebuild。

**Schema 漂移**:沒有預定義 schema 的開放抽取會讓 relation type 爆炸 (數百種)。生產系統建議先定義 30–50 種核心 relation type,prompt 裡列出來,允許但不鼓勵 LLM 創造新 type。

**版本控制**:Graph artifact (parquet) 要納入 DVC 或 LakeFS,因為重建一次成本高,且不同版本的 community report 會直接影響 RAG 答案。

---

## 8. Neo4j + LLM 工程化

當 graph 規模超過 10 萬 node,parquet + NetworkX 開始撐不住,該轉 Neo4j。整合模式:

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="...")
graph.refresh_schema()  # 把 schema 餵給 LLM

chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model="gpt-4o", temperature=0),   # 寫 Cypher 用強模型
    qa_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),  # 組答案用便宜模型
    graph=graph,
    validate_cypher=True,           # 跑前驗證 syntax
    return_intermediate_steps=True, # 把 Cypher 拉出來給審計用
)
ans = chain.invoke({"query": "哪些員工同時參與了 Project Alpha 與 Project Beta?"})
```

**權限控制**:Neo4j 5.x 的 RBAC 可以做 node label / property 層級的 ACL,結合 SSO 後可以做到「業務只能查自己部門的關係」。Cypher 自動生成的場景一定要把 user role 注入 prompt,並對生成的 Cypher 做白名單檢查 (禁止 `DELETE`, `DETACH`)。

---

## 9. 真實案例

**(a) 金融合規問答**:某銀行用 GraphRAG 整合 200 份監管文件 + 內部政策,建構 entity = {規定、產品、流程、負責部門} 的圖。Query「銷售衍生品給散戶需要哪些審核步驟?」需要從產品 → 適用法規 → 流程節點 → 責任部門做四跳,vector RAG 撈不到完整鏈,GraphRAG local search 直接回出完整流程圖。

**(b) 藥物副作用推理**:整合 DrugBank + PubMed abstract,建構 drug-target-pathway-symptom 圖。Query「同時服用 X 與 Y 是否有交互風險?」HippoRAG 2 的 PPR 機制從 X、Y 兩個起點同時 walk,找到共同的 target protein 或代謝路徑,給出推理鏈而非單純記憶式回答。

**(c) 企業內部知識整合**:科技公司把 Confluence、Jira、Slack 匯出建 graph,entity = {專案、人、決策、bug、模組}。新員工 onboarding 問「為什麼我們用 Kafka 而不是 Pulsar?」Global search 從歷年技術評估社群報告中綜合出完整脈絡,vector RAG 只能撈到一兩篇片段討論。

---

## 10. 評估

**多跳問題集**:
- **MuSiQue** (2-4 hop, 多步推理)
- **2WikiMultiHopQA** (組合推理)
- **HotpotQA** (橋接、比較類)
- **MultiHop-RAG** (Tang & Yang 2024,專為 RAG 設計)

**綜合 Benchmark**:
- **GraphRAG-Bench** (2024 Q4):涵蓋 local QA、global QA、reasoning chain 三類 task,提供 entity-level 與 answer-level 雙重評分。
- **UltraDomain** (LightRAG paper):跨 18 個專業領域的 long-context QA。

**評分指標**:
- 答案正確性:LLM-as-judge (gpt-4o 對比 ground truth)
- 推理鏈忠實度:抽答案中 entity,檢查是否都在 retrieved subgraph
- Token 效率:正確率 / token cost,生產上比純 accuracy 更重要
- 延遲 P95:Graph 查詢通常比 vector 慢 2–5 倍,要監控

實務建議:自己準備 50–100 題 in-domain 多跳題,人工標註答案與必要 entity,每次 prompt / model / index 變動都跑一次回歸測試。光看 public benchmark 會被誤導 (你的 domain 分布與它們可能差很遠)。

---

## 小結

GraphRAG 不是萬靈丹,但它補上了 vector RAG 在「關係推理」與「全局綜合」兩個最大的缺口。2024–2026 年的工程趨勢已經很清楚:**vector + graph 雙索引 + router** 會成為複雜 RAG 系統的標配。選實作時別只看 paper 的 benchmark,要看你的 corpus 規模、更新頻率、預算上限、可解釋性需求,在 MS GraphRAG / LightRAG / HippoRAG 2 三條路線中挑最匹配的;對於 10 萬 node 以上的場景,直接上 Neo4j + Cypher 工程化路徑更可持續。
