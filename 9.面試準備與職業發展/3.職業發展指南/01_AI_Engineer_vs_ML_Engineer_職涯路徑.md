# 01 AI Engineer vs ML Engineer 職涯路徑

> **本檔定位**:幫你在 2026 年的 AI 人力市場上選對職位。AI Engineer 和 ML Engineer
> 不是同一條路,薪資、必備技能、日常工作差異甚大。
>
> **Cross-link**:
> - 廣度地圖 → [`../../2024-2026_AI完整領域全景圖.md`](../../2024-2026_AI完整領域全景圖.md)
> - 技術題庫 → [`../1.LLM面試題庫/`](../1.LLM面試題庫/)
> - 熱詞索引 → [`../../FRONTIER_TERMS_INDEX.md`](../../FRONTIER_TERMS_INDEX.md)
> - phantom-mesh 在哪 fit → 本檔 §5「項目經驗 vs LeetCode」

---

## 1. 兩條主路徑的本質差異

| 維度 | **AI Engineer**(LLM 應用工程師) | **ML Engineer** |
|---|---|---|
| 主要工作 | 整合 GPT / Claude / Gemini API、組裝 RAG、做 Agent 編排、評估 prompt | 訓練 / 微調模型、設計訓練 pipeline、MLOps、特徵工程、上線推論服務 |
| 一週 80% 時間在 | 寫 application code、設計 prompt、串向量庫、做 eval、優化延遲與成本 | 跑訓練 job、看 loss 曲線、改 data pipeline、調 hyperparameter、優化 GPU 利用率 |
| 工具棧 | LangChain / LlamaIndex / Pydantic AI / OpenAI SDK / Anthropic SDK / pgvector / Qdrant | PyTorch / JAX / HuggingFace Transformers / Ray / Kubeflow / MLflow / Weights & Biases |
| 數學門檻 | 概念懂即可(知道 attention / cosine sim 在算什麼) | 中–高(需要看懂 paper、自己推導 loss、debug 梯度) |
| 必備經驗 | 至少做過一個 end-to-end LLM 產品(有人在用) | 至少訓練或微調過一個模型,知道怎麼從 data → 部署 |
| 2025 年職缺成長 | **+25.2%**(LinkedIn Emerging Jobs Report 2025) | **+41.8%**(成長最快,2025 美國科技人才報告) |
| 中位數年薪(美國) | **~$149K** | **~$183K**(高 ~23%) |
| 入門年資 | 2–3 年軟體工程經驗即可轉 | 通常需要 ML / DS 背景或研究所訓練 |

**白話總結**:
- **AI Engineer = 軟體工程師的延伸**,主戰場是「把 LLM 變成可用產品」。
- **ML Engineer = 資料科學家的延伸**,主戰場是「讓模型本身變更好」。
- 兩者交集越來越大(SLM 微調、自架推論)、但薪資結構與面試重點仍有別。

---

## 2. 2026 年快速崛起的新興職位

這幾個職位在 2024 年大多還不存在或只是 informal,2026 已開始出現在正式 JD 上。

### 2.1 Context Engineer
- **做什麼**:設計 / 維護 agent 的「context window 餵食策略」 — 哪些 doc 該進 prompt、哪些該放 long-term memory、哪些該 summarise、哪些該 retrieve on demand。
- **為什麼出現**:context window 雖然到 1M-10M tokens,但 cost、latency、attention dilution 仍是 bottleneck。
- **典型公司**:Cognition、Cursor、Lindy、Bolt、各大 agent startup。
- **薪資**:senior 級對齊 staff SWE,矽谷 ~$220K–$320K base + equity。

### 2.2 Forward Deployed Engineer (FDE)
- **做什麼**:駐點客戶端做 customization、寫 integration、把 generic foundation 接到客戶 messy reality。Palantir 模式被 OpenAI / Anthropic / Scale 大量複製。
- **為什麼出現**:enterprise LLM 落地需要既懂技術、又懂商業流程的人。
- **2026 行情**:全職遠端 + 出差,base ~$180K–$280K,bonus / equity 占大宗。

### 2.3 Agent Engineer
- **做什麼**:設計 multi-agent 編排、tool calling schema、agent eval harness、長 horizon task 的 checkpoint 與 recovery。
- **必備**:熟 A2A / MCP / OpenAI Agents SDK / LangGraph / Claude Agent SDK。
- **行情**:對齊 senior LLM eng,~$200K–$300K。

### 2.4 Skill / Plugin Author
- **做什麼**:為 Claude Skills、ChatGPT Apps、Gemini Extensions、Cursor commands 寫「可組合的 capability 模組」。獨立開發者或產品團隊新增的 role。
- **變現**:Skills marketplace、企業 SaaS 加值、open source 知名度。

### 2.5 Eval Engineer
- **做什麼**:設計 LLM 評估資料集、自動化 eval pipeline、追蹤 model regression。
- **為什麼重要**:沒 eval 就無法 A/B test prompt、無法升級 model。FAANG / OpenAI / Anthropic 都在大量招。

---

## 3. 背景轉職路徑(最短路)

### 3.1 軟體工程師 → AI Engineer(**最短路,3–6 個月**)
1. 月 1:打 LLM API 基礎,做 1 個 chatbot demo,熟 streaming SSE、function calling。
2. 月 2:做 1 個 end-to-end RAG(自己的 PDF / 自家公司資料),用 pgvector 或 Qdrant。
3. 月 3–4:做 1 個 agent 專案(瀏覽器自動化、code agent、deep research clone),熟 LangGraph / OpenAI Agents SDK。
4. 月 5:寫成 case study,投產 + 開源 + 在 LinkedIn / X 發 3 篇文章。
5. 月 6:開始投履歷。

**關鍵**:不需要學數學公式、不需要刷 LeetCode hard,但要 **把整套 stack 親手跑過一次**。

### 3.2 資料科學家 / 分析師 → ML Engineer(**自然進階,6–12 個月**)
1. 補軟體工程:Git、CI/CD、Docker、Kubernetes、寫 production-grade Python。
2. 補 MLOps:MLflow / W&B / Ray Train / vLLM / TensorRT-LLM。
3. 補 distributed training:DeepSpeed / FSDP / Megatron 至少跑通一個。
4. 微調 1–2 個開源模型(Qwen / Llama / Mistral)並上線 vLLM。
5. 重寫履歷:把「分析」改寫成「pipeline / 系統」。

### 3.3 數學 / 物理 / 統計背景 → 研究方向(Research Engineer / Scientist)
- 直接念碩博、或進工業實驗室 residency(OpenAI、Anthropic、DeepMind、Meta FAIR、NVIDIA Research、各大學 lab 都有)。
- 必備:讀 paper 速度、復現實驗能力、開源貢獻 / preprint。

### 3.4 前端 / 行動端工程師 → Generative UI / Agent UI 工程師
- 銜接 [`../../20.Generative_UI/`](../../20.Generative_UI/) 全章。
- 熱詞:A2UI、AG-UI、tool-rendered components、streaming UI。
- 行情:senior front-end + AI premium,~$170K–$240K。

---

## 4. 薪資數字參考(2026,僅供方向感)

> 數字會隨景氣浮動,以下取自 levels.fyi、CakeResume、104、Glassdoor、Hired 報告交叉比對。
> 「同職等同 location」是比較基準,跨公司差異仍大。

### 矽谷(USD,total comp:base + bonus + equity per year)

| 職等 | AI Engineer | ML Engineer | Research Scientist |
|---|---|---|---|
| Mid (L4 / SDE II) | $200K–$300K | $230K–$340K | $250K–$380K |
| Senior (L5) | $320K–$480K | $360K–$520K | $400K–$650K |
| Staff (L6) | $480K–$700K | $520K–$780K | $700K–$1.2M |

OpenAI / Anthropic / 頂級 startup 的 staff+ 級可破 $1M(equity 占大宗)。

### 台灣(TWD,annual,base + bonus,大公司 / unicorn)

| 職等 | AI Engineer | ML Engineer |
|---|---|---|
| Mid (3–5 年) | 120 萬 – 180 萬 | 130 萬 – 200 萬 |
| Senior (5–8 年) | 180 萬 – 280 萬 | 200 萬 – 320 萬 |
| Staff (8+ 年) | 280 萬 – 450 萬 | 300 萬 – 500 萬 |

外商台灣分公司(Google、AWS、NVIDIA、各大半導體 AI team)可再 +20–40%。
赴美 H-1B / O-1 / L-1 是另一個跳躍。

---

## 5. 履歷重點:什麼贏什麼(2026 現場觀察)

```
GitHub repo (可重現、有 README、有 demo)     >>  LinkedIn 列證書
End-to-end case study(blog + repo + demo) >>  Coursera / Udemy 結業
phantom-mesh 級的多人協作開源              >>  個人 toy project
解過真實 bug 的 PR (即使是別人的 repo)      >>  自己造 wheel
量化成果(latency 從 X → Y,cost -Z%)       >>  「我用 LangChain 做了一個 chatbot」
LeetCode hard 100 題                       <  寫過一個被 100+ star 的 repo
```

**面試官最想看的三個訊號**:
1. **你能 ship**(不是 demo,是有人在用、有監控、有錯誤處理的東西)。
2. **你會 debug**(看你怎麼描述「最近一次最難 debug 的問題」)。
3. **你跟得上**(熟 2025-2026 的新詞:RLVR、GRPO、A2A、MCP、context engineering、SGLang、長思考 reasoning)。

---

## 6. 不要做的事

- **不要等「準備好」才開始投履歷**:準備永遠不夠,先投 5 間摸底。
- **不要只刷題不做專案**:LLM 領域題目改太快,專案更有複利。
- **不要追新框架追到忘記基本功**:transformer 數學 + Python 系統設計仍是地基。
- **不要忽略「為什麼這個技術」**:面試官最會追問 trade-off,不會 trade-off 等於不會技術。
- **不要在履歷寫「精通」**:寫「有 X 年 production 經驗」+ 具體成果。

---

## 7. 下一步

- 履歷怎麼寫 → [`02_履歷與作品集打造.md`](./02_履歷與作品集打造.md)
- 行為面試怎麼答 → [`03_行為面試_STAR_範例.md`](./03_行為面試_STAR_範例.md)
- 長線怎麼跟進 → [`04_AI_工程師持續學習指南.md`](./04_AI_工程師持續學習指南.md)
- 你還不知道自己想做哪個方向 → 回看全景圖 [`../../2024-2026_AI完整領域全景圖.md`](../../2024-2026_AI完整領域全景圖.md) Part 5「個人學習路徑」三條對照。
