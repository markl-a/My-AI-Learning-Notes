# ColPali、Late Chunking 與 Contextual Retrieval:2024-2025 RAG 三大躍進

> 對應 [全景圖 #12](../../2024-2026_AI完整領域全景圖.md);搭配 [`./GraphRAG_hands_on.md`](./GraphRAG_hands_on.md)

## 1. 三大 2024-2025 RAG 創新總覽

傳統 RAG 在 2023 年定型後,2024 年下半到 2025 年迎來三條獨立但互補的突破路線,各自針對不同的痛點:

| 創新 | 發表 | 解決的痛點 | 核心觀念 |
|------|------|-----------|---------|
| **ColPali** | Illuin/HuggingFace 2024-06 | PDF 圖文混排、OCR 失真、表格/圖表 | 把整頁當圖像,patch-level late interaction |
| **Late Chunking** | Jina AI 2024-09 | chunk 邊界切斷上下文、跨段指代消解失敗 | 先整篇 embed,再 pool 切塊 |
| **Contextual Retrieval** | Anthropic 2024-09 | chunk 失去原文角色、召回率天花板 | LLM 為每 chunk 寫一句「在原文中是什麼」 |

三者並非互斥。實務上,**ColPali 解決「資料進不來」**(視覺內容)、**Late Chunking 解決「邊界切壞」**(語意連續性)、**Contextual Retrieval 解決「chunk 太碎沒語境」**(召回品質),可疊加使用。

---

## 2. ColPali 原理:PaliGemma + ColBERT 的視覺 late interaction

傳統文件 RAG 的 pipeline 是 `PDF → OCR → layout 解析 → 文字 chunk → embedding`,每一階段都會掉資訊,尤其表格、流程圖、手寫註記、複雜版面幾乎必死。ColPali 直接砍掉前三步:

1. **PDF 每頁渲染成圖像**(例如 448×448 或 768×768)
2. 用 **PaliGemma**(Google 的 VLM,SigLIP vision encoder + Gemma LLM)把整頁編碼成 ~1024 個 patch token,每個 token 是 128 維向量
3. Query 用文字 tokenizer 編碼成 ~20 個 token,也是 128 維
4. 比對時用 **ColBERT-style late interaction**:每個 query token 對所有 page patch token 做 MaxSim,再加總

關鍵差異在「late interaction」:不像 bi-encoder 把整頁壓成單一向量(資訊損失大),也不像 cross-encoder 每對 query-doc 都重算(慢)。ColPali 預先算好 page patch 向量,查詢時只做矩陣乘 MaxSim,**保留 patch 級粒度**——也就是能定位到「答案在頁面右下表格」。

在 ViDoRe benchmark(視覺文件檢索)上,ColPali 比 OCR+BM25+ColBERT 的傳統 pipeline 高出 15-20 個 nDCG@5 點,**且不需要 OCR、不需要 layout model、不需要 table extractor**。

---

## 3. ColPali 範例 code:50 行處理 100 頁財報

```python
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
import torch

# 1. 載入模型(約 3GB,fp16 下可放 16GB GPU)
model = ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
).eval()
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

# 2. PDF 100 頁轉成圖像
pages = convert_from_path("tsmc_2024_annual_report.pdf", dpi=150)
print(f"共 {len(pages)} 頁")

# 3. 批次編碼頁面(每頁 ~1024 個 128-d 向量)
page_embeddings = []
with torch.no_grad():
    for i in range(0, len(pages), 4):  # batch=4
        batch = processor.process_images(pages[i:i+4]).to(model.device)
        embs = model(**batch)           # [B, n_patches, 128]
        page_embeddings.extend(list(embs.cpu()))

# 4. 編碼查詢
queries = [
    "2024 年第三季 N3 製程營收占比是多少?",
    "資本支出在先進封裝的分配比例?",
    "美國亞利桑那廠的折舊時程表"
]
with torch.no_grad():
    q_batch = processor.process_queries(queries).to(model.device)
    q_emb = model(**q_batch).cpu()       # [Q, n_qtok, 128]

# 5. Late interaction scoring(MaxSim)
scores = processor.score(q_emb, page_embeddings)  # [Q, N_pages]
for q_idx, q in enumerate(queries):
    top3 = scores[q_idx].topk(3)
    print(f"\n{q}")
    for s, p in zip(top3.values, top3.indices):
        print(f"  p.{p.item()+1}  score={s.item():.2f}")
```

100 頁財報在單張 A100 上索引約 90 秒,查詢延遲約 80 ms。把 top-3 圖頁丟給 Claude/GPT-4o 做 VQA 即得最終答案。

---

## 4. ColQwen2 / ColQwen2.5:2025 多語強化

PaliGemma 主要在英法資料訓練,中文、日文、韓文表現偏弱。2024 年底社群把 backbone 換成 **Qwen2-VL**,推出 **ColQwen2**;2025 年第一季再升級到 **ColQwen2.5**(基於 Qwen2.5-VL-3B/7B)。

升級重點:
- **多語視覺 OCR 內建**:Qwen2.5-VL 對中日韓文字、阿拉伯文、印地文的識別力顯著好過 PaliGemma
- **動態解析度**:不固定 448px,可吃 1280×1280 大圖,圖表細節保留更多
- **長文件 stride**:單次 forward 可吞 4-8 頁拼貼,索引成本降低
- ViDoRe v2 上 ColQwen2.5-7B 約 87 nDCG@5(ColPali-v1.3 約 81)

中文場景(法律、財報、政府文件)建議直接從 **`vidore/colqwen2.5-v0.2`** 起手,把 model 字串替換即可,API 介面相容。

---

## 5. Late Chunking 原理:Jina 2024 的「先 embed 後切塊」

傳統 chunking 流程:`文本 → 切成 512 token 小塊 → 各自獨立 embed`。問題:第 N 個 chunk 裡的「他」「該公司」「上述條款」失去指代對象,embedding 空間中變得模糊。

Late Chunking 反過來:

1. 用 **8K-context embedder**(jina-embeddings-v3、nomic-embed-text-v1.5、bge-m3 都支援)把整篇文章一次 forward
2. 拿到 token-level embeddings,**整篇共享同一份注意力上下文**
3. **再按 chunk 邊界 mean-pool**,得到每個 chunk 的向量

關鍵直覺:Transformer 的 self-attention 已經把「上下文資訊」灌進每個 token embedding,事後 pool 仍保留這些跨段語意。代價只是「整篇要一次塞進 context window」,對 8K embedder 來說 ~6000 字中文文章綽綽有餘,更長則切成 8K 大段再分別 late chunk。

```python
from transformers import AutoModel, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).cuda()

text = open("contract.txt", encoding="utf-8").read()
inputs = tok(text, return_tensors="pt", max_length=8192, truncation=True).to("cuda")
with torch.no_grad():
    token_embs = model(**inputs).last_hidden_state[0]   # [seq, 1024]

# 依句號切 chunk,記錄每個 chunk 的 token span
spans = [(0,128),(128,260),(260,400), ...]  # 由 tokenizer offset_mapping 算出
chunk_embs = [token_embs[s:e].mean(dim=0) for s,e in spans]
```

---

## 6. Late Chunking vs 傳統 chunking 對比實驗

Jina 官方在 LongEmbed / NarrativeQA / SciFact 上做 A/B:

| 資料集 | 傳統 chunking nDCG@10 | Late Chunking nDCG@10 | 提升 |
|--------|-----------------------|----------------------|------|
| NarrativeQA(故事問答,代名詞密集)| 0.413 | 0.491 | **+18.9%** |
| SciFact(科學論文)| 0.652 | 0.701 | +7.5% |
| LongEmbed-QMSum(會議紀錄)| 0.298 | 0.367 | +23.2% |

愈是「指代密集」「跨段語意」場景增益愈大。Late Chunking 的計算成本與傳統 chunking 幾乎相同(forward 次數不變,只是改成單次長 forward),屬於免費午餐型優化。

---

## 7. Anthropic Contextual Retrieval:給每個 chunk 一句「身世」

Anthropic 2024-09-19 公布的方案,**在 chunk embed 之前**,用便宜 LLM(Claude Haiku、GPT-4o-mini)生成 50-100 token 的脈絡前綴:

```
prompt:
<document>{整篇文件}</document>
這是要被檢索的小段:
<chunk>{chunk_text}</chunk>
請寫一句話描述這個 chunk 在原文中扮演什麼角色,以便檢索系統理解,只回那一句。
```

得到例如:「以下段落來自台積電 2024 年 Q3 法說會,描述 N3 製程營收占比及未來 N2 量產時程。」**前綴後再 embed**,等於把「原文中的位置與角色」灌進向量。

**成本控制靠 prompt cache**:整篇文件作為 prefix 快取(Claude 9 折、5 分鐘 TTL),每個 chunk 只付增量 token,實測處理 1M token 文件約 USD 1.02(Haiku 報價)。

**完整 stack**:Contextual Embedding + Contextual BM25 + Voyage/Cohere rerank,Anthropic 報告在自家評測集把「top-20 chunk 未召回率」從 5.7% 降到 1.9%,**召回失敗減少 67%**。

---

## 8. 三者組合策略

不必三個都上,按資料特性挑:

- **資料是 PDF / 投影片 / 掃描件,含大量表格圖表** → 直接 ColPali / ColQwen2.5,跳過 OCR
- **資料是長純文字(法條、論文、技術手冊),指代密集** → Late Chunking + 一般 dense retriever
- **資料量不大但每筆價值高(知識庫、FAQ、合約條款庫)** → Contextual Retrieval + Hybrid + Rerank
- **混合資料源** → 分艙(per-corpus 各跑各的),最後在 reranker 階段統一打分

可疊加路徑:**ColPali 召回頁面 → OCR 該頁 → Late Chunking 切段 → Contextual 前綴 → rerank**,適合超高品質法律/醫療場景。

---

## 9. 真實 case

**法律合約問答(ColPali)**:某律所有 12 萬頁掃描合約 PDF,版面複雜含手簽。傳統 OCR pipeline 表格識別率約 62%。改用 ColQwen2.5 後直接圖像檢索,律師查「賠償上限」「終止條款」top-5 命中率從 71% 升到 93%,索引一次性成本約 6 小時 GPU。

**學術論文 RAG(Late Chunking)**:某大學圖書館建 5 萬篇論文助理,純文字但跨段指代多(「該方法」「上述模型」)。把 chunk 從獨立 embed 改成 jina-v3 Late Chunking,RAGAS context_recall 從 0.68 升到 0.81,改造只動 indexing pipeline,不動下游 LLM。

**客服知識庫(Contextual Retrieval)**:電商 8000 條 FAQ + 政策文件,每條都關鍵。導入 Anthropic 方案後客服 bot 一次答對率 +14 pp,Haiku 預處理一次性成本約 USD 230,後續維護幾乎零成本。

---

## 10. Eval:三個必跑基準

- **ViDoRe**(Visual Document Retrieval Benchmark):ColPali/ColQwen2 系列必跑,涵蓋金融、學術、產業文件,中英法文混合。看 nDCG@5。
- **CRAG**(Meta KDD Cup 2024,Comprehensive RAG):涵蓋 5 領域、8 種問題類型(simple、conditional、multi-hop、aggregation、…),測 end-to-end RAG 質量,看 truthful score(對 - 錯 - 幻覺)。
- **RAGAS** 三指標:`faithfulness`(答案是否忠於檢索內容)、`context_precision`、`context_recall`。Late Chunking 改善 recall,Contextual Retrieval 改善 precision,ColPali 改善 recall(資料進得來)。

評估時必須**固定 LLM、固定 prompt,只換 retrieval 元件**,才能歸因到 ColPali / Late Chunking / Contextual 的真實貢獻,而不是被生成端的雜訊蓋掉。
