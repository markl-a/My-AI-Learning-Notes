# 18. GNN / Graph Learning / 知識圖譜 (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #10
> 圖學習在 LLM 浪潮中沒有消失,反而透過 GraphRAG、Graph Foundation Models、等變網路找到了「不可替代位置」。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **PyTorch + 線代基礎**(對應 repo:[1.從AI到LLM基礎/4.DL](../1.從AI到LLM基礎/4.DL/README.md)、[1.從AI到LLM基礎/1.Math_4_ML](../1.從AI到LLM基礎/1.Math_4_ML/README.md))
> 2. **知道鄰接矩陣 / 圖的基本表示**(線代或離散數學課本級即可)
> 3. **Transformer 基礎**(對應 repo:[2.深入LLM模型工程與LLM運維/1.LLM 基礎與架構](../2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/README.md) — Graph Transformer 沿用 attention)
>
> 完全沒接觸 GNN 的 message passing,先讀 [全景圖 #10](../2024-2026_AI完整領域全景圖.md) 對應章節 + [Stanford CS224W](https://web.stanford.edu/class/cs224w/) 前兩講。
>
> **延伸 / 反向連結**:[12.AI_For_Science](../12.AI_For_Science/README.md)(分子 / 蛋白質皆為等變圖任務) | [3.LLM應用工程/5.進階 RAG 與多元資料檢索](<../3.LLM應用工程/5.進階 RAG 與多元資料檢索/>)(GraphRAG)

---

## 1. GNN 經典與現況

四大經典架構在 2026 年仍是工業界主力 baseline:
- **GCN** — 靜態、小型、可放入記憶體的圖,極難擊敗的簡單基線
- **GraphSAGE** — 鄰居採樣 + inductive,大規模圖(Pinterest、Uber、LinkedIn)的生產首選
- **GAT** — 注意力機制,所有後續 Graph Transformer 的精神祖先
- **GIN** — 分子分類、圖級任務上理論表達力最強(1-WL 上界)

**何時不用 GNN**:
1. 沒有真正的關係結構(把圖片當圖是次佳解)
2. 表格資料用 XGBoost/CatBoost 通常贏
3. 同質性低、特徵主導的資料,MLP + 手工特徵常常打平
4. 圖太大且邊太密時,sampling 方差大於訊號

**心法**:先跑 MLP + node features,再加 GraphSAGE,最後才考慮 GAT/GIN。

## 2. Graph Transformer

OGB 排行榜常勝軍:
- **GraphGPS** — MPNN + Transformer 並行架構與多種 positional encoding (PE)
- **GRIT** — Relative Random Walk PE (RRWP),ZINC 達 SOTA 而不需 message passing
- **Polynormer** (2024) — local-to-global 線性注意力,可擴展百萬節點
- **TokenGT** — 節點與邊都視為 token,理論 2-WL 表達力

2025 G2LFormer 反轉傳統堆疊(淺層 attention + 深層 GNN)解決 over-globalization。

**選型**:中小型圖(< 100K 節點)用 GraphGPS/GRIT;大規模優先 Polynormer。PE 是成敗關鍵——RRWP、Laplacian eigenvectors、SignNet 必試。

## 3. Graph Foundation Models (GFM, 2024-2026 新興)

試圖打造「圖版 GPT」:
- **OFA (One For All)** — 文本描述把多種圖任務統一,需有監督標籤
- **UniGraph / UniGraph2** — 自監督學習跨文本屬性圖的統一嵌入
- **GraphAny** — 第一個真正歸納式 FM——同一個預訓練模型可在任意特徵維度、任意類別數的圖上做節點分類
- **GOFA** — 結合生成式 in-context learning
- **Google Research relational table GFM (2025)** — 瞄準企業資料庫場景

**核心挑戰**:graph vocabulary——如何把異質節點/邊統一 tokenize。

**選型**:GFM 仍以 node classification 為主戰場,生產上建議用 GraphAny 做冷啟動,微調仍要傳統 GNN/GT。

## 4. 大規模圖學習基礎建設

- **PyG 2.0** (2025) — 最活躍生態,新增異質/時序圖、可擴展 feature store
- **DGL 2.1** — GraphBolt 實現整條資料載入 GPU 加速
- **NVIDIA cuGraph-DGL** — 100B edges 上達優於線性的擴展,2-8 倍 data loading 加速
- **Neo4j GDS 2.5+** — 與 PyG 無縫整合,從 Cypher 直接導出訓練圖
- **GraphScope** (阿里) — 大規模分散式圖計算

**選型**:研究/原型用 PyG;百億邊以上選 DGL + cuGraph;若資料已在 Neo4j,GDS → PyG pipeline 是最短路徑。

## 5. 與 LLM 結合:GraphRAG 系列大爆發

**2024-2026 圖學習最熱的軸線**:
- **Microsoft GraphRAG** (2024) — 社群偵測 (Leiden) + 多層摘要,企業 benchmark 從 baseline RAG 32% 提升到 86%
- **LightRAG** (HKU) — 雙層檢索 + 鄰近子圖合併;論文宣稱在特定資料集相對 Microsoft GraphRAG default 設定有大幅成本降低(實際倍數隨 indexing strategy 與資料集差異甚大,讀者應以實測為準)
- **HippoRAG 2** — 神經生物學啟發,**多跳推理便宜 10-30 倍**
- **PathRAG** — flow-based pruning 減少 44% context
- **GraphGPT / InstructGLM** — graph instruction tuning,把節點當 token 注入 LLM

**Text-Attributed Graphs (TAG)** 已成新預設——節點本身帶文字,LLM 編碼 + GNN 結構訊息融合。

**選型**:深度關聯/敘事問答用 GraphRAG;低成本部署用 LightRAG;多跳事實檢索用 HippoRAG 2;企業有 Neo4j 直接 Cypher + LLM 翻譯最務實。

## 6. 應用主場

### 藥物 / 分子
圖是天然結構,**Chemprop (D-MPNN)** 是工業標配;**EquiformerV2** 在 OC20 force prediction 領先;**MACE** 是 ML force field 主流;**SE(3)-Transformer** 啟發了 EquiFold。

### 推薦系統
**PinSAGE** 在 Pinterest 3B 節點/18B 邊上服役;**LightGCN / LightGCN++** 是學術 SOTA baseline;**UltraGCN** 在多個 benchmark 對 LightGCN 有顯著加速與 NDCG 提升,具體數字依資料集差異甚大(原始論文範圍 NDCG@20 +4% 至 +20%,跨 dataset);TikTok / 阿里 / 美團大量用 GNN 做召回。

### 詐欺 / 反洗錢
銀行業主力——NVIDIA 部落格、Elliptic Bitcoin AML、LineMVGNN、LAS-GNN (2025) 顯示異質 GNN + 時序 motif 偵測是業界主流。Continual graph learning 處理 fraud 模式漂移。

## 7. 知識圖譜現代化

Wikidata、Freebase、DBpedia、ConceptNet 等經典 KG 仍是事實基礎,但建構方式從規則/統計轉向 **LLM-driven KG construction**。

**LLM 時代 KG 的價值**:
1. **grounding** — 把幻覺壓在事實上
2. **可解釋性** — 推理路徑可追溯
3. **權限控管** — subgraph-level ACL
4. **多跳關係推理** — vector RAG 的死穴

**兩種主流 GraphRAG**:
- **Microsoft 派** — 社群偵測 + 階層摘要,適合敘事/global sensemaking
- **Neo4j 派** — LLM 直譯 Cypher,適合結構化問答

**工具棧**:Cognee、Graphiti (Zep 開源核心)、LangChain GraphQAChain、LlamaIndex KnowledgeGraphIndex。

## 8. 時序圖 (Temporal Graph)

- **TGN (Temporal Graph Networks)** — 經典基礎,memory module + temporal attention
- **TGAT、JODIE、DyRep** — 各擅勝場
- **Graphiti / Zep 雙時間軸 (bi-temporal) 知識圖** — 同時追蹤「事件發生時間」與「系統知道的時間」,讓 agent memory 可以正確處理歷史修正。Zep 在 LongMemEval 比 MemGPT 提升 18.5% 準確度、降低 90% 延遲

**選型**:推薦/詐欺即時偵測用 TGN;agent 長期記憶用 Graphiti;區塊鏈 AML 用 Wavelet-Temporal Graph Transformer。

## 9. 3D / 幾何 GNN(等變網路)

**SE(3) / E(3)-equivariant networks** 在分子、材料、物理模擬已成主流:
- **EquiformerV2** 在 OC20 與 OC22 引領
- **MACE** 與 **Allegro** 是 ML force field 主流
- **SE(3)-Transformer** 啟發了 EquiFold
- **DiffDock** 用等變擴散做 protein-ligand docking

旋轉/平移不變性是物理先驗,等變 GNN 樣本效率比 invariant baseline 高 5-10×。

## 10. 2025-2026 趨勢:Graph 在 LLM 時代會消失嗎?

**短答:否**。三個不可替代位置:
1. **Grounding** — 純向量 RAG 在 multi-hop、聚合查詢(「X 公司的所有子公司去年總營收?」)必敗,KG 結構查詢一行 Cypher 解決
2. **結構天然資料** — 分子、蛋白、社交網路、交易圖本質就是圖,LLM 無法繞過
3. **可解釋推理** — 合規(醫療/金融/法務)場景需要可追溯的推理鏈,圖路徑就是 explanation

**新趨勢**:GFM 嘗試做圖版 GPT;Graph + LLM 雙塔融合;bi-temporal agent memory;scientific ML 等變網路持續擴張。

**會萎縮的部分**:純結構 node classification(已被 LLM-as-classifier 蠶食)、純文本 KG 補全(LLM 端到端做掉)。

## 11. 何時該學 GNN

**必學族群**:推薦系統工程師、藥物/材料/scientific ML 研究者、銀行反詐騙與 AML、知識圖譜 / GraphRAG 工程師、Web3 鏈上分析、社交網路平台。

**只需學應用面**:純 NLP / CV / LLM 應用工程師——學會用 GraphRAG、看懂 Cypher、理解 entity linking 即可,不必啃 message passing 數學。

**學習路徑**:CS224W (Stanford) → PyG 官方教學 → LightGCN/GraphSAGE 復現 → GraphGPS → 選一個應用領域深入。

---

## 2026 圖學習工程師地圖

| 角色 | 必備技能棧 | 重點論文/工具 |
|---|---|---|
| **GraphRAG 應用工程師** | Neo4j + LLM、LightRAG、entity linking | Microsoft GraphRAG、Graphiti、LangChain |
| **推薦系統 GNN** | PyG/DGL、sampling、雙塔 | LightGCN++、UltraGCN、PinSAGE、TGN |
| **藥物 / 材料 ML** | RDKit、等變網路、MD | Chemprop、EquiformerV2、MACE、DiffDock |
| **金融詐欺 GNN** | 異質圖、時序、可解釋 | TGN、LAS-GNN、LineMVGNN、continual GL |
| **GFM 研究者** | 多任務、in-context、TAG | OFA、GraphAny、UniGraph2、GOFA |
| **基礎建設工程** | cuGraph、distributed sampler | PyG 2.0、DGL 2.1、GraphScope |

**一句話總結**:LLM 沒殺死圖學習,而是讓圖從「獨立 ML 子領域」升級成「LLM 系統的記憶層 + 推理層 + 結構化檢索層」。**2026 年不會 GraphRAG 的 AI 工程師,等同於 2023 年不會 vector DB 的 AI 工程師**。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
