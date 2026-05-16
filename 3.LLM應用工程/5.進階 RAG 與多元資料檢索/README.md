## 進階 RAG 與多元資料檢索

這個模組涵蓋了 RAG 技術的進階應用，包括多種檢索策略、資料整合和智能 Agent 系統。

### 📚 理論文檔

#### 5.1 RAG 技術發展與趨勢
- **2025_RAG技術最新發展.md**
  - 混合檢索（Hybrid Search）
  - GraphRAG 的崛起
  - 多模態 RAG
  - 頂級向量資料庫對比
  - 企業級部署最佳實踐

#### 5.2 RAG 2.0 與多模態系統
- **RAG_2.0_與多模態RAG系統.md**
  - 從 RAG 1.0 到 RAG 2.0 的演進
  - Query Rewriting、HyDE、Reranking
  - 多模態處理（文字、圖像、表格）
  - 自我反思與驗證機制

#### 5.3 完整實作指南
- **完整實作指南.md**
  - 系統化學習路徑
  - 動手實作指導
  - 最佳實踐和陷阱
  - 性能優化技巧

### 💻 實作範例（examples/）

所有範例都是完整、可執行、經過測試的程式碼。

#### 實作 1: Query Rewriting & HyDE
**檔案**: `examples/1_query_rewriting_hyde.py`

✅ **功能**:
- 查詢改寫（Query Rewriting）
- 多查詢生成（Multi-Query Generation）
- HyDE（假設文檔嵌入）
- 混合檢索策略

```python
from query_rewriting_hyde import AdvancedQueryRAG

rag = AdvancedQueryRAG()
rag.ingest_documents(documents)

# 使用混合方法（推薦）
result = rag.query_and_answer(
    query="你的問題",
    method="hybrid",
    top_k=3
)
```

#### 實作 2: SQL Database Integration
**檔案**: `examples/2_sql_integration.py`

✅ **功能**:
- 自然語言轉 SQL（NL2SQL）
- 智能查詢路由
- SQL + 向量檢索混合
- 安全執行機制

```python
from sql_integration import SQLRAGIntegration

system = SQLRAGIntegration(db_path="company.db")
system.create_sample_database()
system.ingest_documents(documents)

# 自動判斷使用 SQL/向量/混合
result = system.hybrid_query("業務部員工的薪資和福利")
```

#### 實作 3: Graph RAG（知識圖譜）
**檔案**: `examples/3_graph_rag.py`

✅ **功能**:
- 自動構建知識圖譜
- 實體和關係抽取
- 多跳推理查詢
- 圖譜可視化

```python
from graph_rag import GraphRAGSystem

graph_rag = GraphRAGSystem()
graph_rag.build_knowledge_graph(documents)
graph_rag.kg_builder.visualize("graph.png")

result = graph_rag.multi_hop_query(
    "OpenAI 和 Transformer 有什麼關係？",
    max_hops=3
)
```

#### 實作 4: LLM + RAG + Agents
**檔案**: `examples/4_agent_collaboration.py`

✅ **功能**:
- ReAct Agent 架構
- 工具整合（RAG、計算、推理）
- 自動工具選擇
- 多 Agent 協作

```python
from agent_collaboration import RAGAgent

agent = RAGAgent()
agent.ingest_documents(documents)
agent.setup_tools()
agent.create_agent()

# Agent 自動選擇和使用工具
result = agent.query("Python 是誰建立的？")
```

### 🚀 快速開始

```bash
# 1. 進入範例目錄
cd "examples"

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 設置 API 金鑰
cp .env.example .env
# 編輯 .env，填入 OPENAI_API_KEY

# 4. 運行第一個範例
python 1_query_rewriting_hyde.py

# 5. 運行測試
python test_query_rewriting.py
```

### 📊 技術對比

| 技術 | 適用場景 | 優勢 | 複雜度 |
|------|---------|------|--------|
| Query Rewriting | 模糊查詢 | 提升準確度 | 低 |
| HyDE | 抽象概念 | 改善語義匹配 | 中 |
| SQL Integration | 結構化資料 | 精確查詢 | 中 |
| Graph RAG | 關係推理 | 多跳查詢 | 高 |
| Agent System | 複雜任務 | 自主決策 | 高 |

### 🎯 學習路徑

1. **基礎** (2-3 小時)
   - 閱讀理論文檔
   - 理解核心概念

2. **實作** (6-8 小時)
   - 逐個運行範例
   - 完成練習任務
   - 通過所有測試

3. **整合** (3-4 小時)
   - 構建完整專案
   - 性能優化
   - 評估和測試

### 📝 檢查清單

完成這個模組後，你應該能夠：

- [ ] 解釋各種進階 RAG 技術的原理
- [ ] 選擇適合場景的技術方案
- [ ] 實作和測試 RAG 系統
- [ ] 整合多種資料源
- [ ] 設計 Agent 系統
- [ ] 優化系統性能

### 📚 更多資源

- **examples/README.md** - 詳細的範例說明
- **完整實作指南.md** - 系統化學習路徑
- [LangChain 文檔](https://python.langchain.com/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

**所有程式碼都經過測試驗證，可直接使用於學習和生產環境。**

## 延伸閱讀
- [18.GNN_Graph_Learning](../../18.GNN_Graph_Learning/README.md) — GraphRAG 的圖學習基礎
- [17.Causal_ML](../../17.Causal_ML/README.md) — Causal RAG / CausalRAG