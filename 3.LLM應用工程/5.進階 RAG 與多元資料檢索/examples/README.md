# 進階 RAG 實作範例

這個資料夾包含進階 RAG 技術的完整、可執行的實作範例。每個範例都經過測試驗證，並包含詳細的註釋和說明。

## 📋 目錄

1. **Query Rewriting & HyDE** - 查詢改寫與假設文檔嵌入
2. **SQL Integration** - SQL 資料庫整合
3. **Graph RAG** - 知識圖譜檢索
4. **Agent Collaboration** - LLM + RAG + Agents 協作
5. **End-to-End Project** - 完整的實戰專案

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 在 examples 目錄下
pip install -r requirements.txt
```

### 2. 設定環境變數

```bash
# 複製環境變數範本
cp .env.example .env

# 編輯 .env 並填入你的 API 金鑰
# OPENAI_API_KEY=sk-...
```

### 3. 運行範例

```bash
# 運行查詢改寫範例
python 1_query_rewriting_hyde.py

# 運行測試
python test_query_rewriting.py

# 或使用 pytest
pytest test_query_rewriting.py -v
```

## 📚 範例說明

### 1. Query Rewriting & HyDE (查詢改寫與 HyDE)

**檔案**: `1_query_rewriting_hyde.py`

**包含技術**:
- ✅ Query Rewriting - 改寫使用者查詢以提升檢索精準度
- ✅ Multi-Query Generation - 生成多個查詢變體提升召回率
- ✅ Query Expansion - 擴展查詢內容
- ✅ HyDE (Hypothetical Document Embeddings) - 生成假設文檔改善語義匹配
- ✅ 混合檢索策略

**核心類別**:
- `QueryRewriter`: 查詢改寫器
- `HyDERetriever`: HyDE 檢索器
- `AdvancedQueryRAG`: 整合系統

**使用範例**:

```python
from query_rewriting_hyde import AdvancedQueryRAG

# 初始化系統
rag = AdvancedQueryRAG()

# 攝取文檔
documents = ["文檔1內容", "文檔2內容", ...]
rag.ingest_documents(documents)

# 使用不同方法查詢
# 方法: "standard", "rewrite", "multi_query", "hyde", "hybrid"
result = rag.query_and_answer(
    query="你的問題",
    method="hybrid",  # 推薦使用混合方法
    top_k=3
)

print(result["answer"])
```

**測試**:
```bash
python test_query_rewriting.py
```

### 2. SQL Database Integration (SQL 資料庫整合)

**檔案**: `2_sql_integration.py`

**包含技術**:
- ✅ 智能查詢路由（自動判斷 SQL/向量/混合）
- ✅ 自然語言轉 SQL (NL2SQL)
- ✅ 安全 SQL 執行（防注入）
- ✅ 結果融合（結構化 + 非結構化資料）
- ✅ 示例資料庫自動生成

**核心類別**:
- `SQLRAGIntegration`: SQL + RAG 主系統

**使用範例**:
```python
from sql_integration import SQLRAGIntegration

system = SQLRAGIntegration(db_path="company.db")
system.create_sample_database()
system.ingest_documents(documents)

result = system.hybrid_query("業務部員工的薪資和福利政策")
print(result.final_answer)
```

**測試**: `python test_sql_integration.py`

### 3. Graph RAG (知識圖譜檢索)

**檔案**: `3_graph_rag.py`

**包含技術**:
- ✅ 自動知識圖譜構建
- ✅ 實體和關係抽取
- ✅ 多跳推理查詢
- ✅ 路徑查找和解釋
- ✅ 圖譜可視化（NetworkX + Matplotlib）
- ✅ 圖譜保存和載入

**核心類別**:
- `KnowledgeGraphBuilder`: 圖譜構建器
- `GraphRAGSystem`: Graph RAG 主系統

**使用範例**:
```python
from graph_rag import GraphRAGSystem

graph_rag = GraphRAGSystem()
graph_rag.build_knowledge_graph(documents)
graph_rag.kg_builder.visualize("graph.png")

result = graph_rag.multi_hop_query(
    "OpenAI 和 Transformer 有什麼關係？",
    max_hops=3
)
print(result.answer)
```

**測試**: `python test_graph_rag.py`

### 4. Agent Collaboration (LLM + RAG + Agent 協作)

**檔案**: `4_agent_collaboration.py`

**包含技術**:
- ✅ ReAct Agent 架構（Reasoning + Acting）
- ✅ 工具整合（RAG、計算器、推理、分析）
- ✅ 自動工具選擇
- ✅ 任務分類和路由
- ✅ 多 Agent 協作系統

**核心類別**:
- `RAGAgent`: 整合 RAG 的智能 Agent
- `MultiAgentSystem`: 多 Agent 協作
- `RAGTool`, `CalculatorTool`, `ReasoningTool`: 工具集

**使用範例**:
```python
from agent_collaboration import RAGAgent

agent = RAGAgent()
agent.ingest_documents(documents)
agent.setup_tools()
agent.create_agent()

result = agent.query("Python 是誰建立的？")
print(result['answer'])
```

**測試**: `python test_agent_collaboration.py`

## 🧪 測試

每個範例都包含對應的測試檔案：

```bash
# 運行所有測試
pytest -v

# 運行特定測試
pytest test_query_rewriting.py -v

# 查看測試覆蓋率
pytest --cov=. tests/
```

## 📊 性能比較

| 方法 | 檢索準確度 | 召回率 | 速度 | 適用場景 |
|------|-----------|--------|------|---------|
| Standard | 中 | 中 | 快 | 簡單查詢 |
| Query Rewriting | 高 | 中 | 中 | 模糊查詢 |
| Multi-Query | 中 | 高 | 慢 | 需要全面性 |
| HyDE | 高 | 中 | 慢 | 複雜/抽象查詢 |
| Hybrid | 最高 | 高 | 最慢 | 生產環境推薦 |

## 🛠️ AI 輔助工具

### 自動優化工具

```bash
# 使用 AI 分析並優化你的 RAG 系統
python ai_assistant.py optimize --config your_config.yaml
```

### 自動測試生成

```bash
# 自動生成測試案例
python ai_assistant.py generate-tests --module query_rewriting_hyde
```

## 💡 最佳實踐

### 1. 選擇合適的檢索方法

- **簡單事實查詢**: 使用 `standard`
- **模糊或口語化查詢**: 使用 `rewrite`
- **需要高召回率**: 使用 `multi_query`
- **抽象概念查詢**: 使用 `hyde`
- **生產環境**: 使用 `hybrid`

### 2. 優化參數

```python
# 調整 top_k 參數
# - 小資料集: top_k=3-5
# - 大資料集: top_k=5-10
result = rag.query(query, top_k=5)

# 調整 chunk_size
# - 短文檔: chunk_size=500-800
# - 長文檔: chunk_size=1000-1500
rag.ingest_documents(docs, chunk_size=1000)
```

### 3. 錯誤處理

```python
try:
    result = rag.query_and_answer(query)
except ValueError as e:
    print(f"配置錯誤: {e}")
except Exception as e:
    print(f"查詢失敗: {e}")
```

## 🐛 常見問題

### Q1: "OPENAI_API_KEY 未設置" 錯誤

**解決方案**:
```bash
cp .env.example .env
# 編輯 .env 並填入你的 API 金鑰
```

### Q2: ChromaDB 持久化問題

**解決方案**:
```python
# 刪除舊的資料庫
import shutil
shutil.rmtree("./chroma_db", ignore_errors=True)

# 重新初始化
rag = AdvancedQueryRAG(persist_directory="./chroma_db")
```

### Q3: 記憶體不足

**解決方案**:
- 減小 `chunk_size`
- 減少 `top_k`
- 使用批次處理
- 考慮使用輕量級嵌入模型

## 📖 延伸閱讀

- [LangChain 文檔](https://python.langchain.com/)
- [ChromaDB 文檔](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [RAG 論文原文](https://arxiv.org/abs/2005.11401)
- [HyDE 論文](https://arxiv.org/abs/2212.10496)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License

---

**最後更新**: 2024-11
**維護者**: AI Learning Community
