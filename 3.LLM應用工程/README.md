# LLM 應用工程

本目錄涵蓋 LLM 從部署到實際應用的完整流程，包括 API 整合、Agent、RAG、推論優化、部署策略等實用技術。

---

## 目錄

1. [LLM 部署與運行基礎](#1-llm-部署與運行基礎)
2. [LLM 作為 API 與應用程式整合](#2-llm-作為-api-與應用程式整合)
3. [Agent 與工具使用](#3-agent-與工具使用)
4. [檢索增強生成 (RAG) 基礎](#4-檢索增強生成-rag-基礎)
5. [進階 RAG 與多元資料檢索](#5-進階-rag-與多元資料檢索)
6. [推論優化 (Inference Optimization)](#6-推論優化-inference-optimization)
7. [LLM 應用部署](#7-llm-應用部署)
8. [LLM 安全與防禦](#8-llm-安全與防禦)
9. [綜合案例與工作流程示範](#9-綜合案例與工作流程示範)
10. [2024-2025 最新發展](#10-2024-2025-最新發展)

---

## 1. LLM 部署與運行基礎

### 1.1 LLM 部署模式概述：API、雲端、在地 (Local)
### 1.2 開源模型 vs. 專有模型 (OpenAI, Anthropic vs. Llama2, GPT-NeoX 等)
### 1.3 實用工具與框架介紹：LM Studio、Ollama、llama.cpp、Hugging Face Spaces
### 1.4 實作示例：
- (程式碼) 使用 Hugging Face Transformers 在本地載入並推論簡單模型
- (程式碼) 使用 OpenAI API 呼叫 GPT-4 生成文字

---

## 2. LLM 作為 API 與應用程式整合

### 2.1 建立 API 接口：OpenAI API、Hugging Face Inference Endpoints
### 2.2 與前端整合：Streamlit、Gradio、WebUI
### 2.3 與自動化工具整合 (例如 UIpath、Zapier)
- 概念與流程範例 (將生成的程式碼自動貼回 IDE、或透過 RPA 工具執行特定任務)
### 2.4 實作示例：
- (程式碼) 使用 Flask + OpenAI API 建立簡易的 LLM RESTful API
- (程式碼) 在 Streamlit 架設 ChatGPT 互動式介面

---

## 3. Agent 與工具使用

### 📚 [AI Agents 與 Agentic Workflows 詳解](./3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md)
完整介紹 2024-2025 年 AI Agents 的發展，包括：
- LangGraph、CrewAI、AutoGPT 等主流框架
- ReAct、Plan-and-Execute 等工作流程模式
- 完整實作範例與最佳實踐

### 3.1 Agent 的概念：ReAct、Toolformer、LangChain Agents
### 3.2 常用代理(Agents)與工具整合 (Python REPL、搜尋引擎、資料庫查詢)
### 3.3 LangChain Functions/Tools 使用範例 (調用外部 API)
### 3.4 實作示例：
- (程式碼) 建立一個 LangChain Agent，能接收使用者指令並自動選擇適合的工具 (如Google Search API 或 Python 執行器)
- (程式碼) 使用 LangChain + SQL 資料庫工具：自動將使用者問題轉為 SQL 查詢並回傳結果

---

## 4. 檢索增強生成 (RAG) 基礎

### 4.1 RAG 流程與原理：向量資料庫、Embeddings、檢索器 (Retriever)
### 4.2 文檔載入器與文件拆分 (PDF、JSON、HTML)
### 4.3 向量資料庫 (Chroma、Pinecone、Milvus) 基礎與設置
### 4.4 實作示例：
- (程式碼) 使用 Hugging Face Sentence Transformers 產生向量嵌入
- (程式碼) 將文檔載入、拆分並存入向量資料庫 (Chroma)
- (程式碼) 實作簡單的 RAG：使用使用者查詢檢索並附加上下文給 LLM

---

## 5. 進階 RAG 與多元資料檢索

### 📚 [RAG 2.0 與多模態 RAG 系統詳解](./5.進階%20RAG%20與多元資料檢索/RAG_2.0_與多模態RAG系統.md)
深入介紹 RAG 2.0 的核心技術：
- 混合檢索 (Hybrid Search)
- 查詢改寫 (Query Rewriting)
- HyDE、Reranking
- 多模態 RAG（文字 + 圖像 + 表格）
- 完整的 RAG 2.0 系統實作

### 5.1 Query Rewriting、HyDE、多查詢檢索器
### 5.2 與結構化數據整合 (SQL, Graph DB)
### 5.3 多工具協作：LLM + RAG + Agents
### 5.4 實作示例：
- (程式碼) 使用 LangChain 將 RAG 與 SQL 查詢合併，在回應中整合結構化資料
- (程式碼) 建立複合管道：先使用 RAG 檢索文本，再用 Agent 從 API 取得最新資料補充回答

---

## 6. 推論優化 (Inference Optimization)

### 6.1 加速推論的技術：量化、Flash Attention、KV Cache、Speculative Decoding
### 6.2 工具與框架：vLLM、Text Generation Inference (TGI)、CTranslate2
### 6.3 實作示例：
- (程式碼) 使用 ExLlama 或 QLoRA 量化模型並比較記憶體消耗與推論速度
- (程式碼) 測試KV Cache對回應時間的影響

---

## 7. LLM 應用部署

### 7.1 原型開發：Gradio、Hugging Face Spaces 快速上線展示
### 7.2 生產部署：Serverless(Lambda) vs. 自建GPU叢集(AWS, GCP, Azure)
### 7.3 邊緣部署：在手機、瀏覽器與IoT環境中運行 LLM (MLC LLM)
### 7.4 實作示例：
- (程式碼) 使用 Gradio 部署簡單的互動介面並分享
- (程式碼) 使用 AWS EC2 + GPU + vLLM 部署一個可擴展的 LLM API

---

## 8. LLM 安全與防禦

### 8.1 Prompt Injection、越獄與資料洩漏風險
### 8.2 OWASP LLM Top 10 資安議題
### 8.3 紅隊測試 (Red Teaming) 與防禦策略
### 8.4 實作示例：
- (程式碼) 在測試環境中嘗試使用 Prompt Injection 並觀察模型反應
- (程式碼) 加入基本的提示過濾與規則設定，減少攻擊面

---

## 9. 綜合案例與工作流程示範

### 9.1 實戰案例：RAG + Agent + 部署的端到端流程
### 9.2 將 LLM 融入自動化工作流程 (UiPath RPA)
- (範例) 運行 LLM 分析程式碼後，將結果程式碼自動插入IDE
### 9.3 部署至生產環境並持續監控、調優與版本控制

---

## 10. 2024-2025 最新發展

### 🔥 核心技術突破

#### 1. **AI Agents 成熟化**
- **從原型到生產**：2025年 AI Agents 從實驗走向實際應用
- **主流框架**：
  - **LangGraph**：圖結構的有狀態工作流
  - **CrewAI**：基於角色的團隊協作
  - **AutoGPT/AutoGen**：自主任務分解執行
- **實際應用**：客戶支持、研究助手、自動化工作流程

#### 2. **RAG 2.0 演進**
- **混合檢索**：向量搜索 + BM25 + 圖結構
- **查詢優化**：Query Rewriting、HyDE、多查詢生成
- **重排序 (Reranking)**：使用專門模型提升相關性
- **多模態 RAG**：無縫處理文字、圖像、音頻、視頻
- **自我反思**：推理驗證與事實檢查

#### 3. **推論優化**
- **Flash Attention 2/3**：更快的注意力計算
- **KV Cache 優化**：減少推理延遲
- **量化技術**：INT8/INT4 量化，保持性能
- **Speculative Decoding**：加速生成速度

#### 4. **部署策略**
- **邊緣部署**：在手機、瀏覽器運行 LLM (WebGPU)
- **Serverless**：AWS Lambda、Azure Functions
- **專用硬件**：Groq、Cerebras 等 AI 芯片

### 📊 實際影響

#### 性能提升
- **RAG 準確率**：從 70% 提升到 85%+（使用 RAG 2.0 技術）
- **Agent 成功率**：複雜任務完成率從 50% 提升到 75%+
- **推理速度**：Flash Attention 帶來 2-4x 加速

#### 成本降低
- **量化模型**：內存需求降低 50-75%
- **混合檢索**：檢索成本降低 30-40%
- **Agent 優化**：減少不必要的 API 調用

### 🛠️ 實用工具

#### 2024-2025 新興工具
- **LangGraph**：複雜工作流編排
- **LangSmith**：Agent 調試與追蹤
- **Chroma DB**：輕量級向量數據庫
- **Weaviate**：多模態向量搜索
- **vLLM**：高效推理引擎
- **Ollama**：本地 LLM 部署

### 📚 學習資源

#### 最新文檔
1. [AI Agents 與 Agentic Workflows (2024-2025)](./3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md)
2. [RAG 2.0 與多模態 RAG 系統](./5.進階%20RAG%20與多元資料檢索/RAG_2.0_與多模態RAG系統.md)

#### 框架文檔
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

#### 社群資源
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [AI Agent Examples](https://github.com/langchain-ai/langchain/tree/master/templates)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

---

## 附錄

### 工具列表與學習資源匯總

#### 開發工具
- **LLM 框架**：LangChain、LlamaIndex、Haystack
- **向量數據庫**：Chroma、Pinecone、Weaviate、Milvus
- **部署工具**：vLLM、TGI、Ollama、llama.cpp
- **前端工具**：Streamlit、Gradio、Chainlit

#### 實用模板
- API 呼叫模板
- RAG 流程模板
- Agent 工具配置模板
- 部署配置範例

### Troubleshooting 記錄

#### 常見錯誤與解決方案
1. **向量數據庫連接問題**
2. **API 速率限制處理**
3. **內存不足問題**
4. **Agent 無限循環**
5. **RAG 幻覺問題**

---

## 總結

透過以上的目錄架構，你可以有條理地學習從 LLM 部署、RAG、Agent，到實際應用整合的各種技術。

### 學習路徑建議

1. **基礎階段**：熟悉 LLM API 調用和基本部署
2. **進階階段**：掌握 RAG 和 Agent 技術
3. **實戰階段**：構建端到端應用
4. **優化階段**：推論優化和生產部署
5. **前沿探索**：追蹤 2024-2025 年最新技術

### 實踐建議

- ✅ 從簡單項目開始（基礎 RAG 或 Agent）
- ✅ 逐步增加複雜度（多模態、混合檢索）
- ✅ 關注性能優化（量化、緩存）
- ✅ 重視安全性（Prompt Injection 防禦）
- ✅ 持續學習新技術（RAG 2.0、AI Agents）

---

**最後更新**：2025年1月
**版本**：3.0（新增 2024-2025 最新內容）
