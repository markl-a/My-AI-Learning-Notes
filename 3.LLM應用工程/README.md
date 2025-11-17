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

### 1.1 LLM 部署模式概述：API、雲端、本地端 (Local)
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
- 以下提供使用 **Responses API** 串接 `gpt-4o-mini` 的最新參考實作（儲存為 `responses_quickstart.py` 即可執行）：

```python
"""Quickstart for the OpenAI Responses API."""

import os
from openai import OpenAI


def main() -> None:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a concise assistant that always returns JSON.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "用 2 點條列說明 LLM 的 RAG 流程"},
                ],
            }
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "string"}}}, "required": ["steps"]}}},
    )

    print(response.output_text)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("請先設定 OPENAI_API_KEY 環境變數")
    main()
```

執行前使用 `pip install openai` 安裝套件，並在 shell 中設定 `export OPENAI_API_KEY=...`；程式會直接輸出 JSON 結構，方便後續與前端或自動化流程串接。

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
- 透過 **LangGraph** 快速建立具備工具調用的 ReAct Agent：

```python
"""LangGraph ReAct agent quickstart."""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


def get_weather(city: str) -> str:
    """簡易的示範工具，回傳指定城市的天氣。"""

    data = {"台北": "晴朗 26°C", "台中": "多雲 24°C", "高雄": "晴時多雲 28°C"}
    return data.get(city, "查無資料")


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(
        llm,
        tools=[get_weather],
        prompt="你是一個嚴謹的助理，回答時同時說明你使用了哪個工具。",
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "請幫我查台北的天氣"}]})
    print(result["messages"][-1].content[0]["text"])


if __name__ == "__main__":
    main()
```

使用前請先安裝 `pip install -U langgraph langchain-openai` 並設定 `OPENAI_API_KEY`；程式會自動在工具與 LLM 之間建立迴圈，適合延伸成更複雜的工作流。

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

### 🔥 2024-2025 核心技術快照

| 範疇 | 亮點 | 實務建議 |
| --- | --- | --- |
| **模型** | GPT-4o / o1、Llama 3、Gemini 1.5 Pro、Claude 3.7 Sonnet、Mistral Large 2、Phi-3.5/4 | 針對任務選擇模型族系，並評估上下文長度、推論成本與授權條款 |
| **Agent** | LangGraph Durable Execution、CrewAI 任務編排、AutoGen v0.3、OpenAI Assistants API、Model Context Protocol | 使用 LangGraph 打底，搭配 MCP 導入內部 API / 資料源，並透過 LangSmith 追蹤行為 |
| **RAG** | GraphRAG、HyDE、BGE reranker、大規模向量資料庫（Weaviate Cloud、Qdrant Hybrid） | 建立多層檢索（向量 + 關鍵字），導入重排序與後處理（壓縮、答案驗證） |
| **推論** | vLLM 0.4.x、SGLang、TensorRT-LLM 0.10、FlashAttention 3、Speculative/Medusa 解碼 | 在雲端使用 vLLM/SGLang 取得高吞吐，在邊緣配合 llama.cpp + GGUF |
| **評測/觀測** | MT-Bench、Arena-Hard、SWE-bench Verified、GAIA、多模態 Benchmarks；OpenTelemetry GenAI semantic conventions | 建立自動化離線評測 + 線上監控，將 span/event 命名標準化，串接 LangSmith 或 Arize |
| **安全** | OWASP LLM Top 10、Prompt Injection Red Team、Guardrails (Guardrails AI、NeMo Guardrails)、安全沙盒（Computer Use） | 建立資料分級與輸出過濾，對 Tool/Function Calling 加入白名單與強制 JSON Schema |

### 📊 成效觀測範例

- **RAG 2.0** 導入重排序與答案驗證後，可將 groundedness 提升至 85% 以上。
- **LangGraph + LangSmith** 可將多步任務的失敗率降低 20-30%，並提供可追溯的節點紀錄。
- **vLLM / SGLang** 在 70B 級模型上提供 2-4x 的吞吐提升，並支援分塊 KV Cache。
- **量化與蒸餾**（如 QLoRA、AWQ、Phi-4、MiniCPM）能將推論成本降低 40%-70%，並利於邊緣部署。

### 🛠️ 重點工具地圖（2024 Q4 版）

- **Agent 編排**：LangGraph、CrewAI、AutoGen、OpenAI Assistants API、Anthropic Workflows。
- **工具協定**：Model Context Protocol（Python SDK、CLI 工具）、OpenAI Function Calling、Azure AI Foundry Toolchain。
- **RAG 平台**：LlamaIndex、LangChain Templates、Microsoft GraphRAG、Weaviate Hybrid Search。
- **推論框架**：vLLM、SGLang、Text Generation Inference、TensorRT-LLM、Ollama、MLC-LLM。
- **監控與評測**：LangSmith、Arize Phoenix、Weights & Biases Traces、OpenTelemetry GenAI、DeepEval、Ragas。

### 📚 延伸學習資源

1. [AI Agents 與 Agentic Workflows (2024-2025)](./3.Agent/AI_Agents_與_Agentic_Workflows_2024-2025.md)
2. [RAG 2.0 與多模態 RAG 系統](./5.進階%20RAG%20與多元資料檢索/RAG_2.0_與多模態RAG系統.md)
3. [2025-10-20~12-28 大致內容整理](../4.相關的更新Blog/2025-10-20~12-28的大致內容.md)

### ✅ 下一步建議

1. 依據應用情境挑選模型（多模態 / 長上下文 / 推理優化）。
2. 以 LangGraph 或 CrewAI 打底，結合 MCP 或工具函式調用串接企業內資料源。
3. 導入多層檢索 + 重排序 + 答案驗證建立穩健的 RAG 2.0 管線。
4. 使用 vLLM / TensorRT-LLM 優化推論成本與延遲，邊緣場景採用 GGUF + WebGPU。
5. 建立自動化評測（MT-Bench、SWE-bench）與觀測性（OpenTelemetry GenAI + LangSmith）。

**最後更新**：2025年1月（同步補充 2024 Q4 ~ 2025 Q1 發布的模型與工具）
