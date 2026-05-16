# 📚 資源索引與工具清單

> 精選AI學習資源、工具、資料集、社群的完整索引

---

## 📖 官方文檔

### LLM框架與庫
| 名稱 | 類型 | 鏈接 | 說明 |
|------|------|------|------|
| **Hugging Face Transformers** | 模型庫 | [Docs](https://huggingface.co/docs/transformers) | 最全面的預訓練模型庫 |
| **LangChain** | 應用框架 | [Docs](https://python.langchain.com/) | LLM應用開發框架 |
| **LlamaIndex** | RAG框架 | [Docs](https://docs.llamaindex.ai/) | 資料索引與檢索 |
| **vLLM** | 推論引擎 | [Docs](https://docs.vllm.ai/) | 高性能LLM推理 |
| **DeepSpeed** | 訓練框架 | [Docs](https://www.deepspeed.ai/) | 大規模分布式訓練 |
| **PyTorch** | 深度學習 | [Docs](https://pytorch.org/docs/) | 主流深度學習框架 |
| **TensorFlow** | 深度學習 | [Docs](https://www.tensorflow.org/) | Google深度學習框架 |

### API服務
| 提供商 | 模型 | 定價 | 文檔 |
|--------|------|------|------|
| **OpenAI** | GPT-4o, o1 | $$ | [API Docs](https://platform.openai.com/docs) |
| **Anthropic** | Claude 3.5 | $$ | [API Docs](https://docs.anthropic.com/) |
| **Google AI** | Gemini 1.5 Pro | $$ | [API Docs](https://ai.google.dev/) |
| **Groq** | Llama, Mixtral | $ | [API Docs](https://console.groq.com/docs) |
| **Together AI** | 多種開源模型 | $ | [API Docs](https://docs.together.ai/) |
| **Replicate** | 開源模型託管 | $ | [API Docs](https://replicate.com/docs) |

---

## 🛠️ 開發工具

### 向量資料庫

| 名稱 | 類型 | 特點 | 適用場景 |
|------|------|------|----------|
| **Chroma** | 開源 | 輕量、易用 | 本地開發、小規模 |
| **Pinecone** | 雲服務 | 高性能、托管 | 生產環境 |
| **Qdrant** | 開源/雲 | 高效、可擴展 | 中大規模應用 |
| **Weaviate** | 開源/雲 | 混合搜索 | 複雜檢索 |
| **Milvus** | 開源 | 企業級 | 大規模部署 |
| **pgvector** | PostgreSQL擴展 | SQL友好 | 已有PostgreSQL |
| **FAISS** | 庫 | Meta出品、快速 | 研究、原型 |

### 開發環境

| 工具 | 用途 | 鏈接 |
|------|------|------|
| **Jupyter** | 交互式開發 | [官網](https://jupyter.org/) |
| **VS Code** | IDE | [官網](https://code.visualstudio.com/) |
| **PyCharm** | Python IDE | [官網](https://www.jetbrains.com/pycharm/) |
| **Cursor** | AI編碼助手 | [官網](https://cursor.sh/) |
| **GitHub Copilot** | AI程式碼補全 | [官網](https://github.com/features/copilot) |

### 實驗追蹤

| 工具 | 特點 | 鏈接 |
|------|------|------|
| **MLflow** | 全生命週期管理 | [Docs](https://mlflow.org/) |
| **Weights & Biases** | 可視化強大 | [Docs](https://docs.wandb.ai/) |
| **TensorBoard** | TensorFlow集成 | [Docs](https://www.tensorflow.org/tensorboard) |
| **LangSmith** | LLM應用監控 | [Docs](https://docs.smith.langchain.com/) |
| **Arize** | AI可觀測性 | [Docs](https://docs.arize.com/) |

---

## 📊 資料集資源

### 通用資料集平台

| 平台 | 說明 | 鏈接 |
|------|------|------|
| **Hugging Face Datasets** | 最全面的資料集庫 | [Hub](https://huggingface.co/datasets) |
| **Kaggle** | 競賽與資料集 | [Datasets](https://www.kaggle.com/datasets) |
| **Papers with Code** | 論文配套資料集 | [Datasets](https://paperswithcode.com/datasets) |
| **Google Dataset Search** | Google資料集搜索 | [Search](https://datasetsearch.research.google.com/) |

### NLP資料集

| 資料集 | 任務 | 規模 | 鏈接 |
|--------|------|------|------|
| **GLUE** | 通用語言理解 | 9個任務 | [官網](https://gluebenchmark.com/) |
| **SuperGLUE** | 進階語言理解 | 8個任務 | [官網](https://super.gluebenchmark.com/) |
| **SQuAD** | 閱讀理解 | 10萬+ | [官網](https://rajpurkar.github.io/SQuAD-explorer/) |
| **MNLI** | 自然語言推理 | 433K | [HF](https://huggingface.co/datasets/multi_nli) |
| **SST-2** | 情感分析 | 70K | [HF](https://huggingface.co/datasets/sst2) |
| **CoNLL** | 命名實體識別 | 多種 | [官網](https://www.conll.org/) |

### 多模態資料集

| 資料集 | 類型 | 說明 | 鏈接 |
|--------|------|------|------|
| **COCO** | 圖像標註 | 物體檢測、分割 | [官網](https://cocodataset.org/) |
| **ImageNet** | 圖像分類 | 1400萬圖像 | [官網](https://www.image-net.org/) |
| **LAION-5B** | 圖文對 | 58億圖文對 | [官網](https://laion.ai/) |
| **Conceptual Captions** | 圖文對 | 330萬 | [GH](https://github.com/google-research-datasets/conceptual-captions) |

---

## 🎓 學習平台

### 在線課程

| 平台 | 特點 | 推薦課程 | 價格 |
|------|------|----------|------|
| **DeepLearning.AI** | Andrew Ng主講 | Deep Learning Specialization | 免費/付費證書 |
| **Fast.ai** | 實踐導向 | Practical Deep Learning | 免費 |
| **Coursera** | 大學合作 | ML Specialization | 免費試聽 |
| **Hugging Face Course** | Transformers專精 | NLP Course | 免費 |
| **Stanford Online** | 頂尖大學 | CS224N, CS229 | 免費 |

### YouTube頻道

| 頻道 | 內容 | 訂閱數 |
|------|------|--------|
| **3Blue1Brown** | 數學直覺 | 6M+ |
| **Two Minute Papers** | 論文解讀 | 1.5M+ |
| **Yannic Kilcher** | 深度論文分析 | 300K+ |
| **Andrej Karpathy** | AI教學 | 300K+ |
| **StatQuest** | 統計與ML | 1M+ |

### 書籍推薦

| 書名 | 作者 | 難度 | 重點 |
|------|------|------|------|
| **Deep Learning** | Goodfellow et al. | 🔴 高 | 深度學習聖經 |
| **Pattern Recognition and ML** | Bishop | 🔴 高 | 機器學習理論 |
| **Hands-On Machine Learning** | Géron | 🟡 中 | 實踐為主 |
| **Neural Networks from Scratch** | Harrison Kinsley | 🟢 低 | 從零實現 |
| **Building LLM Apps** | Various | 🟡 中 | LLM應用 |

---

## 🌐 社群與論壇

### 技術社群

| 社群 | 平台 | 活躍度 | 鏈接 |
|------|------|--------|------|
| **Hugging Face** | Discord | ⭐⭐⭐⭐⭐ | [加入](https://discord.gg/huggingface) |
| **LangChain** | Discord | ⭐⭐⭐⭐⭐ | [加入](https://discord.gg/langchain) |
| **r/MachineLearning** | Reddit | ⭐⭐⭐⭐⭐ | [訪問](https://reddit.com/r/MachineLearning) |
| **r/LocalLLaMA** | Reddit | ⭐⭐⭐⭐ | [訪問](https://reddit.com/r/LocalLLaMA) |
| **AI台灣** | Facebook | ⭐⭐⭐⭐ | 繁體中文社群 |

### 論文追蹤

| 資源 | 說明 | 更新頻率 |
|------|------|----------|
| **arXiv** | 最新預印本 | 每日 |
| **Papers with Code** | 論文+程式碼 | 每日 |
| **Hugging Face Papers** | 每日精選 | 每日 |
| **Arxiv Sanity** | 個性化推薦 | 實時 |

---

## 🔧 實用工具集

### Prompt工程

| 工具 | 功能 | 鏈接 |
|------|------|------|
| **PromptPerfect** | Prompt優化 | [官網](https://promptperfect.jina.ai/) |
| **LangSmith** | Prompt測試 | [官網](https://smith.langchain.com/) |
| **Snorkel** | 資料標註 | [官網](https://snorkel.ai/) |

### 模型評估

| 工具 | 用途 | 鏈接 |
|------|------|------|
| **RAGAS** | RAG評估 | [GitHub](https://github.com/explodinggradients/ragas) |
| **DeepEval** | LLM評估 | [Docs](https://docs.confident-ai.com/) |
| **PromptTools** | Prompt測試 | [GitHub](https://github.com/hegelai/prompttools) |
| **LangChain Eval** | 應用評估 | [Docs](https://python.langchain.com/docs/guides/evaluation) |

### 可視化工具

| 工具 | 功能 | 鏈接 |
|------|------|------|
| **Netron** | 模型可視化 | [GitHub](https://github.com/lutzroeder/netron) |
| **TensorBoard** | 訓練可視化 | [官網](https://www.tensorflow.org/tensorboard) |
| **Streamlit** | 快速應用 | [官網](https://streamlit.io/) |
| **Gradio** | ML Demo | [官網](https://gradio.app/) |

---

## 💻 開源項目

### 推論優化

| 項目 | 說明 | Star | 鏈接 |
|------|------|------|------|
| **vLLM** | 高吞吐推理 | 20K+ | [GitHub](https://github.com/vllm-project/vllm) |
| **TensorRT-LLM** | NVIDIA優化 | 7K+ | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **llama.cpp** | CPU推理 | 60K+ | [GitHub](https://github.com/ggerganov/llama.cpp) |
| **Ollama** | 本地LLM | 80K+ | [GitHub](https://github.com/ollama/ollama) |

### Agent框架

| 項目 | 說明 | Star | 鏈接 |
|------|------|------|------|
| **LangGraph** | 工作流編排 | 5K+ | [GitHub](https://github.com/langchain-ai/langgraph) |
| **CrewAI** | 多Agent | 15K+ | [GitHub](https://github.com/joaomdmoura/crewAI) |
| **AutoGen** | 對話Agent | 28K+ | [GitHub](https://github.com/microsoft/autogen) |
| **MetaGPT** | 軟件開發 | 42K+ | [GitHub](https://github.com/geekan/MetaGPT) |

### RAG工具

| 項目 | 說明 | Star | 鏈接 |
|------|------|------|------|
| **LlamaIndex** | RAG框架 | 33K+ | [GitHub](https://github.com/run-llama/llama_index) |
| **GraphRAG** | 圖譜RAG | 15K+ | [GitHub](https://github.com/microsoft/graphrag) |
| **RAGFlow** | 端到端RAG | 13K+ | [GitHub](https://github.com/infiniflow/ragflow) |

### 多模態

| 項目 | 說明 | Star | 鏈接 |
|------|------|------|------|
| **Stable Diffusion** | 圖像生成 | 65K+ | [GitHub](https://github.com/Stability-AI/stablediffusion) |
| **FLUX** | 最新圖像生成 | 10K+ | [GitHub](https://github.com/black-forest-labs/flux) |
| **AudioCraft** | 音頻生成 | 20K+ | [GitHub](https://github.com/facebookresearch/audiocraft) |
| **Whisper** | 語音識別 | 65K+ | [GitHub](https://github.com/openai/whisper) |

---

## 🎯 學習路線資源映射

### 初學者路徑 (1-3個月)

```
Python基礎
├─ 官方教程: https://docs.python.org/3/tutorial/
├─ 項目內容: 1.從AI到LLM基礎/2.AI_Intro/
└─ 練習平台: LeetCode, HackerRank

機器學習
├─ 課程: Coursera ML Specialization
├─ 項目內容: 1.從AI到LLM基礎/3.ML_&_Data_Analysis/
└─ 實踐: Kaggle競賽

深度學習
├─ 課程: Fast.ai, DeepLearning.AI
├─ 項目內容: 1.從AI到LLM基礎/4.DL/
└─ 框架: PyTorch, TensorFlow
```

### 中級路徑 (3-6個月)

```
LLM基礎
├─ 課程: Hugging Face NLP Course
├─ 項目內容: 2.深入LLM模型工程與LLM運維/
├─ 工具: Transformers, PEFT, vLLM
└─ 論文: 5.AI研究前沿_2024-2025/1.大型語言模型(LLM)/

RAG系統
├─ 文檔: LangChain, LlamaIndex
├─ 項目內容: 3.LLM應用工程/4.RAG基礎/
├─ 工具: Chroma, Pinecone, BGE
└─ 論文: 5.AI研究前沿_2024-2025/4.RAG與檢索/

Agent開發
├─ 框架: LangGraph, CrewAI
├─ 項目內容: 3.LLM應用工程/3.Agent/
├─ 工具: LangSmith, MCP
└─ 論文: 5.AI研究前沿_2024-2025/3.Agent系統/
```

### 高級路徑 (6-12個月)

```
模型訓練
├─ 框架: DeepSpeed, FSDP, Megatron
├─ 項目內容: 2.深入LLM模型工程與LLM運維/
├─ 技術: LoRA, QLoRA, GaLore
└─ 論文: 5.AI研究前沿_2024-2025/5.訓練與優化/

多模態
├─ 工具: Diffusers, AudioCraft, Whisper
├─ 項目內容: 3.LLM應用工程/10.多模態生成/
├─ 論文: 5.AI研究前沿_2024-2025/2.多模態模型/
└─ 項目: 完整的內容生成系統

生產部署
├─ 工具: vLLM, Docker, K8s
├─ 項目內容: 3.LLM應用工程/7.部署/
├─ 監控: LangSmith, Arize, Prometheus
└─ 優化: 量化、快取、並發
```

---

## 🔍 快速查找

### 按技術棧查找

**Python生態**
- NumPy/Pandas → `1.從AI到LLM基礎/3.ML_&_Data_Analysis/`
- PyTorch → `1.從AI到LLM基礎/4.DL/03.Pytorch/`
- TensorFlow → `1.從AI到LLM基礎/4.DL/01.Tensorflow2/`

**LLM工具鏈**
- Transformers → Hugging Face官方文檔
- LangChain → `3.LLM應用工程/1.LangchainDemos/`
- vLLM → `5.AI研究前沿_2024-2025/5.訓練與優化/`

**應用開發**
- RAG → `3.LLM應用工程/4.RAG基礎/`
- Agent → `3.LLM應用工程/3.Agent/`
- 多模態 → `3.LLM應用工程/10.多模態生成/`

### 按問題查找

**"如何訓練LLM？"**
→ `2.深入LLM模型工程與LLM運維/`
→ `5.AI研究前沿_2024-2025/5.訓練與優化/`

**"如何構建RAG系統？"**
→ `3.LLM應用工程/4.RAG基礎/`
→ `5.AI研究前沿_2024-2025/4.RAG與檢索/`

**"如何生成圖片/影片？"**
→ `3.LLM應用工程/10.多模態生成/`
→ `5.AI研究前沿_2024-2025/2.多模態模型/`

**"最新技術趨勢？"**
→ `5.AI研究前沿_2024-2025/`
→ `4.相關的更新Blog/`

---

## 📱 移動端學習

### App推薦

- **Anki**: 間隔重複記憶
- **Notion**: 筆記組織
- **Feedly**: RSS訂閱
- **Pocket**: 稍後閱讀

### 碎片時間利用

- 通勤: 聽播客、看技術影片
- 排隊: 刷論文摘要、技術文章
- 休息: 解LeetCode、看GitHub

---

## 🎁 免費資源清單

### 計算資源

| 平台 | 配置 | 限制 | 鏈接 |
|------|------|------|------|
| **Google Colab** | T4 GPU | 時間限制 | [colab.research.google.com](https://colab.research.google.com/) |
| **Kaggle Notebooks** | P100 GPU | 30h/週 | [kaggle.com/code](https://www.kaggle.com/code) |
| **Hugging Face Spaces** | CPU/GPU | 免費層 | [huggingface.co/spaces](https://huggingface.co/spaces) |
| **GitHub Codespaces** | CPU | 60h/月 | [github.com/codespaces](https://github.com/codespaces) |

### 免費API

| 服務 | 免費額度 | 鏈接 |
|------|----------|------|
| **Groq** | 高速推理 | [console.groq.com](https://console.groq.com/) |
| **Together AI** | $25信用 | [together.ai](https://www.together.ai/) |
| **Replicate** | 少量免費 | [replicate.com](https://replicate.com/) |

---

## 📌 快速參考

### 常用命令

```bash
# Hugging Face下載模型
huggingface-cli download model_name

# 啟動Jupyter
jupyter notebook

# 安裝依賴
pip install -r requirements.txt

# Docker啟動
docker-compose up -d
```

### 常用程式碼片段

**位置**: 參考各模塊的README快速開始部分

---

**持續更新中...** 📚

如有補充建議，歡迎PR！

---

最後更新: 2025-01-19
