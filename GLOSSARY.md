# AI/LLM 術語表 (Glossary)

> 本術語表涵蓋 AI、機器學習、深度學習和大型語言模型領域的核心概念。
> 按字母順序排列，方便查閱。

---

## 目錄
- [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i)
- [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r)
- [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Y](#y) | [Z](#z)

---

## A

### Activation Function (激活函數)
神經網路中用於引入非線性的函數。常見的有 ReLU、Sigmoid、Tanh、GELU 等。

### Adam (Adam 優化器)
一種結合 Momentum 和 RMSprop 優點的自適應學習率優化算法，是深度學習中最常用的優化器之一。

### Agent (代理/智能體)
能夠感知環境並採取行動以實現目標的自主系統。在 LLM 領域，指能使用工具、做決策的 AI 系統。

### Alignment (對齊)
確保 AI 系統的行為符合人類意圖和價值觀的技術。包括 RLHF、DPO、Constitutional AI 等方法。

### Attention Mechanism (注意力機制)
允許模型在處理輸入時動態地關注不同部分的機制。是 Transformer 架構的核心組件。

### AutoML (自動機器學習)
自動化機器學習流程的技術，包括特徵工程、模型選擇、超參數調優等。

### AWQ (Activation-aware Weight Quantization)
一種考慮激活值分佈的權重量化方法，能在保持精度的同時大幅減少模型大小。

---

## B

### Backpropagation (反向傳播)
訓練神經網路的核心算法，通過計算損失函數對每個參數的梯度來更新權重。

### Batch Size (批次大小)
每次訓練迭代中使用的樣本數量。影響訓練速度、記憶體使用和模型收斂。

### BERT (Bidirectional Encoder Representations from Transformers)
Google 開發的預訓練語言模型，使用雙向 Transformer 編碼器，開創了 NLP 預訓練時代。

### BPE (Byte Pair Encoding)
一種子詞分詞算法，通過迭代合併最頻繁的字符對來構建詞彙表。GPT 系列使用此方法。

---

## C

### Chain-of-Thought (CoT, 思維鏈)
一種提示技術，引導 LLM 逐步推理以解決複雜問題，顯著提升推理能力。

### Checkpoint (檢查點)
訓練過程中保存的模型狀態，用於恢復訓練或部署模型。

### CLIP (Contrastive Language-Image Pre-training)
OpenAI 開發的視覺-語言模型，通過對比學習將圖像和文本映射到共同的嵌入空間。

### Constitutional AI (憲法 AI)
Anthropic 提出的對齊方法，使用一組原則（"憲法"）來指導 AI 行為。

### Context Window (上下文窗口)
LLM 一次能處理的最大 token 數量。GPT-4 為 128K，Claude 3 為 200K。

### Cosine Similarity (餘弦相似度)
測量兩個向量之間角度的相似度度量，常用於比較嵌入向量。

### Cross-Entropy Loss (交叉熵損失)
分類任務中常用的損失函數，衡量預測概率分佈與真實分佈之間的差異。

---

## D

### Decoder (解碼器)
Transformer 架構中負責生成輸出序列的部分。GPT 系列是純解碼器架構。

### Dense Layer (全連接層)
神經網路中每個神經元都與前一層所有神經元連接的層。

### Diffusion Model (擴散模型)
通過學習逆轉逐步添加噪聲的過程來生成數據的生成模型。Stable Diffusion、DALL-E 3 基於此技術。

### Distillation (知識蒸餾)
將大模型（教師）的知識轉移到小模型（學生）的技術，用於模型壓縮。

### DPO (Direct Preference Optimization)
直接偏好優化，一種不需要訓練獎勵模型的對齊方法，比 RLHF 更簡單高效。

### Dropout
訓練時隨機丟棄部分神經元的正則化技術，用於防止過擬合。

---

## E

### Embedding (嵌入)
將離散數據（如文字、圖像）映射到連續向量空間的表示方法。

### Encoder (編碼器)
Transformer 架構中負責處理輸入序列的部分。BERT 是純編碼器架構。

### Epoch (訓練週期)
完整遍歷一次訓練數據集的過程。

### Evaluation Metrics (評估指標)
衡量模型性能的標準，如準確率、F1 分數、BLEU、ROUGE、困惑度等。

---

## F

### Few-Shot Learning (小樣本學習)
使用少量示例讓模型學習新任務的能力。GPT-3 展示了強大的 few-shot 能力。

### Fine-Tuning (微調)
在預訓練模型基礎上，使用特定任務數據進行進一步訓練的過程。

### Flash Attention
一種高效的注意力計算方法，通過優化記憶體訪問模式大幅加速 Transformer。

### Foundation Model (基礎模型)
在大規模數據上預訓練的模型，可適應多種下游任務。如 GPT-4、Claude、LLaMA。

### Function Calling (函數調用)
LLM 生成結構化輸出以調用外部函數或 API 的能力。

---

## G

### GAN (Generative Adversarial Network, 生成對抗網絡)
由生成器和判別器組成的生成模型，通過對抗訓練生成逼真數據。

### GELU (Gaussian Error Linear Unit)
一種激活函數，被 BERT、GPT 等模型廣泛使用。

### Gradient Descent (梯度下降)
通過計算損失函數的梯度來迭代更新參數的優化算法。

### GPTQ (GPT Quantization)
一種後訓練量化方法，可將模型壓縮到 4-bit 或更低，同時保持精度。

### GQA (Grouped Query Attention)
分組查詢注意力，在 MHA 和 MQA 之間取得平衡，被 LLaMA 2 等模型採用。

### GraphRAG
結合知識圖譜和 RAG 的檢索增強生成方法，提供更結構化的上下文。

---

## H

### Hallucination (幻覺)
LLM 生成看似合理但實際錯誤或虛構內容的現象。是 LLM 應用的主要挑戰之一。

### Hidden State (隱藏狀態)
神經網路中間層的輸出，包含輸入的學習表示。

### Hyperparameter (超參數)
在訓練前設定的參數，如學習率、批次大小、層數等。需要通過實驗調整。

---

## I

### In-Context Learning (上下文學習)
LLM 通過提示中的示例學習執行新任務的能力，無需更新參數。

### Inference (推理)
使用訓練好的模型進行預測的過程。

### Instruction Tuning (指令微調)
使用指令-回應對微調 LLM，使其更好地遵循人類指令。

---

## J

### JSON Mode (JSON 模式)
LLM 輸出結構化 JSON 格式的能力，便於程序化處理。

---

## K

### KV Cache (鍵值緩存)
在 Transformer 推理時緩存過去的 Key 和 Value，避免重複計算，加速生成。

### Knowledge Distillation (知識蒸餾)
見 [Distillation](#distillation-知識蒸餾)。

---

## L

### LangChain
用於構建 LLM 應用的開源框架，提供鏈式調用、記憶、工具等功能。

### Large Language Model (LLM, 大型語言模型)
參數量達數十億至數萬億的語言模型，如 GPT-4、Claude、LLaMA、Gemini。

### Layer Normalization (層歸一化)
對每一層的激活進行歸一化的技術，穩定訓練過程。

### Learning Rate (學習率)
控制每次參數更新幅度的超參數。過大導致不收斂，過小導致訓練緩慢。

### LoRA (Low-Rank Adaptation)
一種高效微調方法，通過添加低秩矩陣來適應新任務，大幅減少可訓練參數。

### Loss Function (損失函數)
衡量模型預測與真實值之間差距的函數，訓練目標是最小化損失。

---

## M

### MCP (Model Context Protocol)
Anthropic 開發的協議，標準化 AI 模型與外部工具/數據源的交互方式。

### Mixture of Experts (MoE, 專家混合)
將模型分為多個"專家"子網路，每次推理只激活部分專家，提高效率。

### MLOps (機器學習運維)
將 DevOps 實踐應用於機器學習的工程規範，涵蓋訓練、部署、監控全流程。

### Multi-Head Attention (多頭注意力)
將注意力機制分為多個"頭"並行計算，捕獲不同類型的關係。

### Multimodal (多模態)
能處理多種類型數據（文本、圖像、音頻等）的模型，如 GPT-4V、Gemini。

---

## N

### NLP (Natural Language Processing, 自然語言處理)
研究計算機理解和生成人類語言的領域。

### Neural Network (神經網路)
受生物神經系統啟發的機器學習模型，由互連的節點（神經元）組成。

---

## O

### ONNX (Open Neural Network Exchange)
開放的神經網路交換格式，支持跨框架模型部署。

### Overfitting (過擬合)
模型過度學習訓練數據的特定模式，導致泛化能力下降。

---

## P

### Parameter (參數)
模型中可學習的權重和偏置。GPT-4 估計有 1.8 萬億參數。

### Perplexity (困惑度)
衡量語言模型預測能力的指標，越低表示模型越好。

### Pre-Training (預訓練)
在大規模無標註數據上訓練模型學習通用表示的過程。

### Prompt (提示)
給 LLM 的輸入文本，用於引導模型生成期望的輸出。

### Prompt Engineering (提示工程)
設計和優化提示以獲得更好 LLM 輸出的技術和實踐。

### Pruning (剪枝)
移除神經網路中不重要的權重或神經元以壓縮模型的技術。

---

## Q

### QLoRA (Quantized LoRA)
結合量化和 LoRA 的微調方法，可在消費級 GPU 上微調大型模型。

### Quantization (量化)
將模型權重從高精度（如 FP32）轉換為低精度（如 INT8、INT4）的技術。

---

## R

### RAG (Retrieval-Augmented Generation, 檢索增強生成)
結合檢索系統和生成模型的架構，先檢索相關文檔再生成回答，減少幻覺。

### ReAct (Reasoning + Acting)
結合推理和行動的 Agent 框架，讓 LLM 能思考並使用工具。

### Regularization (正則化)
防止過擬合的技術，如 L1/L2 正則化、Dropout 等。

### Reinforcement Learning (強化學習)
通過獎勵信號學習最優行為策略的機器學習範式。

### RLHF (Reinforcement Learning from Human Feedback)
使用人類反饋訓練獎勵模型，再用強化學習對齊 LLM 的技術。

### RNN (Recurrent Neural Network, 循環神經網路)
能處理序列數據的神經網路，通過隱藏狀態傳遞信息。已被 Transformer 取代。

### RoPE (Rotary Position Embedding)
旋轉位置編碼，一種高效的位置編碼方法，被 LLaMA 等模型採用。

---

## S

### Scaling Law (縮放定律)
描述模型性能如何隨參數量、數據量、計算量增長的經驗規律。

### Self-Attention (自注意力)
序列中每個元素都與所有其他元素計算注意力的機制。

### Semantic Search (語義搜索)
基於語義相似度而非關鍵詞匹配的搜索技術，通常使用嵌入向量。

### SFT (Supervised Fine-Tuning, 監督微調)
使用標註數據對預訓練模型進行監督學習的過程。

### Softmax
將向量轉換為概率分佈的函數，常用於分類任務的輸出層。

### Speculative Decoding (投機解碼)
使用小模型預測，大模型驗證的推理加速技術。

---

## T

### Temperature (溫度)
控制 LLM 輸出隨機性的參數。高溫度更隨機，低溫度更確定。

### Tensor (張量)
多維數組，深度學習中數據和參數的基本表示形式。

### TensorRT
NVIDIA 的深度學習推理優化器，可大幅加速 GPU 推理。

### Token (詞元)
LLM 處理文本的基本單位，可以是單詞、子詞或字符。

### Tokenizer (分詞器)
將文本轉換為 token 序列的工具。

### Top-K / Top-P Sampling
控制 LLM 生成多樣性的採樣策略。Top-K 限制候選數，Top-P（Nucleus）限制累積概率。

### Transfer Learning (遷移學習)
將在一個任務上學到的知識應用到相關任務的技術。

### Transformer
基於自注意力機制的神經網路架構，是現代 LLM 的基礎。由 "Attention Is All You Need" 論文提出。

---

## U

### Underfitting (欠擬合)
模型未能充分學習數據模式，導致訓練和測試表現都差。

---

## V

### VAE (Variational Autoencoder, 變分自編碼器)
一種生成模型，學習數據的潛在表示並可生成新樣本。

### Vector Database (向量數據庫)
專門存儲和檢索高維向量的數據庫，如 Pinecone、Milvus、Chroma。

### vLLM
高效的 LLM 推理引擎，使用 PagedAttention 技術優化記憶體管理。

---

## W

### Weight (權重)
神經網路中連接神經元的可學習參數。

### Word Embedding (詞嵌入)
將詞彙映射到連續向量空間的技術，如 Word2Vec、GloVe。

---

## Z

### Zero-Shot Learning (零樣本學習)
無需任何示例，模型直接執行新任務的能力。

---

## 相關資源

- [Hugging Face 文檔](https://huggingface.co/docs)
- [OpenAI 文檔](https://platform.openai.com/docs)
- [Anthropic 文檔](https://docs.anthropic.com)
- [本項目學習路徑](./LEARNING_PATHS.md)

---

> 📝 **貢獻**：歡迎透過 PR 補充更多術語！
