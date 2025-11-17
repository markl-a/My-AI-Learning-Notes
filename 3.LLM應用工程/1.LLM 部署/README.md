# LLM 部署與運行基礎

## 目錄
1. [LLM 部署模式概述](#11-llm-部署模式概述)
2. [開源模型 vs 專有模型](#12-開源模型-vs-專有模型)
3. [實用工具與框架介紹](#13-實用工具與框架介紹)
4. [實作示例](#14-實作示例)
5. [部署策略選擇指南](#15-部署策略選擇指南)
6. [效能監控與優化](#16-效能監控與優化)

---

## 1.1 LLM 部署模式概述

### API 模式

**定義**：透過雲端服務商提供的 API 介面使用 LLM，無需自行管理模型。

**主流服務商**：
- **OpenAI API**：GPT-4、GPT-4o、GPT-4o-mini
- **Anthropic API**：Claude 3.5 Sonnet、Claude 3 Opus、Claude 3 Haiku
- **Google AI**：Gemini 1.5 Pro、Gemini 1.5 Flash
- **Cohere**：Command R+、Command R
- **Azure OpenAI Service**：企業級 OpenAI 模型託管

**優勢**：
- ✅ 零基礎設施成本
- ✅ 快速上線（幾分鐘內可開始使用）
- ✅ 自動擴展
- ✅ 持續更新至最新模型
- ✅ 無需 GPU 資源

**劣勢**：
- ❌ 按使用量計費（長期成本可能較高）
- ❌ 資料需傳送到外部服務
- ❌ 受限於服務商的服務條款
- ❌ 網路延遲
- ❌ 依賴外部服務可用性

**成本估算**（2025年1月）：
| 模型 | 輸入（每百萬 tokens） | 輸出（每百萬 tokens） | 適用場景 |
|------|---------------------|---------------------|---------|
| GPT-4o | $2.50 | $10.00 | 複雜推理、長文本 |
| GPT-4o-mini | $0.15 | $0.60 | 一般任務、高頻呼叫 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 程式碼生成、分析 |
| Gemini 1.5 Pro | $1.25 | $5.00 | 多模態、長上下文 |

**使用範例**：
```python
from openai import OpenAI

# 初始化客戶端
client = OpenAI(api_key="your-api-key")

# 呼叫 API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一個專業的技術助理。"},
        {"role": "user", "content": "解釋什麼是 Transformer 架構"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

### 雲端部署模式

**定義**：在雲端平台上自行部署和管理 LLM，擁有完整控制權。

**主流雲端平台**：

#### AWS（Amazon Web Services）
- **EC2 GPU 執行個體**：p3、p4、g5 系列
- **SageMaker**：託管式機器學習平台
- **Bedrock**：AWS 的 LLM 服務
- **EKS**：容器化部署

**推薦配置**：
```yaml
執行個體類型: g5.12xlarge
GPU: 4x NVIDIA A10G (24GB VRAM each)
vCPU: 48
記憶體: 192 GB
成本: ~$5.67/小時（按需定價）
```

#### GCP（Google Cloud Platform）
- **Compute Engine**：GPU 虛擬機器
- **Vertex AI**：託管式 ML 平台
- **GKE**：Kubernetes 部署
- **TPU**：Google 的專用 AI 晶片

**推薦配置**：
```yaml
執行個體類型: a2-highgpu-4g
GPU: 4x NVIDIA A100 (40GB VRAM each)
vCPU: 48
記憶體: 680 GB
成本: ~$11.07/小時（按需定價）
```

#### Azure
- **Azure Virtual Machines**：NC、ND、NV 系列
- **Azure Machine Learning**：託管式 ML
- **Azure OpenAI Service**：OpenAI 模型的 Azure 版本
- **AKS**：Azure Kubernetes Service

**部署範例（使用 vLLM 在 AWS EC2）**：
```bash
# 1. 啟動 EC2 執行個體（以 g5.12xlarge 為例）

# 2. 安裝依賴
sudo apt update
sudo apt install -y python3-pip nvidia-driver-535

# 3. 安裝 vLLM
pip install vllm

# 4. 部署模型
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```

**優勢**：
- ✅ 完整控制權
- ✅ 資料留在自己的雲端環境
- ✅ 可自訂模型和配置
- ✅ 更好的資料隱私
- ✅ 可選擇特定區域部署

**劣勢**：
- ❌ 需要管理基礎設施
- ❌ 需要 DevOps 技能
- ❌ 固定成本（即使低使用率）
- ❌ 需要監控和維護

---

### 本地端（Local）部署

**定義**：在本地機器或內部資料中心運行 LLM。

**硬體需求**（以 7B 模型為例）：

**最低需求**：
```
GPU: NVIDIA RTX 3060 (12GB VRAM)
RAM: 16GB
儲存: 50GB SSD
模型: 7B 量化模型（4-bit）
```

**推薦配置**：
```
GPU: NVIDIA RTX 4090 (24GB VRAM) 或 A100 (40GB)
RAM: 32GB+
儲存: 1TB NVMe SSD
模型: 7B-13B 模型（8-bit 或 16-bit）
```

**企業級配置**：
```
GPU: 4-8x NVIDIA A100 (80GB VRAM)
RAM: 512GB+
儲存: 4TB+ NVMe SSD
網路: 10Gbps+
模型: 70B+ 模型（全精度或 8-bit）
```

**模型大小與記憶體需求**：
| 模型大小 | FP16 | 8-bit | 4-bit | 推薦 GPU |
|---------|------|-------|-------|---------|
| 7B | 14GB | 7GB | 3.5GB | RTX 3060 12GB+ |
| 13B | 26GB | 13GB | 6.5GB | RTX 4090 24GB+ |
| 30B | 60GB | 30GB | 15GB | A100 40GB+ |
| 70B | 140GB | 70GB | 35GB | 2x A100 40GB+ |

**本地端部署工具**：

#### 1. **Ollama**（最簡單）
```bash
# 安裝 Ollama（macOS/Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 下載並運行模型
ollama run llama2:7b

# Python 使用
import ollama

response = ollama.chat(model='llama2:7b', messages=[
  {
    'role': 'user',
    'content': '什麼是機器學習？',
  },
])
print(response['message']['content'])
```

#### 2. **LM Studio**（圖形介面）
- 下載：https://lmstudio.ai/
- 特點：
  - 圖形化介面，無需程式碼
  - 支援多種模型格式（GGUF、GGML）
  - 內建模型商店
  - 本地 API 伺服器

#### 3. **llama.cpp**（高效能）
```bash
# 編譯 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 下載模型（GGUF 格式）
# 從 Hugging Face 下載，例如：
# https://huggingface.co/TheBloke/Llama-2-7B-GGUF

# 運行模型
./main -m ./models/llama-2-7b.Q4_K_M.gguf \
       -p "什麼是深度學習？" \
       -n 512
```

#### 4. **Text Generation WebUI**（Oobabooga）
```bash
# 安裝
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
bash start_linux.sh  # 或 start_windows.bat, start_macos.sh

# 啟動後訪問 http://localhost:7860
```

**優勢**：
- ✅ 完全離線運行
- ✅ 零 API 成本
- ✅ 完整資料隱私
- ✅ 低延遲（無網路往返）
- ✅ 可完全自訂

**劣勢**：
- ❌ 需要硬體投資
- ❌ 受限於本地算力
- ❌ 需要技術知識
- ❌ 無法輕易擴展

---

## 1.2 開源模型 vs. 專有模型

### 專有模型（Proprietary Models）

**代表模型**：

#### OpenAI 系列
- **GPT-4o**：最新旗艦模型，多模態
- **GPT-4o-mini**：小型高效版本
- **O1 系列**：推理優化模型

**特點**：
- 頂尖性能
- 多模態支援（文字、圖像、語音）
- 128K token 上下文
- 嚴格的安全對齊

#### Anthropic Claude 系列
- **Claude 3.5 Sonnet**：平衡性能與成本
- **Claude 3 Opus**：最強推理能力
- **Claude 3 Haiku**：快速且經濟

**特點**：
- 長上下文支援（200K tokens）
- 優秀的程式碼能力
- Constitutional AI 安全機制
- 更少的幻覺問題

#### Google Gemini
- **Gemini 1.5 Pro**：超長上下文（1M tokens）
- **Gemini 1.5 Flash**：快速推理

**特點**：
- 原生多模態架構
- 超長上下文窗口
- Google 搜尋整合

**專有模型比較**：
| 模型 | 上下文長度 | 強項 | 價格層級 |
|------|-----------|------|---------|
| GPT-4o | 128K | 多模態、通用 | 高 |
| Claude 3.5 Sonnet | 200K | 程式碼、分析 | 中高 |
| Gemini 1.5 Pro | 1M | 長文本、多模態 | 中 |

---

### 開源模型（Open Source Models）

**代表模型**：

#### Meta Llama 系列
- **Llama 3.1（405B）**：最大開源模型
- **Llama 3.1（70B）**：高性能選擇
- **Llama 3.1（8B）**：邊緣部署

**特點**：
- 商業友好授權
- 多語言支援
- 社群生態豐富
- 可完全自訂

**下載與使用**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入 Llama 3.1 8B
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 推論
messages = [
    {"role": "system", "content": "你是一個有幫助的助理。"},
    {"role": "user", "content": "解釋量子計算的基本概念"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

#### Mistral AI 系列
- **Mixtral 8x7B**：混合專家模型（MoE）
- **Mistral 7B**：高效 7B 模型
- **Mistral Small/Large**：專有版本

**特點**：
- 優秀的性能/成本比
- Apache 2.0 授權
- 滑動窗口注意力機制
- 支援長上下文

#### Qwen 系列（阿里巴巴）
- **Qwen2.5（72B）**：最新旗艦
- **Qwen2.5-Coder**：程式碼專用
- **Qwen2.5-Math**：數學專用

**特點**：
- 優秀的中文支援
- 多種專業版本
- 開源且免費

#### Yi 系列（零一萬物）
- **Yi-34B**：中英雙語
- **Yi-VL**：視覺語言模型

**特點**：
- 中文能力強
- 商業友好授權

#### 其他重要開源模型
- **Phi-3/4（Microsoft）**：小型高效模型
- **Gemma（Google）**：輕量級模型
- **Falcon**：開源大型模型
- **BLOOM**：多語言模型

**開源模型比較**：
| 模型 | 參數量 | 授權 | 中文能力 | 推薦用途 |
|------|-------|------|---------|---------|
| Llama 3.1 | 8B-405B | Llama 3 | 良好 | 通用 |
| Mixtral | 8x7B | Apache 2.0 | 一般 | 高效推理 |
| Qwen2.5 | 0.5B-72B | Apache 2.0 | 優秀 | 中文應用 |
| Phi-4 | 14B | MIT | 一般 | 邊緣部署 |

---

### 如何選擇？

**決策樹**：
```
需要最頂尖的性能？
├─ 是 → 專有模型（GPT-4o, Claude 3.5 Sonnet）
└─ 否 ↓
    資料隱私是首要考量？
    ├─ 是 → 開源模型 + 本地部署
    └─ 否 ↓
        預算有限？
        ├─ 是 → 開源模型或小型專有模型（GPT-4o-mini）
        └─ 否 ↓
            需要自訂和微調？
            ├─ 是 → 開源模型
            └─ 否 → 專有模型 API
```

**使用場景建議**：

| 場景 | 推薦方案 | 原因 |
|------|---------|------|
| 快速原型開發 | OpenAI API | 快速、簡單、品質高 |
| 企業級應用（高量） | 開源 + 自建 | 長期成本低、資料可控 |
| 內部工具 | Ollama + 開源模型 | 零 API 成本、隱私保護 |
| 生產環境（中量） | Claude API | 品質穩定、成本合理 |
| 教育/研究 | 開源模型 | 免費、可研究內部機制 |
| 邊緣裝置 | Phi-3/4, Llama 3.1 8B | 小巧高效 |

---

## 1.3 實用工具與框架介紹

### 本地運行工具

#### LM Studio

**特點**：
- 🎨 友善的圖形介面
- 📦 內建模型市場
- 🔌 支援 OpenAI 相容 API
- 💻 跨平台（Windows, Mac, Linux）

**使用流程**：
1. 下載安裝：https://lmstudio.ai/
2. 在模型市場搜尋模型（如 "Llama 3.1"）
3. 下載模型（自動選擇適合的量化版本）
4. 載入模型並開始對話
5. 選擇性啟動本地 API 伺服器

**支援格式**：
- GGUF（推薦）
- GGML

---

#### Ollama

**特點**：
- ⚡ 極簡的命令列工具
- 🐳 Docker 風格的使用體驗
- 🔄 自動管理模型
- 🖥️ 內建 API 伺服器

**安裝**：
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows（使用 WSL 或下載安裝程式）
```

**常用命令**：
```bash
# 運行模型（自動下載）
ollama run llama3.1:8b

# 列出已下載的模型
ollama list

# 刪除模型
ollama rm llama3.1:8b

# 啟動 API 服務（預設在 11434 埠）
ollama serve

# 從 Modelfile 創建自訂模型
ollama create my-model -f Modelfile
```

**Modelfile 範例**：
```dockerfile
# Modelfile
FROM llama3.1:8b

# 設定溫度參數
PARAMETER temperature 0.8

# 設定系統提示
SYSTEM """
你是一個專業的 Python 程式設計助理。
你的回答應該簡潔、準確，並包含實際可執行的程式碼範例。
"""
```

**Python 整合**：
```python
import ollama

# 簡單對話
response = ollama.chat(model='llama3.1:8b', messages=[
    {
        'role': 'user',
        'content': '寫一個 Python 函數計算斐波那契數列',
    },
])
print(response['message']['content'])

# 串流回應
stream = ollama.chat(
    model='llama3.1:8b',
    messages=[{'role': 'user', 'content': '解釋遞迴'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

---

#### llama.cpp

**特點**：
- 🚀 高效能 C++ 實作
- 💾 支援 CPU 推論（無需 GPU）
- 📱 支援行動裝置（iOS, Android）
- 🔧 高度可自訂

**編譯與使用**：
```bash
# 克隆倉庫
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 編譯（CPU 版本）
make

# 編譯（CUDA 版本）
make LLAMA_CUDA=1

# 下載模型（GGUF 格式）
# 從 HuggingFace 下載，例如：
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

# 運行推論
./main \
    -m ./models/llama-2-7b-chat.Q4_K_M.gguf \
    -p "請解釋什麼是深度學習" \
    -n 512 \
    -c 2048 \
    --temp 0.7

# 啟動伺服器模式
./server \
    -m ./models/llama-2-7b-chat.Q4_K_M.gguf \
    -c 2048 \
    --host 0.0.0.0 \
    --port 8080
```

**量化格式說明**：
| 格式 | 位元數 | 檔案大小（7B） | 品質 | 速度 |
|------|-------|--------------|------|------|
| Q2_K | ~2.5 bit | ~2.5GB | 低 | 最快 |
| Q4_K_M | ~4.5 bit | ~4GB | 中 | 快 |
| Q5_K_M | ~5.5 bit | ~5GB | 中高 | 中 |
| Q8_0 | 8 bit | ~7GB | 高 | 中慢 |
| F16 | 16 bit | ~14GB | 最高 | 慢 |

---

#### Hugging Face Spaces

**特點**：
- ☁️ 免費雲端部署
- 🎨 支援多種框架（Gradio, Streamlit）
- 🤝 社群分享
- 🔄 自動部署（Git 整合）

**部署範例（Gradio + Transformers）**：
```python
# app.py
import gradio as gr
from transformers import pipeline

# 載入模型
pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

def generate_response(message, history):
    response = pipe(
        message,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )
    return response[0]['generated_text']

# 創建 Gradio 介面
demo = gr.ChatInterface(
    generate_response,
    title="Llama 2 Chat",
    description="與 Llama 2 7B 對話"
)

if __name__ == "__main__":
    demo.launch()
```

**部署步驟**：
1. 創建 Hugging Face 帳號
2. 創建新的 Space
3. 上傳 `app.py` 和 `requirements.txt`
4. 自動部署並獲得公開 URL

---

### 雲端推論框架

#### vLLM

**特點**：
- ⚡ 高吞吐量（PagedAttention）
- 🔄 連續批次處理
- 📊 2-4x 速度提升
- 🔌 OpenAI 相容 API

**安裝與使用**：
```bash
# 安裝
pip install vllm

# 啟動伺服器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 1 \
    --dtype auto

# Python 使用
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(
    ["解釋什麼是 Transformer", "什麼是注意力機制"],
    sampling_params
)

for output in outputs:
    print(output.outputs[0].text)
```

---

#### Text Generation Inference (TGI)

Hugging Face 開發的高效能推論伺服器。

**特點**：
- 🚀 優化的推論引擎
- 🔄 張量並行支援
- 📊 動態批次處理
- 🐳 Docker 部署

**使用範例**：
```bash
# Docker 運行
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf

# Python 客戶端
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")
response = client.text_generation(
    "解釋什麼是機器學習",
    max_new_tokens=100
)
print(response)
```

---

## 1.4 實作示例

### 示例 1：使用 Hugging Face Transformers 本地載入模型

```python
"""
本地載入並推論 Llama 2 7B Chat 模型
需求：至少 16GB RAM，12GB+ VRAM
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import torch

# 配置 4-bit 量化以降低記憶體需求
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 載入模型和分詞器
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 創建對話管線
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 對話範例
messages = [
    {"role": "system", "content": "你是一個有幫助的 AI 助理。"},
    {"role": "user", "content": "解釋什麼是 Transformer 架構，並說明其重要性。"}
]

# 生成回應
outputs = chat_pipeline(
    messages,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    return_full_text=False
)

print("AI 回應：")
print(outputs[0]['generated_text'])

# 效能測量
import time

prompts = [
    "什麼是深度學習？",
    "解釋 GPU 的作用",
    "什麼是自然語言處理？"
]

start = time.time()
for prompt in prompts:
    result = chat_pipeline(
        [{"role": "user", "content": prompt}],
        max_new_tokens=100
    )
end = time.time()

print(f"\n處理 {len(prompts)} 個查詢耗時：{end - start:.2f} 秒")
print(f"平均每個查詢：{(end - start) / len(prompts):.2f} 秒")
```

---

### 示例 2：使用 OpenAI API 生成文字

```python
"""
使用 OpenAI API 進行各種任務
展示不同參數對輸出的影響
"""

from openai import OpenAI
import json

# 初始化客戶端
client = OpenAI(api_key="your-api-key-here")

# 範例 1：基本文字生成
def basic_generation():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "你是一個專業的技術作家，擅長將複雜概念解釋得清晰易懂。"
            },
            {
                "role": "user",
                "content": "用 3 個段落解釋什麼是 RAG（檢索增強生成）"
            }
        ],
        temperature=0.7,
        max_tokens=500
    )

    print("=== 基本生成 ===")
    print(response.choices[0].message.content)
    print(f"\nTokens 使用：{response.usage.total_tokens}")
    print(f"成本估算：${response.usage.total_tokens / 1_000_000 * 0.60:.6f}")

# 範例 2：結構化輸出（JSON 模式）
def structured_output():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "列出 3 個常見的機器學習演算法，包括名稱、類型和主要用途"
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.5
    )

    print("\n=== 結構化輸出 ===")
    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2, ensure_ascii=False))

# 範例 3：串流回應
def streaming_response():
    print("\n=== 串流回應 ===")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "寫一首關於人工智慧的短詩"}
        ],
        stream=True,
        temperature=0.9
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print()  # 換行

# 範例 4：多輪對話
def multi_turn_conversation():
    print("\n=== 多輪對話 ===")

    messages = [
        {"role": "system", "content": "你是一個 Python 程式設計專家。"}
    ]

    # 第一輪
    messages.append({
        "role": "user",
        "content": "如何在 Python 中讀取 CSV 檔案？"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    assistant_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_response})
    print(f"AI: {assistant_response}\n")

    # 第二輪（延續上下文）
    messages.append({
        "role": "user",
        "content": "如果檔案很大，有更好的方法嗎？"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    print(f"AI: {response.choices[0].message.content}")

# 範例 5：溫度參數比較
def temperature_comparison():
    print("\n=== 溫度參數比較 ===")

    prompt = "解釋什麼是神經網路"

    for temp in [0.0, 0.5, 1.0]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100
        )

        print(f"\nTemperature = {temp}:")
        print(response.choices[0].message.content[:200] + "...")

# 執行所有範例
if __name__ == "__main__":
    basic_generation()
    structured_output()
    streaming_response()
    multi_turn_conversation()
    temperature_comparison()
```

---

### 示例 3：量化模型比較實驗

```python
"""
比較不同量化等級對模型效能和品質的影響
需要：transformers, bitsandbytes, torch
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
import psutil
import os

def get_memory_usage():
    """獲取當前記憶體使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_quantization(model_name, quant_type):
    """測試特定量化配置"""

    print(f"\n{'='*60}")
    print(f"測試配置：{quant_type}")
    print('='*60)

    # 記錄初始記憶體
    mem_before = get_memory_usage()

    # 配置量化
    if quant_type == "16-bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif quant_type == "8-bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:  # 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 記錄載入後記憶體
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    # 測試推論速度
    prompt = "解釋什麼是量子計算"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 預熱
    model.generate(**inputs, max_new_tokens=50)

    # 正式測試
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    end = time.time()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 顯示結果
    print(f"記憶體使用：{mem_used:.2f} MB")
    print(f"推論時間：{end - start:.2f} 秒")
    print(f"Tokens/秒：{100 / (end - start):.2f}")
    print(f"\n生成範例（前 200 字元）：")
    print(response[:200] + "...")

    # 清理
    del model
    torch.cuda.empty_cache()

    return {
        "quant_type": quant_type,
        "memory_mb": mem_used,
        "time_s": end - start,
        "tokens_per_s": 100 / (end - start)
    }

# 執行比較
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    results = []
    for quant in ["4-bit", "8-bit", "16-bit"]:
        result = test_quantization(model_name, quant)
        results.append(result)

    # 總結比較
    print("\n" + "="*60)
    print("總結比較")
    print("="*60)
    print(f"{'量化類型':<15} {'記憶體(MB)':<15} {'速度(tok/s)':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['quant_type']:<15} {r['memory_mb']:<15.2f} {r['tokens_per_s']:<15.2f}")
```

---

## 1.5 部署策略選擇指南

### 成本分析

**API 服務 vs 自建部署成本比較**（以每月 100 萬次請求為例）：

| 方案 | 初期成本 | 每月成本 | 總成本（12個月） | 適用規模 |
|------|---------|---------|----------------|---------|
| OpenAI API | $0 | $300-600 | $3,600-7,200 | 小中型 |
| Cloud VM (AWS g5.xlarge) | $0 | $700 | $8,400 | 中型 |
| 自建伺服器（1x RTX 4090） | $2,500 | $100（電費） | $3,700 | 中大型 |
| 自建叢集（4x A100） | $40,000 | $300 | $43,600 | 大型 |

**損益平衡點分析**：
- API vs 雲端 VM：約 50 萬次請求/月
- API vs 自建（RTX 4090）：約 6 個月
- 雲端 VM vs 自建：約 12 個月

---

## 1.6 效能監控與優化

### 關鍵指標

**推論效能指標**：
1. **首 Token 延遲（Time to First Token, TTFT）**：從請求到第一個 token 的時間
2. **每 Token 延遲（Time per Output Token, TPOT）**：生成每個 token 的時間
3. **總延遲（Total Latency）**：完成整個請求的時間
4. **吞吐量（Throughput）**：每秒處理的請求數或 tokens 數

**監控範例**：
```python
"""
LLM 推論效能監控
"""

import time
from typing import List, Dict
import statistics

class LLMMetrics:
    """LLM 效能指標收集器"""

    def __init__(self):
        self.metrics: List[Dict] = []

    def measure_inference(self, llm_func, prompt: str):
        """測量單次推論的各項指標"""

        # 記錄開始時間
        start_time = time.time()

        # 執行推論（假設返回生成的文字）
        response, token_count = llm_func(prompt)

        # 記錄結束時間
        end_time = time.time()

        # 計算指標
        total_time = end_time - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0

        metric = {
            'prompt_length': len(prompt),
            'response_length': token_count,
            'total_time': total_time,
            'tokens_per_second': tokens_per_second,
            'timestamp': start_time
        }

        self.metrics.append(metric)
        return metric

    def get_summary(self) -> Dict:
        """獲取統計摘要"""

        if not self.metrics:
            return {}

        return {
            'total_requests': len(self.metrics),
            'avg_latency': statistics.mean(m['total_time'] for m in self.metrics),
            'p50_latency': statistics.median(m['total_time'] for m in self.metrics),
            'p95_latency': statistics.quantiles(
                [m['total_time'] for m in self.metrics],
                n=20
            )[18] if len(self.metrics) >= 20 else None,
            'avg_tokens_per_sec': statistics.mean(m['tokens_per_second'] for m in self.metrics),
            'total_tokens': sum(m['response_length'] for m in self.metrics)
        }

    def print_report(self):
        """列印效能報告"""

        summary = self.get_summary()

        print("\n" + "="*60)
        print("LLM 效能報告")
        print("="*60)
        print(f"總請求數：{summary['total_requests']}")
        print(f"平均延遲：{summary['avg_latency']:.3f} 秒")
        print(f"中位數延遲：{summary['p50_latency']:.3f} 秒")
        if summary['p95_latency']:
            print(f"P95 延遲：{summary['p95_latency']:.3f} 秒")
        print(f"平均速度：{summary['avg_tokens_per_sec']:.2f} tokens/秒")
        print(f"總生成 tokens：{summary['total_tokens']}")
        print("="*60)

# 使用範例
metrics = LLMMetrics()

# 假設的 LLM 函數
def mock_llm(prompt):
    import random
    time.sleep(random.uniform(0.5, 2.0))  # 模擬處理時間
    token_count = random.randint(50, 200)
    return "response text", token_count

# 收集指標
for _ in range(10):
    metrics.measure_inference(mock_llm, "測試提示")

# 顯示報告
metrics.print_report()
```

---

## 參考資源

### 官方文件
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [OpenAI API 文件](https://platform.openai.com/docs)
- [vLLM 文件](https://docs.vllm.ai/)
- [Ollama 文件](https://ollama.ai/docs)

### 模型下載
- [Hugging Face Models](https://huggingface.co/models)
- [Ollama Library](https://ollama.ai/library)
- [TheBloke GGUF Models](https://huggingface.co/TheBloke)

### 社群資源
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [Hugging Face Discord](https://huggingface.co/join/discord)
- [LLM 效能排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

**最後更新**：2025年1月
