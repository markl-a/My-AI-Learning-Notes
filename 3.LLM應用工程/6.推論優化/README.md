# 推論優化 (Inference Optimization)

> 完整的 LLM 推論優化學習資源，包含理論知識、實作範例和 AI 輔助工具

## 📋 目錄

- [簡介](#簡介)
- [核心技術](#核心技術)
- [實作目錄](#實作目錄)
- [學習路徑](#學習路徑)
- [效能指標](#效能指標)

## 簡介

LLM 推論優化是將大型語言模型高效部署到生產環境的關鍵技術。本章節涵蓋從基礎理論到進階實作的完整內容。

### 為什麼需要推論優化？

- **降低成本**：減少 GPU 記憶體使用和計算資源
- **提升速度**：縮短回應時間，改善使用者體驗
- **擴展規模**：支援更大的並發請求量
- **節能減排**：降低能源消耗和碳排放

### 優化維度

1. **計算優化**：減少運算量（量化、剪枝）
2. **記憶體優化**：降低記憶體佔用（KV Cache、PagedAttention）
3. **延遲優化**：加速生成速度（Flash Attention、Speculative Decoding）
4. **吞吐量優化**：提升並發能力（批次處理、連續批次）

## 核心技術

### 1. 量化技術 (Quantization)

將模型權重和啟動值從高精度（FP32/FP16）轉換為低精度（INT8/INT4）。

#### 量化方法對比

| 方法 | 精度 | 記憶體節省 | 速度提升 | 精度損失 | 適用場景 |
|------|------|-----------|---------|---------|---------|
| FP16 | 16-bit | 50% | 1.5-2x | 極小 | 通用 |
| INT8 | 8-bit | 75% | 2-4x | 小 | 生產環境 |
| INT4 | 4-bit | 87.5% | 3-6x | 中等 | 資源受限 |
| GPTQ | 3-4 bit | 85-90% | 3-5x | 小 | 大模型壓縮 |
| AWQ | 4-bit | 87.5% | 4-6x | 極小 | 保持精度 |
| GGUF | 2-8 bit | 75-95% | 2-8x | 可調 | 本地部署 |

**詳細內容**：[1.量化技術/README.md](./1.量化技術/README.md)

### 2. KV Cache 機制

快取 Attention 機制中的 Key 和 Value，避免重複計算。

**核心概念**：
- 自迴歸生成時，已生成的 token 的 KV 可以重用
- 記憶體佔用：`batch_size × seq_length × num_layers × hidden_dim × 2`
- 權衡：記憶體使用 vs 計算時間

**詳細內容**：[2.KV-Cache/README.md](./2.KV-Cache/README.md)

### 3. Flash Attention

優化 Attention 計算的記憶體訪問模式，減少 GPU 記憶體帶寬瓶頸。

**關鍵優勢**：
- 記憶體使用：O(N) vs O(N²)
- 速度提升：2-4x（長序列更明顯）
- 無精度損失（數學等價）

**詳細內容**：[3.Flash-Attention/README.md](./3.Flash-Attention/README.md)

### 4. vLLM 與 PagedAttention

高效的 LLM 服務引擎，使用 PagedAttention 管理 KV Cache。

**核心創新**：
- 類似虛擬記憶體的分頁機制
- 接近零浪費的記憶體管理
- 連續批次處理（Continuous Batching）

**詳細內容**：[4.vLLM-部署/README.md](./4.vLLM-部署/README.md)

### 5. Speculative Decoding

使用小模型草稿 + 大模型驗證，加速解碼過程。

**原理**：
1. 小模型快速生成多個候選 token
2. 大模型並行驗證候選
3. 接受正確預測，拒絕錯誤預測

**加速比**：2-3x（無精度損失）

### 6. 其他重要技術

- **模型剪枝**：移除不重要的權重
- **知識蒸餾**：訓練小模型模仿大模型
- **動態批次**：智能組合請求以最大化 GPU 利用率
- **張量並行**：跨 GPU 分割計算

## 實作目錄

```
6.推論優化/
├── README.md                           # 本文件
├── 1.量化技術/
│   ├── README.md                       # 量化理論詳解
│   ├── requirements.txt
│   ├── 01_basic_quantization.py        # 基礎量化實作
│   ├── 02_gptq_quantization.py         # GPTQ 量化
│   ├── 03_awq_quantization.py          # AWQ 量化
│   ├── 04_gguf_quantization.py         # GGUF 格式
│   └── 05_quantization_comparison.py   # 量化方法對比
├── 2.KV-Cache/
│   ├── README.md                       # KV Cache 原理
│   ├── requirements.txt
│   ├── 01_kv_cache_basic.py            # 基礎 KV Cache
│   ├── 02_kv_cache_benchmark.py        # 效能測試
│   └── 03_paged_attention.py           # PagedAttention 演示
├── 3.Flash-Attention/
│   ├── README.md                       # Flash Attention 原理
│   ├── requirements.txt
│   ├── 01_flash_attention_basic.py     # 基礎使用
│   ├── 02_attention_comparison.py      # 與標準 Attention 對比
│   └── 03_flash_attention_v2.py        # Flash Attention 2
├── 4.vLLM-部署/
│   ├── README.md                       # vLLM 完整指南
│   ├── requirements.txt
│   ├── 01_vllm_basic.py                # 基礎部署
│   ├── 02_vllm_api_server.py           # API 服務器
│   ├── 03_vllm_benchmark.py            # 效能測試
│   └── 04_vllm_advanced.py             # 進階配置
├── 5.效能對比測試/
│   ├── README.md                       # 測試方法論
│   ├── requirements.txt
│   ├── 01_latency_benchmark.py         # 延遲測試
│   ├── 02_throughput_benchmark.py      # 吞吐量測試
│   ├── 03_memory_profiling.py          # 記憶體分析
│   └── 04_comprehensive_comparison.py  # 綜合對比
├── 6.AI輔助優化工具/
│   ├── README.md                       # AI 輔助工具說明
│   ├── requirements.txt
│   ├── 01_auto_optimizer.py            # 自動優化建議
│   ├── 02_model_selector.py            # 智能模型選擇
│   └── 03_performance_analyzer.py      # AI 效能分析
└── 7.實戰案例/
    ├── README.md                       # 實戰指南
    ├── requirements.txt
    ├── 01_chatbot_optimization.py      # 聊天機器人優化
    ├── 02_batch_processing.py          # 批次處理優化
    └── 03_production_deployment.py     # 生產環境部署
```

## 學習路徑

### 初級（1-2 週）

1. **理解基礎概念**
   - 閱讀主 README 和各子目錄的理論部分
   - 了解推論優化的重要性和基本方法

2. **量化技術入門**
   - 完成 `1.量化技術/01_basic_quantization.py`
   - 理解 INT8/FP16 量化原理

3. **KV Cache 機制**
   - 學習 `2.KV-Cache/01_kv_cache_basic.py`
   - 運行效能測試了解影響

### 中級（2-3 週）

4. **進階量化方法**
   - GPTQ、AWQ、GGUF 實作
   - 量化方法對比實驗

5. **Flash Attention**
   - 理解記憶體優化原理
   - 對比標準 Attention 效能

6. **vLLM 部署**
   - 搭建 vLLM 服務
   - 配置和調優

### 高級（3-4 週）

7. **綜合效能優化**
   - 完整的效能測試框架
   - 多維度優化對比

8. **AI 輔助工具**
   - 使用 AI 生成優化建議
   - 自動化效能分析

9. **生產環境實戰**
   - 實際場景優化案例
   - 最佳實踐總結

## 效能指標

### 關鍵指標

1. **延遲 (Latency)**
   - TTFT (Time To First Token)：首 token 延遲
   - TPOT (Time Per Output Token)：每 token 生成時間
   - E2E (End-to-End)：端到端延遲

2. **吞吐量 (Throughput)**
   - Tokens/秒：每秒生成的 token 數
   - Requests/秒：每秒處理的請求數
   - Batch 吞吐量：批次處理能力

3. **資源使用**
   - GPU 記憶體佔用
   - GPU 利用率
   - CPU/系統記憶體

4. **成本效益**
   - 每 1M token 成本
   - 每請求成本
   - TCO (Total Cost of Ownership)

### 基準測試

| 優化方法 | 記憶體節省 | 速度提升 | 精度保持 | 實施難度 |
|---------|-----------|---------|---------|---------|
| FP16 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| INT8 量化 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| INT4 量化 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| KV Cache | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Flash Attention | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| vLLM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 常見問題

### Q1: 應該選擇哪種量化方法？

**答**：根據場景選擇：
- **生產環境首選**：AWQ（精度最好）或 GPTQ
- **本地部署**：GGUF（靈活性高）
- **快速原型**：INT8 PTQ（實施最簡單）

### Q2: KV Cache 什麼時候應該禁用？

**答**：
- 短文本生成（<50 tokens）
- 記憶體極度受限時
- 單次生成場景（無對話上下文）

### Q3: vLLM vs TGI 如何選擇？

**答**：
- **vLLM**：吞吐量優先、Python 生態、研究實驗
- **TGI**：延遲優先、企業級支援、生產穩定性

### Q4: Flash Attention 的限制？

**答**：
- 需要支援的 GPU（A100/H100/RTX 3090+）
- CUDA 版本要求（11.6+）
- 某些模型架構可能不支援

## 工具與資源

### 推薦工具

- **[vLLM](https://github.com/vllm-project/vllm)**：高性能推論引擎
- **[Text Generation Inference](https://github.com/huggingface/text-generation-inference)**：HuggingFace 推論服務
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**：CPU 推論優化
- **[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)**：GPTQ 量化
- **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)**：AWQ 量化
- **[Optimum](https://huggingface.co/docs/optimum)**：HuggingFace 優化工具集

### 學習資源

- [vLLM 官方文檔](https://docs.vllm.ai/)
- [Flash Attention 論文](https://arxiv.org/abs/2205.14135)
- [GPTQ 論文](https://arxiv.org/abs/2210.17323)
- [AWQ 論文](https://arxiv.org/abs/2306.00978)
- [PagedAttention 論文](https://arxiv.org/abs/2309.06180)

## 快速開始

### 環境設定

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝基礎依賴
pip install torch transformers accelerate

# 安裝特定工具（根據需要）
pip install vllm                    # vLLM
pip install auto-gptq               # GPTQ 量化
pip install autoawq                 # AWQ 量化
pip install flash-attn              # Flash Attention
```

### 第一個優化實例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 載入模型（FP16）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 啟用 KV Cache（預設啟用）
model.config.use_cache = True

# 生成
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 貢獻與反饋

如果你發現任何問題或有改進建議，歡迎提出 Issue 或 Pull Request！

## 授權

本專案採用 MIT 授權條款。

---

**下一步**：開始學習 [1.量化技術](./1.量化技術/README.md) 📚
