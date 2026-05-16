# 8. 模型部署與運維

> 章節定位:LLM 從訓練完成到「能在 production 穩定服務」的完整工程鏈條 — 推論引擎、容器化、可觀測性、成本管控、安全合規。
> 這是把模型從 research artifact 變成 product 的最後一哩路。

---

## 📌 章節地位

部署與運維 (Deployment & MLOps for LLM) 是 LLM 工程化中最被低估、實際最影響使用體驗與營運成本的環節。一個 30B 模型如果用錯推論引擎,可能 QPS 差 5-10 倍、成本差 3-5 倍。本章節從推論引擎選型開始,涵蓋到完整 production 系統的 observability 與 cost optimization。

學習建議的先決條件:
- 熟悉 Linux / Docker 基本操作
- 了解 HTTP API、gRPC、batching 概念
- 對 GPU 記憶體、KV-cache、quantization 有基本認識

---

## 📚 涵蓋範圍

### 推論引擎 (Inference Engines)

| 引擎 | 強項 | 適用場景 |
|---|---|---|
| **vLLM** | PagedAttention、Continuous batching、高 throughput | 雲端 GPU 大規模服務 |
| **SGLang** | RadixAttention、Structured output、極速首 token | 複雜 prompt / agent workflows |
| **TensorRT-LLM** | NVIDIA 原生極致優化 | H100/H200 上追求單卡極限 |
| **llama.cpp / GGUF** | CPU / 消費級 GPU、量化部署 | 邊緣裝置、個人開發 |
| **Ollama** | 開箱即用、模型管理 | 個人開發、PoC |
| **MLX (Apple)** | Apple Silicon 原生 | Mac 上的 LLM 推理 |
| **TGI (Hugging Face)** | 生態整合佳 | HF 生態系部署 |

### 量化與壓縮部署

- INT8 / INT4 量化 (GPTQ、AWQ、bitsandbytes)
- KV-cache 量化 (FP8 KV、INT4 KV)
- Speculative decoding / Medusa / EAGLE
- 模型蒸餾後的小模型部署

### 容器化與編排

- Docker for LLM (CUDA base images、multi-stage build)
- Kubernetes (KServe、Triton、Ray Serve)
- Serverless GPU (Modal、Runpod、Replicate、Banana)
- Auto-scaling 策略 (latency-based vs queue-based)

### 可觀測性 (Observability)

- **LLM-native**:Langfuse、LangSmith、Helicone、Arize Phoenix、Weave
- **General**:Prometheus + Grafana、OpenTelemetry for LLM
- **追蹤的關鍵指標**:TTFT (Time-To-First-Token)、TPS、p50/p95/p99 latency、token cost、cache hit rate、error rate

### 成本優化策略

- Prompt caching (Anthropic / OpenAI / Gemini)
- 模型路由 (small first → fallback to large)
- Batch API 折扣 (50% off)
- Token usage budget alarm
- Off-peak compute (spot instance)

### 安全與合規

- Prompt injection 防護
- PII / Secret 掃描
- Audit log 與 retention 政策
- 區域化部署 (data residency)

---

## 🗂️ 本章文件清單

### 既有文件
| 文件 | 內容重點 |
|---|---|
| [`8.1-部署環境選擇.md`](./8.1-部署環境選擇.md) | 雲 / 自建 / 邊緣的決策框架 |
| [`8.2-模型服務化與API化.md`](./8.2-模型服務化與API化.md) | API 設計、streaming、batching |
| [`8.3-系統架構維護.md`](./8.3-系統架構維護.md) | 升級、回滾、A/B test |
| [`8.4-資料安全與隱私考量.md`](./8.4-資料安全與隱私考量.md) | PII、加密、合規 |
| [`快速參考指南.md`](./快速參考指南.md) | 一頁式 cheat sheet |
| [`模型部署與運維實戰指南.md`](./模型部署與運維實戰指南.md) | end-to-end 實戰 |
| [`成本優化與Token管理.md`](./成本優化與Token管理.md) | 成本面詳述 |
| [`雲端部署策略指南.md`](./雲端部署策略指南.md) | AWS / GCP / Azure 比較 |

### 2026-05 新增 deep-dive(實戰導向)
| 文件 | 內容重點 |
|---|---|
| [`vLLM_部署實戰.md`](./vLLM_部署實戰.md) | PagedAttention / continuous batching / prefix caching、tp/PP 多卡、AWQ/GPTQ/FP8 部署、EAGLE-3 spec decoding、Prometheus 監控、3 個 production case |
| [`SGLang_TensorRT-LLM_對比.md`](./SGLang_TensorRT-LLM_對比.md) | 三引擎(vLLM/SGLang/TRT-LLM)benchmark、選型決策樹、SGLang `@function` DSL、TRT-LLM build pipeline、2026 趨勢(disagg / EAGLE-3 / CXL / FP4 / MoE) |

---

## 🔗 與其他章節的關係

- **應用側對應**:[`../../3.LLM應用工程/1.LLM 部署/`](../../3.LLM應用工程/1.LLM%20部署/) — 此章與本章主題重複,**建議未來合併或重新分工** (本章偏 infra/MLOps,應用工程偏 product integration)
- **硬體基礎**:[`../../11.AI_Hardware_Compute/`](../../11.AI_Hardware_Compute/) — 推理 infra 的底層 (GPU 選型、interconnect、power)
- **推論優化**:[`../../3.LLM應用工程/6.推論優化/`](../../3.LLM應用工程/6.推論優化/) — 演算法層的優化 (speculative decoding 等)
- **領域全景**:[`../../2024-2026_AI完整領域全景圖.md`](../../2024-2026_AI完整領域全景圖.md) #14 Harness Engineering
- **安全章節**:[`../../3.LLM應用工程/8.LLM安全與防禦/2025_LLM安全最佳實踐.md`](../../3.LLM應用工程/8.LLM安全與防禦/2025_LLM安全最佳實踐.md) — 部署時的 prompt injection / data exfiltration 防護

---

## 📈 建議學習順序

1. **概念奠基**:讀 `8.1-部署環境選擇.md` 建立決策框架
2. **動手 POC**:用 Ollama 或 vLLM 在本機跑通一個 7B 模型,觀察 TTFT / TPS
3. **API 化**:讀 `8.2-模型服務化與API化.md`,實作一個 FastAPI + vLLM 的服務
4. **可觀測性**:接 Langfuse 或 Phoenix,把指標串到 Grafana
5. **成本優化**:讀 `成本優化與Token管理.md`,實際算一筆每 1M token 的 cost breakdown
6. **規模化**:K8s + KServe / Ray Serve 的多副本部署

---

## ⚠️ 已知缺口 (Known Gaps)

- ✅ vLLM / SGLang / TRT-LLM benchmark 對比:**已補**(`SGLang_TensorRT-LLM_對比.md`)
- ✅ vLLM PagedAttention + continuous batching 教學:**已補**(`vLLM_部署實戰.md`)
- ✅ EAGLE-3 speculative decoding 整合:**已補**(`vLLM_部署實戰.md` §8;與 [`../2.文字生成與解碼策略/EAGLE3_Speculative_Decoding_實作.md`](../2.文字生成與解碼策略/EAGLE3_Speculative_Decoding_實作.md) 互補)
- ⏳ **仍缺**:完整 reference architecture(FastAPI + vLLM + Redis + Langfuse + Grafana)的可 deploy 範例(Dockerfile + docker-compose + k8s yaml + helm chart)
- ⏳ **仍缺**:disaggregated prefill / LMCache 完整 hands-on(對應 [`../../11.AI_Hardware_Compute/CXL_Disaggregated_Prefill_KV_Cache.md`](../../11.AI_Hardware_Compute/CXL_Disaggregated_Prefill_KV_Cache.md))
- ⏳ **仍缺**:multi-tenant LoRA serving(S-LoRA、Punica)實作範例
- ⚠️ 與 `3.LLM應用工程/1.LLM 部署/` 主題重疊(下次大重構候選合併目標)

歡迎透過 PR 補充上述缺口。

---

## 🎯 學習產出檢核

讀完本章後,你應該能回答:

- [ ] vLLM 的 PagedAttention 為什麼能大幅提升 throughput?
- [ ] SGLang 的 RadixAttention 在什麼場景下 dominate vLLM?
- [ ] TTFT 與 TPS 哪個對 chat UX 更重要?如何分別優化?
- [ ] INT4 量化會在哪些任務上明顯掉分?如何 mitigate?
- [ ] Prompt caching 如何降低 75-90% 成本?哪些 prompt 結構才能命中?
- [ ] 一個 production LLM 服務,你會監控哪 10 個指標?
