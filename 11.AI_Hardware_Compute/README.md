# 11. AI Hardware / Compute Stack 全景 (2024-2026)

> 對應 [2024-2026 AI 完整領域全景圖](../2024-2026_AI完整領域全景圖.md) #1 + #2
> 12 條軸線拆解「token 從哪裡被生產出來」這個問題。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **DL / Transformer 基礎**(對應 repo:[1.從AI到LLM基礎/4.DL](../1.從AI到LLM基礎/4.DL/README.md))
> 2. **量化、KV cache 等模型壓縮概念**(對應 repo:[2.深入LLM模型工程與LLM運維/7.模型壓縮與優化](../2.深入LLM模型工程與LLM運維/7.模型壓縮與優化/))
> 3. **基本 GPU 程式設計直覺**(CUDA / kernel / memory hierarchy;若 repo 內無,先看:[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 前三章)
>
> 初次接觸請先讀 [全景圖 #1 + #2](../2024-2026_AI完整領域全景圖.md) 對應章節建立硬體棧地圖,再回頭看本檔的逐節 SOTA。
>
> **延伸 / 反向連結**:[3.LLM應用工程/6.推論優化](../3.LLM應用工程/6.推論優化/) | [Neuromorphic_Computing.md](./Neuromorphic_Computing.md)

> **⚠️ 鮮度與可信度說明 / Freshness & Reliability**
> 本章涉及 **GPU 路線圖、HBM 容量、機櫃功耗、TPU/MI300/Apple AI 晶片時程、Rubin/Blackwell 發布月份、各 frontier datacenter 容量** 等高度具體的硬體數字。這些資訊混合三類來源:
> 1. **NVIDIA / AMD / TSMC / SK Hynix 官方規格表 / 財報**(高信任度)
> 2. **媒體報導 / 供應鏈傳聞(Bloomberg / The Information / SemiAnalysis 等)**(屬「報導」非「事實」)
> 3. **AI agent 整理 + 我手動驗證**(可能有誤標,尤其 Apple AI server 晶片量產時程、Rubin Ultra 細節、Stargate 容量數字)
>
> 任何要做硬體選型或投資決策的數字(規格、容量、價格、上市月份、產能爬坡),**請以原廠官方為準**。本章對「Hopper → Blackwell → Rubin 的指數爬坡」這類 *結構性趨勢* 較高信心,對「2026 H2 量產 X 顆」這類 *時點預測* 較低信心。

---

## 1. NVIDIA GPU 路線:從 Hopper 到 Rubin 的指數爬坡

Hopper (H100/H200, 4nm, HBM3/3E 80–141GB) 仍是 2024–2025 訓練主力。**Blackwell (B100/B200)** 於 2024 量產,雙 die 接 NV-HBI 10TB/s 視為單顆 GPU,FP4 算力翻倍。**GB200 NVL72** 為旗艦機櫃:72 顆 B200 + 36 顆 Grace CPU,以第 5 代 NVLink (1.8 TB/s/GPU) 全互連、13.4 TB unified HBM、1.44 ExaFLOPS FP4、120kW 液冷;對比同數量 H100 推理快 30x、能耗降 25x。

**突破**:2024 Q4 推出 **Blackwell Ultra (B300/GB300)**,HBM3E 拉到 288GB,FP4 增 50%。**Vera Rubin (2026 H2)** 採 TSMC 3nm + HBM4 (288GB, 13TB/s),Rubin NVL144 機櫃達 3.6 ExaFLOPS FP4;**Rubin Ultra (2027)** 與 **Feynman (2028)** 矽光子互連入場。**SHARP** 在 NVSwitch 內做 in-network all-reduce,讓 collectives 不再吃 GPU 算力。

**選型**:訓練千億參數以上 → GB200/GB300 NVL72;單機推理 70B 模型 → 8×H200 或 8×B200;單卡開發 → L40S/RTX 6000 Ada。FP4 訓練要搭 Transformer Engine 自動 micro-scaling。

**關鍵詞**:HBM3E、NVLink 5、NVSwitch、SHARP、FP4 micro-scaling、Grace CPU、NVL72、AI Factory

## 2. AMD GPU:MI355X 終於成為 Plan B

**MI300X** (192GB HBM3) 與 **MI325X** (256GB HBM3E) 已切入 Microsoft / Meta / Oracle 推理機隊。**MI350X / MI355X**(Advancing AI 2025 發表)轉 CDNA4 架構、TSMC N3,搭 **288GB HBM3E、8TB/s**;MI350X 1000W 風冷、MI355X 1400W 液冷;新增 MXFP6/MXFP4 datatype。AMD 宣稱對 MI300X 4x AI 算力、35x 推理,並在 like-for-like 推理略勝 H200 1.3x。

**突破**:**UALink Consortium**(AMD/Intel/Google/Meta/Microsoft)制定開放 scale-up 互連,目標取代 NVLink 鎖定。2026 路線 MI400 走 UALoE72(72 卡 pod),2027 MI500 上 UAL256。**ROCm 7** 已對 vLLM/SGLang/PyTorch 2 torch.compile 達到基本可用,但 Triton autotuning 成熟度仍落後 CUDA 一個身段。

**選型**:推理工作負載且模型權重 >180GB → MI300X/MI355X 性價比最好;CI/CD 必須兩家 GPU 都跑,否則某天 ROCm 編譯失敗你會被綁死。

## 3. Google TPU:從 Trillium 到 Ironwood

**TPU v5e**(推理性價比)/**v5p**(訓練)為 Gemini 1.5 主力。**TPU v6 Trillium** (2024) 同 N5 製程下 FLOPs 翻倍、功耗下降。**TPU v7 Ironwood** (2025 GA) 是首款原生 FP8 設計,單晶片性能 10x TPU v5p、4x Trillium,旗艦超級 pod 達 **9,216 chip**(144 cubes × 64 chips),透過 **3D torus + OCS (Optical Circuit Switch)** 動態重組拓撲。

**突破**:**OCS 光學環路交換**讓 Google 可繞過故障 chip、動態重塑 pod 形狀;**Pathways** runtime 做 elastic training + multi-host inference,RL workload 在 TPU 上效率遠勝 GPU。

**選型**:在 GCP 上的 PyTorch 用 `torch_xla` 或 JAX;JAX + Flax/NNX 是 TPU 一等公民。長 context、批次大、需高 HBM 容量 → TPU 比 GPU 划算。

## 4. 大廠自研晶片:Trainium 3 / Maia 200 / MTIA 3

**AWS Trainium2** (2024) 16 chip/instance、96GB HBM/chip;**Trainium3** (2025/12) TSMC 3nm,2.52 PFLOPS FP8、144GB HBM3E、4.9 TB/s,Project Rainier 部署近 50 萬顆,專供 Anthropic 訓練 Claude。**Microsoft Maia 100** (2023, 5nm) 已用於 Copilot;**Maia 200** (2026/1) 3nm、216GB HBM3E、7TB/s、per-dollar +30%。**Meta MTIA v2** 鎖定推薦系統,2026–2027 規劃 **MTIA 300/400/450/500** 四代。

ASIC 對 NVIDIA 形成 CAGR 44.6% 的替代壓力,依 SemiAnalysis 等 2025 估算,推理已佔 hyperscaler AI compute 多數比重(各家統計口徑不一,常見估算 60-67%),大廠財務動機是把自家內部 workload 從 GPU 遷出。TSMC 包辦 92% 先進製程的 hyperscaler ASIC。

## 5. 推理專用 ASIC:速度物理學的重寫

- **Groq LPU**:超大 SRAM(無 HBM)、deterministic dataflow,Llama 3 70B 達 ~275 tok/s,GPT-OSS 120B/Llama 4 Maverick >2500 tok/s。
- **Cerebras WSE-3**:整片 12-inch wafer = 1 顆 chip,21+ PB SRAM 緊鄰計算單元,Llama 3.1-405B 突破 1000 tok/s,LLM 推論速度王。
- **SambaNova RDU**:Reconfigurable Dataflow + 三層記憶體,專攻企業內部多模型 serving。
- **Etched Sohu**:**transformer-only ASIC**,只跑 transformer 但 8 卡盒可達 50 萬 tok/s on Llama 70B(等同 160×H100)。賭注是 transformer 不會被取代。
- **Tenstorrent / Lightmatter**:RISC-V + 光互連,2026 才是檢驗點。

**突破**:打破「memory wall」三條路:把 SRAM 做大(Cerebras)、把 model 編譯進 SRAM(Groq)、把架構固化進矽(Etched)。代價是失去 generality。

**選型**:固定模型 + 極致 TTFT/TPOT → Groq/Cerebras API;業務需要切換模型架構 → 仍 GPU。

## 6. 記憶體:HBM 才是真正的瓶頸

HBM3E 8-Hi/12-Hi 已量產(36GB/stack)。**SK Hynix 62% 市佔**領跑、Micron 21% 超越 Samsung 17%。NVIDIA Blackwell、AMD MI355X、AWS Trainium3 全部優先綁定 SK Hynix。HBM3E 1.2 TB/s/stack。

**突破**:**HBM4** spec 拉到 2048-bit 寬介面、>2 TB/s/stack,首發於 Rubin (2026 H2);SK Hynix 完成開發、Micron 12-layer 36GB。**CXL 3.x** 提供記憶體池化,在推理場景把 KV cache 移到 CXL 模組可省 HBM,但延遲較高。

**心法**:推理瓶頸 90% 在 memory bandwidth(arithmetic intensity 低),不在 FLOPs。看選型先看 GB 與 TB/s,別只看 TFLOPS。

## 7. 互連與網路:三層戰爭

- **Scale-up**(機櫃內 GPU 互連):NVLink 5 (1.8 TB/s) vs UALink vs Broadcom SUE。
- **Scale-out**(跨機櫃):InfiniBand NDR (400G)/XDR (800G) 仍是 NVIDIA 標配;**Ultra Ethernet 1.0** (2025/6 發布 560+ 頁 spec) 由 AMD/Broadcom/Cisco/Meta/Microsoft 推動,提供 RDMA、congestion control、path diversity。
- **DCI**(跨資料中心):光互連、coherent optics。

**突破**:Dell'Oro 預測 **2027 Ethernet 將超越 InfiniBand** 成為 AI 後端網路主流。

## 8. 資料中心:液冷成預設

GB200 NVL72 = **120kW/rack**、Rubin NVL144 預估 150kW+。風冷已死,**Direct-to-Chip 液冷**為標配,**Immersion**(浸沒)在高密度場景興起。**Stargate** (OpenAI + Oracle + SoftBank, 2025/1 宣布) **4 年 $500B、目標 10 GW**,旗艦 Abilene 部署 45 萬顆 GB200,2025 年底已落地近 7 GW、$400B 投資。

## 9. 電力與永續:AI 的真實成本是電費

H100 TDP 700W → B200 1200W → GB200 superchip 2700W → Rubin 預估 ~1.8 kW/GPU。單一 1GW AI 資料中心年耗電約 8.7 TWh ≈ 80 萬戶家庭。

**核電成為主力 baseload**:
- **Microsoft**:20 年 $16B PPA 重啟 Three Mile Island Unit 1(835MW,Crane Clean Energy Center,2028 上線)。
- **Amazon**:Susquehanna 1.92 GW PPA + $500M 投資 X-energy SMR。
- **Google**:Kairos Power 500MW SMR fleet(2030+)。
- **Meta**:Clinton 核電 1.1 GW + Prometheus 計畫 6.6 GW 核電採購。

## 10. 邊緣 AI 晶片:NPU 進入每一台筆電與手機

- **Apple M5** (2025/10):16-core Neural Engine + CPU/GPU 內嵌 Neural Accelerator,Apple Intelligence 在裝置上跑 3B 模型。
- **Qualcomm Snapdragon X2 Elite Extreme** (2026 H1):**80 TOPS NPU**(Hexagon),AI benchmark 88,615 領先全場。
- **Intel Panther Lake** (2025 末) 接續 Lunar Lake (45 TOPS NPU4)。
- **AMD Strix Halo (Ryzen AI Max+)**:統一 128GB LPDDR5X,單台筆電跑 70B 模型。
- **Edge inference 專晶**:Hailo-10、Google Coral、NVIDIA **Jetson Thor**(Blackwell 架構、Robotics)。

**突破**:**Copilot+ PC** 要求 NPU ≥40 TOPS,Windows 11 把 small model 推進 OS。

## 11. 編譯與運行時:Triton 取代 CUDA-only 的時代

**CUDA** 依然是事實標準,但 **OpenAI Triton** 成為跨廠通用 GPU kernel 語言——Python-like DSL、MLIR-based 編譯,**PyTorch 2.x torch.compile** 的 TorchInductor 直接生成 Triton kernel。**ROCm 7** 支援 Triton/torch.compile 但 autotuning 仍弱。**IREE + MLIR** 提供 Vulkan/Metal/CUDA/CPU 多後端,**ExecuTorch** 是 Meta 的 mobile/edge runtime。

## 12. 經濟學:推理超車訓練、Jevons 引爆總支出

per-token 價格曲線(輸入,$/1M tokens):
- GPT-3.5 (2022): $20 → 2024 後 GPT-3.5 級 ~$0.40 → 2025 ~$0.10
- GPT-4 (2023): $30 → GPT-4 Turbo $10 → GPT-4o $5 → GPT-5 nano $0.05
- 整體 **3 年降 100–1000x**

**突破**:依 SemiAnalysis 等 2025 估算,hyperscaler 推理 capex 已逼近或開始超過訓練 capex(各家統計口徑不一,McKinsey/SemiAnalysis 估計約 60/40 並擴大)。**Jevons paradox** 完整應驗:per-token 降 1000x,但 token 用量增 5000x → 總支出 2025 年 +320% YoY。

---

## AI 工程師需懂的硬體最小知識

1. **算術強度** = FLOPs/Byte。Prefill 是 compute-bound、Decode 是 memory-bound。
2. **HBM 容量 > FLOPs**:能不能放下模型 + KV cache 決定能不能跑;速度其次。
3. **Batch size 與 throughput 的 roofline**:小 batch decode 是 BW-bound,加 batch 才把 GPU 餵飽。
4. **Quantization 階梯**:FP16/BF16 → FP8 → FP4/INT4。
5. **Tensor / Pipeline / Expert / Sequence parallelism**:四種並行各有 collective 成本。
6. **CUDA Graph + Continuous batching + PagedAttention**:vLLM/SGLang/TensorRT-LLM 的三大支柱。
7. **FlashAttention-2/3**:從演算法層消除 attention 的 O(N²) memory I/O。
8. **NCCL/RCCL collective**:all-reduce、all-gather、reduce-scatter 是分散式訓練的瓶頸。
9. **Speculative decoding**:小模型猜大模型,2-3x 加速近乎免費。
10. **Quant + KV cache**:長 context 時 KV cache 可能比 weights 還大,要 INT8/FP8 量化。

---

## 2026 AI 硬體選型 5 個關鍵問題

1. 是 prefill-heavy 還是 decode-heavy?→ 決定要 FLOPs 還是 BW。
2. 模型 + KV cache 多大?→ 決定 HBM 容量底線。
3. 是固定模型還是多模型?→ 決定 ASIC vs GPU。
4. Scale-out 規模?→ 決定 IB vs Ultra Ethernet。
5. 電力與冷卻有沒有?→ 決定能否自建 vs 上雲。

**核心 trade-off**:**Generality vs Speed vs $/token**——三角形只能挑兩個。

延伸子主題:[Neuromorphic Computing 神經形態運算](./Neuromorphic_Computing.md)

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
