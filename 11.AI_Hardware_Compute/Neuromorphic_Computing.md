# 神經形態運算 (Neuromorphic Computing) 2024-2026

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #2
> 與 GPU 是「平行宇宙」,不是取代關係。

---

## 1. 基本概念:Spike、Event-Driven 與記憶-運算融合

神經形態運算的核心是放棄馮諾依曼架構的「計算-記憶體分離」,改用三大支柱:**spike-based 通訊**(神經元只在膜電位超過閾值時發射 1-bit 脈衝)、**event-driven 執行**(無事件即零功耗)、**in-memory compute**(突觸權重就地計算)。LIF (Leaky Integrate-and-Fire) 模型是主流抽象。**數位實作正取代類比/混合訊號設計**——犧牲一些生物擬真度,換取可重複性與量產可行性。

## 2. 主流晶片

- **Intel Loihi 2**(Intel 4 製程):單晶片 1M 神經元、120M 突觸,Lava SDK。**Hala Point**(2024/4 部署 Sandia)堆疊 1,152 顆,達 **11.5 億神經元、1,280 億突觸**。
- **IBM NorthPole**(12nm,22B 電晶體):H100 4nm 對比下仍快 **5×/joule**;3B 參數 LLM 推理較 H100 達 **72.7× 能效**。每 token 延遲 < 1ms。
- **BrainChip Akida Gen 2**:8-bit 權重、支援 CNN/DNN/ViT/SNN,2025 推出 Akida Cloud,並獲 $25M 募資。
- **SpiNNaker 2**(GlobalFoundries 22nm):單晶片 152 核;Dresden 部署 5M+ 晶片可達 50-100 億神經元。
- **Innatera Pulsar**(2025 量產):首款 mass-market 神經形態 MCU,600µW 雷達存在偵測。

## 3. 演算法:Surrogate Gradient、ANN-to-SNN、STDP

1. **Surrogate Gradient (SG)**:替不可微的 Heaviside spike 函式套上可微近似(arctan、fast sigmoid),用 BPTT 直接訓練——目前主流。
2. **ANN-to-SNN 轉換**:用 ReLU 訓練 ANN,轉成 IF 神經元 SNN,通常需 100-1000 timesteps。
3. **STDP**:純生物啟發局部學習規則,適合無監督/線上學習。

Meta-SpikeFormer 在 ImageNet 達 **80.0% top-1**,SGLFormer 更達 **83.73%**。

## 4. 軟體生態:PyTorch 派系一統江山

- **PyTorch 派(入門)**:**snnTorch**(教學)、**SpikingJelly**(訓練最快,T=32 較對手快 11×)、**Norse**。
- **硬體 SDK**:Intel **Lava**、BrainChip **MetaTF**、SynSense **Rockpool**、SpiNNaker **PyNN**。

**建議**:研究跑 SpikingJelly;教學讀 snnTorch;要部署 Loihi 才碰 Lava。

## 5. 應用場景:Edge、感測與植入

- **Edge/IoT**:Innatera Pulsar 600µW 雷達 / 400µW 音訊分類——較傳統 MCU 低 100-500×。
- **無人機**:Science Robotics 2024 Loihi + event camera 完整飛控,0.94W idle + 7-12mW @ 200Hz。
- **植入式醫療**:視網膜/耳蝸 prosthetic 用 spiking CNN。
- **機器人視覺**:Nature *Communications Engineering* 2025。

## 6. vs GPU 比較:能效贏,通用性輸

稀疏、事件式工作負載上能效領先 **25-250×**。但:
- 大型 dense 矩陣運算(Transformer FFN、attention)GPU 仍碾壓。
- 訓練幾乎都在 GPU 上做,神經形態晶片只做推理。
- 開發門檻:工具鏈成熟度落後 CUDA 約 10 年。

**結論**:GPU 是訓練/通用 AI 的霸主;神經形態是**推理側、邊緣、低功耗 niche** 的補位。

## 7. 何時值得學 / 何時繞道

**值得學**:
- edge AI/IoT 硬體,目標 < 1mW always-on
- event camera / DVS 視覺
- 計算神經科學/腦機介面
- 賭一條 5-10 年的硬體曲線

**繞道**:
- LLM/Diffusion/AGI 主流方向
- 求快速 deliver 產品(工具鏈不成熟)
- 資料科學家而非系統/硬體工程師

## 3 階段學習路徑

1. **入門 (2 週)**:讀 *Neuromatch* 課程的 SNN section + snnTorch tutorial → 在 MNIST/N-MNIST 上跑通 LIF + SG。
2. **進階 (1-2 月)**:SpikingJelly 復現 Spikformer/Meta-SpikeFormer → 讀 Nature Comms 2025 路線圖 + NeuroBench 論文。
3. **硬體實戰 (3 月+)**:申請 Intel Neuromorphic Research Community (INRC) 拿 Loihi 2 access,或買 Akida PCIe 開發板。

**一句話結論**:在 2026 已從「PPT 階段」進入「商業 edge 量產 + 雲端千億神經元 benchmark」階段,但仍是 GPU 主流生態的**補位者**而非取代者。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
