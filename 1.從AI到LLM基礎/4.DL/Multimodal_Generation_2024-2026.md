# 多模態生成 — Any-to-Any 與 Unified Multimodal (2024-2026)

> 對應 [全景圖](../../2024-2026_AI完整領域全景圖.md) #6
> 從「各模態各自訓練、靠 pipeline 串接」,演化到「單一 backbone、token 雙向流通、生成與理解共享」。

---

## 1. 核心範式:Diffusion → Flow Matching → Hybrid AR

2024 之前主流是 DDPM/DDIM/EDM 路線。2024-2026 全面轉向 **Flow Matching (FM)** 與 **Rectified Flow (RF)**——學習 ODE 速度場,將噪聲到資料的軌跡「拉直」,推理步數降 10-100×。**Stable Diffusion 3、FLUX.1、Sora** 都基於 RF。

**突破**:**Consistency Flow Matching** 強制速度場自洽性,1-4 步即可生成高品質;**Diff2Flow** 證明可直接把預訓練 diffusion 模型「轉檔」為 FM 模型再 finetune,免從頭訓練。

**第三軸**是 **Autoregressive token-based** (Chameleon/Emu3/Janus) 與 **Masked Generative** (MAR/MaskGIT),它們把影像/影片 tokenize 後用 transformer 預測,天然與 LLM 同構——這是 unified multimodal 的關鍵。

**Hybrid (AR + Diffusion)** 由 Transfusion / Show-o 領頭:LLM backbone 處理離散 text token,但對 image 部分採用 diffusion loss——同一個 transformer、兩種損失。

## 2. Any-to-Any 統一多模態模型

2024/05 **GPT-4o** 揭幕「單一 transformer、text/image/audio token 雙向流通」的範式。開源側:
- **Chameleon (Meta)** — fully token-based
- **Emu3 (BAAI)** — 把影片也納入單一 next-token 預測
- **Show-o / Transfusion** — AR+Diffusion 混合
- **Janus / Janus-Pro (DeepSeek, 2025/01)** — 關鍵創新是 **decouple visual encoding**——理解走 SigLIP encoder、生成走 VQ tokenizer,GenEval 達 80% 勝過 SD3-Medium (74%)、DALL-E 3 (67%)
- **OmniGen / OmniGen 2 (BAAI)** — 「text + 多參考圖統一輸入」,可直接做 reference-driven editing 不需 ControlNet
- **Gemini 3 Pro Image** (2025/11) — 把 reasoning 嵌入生成(會先「想」再畫)
- **4M-21 (Apple)** — 21 模態統一

**2025 共識**:「decoupled understanding vs generation encoder」優於完全共享;生成階段保留 diffusion/FM head,理解階段保留 contrastive encoder,但 transformer backbone 共享。

## 3. 影像生成(深化)

閉源 ELO 排序:**Recraft V3 (1172) > FLUX 1.1 Pro (1143) > Ideogram v2 (1102) > Midjourney v6.1/v7 (1093)**。

- **Midjourney v7** (2025/04):藝術/氛圍王
- **FLUX.1 dev/schnell**:開源生態核心(Apache/non-commercial 雙授權)
- **FLUX 1.1 Pro**:企業 API,C2PA 簽章
- **SD 3.5 Large** (Stability):仍是 LoRA/ControlNet 生態唯一可本地完整 fine-tune 的開源旗艦
- **Imagen 3/4** (Google):廣告/品牌素材強
- **DALL-E 3**:已被 GPT-4o native image 取代

**突破**:**typography 與 layout 可靠性**(Ideogram/Recraft)從 demo 進入生產;**vector-native 生成**(Recraft v3 直出 SVG)讓設計師 workflow 從「PS 重畫」變「指令重生」。

## 4. 影片生成(深化)

- **Sora 2** (2025/Q4):依各家 release demo 觀察,在物理真實感與長鏡頭穩定性具備領先表現(具體規格以官方公告為準)
- **Veo 3.1** (Google, 2025/10):同步音訊與 native 4K 領先,API 可用
- **Kling 3.0**:性價比王(~$0.50/clip)
- **Runway Gen-4 / Gen-4.5**:camera control(dolly/crane/focus pull)與 character consistency
- **Pika 2**:短梗創作
- **HunyuanVideo (13B, 騰訊)**:仍是開源霸主
- **Wan 2.2 (阿里)**:引入 MoE(高噪聲/低噪聲 expert 分工),最易部署
- **Mochi 1 (10B)**:Asymmetric DiT
- **LTX-Video**:主打 RTX 4090 即時推理
- **Cosmos (NVIDIA)**:World Foundation Model,定位 robotics/sim

**突破**:I2V (image-to-video) 變成多數工作流的核心入口(配合 FLUX 出首幀);V2V (motion transfer) 用於後期;**character consistency** 透過 reference image + identity embedding 已可做到跨 clip 一致。

## 5. 3D 生成

2025 是 image-to-3D 進入「可用」的轉折年:
- **Microsoft TRELLIS (CVPR'25 Spotlight) → TRELLIS.2 (2025/12, 4B, MIT)**:單圖出帶 PBR 材質 mesh,20s-4min
- **Hunyuan3D 2 (騰訊)**:最強開源替代
- **Tripo 3**:最快(~10s)
- **Meshy 5**:3D 列印適配
- **Rodin**:專業電影級
- **CSM**:場景級
- **3DTopia**:學術線

**突破**:從 NeRF/3DGS「重建」走向 **diffusion-on-latent-voxel 直接生成 mesh**;PBR 材質、UV、retopology 開始一次出。

## 6. 音樂 / 音訊生成

- **Suno v5.5** (2026/03):完整歌曲(verse-chorus、人聲)生成標竿,8 分鐘 + voice cloning
- **Udio**:hip-hop/R&B 人聲品質略勝,UMG / Warner 已授權和解
- **Stable Audio 2.5**:授權合規最安全(instrumental / SFX / 短 loop)
- **MusicGen (Meta)**:研究/開源 baseline

**突破**:商用合規從「灰色」進入「授權模型」階段;跨指(stem separation / 後製可編輯)成標配;與影片生成耦合(Veo 3.1 直接出含同步配樂的 clip)。

## 7. 文件 / 圖表 / 簡報生成

- **Gamma**:AI 簡報市場龍頭
- **Tome AI**:2024 轉向 sales enablement
- **Excalidraw AI**:LLM 生成 mermaid/excalidraw JSON
- **Eraser AI**:工程架構圖

**突破**:輸出從「PNG 圖片」變「**可編輯的結構化文件**」(PPTX/SVG/Markdown/Mermaid)——這是 LLM-as-code-generator 的勝利:讓 LLM 生 DSL 而非 pixel。

## 8. 編輯與控制

生產線常見組合是「**FLUX + LoRA + ControlNet (depth) + IP-Adapter (style ref) + InstantID (face)**」五件套:
- **ControlNet** (canny/depth/pose/normal):結構控制基石
- **IP-Adapter**:「圖像 prompt」,風格/參考圖一行接入
- **InstantID**:單張人臉照即可保 ID 跨姿勢生成
- **LoRA**:數十萬模型生態,SDXL/SD3.5/FLUX 各自獨立
- **Inpainting/Outpainting**:FLUX Fill / SD3.5 Inpaint 像素級可控

**新趨勢**:**unified models (OmniGen/Gemini 3)** 開始挑戰「需要 ControlNet」的前提——但目前可控性仍不如 ControlNet。

## 9. 多模態 RAG 生成迴圈

**Multimodal RAG (mRAG)** 從單純「文字 RAG + caption」進化到 **CLIP/SigLIP 統一向量空間**檢索 + **MLLM 直接讀圖**生成;**VisRAG** 證明直接用 vision encoder 讀整頁 PDF 比 OCR+text-RAG 強。

2025 出現 **iterative / agentic RAG**:模型自己決定「現在要不要 retrieve、從哪個 KB retrieve」(R1-Router 用 RL 學路由)。

**校驗迴圈**:生成 → 用 VLM 看圖 → 比對 prompt → 自我修正,已成為 production 標配。

## 10. 生產化考量

深偽事件 2023 → 2025 從 50 萬增至 800 萬+(900% 兩年),預估 2026 線上媒體 90% 為合成內容。

- **C2PA 2.1** 已成 ISO/IEC 22144 標準,Adobe/MSFT/OpenAI/Google/Meta/BBC/Reuters 全員背書
- **Content Credentials** 簽章每次編輯
- **SynthID (Google DeepMind)** 已覆蓋 100 億+資產,2025/05 上線 Detector Portal
- **EU AI Act 2026/08** 強制機器可讀標記
- 商業授權層:**FLUX 1.1 Pro / Suno Pro / Udio Pro** 走 per-call 計費 + 商用授權保障;**Sora 2 / Veo 3.1** 預設嵌入 C2PA + SynthID 不可關

**心法**:任何上線產品在輸出端寫入 C2PA manifest + SynthID;訓練資料採購要求授權證明;EU 市場 2026/08 前完成合規。

---

## 2026 多模態生成工程師地圖

**第 0 層 基礎理論**:Flow Matching / Rectified Flow / Consistency models;tokenizer 設計;AR vs Diffusion vs Hybrid 三條 loss 路線的選型。

**第 1 層 模型熟練度**(挑 1 主 + 2 輔):
- 影像:FLUX.1 (主) + SD 3.5 + Recraft
- 影片:Wan 2.2 / HunyuanVideo (主) + Sora 2 / Veo 3.1 API
- 3D:TRELLIS.2 (主) + Hunyuan3D 2 + Tripo
- 統一:Janus-Pro / OmniGen 2 + GPT-4o / Gemini 3 Pro Image API
- 音樂:Suno + Stable Audio

**第 2 層 控制與編輯**:ControlNet / IP-Adapter / InstantID / LoRA 訓練 + ComfyUI 節點工程

**第 3 層 系統整合**:multimodal RAG (CLIP/SigLIP + VisRAG);generate-verify loop;agentic orchestration

**第 4 層 生產化**:C2PA / SynthID 嵌入與驗證、訓練資料合規、EU AI Act 文件、watermark robustness 測試、cost/throughput 監控

「2024 = 各模態各自為政,2025 = unified backbone 起飛,2026 = generation+understanding+retrieval 三位一體」——這就是這個橫切面在三年內走完的路。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
