# 12. AI for Science (AI4Science) 全景 (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #8
> 2024 諾貝爾化學獎(Baker + Hassabis + Jumper)正式蓋章:AI 改變了所有科學。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **DL 基礎 + CNN/Transformer**(對應 repo:[1.從AI到LLM基礎/4.DL](../1.從AI到LLM基礎/4.DL/README.md))
> 2. **GNN 基本概念**(等變網路、message passing — 對應 repo:[18.GNN_Graph_Learning](../18.GNN_Graph_Learning/README.md))
> 3. **Diffusion / Flow Matching 基礎**(對應 repo:[Multimodal_Generation_2024-2026.md](../1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md))
>
> 若全部不熟悉,先讀 [全景圖 #8](../2024-2026_AI完整領域全景圖.md) 與 [18.GNN_Graph_Learning](../18.GNN_Graph_Learning/README.md) 建立等變網路與分子表徵直覺。
>
> **延伸 / 反向連結**:[18.GNN_Graph_Learning](../18.GNN_Graph_Learning/README.md)(等變網路、Graph Foundation Models)

---

## 1. 蛋白質與結構生物學

2024/05 DeepMind 與 Isomorphic Labs 共同發表 **AlphaFold 3** 於 *Nature*,將原本只能處理蛋白質骨架的 AlphaFold 2 升級為涵蓋蛋白質、核酸、小分子、離子、修飾殘基的全生物分子複合體預測器,並改用 diffusion-based 架構直接生成原子座標。AlphaFold Server 同步開放給非商業研究者免費使用。

**突破**:相對 AlphaFold-Multimer v2.3,在抗體-抗原預測、蛋白-核酸交互作用、蛋白-配體 docking 都至少提升 50%。隨後生態系爆發:
- **ESM3 (Meta)** — multimodal 序列-結構-功能 LLM(98B 版本可條件式生成新蛋白質)
- **Boltz-1 (MIT, 2024/11)** — 首個與 AlphaFold 3 等價且 MIT 授權可商用,30-60 秒完成一次預測
- **Chai-1 (Chai Discovery, 2024/09)** — PoseBusters 77% 勝過 AlphaFold 3 的 76%
- **RFdiffusion2 (UW IPD, 2024)** — 開源 *de novo* 蛋白設計主力(更新版本 RFdiffusion3 為傳聞中釋出,2026-05 待官方確認)

**實作**:`pip install boltz` + 一張 A100/RTX 4090 + ColabFold MSA pipeline。

## 2. 生物基因組序列模型

**Evo 2 (Arc Institute, 2025/02)**:9.3 兆 DNA 鹼基對訓練,7B 和 40B 兩個版本、最長 1M token context、single-nucleotide 解析度,涵蓋古菌、細菌、真核、噬菌體所有生命域。2026 年正式於 *Nature* 出版。

**突破**:Evo 2 不需任務微調即可預測 BRCA1 致病變異、非編碼突變影響,並可生成完整細菌等級的基因組序列(數十萬至百萬 bp 連貫)。其他重要模型:**Nucleotide Transformer** (2.5B)、**HyenaDNA** (1M context)、**Caduceus** (雙向 Mamba)。

**實作**:Evo 2 已在 NVIDIA BioNeMo 與 build.nvidia.com 提供 API。

## 3. 藥物發現與分子設計

2025 年 AI 製藥從 hype 進入臨床交付期:
- **Isomorphic Labs** 在 2025/03 完成 $600M 融資、累積 $2.1B,2026/01 達沃斯確認首個 AI-designed 分子預計 2026 年底進臨床
- **Recursion + Exscientia** 合併形成 phenomics + 自動化合成的 end-to-end 平台
- **RFdiffusion2** (*Nature Methods* 2025):從功能基團幾何 scaffold 41 個酵素活性位點(前一代僅 16 個)
- **Chai Discovery** 2025 釋出 zero-shot antibody design,16-20% hit rate(傳統 phage display ~1%)

## 4. 材料科學

- **GNoME (DeepMind, *Nature* 2023-2024)**:預測 220 萬新晶體中 38 萬個熱力學穩定材料,等於人類 800 年累積
- **MatterGen (Microsoft, *Nature* 2025/01)**:首個 property-conditional diffusion 生成模型,合作 Shenzhen 團隊已合成出 TaCr2O6,bulk modulus 169 GPa(目標 200 GPa)
- **UMA (Meta FAIR, 2025/07)**:5 億個 3D 原子結構訓練,mixture of linear experts 在 1.4B 參數中只啟動 50M,單一模型橫掃 Matbench Discovery、AdsorbML、MOF、催化劑
- **Equiformer / SevenNet / MACE** — 主要等變 GNN 骨幹

機器學習原子間勢 (MLIP) 取代 DFT 達 $10^4$ 倍加速。

## 5. 氣象與地球科學

- **GraphCast (DeepMind, 2023)** — 神經氣象元年
- **GenCast (DeepMind, *Nature* 2024/12)** — diffusion-based 機率氣象,50 條 15 天 ensemble 在單張 TPU v5 跑 8 分鐘,在 97.2% 評估指標上勝過 ECMWF ENS
- **Aurora (Microsoft, 2024/05)** — 1.3B 參數 3D Swin Transformer,首個跨任務的大氣基礎模型,較 IFS 加速約 5000 倍
- **NeuralGCM (Google + Berkeley)** — ML 嵌入物理 GCM 混合架構

**實作**:`graphcast` 與 `aurora` 皆於 GitHub 開源,從 ERA5 切片 (`xarray` + `zarr`) 即可在單卡推論。

## 6. 數學與形式化證明

**AlphaProof + AlphaGeometry 2 (DeepMind, 2024/07)**:於 2024 IMO 解出 6 題中的 4 題,28/42 分達銀牌,由 Tim Gowers 評審。AlphaProof 攻克競賽中最難、僅 5 位選手破解的題目。2025 *Nature* 論文〈Olympiad-level formal mathematical reasoning with reinforcement learning〉。

2025 年中:
- **DeepSeek-Prover-V2** (671B):MiniF2F-test 88.9%,PutnamBench 49/658
- **Goedel-Prover-V2** (Princeton/Tsinghua, 2025/08):32B 達 MiniF2F 88.1%、8B 版以 84.6% 超越 80 倍大的 DeepSeek

**實作**:Lean 4 + Mathlib + LeanDojo;以 Goedel-Prover-V2-8B 在 RTX 4090 上跑 pass@32。

## 7. 物理模擬與 PDE 求解

**Fourier Neural Operator (FNO)** 仍為核心,2024-2025 衍生 **PINO**(operator + equation loss)、**DiffFNO**(CVPR 2025)、**LOGLO-FNO**(ICLR 2025)。**Mamba Neural Operator**(NeurIPS 2024)以 SSM 取代 transformer 處理長時序 PDE。**PINN** 演進為自適應採樣與 Evo-PINN(2025)。

ML 加速分子動力學由 MACE、SevenNet、UMA 達 $10^3$–$10^5$ 倍加速、覆蓋 ns 到 µs 模擬。

## 8. 天文與粒子物理

**Galaxy Zoo + ML** 已將 SDSS、Euclid、Rubin LSST 的形態分類自動化;**AstroLLaMA / AstroBERT** 將天文文獻與光譜統合進 LLM。LHC ATLAS、CMS 部署 anomaly detection 搜尋 BSM 新物理。LIGO/Virgo 用 ML 做 glitch classification 與低延遲合併事件偵測。

## 9. 科學基座模型 (Foundation Models for Science)

2025 共識:每個科學領域都需要自己的「GPT 等價物」。例:Evo 2(基因組)、ESM3(蛋白)、Aurora(大氣)、UMA(原子勢能)、AstroCLIP(天文)。共同特徵:自監督預訓練、超大規模科學資料、可零樣本/少樣本下游微調、開放權重成為標配。

## 10. 2024 諾貝爾里程碑

**2024 化學獎**:一半授予 **David Baker**(UW)表彰計算蛋白設計;另一半由 **Demis Hassabis** 與 **John Jumper**(DeepMind)共享,表彰 AlphaFold 2。截至 2024/10 已有 190 國、200 萬人使用,並覆蓋 2 億蛋白結構。

**2024 物理獎**:授予 **John Hopfield** 與 **Geoffrey Hinton**,表彰人工神經網路基礎。

兩獎同年頒給 AI 相關工作,正式將 AI 列入主流自然科學典範轉移的歷史座標。

## 11. 工程師如何切入(不做研究的捷徑)

1. **環境**:`conda` + Python 3.11 + PyTorch 2.x + 一張 24GB 顯卡
2. **生物入門**:`biopython` → ColabFold → Boltz-1 → PyMOL / ChimeraX
3. **設計閉環**:RFdiffusion → ProteinMPNN → AlphaFold / Boltz 驗證 → OpenMM 短 MD
4. **基因組**:Evo 2 7B 跑變異 likelihood (build.nvidia.com)
5. **材料/化學**:`ase` + MACE-MP-0 / UMA + MatterGen + Materials Project API
6. **氣象 / PDE**:`xarray` + ERA5 zarr + GraphCast / Aurora
7. **資源**:Hugging Face 的 `chemistry` / `biology` 標籤、NVIDIA BioNeMo、Arc Institute Evo

---

## 2026 AI4Science 入門地圖

| 階段 | 軸線優先順序 | 工具 | 產出指標 |
|---|---|---|---|
| Week 1-2 | 蛋白質結構 | Biopython + Boltz-1 + PyMOL | 重現 1 個 PDB,RMSD < 2 Å |
| Week 3-4 | 蛋白質設計 | RFdiffusion + ProteinMPNN + AlphaFold | mini-binder,AF pLDDT > 80 |
| Week 5-6 | 基因組 | Evo 2 (BioNeMo API) | BRCA1 變異 likelihood ranking |
| Week 7-8 | 材料 | MACE / UMA + MatterGen | 生成 10 個穩定晶體並 DFT 驗證 |
| Week 9-10 | 氣象 | GraphCast / Aurora + ERA5 | 復現一次極端事件預報 |
| Week 11-12 | 數學 | Lean 4 + Goedel-Prover-V2-8B | MiniF2F pass@8 ≥ 50% |

**心法**:AI4Science 已從「研究員專屬」變為「任何具備 PyTorch 基礎的工程師都能跑」。最大紅利在於模型權重 + 推論程式碼幾乎全部開源——你不需要訓練,只需要組合。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
