# 13. Robotics / Embodied AI 全景 (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #7
> 過去 24 個月,具身智能從「實驗室 demo」躍升為「資本市場主敘事」。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **VLM(Vision-Language Models)基礎**(對應 repo:[CV_全景_2024-2026.md](../1.從AI到LLM基礎/4.DL/CV_全景_2024-2026.md))
> 2. **Transformer 基礎**(對應 repo:[2.深入LLM模型工程與LLM運維/1.LLM 基礎與架構](../2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/README.md))
> 3. **Diffusion Policy / Flow Matching 概念**(若 repo 內無,先看:[Multimodal_Generation_2024-2026.md](../1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md) 的 diffusion / flow 章節,以及 [Diffusion Policy 原論文](https://diffusion-policy.cs.columbia.edu/))
>
> 初次接觸機器人控制請先讀 [全景圖 #7](../2024-2026_AI完整領域全景圖.md) 對應章節。
>
> **延伸 / 反向連結**:[3.LLM應用工程/3.Agent](../3.LLM應用工程/3.Agent/)(Computer Use / 數位 agent 與機器人 VLA 同源)

---

## 1. VLA (Vision-Language-Action) 模型深度

VLA 已成為機器人領域 GPT 時刻。從 **RT-2 (2023)** 引入 VLM-as-policy 之後,2024–2026 開源生態爆炸:
- **OpenVLA (7B, Prismatic + Llama2)** — 社群事實基礎
- **RDT-1B** — diffusion-based 雙臂控制
- **Octo** — 輕量 transformer-diffusion,單張 RTX 4090 跑到 15–30 Hz
- **π0 (Physical Intelligence, 2024/10)** — flow-based,VLM backbone + Flow Matching 連續動作生成
- **π0.5 (CoRL 2025 Oral, 開源於 `openpi`)** — co-training (web data + 異構機器人 trajectory + 高層語意預測) 推到能進**沒看過的家**清潔的開放世界泛化
- **π0.6 (2025/11)** — 進一步強化 long-horizon
- **SmolVLA (450M)** — LIBERO 上達到 OpenVLA 9 成性能、單張 RTX 4090 跑 15–30 Hz
- **CogACT** — VLM 與 diffusion action transformer 解耦,真機勝 OpenVLA 55%
- **NVIDIA Gr00t N1 / N1.5** — System 1/System 2 dual architecture,DreamGen 12 任務從 13.1% → 38.3%

**選型**:LeRobot 起步;少量資料用 ACT/SmolVLA 微調,大規模 cross-embodiment 用 π0 / OpenVLA-OFT。

## 2. 人形機器人 (Humanoid)

2025–2026 形成「研究 / 商用 / 平民」三層:
- **Boston Dynamics Electric Atlas** (2024-04 退役液壓版),Hyundai 工廠 2025 試點,CES 2026 宣布開始出貨
- **Apptronik Apollo** — $520M 加碼,Mercedes-Benz、GXO、Jabil 多重綁定
- **Figure 03 (2025/10, $20K)** + **Helix-02 VLA** — 8 小時自主分揀班次,BotQ 工廠產能拉到每小時 1 台,已交付 350+
- **1X Neo** — 主打家用
- **Tesla Optimus Gen 3** — 因設計返工延至 2026 夏量產
- **Unitree G1 ($16K 起)** — 2025 出貨 5,500+,2026 目標 1-2 萬台,**唯一真能買到的全功能平台**
- **Sanctuary AI Phoenix、Agility Digit** — 倉儲 niche

**突破**:Figure 從「VLM + low-level policy」改為**端到端 Helix-02** (S0/S1/S2 分層)。

## 3. 世界模型 (World Models)

三大門派並立:
- **Meta V-JEPA 2 (2025/06, 開源)** — 1M 小時 internet video 預訓練 + 62 小時機器人 fine-tune,DROID 上 zero-shot pick-and-place 達 65-80%
- **DeepMind Genie 3 (2025/08)** — 文字生成 720p / 24fps 可導航互動世界,記憶 1 分鐘;2026/01 對 Ultra 用戶開放
- **NVIDIA Cosmos** (CES 2025, **Cosmos-Predict 2.5** 已支援 30 秒影片 + 多視角) — synthetic data 工廠,1X、Agility、Figure、XPENG、Uber 全部接入
- **World Labs (Fei-Fei Li)** — 3D scene generation
- **Wayve GAIA-2** — 自駕

**突破**:從「離線 generative video」轉向「**可互動 + 可規劃**」。V-JEPA 2 證明 SSL + 少量機器人 fine-tune 就能 zero-shot 規劃。

## 4. 模擬器與訓練平台

- **Isaac Lab** (Isaac Sim + PhysX/Warp/MuJoCo/Newton):GPU 大規模 parallel rollout 王者,4,096 envs 達 82K-94K FPS
- **MuJoCo MJX (Playground)**:8-chip TPU v5 達 270 萬 steps/sec,Apple M3 Max 上 65 萬,sim-to-real 友善
- **Genesis** (2024 末爆紅):實測比宣稱慢 100×,熱度已退
- **Newton** (NVIDIA + DeepMind + Disney 聯合):統一物理引擎,humanoid 70×、manipulation 100× 加速

**選型**:RL 訓練選 Isaac Lab;研究發 paper 選 MuJoCo Playground;sim2real 選 Cosmos + Isaac Lab。

## 5. 資料與 Dataset

- **Open X-Embodiment (OXE)** — 事實 baseline (1M+ trajectories, 22 個 embodiment)
- **DROID** — crowd-sourced scene diversity (V-JEPA 2 / π0 都用)
- **AgiBot World (智元)** — 1,001,552 trajectories / 2,976 hours / 217 tasks,中國最大開源
- **RH20T** — 110K+ contact-rich sequences,含力覺/音訊/動作多模態 40TB
- **LeRobot Dataset Hub (v3.0)** — HF 上 3,000+ datasets,SO-100/101/Koch/LeKiwi 佔過半

## 6. 真實機器人平台

- **LeRobot (Hugging Face)** — democratization 火車頭
- **SO-100 / SO-101 (~$100) + LeKiwi (移動底盤)** — 入門價打到玩具級
- **Stanford ALOHA / Mobile ALOHA** — 雙臂研究黃金標準 ($25K)
- **Franka FR3、UR5/UR10e、Stretch 3 (Hello Robot)** — 工業 / 家用主流
- **Unitree G1** — ROS2 + LeRobot 整合進入學界視野

**建議**:預算 < $500 上 SO-101 雙臂;研究室上 ALOHA + Franka;移動操作上 Stretch 3 / LeKiwi。

## 7. 訓練典範

四大流派並用:
1. **模仿學習 (BC)** — production VLA 主力
2. **Teleop + Diffusion Policy** (Columbia/TRI/MIT) — 多 modal 軌跡
3. **ACT (Action Chunking Transformer, CVAE)** — ALOHA 出圈,10 分鐘 demo 即可 80-90%
4. **RL in sim + Sim2Real** (Isaac Lab + MJX) — locomotion / whole-body
5. **Co-training cross-embodiment** (π0.5 / GR00T) — 人類 ego video + 多機 trajectory + web data

**Action Chunking 成為共識**——一次預測 10-50 步,推理快、軌跡平滑、容錯高。

**選型**:雙臂 → ACT;多模態複雜 → Diffusion Policy;locomotion → PPO in MJX;通用泛化 → π0/OpenVLA fine-tune。

## 8. 2025–2026 突破總清單

6 個里程碑:
1. **π0.5 開放世界泛化** (進沒看過的家清潔)
2. **Figure Helix-02** 端到端 e2e 神經網路 + 8 小時自主作業
3. **NVIDIA GR00T N1.5 開源** (System 1/System 2, DreamGen 12 任務 13.1% → 38.3%)
4. **V-JEPA 2 zero-shot 機器人規劃**
5. **Genie 3 互動世界模型**
6. **Cosmos-Predict 2.5** 統一 Text/Image/Video2World

**共同主題**:**「VLM 後端 + diffusion/flow action head + 大規模 co-training」**已是 winning recipe。

## 9. 產業現狀

- **工廠應用遙遙領先**:Mercedes/BMW/GXO/Hyundai 全部試點
- **家用機器人時程**:Figure 03 最早 2026 Q4,1X Neo 2026,Optimus 不會早於 2027
- **安全認證**:**ISO 10218-1:2025** 整合 collaborative + 功能安全 + cybersecurity,把「固定在 mobile platform 上的 humanoid」納入工業場景;**ISO 25785-1**(專門 humanoid mobility)仍是 Working Draft,2026-2027 才正式發布
- **募資**:Apptronik $520M、Figure $1B+、1X $100M+,2025 humanoid 募資破百億美元

## 10. VLA 對「一般 AI 工程師」意味著什麼

VLA 的核心 abstraction **「pixel/video + language → action token」**已經外溢:
- **Anthropic Claude Computer Use** (Opus 4.6/4.7 在 OSWorld、SWE-Bench 屠榜)
- **OpenAI Operator (CUA)**
- **Google Gemini Computer Use (Project Mariner)**

全是「螢幕截圖 + 指令 → 鍵鼠 action」的 VLA 同構。VLA 範式統一了 robotics 與 software agents——**action space 不同 (joints vs clicks),但 perception/reasoning/chunking 全套技術可遷移**。

**即使不做機器人**,理解 (1) action tokenization、(2) chunked prediction、(3) world model planning,直接受用於 computer-use agent 設計。

---

## 2026 Embodied AI 工程師地圖

| 層級 | 必修 | 推薦 |
|---|---|---|
| **硬體** | SO-101 / LeKiwi (~$300 入門) | Unitree G1 / Franka FR3 |
| **模擬** | Isaac Lab + MuJoCo MJX | Cosmos (synthetic data) |
| **資料** | LeRobotDataset v3 + OXE | DROID / AgiBot World |
| **模型** | OpenVLA / π0 / SmolVLA | GR00T N1.5 / Helix 風格 e2e |
| **演算法** | ACT + Diffusion Policy | Flow matching + co-training |
| **世界模型** | V-JEPA 2 | Genie 3 / Cosmos-Predict |
| **安全** | ISO 10218-1:2025 | ISO 25785-1 (跟蹤) |
| **跨域** | Claude Computer Use API | Operator / Mariner |

**一句話總結**:2024-2026 是「VLA + 世界模型 + 大規模 co-training」三位一體的範式確立期;開源 (LeRobot, OpenVLA, GR00T, V-JEPA 2) 把入場門檻打到個人開發者層級。**VLA 不只是「會動的 LLM」,而是所有 agent 範式的母體**。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
