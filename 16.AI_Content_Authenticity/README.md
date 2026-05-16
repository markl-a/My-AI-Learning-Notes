# 16. AI 內容真實性 / Provenance / Watermarking (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #22
> 從「事後偵測」到「源頭簽章」,2024–2026 是真實性基礎設施正式建構的三年。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **LLM / Diffusion 生成基礎**(對應 repo:[2.深入LLM模型工程與LLM運維/1.LLM 基礎與架構](../2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/README.md)、[Multimodal_Generation_2024-2026.md](../1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md))
> 2. **密碼學雜湊與數位簽章基本概念**(若 repo 內無,先看:[Cloudflare Learning: hashing](https://www.cloudflare.com/learning/ssl/what-is-a-cryptographic-hash-function/))
> 3. **熟悉 watermarking / steganography 的基礎直覺**(可選 — 先讀 [全景圖 #22](../2024-2026_AI完整領域全景圖.md))
>
> 直接進入本檔之前,建議至少先掃過 [全景圖 #22](../2024-2026_AI完整領域全景圖.md) 對應章節建立 C2PA / SynthID 大局觀。
>
> **延伸 / 反向連結**:[14.Voice_Audio_AI](../14.Voice_Audio_AI/README.md)(語音 deepfake) | [19.Synthetic_Data_Engineering](../19.Synthetic_Data_Engineering/README.md)(合成資料溯源) | [Multimodal_Generation_2024-2026.md](../1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md)

---

## 1. C2PA (Content Credentials) 與 ISO/IEC 22144

C2PA 規格從 2024/09 v2.1 演進到 2025/05 **v2.2**,核心擴增包含 **video manifest 處理、雲端 manifest**。技術核心是 COSE 簽章的 manifest store,JPEG 中以 APP11 marker 嵌入(64KB 上限常需分段)。

**Coalition 成員**:Adobe / Microsoft / Google / Meta / OpenAI / BBC / NYT / Sony / Leica / Nikon / Canon / Truepic / Qualcomm。

**突破**:**ISO/DIS 22144 (Authenticity of Information — Content Credentials)** 把 C2PA 拉升為國際標準;JPEG Trust v2 提供互補「信任鏈評估框架」;**C2PA Conformance Program** (2025/05 啟動) 首度提供第三方認證機制。

**實作**:用 `c2pa-rs` / `c2patool` 在內容入口管線簽章;manifest 應包含 `c2pa.actions`、`c2pa.training-mining`、`c2pa.hash.data`;對外發布前以 `c2pa-attacks` 做 fuzz。

## 2. SynthID (Google DeepMind)

覆蓋四種模態:**SynthID-Text、SynthID-Image、SynthID-Audio、SynthID-Video**,整合進 Gemini、Imagen 3、Lyria、Veo。截至 **2025/05 Google I/O 已浮水印超過 100 億**件資產;同月推出 **SynthID Detector Portal**(早期測試者 / 記者 / 研究者申請制)。

**技術**:SynthID-Text 採用 tournament sampling 改寫 token 機率;SynthID-Image 用學習式 perturbation,皆強調感知不變性與抗壓縮。

**實作建議**:對輸出強制呼叫 SynthID embedding API;偵測端應同時跑 SynthID Detector + C2PA verifier 雙軌。

## 3. Meta Video Seal

**2024/12 Meta 開源 Video Seal** (Apache-2.0),**2025/03 升級 v1.0**,支援 256-bit payload、抗模糊 / 裁切 / H.264-265 壓縮;與 AudioSeal、WAM(Watermark-Anything-Model)合併為 **Meta Seal Suite**。

**突破**:時序一致 (temporal-consistent) embedding,在低位元率影片仍可解出;提供 PyTorch + ONNX runtime。

## 4. AudioSeal

Meta FAIR 2024 發表,**localized watermarking** 是核心創新——可在 1ms 解析度標示「哪段是 AI 合成」,而非整段二元判斷。對 MP3 壓縮、變速、剪輯、加噪都保持高 TPR。

**Seal v2** 引入 **chunked detector**,適合 streaming 場景(電話客服、即時直播),回應 AI voice cloning 詐騙浪潮。

## 5. 被動偵測 (Passive Detection)

文字偵測:**GPTZero、Originality.ai、Copyleaks、Turnitin**(教育 / SEO);影音深偽:**Reality Defender(2024 RSA Innovation Sandbox finalist)、Hive Moderation、TrueMedia、Optic、Sensity**;**Microsoft Video Authenticator** 在影格層給操弄機率;**Intel FakeCatcher** 用 PPG 血流訊號達 96% 準確。

**市場**:2025 預估 $0.58B → 2030 $2.06B(CAGR 28.8%)。

**心法**:被動偵測絕不能作唯一信號 — 在 C2PA / SynthID 失效時當第二防線;對抗演進極快,需訂閱式模型更新。

## 6. 訓練資料 opt-out 機制

多協議共存:
- **robots.txt** (per-bot User-Agent disallow)
- **ai.txt** (Spawning,網站根目錄宣告)
- **TDMRep** (W3C `tdmrep.json`,書籍出版業 ~80% 採用)
- **Do Not Train Registry** (Spawning, Have I Been Trained?)
- **ISCC** (ISO 24138:2024 內容指紋,抗 metadata 剝離)
- **TDM·AI Protocol** (Liccium,以 ISCC + Creator Credentials 做加密驗證)

C2PA 官方聲明 **不直接包含 TDM 授權語意**,正確做法是 **TDMRep + ISCC + C2PA manifest 內 `c2pa.training-mining` assertion** 三層疊加。**EU AI Act Article 53** 強制 GPAI 模型供應商辨識並遵守 TDM reservation。

**實作建議**:發布內容三件套 — 網站 `/.well-known/tdmrep.json` + 圖檔嵌入 ISCC + C2PA assertion。

## 7. TDM 例外法律比較

- **EU DSM Article 3**:研究機構無限制 TDM
- **EU DSM Article 4**:商業 TDM 但允許權利人「明確機器可讀方式保留」
- **英國** 2024/12 啟動公眾諮詢,傾向 EU-style opt-out
- **日本** 著作權法 §30-4 全球最寬鬆,允許非享受著作目的之 TDM 幾無 opt-out
- **美國** 仍仰賴 fair use 個案判斷(NYT v. OpenAI 等仍在審)

## 8. 硬體層 attestation

- **Leica M11-P** (2023/10) — 全球首款內建 C2PA 簽章相機
- **Sony Alpha 1/9 III/7 IV/7S III/7R V** — Creators' App 啟用 firmware v3.0+ 簽章
- **Nikon Z9、Canon EOS R5 II** — 跟進
- **Samsung Galaxy S25** (2025/01) — 首款主流 C2PA 手機,Snapdragon 8 Elite 安全處理單元儲存私鑰
- **Google Pixel 10** (2025) — Content Credentials
- **Truepic SDK / Vision** — 任意相機 App 嵌入硬體 attestation

**TEE / SEP 私鑰 + Trust List 階層 CA** 把信任根從「軟體簽章」拉到「裝置序號可驗證」。

## 9. 新聞 / 媒體採用

**Project Origin** (2020, CBC + BBC + NYT + Microsoft) 是 C2PA 編輯側源頭;2025 年 **France Télévisions** 成為全球**首個日常作業層級**全量套用 C2PA 的廣播商;**BBC、AFP、CBC/Radio-Canada、WDR、EBU** 都有試點;**IPTC** 與 C2PA 建立 **Verified News Publishers List**。

## 10. 法規與強制

- **中國** CAC 2025/03 發布《人工智慧生成合成內容標識辦法》+ **GB 45438-2025**,**2025/09/01 生效**。要求顯式 + 隱式標籤
- **EU AI Act Article 50** 強制合成內容須有 machine-readable 標示,**2026/08/02 全面生效**,罰則最高 €15M 或全球營收 3%
- **加州 SB 942 (AI Transparency Act)**:原 2026/01/01,**AB 853 延至 2026/08/02 對齊 EU**,實作指向 C2PA
- **科羅拉多 AI Act**:偏「高風險決策」如就業 / 信貸 / 醫療
- **美國聯邦** 2025 Trump EO 廢止 Biden EO,轉向行業自律

**實作建議**:跨境發布以 **EU + 中國** 為合規下限;同時部署「顯式標籤 (UI badge) + 隱式標籤 (C2PA + SynthID)」雙軌。

## 11. 攻防實驗

浮水印面臨三波攻擊:
1. **簡單擾動**(模糊、JPEG 壓縮、crop)
2. **學習式移除**(VAE remover、SpecGuard、InvisMark)
3. **生成式重寫**(diffusion-based regeneration,2024–2025 突破)— 做分佈級擾動,傳統 robust watermark 無法應對

C2PA 攻擊面包括 **trusted timestamp 竄改**、**「雙重簽章」攻擊**(用合法 issuer 重新簽一份指向偽造內容的 manifest)、**manifest 剝離後重新打包** — `c2pa-attacks` 工具專做這類 fuzzing。

**突破**:**Hash-binding** — 把 watermark payload 與 C2PA manifest content hash 綁定;**ensemble of watermark schemes**(SynthID + Video Seal 並行)增加去除成本。

## 12. 2025–2026 重大進展

- **TikTok**:2025/01 全面整合 C2PA,**已標 13 億+ AI 影片**(全球首個影音平台規模化採用)
- **Meta**:Instagram / Facebook 全面 C2PA 標籤;開源 Video Seal、AudioSeal、Watermark Anything
- **Google + C2PA**:2024 加入 Coalition steering committee;2025 Pixel 10、Google Photos、Google Search "About this image" 整合
- **Adobe Content Credentials Cloud**:2025 推出,對接 Behance、Lightroom、Premiere、Firefly 全家桶
- **YouTube**:創作者上傳 AI 影片強制聲明欄
- **OpenAI**:DALL·E 3、Sora 輸出皆含 C2PA

---

## 2026 AI 真實性工程師地圖

**四層構建**:

1. **Capture / Generation Layer(源頭簽章)**
   - 硬體:Leica / Sony / Nikon / Canon / Samsung S25 / Pixel 10 + Truepic SDK
   - 模型輸出:SynthID API / Video Seal / AudioSeal / 內部 forensic ID
   - 必做:TEE 內生私鑰、capture-time C2PA manifest、模態對應 watermark embedding

2. **Edit / Distribution Layer(編輯保鏈)**
   - 工具:Adobe CC + Content Credentials Cloud、DaVinci、ffmpeg-c2pa
   - 必做:每次轉碼 / 編輯都 append `c2pa.actions` claim;組織身份用 IPTC Verified Publishers
   - opt-out:同步寫入 `c2pa.training-mining` + ISCC + tdmrep.json

3. **Verify / Moderate Layer(平台驗證)**
   - C2PA verifier(`c2pa-rs`、`c2patool`) + SynthID Detector + Video/AudioSeal decoder
   - Passive ensemble:Reality Defender / Hive / FakeCatcher API
   - 必做:multi-signal scoring、calibrated confidence、人類審核回饋環

4. **Compliance / Audit Layer(合規與審計)**
   - 同時滿足 GB 45438-2025、EU AI Act Article 50、SB 942
   - opt-out 守則:遵守 Article 53,訓練前 enforce TDMRep + ai.txt + ISCC 黑名單
   - 攻防:`c2pa-attacks` fuzz、watermark robustness benchmark、定期 revoke + 更換 CA

**最終心法**:沒有單一銀彈 — **provenance (C2PA) 補 watermark 之不足、watermark (SynthID/Seal) 補 metadata 剝離之不足、passive detection 補未簽章內容之不足、法規 (Article 50 / GB 45438) 補技術滲透率之不足**。2026 起,真實性是 ML / 媒體系統設計的一等公民。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
