# 14. Voice AI / Voice Agents / Audio AI (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #18
> 語音 AI 24 個月完成「從串接 pipeline 走向原生語音 LLM」的典範轉移。

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **Transformer 基礎**(對應 repo:[2.深入LLM模型工程與LLM運維/1.LLM 基礎與架構](../2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/README.md))
> 2. **多模態基礎**(對應 repo:[Multimodal_Generation_2024-2026.md](../1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md))
> 3. **串流系統 / WebRTC 概念**(若 repo 內無,先看:[MDN WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) 與 [LiveKit 文件](https://docs.livekit.io/))
>
> 不熟悉 audio signal、Mel-spectrogram、tokenizer 等概念請先看 [全景圖 #18](../2024-2026_AI完整領域全景圖.md) 對應章節。
>
> **延伸 / 反向連結**:[3.LLM應用工程/3.Agent](../3.LLM應用工程/3.Agent/) | [Multimodal_Generation_2024-2026.md](../1.從AI到LLM基礎/4.DL/Multimodal_Generation_2024-2026.md)

> **⚠️ 鮮度與可信度說明 / Freshness & Reliability**
> 本章涉及 **TTS / ASR 模型版本(ElevenLabs v3 / Suno v5 / OpenAI Realtime / Cartesia Sonic / Deepgram Nova-3 等)、發布月份、latency 數據、價格、商業案例規模** 等高頻變動數字。這些資訊混合三類來源:
> 1. **官方文件 / model card / 公開定價頁**(高信任度)
> 2. **媒體報導 / vendor blog / 業內 changelog**(屬「報導」非「事實」,且 vendor 自報指標常無公正基準)
> 3. **AI agent 整理 + 我手動驗證**(可能有誤標,尤其 ElevenLabs v3 GA 月份、Suno 時長、Realtime API 規格細節)
>
> 任何要拿來做選型 / 報價 / 架構決策的數字(latency、價格、context window、語言支援、商用條款),**請以原廠官方文件為準**。本章對「從串接 pipeline 走向原生語音 LLM」這類 *典範轉移敘事* 較高信心,對「2026.03 ElevenLabs v3 GA」這類 *精確時點* 較低信心。

---

## 1. ASR(語音辨識)

三足鼎立:
- **雲端商用**:Deepgram Nova-3 (~6.84% WER 串流, <300ms)、AssemblyAI Universal-2 (LibriSpeech ~2.1% WER)、ElevenLabs Scribe (99 種語言)
- **OpenAI 系**:Whisper v3 large / turbo (4× 加速、WER 退 0.3%)、gpt-4o-transcribe
- **自部署蒸餾**:Distil-Whisper (Whisper-large 6× 加速)、WhisperX (強制對齊 + diarization)

**選型**:電話 8kHz 用 Nova-3;會議寬頻多語用 Scribe / Universal-2;敏感資料離線跑 Distil-Whisper-large-v3。串流場景永遠開 partial transcripts + endpoint detection。

## 2. TTS(語音合成)

- **雲端**:ElevenLabs v3 (Elo ~1178)、OpenAI TTS / Realtime、Cartesia Sonic (TTFB ~90ms)、Google Chirp 3 HD
- **開源**:Sesame CSM-1B (MOS 4.7,非語言聲學線索)、Kokoro-82M (CPU/WebGPU)、F5-TTS、XTTS-v2、Bark、MaskGCT、Orpheus-3B

**ElevenLabs v3** (2026/03 GA):audio tags `[laughs][whispers]`、70+ 語言、`eleven_v3_conversational` 為 agent 優化。

**選型**:Agent → Cartesia 或 ElevenLabs Turbo v2.5;角色/敘事 → ElevenLabs v3 + audio tags;邊緣 → Kokoro;客製 → F5-TTS / CSM。

## 3. 語音克隆 / Voice Cloning

商用:ElevenLabs Instant (60s 樣本) / Professional (30min+)。開源:XTTS-v2、OpenVoice v2、F5-TTS、MaskGCT、Tortoise(已退場)。

**deepfake 爆增**:2024-2025 多起 CEO voice scam,FCC 已將 AI-generated robocall 違法化。ElevenLabs / Resemble / Meta 都在輸出嵌入 audio watermark(AudioSeal、SynthID-Audio)。

**心法**:商用部署一律要求 voice consent / proof-of-life;對外 API 強制 watermark + provenance log。

## 4. Speech-to-Speech 原生模型

- **雲端**:GPT-4o Realtime / GPT-Realtime-2、Gemini 2.5 Flash Native Audio、Hume EVI 3
- **開源**:Moshi (Kyutai, ~200ms 延遲、雙工)、Mini-Omni、Qwen2.5-Omni、GLM-4-Voice

直接吃 audio tokens、吐 audio tokens,保留語氣節奏情緒,端到端 200-600ms。

**價差**:Gemini 2.5 Flash Native Audio ($0.00165/min) vs GPT-4o Realtime (~$0.30/min) — **價差 182x**。

**選型**:電話客服 → Gemini Live 或 Realtime + Cartesia;高端 IVR / coach / therapy → GPT-4o Realtime 或 Hume EVI;研究 / on-prem → Moshi。**工具呼叫**仍弱於文字 LLM,複雜業務流程建議混合架構。

## 5. 即時語音 Agent 框架

兩大開源贏家:
- **Pipecat (Daily.co)** — Python pipeline-frame 模型,DX 對資料/後端工程師友善
- **LiveKit Agents** — WebRTC SFU + Agents 一體,適合 10k+ 同時連線

託管平台:**Vapi / Retell AI / Bland AI** ($0.07-0.15/min);**Cartesia** 從 TTS 上探到 agent stack;**OpenAI Realtime API** 是最簡單的 1:1 起手式。

**選型**:PoC 用 Vapi/Retell 一週上線;量大用 LiveKit;客製 pipeline 用 Pipecat;1:1 demo 用 OpenAI Realtime。turn-taking 必上 Silero VAD + semantic endpoint 雙保險。

## 6. 音樂生成

- **Suno v5.5** — 2M 付費用戶、$300M ARR、8 分鐘長度、Suno Studio DAW、voice cloning、stems 分離
- **Udio** — audiophile 路線、48kHz stereo、inpainting editor
- **Stable Audio 2.0** — 商用授權合規路線
- **MusicGen / Riffusion** — 開源研究底座

業界報導 Suno 已與多家版權方達成框架性合作(具體授權範圍與時間以雙方公告為準),版權戰逐步收斂為「按播放分成」模式。

**選型**:廣告/podcast intro 用 Suno API + 自有 lyrics;遊戲 BGM 用 Stable Audio 2.0(授權乾淨);研究/客製 fine-tune 用 MusicGen-large。

## 7. 音訊事件 / 環境音 / 音效

Meta **AudioGen**、**Stable Audio FX**、**ElevenLabs SFX** 主導 text-to-SFX;**AudioLDM 2、TangoFlux** 是開源後繼。可生成爆炸、雨聲、腳步、UI feedback、Foley 等 0.5-30s clips。

## 8. 延遲工程

傳統 STT→LLM→TTS pipeline 過去 1000-2000ms,2026 透過串流化與 speech-LLM 可降至 200-600ms。**人類自然對話 gap 是 200-300ms**;>300ms 用戶下意識察覺,>500ms 顯著卡頓。

**Latency budget 範例**:VAD 50ms + STT first partial 150ms + LLM TTFT 200ms + TTS TTFB 100ms ≈ 500ms。

**全雙工架構必備 cancellation pipeline** — 使用者一開口立即 (a) 取消 LLM stream,(b) flush TTS buffer,(c) 截斷 outbound audio。

## 9. 生產化考量(電話、合規、成本)

電話側:**Twilio**(成熟、HIPAA/PCI 工具最全、SIP outbound $0.010-0.062/min)vs **Telnyx**(SIP outbound ~$0.005/min、SOC2/HIPAA/PCI-ready)。

Vapi/Retell/Bland 整套打包 $0.07-0.15/min,**LiveKit + 自選 STT/LLM/TTS 自組可壓到 $0.03-0.06/min**。

**PCI compliance 模式**:LLM 偵測到 card-collection intent → 切到 DTMF-only → token 化後回 LLM。HIPAA 強制 on-prem 或 BAA-cover。

## 10. 2026 必知里程碑

- 2024.05 **GPT-4o Realtime** 發布
- 2024.07 **Moshi** 開源全雙工語音 LLM
- 2024.10 **OpenAI Realtime API** GA;Pipecat / LiveKit 跟進
- 2025 上 **Cartesia Sonic**、**Sesame CSM**、**Kokoro** 把 TTS 開源化推到 production-ready
- 2025 下 **Gemini 2.5 Flash Native Audio** 把成本壓到 GPT-4o 的 1/180
- 2026.03 **ElevenLabs v3** GA、**Suno v5.5** 發布 8 分鐘長度
- 2026 H1 多家發布跨語語音翻譯與全雙工模型(具體規格與發布時程隨各家公告為準)

---

## 2026 Voice AI 工程師地圖

stack 5 層:**Transport (WebRTC/SIP)** → **Capture (VAD + STT 或 native audio)** → **Cognition (LLM 或 Speech-LLM + tools)** → **Voice (TTS 或 native audio out)** → **Orchestration (Pipecat/LiveKit/Vapi)**

**選型決策樹**:
- Demo / 1:1 web → OpenAI Realtime API + browser WebRTC
- 客服電話量產 → LiveKit Agents 或 Pipecat + Deepgram Nova-3 + Claude/GPT-4o + Cartesia + Telnyx SIP,目標 $0.05-0.10/min、p50 < 500ms
- 高合規(醫療、金融)→ Azure OpenAI + Azure Speech + Twilio (BAA) + 自托管 redaction
- 隱私 / on-prem → Moshi 或 Pipecat + Whisper-turbo + Llama-3 + Kokoro
- 創作 / 媒體 → ElevenLabs v3 + Suno v5.5 + Stable Audio FX

**核心技能**:VAD/endpointing tuning、turn-taking、interruption cancellation、TTS streaming chunking、latency budgeting、observability、PII redaction、SIP 與 WebRTC 基礎、speech-LLM vs cascaded pipeline trade-off。

**2026 真正的競爭已不在「能不能 demo」,而在「能不能用 $0.05/min、p99 < 800ms、99.5% 通話完成率,在電話另一端取代真人 agent」**。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
