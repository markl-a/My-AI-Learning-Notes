# Case 03 — Real-time Voice Agent 客服系統(類 Bland AI / Vapi / Retell AI / Cresta)

> **題目類型**:電話客服 Voice Agent 系統設計
> **參考真實系統**:Bland AI、Vapi、Retell AI、Cresta、Hume AI EVI
> **同類題庫速覽**:[`../1.LLM面試題庫/04_系統設計題.md` Q3](../1.LLM面試題庫/04_系統設計題.md)
> **姊妹案例**:[Case_01 RAG](./Case_01_Enterprise_RAG_System.md);[Case_02 LLM Gateway](./Case_02_LLM_Gateway_API_Platform.md);[Case_04 Multi-Agent Research](./Case_04_Multi_Agent_Research_System.md)(待補)
> **延伸 deep-dive**:[`14.Voice_Audio_AI/`](../../14.Voice_Audio_AI/README.md)

---

## 題目

> **Design a real-time Voice Agent 客服系統(電話接入):平均 5000 同時通話、peak 15000;p50 端到端對話延遲 < 500ms、p99 < 800ms;支援英 / 中 / 西 / 印地 / 日五大語言;完整錄音 + 即時轉錄 + 情緒分析 + 結構化抽取;雙向整合 Salesforce / HubSpot / Zendesk;HIPAA + PCI 合規(病歷不入 LLM、卡號 DTMF mask);99.9% 可用,單一 region 故障自動切換。**

時間預算:60 分鐘白板 + 10 分鐘 Q&A。
聽眾預期:Staff / Senior Staff Engineer 級,熟悉電話系統 + 即時音訊處理者佳。

---

## 1. Clarification(5 分鐘,候選人主動提問)

Voice Agent 是少數「延遲決定產品死活」的題型 — 慢 200ms 用戶就感覺到「機器人感」。先把 use case 與打斷模式釘死,後面 budget 才有得算。

候選人應該主動問的 **10 個問題**:

1. **主要 use case**:外撥營銷、預約提醒、IVR 升級、客服 inbound、債務催收 — 不同 vertical 對 latency、合規、handoff 邏輯差異極大。
2. **方向**:純 outbound(我們主動撥)、純 inbound(用戶打進)、還是雙向?outbound 還要不要處理「答錄機偵測 + 留言」?
3. **打斷(barge-in)**:用戶開口要不要立即截斷 agent?半雙工(walkie-talkie)還是全雙工(自然對話)?全雙工成本與工程複雜度差一個量級。
4. **通話時長分布**:平均多長、最長多長(會影響 session memory、KV cache、context 截斷策略)?
5. **真人轉接(handoff)**:要不要 warm transfer(帶 context 給真人)?cold transfer 就只是 SIP REFER。
6. **多語切換**:user 撥入時前面 IVR 選語言,還是 agent 自動偵測 + 中途可切?自動偵測會增加 STT 成本與切換延遲。
7. **錄音與保存**:是否錄音?雙聲道(separate channel per side)還是混音?保存多久(7 天 / 90 天 / 7 年 HIPAA)?
8. **CRM 整合深度**:只是查詢(來電 enrich)?還是要 agent 在通話中寫入(建單、改訂單)?寫入要不要強一致?
9. **預算**:per-min all-in 成本目標?$0.10/min 是合理線,> $0.25/min 多半在虧錢做品牌。
10. **品牌音色 / Voice cloning**:用 stock voice(便宜)還是品牌定制音(每 tenant 一條)?後者牽涉同意書與 deep-fake 合規。

**假設用的答案**(以下 design 基於這些):
- Use case 混合:50% inbound 客服、30% outbound 預約提醒、20% IVR 升級
- 全雙工 + barge-in 必須(這是「不像機器人」的關鍵)
- 通話平均 5 分鐘、p99 < 15 分鐘
- Warm transfer 必須(企業客戶要求)
- 多語:user 自選(IVR 前置)+ agent 自動偵測 fallback
- 錄音 90 天熱、7 年冷(HIPAA tenant);PCI tenant 卡號片段不錄
- CRM 雙向(query + write),寫入容忍 5 秒 async
- 成本目標 $0.10/min blended
- Voice cloning 為 enterprise tier opt-in(需簽 voice consent)

---

## 2. Requirements

**Functional**
- SIP inbound + outbound(BYOC trunking + 託管 PSTN 雙模式)
- 即時 STT(streaming partial < 200ms latency)、LLM 推理、TTS streaming(TTFB < 100ms)
- 全雙工 + barge-in(用戶開口立即取消 LLM 與 flush TTS)
- Tool call:查訂單、查預約、改預約、轉真人、發短信
- 多語 5 種:英 / 中 / 西 / 印地 / 日,IVR 選 + STT 自動偵測
- 通話雙聲道錄音 + 即時轉錄 + 情緒分析 + 結構化抽取(call summary、disposition、intent)
- CRM 雙向:Salesforce / HubSpot 來電 enrich + 寫回 call disposition;Zendesk 自動建單
- Warm handoff:WebSocket 推 conversation context + transcript 給 agent desk

**Non-functional**
- **Scale**:5K concurrent calls(平均),15K peak;60K calls/hr peak、1.44M calls/day
- **Latency**:E2E 對話延遲(user 說完最後一個字 → agent 開始播放第一個字)p50 < 500ms、p99 < 800ms
- **Availability**:99.9% per region(月停機 ≤ 43.2 min);單 region 故障 < 30s 切換
- **Cost**:< $0.10/min blended(telephony $0.02 + STT $0.01 + LLM $0.03 + TTS $0.04)
- **合規**:HIPAA(BAA + 自託管 LLM)、PCI-DSS(DTMF mask + 卡號片段不轉錄不錄音)、SOC2、GDPR(EU tenant 資料留歐)
- **錄音保存**:90 天熱(S3 STANDARD)+ 7 年冷(GLACIER),encrypted at rest + KMS per-tenant

---

## 3. Capacity Estimation

```
通話量:
  Concurrent calls:  5K avg / 15K peak
  平均通話時長:       5 min
  Calls/hr:          5K × 60 / 5 = 60K/hr → peak 180K/hr
  Calls/day:         1.44M peak day

音訊頻寬:
  Codec:             Opus 16kHz 32 kbps(雙向)
  Per call b/w:      64 kbps(in+out)
  Aggregate peak:    15K × 64 kbps = ~960 Mbps in/out(媒體層)
  → SIP/RTP 走託管 SBC 或 LiveKit SFU,自家機房只看 transcript stream

LLM token 消耗:
  Per minute:        prompt ~2K tokens(含 system + history + tool defs)+ output ~1K = 3K tok/min
  Per call (5 min):  ~15K tokens
  Daily tokens:      1.44M calls × 15K = ~22B tokens/day peak
  LLM cost:          GPT-4o-mini blended ~$0.30/M weighted
                   → 22B × $0.30/M = ~$6.6K/day → ~$200K/月 LLM 單項

STT(Deepgram Nova-3 streaming):
  Per minute:        $0.0058/min(volume tier)
  Daily:             1.44M × 5 = 7.2M min/day × $0.0058 = ~$42K/day(!)
  → STT 是僅次於 LLM 的第二大成本,大規模時必須評估自託管 Whisper-turbo

TTS:
  字數估算:           agent 講 ~50% 通話時間 = 2.5 min/call
                    × 150 wpm = 375 words/call
  Cartesia Sonic:    $0.00003/字 ≈ $0.011/call
  Daily TTS:         1.44M × $0.011 ≈ $16K/day

儲存:
  錄音:              Opus 32 kbps × 5 min × 1.44M = ~1.7TB/day raw → 壓縮 ~700GB/day
                    × 90 天熱 = ~63 TB(S3 STANDARD)
                    × 7 年冷 = ~1.8 PB(GLACIER Deep Archive,$1/TB-mo ≈ $1.8K/月)
  Transcript:        每通話 ~10KB JSON × 1.44M = ~15 GB/day
                    × 7 年 = ~38 TB(可上 ClickHouse 列存)
  情緒 / 結構化:       每通話 ~2KB metadata × 1.44M = ~3 GB/day

每分鐘成本拆分(目標 $0.10/min):
  Telephony(SIP/PSTN): $0.015
  STT streaming:       $0.006
  LLM(blended):        $0.030
  TTS streaming:       $0.022
  錄音 + 儲存:          $0.005
  Infra + margin:       $0.022
  合計:                $0.100/min ← 達標,但每項都得守住
```

**關鍵發現**:Voice Agent 不像 RAG / Gateway 是 token 燒錢、是**每分鐘四段成本疊加**(telephony + STT + LLM + TTS),任何一段失控整個 unit economics 就崩。架構必須圍繞「latency budget 嚴格拆」「STT/TTS 可替換」「LLM 流量分層」做 trade-off。

---

## 4. High-Level Architecture

```
                                ┌──────────────────────────────────────┐
   PSTN / SIP Trunk  ───SIP───▶ │  Telephony Layer                     │
   (Twilio / Telnyx /            │  - SBC / SIP gateway                 │
    BYOC carrier)                │  - RTP media plane                   │
                                │  - DTMF detection (out-of-band)      │
                                └────────────┬─────────────────────────┘
                                             │ Opus / PCM stream
                                             ▼
                          ┌──────────────────────────────────────────┐
                          │  Media Server (LiveKit SFU / Pipecat)    │
                          │  - WebRTC ↔ SIP bridge                    │
                          │  - Per-call audio pipeline                │
                          │  - VAD + endpointing                      │
                          └────────────┬─────────────────────────────┘
                                       │ audio frames (20ms)
              ┌────────────────────────┼─────────────────────────┐
              │                        │                         │
   ┌──────────▼─────────┐  ┌───────────▼────────────┐  ┌─────────▼─────────┐
   │ STT Streaming      │  │ Recording Sidecar      │  │ DTMF / PCI mask   │
   │ - Deepgram Nova-3  │  │ - Dual-channel WAV →   │  │ - Detect 16-digit │
   │ - Whisper-turbo    │  │   Kafka → S3 (KMS)     │  │   sequence        │
   │   (self-host)      │  │ - Lifecycle to GLACIER │  │ - Beep + redact   │
   │ - Lang auto-detect │  └────────────────────────┘  └───────────────────┘
   └──────────┬─────────┘
              │ partial + final transcript
              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Orchestration / Dialog Manager                              │
   │  - Turn-taking state machine (LISTEN / THINK / SPEAK)        │
   │  - Semantic endpointing (LLM 判句尾)                          │
   │  - Conversation memory (last 10 turns + summary)             │
   │  - Tool router (CRM query / write / handoff)                 │
   │  - PII / PHI redaction (Presidio inline)                     │
   └────────────┬────────────────────────────────────┬───────────┘
                │                                    │
                ▼                                    ▼
   ┌────────────────────────┐          ┌────────────────────────────┐
   │ LLM Provider Layer     │          │ Tool Execution             │
   │ - GPT-4o-mini (default)│          │ - Salesforce / HubSpot API │
   │ - Claude 3.7 Sonnet    │          │ - Zendesk ticket create    │
   │   (hard / escalate)    │          │ - Internal order / booking │
   │ - Llama-3.3-70B self   │          │ - Handoff broker           │
   │   (HIPAA tenant)       │          └────────────────────────────┘
   │ - GPT-4o Realtime      │
   │   (speech-to-speech    │
   │    fast-path tier)     │
   └────────────┬───────────┘
                │ streamed tokens / audio
                ▼
   ┌────────────────────────────────────────────────────────┐
   │  TTS Streaming                                          │
   │  - Cartesia Sonic (TTFB ~90ms)                          │
   │  - ElevenLabs Turbo v2.5 (品牌音 + cloning)             │
   │  - Self-host XTTS-v2 (fallback / 成本敏感)              │
   └────────────┬───────────────────────────────────────────┘
                │ audio chunks
                ▼
        ┌───────────────┐
        │ Media Server  │  ─── RTP ───▶ PSTN ──▶ caller
        │ (TTS playback)│
        └───────────────┘

   Async Pipeline (Kafka):
      transcript stream → ClickHouse (real-time analytics)
                       → Emotion classifier (Hume / 自訓)
                       → Struct extractor (post-call LLM)
                       → CRM webhook out (call disposition)
                       → Handoff WebSocket → agent desk

   Multi-region:  us-east-1 / eu-west-1 / ap-southeast-1
                  Twilio primary ↔ Telnyx secondary(SIP carrier failover)
                  DNS / anycast 控制就近接入
```

---

## 5. Deep Dive

### 5.1 Transport 層(SIP / WebRTC / Media)

**選型 trade-off**:

| 方案 | 優 | 缺 | 適用 |
|---|---|---|---|
| Twilio Programmable Voice | 全球覆蓋、合規最齊、SDK 成熟 | 貴($0.013/min);客製受限 | MVP、中小流量 |
| Telnyx | 便宜($0.0035/min)、API 強 | 部分國家 coverage 弱 | 主規模、北美主要 |
| LiveKit + BYOC SIP | 完全自控、WebRTC native | 自己接 carrier、運維重 | 大規模 + 高度客製 |
| Pipecat + 自架 SBC | 開源最彈性 | 工程量大 | 已有電信經驗團隊 |

**本案選 LiveKit SIP Gateway + Telnyx primary + Twilio fallback**:LiveKit 把 SIP/RTP bridge 到 WebRTC 給上游 pipeline 用,媒體層統一;Telnyx 拿主流量壓成本、Twilio 作為 carrier-level failover(carrier 故障時 SIP REGISTER 切第二家,DNS SRV 控制)。

**踩雷點**:RTP jitter buffer 太大會額外吃 50–100ms latency,Voice Agent 場景 jitter buffer 必須 ≤ 40ms(犧牲一些 packet loss 容忍)。

### 5.2 VAD + Endpointing(雙保險)

「endpointing」(判斷用戶說完了沒)是 Voice Agent 最容易翻車的環節 — 太敏感會搶話,太遲鈍會卡 800ms。雙層判斷:

1. **聲學 VAD(Silero VAD v5)**:每 30ms 一次,延遲 < 10ms,判斷「有沒有人聲」。靜音 > 300ms 觸發初步 endpoint 候選。
2. **語意 endpointing(LLM-based)**:把目前 partial transcript 丟給小模型(Llama-3.2-3B 或 fine-tuned classifier),判斷「這句話語意完整了嗎」。例如 "I want to..." → not done;"I want to book a flight" → done。

兩層 AND 才觸發 turn end → 觸發 LLM 推理。對「嗯...」「let me think...」這類 disfluency 容忍更好,避免提前搶話。

**Trade-off**:語意 endpointing 多 ~50ms latency,但「機器人感」量化指標(用戶被搶話次數)降 70%,值得。

### 5.3 STT 選型 + 多語

| Provider | Latency(partial) | Cost/min | 多語 | 適用 |
|---|---|---|---|---|
| Deepgram Nova-3 | 180–250ms | $0.0058 | 36 種,自動偵測強 | **主流量** |
| AssemblyAI Universal | 250–400ms | $0.0065 | 99 種 | 備援 |
| OpenAI Whisper-large-v3 streaming | 300–500ms | $0.006 | 99 種 | 高準確度需求 |
| Whisper-turbo 自託管(faster-whisper + GPU) | 150–300ms | ~$0.002 攤提 | 99 種 | 大規模、HIPAA |
| Azure Speech | 200–350ms | $0.01 | 100+ | 醫療客戶(BAA) |

選 **Deepgram Nova-3 primary + Whisper-turbo 自託管 fallback**:Nova-3 partial latency 業界最低、語言自動偵測準;自託管 Whisper-turbo 給 HIPAA tenant(BAA 無法走 Deepgram 的 case)。

**Streaming 模式**:WebSocket,Deepgram 端每 100–200ms 推一次 interim transcript;final transcript 觸發後立刻丟下游。

**語言切換**:IVR 階段 user 選好(decision tree),通話中若 STT 偵測到語言切換(連續 3 個 final transcript 都是另一語言),動態切換 STT model + LLM system prompt 中的目標語言。

### 5.4 LLM Orchestration

**分層 routing**:

```
80% default → GPT-4o-mini       ($0.15/$0.60 per M tokens) — 簡單客服
15% hard   → Claude 3.7 Sonnet  ($3/$15 per M tokens) — 多輪推理、複雜 troubleshoot
 5% HIPAA  → Llama-3.3-70B(自託管 vLLM)— 醫療 tenant,prompt 永不出 VPC
```

**Hard query 判定**:對話輪數 > 5、tool call 連環失敗、user sentiment 偏 anger、context > 8K tokens → 升級。

**System prompt 結構**:
```
<role>你是 ACME 公司客服 agent...</role>
<crm_context>{caller_id 對應的 customer record}</crm_context>
<knowledge>{動態檢索的 top-3 FAQ chunks}</knowledge>
<tools>{允許的 tool list}</tools>
<style>用 zh-TW 回答,語氣親切但專業,單句 ≤ 30 字</style>
```

**句長控制**:system prompt 強制要求 agent **一次回應 ≤ 2 句、≤ 60 字**(中文)/ ≤ 30 words(英文)— 句太長會超出用戶耐心、且 TTS 越長越拖。

**KV cache 重用**:同一通話內 system prompt + CRM context 不變,Anthropic prompt cache / GPT-4o automatic cache 命中率 > 80%,input cost 砍半。

### 5.5 TTS Streaming(Cartesia / ElevenLabs)

| Provider | TTFB | Cost | 品牌音 | 適用 |
|---|---|---|---|---|
| Cartesia Sonic-2 | 75–90ms | $0.022/min | 自助 clone | **主流量** |
| ElevenLabs Turbo v2.5 | 150–250ms | $0.05/min | 最自然 | enterprise 品牌 |
| OpenAI TTS-1-hd | 300–500ms | $0.03/min | 6 stock voice | 不建議(慢) |
| XTTS-v2(自託管) | 200–400ms | $0.005 攤提 | 開源 clone | 成本敏感 / on-prem |

選 **Cartesia Sonic 主流量 + ElevenLabs 給品牌客戶**:Cartesia TTFB 是業界最低,差 100ms 就差「機器人感」一個等級。

**Streaming**:LLM token 一邊吐、TTS 一邊合成、Media server 一邊播放。**句子邊界切分** — LLM 吐到逗號 / 句號就把這段文字推給 TTS,而不是等整段完。這是把 E2E latency 從「LLM 全部完成 + TTS 全部完成」(可能 3 秒)壓到「LLM 首句完成 + TTS TTFB」(< 500ms)的關鍵。

### 5.6 Speech-to-Speech 替代路徑

**GPT-4o Realtime / Gemini Live native audio** 把 STT + LLM + TTS 合成一個模型,理論上:
- 延遲省 ~200ms(少兩段串接)
- 情感表達自然(不經過文字 token 損失韻律)
- 成本反而貴($0.06/min input audio + $0.24/min output audio)

**本案 trade-off**:**部分 tier 用、不全用**。
- ✓ 用於:高端 conversational tier(品牌客戶願意付溢價)、不需 fine-grained 控制
- ✗ 不用於:需嚴格 prompt 控制句長 / 用詞 / 合規檢查的客服流程;需要中途插入 tool call 結果再播放(目前 Realtime 對 tool call 結果回播控制弱)

Architecture 上做成可切換 backend:dialog manager 暴露同一介面,底下 plug-in cascaded pipeline 或 speech-to-speech。

### 5.7 Turn-taking 全雙工 + Barge-in

最複雜的工程細節,被搶話一次用戶就掛電話。狀態機:

```
States: IDLE → LISTENING → THINKING → SPEAKING → (interrupted) → LISTENING

Barge-in 觸發條件(用戶開口打斷 agent):
  1. VAD 偵測到 caller voice ≥ 200ms(避免 cough / ambient noise 誤觸發)
  2. STT partial 收到非空字串

Barge-in 動作(必須 < 50ms 完成):
  (a) Cancel LLM stream(HTTP/SSE close)— 停止計費
  (b) Flush TTS buffer(本地 + 已下發到 media server 的 buffer)
  (c) Send RTP silence(讓 caller 立刻聽到 agent 不講了)
  (d) State → LISTENING,丟棄已部分播放的句子(下次 LLM 要知道「我上次只說一半」,寫入 history)
```

**踩雷點**:
- TTS provider 端的 buffer(server 側) 也要 close — Cartesia / ElevenLabs 都有 `cancel` API,沒呼叫的話 server 還在合成、繼續扣費。
- 已下發到 RTP 的 packets 來不及取消(< 100ms 延遲) — accept it,但要在 conversation history 寫「agent 說了 'Sure, the booking is for...' 但被打斷」,讓 LLM 知道下次別重複。
- Echo 問題:caller 端揚聲器播放的 agent 聲音被 mic 收回 → 誤觸發 barge-in。解法:**echo cancellation 在 SBC 端做**(WebRTC AEC 或 Speex AEC)。

### 5.8 PII / PCI / HIPAA 處理

三層防線:

**1. DTMF mode 收卡號(PCI)**:
agent 講「請用鍵盤輸入信用卡號」→ 切換到 DTMF capture 模式 → RTP 內 DTMF 數位 out-of-band 收集,**永遠不進 STT、不進 LLM、不錄音**。直接從 SBC 推到 PCI vault(Stripe / Spreedly),返回 token 給後續流程使用。

**2. Inline PII redaction(Presidio + regex)**:
STT 輸出後、進 LLM 前過一層 Presidio,把 SSN / 信用卡 / 病歷號替換成 `<SSN>` `<CCN>` placeholder。LLM 看到的是 redacted 版本。Outbound TTS 之前再做一次(避免 LLM hallucinate 出 PII)。

**3. HIPAA tenant 隔離(BAA + 自託管)**:
醫療客戶簽 BAA 後路由到獨立 stack:
- STT:Whisper-turbo 自託管(VPC)、不發 Deepgram
- LLM:Llama-3.3-70B 自託管 vLLM、不發 OpenAI / Anthropic
- TTS:XTTS-v2 自託管
- 錄音:per-tenant KMS、physical AZ isolation、access log Splunk → SIEM

對應 control plane 在 routing 階段查 tenant flag,整個媒體路徑分流。代價是 HIPAA tier 成本 ~$0.18/min(差距由企業合約溢價彌補)。

### 5.9 Recording + 即時 Analytics

**錄音 sidecar**:
- 雙聲道 WAV(caller left、agent right)→ 邊錄邊推 Kafka(每 5 秒 chunk)
- S3 multipart upload(per-tenant bucket + KMS)
- 90 天 STANDARD → lifecycle → GLACIER Deep Archive(7 年)
- PCI tenant 在錄音當下 mute 卡號段(DTMF 期間直接寫靜音 sample)

**即時分析 pipeline**(Kafka → consumer 群):
- **Emotion classifier**:用 Hume EVI API 或自訓 wav2vec2-emotion,每 10 秒一次推 sentiment time-series → ClickHouse
- **Live transcript**:Deepgram final transcript 推 ClickHouse + WebSocket 給監聽真人 agent
- **Post-call extractor**:通話結束觸發 LLM 跑 struct extract → `{intent, disposition, customer_satisfaction, action_items, follow_up_required}` → 回寫 CRM
- **Compliance scanner**:離線跑 keyword + LLM 過濾「agent 是否說了違規話術」,觸發 QA 工單

### 5.10 CRM 整合(Salesforce / HubSpot / Zendesk)

**Inbound enrichment**(同步,latency-critical):
SIP INVITE 進來 → 從 From header 拿 caller phone → 並行查 Salesforce / HubSpot(2 秒 timeout)→ 把 customer record 注入 LLM system prompt → agent 開口就知道 user 是誰 / 上次什麼問題。

查失敗 → fallback「請問怎麼稱呼您」,不阻塞通話開始。

**Outbound write**(async,容忍 5 秒):
通話中 agent 透過 tool call 改訂單 / 建單 → 寫入請求進 Kafka → consumer 重試 + idempotency(用 `call_sid + tool_call_id` 當 key)→ 成功後回拋 webhook 給用戶系統。

**Post-call sync**:
通話結束 → struct extract 結果(disposition、summary、tags)→ 並行寫 Salesforce activity + Zendesk ticket(若 disposition = unresolved)+ HubSpot timeline event。

**踩雷點**:Salesforce API 有 daily limit(企業版 15K calls/24hr),5K concurrent 規模下要做 batch + 短時 buffer(把 5 分鐘內同一 account 的多個事件 merge 成一筆 update)。

### 5.11 Handoff to Human

Warm transfer 流程:
1. Agent 偵測升級條件(用戶要求 / sentiment < threshold / tool call 失敗 2 次) → 觸發 `transfer_to_human` tool
2. Handoff broker 查 agent desk 可用真人(Genesys / 自架排隊系統)→ 找到後配對
3. **WebSocket push** 完整 conversation context + live transcript stream + emotion score 給真人 dashboard(真人有 ~10 秒看完摘要)
4. SIP REFER + Replaces(三方通話模式)→ caller 進 conference,agent bot 講「正在為您轉接,我會留在線上協助」
5. 真人接入後 agent bot 進入 listen-only 模式(繼續轉錄、隨時可被叫回)

對應 case 03 sister 系統(real-time assist)就是把 listen-only agent bot 反向用 — 真人在前、AI 在後台即時提示話術。

### 5.12 Multi-region + Failover

**主架構**:3 region active-active(us-east / eu-west / ap-southeast),用戶就近接入(DNS GeoIP + Anycast IP)。

**Failover 層級**:
- **Carrier 層**:Telnyx 不通 → SIP SRV 切 Twilio(< 5s,DNS 控制)
- **Media server 層**:LiveKit 單節點掛 → SFU mesh 內 re-elect,call 不中斷
- **STT/TTS 層**:Provider 5xx > 3 連續 → circuit breaker 切 fallback
- **LLM 層**:延用 [Case_02](./Case_02_LLM_Gateway_API_Platform.md) gateway 同套 hedged request + provider fallback
- **Region 層**:整個 region 掛 → DNS health check 30s 內切其他 region;**進行中通話會掉**(SIP 沒辦法跨 region 無縫遷移,這是現實妥協)— 但 99.9% SLA 含「中斷後 30s 內可重撥成功」

> phantom-mesh 的 **provider fallback 模組** 在這裡擔任 STT / LLM / TTS 三層的統一 circuit breaker + hedged request 邏輯,把 Case_02 Gateway 那套 5 秒切換能力複用過來。

---

## 6. Bottlenecks 與 Mitigation

| Bottleneck | 症狀 | Mitigation |
|---|---|---|
| TTS TTFB > 200ms | 對話有明顯停頓 | 換 Cartesia Sonic(~90ms)或直接 GPT-4o Realtime;預載常用 phrase("讓我查一下"); first-token 預熱 |
| LLM TTFT 慢 | thinking → speaking 切換卡頓 | 用 Anthropic prompt cache + KV reuse;預先送 partial prompt(用戶說到 80% 就 speculative 開推) |
| Streaming SSE 多 provider 格式 | 寫不完的 adapter 與 edge case | **phantom-mesh 統一 SSE parser**(handle Deepgram / OpenAI Realtime / Anthropic / Gemini Live 4 種格式),上層 dialog manager 不感知 provider |
| 通話峰值突發(午餐後 +200%) | LLM provider rate limit、SBC 不夠 | Auto-scale(LiveKit on K8s HPA by concurrent_calls)+ warm pool(預留 20% capacity)+ provider quota 預申請 |
| Carrier 故障 | 整 region 不能撥入 | Twilio ↔ Telnyx dual-trunking + DNS SRV failover < 5s |
| Region 故障 | 進行中通話全掉 | 接受現實(SIP 不可跨 region migrate);用 DNS health check 30s 切新 call 路由 + 自動回撥失敗用戶(opt-in) |
| 錄音 S3 寫入塞爆 | upload throttle / cost spike | Kafka 緩衝 + multipart parallel upload + per-tenant prefix 散熱(避免 hot partition) |
| CRM API daily limit | Salesforce 15K/day 撞牆 | Per-account 短時 buffer(5 min merge) + composite API + 企業客戶簽 unlimited tier |
| 機器人感(用戶提前掛斷) | retention 低 | 監控 average call duration + 掛斷時點分布;A/B test voice / prompt / endpointing 設定 |

---

## 7. Trade-offs(明確表態,別騎牆)

| 決定 | 選 A | 選 B | 我的選擇 |
|---|---|---|---|
| Cascaded(STT+LLM+TTS)vs Speech-to-Speech | 三段串聯(成本可控、細粒度控制) | S2S 一體(低延遲、自然) | **預設 cascaded、品牌高 tier 走 S2S**:大部分流量需要 prompt 控制與 tool call;S2S 給願付溢價的旗艦客戶 |
| 自託管 vs API | 自託管(latency 穩、合規可控、固定成本) | API(輕資產、彈性) | **API 為主、HIPAA 與旗艦 voice 走自託管**:量未到自託管全面甜蜜點($1M/month LLM cost 才是分水嶺) |
| 全雙工 barge-in vs 半雙工 | 全雙工(自然但複雜) | 半雙工(簡單但機器人) | **全雙工**:這是「不像機器人」的核心,值得多 ~30% 工程量 |
| 錄音預設 on/off | 預設錄(品質與 dispute 重要) | 預設不錄(隱私友好) | **預設錄 + 開頭強制告知**(法規要求 two-party consent state 必須告知),HIPAA tenant 提供「不錄」選項但 default 也是錄(治療 dispute) |
| Voice cloning 開放程度 | 開放自助 clone(增長快) | 強制人工審核 + 同意書 | **強制同意書 + 自動 deep-fake detector**:防 fraud 用於詐騙電話 |
| 即時情緒分析 deep model vs 簡單 | 端到端 audio emotion(準) | 文字 sentiment(便宜) | **混合**:文字 sentiment 即時(每輪)+ audio emotion 後處理(每通話)— audio 模型 GPU 成本不適合每 10s 跑全量 |

---

## 8. Extension 題(面試官可能追問)

1. **若 80% 客戶要求 voice cloning(品牌一致音色)**:技術上 ElevenLabs Pro / Cartesia Voice Designer 都能 clone(30s 樣本即可);但合規上是地雷 — 必須收 **voice consent**(被 clone 者書面同意 + face-to-face attestation),並在系統內加 **deepfake watermark**(audio inaudible signature)+ **misuse detector**(偵測 clone 聲音用於詐騙場景就 kill switch)。架構上每 tenant 一條 voice profile,TTS provider 端 voice_id 隔離,絕對禁止跨租戶混用。

2. **真人 + AI 共駕(real-time agent assist)**:把 listen-only 模式反向用 — 真人在前接 caller,AI 在後台即時:(a) 跑 RAG 搜內部知識庫 → 推薦話術;(b) 偵測 next-best-action(refund / discount / escalation);(c) 偵測 compliance violation(real-time 警告真人「剛才那句不能說」)。差異:沒有 TTS 路徑、LLM output 走 WebSocket 給 agent dashboard;但 STT + transcript pipeline 完全複用。對應 Cresta、Observe.AI 的核心產品形態。

3. **HIPAA 客戶要求對話資料絕對不可被外部 LLM 看到**:全自託管 stack — Whisper-turbo + Llama-3.3-70B vLLM + XTTS-v2 + LiveKit BYOC SIP,全部跑在 tenant VPC 或專屬 K8s cluster。Audit log 全量留存 Splunk,KMS 用 tenant CMK(我們無 root access)。代價:成本 ~$0.18/min(vs API tier $0.10)、品質 NPS 略低(自託管 Llama 70B vs GPT-4o-mini 在客服場景大致打平,但 hard query 落後)。商業上靠企業合約溢價彌補。

4. **如何偵測「機器人感」並量化改善**:核心 metric 是 **early hangup rate**(通話 < 30s 掛斷率)、**interrupt-to-handoff ratio**(用戶打斷後直接要求轉真人)、**post-call CSAT**(短信問卷)。深度 metric:對話輪數分布(太少 = 用戶不想互動)、user sentiment 隨時間走勢(越聊越差 = 機器人感累積)、agent 被搶話次數。對應 A/B test 變量:VAD threshold、語意 endpointing model、TTS voice(品牌音 vs stock)、句長控制 prompt。每週更新 routing weights。

---

## phantom-mesh 在本系統的角色(回應 Case_01 / Case_02 一脈)

- **統一 SSE 解析器**:Voice Agent pipeline 中 STT(Deepgram WebSocket)、LLM(OpenAI / Anthropic SSE)、Realtime(OpenAI Realtime WS)、TTS(Cartesia stream)四套 streaming 協定差異,phantom-mesh 的 `stream_parser` 模組把它們 normalize 成統一 event(`{type: partial_transcript | token | audio_chunk, ...}`),上層 dialog manager 不感知 provider。
- **Provider Fallback**:延用 [Case_02 LLM Gateway](./Case_02_LLM_Gateway_API_Platform.md) 的 hedged request + circuit breaker,擴展到 STT / TTS 兩層 — STT 主備(Deepgram → Whisper-turbo)、TTS 主備(Cartesia → ElevenLabs → XTTS)。
- **Cost Tracker per-call**:把 telephony + STT + LLM + TTS 四段成本即時彙整到 per-call ledger,超過 budget threshold(預設 $0.50/call 觸發 alert) 立刻 PagerDuty + 自動降級到便宜 stack(GPT-4o-mini + Cartesia + Whisper-turbo)。對齊 phantom-mesh 的 `cost_attribution` 模組。

---

## 結語(白板下台前 30 秒)

> 「總結:這套 Voice Agent 系統用 LiveKit + Telnyx 接 SIP / WebRTC,STT 走 Deepgram Nova-3 主流量(HIPAA 切自託管 Whisper)、LLM 分層 GPT-4o-mini / Sonnet / 自託管 Llama 三層、TTS 用 Cartesia Sonic 拿低 TTFB。Latency budget 嚴格守:VAD 30ms + STT 200ms + LLM 250ms + TTS 90ms + 網路 100ms = p50 < 500ms,雙層 endpointing + barge-in 把『機器人感』壓下來。合規上 DTMF mask 處理 PCI、Presidio inline redact PII、HIPAA tenant 全自託管 stack。CRM 雙向走 Salesforce / HubSpot inbound enrich + async write back。phantom-mesh 在 streaming 統一、provider fallback、per-call cost attribution 三處複用。最大風險是 TTS TTFB 與 region failover,我會用 Cartesia 預熱池 + 接受『進行中通話不可跨 region』的現實,改用主動回撥補償。下一步兩件事:(1) 推 GPT-4o Realtime fast-path 給旗艦 tier、(2) 上 real-time agent assist 拿企業客戶。」

---

### 面試官最會追問的 3 個 follow-up

1. **「p99 800ms 真的能撐住嗎,給我看每段 budget」** — 答:VAD 30ms + STT final delay 200ms + LLM TTFT(prompt cache hit)200ms + TTS TTFB 90ms + RTP 一個來回 100ms ≈ 620ms,留 180ms 給網路抖動與 jitter buffer;p99 抖到 800ms 的主因是 LLM TTFT spike(provider 端),用 hedged request + Anthropic / Gemini fallback 補。
2. **「全雙工 barge-in 如果 echo 沒消乾淨會怎樣」** — 答:用戶會被 agent 自己的聲音當成「user 在說話」觸發誤打斷,agent 開頭一句就被截斷,直接被掛電話。所以 echo cancellation 必須在 SBC 端(WebRTC AEC)而非用戶端,並且做 **double-talk detection**(同時兩邊有聲音時,VAD 信心分要拉高 threshold)。
3. **「HIPAA tenant 用自託管 Llama 70B,品質真的夠嗎」** — 誠實答:在 routine 客服(預約查詢、FAQ)上跟 GPT-4o-mini 打平甚至更好(因為 fine-tune 過 vertical data);在 hard reasoning 與罕見場景上落後 ~5–8pp,所以醫療場景我會限制 agent 不做診斷類對話、只做 admin / scheduling / triage routing,真正醫療判斷強制 handoff 給真人。

---

> 後續案例 Case_04 ~ Case_05 待補:
> - Case 04:Multi-Agent Research Platform(orchestration、long horizon、self-critique)
> - Case 05:Computer-Use SaaS(sandbox、安全、replay)

返回:[`./README.md`](./README.md) | [`../1.LLM面試題庫/04_系統設計題.md`](../1.LLM面試題庫/04_系統設計題.md)
