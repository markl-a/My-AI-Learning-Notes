> 對應 [全景圖 #3](../../2024-2026_AI完整領域全景圖.md) §5;搭配 [`../1.LLM 基礎與架構/注意力機制最新變體與優化.md`](../1.LLM%20基礎與架構/注意力機制最新變體與優化.md)

# Long Context 完整指南:YaRN / LongRoPE / iRoPE / NSA / MoBA(2024 → 2026)

從 2023 年底 GPT-4 的 32K、Claude 2 的 100K,到 2026 年 Llama 4 Scout 宣稱的 10M context,長上下文已是旗艦模型的標配戰場。這份指南把位置編碼、稀疏注意力、分散式訓練、KV cache 壓縮、部署層優化、評估方法,從理論到工程實踐串成一條線,並對比 2026 年主要旗艦的實際能力。

---

## 1. 長 context 從 128K 到 10M 的演進

- **2023 H2**:GPT-4-32K、Claude 2 100K、Anthropic 推出 prompt caching 概念雛形。MPT-7B-StoryWriter 用 ALiBi 嘗試 65K。
- **2024 H1**:Gemini 1.5 Pro 首次端出 1M context(內部測到 10M),掀起競賽。Llama 3 系列 8K → 128K(YaRN-style 外推 + 繼續預訓練)。
- **2024 H2**:Claude 3.5 Sonnet 200K、GPT-4o 128K、Qwen 2.5 128K → 1M(雙版本)。
- **2025 H1**:DeepSeek V3 / R1 128K;Moonshot Kimi K1.5 推出 MoBA;DeepSeek 發表 NSA 論文。
- **2025 H2**:GPT-5 系列 400K、Claude Sonnet 4 / Opus 4 達 1M(beta)、Gemini 2.5 Pro 2M、xAI Grok 4 fast 2M。
- **2026 Q1**:Llama 4 Scout 用 iRoPE 訓練到 256K、宣稱外推 10M;DeepSeek V3.2 推出 DSA,KV cache 比 NSA 再降 50%。

關鍵轉折是 2024 中:從「把 RoPE base 拉大、硬訓練更長」轉向「位置編碼數學設計 + 注意力稀疏化 + 部署層 PagedAttention」三條腿並進。

---

## 2. 位置編碼譜系

### 2.1 RoPE(Su et al. 2021)基礎

Rotary Position Embedding 把位置編成複數旋轉:對 query / key 第 $i$ 對維度,套用旋轉矩陣 $R(m\theta_i)$,其中 $\theta_i = b^{-2i/d}$,$b$ 是 base(原始 10000)。優點是**相對位置自然編碼**(內積只依賴 $m-n$)、可外推、易於 KV cache。問題是訓練分佈外位置會 OOD,直接外推到 4× 訓練長度就崩。

### 2.2 NTK-aware Scaling(reddit u/bloc97, 2023/06)

直接把 base 從 10000 改成 $10000 \cdot s^{d/(d-2)}$($s$ 為延長倍數),讓高頻維度幾乎不變、低頻維度才被壓縮。**完全免訓練**就能 zero-shot 拉長 2-4×,代價是高頻位置仍有少量退化。

### 2.3 Position Interpolation(Meta, 2023/06)

Chen et al. 直接把位置線性壓縮 $m \to m/s$,等於把訓練分佈內的 2K 位置「拉伸」覆蓋到 8K。需要 1000 步微調,長 context 表現提升明顯,但**高頻短距資訊嚴重糊化**。

### 2.4 YaRN(Nous Research, 2023/11)

Peng et al. 把 NTK 與 PI 合體:對不同頻率維度採取不同插值策略——高頻維度保持原樣(不插值)、低頻維度走 PI、中間頻率用 ramp 過渡;再加上「attention temperature 校正」 $\sqrt{1/t} \log s$ 去補償長序列時 attention entropy 增大的問題。

實測 YaRN 達到同等 perplexity **只需 PI 的 1/10 token、1/2.5 訓練步數**,成為 Llama 2/3、Mistral、Qwen 系列的主流外推方案。YaRN fine-tune 的入門程式碼僅約 50 行(修改 `transformers` 的 `LlamaRotaryEmbedding` 即可),社群多用 `nous-yarn` repo。

### 2.5 LongRoPE(Microsoft, 2024/02)

Ding et al. 觀察 YaRN 用「一條公式」決定所有維度插值率仍非最優,改用**演化搜尋(evolutionary search)在每個維度上找最佳 scaling factor**;再分兩階段:先擴 256K,再擴 2048K。LongRoPE 把 LLaMA2 一路推到 **2M+ context**,在 4K 短任務上幾乎無退化,是目前長度紀錄保持者中最成熟的方案。Phi-3-mini-128K 即採用 LongRoPE。

### 2.6 iRoPE(Llama 4 Scout, 2025/04)

Meta Llama 4 提出 **interleaved RoPE**:奇數層用 RoPE,偶數層完全**不加任何位置編碼(no-PE / NoPE)**,並對 no-PE 層在推論時用 inference-time temperature scaling。其核心洞察:NoPE 層理論上具有長度泛化能力(Kazemnejad 2023),只是會被有 PE 的層帶歪;若把它們**穿插**,就能保留外推能力又不失精度。Scout 只在 **256K** 上訓練,卻能評估到 **10M** 而不崩,是目前外推比最高的設計。

---

## 3. Native Sparse Attention(NSA, DeepSeek 2025/02)

Lu, Yuan et al. 提出**訓練時就稀疏**的 NSA,而非事後剪枝。三條並行通道,每個 query 同時參考:

1. **Compressed branch**:把過去 KV 以 block 為單位平均/MLP 壓成代表 token,粗看全局。
2. **Selected branch**:用 compressed branch 的 score 動態挑出 top-k 個原始 block,精看關鍵段。
3. **Sliding window**:固定大小近鄰,保短距資訊。

三路 attention 輸出加權合併。訓練端用客製 Triton kernel,**FlashAttention-2 forward 9×、backward 6× 加速**,且能端到端訓練(梯度經 selection 用 straight-through)。NSA 是首個在 27B 規模下「速度與品質雙贏」的稀疏注意力方案,目前已被 DeepSeek V3.1 / V3.2 採用為預設機制。

---

## 4. MoBA — Mixture of Block Attention(Moonshot, 2025/02)

Kimi 團隊把 MoE 的 routing 思想搬到 attention:**把 KV 切成 block,用 gating 網路為每個 query 路由到 top-k 個 block**,只在被選中的 block 內做 full attention。與 NSA 對比:

| 維度 | NSA | MoBA |
|------|-----|------|
| 稀疏結構 | 固定三路(壓縮+選擇+窗) | 純 top-k block routing |
| Causal 保持 | 三路內 causal | gating 加 causal mask |
| 與 Full Attn 切換 | 訓練即稀疏 | 可在訓練中切換 full/MoBA |
| 部署 | DeepSeek V3.x | Kimi K1.5 / K2 |

MoBA 的最大優勢是**可與 full attention 互換**,允許在 SFT 階段切回 full,擴大相容性。

---

## 5. DSA — DeepSeek Sparse Attention(V3.2, 2025/Q4)

NSA 之後 DeepSeek 進一步推出 DSA(V3.2 paper),把 selected branch 的選擇粒度做得更細(從 block 級降到 chunk 級)、把 compressed branch 改成**learned latent**,並引入 KV 動態退場機制。最終結果:**長 context 下 KV cache 量比 NSA 再降 50%**,V3.2 在 128K context 推論顯存從約 80GB 降到 40GB 級,對 H800 部署是決定性優化。

---

## 6. Ring Attention / Striped Attention — 分散式長 context 訓練

單卡裝不下百萬 token KV。**Ring Attention**(Liu et al. 2023)把序列切片分發到 N 張卡,query 留在本地,KV 沿 ring 拓樸**逐 hop 環繞**,每張卡只算一段 block;通訊與計算 overlap,理論上序列長度可隨卡數線性放大。**Striped Attention**(Brandon et al. 2023)在 Ring 上修正了 causal mask 不平衡造成的 GPU 閒置(後段 GPU 工作較少),透過交錯切片讓每張卡負載均勻,實測再提速 1.5×。

Gemini 1.5、Llama 4、Qwen 2.5 1M、DeepSeek V3 的長 context 訓練都基於 Ring/Striped 系列(或其變體 Context Parallelism in Megatron-LM、Tensor-Sequence Parallel in DeepSpeed-Ulysses)。

---

## 7. KV cache 壓縮 — H2O / SnapKV / StreamingLLM

訓練端可以用 NSA/MoBA,推論端對既有模型還可後置壓縮:

- **StreamingLLM**(Xiao et al. 2023):保留前 4 個 attention sink + 滑動窗,可無限長串流但會丟中段資訊。
- **H2O**(Zhang et al. 2023):依累積 attention score 動態淘汰「非 Heavy Hitter」KV,壓縮率 5-10× 而 perplexity 幾乎不變。
- **SnapKV**(Li et al. 2024):在 prefill 結束時一次性篩選每個 head 的關鍵 token,decode 階段固定 KV 大小,適合長 prompt + 短 output(摘要、RAG)場景。
- **2025 進展**:KIVI(2-bit KV 量化)、PyramidKV(層越深 KV 越少)、Quest(query-aware 重排)。

實務上 vLLM / SGLang 預設不啟用這些(會傷品質),但 long-document 推論服務常自定 SnapKV plugin。

---

## 8. PagedAttention(vLLM 部署層必備)

KV cache 像作業系統 virtual memory:把 KV 按 16 token 一頁分配,允許**非連續存儲、頁面共享**(同一 system prompt 在多請求間共用)。對比連續 KV,GPU 記憶體碎片從 60-80% 降到 <4%,**throughput 提升 2-4×**。vLLM、SGLang、TensorRT-LLM 都採用,**長 context 服務的事實標準**。配合 prefix caching,使「prompt cache 經濟學」(下節)成為可能。

---

## 9. 長 context vs RAG — 何時塞滿、何時切片

不是 context 越大越好。決策框架:

| 場景 | 推薦 |
|------|------|
| 文件 < 200K、單次查詢 | 直接塞滿 long context |
| 同份 KB 多次查、語料 > 1M token | RAG + prompt cache |
| 需要結構化檢索/引用 | RAG(可審計) |
| Agentic 多輪、上下文累積 | 長 context + cache |
| 高頻短任務 | 短 context + 蒸餾 |

**Prompt cache 經濟學**:Anthropic prompt cache TTL 5 分鐘,cache hit token 計費 0.1× / cache write 1.25×;DeepSeek、OpenAI 也提供類似機制。算式:若同份 prompt $N$ 次內被重用且 $N \geq 2$,cache 比 RAG 還便宜。但若 KB 巨大、查詢稀疏,RAG 仍勝。

**Lost in the Middle**(Liu et al. 2023)現象在 2026 旗艦上已明顯緩解,但仍存在;128K+ 時建議把重要資訊放開頭與結尾。

---

## 10. Effective Context 評估

宣稱的 max context ≠ 實際可用 context。三大基準:

- **RULER**(NVIDIA 2024):13 類任務含 multi-key NIAH、variable tracking、aggregation、QA。標準是「85% 準確率以上才算有效」。Llama 3.1 405B 宣稱 128K,RULER 實測有效約 64K;Gemini 1.5 Pro 1M 實測 128K-256K;Llama 4 Scout 10M 實測有效約 1-2M。
- **Needle-in-a-Haystack-Multi**(NIAH-Multi):同時藏多根針,測 multi-hop 檢索。
- **LongBench v2**(THU, 2024):中英雙語、含程式碼、學術論文、長對話,評估真實推理而非單純檢索。

選模型看 RULER > 看廠商宣稱。

---

## 11. 微調自家模型的長 context(YaRN 路線)

最務實的開源做法,以 Llama 3 8B 從 8K 擴到 64K 為例:

```python
from transformers import AutoModelForCausalLM, AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
cfg.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,                # 8K -> 64K
    "original_max_position_embeddings": 8192,
    "attention_factor": None,     # 用 YaRN 預設
    "beta_fast": 32, "beta_slow": 1,
}
cfg.max_position_embeddings = 65536
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", config=cfg, torch_dtype="bfloat16"
)
# 接著用 1B token 長文本 (PG19, BookSum, RedPajama-Books) 繼續預訓練 ~400 steps
```

繼續預訓練 400-1000 步(批次 4M token)就足夠收斂,單機 8×A100 約 1-2 天。**注意**:資料要真的長(不要短文本拼接的偽長),否則只是學會「忽略遠位置」。

---

## 12. 2026 旗艦長 context 對比

| 模型 | 宣稱 context | RULER 有效 | 位置編碼 | 注意力結構 | 備註 |
|------|------|------|------|------|------|
| **Llama 4 Scout** | 10M | ~1-2M | iRoPE(交錯 RoPE/NoPE) | Hybrid local+global | 256K 訓練、10M 外推 |
| **Gemini 2.5 Pro** | 2M | ~512K | RoPE + 內部稀疏 | Ring + 未公開稀疏 | 多模態長片 |
| **Grok 4 fast** | 2M | ~256K | 改良 RoPE | xAI 內部稀疏 | 速度優先 |
| **Claude Sonnet 4.x / Opus 4.x** | 1M(beta) | ~400K | RoPE 變體 | Anthropic 自研 | prompt cache 成熟 |
| **GPT-5 / 5.1** | 400K | ~256K | 未公開 | 未公開 | tool 使用最強 |
| **DeepSeek V3.2** | 128K | ~96K | RoPE | **DSA**(NSA 升級) | KV cache 最省 |
| **Kimi K2** | 128K-1M | ~128K | RoPE | **MoBA** | 開源 MoE |
| **Qwen 3 Max 1M** | 1M | ~200K | YaRN 變體 | dual chunk attn | 開源 |

選型建議:
- **超長單檔分析(電影/codebase)**:Llama 4 Scout、Gemini 2.5 Pro。
- **可靠 1M 應用 + 商用 SLA**:Claude Opus 4 + prompt cache。
- **開源自部署**:Qwen 3 / DeepSeek V3.2 / Kimi K2,依顯存挑 DSA vs MoBA。
- **成本敏感**:RAG + 短 context Haiku/Flash 仍是 90% 場景的最優解。

---

長 context 是 2024-2026 工程能力的試金石,但「長」永遠不是目的——讓對的資訊在對的地方被模型注意到,才是。
