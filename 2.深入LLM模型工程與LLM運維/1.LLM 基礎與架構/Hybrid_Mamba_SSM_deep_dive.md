> 對應 [全景圖 #3](../../2024-2026_AI完整領域全景圖.md) §1;LLM 核心訓練綜述 [`./LLM_Core_Training_2024-2026.md`](./LLM_Core_Training_2024-2026.md)

# Hybrid Mamba + Attention 深度剖析:後 Transformer 時代的工程現實

當 2017 年的 Transformer 在 2026 年仍是 LLM 主導架構時,真正在生產線上「悄悄替換」掉純 Attention 的,並不是某個爆紅的新典範,而是 **Mamba/SSM 與 Attention 的混合架構**。本篇從控制理論談起,串接 Mamba、Mamba-2、Jamba、Zamba、Falcon Mamba 一路到 2025/11 IBM Granite 4.0,並把它和 Diffusion LLM (Mercury) 放在同一張地圖上。

---

## 1. State Space Models 是什麼:從控制理論到序列建模

State Space Model (SSM) 源自 60 年代控制理論。一個連續系統可寫成:

```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

`h(t)` 是隱藏狀態,`A, B, C, D` 是參數矩陣。離散化後變成一個帶記憶的線性 RNN,等價於把輸入 `x` 與核 `K`(由 `A, B, C` 推導)做一次卷積。和 RNN 不同的是:**只要 `A` 是結構化的(對角化、HiPPO 等),這個卷積核可以用 FFT 在 O(L log L) 內算出**,而推理時又退化成 O(1) 狀態更新。S4 (Gu, 2021) 是第一個把 SSM 拉進深度學習主流的工作,證明它能在 Long Range Arena 把 Transformer 打到地板上。

但 S4 有個致命弱點:`A, B, C` 與輸入無關,所以模型「不會忘記」也「不會挑重點」——對 PathX 這種純訊號任務很強,但語言任務上仍輸 Transformer。

## 2. Mamba (Albert Gu, 2023):Selective SSM + Parallel Scan

Mamba 的關鍵是 **Selective SSM**:讓 `B`、`C`、以及離散化步長 `Δ` 變成輸入相依(input-dependent),也就是說模型可以根據當下的 token 決定「要記住多少、要忘記什麼」。這破壞了原本可用 FFT 卷積的線性時不變性,作者改用 **hardware-aware parallel scan** 在 GPU 上保留 O(L) 訓練、O(1) 推理。

實務意涵:Mamba 在 1.4B / 2.8B 規模下與 Transformer 同等表現,但 **推理吞吐量高 5 倍**,記憶體不隨 context 長度線性成長(只需固定大小的狀態)。

## 3. Mamba-2:State Space Duality (SSD),矩陣形式

2024/5 Tri Dao 與 Albert Gu 的 Mamba-2 論文 *"Transformers are SSMs"* 揭示:**選擇性 SSM 與線性 Attention 是同一族「結構化半可分矩陣」(structured semiseparable matrices) 的不同分解方式**。這個 State Space Duality (SSD) 框架讓 Mamba-2:

- 核心層比 Mamba-1 **快 2-8 倍**(用 matmul 而不是 scan)。
- 狀態維度從 N=16 拉到 N=64 / 128,品質顯著提升。
- 可以用 Tensor Parallelism 等 Transformer 的工程技巧。

這一步是後續 Hybrid 模型能規模化的關鍵——沒有 SSD,工程團隊不會願意把 production 押在 SSM 上。

## 4. Pure Mamba 的失敗點

純 Mamba 在 2024 年很快被發現有三個結構性弱點:

1. **In-context learning (ICL) 弱**:Multi-query associative recall (MQAR)、Phonebook 等需要從上下文「定位並回讀」的任務,Mamba 顯著輸 Transformer——固定大小的隱狀態無法精確索引任意位置。
2. **COPY 任務不穩**:Jelassi et al. (2024) 證明常數大小的 Mamba 不能完美 copy,要 copy 必須讓狀態隨 sequence length 成長,優勢就消失了。
3. **長距離精確回溯失敗**:Needle-in-a-haystack 在 4–8 層 Mamba 之後準確率急速下降,因為資訊被壓縮進有限狀態,精確細節會被覆寫。

簡單講:**Mamba 擅長「壓縮」整段歷史的語意,但不擅長「指向」歷史中的某個 token。** 而 LLM 的 in-context learning 本質上就是後者。

## 5. Hybrid Mamba + Attention 為何贏

Hybrid 架構的直覺很乾脆:**讓 Mamba 做大部分壓縮工作,留少量 Attention 層做精確回溯**。Attention 提供 (a) 任意位置的精確查找、(b) ICL 所需的 token routing;Mamba 提供 (a) 線性記憶體、(b) 長序列吞吐量、(c) 推理時 O(1) 狀態。

實驗一致地顯示:**1:7 或 1:9 (Attention:Mamba) 的混合比例,品質追上純 Transformer,記憶體與吞吐量則接近純 Mamba**。這是個明顯的 Pareto 改善——不是 trade-off,是免費午餐。

## 6. 代表性 Hybrid 模型

### Jamba (AI21, 2024/3)
首個公開規模化的 Hybrid SSM-Transformer。52B 總參數 / 12B 啟動,**Mamba : Attention = 7:1**,加上 MoE FFN,單卡 80GB A100 可處理 **256K context**。Jamba 1.5 後續延伸到 398B 總參數。

### Zamba (Zyphra, 2024)
另一條設計路線:用一個 **shared attention block** 反覆插入 Mamba backbone,而不是交錯排列。7B 參數,32K context,沒有 MoE,主打小規模本機部署。Zamba-2 進一步把 shared block 切成兩個交替使用。

### IBM Granite 4.0 (2025/10–11)
企業端的代表作。架構亮點:
- **Mamba-2 : Transformer = 9:1**(比 Jamba 更激進)。
- **Fine-grained MoE**:H-Tiny 7B 總 / 1B 啟動,H-Small 32B 總 / 9B 啟動,含 **shared experts**(永遠啟動的專家)以穩定基礎能力。
- **無 positional encoding**:沒有 RoPE,理論上 context 無上限,實測到 128K。
- **效能宣稱**:同等品質下記憶體 **降低 70%+**、推理 **快 2 倍**。
- 已在 watsonx.ai、NVIDIA NIM、Ollama、LM Studio、Hugging Face、Replicate 等通路全面上線,定位是「給企業 RAG / agentic 工作流的成本最佳化基座」。

### Falcon Mamba 7B (TII, 2024)
故意維持 **純 Mamba**(無 Attention),驗證「attention-free 是否仍可競爭」。結論:常識與知識基準與同級 Transformer 接近,但推理密集任務輸一截——間接證明 Hybrid 路線才是工程正解。

## 7. 與 Transformer 的工程對比

| 維度 | Transformer (MHA) | Pure Mamba | Hybrid (9:1) |
|---|---|---|---|
| 訓練時複雜度 | O(L²·d) | O(L·d²) | 介於 |
| 推理 KV cache | 隨 L 線性成長 | O(1) 固定狀態 | 僅 1/10 層需 KV |
| 100K context 顯存 | 數十 GB | 數百 MB | 數 GB |
| In-context recall | 強 | 弱 | 強 |
| 長序列 throughput | 低 | 最高 | 接近 Mamba |
| 工程成熟度 | 最高 | 中 | 快速成熟中 |

對企業 inference 而言,**KV cache 從「主導成本」變成「邊緣成本」**,這是 Hybrid 真正動人的點。

## 8. 企業推理場景的具體優勢

企業 RAG / 客服 / Code agent 的典型 workload:**長 context (32K–200K) + 高 QPS + 對延遲敏感**。在這種設定下:

- 純 Transformer 的 KV cache 會撐爆 HBM,batch size 被迫降到個位數。
- Hybrid 模型把 90% 的 layer 換成 O(1) 狀態的 Mamba-2,**單張 H100 可同時服務的 concurrent session 提升一個數量級**。
- IBM 公開的數字是「長 context 高 batch 場景,單位 token 成本降 50–70%」——這是 CFO 會看的數字,不是 benchmark 分數。

## 9. 可執行範例:用 transformers 載入 Granite 4 Hybrid 跑長文件摘要

```python
# pip install -U transformers accelerate torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ibm-granite/granite-4.0-h-micro"   # 3B hybrid,單卡可跑

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

with open("long_report.txt", "r", encoding="utf-8") as f:
    long_doc = f.read()   # 假設 80K tokens 的內部報告

messages = [
    {"role": "system", "content": "你是專業摘要員,請輸出 5 點要點與行動建議。"},
    {"role": "user",   "content": f"以下為內部報告全文,請摘要:\n\n{long_doc}"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
    )

print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

關鍵差異:**整個 80K context 不會炸 KV cache**,因為大部分 layer 是 Mamba-2 的固定狀態。同樣的 prompt 在 Llama-3 8B 上會吃掉接近 40GB KV。

## 10. 訓練 Hybrid 模型的工程要點

- **比例選擇**:Jamba 用 7:1,Granite 4 用 9:1。經驗法則是「品質下限由 Attention 層位置決定,越靠近 input 與 output 越重要」。常見作法是把 Attention 層平均分散,或集中在後段做精細推理。
- **Attention 變體**:大多採 GQA + 不用 RoPE(讓 context 可外推),或保留 RoPE 但配 NoPE 混用。
- **初始化**:Mamba-2 的 `A` 用 HiPPO-style 對角矩陣初始化;`Δ` 的 bias 要小心,過大會讓模型「太敢忘」。
- **訓練資料 schedule**:Hybrid 對長文件、合成 recall 任務、code 都比純 Mamba 受用——這些資料能訓練 Attention 層學會「該查就查」。
- **MoE 結合**:Granite 4 證明 fine-grained MoE + shared experts 與 Hybrid 正交,可同時拿兩邊紅利。
- **蒸餾**:也有團隊從 Transformer 蒸出 Hybrid(Mamba-in-the-Llama 一類工作),省訓練成本。

## 11. 2026 趨勢:Hybrid 會超車 Pure Transformer 嗎?

**支持論點**:
- KV cache 經濟學壓力只會越來越大,context window 持續拉長到 1M+。
- IBM、AI21、Zyphra、Tencent (Hunyuan-T1)、NVIDIA (Nemotron-H) 都已下注 Hybrid;開源權重可用,生態成熟度逼近 Transformer。
- Mamba-2 SSD 框架讓 Hybrid 與 Transformer 共用大部分訓練 / 部署 infra,遷移成本低。

**反對論點**:
- 前沿能力(GPT-5, Claude Sonnet 4.5/4.7, Gemini 3)仍是純 Transformer,模型品質競賽尚未被 Hybrid 拿下。
- ICL、tool use、reasoning 的最強表現仍偏向 Attention-heavy 架構。
- 工程 ecosystem(FlashAttention、PagedAttention、speculative decoding)為 Transformer 高度優化,Hybrid 還在追平。
- 真正的瓶頸可能不是架構,而是 RL post-training 與 data quality——換骨架不一定動到天花板。

誠實的判斷:**Hybrid 會主導 7B–70B 的「成本敏感企業段」,但旗艦前沿模型短期內仍是 Transformer**。兩者並非取代,而是分層共存。

## 12. 並列地圖:三條 Post-Transformer 路線

當前(2026)真正在挑戰純 Transformer 主導的,可以歸成三條路:

1. **Hybrid SSM-Attention**(本篇主軸):Jamba / Granite 4 / Hunyuan-T1。重點是 **效率**,品質與 Transformer 持平。
2. **Diffusion LLM**:Inception Labs Mercury / Mercury 2、LLaDA 2。Transformer 為骨架但 **生成方式換成平行去噪**,Mercury 2 在 H100 上跑出 >1000 tok/s。重點是 **吞吐**,適合 code 與 latency-critical 應用。
3. **Linear Attention / RWKV / RetNet 系**:更極端的「沒有 KV cache」設計,目前仍在 7B 規模驗證,尚未進入企業主流。

三條路線共同信號:**「Attention is all you need」這句話在 2026 不再是字面真理**——你需要的可能是 attention + 一個更便宜的長距機制,或乾脆把 autoregressive 換掉。但 Transformer 不會死,它只是被「拆開重組」進更大的工具箱。

---

**延伸閱讀**:本系列的 [`Mixture_of_Experts_架構詳解.md`](./Mixture_of_Experts_架構詳解.md) 講 Granite 4 用到的 fine-grained MoE;[`注意力機制最新變體與優化.md`](./注意力機制最新變體與優化.md) 講 Hybrid 中那 10% Attention 該用哪種變體;[`LLM_Core_Training_2024-2026.md`](./LLM_Core_Training_2024-2026.md) 提供本篇所在的訓練流程全景。
