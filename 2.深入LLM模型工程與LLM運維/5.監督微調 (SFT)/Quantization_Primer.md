> 📌 此檔為 SFT 章節的「量化前置補課」。完整壓縮技術請見 [`../7.模型壓縮與優化/`](../7.模型壓縮與優化/)。

# 量化前置補課 (Quantization Primer for SFT)

在進入 QLoRA / NF4 之前,你必須先理解「量化 (Quantization)」到底在幹嘛。QLoRA 之所以能讓你在一張 24GB 的 RTX 3090 上微調 70B 模型,核心魔法就是 **4-bit 量化**。本章把所有你會在 SFT 過程中遇到的量化術語一次講清楚。

---

## 1. 為何要量化?

兩個現實問題:

| 問題 | 數字 |
|------|------|
| **模型太大** | Llama-3-70B 在 FP16 下 = 140 GB,單卡放不下 |
| **VRAM 太貴** | A100 80GB 一張要 NT$ 50 萬+;消費卡頂多 24GB |
| **推論太慢** | 記憶體頻寬是瓶頸,參數越多 → token/s 越低 |
| **能耗成本** | INT4 比 FP16 省約 4 倍記憶體頻寬 → 推論延遲降 2-3 倍 |

**量化的本質**:用更少的位元 (bit) 表示同一個權重,代價是精度損失。SFT 訓練時用 4-bit 載入凍結權重 + LoRA adapter 學 BF16,就是 QLoRA 的核心套路。

---

## 2. 數值型別階梯

從高到低、從占記憶體多到少:

```
FP32  ████████████████████████████████  32 bit  (1 sign + 8 exp + 23 mantissa)
BF16  ████████████████                  16 bit  (1 sign + 8 exp + 7  mantissa)  ← 動態範圍同 FP32
FP16  ████████████████                  16 bit  (1 sign + 5 exp + 10 mantissa)  ← 精度高但易溢位
FP8   ████████                           8 bit  (E4M3 / E5M2 兩種變體, H100 原生支援)
INT8  ████████                           8 bit  (整數 -128 ~ 127)
NF4   ████                               4 bit  (NormalFloat4, 為常態分佈最佳化)
INT4  ████                               4 bit  (整數 -8 ~ 7)
FP4   ████                               4 bit  (E2M1, 極端壓縮用)
INT2  ██                                 2 bit  (實驗性,品質崩壞風險高)
```

**重點觀念**:
- **BF16 vs FP16**:BF16 動態範圍跟 FP32 一樣大,訓練時不易 overflow/NaN,**Ampere 後的卡 (A100 / RTX 30/40 系列) 預設都用 BF16**。
- **FP8**:H100 才有原生硬體支援,Ada 也能模擬。
- **NF4**:Tim Dettmers 提出,假設權重服從常態分佈,把 16 個量化點放在分佈密度高的地方,比 INT4 表現好。

---

## 3. Weight-only vs Activation 量化

這是最容易混淆的兩個概念:

| 類別 | 量化對象 | 代表方法 | 適用場景 |
|------|---------|---------|---------|
| **Weight-only** | 只量化權重,activation 維持 FP16 | GPTQ, AWQ, NF4, GGUF | 大多數推論 / QLoRA 微調 |
| **Weight + Activation** | 連 activation 一起量化 | SmoothQuant, INT8 (LLM.int8()), FP8 | 極致推論優化、TensorRT 部署 |

**為什麼大部分用 Weight-only?**
因為 LLM 的 activation 有「離群值 (outlier)」問題 — 少數通道的數值會比其他通道大幾十倍,直接量化會嚴重失真。SmoothQuant 透過數學轉換把離群值搬到權重端,才讓 W8A8 變可行。

**SFT 階段你只會碰 Weight-only**,所以 QLoRA / GPTQ / AWQ 是必修。

---

## 4. GPTQ vs AWQ 對比

兩個目前最主流的 **訓練後量化 (PTQ, Post-Training Quantization)** 方法:

| 維度 | **GPTQ** | **AWQ** |
|------|----------|---------|
| 全名 | Generative Post-Training Quantization | Activation-aware Weight Quantization |
| 原理 | 用 Hessian 矩陣逐層最小化量化誤差 | 觀察 activation 的離群通道,保留前 1% 通道為 FP16 |
| 校準資料 | 需要 (約 128 條 sample) | 需要 (約 32-128 條) |
| 推論速度 | 中等 | **快**(沒有逆向反量化開銷) |
| 量化品質 | 4-bit 還可接受 | **4-bit 普遍更好** |
| 何時用 | 想用 ExLlamaV2 / AutoGPTQ kernel | vLLM / TGI 部署、追求極致吞吐 |

**白話總結**:
- 要部署到 vLLM 推論 → **AWQ**
- 要在 text-generation-webui 跑 → 兩個都可
- 要做 SFT → 都不用,**直接用 NF4 + bitsandbytes**(下節說明)

---

## 5. NF4 + 雙量化 (QLoRA 的核心)

QLoRA 論文 (Dettmers et al., 2023) 提出三個關鍵技巧:

### 5.1 NormalFloat 4-bit (NF4)
- 假設 LLM 權重經過正規化後服從 N(0, 1)
- 用「分位數量化」把 16 個 NF4 點放在常態分佈的等密度區
- 對常態分佈權重的 **資訊理論最優** 4-bit 表示

### 5.2 Double Quantization (雙量化)
- 第一次量化:權重 → NF4
- 量化會產生一個 FP32 的 scale (每 64 個權重一個 scale)
- **第二次量化**:把這些 scale 再量化成 FP8
- 平均每個參數省 **0.37 bit**(70B 模型 → 省約 3GB VRAM)

### 5.3 Paged Optimizer
- 用 NVIDIA Unified Memory 把 optimizer state 在 GPU/CPU 間自動換頁
- 避免梯度 spike 時 OOM

---

## 6. GGUF + K-quants (llama.cpp / Ollama)

GGUF 是 llama.cpp 的格式,專為 **CPU + GPU 混合推論** 設計,Ollama / LM Studio / Jan 都用這個。

**K-quants 命名規則**:`Q{bits}_{variant}`

| 名稱 | 平均 bits/weight | 品質 | 大小 (7B 模型) |
|------|-----------------|------|---------------|
| Q2_K | 2.6 | 差(僅能跑) | 2.7 GB |
| Q3_K_M | 3.4 | 普通 | 3.3 GB |
| **Q4_K_M** | 4.5 | **甜蜜點** | 4.1 GB |
| Q5_K_M | 5.5 | 接近 FP16 | 4.8 GB |
| Q6_K | 6.5 | 幾乎無損 | 5.5 GB |
| Q8_0 | 8.0 | 無損 | 7.2 GB |

**經驗法則**:消費級顯卡推論優先選 **Q4_K_M** 或 **Q5_K_M**。

---

## 7. bitsandbytes 在 SFT 中的角色

`bitsandbytes` 是 Tim Dettmers 開發的函式庫,被 HuggingFace `transformers` 深度整合。它在 SFT 中提供:

- `load_in_8bit=True` — LLM.int8() 量化載入
- `load_in_4bit=True` — NF4 / FP4 量化載入(QLoRA 必備)
- `bnb_4bit_use_double_quant=True` — 啟用雙量化
- `bnb_4bit_compute_dtype=torch.bfloat16` — 反量化後用 BF16 算 forward/backward
- `PagedAdamW8bit` — 分頁 8-bit Adam optimizer

### 完整載入範例 (transformers + bitsandbytes + NF4)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. 設定 4-bit 量化配置 (QLoRA 標準)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 啟用 4-bit 載入
    bnb_4bit_quant_type="nf4",               # 用 NF4 而非 FP4
    bnb_4bit_use_double_quant=True,          # 啟用雙量化 (再省 ~0.4 GB / 7B)
    bnb_4bit_compute_dtype=torch.bfloat16,   # 計算時反量化回 BF16
)

# 2. 載入模型 (自動下載並量化)
model_id = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",                       # 自動分散到可用 GPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. 確認量化後記憶體佔用
print(f"Model footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
# Llama-3-8B in NF4 ≈ 5.4 GB (vs FP16 16 GB)

# 4. 接著就能套 PEFT LoRA 進行 SFT
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)  # 凍結量化權重、開 gradient ckpt
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~21M / total: ~8B  →  僅訓練 0.26% 的參數
```

---

## 8. 記憶體計算公式

**推論記憶體**(僅權重):

```
VRAM (GB) ≈ 參數量 × bytes_per_param / 1e9
```

| 模型大小 | FP32 (4 B) | FP16/BF16 (2 B) | INT8 (1 B) | INT4/NF4 (0.5 B) |
|---------|-----------|----------------|-----------|------------------|
| 7B | 28 GB | 14 GB | 7 GB | **3.5 GB** |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 34B | 136 GB | 68 GB | 34 GB | 17 GB |
| 70B | 280 GB | 140 GB | 70 GB | **35 GB** |

**SFT 訓練還要加**:
- Activation:~2× 權重大小(用 gradient checkpointing 可降到 ~0.5×)
- Optimizer state:AdamW 8-bit ≈ 2 bytes/param;AdamW FP32 ≈ 8 bytes/param
- LoRA gradient:trainable params × 4 bytes

**經驗值**:7B + QLoRA + batch=1 + seq=2048 → 約 **10-12 GB**,RTX 3060 12GB 剛好夠。

---

## 9. 何時量化會傷害品質?

不是所有任務都適合 4-bit。已知量化會明顯掉分的場景:

| 場景 | 原因 | 建議 |
|------|------|------|
| **複雜推理 (MATH, GSM8K)** | 中間 logit 差異被量化抹平 | 至少用 INT8,優先 BF16 |
| **長 context (>32k)** | KV cache 累積誤差放大 | 用 INT8 KV cache 而非 INT4 |
| **Code generation** | 對 token 機率排序敏感 | AWQ INT4 還行,GPTQ INT4 較差 |
| **多語言 / 小語種** | tail token embedding 被壓爛 | 保留 embedding layer 為 FP16 |
| **小模型 (<3B)** | 本身容量小,量化邊際成本高 | 1B 模型直接用 BF16 |
| **RLHF / DPO 訓練** | reward 訊號弱,量化噪音蓋過 | reference model 用 BF16 |

**檢驗準則**:量化後跑 MMLU / HumanEval,**掉分 <1% 算可接受,>3% 就有問題**。

---

## 學習路徑建議

1. 先完成本章 → 看懂上面所有術語
2. 跳到 `進階微調策略_LoRA_QLoRA.md` → 學 QLoRA 完整流程
3. 想深入壓縮技術 → 去 `../7.模型壓縮與優化/` 學 GPTQ/AWQ/SmoothQuant/Distillation
4. 想學部署 → 去 `../8.LLM 推論服務化/` 學 vLLM + AWQ 整合

> **一句話總結**:SFT 時你只需要記住 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)` 這四行,其他細節以後遇到部署需求再回來補。
