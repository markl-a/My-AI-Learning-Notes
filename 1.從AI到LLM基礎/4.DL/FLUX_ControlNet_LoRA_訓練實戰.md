# FLUX.1 + ControlNet + LoRA 訓練實戰

> 對應 [`./Multimodal_Generation_2024-2026.md`](./Multimodal_Generation_2024-2026.md) §3;C2PA 必裝 [`../../16.AI_Content_Authenticity/`](../../16.AI_Content_Authenticity/)

---

## 1. FLUX.1 為何成為 2026 開源影像生成首選

Black Forest Labs(SD 原班核心離開 Stability 後創立)於 2024/08 釋出 **FLUX.1**,2025-2026 已取代 SD 1.5/SDXL/SD3 成為**開源影像生成事實標準**。核心理由:

- **12B 參數**(SDXL 為 2.6B,SD3 Medium 為 2B)。
- **Hybrid transformer 架構**:**double-stream block**(text 與 image token 各自走 attention,再 cross attention 互通)+ **single-stream block**(concat 後共享 attention),取代純 UNet。
- **Rectified Flow Matching**:直接學速度場 `v_θ(x_t, t)`,訓練 loss `||v_θ - (x_1 - x_0)||²`,免 noise schedule,推理 ODE 路徑近線性。
- **文字渲染準確率 ~95%**(SDXL ~40%、SD3 Medium ~78%):靠 T5-XXL 文字編碼 + 大模型容量,海報/招牌/UI mockup 可生產。
- **prompt adherence** Elo 在 Artificial Analysis 連續多月與 Ideogram/Midjourney 同梯。
- 開源權重 + Apache 變體 + 完整 diffusers 整合,LoRA/ControlNet 生態 2025/Q1 後爆發。

## 2. FLUX 三版

| 版本 | 授權 | 步數 | 用途 |
|---|---|---|---|
| **FLUX.1-dev** | FLUX.1 Non-Commercial(僅研究與個人) | 25-50 | 主力本地推理 + LoRA fine-tune 對象 |
| **FLUX.1-schnell** | **Apache 2.0**(可商用) | 1-4 | 蒸餾版,即時預覽、移動端、商用 baseline |
| **FLUX.1.1 [pro]** | 僅 API(BFL/Replicate/fal) | 雲端 | 企業生產,2025/10 起 **預設輸出 C2PA 簽章** |

商用要求高品質又不能呼叫 API → 路徑是:用 dev 訓 LoRA → 把同 LoRA 載到 schnell 跑推理(rank 不變即可遷移,品質略降但合法)。

## 3. 環境準備

```bash
# CUDA 12.4 + PyTorch 2.5+
pip install "diffusers>=0.31" "transformers>=4.45" "peft>=0.13" \
            accelerate sentencepiece protobuf bitsandbytes safetensors \
            controlnet-aux        # canny/depth/pose 預處理器
pip install "ai-toolkit"           # ostris 的 FLUX LoRA 訓練 wrapper(2025 主流)
```

**VRAM 矩陣**(batch=1, 1024²):

| 精度 | 推理 | LoRA 訓練 |
|---|---|---|
| BF16 完整 | 24 GB | 48 GB(H100) |
| **FP8 (e4m3)** | 18 GB | 28 GB(A100) |
| **NF4 量化**(bnb) | **12 GB**(RTX 4080) | **16 GB**(RTX 4090) |
| GGUF Q4(社群) | 8 GB | — |

24 GB 卡(RTX 3090/4090)用 FP8 推理 + NF4 訓練是 2026 主流配方。

## 4. 基礎 inference

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()         # 24GB 卡必加
# pipe.enable_sequential_cpu_offload()  # 12GB 卡再退一階

prompt = (
    "A neon-lit Taipei alley at night, rain reflections, "
    "a vintage red phone booth with the text 'TAIPEI 2026' clearly visible, "
    "cinematic, 35mm, shallow depth of field"
)

image = pipe(
    prompt=prompt,
    height=1024, width=1024,
    guidance_scale=3.5,            # FLUX 慣用 3-4,別用 SD 的 7.5
    num_inference_steps=28,        # dev 建議 25-50
    max_sequence_length=512,       # T5 上限
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("flux_demo.png")
```

關鍵差異 vs SD:**FLUX 用 FlowMatchEulerDiscreteScheduler**(pipeline 預設已綁),不要手動換成 DDIM/DPM;CFG 過高會崩。

## 5. ControlNet 整合(結構控制)

2025 起,**XLabs-AI** 與 **InstantX/Shakker-Labs** 各自釋出 FLUX ControlNet 系列:`canny / depth / pose / openpose / tile / upscale / inpaint`。

```python
import torch, cv2, numpy as np
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector

# 1. 載 ControlNet(以 canny 為例)
controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    torch_dtype=torch.bfloat16,
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.enable_model_cpu_offload()

# 2. 結構圖預處理(可換 Midas/OpenPose)
src = Image.open("ref_pose.jpg").convert("RGB").resize((1024, 1024))
canny = CannyDetector()(src, low_threshold=80, high_threshold=200)
# depth = MidasDetector.from_pretrained("lllyasviel/Annotators")(src)
# pose  = OpenposeDetector.from_pretrained("lllyasviel/Annotators")(src)

# 3. 條件生成
out = pipe(
    prompt="cyberpunk samurai, holographic katana, rim light",
    control_image=canny,
    controlnet_conditioning_scale=0.6,   # 0.4-0.8 是甜蜜帶
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024, width=1024,
).images[0]
out.save("flux_canny.png")
```

**conditioning scale 經驗值**:canny 0.5-0.7、depth 0.4-0.6、pose 0.6-0.8;> 0.9 會「貼邊」(輸出像描邊機產出)。

## 6. IP-Adapter + InstantID(風格 / 人臉 ID)

- **IP-Adapter** = image prompt,把參考圖經 CLIP/SigLIP 編碼後注入 cross-attention,適合**風格/構圖遷移**。
- **InstantID** = face ID embedding + face landmark ControlNet,適合**人臉一致性**(同人不同場景)。FLUX 版由 InstantX 2025/Q2 釋出。

```python
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                    torch_dtype=torch.bfloat16).to("cuda")
pipe.load_ip_adapter("XLabs-AI/flux-ip-adapter",
                     weight_name="ip_adapter.safetensors")
pipe.set_ip_adapter_scale(0.7)         # 0.6-0.8

style_ref = Image.open("artist_style.jpg")
img = pipe(prompt="a wolf howling at the moon",
           ip_adapter_image=style_ref,
           guidance_scale=3.5, num_inference_steps=28).images[0]
```

InstantID 同理,只是改載 face encoder + identity ControlNet,適合「品牌代言人臉」「電商模特換衣」。

## 7. LoRA 訓練

### 資料準備(2025/2026 trigger word 規範)

- **30-50 張角色照**(或 15-25 張風格圖)、解析度 ≥ 1024²、多角度多光照。
- **trigger word**:2025 約定俗成用**罕見但可拼讀的 token**——避免 `sks`(太多人用、污染嚴重),改用 `ohwx`、`v3rity`、自定 `aliceXYZ`。caption 模板:`"a photo of ohwx person, smiling, cafe background"`。
- **caption** 用 BLIP-2 / **JoyCaption Alpha2** / Florence-2 自動產,再人工修;**不要**把 trigger word 之外的角色特徵寫進 caption(寫了 = LoRA 學不到那特徵)。

### 訓練 config(rank/lr/steps)

| 參數 | 建議 |
|---|---|
| LoRA rank | **16-32**(角色 16、風格 32) |
| LoRA alpha | = rank |
| target modules | `to_q, to_k, to_v, to_out, ff.net.0.proj, ff.net.2`(double + single stream 都要) |
| batch size | 1(grad accum 4) |
| learning rate | **1e-4**(AdamW8bit)/ **4e-4**(Prodigy 自適應) |
| steps | **500-2000**(30 張 → 1000、50 張 → 1500) |
| 解析度 | 1024²,bucket [512, 768, 1024, 1280] |
| timestep shift | 3.0(FM 推薦) |

工具:**ai-toolkit**(ostris)、**SimpleTuner**、**kohya-ss/sd-scripts**(`flux_train_network.py`)三者皆可,ai-toolkit 配置最簡。

### 推理載入 LoRA

```python
pipe.load_lora_weights("./output/alice_lora", weight_name="alice_lora.safetensors")
pipe.set_adapters(["default"], adapter_weights=[0.9])  # 0.7-1.0
img = pipe("ohwx person on Mars, spacesuit, golden hour",
           num_inference_steps=28, guidance_scale=3.5).images[0]
```

多 LoRA 疊加:`pipe.set_adapters(["character","style"], adapter_weights=[0.9,0.6])`,**總和勿超 1.5** 否則互相打架。

## 8. 完整 LoRA 訓練腳本(diffusers 0.31+)

```python
# train_flux_lora.py  —— 簡化版,生產建議用 ai-toolkit / diffusers 官方範例
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from diffusers import (FluxPipeline, FluxTransformer2DModel,
                       FlowMatchEulerDiscreteScheduler)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

MODEL_ID, OUT = "black-forest-labs/FLUX.1-dev", Path("./out_lora")
DATA, RES, BS, STEPS, LR = Path("./dataset"), 1024, 1, 1500, 1e-4

class CharSet(torch.utils.data.Dataset):
    def __init__(self, root, res):
        self.imgs = sorted([p for p in root.glob("*.jpg")])
        self.tf = T.Compose([T.Resize(res), T.CenterCrop(res),
                             T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = self.tf(Image.open(self.imgs[i]).convert("RGB"))
        cap = self.imgs[i].with_suffix(".txt").read_text().strip()
        return {"pixel": img, "caption": cap}

acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
vae, t5, clip = pipe.vae, pipe.text_encoder_2, pipe.text_encoder
tok_t5, tok_clip = pipe.tokenizer_2, pipe.tokenizer
transformer: FluxTransformer2DModel = pipe.transformer
sched: FlowMatchEulerDiscreteScheduler = pipe.scheduler

for m in (vae, t5, clip): m.requires_grad_(False)
transformer.requires_grad_(False)

lora_cfg = LoraConfig(r=16, lora_alpha=16,
    target_modules=["to_q","to_k","to_v","to_out.0",
                    "ff.net.0.proj","ff.net.2"],
    init_lora_weights="gaussian")
transformer = get_peft_model(transformer, lora_cfg)
transformer.print_trainable_parameters()

ds = CharSet(DATA, RES); dl = DataLoader(ds, batch_size=BS, shuffle=True)
opt = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                        lr=LR, weight_decay=1e-4)
lr_sch = get_scheduler("cosine", opt, num_warmup_steps=50, num_training_steps=STEPS)

transformer, opt, dl, lr_sch = acc.prepare(transformer, opt, dl, lr_sch)

def encode_prompt(captions):
    t5_in = tok_t5(captions, padding="max_length", max_length=512,
                   truncation=True, return_tensors="pt").input_ids.to(acc.device)
    clip_in = tok_clip(captions, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").input_ids.to(acc.device)
    return t5(t5_in)[0], clip(clip_in, output_hidden_states=False).pooler_output

step = 0; transformer.train()
while step < STEPS:
    for batch in dl:
        with acc.accumulate(transformer):
            pix = batch["pixel"].to(acc.device, dtype=torch.bfloat16)
            with torch.no_grad():
                lat = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
                pe, pp = encode_prompt(batch["caption"])
            noise = torch.randn_like(lat)
            t = torch.rand(lat.shape[0], device=acc.device)
            t_shift = (t * 3.0) / (1 + 2 * t)         # FM timestep shift
            t_exp = t_shift.view(-1,1,1,1)
            noisy = (1 - t_exp) * lat + t_exp * noise
            target = noise - lat                       # rectified-flow target
            pred = transformer(hidden_states=noisy,
                               timestep=t_shift * 1000,
                               encoder_hidden_states=pe,
                               pooled_projections=pp).sample
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            acc.backward(loss); opt.step(); lr_sch.step(); opt.zero_grad()
        if acc.sync_gradients:
            step += 1
            if step % 100 == 0: acc.print(f"step {step}  loss {loss.item():.4f}")
            if step >= STEPS: break

acc.wait_for_everyone()
transformer.save_pretrained(OUT)        # 產出 adapter_model.safetensors
acc.print(f"saved LoRA to {OUT}")
```

實務上**直接用 ai-toolkit**(一份 yaml 搞定),這份手寫版本用來理解 loss/timestep shift 細節。

## 9. 三件套組合 — 生產 pipeline 設計

把 **FLUX-dev + LoRA(角色)+ ControlNet(構圖)+ IP-Adapter(風格)** 串成可重複的生產線:

```
ref_pose.jpg ─► OpenPose ─┐
ref_style.jpg ─────────────┼─► FluxControlNetPipeline ─► 1024² output
prompt + trigger word ─────┤        + LoRA(character)
                           │        + IP-Adapter(style)
seed (固定批次內) ──────────┘
```

權重經驗值:`controlnet=0.6, lora=0.85, ip_adapter=0.5`。**順序很重要**——先 load LoRA 再 set IP-Adapter scale,否則 IP-Adapter 會吃掉 LoRA 的角色特徵。

## 10. 生產 case

- **電商產品換背景**:產品圖跑 SAM-3 摳出主體 → FLUX inpaint + ControlNet-depth 重建陰影,單卡 4090 ~4 秒/張,取代攝影棚拍背景。
- **品牌一致性角色生成**:角色 LoRA + InstantID(臉)+ 服裝 IP-Adapter,廣告主可以「同一虛擬代言人 × 10 國市場 × 5 場景」一次出 50 張,人工修圖 < 10%。
- **廣告 A/B 變體**:同 prompt 改 seed × 200 張 → CLIP-score 與 aesthetic predictor 排序 → top-20 跑 Meta/Google Ads 自動 A/B,CTR 提升通常 15-30%。

## 11. C2PA 整合(必做)

FLUX 1.1 [pro] API 從 2025/10 起預設嵌 C2PA;**自架推理(dev/schnell)必須自己簽**,否則無法上架到要求溯源的平台(Adobe Stock、部分 EU 廣告位)。

```python
# 安裝:pip install c2pa-python
import c2pa
manifest = {
  "claim_generator": "MyStudio/1.0 FLUX.1-dev",
  "title": "ad_hero_2026Q2.png",
  "assertions": [
    {"label":"c2pa.actions", "data":{"actions":[
        {"action":"c2pa.created", "softwareAgent":"FLUX.1-dev+LoRA"}]}},
    {"label":"com.example.ai_model", "data":{
        "model":"FLUX.1-dev", "lora":"alice_v3", "seed":42}}
  ]
}
c2pa.sign_file("flux_demo.png", "flux_demo_signed.png",
               manifest, signer_info=my_signer)
```

詳見 [`../../16.AI_Content_Authenticity/`](../../16.AI_Content_Authenticity/)。**生產流水線必須把 C2PA 簽章寫進 CI**,避免事後補簽漏稿。

## 12. vs SD 3.5 / Midjourney v7

| 維度 | FLUX.1-dev | SD 3.5 Large | Midjourney v7 |
|---|---|---|---|
| 開源權重 | Yes(non-commercial) | **Yes(完整可商用)** | No |
| LoRA / ControlNet 生態 | **2026 最活躍** | 成熟但 v3→v3.5 有斷層 | 無(只能靠 sref/cref) |
| 文字渲染 | **~95%** | ~85% | ~75% |
| 藝術氛圍 | 中上 | 中 | **最強** |
| Prompt 遵從 | **高** | 中高 | 中 |
| 商用授權 | dev 不行、schnell 可 | **完整可商用** | 訂閱即可 |
| 本地最低 VRAM | 12GB(NF4) | 12GB | 雲端 |

**選擇法則**:要 fine-tune + 寫字 + 開源 → FLUX;要完全商用且本地訓練 → SD 3.5;要直接出藝術品 → Midjourney v7。2026 多數工作室是**FLUX 主力 + MJ 補氛圍 + SD 3.5 商用退路**三軌並行。
