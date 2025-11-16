# 現代深度學習技術（2024-2025）

> 🔄 **最後更新：** 2025-01
> 🎯 **目標：** 掌握最新的深度學習技術與趨勢
> 🚀 **狀態：** 持續更新中

---

## 📖 簡介

本目錄涵蓋 2024-2025 年深度學習領域的最新技術和最佳實踐，包括高效訓練方法、參數高效微調、現代架構等。

---

## 📚 技術清單

### 🔥 參數高效微調（PEFT）

#### 1. **LoRA（Low-Rank Adaptation）**
**狀態：規劃中** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- 凍結預訓練模型權重
- 訓練低秩分解矩陣
- 大幅減少可訓練參數（99%+）
- 保持性能，降低記憶體需求

**數學原理：**
```
W' = W + ΔW = W + BA
其中：
- W: 原始權重（凍結）
- B: r × d 矩陣
- A: d × r 矩陣
- r << min(d, d)（秩遠小於維度）
```

**PyTorch 實作：**
```python
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

# 1. 載入基礎模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # 秩
    lora_alpha=32,            # 縮放因子
    lora_dropout=0.1,         # Dropout
    target_modules=["q_proj", "v_proj"],  # 目標層
    bias="none",              # 偏置策略
)

# 3. 應用 LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 輸出：trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

**優勢：**
- ✅ 記憶體需求大幅降低
- ✅ 訓練速度快
- ✅ 可以為不同任務訓練多個 LoRA 適配器
- ✅ 易於分享和部署

**應用場景：**
- LLM 微調（GPT, LLaMA, Mistral）
- Stable Diffusion 微調
- 多任務學習

**推薦資源：**
- 論文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- 庫：[PEFT (Hugging Face)](https://github.com/huggingface/peft)

---

#### 2. **QLoRA（Quantized LoRA）**
**狀態：規劃中** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- 結合 LoRA 和 4-bit 量化
- 在單張 24GB GPU 上微調 65B 參數模型
- NormalFloat (NF4) 量化方法

**實作範例：**
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 載入量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 準備模型進行 k-bit 訓練
model = prepare_model_for_kbit_training(model)

# 應用 LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

**記憶體對比：**
| 模型 | 全參數微調 | LoRA | QLoRA |
|------|-----------|------|-------|
| 7B  | ~120GB    | ~24GB | ~6GB  |
| 13B | ~240GB    | ~48GB | ~10GB |
| 65B | ~1TB+     | ~180GB | ~24GB |

**應用場景：**
- 消費級 GPU 訓練大模型
- 資源受限環境
- 實驗快速迭代

**推薦資源：**
- 論文：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- 庫：[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

#### 3. **Adapter 方法**
**狀態：規劃中**

**核心概念：**
- 在預訓練層之間插入小型網路（Adapter）
- 只訓練 Adapter 參數
- 相比 LoRA 參數更多但更靈活

**架構：**
```
Transformer Layer
├── Self-Attention
│   └── Adapter ← 可訓練
├── Feed-Forward
    └── Adapter ← 可訓練
```

---

### ⚡ 訓練優化技術

#### 4. **混合精度訓練（AMP）**
**狀態：規劃中** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- 使用 FP16 和 FP32 的混合
- 加速訓練（2-3x）
- 減少記憶體使用（~50%）

**PyTorch 實作：**
```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in epochs:
    for batch in dataloader:
        optimizer.zero_grad()

        # 混合精度前向傳播
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)

        # 縮放損失並反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**TensorFlow 實作：**
```python
from tensorflow.keras.mixed_precision import set_global_policy

# 設置全域混合精度策略
set_global_policy('mixed_float16')

model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練自動使用混合精度
model.fit(train_dataset, epochs=10)
```

**速度對比：**
| 模型 | FP32 | FP16 (AMP) | 加速比 |
|------|------|-----------|--------|
| ResNet-50 | 100s | 45s | 2.2x |
| BERT-Base | 200s | 80s | 2.5x |
| GPT-2 | 300s | 120s | 2.5x |

---

#### 5. **梯度累積（Gradient Accumulation）**
**狀態：規劃中**

**核心概念：**
- 模擬大批次訓練
- 在多個小批次上累積梯度
- 減少記憶體需求

**實作：**
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)

    # 歸一化損失
    loss = loss / accumulation_steps
    loss.backward()

    # 每 N 步更新一次
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

#### 6. **Flash Attention**
**狀態：規劃中** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- IO 優化的注意力機制
- 加速 2-4x
- 記憶體需求降低
- 精確（無近似）

**使用（PyTorch）：**
```python
from flash_attn import flash_attn_qkvpacked_func

# 標準注意力
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d)
attn = F.softmax(scores, dim=-1)
output = torch.matmul(attn, V)

# Flash Attention
from flash_attn.flash_attention import FlashAttention
flash_attn = FlashAttention()
output = flash_attn(Q, K, V)
```

**整合到 Transformers：**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",  # 使用 Flash Attention 2
    torch_dtype=torch.float16,
    device_map="auto"
)
```

**性能提升：**
- 訓練速度：2-4x 提升
- 記憶體：減少 10-20x
- 序列長度：支持更長序列（4k → 32k）

---

### 🎨 現代視覺架構

#### 7. **Vision Transformer (ViT)**
**狀態：規劃中** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- 將圖像分割為 patches
- 使用純 Transformer 架構
- 在大規模資料上超越 CNN

**架構：**
```
Image (224×224×3)
↓ 分割為 patches (16×16)
↓ Linear Projection (196 patches × 768 dim)
↓ Position Embedding
↓ Transformer Encoder × N
↓ Classification Head
↓ Output (類別)
```

**PyTorch 實作：**
```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# 載入預訓練模型
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 推理
image = Image.open('cat.jpg')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
```

**從頭實作 ViT：**
```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        # (B, embed_dim, num_patches_h, num_patches_w) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # 添加 Position Embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classification
        x = self.norm(x[:, 0])  # 取 CLS token
        x = self.head(x)

        return x

# 使用
model = ViT(num_classes=1000)
image = torch.randn(1, 3, 224, 224)
output = model(image)  # (1, 1000)
```

**變體：**
- **DeiT：** 數據高效的 ViT
- **Swin Transformer：** 階層式 ViT
- **BEiT：** 自監督預訓練 ViT

---

#### 8. **Diffusion Models（擴散模型）**
**狀態：規劃中** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- 逐步添加噪聲（前向過程）
- 學習去噪（反向過程）
- 生成高品質圖像

**DDPM 簡化實作：**
```python
import torch
import torch.nn as nn

class SimpleDiffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps

        # 定義噪聲調度
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        """前向過程：添加噪聲"""
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def sample(self, model, shape):
        """反向過程：生成圖像"""
        device = next(model.parameters()).device
        x = torch.randn(shape).to(device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # 預測噪聲
            predicted_noise = model(x, t_batch)

            # 去噪
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return x
```

**使用 Stable Diffusion：**
```python
from diffusers import StableDiffusionPipeline
import torch

# 載入模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 生成圖像
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("astronaut.png")
```

**應用：**
- 文生圖（Text-to-Image）
- 圖像修復
- 超解析度
- 圖像編輯

---

### 🛠️ 實用工具與技術

#### 9. **梯度檢查點（Gradient Checkpointing）**
**狀態：規劃中**

**核心概念：**
- 訓練時不儲存所有中間激活
- 需要時重新計算
- 用時間換空間

**PyTorch 實作：**
```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 10)

    def forward(self, x):
        # 使用梯度檢查點
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

**在 Transformers 中使用：**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    gradient_checkpointing=True  # 啟用梯度檢查點
)
```

---

#### 10. **模型量化（Quantization）**
**狀態：規劃中**

**類型：**
- **Post-Training Quantization (PTQ)：** 訓練後量化
- **Quantization-Aware Training (QAT)：** 量化感知訓練

**PyTorch Dynamic Quantization：**
```python
import torch
import torch.quantization

# 動態量化（推理時）
model_fp32 = MyModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},  # 量化的層類型
    dtype=torch.qint8
)

# 模型大小減少 4x，速度提升 2-3x
```

**TensorFlow Lite 量化：**
```python
import tensorflow as tf

# 轉換為 TFLite 並量化
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 儲存
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 📊 技術對比表

| 技術 | 記憶體 | 速度 | 準確率 | 難度 | 適用場景 |
|------|--------|------|--------|------|----------|
| **LoRA** | ⬇️⬇️⬇️ | ⬆️⬆️ | ⬆️ | 低 | LLM 微調 |
| **QLoRA** | ⬇️⬇️⬇️⬇️ | ⬆️ | ⬆️ | 中 | 大模型微調（低資源） |
| **AMP** | ⬇️⬇️ | ⬆️⬆️ | ➡️ | 低 | 所有訓練 |
| **Flash Attn** | ⬇️⬇️⬇️ | ⬆️⬆️⬆️ | ➡️ | 低 | Transformer 訓練 |
| **Gradient Ckpt** | ⬇️⬇️⬇️ | ⬇️ | ➡️ | 低 | 大模型訓練 |
| **Quantization** | ⬇️⬇️⬇️ | ⬆️⬆️ | ⬇️ | 中 | 模型部署 |

---

## 🎯 學習路徑建議

### 初學者（1-3 個月）
1. 掌握混合精度訓練（AMP）
2. 了解梯度累積
3. 實作簡單的遷移學習

### 中級（3-6 個月）
1. 深入 LoRA/QLoRA
2. 實作 Vision Transformer
3. 探索模型量化

### 進階（6+ 個月）
1. 研究 Flash Attention
2. 實作 Diffusion Models
3. 優化分散式訓練
4. 貢獻開源項目

---

## 📚 推薦資源

### 論文
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [DDPM](https://arxiv.org/abs/2006.11239)

### 開源庫
- [PEFT](https://github.com/huggingface/peft) - 參數高效微調
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 8-bit 優化器與量化
- [Diffusers](https://github.com/huggingface/diffusers) - 擴散模型
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### 課程與教程
- [Hugging Face Course](https://huggingface.co/course)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/)

---

## 🔗 相關章節

- **深度學習基礎：** `../00.DL_Path/`
- **PyTorch 實作：** `../03.Pytorch/`
- **Transformer 應用：** `../05.Transformer_lib/`

---

**掌握這些現代技術，讓你的深度學習能力躍升至業界前沿！** 🚀

**持續更新中...** 📖
