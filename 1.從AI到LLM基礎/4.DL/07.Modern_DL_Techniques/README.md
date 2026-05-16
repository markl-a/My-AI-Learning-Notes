# 現代深度學習技術（2024-2025）

> 🔄 **最後更新：** 2025-01-18
> 🎯 **目標：** 掌握最新的深度學習技術與趨勢
> 🚀 **狀態：** 已完成主要內容，持續更新中
> ⭐ **技術涵蓋：** 16+ 種現代技術，包含 2024 年最新研究

---

## 📖 簡介

本目錄涵蓋 2024-2025 年深度學習領域的**最新技術和最佳實踐**，包括：

- ✅ **參數高效微調（PEFT）**：LoRA、DoRA、QLoRA、GaLore、Adapter
- ✅ **訓練優化**：混合精度、Flash Attention、梯度檢查點、梯度累積
- ✅ **分散式訓練**：DeepSpeed ZeRO、FSDP
- ✅ **模型架構**：Vision Transformer、Diffusion Models、MoE
- ✅ **高效推理**：vLLM、Speculative Decoding、Continuous Batching
- ✅ **實戰指南**：最佳實踐、故障排除、超參數調整

**適用對象：**
- 想要高效微調大語言模型的開發者
- 需要優化訓練/推理性能的工程師
- 希望掌握 2024-2025 最新技術的研究者
- 資源受限但想訓練大模型的實踐者

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
**狀態：完成** | **重要性：⭐⭐⭐⭐**

**核心概念：**
- 在預訓練層之間插入小型網路（Adapter）
- 只訓練 Adapter 參數（~3-5% 總參數）
- 相比 LoRA 參數更多但更靈活
- 支持多任務學習和模組化

**架構：**
```
Transformer Layer
├── Self-Attention
│   ├── LayerNorm
│   └── Adapter ← 可訓練 (Down-project → ReLU → Up-project)
├── Feed-Forward
│   ├── LayerNorm
│   └── Adapter ← 可訓練
└── Residual Connection
```

**實作範例：**
```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=64):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Bottleneck architecture
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        # Residual connection
        return x + residual

class TransformerWithAdapter(nn.Module):
    def __init__(self, base_model, adapter_dim=64):
        super().__init__()
        self.base_model = base_model

        # 凍結基礎模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 添加 Adapters
        self.adapters = nn.ModuleList([
            Adapter(768, adapter_dim)
            for _ in range(12)  # 12 層 Transformer
        ])

    def forward(self, x):
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x)
            x = self.adapters[i](x)  # 應用 Adapter
        return x
```

**使用 Hugging Face AdapterHub：**
```python
from transformers import AutoModelForSequenceClassification
from adapters import AutoAdapterModel, AdapterConfig

# 載入模型
model = AutoAdapterModel.from_pretrained("bert-base-uncased")

# 添加 Adapter
adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16)
model.add_adapter("sentiment", config=adapter_config)
model.train_adapter("sentiment")

# 訓練（只訓練 Adapter 參數）
# ... 訓練程式碼 ...

# 儲存 Adapter（只有幾 MB）
model.save_adapter("./my_adapter", "sentiment")

# 載入和切換 Adapter
model.load_adapter("./my_adapter")
model.set_active_adapters("sentiment")
```

**參數對比：**
| 方法 | 可訓練參數 | 記憶體 | 靈活性 |
|------|-----------|--------|--------|
| Full Fine-tuning | 100% | 高 | 高 |
| Adapter | 3-5% | 中 | 高 |
| LoRA | 0.1-1% | 低 | 中 |
| Prefix Tuning | 0.01-0.1% | 低 | 低 |

**優勢：**
- ✅ 模組化設計，易於切換任務
- ✅ 支持多任務學習
- ✅ 相對較少的參數仍保持高性能
- ✅ 廣泛的研究支持和變體

**應用場景：**
- 多任務學習（同時處理多個任務）
- 持續學習（避免災難性遺忘）
- 跨語言遷移學習

**推薦資源：**
- 論文：[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- 庫：[AdapterHub](https://adapterhub.ml/)

---

#### 3.1 **DoRA（Weight-Decomposed Low-Rank Adaptation）** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐** | **2024 年新技術**

**核心概念：**
- LoRA 的改進版本（2024 年提出）
- 將權重分解為大小（magnitude）和方向（direction）
- 性能超越 LoRA，參數量相同
- 更好地逼近全參數微調

**數學原理：**
```
傳統 LoRA: W' = W + BA

DoRA: W' = m · (W + BA) / ||W + BA||
其中：
- m: 可訓練的大小向量
- ||·||: 列向量的範數
```

**PyTorch 實作：**
```python
import torch
import torch.nn as nn

class DoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # 原始權重（凍結）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA 矩陣
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))

        # Magnitude 向量（DoRA 的關鍵）
        self.magnitude = nn.Parameter(torch.ones(out_features, 1))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # LoRA 更新
        lora_update = self.lora_B @ self.lora_A

        # 新權重
        new_weight = self.weight + lora_update

        # 方向歸一化
        weight_norm = torch.norm(new_weight, dim=1, keepdim=True)
        directional_weight = new_weight / (weight_norm + 1e-8)

        # 應用 magnitude
        final_weight = self.magnitude * directional_weight

        return nn.functional.linear(x, final_weight)

# 使用範例
layer = DoRALayer(768, 768, rank=8)
x = torch.randn(4, 768)
output = layer(x)
```

**與 LoRA 對比：**
| 指標 | LoRA | DoRA |
|------|------|------|
| 參數量 | r(d₁+d₂) | r(d₁+d₂)+d₂ |
| 性能 | 基準 | +2-5% |
| 訓練速度 | 快 | 稍慢 |
| 收斂性 | 好 | 更好 |

**實驗結果（LLaMA-7B）：**
- GSM8K: LoRA 42.8% → DoRA 45.6%
- MMLU: LoRA 43.5% → DoRA 46.2%
- 參數量幾乎相同

**推薦資源：**
- 論文：[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) (2024)

---

#### 3.2 **GaLore（Gradient Low-Rank Projection）** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐** | **2024 年新技術**

**核心概念：**
- 訓練全參數，但使用低秩梯度投影
- 記憶體效率接近 LoRA，但保持全參數訓練的性能
- 不需要修改模型架構
- 適用於從頭預訓練大模型

**與傳統方法的區別：**
```
傳統全參數訓練：
- 儲存完整梯度 (d × d)
- 記憶體需求高

LoRA：
- 只訓練低秩矩陣
- 限制模型表達能力

GaLore：
- 投影梯度到低秩空間
- 更新完整參數
- 兼顧性能和記憶體
```

**實作範例：**
```python
import torch
from torch.optim import Optimizer

class GaLoreOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, rank=128, update_proj_gap=200):
        defaults = dict(lr=lr, rank=rank, update_proj_gap=update_proj_gap)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['proj_matrix'] = None

                state['step'] += 1

                # 更新投影矩陣
                if state['step'] % group['update_proj_gap'] == 1:
                    # SVD 分解找到主要方向
                    U, S, V = torch.svd_lowrank(grad, q=group['rank'])
                    state['proj_matrix'] = U

                # 投影梯度
                if state['proj_matrix'] is not None:
                    proj_grad = state['proj_matrix'].T @ grad @ state['proj_matrix']
                    # 反投影回原空間
                    grad = state['proj_matrix'] @ proj_grad @ state['proj_matrix'].T

                # 更新參數
                p.data.add_(grad, alpha=-group['lr'])

# 使用
model = YourLargeModel()
optimizer = GaLoreOptimizer(model.parameters(), lr=1e-4, rank=128)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**記憶體節省：**
| 模型 | 全參數 | LoRA | GaLore |
|------|--------|------|--------|
| 7B | 120GB | 24GB | 28GB |
| 性能 | 100% | 95-98% | 99-100% |

**優勢：**
- ✅ 接近全參數訓練的性能
- ✅ 記憶體需求接近 LoRA
- ✅ 適用於預訓練（不只是微調）
- ✅ 無需修改模型架構

**應用場景：**
- 從頭預訓練 LLM（資源受限）
- 大規模模型微調
- 研究實驗

**推薦資源：**
- 論文：[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) (2024)

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
- **DeiT：** 資料高效的 ViT
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

### 🌐 分散式訓練技術

#### 11. **DeepSpeed ZeRO** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- Zero Redundancy Optimizer（零冗餘優化器）
- 跨 GPU 分割優化器狀態、梯度和參數
- 大幅減少每個 GPU 的記憶體需求
- 三個階段：ZeRO-1、ZeRO-2、ZeRO-3

**ZeRO 階段：**
```
ZeRO-1: 分割優化器狀態
- 記憶體減少：4x
- 通訊量：最小

ZeRO-2: 分割優化器狀態 + 梯度
- 記憶體減少：8x
- 通訊量：中等

ZeRO-3: 分割優化器狀態 + 梯度 + 模型參數
- 記憶體減少：N_gpu x (取決於 GPU 數量)
- 通訊量：最大
- 可在單卡上訓練 100B+ 模型
```

**PyTorch + DeepSpeed 實作：**
```python
# 安裝: pip install deepspeed

# ds_config.json - DeepSpeed 設定檔
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,                    // ZeRO-3
        "offload_optimizer": {
            "device": "cpu",           // 卸載優化器到 CPU
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",           // 卸載參數到 CPU
            "pin_memory": true
        },
        "overlap_comm": true,          // 重疊通訊和計算
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    }
}
```

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型和資料
model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# DeepSpeed 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# 訓練迴圈
for batch in dataloader:
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True)
    outputs = model_engine(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss

    # DeepSpeed 處理反向傳播和優化
    model_engine.backward(loss)
    model_engine.step()
```

**使用 Hugging Face Trainer：**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,
    deepspeed="ds_config.json",  # DeepSpeed 配置
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**記憶體對比（175B 參數模型）：**
| 方法 | GPU 記憶體/卡 | GPU 數量 |
|------|--------------|---------|
| 傳統 DDP | >80GB | 不可行 |
| ZeRO-1 | ~60GB | 32+ |
| ZeRO-2 | ~45GB | 16+ |
| ZeRO-3 | ~20GB | 8+ |
| ZeRO-3 + Offload | ~10GB | 4+ |

**推薦資源：**
- 論文：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- 文檔：[DeepSpeed](https://www.deepspeed.ai/)

---

#### 12. **FSDP（Fully Sharded Data Parallel）** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- PyTorch 原生的分散式訓練方案
- 類似 DeepSpeed ZeRO-3
- 分片模型參數、梯度和優化器狀態
- 與 PyTorch 生態系統深度整合

**PyTorch FSDP 實作：**
```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 混合精度配置
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 自動包裝策略（針對 Transformer）
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer},
)

# 初始化模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# FSDP 包裝
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 預取優化
    cpu_offload=CPUOffload(offload_params=True),  # CPU 卸載
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,
)

# 訓練
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

**使用 Hugging Face Trainer：**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    bf16=True,
    fsdp="full_shard auto_wrap",  # 啟用 FSDP
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_cpu_ram_efficient_loading": True,
    },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**FSDP vs DeepSpeed：**
| 特性 | FSDP | DeepSpeed ZeRO |
|------|------|----------------|
| 生態系統 | PyTorch 原生 | 第三方庫 |
| 易用性 | 簡單 | 需要配置 |
| 功能 | 核心功能 | 更多高級功能 |
| 性能 | 優秀 | 優秀 |
| 社群支持 | PyTorch 團隊 | Microsoft |

**推薦資源：**
- 文檔：[PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- 教程：[Getting Started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

#### 13. **Mixture of Experts (MoE)** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐**

**核心概念：**
- 稀疏激活的模型架構
- 多個專家網路，每次只激活部分專家
- 參數量大但計算量可控
- Switch Transformer、GShard、Mixtral 都使用 MoE

**架構：**
```
輸入 Token
↓
Router（路由網路）
↓
選擇 Top-K 專家（例如 K=2）
↓
專家 1   專家 2   專家 3 ... 專家 N
  ↓         ↓      (未激活)
加權組合輸出
↓
最終輸出
```

**簡化 MoE 實作：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 建立多個專家
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        # 路由網路
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, d_model = x.shape

        # 路由決策
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        # 選擇 Top-K 專家
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # 歸一化權重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 準備輸出
        output = torch.zeros_like(x)

        # 扁平化以便處理
        x_flat = x.view(-1, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)

        # 對每個 token 處理
        for i in range(self.top_k):
            expert_idx = top_k_indices_flat[:, i]
            expert_weight = top_k_weights_flat[:, i]

            # 對每個專家批次處理
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output_flat = output.view(-1, d_model)
                    output_flat[mask] += expert_weight[mask].unsqueeze(-1) * expert_output

        return output

# 使用範例
moe_layer = MoELayer(input_dim=768, hidden_dim=2048, num_experts=8, top_k=2)
x = torch.randn(2, 128, 768)  # (batch, seq_len, d_model)
output = moe_layer(x)
```

**使用現成的 MoE 模型（Mixtral）：**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mixtral 8x7B - 8 個專家，每次激活 2 個
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    device_map="auto",
    load_in_4bit=True,  # 使用 4-bit 量化節省記憶體
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# 推理
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

**MoE 的優勢與挑戰：**

**優勢：**
- ✅ 模型容量大（46B 參數）但計算量小（相當於 12B）
- ✅ 訓練和推理效率高
- ✅ 每個專家可以專精於不同領域

**挑戰：**
- ⚠️ 負載不平衡（某些專家使用過度）
- ⚠️ 通訊開銷（分散式訓練時）
- ⚠️ 訓練穩定性（需要輔助損失）

**負載平衡損失：**
```python
def load_balancing_loss(router_probs, expert_mask):
    """
    鼓勵均勻使用所有專家
    """
    # router_probs: (batch * seq_len, num_experts)
    # expert_mask: (batch * seq_len, num_experts)

    # 每個專家被選中的頻率
    expert_usage = expert_mask.float().mean(0)

    # 每個專家的平均路由概率
    avg_router_prob = router_probs.mean(0)

    # 負載平衡損失
    num_experts = router_probs.shape[-1]
    loss = num_experts * torch.sum(expert_usage * avg_router_prob)

    return loss
```

**實際應用：**
- **Mixtral 8x7B：** Mistral AI 的開源 MoE 模型
- **Switch Transformer：** Google 的 1.6T 參數 MoE
- **GPT-4（推測）：** 可能使用 MoE 架構

**推薦資源：**
- 論文：[Switch Transformers](https://arxiv.org/abs/2101.03961)
- 論文：[Mixtral of Experts](https://arxiv.org/abs/2401.04088) (2024)
- 模型：[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

---

### 🚀 高效推理技術

#### 14. **PagedAttention (vLLM)** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐** | **2023-2024 技術**

**核心概念：**
- 靈感來自虛擬記憶體分頁
- 動態分配 KV Cache 記憶體
- 減少記憶體浪費和碎片化
- 提升吞吐量 2-4x

**傳統 KV Cache 問題：**
```
傳統方法：
- 預分配固定大小記憶體
- 記憶體浪費（max_length - actual_length）
- 無法有效批次處理不同長度的序列

PagedAttention：
- 按需分配記憶體塊（pages）
- 共享 KV Cache（beam search, parallel sampling）
- 接近 100% GPU 記憶體利用率
```

**使用 vLLM：**
```python
# 安裝: pip install vllm

from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # GPU 數量
    max_model_len=4096,      # 最大序列長度
)

# 設置採樣參數
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

# 批次推理
prompts = [
    "Tell me about AI",
    "What is quantum computing?",
    "Explain photosynthesis",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}\nGenerated: {generated_text}\n")
```

**OpenAI 兼容 API 伺服器：**
```bash
# 啟動 vLLM 伺服器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000

# 使用 API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "San Francisco is",
        "max_tokens": 100,
        "temperature": 0
    }'
```

```python
# 使用 OpenAI 客戶端
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM 不需要真實 API key
)

response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="San Francisco is",
    max_tokens=100,
)
print(response.choices[0].text)
```

**性能提升：**
| 指標 | 傳統方法 | vLLM (PagedAttention) |
|------|---------|----------------------|
| 吞吐量 | 1x | 2-4x |
| GPU 記憶體利用率 | 20-40% | 80-90% |
| 延遲 | 基準 | 相似或更好 |
| 批次大小 | 小 | 大 |

**推薦資源：**
- 論文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023)
- 庫：[vLLM](https://github.com/vllm-project/vllm)
- 文檔：[vLLM Documentation](https://docs.vllm.ai/)

---

#### 15. **Speculative Decoding（推測解碼）** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐⭐** | **2023-2024 技術**

**核心概念：**
- 使用小模型快速生成候選 tokens
- 大模型並行驗證這些候選
- 加速推理 2-3x，輸出完全相同
- 無損加速（與原始模型輸出一致）

**工作原理：**
```
1. 小模型（draft model）快速生成 k 個 token 候選
   → [t1, t2, t3, t4, t5]

2. 大模型（target model）一次性驗證所有候選
   → 接受: [t1, t2, t3] ✓
   → 拒絕: [t4, t5] ✗

3. 從第一個被拒絕的位置繼續
   → 大模型生成正確的 t4

4. 重複步驟 1-3
```

**PyTorch 簡化實作：**
```python
import torch
import torch.nn.functional as F

def speculative_decoding(
    target_model,      # 大模型
    draft_model,       # 小模型（快速）
    input_ids,
    max_length=100,
    k=5,               # 每次生成的候選數量
    temperature=1.0,
):
    """
    推測解碼實作
    """
    generated = input_ids.clone()

    while generated.shape[1] < max_length:
        # 步驟 1: 小模型生成 k 個候選 token
        draft_tokens = []
        draft_probs = []
        current_ids = generated

        for _ in range(k):
            with torch.no_grad():
                draft_logits = draft_model(current_ids).logits[:, -1, :]
                draft_prob = F.softmax(draft_logits / temperature, dim=-1)
                draft_token = torch.multinomial(draft_prob, 1)

                draft_tokens.append(draft_token)
                draft_probs.append(draft_prob)
                current_ids = torch.cat([current_ids, draft_token], dim=1)

        # 步驟 2: 大模型並行驗證
        candidate_ids = torch.cat(draft_tokens, dim=1)
        verify_ids = torch.cat([generated, candidate_ids], dim=1)

        with torch.no_grad():
            target_logits = target_model(verify_ids).logits
            target_probs = F.softmax(
                target_logits[:, generated.shape[1]-1:-1, :] / temperature,
                dim=-1
            )

        # 步驟 3: 檢查哪些 token 被接受
        accepted_count = 0
        for i in range(k):
            draft_token = draft_tokens[i]
            target_prob = target_probs[:, i, :]

            # 計算接受概率
            accept_prob = target_prob[:, draft_token] / draft_probs[i][:, draft_token]
            accept_prob = torch.clamp(accept_prob, 0, 1)

            # 決定是否接受
            if torch.rand(1).item() < accept_prob.item():
                accepted_count += 1
            else:
                # 拒絕，從大模型採樣正確的 token
                adjusted_prob = F.relu(target_prob - draft_probs[i])
                adjusted_prob = adjusted_prob / adjusted_prob.sum()
                correct_token = torch.multinomial(adjusted_prob, 1)
                draft_tokens[i] = correct_token
                break

        # 添加接受的 tokens
        accepted_tokens = torch.cat(draft_tokens[:accepted_count+1], dim=1)
        generated = torch.cat([generated, accepted_tokens], dim=1)

    return generated
```

**使用現成庫（Medusa）：**
```python
# 安裝: pip install medusa-llm

from medusa import MedusaModel
from transformers import AutoTokenizer

# 載入模型
model = MedusaModel.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3"
)
tokenizer = AutoTokenizer.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3"
)

# 推理（自動使用 speculative decoding）
prompt = "Explain the theory of relativity"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 傳統生成
# outputs = model.generate(**inputs, max_length=200)

# Speculative decoding（2-3x 更快）
outputs = model.medusa_generate(
    **inputs,
    max_length=200,
    medusa_choices=[[0], [0, 0], [0, 1], [0, 0, 0]],  # 候選配置
)

print(tokenizer.decode(outputs[0]))
```

**使用 draft model 方法：**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 大模型（目標）
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
).to("cuda")

# 小模型（draft）- 同架構但更小
draft_model = AutoModelForCausalLM.from_pretrained(
    "JackFram/llama-68m"  # 小得多的 LLaMA 模型
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 使用 speculative_decoding 函數（上面實作）
prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

outputs = speculative_decoding(
    target_model,
    draft_model,
    input_ids,
    max_length=200,
    k=5
)

print(tokenizer.decode(outputs[0]))
```

**加速效果：**
| 模型大小 | 傳統解碼 | Speculative Decoding | 加速比 |
|---------|---------|---------------------|--------|
| 7B | 100 tokens/s | 200-250 tokens/s | 2-2.5x |
| 13B | 50 tokens/s | 120-150 tokens/s | 2.4-3x |
| 70B | 10 tokens/s | 25-30 tokens/s | 2.5-3x |

**優勢：**
- ✅ 無損加速（輸出完全相同）
- ✅ 不需要額外訓練
- ✅ 適用於各種生成任務
- ✅ 記憶體開銷小

**應用場景：**
- 長文字生成
- 實時對話系統
- 批次推理
- 資源受限環境

**推薦資源：**
- 論文：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2023)
- 論文：[Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774) (2024)
- 庫：[Medusa](https://github.com/FasterDecoding/Medusa)

---

#### 16. **Continuous Batching** 🆕
**狀態：完成** | **重要性：⭐⭐⭐⭐**

**核心概念：**
- 動態批次處理
- 完成的序列立即從批次中移除
- 新請求立即加入批次
- 提升吞吐量和 GPU 利用率

**傳統批次 vs Continuous Batching：**
```
傳統批次處理：
Batch 1: [seq1 (100 tokens), seq2 (200 tokens), seq3 (150 tokens)]
→ 必須等 seq2 完成（200 tokens）才能處理下一個批次
→ seq1, seq3 完成後 GPU 閒置

Continuous Batching：
Step 100: [seq1 ✓, seq2, seq3]
         → seq1 完成，移除
Step 101: [seq2, seq3, seq4 ⭐]  ← 新請求立即加入
         → 持續運行，無閒置時間
```

**使用 TGI（Text Generation Inference）：**
```bash
# 安裝和啟動 TGI
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --max-batch-prefill-tokens 2048 \
    --max-total-tokens 4096
```

```python
# 客戶端使用
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

# 多個並發請求會自動使用 continuous batching
responses = client.text_generation(
    "Tell me about AI",
    max_new_tokens=100,
    stream=False
)
```

**vLLM 也支持 Continuous Batching：**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

# 動態添加請求
prompts = ["First prompt", "Second prompt"]
outputs1 = llm.generate(prompts, sampling_params)

# 可以隨時添加更多請求，不需要等待
more_prompts = ["Third prompt", "Fourth prompt"]
outputs2 = llm.generate(more_prompts, sampling_params)
```

**性能提升：**
- 吞吐量提升：2-10x（取決於請求長度分佈）
- GPU 利用率：70-90%（傳統批次：30-50%）
- 平均延遲：降低 30-50%

**推薦資源：**
- 博客：[Continuous Batching in LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- 庫：[Text Generation Inference](https://github.com/huggingface/text-generation-inference)

---

## 📊 技術對比表

### 訓練技術對比

| 技術 | 記憶體 | 速度 | 準確率 | 難度 | 適用場景 |
|------|--------|------|--------|------|----------|
| **LoRA** | ⬇️⬇️⬇️ | ⬆️⬆️ | ⬆️ | 低 | LLM 微調 |
| **DoRA** 🆕 | ⬇️⬇️⬇️ | ⬆️⬆️ | ⬆️⬆️ | 低 | LLM 微調（性能更好） |
| **QLoRA** | ⬇️⬇️⬇️⬇️ | ⬆️ | ⬆️ | 中 | 大模型微調（低資源） |
| **GaLore** 🆕 | ⬇️⬇️⬇️ | ⬆️ | ⬆️⬆️ | 中 | 預訓練+微調 |
| **Adapter** | ⬇️⬇️ | ⬆️ | ⬆️ | 低 | 多任務學習 |
| **AMP** | ⬇️⬇️ | ⬆️⬆️ | ➡️ | 低 | 所有訓練 |
| **Flash Attn** | ⬇️⬇️⬇️ | ⬆️⬆️⬆️ | ➡️ | 低 | Transformer 訓練 |
| **Gradient Ckpt** | ⬇️⬇️⬇️ | ⬇️ | ➡️ | 低 | 大模型訓練 |
| **Quantization** | ⬇️⬇️⬇️ | ⬆️⬆️ | ⬇️ | 中 | 模型部署 |
| **DeepSpeed ZeRO** 🆕 | ⬇️⬇️⬇️⬇️ | ➡️ | ➡️ | 中 | 分散式訓練 |
| **FSDP** 🆕 | ⬇️⬇️⬇️⬇️ | ➡️ | ➡️ | 中 | 分散式訓練（PyTorch） |
| **MoE** 🆕 | ⬇️ | ⬆️⬆️ | ⬆️⬆️ | 高 | 超大規模模型 |

### 推理技術對比

| 技術 | 記憶體 | 吞吐量 | 延遲 | 難度 | 適用場景 |
|------|--------|--------|------|------|----------|
| **vLLM/PagedAttention** 🆕 | ⬇️⬇️⬇️ | ⬆️⬆️⬆️ | ➡️ | 低 | 生產環境推理 |
| **Speculative Decoding** 🆕 | ➡️ | ⬆️⬆️ | ⬇️⬇️ | 中 | 長文字生成 |
| **Continuous Batching** 🆕 | ➡️ | ⬆️⬆️⬆️ | ⬇️ | 低 | API 服務 |
| **4-bit/8-bit 量化** | ⬇️⬇️⬇️ | ⬆️⬆️ | ➡️ | 低 | 資源受限推理 |
| **Flash Attention** | ⬇️⬇️ | ⬆️⬆️ | ⬇️ | 低 | 長上下文推理 |

---

## 💡 最佳實踐與技巧

### 訓練最佳實踐

#### 1. **PEFT 方法選擇指南**
```
情境 → 推薦方法

消費級 GPU（8-24GB）+ 大模型（7B-70B）：
└─ QLoRA (4-bit) + rank=16-64

充足資源 + 追求最佳性能：
└─ DoRA > LoRA

多任務學習：
└─ Adapter 或 LoRA (多個適配器)

預訓練大模型（資源受限）：
└─ GaLore

需要最小參數量：
└─ LoRA (rank=4-8)
```

#### 2. **超參數設置建議**

**LoRA/DoRA：**
```python
# 小模型（<1B）
lora_config = LoraConfig(r=8, lora_alpha=16)

# 中型模型（1B-10B）
lora_config = LoraConfig(r=16, lora_alpha=32)

# 大型模型（>10B）
lora_config = LoraConfig(r=64, lora_alpha=128)

# 目標模組選擇
target_modules = [
    "q_proj", "v_proj",           # 基礎（最小參數）
    "q_proj", "k_proj", "v_proj", "o_proj",  # 標準
    "q_proj", "k_proj", "v_proj", "o_proj",  # 完整（最佳性能）
    "gate_proj", "up_proj", "down_proj"
]
```

**學習率建議：**
```python
# 全參數微調
lr = 1e-5 to 5e-5

# LoRA/DoRA
lr = 1e-4 to 5e-4  # 可以更高

# GaLore
lr = 1e-4 to 1e-3

# Adapter
lr = 1e-4 to 3e-4
```

#### 3. **記憶體優化組合技巧**

**極限記憶體優化（在 24GB GPU 上訓練 70B 模型）：**
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # QLoRA 4-bit 量化
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,

    # 梯度檢查點
    gradient_checkpointing=True,

    # 混合精度
    bf16=True,

    # 梯度累積（模擬大批次）
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,

    # 其他優化
    optim="paged_adamw_8bit",  # 8-bit 優化器
    max_grad_norm=0.3,
)
```

**組合效果：**
| 技術組合 | 記憶體需求 | 性能影響 |
|---------|-----------|---------|
| 基礎 | 280GB | 100% |
| + AMP | 140GB | ~0% |
| + Gradient Checkpoint | 70GB | -20% 速度 |
| + LoRA | 24GB | -0% 性能 |
| + 4-bit 量化 | 12GB | -2% 性能 |

#### 4. **分散式訓練決策樹**
```
GPU 數量 > 1？
├─ 否 → 使用單卡優化（QLoRA, Gradient Checkpoint）
└─ 是
    ├─ 同一節點（單機多卡）？
    │   ├─ 是 → FSDP 或 DeepSpeed ZeRO-2
    │   └─ 否（跨節點）→ DeepSpeed ZeRO-3
    │
    └─ 模型大小？
        ├─ <10B → DDP（最快）
        ├─ 10B-100B → FSDP/ZeRO-2
        └─ >100B → DeepSpeed ZeRO-3 + CPU Offload
```

### 推理最佳實踐

#### 1. **推理框架選擇**
```
場景 → 推薦方案

生產環境 API 服務：
└─ vLLM 或 TGI（Hugging Face）

單次批次推理：
└─ Transformers + Flash Attention 2

極致速度（可訓練小模型）：
└─ Speculative Decoding

長上下文（>8K tokens）：
└─ vLLM + PagedAttention

資源受限：
└─ llama.cpp（CPU）或 GPTQ/AWQ 量化
```

#### 2. **量化方法選擇**
```python
# GPTQ（適合 Nvidia GPU）
from transformers import GPTQConfig

quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

# AWQ（更快的推理）
from transformers import AwqConfig

quantization_config = AwqConfig(bits=4, fuse_max_seq_len=512)

# bitsandbytes（最簡單）
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

**性能對比：**
| 方法 | 速度 | 記憶體 | 設置難度 | 支持硬體 |
|------|------|--------|---------|---------|
| GPTQ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | Nvidia |
| AWQ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | Nvidia |
| bitsandbytes | ⭐⭐⭐ | ⭐⭐⭐⭐ | 低 | Nvidia |
| GGUF (llama.cpp) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 低 | CPU, Apple Silicon |

---

## 🔧 常見問題與故障排除

### 訓練問題

#### 1. **OOM（Out of Memory）錯誤**

**症狀：**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解決方案（按優先級）：**
```python
# 1. 降低批次大小
per_device_train_batch_size = 1

# 2. 啟用梯度檢查點
gradient_checkpointing = True

# 3. 使用梯度累積
gradient_accumulation_steps = 16  # 有效批次 = 1 * 16

# 4. 使用更激進的量化
load_in_4bit = True  # 從 8-bit 改為 4-bit

# 5. 降低序列長度
max_length = 512  # 從 2048 降低

# 6. 使用更小的 LoRA rank
r = 8  # 從 64 降低

# 7. 清理 GPU 快取
torch.cuda.empty_cache()
```

#### 2. **訓練速度過慢**

**檢查清單：**
```python
# ✅ 啟用混合精度
fp16 = True  # 或 bf16 = True

# ✅ 使用 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)

# ✅ 優化資料載入
dataloader_num_workers = 4
dataloader_pin_memory = True

# ✅ 編譯模型（PyTorch 2.0+）
model = torch.compile(model)

# ✅ 使用更快的優化器
optim = "adamw_torch_fused"  # 比標準 AdamW 快 20-30%
```

#### 3. **Loss 不下降或訓練不穩定**

**診斷步驟：**
```python
# 1. 檢查學習率
# LoRA 可以用更高的學習率
learning_rate = 1e-4  # 試試從 1e-5 提高

# 2. 添加 warmup
warmup_steps = 100  # 或 warmup_ratio = 0.1

# 3. 使用梯度裁剪
max_grad_norm = 1.0

# 4. 檢查資料品質
# 確保標籤正確，沒有 NaN

# 5. 降低 LoRA dropout
lora_dropout = 0.05  # 從 0.1 降低

# 6. 調整 LoRA alpha
lora_alpha = 32  # 嘗試 alpha = 2 * r
```

### 推理問題

#### 1. **推論速度慢**

**優化檢查清單：**
```python
# ✅ 使用專用推理框架
# vLLM, TGI, 而不是 Transformers

# ✅ 啟用 KV Cache
use_cache = True

# ✅ 使用更小的 batch size（單次推理）
batch_size = 1

# ✅ 量化模型
load_in_4bit = True

# ✅ Flash Attention
attn_implementation = "flash_attention_2"

# ✅ 靜態批次大小（避免動態形狀）
# 使用 padding 到固定長度
```

#### 2. **生成品質問題**

**調整生成參數：**
```python
generation_config = GenerationConfig(
    # 溫度控制隨機性
    temperature=0.7,  # <1: 更確定, >1: 更隨機

    # Top-p 採樣
    top_p=0.9,  # 只考慮累積概率前 90% 的 tokens

    # Top-k 採樣
    top_k=50,  # 只考慮概率最高的 50 個 tokens

    # 重複懲罰
    repetition_penalty=1.2,  # >1: 降低重複

    # 長度控制
    max_new_tokens=512,
    min_new_tokens=10,

    # 停止條件
    eos_token_id=tokenizer.eos_token_id,
)
```

#### 3. **記憶體碎片化**

```python
# 定期清理
import gc
gc.collect()
torch.cuda.empty_cache()

# 使用 vLLM（自動管理記憶體）
from vllm import LLM
llm = LLM(model="...")

# 限制 KV Cache 大小
model.generation_config.max_length = 2048
```

### 兼容性問題

#### 1. **Flash Attention 安裝失敗**

```bash
# 方法 1: 使用預編譯版本
pip install flash-attn --no-build-isolation

# 方法 2: 從源碼安裝（需要 CUDA 11.8+）
pip install packaging ninja
pip install flash-attn --no-build-isolation

# 方法 3: 使用 conda
conda install -c conda-forge flash-attention

# 如果仍失敗，使用 sdpa（PyTorch 原生）
attn_implementation = "sdpa"  # 而不是 "flash_attention_2"
```

#### 2. **DeepSpeed 配置錯誤**

```python
# 常見錯誤：設定檔格式
# ❌ 錯誤
"fp16": {"enabled": true}  # Python 布林值

# ✅ 正確（JSON 格式）
"fp16": {"enabled": true}  # JSON 布林值

# 檢查配置
import json
with open("ds_config.json") as f:
    config = json.load(f)  # 確保能正確載入
```

---

## 🎯 學習路徑建議

### 初學者（1-3 個月）
1. ✅ 掌握混合精度訓練（AMP）
2. ✅ 了解梯度累積和梯度檢查點
3. ✅ 實作 LoRA 微調
4. ✅ 學習使用 Hugging Face Transformers

### 中級（3-6 個月）
1. ✅ 深入 LoRA/QLoRA/DoRA
2. ✅ 實作 Vision Transformer
3. ✅ 探索各種模型量化方法
4. ✅ 學習 Flash Attention
5. ✅ 實作簡單的推論優化

### 進階（6-12 個月）
1. ✅ 掌握 DeepSpeed/FSDP 分散式訓練
2. ✅ 研究 MoE 架構
3. ✅ 實作 Diffusion Models
4. ✅ 深入推論優化（vLLM, Speculative Decoding）
5. ✅ 從頭預訓練小型模型

### 專家級（12+ 個月）
1. ✅ 研究最新論文並實作
2. ✅ 貢獻開源項目（PEFT, vLLM, DeepSpeed）
3. ✅ 優化生產環境部署
4. ✅ 開發新的高效訓練/推理方法
5. ✅ 發表研究論文

---

## 📚 推薦資源

### 📄 重要論文

#### PEFT 方法（2022-2024）
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023)
- **[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) (2024)** 🆕
- **[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) (2024)** 🆕
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) (2019) - Adapter

#### 分散式訓練（2019-2024）
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2019)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (2023)

#### 高效推理（2023-2024）
- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023)** 🆕
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2023)** 🆕
- **[Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774) (2024)** 🆕

#### MoE 架構（2021-2024）
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) (2021)
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) (2020)
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088) (2024)** 🆕

#### 視覺與多模態（2020-2023）
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) (2020)
- [Swin Transformer](https://arxiv.org/abs/2103.14030) (2021)
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) (2020)

#### 注意力優化（2022-2024）
- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (2022)
- [Flash Attention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (2023)

### 🛠️ 開源庫與框架

#### 訓練框架
- **[PEFT](https://github.com/huggingface/peft)** - Hugging Face 參數高效微調庫
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** - Microsoft 分散式訓練框架
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** - 8-bit/4-bit 量化庫
- **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** - 高效注意力實作

#### 推理框架 🆕
- **[vLLM](https://github.com/vllm-project/vllm)** - PagedAttention 推論引擎
- **[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)** - Hugging Face 推論服務
- **[Medusa](https://github.com/FasterDecoding/Medusa)** - Speculative Decoding
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - CPU 推理（支持量化）

#### 其他工具
- **[Diffusers](https://github.com/huggingface/diffusers)** - 擴散模型庫
- **[Transformers](https://github.com/huggingface/transformers)** - Hugging Face 核心庫
- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** - LLM 微調工具
- **[LitGPT](https://github.com/Lightning-AI/litgpt)** - Lightning AI 的 GPT 實作

### 📖 課程與教程

#### 官方課程
- **[Hugging Face Course](https://huggingface.co/course)** - 全面的 NLP 和 Transformers 課程
- **[Fast.ai Practical Deep Learning](https://course.fast.ai/)** - 實用深度學習
- **[Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/)** - Transformer 專題
- **[DeepSpeed Tutorials](https://www.deepspeed.ai/tutorials/)** - 分散式訓練教程

#### 實戰教程 🆕
- **[PEFT Examples](https://github.com/huggingface/peft/tree/main/examples)** - PEFT 實戰範例
- **[vLLM Documentation](https://docs.vllm.ai/)** - vLLM 完整文檔
- **[Unsloth](https://github.com/unslothai/unsloth)** - 超快速 LLM 微調（封裝 LoRA/QLoRA）

### 📝 部落格與文章

#### 技術深度解析
- **[Hugging Face Blog](https://huggingface.co/blog)** - 最新技術文章
- **[Towards Data Science](https://towardsdatascience.com/)** - 深度學習文章
- **[Sebastian Raschka's Blog](https://sebastianraschka.com/blog/)** - LLM 研究者的部落格
- **[Lightning AI Blog](https://lightning.ai/blog)** - 訓練優化技術

#### 實用指南 🆕
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [Understanding LoRA](https://lightning.ai/pages/community/tutorial/lora-llm/)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/)

### 🎥 影片資源

- **[Andrej Karpathy - Neural Networks](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)** - 神經網路基礎
- **[Yannic Kilcher](https://www.youtube.com/@YannicKilcher)** - 論文解讀
- **[AI Coffee Break](https://www.youtube.com/@AICoffeeBreak)** - AI 技術解說

---

## 🔗 相關章節

- **深度學習基礎：** `../00.DL_Path/`
- **PyTorch 實作：** `../03.Pytorch/`
- **Transformer 應用：** `../05.Transformer_lib/`
- **優化器詳解：** `../02.Optimizer/`
- **正則化技術：** `../01.Regularization/`

---

## 📝 總結

本文檔涵蓋了 **16+ 種現代深度學習技術**，從訓練到推理，從單卡到分散式，從理論到實踐。

### 🎯 核心要點

**訓練優化：**
- 使用 **LoRA/DoRA** 可以用 1% 參數達到 95%+ 性能
- **QLoRA** 讓 24GB GPU 也能訓練 70B 模型
- **GaLore** 是預訓練大模型的記憶體高效方案
- **Flash Attention** 是 Transformer 訓練的必備優化

**分散式訓練：**
- **FSDP** 是 PyTorch 原生方案，簡單易用
- **DeepSpeed ZeRO** 功能更強大，適合超大模型
- **MoE** 可以讓模型容量倍增但計算量不變

**高效推理：**
- **vLLM** 可將吞吐量提升 2-4x
- **Speculative Decoding** 實現 2-3x 無損加速
- **Continuous Batching** 大幅提升 API 服務效率

### 🚀 實踐建議

1. **從簡單開始**：先掌握 LoRA + AMP + Flash Attention
2. **根據資源選擇**：24GB GPU → QLoRA，多卡 → FSDP/DeepSpeed
3. **優先推論優化**：生產環境必用 vLLM 或 TGI
4. **持續學習**：關注 Hugging Face Blog 和最新論文
5. **動手實踐**：用開源庫快速驗證想法

### 🌟 2024-2025 重點技術

- **DoRA** (2024) - LoRA 的改進版，性能更好
- **GaLore** (2024) - 記憶體高效的預訓練方案
- **Mixtral 8x7B** (2024) - 開源 MoE 模型
- **Medusa** (2024) - 新一代 Speculative Decoding
- **vLLM** - 生產環境推理的事實標準

---

**掌握這些現代技術，讓你的深度學習能力躍升至業界前沿！** 🚀

**文檔將持續更新最新技術...** 📖

---

**貢獻者：** AI Learning Community
**最後驗證：** 2025-01-18
**版本：** v2.0 (大幅增強版)
