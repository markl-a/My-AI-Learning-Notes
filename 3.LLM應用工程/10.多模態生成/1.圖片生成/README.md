# 圖片生成 (Image Generation)

本章節將深入探討AI圖片生成的核心技術，從基礎的Diffusion模型到進階的控制技術。

## 📋 目錄

1. [Stable Diffusion 基礎](#stable-diffusion-基礎)
2. [ControlNet 控制技術](#controlnet-控制技術)
3. [LoRA 訓練與應用](#lora-訓練與應用)
4. [進階技巧](#進階技巧)
5. [實戰案例](#實戰案例)

---

## 🎯 學習目標

完成本章節後，你將能夠：

- ✅ 理解Diffusion模型的工作原理
- ✅ 使用Stable Diffusion生成高品質圖片
- ✅ 運用ControlNet精確控制生成內容
- ✅ 訓練和使用自定義LoRA模型
- ✅ 掌握提示詞工程技巧
- ✅ 構建實用的圖片生成應用

---

## 📚 Stable Diffusion 基礎

### 什麼是Stable Diffusion？

Stable Diffusion是一個基於Latent Diffusion Model (LDM)的文生圖模型，由Stability AI開發。它通過在潛在空間中進行擴散過程，大幅降低了計算成本。

### 核心組件

```
┌─────────────────────────────────────────┐
│        Stable Diffusion 架構             │
├─────────────────────────────────────────┤
│                                         │
│  Text        ┌──────────────┐           │
│  Prompt  →   │ Text Encoder │           │
│              │   (CLIP)     │           │
│              └──────┬───────┘           │
│                     │                   │
│                     ↓                   │
│              ┌──────────────┐           │
│              │    U-Net     │           │
│  Noise   →   │  (Denoiser)  │  →  潛在表示 │
│              └──────────────┘           │
│                     │                   │
│                     ↓                   │
│              ┌──────────────┐           │
│              │  VAE Decoder │           │
│              └──────┬───────┘           │
│                     │                   │
│                     ↓                   │
│                 生成圖片                 │
│                                         │
└─────────────────────────────────────────┘
```

### 基本使用

#### 安裝依賴

```bash
pip install diffusers transformers accelerate torch torchvision xformers
```

#### 基礎圖片生成

```python
from diffusers import StableDiffusionPipeline
import torch

# 加載模型
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# 啟用記憶體優化
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# 生成圖片
prompt = """
a professional portrait photo of a young woman,
natural lighting, bokeh background,
highly detailed, 8k, photorealistic
"""

negative_prompt = """
cartoon, 3d, disfigured, bad art, deformed,
poorly drawn, extra limbs, close up, weird colors
"""

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    width=512,
    height=512,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("generated_portrait.png")
```

### 重要參數說明

| 參數 | 說明 | 推薦值 |
|------|------|--------|
| `num_inference_steps` | 推理步數，步數越多質量越好但速度越慢 | 20-50 |
| `guidance_scale` | CFG scale，控制提示詞遵循程度 | 7-12 |
| `width`, `height` | 圖片尺寸，必須是64的倍數 | 512, 768 |
| `generator` | 隨機數生成器，設定seed可重現結果 | 固定seed |

---

## 🎮 ControlNet 控制技術

### 什麼是ControlNet？

ControlNet是一種條件控制技術，允許你使用邊緣檢測、深度圖、姿態等資訊精確控制生成內容。

### 支持的控制類型

- **Canny Edge** - 邊緣檢測
- **Depth** - 深度圖
- **OpenPose** - 人體姿態
- **Scribble** - 塗鴉線稿
- **Segmentation** - 語義分割
- **Normal Map** - 法線貼圖
- **MLSD** - 直線檢測（建築物）

### 安裝與使用

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image

# 1. 準備控制圖（以Canny邊緣為例）
def get_canny_edge(image_path, low_threshold=100, high_threshold=200):
    """從圖片提取Canny邊緣"""
    image = load_image(image_path)
    image = np.array(image)

    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 應用Canny邊緣檢測
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)

    return Image.fromarray(edges)

# 2. 加載ControlNet模型
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# 3. 建立Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

# 4. 生成圖片
control_image = get_canny_edge("input_image.jpg")
control_image.save("control_canny.png")

prompt = "a luxury sports car, highly detailed, professional photo"
negative_prompt = "low quality, blurry, distorted"

generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_image,
    num_inference_steps=30,
    controlnet_conditioning_scale=1.0,  # 控制強度
).images[0]

generated_image.save("controlnet_output.png")
```

### 多重ControlNet組合

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# 加載多個ControlNet
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

# 建立多控制Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_depth],
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# 準備控制圖
canny_image = get_canny_edge("input.jpg")
depth_image = get_depth_map("input.jpg")  # 需要實現深度估計

# 生成
image = pipe(
    prompt="a beautiful landscape",
    image=[canny_image, depth_image],
    controlnet_conditioning_scale=[0.5, 0.8],  # 各自的控制強度
    num_inference_steps=25
).images[0]

image.save("multi_controlnet_output.png")
```

---

## 🔧 LoRA 訓練與應用

### 什麼是LoRA？

LoRA (Low-Rank Adaptation) 是一種高效的模型微調技術，只需訓練少量參數即可定制化模型風格或特定主題。

### LoRA 優勢

- ✅ **參數少** - 通常只有幾MB到幾十MB
- ✅ **訓練快** - 相比完整微調快10-100倍
- ✅ **可組合** - 可以疊加多個LoRA
- ✅ **易分享** - 文件小，方便傳播

### 使用現有LoRA

```python
from diffusers import StableDiffusionPipeline
import torch

# 加載基礎模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# 加載LoRA權重
pipe.load_lora_weights("path/to/lora_weights.safetensors")

# 設定LoRA強度（0-1）
pipe.fuse_lora(lora_scale=0.8)

pipe = pipe.to("cuda")

# 生成圖片
prompt = "a cute cat in anime style"  # LoRA會影響風格
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("lora_output.png")
```

### 訓練自定義LoRA

#### 準備訓練資料

```bash
# 資料集結構
dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── metadata.csv  # 包含圖片路徑和對應的caption
```

#### 訓練腳本

```python
# train_lora.py
from diffusers import StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

class ImageCaptionDataset(Dataset):
    """圖片-文字對資料集"""
    def __init__(self, metadata_file, image_dir, tokenizer, image_processor):
        self.df = pd.read_csv(metadata_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加載圖片
        image_path = f"{self.image_dir}/{row['image']}"
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(image)

        # 編碼文字
        caption = row['caption']
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0]
        }

def train_lora(
    model_id="runwayml/stable-diffusion-v1-5",
    dataset_path="dataset/metadata.csv",
    image_dir="dataset/images",
    output_dir="lora_output",
    rank=4,  # LoRA rank
    learning_rate=1e-4,
    num_epochs=100,
    batch_size=4
):
    """訓練LoRA模型"""

    # 1. 加載預訓練模型
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # 2. 注入LoRA層
    unet = pipe.unet
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank
        )

    unet.set_attn_processor(lora_attn_procs)

    # 3. 準備訓練參數
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers = lora_layers.to("cuda", dtype=torch.float16)

    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )

    # 4. 準備資料集
    dataset = ImageCaptionDataset(
        dataset_path,
        image_dir,
        pipe.tokenizer,
        pipe.feature_extractor
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # 5. 訓練循環
    unet.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
            input_ids = batch["input_ids"].to("cuda")

            # 編碼圖片到潛在空間
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            # 添加噪聲
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # 獲取文字嵌入
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]

            # 預測噪聲
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states
            ).sample

            # 計算損失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

        # 每10個epoch保存一次
        if (epoch + 1) % 10 == 0:
            save_path = f"{output_dir}/lora_epoch_{epoch+1}.safetensors"
            pipe.save_lora_weights(save_path)
            print(f"Saved LoRA weights to {save_path}")

    # 保存最終模型
    pipe.save_lora_weights(f"{output_dir}/lora_final.safetensors")
    print("Training completed!")

# 執行訓練
if __name__ == "__main__":
    train_lora(
        model_id="runwayml/stable-diffusion-v1-5",
        dataset_path="dataset/metadata.csv",
        image_dir="dataset/images",
        output_dir="my_lora",
        rank=8,
        learning_rate=1e-4,
        num_epochs=100,
        batch_size=2
    )
```

### LoRA 訓練最佳實踐

#### 資料準備

```python
# prepare_dataset.py
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def prepare_training_data(
    input_folder,
    output_folder,
    target_size=512,
    caption_prefix="a photo of"
):
    """
    準備訓練資料
    - 調整圖片大小
    - 生成caption文件
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/images", exist_ok=True)

    captions = []

    for filename in tqdm(os.listdir(input_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 加載並調整圖片
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        # 智能裁剪到正方形
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

        # 調整大小
        img = img.resize((target_size, target_size), Image.LANCZOS)

        # 保存
        output_path = f"{output_folder}/images/{filename}"
        img.save(output_path, quality=95)

        # 生成caption
        # 可以使用BLIP等模型自動生成更好的caption
        caption = f"{caption_prefix} {os.path.splitext(filename)[0].replace('_', ' ')}"
        captions.append({
            "image": filename,
            "caption": caption
        })

    # 保存metadata
    df = pd.DataFrame(captions)
    df.to_csv(f"{output_folder}/metadata.csv", index=False)
    print(f"Prepared {len(captions)} images for training")

# 使用示例
prepare_training_data(
    input_folder="raw_images",
    output_folder="dataset",
    target_size=512,
    caption_prefix="a portrait of person"
)
```

---

## 💡 進階技巧

### 1. Prompt Engineering（提示詞工程）

#### 高品質提示詞結構

```python
def build_quality_prompt(
    subject,
    style="photorealistic",
    quality_tags=True,
    camera_settings=True,
    lighting=True
):
    """構建高品質提示詞"""

    prompt_parts = [subject]

    if style:
        prompt_parts.append(style)

    if lighting:
        lighting_terms = [
            "natural lighting",
            "soft shadows",
            "golden hour"
        ]
        prompt_parts.extend(lighting_terms)

    if camera_settings:
        camera_terms = [
            "bokeh background",
            "shallow depth of field",
            "85mm lens",
            "f/1.8"
        ]
        prompt_parts.extend(camera_terms)

    if quality_tags:
        quality_terms = [
            "highly detailed",
            "sharp focus",
            "8k uhd",
            "professional photography"
        ]
        prompt_parts.extend(quality_terms)

    return ", ".join(prompt_parts)

# 示例
prompt = build_quality_prompt(
    subject="a beautiful woman with long hair",
    style="photorealistic portrait",
    quality_tags=True,
    camera_settings=True,
    lighting=True
)
print(prompt)
# 輸出: "a beautiful woman with long hair, photorealistic portrait,
#        natural lighting, soft shadows, golden hour, bokeh background,
#        shallow depth of field, 85mm lens, f/1.8, highly detailed,
#        sharp focus, 8k uhd, professional photography"
```

#### Negative Prompt 模板

```python
NEGATIVE_PROMPTS = {
    "photorealistic": """
        cartoon, anime, 3d render, drawing, painting,
        low quality, blurry, distorted, deformed,
        bad anatomy, bad proportions, extra limbs,
        duplicate, watermark, signature, text
    """,

    "portrait": """
        multiple people, crowd, far away,
        bad face, ugly face, bad eyes, crossed eyes,
        bad hands, extra fingers, missing fingers,
        low quality, blurry
    """,

    "landscape": """
        people, person, human, character,
        low quality, blurry, foggy,
        oversaturated, undersaturated,
        watermark, text
    """,

    "product": """
        person, people, background clutter,
        low quality, blurry, distorted,
        bad lighting, shadows on product,
        watermark, text
    """
}

# 使用
negative_prompt = NEGATIVE_PROMPTS["photorealistic"]
```

### 2. 批量生成與網格輸出

```python
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

def batch_generate_with_grid(
    prompts,
    output_dir="outputs",
    grid_cols=4,
    **generation_kwargs
):
    """
    批量生成圖片並建立網格預覽

    Args:
        prompts: 提示詞列表
        output_dir: 輸出目錄
        grid_cols: 網格列數
        **generation_kwargs: 傳遞給pipeline的其他參數
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加載模型
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    # 生成圖片
    generated_images = []

    for idx, prompt in enumerate(prompts):
        print(f"Generating {idx+1}/{len(prompts)}: {prompt[:50]}...")

        image = pipe(prompt, **generation_kwargs).images[0]

        # 保存單張圖片
        image_path = f"{output_dir}/image_{idx:04d}.png"
        image.save(image_path)
        generated_images.append(image)

    # 建立網格
    create_image_grid(generated_images, f"{output_dir}/grid.png", cols=grid_cols)
    print(f"Generated {len(prompts)} images in {output_dir}")

def create_image_grid(images, output_path, cols=4):
    """建立圖片網格"""
    n_images = len(images)
    rows = (n_images + cols - 1) // cols

    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, box=(col * w, row * h))

    grid.save(output_path)
    print(f"Grid saved to {output_path}")

# 使用示例
prompts = [
    "a red apple on white background",
    "a blue car in the city",
    "a cute cat sleeping",
    "a mountain landscape at sunset",
    "a modern architecture building",
    "a delicious pizza close-up",
    "a flower in macro photography",
    "a sci-fi spaceship design"
]

batch_generate_with_grid(
    prompts,
    output_dir="batch_outputs",
    grid_cols=4,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512,
    height=512
)
```

### 3. 圖片修復 (Inpainting)

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

def inpaint_image(
    image_path,
    mask_path,
    prompt,
    negative_prompt="",
    strength=0.8
):
    """
    圖片修復/編輯

    Args:
        image_path: 原始圖片路徑
        mask_path: 遮罩圖片路徑（白色區域會被重繪）
        prompt: 描述要生成的內容
        strength: 修改強度 (0-1)
    """
    # 加載模型
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # 加載圖片和遮罩
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # 灰度圖

    # 執行修復
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=50,
        strength=strength,
        guidance_scale=7.5
    ).images[0]

    return result

# 使用示例
result = inpaint_image(
    image_path="photo.jpg",
    mask_path="mask.png",
    prompt="a beautiful flower bouquet",
    negative_prompt="low quality, blurry"
)
result.save("inpainted_result.png")
```

### 4. 圖片放大 (Upscaling)

```python
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch

def upscale_image(image_path, prompt, scale_factor=4):
    """
    使用Stable Diffusion進行圖片放大

    Args:
        image_path: 輸入圖片路徑
        prompt: 圖片描述（幫助保持細節）
        scale_factor: 放大倍數（2或4）
    """
    # 加載放大模型
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    # 加載低解析度圖片
    low_res_img = Image.open(image_path).convert("RGB")

    # 執行放大
    upscaled_image = pipe(
        prompt=prompt,
        image=low_res_img,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    return upscaled_image

# 使用示例
upscaled = upscale_image(
    image_path="low_res.jpg",
    prompt="a high quality portrait photo",
    scale_factor=4
)
upscaled.save("upscaled_4x.png")
```

---

## 🚀 實戰案例

### 案例1：AI頭像生成器

```python
# avatar_generator.py
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random

class AvatarGenerator:
    """AI頭像生成器"""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

    def generate_avatar(
        self,
        gender="female",
        style="realistic",
        age_range="young adult",
        hair_color=None,
        customization=None
    ):
        """
        生成頭像

        Args:
            gender: 性別 (male/female/neutral)
            style: 風格 (realistic/anime/cartoon/artistic)
            age_range: 年齡範圍
            hair_color: 髮色
            customization: 額外定制
        """
        # 構建提示詞
        prompt_parts = []

        # 基礎描述
        if style == "realistic":
            prompt_parts.append(f"professional portrait photo of a {age_range} {gender}")
        elif style == "anime":
            prompt_parts.append(f"anime style portrait of a {age_range} {gender}")
        elif style == "cartoon":
            prompt_parts.append(f"cartoon style avatar of a {age_range} {gender}")
        else:
            prompt_parts.append(f"artistic portrait of a {age_range} {gender}")

        # 髮色
        if hair_color:
            prompt_parts.append(f"{hair_color} hair")

        # 定制化
        if customization:
            prompt_parts.append(customization)

        # 質量標籤
        if style == "realistic":
            prompt_parts.extend([
                "professional photography",
                "studio lighting",
                "bokeh background",
                "highly detailed face",
                "8k uhd",
                "sharp focus"
            ])

        prompt = ", ".join(prompt_parts)

        # Negative prompt
        negative_prompt = """
            multiple people, full body, far away,
            low quality, blurry, distorted face,
            bad anatomy, bad eyes, bad hands,
            duplicate, watermark
        """

        # 生成
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            width=512,
            height=512,
            generator=torch.Generator("cuda").manual_seed(random.randint(0, 2**32))
        ).images[0]

        return image, prompt

# 使用示例
generator = AvatarGenerator()

# 生成不同風格的頭像
avatar1, prompt1 = generator.generate_avatar(
    gender="female",
    style="realistic",
    age_range="young adult",
    hair_color="blonde",
    customization="smiling, blue eyes"
)
avatar1.save("avatar_realistic.png")

avatar2, prompt2 = generator.generate_avatar(
    gender="male",
    style="anime",
    age_range="teenager",
    hair_color="black",
    customization="cool expression, wearing headphones"
)
avatar2.save("avatar_anime.png")

print("Generated avatars:")
print(f"1. {prompt1}")
print(f"2. {prompt2}")
```

### 案例2：產品圖生成工具

```python
# product_image_generator.py
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image

class ProductImageGenerator:
    """電商產品圖生成工具"""

    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    def generate_product_shot(
        self,
        product_description,
        background="white studio",
        angle="front view",
        lighting="professional studio lighting",
        num_variants=4
    ):
        """
        生成產品展示圖

        Args:
            product_description: 產品描述
            background: 背景設置
            angle: 拍攝角度
            lighting: 光照設置
            num_variants: 生成變體數量
        """
        # 構建提示詞
        prompt = f"""
        product photography of {product_description},
        {angle}, {background}, {lighting},
        professional commercial photo, high quality,
        sharp focus, detailed, 8k uhd, clean composition,
        no people, no text
        """

        negative_prompt = """
        person, people, hands holding product,
        low quality, blurry, cluttered background,
        shadows on product, poor lighting,
        watermark, text, logo
        """

        # 生成多個變體
        images = []
        for i in range(num_variants):
            seed = i * 1000
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=8.0,
                width=768,
                height=768,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]
            images.append(image)

        return images, prompt

# 使用示例
generator = ProductImageGenerator()

# 生成手錶產品圖
watch_images, prompt = generator.generate_product_shot(
    product_description="luxury silver watch with leather strap",
    background="white marble surface",
    angle="45 degree angle",
    lighting="soft diffused lighting",
    num_variants=4
)

# 保存
for idx, img in enumerate(watch_images):
    img.save(f"product_watch_{idx+1}.png")

print(f"Generated {len(watch_images)} product images")
print(f"Prompt used: {prompt}")
```

---

## 📚 參考資源

### 官方文檔
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- [Stable Diffusion GitHub](https://github.com/CompVis/stable-diffusion)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### 模型資源
- [Hugging Face Models](https://huggingface.co/models?pipeline_tag=text-to-image)
- [Civitai](https://civitai.com/) - 社群模型和LoRA分享
- [Stability AI](https://stability.ai/)

### 學習資源
- [Fast.ai Diffusion Course](https://www.fast.ai/)
- [Stable Diffusion Art](https://stable-diffusion-art.com/)

---

## ✅ 檢查清單

完成本章節後，你應該能夠：

- [ ] 理解Diffusion模型的基本原理
- [ ] 使用Stable Diffusion生成高品質圖片
- [ ] 掌握重要參數的調節技巧
- [ ] 使用ControlNet進行精確控制
- [ ] 組合多個ControlNet
- [ ] 加載和使用LoRA模型
- [ ] 訓練自定義LoRA
- [ ] 編寫高品質的提示詞
- [ ] 進行圖片修復和編輯
- [ ] 實現圖片放大
- [ ] 構建實用的生成應用

---

## 下一步

完成圖片生成後，建議繼續學習：

1. **影片生成** - 了解如何將靜態圖片擴展到動態影片
2. **音樂生成** - 探索音頻生成技術
3. **實戰項目** - 構建完整的多模態應用

---

最後更新：2024-11-19
難度級別：🟡 中級
預計學習時間：15-20小時
