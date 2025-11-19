"""
Stable Diffusion 基礎圖片生成
基本的文生圖功能，使用 Stable Diffusion v1.5
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def generate_image_basic(
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None,
    output_path: str = "output.png"
):
    """
    使用 Stable Diffusion 生成圖片

    Args:
        prompt: 正向提示詞
        negative_prompt: 負向提示詞
        num_inference_steps: 推理步數（越多質量越好但越慢）
        guidance_scale: 引導強度（控制對提示詞的遵循程度）
        width: 圖片寬度
        height: 圖片高度
        seed: 隨機種子（用於可重現性）
        output_path: 輸出路徑
    """

    # 載入模型
    print("正在載入 Stable Diffusion 模型...")
    model_id = "runwayml/stable-diffusion-v1-5"

    # 檢查是否有 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 建立 pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # 關閉安全檢查器（可選）
    )
    pipe = pipe.to(device)

    # 啟用記憶體優化（如果使用 GPU）
    if device == "cuda":
        pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()  # 需要安裝 xformers

    # 設定隨機種子
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"使用種子: {seed}")

    # 生成圖片
    print(f"正在生成圖片...")
    print(f"提示詞: {prompt}")

    with torch.autocast(device):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]

    # 儲存圖片
    image.save(output_path)
    print(f"圖片已儲存至: {output_path}")

    return image


def batch_generate(prompts: list, output_dir: str = "outputs"):
    """
    批量生成多張圖片

    Args:
        prompts: 提示詞列表
        output_dir: 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        output_path = os.path.join(output_dir, f"image_{i+1:03d}.png")
        generate_image_basic(
            prompt=prompt,
            seed=42 + i,  # 每張圖使用不同的種子
            output_path=output_path
        )


if __name__ == "__main__":
    # 示例 1: 基本使用
    print("=== 示例 1: 基本圖片生成 ===")
    generate_image_basic(
        prompt="a beautiful landscape with mountains and lake, sunset, photorealistic",
        negative_prompt="low quality, blurry, cartoon",
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
        output_path="landscape.png"
    )

    # 示例 2: 人物肖像
    print("\n=== 示例 2: 人物肖像 ===")
    generate_image_basic(
        prompt="professional portrait photo of a smiling woman, natural lighting, high quality",
        negative_prompt="low quality, blurry, distorted face",
        num_inference_steps=50,
        guidance_scale=8.0,
        seed=123,
        output_path="portrait.png"
    )

    # 示例 3: 批量生成
    print("\n=== 示例 3: 批量生成 ===")
    prompts = [
        "a cute cat sitting on a windowsill",
        "a modern city skyline at night",
        "a cozy coffee shop interior",
        "a fantasy castle in the clouds",
        "a vintage car on a desert road"
    ]
    batch_generate(prompts, output_dir="batch_outputs")

    print("\n所有圖片生成完成！")
