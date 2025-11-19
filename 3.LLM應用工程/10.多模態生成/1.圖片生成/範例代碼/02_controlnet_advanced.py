"""
ControlNet 進階控制
使用 ControlNet 實現精確的圖片生成控制
支持多種控制類型：Canny邊緣、深度圖、姿態等
"""

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from controlnet_aux import CannyDetector, OpenposeDetector, HEDdetector


class ControlNetGenerator:
    """ControlNet 圖片生成器"""

    def __init__(self, controlnet_type: str = "canny"):
        """
        初始化 ControlNet

        Args:
            controlnet_type: 控制類型 (canny, openpose, hed, depth 等)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")

        # ControlNet 模型映射
        controlnet_models = {
            "canny": "lllyasviel/sd-controlnet-canny",
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "hed": "lllyasviel/sd-controlnet-hed",
            "depth": "lllyasviel/sd-controlnet-depth",
            "normal": "lllyasviel/sd-controlnet-normal",
            "scribble": "lllyasviel/sd-controlnet-scribble",
            "seg": "lllyasviel/sd-controlnet-seg"
        }

        if controlnet_type not in controlnet_models:
            raise ValueError(f"不支持的控制類型: {controlnet_type}")

        # 載入 ControlNet
        print(f"正在載入 ControlNet ({controlnet_type})...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_models[controlnet_type],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # 載入 Stable Diffusion Pipeline
        print("正在載入 Stable Diffusion Pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )

        # 使用更快的調度器
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to(self.device)

        # 記憶體優化
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()

        self.controlnet_type = controlnet_type

    def preprocess_canny(self, image: Image.Image, low_threshold: int = 100,
                        high_threshold: int = 200) -> Image.Image:
        """
        Canny 邊緣檢測預處理

        Args:
            image: 輸入圖片
            low_threshold: Canny 低閾值
            high_threshold: Canny 高閾值
        """
        # 轉換為 numpy 陣列
        image_np = np.array(image)

        # Canny 邊緣檢測
        edges = cv2.Canny(image_np, low_threshold, high_threshold)

        # 轉換回 PIL Image
        edges = Image.fromarray(edges)
        return edges

    def preprocess_openpose(self, image: Image.Image) -> Image.Image:
        """OpenPose 姿態檢測預處理"""
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        pose = detector(image)
        return pose

    def preprocess_hed(self, image: Image.Image) -> Image.Image:
        """HED 邊緣檢測預處理"""
        detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        edges = detector(image)
        return edges

    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str = "low quality, blurry",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = None
    ) -> Image.Image:
        """
        生成圖片

        Args:
            control_image: 控制圖片（已預處理）
            prompt: 提示詞
            negative_prompt: 負向提示詞
            num_inference_steps: 推理步數
            guidance_scale: 引導強度
            controlnet_conditioning_scale: ControlNet 控制強度（0-2，越高控制越強）
            seed: 隨機種子
        """
        # 設定種子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成圖片
        print(f"正在生成圖片...")
        print(f"提示詞: {prompt}")
        print(f"ControlNet 強度: {controlnet_conditioning_scale}")

        with torch.autocast(self.device):
            output = self.pipe(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator
            )

        return output.images[0]


def example_canny_edge():
    """示例 1: Canny 邊緣控制"""
    print("=== 示例 1: Canny 邊緣控制 ===")

    # 載入參考圖片（這裡假設有一張參考圖）
    # 實際使用時請替換為您的圖片路徑
    reference_image = Image.new("RGB", (512, 512), color="white")
    # reference_image = Image.open("your_reference_image.jpg")

    # 初始化生成器
    generator = ControlNetGenerator(controlnet_type="canny")

    # 預處理：提取邊緣
    canny_image = generator.preprocess_canny(reference_image)
    canny_image.save("canny_edges.png")
    print("邊緣圖已儲存: canny_edges.png")

    # 生成新圖片
    result = generator.generate(
        control_image=canny_image,
        prompt="a beautiful anime girl, high quality, detailed",
        negative_prompt="low quality, blurry, ugly",
        num_inference_steps=30,
        controlnet_conditioning_scale=1.0,
        seed=42
    )

    result.save("canny_result.png")
    print("生成結果已儲存: canny_result.png")


def example_style_transfer():
    """示例 2: 風格轉換"""
    print("\n=== 示例 2: 保持構圖的風格轉換 ===")

    # 載入原圖
    original = Image.new("RGB", (512, 512), color="white")
    # original = Image.open("original_photo.jpg")

    # 初始化
    generator = ControlNetGenerator(controlnet_type="canny")

    # 提取邊緣
    edges = generator.preprocess_canny(original, low_threshold=50, high_threshold=150)

    # 不同風格的提示詞
    styles = [
        ("oil painting style, vibrant colors", "oil_painting.png"),
        ("watercolor style, soft colors", "watercolor.png"),
        ("anime style, cell shading", "anime.png"),
        ("cyberpunk style, neon lights", "cyberpunk.png")
    ]

    for prompt, filename in styles:
        result = generator.generate(
            control_image=edges,
            prompt=prompt,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.8,
            seed=42
        )
        result.save(filename)
        print(f"已生成: {filename}")


def example_multiple_strength():
    """示例 3: 不同控制強度的效果"""
    print("\n=== 示例 3: 不同 ControlNet 強度 ===")

    # 載入參考圖
    reference = Image.new("RGB", (512, 512), color="white")

    # 初始化
    generator = ControlNetGenerator(controlnet_type="canny")
    edges = generator.preprocess_canny(reference)

    prompt = "a beautiful landscape with mountains and lake"

    # 測試不同的控制強度
    strengths = [0.3, 0.5, 0.8, 1.0, 1.5]

    for strength in strengths:
        result = generator.generate(
            control_image=edges,
            prompt=prompt,
            controlnet_conditioning_scale=strength,
            seed=42
        )
        result.save(f"strength_{strength}.png")
        print(f"已生成 (強度={strength}): strength_{strength}.png")


if __name__ == "__main__":
    print("ControlNet 進階控制示例")
    print("=" * 50)

    # 運行示例
    # 注意: 這些示例需要有實際的參考圖片才能正常運行
    # 請將 Image.new() 替換為 Image.open("your_image.jpg")

    example_canny_edge()
    # example_style_transfer()
    # example_multiple_strength()

    print("\n所有示例完成！")
    print("\n使用提示:")
    print("1. 替換示例中的 Image.new() 為實際圖片")
    print("2. 調整 controlnet_conditioning_scale 控制強度")
    print("3. 嘗試不同的預處理參數獲得最佳效果")
