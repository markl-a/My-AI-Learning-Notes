"""
Stable Video Diffusion 影片生成
從單張圖片生成短視頻
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import os


class VideoGenerator:
    """影片生成器"""

    def __init__(self, model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"):
        """
        初始化影片生成器

        Args:
            model_id: 模型 ID
                - stabilityai/stable-video-diffusion-img2vid: 14 幀版本
                - stabilityai/stable-video-diffusion-img2vid-xt: 25 幀版本（推薦）
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")

        if self.device == "cpu":
            print("警告: CPU 生成影片會非常慢，強烈建議使用 GPU")

        # 載入模型
        print(f"正在載入模型: {model_id}")
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None
        )
        self.pipe = self.pipe.to(self.device)

        # 記憶體優化
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            # self.pipe.enable_attention_slicing()

    def generate_video(
        self,
        image: Image.Image,
        num_frames: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 8,
        seed: int = 42,
        output_path: str = "output.mp4"
    ) -> str:
        """
        從圖片生成影片

        Args:
            image: 輸入圖片
            num_frames: 生成幀數 (14 或 25)
            fps: 影片幀率
            motion_bucket_id: 運動強度 (1-255，數值越大運動越明顯)
            noise_aug_strength: 噪聲增強強度 (0-1)
            decode_chunk_size: 解碼批次大小（影響記憶體使用）
            seed: 隨機種子
            output_path: 輸出路徑

        Returns:
            輸出路徑
        """
        # 確保圖片尺寸是 8 的倍數
        width, height = image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        image = image.resize((width, height))

        print(f"圖片尺寸: {width}x{height}")
        print(f"生成幀數: {num_frames}")
        print(f"運動強度: {motion_bucket_id}")

        # 設定種子
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成影片幀
        print("正在生成影片...")
        frames = self.pipe(
            image=image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            generator=generator
        ).frames[0]

        # 匯出影片
        export_to_video(frames, output_path, fps=fps)
        print(f"影片已儲存至: {output_path}")

        return output_path

    def batch_generate(
        self,
        images: list,
        output_dir: str = "videos",
        **kwargs
    ):
        """
        批量生成影片

        Args:
            images: 圖片列表（PIL Image 或路徑）
            output_dir: 輸出目錄
            **kwargs: 傳遞給 generate_video 的參數
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate(images):
            if isinstance(img, str):
                img = Image.open(img)

            output_path = os.path.join(output_dir, f"video_{i+1:03d}.mp4")
            self.generate_video(
                image=img,
                output_path=output_path,
                **kwargs
            )


def example_basic_generation():
    """示例 1: 基本影片生成"""
    print("=== 示例 1: 基本影片生成 ===")

    # 創建或載入圖片
    # 方法 1: 從本地載入
    # image = Image.open("input_image.jpg")

    # 方法 2: 從 URL 載入
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    image = load_image(image_url)

    # 初始化生成器
    generator = VideoGenerator()

    # 生成影片
    generator.generate_video(
        image=image,
        num_frames=25,
        fps=7,
        motion_bucket_id=127,
        output_path="basic_video.mp4"
    )


def example_motion_control():
    """示例 2: 控制運動強度"""
    print("\n=== 示例 2: 不同運動強度 ===")

    # 載入圖片
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    image = load_image(image_url)

    generator = VideoGenerator()

    # 測試不同的運動強度
    motion_strengths = [50, 100, 150, 200]

    for motion in motion_strengths:
        print(f"\n生成運動強度 {motion} 的影片...")
        generator.generate_video(
            image=image,
            num_frames=25,
            fps=7,
            motion_bucket_id=motion,
            output_path=f"motion_{motion}.mp4"
        )


def example_long_video():
    """示例 3: 生成較長的影片（通過連接多個片段）"""
    print("\n=== 示例 3: 生成長影片 ===")

    from moviepy.editor import VideoFileClip, concatenate_videoclips

    # 載入圖片
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    image = load_image(image_url)

    generator = VideoGenerator()

    # 生成多個片段
    num_segments = 3
    segment_paths = []

    for i in range(num_segments):
        print(f"\n生成片段 {i+1}/{num_segments}...")
        output_path = f"segment_{i+1}.mp4"
        generator.generate_video(
            image=image,
            num_frames=25,
            fps=7,
            seed=42 + i,  # 每個片段使用不同種子
            output_path=output_path
        )
        segment_paths.append(output_path)

    # 連接片段
    print("\n正在連接影片片段...")
    clips = [VideoFileClip(path) for path in segment_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("long_video.mp4", fps=7)

    # 清理臨時文件
    for path in segment_paths:
        os.remove(path)

    print("長影片已生成: long_video.mp4")


def example_image_sequence():
    """示例 4: 從圖片序列生成影片序列"""
    print("\n=== 示例 4: 圖片序列處理 ===")

    # 假設有多張圖片
    image_urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png",
        # 添加更多圖片 URL
    ]

    generator = VideoGenerator()

    # 批量處理
    images = [load_image(url) for url in image_urls]
    generator.batch_generate(
        images=images,
        output_dir="sequence_videos",
        num_frames=25,
        fps=7,
        motion_bucket_id=127
    )


if __name__ == "__main__":
    print("Stable Video Diffusion 影片生成示例")
    print("=" * 60)
    print("\n注意: 影片生成需要大量 GPU 記憶體 (建議 12GB+)")
    print("      如果記憶體不足，可以減少 num_frames 或使用 CPU offload\n")

    # 運行基本示例
    example_basic_generation()

    # 運行其他示例（取消註釋以執行）
    # example_motion_control()
    # example_long_video()  # 需要安裝 moviepy
    # example_image_sequence()

    print("\n所有示例完成！")
    print("\n調參建議:")
    print("1. motion_bucket_id: 控制運動幅度 (50-200)")
    print("2. noise_aug_strength: 控制變化程度 (0.0-0.1)")
    print("3. num_frames: 影響影片長度和記憶體使用")
    print("4. fps: 影響播放速度 (建議 6-12)")
