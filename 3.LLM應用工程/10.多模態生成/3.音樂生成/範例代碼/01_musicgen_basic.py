"""
MusicGen 音樂生成
使用 Meta 的 MusicGen 模型生成音樂
"""

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import numpy as np
from typing import List, Optional


class MusicGenerator:
    """音樂生成器"""

    def __init__(self, model_size: str = "small"):
        """
        初始化音樂生成器

        Args:
            model_size: 模型大小
                - small: 最快，品質較低 (300M)
                - medium: 平衡 (1.5B)
                - large: 最佳品質，最慢 (3.3B)
                - melody: 支持旋律條件 (1.5B)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")

        # 模型映射
        model_map = {
            "small": "facebook/musicgen-small",
            "medium": "facebook/musicgen-medium",
            "large": "facebook/musicgen-large",
            "melody": "facebook/musicgen-melody"
        }

        if model_size not in model_map:
            raise ValueError(f"不支持的模型大小: {model_size}")

        model_id = model_map[model_size]

        # 載入模型和處理器
        print(f"正在載入模型: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        self.model = self.model.to(self.device)

        self.model_size = model_size
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate

    def generate(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        do_sample: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        生成音樂

        Args:
            prompt: 音樂描述文字
            duration: 音樂長度（秒）
            temperature: 溫度參數（越高越有創意，但可能不連貫）
            top_k: Top-K 採樣
            top_p: Top-P 採樣
            guidance_scale: 引導強度（對提示詞的遵循程度）
            do_sample: 是否使用採樣（False 則使用貪婪解碼）
            seed: 隨機種子

        Returns:
            音訊陣列 (sampling_rate, audio_values)
        """
        print(f"\n生成音樂:")
        print(f"  提示詞: {prompt}")
        print(f"  時長: {duration}秒")

        # 處理輸入
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 計算最大長度（tokens）
        max_new_tokens = int(duration * self.model.config.audio_encoder.frame_rate)

        # 設定種子
        if seed is not None:
            torch.manual_seed(seed)

        # 生成音樂
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale
            )

        # 轉換為 numpy 陣列
        audio_array = audio_values[0, 0].cpu().numpy()

        return audio_array

    def save_audio(
        self,
        audio_array: np.ndarray,
        output_path: str,
        format: str = "wav"
    ):
        """
        儲存音訊文件

        Args:
            audio_array: 音訊陣列
            output_path: 輸出路徑
            format: 音訊格式 (wav, mp3)
        """
        if format == "wav":
            scipy.io.wavfile.write(
                output_path,
                rate=self.sampling_rate,
                data=audio_array
            )
        elif format == "mp3":
            # 需要安裝 pydub 和 ffmpeg
            from pydub import AudioSegment
            temp_wav = "temp_audio.wav"
            scipy.io.wavfile.write(temp_wav, rate=self.sampling_rate, data=audio_array)
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format="mp3")
            import os
            os.remove(temp_wav)
        else:
            raise ValueError(f"不支持的格式: {format}")

        print(f"音訊已儲存至: {output_path}")

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        **kwargs
    ):
        """生成並儲存音樂（便捷方法）"""
        audio = self.generate(prompt, **kwargs)
        self.save_audio(audio, output_path)
        return audio


def example_basic_generation():
    """示例 1: 基本音樂生成"""
    print("=== 示例 1: 基本音樂生成 ===")

    generator = MusicGenerator(model_size="small")

    # 生成不同風格的音樂
    prompts = [
        "upbeat pop music with electric guitar",
        "calm piano melody for relaxation",
        "energetic electronic dance music",
        "acoustic folk song with guitar"
    ]

    for i, prompt in enumerate(prompts):
        output_path = f"music_{i+1}.wav"
        generator.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            duration=10.0,
            seed=42
        )


def example_different_parameters():
    """示例 2: 不同參數效果"""
    print("\n=== 示例 2: 參數調整 ===")

    generator = MusicGenerator(model_size="small")
    prompt = "jazz music with saxophone"

    # 測試不同溫度
    temperatures = [0.8, 1.0, 1.2]
    for temp in temperatures:
        print(f"\n溫度 = {temp}")
        generator.generate_and_save(
            prompt=prompt,
            output_path=f"jazz_temp_{temp}.wav",
            duration=8.0,
            temperature=temp,
            seed=42
        )

    # 測試不同引導強度
    guidance_scales = [2.0, 3.0, 5.0]
    for scale in guidance_scales:
        print(f"\n引導強度 = {scale}")
        generator.generate_and_save(
            prompt=prompt,
            output_path=f"jazz_guidance_{scale}.wav",
            duration=8.0,
            guidance_scale=scale,
            seed=42
        )


def example_music_styles():
    """示例 3: 各種音樂風格"""
    print("\n=== 示例 3: 音樂風格庫 ===")

    generator = MusicGenerator(model_size="small")

    # 詳細的風格描述模板
    styles = {
        "古典": "classical orchestral music with strings and piano, elegant and sophisticated",
        "搖滾": "energetic rock music with electric guitar, bass and drums, powerful and intense",
        "爵士": "smooth jazz music with saxophone and piano, relaxed and sophisticated",
        "電子": "electronic dance music with synthesizers, upbeat and energetic",
        "環境音樂": "ambient atmospheric music, calm and meditative",
        "流行": "catchy pop music with melody, upbeat and cheerful",
        "嘻哈": "hip hop beat with bass and drums, rhythmic and groovy",
        "鄉村": "country music with acoustic guitar, warm and storytelling",
        "雷鬼": "reggae music with offbeat rhythm, relaxed and tropical",
        "金屬": "heavy metal music with distorted guitars, aggressive and powerful"
    }

    for style_name, prompt in styles.items():
        print(f"\n生成 {style_name} 風格...")
        generator.generate_and_save(
            prompt=prompt,
            output_path=f"style_{style_name}.wav",
            duration=10.0,
            seed=42
        )


def example_loop_generation():
    """示例 4: 生成循環音樂"""
    print("\n=== 示例 4: 循環音樂生成 ===")

    generator = MusicGenerator(model_size="small")

    # 生成適合循環的音樂
    loop_prompts = [
        "short looping drum beat, 4 bar loop",
        "looping bass line, repetitive and groovy",
        "ambient pad loop, atmospheric and continuous",
        "melodic synth loop, catchy and repetitive"
    ]

    for i, prompt in enumerate(loop_prompts):
        generator.generate_and_save(
            prompt=prompt,
            output_path=f"loop_{i+1}.wav",
            duration=4.0,  # 短循環
            temperature=0.9,
            seed=42
        )


def example_batch_generation():
    """示例 5: 批量生成"""
    print("\n=== 示例 5: 批量生成音樂庫 ===")

    generator = MusicGenerator(model_size="small")

    # 為遊戲或影片創建音樂庫
    music_library = {
        "background": [
            "calm background music for cafe",
            "peaceful background music for study",
            "uplifting background music for video"
        ],
        "action": [
            "intense action music with drums",
            "fast-paced chase music",
            "epic battle music with orchestra"
        ],
        "emotional": [
            "sad emotional piano music",
            "hopeful uplifting music",
            "romantic music with strings"
        ]
    }

    import os
    for category, prompts in music_library.items():
        os.makedirs(f"music_library/{category}", exist_ok=True)

        for i, prompt in enumerate(prompts):
            output_path = f"music_library/{category}/track_{i+1}.wav"
            print(f"\n生成 {category}/{i+1}...")
            generator.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                duration=15.0,
                seed=42 + i
            )


if __name__ == "__main__":
    print("MusicGen 音樂生成示例")
    print("=" * 60)
    print("\n注意:")
    print("1. 首次運行會下載模型（small: ~300MB, medium: ~1.5GB）")
    print("2. 建議使用 GPU 加速生成")
    print("3. 生成時間取決於音樂長度和模型大小\n")

    # 運行基本示例
    example_basic_generation()

    # 運行其他示例（取消註釋以執行）
    # example_different_parameters()
    # example_music_styles()
    # example_loop_generation()
    # example_batch_generation()

    print("\n所有示例完成！")
    print("\n提示詞技巧:")
    print("1. 描述樂器: 'with piano', 'with guitar'")
    print("2. 描述節奏: 'upbeat', 'slow', 'energetic'")
    print("3. 描述情緒: 'happy', 'sad', 'peaceful'")
    print("4. 描述風格: 'jazz', 'rock', 'classical'")
    print("5. 組合使用: 'calm piano jazz music for relaxation'")
