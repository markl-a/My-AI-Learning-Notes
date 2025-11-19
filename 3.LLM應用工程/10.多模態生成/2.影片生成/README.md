# 影片生成 (Video Generation)

本章節將深入探討AI影片生成技術，從圖片到視頻的擴展，涵蓋最新的視頻生成模型和技術。

## 📋 目錄

1. [視頻生成基礎](#視頻生成基礎)
2. [Stable Video Diffusion](#stable-video-diffusion)
3. [AnimateDiff](#animatediff)
4. [Text-to-Video](#text-to-video)
5. [視頻編輯與處理](#視頻編輯與處理)
6. [實戰案例](#實戰案例)

---

## 🎯 學習目標

完成本章節後，你將能夠：

- ✅ 理解視頻生成的基本原理
- ✅ 使用Stable Video Diffusion生成流暢視頻
- ✅ 運用AnimateDiff創建動畫
- ✅ 實現文本到視頻的生成
- ✅ 編輯和後處理生成的視頻
- ✅ 構建視頻生成應用

---

## 📚 視頻生成基礎

### 視頻生成的挑戰

與圖片生成相比，視頻生成面臨更多挑戰：

1. **時間一致性** - 幀與幀之間需要保持連貫
2. **運動合理性** - 物體運動需符合物理規律
3. **計算成本** - 需要生成多幀圖片
4. **記憶體需求** - 同時處理多幀數據

### 主要技術方案

```
視頻生成技術路線
├── Image-to-Video (圖片到視頻)
│   ├── Stable Video Diffusion (SVD)
│   └── DynamiCrafter
│
├── Text-to-Video (文本到視頻)
│   ├── ModelScope
│   ├── ZeroScope
│   └── AnimateDiff
│
└── Video-to-Video (視頻到視頻)
    ├── Runway Gen-2
    └── Pika Labs
```

---

## 🎬 Stable Video Diffusion

### 什麼是SVD？

Stable Video Diffusion (SVD) 是Stability AI推出的圖片到視頻模型，基於Stable Diffusion架構，專門針對視頻生成進行優化。

### 核心特點

- ✅ **高質量輸出** - 生成流暢自然的視頻
- ✅ **可控性強** - 可控制運動幅度和方向
- ✅ **多分辨率** - 支持不同分辨率輸出
- ✅ **開源免費** - 可本地部署

### 安裝與設置

```bash
# 安裝依賴
pip install diffusers transformers accelerate torch torchvision
pip install opencv-python pillow imageio imageio-ffmpeg

# 可選：安裝xformers加速
pip install xformers
```

### 基本使用

```python
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch

# 加載模型
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")

# 啟用記憶體優化
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# 加載輸入圖片
image = load_image("input_image.jpg")
image = image.resize((1024, 576))  # SVD推薦的寬高比

# 生成視頻（默認25幀）
frames = pipe(
    image=image,
    num_frames=25,
    decode_chunk_size=8,  # 降低記憶體使用
    num_inference_steps=25,
    fps=7,
    motion_bucket_id=127,  # 運動強度 (0-255)
    noise_aug_strength=0.02  # 噪聲增強強度
).frames[0]

# 導出視頻
export_to_video(frames, "output_video.mp4", fps=7)
print("Video generated successfully!")
```

### 參數調節指南

#### motion_bucket_id (運動強度)

```python
# 測試不同運動強度
motion_levels = {
    "subtle": 20,      # 微妙的運動
    "gentle": 60,      # 溫和的運動
    "moderate": 127,   # 中等運動（默認）
    "strong": 180,     # 強烈運動
    "extreme": 255     # 極度運動
}

for name, motion_id in motion_levels.items():
    frames = pipe(
        image=image,
        motion_bucket_id=motion_id,
        num_frames=25
    ).frames[0]

    export_to_video(frames, f"video_{name}_motion.mp4", fps=7)
```

#### 調整視頻長度和幀率

```python
def generate_custom_video(
    image_path,
    duration_seconds=3,
    fps=15,
    motion_strength=127
):
    """
    生成自定義時長和幀率的視頻

    Args:
        image_path: 輸入圖片路徑
        duration_seconds: 視頻時長（秒）
        fps: 幀率
        motion_strength: 運動強度 (0-255)
    """
    # 計算需要的幀數
    num_frames = duration_seconds * fps

    # SVD-XT最多支持25幀，需要分段生成長視頻
    max_frames_per_segment = 25

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    image = load_image(image_path).resize((1024, 576))

    if num_frames <= max_frames_per_segment:
        # 單次生成
        frames = pipe(
            image=image,
            num_frames=num_frames,
            motion_bucket_id=motion_strength,
            fps=fps
        ).frames[0]
    else:
        # 分段生成並拼接
        all_frames = []
        segments = (num_frames + max_frames_per_segment - 1) // max_frames_per_segment

        for i in range(segments):
            frames_this_segment = min(max_frames_per_segment, num_frames - i * max_frames_per_segment)

            segment_frames = pipe(
                image=image if i == 0 else all_frames[-1],  # 使用上一段的最後一幀
                num_frames=frames_this_segment,
                motion_bucket_id=motion_strength,
                fps=fps
            ).frames[0]

            all_frames.extend(segment_frames)

        frames = all_frames

    export_to_video(frames, "custom_video.mp4", fps=fps)
    return frames

# 使用示例：生成10秒、30fps的視頻
frames = generate_custom_video(
    "input.jpg",
    duration_seconds=10,
    fps=30,
    motion_strength=100
)
```

### 進階控制技巧

#### 1. 條件引導視頻生成

```python
from PIL import Image
import numpy as np

def generate_with_motion_control(
    image_path,
    motion_mask_path=None,
    camera_motion="zoom_in"
):
    """
    帶運動控制的視頻生成

    Args:
        image_path: 輸入圖片
        motion_mask_path: 運動遮罩（可選）
        camera_motion: 相機運動類型
    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    image = load_image(image_path).resize((1024, 576))

    # 根據相機運動類型調整參數
    motion_params = {
        "zoom_in": {"motion_bucket_id": 180, "noise_aug_strength": 0.01},
        "zoom_out": {"motion_bucket_id": 160, "noise_aug_strength": 0.015},
        "pan_left": {"motion_bucket_id": 140, "noise_aug_strength": 0.02},
        "pan_right": {"motion_bucket_id": 140, "noise_aug_strength": 0.02},
        "static": {"motion_bucket_id": 50, "noise_aug_strength": 0.005}
    }

    params = motion_params.get(camera_motion, motion_params["static"])

    frames = pipe(
        image=image,
        num_frames=25,
        **params
    ).frames[0]

    export_to_video(frames, f"video_{camera_motion}.mp4", fps=7)
    return frames

# 生成不同相機運動的視頻
for motion in ["zoom_in", "pan_left", "static"]:
    generate_with_motion_control("landscape.jpg", camera_motion=motion)
```

#### 2. 批量圖片到視頻轉換

```python
import os
from pathlib import Path

def batch_image_to_video(
    input_folder,
    output_folder,
    motion_strength=127,
    fps=7
):
    """批量將圖片轉換為視頻"""

    os.makedirs(output_folder, exist_ok=True)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    # 支持的圖片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    for img_file in Path(input_folder).iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue

        print(f"Processing {img_file.name}...")

        try:
            image = load_image(str(img_file)).resize((1024, 576))

            frames = pipe(
                image=image,
                num_frames=25,
                motion_bucket_id=motion_strength,
                fps=fps
            ).frames[0]

            output_path = f"{output_folder}/{img_file.stem}.mp4"
            export_to_video(frames, output_path, fps=fps)

            print(f"✓ Saved to {output_path}")

        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")

    print("Batch processing completed!")

# 使用
batch_image_to_video(
    input_folder="input_images",
    output_folder="output_videos",
    motion_strength=127,
    fps=7
)
```

---

## 🎨 AnimateDiff

### 什麼是AnimateDiff？

AnimateDiff是一個將任何Stable Diffusion模型變成視頻生成模型的插件，通過添加時間層來實現動畫生成。

### 核心優勢

- ✅ **兼容性強** - 可與任何SD模型和LoRA組合
- ✅ **靈活控制** - 支持MotionLoRA精確控制運動
- ✅ **社群資源豐富** - 大量預訓練運動模塊

### 安裝

```bash
pip install diffusers transformers accelerate torch
pip install git+https://github.com/guoyww/AnimateDiff.git
```

### 基本使用

```python
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
import torch

# 加載運動適配器
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)

# 創建AnimateDiff Pipeline
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    beta_schedule="linear",
    clip_sample=False
)
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# 生成動畫
prompt = "a cat walking on the street, high quality, detailed"
negative_prompt = "low quality, blurry, static"

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=16,
    num_inference_steps=25,
    guidance_scale=7.5
).frames[0]

# 導出為GIF
export_to_gif(frames, "animated_cat.gif", fps=8)
```

### 使用MotionLoRA

```python
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
import torch

# 加載基礎組件
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)

pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# 加載MotionLoRA（控制特定運動）
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-in",
    adapter_name="zoom_in"
)

# 生成帶特定運動的視頻
frames = pipe(
    prompt="a beautiful landscape, mountains and lake",
    num_frames=16,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8}  # LoRA強度
).frames[0]

export_to_gif(frames, "zoom_in_landscape.gif")
```

### 組合多個LoRA

```python
def generate_with_style_and_motion(
    prompt,
    style_lora_path,
    motion_lora_path,
    output_path="output.gif"
):
    """
    組合風格LoRA和運動LoRA生成動畫

    Args:
        prompt: 提示詞
        style_lora_path: 風格LoRA路徑
        motion_lora_path: 運動LoRA路徑
        output_path: 輸出路徑
    """
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=torch.float16
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # 加載風格LoRA
    pipe.load_lora_weights(style_lora_path, adapter_name="style")

    # 加載運動LoRA
    pipe.load_lora_weights(motion_lora_path, adapter_name="motion")

    # 設置兩個LoRA的權重
    pipe.set_adapters(["style", "motion"], adapter_weights=[0.8, 0.6])

    frames = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=7.5
    ).frames[0]

    export_to_gif(frames, output_path)
    return frames

# 使用示例
generate_with_style_and_motion(
    prompt="a cyberpunk city at night, neon lights",
    style_lora_path="path/to/cyberpunk_style_lora",
    motion_lora_path="guoyww/animatediff-motion-lora-pan-left",
    output_path="cyberpunk_city.gif"
)
```

---

## 📝 Text-to-Video

### 使用ModelScope

```python
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

# 加載ModelScope Text-to-Video模型
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# 生成視頻
prompt = "a panda eating bamboo in a bamboo forest"

video_frames = pipe(
    prompt=prompt,
    num_inference_steps=25,
    num_frames=16,
    height=320,
    width=576
).frames[0]

# 導出視頻
export_to_video(video_frames, "panda_video.mp4", fps=8)
```

### 進階：長視頻生成

```python
def generate_long_video(
    prompt,
    total_frames=64,
    frames_per_batch=16,
    overlap_frames=4
):
    """
    生成長視頻（通過分段生成並平滑過渡）

    Args:
        prompt: 提示詞
        total_frames: 總幀數
        frames_per_batch: 每批生成的幀數
        overlap_frames: 重疊幀數（用於平滑過渡）
    """
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    all_frames = []
    num_batches = (total_frames + frames_per_batch - 1) // frames_per_batch

    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")

        # 生成這一批幀
        frames = pipe(
            prompt=prompt,
            num_frames=frames_per_batch,
            num_inference_steps=25
        ).frames[0]

        if i == 0:
            # 第一批：全部保留
            all_frames.extend(frames)
        else:
            # 後續批次：跳過重疊部分
            all_frames.extend(frames[overlap_frames:])

    # 限制到目標幀數
    all_frames = all_frames[:total_frames]

    export_to_video(all_frames, "long_video.mp4", fps=8)
    return all_frames

# 生成64幀的長視頻
frames = generate_long_video(
    prompt="a beautiful sunset over the ocean, waves gently rolling",
    total_frames=64,
    frames_per_batch=16,
    overlap_frames=4
)
```

---

## 🛠️ 視頻編輯與處理

### 視頻插幀（提高幀率）

```python
import cv2
import numpy as np
from PIL import Image

def interpolate_frames(video_path, output_path, target_fps=30):
    """
    使用光流法進行視頻插幀

    Args:
        video_path: 輸入視頻路徑
        output_path: 輸出視頻路徑
        target_fps: 目標幀率
    """
    # 讀取視頻
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 計算插值倍數
    interpolation_factor = target_fps / source_fps

    # 創建視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    # 讀取第一幀
    ret, prev_frame = cap.read()
    if not ret:
        return

    out.write(prev_frame)

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        # 計算需要插入的幀數
        num_interpolated = int(interpolation_factor) - 1

        # 簡單線性插值（可以使用更高級的光流算法）
        for i in range(1, num_interpolated + 1):
            alpha = i / (num_interpolated + 1)
            interpolated = cv2.addWeighted(
                prev_frame, 1 - alpha,
                next_frame, alpha,
                0
            )
            out.write(interpolated)

        out.write(next_frame)
        prev_frame = next_frame

    cap.release()
    out.release()
    print(f"Interpolated video saved to {output_path}")

# 使用
interpolate_frames("input.mp4", "output_30fps.mp4", target_fps=30)
```

### 視頻穩定化

```python
def stabilize_video(input_path, output_path):
    """使用OpenCV穩定視頻"""
    import cv2

    cap = cv2.VideoCapture(input_path)

    # 獲取視頻信息
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 讀取第一幀
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames-1, 3), np.float32)

    # 計算幀間變換
    for i in range(n_frames-2):
        ret, curr = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # 檢測特徵點
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )

        # 光流追蹤
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )

        # 過濾有效點
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # 計算變換矩陣
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

        if m is not None:
            dx = m[0,2]
            dy = m[1,2]
            da = np.arctan2(m[1,0], m[0,0])
        else:
            dx = dy = da = 0

        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

    # 計算平滑軌跡
    trajectory = np.cumsum(transforms, axis=0)

    # 應用移動平均平滑
    smoothed_trajectory = smooth(trajectory)

    # 計算穩定化變換
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # 應用穩定化
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(n_frames-1):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, da = transforms_smooth[i]

        m = np.array([[np.cos(da), -np.sin(da), dx],
                      [np.sin(da), np.cos(da), dy]])

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        out.write(frame_stabilized)

    cap.release()
    out.release()

def smooth(trajectory, smoothing_radius=50):
    """移動平均平滑"""
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        for j in range(smoothing_radius, len(trajectory) - smoothing_radius):
            smoothed_trajectory[j, i] = np.mean(
                trajectory[j - smoothing_radius:j + smoothing_radius + 1, i]
            )
    return smoothed_trajectory

# 使用
stabilize_video("shaky_video.mp4", "stabilized_video.mp4")
```

### 添加音樂和音效

```python
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

def add_audio_to_video(
    video_path,
    audio_path,
    output_path,
    audio_start=0,
    audio_volume=1.0
):
    """
    為視頻添加音頻

    Args:
        video_path: 視頻文件路徑
        audio_path: 音頻文件路徑
        output_path: 輸出路徑
        audio_start: 音頻開始時間（秒）
        audio_volume: 音頻音量 (0-1)
    """
    # 加載視頻
    video = VideoFileClip(video_path)

    # 加載音頻
    audio = AudioFileClip(audio_path)

    # 調整音頻長度匹配視頻
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    elif audio.duration < video.duration:
        # 循環音頻
        n_loops = int(video.duration / audio.duration) + 1
        audio = CompositeAudioClip([audio] * n_loops).subclip(0, video.duration)

    # 調整音量
    audio = audio.volumex(audio_volume)

    # 設置音頻起始時間
    audio = audio.set_start(audio_start)

    # 合成
    final_video = video.set_audio(audio)

    # 導出
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=video.fps
    )

    print(f"Video with audio saved to {output_path}")

# 使用
add_audio_to_video(
    "generated_video.mp4",
    "background_music.mp3",
    "final_video.mp4",
    audio_volume=0.5
)
```

---

## 🚀 實戰案例

### 案例1：短視頻自動生成系統

```python
# short_video_generator.py
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw, ImageFont
import torch

class ShortVideoGenerator:
    """短視頻自動生成系統"""

    def __init__(self):
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

    def add_text_overlay(self, frame, text, position="bottom"):
        """在幀上添加文字"""
        draw = ImageDraw.Draw(frame)

        # 嘗試加載字體
        try:
            font = ImageFont.truetype("Arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        # 計算文字位置
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if position == "bottom":
            x = (frame.width - text_width) // 2
            y = frame.height - text_height - 50
        elif position == "top":
            x = (frame.width - text_width) // 2
            y = 50
        else:  # center
            x = (frame.width - text_width) // 2
            y = (frame.height - text_height) // 2

        # 繪製文字陰影
        draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
        # 繪製文字
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        return frame

    def generate_short_video(
        self,
        image_path,
        title_text,
        output_path="short_video.mp4",
        duration=3,
        fps=15
    ):
        """
        生成帶標題的短視頻

        Args:
            image_path: 輸入圖片路徑
            title_text: 標題文字
            output_path: 輸出路徑
            duration: 視頻時長（秒）
            fps: 幀率
        """
        # 加載圖片
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576))

        # 生成視頻幀
        num_frames = min(25, duration * fps)  # SVD限制

        frames = self.pipe(
            image=image,
            num_frames=num_frames,
            motion_bucket_id=100,
            fps=fps
        ).frames[0]

        # 添加文字覆蓋
        frames_with_text = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_with_text = self.add_text_overlay(
                frame_pil,
                title_text,
                position="bottom"
            )
            frames_with_text.append(frame_with_text)

        # 導出
        export_to_video(frames_with_text, output_path, fps=fps)
        print(f"Short video saved to {output_path}")

        return frames_with_text

# 使用示例
generator = ShortVideoGenerator()

generator.generate_short_video(
    image_path="product.jpg",
    title_text="新品上市！限時優惠",
    output_path="product_promo.mp4",
    duration=3,
    fps=15
)
```

### 案例2：故事視頻生成器

```python
# story_video_generator.py
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, export_to_video
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch

class StoryVideoGenerator:
    """基於故事腳本的視頻生成器"""

    def __init__(self):
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=torch.float16
        )

        self.pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to("cuda")

    def generate_scene(self, prompt, num_frames=16):
        """生成單個場景"""
        frames = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=25,
            guidance_scale=7.5
        ).frames[0]

        return frames

    def generate_story_video(
        self,
        story_script,
        output_path="story_video.mp4",
        fps=8
    ):
        """
        根據故事腳本生成視頻

        Args:
            story_script: 故事腳本列表
                          [{"prompt": "...", "duration": 2}, ...]
            output_path: 輸出路徑
            fps: 幀率
        """
        temp_clips = []

        for i, scene in enumerate(story_script):
            print(f"Generating scene {i+1}/{len(story_script)}...")

            prompt = scene["prompt"]
            duration = scene.get("duration", 2)
            num_frames = int(duration * fps)

            # 生成場景
            frames = self.generate_scene(prompt, num_frames=min(16, num_frames))

            # 保存為臨時視頻
            temp_path = f"temp_scene_{i}.mp4"
            export_to_video(frames, temp_path, fps=fps)
            temp_clips.append(temp_path)

        # 合併所有場景
        clips = [VideoFileClip(path) for path in temp_clips]
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(output_path, fps=fps)

        # 清理臨時文件
        import os
        for path in temp_clips:
            os.remove(path)

        print(f"Story video saved to {output_path}")

# 使用示例
generator = StoryVideoGenerator()

# 定義故事腳本
story = [
    {
        "prompt": "a peaceful forest at dawn, birds flying, misty atmosphere",
        "duration": 3
    },
    {
        "prompt": "a deer walking through the forest, sunlight filtering through trees",
        "duration": 3
    },
    {
        "prompt": "a crystal clear stream flowing over rocks, surrounded by green moss",
        "duration": 3
    },
    {
        "prompt": "close-up of a butterfly landing on a flower, beautiful colors",
        "duration": 2
    }
]

generator.generate_story_video(
    story_script=story,
    output_path="nature_story.mp4",
    fps=8
)
```

---

## 📚 參考資源

### 官方文檔
- [Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)
- [AnimateDiff GitHub](https://github.com/guoyww/AnimateDiff)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/stable_video_diffusion)

### 模型資源
- [Hugging Face Video Models](https://huggingface.co/models?pipeline_tag=text-to-video)
- [CivitAI Motion Modules](https://civitai.com/models?tag=motion%20module)

### 學習資源
- [Two Minute Papers - Video Generation](https://www.youtube.com/@TwoMinutePapers)
- [Stable Diffusion Art - Video Guide](https://stable-diffusion-art.com/video/)

---

## ✅ 檢查清單

完成本章節後，你應該能夠：

- [ ] 理解視頻生成的核心挑戰
- [ ] 使用Stable Video Diffusion生成流暢視頻
- [ ] 調節運動強度和視頻參數
- [ ] 使用AnimateDiff創建動畫
- [ ] 組合風格和運動LoRA
- [ ] 實現文本到視頻生成
- [ ] 進行視頻插幀和穩定化
- [ ] 添加音頻和文字覆蓋
- [ ] 構建完整的視頻生成應用

---

## 下一步

完成影片生成後，建議繼續學習：

1. **音樂生成** - 為你的視頻添加自動生成的背景音樂
2. **實戰項目** - 構建端到端的多模態內容生成系統

---

最後更新：2024-11-19
難度級別：🔴 高級
預計學習時間：12-15小時
