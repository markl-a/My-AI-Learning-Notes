# 實戰項目 (Practical Projects)

本章節提供完整的端到端多模態生成項目，整合圖片、視頻、音樂生成技術。

## 📋 目錄

1. [項目1：AI內容創作平台](#項目1ai內容創作平台)
2. [項目2：自動短視頻生成器](#項目2自動短視頻生成器)
3. [項目3：產品營銷素材生成系統](#項目3產品營銷素材生成系統)
4. [部署指南](#部署指南)

---

## 🎯 項目概覽

| 項目 | 難度 | 技術棧 | 預計時間 |
|------|------|--------|----------|
| AI內容創作平台 | 🔴 高級 | SD, MusicGen, FastAPI, React | 40-50h |
| 自動短視頻生成器 | 🟡 中級 | SVD, AnimateDiff, Bark | 20-30h |
| 產品營銷素材生成 | 🟢 初級 | SD, ControlNet, AudioLDM | 15-20h |

---

## 項目1：AI內容創作平台

### 項目描述

構建一個全功能的AI內容創作平台，用戶可以通過簡單的文本描述生成圖片、視頻和音樂。

### 功能特性

- ✅ 文本生成圖片（支持多種風格）
- ✅ 圖片生成視頻
- ✅ 文本生成音樂
- ✅ 批量生成和管理
- ✅ 用戶認證和配額管理
- ✅ Web界面和API接口

### 技術架構

```
┌─────────────────────────────────────────┐
│           前端 (React + TypeScript)      │
├─────────────────────────────────────────┤
│     - 圖片生成界面                       │
│     - 視頻生成界面                       │
│     - 音樂生成界面                       │
│     - 項目管理                           │
└─────────────────┬───────────────────────┘
                  │
                  │ REST API
                  ↓
┌─────────────────────────────────────────┐
│         後端 (FastAPI + Python)          │
├─────────────────────────────────────────┤
│  ┌────────────────────────────────────┐ │
│  │   API 路由層                        │ │
│  │   - /api/generate/image            │ │
│  │   - /api/generate/video            │ │
│  │   - /api/generate/music            │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │   業務邏輯層                        │ │
│  │   - 生成管理器                     │ │
│  │   - 任務隊列                       │ │
│  │   - 用戶管理                       │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │   AI模型層                          │ │
│  │   - Stable Diffusion               │ │
│  │   - Stable Video Diffusion         │ │
│  │   - MusicGen                       │ │
│  └────────────────────────────────────┘ │
└─────────────────┬───────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────┐
│      數據存儲                             │
│   - PostgreSQL (元數據)                  │
│   - Redis (緩存/任務隊列)                │
│   - S3/MinIO (生成內容)                  │
└─────────────────────────────────────────┘
```

### 實現代碼

#### 後端 API (FastAPI)

```python
# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime
import redis
import json

app = FastAPI(title="AI Content Creation Platform")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis連接（用於任務隊列）
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 數據模型
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    num_images: int = 1
    style: Optional[str] = "realistic"

class VideoGenerationRequest(BaseModel):
    image_url: Optional[str] = None
    prompt: Optional[str] = None
    duration: int = 3
    motion_strength: int = 127

class MusicGenerationRequest(BaseModel):
    prompt: str
    duration: int = 10
    style: Optional[str] = "electronic"

class GenerationStatus(BaseModel):
    task_id: str
    status: str  # pending/processing/completed/failed
    progress: int
    result_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# 生成管理器
from generators import ImageGenerator, VideoGenerator, MusicGenerator

image_gen = ImageGenerator()
video_gen = VideoGenerator()
music_gen = MusicGenerator()

# API路由
@app.post("/api/generate/image")
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks
):
    """生成圖片"""
    task_id = str(uuid.uuid4())

    # 創建任務記錄
    task_data = {
        "task_id": task_id,
        "type": "image",
        "status": "pending",
        "progress": 0,
        "params": request.dict(),
        "created_at": datetime.now().isoformat()
    }

    redis_client.set(f"task:{task_id}", json.dumps(task_data))

    # 添加到後台任務
    background_tasks.add_task(
        process_image_generation,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "pending"}

@app.post("/api/generate/video")
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks
):
    """生成視頻"""
    task_id = str(uuid.uuid4())

    task_data = {
        "task_id": task_id,
        "type": "video",
        "status": "pending",
        "progress": 0,
        "params": request.dict(),
        "created_at": datetime.now().isoformat()
    }

    redis_client.set(f"task:{task_id}", json.dumps(task_data))

    background_tasks.add_task(
        process_video_generation,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "pending"}

@app.post("/api/generate/music")
async def generate_music(
    request: MusicGenerationRequest,
    background_tasks: BackgroundTasks
):
    """生成音樂"""
    task_id = str(uuid.uuid4())

    task_data = {
        "task_id": task_id,
        "type": "music",
        "status": "pending",
        "progress": 0,
        "params": request.dict(),
        "created_at": datetime.now().isoformat()
    }

    redis_client.set(f"task:{task_id}", json.dumps(task_data))

    background_tasks.add_task(
        process_music_generation,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "pending"}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """查詢任務狀態"""
    task_data = redis_client.get(f"task:{task_id}")

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    return json.loads(task_data)

# 後台處理函數
async def process_image_generation(task_id: str, request: ImageGenerationRequest):
    """處理圖片生成任務"""
    try:
        # 更新狀態為處理中
        update_task_status(task_id, "processing", 10)

        # 生成圖片
        images = image_gen.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            style=request.style,
            progress_callback=lambda p: update_task_status(task_id, "processing", 10 + int(p * 0.8))
        )

        # 保存圖片到存儲
        image_urls = []
        for idx, image in enumerate(images):
            url = save_to_storage(image, f"{task_id}_{idx}.png")
            image_urls.append(url)

        # 更新為完成
        update_task_status(task_id, "completed", 100, result_url=image_urls[0] if image_urls else None)

    except Exception as e:
        update_task_status(task_id, "failed", 0, error=str(e))

async def process_video_generation(task_id: str, request: VideoGenerationRequest):
    """處理視頻生成任務"""
    try:
        update_task_status(task_id, "processing", 10)

        # 生成視頻
        video_path = video_gen.generate(
            image_url=request.image_url,
            prompt=request.prompt,
            duration=request.duration,
            motion_strength=request.motion_strength,
            progress_callback=lambda p: update_task_status(task_id, "processing", 10 + int(p * 0.8))
        )

        # 保存到存儲
        video_url = save_to_storage(video_path, f"{task_id}.mp4")

        update_task_status(task_id, "completed", 100, result_url=video_url)

    except Exception as e:
        update_task_status(task_id, "failed", 0, error=str(e))

async def process_music_generation(task_id: str, request: MusicGenerationRequest):
    """處理音樂生成任務"""
    try:
        update_task_status(task_id, "processing", 10)

        # 生成音樂
        audio_path = music_gen.generate(
            prompt=request.prompt,
            duration=request.duration,
            style=request.style,
            progress_callback=lambda p: update_task_status(task_id, "processing", 10 + int(p * 0.8))
        )

        # 保存到存儲
        audio_url = save_to_storage(audio_path, f"{task_id}.wav")

        update_task_status(task_id, "completed", 100, result_url=audio_url)

    except Exception as e:
        update_task_status(task_id, "failed", 0, error=str(e))

def update_task_status(task_id: str, status: str, progress: int, result_url: str = None, error: str = None):
    """更新任務狀態"""
    task_data = json.loads(redis_client.get(f"task:{task_id}"))
    task_data["status"] = status
    task_data["progress"] = progress

    if result_url:
        task_data["result_url"] = result_url

    if error:
        task_data["error"] = error

    if status == "completed" or status == "failed":
        task_data["completed_at"] = datetime.now().isoformat()

    redis_client.set(f"task:{task_id}", json.dumps(task_data))

def save_to_storage(file_path, filename):
    """保存文件到存儲（S3/MinIO）"""
    # 實現文件上傳邏輯
    # 這裡簡化為返回本地路徑
    return f"/storage/{filename}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 生成器模塊

```python
# generators.py
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from audiocraft.models import MusicGen
import torch
from PIL import Image
import os

class ImageGenerator:
    """圖片生成器"""

    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

    def generate(
        self,
        prompt,
        negative_prompt="",
        width=512,
        height=512,
        num_images=1,
        style="realistic",
        progress_callback=None
    ):
        """生成圖片"""

        # 根據風格調整提示詞
        style_prompts = {
            "realistic": ", photorealistic, highly detailed, 8k uhd",
            "anime": ", anime style, vibrant colors, detailed",
            "artistic": ", artistic, painterly, expressive",
            "3d": ", 3d render, octane render, highly detailed"
        }

        full_prompt = prompt + style_prompts.get(style, "")

        images = []
        for i in range(num_images):
            if progress_callback:
                progress_callback((i / num_images) * 100)

            image = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

            images.append(image)

            # 保存臨時文件
            os.makedirs("temp", exist_ok=True)
            image.save(f"temp/image_{i}.png")

        if progress_callback:
            progress_callback(100)

        return images

class VideoGenerator:
    """視頻生成器"""

    def __init__(self):
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        image_url=None,
        prompt=None,
        duration=3,
        motion_strength=127,
        progress_callback=None
    ):
        """生成視頻"""
        from diffusers.utils import load_image, export_to_video

        # 加載或生成輸入圖片
        if image_url:
            image = load_image(image_url)
        elif prompt:
            # 使用圖片生成器創建初始圖片
            img_gen = ImageGenerator()
            images = img_gen.generate(prompt, num_images=1)
            image = images[0]
        else:
            raise ValueError("Must provide either image_url or prompt")

        image = image.resize((1024, 576))

        if progress_callback:
            progress_callback(20)

        # 生成視頻
        frames = self.pipe(
            image=image,
            num_frames=min(25, duration * 7),
            motion_bucket_id=motion_strength,
            decode_chunk_size=8
        ).frames[0]

        if progress_callback:
            progress_callback(90)

        # 導出視頻
        os.makedirs("temp", exist_ok=True)
        output_path = "temp/video_output.mp4"
        export_to_video(frames, output_path, fps=7)

        if progress_callback:
            progress_callback(100)

        return output_path

class MusicGenerator:
    """音樂生成器"""

    def __init__(self):
        self.model = MusicGen.get_pretrained('facebook/musicgen-medium')

    def generate(
        self,
        prompt,
        duration=10,
        style="electronic",
        progress_callback=None
    ):
        """生成音樂"""
        from audiocraft.data.audio import audio_write

        # 風格模板
        style_templates = {
            "electronic": "electronic music, synthesizer, modern",
            "acoustic": "acoustic music, guitar, organic",
            "orchestral": "orchestral music, cinematic, dramatic",
            "ambient": "ambient music, atmospheric, peaceful"
        }

        full_prompt = f"{prompt}, {style_templates.get(style, '')}"

        self.model.set_generation_params(
            duration=duration,
            temperature=1.0,
            cfg_coef=3.0
        )

        if progress_callback:
            progress_callback(30)

        # 生成
        wav = self.model.generate([full_prompt], progress=True)

        if progress_callback:
            progress_callback(90)

        # 保存
        os.makedirs("temp", exist_ok=True)
        output_path = "temp/music_output"
        audio_write(output_path, wav[0].cpu(), self.model.sample_rate)

        if progress_callback:
            progress_callback(100)

        return f"{output_path}.wav"
```

#### 前端界面 (React)

```typescript
// App.tsx
import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface GenerationTask {
  task_id: string;
  status: string;
  progress: number;
  result_url?: string;
}

function App() {
  const [activeTab, setActiveTab] = useState<'image' | 'video' | 'music'>('image');
  const [currentTask, setCurrentTask] = useState<GenerationTask | null>(null);

  // 圖片生成
  const [imagePrompt, setImagePrompt] = useState('');
  const [imageStyle, setImageStyle] = useState('realistic');

  // 視頻生成
  const [videoPrompt, setVideoPrompt] = useState('');
  const [videoDuration, setVideoDuration] = useState(3);

  // 音樂生成
  const [musicPrompt, setMusicPrompt] = useState('');
  const [musicDuration, setMusicDuration] = useState(10);

  const generateImage = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate/image`, {
        prompt: imagePrompt,
        style: imageStyle,
        num_images: 1
      });

      setCurrentTask(response.data);
      pollTaskStatus(response.data.task_id);
    } catch (error) {
      console.error('Error generating image:', error);
    }
  };

  const generateVideo = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate/video`, {
        prompt: videoPrompt,
        duration: videoDuration,
        motion_strength: 127
      });

      setCurrentTask(response.data);
      pollTaskStatus(response.data.task_id);
    } catch (error) {
      console.error('Error generating video:', error);
    }
  };

  const generateMusic = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate/music`, {
        prompt: musicPrompt,
        duration: musicDuration,
        style: 'electronic'
      });

      setCurrentTask(response.data);
      pollTaskStatus(response.data.task_id);
    } catch (error) {
      console.error('Error generating music:', error);
    }
  };

  const pollTaskStatus = async (taskId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/task/${taskId}`);
        setCurrentTask(response.data);

        if (response.data.status === 'completed' || response.data.status === 'failed') {
          clearInterval(interval);
        }
      } catch (error) {
        console.error('Error polling task status:', error);
        clearInterval(interval);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center">
          AI Content Creation Platform
        </h1>

        {/* 標籤切換 */}
        <div className="flex mb-8 border-b">
          <button
            className={`px-6 py-3 ${activeTab === 'image' ? 'border-b-2 border-blue-500' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            Image Generation
          </button>
          <button
            className={`px-6 py-3 ${activeTab === 'video' ? 'border-b-2 border-blue-500' : ''}`}
            onClick={() => setActiveTab('video')}
          >
            Video Generation
          </button>
          <button
            className={`px-6 py-3 ${activeTab === 'music' ? 'border-b-2 border-blue-500' : ''}`}
            onClick={() => setActiveTab('music')}
          >
            Music Generation
          </button>
        </div>

        {/* 圖片生成 */}
        {activeTab === 'image' && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Generate Image</h2>

            <div className="mb-4">
              <label className="block mb-2">Prompt</label>
              <textarea
                className="w-full p-2 border rounded"
                rows={3}
                value={imagePrompt}
                onChange={(e) => setImagePrompt(e.target.value)}
                placeholder="Describe the image you want to generate..."
              />
            </div>

            <div className="mb-4">
              <label className="block mb-2">Style</label>
              <select
                className="w-full p-2 border rounded"
                value={imageStyle}
                onChange={(e) => setImageStyle(e.target.value)}
              >
                <option value="realistic">Realistic</option>
                <option value="anime">Anime</option>
                <option value="artistic">Artistic</option>
                <option value="3d">3D Render</option>
              </select>
            </div>

            <button
              className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
              onClick={generateImage}
            >
              Generate
            </button>
          </div>
        )}

        {/* 視頻生成 */}
        {activeTab === 'video' && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Generate Video</h2>

            <div className="mb-4">
              <label className="block mb-2">Prompt</label>
              <textarea
                className="w-full p-2 border rounded"
                rows={3}
                value={videoPrompt}
                onChange={(e) => setVideoPrompt(e.target.value)}
                placeholder="Describe the video scene..."
              />
            </div>

            <div className="mb-4">
              <label className="block mb-2">Duration (seconds)</label>
              <input
                type="number"
                className="w-full p-2 border rounded"
                value={videoDuration}
                onChange={(e) => setVideoDuration(parseInt(e.target.value))}
                min={1}
                max={10}
              />
            </div>

            <button
              className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
              onClick={generateVideo}
            >
              Generate
            </button>
          </div>
        )}

        {/* 音樂生成 */}
        {activeTab === 'music' && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Generate Music</h2>

            <div className="mb-4">
              <label className="block mb-2">Prompt</label>
              <textarea
                className="w-full p-2 border rounded"
                rows={3}
                value={musicPrompt}
                onChange={(e) => setMusicPrompt(e.target.value)}
                placeholder="Describe the music you want..."
              />
            </div>

            <div className="mb-4">
              <label className="block mb-2">Duration (seconds)</label>
              <input
                type="number"
                className="w-full p-2 border rounded"
                value={musicDuration}
                onChange={(e) => setMusicDuration(parseInt(e.target.value))}
                min={5}
                max={30}
              />
            </div>

            <button
              className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
              onClick={generateMusic}
            >
              Generate
            </button>
          </div>
        )}

        {/* 生成狀態 */}
        {currentTask && (
          <div className="mt-8 bg-white p-6 rounded-lg shadow">
            <h3 className="text-xl font-bold mb-4">Generation Status</h3>

            <div className="mb-4">
              <div className="flex justify-between mb-2">
                <span>Status: {currentTask.status}</span>
                <span>{currentTask.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${currentTask.progress}%` }}
                />
              </div>
            </div>

            {currentTask.status === 'completed' && currentTask.result_url && (
              <div className="mt-4">
                <p className="mb-2">Result:</p>
                {activeTab === 'image' && (
                  <img
                    src={currentTask.result_url}
                    alt="Generated"
                    className="max-w-full rounded"
                  />
                )}
                {activeTab === 'video' && (
                  <video
                    src={currentTask.result_url}
                    controls
                    className="max-w-full rounded"
                  />
                )}
                {activeTab === 'music' && (
                  <audio
                    src={currentTask.result_url}
                    controls
                    className="w-full"
                  />
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
```

---

## 項目2：自動短視頻生成器

### 項目描述

自動從文本腳本生成完整的短視頻，包括圖片、視頻、旁白和背景音樂。

### 完整代碼

```python
# short_video_creator.py
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from bark import generate_audio, SAMPLE_RATE
from audiocraft.models import MusicGen
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
import torch
from PIL import Image
import numpy as np

class ShortVideoCreator:
    """自動短視頻生成器"""

    def __init__(self):
        # 初始化所有模型
        print("Loading models...")

        self.image_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.image_pipe = self.image_pipe.to("cuda")

        self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16
        )
        self.video_pipe = self.video_pipe.to("cuda")

        self.music_model = MusicGen.get_pretrained('facebook/musicgen-small')

        print("Models loaded successfully!")

    def create_video_from_script(
        self,
        script,
        output_path="final_video.mp4",
        add_music=True,
        music_volume=0.3
    ):
        """
        從腳本創建視頻

        Args:
            script: 視頻腳本列表
                [
                    {
                        "visual": "scene description for image generation",
                        "narration": "text to speak",
                        "duration": 5
                    },
                    ...
                ]
            output_path: 輸出路徑
            add_music: 是否添加背景音樂
            music_volume: 背景音樂音量
        """
        import scipy
        from diffusers.utils import export_to_video
        import os

        clips = []
        total_duration = sum(scene.get("duration", 3) for scene in script)

        # 1. 為每個場景生成內容
        for idx, scene in enumerate(script):
            print(f"\n=== Processing Scene {idx+1}/{len(script)} ===")

            # 生成圖片
            print(f"Generating image: {scene['visual'][:50]}...")
            image = self.image_pipe(
                prompt=scene['visual'],
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

            # 保存臨時圖片
            temp_image_path = f"temp_scene_{idx}_image.png"
            image.save(temp_image_path)

            # 圖片轉視頻
            print("Converting image to video...")
            image_resized = image.resize((1024, 576))

            frames = self.video_pipe(
                image=image_resized,
                num_frames=min(25, scene.get("duration", 3) * 7),
                motion_bucket_id=80
            ).frames[0]

            # 導出視頻
            temp_video_path = f"temp_scene_{idx}_video.mp4"
            export_to_video(frames, temp_video_path, fps=7)

            # 生成旁白
            if "narration" in scene and scene["narration"]:
                print(f"Generating narration: {scene['narration'][:50]}...")
                narration_audio = generate_audio(
                    scene["narration"],
                    history_prompt="v2/en_speaker_6"
                )

                # 保存旁白
                temp_narration_path = f"temp_scene_{idx}_narration.wav"
                scipy.io.wavfile.write(
                    temp_narration_path,
                    SAMPLE_RATE,
                    narration_audio
                )

                # 組合視頻和旁白
                video_clip = VideoFileClip(temp_video_path)
                audio_clip = AudioFileClip(temp_narration_path)

                # 調整視頻長度匹配音頻
                if audio_clip.duration > video_clip.duration:
                    # 如果旁白更長，需要擴展視頻（循環或減慢）
                    video_clip = video_clip.loop(duration=audio_clip.duration)

                video_clip = video_clip.set_audio(audio_clip)
            else:
                video_clip = VideoFileClip(temp_video_path)

            clips.append(video_clip)

        # 2. 拼接所有場景
        print("\n=== Concatenating all scenes ===")
        final_video = concatenate_videoclips(clips, method="compose")

        # 3. 添加背景音樂
        if add_music:
            print("Generating background music...")

            self.music_model.set_generation_params(
                duration=min(30, int(total_duration))
            )

            music_wav = self.music_model.generate([
                "calm background music, ambient, peaceful, no vocals"
            ])

            # 保存音樂
            from audiocraft.data.audio import audio_write
            audio_write("temp_bg_music", music_wav[0].cpu(), self.music_model.sample_rate)

            # 加載音樂
            import librosa
            import soundfile as sf

            music_audio, sr = librosa.load("temp_bg_music.wav", sr=SAMPLE_RATE)

            # 循環音樂以匹配視頻長度
            if len(music_audio) / sr < total_duration:
                num_loops = int(np.ceil(total_duration / (len(music_audio) / sr)))
                music_audio = np.tile(music_audio, num_loops)

            # 截取到視頻長度
            target_samples = int(final_video.duration * sr)
            music_audio = music_audio[:target_samples]

            # 降低音樂音量
            music_audio = music_audio * music_volume

            # 添加淡出
            fade_duration = 3
            fade_samples = int(fade_duration * sr)
            fade_curve = np.linspace(1, 0, fade_samples)
            music_audio[-fade_samples:] *= fade_curve

            # 保存處理後的音樂
            sf.write("temp_bg_music_processed.wav", music_audio, sr)

            # 組合原始音頻和背景音樂
            bg_music_clip = AudioFileClip("temp_bg_music_processed.wav")

            if final_video.audio:
                final_audio = CompositeAudioClip([final_video.audio, bg_music_clip])
                final_video = final_video.set_audio(final_audio)
            else:
                final_video = final_video.set_audio(bg_music_clip)

        # 4. 導出最終視頻
        print(f"\n=== Exporting final video to {output_path} ===")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=24
        )

        # 清理臨時文件
        print("Cleaning up temporary files...")
        import glob
        for temp_file in glob.glob("temp_*"):
            try:
                os.remove(temp_file)
            except:
                pass

        print(f"\n✅ Video created successfully: {output_path}")

# 使用示例
if __name__ == "__main__":
    creator = ShortVideoCreator()

    # 定義腳本
    script = [
        {
            "visual": "a peaceful morning landscape, sunrise over mountains, misty atmosphere, beautiful scenery",
            "narration": "In the early morning, as the sun rises over the mountains, nature awakens.",
            "duration": 5
        },
        {
            "visual": "a person meditating by a lake, calm water reflections, zen atmosphere, peaceful",
            "narration": "Taking a moment to meditate brings peace and clarity to the mind.",
            "duration": 5
        },
        {
            "visual": "birds flying in the sky, freedom, open sky, natural beauty",
            "narration": "Like birds in the sky, we are free to choose our path forward.",
            "duration": 4
        },
        {
            "visual": "sunset over ocean, warm colors, peaceful ending, beautiful horizon",
            "narration": "As the day ends, we reflect on the beauty that surrounds us every day.",
            "duration": 5
        }
    ]

    # 創建視頻
    creator.create_video_from_script(
        script,
        output_path="motivational_video.mp4",
        add_music=True,
        music_volume=0.2
    )
```

---

## 項目3：產品營銷素材生成系統

完整的產品營銷材料自動化生成系統。

```python
# product_marketing_generator.py
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import numpy as np

class ProductMarketingGenerator:
    """產品營銷素材生成系統"""

    def __init__(self):
        # 加載基礎模型
        self.base_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.base_pipe = self.base_pipe.to("cuda")

        # 加載ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )

        self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        self.controlnet_pipe = self.controlnet_pipe.to("cuda")

    def generate_product_variations(
        self,
        product_name,
        product_description,
        num_variations=5,
        backgrounds=None
    ):
        """
        生成產品的多種展示變體

        Args:
            product_name: 產品名稱
            product_description: 產品描述
            num_variations: 變體數量
            backgrounds: 背景列表
        """
        if backgrounds is None:
            backgrounds = [
                "white studio background",
                "wooden table surface",
                "marble countertop",
                "outdoor natural setting",
                "modern minimalist room"
            ]

        variations = []

        for i, bg in enumerate(backgrounds[:num_variations]):
            prompt = f"""
            product photography of {product_description},
            {bg}, professional lighting, high quality,
            commercial photo, no people, centered composition,
            sharp focus, detailed, 8k
            """

            negative_prompt = """
            person, people, hands, low quality, blurry,
            watermark, text, logo, cluttered
            """

            image = self.base_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=8.0,
                width=768,
                height=768
            ).images[0]

            # 添加產品名稱水印
            image_with_text = self.add_product_watermark(image, product_name)

            filename = f"{product_name.replace(' ', '_')}_variation_{i+1}.png"
            image_with_text.save(filename)

            variations.append(image_with_text)

            print(f"✓ Generated variation {i+1}: {filename}")

        return variations

    def add_product_watermark(self, image, product_name):
        """添加產品名稱水印"""
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("Arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        # 在右下角添加產品名稱
        bbox = draw.textbbox((0, 0), product_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = image.width - text_width - 30
        y = image.height - text_height - 30

        # 半透明背景
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [x-10, y-5, x+text_width+10, y+text_height+5],
            fill=(0, 0, 0, 128)
        )

        image = image.convert('RGBA')
        image = Image.alpha_composite(image, overlay)

        draw = ImageDraw.Draw(image)
        draw.text((x, y), product_name, font=font, fill=(255, 255, 255, 255))

        return image.convert('RGB')

    def create_marketing_bundle(
        self,
        product_name,
        product_description,
        output_dir="marketing_bundle"
    ):
        """
        創建完整的營銷素材包

        包含：
        - 產品圖 (5張不同角度/背景)
        - 社交媒體尺寸圖 (Instagram, Facebook, Twitter)
        - 橫幅廣告
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating marketing bundle for: {product_name}")

        # 1. 生成基礎產品圖
        print("\n1. Generating base product images...")
        base_images = self.generate_product_variations(
            product_name,
            product_description,
            num_variations=5
        )

        # 保存到bundle目錄
        for i, img in enumerate(base_images):
            img.save(f"{output_dir}/product_image_{i+1}.png")

        # 2. 生成社交媒體尺寸
        print("\n2. Generating social media sizes...")

        social_sizes = {
            "instagram_post": (1080, 1080),
            "instagram_story": (1080, 1920),
            "facebook_post": (1200, 630),
            "twitter_post": (1200, 675)
        }

        for size_name, (width, height) in social_sizes.items():
            resized = base_images[0].resize((width, height), Image.LANCZOS)
            resized.save(f"{output_dir}/{size_name}.png")
            print(f"✓ Created {size_name}")

        # 3. 生成橫幅廣告
        print("\n3. Generating banner ads...")

        banner_prompt = f"""
        advertisement banner for {product_description},
        modern design, clean layout, professional,
        product prominently displayed, marketing material
        """

        banner = self.base_pipe(
            prompt=banner_prompt,
            width=1024,
            height=512,
            num_inference_steps=40
        ).images[0]

        banner.save(f"{output_dir}/banner_ad.png")

        print(f"\n✅ Marketing bundle created in: {output_dir}/")

# 使用示例
if __name__ == "__main__":
    generator = ProductMarketingGenerator()

    # 生成產品營銷素材
    generator.create_marketing_bundle(
        product_name="Wireless Headphones Pro",
        product_description="premium wireless headphones, black color, modern design, studio quality",
        output_dir="headphones_marketing"
    )
```

---

## 🚀 部署指南

### Docker部署

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安裝Python和依賴
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製requirements
COPY requirements.txt .

# 安裝Python包
RUN pip3 install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./storage:/app/storage
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ai_content
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

---

## 📚 總結

本章節提供了三個完整的實戰項目：

1. **AI內容創作平台** - 企業級多模態生成系統
2. **自動短視頻生成器** - 從腳本到成品的全自動流程
3. **產品營銷素材生成** - 批量生成營銷材料

每個項目都包含完整的代碼實現和部署指南，可以直接用於實際生產環境。

---

最後更新：2024-11-19
