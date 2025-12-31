# 語音與音訊 AI (Voice and Audio AI)

## 概述

語音 AI 在 2025 年已成為人機互動的重要介面。從語音助手到即時翻譯，語音技術正在改變我們與 AI 互動的方式。

## 語音轉文字 (Speech-to-Text)

### OpenAI Whisper API

```python
from openai import OpenAI
from pathlib import Path
import tempfile

class WhisperTranscriber:
    """Whisper 語音轉文字"""

    def __init__(self):
        self.client = OpenAI()

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        prompt: str = None,
        response_format: str = "json"  # json, text, srt, vtt, verbose_json
    ) -> dict:
        """轉錄音訊檔案"""
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                prompt=prompt,
                response_format=response_format
            )

        if response_format == "json":
            return {"text": response.text}
        elif response_format == "verbose_json":
            return {
                "text": response.text,
                "segments": response.segments,
                "language": response.language,
                "duration": response.duration
            }
        else:
            return {"text": response}

    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: str = None
    ) -> list[dict]:
        """帶時間戳的轉錄"""
        result = self.transcribe(
            audio_path,
            language=language,
            response_format="verbose_json"
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })

        return segments

    def translate(self, audio_path: str) -> str:
        """翻譯音訊為英文"""
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.translations.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text

# 使用範例
transcriber = WhisperTranscriber()

# 基本轉錄
result = transcriber.transcribe("meeting.mp3", language="zh")
print(result["text"])

# 帶時間戳
segments = transcriber.transcribe_with_timestamps("podcast.mp3")
for seg in segments:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
```

### 本地 Whisper 模型

```python
import whisper
import torch
from typing import Optional

class LocalWhisperTranscriber:
    """本地 Whisper 模型"""

    def __init__(
        self,
        model_size: str = "base",  # tiny, base, small, medium, large
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisper.load_model(model_size, device=device)
        self.device = device

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"  # transcribe, translate
    ) -> dict:
        """轉錄音訊"""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False
        )

        return {
            "text": result["text"],
            "language": result.get("language"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result["segments"]
            ]
        }

    def detect_language(self, audio_path: str) -> str:
        """偵測語言"""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)

        return max(probs, key=probs.get)

# 使用範例
local_transcriber = LocalWhisperTranscriber("medium")
result = local_transcriber.transcribe("audio.wav", language="zh")
```

### 即時語音轉錄

```python
import sounddevice as sd
import numpy as np
import queue
import threading
from openai import OpenAI
import tempfile
import wave

class RealtimeTranscriber:
    """即時語音轉錄"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 5.0  # 每段音訊長度（秒）
    ):
        self.client = OpenAI()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcripts = []

    def _audio_callback(self, indata, frames, time, status):
        """音訊回調"""
        if status:
            print(f"音訊狀態: {status}")
        self.audio_queue.put(indata.copy())

    def _save_audio_chunk(self, audio_data: np.ndarray) -> str:
        """儲存音訊片段"""
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.sample_rate)
                wav.writeframes(
                    (audio_data * 32767).astype(np.int16).tobytes()
                )
            return f.name

    def _transcribe_worker(self):
        """轉錄工作執行緒"""
        buffer = []

        while self.is_recording or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                buffer.extend(chunk.flatten())

                if len(buffer) >= self.chunk_size:
                    # 處理音訊
                    audio_data = np.array(buffer[:self.chunk_size])
                    buffer = buffer[self.chunk_size:]

                    # 儲存並轉錄
                    audio_path = self._save_audio_chunk(audio_data)

                    try:
                        with open(audio_path, "rb") as f:
                            response = self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                language="zh"
                            )

                        if response.text.strip():
                            self.transcripts.append(response.text)
                            print(f"[轉錄] {response.text}")
                    finally:
                        Path(audio_path).unlink(missing_ok=True)

            except queue.Empty:
                continue

    def start(self, duration: float = None):
        """開始錄音和轉錄"""
        self.is_recording = True
        self.transcripts = []

        # 啟動轉錄執行緒
        transcribe_thread = threading.Thread(target=self._transcribe_worker)
        transcribe_thread.start()

        # 開始錄音
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback
        ):
            print("開始錄音... (按 Ctrl+C 停止)")
            try:
                if duration:
                    sd.sleep(int(duration * 1000))
                else:
                    while True:
                        sd.sleep(100)
            except KeyboardInterrupt:
                pass

        self.is_recording = False
        transcribe_thread.join()

        return " ".join(self.transcripts)

# 使用範例
realtime = RealtimeTranscriber(chunk_duration=3.0)
transcript = realtime.start(duration=30)  # 錄製 30 秒
print(f"\n完整轉錄: {transcript}")
```

## 文字轉語音 (Text-to-Speech)

### OpenAI TTS

```python
from openai import OpenAI
from pathlib import Path

class OpenAITTS:
    """OpenAI 文字轉語音"""

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(self):
        self.client = OpenAI()

    def speak(
        self,
        text: str,
        output_path: str,
        voice: str = "alloy",
        model: str = "tts-1",  # tts-1, tts-1-hd
        speed: float = 1.0  # 0.25 to 4.0
    ) -> str:
        """生成語音"""
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed
        )

        response.stream_to_file(output_path)
        return output_path

    def speak_streaming(
        self,
        text: str,
        voice: str = "alloy"
    ):
        """串流語音生成"""
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        for chunk in response.iter_bytes(chunk_size=4096):
            yield chunk

    def generate_all_voices(
        self,
        text: str,
        output_dir: str
    ) -> list[str]:
        """用所有聲音生成"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = []
        for voice in self.VOICES:
            file_path = output_path / f"{voice}.mp3"
            self.speak(text, str(file_path), voice=voice)
            files.append(str(file_path))
            print(f"已生成: {voice}")

        return files

# 使用範例
tts = OpenAITTS()

# 基本使用
tts.speak(
    "你好，歡迎使用語音合成服務。",
    "output.mp3",
    voice="nova"
)

# HD 品質
tts.speak(
    "這是高品質語音輸出。",
    "output_hd.mp3",
    voice="alloy",
    model="tts-1-hd"
)
```

### Edge TTS（免費替代方案）

```python
import edge_tts
import asyncio
from typing import Optional

class EdgeTTS:
    """Edge TTS 免費語音合成"""

    # 中文語音
    CHINESE_VOICES = {
        "zh-TW-HsiaoChenNeural": "台灣女聲",
        "zh-TW-YunJheNeural": "台灣男聲",
        "zh-CN-XiaoxiaoNeural": "中國女聲",
        "zh-CN-YunxiNeural": "中國男聲"
    }

    @staticmethod
    async def speak_async(
        text: str,
        output_path: str,
        voice: str = "zh-TW-HsiaoChenNeural",
        rate: str = "+0%",  # -50% to +100%
        pitch: str = "+0Hz"  # -50Hz to +50Hz
    ) -> str:
        """異步語音生成"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch
        )
        await communicate.save(output_path)
        return output_path

    @classmethod
    def speak(
        cls,
        text: str,
        output_path: str,
        voice: str = "zh-TW-HsiaoChenNeural",
        rate: str = "+0%"
    ) -> str:
        """同步語音生成"""
        return asyncio.run(
            cls.speak_async(text, output_path, voice, rate)
        )

    @staticmethod
    async def list_voices(language: str = "zh") -> list[dict]:
        """列出可用語音"""
        voices = await edge_tts.list_voices()
        return [
            v for v in voices
            if v["Locale"].startswith(language)
        ]

# 使用範例
edge = EdgeTTS()

# 生成語音
edge.speak(
    "這是使用 Edge TTS 生成的語音。",
    "edge_output.mp3",
    voice="zh-TW-HsiaoChenNeural"
)

# 調整語速
edge.speak(
    "這是加快語速的語音。",
    "fast_output.mp3",
    rate="+20%"
)
```

## 即時語音對話

### 語音對話系統

```python
from openai import OpenAI
import sounddevice as sd
import numpy as np
import queue
import tempfile
import wave
from pathlib import Path
import threading
import pygame

class VoiceConversation:
    """語音對話系統"""

    def __init__(
        self,
        system_prompt: str = "你是一個友善的語音助手。請用簡短的句子回答。",
        voice: str = "nova"
    ):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.voice = voice
        self.conversation_history = []

        # 音訊設定
        self.sample_rate = 16000
        self.channels = 1

        # 初始化 pygame 用於播放
        pygame.mixer.init()

    def _record_audio(
        self,
        duration: float = 5.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5
    ) -> np.ndarray:
        """錄製音訊（帶靜音檢測）"""
        print("🎤 正在聽...")

        audio_data = []
        silence_samples = 0
        max_silence = int(silence_duration * self.sample_rate)

        def callback(indata, frames, time, status):
            nonlocal silence_samples
            audio_data.extend(indata[:, 0])

            # 靜音檢測
            volume = np.abs(indata).mean()
            if volume < silence_threshold:
                silence_samples += frames
            else:
                silence_samples = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback
        ):
            start_time = sd.sleep(int(duration * 1000))

            # 等待說話開始或達到最大時間
            while len(audio_data) < duration * self.sample_rate:
                if silence_samples > max_silence and len(audio_data) > self.sample_rate:
                    break
                sd.sleep(100)

        return np.array(audio_data)

    def _audio_to_text(self, audio_data: np.ndarray) -> str:
        """音訊轉文字"""
        # 儲存臨時檔案
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(
                    (audio_data * 32767).astype(np.int16).tobytes()
                )
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="zh"
                )
            return response.text
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _get_response(self, user_message: str) -> str:
        """取得 AI 回應"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history
            ],
            max_tokens=150  # 保持回應簡短
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def _text_to_speech(self, text: str) -> str:
        """文字轉語音"""
        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as f:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text
            )
            response.stream_to_file(f.name)
            return f.name

    def _play_audio(self, audio_path: str):
        """播放音訊"""
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        Path(audio_path).unlink(missing_ok=True)

    def chat(self, text_input: bool = False) -> str:
        """進行一輪對話"""
        # 取得使用者輸入
        if text_input:
            user_message = input("你: ")
        else:
            audio = self._record_audio()
            user_message = self._audio_to_text(audio)
            print(f"你: {user_message}")

        if not user_message.strip():
            return ""

        # 取得回應
        print("🤔 思考中...")
        response = self._get_response(user_message)
        print(f"AI: {response}")

        # 播放語音
        print("🔊 播放中...")
        audio_path = self._text_to_speech(response)
        self._play_audio(audio_path)

        return response

    def start_conversation(self, max_turns: int = 10):
        """開始對話"""
        print("=" * 50)
        print("語音對話已啟動！說「結束」來停止。")
        print("=" * 50)

        for _ in range(max_turns):
            response = self.chat()

            if "結束" in response or "再見" in response:
                print("對話結束。再見！")
                break

# 使用範例
conversation = VoiceConversation(
    system_prompt="你是一個台灣的 AI 助手。請用繁體中文簡短回答。",
    voice="nova"
)

# 開始對話
conversation.start_conversation()
```

## 音訊 RAG 系統

### 音訊內容索引與搜尋

```python
from dataclasses import dataclass
from typing import Optional
import chromadb
from openai import OpenAI
import hashlib
from pathlib import Path

@dataclass
class AudioDocument:
    """音訊文件"""
    id: str
    path: str
    transcript: str
    duration: float
    segments: list[dict]
    metadata: dict

class AudioRAG:
    """音訊 RAG 系統"""

    def __init__(
        self,
        collection_name: str = "audio_rag",
        persist_dir: str = "./chroma_audio"
    ):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_id(self, audio_path: str) -> str:
        """生成音訊 ID"""
        with open(audio_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _transcribe(self, audio_path: str) -> dict:
        """轉錄音訊"""
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )

        return {
            "text": response.text,
            "duration": response.duration,
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in response.segments
            ]
        }

    def _get_embedding(self, text: str) -> list[float]:
        """取得文字嵌入"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def add_audio(
        self,
        audio_path: str,
        metadata: Optional[dict] = None
    ) -> str:
        """新增音訊到索引"""
        audio_id = self._generate_id(audio_path)

        # 檢查是否已存在
        existing = self.collection.get(ids=[audio_id])
        if existing['ids']:
            return audio_id

        # 轉錄
        print(f"轉錄中: {audio_path}")
        transcript_data = self._transcribe(audio_path)

        # 為每個片段建立索引
        segment_ids = []
        segment_texts = []
        segment_embeddings = []
        segment_metadatas = []

        for i, seg in enumerate(transcript_data["segments"]):
            seg_id = f"{audio_id}_seg_{i}"
            segment_ids.append(seg_id)
            segment_texts.append(seg["text"])
            segment_embeddings.append(self._get_embedding(seg["text"]))
            segment_metadatas.append({
                "audio_id": audio_id,
                "audio_path": str(Path(audio_path).absolute()),
                "segment_index": i,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "filename": Path(audio_path).name,
                **(metadata or {})
            })

        # 批次新增
        if segment_ids:
            self.collection.add(
                ids=segment_ids,
                embeddings=segment_embeddings,
                documents=segment_texts,
                metadatas=segment_metadatas
            )

        # 也新增完整轉錄
        self.collection.add(
            ids=[audio_id],
            embeddings=[self._get_embedding(transcript_data["text"])],
            documents=[transcript_data["text"]],
            metadatas=[{
                "audio_path": str(Path(audio_path).absolute()),
                "duration": transcript_data["duration"],
                "type": "full_transcript",
                "filename": Path(audio_path).name,
                **(metadata or {})
            }]
        )

        return audio_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        segment_level: bool = True
    ) -> list[dict]:
        """搜尋音訊內容"""
        query_embedding = self._get_embedding(query)

        # 根據搜尋層級過濾
        where_filter = None
        if segment_level:
            where_filter = {"type": {"$ne": "full_transcript"}}
        else:
            where_filter = {"type": "full_transcript"}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })

        return formatted

    def rag_query(
        self,
        query: str,
        n_results: int = 5
    ) -> str:
        """RAG 查詢"""
        # 搜尋相關片段
        results = self.search(query, n_results, segment_level=True)

        if not results:
            return "找不到相關音訊內容"

        # 構建上下文
        context_parts = []
        for i, r in enumerate(results):
            meta = r['metadata']
            context_parts.append(
                f"來源 {i+1}: {meta['filename']}\n"
                f"時間: {meta.get('start_time', 0):.1f}s - {meta.get('end_time', 0):.1f}s\n"
                f"內容: {r['text']}"
            )

        context = "\n\n".join(context_parts)

        # 生成回答
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一個音訊內容分析助手。根據提供的音訊轉錄內容回答問題。"
                },
                {
                    "role": "user",
                    "content": f"參考資料:\n{context}\n\n問題: {query}"
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content

# 使用範例
audio_rag = AudioRAG()

# 建立索引
audio_files = [
    "meetings/meeting_2024_01.mp3",
    "meetings/meeting_2024_02.mp3",
    "podcasts/episode_01.mp3"
]

for audio in audio_files:
    if Path(audio).exists():
        audio_rag.add_audio(audio)

# 搜尋
results = audio_rag.search("專案進度討論")
for r in results:
    print(f"[{r['metadata']['filename']}] {r['text'][:50]}...")

# RAG 查詢
answer = audio_rag.rag_query("上次會議討論了哪些重點？")
print(answer)
```

## 會議記錄系統

### 完整會議分析

```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class MeetingAnalysis:
    """會議分析結果"""
    title: str
    date: datetime
    duration: float
    participants: list[str]
    summary: str
    key_points: list[str]
    action_items: list[dict]
    topics: list[dict]
    transcript: str

class MeetingAnalyzer:
    """會議分析器"""

    def __init__(self):
        self.client = OpenAI()

    def transcribe_meeting(self, audio_path: str) -> dict:
        """轉錄會議"""
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )

        return {
            "text": response.text,
            "duration": response.duration,
            "segments": response.segments
        }

    def analyze_meeting(
        self,
        transcript: str,
        meeting_context: str = ""
    ) -> dict:
        """分析會議內容"""
        prompt = f"""分析以下會議記錄：

{f"會議背景: {meeting_context}" if meeting_context else ""}

會議記錄：
{transcript}

請以 JSON 格式輸出分析結果：
```json
{{
    "title": "會議主題",
    "summary": "會議摘要（100字內）",
    "key_points": [
        "重點1",
        "重點2"
    ],
    "action_items": [
        {{
            "task": "任務描述",
            "assignee": "負責人（如有提及）",
            "deadline": "期限（如有提及）"
        }}
    ],
    "topics_discussed": [
        {{
            "topic": "議題名稱",
            "summary": "討論摘要",
            "decisions": ["決定事項"]
        }}
    ],
    "participants_mentioned": ["參與者名稱"],
    "follow_up_needed": ["需要後續追蹤的事項"]
}}
```"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )

        result = response.choices[0].message.content

        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            return json.loads(result.strip())
        except:
            return {"raw_analysis": result}

    def process_meeting(
        self,
        audio_path: str,
        meeting_context: str = ""
    ) -> MeetingAnalysis:
        """完整處理會議"""
        # 轉錄
        print("轉錄會議中...")
        transcript_data = self.transcribe_meeting(audio_path)

        # 分析
        print("分析會議內容...")
        analysis = self.analyze_meeting(
            transcript_data["text"],
            meeting_context
        )

        return MeetingAnalysis(
            title=analysis.get("title", "未命名會議"),
            date=datetime.now(),
            duration=transcript_data["duration"],
            participants=analysis.get("participants_mentioned", []),
            summary=analysis.get("summary", ""),
            key_points=analysis.get("key_points", []),
            action_items=analysis.get("action_items", []),
            topics=analysis.get("topics_discussed", []),
            transcript=transcript_data["text"]
        )

    def generate_minutes(
        self,
        analysis: MeetingAnalysis,
        format: str = "markdown"
    ) -> str:
        """生成會議紀錄"""
        if format == "markdown":
            return self._generate_markdown_minutes(analysis)
        else:
            return self._generate_text_minutes(analysis)

    def _generate_markdown_minutes(
        self,
        analysis: MeetingAnalysis
    ) -> str:
        """生成 Markdown 格式會議紀錄"""
        minutes = f"""# {analysis.title}

**日期**: {analysis.date.strftime("%Y-%m-%d %H:%M")}
**時長**: {analysis.duration / 60:.1f} 分鐘
**參與者**: {", ".join(analysis.participants) if analysis.participants else "未記錄"}

## 會議摘要

{analysis.summary}

## 重點討論

"""
        for point in analysis.key_points:
            minutes += f"- {point}\n"

        minutes += "\n## 討論議題\n\n"
        for topic in analysis.topics:
            minutes += f"### {topic.get('topic', '議題')}\n\n"
            minutes += f"{topic.get('summary', '')}\n\n"
            if topic.get('decisions'):
                minutes += "**決定事項**:\n"
                for decision in topic['decisions']:
                    minutes += f"- {decision}\n"
            minutes += "\n"

        minutes += "## 行動項目\n\n"
        minutes += "| 任務 | 負責人 | 期限 |\n"
        minutes += "|------|--------|------|\n"
        for item in analysis.action_items:
            minutes += f"| {item.get('task', '')} | {item.get('assignee', 'TBD')} | {item.get('deadline', 'TBD')} |\n"

        return minutes

    def _generate_text_minutes(
        self,
        analysis: MeetingAnalysis
    ) -> str:
        """生成純文字格式會議紀錄"""
        return f"""
會議紀錄: {analysis.title}
{"=" * 50}
日期: {analysis.date.strftime("%Y-%m-%d %H:%M")}
時長: {analysis.duration / 60:.1f} 分鐘

摘要:
{analysis.summary}

重點:
{chr(10).join(f"• {p}" for p in analysis.key_points)}

行動項目:
{chr(10).join(f"• {item['task']} (負責: {item.get('assignee', 'TBD')}, 期限: {item.get('deadline', 'TBD')})" for item in analysis.action_items)}
"""

# 使用範例
analyzer = MeetingAnalyzer()

# 處理會議
analysis = analyzer.process_meeting(
    "weekly_standup.mp3",
    meeting_context="週例會，討論專案進度"
)

# 生成會議紀錄
minutes = analyzer.generate_minutes(analysis, format="markdown")
print(minutes)

# 儲存
with open("meeting_minutes.md", "w") as f:
    f.write(minutes)
```

## 最佳實踐

### 1. 音訊品質優化

```python
import subprocess
from pathlib import Path

def optimize_audio_for_transcription(
    input_path: str,
    output_path: str
) -> str:
    """優化音訊以提高轉錄品質"""
    # 使用 ffmpeg 進行預處理
    # - 轉換為 16kHz 單聲道
    # - 正規化音量
    # - 降噪

    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000",  # 取樣率
        "-ac", "1",      # 單聲道
        "-af", "highpass=f=200,lowpass=f=3000,volume=2",  # 濾波和增益
        "-y", output_path
    ]

    subprocess.run(cmd, capture_output=True)
    return output_path
```

### 2. 成本估算

```python
def estimate_whisper_cost(duration_seconds: float) -> float:
    """估算 Whisper API 成本"""
    # Whisper API 定價: $0.006 / 分鐘
    minutes = duration_seconds / 60
    return minutes * 0.006

def estimate_tts_cost(text: str, model: str = "tts-1") -> float:
    """估算 TTS 成本"""
    # tts-1: $15 / 1M 字元
    # tts-1-hd: $30 / 1M 字元
    char_count = len(text)
    rate = 15 if model == "tts-1" else 30
    return (char_count / 1_000_000) * rate
```

## 延伸閱讀

- [OpenAI Speech to Text](https://platform.openai.com/docs/guides/speech-to-text)
- [OpenAI Text to Speech](https://platform.openai.com/docs/guides/text-to-speech)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Edge TTS](https://github.com/rany2/edge-tts)
- [SpeechRecognition Library](https://pypi.org/project/SpeechRecognition/)
