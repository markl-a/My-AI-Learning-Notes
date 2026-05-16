# 音樂生成 (Music & Audio Generation)

本章節將深入探討AI音樂和音頻生成技術，從基礎的音樂生成到進階的語音合成和音效設計。

## 📋 目錄

1. [音頻生成基礎](#音頻生成基礎)
2. [MusicGen - 音樂生成](#musicgen---音樂生成)
3. [AudioLDM - 音效生成](#audioldm---音效生成)
4. [Bark - 語音合成](#bark---語音合成)
5. [音頻處理與編輯](#音頻處理與編輯)
6. [實戰案例](#實戰案例)

---

## 🎯 學習目標

完成本章節後，你將能夠：

- ✅ 理解音頻生成的基本原理
- ✅ 使用MusicGen生成各種風格的音樂
- ✅ 運用AudioLDM建立音效
- ✅ 使用Bark進行語音合成
- ✅ 處理和編輯生成的音頻
- ✅ 構建音頻生成應用

---

## 📚 音頻生成基礎

### 音頻生成的核心概念

```
音頻信號基礎
├── 採樣率 (Sample Rate)
│   ├── 8kHz - 語音
│   ├── 16kHz - 語音高品質
│   ├── 44.1kHz - CD音質
│   └── 48kHz - 專業音頻
│
├── 位深度 (Bit Depth)
│   ├── 16-bit - 標準
│   └── 24-bit - 高保真
│
└── 聲道 (Channels)
    ├── Mono - 單聲道
    └── Stereo - 立體聲
```

### 音頻生成技術分類

1. **音樂生成**
   - 旋律生成
   - 伴奏生成
   - 完整編曲

2. **音效生成**
   - 環境音
   - 特效音
   - 擬音效果

3. **語音合成**
   - 文字轉語音 (TTS)
   - 語音克隆
   - 情感語音

---

## 🎵 MusicGen - 音樂生成

### 什麼是MusicGen？

MusicGen是Meta開發的音樂生成模型，可以根據文字描述生成高品質的音樂片段。

### 核心特點

- ✅ **高音質輸出** - 支持高達48kHz採樣率
- ✅ **多樣性** - 支持各種音樂風格
- ✅ **可控性** - 可控制旋律、節奏、風格
- ✅ **條件生成** - 支持音頻條件引導

### 安裝

```bash
pip install transformers torch torchaudio scipy
pip install audiocraft  # Meta的音頻生成庫
```

### 基本使用

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

# 加載模型
# 可選: small (300M), medium (1.5B), large (3.3B), melody (1.5B)
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# 設置生成參數
model.set_generation_params(
    duration=10,  # 生成時長（秒）
    temperature=1.0,  # 隨機性 (0.1-2.0)
    top_k=250,  # 採樣策略
    top_p=0.95,
    cfg_coef=3.0  # 提示詞遵循程度
)

# 生成音樂
descriptions = [
    "upbeat electronic dance music with a strong beat",
    "relaxing piano melody with ambient background",
    "rock music with electric guitar solo",
]

# 批量生成
wav = model.generate(descriptions)

# 保存音頻
for idx, one_wav in enumerate(wav):
    audio_write(
        f'generated_music_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness",  # 音量標準化
        loudness_compressor=True
    )
    print(f"Saved: generated_music_{idx}.wav")
```

### 旋律條件生成

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio

# 加載melody模型（支持旋律條件）
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=15)

# 加載參考旋律
melody, sr = torchaudio.load("reference_melody.wav")

# 重採樣到模型需要的採樣率
if sr != model.sample_rate:
    melody = torchaudio.functional.resample(melody, sr, model.sample_rate)

# 基於旋律生成音樂
descriptions = [
    "jazz arrangement with saxophone and piano",
    "orchestral version with strings",
    "electronic remix with synth bass"
]

# 生成（使用旋律作為條件）
wav = model.generate_with_chroma(
    descriptions=descriptions,
    melody_wavs=melody[None].expand(len(descriptions), -1, -1),
    melody_sample_rate=model.sample_rate,
    progress=True
)

# 保存
for idx, one_wav in enumerate(wav):
    audio_write(f'melody_based_{idx}', one_wav.cpu(), model.sample_rate)
```

### 音樂風格控制

```python
def generate_music_by_style(
    style,
    duration=30,
    num_variations=3,
    output_prefix="music"
):
    """
    根據風格生成音樂

    Args:
        style: 音樂風格
        duration: 時長（秒）
        num_variations: 變體數量
        output_prefix: 輸出文件前綴
    """
    # 風格模板
    style_templates = {
        "lo-fi": "lo-fi hip hop beat, chill, relaxing, soft piano, vinyl crackle",
        "epic": "epic orchestral music, cinematic, dramatic, powerful strings and brass",
        "jazz": "smooth jazz, saxophone, piano, double bass, brush drums",
        "edm": "electronic dance music, energetic, synthesizer, heavy bass, 128 bpm",
        "ambient": "ambient atmospheric music, ethereal pads, no drums, peaceful",
        "rock": "rock music, electric guitar, drums, bass guitar, energetic",
        "classical": "classical piano composition, elegant, expressive, romantic era",
        "funk": "funky groove, slap bass, rhythmic guitar, horn section, 110 bpm"
    }

    base_description = style_templates.get(
        style.lower(),
        f"{style} music"
    )

    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        cfg_coef=3.0
    )

    # 生成多個變體
    descriptions = [base_description] * num_variations

    wav = model.generate(descriptions, progress=True)

    # 保存
    for idx, one_wav in enumerate(wav):
        filename = f"{output_prefix}_{style}_{idx+1}"
        audio_write(filename, one_wav.cpu(), model.sample_rate)
        print(f"✓ Generated: {filename}.wav")

# 使用示例
generate_music_by_style("lo-fi", duration=30, num_variations=3)
generate_music_by_style("epic", duration=20, num_variations=2)
```

### 進階：多段音樂拼接

```python
from audiocraft.models import MusicGen
import torch
import torchaudio

def generate_music_segments(
    segments_config,
    output_path="complete_music.wav",
    transition_duration=2
):
    """
    生成並拼接多個音樂段落

    Args:
        segments_config: 段落配置列表
            [{"description": "...", "duration": 10}, ...]
        output_path: 輸出路徑
        transition_duration: 過渡時長（秒）
    """
    model = MusicGen.get_pretrained('facebook/musicgen-medium')

    all_segments = []

    for i, config in enumerate(segments_config):
        print(f"Generating segment {i+1}/{len(segments_config)}...")

        model.set_generation_params(duration=config["duration"])

        description = config["description"]

        # 生成段落
        wav = model.generate([description], progress=True)
        segment = wav[0]

        all_segments.append(segment)

    # 拼接音頻（帶淡入淡出過渡）
    final_audio = crossfade_segments(
        all_segments,
        model.sample_rate,
        transition_duration
    )

    # 保存
    torchaudio.save(
        output_path,
        final_audio.cpu(),
        model.sample_rate
    )

    print(f"Complete music saved to {output_path}")

def crossfade_segments(segments, sample_rate, transition_duration):
    """使用淡入淡出拼接音頻段落"""
    transition_samples = int(transition_duration * sample_rate)

    result = segments[0]

    for segment in segments[1:]:
        # 建立淡出淡入曲線
        fadeout = torch.linspace(1, 0, transition_samples)
        fadein = torch.linspace(0, 1, transition_samples)

        # 應用淡出到前一段的末尾
        result[:, -transition_samples:] *= fadeout

        # 應用淡入到當前段的開頭
        segment[:, :transition_samples] *= fadein

        # 重疊部分相加
        result[:, -transition_samples:] += segment[:, :transition_samples]

        # 拼接剩餘部分
        result = torch.cat([result, segment[:, transition_samples:]], dim=1)

    return result

# 使用示例：建立多段音樂作品
segments = [
    {
        "description": "gentle piano introduction, slow tempo, melancholic",
        "duration": 15
    },
    {
        "description": "building up with strings, crescendo, emotional",
        "duration": 20
    },
    {
        "description": "full orchestral climax, powerful, dramatic",
        "duration": 25
    },
    {
        "description": "peaceful ending, soft piano and strings, fade out",
        "duration": 15
    }
]

generate_music_segments(segments, "orchestral_piece.wav", transition_duration=3)
```

---

## 🔊 AudioLDM - 音效生成

### 什麼是AudioLDM？

AudioLDM是基於Latent Diffusion的音頻生成模型，專門用於生成音效和環境音。

### 安裝

```bash
pip install diffusers transformers scipy
```

### 基本使用

```python
from diffusers import AudioLDMPipeline
import torch
import scipy

# 加載模型
pipe = AudioLDMPipeline.from_pretrained(
    "cvssp/audioldm-s-full-v2",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 生成音效
prompt = "dog barking in the distance, outdoor environment, realistic"

audio = pipe(
    prompt,
    num_inference_steps=50,
    audio_length_in_s=5.0,  # 音頻長度（秒）
    guidance_scale=2.5
).audios[0]

# 保存音頻
scipy.io.wavfile.write(
    "dog_bark.wav",
    rate=16000,
    data=audio
)
```

### 音效類別生成

```python
def generate_sound_effects(sound_type, num_variations=3):
    """
    生成特定類型的音效

    Args:
        sound_type: 音效類型
        num_variations: 變體數量
    """
    # 音效模板
    sound_templates = {
        "nature": [
            "gentle rain falling on leaves",
            "ocean waves crashing on beach",
            "birds chirping in forest",
            "wind blowing through trees",
            "thunder rumbling in distance"
        ],
        "urban": [
            "city traffic ambience",
            "subway train arriving",
            "cafe background chatter",
            "construction site sounds",
            "police siren passing by"
        ],
        "home": [
            "door creaking open",
            "footsteps on wooden floor",
            "clock ticking",
            "kettle boiling water",
            "cat meowing"
        ],
        "sci-fi": [
            "spaceship engine humming",
            "laser gun shooting",
            "alien creature sound",
            "futuristic computer beep",
            "teleportation effect"
        ],
        "musical": [
            "drum kit being played",
            "acoustic guitar strumming",
            "piano chord progression",
            "violin melody",
            "synthesizer arpeggio"
        ]
    }

    prompts = sound_templates.get(sound_type, [f"{sound_type} sound"])

    pipe = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-s-full-v2",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    for idx, prompt in enumerate(prompts[:num_variations]):
        print(f"Generating: {prompt}")

        audio = pipe(
            prompt,
            num_inference_steps=50,
            audio_length_in_s=5.0,
            guidance_scale=2.5,
            num_waveforms_per_prompt=1
        ).audios[0]

        # 保存
        output_file = f"{sound_type}_{idx+1}.wav"
        scipy.io.wavfile.write(output_file, rate=16000, data=audio)
        print(f"✓ Saved: {output_file}")

# 生成不同類型的音效
generate_sound_effects("nature", num_variations=5)
generate_sound_effects("sci-fi", num_variations=3)
```

### 長音效生成

```python
def generate_long_soundscape(
    description,
    total_duration=30,
    segment_duration=10,
    overlap_duration=2
):
    """
    生成長時間的音景

    Args:
        description: 音效描述
        total_duration: 總時長（秒）
        segment_duration: 每段時長（秒）
        overlap_duration: 重疊時長（秒）
    """
    import numpy as np

    pipe = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-s-full-v2",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # 計算需要生成的段數
    num_segments = int(np.ceil(total_duration / (segment_duration - overlap_duration)))

    all_audio = []
    sample_rate = 16000

    for i in range(num_segments):
        print(f"Generating segment {i+1}/{num_segments}...")

        audio = pipe(
            description,
            num_inference_steps=50,
            audio_length_in_s=segment_duration,
            guidance_scale=2.5
        ).audios[0]

        all_audio.append(audio)

    # 拼接音頻（帶重疊）
    overlap_samples = int(overlap_duration * sample_rate)
    final_audio = all_audio[0]

    for audio in all_audio[1:]:
        # 建立淡入淡出
        fadeout = np.linspace(1, 0, overlap_samples)
        fadein = np.linspace(0, 1, overlap_samples)

        # 混合重疊部分
        overlap_mix = (
            final_audio[-overlap_samples:] * fadeout +
            audio[:overlap_samples] * fadein
        )

        # 拼接
        final_audio = np.concatenate([
            final_audio[:-overlap_samples],
            overlap_mix,
            audio[overlap_samples:]
        ])

    # 截取到目標時長
    target_samples = int(total_duration * sample_rate)
    final_audio = final_audio[:target_samples]

    # 保存
    scipy.io.wavfile.write(
        "long_soundscape.wav",
        rate=sample_rate,
        data=final_audio
    )

    print(f"Generated {total_duration}s soundscape")

# 生成30秒的雨聲環境音
generate_long_soundscape(
    description="continuous rainfall, thunder occasionally, ambient outdoor",
    total_duration=30,
    segment_duration=10,
    overlap_duration=3
)
```

---

## 🎙️ Bark - 語音合成

### 什麼是Bark？

Bark是Suno開發的多語言文字轉語音模型，支持語音生成、音樂生成和非語言聲音。

### 核心特點

- ✅ **多語言支持** - 支持多種語言包括中文
- ✅ **情感表達** - 可以表達不同情緒
- ✅ **非語言聲音** - 支持笑聲、嘆息等
- ✅ **音樂生成** - 可以哼唱和唱歌

### 安裝

```bash
pip install git+https://github.com/suno-ai/bark.git
pip install scipy numpy transformers
```

### 基本使用

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np

# 預加載模型
preload_models()

# 基本文字轉語音
text_prompt = """
    Hello, I am Bark, a text-to-speech model created by Suno.
    I can speak in many different voices and languages.
"""

audio_array = generate_audio(text_prompt)

# 保存音頻
write_wav("bark_output.wav", SAMPLE_RATE, audio_array)
print("Audio saved!")
```

### 語音預設與控制

```python
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav

# Bark語音預設格式: [language]_[gender]_[id]
# 例如: en_speaker_0, zh_speaker_0, es_speaker_3

def generate_with_voice(
    text,
    voice_preset="v2/en_speaker_6",
    output_file="output.wav"
):
    """
    使用特定語音預設生成音頻

    Args:
        text: 要轉換的文字
        voice_preset: 語音預設
        output_file: 輸出文件
    """
    # 生成音頻
    audio_array = generate_audio(
        text,
        history_prompt=voice_preset
    )

    # 保存
    write_wav(output_file, SAMPLE_RATE, audio_array)
    print(f"Generated with voice {voice_preset}: {output_file}")

# 測試不同語音
voices = [
    "v2/en_speaker_0",  # 男聲
    "v2/en_speaker_1",  # 女聲
    "v2/en_speaker_6",  # 年輕男聲
    "v2/en_speaker_9",  # 成熟女聲
]

text = "The quick brown fox jumps over the lazy dog."

for idx, voice in enumerate(voices):
    generate_with_voice(
        text,
        voice_preset=voice,
        output_file=f"voice_{idx}.wav"
    )
```

### 情感與非語言聲音

```python
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav

# Bark支持特殊標記來控制情感和非語言聲音
# [laughter] - 笑聲
# [laughs] - 笑
# [sighs] - 嘆息
# [music] - 音樂
# [gasps] - 喘息
# [clears throat] - 清嗓子
# CAPITALIZATION - 強調

def generate_emotional_speech(output_file="emotional.wav"):
    """生成帶情感的語音"""

    text = """
    [clears throat] Ladies and gentlemen, [laughs]
    I have some AMAZING news to share with you today!
    [gasps] You won't believe what just happened!
    [sighs] But first, let me tell you a story...
    """

    audio = generate_audio(text)
    write_wav(output_file, SAMPLE_RATE, audio)
    print(f"Emotional speech saved to {output_file}")

# 生成音樂/歌唱
def generate_singing(output_file="singing.wav"):
    """生成歌唱"""

    text = """
    ♪ La la la la la ♪
    ♪ Do re mi fa so la ti do ♪
    [music] [clears throat] That was beautiful!
    """

    audio = generate_audio(text)
    write_wav(output_file, SAMPLE_RATE, audio)
    print(f"Singing saved to {output_file}")

generate_emotional_speech()
generate_singing()
```

### 多語言語音生成

```python
def generate_multilingual_speech():
    """生成多語言語音"""

    languages = {
        "english": {
            "text": "Hello! How are you today?",
            "preset": "v2/en_speaker_6"
        },
        "chinese": {
            "text": "你好！今天過得怎麼樣？",
            "preset": "v2/zh_speaker_0"
        },
        "spanish": {
            "text": "¡Hola! ¿Cómo estás hoy?",
            "preset": "v2/es_speaker_0"
        },
        "french": {
            "text": "Bonjour! Comment allez-vous aujourd'hui?",
            "preset": "v2/fr_speaker_0"
        },
        "german": {
            "text": "Hallo! Wie geht es dir heute?",
            "preset": "v2/de_speaker_0"
        }
    }

    for lang, config in languages.items():
        print(f"Generating {lang}...")

        audio = generate_audio(
            config["text"],
            history_prompt=config["preset"]
        )

        output_file = f"speech_{lang}.wav"
        write_wav(output_file, SAMPLE_RATE, audio)
        print(f"✓ Saved: {output_file}")

generate_multilingual_speech()
```

### 長文字語音生成

```python
def generate_long_form_speech(
    text,
    voice_preset="v2/en_speaker_6",
    output_file="long_speech.wav",
    segment_length=200  # 字符數
):
    """
    生成長文字語音（分段處理）

    Args:
        text: 長文字
        voice_preset: 語音預設
        output_file: 輸出文件
        segment_length: 每段字符數
    """
    import numpy as np

    # 分割文字為段落
    sentences = text.split('. ')
    segments = []
    current_segment = ""

    for sentence in sentences:
        if len(current_segment) + len(sentence) < segment_length:
            current_segment += sentence + ". "
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence + ". "

    if current_segment:
        segments.append(current_segment.strip())

    # 生成每個段落
    audio_segments = []

    for i, segment in enumerate(segments):
        print(f"Generating segment {i+1}/{len(segments)}...")

        audio = generate_audio(
            segment,
            history_prompt=voice_preset
        )

        audio_segments.append(audio)

    # 拼接所有段落
    # 添加短暫的靜音間隔
    silence = np.zeros(int(0.5 * SAMPLE_RATE))  # 0.5秒靜音

    final_audio = audio_segments[0]
    for audio in audio_segments[1:]:
        final_audio = np.concatenate([final_audio, silence, audio])

    # 保存
    write_wav(output_file, SAMPLE_RATE, final_audio)
    print(f"Long-form speech saved to {output_file}")

# 使用示例
long_text = """
Artificial intelligence is transforming the world as we know it.
From healthcare to finance, from transportation to entertainment,
AI is making a significant impact. Machine learning algorithms
can now recognize patterns, make predictions, and even create
original content. The future of AI is bright and full of possibilities.
As we continue to develop more advanced systems, we must also
consider the ethical implications and ensure that AI benefits
all of humanity.
"""

generate_long_form_speech(
    long_text,
    voice_preset="v2/en_speaker_6",
    output_file="ai_speech.wav"
)
```

---

## 🎛️ 音頻處理與編輯

### 基礎音頻處理

```python
import librosa
import soundfile as sf
import numpy as np

def adjust_audio_properties(
    input_file,
    output_file,
    target_sr=None,
    volume_factor=1.0,
    trim_silence=True,
    normalize=True
):
    """
    調整音頻屬性

    Args:
        input_file: 輸入文件
        output_file: 輸出文件
        target_sr: 目標採樣率
        volume_factor: 音量倍數
        trim_silence: 是否裁剪靜音
        normalize: 是否標準化音量
    """
    # 加載音頻
    audio, sr = librosa.load(input_file, sr=target_sr)

    # 裁剪靜音
    if trim_silence:
        audio, _ = librosa.effects.trim(
            audio,
            top_db=20,  # 靜音閾值
            frame_length=2048,
            hop_length=512
        )

    # 調整音量
    audio = audio * volume_factor

    # 標準化
    if normalize:
        audio = librosa.util.normalize(audio)

    # 保存
    sf.write(output_file, audio, sr)
    print(f"Processed audio saved to {output_file}")

# 使用
adjust_audio_properties(
    "input.wav",
    "output.wav",
    target_sr=44100,
    volume_factor=1.2,
    trim_silence=True,
    normalize=True
)
```

### 音頻特效

```python
def apply_audio_effects(
    input_file,
    output_file,
    effect="reverb"
):
    """
    應用音頻特效

    Args:
        input_file: 輸入文件
        output_file: 輸出文件
        effect: 特效類型 (reverb/echo/pitch_shift/time_stretch)
    """
    audio, sr = librosa.load(input_file)

    if effect == "reverb":
        # 簡單混響效果
        delay = int(0.1 * sr)  # 100ms延遲
        reverb = np.zeros(len(audio) + delay)
        reverb[:len(audio)] = audio
        reverb[delay:] += audio * 0.3  # 添加延遲信號
        processed = reverb[:len(audio)]

    elif effect == "echo":
        # 回聲效果
        delay = int(0.3 * sr)
        echo = np.copy(audio)
        if len(audio) > delay:
            echo[delay:] += audio[:-delay] * 0.5
        processed = echo

    elif effect == "pitch_shift":
        # 音高變換（升高2個半音）
        processed = librosa.effects.pitch_shift(
            audio,
            sr=sr,
            n_steps=2
        )

    elif effect == "time_stretch":
        # 時間拉伸（加速1.2倍）
        processed = librosa.effects.time_stretch(audio, rate=1.2)

    else:
        processed = audio

    # 標準化
    processed = librosa.util.normalize(processed)

    # 保存
    sf.write(output_file, processed, sr)
    print(f"Applied {effect} effect: {output_file}")

# 測試不同效果
effects = ["reverb", "echo", "pitch_shift", "time_stretch"]
for effect in effects:
    apply_audio_effects(
        "input.wav",
        f"output_{effect}.wav",
        effect=effect
    )
```

### 音頻混合

```python
def mix_audio_tracks(
    tracks,
    volumes,
    output_file="mixed.wav",
    target_sr=44100
):
    """
    混合多個音軌

    Args:
        tracks: 音軌文件列表
        volumes: 各音軌音量 (0-1)
        output_file: 輸出文件
        target_sr: 目標採樣率
    """
    # 加載所有音軌
    audio_tracks = []
    max_length = 0

    for track_file in tracks:
        audio, sr = librosa.load(track_file, sr=target_sr)
        audio_tracks.append(audio)
        max_length = max(max_length, len(audio))

    # 填充到相同長度
    for i in range(len(audio_tracks)):
        if len(audio_tracks[i]) < max_length:
            padding = max_length - len(audio_tracks[i])
            audio_tracks[i] = np.pad(
                audio_tracks[i],
                (0, padding),
                mode='constant'
            )

    # 混合
    mixed = np.zeros(max_length)
    for audio, volume in zip(audio_tracks, volumes):
        mixed += audio * volume

    # 標準化避免削波
    mixed = librosa.util.normalize(mixed)

    # 保存
    sf.write(output_file, mixed, target_sr)
    print(f"Mixed audio saved to {output_file}")

# 使用示例：混合音樂、旁白和音效
mix_audio_tracks(
    tracks=["music.wav", "narration.wav", "sfx.wav"],
    volumes=[0.3, 0.7, 0.4],
    output_file="final_mix.wav"
)
```

---

## 🚀 實戰案例

### 案例1：播客自動生成器

```python
# podcast_generator.py
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import numpy as np
import librosa
import soundfile as sf

class PodcastGenerator:
    """播客自動生成器"""

    def __init__(self):
        self.music_model = MusicGen.get_pretrained('facebook/musicgen-small')

    def generate_intro_music(self, duration=10):
        """生成開場音樂"""
        self.music_model.set_generation_params(duration=duration)

        wav = self.music_model.generate([
            "upbeat podcast intro music, energetic, modern, professional"
        ])

        # 保存臨時文件
        audio_write("temp_intro", wav[0].cpu(), self.music_model.sample_rate)

        # 加載並返回
        audio, sr = librosa.load("temp_intro.wav", sr=SAMPLE_RATE)
        return audio

    def generate_outro_music(self, duration=10):
        """生成結尾音樂"""
        self.music_model.set_generation_params(duration=duration)

        wav = self.music_model.generate([
            "podcast outro music, calm, reflective, fade out"
        ])

        audio_write("temp_outro", wav[0].cpu(), self.music_model.sample_rate)
        audio, sr = librosa.load("temp_outro.wav", sr=SAMPLE_RATE)
        return audio

    def generate_speech_segment(self, text, voice="v2/en_speaker_6"):
        """生成語音段落"""
        audio = generate_audio(text, history_prompt=voice)
        return audio

    def create_podcast(
        self,
        title,
        host_text,
        segments,
        output_file="podcast.wav"
    ):
        """
        建立完整播客

        Args:
            title: 播客標題
            host_text: 主持人開場白
            segments: 內容段落列表
            output_file: 輸出文件
        """
        print("Generating podcast components...")

        # 生成音樂
        intro_music = self.generate_intro_music(duration=8)
        outro_music = self.generate_outro_music(duration=8)

        # 生成開場白
        intro_speech = self.generate_speech_segment(
            f"Welcome to {title}. {host_text}",
            voice="v2/en_speaker_6"
        )

        # 生成內容段落
        segment_audios = []
        for i, segment in enumerate(segments):
            print(f"Generating segment {i+1}/{len(segments)}...")
            audio = self.generate_speech_segment(
                segment["text"],
                voice=segment.get("voice", "v2/en_speaker_6")
            )
            segment_audios.append(audio)

            # 段落間添加短暫靜音
            silence = np.zeros(int(1.0 * SAMPLE_RATE))
            segment_audios.append(silence)

        # 生成結束語
        outro_speech = self.generate_speech_segment(
            "Thank you for listening! Don't forget to subscribe!",
            voice="v2/en_speaker_6"
        )

        # 組合所有元素
        # 1. 開場音樂（淡出）
        intro_music = self.apply_fade(intro_music, fade_in=True, fade_out=True)

        # 2. 拼接：intro music + intro speech
        podcast = np.concatenate([intro_music, intro_speech])

        # 3. 添加所有內容段落
        for audio in segment_audios:
            podcast = np.concatenate([podcast, audio])

        # 4. 添加結束語和結尾音樂
        outro_music = self.apply_fade(outro_music, fade_in=True, fade_out=True)
        podcast = np.concatenate([podcast, outro_speech, outro_music])

        # 標準化
        podcast = librosa.util.normalize(podcast)

        # 保存
        write_wav(output_file, SAMPLE_RATE, podcast)
        print(f"Podcast saved to {output_file}")

    def apply_fade(self, audio, fade_in=True, fade_out=True, duration=2.0):
        """應用淡入淡出"""
        fade_samples = int(duration * SAMPLE_RATE)

        if fade_in:
            fade_in_curve = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in_curve

        if fade_out:
            fade_out_curve = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out_curve

        return audio

# 使用示例
generator = PodcastGenerator()

segments = [
    {
        "text": "Today we're discussing artificial intelligence and its impact on society.",
        "voice": "v2/en_speaker_6"
    },
    {
        "text": "AI has the potential to transform every aspect of our lives, from healthcare to education.",
        "voice": "v2/en_speaker_6"
    },
    {
        "text": "[clears throat] But we must also consider the ethical implications.",
        "voice": "v2/en_speaker_9"
    }
]

generator.create_podcast(
    title="Tech Talk Podcast",
    host_text="I'm your host, and today we have a fascinating discussion.",
    segments=segments,
    output_file="tech_talk_episode_001.wav"
)
```

### 案例2：背景音樂生成器

```python
# background_music_generator.py
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

class BackgroundMusicGenerator:
    """為影片生成背景音樂"""

    def __init__(self):
        self.model = MusicGen.get_pretrained('facebook/musicgen-medium')

    def generate_for_video(
        self,
        video_duration,
        mood="upbeat",
        genre="electronic",
        output_file="bg_music.wav"
    ):
        """
        根據影片時長和情緒生成背景音樂

        Args:
            video_duration: 影片時長（秒）
            mood: 情緒 (upbeat/calm/dramatic/playful/mysterious)
            genre: 風格 (electronic/acoustic/orchestral/ambient)
            output_file: 輸出文件
        """
        # 構建提示詞
        mood_descriptors = {
            "upbeat": "energetic, positive, motivating",
            "calm": "peaceful, relaxing, gentle",
            "dramatic": "intense, cinematic, powerful",
            "playful": "fun, cheerful, lighthearted",
            "mysterious": "suspenseful, dark, intriguing"
        }

        genre_descriptors = {
            "electronic": "synthesizer, modern, digital",
            "acoustic": "guitar, piano, organic",
            "orchestral": "strings, brass, classical",
            "ambient": "atmospheric, ethereal, spacious"
        }

        prompt = f"""
        {mood_descriptors.get(mood, mood)} {genre} background music,
        {genre_descriptors.get(genre, "")},
        seamless loop, no vocals, suitable for video content
        """

        # 設置生成參數
        self.model.set_generation_params(
            duration=min(30, video_duration),  # 最多30秒
            temperature=1.0,
            cfg_coef=3.0
        )

        # 生成
        wav = self.model.generate([prompt], progress=True)

        # 如果影片更長，需要循環音樂
        if video_duration > 30:
            # 保存基礎循環
            audio_write("temp_loop", wav[0].cpu(), self.model.sample_rate)

            # 循環拼接
            import librosa
            import soundfile as sf

            loop_audio, sr = librosa.load("temp_loop.wav")

            # 計算需要多少次循環
            num_loops = int(np.ceil(video_duration / 30))

            # 拼接
            extended_audio = np.tile(loop_audio, num_loops)

            # 截取到準確時長
            target_samples = int(video_duration * sr)
            extended_audio = extended_audio[:target_samples]

            # 應用淡出
            fade_duration = 3  # 3秒淡出
            fade_samples = int(fade_duration * sr)
            fade_curve = np.linspace(1, 0, fade_samples)
            extended_audio[-fade_samples:] *= fade_curve

            # 保存
            sf.write(output_file, extended_audio, sr)
        else:
            audio_write(output_file.replace('.wav', ''), wav[0].cpu(), self.model.sample_rate)

        print(f"Background music saved to {output_file}")

    def generate_playlist(
        self,
        num_tracks=5,
        duration=30,
        theme="work_focus"
    ):
        """
        生成音樂播放列表

        Args:
            num_tracks: 曲目數量
            duration: 每首時長
            theme: 主題
        """
        themes = {
            "work_focus": [
                "calm electronic music for concentration",
                "minimal ambient background music",
                "lo-fi beats for studying",
                "peaceful piano for productivity",
                "soft jazz for focus"
            ],
            "workout": [
                "high energy electronic music, 130 bpm",
                "motivating rock music, intense",
                "upbeat hip hop beats",
                "powerful drum and bass",
                "energetic EDM workout music"
            ],
            "relaxation": [
                "peaceful ambient music, nature sounds",
                "calm meditation music",
                "soothing spa music",
                "gentle acoustic guitar",
                "soft piano lullaby"
            ]
        }

        prompts = themes.get(theme, themes["work_focus"])

        self.model.set_generation_params(duration=duration)

        for idx, prompt in enumerate(prompts[:num_tracks]):
            print(f"Generating track {idx+1}/{num_tracks}...")

            wav = self.model.generate([prompt], progress=True)

            filename = f"{theme}_track_{idx+1:02d}"
            audio_write(filename, wav[0].cpu(), self.model.sample_rate)

            print(f"✓ Saved: {filename}.wav")

# 使用示例
generator = BackgroundMusicGenerator()

# 為影片生成背景音樂
generator.generate_for_video(
    video_duration=120,  # 2分鐘
    mood="upbeat",
    genre="electronic",
    output_file="video_bg_music.wav"
)

# 生成工作專注播放列表
generator.generate_playlist(
    num_tracks=5,
    duration=180,  # 3分鐘每首
    theme="work_focus"
)
```

---

## 📚 參考資源

### 官方文檔
- [MusicGen GitHub](https://github.com/facebookresearch/audiocraft)
- [AudioLDM Paper](https://arxiv.org/abs/2301.12503)
- [Bark GitHub](https://github.com/suno-ai/bark)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

### 模型資源
- [Hugging Face Audio Models](https://huggingface.co/models?pipeline_tag=text-to-audio)
- [AudioCraft Models](https://huggingface.co/facebook)

### 學習資源
- [Digital Signal Processing Course](https://www.coursera.org/learn/dsp)
- [Music Information Retrieval](https://www.audiolabs-erlangen.de/resources/MIR)

---

## ✅ 檢查清單

完成本章節後，你應該能夠：

- [ ] 理解音頻生成的基本概念
- [ ] 使用MusicGen生成不同風格的音樂
- [ ] 控制音樂的情緒和風格
- [ ] 使用AudioLDM生成各種音效
- [ ] 建立長時間的音景
- [ ] 使用Bark進行文字轉語音
- [ ] 生成多語言和情感豐富的語音
- [ ] 處理和編輯音頻文件
- [ ] 混合多個音軌
- [ ] 構建實用的音頻生成應用

---

## 下一步

完成音樂生成後，建議：

1. **實戰項目** - 構建完整的多模態內容生成系統
2. **整合應用** - 將圖片、影片、音頻生成整合到實際應用中
3. **探索更多** - 研究最新的音頻生成技術和模型

---

最後更新：2024-11-19
難度級別：🔴 高級
預計學習時間：10-12小時
