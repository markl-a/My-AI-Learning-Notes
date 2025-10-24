# ASUS ROG Flow Z13 2025 AI 工具完整指南
## AMD AI MAX+ 395 (128GB RAM) 最佳化配置

---

## 🎯 您的硬體優勢

### 超強規格總覽

| 元件 | 規格 | AI 能力 | 特殊優勢 |
|------|------|---------|----------|
| **CPU** | 16 核 Zen 5 | 通用 AI | 多線程處理 |
| **GPU** | Radeon 8060S (40 CU) | 高性能 AI | 可分配高達 96GB VRAM！ |
| **NPU** | XDNA2 架構 | 50 TOPS | 低功耗推論 |
| **RAM** | 128GB LPDDR5X | 超大容量 | 可運行 70B+ 參數模型 |

> **獨特優勢**: 您的 128GB RAM 配置可以分配高達 96GB 作為 VRAM，這是絕大多數筆電無法達到的！

---

## 🚀 推薦 AI 工具清單

### 1️⃣ 大型語言模型 (LLM)

#### **LM Studio** ⭐ 強烈推薦
- **特點**: 圖形化介面，無需技術知識
- **GPU/NPU**: 完整支援 AMD GPU 加速
- **優勢**: 可運行 70B 參數模型
```bash
# 下載: https://lmstudio.ai/
# 模型建議:
- Llama 3.1 70B (Q4 量化)
- Mixtral 8x22B
- Qwen 2.5 72B
```

#### **Ollama** ⭐ 推薦
- **特點**: 命令列工具，Docker 化管理
- **GPU/NPU**: GPU 加速支援
```bash
# 安裝
curl -fsSL https://ollama.com/install.sh | sh

# 運行大模型
ollama run llama3.1:70b
ollama run mixtral:8x22b
```

#### **Text Generation WebUI (oobabooga)**
- **特點**: 功能最完整的 WebUI
- **GPU/NPU**: DirectML/Vulkan 支援
```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
python -m pip install -r requirements_amd.txt
```

#### **AMD GAIA** ⭐ AMD 原生
- **特點**: AMD 官方工具，NPU+GPU 混合加速
- **GPU/NPU**: 完整 Hybrid 模式支援
```bash
# 從 GitHub 下載
https://github.com/amd/gaia/releases
```

#### **KoboldCpp**
- **特點**: 故事生成特化
- **GPU/NPU**: CLBlast/Vulkan 支援
```bash
# 支援 AMD GPU
https://github.com/LostRuins/koboldcpp
```

---

### 2️⃣ 圖像生成 AI

#### **AMD Amuse** ⭐ AMD 原生
- **特點**: AMD 官方 Stable Diffusion 工具
- **GPU/NPU**: CPU+GPU+NPU 三重加速
- **優勢**: 超解析度使用 NPU 加速

#### **ComfyUI** ⭐ 專業推薦
- **特點**: 節點式工作流程
- **GPU/NPU**: DirectML 後端支援
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install torch-directml
pip install -r requirements.txt
python main.py --directml
```

#### **Stable Diffusion WebUI (AUTOMATIC1111)**
- **特點**: 最流行的 SD WebUI
- **GPU/NPU**: DirectML 支援
```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
# 編輯 webui-user.bat，加入:
set COMMANDLINE_ARGS=--use-directml --medvram
```

#### **Fooocus**
- **特點**: 簡化版 SD，易用
- **GPU/NPU**: AMD GPU 支援
```bash
git clone https://github.com/lllyasviel/Fooocus
cd Fooocus
pip install -r requirements_versions.txt
python entry_with_update.py --directml
```

#### **InvokeAI**
- **特點**: 專業級介面
- **GPU/NPU**: 多後端支援
```bash
pip install invokeai
invokeai --web --precision float16
```

---

### 3️⃣ 影片/音訊 AI

#### **Whisper (OpenAI)**
- **特點**: 語音轉文字
- **GPU/NPU**: GPU 加速
```python
import whisper
model = whisper.load_model("large-v3")
# 可處理長音訊，128GB RAM 無壓力
```

#### **Real-ESRGAN**
- **特點**: 影像/影片升頻
- **GPU/NPU**: DirectML 支援
```bash
pip install realesrgan
# NPU 可加速超解析度
```

#### **Riffusion**
- **特點**: AI 音樂生成
- **GPU/NPU**: GPU 加速

---

### 4️⃣ 程式開發 AI

#### **Continue.dev**
- **特點**: VS Code AI 助手
- **GPU/NPU**: 本地模型支援
```bash
# VS Code 擴充功能
# 可連接到 LM Studio 或 Ollama
```

#### **Tabby**
- **特點**: 自架程式碼補全
- **GPU/NPU**: GPU 加速推論
```bash
docker run -it --gpus all \
  -p 8080:8080 \
  -v $HOME/.tabby:/data \
  tabbyml/tabby serve --model StarCoder-7B
```

#### **CodeGPT**
- **特點**: 多 IDE 支援
- **GPU/NPU**: 連接本地模型

---

### 5️⃣ 專業/研究工具

#### **PyTorch + DirectML** ⭐ 已安裝
- **用途**: 深度學習開發
- **GPU/NPU**: 完整 GPU 支援
```python
import torch_directml
device = torch_directml.device()
# 可訓練大型模型
```

#### **ONNX Runtime** ⭐ 已安裝
- **用途**: 跨框架推論
- **GPU/NPU**: NPU + GPU 雙支援
```python
# NPU 推論
providers=['VitisAIExecutionProvider']
# GPU 推論
providers=['DmlExecutionProvider']
```

#### **JAX**
- **用途**: 高性能機器學習
- **GPU/NPU**: 實驗性 AMD 支援
```bash
pip install jax[cpu]
# AMD GPU 支援開發中
```

#### **MLC-LLM**
- **用途**: 高效能 LLM 部署
- **GPU/NPU**: Vulkan 後端
```bash
pip install mlc-llm
python -m mlc_llm compile model.onnx --device vulkan
```

---

## 💡 針對 128GB RAM 的特殊配置

### 超大模型運行配置

```python
# LM Studio 設定
{
  "n_gpu_layers": -1,  # 全部載入 GPU
  "n_ctx": 32768,      # 超長上下文
  "gpu_memory": 96000, # 96GB VRAM
  "cpu_threads": 16    # 全部 CPU 核心
}

# llama.cpp 參數
./main -m model.gguf \
  -ngl 999 \           # 所有層到 GPU
  -c 32768 \           # 32K 上下文
  -b 2048 \            # 大批次
  -t 16                # 16 線程
```

### 多模型並行運行

```python
# 同時運行多個模型
# 模型 1: 聊天 (30GB)
ollama run llama3.1:70b --port 11434

# 模型 2: 程式碼 (20GB)
ollama run codellama:34b --port 11435

# 模型 3: 圖像生成 (10GB)
python sd_webui.py --port 7860

# 還有 68GB 可用！
```

---

## 📊 效能基準參考

### 您的系統預期效能

| 模型類型 | 模型大小 | Token/秒 | 記憶體使用 |
|---------|---------|----------|------------|
| Llama 3.1 70B (Q4) | 35GB | 8-12 | GPU 35GB |
| Mixtral 8x22B (Q4) | 65GB | 5-8 | GPU 65GB |
| SDXL + ControlNet | 10GB | 2-4 img/min | GPU 10GB |
| Whisper Large v3 | 3GB | 即時轉錄 | GPU 3GB |

---

## 🔧 優化建議

### 1. GPU 記憶體分配
```bash
# Windows 系統變數
set GPU_MAX_HEAP_SIZE=100
set GPU_MAX_ALLOC_PERCENT=100

# 允許 GPU 使用最大記憶體
```

### 2. NPU 優先任務
- 影像超解析度
- 背景去除
- 即時翻譯
- 輕量推論

### 3. GPU 優先任務
- 大模型推論
- 圖像生成
- 模型訓練
- 影片處理

### 4. 電源管理
```powershell
# 最高效能模式
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 關閉 GPU 節能
AMD Software: Adrenalin Edition → 效能 → 調校 → 手動
```

---

## 🎮 遊戲 + AI 組合

### 可同時運行
1. **遊戲**: 使用 GPU 前 20GB
2. **AI 助手**: LLM 使用 GPU 30GB
3. **語音辨識**: Whisper 使用 NPU
4. **串流編碼**: 使用硬體編碼器

---

## 📝 快速開始腳本

創建 `start_ai_suite.bat`:

```batch
@echo off
echo Starting AI Suite for ASUS Z13 2025...

REM 啟動 LM Studio
start "LM Studio" "C:\Program Files\LM Studio\LM Studio.exe"

REM 啟動 Ollama
start "Ollama" ollama serve

REM 啟動 ComfyUI
cd /d C:\AI\ComfyUI
start "ComfyUI" python main.py --directml

REM 啟動 GAIA
start "GAIA" C:\GAIA\gaia-gui.exe

echo All AI tools started!
pause
```

---

## 🔗 資源下載

### 必裝工具
1. [LM Studio](https://lmstudio.ai/)
2. [AMD GAIA](https://github.com/amd/gaia/releases)
3. [Ollama](https://ollama.com/)
4. [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### 模型資源
1. [HuggingFace](https://huggingface.co/models)
2. [CivitAI](https://civitai.com/) (SD 模型)
3. [TheBloke](https://huggingface.co/TheBloke) (量化 LLM)

---

## 總結

您的 ASUS Z13 2025 (128GB) 配置是目前最強的 AI 筆電之一：

✅ **可運行 70B+ 參數 LLM** (大多數筆電最多 13B)
✅ **96GB VRAM 分配能力** (RTX 4090 只有 24GB)
✅ **NPU+GPU 雙加速** (50 TOPS NPU + 強大 GPU)
✅ **同時多工運行** (LLM + SD + Whisper 無壓力)

建議優先安裝：
1. **LM Studio** - 最簡單的 LLM 工具
2. **AMD GAIA** - 充分利用 NPU
3. **ComfyUI** - 專業圖像生成
4. **Ollama** - 多模型管理

您的硬體配置幾乎可以運行目前所有開源 AI 模型！

---

最後更新：2024年10月24日