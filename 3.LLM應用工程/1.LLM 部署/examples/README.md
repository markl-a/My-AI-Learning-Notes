# LLM 部署基礎示例

本目錄包含各種 LLM 部署和使用的基礎示例代碼，每個示例都是獨立可運行的。

## 📋 示例列表

### 1. Ollama 本地部署 (`01_ollama_basic.py`)
- **說明**：使用 Ollama 在本地運行開源模型
- **依賴**：Ollama（需先安裝）
- **特色**：簡單易用，無需 GPU 配置
- **適用場景**：快速原型開發、本地測試

### 2. OpenAI API 使用 (`02_openai_api.py`)
- **說明**：通過 OpenAI API 使用 GPT 模型
- **依賴**：OpenAI API key
- **特色**：展示基礎調用、結構化輸出、多輪對話
- **適用場景**：生產環境、高品質需求

### 3. Hugging Face 本地模型 (`03_huggingface_local.py`)
- **說明**：使用 Transformers 加載和運行本地模型
- **依賴**：GPU（推薦）、Transformers、PyTorch
- **特色**：4-bit 量化、性能優化
- **適用場景**：完全控制、數據隱私

### 4. 流式響應 (`04_streaming_response.py`)
- **說明**：實現流式輸出，提升用戶體驗
- **依賴**：支持流式的 LLM 提供商
- **特色**：實時顯示生成內容
- **適用場景**：聊天應用、交互式界面

### 5. 批量推理 (`05_batch_inference.py`)
- **說明**：高效處理大量請求
- **依賴**：本地模型或 API
- **特色**：並行處理、進度追蹤、性能統計
- **適用場景**：數據處理、批次任務

### 6. 模型比較工具 (`06_model_comparison.py`)
- **說明**：比較不同模型的性能和輸出質量
- **依賴**：多個模型（API 或本地）
- **特色**：並排比較、性能分析、成本估算
- **適用場景**：模型選擇、性能評估

## 🚀 快速開始

### 安裝依賴

```bash
# 進入部署目錄
cd "3.LLM應用工程/1.LLM 部署"

# 安裝 Python 依賴
pip install -r requirements.txt
```

### 配置環境變數

```bash
# 複製環境變數模板
cp .env.example .env

# 編輯 .env 文件，填入你的 API keys
nano .env
```

### 運行示例

```bash
# 運行特定示例
python examples/01_ollama_basic.py

# 或者使用 Python 模組方式
python -m examples.02_openai_api
```

## 📝 使用說明

### Ollama 本地部署

需要先安裝 Ollama：

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 下載模型
ollama pull llama3.1:8b

# 啟動服務（通常會自動啟動）
ollama serve
```

### API Keys 獲取

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google AI**: https://makersuite.google.com/app/apikey

### GPU 設置

如果使用本地模型，確保已安裝 CUDA：

```bash
# 檢查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 查看 GPU 信息
nvidia-smi
```

## 💡 最佳實踐

### 1. 開發階段
- 使用 Ollama 或小型模型快速迭代
- 使用 API 進行原型驗證

### 2. 測試階段
- 比較不同模型的性能和成本
- 測試邊界情況和錯誤處理

### 3. 生產階段
- 根據需求選擇合適的部署方式
- 實施監控和日誌記錄
- 考慮成本和性能平衡

## 🔧 故障排除

### Ollama 連接失敗
```bash
# 檢查 Ollama 是否運行
curl http://localhost:11434/api/tags

# 重啟 Ollama
ollama serve
```

### CUDA 記憶體不足
```python
# 使用更激進的量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 或使用更小的模型
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 而非 70B
```

### API 速率限制
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_api():
    # API 調用代碼
    pass
```

## 📚 延伸閱讀

- [Ollama 官方文檔](https://ollama.ai/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [OpenAI API 參考](https://platform.openai.com/docs)
- [vLLM 性能優化](https://docs.vllm.ai/)

## 🤝 貢獻

歡迎提交問題和改進建議！如果你有好的示例代碼，請創建 Pull Request。
