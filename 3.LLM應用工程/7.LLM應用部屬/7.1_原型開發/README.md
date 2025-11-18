# 7.1 原型開發：快速上線展示

## 概述

原型開發階段的目標是**快速驗證想法**，用最少的時間和資源構建可演示的 LLM 應用。本章節介紹三大主流工具：Gradio、Streamlit 和 Hugging Face Spaces。

## 🎯 學習目標

- 掌握 Gradio 和 Streamlit 的基礎和進階用法
- 能夠在 10 分鐘內部署一個可分享的 LLM 應用
- 學會整合多種 AI 服務（OpenAI, Claude, 本地模型）
- 理解如何添加 AI 輔助功能（摘要、翻譯、情感分析等）

## 🛠️ 工具對比

| 特性 | Gradio | Streamlit | Hugging Face Spaces |
|------|--------|-----------|---------------------|
| **學習曲線** | ⭐⭐ 簡單 | ⭐⭐⭐ 中等 | ⭐ 非常簡單 |
| **UI 靈活性** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中等（基於 Gradio/Streamlit） |
| **ML 專用功能** | ⭐⭐⭐⭐⭐ 很強 | ⭐⭐⭐ 一般 | ⭐⭐⭐⭐⭐ 很強 |
| **分享便利性** | ⭐⭐⭐⭐ 好 | ⭐⭐⭐ 一般 | ⭐⭐⭐⭐⭐ 最好 |
| **社群支持** | ⭐⭐⭐⭐⭐ 很好 | ⭐⭐⭐⭐⭐ 很好 | ⭐⭐⭐⭐ 好 |
| **免費託管** | ❌ | ❌ | ✅ 永久免費 |
| **最佳用途** | ML 模型演示 | 數據儀表板 | 公開分享模型 |

## 📁 目錄結構

```
7.1_原型開發/
├── README.md                          # 本文件
├── examples/
│   ├── gradio/
│   │   ├── 01_basic_chat.py          # 基礎聊天機器人
│   │   ├── 02_multi_model_chat.py    # 多模型切換
│   │   ├── 03_streaming_chat.py      # 流式回應
│   │   ├── 04_ai_assistant.py        # AI 輔助功能（摘要、翻譯等）
│   │   ├── 05_image_chat.py          # 多模態：圖像+文字
│   │   └── requirements.txt          # Gradio 依賴
│   └── streamlit/
│       ├── 01_basic_chat.py          # 基礎聊天應用
│       ├── 02_rag_demo.py            # RAG 文檔問答
│       ├── 03_data_analysis.py       # 數據分析助手
│       └── requirements.txt          # Streamlit 依賴
└── huggingface_spaces_guide.md        # HF Spaces 部署指南
```

---

## 🚀 Gradio：最適合 ML 模型的 UI 框架

### 為什麼選擇 Gradio？

1. **專為機器學習設計**：原生支持各種 ML 輸入輸出類型
2. **快速分享**：`share=True` 一鍵生成公開連結
3. **美觀的 UI**：自動生成專業的界面
4. **社群整合**：與 Hugging Face 深度集成

### 核心概念

#### 1. Interface API（簡單場景）

適合單一輸入輸出的場景：

```python
import gradio as gr

def greet(name):
    return f"你好, {name}！"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

#### 2. ChatInterface API（聊天場景）

專為聊天機器人設計：

```python
import gradio as gr

def respond(message, history):
    return f"你說：{message}"

demo = gr.ChatInterface(fn=respond)
demo.launch()
```

#### 3. Blocks API（複雜場景）

完全自定義的布局：

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# 我的 LLM 應用")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="輸入")
            submit_btn = gr.Button("送出")
        with gr.Column():
            output_text = gr.Textbox(label="輸出")

    submit_btn.click(fn=lambda x: x.upper(), inputs=input_text, outputs=output_text)

demo.launch()
```

### Gradio 範例概覽

#### 範例 1：基礎聊天機器人 (`01_basic_chat.py`)
- 整合 OpenAI 和 Claude API
- 環境變數管理 API 金鑰
- 錯誤處理和重試機制

#### 範例 2：多模型切換 (`02_multi_model_chat.py`)
- 支持 GPT-3.5、GPT-4、Claude 3 模型切換
- 系統提示詞自定義
- 模型參數調整（temperature, max_tokens）

#### 範例 3：流式回應 (`03_streaming_chat.py`)
- 實時顯示生成過程
- 更好的用戶體驗
- 降低感知延遲

#### 範例 4：AI 輔助功能 (`04_ai_assistant.py`)
- 📝 文本摘要
- 🌐 多語言翻譯
- 😊 情感分析
- 🔑 關鍵詞提取
- ✍️ 文本改寫

#### 範例 5：多模態聊天 (`05_image_chat.py`)
- 支持圖像+文字輸入
- 整合 GPT-4 Vision
- 圖像描述和分析

---

## 🎨 Streamlit：構建數據應用的利器

### 為什麼選擇 Streamlit？

1. **數據科學友好**：原生支持 Pandas, Matplotlib, Plotly
2. **組件豐富**：豐富的 UI 組件庫
3. **狀態管理**：內建 session state 管理
4. **部署簡單**：Streamlit Cloud 免費託管

### 核心概念

#### 1. 基礎組件

```python
import streamlit as st

# 文字
st.title("我的應用")
st.markdown("### 副標題")
st.text("純文字")

# 輸入
name = st.text_input("你的名字")
age = st.number_input("年齡", min_value=0, max_value=120)
choice = st.selectbox("選擇", ["選項 1", "選項 2"])

# 按鈕
if st.button("點我"):
    st.write(f"你好, {name}！")
```

#### 2. 狀態管理

```python
import streamlit as st

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 讀取和修改狀態
st.session_state.messages.append({"role": "user", "content": "你好"})
```

#### 3. 快取機制

```python
import streamlit as st

@st.cache_data  # 快取數據
def load_data():
    return expensive_computation()

@st.cache_resource  # 快取模型等資源
def load_model():
    return load_large_model()
```

### Streamlit 範例概覽

#### 範例 1：基礎聊天應用 (`01_basic_chat.py`)
- 對話歷史顯示
- Session state 管理
- 清除對話功能

#### 範例 2：RAG 文檔問答 (`02_rag_demo.py`)
- 文檔上傳和解析
- 向量化檢索
- 引用來源顯示

#### 範例 3：數據分析助手 (`03_data_analysis.py`)
- CSV 數據上傳
- AI 自動分析
- 圖表視覺化

---

## 🤗 Hugging Face Spaces：免費的模型託管平台

### 什麼是 Hugging Face Spaces？

Hugging Face Spaces 是一個免費的機器學習應用託管平台，支持：
- **Gradio** 和 **Streamlit** 應用
- **Docker** 自定義環境
- **免費 GPU**（需申請）
- **永久運行**

### 部署流程

#### 方法 1：從本地上傳

1. **創建 Space**
   - 訪問 https://huggingface.co/spaces
   - 點擊 "Create new Space"
   - 選擇 SDK（Gradio/Streamlit）

2. **準備文件**
   ```
   my-space/
   ├── app.py              # 主程式（必須）
   ├── requirements.txt    # Python 依賴
   └── README.md          # Space 描述
   ```

3. **上傳到 Space**
   ```bash
   git clone https://huggingface.co/spaces/你的用戶名/space名稱
   cd space名稱
   cp /path/to/your/app.py ./
   cp /path/to/your/requirements.txt ./
   git add .
   git commit -m "Initial commit"
   git push
   ```

#### 方法 2：直接在線編輯

1. 在 Space 頁面點擊 "Files" → "Add file"
2. 直接在瀏覽器中編輯 `app.py`
3. 保存後自動部署

### 配置文件

#### README.md（Space 元數據）

```yaml
---
title: 我的聊天機器人
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 我的 LLM 聊天機器人

這是一個基於 GPT-3.5 的智能對話系統。
```

#### requirements.txt

```txt
gradio==4.0.0
openai==1.0.0
anthropic==0.5.0
python-dotenv==1.0.0
```

### 環境變數（Secrets）

在 Space 設置中添加：
1. 點擊 Settings → Repository secrets
2. 添加 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 等

在代碼中使用：
```python
import os
openai_api_key = os.environ.get("OPENAI_API_KEY")
```

### 申請免費 GPU

1. 在 Space 設置中點擊 "Hardware"
2. 選擇 "T4 small"（免費）
3. 提交申請並等待審核

---

## 💡 最佳實踐

### 1. API 金鑰安全

❌ **錯誤做法**：
```python
openai.api_key = "sk-xxxxxxxxxxxxxxxx"  # 硬編碼
```

✅ **正確做法**：
```python
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

### 2. 錯誤處理

```python
import gradio as gr

def chat(message, history):
    try:
        response = call_llm_api(message)
        return response
    except Exception as e:
        return f"❌ 錯誤：{str(e)}"
```

### 3. 速率限制

```python
import time
from functools import wraps

def rate_limit(max_calls=10, time_window=60):
    calls = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - time_window]

            if len(calls) >= max_calls:
                raise Exception("請求過於頻繁，請稍後再試")

            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=10, time_window=60)
def chat(message, history):
    # ...
```

### 4. 日誌記錄

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def chat(message, history):
    logging.info(f"用戶輸入：{message}")
    response = call_llm(message)
    logging.info(f"模型回應：{response[:50]}...")
    return response
```

---

## 🎓 實戰練習

### 練習 1：基礎聊天機器人（30 分鐘）
1. 使用 Gradio ChatInterface 創建聊天機器人
2. 整合 OpenAI API
3. 添加錯誤處理
4. 部署到 Hugging Face Spaces

### 練習 2：多功能 AI 助手（1 小時）
1. 使用 Gradio Blocks 創建多標籤頁應用
2. 實現：聊天、摘要、翻譯三個功能
3. 添加模型選擇（GPT-3.5 vs Claude）
4. 實現流式回應

### 練習 3：RAG 文檔問答（2 小時）
1. 使用 Streamlit 創建文檔上傳界面
2. 整合向量資料庫（FAISS/Chroma）
3. 實現語義檢索
4. 顯示引用來源

---

## 🔗 參考資源

### 官方文檔
- [Gradio 文檔](https://www.gradio.app/docs/)
- [Streamlit 文檔](https://docs.streamlit.io/)
- [Hugging Face Spaces 文檔](https://huggingface.co/docs/hub/spaces)

### 優秀範例
- [Gradio 官方範例庫](https://www.gradio.app/guides/quickstart)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [HF Spaces 熱門應用](https://huggingface.co/spaces)

### 視頻教程
- [Gradio 速成課程](https://www.youtube.com/watch?v=RiCQzBluTxU)
- [Streamlit 入門](https://www.youtube.com/watch?v=8M20LyCZDOY)

---

## 🐛 常見問題

### Q1: Gradio `share=True` 無法生成公開連結？

**A:** 可能是網路防火牆問題。解決方案：
```bash
# 使用 ngrok 替代
pip install pyngrok
```

```python
from pyngrok import ngrok

public_url = ngrok.connect(7860)
print(f"公開 URL: {public_url}")

demo.launch(server_port=7860, share=False)
```

### Q2: Streamlit 應用刷新後狀態丟失？

**A:** 使用 `st.session_state` 保存狀態：
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

### Q3: Hugging Face Space 部署後一直 Building？

**A:** 檢查：
1. `requirements.txt` 語法是否正確
2. `app.py` 是否有語法錯誤
3. 查看 Space 的 "Logs" 標籤頁

### Q4: API 調用超時？

**A:** 設置合理的 timeout：
```python
import openai
openai.api_timeout = 30  # 30 秒超時
```

---

## 🎯 下一步

完成本章節後，你可以：
- ✅ 在 10 分鐘內部署一個 LLM 應用原型
- ✅ 選擇適合的框架（Gradio vs Streamlit）
- ✅ 部署到 Hugging Face Spaces 並分享給他人

**準備好了嗎？** 前往 [examples/gradio/](./examples/gradio/) 開始實戰！

或者繼續學習 [7.2 生產部署](../7.2_生產部屬/README.md) 了解如何將原型升級為生產級服務。
