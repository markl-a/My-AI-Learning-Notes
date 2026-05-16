# Hugging Face Spaces 部署完整指南

## 🌟 什麼是 Hugging Face Spaces？

Hugging Face Spaces 是一個**免費**的機器學習應用託管平台，讓你可以輕鬆部署和分享 ML 應用。

### 核心優勢

| 特性 | 說明 |
|------|------|
| **完全免費** | 基礎版永久免費，包含 CPU 運算資源 |
| **支持多框架** | Gradio、Streamlit、Docker |
| **免費 GPU** | 可申請免費的 T4 GPU（需審核）|
| **自動部署** | Git push 即自動構建和部署 |
| **公開分享** | 獲得公開 URL，可嵌入網站 |
| **社群整合** | 與 Hugging Face Hub 深度整合 |

---

## 🚀 快速開始：3 種部署方式

### 方式 1：網頁直接建立（最簡單，5 分鐘）

#### 步驟 1：建立 Space

1. 訪問 https://huggingface.co/spaces
2. 點擊右上角 "Create new Space"
3. 填寫資訊：
   - **Space name**：你的應用名稱（例如：`my-chatbot`）
   - **License**：選擇 MIT 或 Apache 2.0
   - **SDK**：選擇 `Gradio` 或 `Streamlit`
   - **Visibility**：Public（公開）或 Private（私密）

#### 步驟 2：在線編輯程式碼

1. 建立後會自動生成一個基礎模板
2. 點擊 "Files" → "app.py" → "Edit"
3. 將你的程式碼貼上去
4. 點擊 "Commit changes to main"

#### 步驟 3：添加依賴

1. 點擊 "Files" → "Add file" → "Create a new file"
2. 文件名：`requirements.txt`
3. 內容：
   ```txt
   gradio==4.0.0
   openai==1.0.0
   anthropic==0.5.0
   ```
4. 保存並提交

#### 步驟 4：設置 API 金鑰（Secrets）

1. 點擊 "Settings" → "Repository secrets"
2. 點擊 "New secret"
3. 添加：
   - Name: `OPENAI_API_KEY`
   - Value: `sk-...`（你的 API 金鑰）
4. 點擊 "Add"

✅ **完成！** Space 會自動構建和部署，幾分鐘後就可以訪問了。

---

### 方式 2：Git 命令行部署（適合開發者）

#### 步驟 1：建立 Space（同方式 1）

在網頁上建立一個空的 Space。

#### 步驟 2：克隆 Space 倉庫

```bash
# 克隆你的 Space
git clone https://huggingface.co/spaces/你的用戶名/你的space名稱
cd 你的space名稱
```

#### 步驟 3：添加文件

```bash
# 複製你的應用文件
cp /path/to/your/app.py ./
cp /path/to/your/requirements.txt ./

# 或者直接建立文件
cat > app.py << 'EOF'
import gradio as gr

def greet(name):
    return f"你好, {name}！"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
EOF

cat > requirements.txt << 'EOF'
gradio==4.0.0
EOF
```

#### 步驟 4：提交並推送

```bash
# 添加文件到 Git
git add app.py requirements.txt

# 提交
git commit -m "Initial commit: Add chatbot app"

# 推送到 Hugging Face
git push
```

#### 步驟 5：查看部署

訪問 `https://huggingface.co/spaces/你的用戶名/你的space名稱`

---

### 方式 3：使用 huggingface_hub Python 庫（自動化）

#### 安裝工具

```bash
pip install huggingface_hub
```

#### 登錄

```bash
huggingface-cli login
```

輸入你的 Hugging Face Token（在 https://huggingface.co/settings/tokens 獲取）

#### Python 腳本部署

```python
from huggingface_hub import HfApi, create_repo

# 初始化 API
api = HfApi()

# 建立 Space
repo_id = "你的用戶名/my-chatbot"
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",  # 或 "streamlit"
    private=False
)

# 上傳文件
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=repo_id,
    repo_type="space"
)

print(f"✅ 部署完成！訪問：https://huggingface.co/spaces/{repo_id}")
```

---

## 📁 Space 文件結構

### Gradio Space 最小結構

```
my-gradio-space/
├── app.py              # 主程式（必須，文件名必須是 app.py）
├── requirements.txt    # Python 依賴（可選）
└── README.md          # Space 描述（可選但推薦）
```

### Streamlit Space 最小結構

```
my-streamlit-space/
├── app.py              # 主程式（必須）
├── requirements.txt    # Python 依賴（可選）
└── README.md          # Space 描述（可選）
```

### 完整專業結構

```
my-professional-space/
├── app.py              # 主程式
├── requirements.txt    # Python 依賴
├── README.md          # Space 描述和使用說明
├── .gitignore         # Git 忽略文件
├── utils/             # 工具模塊（可選）
│   ├── __init__.py
│   ├── llm_utils.py
│   └── data_utils.py
├── assets/            # 靜態資源（可選）
│   ├── logo.png
│   └── styles.css
└── examples/          # 示例文件（可選）
    └── sample.txt
```

---

## 🔧 README.md 配置（重要！）

README.md 的開頭需要包含 YAML 元資料，用於配置 Space：

### Gradio Space README.md

```yaml
---
title: 我的聊天機器人
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 我的 LLM 聊天機器人

這是一個基於 GPT-3.5 的智能對話系統。

## 功能特色

- ✅ 支持多輪對話
- ✅ 流式回應
- ✅ 自定義系統提示詞

## 使用方法

1. 在輸入框中輸入問題
2. 點擊發送或按 Enter
3. 等待 AI 回應

## 技術棧

- Gradio 4.0.0
- OpenAI GPT-3.5 Turbo
```

### Streamlit Space README.md

```yaml
---
title: 資料分析助手
emoji: 📊
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 資料分析 AI 助手

使用 AI 自動分析你的資料！
```

### README 配置選項說明

| 選項 | 說明 | 範例值 |
|------|------|--------|
| `title` | Space 標題 | "我的聊天機器人" |
| `emoji` | 顯示的圖標 | 🤖 😊 📊 🎨 |
| `colorFrom` | 漸變起始顏色 | blue, red, green, yellow, purple |
| `colorTo` | 漸變結束顏色 | 同上 |
| `sdk` | 使用的框架 | gradio, streamlit, docker |
| `sdk_version` | SDK 版本 | 4.0.0, 1.28.0 |
| `app_file` | 主文件名 | app.py（預設值）|
| `pinned` | 是否置頂到個人頁 | true, false |
| `license` | 開源許可證 | mit, apache-2.0, gpl-3.0 |

---

## 🔐 管理 Secrets（API 金鑰）

### 設置 Secrets

1. 進入你的 Space 頁面
2. 點擊 "Settings"
3. 找到 "Repository secrets"
4. 點擊 "New secret"
5. 輸入：
   - Name: `OPENAI_API_KEY`
   - Value: `sk-proj-...`
6. 點擊 "Add"

### 在程式碼中使用 Secrets

```python
import os

# 讀取環境變數
api_key = os.environ.get("OPENAI_API_KEY")

# 或使用 Streamlit（僅 Streamlit Space）
import streamlit as st
api_key = st.secrets.get("OPENAI_API_KEY")  # 需要 .streamlit/secrets.toml
```

### 多個 Secrets 管理

```python
import os

# 讀取多個 API 金鑰
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# 檢查是否存在
if not OPENAI_KEY:
    raise ValueError("請設置 OPENAI_API_KEY")
```

---

## 🎨 申請免費 GPU

### GPU 等級

| 等級 | GPU 型號 | 顯存 | 費用 | 適用場景 |
|------|----------|------|------|----------|
| **CPU** | - | - | 免費 | 輕量應用、API 呼叫 |
| **T4 small** | NVIDIA T4 | 16GB | 免費（需申請）| 中小型模型推論 |
| **T4 medium** | NVIDIA T4 | 16GB | $0.60/小時 | 大型模型 |
| **A10G small** | NVIDIA A10G | 24GB | $1.05/小時 | 高性能需求 |

### 申請步驟

1. 進入 Space Settings
2. 點擊 "Hardware" 標籤
3. 選擇 "T4 small"
4. 點擊 "Request for free upgrade"
5. 填寫申請理由（英文）：
   ```
   I'm building an educational LLM application for demonstrating
   model inference techniques. This Space will be used to showcase
   best practices in LLM deployment and will benefit the community.
   ```
6. 提交並等待審核（通常 1-3 天）

### 使用 GPU 的程式碼示例

```python
import torch
import gradio as gr

# 檢測 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備: {device}")

# 加載模型到 GPU
model = AutoModel.from_pretrained("模型名稱").to(device)
```

---

## 🔄 自動部署和 CI/CD

### Git Push 自動部署

每次 `git push` 到 main 分支時，Space 會自動：
1. 拉取最新程式碼
2. 安裝 requirements.txt 中的依賴
3. 重新啟動應用

### 查看構建日誌

1. 進入 Space 頁面
2. 點擊 "Logs" 標籤
3. 查看實時構建和運行日誌

### 處理構建錯誤

常見錯誤和解決方法：

#### 錯誤 1：依賴安裝失敗

```
ERROR: Could not find a version that satisfies the requirement xxx
```

**解決方案：**
- 檢查 `requirements.txt` 中的包名和版本號
- 使用 `==` 固定版本，而不是 `>=`

#### 錯誤 2：應用啟動失敗

```
ModuleNotFoundError: No module named 'xxx'
```

**解決方案：**
- 確保 `requirements.txt` 包含所有依賴
- 重新推送程式碼觸發重新構建

#### 錯誤 3：API 金鑰錯誤

```
AuthenticationError: Invalid API key
```

**解決方案：**
- 檢查 Secrets 中的 API 金鑰是否正確
- 確保程式碼中正確讀取環境變數

---

## 📊 Space 分析和監控

### 查看訪問統計

Hugging Face 提供基礎的訪問統計：

1. 進入 Space 頁面
2. 點擊右上角的統計圖標
3. 查看：
   - 總訪問量
   - 每日活躍用戶
   - 地理分佈

### 添加自定義分析

可以整合 Google Analytics：

```python
import gradio as gr

# 在 Gradio Blocks 中添加 Google Analytics
with gr.Blocks(analytics_enabled=True, head="""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""") as demo:
    # 你的應用程式碼
    pass
```

---

## 🌐 嵌入 Space 到網站

### 使用 iframe

```html
<iframe
  src="https://你的用戶名-你的space名稱.hf.space"
  frameborder="0"
  width="100%"
  height="600"
></iframe>
```

### Gradio 嵌入

Gradio 提供更好的嵌入支持：

```html
<gradio-app src="https://你的用戶名-你的space名稱.hf.space"></gradio-app>
<script
  type="module"
  src="https://gradio.s3-us-west-2.amazonaws.com/4.0.0/gradio.js"
></script>
```

---

## 🚀 進階技巧

### 1. 使用 Docker SDK（完全自定義）

建立 `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

README.md 配置：

```yaml
---
title: My Docker Space
sdk: docker
app_port: 7860
---
```

### 2. 多頁面應用

使用 Gradio Blocks 建立多頁面：

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("聊天"):
            # 聊天介面
            pass
        with gr.Tab("設置"):
            # 設置介面
            pass

demo.launch()
```

### 3. 持久化存儲

Space 重啟後會清空資料，使用外部存儲：

```python
import gradio as gr
from huggingface_hub import hf_hub_download

# 下載資料文件
data_path = hf_hub_download(
    repo_id="你的用戶名/你的資料集",
    filename="data.json",
    repo_type="dataset"
)
```

### 4. 速率限制

防止濫用：

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
                raise gr.Error("請求過於頻繁，請稍後再試")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=10, time_window=60)
def chat(message, history):
    # 你的聊天邏輯
    pass
```

---

## 🐛 常見問題排查

### Q1: Space 一直顯示 "Building..."

**可能原因：**
- 依賴安裝時間過長
- 網路問題
- requirements.txt 錯誤

**解決方案：**
1. 查看 "Logs" 標籤的構建日誌
2. 簡化 requirements.txt，只保留必要的包
3. 等待 5-10 分鐘再檢查

### Q2: 應用啟動後顯示錯誤

**檢查清單：**
- [ ] app.py 語法是否正確
- [ ] 所有依賴是否在 requirements.txt 中
- [ ] API 金鑰是否正確設置
- [ ] 端口是否正確（Gradio 預設 7860）

### Q3: 如何升級到付費 GPU？

**步驟：**
1. Settings → Hardware
2. 選擇付費 GPU 等級
3. 添加付款方式
4. 確認升級

### Q4: 可以使用自定義域名嗎？

**答案：** 目前 Hugging Face Spaces 不支持自定義域名，但你可以：
- 使用 iframe 嵌入到你的網站
- 使用反向代理（需要自己的伺服器）

---

## 📚 實戰範例

### 範例 1：部署 Gradio 聊天機器人

```bash
# 1. 建立 Space（在網頁上操作）

# 2. 克隆倉庫
git clone https://huggingface.co/spaces/你的用戶名/chatbot
cd chatbot

# 3. 建立文件
cat > app.py << 'EOF'
import os
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def chat(message, history):
    messages = [{"role": "system", "content": "你是一個友善的助手"}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(fn=chat, title="我的聊天機器人")
demo.launch()
EOF

cat > requirements.txt << 'EOF'
gradio==4.0.0
openai==1.0.0
EOF

cat > README.md << 'EOF'
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

# 我的聊天機器人

基於 GPT-3.5 的智能對話系統
EOF

# 4. 提交並推送
git add .
git commit -m "Add chatbot"
git push

# 5. 在網頁上設置 OPENAI_API_KEY

# ✅ 完成！
```

---

## 🎯 最佳實踐

### 1. 安全性

- ✅ 永遠不要在程式碼中硬編碼 API 金鑰
- ✅ 使用 Secrets 管理敏感資訊
- ✅ 添加速率限制防止濫用
- ✅ 驗證用戶輸入

### 2. 性能優化

- ✅ 使用 `@st.cache_data` (Streamlit) 或 `gr.State` (Gradio) 快取資料
- ✅ 延遲加載大型模型
- ✅ 優化圖片和資源大小
- ✅ 使用 GPU 加速推論

### 3. 用戶體驗

- ✅ 提供清晰的錯誤訊息
- ✅ 添加載入動畫
- ✅ 使用範例引導用戶
- ✅ 寫詳細的 README

### 4. 維護性

- ✅ 固定依賴版本（使用 `==`）
- ✅ 添加註釋和文檔
- ✅ 定期更新依賴
- ✅ 監控日誌和錯誤

---

## 📖 延伸閱讀

- [Hugging Face Spaces 官方文檔](https://huggingface.co/docs/hub/spaces)
- [Gradio 文檔](https://www.gradio.app/docs/)
- [Streamlit 文檔](https://docs.streamlit.io/)
- [Space 範例集合](https://huggingface.co/spaces)

---

**最後更新：** 2024 年 11 月

**作者：** AI Learning Notes
