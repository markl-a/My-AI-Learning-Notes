# Building Generative AI Applications with Gradio

## 📋 課程概述

學習使用 Gradio 快速建立 AI 應用介面並部署。

### 課程目標
- 掌握 Gradio 基本用法
- 建立互動式 AI 介面
- 學習部署和分享應用
- 實作多種 AI 應用

### 課程時長
約 1 小時

## 🎯 Gradio 快速上手

```python
import gradio as gr
from openai import OpenAI

client = OpenAI()

def chat(message, history):
    """聊天功能"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# 建立介面
demo = gr.ChatInterface(
    fn=chat,
    title="AI 聊天機器人",
    description="使用 GPT-3.5 的聊天機器人"
)

# 啟動
demo.launch()
```

## 💡 實用範例

- 文本摘要工具
- 圖像生成器
- 程式碼解釋器
- 翻譯助手

---

**課程連結**：[DeepLearning.ai - Gradio Applications](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)

**完成日期**：2025-01-17
