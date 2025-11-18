"""
流式回應聊天機器人

這個範例展示如何實現流式回應（Streaming），
讓 AI 的回覆逐字顯示，提供更好的用戶體驗。

特點：
- 實時顯示生成過程
- 降低感知延遲
- 支持 OpenAI 和 Claude 的流式 API

運行方式：
1. 設置環境變數：export OPENAI_API_KEY=your_key
2. 執行：python 03_streaming_chat.py

作者：AI Learning Notes
日期：2024-11
"""

import os
from typing import List, Tuple, Generator
import gradio as gr
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def stream_openai(message: str, history: List[Tuple[str, str]]) -> Generator[str, None, None]:
    """
    使用 OpenAI API 流式生成回應

    Args:
        message: 用戶輸入
        history: 對話歷史

    Yields:
        逐步生成的文本
    """
    if not OPENAI_API_KEY:
        yield "❌ 錯誤：請設置 OPENAI_API_KEY 環境變數"
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # 構建訊息列表
        messages = [
            {"role": "system", "content": "你是一個友善且樂於助人的 AI 助手。"}
        ]

        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})

        messages.append({"role": "user", "content": message})

        # 流式調用 API
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True  # 啟用流式輸出
        )

        # 逐塊生成回應
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield full_response

    except Exception as e:
        yield f"❌ 錯誤：{str(e)}"


def stream_claude(message: str, history: List[Tuple[str, str]]) -> Generator[str, None, None]:
    """
    使用 Claude API 流式生成回應

    Args:
        message: 用戶輸入
        history: 對話歷史

    Yields:
        逐步生成的文本
    """
    if not ANTHROPIC_API_KEY:
        yield "❌ 錯誤：請設置 ANTHROPIC_API_KEY 環境變數"
        return

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # 構建訊息列表
        messages = []
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})

        messages.append({"role": "user", "content": message})

        # 流式調用 API
        full_response = ""
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield full_response

    except Exception as e:
        yield f"❌ 錯誤：{str(e)}"


def create_demo():
    """創建 Gradio 演示介面"""

    # 確定可用模型
    if OPENAI_API_KEY and ANTHROPIC_API_KEY:
        model_options = ["OpenAI GPT-3.5", "Claude 3.5 Sonnet"]
        default_model = "OpenAI GPT-3.5"
    elif OPENAI_API_KEY:
        model_options = ["OpenAI GPT-3.5"]
        default_model = "OpenAI GPT-3.5"
    elif ANTHROPIC_API_KEY:
        model_options = ["Claude 3.5 Sonnet"]
        default_model = "Claude 3.5 Sonnet"
    else:
        model_options = ["無可用模型"]
        default_model = "無可用模型"

    def chat_stream(message: str, history: List[Tuple[str, str]], model: str):
        """根據選擇的模型進行流式對話"""
        if model == "OpenAI GPT-3.5":
            yield from stream_openai(message, history)
        elif model == "Claude 3.5 Sonnet":
            yield from stream_claude(message, history)
        else:
            yield "❌ 請先設置 API 金鑰"

    # 使用 Blocks API
    with gr.Blocks(title="流式聊天機器人", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌊 流式回應聊天機器人

            體驗實時的 AI 回應生成過程！

            ## ✨ 特點

            - **即時反饋**：看著 AI 逐字生成回應
            - **更好體驗**：降低等待的焦慮感
            - **支持多模型**：OpenAI 和 Claude 都支持流式輸出

            ## 🎯 適用場景

            - 長文本生成（文章、代碼等）
            - 互動性強的應用
            - 需要即時反饋的場景
            """
        )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=model_options,
                value=default_model,
                label="🤖 選擇 AI 模型",
                info="選擇不同的模型體驗流式輸出"
            )

        chatbot = gr.Chatbot(
            label="💬 對話窗口",
            height=450,
            show_copy_button=True,
            avatar_images=(None, "🤖")  # 用戶和 AI 的頭像
        )

        with gr.Row():
            msg = gr.Textbox(
                label="✍️ 輸入訊息",
                placeholder="輸入你的問題，觀察 AI 的實時回應...",
                scale=4,
                lines=2
            )
            submit = gr.Button("📤 發送", scale=1, variant="primary")

        with gr.Row():
            clear = gr.Button("🗑️ 清除對話")
            retry = gr.Button("🔄 重新生成")

        # API 狀態指示
        with gr.Accordion("📊 API 狀態", open=False):
            gr.Markdown(
                f"""
                | API 提供商 | 狀態 | 模型 |
                |------------|------|------|
                | OpenAI     | {'✅ 已配置' if OPENAI_API_KEY else '❌ 未配置'} | GPT-3.5 Turbo |
                | Anthropic  | {'✅ 已配置' if ANTHROPIC_API_KEY else '❌ 未配置'} | Claude 3.5 Sonnet |
                """
            )

        # 範例提示
        with gr.Accordion("💡 範例問題", open=True):
            gr.Examples(
                examples=[
                    "寫一篇關於人工智能未來發展的短文（200字）",
                    "用 Python 實現一個簡單的 Web 爬蟲",
                    "解釋量子計算的基本原理",
                    "創作一首關於秋天的現代詩",
                    "列出學習機器學習的完整路線圖"
                ],
                inputs=msg,
                label="點擊任一問題開始對話"
            )

        # 使用說明
        with gr.Accordion("📖 使用說明", open=False):
            gr.Markdown(
                """
                ### 環境配置

                ```bash
                # 設置 OpenAI API Key
                export OPENAI_API_KEY="sk-..."

                # 或設置 Claude API Key
                export ANTHROPIC_API_KEY="sk-ant-..."
                ```

                ### 特點說明

                1. **流式輸出**：回應會逐字顯示，而不是等待全部生成完成
                2. **降低延遲**：用戶可以更早看到開始的內容
                3. **中斷處理**：可以隨時停止生成（刷新頁面）

                ### 技術細節

                - OpenAI: 使用 `stream=True` 參數
                - Claude: 使用 `messages.stream()` 方法
                - Gradio: 使用 generator 函數實現流式更新
                """
            )

        def respond(message, chat_history, model):
            """處理用戶輸入並流式返回回應"""
            if not message.strip():
                return chat_history

            # 添加用戶訊息
            chat_history.append([message, ""])

            # 流式生成 AI 回應
            for response in chat_stream(message, chat_history[:-1], model):
                chat_history[-1][1] = response
                yield chat_history

        # 事件綁定
        submit_event = submit.click(
            respond,
            [msg, chatbot, model_selector],
            [chatbot]
        )
        msg_event = msg.submit(
            respond,
            [msg, chatbot, model_selector],
            [chatbot]
        )

        # 發送後清空輸入框
        submit.click(lambda: "", None, msg)
        msg.submit(lambda: "", None, msg)

        # 清除對話
        clear.click(lambda: [], None, chatbot, queue=False)

        # 重新生成（移除最後一條回應並重新發送）
        def regenerate(chat_history, model):
            if not chat_history:
                return chat_history

            last_user_msg = chat_history[-1][0]
            chat_history = chat_history[:-1]

            # 重新生成
            chat_history.append([last_user_msg, ""])
            for response in chat_stream(last_user_msg, chat_history[:-1], model):
                chat_history[-1][1] = response
                yield chat_history

        retry.click(regenerate, [chatbot, model_selector], [chatbot])

    return demo


if __name__ == "__main__":
    demo = create_demo()

    # 啟動應用
    demo.launch(
        share=False,  # 改為 True 可生成公開連結
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        # 啟用佇列以支持流式輸出
        # Gradio 會自動處理流式響應
    )
