"""
基礎聊天機器人範例

這個範例展示如何使用 Gradio 建立一個簡單的聊天機器人，
支持 OpenAI 和 Anthropic Claude API。

運行方式：
1. 設置環境變數：export OPENAI_API_KEY=your_key
2. 執行：python 01_basic_chat.py
3. 在瀏覽器打開顯示的 URL

作者：AI Learning Notes
日期：2024-11
"""

import os
from typing import List, Tuple
import gradio as gr
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 檢查 API 金鑰
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def chat_with_openai(message: str, history: List[Tuple[str, str]]) -> str:
    """
    使用 OpenAI API 進行對話

    Args:
        message: 用戶輸入的訊息
        history: 對話歷史，格式為 [(用戶訊息, AI回應), ...]

    Returns:
        AI 的回應
    """
    if not OPENAI_API_KEY:
        return "❌ 錯誤：請設置 OPENAI_API_KEY 環境變數"

    try:
        # 延遲導入，避免在沒有 API key 時報錯
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # 構建對話訊息列表
        messages = [
            {"role": "system", "content": "你是一個友善且樂於助人的 AI 助手。"}
        ]

        # 添加歷史對話
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})

        # 添加當前用戶訊息
        messages.append({"role": "user", "content": message})

        # 調用 OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ 錯誤：{str(e)}"


def chat_with_claude(message: str, history: List[Tuple[str, str]]) -> str:
    """
    使用 Anthropic Claude API 進行對話

    Args:
        message: 用戶輸入的訊息
        history: 對話歷史

    Returns:
        AI 的回應
    """
    if not ANTHROPIC_API_KEY:
        return "❌ 錯誤：請設置 ANTHROPIC_API_KEY 環境變數"

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # 構建對話訊息列表
        messages = []

        # 添加歷史對話
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})

        # 添加當前用戶訊息
        messages.append({"role": "user", "content": message})

        # 調用 Claude API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages
        )

        return response.content[0].text

    except Exception as e:
        return f"❌ 錯誤：{str(e)}"


# 創建 Gradio 聊天介面
def create_demo():
    """創建 Gradio 演示介面"""

    # 定義可用的模型選項
    if OPENAI_API_KEY and ANTHROPIC_API_KEY:
        model_options = ["OpenAI GPT-3.5", "Anthropic Claude"]
        default_model = "OpenAI GPT-3.5"
    elif OPENAI_API_KEY:
        model_options = ["OpenAI GPT-3.5"]
        default_model = "OpenAI GPT-3.5"
    elif ANTHROPIC_API_KEY:
        model_options = ["Anthropic Claude"]
        default_model = "Anthropic Claude"
    else:
        model_options = ["無可用模型"]
        default_model = "無可用模型"

    def chat_wrapper(message: str, history: List[Tuple[str, str]], model: str) -> str:
        """包裝函數，根據選擇的模型調用對應的 API"""
        if model == "OpenAI GPT-3.5":
            return chat_with_openai(message, history)
        elif model == "Anthropic Claude":
            return chat_with_claude(message, history)
        else:
            return "❌ 請先設置 API 金鑰"

    # 使用 Blocks API 創建自定義介面
    with gr.Blocks(title="LLM 聊天機器人", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 LLM 聊天機器人

            這是一個基礎的聊天機器人範例，支持 OpenAI 和 Claude API。

            **設置步驟：**
            1. 設置環境變數 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`
            2. 選擇模型並開始對話

            **功能特色：**
            - ✅ 支持多個 AI 模型
            - ✅ 保留對話歷史
            - ✅ 錯誤處理
            """
        )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=model_options,
                value=default_model,
                label="選擇 AI 模型",
                info="需要對應的 API 金鑰"
            )

        chatbot = gr.Chatbot(
            label="對話窗口",
            height=400,
            show_copy_button=True
        )

        with gr.Row():
            msg = gr.Textbox(
                label="輸入訊息",
                placeholder="在這裡輸入你的問題...",
                scale=4
            )
            submit = gr.Button("發送", scale=1, variant="primary")

        with gr.Row():
            clear = gr.Button("清除對話")

        # 顯示當前 API 狀態
        api_status = gr.Markdown(
            f"""
            **API 狀態：**
            - OpenAI: {'✅ 已設置' if OPENAI_API_KEY else '❌ 未設置'}
            - Anthropic: {'✅ 已設置' if ANTHROPIC_API_KEY else '❌ 未設置'}
            """
        )

        # 範例問題
        gr.Examples(
            examples=[
                "你好！請介紹一下自己",
                "解釋什麼是機器學習",
                "寫一個 Python 函數來計算斐波那契數列",
                "給我 5 個提高工作效率的建議"
            ],
            inputs=msg,
            label="範例問題"
        )

        def respond(message, chat_history, model):
            """處理用戶輸入並返回回應"""
            if not message.strip():
                return "", chat_history

            # 獲取 AI 回應
            bot_message = chat_wrapper(message, chat_history, model)

            # 更新對話歷史
            chat_history.append((message, bot_message))

            return "", chat_history

        # 綁定事件
        submit.click(respond, [msg, chatbot, model_selector], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, model_selector], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo


if __name__ == "__main__":
    # 創建並啟動應用
    demo = create_demo()

    # 啟動選項
    demo.launch(
        share=False,  # 設為 True 可生成公開分享連結
        server_name="0.0.0.0",  # 允許外部訪問
        server_port=7860,  # 指定端口
        show_error=True  # 顯示詳細錯誤訊息
    )
