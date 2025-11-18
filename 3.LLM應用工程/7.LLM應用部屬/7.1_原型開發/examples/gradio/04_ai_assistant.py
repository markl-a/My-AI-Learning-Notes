"""
AI 多功能助手

這個範例展示如何構建一個多功能的 AI 助手，包含：
- 📝 文本摘要
- 🌐 多語言翻譯
- 😊 情感分析
- 🔑 關鍵詞提取
- ✍️ 文本改寫
- 📊 文本分類

使用 Gradio Tabs 組織多個功能，展示 AI 輔助工具的實際應用。

運行方式：
python 04_ai_assistant.py

作者：AI Learning Notes
日期：2024-11
"""

import os
from typing import Dict, List
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def call_llm(prompt: str, system_message: str = None, model: str = "openai") -> str:
    """
    通用 LLM 調用函數

    Args:
        prompt: 用戶提示詞
        system_message: 系統提示詞
        model: 使用的模型 ("openai" 或 "claude")

    Returns:
        LLM 的回應
    """
    try:
        if model == "openai" and OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content

        elif model == "claude" and ANTHROPIC_API_KEY:
            from anthropic import Anthropic
            client = Anthropic(api_key=ANTHROPIC_API_KEY)

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                system=system_message if system_message else "",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        else:
            return "❌ 錯誤：請設置對應的 API 金鑰"

    except Exception as e:
        return f"❌ 錯誤：{str(e)}"


# ========== 功能 1：文本摘要 ==========
def summarize_text(text: str, length: str, model: str) -> str:
    """
    生成文本摘要

    Args:
        text: 要摘要的文本
        length: 摘要長度 (短/中/長)
        model: 使用的模型

    Returns:
        摘要結果
    """
    if not text.strip():
        return "❌ 請輸入要摘要的文本"

    length_map = {
        "短（1-2 句）": "用 1-2 句話",
        "中（3-5 句）": "用 3-5 句話",
        "長（一段）": "用一段文字（約 100-150 字）"
    }

    prompt = f"""請{length_map[length]}總結以下文本的核心內容：

文本：
{text}

摘要："""

    system_message = "你是一個專業的文本摘要助手，擅長提取關鍵信息並生成簡潔的摘要。"

    return call_llm(prompt, system_message, model)


# ========== 功能 2：多語言翻譯 ==========
def translate_text(text: str, target_lang: str, model: str) -> str:
    """
    翻譯文本到目標語言

    Args:
        text: 要翻譯的文本
        target_lang: 目標語言
        model: 使用的模型

    Returns:
        翻譯結果
    """
    if not text.strip():
        return "❌ 請輸入要翻譯的文本"

    prompt = f"""請將以下文本翻譯成{target_lang}，保持原意並使用自然流暢的表達：

原文：
{text}

譯文："""

    system_message = "你是一個專業的翻譯助手，精通多種語言。"

    return call_llm(prompt, system_message, model)


# ========== 功能 3：情感分析 ==========
def analyze_sentiment(text: str, model: str) -> str:
    """
    分析文本的情感傾向

    Args:
        text: 要分析的文本
        model: 使用的模型

    Returns:
        情感分析結果
    """
    if not text.strip():
        return "❌ 請輸入要分析的文本"

    prompt = f"""請分析以下文本的情感傾向，並提供：
1. 總體情感（正面/中性/負面）
2. 情感強度（1-10 分）
3. 主要情緒（如：喜悅、憤怒、悲傷等）
4. 簡要說明

文本：
{text}

分析結果："""

    system_message = "你是一個專業的情感分析專家，能夠準確識別文本中的情感和情緒。"

    return call_llm(prompt, system_message, model)


# ========== 功能 4：關鍵詞提取 ==========
def extract_keywords(text: str, num_keywords: int, model: str) -> str:
    """
    從文本中提取關鍵詞

    Args:
        text: 要分析的文本
        num_keywords: 提取的關鍵詞數量
        model: 使用的模型

    Returns:
        關鍵詞列表
    """
    if not text.strip():
        return "❌ 請輸入要分析的文本"

    prompt = f"""請從以下文本中提取 {num_keywords} 個最重要的關鍵詞或短語，
並為每個關鍵詞簡短說明其重要性。

文本：
{text}

關鍵詞："""

    system_message = "你是一個專業的文本分析專家，擅長識別文本中的核心概念和關鍵信息。"

    return call_llm(prompt, system_message, model)


# ========== 功能 5：文本改寫 ==========
def rewrite_text(text: str, style: str, model: str) -> str:
    """
    以不同風格改寫文本

    Args:
        text: 要改寫的文本
        style: 改寫風格
        model: 使用的模型

    Returns:
        改寫後的文本
    """
    if not text.strip():
        return "❌ 請輸入要改寫的文本"

    style_instructions = {
        "正式專業": "改寫成正式、專業的商務語言",
        "輕鬆口語": "改寫成輕鬆、口語化的表達方式",
        "學術論文": "改寫成學術論文的嚴謹風格",
        "創意文案": "改寫成吸引人的創意文案",
        "簡潔精煉": "改寫得更加簡潔精煉，去除冗餘"
    }

    prompt = f"""請{style_instructions[style]}。保持原意但改變表達方式。

原文：
{text}

改寫："""

    system_message = "你是一個專業的文字編輯，擅長以不同風格改寫文本。"

    return call_llm(prompt, system_message, model)


# ========== 功能 6：文本分類 ==========
def classify_text(text: str, categories: str, model: str) -> str:
    """
    將文本分類到指定類別

    Args:
        text: 要分類的文本
        categories: 候選類別（逗號分隔）
        model: 使用的模型

    Returns:
        分類結果
    """
    if not text.strip():
        return "❌ 請輸入要分類的文本"

    if not categories.strip():
        return "❌ 請輸入候選類別"

    prompt = f"""請將以下文本分類到最合適的類別，並說明理由。

候選類別：{categories}

文本：
{text}

分類結果："""

    system_message = "你是一個專業的文本分類專家。"

    return call_llm(prompt, system_message, model)


# ========== 創建 Gradio 界面 ==========
def create_demo():
    """創建多標籤頁的 AI 助手界面"""

    # 確定可用模型
    if OPENAI_API_KEY and ANTHROPIC_API_KEY:
        model_options = ["openai", "claude"]
    elif OPENAI_API_KEY:
        model_options = ["openai"]
    elif ANTHROPIC_API_KEY:
        model_options = ["claude"]
    else:
        model_options = []

    with gr.Blocks(title="AI 多功能助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🛠️ AI 多功能助手

            一個集成多種 AI 輔助功能的工具箱，幫助你高效處理文本任務。

            **功能列表：**
            📝 文本摘要 | 🌐 翻譯 | 😊 情感分析 | 🔑 關鍵詞提取 | ✍️ 文本改寫 | 📊 文本分類
            """
        )

        # 全局模型選擇器
        with gr.Row():
            global_model = gr.Radio(
                choices=model_options,
                value=model_options[0] if model_options else None,
                label="🤖 選擇 AI 模型",
                info="OpenAI 或 Claude"
            )

        with gr.Tabs():
            # ===== Tab 1: 文本摘要 =====
            with gr.Tab("📝 文本摘要"):
                gr.Markdown("### 自動生成文本摘要，快速了解核心內容")

                with gr.Row():
                    with gr.Column(scale=2):
                        summary_input = gr.Textbox(
                            label="輸入文本",
                            placeholder="貼上你想要摘要的文章、報告或任何長文本...",
                            lines=10
                        )
                        summary_length = gr.Radio(
                            choices=["短（1-2 句）", "中（3-5 句）", "長（一段）"],
                            value="中（3-5 句）",
                            label="摘要長度"
                        )
                        summary_btn = gr.Button("生成摘要", variant="primary")

                    with gr.Column(scale=2):
                        summary_output = gr.Textbox(
                            label="摘要結果",
                            lines=10,
                            show_copy_button=True
                        )

                gr.Examples(
                    examples=[
                        ["人工智能（AI）正在改變我們的世界。從醫療診斷到自動駕駛，從語言翻譯到創意生成，AI 的應用已經滲透到生活的方方面面。然而，隨著技術的快速發展，我們也面臨著隱私保護、就業影響、算法偏見等挑戰。如何在享受 AI 帶來的便利的同時，確保技術的負責任發展，是我們這個時代需要共同思考的重要問題。"]
                    ],
                    inputs=summary_input
                )

                summary_btn.click(
                    summarize_text,
                    inputs=[summary_input, summary_length, global_model],
                    outputs=summary_output
                )

            # ===== Tab 2: 多語言翻譯 =====
            with gr.Tab("🌐 多語言翻譯"):
                gr.Markdown("### 支持多種語言的高質量翻譯")

                with gr.Row():
                    with gr.Column(scale=2):
                        translate_input = gr.Textbox(
                            label="輸入文本",
                            placeholder="輸入要翻譯的文本...",
                            lines=8
                        )
                        target_lang = gr.Dropdown(
                            choices=["英文", "中文（繁體）", "中文（簡體）", "日文", "韓文", "法文", "德文", "西班牙文"],
                            value="英文",
                            label="目標語言"
                        )
                        translate_btn = gr.Button("翻譯", variant="primary")

                    with gr.Column(scale=2):
                        translate_output = gr.Textbox(
                            label="翻譯結果",
                            lines=8,
                            show_copy_button=True
                        )

                translate_btn.click(
                    translate_text,
                    inputs=[translate_input, target_lang, global_model],
                    outputs=translate_output
                )

            # ===== Tab 3: 情感分析 =====
            with gr.Tab("😊 情感分析"):
                gr.Markdown("### 分析文本的情感傾向和情緒強度")

                with gr.Row():
                    with gr.Column(scale=2):
                        sentiment_input = gr.Textbox(
                            label="輸入文本",
                            placeholder="輸入要分析情感的文本（評論、社交媒體內容等）...",
                            lines=8
                        )
                        sentiment_btn = gr.Button("分析情感", variant="primary")

                    with gr.Column(scale=2):
                        sentiment_output = gr.Textbox(
                            label="分析結果",
                            lines=8,
                            show_copy_button=True
                        )

                gr.Examples(
                    examples=[
                        ["這部電影真是太棒了！劇情緊湊，演員表現出色，視覺效果令人驚嘆。強烈推薦！"],
                        ["今天的服務實在太差了，等了一個小時還沒上菜，態度還很惡劣。非常失望。"],
                        ["產品質量還可以，但價格偏高，性價比一般般。"]
                    ],
                    inputs=sentiment_input
                )

                sentiment_btn.click(
                    analyze_sentiment,
                    inputs=[sentiment_input, global_model],
                    outputs=sentiment_output
                )

            # ===== Tab 4: 關鍵詞提取 =====
            with gr.Tab("🔑 關鍵詞提取"):
                gr.Markdown("### 從文本中提取核心關鍵詞和概念")

                with gr.Row():
                    with gr.Column(scale=2):
                        keywords_input = gr.Textbox(
                            label="輸入文本",
                            placeholder="輸入文章、報告或任何文本...",
                            lines=8
                        )
                        num_keywords = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=5,
                            step=1,
                            label="關鍵詞數量"
                        )
                        keywords_btn = gr.Button("提取關鍵詞", variant="primary")

                    with gr.Column(scale=2):
                        keywords_output = gr.Textbox(
                            label="關鍵詞結果",
                            lines=8,
                            show_copy_button=True
                        )

                keywords_btn.click(
                    extract_keywords,
                    inputs=[keywords_input, num_keywords, global_model],
                    outputs=keywords_output
                )

            # ===== Tab 5: 文本改寫 =====
            with gr.Tab("✍️ 文本改寫"):
                gr.Markdown("### 以不同風格改寫文本，保持原意")

                with gr.Row():
                    with gr.Column(scale=2):
                        rewrite_input = gr.Textbox(
                            label="輸入文本",
                            placeholder="輸入要改寫的文本...",
                            lines=8
                        )
                        rewrite_style = gr.Radio(
                            choices=["正式專業", "輕鬆口語", "學術論文", "創意文案", "簡潔精煉"],
                            value="正式專業",
                            label="改寫風格"
                        )
                        rewrite_btn = gr.Button("改寫", variant="primary")

                    with gr.Column(scale=2):
                        rewrite_output = gr.Textbox(
                            label="改寫結果",
                            lines=8,
                            show_copy_button=True
                        )

                rewrite_btn.click(
                    rewrite_text,
                    inputs=[rewrite_input, rewrite_style, global_model],
                    outputs=rewrite_output
                )

            # ===== Tab 6: 文本分類 =====
            with gr.Tab("📊 文本分類"):
                gr.Markdown("### 將文本自動分類到指定類別")

                with gr.Row():
                    with gr.Column(scale=2):
                        classify_input = gr.Textbox(
                            label="輸入文本",
                            placeholder="輸入要分類的文本...",
                            lines=6
                        )
                        classify_categories = gr.Textbox(
                            label="候選類別（逗號分隔）",
                            placeholder="例如：科技, 體育, 娛樂, 政治, 經濟",
                            value="科技, 體育, 娛樂, 政治, 經濟"
                        )
                        classify_btn = gr.Button("分類", variant="primary")

                    with gr.Column(scale=2):
                        classify_output = gr.Textbox(
                            label="分類結果",
                            lines=8,
                            show_copy_button=True
                        )

                classify_btn.click(
                    classify_text,
                    inputs=[classify_input, classify_categories, global_model],
                    outputs=classify_output
                )

        # API 狀態
        with gr.Accordion("📊 API 狀態", open=False):
            gr.Markdown(
                f"""
                | 提供商 | 狀態 | 可用模型 |
                |--------|------|----------|
                | OpenAI | {'✅ 已配置' if OPENAI_API_KEY else '❌ 未配置'} | GPT-3.5 Turbo |
                | Anthropic | {'✅ 已配置' if ANTHROPIC_API_KEY else '❌ 未配置'} | Claude 3.5 Sonnet |

                **設置方法：**
                ```bash
                export OPENAI_API_KEY="your-key"
                export ANTHROPIC_API_KEY="your-key"
                ```
                """
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
