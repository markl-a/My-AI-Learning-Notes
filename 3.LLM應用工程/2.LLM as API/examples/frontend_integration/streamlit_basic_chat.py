"""
Streamlit 基礎聊天機器人
支援多個 LLM 提供商（OpenAI, Anthropic, Gemini）
"""

import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai

# 載入環境變數
load_dotenv()

# 頁面配置
st.set_page_config(
    page_title="AI 聊天助理",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# 初始化 API 客戶端
@st.cache_resource
def init_clients():
    """初始化所有 API 客戶端"""
    clients = {}

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        clients['openai'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        clients['anthropic'] = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        clients['gemini'] = genai

    return clients


def get_openai_response(client, messages, model, temperature):
    """獲取 OpenAI 回應"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        full_response = ""
        message_placeholder = st.empty()

        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"OpenAI 錯誤: {e}")
        return None


def get_anthropic_response(client, messages, model, temperature):
    """獲取 Anthropic 回應"""
    try:
        # 轉換消息格式
        api_messages = []
        for msg in messages:
            if msg["role"] != "system":
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # 串流回應
        full_response = ""
        message_placeholder = st.empty()

        with client.messages.stream(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Anthropic 錯誤: {e}")
        return None


def get_gemini_response(genai_module, messages, model, temperature):
    """獲取 Gemini 回應"""
    try:
        model_instance = genai_module.GenerativeModel(model)

        # 轉換消息格式
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        # 串流回應
        response = model_instance.generate_content(
            prompt,
            stream=True,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 2048
            }
        )

        full_response = ""
        message_placeholder = st.empty()

        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Gemini 錯誤: {e}")
        return None


def main():
    """主應用程式"""

    # 標題
    st.markdown('<h1 class="main-header">🤖 AI 聊天助理</h1>', unsafe_allow_html=True)

    # 初始化客戶端
    clients = init_clients()

    if not clients:
        st.error("❌ 未找到有效的 API keys。請設定環境變數。")
        st.info("""
        請在 `.env` 文件中設定以下環境變數：
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - GOOGLE_API_KEY
        """)
        return

    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 設定")

        # 選擇提供商
        available_providers = list(clients.keys())
        provider_names = {
            'openai': 'OpenAI',
            'anthropic': 'Anthropic Claude',
            'gemini': 'Google Gemini'
        }

        provider = st.selectbox(
            "選擇 AI 提供商",
            available_providers,
            format_func=lambda x: provider_names.get(x, x)
        )

        # 選擇模型
        model_options = {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
            'anthropic': [
                'claude-3-5-sonnet-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229'
            ],
            'gemini': [
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-1.0-pro'
            ]
        }

        model = st.selectbox(
            "選擇模型",
            model_options.get(provider, [])
        )

        # 溫度設定
        temperature = st.slider(
            "Temperature（創意度）",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="較高的值會讓回應更有創意，較低的值會讓回應更確定"
        )

        # 系統提示
        system_prompt = st.text_area(
            "系統提示（可選）",
            value="你是一個專業且友善的 AI 助理。",
            height=100,
            help="設定 AI 的行為和角色"
        )

        st.divider()

        # 清除對話按鈕
        if st.button("🗑️ 清除對話", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # 統計資訊
        st.subheader("📊 對話統計")
        message_count = len(st.session_state.get('messages', []))
        st.metric("訊息數量", message_count)

        # 預設提示
        st.subheader("💡 建議問題")
        example_prompts = [
            "解釋量子計算的基本原理",
            "寫一個 Python 快速排序演算法",
            "介紹台灣的夜市文化",
            "如何學習機器學習？",
            "解釋什麼是 Docker"
        ]

        for prompt in example_prompts:
            if st.button(prompt, key=f"example_{prompt}", use_container_width=True):
                st.session_state.example_prompt = prompt

    # 初始化對話歷史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示對話歷史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 處理範例提示
    if "example_prompt" in st.session_state:
        prompt = st.session_state.example_prompt
        del st.session_state.example_prompt
    else:
        # 使用者輸入
        prompt = st.chat_input("輸入你的問題...")

    # 處理使用者輸入
    if prompt:
        # 添加使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成 AI 回應
        with st.chat_message("assistant"):
            # 準備消息
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(st.session_state.messages)

            # 根據提供商生成回應
            if provider == 'openai':
                response = get_openai_response(
                    clients['openai'],
                    messages,
                    model,
                    temperature
                )
            elif provider == 'anthropic':
                response = get_anthropic_response(
                    clients['anthropic'],
                    messages,
                    model,
                    temperature
                )
            elif provider == 'gemini':
                response = get_gemini_response(
                    clients['gemini'],
                    messages,
                    model,
                    temperature
                )

            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

    # 頁腳
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🤖 提供商: {provider_names.get(provider, provider)}")
    with col2:
        st.caption(f"📦 模型: {model}")
    with col3:
        st.caption(f"🌡️ Temperature: {temperature}")


if __name__ == "__main__":
    main()
