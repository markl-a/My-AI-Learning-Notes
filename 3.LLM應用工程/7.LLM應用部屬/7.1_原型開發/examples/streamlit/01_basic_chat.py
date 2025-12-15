"""
Streamlit 聊天機器人範例

這個範例展示如何使用 Streamlit 構建一個聊天機器人應用，
展示 Streamlit 的 session state 管理和聊天界面組件。

特點：
- 使用 st.chat_message 和 st.chat_input
- Session state 管理對話歷史
- 支持清除對話和下載歷史記錄
- 支持 OpenAI 和 Claude API

運行方式：
streamlit run 01_basic_chat.py

作者：AI Learning Notes
日期：2024-11
"""

import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ========== 頁面配置 ==========
st.set_page_config(
    page_title="AI 聊天機器人",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== 初始化 Session State ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    if OPENAI_API_KEY:
        st.session_state.model = "openai"
    elif ANTHROPIC_API_KEY:
        st.session_state.model = "claude"
    else:
        st.session_state.model = None


# ========== 側邊欄設置 ==========
with st.sidebar:
    st.title("⚙️ 設置")

    # 模型選擇
    model_options = []
    if OPENAI_API_KEY:
        model_options.append("OpenAI GPT-3.5")
    if ANTHROPIC_API_KEY:
        model_options.append("Anthropic Claude")

    if model_options:
        selected_model = st.selectbox(
            "選擇 AI 模型",
            model_options,
            index=0
        )

        # 更新 session state
        if "OpenAI" in selected_model:
            st.session_state.model = "openai"
        else:
            st.session_state.model = "claude"
    else:
        st.error("❌ 請設置 API 金鑰")
        st.info(
            """
            請設置以下環境變數之一：
            - `OPENAI_API_KEY`
            - `ANTHROPIC_API_KEY`
            """
        )

    st.divider()

    # 模型參數
    st.subheader("🎛️ 模型參數")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="控制回應的隨機性。值越高，回應越有創意但可能不太準確。"
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="回應的最大長度"
    )

    st.divider()

    # 系統提示詞
    st.subheader("💬 系統提示詞")
    system_prompt = st.text_area(
        "自定義 AI 的行為",
        value="你是一個友善且樂於助人的 AI 助手。",
        height=100,
        help="定義 AI 的角色和回應風格"
    )

    st.divider()

    # 對話管理
    st.subheader("📊 對話管理")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ 清除對話", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("📥 下載歷史", use_container_width=True):
            if st.session_state.messages:
                # 生成 JSON 格式的對話歷史
                chat_history = {
                    "export_time": datetime.now().isoformat(),
                    "model": st.session_state.model,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="下載 JSON",
                    data=json.dumps(chat_history, ensure_ascii=False, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # 統計信息
    st.divider()
    st.subheader("📈 統計")
    st.metric("對話輪數", len(st.session_state.messages) // 2)

    # API 狀態
    st.divider()
    st.subheader("🔌 API 狀態")
    st.markdown(
        f"""
        - OpenAI: {'✅' if OPENAI_API_KEY else '❌'}
        - Anthropic: {'✅' if ANTHROPIC_API_KEY else '❌'}
        """
    )


# ========== LLM 調用函數 ==========
def call_openai(messages_list, temperature, max_tokens):
    """調用 OpenAI API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_list,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ OpenAI API 錯誤：{str(e)}"


def call_claude(messages_list, temperature, max_tokens):
    """調用 Claude API"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Claude API 格式不同，需要分離 system message
        system_msg = ""
        claude_messages = []

        for msg in messages_list:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append(msg)

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=claude_messages
        )

        return response.content[0].text

    except Exception as e:
        return f"❌ Claude API 錯誤：{str(e)}"


def get_ai_response(user_message, temperature, max_tokens, system_prompt):
    """根據選擇的模型獲取 AI 回應"""

    # 構建訊息列表
    messages_list = [{"role": "system", "content": system_prompt}]

    # 添加歷史對話
    for msg in st.session_state.messages:
        messages_list.append({"role": msg["role"], "content": msg["content"]})

    # 添加當前用戶訊息
    messages_list.append({"role": "user", "content": user_message})

    # 調用對應的 API
    if st.session_state.model == "openai":
        return call_openai(messages_list, temperature, max_tokens)
    elif st.session_state.model == "claude":
        return call_claude(messages_list, temperature, max_tokens)
    else:
        return "❌ 請先設置 API 金鑰"


# ========== 主界面 ==========
st.title("🤖 AI 聊天機器人")
st.caption("基於 Streamlit 構建的 LLM 聊天應用")

# 顯示歡迎訊息
if len(st.session_state.messages) == 0:
    st.info(
        """
        👋 歡迎使用 AI 聊天機器人！

        **功能特色：**
        - ✅ 支持 OpenAI 和 Claude 模型
        - ✅ 可自定義系統提示詞
        - ✅ 調整模型參數（temperature, max_tokens）
        - ✅ 下載對話歷史

        **開始對話：** 在下方輸入框中輸入你的問題
        """
    )

# 顯示對話歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天輸入
if prompt := st.chat_input("在這裡輸入你的問題..."):
    # 檢查是否有可用的 API
    if not st.session_state.model:
        st.error("❌ 請先設置 API 金鑰")
        st.stop()

    # 顯示用戶訊息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 添加到歷史記錄
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 獲取 AI 回應
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = get_ai_response(prompt, temperature, max_tokens, system_prompt)
            st.markdown(response)

    # 添加到歷史記錄
    st.session_state.messages.append({"role": "assistant", "content": response})


# ========== 底部信息 ==========
st.divider()
st.caption(
    """
    **提示：** 你可以在側邊欄調整模型參數和系統提示詞來改變 AI 的行為。
    """
)
