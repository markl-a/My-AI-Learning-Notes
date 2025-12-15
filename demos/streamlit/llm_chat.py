"""
LLM 聊天應用 Streamlit 演示

這個演示展示了如何使用 Streamlit 構建一個 LLM 聊天界面，
支持多種模型、對話歷史管理和參數調整。

運行方式：
    streamlit run demos/streamlit/llm_chat.py

訪問：http://localhost:8501
"""

import streamlit as st
import os
from typing import List, Dict
from datetime import datetime

# 檢查依賴
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain.callbacks.base import BaseCallbackHandler
except ImportError as e:
    st.error(f"❌ 缺少依賴：{e}")
    st.info("請安裝: pip install langchain langchain-community langchain-openai streamlit")
    st.stop()

# ==================== 頁面配置 ====================

st.set_page_config(
    page_title="LLM 聊天助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定義 CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .system-message {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 輔助函數 ====================

class StreamHandler(BaseCallbackHandler):
    """流式輸出處理器"""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def init_session_state():
    """初始化會話狀態"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0

    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0


def get_llm(model_name: str, temperature: float, max_tokens: int, streaming: bool = True):
    """獲取 LLM 實例

    Args:
        model_name: 模型名稱
        temperature: 溫度參數
        max_tokens: 最大 token 數
        streaming: 是否流式輸出

    Returns:
        LLM 實例
    """
    if model_name.startswith("gpt") or model_name.startswith("claude"):
        # OpenAI / Anthropic
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )
    else:
        # Ollama 本地模型
        return Ollama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434"
        )


def format_message(role: str, content: str) -> str:
    """格式化消息顯示

    Args:
        role: 角色（user/assistant/system）
        content: 消息內容

    Returns:
        格式化的 HTML
    """
    icon_map = {
        "user": "👤",
        "assistant": "🤖",
        "system": "⚙️"
    }

    class_map = {
        "user": "user-message",
        "assistant": "assistant-message",
        "system": "system-message"
    }

    return f"""
    <div class="chat-message {class_map[role]}">
        <strong>{icon_map[role]} {role.title()}:</strong><br>
        {content}
    </div>
    """


def export_conversation() -> str:
    """導出對話歷史

    Returns:
        Markdown 格式的對話
    """
    export_text = f"# 對話歷史\n\n"
    export_text += f"**導出時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    export_text += f"**對話輪數**: {st.session_state.conversation_count}\n\n"
    export_text += "---\n\n"

    for i, msg in enumerate(st.session_state.messages):
        role = "用戶" if msg["role"] == "user" else "助手"
        export_text += f"### {role} ({i+1})\n\n{msg['content']}\n\n"

    return export_text


# ==================== 主應用 ====================

def main():
    """主應用函數"""

    # 初始化
    init_session_state()

    # 標題
    st.markdown('<div class="main-header">🤖 LLM 聊天助手</div>', unsafe_allow_html=True)

    # ==================== 側邊欄 ====================
    with st.sidebar:
        st.header("⚙️ 設置")

        # 模型選擇
        model_provider = st.radio(
            "選擇模型提供商",
            ["OpenAI", "Ollama (本地)"],
            help="OpenAI 需要 API 金鑰"
        )

        if model_provider == "OpenAI":
            model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini"]
        else:
            model_options = ["llama3.2", "llama3.1", "phi3", "mistral"]

        selected_model = st.selectbox("選擇模型", model_options)

        # 參數調整
        st.subheader("🎛️ 模型參數")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="控制輸出的隨機性，越高越隨機"
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="最大生成 token 數"
        )

        # 系統提示詞
        st.subheader("💬 系統提示詞")
        system_prompt = st.text_area(
            "設置助手的行為",
            value="你是一個專業、友好的 AI 助手，擅長解答各種問題。",
            height=100
        )

        # 統計信息
        st.divider()
        st.subheader("📊 統計")
        st.metric("對話輪數", st.session_state.conversation_count)
        st.metric("消息數量", len(st.session_state.messages))

        # 操作按鈕
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ 清除歷史", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_count = 0
                st.rerun()

        with col2:
            if st.button("💾 導出對話", use_container_width=True):
                export_text = export_conversation()
                st.download_button(
                    label="📥 下載",
                    data=export_text,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    # ==================== 主聊天區域 ====================

    # 顯示歷史消息
    for message in st.session_state.messages:
        st.markdown(
            format_message(message["role"], message["content"]),
            unsafe_allow_html=True
        )

    # 用戶輸入
    user_input = st.chat_input("輸入您的問題...")

    if user_input:
        # 添加用戶消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(format_message("user", user_input), unsafe_allow_html=True)

        # 準備消息歷史
        messages_for_llm = []

        # 添加系統消息
        if system_prompt:
            messages_for_llm.append(SystemMessage(content=system_prompt))

        # 添加對話歷史
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages_for_llm.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_for_llm.append(AIMessage(content=msg["content"]))

        # 生成回覆
        with st.spinner("🤔 思考中..."):
            try:
                # 創建 LLM
                llm = get_llm(selected_model, temperature, max_tokens)

                # 獲取回覆
                if hasattr(llm, 'stream'):
                    # 流式輸出
                    response_container = st.empty()
                    full_response = ""

                    for chunk in llm.stream(messages_for_llm):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                        else:
                            full_response += str(chunk)
                        response_container.markdown(
                            format_message("assistant", full_response),
                            unsafe_allow_html=True
                        )

                    response = full_response
                else:
                    # 非流式輸出
                    response = llm.invoke(messages_for_llm)
                    if hasattr(response, 'content'):
                        response = response.content
                    st.markdown(format_message("assistant", response), unsafe_allow_html=True)

                # 保存回覆
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.conversation_count += 1

            except Exception as e:
                st.error(f"❌ 錯誤：{str(e)}")

                if "OPENAI_API_KEY" in str(e):
                    st.info("💡 請設置 OPENAI_API_KEY 環境變量")
                elif "Ollama" in str(e):
                    st.info("💡 請確保 Ollama 正在運行：`ollama serve`")

    # ==================== 功能標籤頁 ====================

    with st.expander("📖 使用指南"):
        st.markdown("""
        ### 如何使用

        1. **選擇模型**：在左側選擇您想使用的 LLM 模型
        2. **調整參數**：根據需求調整 temperature 和 max tokens
        3. **設置系統提示**：自定義助手的行為和風格
        4. **開始對話**：在下方輸入框輸入您的問題
        5. **查看回覆**：助手會根據您的問題生成回覆
        6. **管理歷史**：可以清除或導出對話歷史

        ### 參數說明

        - **Temperature**: 控制輸出的創造性和隨機性
          - 0.0-0.3：更保守、更一致
          - 0.4-0.7：平衡創造性和一致性
          - 0.8-2.0：更有創造性、更隨機

        - **Max Tokens**: 限制回覆的最大長度
          - 一般對話：1000-2000
          - 長文生成：3000-4000

        ### 技術棧

        - **框架**: Streamlit
        - **LLM**: OpenAI API / Ollama
        - **對話管理**: LangChain
        """)

    with st.expander("💡 示例問題"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **通用問答**
            - 什麼是機器學習？
            - 解釋深度學習的基本概念
            - RAG 系統的工作原理是什麼？

            **代碼協助**
            - 用 Python 寫一個快速排序
            - 解釋這段代碼的作用
            - 如何優化這個算法？
            """)

        with col2:
            st.markdown("""
            **學習輔助**
            - 總結這篇文章的要點
            - 幫我理解 Transformer 架構
            - 推薦 AI 學習路徑

            **創意寫作**
            - 寫一篇技術博客大綱
            - 生成項目README
            - 設計一個學習計劃
            """)


# ==================== 運行應用 ====================

if __name__ == "__main__":
    # 檢查環境
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ 未檢測到 OPENAI_API_KEY，某些功能可能無法使用")
        st.info("💡 使用 Ollama 本地模型可以避免此問題")

    main()
