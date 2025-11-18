"""
Streamlit 視覺理解聊天機器人
支援圖片上傳和圖片分析（使用 GPT-4 Vision, Claude, Gemini）
"""

import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
import base64
from PIL import Image
import io
from typing import Optional

# 載入環境變數
load_dotenv()

# 頁面配置
st.set_page_config(
    page_title="AI 視覺助理",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #7c4dff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .feature-box {
        background-color: #f3e5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


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


def encode_image_base64(image_bytes) -> str:
    """將圖片編碼為 base64"""
    return base64.b64encode(image_bytes).decode('utf-8')


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """調整圖片大小以避免超過 API 限制"""
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def analyze_image_openai(client, image_bytes, prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """使用 OpenAI GPT-4 Vision 分析圖片"""
    try:
        # 編碼圖片
        base64_image = encode_image_base64(image_bytes)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
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


def analyze_image_anthropic(client, image_bytes, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Optional[str]:
    """使用 Anthropic Claude 分析圖片"""
    try:
        # 編碼圖片
        base64_image = encode_image_base64(image_bytes)

        # 檢測圖片格式
        image = Image.open(io.BytesIO(image_bytes))
        media_type = f"image/{image.format.lower()}"
        if media_type == "image/jpg":
            media_type = "image/jpeg"

        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        response = message.content[0].text
        st.markdown(response)
        return response

    except Exception as e:
        st.error(f"Anthropic 錯誤: {e}")
        return None


def analyze_image_gemini(genai_module, image_bytes, prompt: str, model_name: str = "gemini-1.5-pro") -> Optional[str]:
    """使用 Google Gemini 分析圖片"""
    try:
        # 載入圖片
        image = Image.open(io.BytesIO(image_bytes))

        model = genai_module.GenerativeModel(model_name)

        response = model.generate_content(
            [prompt, image],
            stream=True
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
    st.markdown('<h1 class="main-header">👁️ AI 視覺助理</h1>', unsafe_allow_html=True)
    st.markdown("上傳圖片，讓 AI 為你分析和理解圖片內容")

    # 初始化客戶端
    clients = init_clients()

    if not clients:
        st.error("❌ 未找到有效的 API keys。請設定環境變數。")
        return

    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 設定")

        # 選擇提供商
        available_providers = list(clients.keys())
        provider_names = {
            'openai': 'OpenAI GPT-4 Vision',
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
            'openai': ['gpt-4o', 'gpt-4o-mini'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
            'gemini': ['gemini-1.5-pro', 'gemini-1.5-flash']
        }

        model = st.selectbox(
            "選擇模型",
            model_options.get(provider, [])
        )

        st.divider()

        # 預設分析任務
        st.subheader("🎯 分析任務")

        analysis_tasks = {
            "詳細描述": "請詳細描述這張圖片的內容，包括場景、物體、顏色、氛圍等。",
            "OCR 文字識別": "請識別並提取圖片中的所有文字內容。",
            "物體檢測": "請列出圖片中的所有物體，並描述它們的位置和特徵。",
            "情感分析": "請分析圖片所傳達的情感和氛圍。",
            "藝術評論": "請從藝術角度評論這張圖片，包括構圖、色彩運用、主題等。",
            "技術分析": "請從攝影技術角度分析這張圖片，包括光線、焦距、景深等。",
            "自定義": ""
        }

        selected_task = st.selectbox(
            "選擇分析任務",
            list(analysis_tasks.keys())
        )

        if selected_task == "自定義":
            custom_prompt = st.text_area(
                "自定義提示",
                placeholder="輸入你想問關於圖片的問題...",
                height=100
            )
            current_prompt = custom_prompt
        else:
            current_prompt = analysis_tasks[selected_task]
            st.info(f"📝 提示: {current_prompt}")

        st.divider()

        # 預設範例圖片
        st.subheader("💡 範例圖片")

        example_images = {
            "自然風景": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "都市建築": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/800px-New_york_times_square-terabass.jpg",
            "人物肖像": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/TypicalPeruvianGirl.jpg/800px-TypicalPeruvianGirl.jpg"
        }

        selected_example = st.selectbox(
            "選擇範例（可選）",
            ["無"] + list(example_images.keys())
        )

        if selected_example != "無":
            st.image(example_images[selected_example], width=200)
            if st.button("使用此範例", use_container_width=True):
                st.session_state.example_url = example_images[selected_example]

        st.divider()

        # 統計
        if 'image_count' in st.session_state:
            st.metric("已分析圖片", st.session_state.image_count)

        # 清除按鈕
        if st.button("🗑️ 清除歷史", use_container_width=True):
            for key in ['messages', 'image_count']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # 初始化會話狀態
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "image_count" not in st.session_state:
        st.session_state.image_count = 0

    # 主要內容區域
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 上傳圖片")

        uploaded_file = st.file_uploader(
            "選擇圖片文件",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="支援 PNG, JPG, JPEG, WEBP 格式"
        )

        if uploaded_file:
            # 顯示上傳的圖片
            image = Image.open(uploaded_file)
            st.image(image, caption="已上傳的圖片", use_container_width=True)

            # 圖片資訊
            st.info(f"""
            **圖片資訊**
            - 格式: {image.format}
            - 尺寸: {image.size[0]} x {image.size[1]}
            - 模式: {image.mode}
            """)

            # 分析按鈕
            if st.button("🔍 開始分析", type="primary", use_container_width=True):
                if not current_prompt:
                    st.warning("請選擇分析任務或輸入自定義提示")
                else:
                    # 調整圖片大小
                    resized_image = resize_image(image)

                    # 轉換為 bytes
                    img_byte_arr = io.BytesIO()
                    resized_image.save(img_byte_arr, format=image.format or 'PNG')
                    image_bytes = img_byte_arr.getvalue()

                    # 分析圖片
                    with st.spinner("🤖 AI 正在分析圖片..."):
                        if provider == 'openai':
                            result = analyze_image_openai(
                                clients['openai'],
                                image_bytes,
                                current_prompt,
                                model
                            )
                        elif provider == 'anthropic':
                            result = analyze_image_anthropic(
                                clients['anthropic'],
                                image_bytes,
                                current_prompt,
                                model
                            )
                        elif provider == 'gemini':
                            result = analyze_image_gemini(
                                clients['gemini'],
                                image_bytes,
                                current_prompt,
                                model
                            )

                        if result:
                            # 保存到歷史
                            st.session_state.messages.append({
                                "image": image,
                                "prompt": current_prompt,
                                "result": result,
                                "provider": provider,
                                "model": model
                            })
                            st.session_state.image_count += 1
                            st.success("✅ 分析完成！")

    with col2:
        st.subheader("💬 分析結果")

        if st.session_state.messages:
            # 顯示最新的結果
            latest = st.session_state.messages[-1]

            with st.container():
                st.markdown("**分析結果：**")
                st.markdown(latest["result"])

                st.divider()

                st.markdown(f"""
                **詳細資訊：**
                - 提供商: {provider_names.get(latest['provider'], latest['provider'])}
                - 模型: {latest['model']}
                - 任務: {latest['prompt'][:50]}...
                """)

            # 歷史記錄
            if len(st.session_state.messages) > 1:
                with st.expander(f"📚 查看歷史記錄 ({len(st.session_state.messages)-1} 個)"):
                    for i, msg in enumerate(reversed(st.session_state.messages[:-1]), 1):
                        st.markdown(f"**#{len(st.session_state.messages) - i}**")
                        st.image(msg["image"], width=150)
                        st.caption(f"{msg['prompt'][:50]}...")
                        st.text(f"{msg['result'][:100]}...")
                        st.divider()
        else:
            st.info("""
            ### 👋 歡迎使用 AI 視覺助理！

            **開始步驟：**
            1. 📤 上傳一張圖片
            2. 🎯 選擇分析任務
            3. 🔍 點擊「開始分析」

            **支援的功能：**
            - 🖼️ 圖片詳細描述
            - 📝 OCR 文字識別
            - 🎯 物體檢測
            - 😊 情感分析
            - 🎨 藝術評論
            - 📷 技術分析

            **支援的格式：**
            PNG, JPG, JPEG, WEBP
            """)

    # 頁腳
    st.markdown("---")
    st.caption(f"🤖 當前使用: {provider_names.get(provider, provider)} - {model}")


if __name__ == "__main__":
    main()
