"""
Google Gemini API 基礎使用示例
展示基本對話、串流、多模態輸入和安全設定
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict
import PIL.Image
import requests
from io import BytesIO

# 載入環境變數
load_dotenv()

# 配置 API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def basic_chat():
    """基本對話示例"""
    print("\n=== 基本對話 ===")

    model = genai.GenerativeModel('gemini-1.5-pro')

    response = model.generate_content(
        "解釋什麼是大型語言模型（LLM），並說明其工作原理"
    )

    print(f"回應: {response.text}")

    # 顯示詳細資訊
    try:
        print(f"\n使用統計:")
        print(f"  - Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"  - Candidates tokens: {response.usage_metadata.candidates_token_count}")
        print(f"  - Total tokens: {response.usage_metadata.total_token_count}")
    except:
        pass

    return response


def streaming_chat():
    """串流回應示例"""
    print("\n\n=== 串流回應 ===")
    print("AI: ", end="", flush=True)

    model = genai.GenerativeModel('gemini-1.5-pro')

    response = model.generate_content(
        "寫一個 Python 程式來實現二分搜尋演算法，包含詳細註解",
        stream=True
    )

    full_response = ""
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_response += chunk.text

    print("\n")
    return full_response


def chat_session_example():
    """對話會話示例（多輪對話）"""
    print("\n\n=== 多輪對話會話 ===")

    model = genai.GenerativeModel('gemini-1.5-pro')

    # 開始對話會話
    chat = model.start_chat(history=[])

    # 第一輪
    message1 = "我想學習資料結構，請給我一些建議"
    print(f"\n用戶: {message1}")
    response1 = chat.send_message(message1)
    print(f"AI: {response1.text[:200]}...\n")

    # 第二輪（AI 會記住上下文）
    message2 = "應該先學習哪一個資料結構？"
    print(f"用戶: {message2}")
    response2 = chat.send_message(message2)
    print(f"AI: {response2.text[:200]}...\n")

    # 第三輪
    message3 = "能給我一個實作範例嗎？"
    print(f"用戶: {message3}")
    response3 = chat.send_message(message3)
    print(f"AI: {response3.text[:300]}...\n")

    # 顯示完整對話歷史
    print("=== 對話歷史 ===")
    for i, message in enumerate(chat.history):
        role = "用戶" if message.role == "user" else "AI"
        print(f"{i+1}. {role}: {message.parts[0].text[:100]}...")

    return chat


def vision_example():
    """視覺理解示例"""
    print("\n\n=== 視覺理解示例 ===")

    model = genai.GenerativeModel('gemini-1.5-pro')

    # 使用 URL 載入圖片
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    try:
        # 下載圖片
        response_img = requests.get(image_url)
        img = PIL.Image.open(BytesIO(response_img.content))

        print(f"圖片 URL: {image_url}")
        print(f"圖片尺寸: {img.size}")

        # 分析圖片
        response = model.generate_content([
            "請詳細描述這張圖片，包括：場景、主要元素、顏色、天氣和整體氛圍。",
            img
        ])

        print(f"\nAI 分析:\n{response.text}")

        return response
    except Exception as e:
        print(f"錯誤: {e}")
        return None


def multi_image_example():
    """多圖片分析示例"""
    print("\n\n=== 多圖片分析示例 ===")

    model = genai.GenerativeModel('gemini-1.5-pro')

    # 使用多個圖片 URL
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Placeholder_view_vector.svg/800px-Placeholder_view_vector.svg.png"
    ]

    try:
        images = []
        for url in image_urls:
            response = requests.get(url)
            img = PIL.Image.open(BytesIO(response.content))
            images.append(img)

        print(f"載入了 {len(images)} 張圖片")

        # 分析多張圖片
        response = model.generate_content([
            "比較這些圖片的異同，並描述它們的主要特徵。",
            *images
        ])

        print(f"\nAI 分析:\n{response.text}")

        return response
    except Exception as e:
        print(f"錯誤: {e}")
        return None


def safety_settings_example():
    """安全設定示例"""
    print("\n\n=== 安全設定示例 ===")

    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    # 配置安全設定
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        safety_settings=safety_settings
    )

    response = model.generate_content(
        "解釋網路安全中的常見威脅有哪些"
    )

    print(f"回應: {response.text[:300]}...")

    # 顯示安全評分
    print("\n安全評分:")
    try:
        for rating in response.candidates[0].safety_ratings:
            print(f"  - {rating.category.name}: {rating.probability.name}")
    except:
        print("  無安全評分資訊")

    return response


def system_instruction_example():
    """系統指令示例"""
    print("\n\n=== 系統指令示例 ===")

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        system_instruction="""你是一個專業的 Python 教師。
        當回答問題時：
        1. 使用簡單易懂的語言
        2. 提供實際的程式碼範例
        3. 解釋每個步驟的原因
        4. 指出常見的錯誤和最佳實踐"""
    )

    response = model.generate_content(
        "如何在 Python 中處理檔案讀寫？"
    )

    print(f"回應:\n{response.text}")

    return response


def json_mode_example():
    """JSON 模式示例"""
    print("\n\n=== JSON 模式示例 ===")

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        generation_config={
            "response_mime_type": "application/json"
        }
    )

    prompt = """列出 3 個 Python 學習資源，以 JSON 格式回應：
    {
        "resources": [
            {
                "name": "資源名稱",
                "type": "書籍/課程/網站",
                "difficulty": "初級/中級/高級",
                "description": "簡短描述",
                "url": "連結（如果有）"
            }
        ]
    }
    """

    response = model.generate_content(prompt)

    print(f"JSON 回應:\n{response.text}")

    # 解析 JSON
    try:
        import json
        data = json.loads(response.text)
        print("\n解析成功！")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nJSON 解析錯誤: {e}")

    return response


def parameters_example():
    """參數調整示例"""
    print("\n\n=== 參數調整示例 ===")

    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 40,
        "max_output_tokens": 1024,
    }

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        generation_config=generation_config
    )

    response = model.generate_content(
        "寫一個創意的短篇科幻故事，關於AI和人類共存的未來"
    )

    print(f"創意輸出:\n{response.text[:500]}...\n")

    print("使用參數:")
    print(f"  - Temperature: {generation_config['temperature']} (高創意)")
    print(f"  - Top-p: {generation_config['top_p']}")
    print(f"  - Top-k: {generation_config['top_k']}")

    return response


def count_tokens_example():
    """Token 計數示例"""
    print("\n\n=== Token 計數示例 ===")

    model = genai.GenerativeModel('gemini-1.5-pro')

    # 計算單一文本的 tokens
    text = "請解釋機器學習中的梯度下降演算法" * 10

    token_count = model.count_tokens(text)

    print(f"文本長度: {len(text)} 字符")
    print(f"Token 數量: {token_count.total_tokens}")

    # 計算對話的 tokens
    chat = model.start_chat(history=[])
    messages = [
        "你好",
        "我想學習 Python",
        "推薦一些資源"
    ]

    for msg in messages:
        chat.send_message(msg)

    total_tokens = model.count_tokens(chat.history)
    print(f"\n對話總 tokens: {total_tokens.total_tokens}")

    return token_count


def list_models():
    """列出可用模型"""
    print("\n\n=== 可用的 Gemini 模型 ===")

    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"\n模型名稱: {model.name}")
            print(f"  顯示名稱: {model.display_name}")
            print(f"  描述: {model.description}")
            print(f"  輸入 token 限制: {model.input_token_limit}")
            print(f"  輸出 token 限制: {model.output_token_limit}")


def main():
    """主程式"""
    print("=" * 60)
    print("Google Gemini API 基礎使用示例")
    print("=" * 60)

    try:
        # 0. 列出可用模型
        list_models()

        # 1. 基本對話
        basic_chat()

        # 2. 串流回應
        streaming_chat()

        # 3. 對話會話
        chat_session_example()

        # 4. 視覺理解
        vision_example()

        # 5. 多圖片分析
        multi_image_example()

        # 6. 安全設定
        safety_settings_example()

        # 7. 系統指令
        system_instruction_example()

        # 8. JSON 模式
        json_mode_example()

        # 9. 參數調整
        parameters_example()

        # 10. Token 計數
        count_tokens_example()

        print("\n" + "=" * 60)
        print("所有示例執行完成！")
        print("=" * 60)

        print("\nGemini 的優勢：")
        print("✓ 強大的多模態能力（文字、圖片、影片）")
        print("✓ 長上下文支援（最高 2M tokens）")
        print("✓ 免費層級慷慨")
        print("✓ 快速的回應速度")
        print("✓ 內建安全過濾")
        print("✓ 原生 JSON 輸出模式")

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n請確保：")
        print("1. 已設定 GOOGLE_API_KEY 環境變數")
        print("2. API key 有效且有足夠的配額")
        print("3. 已安裝必要套件: pip install google-generativeai pillow")
        print("4. 網路連線正常")


if __name__ == "__main__":
    main()
