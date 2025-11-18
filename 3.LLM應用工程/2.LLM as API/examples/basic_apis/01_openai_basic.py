"""
OpenAI API 基礎使用示例
展示基本對話、串流、函數呼叫和結構化輸出
"""

import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 初始化客戶端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def basic_chat():
    """基本對話示例"""
    print("\n=== 基本對話 ===")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一個專業的 Python 程式設計助理。"},
            {"role": "user", "content": "解釋什麼是裝飾器（decorator）並給一個實用的例子"}
        ],
        temperature=0.7,
        max_tokens=500
    )

    print(f"回應: {response.choices[0].message.content}")
    print(f"\n使用 tokens: {response.usage.total_tokens}")
    print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  - Completion tokens: {response.usage.completion_tokens}")

    return response


def streaming_chat():
    """串流回應示例"""
    print("\n\n=== 串流回應 ===")
    print("AI: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "用一段話介紹台灣的夜市文化"}
        ],
        stream=True,
        temperature=0.7
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print("\n")
    return full_response


def function_calling_example():
    """函數呼叫示例"""
    print("\n=== 函數呼叫 ===")

    # 定義可用的函數
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "獲取指定城市的天氣資訊",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名稱，例如：台北、高雄"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "溫度單位"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "執行數學計算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "數學表達式，例如：'2 + 2' 或 '10 * 5'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # 模擬天氣函數
    def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
        """模擬獲取天氣數據"""
        # 實際應用中應該調用真實的天氣 API
        weather_data = {
            "台北": {"temperature": 28, "condition": "多雲", "humidity": 75},
            "高雄": {"temperature": 32, "condition": "晴天", "humidity": 65},
        }

        data = weather_data.get(location, {"temperature": 25, "condition": "未知", "humidity": 70})

        if unit == "fahrenheit":
            data["temperature"] = data["temperature"] * 9/5 + 32
            data["unit"] = "°F"
        else:
            data["unit"] = "°C"

        return data

    def calculate(expression: str) -> float:
        """安全地計算數學表達式"""
        try:
            # 注意：在生產環境中應該使用更安全的方法
            return eval(expression, {"__builtins__": {}})
        except:
            return "計算錯誤"

    # 可用的函數映射
    available_functions = {
        "get_weather": get_weather,
        "calculate": calculate
    }

    # 用戶查詢
    user_query = "台北現在天氣如何？另外幫我算 123 * 456"
    print(f"用戶問題: {user_query}")

    messages = [{"role": "user", "content": user_query}]

    # 第一次 API 呼叫
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 檢查是否需要呼叫函數
    if tool_calls:
        messages.append(response_message)

        # 執行所有函數呼叫
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n呼叫函數: {function_name}")
            print(f"參數: {function_args}")

            # 執行函數
            function_response = available_functions[function_name](**function_args)

            print(f"結果: {function_response}")

            # 添加函數回應到對話
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False)
            })

        # 第二次 API 呼叫，獲取最終回應
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        final_answer = second_response.choices[0].message.content
        print(f"\n最終回應: {final_answer}")

        return final_answer

    return response_message.content


def structured_output():
    """結構化輸出示例（JSON 模式）"""
    print("\n\n=== 結構化輸出 ===")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "你是一個旅遊助理。以 JSON 格式提供景點資訊。"
            },
            {
                "role": "user",
                "content": """列出 3 個台灣著名景點，包含以下資訊：
                - name: 景點名稱
                - location: 所在城市
                - description: 簡短描述（不超過50字）
                - category: 類別（自然景觀/歷史古蹟/現代建築/夜市美食）
                - rating: 推薦評分（1-5）

                以 JSON 格式回應，包含一個 attractions 陣列。"""
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )

    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def multi_turn_conversation():
    """多輪對話示例"""
    print("\n\n=== 多輪對話 ===")

    # 維護對話歷史
    conversation_history = [
        {"role": "system", "content": "你是一個友善的 AI 助理，擅長記住對話上下文。"}
    ]

    # 模擬多輪對話
    user_inputs = [
        "我想學習 Python",
        "我應該從哪裡開始？",
        "推薦一些初學者專案"
    ]

    for user_input in user_inputs:
        print(f"\n用戶: {user_input}")

        # 添加用戶消息到歷史
        conversation_history.append({"role": "user", "content": user_input})

        # 生成回應
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=200
        )

        assistant_message = response.choices[0].message.content
        print(f"AI: {assistant_message}")

        # 添加 AI 回應到歷史
        conversation_history.append({"role": "assistant", "content": assistant_message})

    return conversation_history


def vision_example():
    """視覺理解示例（GPT-4 Vision）"""
    print("\n\n=== 視覺理解示例 ===")

    # 使用公開的圖片 URL 作為示例
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "請詳細描述這張圖片的內容，包括場景、顏色、氛圍等。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        print(f"圖片 URL: {image_url}")
        print(f"\nAI 分析: {response.choices[0].message.content}")

        return response
    except Exception as e:
        print(f"視覺理解功能需要 GPT-4o 或 GPT-4o-mini 模型")
        print(f"錯誤: {e}")
        return None


def main():
    """主程式"""
    print("=" * 60)
    print("OpenAI API 基礎使用示例")
    print("=" * 60)

    try:
        # 1. 基本對話
        basic_chat()

        # 2. 串流回應
        streaming_chat()

        # 3. 函數呼叫
        function_calling_example()

        # 4. 結構化輸出
        structured_output()

        # 5. 多輪對話
        multi_turn_conversation()

        # 6. 視覺理解
        vision_example()

        print("\n" + "=" * 60)
        print("所有示例執行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n請確保：")
        print("1. 已設定 OPENAI_API_KEY 環境變數")
        print("2. API key 有效且有足夠的配額")
        print("3. 網路連線正常")


if __name__ == "__main__":
    main()
