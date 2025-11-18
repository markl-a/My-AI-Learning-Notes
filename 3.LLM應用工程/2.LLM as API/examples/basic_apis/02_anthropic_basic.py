"""
Anthropic Claude API 基礎使用示例
展示基本對話、串流、多輪對話和視覺理解
"""

import os
import base64
import anthropic
from dotenv import load_dotenv
from typing import List, Dict

# 載入環境變數
load_dotenv()

# 初始化客戶端
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def basic_chat():
    """基本對話示例"""
    print("\n=== 基本對話 ===")

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "解釋什麼是函數式編程，並給出 Python 中的實際例子"
            }
        ]
    )

    response_text = message.content[0].text
    print(f"回應: {response_text}")

    print(f"\n使用統計:")
    print(f"  - Input tokens: {message.usage.input_tokens}")
    print(f"  - Output tokens: {message.usage.output_tokens}")
    print(f"  - 模型: {message.model}")
    print(f"  - Stop reason: {message.stop_reason}")

    return message


def streaming_chat():
    """串流回應示例"""
    print("\n\n=== 串流回應 ===")
    print("AI: ", end="", flush=True)

    full_response = ""

    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "寫一個 Python 的快速排序演算法，包含詳細註解"
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print("\n")
    return full_response


def system_prompt_example():
    """系統提示示例"""
    print("\n\n=== 使用系統提示 ===")

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="""你是一個專業的程式碼審查專家。
        當審查程式碼時，請：
        1. 指出潛在的問題和改進空間
        2. 提供具體的修改建議
        3. 解釋為什麼要這樣修改
        4. 給出改進後的程式碼""",
        messages=[
            {
                "role": "user",
                "content": """請審查這段 Python 程式碼：

```python
def get_user(id):
    user = db.query("SELECT * FROM users WHERE id = " + str(id))
    return user
```"""
            }
        ]
    )

    print(message.content[0].text)
    return message


def multi_turn_conversation():
    """多輪對話示例"""
    print("\n\n=== 多輪對話 ===")

    # Claude 的對話格式需要 user 和 assistant 交替
    messages = [
        {
            "role": "user",
            "content": "我正在學習機器學習，請推薦學習路徑"
        }
    ]

    # 第一輪
    print("\n用戶: 我正在學習機器學習，請推薦學習路徑")
    response1 = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=messages
    )

    assistant_message1 = response1.content[0].text
    print(f"AI: {assistant_message1[:200]}...\n")

    # 添加 AI 回應到對話歷史
    messages.append({
        "role": "assistant",
        "content": assistant_message1
    })

    # 第二輪
    messages.append({
        "role": "user",
        "content": "我已經了解 Python 基礎，應該先學習哪個演算法？"
    })

    print("用戶: 我已經了解 Python 基礎，應該先學習哪個演算法？")
    response2 = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=messages
    )

    assistant_message2 = response2.content[0].text
    print(f"AI: {assistant_message2[:200]}...\n")

    return messages


def vision_example():
    """視覺理解示例"""
    print("\n\n=== 視覺理解示例 ===")

    # 使用公開的圖片 URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": "請詳細描述這張圖片，包括場景、自然元素、顏色和整體氛圍。"
                        }
                    ]
                }
            ]
        )

        print(f"圖片 URL: {image_url}")
        print(f"\nAI 分析: {message.content[0].text}")

        return message
    except Exception as e:
        print(f"錯誤: {e}")
        return None


def long_context_example():
    """長文本處理示例（展示 Claude 的 200K token 上下文）"""
    print("\n\n=== 長文本處理示例 ===")

    # 模擬長文檔
    long_document = """
    # Python 最佳實踐指南

    ## 1. 程式碼風格
    遵循 PEP 8 風格指南...

    ## 2. 錯誤處理
    使用適當的異常處理...

    ## 3. 測試
    編寫單元測試和整合測試...

    """ * 20  # 重複以模擬長文檔

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""以下是一份技術文檔：

{long_document}

請總結這份文檔的主要要點，並列出 3 個最重要的建議。"""
            }
        ]
    )

    print(f"文檔長度: {len(long_document)} 字符")
    print(f"Input tokens: {message.usage.input_tokens}")
    print(f"\n摘要: {message.content[0].text}")

    return message


def thinking_example():
    """思考過程示例（使用 extended thinking）"""
    print("\n\n=== 思考過程示例 ===")

    # 注意：這需要支援 extended thinking 的模型
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": """解決這個數學問題：

                    如果一個水池有兩個進水管和一個出水管。
                    - A 管單獨開需要 3 小時注滿
                    - B 管單獨開需要 6 小時注滿
                    - C 管單獨開需要 4 小時排空

                    如果三個管子同時打開，需要多長時間注滿水池？

                    請詳細說明你的思考過程和計算步驟。"""
                }
            ],
            temperature=0  # 使用低溫度以獲得更確定的答案
        )

        print(message.content[0].text)
        return message

    except Exception as e:
        print(f"錯誤: {e}")
        return None


def structured_output_example():
    """結構化輸出示例"""
    print("\n\n=== 結構化輸出示例 ===")

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": """請分析以下程式碼並以 JSON 格式回應：

```python
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
```

請提供：
{
    "function_name": "函數名稱",
    "purpose": "函數用途",
    "complexity": "時間複雜度",
    "issues": ["潛在問題列表"],
    "improvements": ["改進建議列表"]
}

只回傳 JSON，不要其他說明文字。"""
            }
        ],
        temperature=0
    )

    response_text = message.content[0].text
    print(response_text)

    # 嘗試解析 JSON
    try:
        import json
        # 提取 JSON 部分
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            print("\n解析成功的 JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nJSON 解析提示: {e}")

    return message


def batch_processing_example():
    """批次處理示例"""
    print("\n\n=== 批次處理示例 ===")

    questions = [
        "什麼是遞迴？",
        "什麼是動態規劃？",
        "什麼是貪婪演算法？"
    ]

    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n處理問題 {i}/{len(questions)}: {question}")

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": f"用一句話解釋：{question}"
                }
            ]
        )

        answer = message.content[0].text
        print(f"回答: {answer}")

        results.append({
            "question": question,
            "answer": answer,
            "tokens": message.usage.output_tokens
        })

    print(f"\n總共處理了 {len(results)} 個問題")
    total_tokens = sum(r["tokens"] for r in results)
    print(f"總計使用 {total_tokens} 個 output tokens")

    return results


def main():
    """主程式"""
    print("=" * 60)
    print("Anthropic Claude API 基礎使用示例")
    print("=" * 60)

    try:
        # 1. 基本對話
        basic_chat()

        # 2. 串流回應
        streaming_chat()

        # 3. 系統提示
        system_prompt_example()

        # 4. 多輪對話
        multi_turn_conversation()

        # 5. 視覺理解
        vision_example()

        # 6. 長文本處理
        long_context_example()

        # 7. 思考過程
        thinking_example()

        # 8. 結構化輸出
        structured_output_example()

        # 9. 批次處理
        batch_processing_example()

        print("\n" + "=" * 60)
        print("所有示例執行完成！")
        print("=" * 60)

        print("\nClaude 的優勢：")
        print("✓ 200K token 上下文窗口")
        print("✓ 優秀的程式碼生成和分析能力")
        print("✓ Constitutional AI 安全機制")
        print("✓ 支援視覺理解")
        print("✓ 自然的多輪對話能力")

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n請確保：")
        print("1. 已設定 ANTHROPIC_API_KEY 環境變數")
        print("2. API key 有效且有足夠的配額")
        print("3. 網路連線正常")


if __name__ == "__main__":
    main()
