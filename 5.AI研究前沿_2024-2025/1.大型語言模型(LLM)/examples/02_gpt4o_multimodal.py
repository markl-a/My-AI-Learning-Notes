"""
GPT-4o 多模態應用
支持文字、圖片、音頻的統一處理
包含視覺理解、工具調用、結構化輸出等功能
"""

from openai import OpenAI
from typing import List, Dict, Optional, Union
import base64
import json
from pathlib import Path
import requests


class GPT4oClient:
    """GPT-4o 客戶端封裝"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 GPT-4o 客戶端

        Args:
            api_key: OpenAI API Key（可從環境變數讀取）
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        print(f"GPT-4o 客戶端初始化完成")
        print(f"模型: {self.model}")
        print(f"支持: 文字、圖片、音頻")

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        基本對話

        Args:
            messages: 對話訊息
            temperature: 溫度
            max_tokens: 最大token數
            response_format: 回應格式（如 {"type": "json_object"}）

        Returns:
            回覆內容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )

        return response.choices[0].message.content

    def vision_analysis(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt: str = "請描述這張圖片",
        detail: str = "auto"
    ) -> str:
        """
        視覺理解

        Args:
            image_path: 本地圖片路徑
            image_url: 圖片URL
            prompt: 提示詞
            detail: 細節層級 (low, high, auto)

        Returns:
            分析結果
        """
        # 準備圖片
        if image_path:
            image_data = self._encode_image(image_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": detail
                }
            }
        elif image_url:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": detail
                }
            }
        else:
            raise ValueError("必須提供 image_path 或 image_url")

        # 構建訊息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        ]

        return self.chat(messages)

    def function_calling(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: str = "auto"
    ) -> Dict:
        """
        工具調用

        Args:
            messages: 對話訊息
            tools: 工具定義
            tool_choice: 工具選擇策略 (auto, required, none)

        Returns:
            包含工具調用和結果的完整回應
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )

        message = response.choices[0].message

        # 檢查是否有工具調用
        if message.tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in message.tool_calls
                ],
                "message": message
            }
        else:
            return {
                "type": "text",
                "content": message.content
            }

    def structured_output(
        self,
        prompt: str,
        schema: Dict
    ) -> Dict:
        """
        結構化輸出（JSON Schema）

        Args:
            prompt: 提示詞
            schema: JSON Schema

        Returns:
            結構化數據
        """
        messages = [
            {
                "role": "system",
                "content": "你是一個數據提取助手，請按照指定格式輸出JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.chat(
            messages,
            response_format={"type": "json_object"}
        )

        return json.loads(response)

    def multi_image_analysis(
        self,
        images: List[Union[str, Dict]],
        prompt: str = "比較這些圖片的異同"
    ) -> str:
        """
        多圖片分析

        Args:
            images: 圖片列表（路徑或URL）
            prompt: 提示詞

        Returns:
            分析結果
        """
        content = [{"type": "text", "text": prompt}]

        for img in images:
            if isinstance(img, str):
                if img.startswith("http"):
                    # URL
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                else:
                    # 本地文件
                    image_data = self._encode_image(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })

        messages = [{"role": "user", "content": content}]
        return self.chat(messages)

    def _encode_image(self, image_path: str) -> str:
        """將圖片編碼為 base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def example_basic_chat():
    """示例 1: 基本對話"""
    print("=== 示例 1: GPT-4o 基本對話 ===\n")

    client = GPT4oClient()

    messages = [
        {"role": "user", "content": "解釋一下 GPT-4o 相比 GPT-4 的主要改進"}
    ]

    response = client.chat(messages)
    print(f"用戶: {messages[0]['content']}")
    print(f"GPT-4o: {response}\n")


def example_vision_understanding():
    """示例 2: 視覺理解"""
    print("=== 示例 2: 圖片理解 ===\n")

    client = GPT4oClient()

    # 使用線上圖片
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = client.vision_analysis(
        image_url=image_url,
        prompt="詳細描述這張圖片中的場景、物體和氛圍",
        detail="high"
    )

    print(f"圖片URL: {image_url}")
    print(f"分析結果: {response}\n")


def example_ocr():
    """示例 3: 光學字符識別（OCR）"""
    print("=== 示例 3: OCR 文字識別 ===\n")

    client = GPT4oClient()

    # 假設有一張包含文字的圖片
    response = client.vision_analysis(
        image_url="https://example.com/document.jpg",
        prompt="請提取圖片中的所有文字，保持原有格式"
    )

    print(f"識別的文字:\n{response}\n")


def example_function_calling():
    """示例 4: 工具調用"""
    print("=== 示例 4: Function Calling ===\n")

    client = GPT4oClient()

    # 定義工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "獲取指定城市的當前天氣",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名稱，例如：台北、東京"
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
                "name": "search_restaurants",
                "description": "搜索附近的餐廳",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "cuisine": {"type": "string"},
                        "price_range": {"type": "string", "enum": ["$", "$$", "$$$"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "台北現在的天氣如何？"}
    ]

    result = client.function_calling(messages, tools)

    print(f"用戶: {messages[0]['content']}")
    print(f"工具調用結果: {json.dumps(result, ensure_ascii=False, indent=2)}\n")


def example_structured_output():
    """示例 5: 結構化輸出"""
    print("=== 示例 5: 結構化數據提取 ===\n")

    client = GPT4oClient()

    prompt = """
從以下文本中提取信息：

"Apple 公司在 2024 年第一季度營收達到 900 億美元，
同比增長 5%，iPhone 銷售量為 5000 萬台。
CEO Tim Cook 表示對未來充滿信心。"

請提取：公司名稱、時間、營收、增長率、產品、銷售量、CEO姓名
"""

    schema = {
        "company": "string",
        "period": "string",
        "revenue": "string",
        "growth_rate": "string",
        "product": "string",
        "sales_volume": "string",
        "ceo": "string"
    }

    result = client.structured_output(prompt, schema)

    print("提取的結構化數據:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def example_multi_image():
    """示例 6: 多圖片比較"""
    print("=== 示例 6: 多圖片分析 ===\n")

    client = GPT4oClient()

    images = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg"
    ]

    response = client.multi_image_analysis(
        images=images,
        prompt="比較這三張圖片，找出相同點和不同點"
    )

    print(f"分析 {len(images)} 張圖片")
    print(f"比較結果: {response}\n")


def example_code_analysis():
    """示例 7: 代碼分析"""
    print("=== 示例 7: 代碼理解與優化 ===\n")

    client = GPT4oClient()

    code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

    messages = [
        {
            "role": "user",
            "content": f"分析以下代碼的時間複雜度，並提供優化建議：\n\n```python\n{code}\n```"
        }
    ]

    response = client.chat(messages)
    print(f"代碼分析:\n{response}\n")


def example_json_mode():
    """示例 8: JSON 模式"""
    print("=== 示例 8: JSON 模式輸出 ===\n")

    client = GPT4oClient()

    messages = [
        {
            "role": "system",
            "content": "你是一個數據提取助手，以JSON格式輸出。"
        },
        {
            "role": "user",
            "content": """提取以下產品的信息：

產品：iPhone 15 Pro
價格：NT$ 36,900
顏色：自然鈦金屬
儲存：256GB
特色：A17 Pro 晶片、鈦金屬設計、Pro 相機系統

以JSON格式輸出。"""
        }
    ]

    response = client.chat(
        messages,
        response_format={"type": "json_object"}
    )

    print("JSON 輸出:")
    print(json.dumps(json.loads(response), ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    print("GPT-4o 多模態應用示例")
    print("=" * 60)
    print()

    # 注意：需要設置 OPENAI_API_KEY 環境變數
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 請設置 OPENAI_API_KEY 環境變數")
        print("export OPENAI_API_KEY='your-api-key'\n")

    try:
        # 運行示例（需要有效的 API Key）
        example_basic_chat()
        # example_vision_understanding()
        # example_function_calling()
        # example_structured_output()
        # example_json_mode()
        # example_code_analysis()

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n注意:")
        print("1. 需要有效的 OpenAI API Key")
        print("2. GPT-4o 為付費服務")
        print("3. 確保網絡連接正常")

    print("\nGPT-4o 特性:")
    print("✓ 多模態統一模型（文字、圖片、音頻）")
    print("✓ 128K 上下文長度")
    print("✓ 原生視覺理解能力")
    print("✓ 強大的工具調用")
    print("✓ 結構化輸出（JSON Schema）")
    print("✓ 速度快、成本低（相比GPT-4 Turbo）")

    print("\n定價（參考）:")
    print("- 輸入: $5 / 1M tokens")
    print("- 輸出: $15 / 1M tokens")
    print("- 圖片: 根據尺寸和細節級別計費")

    print("\n使用建議:")
    print("1. 視覺任務優先使用 GPT-4o")
    print("2. 需要實時響應時使用 GPT-4o")
    print("3. 成本敏感時選擇 GPT-4o mini")
    print("4. 複雜推理任務可考慮 o1 系列")
