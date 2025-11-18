"""
AI 輔助數據生成工具
使用 LLM API (OpenAI/Anthropic) 自動生成高質量的訓練數據
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio
from anthropic import Anthropic
import openai


@dataclass
class InstructionExample:
    """指令微調數據樣本"""
    instruction: str
    input: str
    output: str
    metadata: Optional[Dict] = None


class AIDataGenerator:
    """使用 AI 生成訓練數據"""

    def __init__(self, api_key: str = None, provider: str = "anthropic"):
        """
        初始化生成器

        Args:
            api_key: API 密鑰（如果為 None，則從環境變量讀取）
            provider: API 提供者 ('anthropic' 或 'openai')
        """
        self.provider = provider

        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
        elif provider == "openai":
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4"
        else:
            raise ValueError(f"不支持的提供者: {provider}")

    def generate_examples_from_topic(
        self,
        topic: str,
        num_examples: int = 10,
        example_types: List[str] = None
    ) -> List[InstructionExample]:
        """
        根據主題生成訓練樣本

        Args:
            topic: 主題（例如："客服對話"、"代碼生成"、"數學問題"）
            num_examples: 生成數量
            example_types: 樣本類型列表（例如：["問答", "總結", "翻譯"]）

        Returns:
            生成的訓練樣本列表
        """
        if example_types is None:
            example_types = ["問答", "解釋", "總結", "分析", "建議"]

        prompt = self._create_generation_prompt(topic, num_examples, example_types)

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        else:  # openai
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content

        return self._parse_generated_examples(content, topic)

    def _create_generation_prompt(
        self,
        topic: str,
        num_examples: int,
        example_types: List[str]
    ) -> str:
        """創建生成提示"""
        return f"""請為"{topic}"主題生成 {num_examples} 個高質量的指令微調訓練樣本。

樣本類型包括：{', '.join(example_types)}

每個樣本應該包含：
1. instruction: 明確的指令或問題
2. input: 可選的輸入上下文（如果不需要則為空）
3. output: 高質量的期望輸出

要求：
- 樣本應該多樣化，涵蓋不同的子主題和難度
- 輸出應該詳細、準確、有幫助
- 格式應該一致
- 避免重複和過於相似的樣本

請以 JSON 格式返回，格式如下：
```json
[
  {{
    "instruction": "...",
    "input": "...",
    "output": "..."
  }},
  ...
]
```

只返回 JSON，不要其他說明文字。"""

    def _parse_generated_examples(
        self,
        content: str,
        topic: str
    ) -> List[InstructionExample]:
        """解析生成的樣本"""
        # 提取 JSON 內容
        try:
            # 嘗試找到 JSON 代碼塊
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()

            examples_data = json.loads(json_str)

            examples = []
            for data in examples_data:
                example = InstructionExample(
                    instruction=data["instruction"],
                    input=data.get("input", ""),
                    output=data["output"],
                    metadata={"topic": topic, "generated": True}
                )
                examples.append(example)

            return examples

        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤: {e}")
            print(f"內容: {content}")
            return []

    def generate_variations(
        self,
        original_example: InstructionExample,
        num_variations: int = 3
    ) -> List[InstructionExample]:
        """
        為現有樣本生成變體

        Args:
            original_example: 原始樣本
            num_variations: 變體數量

        Returns:
            變體樣本列表
        """
        prompt = f"""請為以下訓練樣本生成 {num_variations} 個變體。
變體應該：
- 保持相同的主題和難度
- 使用不同的措辭和表達方式
- 提供略有不同但同樣正確的答案

原始樣本：
指令: {original_example.instruction}
輸入: {original_example.input}
輸出: {original_example.output}

請以 JSON 格式返回變體，格式與原始樣本相同。"""

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content

        return self._parse_generated_examples(content, "variation")


class DataAugmentor:
    """數據增強工具"""

    @staticmethod
    def back_translate(
        text: str,
        intermediate_lang: str = "en"
    ) -> str:
        """
        回譯增強（需要翻譯 API）
        中文 -> 英文 -> 中文
        """
        # 這裡需要集成翻譯 API
        # 示例：使用 Google Translate 或其他服務
        pass

    @staticmethod
    def paraphrase_with_llm(
        text: str,
        client: Anthropic
    ) -> str:
        """使用 LLM 重寫"""
        prompt = f"請用不同的方式重新表述以下內容，保持原意不變：\n\n{text}"

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


def save_examples(examples: List[InstructionExample], output_file: str):
    """保存樣本到 JSON 文件"""
    data = [
        {
            "instruction": ex.instruction,
            "input": ex.input,
            "output": ex.output,
            "metadata": ex.metadata
        }
        for ex in examples
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(examples)} 個樣本到 {output_file}")


def main():
    """示例使用"""
    # 初始化生成器
    generator = AIDataGenerator(provider="anthropic")

    # 生成客服對話樣本
    print("生成客服對話訓練數據...")
    customer_service_examples = generator.generate_examples_from_topic(
        topic="電商客服對話",
        num_examples=20,
        example_types=["退換貨諮詢", "物流查詢", "產品問題", "投訴處理"]
    )

    save_examples(customer_service_examples, "customer_service_data.json")

    # 生成代碼解釋樣本
    print("\n生成代碼解釋訓練數據...")
    code_examples = generator.generate_examples_from_topic(
        topic="Python 代碼解釋和調試",
        num_examples=15,
        example_types=["代碼解釋", "錯誤修復", "優化建議"]
    )

    save_examples(code_examples, "code_explanation_data.json")

    # 為樣本生成變體
    if customer_service_examples:
        print("\n生成樣本變體...")
        variations = generator.generate_variations(
            customer_service_examples[0],
            num_variations=3
        )
        save_examples(variations, "variations_data.json")


if __name__ == "__main__":
    main()
