"""
數據格式轉換工具
支持多種常見的 SFT 數據格式之間的轉換
"""

import json
from typing import List, Dict, Optional
from enum import Enum


class DataFormat(Enum):
    """支持的數據格式"""
    ALPACA = "alpaca"  # Alpaca 格式
    SHAREGPT = "sharegpt"  # ShareGPT 格式
    OPENAI = "openai"  # OpenAI 格式
    CONVERSATION = "conversation"  # 多輪對話格式
    SIMPLE = "simple"  # 簡單格式 (instruction, output)


class DataFormatter:
    """數據格式轉換器"""

    @staticmethod
    def alpaca_to_sharegpt(alpaca_data: List[Dict]) -> List[Dict]:
        """
        Alpaca 格式轉 ShareGPT 格式

        Alpaca: {"instruction": "...", "input": "...", "output": "..."}
        ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        """
        sharegpt_data = []

        for item in alpaca_data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            # 組合 instruction 和 input
            if input_text:
                user_message = f"{instruction}\n\n{input_text}"
            else:
                user_message = instruction

            sharegpt_item = {
                "conversations": [
                    {"from": "human", "value": user_message},
                    {"from": "gpt", "value": output}
                ]
            }

            sharegpt_data.append(sharegpt_item)

        return sharegpt_data

    @staticmethod
    def sharegpt_to_openai(sharegpt_data: List[Dict]) -> List[Dict]:
        """
        ShareGPT 格式轉 OpenAI 格式

        ShareGPT: {"conversations": [...]}
        OpenAI: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        openai_data = []

        for item in sharegpt_data:
            conversations = item.get("conversations", [])

            messages = []
            for conv in conversations:
                role = "user" if conv["from"] in ["human", "user"] else "assistant"
                messages.append({
                    "role": role,
                    "content": conv["value"]
                })

            openai_data.append({"messages": messages})

        return openai_data

    @staticmethod
    def alpaca_to_openai(alpaca_data: List[Dict]) -> List[Dict]:
        """
        Alpaca 格式轉 OpenAI 格式
        """
        sharegpt_data = DataFormatter.alpaca_to_sharegpt(alpaca_data)
        return DataFormatter.sharegpt_to_openai(sharegpt_data)

    @staticmethod
    def add_system_prompt(
        data: List[Dict],
        system_prompt: str,
        format_type: str = "openai"
    ) -> List[Dict]:
        """
        為數據添加系統提示詞

        Args:
            data: 數據列表
            system_prompt: 系統提示詞
            format_type: 數據格式類型
        """
        if format_type == "openai":
            for item in data:
                messages = item.get("messages", [])
                # 在開頭插入系統消息
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt
                })
                item["messages"] = messages

        return data

    @staticmethod
    def apply_chat_template(
        data: List[Dict],
        template_name: str = "alpaca"
    ) -> List[Dict]:
        """
        應用聊天模板

        Args:
            data: Alpaca 格式數據
            template_name: 模板名稱 ('alpaca', 'vicuna', 'chatml')
        """
        templates = {
            "alpaca": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n{output}"
            ),
            "alpaca_with_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{output}"
            ),
            "vicuna": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                "USER: {instruction}\n"
                "ASSISTANT: {output}"
            ),
            "chatml": (
                "<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n{output}<|im_end|>"
            )
        }

        formatted_data = []

        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            # 選擇模板
            if input_text and template_name == "alpaca":
                template = templates["alpaca_with_input"]
            else:
                template = templates.get(template_name, templates["alpaca"])

            # 格式化文本
            formatted_text = template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )

            formatted_data.append({
                "text": formatted_text,
                "metadata": {
                    "template": template_name,
                    "original": item
                }
            })

        return formatted_data

    @staticmethod
    def split_train_val(
        data: List[Dict],
        val_ratio: float = 0.1,
        shuffle: bool = True
    ) -> tuple[List[Dict], List[Dict]]:
        """
        分割訓練集和驗證集

        Args:
            data: 數據列表
            val_ratio: 驗證集比例
            shuffle: 是否打亂

        Returns:
            (train_data, val_data)
        """
        import random

        if shuffle:
            data = data.copy()
            random.shuffle(data)

        split_idx = int(len(data) * (1 - val_ratio))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        return train_data, val_data

    @staticmethod
    def merge_datasets(datasets: List[List[Dict]]) -> List[Dict]:
        """
        合併多個數據集

        Args:
            datasets: 數據集列表

        Returns:
            合併後的數據集
        """
        merged = []
        for dataset in datasets:
            merged.extend(dataset)

        return merged

    @staticmethod
    def filter_by_length(
        data: List[Dict],
        min_length: int = 10,
        max_length: int = 2048,
        field: str = "output"
    ) -> List[Dict]:
        """
        根據長度過濾數據

        Args:
            data: 數據列表
            min_length: 最小長度
            max_length: 最大長度
            field: 要檢查的字段

        Returns:
            過濾後的數據
        """
        filtered = []

        for item in data:
            text = item.get(field, "")
            if min_length <= len(text) <= max_length:
                filtered.append(item)

        return filtered

    @staticmethod
    def balance_dataset(
        data: List[Dict],
        category_field: str,
        max_per_category: Optional[int] = None
    ) -> List[Dict]:
        """
        平衡數據集的類別分佈

        Args:
            data: 數據列表
            category_field: 類別字段名
            max_per_category: 每個類別的最大樣本數

        Returns:
            平衡後的數據集
        """
        from collections import defaultdict
        import random

        # 按類別分組
        category_groups = defaultdict(list)
        for item in data:
            category = item.get(category_field, "unknown")
            category_groups[category].append(item)

        # 如果沒有指定最大數量，使用最小類別的數量
        if max_per_category is None:
            max_per_category = min(len(items) for items in category_groups.values())

        # 從每個類別中採樣
        balanced = []
        for category, items in category_groups.items():
            if len(items) > max_per_category:
                sampled = random.sample(items, max_per_category)
            else:
                sampled = items
            balanced.extend(sampled)

        return balanced


def convert_format(
    input_file: str,
    output_file: str,
    input_format: str,
    output_format: str
):
    """
    轉換數據格式

    Args:
        input_file: 輸入文件路徑
        output_file: 輸出文件路徑
        input_format: 輸入格式 (alpaca, sharegpt, openai)
        output_format: 輸出格式 (alpaca, sharegpt, openai)
    """
    # 讀取數據
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatter = DataFormatter()

    # 轉換格式
    if input_format == "alpaca" and output_format == "sharegpt":
        converted = formatter.alpaca_to_sharegpt(data)
    elif input_format == "alpaca" and output_format == "openai":
        converted = formatter.alpaca_to_openai(data)
    elif input_format == "sharegpt" and output_format == "openai":
        converted = formatter.sharegpt_to_openai(data)
    else:
        raise ValueError(f"不支持的轉換: {input_format} -> {output_format}")

    # 保存數據
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"已轉換 {len(converted)} 個樣本")
    print(f"從 {input_format} 格式轉換為 {output_format} 格式")
    print(f"保存到: {output_file}")


def main():
    """示例使用"""
    # 示例數據
    alpaca_data = [
        {
            "instruction": "解釋什麼是機器學習",
            "input": "",
            "output": "機器學習是人工智慧的一個分支，它使計算機能夠從數據中學習..."
        },
        {
            "instruction": "將以下文本翻譯成英文",
            "input": "你好，世界",
            "output": "Hello, World"
        }
    ]

    formatter = DataFormatter()

    # 轉換為 ShareGPT 格式
    print("轉換為 ShareGPT 格式:")
    sharegpt = formatter.alpaca_to_sharegpt(alpaca_data)
    print(json.dumps(sharegpt[0], ensure_ascii=False, indent=2))

    # 轉換為 OpenAI 格式
    print("\n轉換為 OpenAI 格式:")
    openai_format = formatter.alpaca_to_openai(alpaca_data)
    print(json.dumps(openai_format[0], ensure_ascii=False, indent=2))

    # 應用模板
    print("\n應用 Alpaca 模板:")
    templated = formatter.apply_chat_template(alpaca_data, template_name="alpaca")
    print(templated[0]["text"])

    # 分割數據集
    print("\n分割訓練集和驗證集:")
    train, val = formatter.split_train_val(alpaca_data, val_ratio=0.2)
    print(f"訓練集: {len(train)} 樣本")
    print(f"驗證集: {len(val)} 樣本")


if __name__ == "__main__":
    main()
