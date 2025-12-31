"""
Llama 3.1 完整使用指南
包含推理、微調、工具調用等功能
支持 8B、70B、405B 模型
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from typing import List, Dict, Optional, Callable
import json


class Llama31Model:
    """Llama 3.1 模型封裝"""

    def __init__(
        self,
        model_size: str = "8b",
        quantization: Optional[str] = None,
        device: str = "auto"
    ):
        """
        初始化 Llama 3.1 模型

        Args:
            model_size: 模型大小 (8b, 70b, 405b)
            quantization: 量化方式 (4bit, 8bit, None)
            device: 設備 (auto, cuda, cpu)
        """
        # 模型映射
        model_map = {
            "8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "405b": "meta-llama/Meta-Llama-3.1-405B-Instruct"
        }

        if model_size not in model_map:
            raise ValueError(f"不支持的模型大小: {model_size}")

        self.model_id = model_map[model_size]
        self.model_size = model_size
        print(f"正在載入 Llama 3.1 {model_size.upper()} 模型...")

        # 量化配置
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("使用 4-bit 量化")
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("使用 8-bit 量化")

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map=device,
            torch_dtype=torch.bfloat16 if quantization is None else None,
            trust_remote_code=True
        )

        print(f"模型載入完成！")
        print(f"上下文長度: 128K tokens")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str:
        """
        聊天對話

        Args:
            messages: 對話歷史 [{"role": "user", "content": "..."}]
            max_new_tokens: 最大生成長度
            temperature: 溫度
            top_p: Top-P 採樣
            stream: 是否流式輸出

        Returns:
            回覆文本
        """
        # 應用聊天模板
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解碼（只返回新生成的部分）
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response

    def function_calling(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        max_iterations: int = 5
    ) -> Dict:
        """
        工具調用（Function Calling）

        Args:
            messages: 對話歷史
            tools: 可用工具定義
            max_iterations: 最大迭代次數

        Returns:
            最終結果
        """
        # 構建工具描述
        tools_description = self._format_tools(tools)

        # 添加系統提示
        system_message = {
            "role": "system",
            "content": f"""你是一個helpful assistant，可以使用以下工具：

{tools_description}

當需要使用工具時，請以JSON格式回覆：
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}

當不需要使用工具時，直接回覆用戶。"""
        }

        full_messages = [system_message] + messages
        conversation_history = list(full_messages)

        for iteration in range(max_iterations):
            # 生成回覆
            response = self.chat(conversation_history)

            # 檢查是否是工具調用
            tool_call = self._parse_tool_call(response)

            if tool_call is None:
                # 不是工具調用，直接返回
                return {
                    "type": "text",
                    "content": response,
                    "iterations": iteration + 1
                }

            # 執行工具
            tool_result = self._execute_tool(tool_call, tools)

            # 添加到對話歷史
            conversation_history.append({
                "role": "assistant",
                "content": response
            })
            conversation_history.append({
                "role": "user",
                "content": f"工具執行結果: {json.dumps(tool_result, ensure_ascii=False)}"
            })

        return {
            "type": "error",
            "content": "達到最大迭代次數",
            "iterations": max_iterations
        }

    def _format_tools(self, tools: List[Dict]) -> str:
        """格式化工具描述"""
        formatted = []
        for tool in tools:
            formatted.append(f"- {tool['name']}: {tool['description']}")
            if 'parameters' in tool:
                formatted.append(f"  參數: {tool['parameters']}")
        return "\n".join(formatted)

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """解析工具調用"""
        try:
            # 嘗試找到 JSON 格式的工具調用
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                tool_call = json.loads(json_str)
                if 'tool' in tool_call:
                    return tool_call
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _execute_tool(self, tool_call: Dict, tools: List[Dict]) -> Dict:
        """執行工具"""
        tool_name = tool_call.get('tool')

        # 查找工具
        for tool in tools:
            if tool['name'] == tool_name:
                if 'function' in tool:
                    # 執行函數
                    try:
                        result = tool['function'](**tool_call.get('arguments', {}))
                        return {"status": "success", "result": result}
                    except Exception as e:
                        return {"status": "error", "error": str(e)}

        return {"status": "error", "error": f"工具 {tool_name} 不存在"}


def example_basic_chat():
    """示例 1: 基本對話"""
    print("=== 示例 1: Llama 3.1 基本對話 ===\n")

    # 使用 8B 模型（較快）
    llm = Llama31Model(model_size="8b", quantization="4bit")

    # 簡單對話
    messages = [
        {"role": "user", "content": "什麼是 Transformer 架構？"}
    ]

    response = llm.chat(messages)
    print(f"用戶: {messages[0]['content']}")
    print(f"Llama 3.1: {response}\n")


def example_multi_turn_chat():
    """示例 2: 多輪對話"""
    print("=== 示例 2: 多輪對話 ===\n")

    llm = Llama31Model(model_size="8b", quantization="4bit")

    # 多輪對話
    conversation = [
        {"role": "user", "content": "請介紹一下 Python 的裝飾器"},
    ]

    # 第一輪
    response1 = llm.chat(conversation)
    print(f"用戶: {conversation[0]['content']}")
    print(f"助手: {response1}\n")

    # 第二輪
    conversation.append({"role": "assistant", "content": response1})
    conversation.append({"role": "user", "content": "給我一個具體的例子"})

    response2 = llm.chat(conversation)
    print(f"用戶: {conversation[-1]['content']}")
    print(f"助手: {response2}\n")


def example_function_calling():
    """示例 3: 工具調用"""
    print("=== 示例 3: Function Calling ===\n")

    llm = Llama31Model(model_size="8b", quantization="4bit")

    # 定義工具
    def get_weather(city: str) -> str:
        """獲取天氣（模擬）"""
        weather_data = {
            "台北": "晴天，25°C",
            "東京": "多雲，18°C",
            "紐約": "下雨，15°C"
        }
        return weather_data.get(city, "無數據")

    def calculate(operation: str, a: float, b: float) -> float:
        """計算器"""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        else:
            raise ValueError(f"不支持的運算: {operation}")

    tools = [
        {
            "name": "get_weather",
            "description": "獲取指定城市的天氣",
            "parameters": {"city": "城市名稱"},
            "function": get_weather
        },
        {
            "name": "calculate",
            "description": "執行數學運算",
            "parameters": {
                "operation": "運算類型 (add, multiply)",
                "a": "第一個數字",
                "b": "第二個數字"
            },
            "function": calculate
        }
    ]

    # 測試工具調用
    messages = [
        {"role": "user", "content": "台北現在的天氣如何？"}
    ]

    result = llm.function_calling(messages, tools)
    print(f"用戶: {messages[0]['content']}")
    print(f"結果: {result}\n")


def example_long_context():
    """示例 4: 長上下文處理（128K）"""
    print("=== 示例 4: 長上下文處理 ===\n")

    llm = Llama31Model(model_size="8b", quantization="4bit")

    # 生成長文本
    long_document = """
這是一份關於人工智能發展的詳細報告。

第一章：AI的歷史
人工智能的概念最早由Alan Turing提出...
[此處假設有大量文本]

第二章：深度學習革命
2012年，AlexNet在ImageNet上的突破...
[此處假設有大量文本]

第三章：大型語言模型
2017年，Transformer架構的提出改變了NLP領域...
GPT系列、BERT、T5等模型的出現...
[此處假設有大量文本]

結論：
AI技術正在快速發展，未來將會...
""" * 10  # 模擬長文本

    messages = [
        {"role": "user", "content": f"請閱讀以下文檔並總結：\n\n{long_document[:2000]}...（文檔過長已截斷）\n\n請問這份文檔主要講了什麼？"}
    ]

    response = llm.chat(messages, max_new_tokens=256)
    print(f"文檔長度: ~{len(long_document)} 字符")
    print(f"總結: {response}\n")


def example_code_generation():
    """示例 5: 代碼生成"""
    print("=== 示例 5: 代碼生成 ===\n")

    llm = Llama31Model(model_size="8b", quantization="4bit")

    messages = [
        {
            "role": "user",
            "content": "寫一個 Python 函數，實現二分搜索算法，包含詳細註釋"
        }
    ]

    response = llm.chat(messages, temperature=0.3)  # 較低溫度以提高準確性
    print(f"用戶: {messages[0]['content']}")
    print(f"生成的代碼:\n{response}\n")


def example_structured_output():
    """示例 6: 結構化輸出"""
    print("=== 示例 6: 結構化輸出（JSON）===\n")

    llm = Llama31Model(model_size="8b", quantization="4bit")

    messages = [
        {
            "role": "user",
            "content": """請將以下信息提取為JSON格式：

張三，35歲，軟體工程師，住在台北市，興趣是編程和攝影。

JSON格式：{"name": "...", "age": ..., "occupation": "...", "city": "...", "hobbies": [...]}"""
        }
    ]

    response = llm.chat(messages, temperature=0.1)
    print(f"提取結果:\n{response}\n")


if __name__ == "__main__":
    print("Llama 3.1 完整使用示例")
    print("=" * 60)
    print()

    # 注意：這些示例需要較大的GPU記憶體
    # 8B 模型（4-bit量化）: ~5GB VRAM
    # 70B 模型（4-bit量化）: ~40GB VRAM
    # 405B 模型（4-bit量化）: ~200GB+ VRAM

    try:
        # 運行示例（選擇性執行）
        example_basic_chat()
        # example_multi_turn_chat()
        # example_function_calling()
        # example_long_context()
        # example_code_generation()
        # example_structured_output()

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n注意:")
        print("1. 需要 HuggingFace 賬號並接受 Llama 3.1 許可")
        print("2. 需要足夠的 GPU 記憶體")
        print("3. 首次運行會下載模型（8B ~16GB, 70B ~140GB）")

    print("\nLlama 3.1 特性:")
    print("✓ 128K 上下文長度")
    print("✓ 多語言支持（包含繁體中文）")
    print("✓ 工具調用能力")
    print("✓ 優秀的代碼生成")
    print("✓ 商業友好許可")

    print("\n模型選擇建議:")
    print("- 8B: 個人使用、快速原型（單卡RTX 4090）")
    print("- 70B: 生產應用、高質量輸出（多卡A100）")
    print("- 405B: 研究、最佳性能（大規模集群）")
