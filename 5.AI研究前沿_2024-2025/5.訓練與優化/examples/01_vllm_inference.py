"""
vLLM 高性能推理
使用 vLLM 實現高吞吐量的 LLM 推理
支持 PagedAttention、Continuous Batching 等優化技術
"""

from vllm import LLM, SamplingParams
from typing import List, Dict, Optional
import time


class VLLMInference:
    """vLLM 推理引擎"""

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False
    ):
        """
        初始化 vLLM 推理引擎

        Args:
            model_name: 模型名稱或路徑
            tensor_parallel_size: 張量並行大小（GPU數量）
            gpu_memory_utilization: GPU 記憶體使用率 (0.0-1.0)
            max_model_len: 最大序列長度
            trust_remote_code: 是否信任遠程代碼
        """
        print(f"正在載入模型: {model_name}")
        print(f"張量並行: {tensor_parallel_size}")
        print(f"GPU 記憶體使用率: {gpu_memory_utilization}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code
        )

        self.model_name = model_name
        print("模型載入完成！")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        n: int = 1,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        """
        批量生成文本

        Args:
            prompts: 提示詞列表
            max_tokens: 最大生成token數
            temperature: 溫度（0-2，越高越隨機）
            top_p: Top-P 採樣
            top_k: Top-K 採樣
            presence_penalty: 存在懲罰（-2.0 到 2.0）
            frequency_penalty: 頻率懲罰（-2.0 到 2.0）
            n: 每個提示生成n個回覆
            stop: 停止詞列表

        Returns:
            生成的文本列表
        """
        # 配置採樣參數
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop
        )

        # 開始計時
        start_time = time.time()

        # 批量生成
        outputs = self.llm.generate(prompts, sampling_params)

        # 計算時間和吞吐量
        elapsed_time = time.time() - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / elapsed_time

        print(f"\n生成統計:")
        print(f"  - 提示數量: {len(prompts)}")
        print(f"  - 生成時間: {elapsed_time:.2f}秒")
        print(f"  - 總token數: {total_tokens}")
        print(f"  - 吞吐量: {throughput:.2f} tokens/秒")

        # 提取生成的文本
        results = []
        for output in outputs:
            for completion in output.outputs:
                results.append(completion.text)

        return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        聊天模式（單輪對話）

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成長度
            temperature: 溫度

        Returns:
            回覆文本
        """
        # 構建提示（簡化版，實際應根據模型的 chat template）
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"System: {content}\n"

        prompt += "Assistant:"

        # 生成回覆
        results = self.generate(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nUser:", "\n\n"]
        )

        return results[0] if results else ""


def example_basic_generation():
    """示例 1: 基本文本生成"""
    print("=== 示例 1: 基本文本生成 ===")

    # 使用小模型進行測試
    engine = VLLMInference(
        model_name="facebook/opt-125m",  # 小模型，快速測試
        tensor_parallel_size=1
    )

    # 單個提示
    prompts = ["Once upon a time,"]
    results = engine.generate(
        prompts=prompts,
        max_tokens=100,
        temperature=0.8
    )

    print(f"\n提示: {prompts[0]}")
    print(f"生成: {results[0]}")


def example_batch_inference():
    """示例 2: 批量推理"""
    print("\n=== 示例 2: 批量推理（展示高吞吐量）===")

    engine = VLLMInference(
        model_name="facebook/opt-125m",
        gpu_memory_utilization=0.9  # 高記憶體使用率以支持更大批次
    )

    # 批量提示
    prompts = [
        "The future of AI is",
        "In a galaxy far away,",
        "The secret to happiness is",
        "Climate change will",
        "Technology has changed",
        "The meaning of life",
        "Artificial intelligence can",
        "In the year 2050,"
    ] * 10  # 80個提示

    # 批量生成
    print(f"\n批量生成 {len(prompts)} 個提示...")
    results = engine.generate(
        prompts=prompts,
        max_tokens=50,
        temperature=0.7
    )

    # 顯示部分結果
    print("\n部分結果:")
    for i in range(min(3, len(results))):
        print(f"\n{i+1}. 提示: {prompts[i]}")
        print(f"   生成: {results[i][:100]}...")


def example_different_parameters():
    """示例 3: 不同採樣參數"""
    print("\n=== 示例 3: 不同採樣參數 ===")

    engine = VLLMInference(model_name="facebook/opt-125m")

    prompt = "Artificial intelligence is"

    # 測試不同溫度
    print("\n不同溫度的效果:")
    for temp in [0.0, 0.5, 1.0, 1.5]:
        result = engine.generate(
            prompts=[prompt],
            max_tokens=30,
            temperature=temp
        )[0]
        print(f"\n溫度={temp}:")
        print(f"{result[:100]}")


def example_streaming_like():
    """示例 4: 模擬流式輸出"""
    print("\n=== 示例 4: 多樣性生成（每個提示生成多個回覆）===")

    engine = VLLMInference(model_name="facebook/opt-125m")

    prompt = "The best way to learn programming is"

    # 生成多個候選回覆
    results = engine.generate(
        prompts=[prompt],
        max_tokens=50,
        temperature=0.8,
        n=5  # 生成5個不同的回覆
    )

    print(f"\n提示: {prompt}\n")
    print("生成的5個不同回覆:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result}")


def example_chat_mode():
    """示例 5: 聊天模式"""
    print("\n=== 示例 5: 聊天模式 ===")

    engine = VLLMInference(model_name="facebook/opt-125m")

    # 模擬多輪對話
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]

    response = engine.chat(
        messages=conversation,
        max_tokens=100,
        temperature=0.7
    )

    print("\nUser: What is Python?")
    print(f"Assistant: {response}")


def example_production_deployment():
    """示例 6: 生產環境配置"""
    print("\n=== 示例 6: 生產環境推薦配置 ===")

    print("""
生產環境 vLLM 推薦配置:

1. 高吞吐量配置（批處理服務）:
   - gpu_memory_utilization: 0.95
   - max_num_batched_tokens: 8192 (根據GPU調整)
   - max_num_seqs: 256 (同時處理的序列數)

2. 低延遲配置（實時服務）:
   - gpu_memory_utilization: 0.8
   - max_num_seqs: 64
   - 使用較小的batch size

3. 多GPU配置:
   - tensor_parallel_size: GPU數量
   - 適合大模型（70B+）

4. 長上下文配置:
   - max_model_len: 根據需求設定
   - gpu_memory_utilization: 0.85-0.90
   - 啟用 FlashAttention

示例代碼:
    """)

    print("""
# 生產環境示例
engine = VLLMInference(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,  # 使用2個GPU
    gpu_memory_utilization=0.95,  # 高記憶體使用率
    max_model_len=4096,  # 支持長上下文
    trust_remote_code=True
)
    """)


if __name__ == "__main__":
    print("vLLM 高性能推理示例")
    print("=" * 60)
    print("\nvLLM 特性:")
    print("1. PagedAttention - 高效的KV緩存管理")
    print("2. Continuous Batching - 動態批處理")
    print("3. 高吞吐量 - 比HuggingFace快15-24x")
    print("4. 張量並行 - 支持多GPU推理")
    print("5. 流式輸出 - 支持實時生成")
    print()

    # 運行示例
    try:
        example_basic_generation()
        # example_batch_inference()
        # example_different_parameters()
        # example_streaming_like()
        # example_chat_mode()
        example_production_deployment()

    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n注意:")
        print("1. 確保已安裝 vllm: pip install vllm")
        print("2. 需要 NVIDIA GPU 和 CUDA")
        print("3. 某些模型可能需要登錄 HuggingFace")

    print("\n所有示例完成！")
    print("\n性能優化建議:")
    print("1. 調整 gpu_memory_utilization 平衡記憶體和吞吐量")
    print("2. 使用 tensor_parallel 處理大模型")
    print("3. 啟用 FlashAttention 加速推理")
    print("4. 合理設置 max_num_batched_tokens")
    print("5. 根據需求選擇適當的模型大小")
