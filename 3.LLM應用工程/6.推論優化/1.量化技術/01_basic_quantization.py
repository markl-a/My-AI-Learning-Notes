"""
基礎量化實作
演示 FP16 和 INT8 量化的基本使用，對比記憶體和推論速度
"""

import torch
import time
import psutil
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Dict, Tuple
import gc


class QuantizationBenchmark:
    """量化基準測試類"""

    def __init__(self, model_name: str = "gpt2"):
        """
        初始化

        Args:
            model_name: 模型名稱，預設使用 GPT-2（小模型，快速測試）
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 測試提示
        self.test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant land",
            "The key to success in life is",
            "Climate change is one of the most pressing",
            "The development of quantum computing will"
        ]

    def get_memory_usage(self) -> Dict[str, float]:
        """
        獲取記憶體使用情況

        Returns:
            包含 CPU 和 GPU 記憶體使用的字典
        """
        memory_info = {
            'cpu_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }

        if torch.cuda.is_available():
            memory_info['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return memory_info

    def load_fp32_model(self) -> AutoModelForCausalLM:
        """載入 FP32 模型"""
        print("\n" + "="*50)
        print("載入 FP32 模型")
        print("="*50)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            model = model.to(self.device)

        memory = self.get_memory_usage()
        print(f"記憶體使用: CPU={memory['cpu_mb']:.2f} MB", end="")
        if 'gpu_mb' in memory:
            print(f", GPU={memory['gpu_mb']:.2f} MB")
        else:
            print()

        return model

    def load_fp16_model(self) -> AutoModelForCausalLM:
        """載入 FP16 模型"""
        print("\n" + "="*50)
        print("載入 FP16 模型")
        print("="*50)

        if self.device == "cpu":
            print("警告: CPU 不支援 FP16，使用 FP32 代替")
            return self.load_fp32_model()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        memory = self.get_memory_usage()
        print(f"記憶體使用: CPU={memory['cpu_mb']:.2f} MB", end="")
        if 'gpu_mb' in memory:
            print(f", GPU={memory['gpu_mb']:.2f} MB")
        else:
            print()

        return model

    def load_int8_model(self) -> AutoModelForCausalLM:
        """載入 INT8 量化模型（使用 bitsandbytes）"""
        print("\n" + "="*50)
        print("載入 INT8 量化模型")
        print("="*50)

        if self.device == "cpu":
            print("警告: bitsandbytes INT8 需要 GPU，改用動態量化")
            return self.load_dynamic_int8_model()

        try:
            # 配置 8-bit 量化
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )

            memory = self.get_memory_usage()
            print(f"記憶體使用: CPU={memory['cpu_mb']:.2f} MB", end="")
            if 'gpu_mb' in memory:
                print(f", GPU={memory['gpu_mb']:.2f} MB")
            else:
                print()

            return model

        except Exception as e:
            print(f"INT8 量化失敗: {e}")
            print("回退到動態量化")
            return self.load_dynamic_int8_model()

    def load_dynamic_int8_model(self) -> torch.nn.Module:
        """載入動態 INT8 量化模型（CPU 友好）"""
        print("使用 PyTorch 動態量化")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )

        # 動態量化 Linear 層
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        model = model.to(self.device)

        memory = self.get_memory_usage()
        print(f"記憶體使用: CPU={memory['cpu_mb']:.2f} MB", end="")
        if 'gpu_mb' in memory:
            print(f", GPU={memory['gpu_mb']:.2f} MB")
        else:
            print()

        return model

    def benchmark_generation(
        self,
        model: torch.nn.Module,
        model_name: str,
        num_runs: int = 5
    ) -> Tuple[float, float]:
        """
        基準測試生成速度

        Args:
            model: 模型
            model_name: 模型名稱（用於顯示）
            num_runs: 運行次數

        Returns:
            (平均延遲, 標準差)
        """
        print(f"\n測試 {model_name} 生成速度...")

        latencies = []

        for i, prompt in enumerate(self.test_prompts[:num_runs]):
            inputs = self.tokenizer(prompt, return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 預熱
            if i == 0:
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False
                    )
                if self.device == "cuda":
                    torch.cuda.synchronize()

            # 計時
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )

            if self.device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 轉換為毫秒
            latencies.append(latency)

            # 顯示生成結果（第一次）
            if i == 0:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"範例輸出: {generated_text[:100]}...")

        avg_latency = sum(latencies) / len(latencies)
        std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5

        print(f"平均延遲: {avg_latency:.2f} ms (±{std_latency:.2f} ms)")

        return avg_latency, std_latency

    def clear_memory(self):
        """清理記憶體"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run_comparison(self):
        """運行完整對比"""
        print("\n" + "="*70)
        print("量化方法對比測試")
        print("="*70)
        print(f"模型: {self.model_name}")
        print(f"設備: {self.device}")

        results = {}

        # 測試 FP32
        try:
            print("\n" + "-"*70)
            print("測試 1/3: FP32")
            print("-"*70)

            mem_before = self.get_memory_usage()
            model_fp32 = self.load_fp32_model()
            mem_after = self.get_memory_usage()

            mem_diff = mem_after['gpu_mb'] - mem_before['gpu_mb'] if 'gpu_mb' in mem_after else mem_after['cpu_mb'] - mem_before['cpu_mb']

            avg_lat, std_lat = self.benchmark_generation(model_fp32, "FP32")

            results['FP32'] = {
                'memory_mb': mem_diff,
                'latency_ms': avg_lat,
                'latency_std': std_lat
            }

            del model_fp32
            self.clear_memory()

        except Exception as e:
            print(f"FP32 測試失敗: {e}")

        # 測試 FP16
        if self.device == "cuda":
            try:
                print("\n" + "-"*70)
                print("測試 2/3: FP16")
                print("-"*70)

                mem_before = self.get_memory_usage()
                model_fp16 = self.load_fp16_model()
                mem_after = self.get_memory_usage()

                mem_diff = mem_after['gpu_mb'] - mem_before['gpu_mb']

                avg_lat, std_lat = self.benchmark_generation(model_fp16, "FP16")

                results['FP16'] = {
                    'memory_mb': mem_diff,
                    'latency_ms': avg_lat,
                    'latency_std': std_lat
                }

                del model_fp16
                self.clear_memory()

            except Exception as e:
                print(f"FP16 測試失敗: {e}")

        # 測試 INT8
        try:
            print("\n" + "-"*70)
            print("測試 3/3: INT8")
            print("-"*70)

            mem_before = self.get_memory_usage()
            model_int8 = self.load_int8_model()
            mem_after = self.get_memory_usage()

            mem_diff = mem_after['gpu_mb'] - mem_before['gpu_mb'] if 'gpu_mb' in mem_after else mem_after['cpu_mb'] - mem_before['cpu_mb']

            avg_lat, std_lat = self.benchmark_generation(model_int8, "INT8")

            results['INT8'] = {
                'memory_mb': mem_diff,
                'latency_ms': avg_lat,
                'latency_std': std_lat
            }

            del model_int8
            self.clear_memory()

        except Exception as e:
            print(f"INT8 測試失敗: {e}")

        # 顯示結果摘要
        self.print_results(results)

        return results

    def print_results(self, results: Dict):
        """打印結果摘要"""
        print("\n" + "="*70)
        print("結果摘要")
        print("="*70)

        if not results:
            print("沒有測試結果")
            return

        # 表頭
        print(f"{'方法':<10} {'記憶體 (MB)':<15} {'延遲 (ms)':<15} {'相對速度':<15}")
        print("-" * 70)

        # 基準（FP32）
        baseline_latency = results.get('FP32', {}).get('latency_ms', 1)
        baseline_memory = results.get('FP32', {}).get('memory_mb', 1)

        for method, metrics in results.items():
            memory = metrics['memory_mb']
            latency = metrics['latency_ms']
            speedup = baseline_latency / latency
            memory_ratio = memory / baseline_memory

            print(f"{method:<10} {memory:<15.2f} {latency:<15.2f} {speedup:<15.2f}x")

        print("="*70)

        # 建議
        print("\n💡 建議:")
        if 'FP16' in results and 'FP32' in results:
            fp16_speedup = results['FP32']['latency_ms'] / results['FP16']['latency_ms']
            fp16_mem_save = (1 - results['FP16']['memory_mb'] / results['FP32']['memory_mb']) * 100
            print(f"✅ FP16: 速度提升 {fp16_speedup:.2f}x，記憶體節省 {fp16_mem_save:.1f}%")

        if 'INT8' in results and 'FP32' in results:
            int8_speedup = results['FP32']['latency_ms'] / results['INT8']['latency_ms']
            int8_mem_save = (1 - results['INT8']['memory_mb'] / results['FP32']['memory_mb']) * 100
            print(f"✅ INT8: 速度提升 {int8_speedup:.2f}x，記憶體節省 {int8_mem_save:.1f}%")

        print("\n📚 下一步:")
        print("  - 試試更大的模型（如 LLaMA-7B）")
        print("  - 學習 GPTQ/AWQ 4-bit 量化（更高壓縮比）")
        print("  - 評估量化對模型精度的影響")


def main():
    """主函數"""
    print("""
╔════════════════════════════════════════════════════════════╗
║           LLM 量化技術 - 基礎量化實作                       ║
║                                                            ║
║  本範例演示:                                                ║
║  1. FP32/FP16/INT8 量化方法                                ║
║  2. 記憶體使用對比                                          ║
║  3. 推論速度對比                                            ║
║                                                            ║
║  注意: 首次運行會下載模型，可能需要幾分鐘                    ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 選擇模型（可修改）
    # 選項: "gpt2", "gpt2-medium", "facebook/opt-125m", "facebook/opt-1.3b"
    model_name = "gpt2"  # 使用小模型進行快速測試

    # 如果有 GPU 且記憶體充足，可以使用更大的模型
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"檢測到 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {gpu_memory:.1f} GB")

        if gpu_memory > 16:
            print("建議使用更大的模型來看到更明顯的差異")
            print("可以嘗試: facebook/opt-1.3b 或 facebook/opt-2.7b")

    # 運行測試
    benchmark = QuantizationBenchmark(model_name=model_name)
    results = benchmark.run_comparison()

    print("\n✅ 測試完成！")
    print("\n🔗 相關資源:")
    print("  - GPTQ 量化: 02_gptq_quantization.py")
    print("  - AWQ 量化: 03_awq_quantization.py")
    print("  - 量化對比: 05_quantization_comparison.py")


if __name__ == "__main__":
    main()
