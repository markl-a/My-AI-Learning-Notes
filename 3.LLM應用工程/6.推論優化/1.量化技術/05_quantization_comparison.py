"""
量化方法綜合對比
對比不同量化方法的效能、記憶體使用和精度
"""

import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BenchmarkResult:
    """基準測試結果"""
    method: str
    memory_mb: float
    latency_ms: float
    tokens_per_second: float
    model_size_mb: float
    perplexity: float = 0.0
    success: bool = True
    error: str = ""


class QuantizationComparison:
    """量化方法對比工具"""

    def __init__(self, model_name: str = "gpt2", output_dir: str = "./benchmark_results"):
        """
        初始化

        Args:
            model_name: 模型名稱
            output_dir: 結果輸出目錄
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(output_dir, exist_ok=True)

        print(f"量化方法對比工具")
        print(f"模型: {model_name}")
        print(f"設備: {self.device}")

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 測試數據
        self.test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a faraway land",
            "The key to success in machine learning",
            "Climate change is one of the most pressing",
            "The development of quantum computing"
        ]

        self.results: List[BenchmarkResult] = []

    def get_gpu_memory(self) -> float:
        """獲取 GPU 記憶體使用（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def clear_memory(self):
        """清理記憶體"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def estimate_model_size(self, model) -> float:
        """
        估算模型大小（MB）

        Args:
            model: 模型

        Returns:
            模型大小（MB）
        """
        total_params = 0
        param_size = 0

        for param in model.parameters():
            total_params += param.numel()
            param_size += param.numel() * param.element_size()

        # 加上 buffer
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        total_size_mb = (param_size + buffer_size) / 1024 / 1024

        print(f"  參數數量: {total_params:,}")
        print(f"  模型大小: {total_size_mb:.2f} MB")

        return total_size_mb

    def benchmark_model(
        self,
        model,
        method_name: str,
        num_runs: int = 5,
        max_new_tokens: int = 50
    ) -> BenchmarkResult:
        """
        測試模型性能

        Args:
            model: 模型
            method_name: 方法名稱
            num_runs: 運行次數
            max_new_tokens: 生成 token 數

        Returns:
            測試結果
        """
        print(f"\n{'='*60}")
        print(f"測試 {method_name}")
        print(f"{'='*60}")

        try:
            # 記憶體使用
            self.clear_memory()
            mem_before = self.get_gpu_memory()

            # 模型大小
            model_size = self.estimate_model_size(model)

            # 推論速度測試
            latencies = []
            total_tokens = 0

            print(f"\n運行 {num_runs} 次推論測試...")

            for i, prompt in enumerate(self.test_prompts[:num_runs]):
                inputs = self.tokenizer(prompt, return_tensors="pt")

                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # 預熱
                if i == 0:
                    with torch.no_grad():
                        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                    if self.device == "cuda":
                        torch.cuda.synchronize()

                # 計時
                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )

                if self.device == "cuda":
                    torch.cuda.synchronize()

                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

                num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                total_tokens += num_tokens

                print(f"  運行 {i+1}: {latency:.2f} ms ({num_tokens} tokens)")

            mem_after = self.get_gpu_memory()
            memory_used = mem_after - mem_before

            avg_latency = sum(latencies) / len(latencies)
            tokens_per_second = (total_tokens / sum(latencies)) * 1000

            print(f"\n結果:")
            print(f"  平均延遲: {avg_latency:.2f} ms")
            print(f"  吞吐量: {tokens_per_second:.2f} tokens/s")
            print(f"  記憶體使用: {memory_used:.2f} MB")
            print(f"  模型大小: {model_size:.2f} MB")

            result = BenchmarkResult(
                method=method_name,
                memory_mb=max(memory_used, model_size),  # 使用較大的值
                latency_ms=avg_latency,
                tokens_per_second=tokens_per_second,
                model_size_mb=model_size,
                success=True
            )

            return result

        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()

            return BenchmarkResult(
                method=method_name,
                memory_mb=0,
                latency_ms=0,
                tokens_per_second=0,
                model_size_mb=0,
                success=False,
                error=str(e)
            )

    def test_fp32(self):
        """測試 FP32"""
        print("\n" + "🔹"*30)
        print("測試 1/4: FP32 (基準)")
        print("🔹"*30)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cpu":
                model = model.to(self.device)

            result = self.benchmark_model(model, "FP32")
            self.results.append(result)

            del model
            self.clear_memory()

        except Exception as e:
            print(f"FP32 測試失敗: {e}")

    def test_fp16(self):
        """測試 FP16"""
        if self.device == "cpu":
            print("\n⏭️  跳過 FP16 (CPU 不支援)")
            return

        print("\n" + "🔹"*30)
        print("測試 2/4: FP16")
        print("🔹"*30)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            result = self.benchmark_model(model, "FP16")
            self.results.append(result)

            del model
            self.clear_memory()

        except Exception as e:
            print(f"FP16 測試失敗: {e}")

    def test_int8(self):
        """測試 INT8"""
        print("\n" + "🔹"*30)
        print("測試 3/4: INT8")
        print("🔹"*30)

        try:
            if self.device == "cuda":
                # GPU: 使用 bitsandbytes
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                # CPU: 使用動態量化
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                model = model.to(self.device)

            result = self.benchmark_model(model, "INT8")
            self.results.append(result)

            del model
            self.clear_memory()

        except Exception as e:
            print(f"INT8 測試失敗: {e}")
            import traceback
            traceback.print_exc()

    def test_int4(self):
        """測試 INT4 (4-bit)"""
        if self.device == "cpu":
            print("\n⏭️  跳過 INT4 (需要 GPU)")
            return

        print("\n" + "🔹"*30)
        print("測試 4/4: INT4 (4-bit)")
        print("🔹"*30)

        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )

            result = self.benchmark_model(model, "INT4")
            self.results.append(result)

            del model
            self.clear_memory()

        except Exception as e:
            print(f"INT4 測試失敗: {e}")
            import traceback
            traceback.print_exc()

    def run_all_tests(self):
        """運行所有測試"""
        print("\n" + "="*70)
        print("開始量化方法對比測試")
        print("="*70)

        self.test_fp32()
        self.test_fp16()
        self.test_int8()
        self.test_int4()

        print("\n" + "="*70)
        print("所有測試完成")
        print("="*70)

    def print_summary(self):
        """打印結果摘要"""
        print("\n" + "="*70)
        print("測試結果摘要")
        print("="*70)

        if not self.results:
            print("沒有測試結果")
            return

        # 找到基準（FP32）
        baseline = None
        for result in self.results:
            if result.method == "FP32" and result.success:
                baseline = result
                break

        # 表頭
        print(f"\n{'方法':<10} {'記憶體':<12} {'延遲':<15} {'吞吐量':<15} {'相對速度':<12}")
        print("-" * 70)

        for result in self.results:
            if not result.success:
                print(f"{result.method:<10} ❌ 失敗: {result.error[:40]}")
                continue

            speedup = ""
            if baseline and baseline.latency_ms > 0:
                speedup = f"{baseline.latency_ms / result.latency_ms:.2f}x"

            print(
                f"{result.method:<10} "
                f"{result.memory_mb:<11.1f}MB "
                f"{result.latency_ms:<14.2f}ms "
                f"{result.tokens_per_second:<14.2f}t/s "
                f"{speedup:<12}"
            )

        print("-" * 70)

        # 建議
        print("\n💡 結論:")

        if baseline:
            for result in self.results:
                if not result.success or result.method == "FP32":
                    continue

                speedup = baseline.latency_ms / result.latency_ms
                mem_save = (1 - result.memory_mb / baseline.memory_mb) * 100

                print(f"\n{result.method}:")
                print(f"  ✅ 速度提升: {speedup:.2f}x")
                print(f"  ✅ 記憶體節省: {mem_save:.1f}%")
                print(f"  ✅ 模型大小: {result.model_size_mb:.1f} MB")

        # 推薦
        print("\n📊 推薦:")
        print("  • 追求精度: FP16")
        print("  • 平衡性能: INT8")
        print("  • 極致壓縮: INT4")
        print("  • CPU 部署: INT8 動態量化")

    def save_results(self):
        """儲存結果到 JSON"""
        output_file = os.path.join(self.output_dir, "quantization_comparison.json")

        data = {
            "model_name": self.model_name,
            "device": self.device,
            "results": [asdict(r) for r in self.results]
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 結果已儲存到: {output_file}")

    def plot_results(self):
        """繪製結果圖表"""
        if not self.results:
            return

        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return

        # 設置風格
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"量化方法對比 - {self.model_name}", fontsize=16, fontweight='bold')

        methods = [r.method for r in successful_results]

        # 1. 記憶體使用
        ax1 = axes[0, 0]
        memory_data = [r.memory_mb for r in successful_results]
        bars1 = ax1.bar(methods, memory_data, color='skyblue', edgecolor='navy')
        ax1.set_ylabel('Memory (MB)', fontsize=12)
        ax1.set_title('Memory Usage', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # 添加數值標籤
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')

        # 2. 延遲
        ax2 = axes[0, 1]
        latency_data = [r.latency_ms for r in successful_results]
        bars2 = ax2.bar(methods, latency_data, color='lightcoral', edgecolor='darkred')
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_title('Inference Latency', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')

        # 3. 吞吐量
        ax3 = axes[1, 0]
        throughput_data = [r.tokens_per_second for r in successful_results]
        bars3 = ax3.bar(methods, throughput_data, color='lightgreen', edgecolor='darkgreen')
        ax3.set_ylabel('Throughput (tokens/s)', fontsize=12)
        ax3.set_title('Throughput', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')

        # 4. 相對速度（與 FP32 對比）
        ax4 = axes[1, 1]
        baseline = next((r for r in successful_results if r.method == "FP32"), None)
        if baseline and baseline.latency_ms > 0:
            speedup_data = [baseline.latency_ms / r.latency_ms for r in successful_results]
            bars4 = ax4.bar(methods, speedup_data, color='plum', edgecolor='purple')
            ax4.set_ylabel('Speedup (vs FP32)', fontsize=12)
            ax4.set_title('Relative Speed', fontsize=14, fontweight='bold')
            ax4.axhline(y=1.0, color='r', linestyle='--', label='FP32 Baseline')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)

            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x',
                        ha='center', va='bottom')

        plt.tight_layout()

        # 儲存圖表
        output_file = os.path.join(self.output_dir, "quantization_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 圖表已儲存到: {output_file}")

        # 顯示（如果在互動環境中）
        try:
            plt.show()
        except Exception:
            pass


def main():
    """主函數"""
    print("""
╔════════════════════════════════════════════════════════════╗
║              量化方法綜合對比                               ║
║                                                            ║
║  對比以下量化方法:                                         ║
║  • FP32 (基準)                                             ║
║  • FP16                                                    ║
║  • INT8                                                    ║
║  • INT4 (4-bit)                                            ║
║                                                            ║
║  測試指標:                                                 ║
║  • 記憶體使用                                              ║
║  • 推論延遲                                                ║
║  • 吞吐量                                                  ║
║  • 模型大小                                                ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 選擇模型
    model_name = "gpt2"  # 可改為 "facebook/opt-125m", "gpt2-medium" 等

    print(f"\n使用模型: {model_name}")
    print("開始測試...\n")

    # 初始化對比工具
    comparison = QuantizationComparison(model_name=model_name)

    # 運行所有測試
    comparison.run_all_tests()

    # 顯示摘要
    comparison.print_summary()

    # 儲存結果
    comparison.save_results()

    # 繪製圖表
    try:
        comparison.plot_results()
    except Exception as e:
        print(f"⚠️  繪製圖表失敗: {e}")

    print("\n✅ 測試完成!")
    print("\n🔗 下一步:")
    print("  - 試試更大的模型觀察更明顯的差異")
    print("  - 學習 KV Cache: ../2.KV-Cache/")
    print("  - 部署優化模型: ../4.vLLM-部署/")


if __name__ == "__main__":
    main()
