"""
KV Cache 基礎實作
演示 KV Cache 如何加速自迴歸生成
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class KVCacheDemonstration:
    """KV Cache 演示類"""

    def __init__(self, model_name: str = "gpt2"):
        """
        初始化

        Args:
            model_name: 模型名稱
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"KV Cache 演示")
        print(f"模型: {model_name}")
        print(f"設備: {self.device}")

        # 載入模型和 tokenizer
        print("\n載入模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ 模型載入完成")

    def generate_without_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Tuple[str, float, int]:
        """
        不使用 KV Cache 生成（低效）

        Args:
            prompt: 輸入提示
            max_new_tokens: 最大生成 token 數

        Returns:
            (生成文本, 延遲(ms), 計算次數估計)
        """
        print("\n" + "="*60)
        print("無 KV Cache 生成（低效模式）")
        print("="*60)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # 禁用 KV Cache
        self.model.config.use_cache = False

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False  # 明確禁用
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency = (time.time() - start_time) * 1000

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_length = outputs.shape[1]
        new_tokens = total_length - input_length

        # 估算計算次數（近似）
        # 每個時間步需要計算當前所有 token 的 attention
        compute_ops = sum(range(input_length, total_length + 1))

        print(f"輸入長度: {input_length} tokens")
        print(f"生成長度: {new_tokens} tokens")
        print(f"總延遲: {latency:.2f} ms")
        print(f"平均每 token: {latency / new_tokens:.2f} ms")
        print(f"估計計算次數: {compute_ops}")
        print(f"\n生成文本:\n{generated_text}\n")

        return generated_text, latency, compute_ops

    def generate_with_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Tuple[str, float, int]:
        """
        使用 KV Cache 生成（高效）

        Args:
            prompt: 輸入提示
            max_new_tokens: 最大生成 token 數

        Returns:
            (生成文本, 延遲(ms), 計算次數估計)
        """
        print("\n" + "="*60)
        print("有 KV Cache 生成（高效模式）")
        print("="*60)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # 啟用 KV Cache
        self.model.config.use_cache = True

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True  # 明確啟用
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency = (time.time() - start_time) * 1000

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_length = outputs.shape[1]
        new_tokens = total_length - input_length

        # 使用 cache 後，每個時間步只計算 1 個新 token
        compute_ops = input_length + new_tokens  # 首次計算 + 每步 1 次

        print(f"輸入長度: {input_length} tokens")
        print(f"生成長度: {new_tokens} tokens")
        print(f"總延遲: {latency:.2f} ms")
        print(f"平均每 token: {latency / new_tokens:.2f} ms")
        print(f"估計計算次數: {compute_ops}")
        print(f"\n生成文本:\n{generated_text}\n")

        return generated_text, latency, compute_ops

    def manual_generation_demo(self, prompt: str, max_new_tokens: int = 10):
        """
        手動逐步生成演示，展示 KV Cache 的工作過程

        Args:
            prompt: 輸入提示
            max_new_tokens: 最大生成 token 數
        """
        print("\n" + "="*60)
        print("手動逐步生成演示（展示 KV Cache 工作原理）")
        print("="*60)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        self.model.config.use_cache = True
        past_key_values = None

        print(f"初始提示: {prompt}")
        print(f"初始 token IDs: {input_ids[0].tolist()}\n")

        generated_ids = input_ids.clone()

        for step in range(max_new_tokens):
            print(f"--- 步驟 {step + 1} ---")

            # 第一步：處理整個輸入
            # 後續步驟：只處理新生成的 token
            if step == 0:
                current_input = input_ids
                print(f"處理完整輸入 ({current_input.shape[1]} tokens)")
            else:
                current_input = next_token.unsqueeze(0)
                print(f"只處理新 token (1 token)")

            # 前向傳播
            with torch.no_grad():
                outputs = self.model(
                    current_input,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            # 獲取下一個 token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # 更新 past_key_values（KV Cache）
            past_key_values = outputs.past_key_values

            # 如果有 cache，顯示其大小
            if past_key_values:
                first_layer_cache = past_key_values[0]  # (key, value)
                cache_seq_len = first_layer_cache[0].shape[2]  # K 的序列長度
                print(f"KV Cache 序列長度: {cache_seq_len}")
                print(f"Cache 層數: {len(past_key_values)}")

            # 解碼新 token
            generated_token_text = self.tokenizer.decode(next_token)
            print(f"生成 token: '{generated_token_text}' (ID: {next_token.item()})")

            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            # 檢查是否結束
            if next_token.item() == self.tokenizer.eos_token_id:
                print("遇到 EOS token，停止生成")
                break

            print()

        # 顯示完整生成文本
        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("="*60)
        print(f"完整生成文本:\n{final_text}")
        print("="*60)

    def compare_performance(
        self,
        prompt: str,
        max_new_tokens_list: list = [20, 50, 100]
    ):
        """
        對比不同生成長度下的性能

        Args:
            prompt: 輸入提示
            max_new_tokens_list: 不同的生成長度列表
        """
        print("\n" + "="*60)
        print("性能對比測試")
        print("="*60)

        results = {
            "max_tokens": [],
            "without_cache_ms": [],
            "with_cache_ms": [],
            "speedup": []
        }

        for max_tokens in max_new_tokens_list:
            print(f"\n{'>'*60}")
            print(f"測試生成長度: {max_tokens} tokens")
            print(f"{'<'*60}")

            # 無 Cache
            _, latency_no_cache, _ = self.generate_without_cache(prompt, max_tokens)

            # 有 Cache
            _, latency_cache, _ = self.generate_with_cache(prompt, max_tokens)

            speedup = latency_no_cache / latency_cache

            results["max_tokens"].append(max_tokens)
            results["without_cache_ms"].append(latency_no_cache)
            results["with_cache_ms"].append(latency_cache)
            results["speedup"].append(speedup)

            print(f"\n📊 結果:")
            print(f"  無 Cache: {latency_no_cache:.2f} ms")
            print(f"  有 Cache: {latency_cache:.2f} ms")
            print(f"  加速比: {speedup:.2f}x")

        # 顯示總結
        self._print_comparison_summary(results)

        # 繪製圖表
        self._plot_comparison(results)

        return results

    def _print_comparison_summary(self, results: Dict):
        """打印對比摘要"""
        print("\n" + "="*60)
        print("性能對比摘要")
        print("="*60)

        print(f"\n{'生成長度':<12} {'無 Cache (ms)':<15} {'有 Cache (ms)':<15} {'加速比':<10}")
        print("-" * 60)

        for i in range(len(results["max_tokens"])):
            print(
                f"{results['max_tokens'][i]:<12} "
                f"{results['without_cache_ms'][i]:<15.2f} "
                f"{results['with_cache_ms'][i]:<15.2f} "
                f"{results['speedup'][i]:<10.2f}x"
            )

        print("-" * 60)

        avg_speedup = sum(results["speedup"]) / len(results["speedup"])
        print(f"\n平均加速比: {avg_speedup:.2f}x")

        print("\n💡 觀察:")
        print("  • 生成長度越長，KV Cache 的加速效果越明顯")
        print("  • KV Cache 將 O(n²) 的計算複雜度降低到 O(n)")
        print("  • 生產環境中應該始終啟用 KV Cache")

    def _plot_comparison(self, results: Dict):
        """繪製對比圖表"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 圖 1: 延遲對比
            x = results["max_tokens"]
            ax1.plot(x, results["without_cache_ms"], marker='o', label='無 Cache', linewidth=2)
            ax1.plot(x, results["with_cache_ms"], marker='s', label='有 Cache', linewidth=2)
            ax1.set_xlabel('生成 Token 數', fontsize=12)
            ax1.set_ylabel('延遲 (ms)', fontsize=12)
            ax1.set_title('KV Cache 延遲對比', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 圖 2: 加速比
            ax2.plot(x, results["speedup"], marker='D', color='green', linewidth=2)
            ax2.axhline(y=1.0, color='r', linestyle='--', label='基準線 (1x)')
            ax2.set_xlabel('生成 Token 數', fontsize=12)
            ax2.set_ylabel('加速比', fontsize=12)
            ax2.set_title('KV Cache 加速比', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('kv_cache_comparison.png', dpi=300, bbox_inches='tight')
            print("\n📊 圖表已儲存: kv_cache_comparison.png")

            try:
                plt.show()
            except:
                pass

        except Exception as e:
            print(f"⚠️  繪圖失敗: {e}")


def main():
    """主函數"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                KV Cache 基礎演示                            ║
║                                                            ║
║  本範例演示:                                                ║
║  1. KV Cache 的工作原理                                    ║
║  2. 有/無 Cache 的性能對比                                  ║
║  3. 手動逐步生成過程                                        ║
║  4. 不同生成長度的加速效果                                  ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 初始化
    demo = KVCacheDemonstration(model_name="gpt2")

    # 測試提示
    prompt = "The future of artificial intelligence is"

    # 演示 1: 手動逐步生成
    print("\n" + "🔹"*30)
    print("演示 1: 手動逐步生成（理解 KV Cache 工作原理）")
    print("🔹"*30)
    demo.manual_generation_demo(prompt, max_new_tokens=10)

    input("\n按 Enter 繼續到演示 2...")

    # 演示 2: 性能對比
    print("\n" + "🔹"*30)
    print("演示 2: 性能對比（不同生成長度）")
    print("🔹"*30)
    results = demo.compare_performance(
        prompt,
        max_new_tokens_list=[20, 50, 100]
    )

    print("\n✅ 所有演示完成！")
    print("\n🔗 下一步:")
    print("  - 進階 KV Cache: 02_kv_cache_benchmark.py")
    print("  - Flash Attention: ../3.Flash-Attention/")
    print("  - vLLM（PagedAttention）: ../4.vLLM-部署/")


if __name__ == "__main__":
    main()
