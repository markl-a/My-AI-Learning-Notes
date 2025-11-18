"""
GPTQ 量化實作
使用 AutoGPTQ 對大型語言模型進行 4-bit 量化
"""

import torch
import time
import os
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Optional
import json


class GPTQQuantizer:
    """GPTQ 量化工具類"""

    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False
    ):
        """
        初始化 GPTQ 量化器

        Args:
            model_name: 模型名稱或路徑
            bits: 量化位元數 (2, 3, 4, 8)
            group_size: 分組大小，-1 表示不分組
            desc_act: 是否按降序排列激活值（可能提升精度）
        """
        self.model_name = model_name
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act

        print(f"初始化 GPTQ 量化器")
        print(f"  模型: {model_name}")
        print(f"  位元數: {bits}-bit")
        print(f"  分組大小: {group_size}")
        print(f"  降序激活: {desc_act}")

        # 檢查 AutoGPTQ 是否可用
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            self.AutoGPTQForCausalLM = AutoGPTQForCausalLM
            self.BaseQuantizeConfig = BaseQuantizeConfig
            print("✅ AutoGPTQ 已安裝")
        except ImportError:
            print("❌ 未安裝 AutoGPTQ")
            print("請安裝: pip install auto-gptq")
            raise

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_calibration_data(
        self,
        num_samples: int = 128,
        max_length: int = 512
    ) -> List[dict]:
        """
        準備校準數據

        Args:
            num_samples: 樣本數量
            max_length: 最大序列長度

        Returns:
            校準數據列表
        """
        print(f"\n準備校準數據 (共 {num_samples} 個樣本)...")

        try:
            from datasets import load_dataset

            # 使用 C4 數據集（常用於 LLM 校準）
            print("載入 C4 數據集...")
            dataset = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            calibration_data = []

            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break

                text = sample["text"]

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                )

                calibration_data.append({
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                })

                if (i + 1) % 32 == 0:
                    print(f"  已處理 {i + 1}/{num_samples} 個樣本")

            print(f"✅ 校準數據準備完成: {len(calibration_data)} 個樣本")
            return calibration_data

        except Exception as e:
            print(f"⚠️  無法載入 C4 數據集: {e}")
            print("使用預設文本作為校準數據...")

            # 備用：使用預設文本
            example_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "Machine learning models require large amounts of data.",
                "Natural language processing has made significant progress.",
                "Deep learning enables computers to learn from experience.",
            ] * (num_samples // 5 + 1)

            calibration_data = []
            for text in example_texts[:num_samples]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                )
                calibration_data.append({
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                })

            return calibration_data

    def quantize(
        self,
        output_dir: str,
        calibration_data: Optional[List[dict]] = None,
        num_samples: int = 128
    ):
        """
        執行 GPTQ 量化

        Args:
            output_dir: 輸出目錄
            calibration_data: 校準數據（如果為 None，將自動生成）
            num_samples: 校準樣本數（如果需要生成）
        """
        print("\n" + "="*60)
        print("開始 GPTQ 量化")
        print("="*60)

        # 準備校準數據
        if calibration_data is None:
            calibration_data = self.prepare_calibration_data(num_samples=num_samples)

        # 配置量化參數
        quantize_config = self.BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            damp_percent=0.01,
            sym=True,  # 對稱量化
            true_sequential=True  # 順序量化（更準確但較慢）
        )

        print(f"\n量化配置:")
        print(f"  bits: {quantize_config.bits}")
        print(f"  group_size: {quantize_config.group_size}")
        print(f"  desc_act: {quantize_config.desc_act}")

        # 載入模型
        print(f"\n載入原始模型...")
        start_time = time.time()

        model = self.AutoGPTQForCausalLM.from_pretrained(
            self.model_name,
            quantize_config=quantize_config,
            trust_remote_code=True
        )

        load_time = time.time() - start_time
        print(f"✅ 模型載入完成 ({load_time:.2f} 秒)")

        # 執行量化
        print(f"\n執行量化...")
        print("⏳ 這可能需要幾分鐘到幾十分鐘，取決於模型大小和校準數據量...")

        start_time = time.time()

        model.quantize(
            calibration_data,
            use_triton=False,  # Triton 可能不穩定
            batch_size=1,
            use_cuda_fp16=torch.cuda.is_available()
        )

        quantize_time = time.time() - start_time
        print(f"✅ 量化完成 ({quantize_time:.2f} 秒)")

        # 儲存量化模型
        print(f"\n儲存量化模型到 {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        model.save_quantized(
            output_dir,
            use_safetensors=True
        )

        # 儲存 tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # 儲存量化資訊
        info = {
            "model_name": self.model_name,
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "quantize_time": quantize_time,
            "num_calibration_samples": len(calibration_data)
        }

        with open(os.path.join(output_dir, "quantization_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"✅ 模型已儲存")
        print(f"\n量化摘要:")
        print(f"  原始模型: {self.model_name}")
        print(f"  量化位元: {self.bits}-bit")
        print(f"  量化時間: {quantize_time:.2f} 秒")
        print(f"  輸出目錄: {output_dir}")

        return model

    def load_quantized_model(self, model_path: str):
        """
        載入已量化的模型

        Args:
            model_path: 量化模型路徑

        Returns:
            量化模型
        """
        print(f"\n載入量化模型: {model_path}")

        model = self.AutoGPTQForCausalLM.from_quantized(
            model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_triton=False,
            use_safetensors=True,
            trust_remote_code=True
        )

        print("✅ 量化模型載入完成")
        return model

    def benchmark(
        self,
        model,
        test_prompts: List[str] = None,
        max_new_tokens: int = 50
    ):
        """
        測試量化模型性能

        Args:
            model: 模型
            test_prompts: 測試提示列表
            max_new_tokens: 最大生成 token 數
        """
        if test_prompts is None:
            test_prompts = [
                "The future of artificial intelligence is",
                "Once upon a time",
                "The key to success is"
            ]

        print("\n" + "="*60)
        print("性能測試")
        print("="*60)

        device = next(model.parameters()).device
        latencies = []

        for i, prompt in enumerate(test_prompts):
            print(f"\n提示 {i+1}/{len(test_prompts)}: {prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            # 計時
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成: {generated_text}")
            print(f"延遲: {latency:.2f} ms")

        avg_latency = sum(latencies) / len(latencies)
        print(f"\n平均延遲: {avg_latency:.2f} ms")

        return avg_latency


def demo_quick_quantization():
    """快速量化演示（使用小模型）"""
    print("""
╔════════════════════════════════════════════════════════════╗
║           GPTQ 量化 - 快速演示                              ║
║                                                            ║
║  使用小模型 (GPT-2) 進行快速測試                            ║
║  實際應用中應使用更大的模型                                 ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 使用小模型進行快速演示
    model_name = "gpt2"  # 可改為 "facebook/opt-125m"
    output_dir = "./models/gpt2-gptq-4bit"

    try:
        # 初始化量化器
        quantizer = GPTQQuantizer(
            model_name=model_name,
            bits=4,
            group_size=128,
            desc_act=False
        )

        # 執行量化（使用少量校準數據進行演示）
        model = quantizer.quantize(
            output_dir=output_dir,
            num_samples=32  # 演示用少量樣本
        )

        # 測試性能
        quantizer.benchmark(model)

        print("\n✅ 快速演示完成！")
        print(f"量化模型已儲存到: {output_dir}")

    except Exception as e:
        print(f"❌ 量化失敗: {e}")
        import traceback
        traceback.print_exc()


def demo_load_and_test():
    """載入並測試已量化的模型"""
    print("""
╔════════════════════════════════════════════════════════════╗
║           GPTQ 量化 - 載入測試                              ║
║                                                            ║
║  載入已量化的模型並測試性能                                 ║
╚════════════════════════════════════════════════════════════╝
    """)

    model_path = "./models/gpt2-gptq-4bit"

    if not os.path.exists(model_path):
        print(f"❌ 找不到量化模型: {model_path}")
        print("請先運行快速量化演示")
        return

    try:
        quantizer = GPTQQuantizer(model_name=model_path, bits=4, group_size=128)
        model = quantizer.load_quantized_model(model_path)

        # 測試
        quantizer.benchmark(model, test_prompts=[
            "Artificial intelligence will",
            "The best way to learn is",
            "In the year 2050,"
        ])

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函數"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                    GPTQ 量化實作                            ║
║                                                            ║
║  GPTQ 是專為大型語言模型設計的訓練後量化方法                ║
║  - 支援 4-bit 甚至 3-bit 量化                               ║
║  - 精度損失小                                              ║
║  - 適合大模型壓縮                                           ║
║                                                            ║
║  選擇演示模式:                                             ║
║  1. 快速量化演示 (推薦新手)                                ║
║  2. 載入並測試已量化的模型                                 ║
║                                                            ║
║  注意: 首次運行需要下載模型和數據集                         ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 檢查環境
    if not torch.cuda.is_available():
        print("⚠️  警告: 未檢測到 CUDA，GPTQ 需要 GPU")
        print("建議在有 GPU 的環境中運行")
        return

    print("\n請選擇:")
    print("1. 快速量化演示")
    print("2. 載入並測試")
    print("0. 退出")

    choice = input("\n輸入選項 (預設 1): ").strip() or "1"

    if choice == "1":
        demo_quick_quantization()
    elif choice == "2":
        demo_load_and_test()
    else:
        print("退出")

    print("\n🔗 下一步:")
    print("  - 試試 AWQ 量化: 03_awq_quantization.py")
    print("  - 量化方法對比: 05_quantization_comparison.py")
    print("  - 使用量化模型部署: ../4.vLLM-部署/")


if __name__ == "__main__":
    main()
