"""
QLoRA 高效微調
使用 QLoRA (4-bit 量化 + LoRA) 在單張 GPU 上微調大型模型
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from typing import Optional


class QLoRATrainer:
    """QLoRA 微調訓練器"""

    def __init__(
        self,
        model_name: str = "facebook/opt-1.3b",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None
    ):
        """
        初始化 QLoRA 訓練器

        Args:
            model_name: 基礎模型名稱
            lora_r: LoRA 秩（越大能力越強，但參數越多）
            lora_alpha: LoRA 縮放參數
            lora_dropout: LoRA dropout 率
            target_modules: 要應用 LoRA 的模塊（None 則自動檢測）
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # 4-bit 量化配置
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # 雙重量化
            bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16  # 計算時使用 bfloat16
        )

        # LoRA 配置
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,  # None 則自動檢測
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """載入並準備模型"""
        print(f"正在載入模型: {self.model_name}")
        print(f"使用 4-bit 量化...")

        # 載入量化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",  # 自動分配到可用設備
            trust_remote_code=True
        )

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 準備模型用於 k-bit 訓練
        self.model = prepare_model_for_kbit_training(self.model)

        # 應用 LoRA
        print(f"應用 LoRA (r={self.lora_r}, alpha={self.lora_alpha})...")
        self.model = get_peft_model(self.model, self.lora_config)

        # 打印可訓練參數
        self.print_trainable_parameters()

        return self.model, self.tokenizer

    def print_trainable_parameters(self):
        """打印可訓練參數信息"""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"\n模型參數:")
        print(f"  總參數: {all_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  可訓練比例: {100 * trainable_params / all_params:.2f}%")
        print(f"  記憶體節省: ~{all_params / trainable_params:.1f}x\n")

    def prepare_dataset(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        max_length: int = 512,
        num_samples: Optional[int] = None
    ):
        """
        準備訓練數據集

        Args:
            dataset_name: 數據集名稱
            max_length: 最大序列長度
            num_samples: 使用的樣本數（None 則使用全部）

        Returns:
            處理後的數據集
        """
        print(f"正在載入數據集: {dataset_name}")

        # 載入數據集
        dataset = load_dataset(dataset_name, split="train")

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"數據集大小: {len(dataset)}")

        # 數據處理函數
        def tokenize_function(examples):
            # 根據數據集格式調整
            if "instruction" in examples:
                # Alpaca 格式
                texts = []
                for inst, inp, out in zip(
                    examples["instruction"],
                    examples.get("input", [""] * len(examples["instruction"])),
                    examples["output"]
                ):
                    if inp:
                        text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                    else:
                        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    texts.append(text)
            else:
                # 其他格式
                texts = examples["text"]

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # 處理數據集
        print("正在處理數據集...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(
        self,
        train_dataset,
        output_dir: str = "./qlora-output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100
    ):
        """
        執行訓練

        Args:
            train_dataset: 訓練數據集
            output_dir: 輸出目錄
            num_train_epochs: 訓練輪數
            per_device_train_batch_size: 每個設備的批次大小
            gradient_accumulation_steps: 梯度累積步數
            learning_rate: 學習率
            warmup_steps: 熱身步數
            logging_steps: 日誌記錄步數
            save_steps: 保存檢查點步數
        """
        print("\n開始訓練...")

        # 訓練參數
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=True,  # 使用混合精度訓練
            optim="paged_adamw_8bit",  # 8-bit Adam 優化器
            logging_dir=f"{output_dir}/logs",
            report_to="none"  # 不使用外部日誌工具
        )

        # 創建 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )

        # 訓練
        trainer.train()

        # 保存模型
        print(f"\n保存模型到: {output_dir}")
        trainer.save_model(output_dir)

        return trainer

    def inference(self, prompt: str, max_length: int = 200):
        """
        推理

        Args:
            prompt: 輸入提示
            max_length: 最大生成長度

        Returns:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def example_basic_finetuning():
    """示例 1: 基本微調流程"""
    print("=== 示例 1: QLoRA 基本微調 ===\n")

    # 創建訓練器（使用小模型進行快速測試）
    trainer = QLoRATrainer(
        model_name="facebook/opt-350m",  # 小模型用於測試
        lora_r=8,
        lora_alpha=32
    )

    # 載入模型
    model, tokenizer = trainer.load_model()

    # 準備數據（使用少量樣本進行快速測試）
    print("\n準備訓練數據...")
    # 創建示例數據
    from datasets import Dataset
    examples = {
        "instruction": [
            "What is Python?",
            "Explain machine learning",
            "How to write a function?"
        ],
        "input": ["", "", ""],
        "output": [
            "Python is a high-level programming language.",
            "Machine learning is a subset of AI.",
            "You can define a function using the def keyword."
        ]
    }
    dataset = Dataset.from_dict(examples)

    # 處理數據（需要先定義 tokenizer）
    def tokenize_function(examples):
        texts = [f"Q: {q}\nA: {a}" for q, a in zip(examples["instruction"], examples["output"])]
        return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_dataset = tokenized_dataset.add_column(
        "labels",
        tokenized_dataset["input_ids"]
    )

    print("\n注意: 這是一個最小化示例")
    print("實際訓練需要:")
    print("1. 更大的數據集")
    print("2. 更多的訓練步數")
    print("3. 適當的超參數調整\n")


def example_memory_comparison():
    """示例 2: 記憶體使用對比"""
    print("=== 示例 2: QLoRA 記憶體優勢 ===\n")

    print("記憶體使用對比 (7B 模型):")
    print("-" * 60)
    print(f"{'方法':<25} {'GPU記憶體':<15} {'可訓練參數'}")
    print("-" * 60)
    print(f"{'全量微調 (FP16)':<25} {'~28 GB':<15} {'~7B'}")
    print(f"{'LoRA (FP16)':<25} {'~28 GB':<15} {'~8M'}")
    print(f"{'QLoRA (4-bit)':<25} {'~5 GB':<15} {'~8M'}")
    print("-" * 60)

    print("\nQLoRA 優勢:")
    print("✓ 記憶體使用降低 ~5-6x")
    print("✓ 單卡可微調 65B+ 模型")
    print("✓ 性能接近全量微調")
    print("✓ 訓練速度略慢 (~25%)")


def example_hyperparameters():
    """示例 3: 超參數建議"""
    print("\n=== 示例 3: QLoRA 超參數建議 ===\n")

    print("LoRA 超參數:")
    print("-" * 60)
    print("r (秩):")
    print("  - 小模型 (7B以下): 8-16")
    print("  - 中型模型 (7B-13B): 16-32")
    print("  - 大模型 (13B+): 32-64")
    print()
    print("alpha: 通常設為 r 的 2-4 倍")
    print("  - r=8  -> alpha=16-32")
    print("  - r=16 -> alpha=32-64")
    print()
    print("target_modules:")
    print("  - LLaMA: [\"q_proj\", \"v_proj\", \"k_proj\", \"o_proj\"]")
    print("  - GPT: [\"c_attn\", \"c_proj\"]")
    print("  - OPT: [\"q_proj\", \"v_proj\"]")

    print("\n訓練超參數:")
    print("-" * 60)
    print("學習率: 1e-4 到 5e-4 (比全量微調高)")
    print("批次大小: 4-8 (使用梯度累積)")
    print("訓練輪數: 3-5")
    print("warmup_ratio: 0.03-0.05")


def example_production_tips():
    """示例 4: 生產環境建議"""
    print("\n=== 示例 4: 生產環境最佳實踐 ===\n")

    print("1. 數據準備:")
    print("   - 確保數據質量高於數量")
    print("   - 使用適當的提示模板")
    print("   - 平衡不同類型的樣本")

    print("\n2. 訓練策略:")
    print("   - 使用 gradient checkpointing 節省記憶體")
    print("   - 啟用 8-bit Adam 優化器")
    print("   - 使用混合精度訓練 (fp16/bf16)")

    print("\n3. 評估與調優:")
    print("   - 定期評估驗證集")
    print("   - 監控過擬合")
    print("   - 嘗試不同的 LoRA 秩")

    print("\n4. 部署:")
    print("   - 合併 LoRA 權重到基礎模型")
    print("   - 或保持分離以支持多任務")
    print("   - 使用 vLLM 或 TGI 進行推理")


if __name__ == "__main__":
    print("QLoRA 高效微調示例")
    print("=" * 60)

    # 運行示例
    example_basic_finetuning()
    example_memory_comparison()
    example_hyperparameters()
    example_production_tips()

    print("\n所有示例完成！")
    print("\n安裝依賴:")
    print("pip install transformers peft bitsandbytes datasets accelerate")
    print("\n注意:")
    print("1. 需要支持 CUDA 的 GPU")
    print("2. bitsandbytes 需要 CUDA 11.1+")
    print("3. 建議使用 PyTorch 2.0+")
