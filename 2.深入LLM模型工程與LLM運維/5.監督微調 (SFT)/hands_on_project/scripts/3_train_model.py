#!/usr/bin/env python3
"""
訓練腳本：使用 QLoRA 訓練客服機器人模型

使用方法:
    python 3_train_model.py --model_name gpt2 --train_data data/train.json
"""

import os
import sys
import argparse
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """載入訓練數據"""
    logger.info(f"載入數據: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"載入了 {len(data)} 個樣本")
    return data


def format_prompt(example):
    """格式化提示詞"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""以下是一個客服對話任務。請根據指令和輸入，提供專業、友好的回答。

### 指令:
{instruction}

### 輸入:
{input_text}

### 回答:
{output}"""
    else:
        prompt = f"""以下是一個客服對話任務。請根據指令提供專業、友好的回答。

### 指令:
{instruction}

### 回答:
{output}"""

    return {"text": prompt}


def setup_model_and_tokenizer(
    model_name,
    use_qlora=True,
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.1
):
    """設置模型和 tokenizer"""

    logger.info(f"載入模型: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # 模型配置
    if use_qlora:
        logger.info("使用 QLoRA 進行 4-bit 量化訓練")

        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 載入量化模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 準備模型進行訓練
        model = prepare_model_for_kbit_training(model)

    else:
        logger.info("使用標準 LoRA 訓練")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    # LoRA 配置
    logger.info(f"LoRA 配置: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    # 根據模型類型選擇目標模塊
    if "gpt2" in model_name.lower():
        target_modules = ["c_attn"]
    elif "llama" in model_name.lower():
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    elif "mistral" in model_name.lower():
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    else:
        # 默認
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 應用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    model,
    tokenizer,
    train_data,
    val_data=None,
    output_dir="./models/customer_service_bot",
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=512,
    logging_steps=10,
    save_steps=100,
    use_wandb=False
):
    """訓練模型"""

    logger.info("準備訓練數據...")

    # 格式化數據
    train_formatted = [format_prompt(ex) for ex in train_data]
    train_dataset = Dataset.from_list(train_formatted)

    if val_data:
        val_formatted = [format_prompt(ex) for ex in val_data]
        val_dataset = Dataset.from_list(val_formatted)
    else:
        val_dataset = None

    # 訓練參數 (TRL >= 0.12 後改用 SFTConfig,把 dataset_text_field / max_seq_length 等
    # SFT 專屬欄位與 TrainingArguments 合併到同一個 config)
    logger.info("配置訓練參數...")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    # 創建 Trainer (新版 SFTTrainer 用 processing_class 取代 tokenizer)
    logger.info("創建 SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=None,  # 已經應用了 PEFT
        processing_class=tokenizer,
        args=training_args
    )

    # 開始訓練
    logger.info("開始訓練...")
    logger.info(f"  訓練樣本數: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"  驗證樣本數: {len(val_dataset)}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  有效 batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {learning_rate}")

    trainer.train()

    # 保存最終模型
    logger.info(f"保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("訓練完成！")


def main():
    parser = argparse.ArgumentParser(description="訓練客服機器人模型")

    # 模型參數
    parser.add_argument("--model_name", type=str, default="gpt2",
                       help="基座模型名稱")
    parser.add_argument("--use_qlora", action="store_true",
                       help="使用 QLoRA (4-bit 量化)")

    # 數據參數
    parser.add_argument("--train_data", type=str, required=True,
                       help="訓練數據文件路徑")
    parser.add_argument("--val_data", type=str, default=None,
                       help="驗證數據文件路徑")

    # LoRA 參數
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")

    # 訓練參數
    parser.add_argument("--output_dir", type=str, default="./models/customer_service_bot",
                       help="輸出目錄")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="訓練 epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="每個設備的 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="梯度累積步數")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="學習率")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="最大序列長度")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日誌記錄步數")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="保存檢查點步數")

    # 其他
    parser.add_argument("--use_wandb", action="store_true",
                       help="使用 Weights & Biases 進行記錄")

    args = parser.parse_args()

    # 載入數據
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data) if args.val_data else None

    # 設置模型
    model, tokenizer = setup_model_and_tokenizer(
        model_name=args.model_name,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # 訓練
    train(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb
    )


if __name__ == "__main__":
    main()
