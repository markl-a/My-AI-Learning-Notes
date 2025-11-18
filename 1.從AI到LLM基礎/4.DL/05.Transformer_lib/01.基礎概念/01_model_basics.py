"""
Transformers 模型基礎示例
展示如何載入、使用和保存模型
"""

from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch

def example_1_load_model():
    """範例 1：載入預訓練模型"""
    print("=" * 50)
    print("範例 1：載入預訓練模型")
    print("=" * 50)

    # 載入模型和 tokenizer
    model_name = "bert-base-chinese"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"✓ 成功載入模型: {model_name}")
    print(f"✓ 模型類型: {type(model).__name__}")
    print(f"✓ 參數數量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Vocab 大小: {len(tokenizer)}")

    return model, tokenizer


def example_2_model_inference():
    """範例 2：模型推理"""
    print("\n" + "=" * 50)
    print("範例 2：模型推理")
    print("=" * 50)

    model_name = "bert-base-chinese"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 準備輸入
    text = "Hugging Face Transformers 是一個很棒的庫"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"輸入文本: {text}")
    print(f"Token IDs shape: {inputs['input_ids'].shape}")

    # 執行推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # 獲取最後一層的隱藏狀態
    last_hidden_state = outputs.last_hidden_state

    print(f"輸出形狀: {last_hidden_state.shape}")
    print(f"  - Batch size: {last_hidden_state.shape[0]}")
    print(f"  - Sequence length: {last_hidden_state.shape[1]}")
    print(f"  - Hidden size: {last_hidden_state.shape[2]}")


def example_3_different_precisions():
    """範例 3：不同精度載入"""
    print("\n" + "=" * 50)
    print("範例 3：不同精度載入")
    print("=" * 50)

    model_name = "gpt2"

    # FP32 (預設)
    model_fp32 = AutoModel.from_pretrained(model_name)
    size_fp32 = sum(p.numel() * p.element_size() for p in model_fp32.parameters()) / (1024 ** 2)
    print(f"FP32 模型大小: {size_fp32:.2f} MB")

    # FP16
    model_fp16 = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    size_fp16 = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / (1024 ** 2)
    print(f"FP16 模型大小: {size_fp16:.2f} MB")
    print(f"記憶體節省: {(1 - size_fp16/size_fp32) * 100:.1f}%")


def example_4_save_and_load():
    """範例 4：保存和載入模型"""
    print("\n" + "=" * 50)
    print("範例 4：保存和載入模型")
    print("=" * 50)

    model_name = "bert-base-chinese"
    save_path = "./saved_model"

    # 載入模型
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 保存模型
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✓ 模型已保存到: {save_path}")

    # 從本地載入
    loaded_model = AutoModel.from_pretrained(save_path, local_files_only=True)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True)
    print(f"✓ 模型已從本地載入")

    # 驗證
    print(f"✓ 參數數量匹配: {sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in loaded_model.parameters())}")


def example_5_model_config():
    """範例 5：模型配置"""
    print("\n" + "=" * 50)
    print("範例 5：模型配置")
    print("=" * 50)

    model_name = "bert-base-chinese"

    # 載入配置
    config = AutoConfig.from_pretrained(model_name)

    print("模型配置:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Num hidden layers: {config.num_hidden_layers}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Hidden dropout: {config.hidden_dropout_prob}")
    print(f"  - Attention dropout: {config.attention_probs_dropout_prob}")

    # 修改配置並創建新模型
    config.hidden_dropout_prob = 0.2
    config.num_hidden_layers = 6  # 減少層數

    model = AutoModel.from_config(config)
    print(f"\n✓ 使用自定義配置創建模型")
    print(f"  - 層數: {config.num_hidden_layers}")
    print(f"  - 參數數量: {sum(p.numel() for p in model.parameters()):,}")


def example_6_device_placement():
    """範例 6：設備分配"""
    print("\n" + "=" * 50)
    print("範例 6：設備分配")
    print("=" * 50)

    model_name = "gpt2"

    # 檢查可用設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"可用設備: {device}")

    if device == "cuda":
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 方法 1：手動移動到設備
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    print(f"✓ 模型已移動到 {device}")

    # 方法 2：使用 device_map（自動分配）
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    print(f"✓ 使用 device_map='auto' 自動分配設備")


def main():
    """執行所有範例"""
    print("\n🚀 Transformers 模型基礎示例\n")

    try:
        example_1_load_model()
        example_2_model_inference()
        example_3_different_precisions()
        example_4_save_and_load()
        example_5_model_config()
        example_6_device_placement()

        print("\n" + "=" * 50)
        print("✅ 所有範例執行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
