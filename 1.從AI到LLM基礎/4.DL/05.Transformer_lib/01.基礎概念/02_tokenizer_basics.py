"""
Tokenizer 基礎示例
展示如何使用 tokenizer 處理文本
"""

from transformers import AutoTokenizer
import torch


def example_1_basic_tokenization():
    """範例 1：基本分詞"""
    print("=" * 50)
    print("範例 1：基本分詞")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    text = "我愛使用 Hugging Face Transformers！"

    # 基本編碼
    encoding = tokenizer(text)

    print(f"原始文本: {text}")
    print(f"Token IDs: {encoding['input_ids']}")
    print(f"Attention Mask: {encoding['attention_mask']}")

    # 解碼
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    print(f"Tokens: {tokens}")

    # 還原文本
    decoded_text = tokenizer.decode(encoding['input_ids'])
    print(f"還原文本: {decoded_text}")


def example_2_padding_truncation():
    """範例 2：填充和截斷"""
    print("\n" + "=" * 50)
    print("範例 2：填充和截斷")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    texts = [
        "短句子",
        "這是一個中等長度的句子",
        "這是一個非常非常非常長的句子，用來展示截斷功能的作用，當句子超過最大長度時會被截斷"
    ]

    # 不同的填充和截斷策略
    print("\n1. 不填充，不截斷:")
    encodings = tokenizer(texts)
    for i, enc in enumerate(encodings['input_ids']):
        print(f"  文本 {i+1} 長度: {len(enc)}")

    print("\n2. 填充到批次最大長度:")
    encodings = tokenizer(texts, padding=True)
    for i, enc in enumerate(encodings['input_ids']):
        print(f"  文本 {i+1} 長度: {len(enc)}")

    print("\n3. 填充到固定長度並截斷:")
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=20)
    for i, enc in enumerate(encodings['input_ids']):
        print(f"  文本 {i+1} 長度: {len(enc)}")
        print(f"  Attention Mask: {encodings['attention_mask'][i]}")


def example_3_different_tokenizers():
    """範例 3：不同類型的 tokenizer"""
    print("\n" + "=" * 50)
    print("範例 3：不同類型的 Tokenizer")
    print("=" * 50)

    text = "unbelievable transformation"

    # WordPiece (BERT)
    print("\n1. WordPiece (BERT):")
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens_bert = tokenizer_bert.tokenize(text)
    print(f"  Tokens: {tokens_bert}")

    # BPE (GPT-2)
    print("\n2. BPE (GPT-2):")
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    tokens_gpt2 = tokenizer_gpt2.tokenize(text)
    print(f"  Tokens: {tokens_gpt2}")

    # SentencePiece (T5)
    print("\n3. SentencePiece (T5):")
    tokenizer_t5 = AutoTokenizer.from_pretrained("t5-small")
    tokens_t5 = tokenizer_t5.tokenize(text)
    print(f"  Tokens: {tokens_t5}")


def example_4_special_tokens():
    """範例 4：特殊標記"""
    print("\n" + "=" * 50)
    print("範例 4：特殊標記")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    print("預設特殊標記:")
    print(f"  PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  UNK: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"  CLS: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
    print(f"  SEP: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
    print(f"  MASK: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")

    # 添加特殊標記
    print("\n添加自定義特殊標記:")
    special_tokens = {'additional_special_tokens': ['[USER]', '[ASSISTANT]']}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"  添加了 {num_added} 個新標記")
    print(f"  新的 vocab 大小: {len(tokenizer)}")

    # 使用新標記
    text = "[USER] 你好 [ASSISTANT] 您好！"
    encoding = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    print(f"\n  原文: {text}")
    print(f"  Tokens: {tokens}")


def example_5_batch_encoding():
    """範例 5：批次編碼"""
    print("\n" + "=" * 50)
    print("範例 5：批次編碼")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    texts = [
        "第一句話",
        "第二句話",
        "第三句話",
    ]

    # PyTorch 格式
    print("1. PyTorch 格式:")
    pt_encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    print(f"  Input IDs shape: {pt_encoding['input_ids'].shape}")
    print(f"  Attention Mask shape: {pt_encoding['attention_mask'].shape}")
    print(f"  Type: {type(pt_encoding['input_ids'])}")

    # TensorFlow 格式
    print("\n2. TensorFlow 格式:")
    tf_encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )
    print(f"  Input IDs shape: {tf_encoding['input_ids'].shape}")
    print(f"  Type: {type(tf_encoding['input_ids'])}")

    # NumPy 格式
    print("\n3. NumPy 格式:")
    np_encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np"
    )
    print(f"  Input IDs shape: {np_encoding['input_ids'].shape}")
    print(f"  Type: {type(np_encoding['input_ids'])}")


def example_6_fast_tokenizer_features():
    """範例 6：Fast Tokenizer 的特殊功能"""
    print("\n" + "=" * 50)
    print("範例 6：Fast Tokenizer 的特殊功能")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)

    text = "我愛 Hugging Face"

    # 獲取 offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True)

    print(f"原始文本: {text}")
    print(f"\nTokens 和對應的文本位置:")
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])

    for token, (start, end) in zip(tokens, encoding['offset_mapping']):
        if start == end:  # 特殊標記
            print(f"  {token:15} -> 特殊標記")
        else:
            original_text = text[start:end]
            print(f"  {token:15} -> '{original_text}' (位置 {start}-{end})")


def example_7_encode_decode():
    """範例 7：編碼和解碼"""
    print("\n" + "=" * 50)
    print("範例 7：編碼和解碼")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    text = "這是一個測試句子"

    print(f"原始文本: {text}\n")

    # 編碼
    print("1. 編碼方法:")
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"  encode(): {token_ids}")

    token_ids_2 = tokenizer(text)['input_ids']
    print(f"  __call__(): {token_ids_2}")

    # 解碼
    print("\n2. 解碼方法:")
    decoded_text = tokenizer.decode(token_ids)
    print(f"  decode(): {decoded_text}")

    decoded_text_no_special = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"  decode(skip_special_tokens=True): {decoded_text_no_special}")

    # 批次解碼
    print("\n3. 批次解碼:")
    batch_ids = [token_ids, token_ids]
    decoded_batch = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
    for i, text in enumerate(decoded_batch):
        print(f"  文本 {i+1}: {text}")


def example_8_tokenizer_performance():
    """範例 8：Tokenizer 性能比較"""
    print("\n" + "=" * 50)
    print("範例 8：Fast vs Slow Tokenizer 性能比較")
    print("=" * 50)

    import time

    texts = ["這是一個測試句子" for _ in range(1000)]

    # Slow Tokenizer
    print("1. Slow Tokenizer (Python):")
    tokenizer_slow = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
    start = time.time()
    _ = tokenizer_slow(texts, padding=True, truncation=True)
    time_slow = time.time() - start
    print(f"  時間: {time_slow:.4f} 秒")

    # Fast Tokenizer
    print("\n2. Fast Tokenizer (Rust):")
    tokenizer_fast = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)
    start = time.time()
    _ = tokenizer_fast(texts, padding=True, truncation=True)
    time_fast = time.time() - start
    print(f"  時間: {time_fast:.4f} 秒")

    print(f"\n  加速比: {time_slow / time_fast:.2f}x")


def main():
    """執行所有範例"""
    print("\n🚀 Tokenizer 基礎示例\n")

    try:
        example_1_basic_tokenization()
        example_2_padding_truncation()
        example_3_different_tokenizers()
        example_4_special_tokens()
        example_5_batch_encoding()
        example_6_fast_tokenizer_features()
        example_7_encode_decode()
        example_8_tokenizer_performance()

        print("\n" + "=" * 50)
        print("✅ 所有範例執行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
