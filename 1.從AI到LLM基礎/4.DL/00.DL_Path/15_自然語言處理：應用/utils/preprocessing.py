"""
數據預處理工具模組
Data Preprocessing Utilities for NLP

提供文本清理、分詞、詞表構建等功能
"""

import re
import string
from typing import List, Dict, Optional, Union
from collections import Counter
import numpy as np


class TextCleaner:
    """文本清理器"""

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emoji: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True,
    ):
        """
        Args:
            lowercase: 是否轉換為小寫
            remove_html: 是否移除 HTML 標籤
            remove_urls: 是否移除 URL
            remove_emoji: 是否移除表情符號
            remove_punctuation: 是否移除標點符號
            remove_numbers: 是否移除數字
            remove_extra_spaces: 是否移除多餘空格
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emoji = remove_emoji
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces

    def clean(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            return ""

        # 移除 HTML 標籤
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)

        # 移除 URL
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)

        # 移除表情符號
        if self.remove_emoji:
            # Unicode emoji pattern
            emoji_pattern = re.compile(
                "["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE
            )
            text = emoji_pattern.sub(r'', text)

        # 轉換為小寫
        if self.lowercase:
            text = text.lower()

        # 移除標點符號
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # 移除數字
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # 移除多餘空格
        if self.remove_extra_spaces:
            text = ' '.join(text.split())

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清理文本"""
        return [self.clean(text) for text in texts]


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emoji: bool = False,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
) -> str:
    """
    清理單個文本 (便捷函數)

    Args:
        text: 輸入文本
        其他參數同 TextCleaner

    Returns:
        清理後的文本

    Example:
        >>> text = "Check out https://example.com <br> Great product! 😊"
        >>> clean_text(text, remove_emoji=True)
        'check out great product!'
    """
    cleaner = TextCleaner(
        lowercase=lowercase,
        remove_html=remove_html,
        remove_urls=remove_urls,
        remove_emoji=remove_emoji,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
    )
    return cleaner.clean(text)


def tokenize_english(text: str, method: str = 'simple') -> List[str]:
    """
    英文分詞

    Args:
        text: 輸入文本
        method: 分詞方法 ('simple', 'nltk', 'spacy')

    Returns:
        詞元列表

    Example:
        >>> tokenize_english("Hello, world!")
        ['hello', 'world']
    """
    if method == 'simple':
        # 簡單的基於空格和標點的分詞
        text = text.lower()
        # 保留字母和空格
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.split()

    elif method == 'nltk':
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            return word_tokenize(text.lower())
        except ImportError:
            print("NLTK not installed. Falling back to simple tokenization.")
            return tokenize_english(text, method='simple')

    elif method == 'spacy':
        try:
            import spacy
            # 需要先下載: python -m spacy download en_core_web_sm
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            return [token.text.lower() for token in doc]
        except ImportError:
            print("spaCy not installed. Falling back to simple tokenization.")
            return tokenize_english(text, method='simple')

    else:
        raise ValueError(f"Unknown tokenization method: {method}")


def tokenize_chinese(text: str, method: str = 'jieba') -> List[str]:
    """
    中文分詞

    Args:
        text: 輸入文本
        method: 分詞方法 ('jieba', 'pkuseg', 'hanlp')

    Returns:
        詞元列表

    Example:
        >>> tokenize_chinese("我愛自然語言處理")
        ['我', '愛', '自然語言處理']
    """
    if method == 'jieba':
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            print("jieba not installed. Install with: pip install jieba")
            return list(text)  # 字符級別

    elif method == 'pkuseg':
        try:
            import pkuseg
            seg = pkuseg.pkuseg()
            return seg.cut(text)
        except ImportError:
            print("pkuseg not installed. Install with: pip install pkuseg")
            return tokenize_chinese(text, method='jieba')

    elif method == 'hanlp':
        try:
            import hanlp
            tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
            return tokenizer(text)
        except ImportError:
            print("HanLP not installed. Install with: pip install hanlp")
            return tokenize_chinese(text, method='jieba')

    else:
        raise ValueError(f"Unknown tokenization method: {method}")


class Vocabulary:
    """詞表類"""

    def __init__(
        self,
        tokens: Optional[List[List[str]]] = None,
        min_freq: int = 0,
        reserved_tokens: Optional[List[str]] = None,
    ):
        """
        構建詞表

        Args:
            tokens: 詞元列表的列表
            min_freq: 最小詞頻
            reserved_tokens: 保留詞元 (如 <pad>, <unk>, <bos>, <eos>)
        """
        self.token_to_idx = {}
        self.idx_to_token = []
        self.token_freqs = Counter()

        # 添加保留詞元
        if reserved_tokens:
            self.idx_to_token = reserved_tokens.copy()
        else:
            self.idx_to_token = ['<unk>', '<pad>']

        # 構建詞表
        if tokens:
            self._build_vocab(tokens, min_freq)

        # 構建映射
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'

    def _build_vocab(self, tokens: List[List[str]], min_freq: int):
        """構建詞表"""
        # 統計詞頻
        for token_list in tokens:
            self.token_freqs.update(token_list)

        # 過濾低頻詞
        for token, freq in self.token_freqs.items():
            if freq >= min_freq and token not in self.idx_to_token:
                self.idx_to_token.append(token)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """獲取詞元索引"""
        if isinstance(tokens, str):
            return self.token_to_idx.get(tokens, self.token_to_idx[self.unk_token])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        """索引轉詞元"""
        if isinstance(indices, int):
            return self.idx_to_token[indices]
        return [self.to_tokens(idx) for idx in indices]

    def save(self, filepath: str):
        """保存詞表"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'idx_to_token': self.idx_to_token,
                'token_freqs': dict(self.token_freqs),
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """加載詞表"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = cls()
        vocab.idx_to_token = data['idx_to_token']
        vocab.token_freqs = Counter(data['token_freqs'])
        vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
        return vocab


def pad_sequences(
    sequences: List[List[int]],
    max_length: Optional[int] = None,
    padding: str = 'post',
    truncating: str = 'post',
    value: int = 0,
) -> np.ndarray:
    """
    序列填充

    Args:
        sequences: 序列列表
        max_length: 最大長度 (None 則使用最長序列的長度)
        padding: 填充位置 ('pre' 或 'post')
        truncating: 截斷位置 ('pre' 或 'post')
        value: 填充值

    Returns:
        填充後的數組 shape: (len(sequences), max_length)

    Example:
        >>> seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> pad_sequences(seqs, max_length=4)
        array([[1, 2, 3, 0],
               [4, 5, 0, 0],
               [6, 7, 8, 9]])
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = np.full((len(sequences), max_length), value, dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq = seq[:max_length]  # 截斷

        if not seq:
            continue

        if truncating == 'pre':
            seq = seq[-max_length:]
        else:  # post
            seq = seq[:max_length]

        if padding == 'post':
            padded[i, :len(seq)] = seq
        else:  # pre
            padded[i, -len(seq):] = seq

    return padded


def create_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    創建 n-gram

    Args:
        tokens: 詞元列表
        n: n-gram 的 n

    Returns:
        n-gram 列表

    Example:
        >>> create_ngrams(['I', 'love', 'NLP'], n=2)
        ['I love', 'love NLP']
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


def augment_text(
    text: str,
    method: str = 'synonym',
    num_aug: int = 1,
) -> List[str]:
    """
    文本數據增強

    Args:
        text: 輸入文本
        method: 增強方法 ('synonym', 'back_translation', 'random_insertion')
        num_aug: 增強樣本數

    Returns:
        增強後的文本列表

    Example:
        >>> augment_text("I love this movie", method='synonym', num_aug=2)
        ['I adore this movie', 'I love this film']
    """
    augmented = []

    if method == 'synonym':
        # 使用 nlpaug 進行同義詞替換
        try:
            import nlpaug.augmenter.word as naw
            aug = naw.SynonymAug(aug_src='wordnet')
            for _ in range(num_aug):
                augmented.append(aug.augment(text))
        except ImportError:
            print("nlpaug not installed. Install with: pip install nlpaug")
            augmented = [text] * num_aug

    elif method == 'back_translation':
        # 回譯增強 (需要翻譯模型)
        print("Back translation requires translation model (e.g., Helsinki-NLP)")
        augmented = [text] * num_aug

    elif method == 'random_insertion':
        # 隨機插入
        try:
            import nlpaug.augmenter.word as naw
            aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased')
            for _ in range(num_aug):
                augmented.append(aug.augment(text))
        except ImportError:
            print("nlpaug not installed.")
            augmented = [text] * num_aug

    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    return augmented


# 便捷函數
def create_vocab(
    texts: List[str],
    tokenizer_fn=None,
    min_freq: int = 2,
    reserved_tokens: Optional[List[str]] = None,
) -> Vocabulary:
    """
    從文本列表創建詞表

    Args:
        texts: 文本列表
        tokenizer_fn: 分詞函數
        min_freq: 最小詞頻
        reserved_tokens: 保留詞元

    Returns:
        Vocabulary 對象

    Example:
        >>> texts = ["I love NLP", "NLP is great"]
        >>> vocab = create_vocab(texts, tokenizer_fn=tokenize_english)
        >>> len(vocab)
        6  # <unk>, <pad>, I, love, NLP, is, great
    """
    if tokenizer_fn is None:
        tokenizer_fn = tokenize_english

    # 分詞
    tokenized_texts = [tokenizer_fn(text) for text in texts]

    # 創建詞表
    vocab = Vocabulary(
        tokens=tokenized_texts,
        min_freq=min_freq,
        reserved_tokens=reserved_tokens,
    )

    return vocab


if __name__ == '__main__':
    # 測試代碼
    print("=" * 50)
    print("測試文本清理")
    print("=" * 50)

    text = "Check out https://example.com <br> Great product! 😊 Price: $99"
    print(f"原始文本: {text}")
    print(f"清理後: {clean_text(text, remove_emoji=True, remove_punctuation=True)}")

    print("\n" + "=" * 50)
    print("測試英文分詞")
    print("=" * 50)

    text = "Natural Language Processing is amazing!"
    print(f"原始文本: {text}")
    print(f"分詞結果: {tokenize_english(text)}")

    print("\n" + "=" * 50)
    print("測試詞表構建")
    print("=" * 50)

    texts = ["I love NLP", "NLP is great", "I love AI"]
    vocab = create_vocab(texts, min_freq=1)
    print(f"詞表大小: {len(vocab)}")
    print(f"詞表: {vocab.idx_to_token}")
    print(f"'NLP' 的索引: {vocab['NLP']}")
    print(f"索引 2 的詞元: {vocab.to_tokens(2)}")

    print("\n" + "=" * 50)
    print("測試序列填充")
    print("=" * 50)

    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    padded = pad_sequences(seqs, max_length=5, value=0)
    print(f"原始序列: {seqs}")
    print(f"填充後:\n{padded}")

    print("\n✅ 所有測試完成！")
