# 練習 2.1：文檔切分策略

**難度**: ⭐⭐ 基礎
**預計時間**: 1 小時
**前置知識**: Python、基本 NLP 概念

## 學習目標

完成本練習後，你將能夠：

- [ ] 理解文檔切分對 RAG 效果的影響
- [ ] 實現多種切分策略
- [ ] 根據文檔類型選擇合適的切分方法
- [ ] 評估切分品質

## 背景知識

文檔切分 (Chunking) 是 RAG 系統的關鍵步驟。好的切分應該：

1. **語義完整**：每個 chunk 包含完整的語義單元
2. **大小適中**：太大增加噪音，太小丟失上下文
3. **適當重疊**：避免信息在邊界處斷裂

## 練習任務

### 任務 1：實現基礎切分器

```python
from typing import List

def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    固定大小切分

    Args:
        text: 原始文本
        chunk_size: 每個 chunk 的字符數
        overlap: 重疊字符數

    Returns:
        切分後的 chunk 列表
    """
    # TODO: 實現固定大小切分
    chunks = []
    # 你的代碼
    return chunks

def sentence_chunking(text: str, max_sentences: int = 5) -> List[str]:
    """
    按句子切分

    Args:
        text: 原始文本
        max_sentences: 每個 chunk 的最大句子數

    Returns:
        切分後的 chunk 列表
    """
    # TODO: 實現句子切分
    chunks = []
    # 你的代碼
    return chunks

def semantic_chunking(text: str, similarity_threshold: float = 0.5) -> List[str]:
    """
    語義切分：在語義變化處分割

    Args:
        text: 原始文本
        similarity_threshold: 相似度閾值

    Returns:
        切分後的 chunk 列表
    """
    # TODO: 使用 embedding 實現語義切分
    chunks = []
    # 你的代碼
    return chunks
```

### 任務 2：比較不同策略

使用以下測試文檔：

```python
test_document = """
# 機器學習簡介

機器學習是人工智能的一個分支，專注於開發能夠從數據中學習的算法。
它使計算機能夠在沒有明確編程的情況下執行任務。

## 監督學習

監督學習使用標記的訓練數據。常見算法包括：
- 線性回歸
- 決策樹
- 神經網絡

## 非監督學習

非監督學習處理未標記的數據。主要應用：
- 聚類分析
- 降維
- 異常檢測

## 深度學習

深度學習是機器學習的子集，使用多層神經網絡。
近年來在圖像識別、自然語言處理等領域取得重大突破。
"""

# 測試三種切分方法
results = {
    "fixed": fixed_size_chunking(test_document, chunk_size=200, overlap=20),
    "sentence": sentence_chunking(test_document, max_sentences=3),
    "semantic": semantic_chunking(test_document)
}

# 分析結果
for method, chunks in results.items():
    print(f"\n=== {method} ({len(chunks)} chunks) ===")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:50]}...")
```

### 任務 3：Markdown 感知切分

針對 Markdown 文檔，實現結構感知的切分：

```python
def markdown_chunking(text: str) -> List[dict]:
    """
    Markdown 結構感知切分

    Returns:
        包含 metadata 的 chunk 列表
        [{"content": "...", "header": "標題", "level": 2}, ...]
    """
    # TODO: 實現 Markdown 切分
    # 提示：按標題層級切分，保留標題信息
    pass
```

### 任務 4：評估切分品質

實現切分品質評估：

```python
def evaluate_chunking(chunks: List[str], questions: List[str]) -> dict:
    """
    評估切分品質

    指標：
    - 平均 chunk 大小
    - 大小標準差（越小越一致）
    - 語義完整性分數
    """
    # TODO: 實現評估邏輯
    metrics = {
        "avg_size": 0,
        "size_std": 0,
        "semantic_score": 0
    }
    return metrics
```

## 驗證方法

- [ ] `fixed_size_chunking` 輸出 chunk 大小一致（±5%）
- [ ] `sentence_chunking` 每個 chunk 包含完整句子
- [ ] `markdown_chunking` 正確識別並保留標題信息
- [ ] 評估函數能區分不同切分策略的品質

## 參考資料

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Semantic Chunking 論文](https://arxiv.org/abs/2312.06648)

## 下一步

完成本練習後，繼續學習：
- [練習 2.2：向量檢索優化](./02-retrieval.md)
