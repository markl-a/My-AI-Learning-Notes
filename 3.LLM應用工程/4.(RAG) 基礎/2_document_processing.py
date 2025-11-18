"""
文檔處理與拆分範例
展示如何載入不同格式的文檔並進行智能拆分
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Document:
    """文檔類"""
    content: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextSplitter:
    """文本拆分器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文本拆分器

        Args:
            chunk_size: 每個塊的最大字符數
            chunk_overlap: 塊之間的重疊字符數
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        將文本拆分為多個塊

        Args:
            text: 要拆分的文本

        Returns:
            拆分後的文本塊列表
        """
        chunks = []
        start = 0

        while start < len(text):
            # 計算結束位置
            end = start + self.chunk_size

            # 如果不是最後一塊，嘗試在合適的位置斷開
            if end < len(text):
                # 尋找最近的句號、換行或空格
                for separator in ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ', ' ']:
                    pos = text.rfind(separator, start, end)
                    if pos != -1:
                        end = pos + len(separator)
                        break

            # 提取塊
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 移動到下一個塊，考慮重疊
            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        拆分多個文檔

        Args:
            documents: 文檔列表

        Returns:
            拆分後的文檔列表
        """
        split_docs = []

        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk_id'] = i
                metadata['total_chunks'] = len(chunks)
                split_docs.append(Document(content=chunk, metadata=metadata))

        return split_docs


class RecursiveTextSplitter(TextSplitter):
    """遞歸文本拆分器，按層次結構拆分"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 separators: List[str] = None):
        """
        初始化遞歸文本拆分器

        Args:
            chunk_size: 每個塊的最大字符數
            chunk_overlap: 塊之間的重疊字符數
            separators: 分隔符列表，按優先級排序
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "。", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        遞歸拆分文本

        Args:
            text: 要拆分的文本

        Returns:
            拆分後的文本塊列表
        """
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        遞歸拆分文本的內部方法

        Args:
            text: 要拆分的文本
            separators: 可用的分隔符列表

        Returns:
            拆分後的文本塊列表
        """
        final_chunks = []

        # 選擇當前分隔符
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []

        # 按分隔符拆分
        if separator:
            splits = text.split(separator)
        else:
            splits = [text]

        # 處理每個拆分
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # 如果還有分隔符，繼續遞歸拆分
                if new_separators:
                    good_splits.extend(self._split_text(split, new_separators))
                else:
                    # 沒有分隔符了，直接按大小拆分
                    good_splits.append(split)

        # 合併小塊
        merged_chunks = []
        current_chunk = ""

        for split in good_splits:
            if not split.strip():
                continue

            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks


class DocumentLoader:
    """文檔載入器"""

    @staticmethod
    def load_text(file_path: str) -> Document:
        """載入文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={'source': file_path, 'type': 'text'}
        )

    @staticmethod
    def load_json(file_path: str, content_field: str = 'content') -> List[Document]:
        """載入 JSON 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and content_field in item:
                    metadata = {k: v for k, v in item.items() if k != content_field}
                    metadata['source'] = file_path
                    metadata['type'] = 'json'
                    documents.append(Document(
                        content=item[content_field],
                        metadata=metadata
                    ))
        elif isinstance(data, dict) and content_field in data:
            metadata = {k: v for k, v in data.items() if k != content_field}
            metadata['source'] = file_path
            metadata['type'] = 'json'
            documents.append(Document(
                content=data[content_field],
                metadata=metadata
            ))

        return documents

    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """載入 Markdown 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={'source': file_path, 'type': 'markdown'}
        )


def demo_basic_splitting():
    """基礎拆分演示"""
    print("=" * 60)
    print("基礎文本拆分演示")
    print("=" * 60)

    # 準備長文本
    long_text = """機器學習是人工智慧的一個重要分支。它使計算機能夠從數據中學習，而無需明確編程。

深度學習是機器學習的一個子集，它使用多層神經網絡來處理和學習數據中的複雜模式。深度學習在圖像識別、語音識別和自然語言處理等領域取得了突破性進展。

自然語言處理（NLP）專注於使計算機能夠理解、解釋和生成人類語言。現代 NLP 系統使用深度學習技術，如 Transformer 模型，來完成諸如機器翻譯、文本摘要和問答等任務。

計算機視覺是另一個重要的 AI 領域，它使計算機能夠從圖像和視頻中提取有意義的信息。卷積神經網絡（CNN）是計算機視覺中最常用的深度學習架構。

強化學習是一種機器學習方法，其中智能體通過與環境互動來學習最優策略。它已成功應用於遊戲、機器人控制和自動駕駛等領域。"""

    print(f"\n原始文本長度: {len(long_text)} 字符")
    print("\n" + "-" * 60)

    # 基礎拆分器
    splitter = TextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_text(long_text)

    print(f"\n使用基礎拆分器（chunk_size=150, overlap=20）")
    print(f"拆分為 {len(chunks)} 個塊:\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"塊 {i} (長度: {len(chunk)}):")
        print(f"{chunk}\n")
        print("-" * 60)


def demo_recursive_splitting():
    """遞歸拆分演示"""
    print("\n" + "=" * 60)
    print("遞歸文本拆分演示")
    print("=" * 60)

    text = """# 深度學習框架

## TensorFlow
TensorFlow 是由 Google 開發的開源機器學習框架。它提供了全面的工具和庫，用於構建和部署機器學習模型。

### 主要特點
- 支持多種硬件平台
- 靈活的架構
- 強大的生態系統

## PyTorch
PyTorch 是由 Facebook 開發的開源深度學習框架。它以動態計算圖和簡潔的 API 而聞名。

### 主要特點
- 動態計算圖
- Pythonic 的 API
- 強大的 GPU 加速"""

    print(f"\n原始文本長度: {len(text)} 字符")
    print("\n原始文本:")
    print(text)
    print("\n" + "-" * 60)

    # 遞歸拆分器
    splitter = RecursiveTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    print(f"\n使用遞歸拆分器（chunk_size=200, overlap=30）")
    print(f"拆分為 {len(chunks)} 個塊:\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"塊 {i} (長度: {len(chunk)}):")
        print(chunk)
        print("\n" + "-" * 60)


def demo_document_loading():
    """文檔載入演示"""
    print("\n" + "=" * 60)
    print("文檔載入演示")
    print("=" * 60)

    # 創建測試文件
    test_dir = Path("3.LLM應用工程/4.(RAG) 基礎/test_data")
    test_dir.mkdir(exist_ok=True)

    # 創建測試文本文件
    text_file = test_dir / "sample.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("這是一個測試文本文件。\n它包含多行內容。\n用於演示文檔載入功能。")

    # 創建測試 JSON 文件
    json_file = test_dir / "sample.json"
    test_data = [
        {"content": "第一篇文檔", "title": "文檔1", "author": "作者A"},
        {"content": "第二篇文檔", "title": "文檔2", "author": "作者B"}
    ]
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    # 載入文檔
    loader = DocumentLoader()

    print("\n1. 載入文本文件:")
    text_doc = loader.load_text(str(text_file))
    print(f"   內容: {text_doc.content}")
    print(f"   元數據: {text_doc.metadata}")

    print("\n2. 載入 JSON 文件:")
    json_docs = loader.load_json(str(json_file))
    for i, doc in enumerate(json_docs, 1):
        print(f"\n   文檔 {i}:")
        print(f"   內容: {doc.content}")
        print(f"   元數據: {doc.metadata}")

    # 拆分文檔
    print("\n3. 拆分文檔:")
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    split_docs = splitter.split_documents([text_doc])

    print(f"   原始文檔被拆分為 {len(split_docs)} 個塊:")
    for i, doc in enumerate(split_docs, 1):
        print(f"\n   塊 {i}:")
        print(f"   內容: {doc.content}")
        print(f"   元數據: {doc.metadata}")


def demo_metadata_preservation():
    """元數據保留演示"""
    print("\n" + "=" * 60)
    print("元數據保留演示")
    print("=" * 60)

    # 創建帶元數據的文檔
    doc = Document(
        content="這是一篇關於機器學習的長文章。" * 20,
        metadata={
            'title': '機器學習簡介',
            'author': '張三',
            'date': '2024-01-01',
            'category': '技術文檔'
        }
    )

    print("\n原始文檔元數據:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")

    # 拆分文檔
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    split_docs = splitter.split_documents([doc])

    print(f"\n文檔被拆分為 {len(split_docs)} 個塊")
    print("\n拆分後每個塊的元數據:")

    for i, split_doc in enumerate(split_docs[:3], 1):  # 只顯示前3個
        print(f"\n塊 {i}:")
        for key, value in split_doc.metadata.items():
            print(f"  {key}: {value}")
        print(f"  內容預覽: {split_doc.content[:50]}...")


def main():
    """主函數"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 13 + "文檔處理與拆分範例" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")

    demo_basic_splitting()
    demo_recursive_splitting()
    demo_document_loading()
    demo_metadata_preservation()

    print("\n\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n重點回顧:")
    print("1. 合適的塊大小和重疊能保證檢索質量")
    print("2. 遞歸拆分按層次結構拆分，保持語義完整性")
    print("3. 元數據在拆分過程中被保留，方便追溯來源")
    print("4. 不同格式的文檔需要不同的載入器處理")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
