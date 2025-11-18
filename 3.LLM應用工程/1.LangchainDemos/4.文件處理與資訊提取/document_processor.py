"""
文件處理與資訊提取
支援多種文件格式：PDF、Word、TXT、網頁等
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 添加父目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_environment, get_llm, setup_langsmith

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DocumentProcessor:
    """文件處理器"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        load_environment()
        setup_langsmith()
        self.llm = get_llm(model=model_name)
        print("✓ 文件處理器初始化完成")

    def load_document(self, file_path: str) -> List[Dict]:
        """
        載入文件

        Args:
            file_path: 文件路徑或 URL

        Returns:
            文件內容列表
        """
        path = Path(file_path)

        # 根據檔案類型選擇適當的 loader
        if file_path.startswith("http"):
            loader = WebBaseLoader(file_path)
        elif path.suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif path.suffix == ".txt":
            loader = TextLoader(file_path)
        elif path.suffix in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"不支援的檔案格式: {path.suffix}")

        documents = loader.load()
        print(f"✓ 已載入 {len(documents)} 個文件片段")
        return documents

    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        總結文件內容

        Args:
            text: 要總結的文字
            max_length: 總結的最大字數

        Returns:
            總結內容
        """
        template = """請用繁體中文總結以下內容，不超過 {max_length} 字。
重點要清晰、簡潔。

內容：
{text}

總結："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        summary = chain.invoke({"text": text, "max_length": max_length})
        return summary.strip()

    def extract_info(
        self,
        text: str,
        info_type: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        從文件中提取特定資訊

        Args:
            text: 文件內容
            info_type: 資訊類型（如：人名、日期、金額等）
            custom_prompt: 自訂提示詞

        Returns:
            提取的資訊
        """
        if custom_prompt:
            template = custom_prompt + "\n\n內容：\n{text}"
        else:
            template = """請從以下內容中提取所有的 {info_type}。
以列表格式返回，用繁體中文。

內容：
{text}

提取結果："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({"text": text, "info_type": info_type})
        return result.strip()

    def answer_questions(self, documents: List, question: str) -> str:
        """
        基於文件回答問題

        Args:
            documents: 文件列表
            question: 問題

        Returns:
            答案
        """
        # 合併文件內容
        text = "\n\n".join([doc.page_content for doc in documents])

        template = """基於以下文件內容回答問題。
如果文件中沒有相關資訊，請誠實說不知道。
請用繁體中文回答。

文件內容：
{text}

問題：{question}

答案："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({"text": text, "question": question})
        return answer.strip()

    def translate(self, text: str, target_language: str = "繁體中文") -> str:
        """
        翻譯文件

        Args:
            text: 要翻譯的文字
            target_language: 目標語言

        Returns:
            翻譯結果
        """
        template = """請將以下內容翻譯成{target_language}。
保持原文的格式和結構。

原文：
{text}

譯文："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        translation = chain.invoke({
            "text": text,
            "target_language": target_language
        })
        return translation.strip()


# 示範範例
def demo_summarization():
    """示範文件總結"""
    print("=" * 80)
    print("示範：文件總結")
    print("=" * 80)

    processor = DocumentProcessor()

    # 範例文件
    text = """
    人工智慧（Artificial Intelligence, AI）是電腦科學的一個分支，
    目標是創建能夠執行通常需要人類智慧才能完成的任務的系統。
    這些任務包括視覺感知、語音識別、決策制定和語言翻譯等。

    機器學習是 AI 的一個子領域，它使用統計技術讓電腦系統能夠從數據中「學習」，
    而不需要被明確編程。深度學習則是機器學習的一個分支，
    使用人工神經網路來模仿人腦的學習過程。

    近年來，AI 技術取得了巨大進展，特別是在自然語言處理、
    電腦視覺和強化學習等領域。ChatGPT 等大型語言模型的出現，
    更是將 AI 應用推向了新的高度。
    """

    summary = processor.summarize(text, max_length=100)
    print(f"\n原文長度: {len(text)} 字")
    print(f"\n總結:\n{summary}")


def demo_info_extraction():
    """示範資訊提取"""
    print("\n" + "=" * 80)
    print("示範：資訊提取")
    print("=" * 80)

    processor = DocumentProcessor()

    text = """
    會議記錄
    日期：2024年1月15日
    時間：下午2:00-4:00
    地點：會議室A
    出席人員：張三、李四、王五

    討論事項：
    1. 新產品開發計畫
    2. Q1季度預算分配
    3. 市場推廣策略

    決議：
    - 張三負責產品開發
    - 李四處理預算事宜
    - 王五規劃市場活動
    - 下次會議時間：2024年2月1日
    """

    # 提取日期
    dates = processor.extract_info(text, "日期和時間")
    print(f"\n日期時間:\n{dates}")

    # 提取人名
    names = processor.extract_info(text, "人名")
    print(f"\n出席人員:\n{names}")

    # 提取決議事項
    decisions = processor.extract_info(
        text,
        custom_prompt="請列出會議中的所有決議事項"
    )
    print(f"\n決議事項:\n{decisions}")


def demo_qa():
    """示範文件問答"""
    print("\n" + "=" * 80)
    print("示範：文件問答")
    print("=" * 80)

    processor = DocumentProcessor()

    # 建立範例文件
    from langchain.schema import Document

    documents = [
        Document(page_content="""
        Python 是一種高階、通用的程式語言。
        它由 Guido van Rossum 創建，於1991年首次發布。
        Python 的設計哲學強調程式碼的可讀性，
        並使用大量的空白字元。
        """),
        Document(page_content="""
        Python 支援多種程式設計範式，包括結構化、物件導向和函數式程式設計。
        它擁有豐富的標準庫，常被稱為「內建電池」的語言。
        Python 廣泛應用於網頁開發、數據分析、機器學習、
        人工智慧等領域。
        """)
    ]

    # 提問
    questions = [
        "Python 是誰創建的？",
        "Python 有哪些特點？",
        "Python 可以用在哪些領域？"
    ]

    for question in questions:
        print(f"\n問題: {question}")
        answer = processor.answer_questions(documents, question)
        print(f"答案: {answer}")


if __name__ == "__main__":
    try:
        demo_summarization()
        demo_info_extraction()
        demo_qa()

        print("\n" + "=" * 80)
        print("✓ 所有示範執行完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
