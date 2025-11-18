# 資料集準備與建立

## 目錄
1. [原始資料收集與清理](#31-原始資料收集與清理)
2. [Instruction Dataset 準備與格式化](#32-instruction-dataset-準備與格式化)
3. [資料增強技術](#33-資料增強技術)
4. [使用現有資料集](#34-使用現有資料集)
5. [資料品質評估與過濾](#35-資料品質評估與過濾)
6. [RLHF/RLAIF 資料集準備](#36-rlhfrlaif-資料集準備)
7. [資料標註流程與工具](#37-資料標註流程與工具)
8. [資料集版本控制與管理](#38-資料集版本控制與管理)
9. [資料隱私與合規性](#39-資料隱私與合規性)
10. [實作範例](#310-實作範例)
11. [實際案例研究](#311-實際案例研究)
12. [速查表與最佳實踐](#312-速查表與最佳實踐)

---

## 3.1 原始資料收集與清理

### 3.1.1 資料來源

**公開資料來源**：
- **網路爬取**：Wikipedia、Reddit、Stack Overflow、新聞網站
- **學術資料庫**：arXiv、PubMed、學術期刊
- **開源專案**：GitHub、GitLab（程式碼資料集）
- **公開資料集**：Common Crawl、The Pile、C4
- **社群平台**：Twitter、論壇、問答平台

**私有資料來源**：
- 企業內部文件
- 客服對話記錄
- 產品文檔
- 技術支援記錄

### 3.1.2 網路爬蟲實作

#### 基礎爬蟲（使用 BeautifulSoup）

```python
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict
import json

class WebScraper:
    """簡單的網頁爬蟲"""

    def __init__(self, user_agent=None):
        self.session = requests.Session()
        if user_agent:
            self.session.headers.update({'User-Agent': user_agent})
        else:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

    def fetch_page(self, url: str, timeout=10):
        """抓取網頁內容"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except requests.RequestException as e:
            print(f"抓取失敗 {url}: {e}")
            return None

    def extract_text(self, html: str, remove_tags=['script', 'style', 'nav', 'footer']):
        """從 HTML 中提取純文字"""
        soup = BeautifulSoup(html, 'html.parser')

        # 移除不需要的標籤
        for tag in remove_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # 提取文字
        text = soup.get_text(separator='\n', strip=True)

        # 清理多餘空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text

    def extract_article(self, html: str):
        """提取文章內容（針對新聞、部落格等）"""
        soup = BeautifulSoup(html, 'html.parser')

        # 嘗試找到主要內容區域
        article = None
        for tag in ['article', 'main', 'div.content', 'div.post']:
            article = soup.find(tag)
            if article:
                break

        if not article:
            article = soup

        # 提取標題
        title = None
        for tag in ['h1', 'h2', 'title']:
            title_elem = article.find(tag)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break

        # 提取段落
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p')]
        content = '\n\n'.join([p for p in paragraphs if len(p) > 50])

        return {
            'title': title,
            'content': content,
            'length': len(content)
        }

# 使用範例
scraper = WebScraper()

# 抓取單個頁面
url = "https://zh.wikipedia.org/wiki/人工智慧"
html = scraper.fetch_page(url)

if html:
    # 方法1：提取所有文字
    text = scraper.extract_text(html)
    print(f"提取的文字長度: {len(text)}")

    # 方法2：提取文章結構
    article = scraper.extract_article(html)
    print(f"標題: {article['title']}")
    print(f"內容長度: {article['length']}")
```

#### 進階爬蟲（處理動態網頁）

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

class DynamicScraper:
    """處理動態網頁的爬蟲（需要 Selenium）"""

    def __init__(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

    def fetch_dynamic_page(self, url: str, wait_for_element=None):
        """抓取動態渲染的網頁"""
        self.driver.get(url)

        # 等待特定元素出現
        if wait_for_element:
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
            )

        # 滾動到底部（觸發懶加載）
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        return self.driver.page_source

    def close(self):
        self.driver.quit()

# 使用範例（需要安裝 ChromeDriver）
# scraper = DynamicScraper(headless=True)
# html = scraper.fetch_dynamic_page("https://example.com", wait_for_element=".content")
# scraper.close()
```

### 3.1.3 資料清理

#### 文字清理函數

```python
import re
from typing import List
import unicodedata

class TextCleaner:
    """文字清理工具"""

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """移除 HTML 標籤"""
        return re.sub(r'<[^>]+>', '', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """移除 URL"""
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """移除電子郵件地址"""
        return re.sub(r'\S+@\S+', '', text)

    @staticmethod
    def remove_special_chars(text: str, keep_chinese=True, keep_english=True, keep_numbers=True) -> str:
        """移除特殊字元"""
        pattern_parts = []
        if keep_chinese:
            pattern_parts.append(r'\u4e00-\u9fff')
        if keep_english:
            pattern_parts.append(r'a-zA-Z')
        if keep_numbers:
            pattern_parts.append(r'0-9')

        pattern = f'[^{" ".join(pattern_parts)}\\s.,!?;:()「」『』""、。！?；:（）]'
        return re.sub(pattern, '', text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """規範化空白字元"""
        # 移除多餘的空格
        text = re.sub(r' +', ' ', text)
        # 移除多餘的換行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 移除行首尾空白
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()

    @staticmethod
    def remove_repeated_chars(text: str, max_repeat=2) -> str:
        """移除重複字元（如：哈哈哈哈哈 -> 哈哈）"""
        def replace_func(match):
            char = match.group(1)
            return char * min(len(match.group(0)), max_repeat)

        return re.sub(r'(.)\1{2,}', replace_func, text)

    @staticmethod
    def filter_by_length(text: str, min_length=10, max_length=10000) -> bool:
        """根據長度過濾文字"""
        return min_length <= len(text) <= max_length

    @staticmethod
    def detect_language(text: str) -> str:
        """簡單的語言檢測"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'\S', text))

        if total_chars == 0:
            return 'unknown'

        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.5:
            return 'zh'
        elif chinese_ratio < 0.1:
            return 'en'
        else:
            return 'mixed'

    @classmethod
    def clean_text(cls, text: str,
                   remove_html=True,
                   remove_urls=True,
                   remove_emails=True,
                   normalize_ws=True,
                   remove_repeated=True) -> str:
        """綜合清理函數"""
        if remove_html:
            text = cls.remove_html_tags(text)
        if remove_urls:
            text = cls.remove_urls(text)
        if remove_emails:
            text = cls.remove_emails(text)
        if remove_repeated:
            text = cls.remove_repeated_chars(text)
        if normalize_ws:
            text = cls.normalize_whitespace(text)

        return text

# 使用範例
cleaner = TextCleaner()

raw_text = """
<p>這是一個範例文字 http://example.com 包含 HTML 標籤和 URL</p>
聯絡我們: example@email.com
哈哈哈哈哈哈哈哈哈
"""

cleaned = cleaner.clean_text(raw_text)
print("清理後的文字:")
print(cleaned)

# 語言檢測
lang = cleaner.detect_language("這是中文文字 with some English")
print(f"檢測到的語言: {lang}")
```

#### 去重複

```python
from collections import defaultdict
import hashlib
from typing import List, Set

class Deduplicator:
    """文字去重複工具"""

    @staticmethod
    def exact_dedup(texts: List[str]) -> List[str]:
        """精確去重（完全相同的文字）"""
        seen = set()
        unique_texts = []

        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def hash_dedup(texts: List[str]) -> List[str]:
        """基於雜湊的去重"""
        seen_hashes = set()
        unique_texts = []

        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def fuzzy_dedup(texts: List[str], similarity_threshold=0.9) -> List[str]:
        """模糊去重（基於相似度）"""
        from difflib import SequenceMatcher

        unique_texts = []

        for text in texts:
            is_duplicate = False
            for unique_text in unique_texts:
                similarity = SequenceMatcher(None, text, unique_text).ratio()
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_texts.append(text)

        return unique_texts

    @staticmethod
    def n_gram_dedup(texts: List[str], n=3, overlap_threshold=0.8) -> List[str]:
        """基於 n-gram 的去重"""
        def get_ngrams(text, n):
            words = text.split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

        unique_texts = []

        for text in texts:
            text_ngrams = get_ngrams(text, n)
            is_duplicate = False

            for unique_text in unique_texts:
                unique_ngrams = get_ngrams(unique_text, n)
                if len(text_ngrams) == 0 or len(unique_ngrams) == 0:
                    continue

                overlap = len(text_ngrams & unique_ngrams) / min(len(text_ngrams), len(unique_ngrams))
                if overlap >= overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_texts.append(text)

        return unique_texts

# 使用範例
texts = [
    "這是第一段文字",
    "這是第一段文字",  # 完全重複
    "這是第一段文字，稍有不同",  # 相似
    "這是完全不同的內容",
]

dedup = Deduplicator()

# 精確去重
exact_unique = dedup.exact_dedup(texts)
print(f"精確去重: {len(exact_unique)} 條")

# 模糊去重
fuzzy_unique = dedup.fuzzy_dedup(texts, similarity_threshold=0.8)
print(f"模糊去重: {len(fuzzy_unique)} 條")
```

---

## 3.2 Instruction Dataset 準備與格式化

### 3.2.1 Instruction Tuning 資料格式

**標準格式**：

```json
{
  "instruction": "任務描述或指令",
  "input": "輸入內容（可選）",
  "output": "期望的輸出"
}
```

**範例**：

```json
[
  {
    "instruction": "將以下句子翻譯成英文",
    "input": "機器學習是人工智慧的一個分支。",
    "output": "Machine learning is a branch of artificial intelligence."
  },
  {
    "instruction": "解釋什麼是量子計算",
    "input": "",
    "output": "量子計算是利用量子力學現象（如疊加態和糾纏）進行計算的技術。與傳統計算機使用位元（0或1）不同，量子計算機使用量子位元（qubit），可以同時表示0和1，這使得某些特定問題的計算速度遠超傳統計算機。"
  }
]
```

### 3.2.2 資料集轉換工具

```python
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import random

@dataclass
class InstructionExample:
    """Instruction 範例資料結構"""
    instruction: str
    input: str
    output: str

    def to_dict(self) -> Dict:
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output
        }

    def to_alpaca_format(self) -> str:
        """轉換為 Alpaca 格式"""
        if self.input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{self.instruction}

### Input:
{self.input}

### Response:
{self.output}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{self.instruction}

### Response:
{self.output}"""

    def to_chatml_format(self) -> List[Dict]:
        """轉換為 ChatML 格式"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        if self.input:
            user_content = f"{self.instruction}\n\n{self.input}"
        else:
            user_content = self.instruction

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.output})

        return messages

class InstructionDatasetBuilder:
    """Instruction 資料集建構工具"""

    def __init__(self):
        self.examples: List[InstructionExample] = []

    def add_example(self, instruction: str, output: str, input: str = ""):
        """添加一個範例"""
        self.examples.append(InstructionExample(instruction, input, output))

    def add_qa_pair(self, question: str, answer: str):
        """添加問答對"""
        self.add_example(
            instruction="請回答以下問題",
            input=question,
            output=answer
        )

    def add_translation_pair(self, source: str, target: str, source_lang="中文", target_lang="英文"):
        """添加翻譯對"""
        self.add_example(
            instruction=f"將以下{source_lang}翻譯成{target_lang}",
            input=source,
            output=target
        )

    def add_summarization_pair(self, text: str, summary: str):
        """添加摘要對"""
        self.add_example(
            instruction="請為以下文字寫一個摘要",
            input=text,
            output=summary
        )

    def from_json(self, json_path: str):
        """從 JSON 文件載入"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            self.examples.append(InstructionExample(
                instruction=item['instruction'],
                input=item.get('input', ''),
                output=item['output']
            ))

    def to_json(self, output_path: str):
        """儲存為 JSON"""
        data = [ex.to_dict() for ex in self.examples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def to_alpaca_format(self, output_path: str):
        """儲存為 Alpaca 格式的純文字"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in self.examples:
                f.write(ex.to_alpaca_format())
                f.write('\n\n' + '='*80 + '\n\n')

    def split_train_val(self, val_ratio=0.1, shuffle=True):
        """分割訓練集和驗證集"""
        examples = self.examples.copy()
        if shuffle:
            random.shuffle(examples)

        split_idx = int(len(examples) * (1 - val_ratio))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        train_builder = InstructionDatasetBuilder()
        train_builder.examples = train_examples

        val_builder = InstructionDatasetBuilder()
        val_builder.examples = val_examples

        return train_builder, val_builder

    def __len__(self):
        return len(self.examples)

# 使用範例
builder = InstructionDatasetBuilder()

# 添加不同類型的範例
builder.add_qa_pair(
    question="什麼是深度學習?",
    answer="深度學習是機器學習的一個子領域，使用多層神經網路來學習資料的表示。"
)

builder.add_translation_pair(
    source="你好，世界！",
    target="Hello, World!"
)

builder.add_example(
    instruction="將以下程式碼轉換為 Python",
    input="for (int i = 0; i < 10; i++) { print(i); }",
    output="for i in range(10):\n    print(i)"
)

# 儲存
builder.to_json("instruction_dataset.json")

# 分割訓練集和驗證集
train, val = builder.split_train_val(val_ratio=0.1)
print(f"訓練集: {len(train)} 條，驗證集: {len(val)} 條")
```

### 3.2.3 多輪對話格式

```python
from typing import List, Dict

class ConversationDatasetBuilder:
    """多輪對話資料集建構工具"""

    def __init__(self):
        self.conversations: List[List[Dict]] = []

    def add_conversation(self, messages: List[Dict[str, str]]):
        """
        添加一個對話
        messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        self.conversations.append(messages)

    def from_chat_logs(self, chat_logs: List[List[str]]):
        """
        從聊天記錄轉換
        chat_logs: [["user_msg", "assistant_msg", "user_msg", "assistant_msg", ...]]
        """
        for log in chat_logs:
            messages = []
            for i, msg in enumerate(log):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})
            self.conversations.append(messages)

    def to_sharegpt_format(self) -> List[Dict]:
        """轉換為 ShareGPT 格式"""
        formatted = []
        for conv in self.conversations:
            formatted.append({
                "conversations": conv
            })
        return formatted

    def to_jsonl(self, output_path: str):
        """儲存為 JSONL 格式（每行一個對話）"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in self.conversations:
                json.dump({"messages": conv}, f, ensure_ascii=False)
                f.write('\n')

    def to_training_format(self, tokenizer, max_length=2048):
        """轉換為訓練格式（需要 tokenizer）"""
        formatted_data = []

        for conv in self.conversations:
            # 使用 tokenizer 的 apply_chat_template
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_data.append(text)

        return formatted_data

# 使用範例
conv_builder = ConversationDatasetBuilder()

# 添加單輪對話
conv_builder.add_conversation([
    {"role": "user", "content": "什麼是機器學習？"},
    {"role": "assistant", "content": "機器學習是一種人工智慧技術..."}
])

# 添加多輪對話
conv_builder.add_conversation([
    {"role": "user", "content": "Python 怎麼讀取檔案？"},
    {"role": "assistant", "content": "使用 open() 函數..."},
    {"role": "user", "content": "能給個範例嗎？"},
    {"role": "assistant", "content": "當然！這是範例：\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"}
])

# 儲存
conv_builder.to_jsonl("conversations.jsonl")
print(f"已建立 {len(conv_builder.conversations)} 個對話")
```

---

## 3.3 資料增強技術

### 3.3.1 Evol-Instruct

**概念**：透過 LLM 自動演化和擴展指令，提升資料集多樣性和複雜度。

```python
from openai import OpenAI
import os

class EvolInstructGenerator:
    """Evol-Instruct 資料增強工具"""

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def evolve_instruction(self, instruction: str, method: str = "deepen") -> str:
        """
        演化指令

        methods:
        - deepen: 增加深度（增加推理步驟）
        - breadth: 增加廣度（拓展主題）
        - constraint: 添加約束條件
        - concretize: 具體化（添加更多細節）
        - reasoning: 增加推理需求
        """

        prompts = {
            "deepen": f"""請將以下指令改寫得更深入，增加複雜度和推理步驟：

原始指令：{instruction}

改寫後的指令：""",

            "breadth": f"""請將以下指令改寫，拓展到相關但不同的主題：

原始指令：{instruction}

改寫後的指令：""",

            "constraint": f"""請為以下指令添加更多約束條件或限制：

原始指令：{instruction}

改寫後的指令：""",

            "concretize": f"""請將以下指令具體化，添加更多細節和實際例子：

原始指令：{instruction}

改寫後的指令：""",

            "reasoning": f"""請改寫以下指令，使其需要更多推理和分析：

原始指令：{instruction}

改寫後的指令："""
        }

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompts[method]}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def generate_response(self, instruction: str, input_text: str = "") -> str:
        """為指令生成回應"""
        if input_text:
            prompt = f"指令：{instruction}\n\n輸入：{input_text}\n\n回應："
        else:
            prompt = f"指令：{instruction}\n\n回應："

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()

    def evolve_dataset(self, instructions: List[str], methods: List[str] = None):
        """批次演化資料集"""
        if methods is None:
            methods = ["deepen", "breadth", "constraint", "concretize", "reasoning"]

        evolved_dataset = []

        for instruction in instructions:
            # 原始指令
            evolved_dataset.append({
                "instruction": instruction,
                "method": "original"
            })

            # 演化版本
            for method in methods:
                try:
                    evolved = self.evolve_instruction(instruction, method)
                    evolved_dataset.append({
                        "instruction": evolved,
                        "method": method,
                        "original": instruction
                    })
                except Exception as e:
                    print(f"演化失敗 ({method}): {e}")

        return evolved_dataset

# 使用範例（需要 OpenAI API key）
# generator = EvolInstructGenerator()
#
# original_instruction = "解釋什麼是機器學習"
# evolved = generator.evolve_instruction(original_instruction, method="deepen")
# print(f"演化後: {evolved}")
```

### 3.3.2 合成資料生成

```python
class SyntheticDataGenerator:
    """合成資料生成器"""

    def __init__(self, llm_client=None):
        self.client = llm_client

    def generate_qa_pairs(self, topic: str, num_pairs: int = 10):
        """生成問答對"""
        prompt = f"""請生成 {num_pairs} 個關於「{topic}」的高質量問答對。

格式：
Q: 問題
A: 答案

要求：
1. 問題應該涵蓋不同難度
2. 答案應該準確且詳細
3. 包含不同類型的問題（概念、應用、比較等）

生成："""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000
        )

        return self.parse_qa_pairs(response.choices[0].message.content)

    def parse_qa_pairs(self, text: str) -> List[Dict]:
        """解析問答對"""
        pairs = []
        lines = text.split('\n')

        current_q = None
        current_a = None

        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('問:'):
                if current_q and current_a:
                    pairs.append({"question": current_q, "answer": current_a})
                current_q = line[2:].strip()
                current_a = None
            elif line.startswith('A:') or line.startswith('答:'):
                current_a = line[2:].strip()

        if current_q and current_a:
            pairs.append({"question": current_q, "answer": current_a})

        return pairs

    def generate_code_examples(self, task: str, language: str = "Python"):
        """生成程式碼範例"""
        prompt = f"""請生成 5 個關於「{task}」的 {language} 程式碼範例。

每個範例應包含：
1. 任務描述
2. 程式碼實作
3. 使用範例

格式：
### 範例 N
**任務**: ...
**程式碼**:
```python
...
```
**使用**:
```python
...
```
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content

    def back_translate(self, text: str, intermediate_lang: str = "en"):
        """回譯增強（中文->英文->中文）"""
        # 第一次翻譯
        to_en_prompt = f"Translate the following Chinese text to English:\n\n{text}"
        response1 = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": to_en_prompt}],
            temperature=0.3
        )
        english_text = response1.choices[0].message.content

        # 回譯
        to_zh_prompt = f"Translate the following English text to Chinese:\n\n{english_text}"
        response2 = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": to_zh_prompt}],
            temperature=0.3
        )
        back_translated = response2.choices[0].message.content

        return {
            "original": text,
            "english": english_text,
            "back_translated": back_translated
        }

# 使用範例
# generator = SyntheticDataGenerator(llm_client=OpenAI())
# qa_pairs = generator.generate_qa_pairs("機器學習", num_pairs=5)
```

### 3.3.3 簡單的資料增強

```python
import random
import jieba

class SimpleDataAugmentation:
    """簡單的資料增強技術（不需要 LLM）"""

    @staticmethod
    def synonym_replacement(text: str, n=1):
        """同義詞替換"""
        # 簡化版本，實際應用應使用同義詞詞典
        synonyms = {
            "機器學習": ["ML", "機器學習技術", "機器學習方法"],
            "深度學習": ["DL", "深度神經網路", "深度學習技術"],
            "模型": ["模型", "模型架構", "演算法模型"],
        }

        for word, syns in synonyms.items():
            if word in text and random.random() < 0.3:
                text = text.replace(word, random.choice(syns), 1)

        return text

    @staticmethod
    def random_insertion(text: str, n=1):
        """隨機插入"""
        words = list(jieba.cut(text))

        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)

        return ''.join(words)

    @staticmethod
    def random_swap(text: str, n=1):
        """隨機交換"""
        words = list(jieba.cut(text))

        for _ in range(n):
            if len(words) < 2:
                return text

            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ''.join(words)

    @staticmethod
    def random_deletion(text: str, p=0.1):
        """隨機刪除"""
        words = list(jieba.cut(text))

        if len(words) == 1:
            return text

        new_words = [word for word in words if random.random() > p]

        if len(new_words) == 0:
            return random.choice(words)

        return ''.join(new_words)

    @classmethod
    def augment(cls, text: str, num_aug=4):
        """綜合增強"""
        augmented_texts = [text]  # 包含原始文字

        methods = [
            cls.synonym_replacement,
            cls.random_insertion,
            cls.random_swap,
            cls.random_deletion
        ]

        for _ in range(num_aug):
            method = random.choice(methods)
            augmented = method(text)
            if augmented != text:
                augmented_texts.append(augmented)

        return list(set(augmented_texts))  # 去重

# 使用範例
aug = SimpleDataAugmentation()
original = "機器學習是人工智慧的一個重要分支"
augmented = aug.augment(original, num_aug=3)

print("原始:", original)
print("增強版本:")
for i, text in enumerate(augmented[1:], 1):
    print(f"{i}. {text}")
```

---

## 3.4 使用現有資料集

### 3.4.1 Hugging Face Datasets

```python
from datasets import load_dataset, concatenate_datasets
import pandas as pd

class DatasetLoader:
    """資料集載入工具"""

    @staticmethod
    def load_alpaca():
        """載入 Alpaca 資料集"""
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        return dataset

    @staticmethod
    def load_dolly():
        """載入 Dolly 15k 資料集"""
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        return dataset

    @staticmethod
    def load_chinese_datasets():
        """載入中文資料集"""
        datasets_info = {
            "belle": "BelleGroup/train_1M_CN",
            "chinese_alpaca": "shibing624/alpaca-zh",
            "cvalues": "CValues/cvalues-llama",
        }

        loaded_datasets = {}
        for name, path in datasets_info.items():
            try:
                dataset = load_dataset(path, split="train")
                loaded_datasets[name] = dataset
                print(f"已載入 {name}: {len(dataset)} 條")
            except Exception as e:
                print(f"載入 {name} 失敗: {e}")

        return loaded_datasets

    @staticmethod
    def load_from_json(json_path: str):
        """從 JSON 載入"""
        dataset = load_dataset('json', data_files=json_path, split='train')
        return dataset

    @staticmethod
    def merge_datasets(datasets: List, sampling_ratios: List[float] = None):
        """合併多個資料集"""
        if sampling_ratios:
            assert len(datasets) == len(sampling_ratios)
            sampled_datasets = []
            for dataset, ratio in zip(datasets, sampling_ratios):
                sample_size = int(len(dataset) * ratio)
                sampled = dataset.shuffle(seed=42).select(range(sample_size))
                sampled_datasets.append(sampled)
            datasets = sampled_datasets

        merged = concatenate_datasets(datasets)
        return merged

# 使用範例
loader = DatasetLoader()

# 載入英文資料集
# alpaca = loader.load_alpaca()
# print(f"Alpaca: {len(alpaca)} 條")

# 載入中文資料集
# chinese_ds = loader.load_chinese_datasets()

# 合併資料集（不同比例採樣）
# merged = loader.merge_datasets(
#     [chinese_ds['belle'], chinese_ds['chinese_alpaca']],
#     sampling_ratios=[0.5, 1.0]  # belle 取 50%, chinese_alpaca 全取
# )
```

### 3.4.2 繁體中文化

```python
from opencc import OpenCC
from typing import List, Dict

class TraditionalChineseConverter:
    """繁體中文轉換工具"""

    def __init__(self):
        # 簡體轉繁體
        self.s2t = OpenCC('s2t')  # Simplified to Traditional
        # 繁體轉簡體
        self.t2s = OpenCC('t2s')  # Traditional to Simplified
        # 簡體轉台灣正體
        self.s2tw = OpenCC('s2tw')
        # 簡體轉香港繁體
        self.s2hk = OpenCC('s2hk')

    def convert_text(self, text: str, mode='s2tw') -> str:
        """
        轉換文字

        modes:
        - s2t: 簡體到繁體
        - s2tw: 簡體到台灣正體
        - s2hk: 簡體到香港繁體
        - t2s: 繁體到簡體
        """
        converter = getattr(self, mode)
        return converter.convert(text)

    def convert_dataset(self, dataset, fields: List[str], mode='s2tw'):
        """轉換資料集"""
        def convert_example(example):
            for field in fields:
                if field in example and example[field]:
                    example[field] = self.convert_text(example[field], mode)
            return example

        return dataset.map(convert_example)

    def convert_instruction_dataset(self, examples: List[Dict], mode='s2tw') -> List[Dict]:
        """轉換 instruction 資料集"""
        converted = []
        for ex in examples:
            converted_ex = {
                'instruction': self.convert_text(ex['instruction'], mode),
                'input': self.convert_text(ex.get('input', ''), mode),
                'output': self.convert_text(ex['output'], mode)
            }
            converted.append(converted_ex)
        return converted

# 使用範例
converter = TraditionalChineseConverter()

# 轉換單個文字
simplified = "机器学习是人工智能的一个分支"
traditional = converter.convert_text(simplified, mode='s2tw')
print(f"簡體: {simplified}")
print(f"繁體: {traditional}")

# 轉換資料集
examples = [
    {
        "instruction": "解释什么是机器学习",
        "input": "",
        "output": "机器学习是一种人工智能技术..."
    }
]

converted_examples = converter.convert_instruction_dataset(examples)
print("\n轉換後的資料集:")
for ex in converted_examples:
    print(ex)
```

---

## 3.5 資料品質評估與過濾

### 3.5.1 品質評估指標

```python
import re
from typing import Dict, List
import numpy as np

class DataQualityEvaluator:
    """資料品質評估工具"""

    @staticmethod
    def length_stats(text: str) -> Dict:
        """長度統計"""
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }

    @staticmethod
    def calculate_diversity(texts: List[str]) -> Dict:
        """計算資料集多樣性"""
        all_words = []
        for text in texts:
            all_words.extend(text.split())

        unique_words = set(all_words)

        return {
            'total_words': len(all_words),
            'unique_words': len(unique_words),
            'diversity_ratio': len(unique_words) / len(all_words) if all_words else 0
        }

    @staticmethod
    def check_quality(text: str) -> Dict[str, bool]:
        """檢查文字品質"""
        checks = {
            'min_length': len(text) >= 10,
            'max_length': len(text) <= 10000,
            'not_empty': len(text.strip()) > 0,
            'has_letters': bool(re.search(r'[a-zA-Z\u4e00-\u9fff]', text)),
            'reasonable_punct': text.count('!') + text.count('?') < len(text) * 0.1,
            'no_excessive_caps': sum(1 for c in text if c.isupper()) < len(text) * 0.5,
            'no_excessive_repeats': not bool(re.search(r'(.)\1{10,}', text)),
        }

        checks['passed'] = all(checks.values())
        return checks

    @staticmethod
    def detect_language_quality(text: str) -> Dict:
        """語言品質檢測"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'\S', text))

        return {
            'chinese_ratio': chinese_chars / total_chars if total_chars > 0 else 0,
            'english_ratio': english_chars / total_chars if total_chars > 0 else 0,
            'has_mixed_lang': chinese_chars > 0 and english_chars > 0,
            'primary_language': 'zh' if chinese_chars > english_chars else 'en'
        }

    @classmethod
    def filter_dataset(cls, examples: List[Dict],
                      min_length=10,
                      max_length=10000,
                      quality_checks=True) -> List[Dict]:
        """過濾資料集"""
        filtered = []

        for ex in examples:
            # 檢查所有文字欄位
            all_text = ' '.join([
                ex.get('instruction', ''),
                ex.get('input', ''),
                ex.get('output', '')
            ])

            # 長度檢查
            if len(all_text) < min_length or len(all_text) > max_length:
                continue

            # 品質檢查
            if quality_checks:
                quality = cls.check_quality(all_text)
                if not quality['passed']:
                    continue

            filtered.append(ex)

        return filtered

# 使用範例
evaluator = DataQualityEvaluator()

# 評估單個文字
text = "這是一個測試文字，用來評估資料品質。"
stats = evaluator.length_stats(text)
quality = evaluator.check_quality(text)
lang = evaluator.detect_language_quality(text)

print("長度統計:", stats)
print("品質檢查:", quality)
print("語言檢測:", lang)

# 過濾資料集
examples = [
    {"instruction": "測試", "input": "", "output": "太短"},  # 會被過濾
    {"instruction": "這是一個有效的指令", "input": "輸入內容", "output": "這是一個有效的輸出內容"},
]

filtered = evaluator.filter_dataset(examples, min_length=20)
print(f"\n過濾前: {len(examples)} 條")
print(f"過濾後: {len(filtered)} 條")
```

### 3.5.2 毒性和偏見檢測

```python
class ToxicityDetector:
    """毒性內容檢測"""

    def __init__(self):
        # 敏感詞列表（實際應用應使用更完整的詞庫）
        self.toxic_words = set([
            # 添加敏感詞
        ])

        self.bias_patterns = [
            # 性別偏見
            r'(男|女)(生|性)+(比較|更加|總是)',
            # 種族偏見
            # ... 添加更多模式
        ]

    def check_toxicity(self, text: str) -> Dict:
        """檢查毒性內容"""
        text_lower = text.lower()

        found_toxic = [word for word in self.toxic_words if word in text_lower]

        return {
            'is_toxic': len(found_toxic) > 0,
            'toxic_words': found_toxic,
            'toxicity_score': len(found_toxic) / len(text.split()) if text.split() else 0
        }

    def check_bias(self, text: str) -> Dict:
        """檢查偏見內容"""
        found_patterns = []

        for pattern in self.bias_patterns:
            if re.search(pattern, text):
                found_patterns.append(pattern)

        return {
            'has_bias': len(found_patterns) > 0,
            'bias_patterns': found_patterns
        }

    def is_safe(self, text: str) -> bool:
        """綜合安全檢查"""
        toxicity = self.check_toxicity(text)
        bias = self.check_bias(text)

        return not (toxicity['is_toxic'] or bias['has_bias'])

# 使用範例
detector = ToxicityDetector()

test_text = "這是一個測試文字"
is_safe = detector.is_safe(test_text)
print(f"文字安全性: {is_safe}")
```

---

## 3.6 RLHF/RLAIF 資料集準備

### 3.6.1 RLHF 資料格式

**Reinforcement Learning from Human Feedback (RLHF)** 需要 **偏好資料**（Preference Data），用於訓練獎勵模型。

**標準格式**：

```json
{
  "prompt": "使用者的提示或問題",
  "chosen": "較好的回應",
  "rejected": "較差的回應"
}
```

**範例**：

```json
[
  {
    "prompt": "請解釋量子計算的基本原理",
    "chosen": "量子計算利用量子力學的疊加態和糾纏等現象進行計算。與傳統位元只能是 0 或 1 不同，量子位元（qubit）可以同時處於 0 和 1 的疊加態，使得量子電腦能夠並行處理大量計算。這種特性使量子計算在某些特定問題上，如質因數分解、資料庫搜尋等，能展現出指數級的加速效果。",
    "rejected": "量子計算就是很快的計算機。"
  }
]
```

### 3.6.2 收集偏好資料

```python
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PreferenceExample:
    """偏好範例資料結構"""
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict = None

    def to_dict(self) -> Dict:
        data = {
            'prompt': self.prompt,
            'chosen': self.chosen,
            'rejected': self.rejected
        }
        if self.metadata:
            data['metadata'] = self.metadata
        return data

class PreferenceDatasetBuilder:
    """RLHF 偏好資料集建構工具"""

    def __init__(self):
        self.examples: List[PreferenceExample] = []

    def add_preference(self, prompt: str, chosen: str, rejected: str,
                      metadata: Dict = None):
        """添加一個偏好範例"""
        example = PreferenceExample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata or {}
        )
        self.examples.append(example)

    def from_comparison_data(self, comparisons: List[Dict]):
        """
        從比較資料轉換
        comparisons: [{
            "prompt": "...",
            "responses": ["response_1", "response_2", ...],
            "rankings": [1, 2, ...]  # 1 是最好的
        }]
        """
        for comp in comparisons:
            prompt = comp['prompt']
            responses = comp['responses']
            rankings = comp['rankings']

            # 找出最好和最差的回應
            best_idx = rankings.index(min(rankings))
            worst_idx = rankings.index(max(rankings))

            self.add_preference(
                prompt=prompt,
                chosen=responses[best_idx],
                rejected=responses[worst_idx],
                metadata={
                    'rankings': rankings,
                    'num_responses': len(responses)
                }
            )

    def from_rating_data(self, ratings: List[Dict], threshold: float = 3.5):
        """
        從評分資料轉換
        ratings: [{
            "prompt": "...",
            "response": "...",
            "rating": 4.5  # 1-5 分
        }]
        """
        # 按 prompt 分組
        grouped = {}
        for item in ratings:
            prompt = item['prompt']
            if prompt not in grouped:
                grouped[prompt] = []
            grouped[prompt].append((item['response'], item['rating']))

        # 為每個 prompt 建立偏好對
        for prompt, responses_ratings in grouped.items():
            # 至少需要 2 個回應
            if len(responses_ratings) < 2:
                continue

            # 找出高分和低分的回應
            high_rated = [r for r, rating in responses_ratings if rating >= threshold]
            low_rated = [r for r, rating in responses_ratings if rating < threshold]

            if high_rated and low_rated:
                # 隨機配對
                import random
                self.add_preference(
                    prompt=prompt,
                    chosen=random.choice(high_rated),
                    rejected=random.choice(low_rated)
                )

    def create_synthetic_negatives(self, instructions: List[Dict],
                                   method: str = "truncate"):
        """
        為現有資料集建立合成負樣本

        methods:
        - truncate: 截斷回應
        - corrupt: 破壞回應品質
        - irrelevant: 生成不相關回應
        """
        for inst in instructions:
            prompt = inst['instruction']
            if inst.get('input'):
                prompt += f"\n\n{inst['input']}"

            chosen = inst['output']

            # 生成負樣本
            if method == "truncate":
                # 截斷到一半
                rejected = chosen[:len(chosen)//2] + "..."
            elif method == "corrupt":
                # 添加語法錯誤或不完整
                rejected = chosen[:len(chosen)//3]
            elif method == "irrelevant":
                # 使用不相關的回應（需要從其他範例中取）
                rejected = "抱歉，我無法回答這個問題。"

            self.add_preference(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata={'method': 'synthetic', 'type': method}
            )

    def to_json(self, output_path: str):
        """儲存為 JSON"""
        data = [ex.to_dict() for ex in self.examples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def to_dpo_format(self):
        """
        轉換為 DPO (Direct Preference Optimization) 格式
        用於 trl 庫的 DPOTrainer
        """
        formatted = []
        for ex in self.examples:
            formatted.append({
                "prompt": ex.prompt,
                "chosen": ex.chosen,
                "rejected": ex.rejected
            })
        return formatted

    def __len__(self):
        return len(self.examples)

# 使用範例
builder = PreferenceDatasetBuilder()

# 方法1: 直接添加偏好對
builder.add_preference(
    prompt="解釋機器學習",
    chosen="機器學習是一種人工智慧技術，透過資料學習模式...",
    rejected="機器學習就是電腦學習。"
)

# 方法2: 從評分資料轉換
rating_data = [
    {"prompt": "什麼是 Python?", "response": "Python 是一種高階程式語言...", "rating": 4.5},
    {"prompt": "什麼是 Python?", "response": "一種蛇。", "rating": 1.0},
]
builder.from_rating_data(rating_data)

# 儲存
builder.to_json("preference_data.json")
print(f"建立了 {len(builder)} 個偏好範例")
```

### 3.6.3 RLAIF - 使用 AI 回饋

**RLAIF (Reinforcement Learning from AI Feedback)** 使用 LLM 來評估和排序回應，減少人工標註成本。

```python
from openai import OpenAI
import os
from typing import List, Tuple

class RLAIFDataGenerator:
    """使用 AI 生成偏好資料"""

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_multiple_responses(self, prompt: str, n: int = 4) -> List[str]:
        """為同一個 prompt 生成多個回應"""
        responses = []

        for i in range(n):
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 + (i * 0.1),  # 調整溫度以增加多樣性
                max_tokens=500
            )
            responses.append(response.choices[0].message.content)

        return responses

    def judge_responses(self, prompt: str, response_a: str,
                       response_b: str) -> Tuple[str, str, str]:
        """
        使用 LLM 判斷哪個回應更好
        返回: (chosen, rejected, reasoning)
        """
        judge_prompt = f"""請評估以下兩個回應哪個更好。

問題: {prompt}

回應 A:
{response_a}

回應 B:
{response_b}

請從以下方面評估:
1. 準確性
2. 完整性
3. 清晰度
4. 實用性

請選擇 A 或 B，並說明理由。
格式: 選擇: [A/B]
理由: ...
"""

        response = self.client.chat.completions.create(
            model="gpt-4",  # 使用更強的模型作為評審
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3
        )

        judgment = response.choices[0].message.content

        # 解析結果
        if "選擇: A" in judgment or "選擇：A" in judgment:
            return response_a, response_b, judgment
        else:
            return response_b, response_a, judgment

    def create_preference_dataset(self, prompts: List[str]) -> List[Dict]:
        """為一組 prompts 建立偏好資料集"""
        preference_data = []

        for i, prompt in enumerate(prompts):
            print(f"處理 prompt {i+1}/{len(prompts)}")

            # 生成多個回應
            responses = self.generate_multiple_responses(prompt, n=2)

            # 評判
            chosen, rejected, reasoning = self.judge_responses(
                prompt, responses[0], responses[1]
            )

            preference_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "method": "rlaif",
                    "reasoning": reasoning
                }
            })

        return preference_data

    def rank_responses(self, prompt: str, responses: List[str]) -> List[int]:
        """對多個回應進行排序"""
        rank_prompt = f"""請對以下回應進行排序（從最好到最差）。

問題: {prompt}

回應:
"""
        for i, resp in enumerate(responses):
            rank_prompt += f"\n{i+1}. {resp}\n"

        rank_prompt += "\n請返回排序結果，格式: [1, 3, 2, 4] (數字代表回應的編號)"

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": rank_prompt}],
            temperature=0.2
        )

        # 解析排序結果
        import re
        result = response.choices[0].message.content
        matches = re.findall(r'\[([\d,\s]+)\]', result)
        if matches:
            ranking = [int(x.strip()) for x in matches[0].split(',')]
            return ranking

        return list(range(1, len(responses) + 1))

# 使用範例（需要 OpenAI API key）
# generator = RLAIFDataGenerator()
# prompts = ["解釋什麼是深度學習", "Python 如何讀取 CSV 檔案?"]
# preference_data = generator.create_preference_dataset(prompts)
```

### 3.6.4 Constitutional AI 資料

**Constitutional AI** 透過定義「憲法」（規則）來指導 AI 的行為。

```python
class ConstitutionalAIDataBuilder:
    """Constitutional AI 資料集建構"""

    def __init__(self, client=None):
        self.client = client
        self.constitution = []

    def add_principle(self, principle: str):
        """添加一個憲法原則"""
        self.constitution.append(principle)

    def critique_and_revise(self, prompt: str, response: str,
                           principle: str) -> Tuple[str, str]:
        """
        根據憲法原則批評並修訂回應
        返回: (critique, revised_response)
        """
        critique_prompt = f"""根據以下原則評估回應:

原則: {principle}

問題: {prompt}
回應: {response}

請指出回應是否違反了這個原則，並給出批評。
"""

        critique_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0.3
        )
        critique = critique_response.choices[0].message.content

        # 修訂回應
        revise_prompt = f"""根據以下批評修訂回應:

原則: {principle}
問題: {prompt}
原始回應: {response}
批評: {critique}

請提供一個改進的回應。
"""

        revise_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": revise_prompt}],
            temperature=0.3
        )
        revised = revise_response.choices[0].message.content

        return critique, revised

    def create_constitutional_dataset(self, initial_data: List[Dict]) -> List[Dict]:
        """建立 Constitutional AI 資料集"""
        constitutional_data = []

        for item in initial_data:
            prompt = item['prompt']
            response = item['response']

            # 對每個原則進行批評和修訂
            for principle in self.constitution:
                critique, revised = self.critique_and_revise(
                    prompt, response, principle
                )

                constitutional_data.append({
                    "prompt": prompt,
                    "original_response": response,
                    "principle": principle,
                    "critique": critique,
                    "revised_response": revised
                })

        return constitutional_data

# 使用範例
# builder = ConstitutionalAIDataBuilder(client=OpenAI())
#
# # 添加憲法原則
# builder.add_principle("回應應該準確且有事實依據")
# builder.add_principle("回應應該有禮貌且尊重")
# builder.add_principle("回應應該避免有害或危險的建議")
```

---

## 3.7 資料標註流程與工具

### 3.7.1 標註工作流程

**典型的資料標註流程**：

1. **準備階段**
   - 定義標註任務
   - 建立標註指南
   - 選擇標註工具
   - 招募並培訓標註者

2. **標註階段**
   - 分配任務
   - 多人標註（提高可靠性）
   - 品質控制檢查
   - 解決衝突

3. **驗證階段**
   - 計算標註者間一致性
   - 專家審核
   - 修正錯誤標註

4. **交付階段**
   - 匯出標註資料
   - 格式轉換
   - 資料驗證

### 3.7.2 標註品質控制

```python
from collections import defaultdict
from typing import List, Dict, Set
import numpy as np
from sklearn.metrics import cohen_kappa_score

class AnnotationQualityControl:
    """標註品質控制工具"""

    @staticmethod
    def calculate_agreement(annotations: List[List[int]]) -> float:
        """
        計算標註者間一致性（Fleiss' Kappa）
        annotations: [[1, 0, 1], [1, 1, 1], ...]  # 每行是不同標註者對同一項目的標註
        """
        n_items = len(annotations)
        n_raters = len(annotations[0])

        # 計算每個類別的比例
        p_j = defaultdict(int)
        for item_annotations in annotations:
            for label in item_annotations:
                p_j[label] += 1

        for label in p_j:
            p_j[label] /= (n_items * n_raters)

        # 計算 P_bar (observed agreement)
        P_bar = 0
        for item_annotations in annotations:
            n_j = defaultdict(int)
            for label in item_annotations:
                n_j[label] += 1

            sum_nj_squared = sum(count**2 for count in n_j.values())
            P_i = (sum_nj_squared - n_raters) / (n_raters * (n_raters - 1))
            P_bar += P_i

        P_bar /= n_items

        # 計算 P_e_bar (expected agreement)
        P_e_bar = sum(p**2 for p in p_j.values())

        # Fleiss' Kappa
        if P_e_bar == 1:
            return 1.0
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

        return kappa

    @staticmethod
    def identify_difficult_examples(annotations: Dict[str, List],
                                    threshold: float = 0.5) -> List[str]:
        """
        識別困難的標註範例（標註者意見分歧的）
        annotations: {example_id: [label1, label2, label3]}
        """
        difficult = []

        for example_id, labels in annotations.items():
            # 計算一致性
            most_common = max(set(labels), key=labels.count)
            agreement_rate = labels.count(most_common) / len(labels)

            if agreement_rate < threshold:
                difficult.append(example_id)

        return difficult

    @staticmethod
    def calculate_annotator_performance(annotations: Dict[str, Dict[str, int]],
                                       gold_labels: Dict[str, int]) -> Dict[str, float]:
        """
        計算每個標註者的表現
        annotations: {example_id: {annotator_id: label}}
        gold_labels: {example_id: correct_label}
        """
        performance = defaultdict(lambda: {"correct": 0, "total": 0})

        for example_id, annotator_labels in annotations.items():
            if example_id not in gold_labels:
                continue

            gold_label = gold_labels[example_id]

            for annotator_id, label in annotator_labels.items():
                performance[annotator_id]["total"] += 1
                if label == gold_label:
                    performance[annotator_id]["correct"] += 1

        # 計算準確率
        scores = {}
        for annotator_id, stats in performance.items():
            scores[annotator_id] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        return scores

# 使用範例
qc = AnnotationQualityControl()

# 範例標註資料（3 個標註者對 5 個項目的標註）
annotations = [
    [1, 1, 1],  # 項目 1: 完全一致
    [1, 1, 0],  # 項目 2: 部分一致
    [0, 0, 0],  # 項目 3: 完全一致
    [1, 0, 1],  # 項目 4: 部分一致
    [1, 1, 1],  # 項目 5: 完全一致
]

kappa = qc.calculate_agreement(annotations)
print(f"Fleiss' Kappa: {kappa:.3f}")

# 解釋
if kappa > 0.8:
    print("一致性: 幾乎完美")
elif kappa > 0.6:
    print("一致性: 實質性")
elif kappa > 0.4:
    print("一致性: 中等")
else:
    print("一致性: 較差")
```

### 3.7.3 開源標註工具

```python
"""
推薦的開源標註工具:

1. **Label Studio**
   - 網址: https://labelstud.io/
   - 支援: 文字分類、NER、問答、圖片標註等
   - 特色: 介面友善、可自訂、支援 ML 輔助標註

   安裝:
   pip install label-studio
   label-studio start

2. **Doccano**
   - 網址: https://github.com/doccano/doccano
   - 支援: 文字分類、序列標註、seq2seq
   - 特色: 輕量、開源、易部署

   安裝:
   pip install doccano
   doccano init
   doccano createuser --username admin --password pass
   doccano webserver --port 8000

3. **Argilla** (前身為 rubrix)
   - 網址: https://github.com/argilla-io/argilla
   - 支援: 文字分類、NER、問答、生成任務
   - 特色: ML 友善、支援主動學習、整合 HuggingFace

   安裝:
   pip install argilla

4. **Prodigy** (商業)
   - 網址: https://prodi.gy/
   - 特色: 主動學習、高效率、scriptable

5. **CVAT** (Computer Vision Annotation Tool)
   - 網址: https://github.com/opencv/cvat
   - 支援: 圖片、影片標註
   - 特色: 功能強大、適合電腦視覺任務
"""

# Argilla 使用範例
"""
import argilla as rg

# 連接到 Argilla 伺服器
rg.init(api_url="http://localhost:6900", api_key="your_api_key")

# 建立文字分類資料集
dataset = rg.DatasetForTextClassification([
    rg.TextClassificationRecord(
        text="這是一個很棒的產品！",
        prediction=[("positive", 0.9), ("negative", 0.1)],
        annotation="positive",
        metadata={"source": "review_site"}
    )
])

# 記錄到 Argilla
rg.log(dataset, name="product_reviews")
"""
```

### 3.7.4 標註指南範例

**標註指南模板**：

```markdown
# 資料標註指南

## 任務概述
[簡述標註任務的目的和重要性]

## 標註類別

### 類別 1: 正面情緒
**定義**: 表達積極、正面、滿意的情緒
**範例**:
- "這個產品太棒了！"
- "服務很好，非常滿意。"

### 類別 2: 負面情緒
**定義**: 表達消極、負面、不滿的情緒
**範例**:
- "產品品質很差。"
- "客服態度不好。"

### 類別 3: 中性
**定義**: 沒有明顯情緒傾向，陳述事實
**範例**:
- "產品已收到。"
- "這是一支藍色的筆。"

## 特殊情況

### 混合情緒
當文字包含多種情緒時，選擇主導情緒。

範例: "產品不錯，但價格太貴。" → 根據重點選擇

### 諷刺
注意識別諷刺語氣，應根據實際意圖標註。

範例: "真是太'棒'了，又壞了。" → 負面

### 不確定
如果無法判斷，選擇「不確定」並留言說明。

## 標註流程

1. 仔細閱讀整個文字
2. 識別關鍵情緒詞彙
3. 考慮整體語境
4. 選擇最合適的類別
5. 如有疑問，諮詢管理員

## 品質要求

- 準確性: 嚴格按照定義標註
- 一致性: 相似案例應標註相同
- 完整性: 不要跳過任何項目

## 常見錯誤

1. ❌ 只看關鍵詞，忽略語境
2. ❌ 受個人偏好影響
3. ❌ 標註過快，不仔細閱讀

## 聯絡方式

有問題請聯絡: [聯絡資訊]
```

---

## 3.8 資料集版本控制與管理

### 3.8.1 為什麼需要版本控制

**資料集版本控制的重要性**：

1. **可重現性**: 確保訓練結果可以重現
2. **追溯性**: 了解資料集的變更歷史
3. **協作**: 多人協作時避免衝突
4. **實驗管理**: 追蹤不同資料集版本的模型表現

### 3.8.2 資料集版本控制工具

```python
"""
推薦的資料集版本控制工具:

1. **DVC (Data Version Control)**
   - 網址: https://dvc.org/
   - 類似 Git，專為資料和模型設計
   - 支援雲端儲存 (S3, GCS, Azure, etc.)

   安裝:
   pip install dvc

   基本使用:
   dvc init
   dvc add data/training_data.json
   git add data/training_data.json.dvc .gitignore
   git commit -m "Add training data"
   dvc push

2. **LakeFS**
   - 網址: https://lakefs.io/
   - 像 Git 一樣的資料湖版本控制
   - 支援大規模資料

3. **MLflow**
   - 網址: https://mlflow.org/
   - 除了版本控制，還包含實驗追蹤

4. **Pachyderm**
   - 網址: https://www.pachyderm.com/
   - 資料版本控制 + 管線自動化

5. **HuggingFace Datasets Hub**
   - 網址: https://huggingface.co/datasets
   - 公開分享資料集
   - 內建版本控制
"""
```

### 3.8.3 資料集元資料管理

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import json
import hashlib

@dataclass
class DatasetMetadata:
    """資料集元資料"""
    name: str
    version: str
    created_at: str
    description: str
    size: int  # 資料條數
    format: str
    license: str
    authors: List[str]
    source: List[str]
    processing_steps: List[Dict]
    statistics: Dict
    checksum: str  # 資料完整性校驗
    tags: List[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

class DatasetVersionManager:
    """資料集版本管理器"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.versions = []

    def calculate_checksum(self, data: List[Dict]) -> str:
        """計算資料集的校驗和"""
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def calculate_statistics(self, data: List[Dict]) -> Dict:
        """計算資料集統計資訊"""
        stats = {
            "total_examples": len(data),
            "avg_input_length": 0,
            "avg_output_length": 0,
            "max_input_length": 0,
            "max_output_length": 0,
        }

        if not data:
            return stats

        input_lengths = []
        output_lengths = []

        for item in data:
            input_text = item.get('input', '') or item.get('instruction', '')
            output_text = item.get('output', '')

            input_len = len(input_text)
            output_len = len(output_text)

            input_lengths.append(input_len)
            output_lengths.append(output_len)

        stats["avg_input_length"] = sum(input_lengths) / len(input_lengths)
        stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        stats["max_input_length"] = max(input_lengths)
        stats["max_output_length"] = max(output_lengths)

        return stats

    def create_version(self, data: List[Dict],
                      version: str,
                      description: str,
                      authors: List[str],
                      source: List[str],
                      processing_steps: List[Dict],
                      license: str = "Unknown") -> DatasetMetadata:
        """建立新版本"""
        metadata = DatasetMetadata(
            name=self.dataset_name,
            version=version,
            created_at=datetime.now().isoformat(),
            description=description,
            size=len(data),
            format="json",
            license=license,
            authors=authors,
            source=source,
            processing_steps=processing_steps,
            statistics=self.calculate_statistics(data),
            checksum=self.calculate_checksum(data),
            tags=[]
        )

        self.versions.append(metadata)
        return metadata

    def compare_versions(self, version1: DatasetMetadata,
                        version2: DatasetMetadata) -> Dict:
        """比較兩個版本的差異"""
        diff = {
            "size_change": version2.size - version1.size,
            "size_change_pct": ((version2.size - version1.size) / version1.size * 100) if version1.size > 0 else 0,
            "checksum_changed": version1.checksum != version2.checksum,
            "processing_steps_added": len(version2.processing_steps) - len(version1.processing_steps),
        }

        return diff

# 使用範例
manager = DatasetVersionManager("my_instruction_dataset")

# 建立初始版本
data_v1 = [
    {"instruction": "...", "output": "..."},
    # ... 更多資料
]

metadata_v1 = manager.create_version(
    data=data_v1,
    version="v1.0.0",
    description="初始版本，包含基礎指令資料",
    authors=["Team A"],
    source=["Wikipedia", "Custom QA"],
    processing_steps=[
        {"step": "data_collection", "date": "2024-01-01"},
        {"step": "cleaning", "method": "remove_duplicates"}
    ],
    license="CC-BY-4.0"
)

# 儲存元資料
metadata_v1.to_json("metadata_v1.0.0.json")

print(f"版本: {metadata_v1.version}")
print(f"大小: {metadata_v1.size} 條")
print(f"校驗和: {metadata_v1.checksum[:16]}...")
print(f"統計: {metadata_v1.statistics}")
```

### 3.8.4 資料集變更日誌 (CHANGELOG)

**CHANGELOG.md 範例**：

```markdown
# 資料集變更日誌

## [v2.0.0] - 2024-03-15

### 新增
- 添加 5,000 條程式碼相關指令
- 添加多輪對話資料

### 變更
- 更新所有資料為繁體中文
- 改進輸出格式的一致性

### 移除
- 移除 500 條低品質資料

### 修正
- 修正標點符號錯誤
- 修正 50 條事實性錯誤

## [v1.5.0] - 2024-02-01

### 新增
- 添加 3,000 條摘要任務資料

### 變更
- 改進去重複演算法

## [v1.0.0] - 2024-01-01

### 新增
- 初始版本
- 包含 10,000 條基礎指令資料
```

---

## 3.9 資料隱私與合規性

### 3.9.1 隱私考量

**資料收集時的隱私問題**：

1. **個人識別資訊 (PII)**
   - 姓名、地址、電話號碼
   - 電子郵件、身分證號
   - 信用卡資訊

2. **敏感資訊**
   - 醫療記錄
   - 財務資訊
   - 私人通訊

3. **法規合規**
   - GDPR (歐盟)
   - CCPA (加州)
   - 個資法 (台灣)

### 3.9.2 PII 偵測與移除

```python
import re
from typing import List, Dict, Tuple

class PIIDetector:
    """個人識別資訊偵測與移除"""

    def __init__(self):
        # 編譯正則表達式模式
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_tw': re.compile(r'\b0\d{1,2}-?\d{3,4}-?\d{4}\b'),
            'phone_mobile_tw': re.compile(r'\b09\d{2}-?\d{3}-?\d{3}\b'),
            'credit_card': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
            'taiwan_id': re.compile(r'\b[A-Z][12]\d{8}\b'),  # 台灣身分證號
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """偵測文字中的 PII"""
        detected = {}

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches

        return detected

    def remove_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """移除 PII"""
        for pattern in self.patterns.values():
            text = pattern.sub(replacement, text)

        return text

    def anonymize_names(self, text: str, name_list: List[str]) -> str:
        """匿名化人名"""
        for name in name_list:
            text = text.replace(name, "[NAME]")

        return text

    def check_dataset(self, dataset: List[Dict]) -> Dict:
        """檢查整個資料集的 PII"""
        report = {
            "total_examples": len(dataset),
            "examples_with_pii": 0,
            "pii_counts": {},
        }

        for item in dataset:
            # 檢查所有文字欄位
            all_text = ' '.join([
                str(v) for v in item.values() if isinstance(v, str)
            ])

            detected = self.detect_pii(all_text)

            if detected:
                report["examples_with_pii"] += 1
                for pii_type, matches in detected.items():
                    if pii_type not in report["pii_counts"]:
                        report["pii_counts"][pii_type] = 0
                    report["pii_counts"][pii_type] += len(matches)

        return report

    def clean_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """清理資料集中的 PII"""
        cleaned = []

        for item in dataset:
            cleaned_item = {}
            for key, value in item.items():
                if isinstance(value, str):
                    cleaned_item[key] = self.remove_pii(value)
                else:
                    cleaned_item[key] = value
            cleaned.append(cleaned_item)

        return cleaned

# 使用範例
detector = PIIDetector()

# 偵測 PII
text = "請聯絡我，我的電話是 0912-345-678，email 是 john@example.com"
detected = detector.detect_pii(text)
print(f"偵測到的 PII: {detected}")

# 移除 PII
cleaned = detector.remove_pii(text)
print(f"清理後: {cleaned}")

# 檢查資料集
dataset = [
    {"instruction": "...", "output": "我的電話是 0912-345-678"},
    {"instruction": "...", "output": "正常的回應"},
]

report = detector.check_dataset(dataset)
print(f"\n資料集 PII 報告:")
print(f"  總數: {report['total_examples']}")
print(f"  包含 PII: {report['examples_with_pii']}")
print(f"  PII 類型: {report['pii_counts']}")

# 清理資料集
cleaned_dataset = detector.clean_dataset(dataset)
```

### 3.9.3 資料使用授權

**授權選擇指南**：

| 授權類型 | 商業使用 | 修改 | 重新分發 | 歸屬要求 | 適用場景 |
|---------|---------|------|---------|---------|---------|
| CC0 | ✅ | ✅ | ✅ | ❌ | 完全開放 |
| CC-BY | ✅ | ✅ | ✅ | ✅ | 需要署名 |
| CC-BY-SA | ✅ | ✅ | ✅ | ✅ | 相同授權 |
| CC-BY-NC | ❌ | ✅ | ✅ | ✅ | 非商業用 |
| MIT | ✅ | ✅ | ✅ | ✅ | 程式碼 |
| Apache 2.0 | ✅ | ✅ | ✅ | ✅ | 程式碼+專利 |

**建議**：
- 研究/教育用途: CC-BY 或 CC-BY-SA
- 商業應用: 確保所有來源資料都允許商業使用
- 混合資料: 使用最嚴格的授權

---

## 3.10 實作範例

### 3.10.1 完整的資料處理流程

```python
import json
from pathlib import Path
from typing import List, Dict
import random

class DatasetPipeline:
    """完整的資料集處理流程"""

    def __init__(self, output_dir="./processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.cleaner = TextCleaner()
        self.deduplicator = Deduplicator()
        self.evaluator = DataQualityEvaluator()
        self.converter = TraditionalChineseConverter()

    def process_raw_texts(self, raw_texts: List[str]) -> List[str]:
        """處理原始文字"""
        print("步驟 1: 清理文字")
        cleaned = [self.cleaner.clean_text(text) for text in raw_texts]
        print(f"  清理完成: {len(cleaned)} 條")

        print("步驟 2: 過濾短文字")
        filtered = [text for text in cleaned if self.cleaner.filter_by_length(text, min_length=50)]
        print(f"  過濾完成: {len(filtered)} 條")

        print("步驟 3: 去重複")
        unique = self.deduplicator.exact_dedup(filtered)
        print(f"  去重完成: {len(unique)} 條")

        return unique

    def create_instruction_dataset(self,
                                  qa_pairs: List[Dict],
                                  convert_to_traditional=True) -> List[Dict]:
        """建立 instruction 資料集"""
        print("建立 Instruction 資料集")

        dataset = []
        for qa in qa_pairs:
            example = {
                'instruction': "請回答以下問題",
                'input': qa['question'],
                'output': qa['answer']
            }
            dataset.append(example)

        print(f"  建立了 {len(dataset)} 條指令")

        if convert_to_traditional:
            print("  轉換為繁體中文")
            dataset = self.converter.convert_instruction_dataset(dataset)

        return dataset

    def quality_control(self, dataset: List[Dict]) -> List[Dict]:
        """品質控制"""
        print("執行品質控制")

        before_count = len(dataset)
        filtered = self.evaluator.filter_dataset(
            dataset,
            min_length=20,
            max_length=2000,
            quality_checks=True
        )
        after_count = len(filtered)

        print(f"  過濾了 {before_count - after_count} 條低品質資料")
        print(f"  保留了 {after_count} 條高品質資料")

        return filtered

    def split_and_save(self, dataset: List[Dict],
                      val_ratio=0.1,
                      test_ratio=0.1,
                      shuffle=True):
        """分割並儲存資料集"""
        print("分割並儲存資料集")

        if shuffle:
            random.shuffle(dataset)

        total = len(dataset)
        test_size = int(total * test_ratio)
        val_size = int(total * val_ratio)
        train_size = total - test_size - val_size

        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]

        # 儲存
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            print(f"  {split_name}: {len(split_data)} 條 -> {output_path}")

        return splits

    def generate_statistics(self, dataset: List[Dict]):
        """生成統計資訊"""
        print("\n資料集統計:")
        print(f"  總數: {len(dataset)}")

        # 長度統計
        lengths = [len(ex['output']) for ex in dataset]
        print(f"  輸出長度 - 平均: {np.mean(lengths):.0f}, 中位數: {np.median(lengths):.0f}")
        print(f"  輸出長度 - 最小: {min(lengths)}, 最大: {max(lengths)}")

        # 語言分佈
        languages = [self.evaluator.detect_language_quality(ex['output'])['primary_language']
                    for ex in dataset]
        lang_counts = {lang: languages.count(lang) for lang in set(languages)}
        print(f"  語言分佈: {lang_counts}")

# 完整範例
def main():
    """完整的資料處理流程範例"""

    # 初始化 pipeline
    pipeline = DatasetPipeline(output_dir="./my_dataset")

    # 模擬原始資料
    raw_qa_pairs = [
        {"question": "什么是机器学习？", "answer": "机器学习是人工智能的一个分支..."},
        {"question": "什么是深度学习？", "answer": "深度学习是机器学习的一个子领域..."},
        # ... 更多資料
    ]

    # 步驟 1: 建立 instruction 資料集
    dataset = pipeline.create_instruction_dataset(
        raw_qa_pairs,
        convert_to_traditional=True
    )

    # 步驟 2: 品質控制
    dataset = pipeline.quality_control(dataset)

    # 步驟 3: 分割並儲存
    splits = pipeline.split_and_save(
        dataset,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # 步驟 4: 生成統計
    pipeline.generate_statistics(dataset)

    print("\n資料處理完成！")

# 執行
# main()
```

### 3.10.2 從多個來源整合資料

```python
class MultiSourceDataIntegrator:
    """多來源資料整合"""

    def __init__(self):
        self.datasets = {}

    def add_source(self, name: str, data: List[Dict], weight: float = 1.0):
        """添加資料來源"""
        self.datasets[name] = {
            'data': data,
            'weight': weight
        }
        print(f"添加來源 '{name}': {len(data)} 條，權重 {weight}")

    def merge(self, max_samples_per_source: int = None) -> List[Dict]:
        """合併所有資料來源"""
        merged = []

        for name, info in self.datasets.items():
            data = info['data']
            weight = info['weight']

            # 根據權重採樣
            if max_samples_per_source:
                sample_size = min(int(max_samples_per_source * weight), len(data))
                data = random.sample(data, sample_size)

            # 添加來源標記
            for item in data:
                item['source'] = name
                merged.append(item)

        # 打亂
        random.shuffle(merged)

        print(f"\n合併完成: 總共 {len(merged)} 條")
        return merged

    def balance_sources(self) -> List[Dict]:
        """平衡各資料來源（每個來源採樣相同數量）"""
        min_size = min(len(info['data']) for info in self.datasets.values())

        balanced = []
        for name, info in self.datasets.items():
            sampled = random.sample(info['data'], min_size)
            for item in sampled:
                item['source'] = name
                balanced.append(item)

        random.shuffle(balanced)

        print(f"平衡完成: 每個來源 {min_size} 條，總共 {len(balanced)} 條")
        return balanced

# 使用範例
integrator = MultiSourceDataIntegrator()

# 添加不同來源
integrator.add_source("wikipedia", [{"instruction": "...", "output": "..."}] * 1000, weight=1.0)
integrator.add_source("qa_pairs", [{"instruction": "...", "output": "..."}] * 500, weight=1.5)
integrator.add_source("custom", [{"instruction": "...", "output": "..."}] * 200, weight=0.5)

# 合併（根據權重）
merged = integrator.merge(max_samples_per_source=500)

# 或平衡合併
# balanced = integrator.balance_sources()
```

---

## 3.11 實際案例研究

### 3.11.1 案例 1：台灣繁體中文客服機器人資料集

**背景**：為台灣電商平台建立客服機器人訓練資料集

**需求**：
- 繁體中文
- 涵蓋常見客服場景（訂單、退貨、產品諮詢等）
- 10,000 條高品質對話資料

**實施步驟**：

1. **資料收集** (2 週)
   - 收集歷史客服對話記錄（匿名化處理）
   - 爬取常見問答網站的相關內容
   - 團隊成員貢獻種子問題

2. **資料清理** (1 週)
   ```python
   # 移除個人資訊
   pii_detector = PIIDetector()
   cleaned_data = pii_detector.clean_dataset(raw_data)

   # 繁體中文轉換
   converter = TraditionalChineseConverter()
   tw_data = converter.convert_dataset(cleaned_data,
                                        fields=['question', 'answer'],
                                        mode='s2tw')
   ```

3. **資料增強** (2 週)
   - 使用 LLM 生成相似問題的變體
   - 針對不同語氣重寫（正式/口語）
   - 添加錯別字和口語表達變體

4. **資料標註** (3 週)
   - 使用 Label Studio 進行意圖標註
   - 3 位標註者交叉驗證
   - 專家審核有爭議的案例

5. **品質控制** (1 週)
   - 計算標註者一致性（Kappa > 0.75）
   - 過濾低品質資料
   - 平衡各類別資料

**成果**：
- 最終資料集：12,500 條（超過目標）
- 涵蓋 15 個主要客服場景
- 標註一致性：0.82（優良）
- 訓練後模型準確率：87%

**經驗教訓**：
- ✅ PII 移除非常重要，避免法律問題
- ✅ 繁體中文轉換要人工審核關鍵詞彙
- ⚠️ 口語化表達需要更多樣化
- ⚠️ 長尾問題需要更多資料

### 3.11.2 案例 2：程式碼生成資料集（中英混合）

**背景**：為 Python 程式碼生成建立訓練資料集

**需求**：
- 中文指令 + Python 程式碼
- 涵蓋基礎到進階程式設計任務
- 5,000 條指令-程式碼對

**實施步驟**：

1. **資料來源**
   - GitHub 公開專案（爬取 + 清理）
   - LeetCode 中文題解
   - Stack Overflow 中文問答
   - 教學網站程式碼範例

2. **自動化處理**
   ```python
   # 從 GitHub 提取
   def extract_code_pairs(repo_path):
       pairs = []
       # 尋找有 docstring 的函數
       for file in glob_python_files(repo_path):
           functions = extract_functions_with_docs(file)
           for func in functions:
               # 將 docstring 轉為指令
               instruction = convert_docstring_to_instruction(func.docstring)
               pairs.append({
                   'instruction': instruction,
                   'output': func.code
               })
       return pairs
   ```

3. **合成資料生成**
   - 使用 GPT-4 生成多樣化的程式設計任務
   - 針對每個任務生成中文說明
   - 自動生成測試案例

4. **品質驗證**
   ```python
   # 程式碼可執行性檢查
   def validate_code(code: str, test_cases: List[Dict]) -> bool:
       try:
           exec(code)
           # 執行測試案例
           for test in test_cases:
               result = eval(f"func({test['input']})")
               if result != test['expected']:
                   return False
           return True
       except:
           return False
   ```

5. **資料分層**
   - 初級（40%）：基本語法、簡單函數
   - 中級（40%）：資料結構、演算法
   - 進階（20%）：複雜系統、優化

**成果**：
- 最終資料集：6,800 條
- 98% 程式碼可執行
- 涵蓋 20+ 程式設計主題
- 模型 pass@1 準確率：65%

**經驗教訓**：
- ✅ 自動化可執行性檢查節省大量時間
- ✅ 真實專案的程式碼品質較高
- ⚠️ 需要平衡不同難度等級
- ⚠️ 中文技術詞彙需要標準化

### 3.11.3 案例 3：醫療問答資料集（高品質小資料）

**背景**：建立醫療健康諮詢的問答資料集

**需求**：
- 高準確性（涉及健康安全）
- 專業審核
- 1,000 條精品資料

**特殊考量**：
- 醫療資訊必須準確
- 需要專業醫師審核
- 避免給出診斷性建議

**實施步驟**：

1. **資料收集** (嚴格篩選)
   - 衛生福利部官方資訊
   - 醫學期刊科普文章
   - 醫院官網衛教資訊
   - 排除個人醫療建議

2. **專業標註** (4 週)
   - 醫師編寫標準答案
   - 標註資訊類型（症狀、預防、治療、緊急）
   - 添加免責聲明

3. **多輪審核**
   ```
   第一輪：初級醫護人員編寫
   第二輪：主治醫師審核修正
   第三輪：不同科別醫師交叉檢查
   最終輪：法律顧問確認合規性
   ```

4. **安全過濾**
   ```python
   # 確保不包含診斷性語言
   forbidden_patterns = [
       r"你(患有|得了|有)",
       r"建議(立即|馬上)?服用",
       r"診斷(為|是)",
   ]

   def check_safety(answer: str) -> bool:
       for pattern in forbidden_patterns:
           if re.search(pattern, answer):
               return False
       return True
   ```

5. **資料格式**
   ```json
   {
     "question": "如何預防流感？",
     "answer": "預防流感的方法包括：1. 接種流感疫苗...",
     "category": "預防保健",
     "verified_by": "家醫科醫師",
     "last_reviewed": "2024-01-15",
     "disclaimer": "此資訊僅供參考，不能取代專業醫療建議..."
   }
   ```

**成果**：
- 最終資料集：1,200 條
- 100% 專業審核
- 涵蓋 8 個醫療主題
- 零醫療糾紛風險

**經驗教訓**：
- ✅ 品質 >> 數量（在醫療領域尤其重要）
- ✅ 多輪專業審核確保準確性
- ✅ 明確的免責聲明
- ⚠️ 成本高（專業審核費用）
- ⚠️ 耗時長（嚴格審核流程）

---

## 3.12 速查表與最佳實踐

### 3.12.1 資料集準備流程速查表

```
┌─────────────────────────────────────────────────────────────┐
│                    資料集準備檢查清單                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ 階段 1: 規劃 (Planning)                                       │
│ ☐ 定義任務目標和範圍                                          │
│ ☐ 確定所需資料量                                              │
│ ☐ 評估預算和時程                                              │
│ ☐ 確認資料授權和合規性                                        │
│                                                               │
│ 階段 2: 收集 (Collection)                                     │
│ ☐ 識別資料來源                                                │
│ ☐ 收集原始資料                                                │
│ ☐ 檢查資料授權                                                │
│ ☐ 備份原始資料                                                │
│                                                               │
│ 階段 3: 清理 (Cleaning)                                       │
│ ☐ 移除 HTML 標籤和特殊字元                                    │
│ ☐ 移除 PII（個人識別資訊）                                    │
│ ☐ 標準化格式                                                  │
│ ☐ 去除重複資料                                                │
│ ☐ 過濾低品質內容                                              │
│                                                               │
│ 階段 4: 標註 (Annotation)                                     │
│ ☐ 建立標註指南                                                │
│ ☐ 培訓標註者                                                  │
│ ☐ 多人標註（提高可靠性）                                      │
│ ☐ 計算標註者一致性                                            │
│ ☐ 解決衝突案例                                                │
│                                                               │
│ 階段 5: 驗證 (Validation)                                     │
│ ☐ 品質評估                                                    │
│ ☐ 統計分析（長度、多樣性等）                                  │
│ ☐ 專家審核樣本                                                │
│ ☐ 測試案例驗證                                                │
│                                                               │
│ 階段 6: 版本控制 (Versioning)                                 │
│ ☐ 建立元資料檔案                                              │
│ ☐ 計算校驗和                                                  │
│ ☐ 編寫 CHANGELOG                                              │
│ ☐ 使用版本控制工具（DVC/Git）                                 │
│                                                               │
│ 階段 7: 交付 (Delivery)                                       │
│ ☐ 分割訓練/驗證/測試集                                        │
│ ☐ 轉換為所需格式                                              │
│ ☐ 生成資料集卡片（Dataset Card）                              │
│ ☐ 文檔化處理流程                                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.12.2 資料品質評估指標

| 指標 | 說明 | 建議值 | 計算方式 |
|------|------|--------|----------|
| **覆蓋率** | 資料涵蓋的主題/任務類型 | > 80% | 已涵蓋任務 / 總任務數 |
| **多樣性** | 詞彙和句式的多樣性 | > 0.6 | 獨特詞彙數 / 總詞彙數 |
| **準確性** | 標註或內容的正確性 | > 95% | 正確項目數 / 總項目數 |
| **一致性** | 標註者間的一致性 | Kappa > 0.7 | Fleiss' Kappa |
| **完整性** | 資料欄位的完整度 | > 98% | 完整記錄 / 總記錄數 |
| **平衡性** | 各類別的資料分佈 | 1:3 以內 | 最小類別 / 最大類別 |

### 3.12.3 常見問題與解決方案

**問題 1：資料量不足**
```
解決方案：
1. 資料增強（back-translation、paraphrasing）
2. 合成資料生成（使用 LLM）
3. Few-shot learning（減少所需資料量）
4. 遷移學習（使用類似領域的資料）
```

**問題 2：類別不平衡**
```
解決方案：
1. 過採樣（over-sampling）少數類別
2. 欠採樣（under-sampling）多數類別
3. 合成少數類別資料（SMOTE、ADASYN）
4. 調整損失函數權重
```

**問題 3：標註成本高**
```
解決方案：
1. 主動學習（選擇最有價值的樣本）
2. 半監督學習（利用未標註資料）
3. 弱監督學習（使用規則或啟發式標註）
4. RLAIF（使用 AI 輔助標註）
```

**問題 4：資料品質參差不齊**
```
解決方案：
1. 建立明確的品質標準
2. 多輪審核流程
3. 使用 LLM 輔助品質評估
4. 統計異常值檢測
5. 人工抽樣審核
```

**問題 5：隱私和合規問題**
```
解決方案：
1. PII 自動偵測和移除
2. 資料匿名化
3. 差分隱私技術
4. 法律顧問審核
5. 明確的使用授權
```

### 3.12.4 工具選擇指南

**資料收集**：
- 網頁爬取：Scrapy, BeautifulSoup, Selenium
- API 介接：requests, httpx
- 資料庫查詢：SQLAlchemy, PyMongo

**資料清理**：
- 文字處理：pandas, re, ftfy
- 去重複：datasketch (MinHash), dedupe
- PII 移除：presidio, scrubadub

**資料標註**：
- 開源工具：Label Studio, Doccano, Argilla
- 商業工具：Prodigy, Labelbox, Scale AI
- 眾包平台：Amazon MTurk, Figure Eight

**資料增強**：
- NLP 增強：nlpaug, TextAttack
- LLM 生成：OpenAI API, Anthropic Claude
- 翻譯：Google Translate API, DeepL

**版本控制**：
- 資料版本：DVC, LakeFS, Pachyderm
- 程式碼版本：Git, GitHub, GitLab
- 實驗追蹤：MLflow, Weights & Biases

**品質檢查**：
- 統計分析：pandas, numpy, scipy
- 視覺化：matplotlib, seaborn, plotly
- LLM 評估：OpenAI GPT-4, Claude

### 3.12.5 最佳實踐總結

#### ✅ 應該做的事

1. **及早定義品質標準**
   - 在收集資料前就明確品質要求
   - 建立可量化的評估指標

2. **保留原始資料**
   - 永遠備份未處理的原始資料
   - 記錄所有轉換步驟

3. **多樣性優先**
   - 涵蓋不同場景、語氣、難度
   - 避免過度專注單一領域

4. **迭代改進**
   - 先建立小規模 MVP 資料集
   - 根據模型表現逐步改進

5. **文檔化一切**
   - 記錄資料來源和處理流程
   - 維護詳細的 CHANGELOG

6. **合規性第一**
   - 確保資料使用授權
   - 移除個人隱私資訊

7. **人工審核關鍵樣本**
   - 不完全依賴自動化
   - 定期抽樣人工檢查

#### ❌ 應該避免的事

1. **不要忽視資料授權**
   - 未經授權使用受版權保護的資料
   - 可能導致法律問題

2. **不要過度依賴單一來源**
   - 會導致偏見和過擬合
   - 限制模型泛化能力

3. **不要跳過清理步驟**
   - 「垃圾進，垃圾出」
   - 髒資料會嚴重影響模型品質

4. **不要忽視長尾分佈**
   - 稀有但重要的案例也需要涵蓋
   - 平衡是關鍵

5. **不要過度增強**
   - 合成資料可能引入人工偏見
   - 保持一定比例的真實資料

6. **不要忽視版本控制**
   - 沒有版本控制會導致實驗不可重現
   - 無法追溯問題

7. **不要單獨工作**
   - 多人參與提高品質
   - 交叉檢查發現盲點

### 3.12.6 ROI 評估

**時間投入建議**：

```
小型專案（1K-5K 資料）：
├─ 收集：20%（1 週）
├─ 清理：25%（1.5 週）
├─ 標註：30%（2 週）
├─ 驗證：15%（1 週）
└─ 文檔：10%（0.5 週）
總計：6 週

中型專案（10K-50K 資料）：
├─ 收集：20%（3 週）
├─ 清理：20%（3 週）
├─ 標註：35%（5 週）
├─ 驗證：15%（2 週）
└─ 文檔：10%（1.5 週）
總計：14.5 週

大型專案（100K+ 資料）：
├─ 收集：15%（6 週）
├─ 清理：20%（8 週）
├─ 標註：40%（16 週）
├─ 驗證：15%（6 週）
└─ 文檔：10%（4 週）
總計：40 週
```

**成本估算**：

| 項目 | 預算佔比 | 說明 |
|------|---------|------|
| 資料收集 | 10-15% | API 費用、爬蟲維護 |
| 資料標註 | 40-50% | 標註人力成本最高 |
| 工具授權 | 10-15% | 標註工具、API 訂閱 |
| 品質審核 | 15-20% | 專家審核費用 |
| 基礎設施 | 5-10% | 儲存、計算資源 |
| 雜項 | 5-10% | 預留緩衝 |

**品質 vs 成本權衡**：

```
情境 A：高品質小資料集
- 1,000 條專家審核資料
- 成本：高（$10-50/條）
- 適用：醫療、法律、金融等高風險領域

情境 B：中品質中型資料集
- 10,000 條專業標註資料
- 成本：中（$1-5/條）
- 適用：一般商業應用

情境 C：可接受品質大型資料集
- 100,000 條眾包/半自動標註
- 成本：低（$0.1-0.5/條）
- 適用：預訓練、通用任務
```

---

## 參考資源

### 開源資料集

#### 英文資料集

**Instruction Following**：
- **Alpaca** (52K): https://github.com/tatsu-lab/stanford_alpaca
  - Stanford 的 instruction-following 資料集
  - 使用 Self-Instruct 方法生成

- **Dolly 15k**: https://huggingface.co/datasets/databricks/databricks-dolly-15k
  - Databricks 員工標註的高品質資料
  - 涵蓋多種任務類型

- **OpenAssistant Conversations** (161K): https://huggingface.co/datasets/OpenAssistant/oasst1
  - 大規模對話資料集
  - 包含多輪對話和品質評分

- **ShareGPT**: https://huggingface.co/datasets/RyokoAI/ShareGPT52K
  - 真實使用者與 ChatGPT 的對話
  - 自然且多樣化

**RLHF/Preference 資料**：
- **Anthropic HH-RLHF**: https://huggingface.co/datasets/Anthropic/hh-rlhf
  - Human preference 資料
  - 有幫助且無害的對話

- **OpenAI WebGPT**: https://huggingface.co/datasets/openai/webgpt_comparisons
  - 網頁搜尋相關的偏好資料

- **Stanford SHP** (385K): https://huggingface.co/datasets/stanfordnlp/SHP
  - 來自 Reddit 的偏好資料
  - 18 個不同領域

**程式碼**：
- **CodeAlpaca** (20K): https://github.com/sahil280114/codealpaca
  - 程式碼生成 instruction 資料

- **Code Contests**: https://huggingface.co/datasets/deepmind/code_contests
  - 程式競賽問題和解答

#### 中文/繁體中文資料集

**Instruction 資料**：
- **BELLE** (0.5M-2M): https://github.com/LianjiaTech/BELLE
  - 大規模中文 instruction 資料
  - 多個子集可選

- **Chinese-Alpaca**: https://github.com/ymcui/Chinese-LLaMA-Alpaca
  - 中文版 Alpaca 資料
  - 適合繁體中文轉換

- **COIG** (298K): https://huggingface.co/datasets/BAAI/COIG
  - 中文開源 instruction 資料集
  - 整合多個來源

- **Firefly** (1.1M): https://github.com/yangjianxin1/Firefly
  - 大規模中文對話資料
  - 涵蓋 23 種任務

- **Guanaco**: https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
  - 多語言（包含繁體中文）

**台灣特有資源**：
- **TAIDE 資料集**: https://taide.tw/
  - 台灣可信任生成式 AI 對話引擎
  - 繁體中文優化

- **中華文化語料庫**: 學術機構合作建立
  - 台灣特有文化內容

### 工具與庫

#### 資料收集
- **Scrapy**: https://scrapy.org/ - 專業爬蟲框架
- **BeautifulSoup**: https://www.crummy.com/software/BeautifulSoup/ - HTML 解析
- **Selenium**: https://www.selenium.dev/ - 動態網頁爬取
- **Playwright**: https://playwright.dev/ - 現代化瀏覽器自動化
- **requests-html**: https://github.com/psf/requests-html - 簡化的爬蟲工具

#### 資料清理與處理
- **pandas**: https://pandas.pydata.org/ - 資料分析和處理
- **ftfy**: https://github.com/rspeer/python-ftfy - 修正文字編碼問題
- **OpenCC**: https://github.com/BYVoid/OpenCC - 繁簡轉換（支援台灣正體）
- **jieba**: https://github.com/fxsjy/jieba - 中文分詞
- **ckiptagger**: https://github.com/ckiplab/ckiptagger - 中研院中文處理工具
- **datasketch**: https://github.com/ekzhu/datasketch - 大規模去重（MinHash）

#### 資料標註
- **Label Studio**: https://labelstud.io/ - 多功能標註平台
- **Doccano**: https://github.com/doccano/doccano - 文字標註工具
- **Argilla**: https://github.com/argilla-io/argilla - ML 友善標註平台
- **Prodigy**: https://prodi.gy/ - 商業標註工具（支援主動學習）
- **LabelImg**: https://github.com/heartexlabs/labelImg - 圖片標註

#### 資料增強
- **nlpaug**: https://github.com/makcedward/nlpaug - NLP 資料增強
- **TextAttack**: https://github.com/QData/TextAttack - 對抗式資料增強
- **EDA**: https://github.com/jasonwei20/eda_nlp - 簡單資料增強

#### 版本控制與管理
- **DVC**: https://dvc.org/ - 資料版本控制
- **LakeFS**: https://lakefs.io/ - 資料湖版本控制
- **Pachyderm**: https://www.pachyderm.com/ - 資料管線自動化
- **MLflow**: https://mlflow.org/ - ML 實驗追蹤

#### 品質檢查
- **Great Expectations**: https://greatexpectations.io/ - 資料驗證
- **Presidio**: https://github.com/microsoft/presidio - PII 偵測與移除
- **scrubadub**: https://github.com/LeapBeyond/scrubadub - PII 清理

#### Hugging Face 生態系
- **Datasets**: https://huggingface.co/docs/datasets - 資料集載入和處理
- **Transformers**: https://huggingface.co/docs/transformers - 模型和資料處理
- **TRL**: https://github.com/huggingface/trl - RLHF 訓練工具

### 重要論文

#### Instruction Tuning
1. **FLAN** - "Finetuned Language Models Are Zero-Shot Learners" (Wei et al., 2022)
   - 提出 instruction tuning 概念
   - https://arxiv.org/abs/2109.01652

2. **Self-Instruct** - "Self-Instruct: Aligning Language Model with Self Generated Instructions" (Wang et al., 2022)
   - 使用 LLM 自動生成 instruction 資料
   - https://arxiv.org/abs/2212.10560

3. **Alpaca** - "Alpaca: A Strong, Replicable Instruction-Following Model" (Stanford, 2023)
   - 低成本 instruction tuning
   - https://crfm.stanford.edu/2023/03/13/alpaca.html

4. **Evol-Instruct** - "WizardLM: Empowering Large Language Models to Follow Complex Instructions" (Xu et al., 2023)
   - 演化式 instruction 生成
   - https://arxiv.org/abs/2304.12244

#### RLHF & Alignment
5. **InstructGPT** - "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
   - OpenAI 的 RLHF 方法
   - https://arxiv.org/abs/2203.02155

6. **Constitutional AI** - "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
   - Anthropic 的 RLAIF 方法
   - https://arxiv.org/abs/2212.08073

7. **DPO** - "Direct Preference Optimization" (Rafailov et al., 2023)
   - 無需 reward model 的偏好優化
   - https://arxiv.org/abs/2305.18290

#### 資料品質與過濾
8. **DataComp** - "DataComp: In search of the next generation of multimodal datasets" (Gadre et al., 2023)
   - 大規模資料集品質研究
   - https://arxiv.org/abs/2304.14108

9. **LIMA** - "Less Is More for Alignment" (Zhou et al., 2023)
   - 1000 條高品質資料的效果
   - https://arxiv.org/abs/2305.11206

10. **Textbooks Are All You Need** (Gunasekar et al., 2023)
    - 高品質合成資料的重要性
    - https://arxiv.org/abs/2306.11644

#### 資料增強
11. **Back-Translation** - "Improving Neural Machine Translation Models with Monolingual Data" (Sennrich et al., 2016)
    - https://arxiv.org/abs/1511.06709

12. **EDA** - "Easy Data Augmentation Techniques" (Wei & Zou, 2019)
    - https://arxiv.org/abs/1901.11196

### 線上資源與教學

#### 官方文檔
- **Hugging Face 資料集教學**: https://huggingface.co/docs/datasets/tutorial
- **OpenAI Fine-tuning Guide**: https://platform.openai.com/docs/guides/fine-tuning
- **Anthropic 資料準備指南**: https://docs.anthropic.com/claude/docs

#### 部落格文章
- **How to create an instruction dataset**: https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2
- **RLHF: Reinforcement Learning from Human Feedback**: https://huggingface.co/blog/rlhf
- **Building a dataset from scratch**: https://www.surgehq.ai/blog/how-to-build-a-dataset-from-scratch

#### 社群與論壇
- **r/LocalLLaMA**: https://www.reddit.com/r/LocalLLaMA/ - LLM 社群
- **Hugging Face Forums**: https://discuss.huggingface.co/ - 技術討論
- **LangChain Discord**: https://discord.gg/langchain - AI 開發者社群

### 付費服務

#### 標註服務
- **Scale AI**: https://scale.com/ - 專業標註服務
- **Labelbox**: https://labelbox.com/ - 標註平台
- **Figure Eight (Appen)**: https://appen.com/ - 眾包標註
- **Amazon MTurk**: https://www.mturk.com/ - 眾包平台

#### API 服務
- **OpenAI API**: https://platform.openai.com/ - GPT 系列
- **Anthropic API**: https://www.anthropic.com/api - Claude 系列
- **Cohere**: https://cohere.ai/ - 企業級 NLP API

---

## 總結

資料集準備是 LLM 訓練的基礎，品質直接影響模型性能：

### 核心要點

1. **資料品質 > 資料數量**
   - 1000 條高品質資料勝過 10000 條低品質資料
   - 投入時間進行清理和過濾
   - 人工審核關鍵樣本

2. **多樣性很重要**
   - 涵蓋不同主題、難度、格式
   - 使用資料增強技術
   - 整合多個資料來源

3. **格式標準化**
   - 統一資料格式
   - 與目標模型的訓練格式匹配
   - 明確區分輸入和輸出

4. **持續迭代**
   - 收集 -> 清理 -> 評估 -> 改進
   - 根據模型表現調整資料
   - 記錄資料處理流程

5. **台灣在地化**
   - 使用繁體中文
   - 調整文化相關內容
   - 考慮台灣使用者習慣

### 實務建議

**起步階段**：
- 從現有開源資料集開始
- 使用工具自動轉換為繁體中文
- 建立小規模高品質資料集（1000-5000 條）

**成長階段**：
- 收集領域特定資料
- 使用 LLM 協助生成合成資料
- 建立資料審核流程

**成熟階段**：
- 建立自動化資料處理 pipeline
- 持續監控資料品質
- 根據使用者回饋優化資料集
