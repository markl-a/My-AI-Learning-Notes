# 資料集準備與建立

## 目錄
1. [原始資料收集與清理](#31-原始資料收集與清理)
2. [Instruction Dataset 準備與格式化](#32-instruction-dataset-準備與格式化)
3. [資料增強技術](#33-資料增強技術)
4. [使用現有資料集](#34-使用現有資料集)
5. [資料品質評估與過濾](#35-資料品質評估與過濾)
6. [實作範例](#36-實作範例)

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

## 3.6 實作範例

### 3.6.1 完整的資料處理流程

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

### 3.6.2 從多個來源整合資料

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

## 參考資源

### 開源資料集

**英文**：
- **Alpaca**: 52K instruction-following 範例
- **Dolly 15k**: 人工標註的 instruction 資料
- **ShareGPT**: 真實使用者與 ChatGPT 的對話
- **OpenAssistant**: 開源對話資料集

**中文**：
- **BELLE**: 中文 instruction 資料集
- **Chinese-Alpaca**: 中文 Alpaca 資料
- **COIG**: 中文開源 instruction 資料集
- **Firefly**: 中文對話資料集

### 工具與庫

- **Hugging Face Datasets**: https://huggingface.co/docs/datasets
- **OpenCC**: https://github.com/BYVoid/OpenCC（繁簡轉換）
- **jieba**: https://github.com/fxsjy/jieba（中文分詞）
- **BeautifulSoup**: https://www.crummy.com/software/BeautifulSoup/
- **Scrapy**: https://scrapy.org/（專業爬蟲框架）

### 論文

1. **Self-Instruct**: "Self-Instruct: Aligning Language Model with Self Generated Instructions" (Wang et al., 2022)
2. **Evol-Instruct**: "WizardLM: Empowering Large Language Models to Follow Complex Instructions" (Xu et al., 2023)
3. **Alpaca**: "Alpaca: A Strong, Replicable Instruction-Following Model" (Stanford, 2023)

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
