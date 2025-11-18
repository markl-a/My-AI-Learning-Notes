# GitHub Copilot 最佳實踐指南

> 如何最大化利用 AI 助手提升 NLP 開發效率

## 📚 目錄

- [什麼是 GitHub Copilot](#什麼是-github-copilot)
- [基礎使用](#基礎使用)
- [高級技巧](#高級技巧)
- [實際案例](#實際案例)
- [最佳實踐](#最佳實踐)
- [常見問題](#常見問題)

## 什麼是 GitHub Copilot

GitHub Copilot 是一個 AI 程式設計助手，由 OpenAI Codex 驅動，可以：
- 📝 自動補全代碼
- 🧪 生成測試用例
- 📖 撰寫文檔
- 🐛 解釋代碼
- 🔄 重構建議

## 基礎使用

### 1. 函數生成

**技巧:** 用清晰的注釋描述你想要的功能

```python
# Function to tokenize Chinese text using jieba and return a list of tokens
```

**Copilot 建議:**
```python
def tokenize_chinese(text):
    """
    Tokenize Chinese text using jieba

    Args:
        text (str): Input Chinese text

    Returns:
        list: List of tokens
    """
    import jieba
    return list(jieba.cut(text))
```

### 2. 類實現

```python
# Class for sentiment analysis with BERT
# Should have methods: __init__, load_model, predict, batch_predict
```

**Copilot 建議:**
```python
class BERTSentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return prediction

    def batch_predict(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()
        return predictions
```

### 3. 數據處理

```python
# Function to clean text: lowercase, remove URLs, remove HTML tags, remove extra spaces
```

**Copilot 建議:**
```python
import re

def clean_text(text):
    """Clean text for NLP processing"""
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove extra spaces
    text = ' '.join(text.split())

    return text.strip()
```

## 高級技巧

### 1. 上下文感知

Copilot 會根據你已有的代碼提供更相關的建議。

**示例:**

```python
import torch
from transformers import AutoTokenizer, AutoModel

class TextEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    # Method to encode a single text
    # Copilot 知道你正在使用 Transformers，會提供相應的建議
```

**Copilot 建議:**
```python
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
```

### 2. 測試用例生成

```python
# 給定一個函數
def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# 輸入注釋來生成測試
# Test cases for calculate_f1_score function using pytest
```

**Copilot 建議:**
```python
import pytest

def test_calculate_f1_score_normal_case():
    assert calculate_f1_score(0.5, 0.5) == 0.5
    assert calculate_f1_score(0.8, 0.6) == pytest.approx(0.685, 0.01)

def test_calculate_f1_score_edge_cases():
    assert calculate_f1_score(0, 0) == 0.0
    assert calculate_f1_score(1, 1) == 1.0

def test_calculate_f1_score_zero_precision():
    assert calculate_f1_score(0, 0.5) == 0.0

def test_calculate_f1_score_zero_recall():
    assert calculate_f1_score(0.5, 0) == 0.0
```

### 3. 文檔字符串生成

**技巧:** 定義函數後，在函數內輸入三個引號，Copilot 會自動生成文檔

```python
def train_sentiment_model(model, train_loader, val_loader, epochs=5, lr=2e-5):
    """
    # 在這裡輸入三個引號，Copilot 會生成完整的 docstring
```

**Copilot 建議:**
```python
def train_sentiment_model(model, train_loader, val_loader, epochs=5, lr=2e-5):
    """
    Train a sentiment analysis model

    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of training epochs (default: 5)
        lr (float): Learning rate (default: 2e-5)

    Returns:
        dict: Training history containing loss and accuracy for each epoch

    Example:
        >>> model = BERTForSequenceClassification.from_pretrained('bert-base-uncased')
        >>> history = train_sentiment_model(model, train_loader, val_loader)
    """
    # Training code here...
```

### 4. 多行建議

**技巧:** 使用 Alt + ] 或 Alt + [ 在多個建議之間切換

```python
# Complete data preprocessing pipeline for NLP
```

Copilot 可能會提供多個不同的實現方案，你可以選擇最合適的。

## 實際案例

### 案例 1: 快速構建文本分類器

```python
# Build a complete text classification pipeline with the following steps:
# 1. Load data
# 2. Preprocess text
# 3. Create DataLoader
# 4. Define model
# 5. Train model
# 6. Evaluate
```

**Copilot 會生成一個完整的流程框架:**

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

class TextClassificationPipeline:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def load_data(self, texts, labels, batch_size=16):
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'],
                               torch.tensor(labels))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, train_loader, epochs=3, lr=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy
```

### 案例 2: 數據增強函數

```python
# Function to augment text data using:
# 1. Synonym replacement
# 2. Random insertion
# 3. Random swap
# 4. Random deletion
```

**Copilot 建議:**
```python
import random
import nltk
from nltk.corpus import wordnet

def augment_text(text, augmentation_type='synonym', num_aug=1):
    """
    Augment text data using various techniques

    Args:
        text (str): Input text
        augmentation_type (str): Type of augmentation
        num_aug (int): Number of words to augment

    Returns:
        str: Augmented text
    """
    words = text.split()

    if augmentation_type == 'synonym':
        for _ in range(num_aug):
            idx = random.randint(0, len(words)-1)
            synonyms = get_synonyms(words[idx])
            if synonyms:
                words[idx] = random.choice(synonyms)

    elif augmentation_type == 'insertion':
        for _ in range(num_aug):
            idx = random.randint(0, len(words))
            word_to_insert = random.choice(words)
            words.insert(idx, word_to_insert)

    elif augmentation_type == 'swap':
        for _ in range(num_aug):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]

    elif augmentation_type == 'deletion':
        for _ in range(num_aug):
            if len(words) > 1:
                idx = random.randint(0, len(words)-1)
                words.pop(idx)

    return ' '.join(words)

def get_synonyms(word):
    """Get synonyms from WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)
```

## 最佳實踐

### 1. 編寫清晰的注釋

✅ **好的注釋:**
```python
# Function to calculate precision, recall, and F1 score
# Args: y_true (list), y_pred (list)
# Returns: dict with 'precision', 'recall', 'f1' keys
```

❌ **不好的注釋:**
```python
# calc metrics
```

### 2. 提供類型提示

Copilot 能更好地理解你的意圖：

```python
from typing import List, Dict, Tuple

def process_batch(texts: List[str], labels: List[int]) -> Dict[str, torch.Tensor]:
    # Copilot 知道返回類型，會生成相應的代碼
```

### 3. 使用示例驅動

```python
# Function to extract named entities from text
# Example:
#   Input: "Apple Inc. is located in Cupertino."
#   Output: [{'text': 'Apple Inc.', 'type': 'ORG'},
#            {'text': 'Cupertino', 'type': 'LOC'}]
```

### 4. 迭代改進

不要期望第一次建議就完美：
1. 接受 Copilot 的建議
2. 調整注釋使其更精確
3. 重新生成獲得更好的結果

### 5. 結合其他工具

```python
# 使用 Copilot 生成代碼框架
# 使用 ChatGPT/Claude 解釋複雜邏輯
# 使用 IDE 調試工具優化性能
```

## 常見問題

### Q1: Copilot 生成的代碼總是正確的嗎？

**A:** 不一定。Copilot 是輔助工具，需要人工審查：
- ✅ 檢查邏輯正確性
- ✅ 驗證邊界情況
- ✅ 測試異常處理
- ✅ 確保代碼安全性

### Q2: 如何提高 Copilot 建議質量？

**A:**
1. 提供清晰的上下文
2. 使用描述性的變量名
3. 添加類型提示
4. 參考現有代碼風格
5. 給出具體示例

### Q3: Copilot 能幫助學習嗎？

**A:** 可以！
- 📚 學習常見模式
- 🔍 發現新的庫和方法
- 💡 獲得實現思路
- 但要理解代碼而不是盲目複製

### Q4: 什麼情況不適合用 Copilot？

**A:**
- 需要深入理解的算法
- 關鍵的業務邏輯
- 安全敏感的代碼
- 需要創新性的解決方案

## 快捷鍵

### VS Code

- `Tab` - 接受建議
- `Esc` - 拒絕建議
- `Alt + ]` - 下一個建議
- `Alt + [` - 上一個建議
- `Ctrl + Enter` - 打開 Copilot 面板（查看多個建議）

### PyCharm

- `Tab` - 接受建議
- `Esc` - 拒絕建議

## 進階應用

### 1. 生成正則表達式

```python
# Regex to match email addresses
email_pattern = # Copilot 會建議完整的正則表達式
```

### 2. 單元測試框架

```python
# Setup pytest fixture for loading test data
@pytest.fixture
def # Copilot 會生成完整的 fixture
```

### 3. 錯誤處理

```python
def load_model(model_path):
    # Add comprehensive error handling for:
    # - File not found
    # - Invalid model format
    # - CUDA out of memory
```

## 總結

GitHub Copilot 是強大的生產力工具，但需要正確使用：

✅ **做:**
- 用於生成樣板代碼
- 輔助學習新技術
- 加速原型開發
- 生成測試和文檔

❌ **不要:**
- 盲目信任建議
- 跳過代碼審查
- 忽視安全考慮
- 放棄思考

**記住:** Copilot 是副駕駛，你才是駕駛員！🚗

---

**下一步:** 查看 [AI 輔助 Prompt 工程](./llm_prompts.md) 了解如何使用 ChatGPT/Claude 輔助開發。
