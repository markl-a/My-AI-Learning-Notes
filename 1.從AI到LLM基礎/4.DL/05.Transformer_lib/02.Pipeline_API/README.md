# Pipeline API - 快速上手指南

Pipeline API 是 Transformers 最簡單的使用方式，只需 2-3 行代碼即可完成複雜的 NLP、CV、Audio 任務。

## 📚 學習目標

- 掌握 Pipeline API 的基本使用
- 了解各種任務類型和對應的 pipeline
- 學習自定義 pipeline 參數
- 掌握批次處理和性能優化

## 支援的任務類型（2025 最新）

### NLP 任務
- `text-classification` - 文本分類/情感分析
- `token-classification` - 命名實體識別 (NER)
- `question-answering` - 問答系統
- `text-generation` - 文本生成
- `summarization` - 文本摘要
- `translation` - 機器翻譯
- `zero-shot-classification` - 零樣本分類
- `fill-mask` - 填空
- `text2text-generation` - 文本到文本生成

### Computer Vision 任務
- `image-classification` - 圖像分類
- `object-detection` - 目標檢測
- `image-segmentation` - 圖像分割
- `depth-estimation` - 深度估計
- `zero-shot-image-classification` - 零樣本圖像分類

### Audio 任務
- `automatic-speech-recognition` - 語音識別
- `audio-classification` - 音頻分類
- `text-to-speech` - 文本轉語音

### Multimodal 任務
- `visual-question-answering` - 視覺問答
- `image-to-text` - 圖像描述
- `document-question-answering` - 文檔問答

## 快速開始

### 基本用法

```python
from transformers import pipeline

# 1. 創建 pipeline（自動下載模型）
classifier = pipeline("sentiment-analysis")

# 2. 使用 pipeline
result = classifier("I love Hugging Face!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# 3. 批次處理
results = pipeline([
    "This is great!",
    "This is terrible."
])
```

### 指定模型

```python
# 使用特定模型
classifier = pipeline(
    "sentiment-analysis",
    model="bert-base-chinese",
    tokenizer="bert-base-chinese"
)

# 或使用模型 ID
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
```

### 設備配置

```python
import torch

# 使用 GPU
classifier = pipeline("sentiment-analysis", device=0)  # GPU 0

# 使用 CPU
classifier = pipeline("sentiment-analysis", device=-1)

# 自動選擇
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device=device)
```

## 常見任務示例

### 1. 文本分類

```python
classifier = pipeline("text-classification")
result = classifier("這個產品非常棒！")
print(result)
```

### 2. 命名實體識別 (NER)

```python
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
entities = ner(text)
for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.4f})")
```

### 3. 問答系統

```python
qa = pipeline("question-answering")
context = "Transformers was created by Hugging Face. It provides thousands of pretrained models."
question = "Who created Transformers?"
answer = qa(question=question, context=context)
print(f"Answer: {answer['answer']} (score: {answer['score']:.4f})")
```

### 4. 文本生成（2025 最新模型）

```python
# 使用 LLaMA 3
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

prompt = "Explain quantum computing in simple terms:"
output = generator(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
print(output[0]['generated_text'])
```

### 5. 圖像分類

```python
from PIL import Image

classifier = pipeline("image-classification")
image = Image.open("cat.jpg")
results = classifier(image)
for result in results:
    print(f"{result['label']}: {result['score']:.4f}")
```

### 6. 語音識別（Whisper V3）

```python
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3"
)

# 從文件
result = transcriber("audio.mp3")
print(result["text"])

# 從 URL
result = transcriber("https://example.com/audio.wav")
```

### 7. 零樣本分類

```python
classifier = pipeline("zero-shot-classification")
text = "This tutorial explains how to use Transformers."
candidate_labels = ["education", "politics", "sports", "technology"]

result = classifier(text, candidate_labels)
print(f"Labels: {result['labels']}")
print(f"Scores: {result['scores']}")
```

## 進階用法

### 批次處理優化

```python
from transformers import pipeline

# 創建 pipeline
classifier = pipeline("sentiment-analysis", device=0)

# 大批次數據
texts = ["text " + str(i) for i in range(1000)]

# 設置批次大小
results = classifier(texts, batch_size=32)
```

### 流式生成

```python
generator = pipeline("text-generation", model="gpt2")

# 流式輸出
for output in generator("Once upon a time", max_length=50, num_return_sequences=1):
    print(output['generated_text'], end='', flush=True)
```

### 自定義參數

```python
classifier = pipeline(
    "sentiment-analysis",
    model="bert-base-chinese",
    tokenizer="bert-base-chinese",
    framework="pt",  # 'pt' 或 'tf'
    device=0,
    batch_size=16,
    max_length=512,
    truncation=True,
    padding=True
)
```

### 訪問底層模型

```python
classifier = pipeline("sentiment-analysis")

# 訪問模型
model = classifier.model
tokenizer = classifier.tokenizer

# 自定義推理
inputs = tokenizer("Some text", return_tensors="pt")
outputs = model(**inputs)
```

## 性能優化技巧

### 1. 使用批次處理

```python
# 慢：逐一處理
for text in texts:
    result = classifier(text)

# 快：批次處理
results = classifier(texts, batch_size=32)
```

### 2. 使用 FP16

```python
import torch

classifier = pipeline(
    "text-classification",
    model="bert-base-chinese",
    torch_dtype=torch.float16,
    device=0
)
```

### 3. 預載入模型

```python
# 初始化時載入模型
classifier = pipeline("sentiment-analysis", device=0)

# 預熱
_ = classifier("warm up")

# 實際使用
results = classifier(your_texts)
```

## 錯誤處理

```python
from transformers import pipeline

try:
    classifier = pipeline("sentiment-analysis")
    result = classifier("Some text")
except Exception as e:
    print(f"Error: {e}")
    # 降級處理
    classifier = pipeline("sentiment-analysis", device=-1)  # 使用 CPU
```

## 常見問題

**Q: Pipeline 支援哪些框架？**
A: PyTorch、TensorFlow 和 JAX。

**Q: 如何選擇合適的模型？**
A: 訪問 [Hugging Face Model Hub](https://huggingface.co/models) 根據任務和語言篩選。

**Q: Pipeline 會自動下載模型嗎？**
A: 是的，首次使用會自動下載並快取。

**Q: 如何離線使用？**
A: 提前下載模型並使用本地路徑：
```python
classifier = pipeline("sentiment-analysis", model="./local_model")
```

## 延伸閱讀

- [Pipeline 官方文檔](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [任務指南](https://huggingface.co/docs/transformers/task_summary)
- [模型列表](https://huggingface.co/models)

## 下一步

- 查看 [示例代碼](./examples/) 了解更多用法
- 前往 [03. 模型微調](../03.模型微調/) 學習如何訓練自己的模型
