# 實戰項目

通過實際項目鞏固和應用所學知識。

## 📚 項目列表

1. [情感分析系統](#項目1-情感分析系統)
2. [問答機器人](#項目2-問答機器人)
3. [文字摘要應用](#項目3-文字摘要應用)
4. [多語言翻譯工具](#項目4-多語言翻譯工具)
5. [語音助手](#項目5-語音助手)

---

## 項目 1: 情感分析系統

### 項目目標
構建一個能夠分析中文文字情感（正面/負面/中性）的系統。

### 技術棧
- BERT-base-chinese
- PyTorch
- Gradio (UI)
- FastAPI (API)

### 實現步驟

#### 1. 資料準備

```python
from datasets import load_dataset, Dataset
import pandas as pd

# 載入或建立資料集
def load_sentiment_data():
    # 方法 1: 從 Hugging Face Hub
    dataset = load_dataset("tyqiangz/multilingual-sentiments", "chinese")

    # 方法 2: 從本地 CSV
    df = pd.read_csv("sentiment_data.csv")
    dataset = Dataset.from_pandas(df)

    return dataset

dataset = load_sentiment_data()
```

#### 2. 模型微調

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# 載入模型
model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 資料預處理
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess, batched=True)

# 訓練
training_args = TrainingArguments(
    output_dir="./sentiment-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./sentiment-final")
```

#### 3. 部署為 Web 應用

```python
import gradio as gr
from transformers import pipeline

# 載入模型
sentiment_analyzer = pipeline(
    "text-classification",
    model="./sentiment-final",
    device=0
)

def predict_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return {
        "情感": result["label"],
        "信心度": f"{result['score']:.4f}"
    }

# 建立 Gradio 界面
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="輸入文字", placeholder="輸入要分析的文字..."),
    outputs=gr.JSON(label="分析結果"),
    title="中文情感分析系統",
    description="基於 BERT 的中文情感分析",
    examples=[
        ["這個產品非常好用！"],
        ["質量太差了，很失望。"],
        ["還可以，沒有特別驚艷。"]
    ]
)

demo.launch()
```

---

## 項目 2: 問答機器人

### 項目目標
基於文檔的智能問答系統，能夠回答關於特定領域的問題。

### 技術棧
- LLaMA 3 / Qwen 2.5
- LangChain
- FAISS (向量資料庫)
- Streamlit (UI)

### 實現步驟

#### 1. 文檔處理與向量化

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PDFLoader

# 載入文檔
def load_documents(file_paths):
    documents = []
    for path in file_paths:
        if path.endswith('.pdf'):
            loader = PDFLoader(path)
        else:
            loader = TextLoader(path)
        documents.extend(loader.load())
    return documents

# 分割文檔
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = load_documents(["doc1.pdf", "doc2.txt"])
splits = text_splitter.split_documents(docs)

# 建立向量資料庫
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("./vectorstore")
```

#### 2. 問答鏈

```python
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 載入 LLM
llm_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 建立問答鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# 使用
def answer_question(question):
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }

# 測試
result = answer_question("什麼是 Transformer 架構？")
print(f"回答: {result['answer']}")
print(f"來源: {result['sources']}")
```

#### 3. Streamlit UI

```python
import streamlit as st

st.title("📚 智能文檔問答系統")

# 文件上傳
uploaded_files = st.file_uploader(
    "上傳文檔",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    # 處理上傳的文件
    process_documents(uploaded_files)
    st.success("文檔已處理完成！")

# 問題輸入
question = st.text_input("輸入你的問題：")

if st.button("提問") and question:
    with st.spinner("思考中..."):
        result = answer_question(question)

    st.write("### 回答")
    st.write(result["answer"])

    st.write("### 參考來源")
    for i, source in enumerate(result["sources"], 1):
        with st.expander(f"來源 {i}"):
            st.write(source)
```

---

## 項目 3: 文字摘要應用

### 項目目標
自動生成長文字的摘要，支持新聞、文章、報告等。

### 實現

```python
from transformers import pipeline
import gradio as gr

# 載入摘要模型
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0
)

def summarize_text(text, max_length=150, min_length=50):
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )[0]['summary_text']
    return summary

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 📝 智能文字摘要")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="輸入文字",
                placeholder="粘貼要摘要的長文字...",
                lines=15
            )
            max_length = gr.Slider(50, 300, value=150, label="最大長度")
            min_length = gr.Slider(20, 100, value=50, label="最小長度")
            submit_btn = gr.Button("生成摘要", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="摘要結果", lines=10)

    submit_btn.click(
        fn=summarize_text,
        inputs=[input_text, max_length, min_length],
        outputs=output_text
    )

demo.launch()
```

---

## 項目 4: 多語言翻譯工具

### 項目目標
支持 200+ 語言的實時翻譯工具。

### 實現

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# 載入 NLLB-200 模型
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 語言映射
LANGUAGES = {
    "英語": "eng_Latn",
    "中文": "zho_Hans",
    "日語": "jpn_Jpan",
    "韓語": "kor_Hang",
    "法語": "fra_Latn",
    "德語": "deu_Latn",
    "西班牙語": "spa_Latn",
}

def translate(text, source_lang, target_lang):
    # 設置源語言
    tokenizer.src_lang = LANGUAGES[source_lang]

    # 編碼
    inputs = tokenizer(text, return_tensors="pt")

    # 翻譯
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGES[target_lang]],
        max_length=512
    )

    # 解碼
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result

# Gradio 界面
demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="輸入文字", lines=5),
        gr.Dropdown(choices=list(LANGUAGES.keys()), label="源語言"),
        gr.Dropdown(choices=list(LANGUAGES.keys()), label="目標語言"),
    ],
    outputs=gr.Textbox(label="翻譯結果", lines=5),
    title="🌍 多語言翻譯工具",
    description="支持 200+ 語言的神經機器翻譯",
)

demo.launch()
```

---

## 項目 5: 語音助手

### 項目目標
構建一個完整的語音助手，包含語音識別、理解和合成。

### 實現

```python
import torch
from transformers import pipeline
import gradio as gr

# 1. 語音識別
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=0
)

# 2. 語言理解（使用 LLM）
chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 3. 語音合成
tts = pipeline(
    "text-to-speech",
    model="microsoft/speecht5_tts",
    device=0
)

def voice_assistant(audio):
    # 步驟 1: 語音轉文字
    transcription = asr(audio)["text"]
    print(f"用戶說: {transcription}")

    # 步驟 2: 生成回應
    prompt = f"用戶: {transcription}\n助手:"
    response = chatbot(prompt, max_new_tokens=256)[0]['generated_text']
    response_text = response.split("助手:")[-1].strip()
    print(f"助手回答: {response_text}")

    # 步驟 3: 文字轉語音
    speech = tts(response_text)

    return transcription, response_text, speech

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 AI 語音助手")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(source="microphone", type="filepath", label="說話")
            submit_btn = gr.Button("提交", variant="primary")

        with gr.Column():
            transcription_output = gr.Textbox(label="識別文字")
            response_output = gr.Textbox(label="助手回應")
            audio_output = gr.Audio(label="語音回應")

    submit_btn.click(
        fn=voice_assistant,
        inputs=audio_input,
        outputs=[transcription_output, response_output, audio_output]
    )

demo.launch()
```

---

## 項目擴展建議

### 1. 部署選項

- **本地部署**: Gradio / Streamlit
- **雲端部署**: Hugging Face Spaces / AWS / GCP
- **API 服務**: FastAPI + Docker
- **移動端**: ONNX Runtime + React Native

### 2. 性能優化

- 使用模型量化（GPTQ/AWQ）減少記憶體
- 使用 vLLM 提高推論速度
- 實現批次處理
- 添加快取機制

### 3. 功能增強

- 添加用戶反饋機制
- 實現多輪對話
- 支持多模態輸入
- 添加個性化定制

---

## 學習資源

- [Hugging Face Course](https://huggingface.co/course)
- [LangChain 文檔](https://python.langchain.com/)
- [Gradio 教程](https://gradio.app/docs/)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)

---

## 下一步

恭喜你完成所有學習內容！現在你可以：

1. ✅ 選擇一個項目開始實作
2. ✅ 參與開源項目貢獻
3. ✅ 在 Hugging Face Hub 分享你的模型
4. ✅ 探索最新的研究論文
5. ✅ 構建自己的 AI 產品

繼續學習，持續進步！🚀
