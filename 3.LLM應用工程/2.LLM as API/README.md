# LLM 作為 API 與應用程式整合

## 目錄
1. [建立 API 接口](#21-建立-api-接口)
2. [與前端整合](#22-與前端整合)
3. [與自動化工具整合](#23-與自動化工具整合)
4. [實作示例](#24-實作示例)
5. [生產環境最佳實踐](#25-生產環境最佳實踐)
6. [錯誤處理與重試策略](#26-錯誤處理與重試策略)

---

## 2.1 建立 API 接口

### OpenAI API

**特點**：
- RESTful API 設計
- 支援多種模型（GPT-4o, GPT-4o-mini, O1 系列）
- 串流與非串流模式
- 完整的錯誤處理
- Token 使用計量

**基礎使用**：
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 基本對話
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一個專業的技術助理。"},
        {"role": "user", "content": "解釋什麼是 API"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
print(f"使用 tokens: {response.usage.total_tokens}")
```

**進階功能**：

#### 1. 函數呼叫（Function Calling）
```python
functions = [
    {
        "name": "get_weather",
        "description": "獲取指定城市的天氣資訊",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名稱，例如：台北"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "溫度單位"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "台北現在天氣如何？"}],
    functions=functions,
    function_call="auto"
)

# 檢查是否需要呼叫函數
if response.choices[0].message.function_call:
    function_name = response.choices[0].message.function_call.name
    function_args = json.loads(response.choices[0].message.function_call.arguments)

    # 執行函數
    weather_data = get_weather(**function_args)

    # 將結果返回給模型
    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "台北現在天氣如何？"},
            response.choices[0].message,
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(weather_data)
            }
        ]
    )

    print(second_response.choices[0].message.content)
```

#### 2. 串流回應
```python
def stream_response():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "介紹台灣的夜市文化"}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
```

#### 3. 結構化輸出（JSON 模式）
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "列出 5 個台灣著名景點，包含名稱、位置和簡介"
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.3
)

import json
result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

### Hugging Face Inference Endpoints

**特點**：
- 支援所有 Hugging Face 模型
- 自動擴展
- 按使用量計費
- 簡單的 API 介面

**使用 Inference API（免費層級）**：
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "解釋什麼是機器學習",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95
    }
})

print(output[0]['generated_text'])
```

**使用 Inference Endpoints（付費，專用端點）**：
```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="https://your-endpoint.aws.endpoints.huggingface.cloud",
    token=HF_API_TOKEN
)

response = client.text_generation(
    "寫一個 Python 函式計算階乘",
    max_new_tokens=200,
    temperature=0.5
)

print(response)
```

---

### Anthropic Claude API

**特點**：
- 200K token 上下文窗口
- 優秀的程式碼生成能力
- Constitutional AI 安全機制
- 支援多輪對話

**基礎使用**：
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "解釋 Rust 的所有權系統"}
    ]
)

print(message.content[0].text)
```

**串流回應**：
```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "寫一個排序演算法"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## 2.2 與前端整合

### Streamlit 整合

**特點**：
- 純 Python 程式碼
- 快速原型開發
- 內建元件豐富
- 自動重載

**完整聊天機器人範例**：
```python
# app.py
import streamlit as st
from openai import OpenAI

# 頁面配置
st.set_page_config(
    page_title="AI 聊天助理",
    page_icon="🤖",
    layout="wide"
)

# 標題
st.title("🤖 AI 聊天助理")

# 初始化 OpenAI 客戶端
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = get_openai_client()

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示對話歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 使用者輸入
if prompt := st.chat_input("輸入你的問題..."):
    # 添加使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回應
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 串流回應
        for response in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            if response.choices[0].delta.content:
                full_response += response.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 側邊欄設定
with st.sidebar:
    st.header("設定")

    if st.button("清除對話"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("對話統計")
    st.metric("訊息數量", len(st.session_state.messages))
```

**部署 Streamlit 應用**：
```bash
# 本地運行
streamlit run app.py

# 部署到 Streamlit Cloud
# 1. 推送程式碼到 GitHub
# 2. 連接到 streamlit.io
# 3. 選擇 repository 和 branch
# 4. 設定 secrets (OPENAI_API_KEY)
```

---

### Gradio 整合

**特點**：
- 自動生成介面
- 支援多種輸入類型
- 可嵌入網頁
- 分享功能

**聊天介面範例**：
```python
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def chat(message, history):
    """處理對話"""

    # 構建訊息歷史
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    # 生成回應
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content

# 建立介面
demo = gr.ChatInterface(
    chat,
    title="AI 聊天機器人",
    description="與 GPT-4o-mini 對話",
    theme=gr.themes.Soft(),
    examples=[
        "解釋什麼是量子計算",
        "寫一個 Python 排序函數",
        "介紹台灣的歷史"
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True 生成公開連結
```

**RAG 應用範例**：
```python
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 初始化
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

def answer_question(question):
    """回答問題並顯示來源"""
    result = qa_chain({"query": question})

    answer = result['result']
    sources = "\n\n**參考來源：**\n"
    for i, doc in enumerate(result['source_documents'], 1):
        sources += f"{i}. {doc.metadata.get('source', 'Unknown')}\n"
        sources += f"   {doc.page_content[:200]}...\n\n"

    return answer + sources

# 建立介面
interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="輸入你的問題..."),
    outputs=gr.Markdown(),
    title="RAG 問答系統",
    description="基於知識庫回答問題",
    examples=[
        "什麼是 Transformer 架構？",
        "RAG 的優勢是什麼？"
    ]
)

interface.launch()
```

---

### React + FastAPI 整合

**後端（FastAPI）**：
```python
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List

app = FastAPI()

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 開發伺服器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key="your-api-key")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

class ChatResponse(BaseModel):
    message: str
    tokens_used: int

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天端點"""
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[msg.dict() for msg in request.messages],
            temperature=request.temperature
        )

        return ChatResponse(
            message=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 運行：uvicorn main:app --reload
```

**前端（React）**：
```jsx
// frontend/src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        messages: updatedMessages
      });

      setMessages([
        ...updatedMessages,
        { role: 'assistant', content: response.data.message }
      ]);
    } catch (error) {
      console.error('Error:', error);
      alert('發生錯誤，請稍後再試');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <strong>{msg.role === 'user' ? '你' : 'AI'}：</strong>
              {msg.content}
            </div>
          ))}
        </div>

        <div className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="輸入訊息..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading}>
            {loading ? '傳送中...' : '傳送'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
```

---

## 2.3 與自動化工具整合

### 與 RPA（UiPath）整合

**概念**：使用 LLM 增強 RPA 流程，實現智能自動化。

**範例：自動處理電子郵件**：
```python
# email_processor.py
from openai import OpenAI
import imaplib
import email
from email.header import decode_header

client = OpenAI(api_key="your-api-key")

def process_email(email_content):
    """使用 LLM 分析並分類郵件"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """你是一個郵件分類助理。分析郵件內容並：
                1. 分類（工作、個人、垃圾郵件、重要）
                2. 提取關鍵資訊
                3. 建議回覆行動

                以 JSON 格式回應。"""
            },
            {
                "role": "user",
                "content": f"郵件內容：\n{email_content}"
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def fetch_and_process_emails():
    """連接 IMAP 並處理郵件"""

    # 連接郵件伺服器
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login("your-email@gmail.com", "your-password")
    mail.select("inbox")

    # 搜尋未讀郵件
    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    for email_id in email_ids[:10]:  # 處理前 10 封
        # 取得郵件
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        # 提取內容
        subject = decode_header(msg["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()

        # 使用 LLM 分析
        analysis = process_email(f"主旨：{subject}\n\n內容：{body}")

        print(f"郵件分析：{analysis}")

        # 根據分析結果執行動作（標記、轉發、回覆等）
        if analysis.get('category') == '重要':
            # 標記為重要
            mail.store(email_id, '+FLAGS', '\\Flagged')

if __name__ == "__main__":
    fetch_and_process_emails()
```

---

### 與 Zapier 整合

**使用 Webhooks**：
```python
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key="your-api-key")

@app.route('/webhook', methods=['POST'])
def zapier_webhook():
    """接收 Zapier webhook 請求"""

    data = request.json
    user_input = data.get('input_text', '')

    # 使用 LLM 處理
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一個內容摘要助理。"},
            {"role": "user", "content": f"請摘要以下內容：\n{user_input}"}
        ],
        max_tokens=200
    )

    summary = response.choices[0].message.content

    # 返回給 Zapier
    return jsonify({
        "summary": summary,
        "original_length": len(user_input),
        "summary_length": len(summary)
    })

if __name__ == '__main__':
    app.run(port=5000)
```

**Zapier 設定流程**：
1. 觸發器：新的 Google Docs 文件
2. 動作：Webhooks POST 到上述端點
3. 動作：將摘要儲存到 Notion

---

## 2.4 實作示例

### 示例 1：Flask + OpenAI API 建立 RESTful API

```python
# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)  # 允許跨域請求

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天端點"""
    try:
        data = request.json
        messages = data.get('messages', [])
        model = data.get('model', 'gpt-4o-mini')
        temperature = data.get('temperature', 0.7)

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return jsonify({
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """文字摘要端點"""
    try:
        data = request.json
        text = data.get('text', '')
        max_length = data.get('max_length', 200)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"你是一個專業的摘要助理。請將以下文字摘要為不超過 {max_length} 字的內容。"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )

        return jsonify({
            "summary": response.choices[0].message.content,
            "original_length": len(text),
            "tokens_used": response.usage.total_tokens
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """翻譯端點"""
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', '英文')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"你是一個專業的翻譯助理。請將以下文字翻譯成{target_lang}。"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )

        return jsonify({
            "translation": response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康檢查端點"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**測試 API**：
```bash
# 聊天
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'

# 摘要
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "這是一段很長的文字..."
  }'

# 翻譯
curl -X POST http://localhost:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "target_lang": "中文"
  }'
```

---

### 示例 2：完整的 Responses API 實作

```python
"""
使用 OpenAI Responses API 的完整範例
適合結構化輸出和高級應用
"""

import os
from openai import OpenAI
from typing import List, Dict
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def structured_rag_response(query: str) -> Dict:
    """使用 Responses API 進行結構化 RAG 回應"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""你是一個專業的 RAG 系統助理。
        分析使用者查詢並提供結構化回應，包括：
        1. 理解的查詢意圖
        2. 建議的檢索策略
        3. 預期的回答類型
        """,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": query}
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rag_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string"},
                        "retrieval_strategy": {"type": "string"},
                        "expected_answer_type": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["intent", "retrieval_strategy", "expected_answer_type", "keywords"]
                }
            }
        }
    )

    return json.loads(response.output_text)

def code_generation_response(description: str) -> str:
    """生成程式碼"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""你是一個 Python 程式碼生成專家。
        生成乾淨、有註解、可執行的 Python 程式碼。
        包含錯誤處理和類型提示。""",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"需求：{description}"}
                ]
            }
        ]
    )

    return response.output_text

# 測試
if __name__ == "__main__":
    # RAG 查詢分析
    rag_result = structured_rag_response("解釋 Transformer 的注意力機制")
    print("RAG 分析結果：")
    print(json.dumps(rag_result, indent=2, ensure_ascii=False))

    # 程式碼生成
    code = code_generation_response("寫一個函數計算兩個日期之間的天數差")
    print("\n生成的程式碼：")
    print(code)
```

---

## 2.5 生產環境最佳實踐

### 1. API 金鑰管理

```python
# 使用環境變數
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("請設定 OPENAI_API_KEY 環境變數")

# 使用 Secrets 管理（雲端部署）
# AWS Secrets Manager 範例
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        return json.loads(get_secret_value_response['SecretString'])
    except ClientError as e:
        raise e

secrets = get_secret("prod/openai/api-key")
OPENAI_API_KEY = secrets['api_key']
```

---

### 2. 速率限制與配額管理

```python
from ratelimit import limits, sleep_and_retry
import time

# 每分鐘最多 60 次請求
CALLS_PER_MINUTE = 60

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def call_openai_api(messages):
    """帶速率限制的 API 呼叫"""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

# 使用 Token bucket 更精細的控制
class TokenBucket:
    def __init__(self, tokens_per_second):
        self.capacity = tokens_per_second
        self.tokens = tokens_per_second
        self.last_update = time.time()

    def consume(self, tokens):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.capacity)
        self.last_update = now

        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens):
        if tokens <= self.tokens:
            return 0
        return (tokens - self.tokens) / self.capacity

bucket = TokenBucket(tokens_per_second=10)

def rate_limited_call(messages):
    # 估算 tokens
    estimated_tokens = sum(len(m['content']) // 4 for m in messages)

    wait = bucket.wait_time(estimated_tokens)
    if wait > 0:
        time.sleep(wait)

    bucket.consume(estimated_tokens)
    return client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

---

### 3. 快取策略

```python
from functools import lru_cache
import hashlib
import redis
import json

# 本地快取（LRU）
@lru_cache(maxsize=1000)
def cached_completion(prompt: str, model: str = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Redis 快取
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_cached_response(prompt: str, model: str = "gpt-4o-mini", ttl: int = 3600):
    """使用 Redis 快取回應"""

    # 生成快取鍵
    cache_key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()

    # 檢查快取
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 呼叫 API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    result = {
        "response": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    }

    # 儲存到快取
    redis_client.setex(cache_key, ttl, json.dumps(result))

    return result
```

---

## 2.6 錯誤處理與重試策略

### 完整的錯誤處理

```python
from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

client = OpenAI(api_key="your-api-key")

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def robust_completion(messages, model="gpt-4o-mini", **kwargs):
    """帶重試機制的 API 呼叫"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response

    except RateLimitError as e:
        print(f"速率限制錯誤：{e}")
        raise  # 重試

    except APITimeoutError as e:
        print(f"請求超時：{e}")
        raise  # 重試

    except OpenAIError as e:
        print(f"OpenAI 錯誤：{e}")
        # 不重試其他錯誤
        raise

# 使用範例
try:
    response = robust_completion(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"最終失敗：{e}")
```

---

## 參考資源

### 官方文件
- [OpenAI API 文件](https://platform.openai.com/docs)
- [Anthropic API 文件](https://docs.anthropic.com/)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/)
- [Streamlit 文件](https://docs.streamlit.io/)
- [Gradio 文件](https://gradio.app/docs/)

### 範例程式碼
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)

---

**最後更新**：2025年1月
