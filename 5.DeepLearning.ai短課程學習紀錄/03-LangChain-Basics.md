# LangChain for LLM Application Development

## 📋 課程概述

這門課程由 Harrison Chase (LangChain 創辦人) 和 Andrew Ng 共同教授，深入介紹 LangChain 框架的核心概念和實際應用。

### 課程目標
- 掌握 LangChain 的核心元件
- 學習如何建構複雜的 LLM 應用
- 理解 Models, Prompts, Chains, Memory, Agents 的使用
- 實作問答系統和文檔分析工具

### 適合對象
- Python 開發者
- 想要快速建構 LLM 應用的工程師
- AI 產品經理和研究人員

### 課程時長
約 1 小時

## 🎯 LangChain 核心概念

LangChain 是一個用於開發由語言模型驅動的應用程式的框架。它提供了一系列模組化的元件，可以組合使用來建立複雜的應用。

### 核心元件架構

```
┌─────────────────────────────────────────┐
│         LangChain 應用架構              │
├─────────────────────────────────────────┤
│  Models (模型)                          │
│  - LLMs                                 │
│  - Chat Models                          │
│  - Text Embedding Models                │
├─────────────────────────────────────────┤
│  Prompts (提示模板)                     │
│  - Prompt Templates                     │
│  - Few-shot Examples                    │
│  - Output Parsers                       │
├─────────────────────────────────────────┤
│  Chains (鏈)                            │
│  - LLM Chain                            │
│  - Sequential Chain                     │
│  - Router Chain                         │
├─────────────────────────────────────────┤
│  Memory (記憶)                          │
│  - Conversation Buffer                  │
│  - Conversation Summary                 │
│  - Entity Memory                        │
├─────────────────────────────────────────┤
│  Agents (代理)                          │
│  - Zero-shot ReAct                      │
│  - Conversational ReAct                 │
│  - Custom Agents                        │
└─────────────────────────────────────────┘
```

## 🔧 環境設定

### 安裝套件

```bash
pip install langchain langchain-openai
pip install python-dotenv
pip install tiktoken  # OpenAI 的 tokenizer
```

### 基本設定

```python
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定 API 金鑰
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
```

## 1️⃣ Models（模型）

### Chat Models

LangChain 支援多種聊天模型，包括 OpenAI、Anthropic、Google 等。

```python
from langchain_openai import ChatOpenAI

# 初始化聊天模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,  # 控制輸出的隨機性
    max_tokens=100     # 限制輸出長度
)

# 簡單呼叫
response = llm.invoke("台灣最高的山是什麼？")
print(response.content)
```

### 使用訊息格式

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="你是一位專業的台灣旅遊導遊。"),
    HumanMessage(content="推薦我三個台北必去的景點。")
]

response = llm.invoke(messages)
print(response.content)
```

### 批次處理

```python
# 批次呼叫多個提示
batch_messages = [
    [HumanMessage(content="什麼是機器學習？")],
    [HumanMessage(content="什麼是深度學習？")],
    [HumanMessage(content="什麼是強化學習？")]
]

responses = llm.batch(batch_messages)
for response in responses:
    print(f"- {response.content}\n")
```

### 串流輸出

```python
# 串流方式接收回應
for chunk in llm.stream("寫一首關於台灣的短詩"):
    print(chunk.content, end="", flush=True)
```

## 2️⃣ Prompt Templates（提示模板）

提示模板讓你可以重複使用提示結構，只需替換變數。

### 基本提示模板

```python
from langchain_core.prompts import PromptTemplate

# 建立提示模板
template = """
你是一位{role}。
請回答以下問題：{question}

請用繁體中文回答，並保持{tone}的語氣。
"""

prompt = PromptTemplate(
    input_variables=["role", "question", "tone"],
    template=template
)

# 使用模板
formatted_prompt = prompt.format(
    role="Python 程式設計專家",
    question="如何使用列表推導式？",
    tone="友善且易懂"
)

print(formatted_prompt)
```

### ChatPromptTemplate

專門用於聊天模型的提示模板。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{expertise}專家。請用繁體中文回答。"),
    ("human", "{user_input}")
])

# 格式化訊息
messages = chat_template.format_messages(
    expertise="資料科學",
    user_input="解釋什麼是交叉驗證"
)

response = llm.invoke(messages)
print(response.content)
```

### Few-Shot 提示模板

提供範例來引導模型輸出。

```python
from langchain_core.prompts import FewShotPromptTemplate

# 定義範例
examples = [
    {
        "question": "2 + 2",
        "answer": "4"
    },
    {
        "question": "5 * 3",
        "answer": "15"
    }
]

# 範例格式
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="問題：{question}\n答案：{answer}"
)

# Few-shot 模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="以下是一些數學問題和答案的範例：",
    suffix="問題：{input}\n答案：",
    input_variables=["input"]
)

print(few_shot_prompt.format(input="7 + 8"))
```

### 輸出解析器（Output Parsers）

將模型輸出轉換為結構化格式。

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. 字串解析器（預設）
string_parser = StrOutputParser()

# 2. JSON 解析器
class Person(BaseModel):
    name: str = Field(description="人物的名字")
    age: int = Field(description="人物的年齡")
    occupation: str = Field(description="人物的職業")

json_parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "從使用者的描述中提取資訊。\n{format_instructions}"),
    ("human", "{description}")
])

# 將格式指令加入提示
prompt = prompt.partial(format_instructions=json_parser.get_format_instructions())

chain = prompt | llm | json_parser

result = chain.invoke({
    "description": "我的朋友小明今年 28 歲，是一位軟體工程師。"
})

print(result)
# 輸出：{'name': '小明', 'age': 28, 'occupation': '軟體工程師'}
```

### 列表解析器

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

list_parser = CommaSeparatedListOutputParser()

format_instructions = list_parser.get_format_instructions()

prompt = PromptTemplate(
    template="列出 {count} 個{topic}。\n{format_instructions}",
    input_variables=["count", "topic"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | llm | list_parser

result = chain.invoke({"count": 5, "topic": "台灣的夜市"})
print(result)
# 輸出：['士林夜市', '饒河街夜市', '逢甲夜市', '六合夜市', '花園夜市']
```

## 3️⃣ Chains（鏈）

Chains 允許你將多個元件串接在一起，建立複雜的工作流程。

### LLMChain（基本鏈）

```python
from langchain.chains import LLMChain

# 使用 LCEL (LangChain Expression Language) 語法
prompt = ChatPromptTemplate.from_template("告訴我一個關於{topic}的有趣事實")

# 建立鏈：使用 | 運算子
chain = prompt | llm | StrOutputParser()

# 執行鏈
result = chain.invoke({"topic": "台灣黑熊"})
print(result)
```

### Sequential Chain（順序鏈）

將多個鏈按順序連接。

```python
from langchain.chains import SequentialChain

# 第一個鏈：生成故事大綱
outline_prompt = ChatPromptTemplate.from_template(
    "為一個關於{topic}的故事寫一個簡短大綱"
)
outline_chain = outline_prompt | llm | StrOutputParser()

# 第二個鏈：根據大綱寫故事
story_prompt = ChatPromptTemplate.from_template(
    "根據以下大綱，寫一個完整的短篇故事：\n{outline}"
)
story_chain = story_prompt | llm | StrOutputParser()

# 組合鏈（使用 RunnablePassthrough）
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    {"outline": outline_chain}
    | story_chain
)

result = full_chain.invoke({"topic": "時空旅行"})
print(result)
```

### 更複雜的順序鏈範例

```python
from operator import itemgetter

# 鏈 1：翻譯成英文
translate_prompt = ChatPromptTemplate.from_template(
    "將以下中文翻譯成英文：{chinese_text}"
)

# 鏈 2：總結內容
summarize_prompt = ChatPromptTemplate.from_template(
    "用一句話總結：{english_text}"
)

# 鏈 3：分析情感
sentiment_prompt = ChatPromptTemplate.from_template(
    "分析以下文字的情感（正面/負面/中性）：{summary}"
)

# 組合鏈
multi_chain = (
    {"english_text": translate_prompt | llm | StrOutputParser()}
    | RunnablePassthrough.assign(
        summary=itemgetter("english_text") | summarize_prompt | llm | StrOutputParser()
    )
    | {"sentiment": sentiment_prompt | llm | StrOutputParser()}
)

result = multi_chain.invoke({
    "chinese_text": "今天天氣很好，我很開心去公園散步。"
})
print(result)
```

### Router Chain（路由鏈）

根據輸入動態選擇不同的處理鏈。

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import LLMChain

# 定義不同領域的提示
physics_template = """
你是一位物理學家。
請回答以下物理問題：{input}
"""

math_template = """
你是一位數學家。
請回答以下數學問題：{input}
"""

history_template = """
你是一位歷史學家。
請回答以下歷史問題：{input}
"""

# 創建提示資訊
prompt_infos = [
    {
        "name": "physics",
        "description": "適合回答物理相關問題",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "適合回答數學相關問題",
        "prompt_template": math_template
    },
    {
        "name": "history",
        "description": "適合回答歷史相關問題",
        "prompt_template": history_template
    }
]

# 注意：MultiPromptChain 在新版本中可能需要不同的實作方式
# 這裡展示概念，實際使用時可能需要調整
```

### Transform Chain（轉換鏈）

對輸入進行預處理或轉換。

```python
from langchain_core.runnables import RunnableLambda

def preprocess_text(text):
    """文字預處理函數"""
    # 移除多餘空白、轉小寫等
    return text.strip().lower()

def postprocess_response(response):
    """回應後處理函數"""
    # 格式化輸出
    return f"📝 {response}"

# 建立包含預處理和後處理的鏈
preprocess = RunnableLambda(preprocess_text)
postprocess = RunnableLambda(postprocess_response)

full_chain = (
    preprocess
    | {"input": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("解釋：{input}")
    | llm
    | StrOutputParser()
    | postprocess
)

result = full_chain.invoke("   MACHINE LEARNING   ")
print(result)
```

## 4️⃣ Memory（記憶）

Memory 元件讓你的應用能夠記住之前的對話。

### ConversationBufferMemory（緩衝記憶）

儲存完整的對話歷史。

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 初始化記憶
memory = ConversationBufferMemory()

# 建立對話鏈
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 顯示詳細資訊
)

# 進行多輪對話
print(conversation.predict(input="你好，我叫小明"))
print(conversation.predict(input="我喜歡登山"))
print(conversation.predict(input="你還記得我的名字嗎？"))

# 查看記憶內容
print("\n對話歷史：")
print(memory.buffer)
```

### ConversationBufferWindowMemory（視窗記憶）

只保留最近 k 輪對話。

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 2 輪對話
window_memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm,
    memory=window_memory,
    verbose=True
)

# 進行對話
conversation.predict(input="嗨，我是王大明")
conversation.predict(input="我是工程師")
conversation.predict(input="我在台北工作")
conversation.predict(input="你知道我的職業嗎？")  # 記得
conversation.predict(input="你知道我的名字嗎？")  # 可能不記得（超出視窗）
```

### ConversationSummaryMemory（摘要記憶）

將對話歷史摘要後儲存，節省 token。

```python
from langchain.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)

conversation = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

conversation.predict(input="我剛從日本旅遊回來，去了東京、京都和大阪。")
conversation.predict(input="我最喜歡京都的金閣寺和清水寺。")
conversation.predict(input="我買了很多伴手禮，包括白色戀人、薯條三兄弟。")

# 查看摘要
print("\n對話摘要：")
print(summary_memory.buffer)
```

### ConversationSummaryBufferMemory（混合記憶）

結合完整記憶和摘要記憶。

```python
from langchain.memory import ConversationSummaryBufferMemory

hybrid_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,  # 超過此 token 數就摘要
    return_messages=True
)

conversation = ConversationChain(
    llm=llm,
    memory=hybrid_memory,
    verbose=True
)
```

### 在 LCEL 中使用 Memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# 儲存每個 session 的對話歷史
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 建立帶有記憶的鏈
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友善的助理。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# 加入記憶功能
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用（需要提供 session_id）
config = {"configurable": {"session_id": "user_123"}}

response1 = chain_with_history.invoke(
    {"input": "嗨，我叫 Alice"},
    config=config
)
print(response1.content)

response2 = chain_with_history.invoke(
    {"input": "你記得我的名字嗎？"},
    config=config
)
print(response2.content)
```

## 5️⃣ Agents（代理）

Agents 可以根據使用者輸入動態決定使用哪些工具。

### 建立基本工具

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain import hub

# 定義工具函數
def get_current_weather(location: str) -> str:
    """取得指定地點的天氣資訊"""
    # 實際應用中應該呼叫天氣 API
    weather_data = {
        "台北": "晴天，氣溫 25°C",
        "台中": "多雲，氣溫 27°C",
        "高雄": "晴天，氣溫 29°C"
    }
    return weather_data.get(location, "查無此地點的天氣資訊")

def calculate(expression: str) -> str:
    """計算數學表達式"""
    try:
        result = eval(expression)
        return f"計算結果：{result}"
    except:
        return "計算錯誤"

def search_taiwan_info(query: str) -> str:
    """搜尋台灣相關資訊"""
    # 模擬搜尋結果
    return f"關於「{query}」的資訊：台灣是一個美麗的島嶼..."

# 建立工具列表
tools = [
    Tool(
        name="天氣查詢",
        func=get_current_weather,
        description="查詢台灣各城市的天氣。輸入：城市名稱（例如：台北）"
    ),
    Tool(
        name="計算機",
        func=calculate,
        description="執行數學計算。輸入：數學表達式（例如：2+2*3）"
    ),
    Tool(
        name="台灣資訊搜尋",
        func=search_taiwan_info,
        description="搜尋台灣相關的資訊。輸入：搜尋關鍵字"
    )
]
```

### 建立 ReAct Agent

```python
# 使用 LangChain Hub 的 ReAct 提示模板
react_prompt = hub.pull("hwchase17/react")

# 建立 agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# 建立 agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

# 執行 agent
result = agent_executor.invoke({
    "input": "台北的天氣如何？另外，幫我計算 15 * 8 + 20"
})

print(result["output"])
```

### 自訂 Agent 提示

```python
agent_prompt = PromptTemplate.from_template("""
你是一位智慧助理，可以使用以下工具：

{tools}

工具名稱：{tool_names}

請使用以下格式回答：

Question: 你需要回答的問題
Thought: 你應該思考要做什麼
Action: 要執行的動作，應該是 [{tool_names}] 中的一個
Action Input: 動作的輸入
Observation: 動作的結果
... (這個 Thought/Action/Action Input/Observation 可以重複 N 次)
Thought: 我現在知道最終答案了
Final Answer: 原始問題的最終答案

開始！

Question: {input}
Thought: {agent_scratchpad}
""")
```

### 帶有記憶的 Agent

```python
from langchain.memory import ConversationBufferMemory

# 建立記憶（需要返回訊息）
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 建立對話式 agent
from langchain.agents import AgentType, initialize_agent

conversational_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 多輪對話
print(conversational_agent.invoke({"input": "台北今天天氣如何？"}))
print(conversational_agent.invoke({"input": "那台中呢？"}))
print(conversational_agent.invoke({"input": "你剛才查了哪些城市的天氣？"}))
```

## 💡 實戰專案：文檔問答系統

### 專案架構

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

class DocumentQASystem:
    def __init__(self, document_path):
        """初始化文檔問答系統"""
        self.document_path = document_path
        self.vectorstore = None
        self.qa_chain = None

        # 載入並處理文檔
        self.load_and_process_documents()

        # 建立問答鏈
        self.create_qa_chain()

    def load_and_process_documents(self):
        """載入並處理文檔"""
        # 1. 載入文檔
        loader = TextLoader(self.document_path, encoding='utf-8')
        documents = loader.load()

        # 2. 分割文檔
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # 3. 建立向量儲存
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="doc_qa"
        )

        print(f"✅ 已載入並處理 {len(splits)} 個文檔片段")

    def create_qa_chain(self):
        """建立問答鏈"""
        # 建立檢索器
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # 返回最相關的 3 個片段
        )

        # 建立問答鏈
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # 將所有相關片段放入單一提示
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

    def ask(self, question):
        """提問"""
        if not self.qa_chain:
            return "系統尚未初始化"

        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

# 使用範例
# qa_system = DocumentQASystem("my_document.txt")
# result = qa_system.ask("這份文檔的主要內容是什麼？")
# print(result["answer"])
```

### 進階：自訂問答提示

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 自訂提示模板
qa_template = """
你是一位專業的文檔分析助理。請根據以下上下文資訊回答問題。

規則：
1. 只根據提供的上下文回答
2. 如果上下文中沒有相關資訊，請說「根據提供的資訊無法回答此問題」
3. 使用繁體中文回答
4. 引用來源時請標註

上下文：
{context}

問題：{question}

詳細回答：
"""

QA_PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"]
)

# 使用自訂提示建立鏈
custom_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)
```

## 📊 實用工具與技巧

### 1. Callbacks（回調）

監控鏈的執行過程。

```python
from langchain.callbacks import StdOutCallbackHandler

# 使用回調顯示詳細資訊
handler = StdOutCallbackHandler()

chain = prompt | llm | StrOutputParser()
result = chain.invoke(
    {"topic": "人工智慧"},
    config={"callbacks": [handler]}
)
```

### 2. 快取機制

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 設定快取
set_llm_cache(InMemoryCache())

# 第一次呼叫（較慢）
result1 = llm.invoke("什麼是機器學習？")

# 第二次呼叫相同問題（從快取取得，很快）
result2 = llm.invoke("什麼是機器學習？")
```

### 3. Token 計數

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = llm.invoke("解釋量子計算")
    print(f"總 tokens: {cb.total_tokens}")
    print(f"提示 tokens: {cb.prompt_tokens}")
    print(f"完成 tokens: {cb.completion_tokens}")
    print(f"總成本: ${cb.total_cost:.4f}")
```

### 4. 錯誤處理

```python
from langchain_core.runnables import RunnableLambda

def safe_invoke(chain, input_data, default_response="發生錯誤"):
    """安全執行鏈，捕捉錯誤"""
    try:
        return chain.invoke(input_data)
    except Exception as e:
        print(f"錯誤：{e}")
        return default_response

# 使用
result = safe_invoke(chain, {"topic": "AI"})
```

## ✅ 最佳實踐總結

### 1. 選擇合適的元件
- **簡單任務**：使用基本的 LLMChain
- **多步驟任務**：使用 SequentialChain
- **需要記憶**：使用 Memory 元件
- **動態決策**：使用 Agents

### 2. Prompt Engineering
- 使用 PromptTemplate 重用提示
- 提供清晰的指令和範例
- 使用輸出解析器獲得結構化資料

### 3. 效能優化
- 使用快取減少 API 呼叫
- 監控 token 使用量
- 批次處理多個請求

### 4. 生產環境考量
- 實作錯誤處理
- 添加日誌記錄
- 使用回調監控
- 定期評估輸出品質

## 📚 延伸學習

- **LangChain Chat with Data**：學習文檔處理和 RAG
- **LangChain Agents**：深入了解 Function Calling 和工具使用
- **LangGraph**：建構複雜的狀態機工作流程
- **LangSmith**：監控和除錯 LangChain 應用

---

**課程連結**：[DeepLearning.ai - LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)

**完成日期**：2025-01-17
