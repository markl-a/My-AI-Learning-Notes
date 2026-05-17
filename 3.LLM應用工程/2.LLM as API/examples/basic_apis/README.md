# 基礎 LLM API 使用示例

> **⚠️ 教學示範 — `01_openai_basic.py` 的 calculator tool 用 AST `safe_eval`**
>
> `01_openai_basic.py:152-175` 自實作 `safe_eval`(AST + 白名單運算符)當 OpenAI function calling 的 calculator tool 示範。雖比 `eval()` 安全,production 請改用 `simpleeval`/`numexpr` 套件,並對所有 tool call 結果加 schema validate。

本目錄包含多個主流 LLM API 的完整使用示例，所有程式碼都經過測試並可直接運行。

## 📁 文件說明

- `01_openai_basic.py` - OpenAI API 完整示例
- `02_anthropic_basic.py` - Anthropic Claude API 完整示例
- `03_gemini_basic.py` - Google Gemini API 完整示例
- `04_api_comparison.py` - API 性能與成本比較工具

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd "3.LLM應用工程/2.LLM as API"
pip install -r requirements.txt
```

### 2. 設定環境變數

複製 `.env.example` 並重命名為 `.env`：

```bash
cp .env.example .env
```

編輯 `.env` 文件，填入你的 API keys：

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. 運行示例

```bash
# OpenAI 示例
python examples/basic_apis/01_openai_basic.py

# Anthropic 示例
python examples/basic_apis/02_anthropic_basic.py

# Gemini 示例
python examples/basic_apis/03_gemini_basic.py

# API 比較
python examples/basic_apis/04_api_comparison.py
```

## 📚 功能覆蓋

### OpenAI API (`01_openai_basic.py`)

✅ 基本對話
✅ 串流回應
✅ 函數呼叫（Function Calling）
✅ 結構化輸出（JSON 模式）
✅ 多輪對話
✅ 視覺理解（GPT-4 Vision）

**使用場景：**
- 一般對話應用
- 需要函數呼叫的智能助理
- 需要結構化資料輸出
- 多模態應用（文字+圖像）

### Anthropic Claude API (`02_anthropic_basic.py`)

✅ 基本對話
✅ 串流回應
✅ 系統提示（System Prompt）
✅ 多輪對話
✅ 視覺理解
✅ 長文字處理（200K tokens）
✅ 結構化輸出
✅ 批次處理

**使用場景：**
- 程式碼生成和審查
- 長文檔分析和摘要
- 需要高品質推理的任務
- 多輪複雜對話

### Google Gemini API (`03_gemini_basic.py`)

✅ 基本對話
✅ 串流回應
✅ 對話會話（Chat Session）
✅ 視覺理解
✅ 多圖片分析
✅ 安全設定
✅ 系統指令
✅ JSON 模式
✅ Token 計數

**使用場景：**
- 多模態應用（文字+圖像+影片）
- 免費層級使用
- 需要快速回應
- 需要長上下文（最高 2M tokens）

### API 比較工具 (`04_api_comparison.py`)

✅ 性能比較（延遲測試）
✅ Token 使用分析
✅ 成本估算
✅ 批次測試
✅ 場景測試（程式碼生成、摘要、翻譯等）

## 💡 程式碼示例

### OpenAI 基本對話

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一個專業的助理"},
        {"role": "user", "content": "解釋什麼是 API"}
    ]
)

print(response.choices[0].message.content)
```

### Anthropic Claude 串流回應

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "寫一個排序演算法"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Gemini 對話會話

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-1.5-pro')

chat = model.start_chat(history=[])
response = chat.send_message("你好！")
print(response.text)
```

## 📊 API 比較

| 特性 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 上下文長度 | 128K | 200K | 2M |
| 多模態 | ✅ (文字+圖像) | ✅ (文字+圖像) | ✅ (文字+圖像+影片) |
| 串流支援 | ✅ | ✅ | ✅ |
| 函數呼叫 | ✅ | ⚠️ (有限) | ✅ |
| JSON 模式 | ✅ | ⚠️ (需提示) | ✅ |
| 免費層級 | ❌ | ❌ | ✅ (有限) |
| 價格/1M tokens | $0.15-0.60 | $3-15 | $3.50-10.50 |

## 🎯 最佳實踐

### 1. API Key 安全

```python
# ❌ 不要這樣做
api_key = "sk-..."

# ✅ 使用環境變數
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### 2. 錯誤處理

```python
from openai import OpenAI, OpenAIError

try:
    response = client.chat.completions.create(...)
except OpenAIError as e:
    print(f"API 錯誤: {e}")
    # 處理錯誤
```

### 3. 成本優化

- 使用較小的模型（如 gpt-4o-mini）進行測試
- 設定 `max_tokens` 限制輸出長度
- 使用快取減少重複請求
- 批次處理多個請求

### 4. 性能優化

- 使用串流回應改善用戶體驗
- 實作重試機制處理臨時錯誤
- 使用非同步請求處理並發
- 監控 token 使用和成本

## 🐛 常見問題

### 問題：API Key 無效

**解決方案：**
1. 確認 `.env` 文件在正確位置
2. 檢查 API Key 是否有效
3. 確認 API Key 有足夠的配額

### 問題：Rate Limit 錯誤

**解決方案：**
- 實作指數退避重試
- 使用速率限制器
- 升級 API 計劃

### 問題：回應品質不佳

**解決方案：**
- 調整 temperature（0.0-2.0）
- 使用更好的提示詞
- 嘗試不同的模型
- 添加系統提示

## 📖 延伸閱讀

- [OpenAI API 文件](https://platform.openai.com/docs)
- [Anthropic API 文件](https://docs.anthropic.com/)
- [Google Gemini API 文件](https://ai.google.dev/docs)

## 🤝 貢獻

如果你發現任何問題或有改進建議，歡迎提出 Issue 或 Pull Request！

## 📝 授權

MIT License

---

**最後更新：** 2025年1月
