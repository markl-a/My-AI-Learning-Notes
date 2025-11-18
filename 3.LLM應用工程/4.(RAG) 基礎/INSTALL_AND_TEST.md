# RAG 基礎範例 - 安裝和測試指南

## 快速開始

### 1. 安裝依賴

所有範例程式需要以下依賴：

```bash
# 基礎依賴（必需）
pip install sentence-transformers scikit-learn numpy

# 這可能需要 5-10 分鐘，取決於網速和系統性能
```

**完整安裝（包含所有可選依賴）**：

```bash
pip install -r requirements.txt
```

### 2. 驗證安裝

安裝完成後，運行以下命令驗證：

```bash
# 驗證基礎庫
python -c "import sentence_transformers; print('✓ Sentence Transformers')"
python -c "import sklearn; print('✓ Scikit-learn')"
python -c "import numpy; print('✓ NumPy')"
```

### 3. 語法檢查

在安裝依賴之前，可以先驗證代碼語法：

```bash
# 檢查所有 Python 文件的語法
for file in *.py; do
    python -m py_compile "$file" && echo "✓ $file" || echo "✗ $file"
done
```

### 4. 運行範例

#### 方法一：運行單個範例

```bash
# 範例 1: 基礎嵌入向量
python 1_basic_embeddings.py

# 範例 2: 文檔處理與拆分
python 2_document_processing.py

# 範例 3: 向量資料庫
python 3_vector_databases.py

# 範例 4: 完整 RAG 系統
python 4_complete_rag_system.py

# 範例 5: 進階 RAG 技術
python 5_advanced_rag_techniques.py

# 範例 6: 實戰問答系統
python 6_practical_qa_system.py
```

#### 方法二：運行所有範例

```bash
chmod +x run_all_examples.sh
./run_all_examples.sh
```

## 範例說明

### 1. 基礎嵌入向量 (`1_basic_embeddings.py`)

**運行時間**: ~30-60 秒（首次運行需要下載模型）

**輸出內容**:
- 基礎嵌入向量演示
- 多語言嵌入向量演示
- 語義搜索演示

**首次運行注意**:
- 會自動下載 `all-MiniLM-L6-v2` 模型（約 80MB）
- 會自動下載 `paraphrase-multilingual-MiniLM-L12-v2` 模型（約 400MB）
- 下載位置: `~/.cache/huggingface/`

### 2. 文檔處理與拆分 (`2_document_processing.py`)

**運行時間**: ~5-10 秒

**輸出內容**:
- 基礎文本拆分演示
- 遞歸文本拆分演示
- 文檔載入演示
- 元數據保留演示

**會創建的文件/目錄**:
- `test_data/` - 測試數據目錄

### 3. 向量資料庫 (`3_vector_databases.py`)

**運行時間**: ~40-90 秒

**輸出內容**:
- 簡單向量資料庫演示
- FAISS 向量資料庫演示（如果已安裝）
- 性能比較

**會創建的文件/目錄**:
- `test_data/simple_db.pkl` - 持久化的向量資料庫

**可選優化**:
```bash
# 安裝 FAISS 以獲得更好的性能
pip install faiss-cpu
```

### 4. 完整 RAG 系統 (`4_complete_rag_system.py`)

**運行時間**: ~60-120 秒

**輸出內容**:
- 基礎 RAG 系統演示
- 自定義知識庫演示
- 問答示例

**注意**:
- 使用模擬的 LLM 回答（SimpleLLM）
- 要使用真實 LLM，需要設置 API 密鑰

### 5. 進階 RAG 技術 (`5_advanced_rag_techniques.py`)

**運行時間**: ~90-180 秒

**輸出內容**:
- BM25 稀疏檢索演示
- 混合檢索演示
- 重排序演示
- 查詢擴展演示
- 完整的進階 RAG 管道

**可選依賴**:
```bash
# 安裝 Cross-Encoder 以獲得更好的重排序效果
pip install sentence-transformers
```

### 6. 實戰問答系統 (`6_practical_qa_system.py`)

**運行時間**: ~60-120 秒

**輸出內容**:
- 多文檔問答系統完整演示
- 自動創建測試知識庫
- 多輪問答示例

**會創建的文件/目錄**:
- `test_data/knowledge_base/` - 測試知識庫目錄
  - `ml_basics.txt`
  - `dl_frameworks.txt`
  - `nlp_guide.txt`

## 常見問題

### Q1: 安裝 sentence-transformers 太慢怎麼辦？

**A**: Sentence-transformers 依賴較多，安裝可能需要 5-10 分鐘。建議：

```bash
# 使用清華鏡像加速
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers scikit-learn numpy
```

### Q2: 運行時提示 "ModuleNotFoundError"？

**A**: 確保已安裝所有必需的依賴：

```bash
pip install sentence-transformers scikit-learn numpy
```

### Q3: 首次運行為什麼這麼慢？

**A**: 首次運行需要下載預訓練模型，後續運行會使用緩存的模型，速度會快很多。

### Q4: 如何使用真實的 LLM API？

**A**: 在範例 4 和 6 中，可以配置環境變量使用真實 API：

```bash
# 設置 OpenAI API 密鑰
export OPENAI_API_KEY="your-api-key"

# 或在代碼中修改
llm = OpenAILLM(api_key="your-api-key", model="gpt-4")
```

### Q5: 如何清理生成的文件？

**A**: 運行以下命令清理測試數據：

```bash
rm -rf test_data/
rm -rf __pycache__/
```

## 性能優化建議

### 1. 安裝 FAISS

FAISS 是 Facebook 開發的高效向量搜索庫：

```bash
pip install faiss-cpu
# 或 GPU 版本
pip install faiss-gpu
```

### 2. 使用更小的模型

如果硬件資源有限，可以使用更小的嵌入模型：

- `all-MiniLM-L6-v2` (80MB) - 已在範例中使用
- `paraphrase-MiniLM-L3-v2` (60MB) - 更小更快

### 3. 批量處理

處理大量文檔時，可以使用批量編碼：

```python
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

## 進階使用

### 集成 Ollama 本地模型

```bash
# 1. 安裝 Ollama
curl https://ollama.ai/install.sh | sh

# 2. 下載模型
ollama pull llama2

# 3. 在代碼中使用
# 修改 RAGSystem 使用 Ollama API
```

### 添加 Gradio 界面

```bash
# 安裝 Gradio
pip install gradio

# 創建 Web 界面
# 參考 6_practical_qa_system.py 並添加 Gradio 包裝
```

## 故障排除

### 問題：無法下載模型

**解決方案**:

```bash
# 設置 Hugging Face 鏡像
export HF_ENDPOINT=https://hf-mirror.com

# 或手動下載模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
```

### 問題：內存不足

**解決方案**:

1. 使用更小的批次大小
2. 使用更小的模型
3. 分批處理文檔

### 問題：編碼錯誤

**解決方案**:

```python
# 確保使用 UTF-8 編碼
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
```

## 測試檢查清單

- [ ] 所有依賴已安裝
- [ ] 語法檢查通過
- [ ] 範例 1 運行成功
- [ ] 範例 2 運行成功
- [ ] 範例 3 運行成功
- [ ] 範例 4 運行成功
- [ ] 範例 5 運行成功
- [ ] 範例 6 運行成功
- [ ] 測試數據已清理

## 資源

- [Sentence Transformers 文檔](https://www.sbert.net/)
- [FAISS 文檔](https://faiss.ai/)
- [Hugging Face 模型庫](https://huggingface.co/models)
- [LangChain 文檔](https://python.langchain.com/)

## 授權

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！
