# RNN 實戰項目集

> **完整的端到端 RNN 應用項目**
>
> 從資料準備到模型部署的完整實踐

## 📚 項目概覽

本資料夾包含三個完整的 RNN 實戰項目，覆蓋了 RNN 在不同領域的應用：

| 項目 | 難度 | 時間 | 主要技術 | 應用場景 |
|------|------|------|---------|---------|
| 1. 情感分析 | ⭐⭐⭐ | 2-3h | Bi-LSTM, Attention | NLP |
| 2. 文字生成 | ⭐⭐⭐ | 2-3h | Char-LSTM, 溫度採樣 | 創意 AI |
| 3. 時間序列預測 | ⭐⭐⭐⭐ | 3-4h | LSTM, Seq2Seq | 金融/IoT |

---

## 🎯 學習目標

完成這些項目後，你將能夠：

- ✅ 端到端構建 RNN 應用
- ✅ 處理真實世界的資料
- ✅ 調試和優化模型
- ✅ 部署模型到生產環境
- ✅ 使用 AI 工具輔助開發

---

## 📖 項目詳情

### 1. 情感分析 (sentiment_analysis.ipynb)

**任務**: 對電影評論進行正負面情感分類

**亮點**:
- 完整的文字預處理流程
- 三種模型架構對比（Simple LSTM, Bi-LSTM, Attention）
- 注意力權重可視化
- 模型解釋和部署

**核心技術**:
```python
class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.bilstm = nn.LSTM(bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
```

**資料集**: IMDB 電影評論 (50k 樣本)

**性能指標**:
- 準確率: ~90%
- F1 分數: ~0.88

---

### 2. 文字生成 (text_generation.ipynb)

**任務**: 生成莎士比亞風格的文字

**亮點**:
- 字符級語言模型
- 溫度採樣實驗
- Top-K 採樣對比
- 生成品質評估

**核心技術**:
```python
def generate_text(model, seed, temperature=1.0):
    # 溫度控制隨機性
    logits = output / temperature
    probs = F.softmax(logits, dim=0)
    next_char = torch.multinomial(probs, 1)
```

**資料集**: 莎士比亞全集

**生成示例**:
```
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer...
```

---

### 3. 時間序列預測 (time_series_forecasting.ipynb)

**任務**: 預測股票價格/能源需求

**亮點**:
- 多變量時間序列處理
- Seq2Seq 預測架構
- 滑動窗口技術
- 預測不確定性量化

**核心技術**:
```python
class Seq2SeqForecaster(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(...)
        self.decoder = nn.LSTM(...)

    def forward(self, past, future_steps):
        # 編碼歷史
        _, (h, c) = self.encoder(past)
        # 解碼預測
        predictions = self.decoder(h, steps=future_steps)
```

**資料集**: 股票價格（S&P 500）或能源消耗資料

**性能指標**:
- MAE: < 5%
- RMSE: < 8%

---

## 🚀 快速開始

### 環境配置

```bash
# 建立虛擬環境
conda create -n rnn-projects python=3.8
conda activate rnn-projects

# 安裝依賴
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install jupyter notebook
pip install tqdm scikit-learn
```

### 運行項目

```bash
# 啟動 Jupyter Notebook
jupyter notebook

# 打開任意項目文件並運行
```

---

## 📊 項目結構

```
10_RNN實戰項目/
├── README.md                          # 本文件
├── 1_sentiment_analysis.ipynb         # 情感分析
├── 2_text_generation.ipynb            # 文字生成
├── 3_time_series_forecasting.ipynb    # 時間序列預測
├── data/                              # 資料資料夾
│   ├── imdb/                          # IMDB 資料
│   ├── shakespeare/                   # 莎士比亞文字
│   └── stocks/                        # 股票資料
├── models/                            # 保存的模型
│   ├── sentiment_model.pth
│   ├── text_gen_model.pth
│   └── forecasting_model.pth
└── utils/                             # 工具函數
    ├── data_utils.py                  # 資料處理
    ├── model_utils.py                 # 模型工具
    └── visualization.py               # 可視化
```

---

## 💡 學習路徑

### 初學者路徑

```mermaid
graph LR
    A[情感分析] --> B[文字生成]
    B --> C[時間序列預測]
```

**理由**: 從簡單的分類任務開始，逐步過渡到生成任務，最後處理複雜的序列預測

### 進階路徑

根據興趣選擇：
- **NLP 方向**: 情感分析 → 文字生成
- **時序分析**: 時間序列預測
- **全棧 AI**: 完成所有項目

---

## 🤖 AI 輔助開發

### 使用 ChatGPT/Claude 的場景

#### 1. 調試錯誤

**提示詞**:
```
我的 LSTM 模型訓練時出現以下錯誤：
[粘貼錯誤資訊和相關程式碼]

可能的原因和解決方案是什麼？
```

#### 2. 性能優化

**提示詞**:
```
我的情感分析模型準確率只有 75%，配置如下：
- 模型: Bi-LSTM
- 資料量: 5000 條
- 超參數: [列出超參數]

如何提高性能？請提供具體的改進建議。
```

#### 3. 程式碼改進

**提示詞**:
```
請審查以下程式碼並建議改進：
[粘貼程式碼]

關注點：
1. 程式碼效率
2. 最佳實踐
3. 潛在的bug
```

#### 4. 架構設計

**提示詞**:
```
我想構建一個時間序列預測模型，預測未來 24 小時的電力需求。
輸入：過去 168 小時的資料（7天）
輸出：未來 24 小時的預測

請建議：
1. 模型架構（LSTM vs GRU vs Transformer）
2. 如何處理多變量輸入
3. 如何量化預測不確定性
```

---

## 🛠️ 常見問題

### Q1: 模型訓練太慢怎麼辦？

**A**: 優化策略：

```python
# 1. 使用更小的批次大小
batch_size = 32  # 從 128 減少到 32

# 2. 減少序列長度
max_seq_len = 50  # 從 100 減少到 50

# 3. 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 4. 使用混合精度訓練
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

### Q2: 模型過擬合怎麼辦？

**A**: 正則化技術：

```python
# 1. Dropout
self.dropout = nn.Dropout(0.5)

# 2. 權重衰減
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 3. Early Stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break

# 4. 資料增強
# - 同義詞替換
# - 回譯
# - 隨機刪除
```

---

### Q3: 如何評估生成品質？

**A**: 多維度評估：

```python
# 1. 困惑度（Perplexity）
perplexity = torch.exp(loss)

# 2. BLEU 分數（與參考文字比較）
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([reference], generated)

# 3. 多樣性指標
unique_chars = len(set(generated_text))
unique_words = len(set(generated_text.split()))

# 4. 人工評估
# - 流暢性: 1-5 分
# - 連貫性: 1-5 分
# - 創造性: 1-5 分
```

---

### Q4: 如何部署模型？

**A**: 部署步驟：

```python
# 1. 保存完整模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'config': config
}, 'model_complete.pth')

# 2. 使用 TorchScript
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 3. 轉換為 ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# 4. 建立 API（Flask/FastAPI）
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(text: str):
    # 加載模型並預測
    result = model.predict(text)
    return {"prediction": result}
```

---

## 📈 性能基準

### 硬件配置對比

| 配置 | 情感分析 | 文字生成 | 時序預測 |
|------|---------|---------|---------|
| CPU (i5) | ~10 min | ~15 min | ~20 min |
| GPU (GTX 1060) | ~2 min | ~3 min | ~4 min |
| GPU (RTX 3080) | ~1 min | ~1.5 min | ~2 min |

### 模型大小

| 項目 | 參數量 | 模型大小 | 推論速度 |
|------|-------|---------|---------|
| 情感分析 | ~500K | ~2 MB | ~5 ms |
| 文字生成 | ~1M | ~4 MB | ~10 ms |
| 時序預測 | ~800K | ~3 MB | ~3 ms |

---

## 🎓 進階挑戰

完成基礎項目後的挑戰任務：

### Challenge 1: 多語言情感分析
- 擴展到中文、日文等語言
- 使用多語言詞嵌入（mBERT）
- 跨語言遷移學習

### Challenge 2: 對話生成
- 構建聊天機器人
- 實現多輪對話
- 添加個性化回覆

### Challenge 3: 異常檢測
- 使用時序模型檢測異常
- 實時預警系統
- 可解釋性分析

### Challenge 4: 多模態學習
- 結合文字和圖像
- 影片字幕生成
- 圖像描述生成

---

## 📚 延伸資源

### 推薦課程
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### 優質論文
- **Sentiment Analysis**: "Recursive Deep Models for Semantic Compositionality"
- **Text Generation**: "The Curious Case of Neural Text Degeneration"
- **Time Series**: "Deep Learning for Time Series Forecasting"

### 開源項目
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [AllenNLP](https://github.com/allenai/allennlp)
- [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting)

---

## 🤝 貢獻

發現問題或有改進建議？

1. Fork 本倉庫
2. 建立功能分支
3. 提交 Pull Request

---

## 📝 更新日誌

### v2.0 (2025-01-18)
- ✨ 新增三個完整的實戰項目
- 🎨 添加互動式可視化
- 📚 完善文檔和教程
- 🤖 集成 AI 輔助學習

---

## 💬 討論和支持

- **GitHub Issues**: 報告問題
- **Discussions**: 討論想法
- **Email**: your-email@example.com

---

## 📄 授權

MIT License

---

**祝學習愉快！記得實踐是最好的老師 🚀**

---

## 🎯 下一步

完成這些項目後，建議：

1. **深入 Transformer**: 學習現代序列模型
2. **部署實戰**: 將模型部署到雲端
3. **參加競賽**: Kaggle, 天池等
4. **貢獻開源**: 回饋社群

---

**Happy Coding! 💻✨**
