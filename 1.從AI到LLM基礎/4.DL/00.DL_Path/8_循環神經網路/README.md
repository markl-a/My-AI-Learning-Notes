# 循環神經網路（RNN）學習指南

> **版本**: 2.0 (2025)
> **難度**: ⭐⭐⭐ 中級
> **預計學習時間**: 2-3 週

## 📚 目錄

- [學習目標](#學習目標)
- [前置知識](#前置知識)
- [學習路徑](#學習路徑)
- [內容結構](#內容結構)
- [實作建議](#實作建議)
- [AI 輔助學習](#ai-輔助學習)
- [實際應用](#實際應用)
- [常見問題](#常見問題)
- [延伸資源](#延伸資源)

---

## 🎯 學習目標

完成本章節學習後，你將能夠：

1. ✅ 理解序列數據的特性和處理方法
2. ✅ 掌握 RNN 的基本原理和數學推導
3. ✅ 從零實現一個完整的 RNN 模型
4. ✅ 使用 PyTorch 高級 API 構建 RNN
5. ✅ 理解梯度消失/爆炸問題及解決方案
6. ✅ 應用 RNN 解決實際問題（文本生成、序列預測等）

---

## 📋 前置知識

### 必備知識
- ✅ Python 編程基礎
- ✅ NumPy 數組操作
- ✅ PyTorch 基礎（張量、自動微分）
- ✅ 深度學習基礎（MLP、反向傳播）
- ✅ 基本的微積分和線性代數

### 推薦但非必需
- 自然語言處理基礎
- 時間序列分析概念

---

## 🗺️ 學習路徑

### 階段一：序列數據基礎（第 1-3 節）
**目標**: 理解序列數據的特殊性

```mermaid
graph LR
    A[1. 序列模型] --> B[2. 文本預處理]
    B --> C[3. 語言模型]
```

**學習重點**:
- 序列數據 vs 表格數據的差異
- 文本的數字化表示方法
- N-gram 語言模型

**預計時間**: 3-4 天

---

### 階段二：RNN 核心概念（第 4-6 節）
**目標**: 掌握 RNN 的原理和實作

```mermaid
graph LR
    A[4. RNN 理論] --> B[5. 從零實現]
    B --> C[6. PyTorch 實現]
```

**學習重點**:
- 隱藏狀態的概念
- RNN 的前向傳播和反向傳播
- 獨熱編碼（One-hot Encoding）
- 困惑度（Perplexity）評估指標

**預計時間**: 5-7 天

---

### 階段三：訓練技巧（第 7 節）
**目標**: 理解 RNN 訓練中的挑戰

```mermaid
graph LR
    A[BPTT 算法] --> B[梯度裁剪]
    B --> C[穩定訓練技巧]
```

**學習重點**:
- 通過時間反向傳播（BPTT）
- 梯度消失和梯度爆炸
- 梯度裁剪技術

**預計時間**: 2-3 天

---

## 📖 內容結構

### 0. 索引（0_index.ipynb）
- 章節概述
- RNN 的應用場景
- 學習路線圖

### 1. 序列模型（1_sequence.ipynb）
**核心概念**:
- 序列數據的特性
- 自回歸模型
- 馬爾可夫模型

**關鍵代碼**:
```python
# 序列預測示例
def sequence_prediction(data, tau=4):
    """使用過去 tau 個時間步預測未來"""
    T = len(data) - tau
    X = torch.zeros((T, tau))
    y = torch.zeros(T)
    for t in range(T):
        X[t] = data[t:t+tau]
        y[t] = data[t+tau]
    return X, y
```

---

### 2. 文本預處理（2_text_preprocessing.ipynb）
**核心概念**:
- 文本讀取和清洗
- 分詞（Tokenization）
- 詞表構建
- 數據批處理

**關鍵代碼**:
```python
# 構建詞表
class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        # 統計詞頻
        counter = collections.Counter(tokens)
        # 按頻率排序
        self.token_freqs = sorted(counter.items(),
                                 key=lambda x: x[1],
                                 reverse=True)
        # 建立詞到索引的映射
        self.token_to_idx = {}
        self.idx_to_token = []
```

---

### 3. 語言模型和數據集（3_language_models_and_dataset.ipynb）
**核心概念**:
- 語言模型的數學定義
- N-gram 模型
- 序列數據的批處理策略
- 隨機採樣 vs 順序分區

**重要公式**:
```
P(x₁, x₂, ..., xₜ) = ∏ P(xₜ | x₁, ..., xₜ₋₁)
```

---

### 4. RNN 基礎（4_rnn.ipynb）
**核心概念**:
- 隱藏狀態的定義
- RNN 的計算圖
- 參數共享機制

**數學推導**:
```
隱藏狀態更新：H_t = φ(X_t W_xh + H_{t-1} W_hh + b_h)
輸出計算：     O_t = H_t W_hq + b_q
```

**關鍵洞察**:
- RNN 通過隱藏狀態在時間步之間傳遞信息
- 參數在所有時間步共享，不隨序列長度增長

---

### 5. 從零實現 RNN（5_rnn-scratch.ipynb）
**完整實現流程**:

```python
# 1. 初始化模型參數
def get_params(vocab_size, num_hiddens, device):
    # 輸入到隱藏層
    W_xh = torch.randn(vocab_size, num_hiddens) * 0.01
    # 隱藏層到隱藏層
    W_hh = torch.randn(num_hiddens, num_hiddens) * 0.01
    # 偏置
    b_h = torch.zeros(num_hiddens)
    # 隱藏層到輸出層
    W_hq = torch.randn(num_hiddens, vocab_size) * 0.01
    b_q = torch.zeros(vocab_size)
    return [W_xh, W_hh, b_h, W_hq, b_q]

# 2. RNN 前向傳播
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 3. 梯度裁剪
def grad_clipping(net, theta):
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2))
                         for p in net.params))
    if norm > theta:
        for param in net.params:
            param.grad[:] *= theta / norm
```

**學習要點**:
- 理解每個參數矩陣的作用
- 掌握梯度裁剪的必要性
- 學會調試 RNN 訓練過程

---

### 6. 簡潔實現（6_rnn-concise.ipynb）
**使用 PyTorch API**:

```python
import torch.nn as nn

# 創建 RNN 層
rnn_layer = nn.RNN(input_size=vocab_size,
                   hidden_size=num_hiddens)

# 完整模型
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = rnn_layer.hidden_size
        self.linear = nn.Linear(self.num_hiddens, vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T, self.vocab_size).float()
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state
```

**對比分析**:
| 特性 | 從零實現 | PyTorch 實現 |
|------|---------|-------------|
| 靈活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 運行速度 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 學習價值 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 工業應用 | ⭐ | ⭐⭐⭐⭐⭐ |

---

### 7. 通過時間反向傳播（7_bptt.ipynb）
**核心概念**:
- BPTT 算法原理
- 計算複雜度分析
- 截斷 BPTT

**數學推導**:
```
損失函數: L = (1/T) Σ l(y_t, ŷ_t)
梯度: ∂L/∂W_hh = Σ (∂L/∂H_t) × (∂H_t/∂W_hh)
```

**重要問題**:
1. **梯度消失**: 當序列很長時，梯度會指數級衰減
2. **梯度爆炸**: 參數不當初始化導致梯度指數級增長

**解決方案**:
- 梯度裁剪（處理梯度爆炸）
- LSTM/GRU（處理梯度消失，見第 9 章）

---

## 💡 實作建議

### 1. 循序漸進的學習策略

#### Week 1: 理論基礎
- [ ] 完整閱讀第 1-3 節的理論部分
- [ ] 手動推導 RNN 的前向傳播公式
- [ ] 理解困惑度的計算方法

#### Week 2: 動手實作
- [ ] 運行第 5 節的從零實現代碼
- [ ] 嘗試修改超參數觀察效果
- [ ] 在自己的數據集上訓練模型

#### Week 3: 深入理解
- [ ] 比較從零實現和 PyTorch 實現的差異
- [ ] 研究梯度裁剪對訓練的影響
- [ ] 嘗試實現一個簡單的應用

---

### 2. 調試技巧

```python
# 檢查隱藏狀態的形狀
print(f"Hidden state shape: {H.shape}")  # 應該是 (batch_size, num_hiddens)

# 監控梯度範數
total_norm = 0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f'Gradient norm: {total_norm}')

# 可視化訓練損失
import matplotlib.pyplot as plt
plt.plot(train_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()
```

---

### 3. 超參數調優指南

| 超參數 | 推薦範圍 | 影響 |
|--------|---------|------|
| `num_hiddens` | 128-512 | 模型容量 |
| `learning_rate` | 0.001-1.0 | 收斂速度 |
| `batch_size` | 32-128 | 訓練穩定性 |
| `num_steps` | 32-64 | 序列長度 |
| `clip_theta` | 1.0-5.0 | 梯度裁剪閾值 |

**調優策略**:
1. 先用小模型快速驗證
2. 觀察訓練損失曲線
3. 使用驗證集避免過擬合
4. 記錄實驗結果

---

## 🤖 AI 輔助學習

### 與 ChatGPT/Claude 互動的提示詞

#### 1. 理解概念
```
我正在學習循環神經網路（RNN）。請用簡單的比喻解釋以下概念：
1. 隱藏狀態是什麼？
2. 為什麼 RNN 適合處理序列數據？
3. 梯度消失問題是如何產生的？

請用日常生活的例子來說明，並給出具體的計算示例。
```

#### 2. 代碼調試
```
我的 RNN 模型訓練時遇到以下問題：
[粘貼錯誤信息]

代碼如下：
[粘貼相關代碼]

可能的原因是什麼？如何修復？
```

#### 3. 深入探討
```
請幫我分析 RNN 和 Transformer 在處理序列數據時的差異：
1. 計算複雜度
2. 並行化能力
3. 長程依賴建模
4. 適用場景

並給出實際的應用案例。
```

---

### 使用 AI 工具增強學習

#### 1. **代碼補全和生成** (GitHub Copilot, Cursor)
```python
# 提示：實現一個 RNN 進行情感分析
class SentimentRNN(nn.Module):
    # Copilot 會自動補全完整實現
```

#### 2. **可視化工具**
```python
# 使用 TensorBoard 可視化訓練過程
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/rnn_experiment')
for epoch in range(num_epochs):
    loss = train_one_epoch()
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_histogram('RNN/weights', model.rnn.weight_hh_l0, epoch)
```

#### 3. **自動化實驗追蹤** (Weights & Biases)
```python
import wandb

wandb.init(project="rnn-learning", config={
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 32
})

# 訓練時自動記錄
wandb.log({"loss": loss, "perplexity": ppl})
```

---

## 🚀 實際應用

### 1. 文本生成
**應用場景**: 自動寫作、對話系統、代碼生成

```python
def generate_text(model, start_text, num_chars=100):
    model.eval()
    chars = [ch for ch in start_text]
    state = model.begin_state(batch_size=1, device=device)

    for i in range(num_chars):
        X = torch.tensor([vocab[chars[-1]]], device=device).reshape(1, 1)
        Y, state = model(X, state)
        next_char_idx = Y.argmax(dim=1).item()
        chars.append(vocab.idx_to_token[next_char_idx])

    return ''.join(chars)

# 使用示例
generated = generate_text(model, "The time traveller ", 200)
print(generated)
```

---

### 2. 情感分析
**應用場景**: 社交媒體監控、用戶反饋分析、電影評論分類

```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # 二分類：正面/負面

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        # 使用最後一個時間步的輸出
        return self.fc(hidden.squeeze(0))
```

---

### 3. 時間序列預測
**應用場景**: 股票價格預測、天氣預報、設備故障預警

```python
class TimeSeriesRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        output, hidden = self.rnn(x)
        # 預測下一個時間步
        return self.fc(output[:, -1, :])

# 訓練示例
model = TimeSeriesRNN(input_dim=1, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

### 4. 序列標註（命名實體識別）
**應用場景**: NER、詞性標註、分詞

```python
class SequenceLabelingRNN(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tag_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        # 對每個時間步進行分類
        return self.fc(output)
```

---

## ❓ 常見問題

### Q1: RNN 和 CNN 有什麼區別？
**A**:
- **CNN**: 處理空間結構數據（圖像），使用局部連接和權重共享
- **RNN**: 處理序列數據（文本、時間序列），通過隱藏狀態捕獲時間依賴

**選擇建議**:
- 圖像處理 → CNN
- 文本、時間序列 → RNN/LSTM/Transformer
- 視頻處理 → CNN + RNN 或 3D CNN

---

### Q2: 為什麼訓練 RNN 這麼困難？
**A**: 主要挑戰：
1. **梯度消失/爆炸**: 長序列導致梯度傳播不穩定
2. **長程依賴**: 難以捕獲距離較遠的依賴關係
3. **訓練速度慢**: 序列計算難以並行化

**解決方案**:
- 使用梯度裁剪
- 採用 LSTM/GRU 架構
- 考慮使用 Transformer（如果適用）

---

### Q3: 什麼時候應該使用 RNN vs Transformer？
**A**:

| 場景 | 推薦模型 | 原因 |
|------|---------|------|
| 小數據集 | RNN | 參數少，不易過擬合 |
| 大數據集 | Transformer | 並行訓練，效果更好 |
| 實時推理 | RNN | 逐步處理，延遲低 |
| 批量處理 | Transformer | 並行計算，速度快 |
| 超長序列 | Transformer | 注意力機制處理長程依賴 |

---

### Q4: 如何選擇隱藏層大小？
**A**: 經驗法則：
- **小數據集**: 64-128
- **中等數據集**: 256-512
- **大數據集**: 512-1024

**權衡**:
- 更大的隱藏層 → 更強的表示能力，但容易過擬合
- 使用驗證集選擇最優大小

---

### Q5: 獨熱編碼 vs 詞嵌入，哪個更好？
**A**:

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| 獨熱編碼 | 簡單，無需預訓練 | 高維稀疏，無語義信息 | 小詞表，教學演示 |
| 詞嵌入 | 低維稠密，包含語義 | 需要訓練或預訓練 | 實際應用，大詞表 |

**推薦**: 實際項目中使用詞嵌入（Word2Vec, GloVe, BERT）

---

## 📚 延伸資源

### 官方文檔
- [PyTorch RNN 教程](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [動手學深度學習](https://d2l.ai/chapter_recurrent-neural-networks/)

### 優質課程
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Fast.ai NLP Course](https://www.fast.ai/posts/2019-07-08-fastai-nlp.html)

### 論文閱讀
- **RNN 原始論文**: Rumelhart et al., "Learning representations by back-propagating errors" (1986)
- **LSTM**: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- **GRU**: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)

### 實戰項目
- [Awesome RNN](https://github.com/kjw0612/awesome-rnn): RNN 相關資源合集
- [PyTorch Examples](https://github.com/pytorch/examples/tree/main/word_language_model): 官方語言模型示例

### 工具和庫
- **Transformers**: Hugging Face 提供的預訓練模型
- **AllenNLP**: NLP 研究工具包
- **spaCy**: 工業級 NLP 庫

---

## 🎓 學習檢查清單

完成以下任務以確保掌握 RNN 基礎：

- [ ] 能解釋 RNN 的工作原理（用自己的話）
- [ ] 能從零實現一個簡單的 RNN
- [ ] 理解梯度消失/爆炸的原因和解決方法
- [ ] 能使用 PyTorch API 快速構建 RNN 模型
- [ ] 完成至少一個實際項目（文本生成/情感分析等）
- [ ] 能調試 RNN 訓練中的常見問題
- [ ] 理解 RNN 的局限性和適用場景

---

## 💬 社群和討論

遇到問題或想分享學習心得？

- **GitHub Issues**: 在本倉庫提交問題
- **Stack Overflow**: 使用標籤 `pytorch` `rnn` `lstm`
- **Reddit**: r/MachineLearning, r/deeplearning
- **Discord/Slack**: 加入深度學習社群

---

## 📝 下一步

完成本章後，建議繼續學習：
1. **第 9 章**: 現代循環神經網路（LSTM、GRU、雙向RNN）
2. **注意力機制**: 為 Transformer 打基礎
3. **Seq2Seq 模型**: 機器翻譯、對話系統

---

**最後更新**: 2025-11-18
**貢獻者**: AI Learning Community
**授權**: MIT License

---

**祝學習愉快！🚀**
