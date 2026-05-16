# 先修知識檢查清單 (Prerequisites Checklist)

> 在開始學習本專案內容前，請先評估自己的知識基礎。
> 此清單幫助你找到適合的起點，避免學習過程中遇到障礙。

---

## 📊 快速自我評估

### 你是哪類學習者？

| 類型 | 描述 | 建議起點 |
|------|------|---------|
| 🌱 **完全新手** | 無程式設計經驗，對 AI 感興趣 | [Level 0](#level-0-程式設計基礎) 開始 |
| 🌿 **有程式基礎** | 會 Python，但沒學過 ML | [Level 1](#level-1-機器學習基礎) 開始 |
| 🌳 **有 ML 經驗** | 了解 ML/DL，想學 LLM 應用 | [Level 2](#level-2-深度學習基礎) 快速複習後進入 LLM |
| 🌲 **AI 工程師** | 有 LLM 經驗，想深入研究 | 直接進入 [進階主題](#level-4-llm-進階) |

---

## Level 0: 程式設計基礎

### Python 基礎 ✅ 必須

在學習 AI/ML 之前，你需要熟悉 Python：

#### 檢查清單

- [ ] **變數與資料類型**
  - 整數、浮點數、字串、布林值
  - 列表 (list)、字典 (dict)、元組 (tuple)、集合 (set)

- [ ] **控制流程**
  - if/elif/else 條件判斷
  - for/while 迴圈
  - break/continue/pass

- [ ] **函數**
  - 定義和呼叫函數
  - 參數和回傳值
  - *args 和 **kwargs
  - Lambda 函數

- [ ] **物件導向程式設計 (OOP)**
  - 類別 (class) 和物件 (object)
  - 繼承 (inheritance)
  - 方法 (method) 和屬性 (attribute)

- [ ] **模組與套件**
  - import 語句
  - pip 套件管理
  - 虛擬環境 (venv/conda)

- [ ] **檔案處理**
  - 讀寫文字檔案
  - JSON/CSV 處理

- [ ] **例外處理**
  - try/except/finally
  - 自定義例外

#### 自測題

```python
# 如果你能理解並修改以下程式碼，Python 基礎足夠
class DataProcessor:
    def __init__(self, data: list[dict]):
        self.data = data

    def filter_by_key(self, key: str, value) -> list[dict]:
        return [item for item in self.data if item.get(key) == value]

    def aggregate(self, key: str) -> dict:
        result = {}
        for item in self.data:
            k = item.get(key)
            result[k] = result.get(k, 0) + 1
        return result

# 你能解釋這段程式碼在做什麼嗎？
# 你能添加一個新方法嗎？
```

#### 補充資源

如果上述有未掌握的部分：
- 📚 [Python 官方教程](https://docs.python.org/zh-tw/3/tutorial/)
- 📚 本專案：[`1.從AI到LLM基礎/2.AI_Intro/1.快速入門python.ipynb`](./1.從AI到LLM基礎/2.AI_Intro/1.快速入門python.ipynb)
- 🎬 [莫煩 Python](https://mofanpy.com/)

---

## Level 1: 機器學習基礎

### 數學基礎 ✅ 重要

#### 線性代數（必須）

- [ ] **向量和矩陣**
  - 向量的定義和運算（加減、點積）
  - 矩陣的定義和運算（加減、乘法、轉置）
  - 向量和矩陣的維度

- [ ] **矩陣運算**
  - 矩陣乘法（如何計算、維度規則）
  - 單位矩陣和逆矩陣
  - 行列式（基本概念）

- [ ] **特徵值和特徵向量**
  - 基本概念（可選深入）
  - PCA 的數學基礎

#### 自測題

```
Q1: 若 A 是 3×4 矩陣，B 是 4×2 矩陣，AB 的維度是？
A: 3×2

Q2: 什麼是向量的點積？兩個垂直向量的點積是多少？
A: 點積是對應元素相乘再求和；垂直向量點積為 0

Q3: 為什麼深度學習中要用矩陣運算？
A: 因為批次處理和並行計算效率高
```

#### 微積分（推薦）

- [ ] **導數**
  - 導數的定義和意義（斜率、變化率）
  - 常見函數的導數
  - 鏈式法則（Chain Rule）—— 反向傳播的基礎

- [ ] **偏導數**
  - 多變數函數的偏導
  - 梯度向量的概念

- [ ] **最佳化**
  - 梯度下降的直覺理解
  - 局部最小值 vs 全局最小值

#### 自測題

```
Q1: 為什麼訓練神經網路需要計算梯度？
A: 梯度指向損失函式增長最快的方向，負梯度方向是下降最快的方向

Q2: 什麼是鏈式法則？為什麼它對深度學習重要？
A: 複合函數求導的規則，是反向傳播演算法的數學基礎

Q3: 學習率如果設太大會發生什麼？
A: 可能跳過最優點，導致損失不收斂
```

#### 機率與統計（推薦）

- [ ] **基礎概念**
  - 機率的定義
  - 條件機率和貝氏定理
  - 期望值和變異數

- [ ] **機率分佈**
  - 正態分佈（高斯分佈）
  - 伯努利分佈、二項分佈

- [ ] **統計推論**
  - 最大似然估計（MLE）的概念
  - 過擬合和正則化的統計解釋

#### 補充資源

- 📚 本專案：[`1.從AI到LLM基礎/1.Math_4_ML/`](./1.從AI到LLM基礎/1.Math_4_ML/)
- 🎬 [3Blue1Brown - 線性代數的本質](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- 🎬 [StatQuest - 統計學習](https://www.youtube.com/c/joshstarmer)

---

### Python 資料科學套件 ✅ 必須

#### NumPy

- [ ] 建立陣列（array）
- [ ] 陣列運算（向量化操作）
- [ ] 索引和切片
- [ ] 廣播（Broadcasting）
- [ ] 常用函數（sum、mean、reshape、dot）

#### Pandas

- [ ] DataFrame 和 Series
- [ ] 讀取資料（CSV、JSON）
- [ ] 資料篩選和查詢
- [ ] 分組和聚合（groupby）
- [ ] 資料清洗（處理缺失值）

#### Matplotlib/Seaborn

- [ ] 基本繪圖（折線圖、散點圖）
- [ ] 直方圖和分佈圖
- [ ] 熱力圖
- [ ] 子圖和圖例

#### 自測題

```python
import numpy as np
import pandas as pd

# 你能解釋以下程式碼的輸出嗎？
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)           # ?
print(arr.T.shape)         # ?
print(arr.sum(axis=0))     # ?
print(arr @ arr.T)         # ?

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Alice', 'Bob'],
    'score': [85, 90, 88, 92]
})
print(df.groupby('name')['score'].mean())  # ?
```

#### 補充資源

- 📚 本專案：[`1.從AI到LLM基礎/2.AI_Intro/2.Python的ML相關模塊套件使用.ipynb`](./1.從AI到LLM基礎/2.AI_Intro/2.Python的ML相關模塊套件使用.ipynb)
- 📚 [NumPy 官方教程](https://numpy.org/doc/stable/user/quickstart.html)
- 📚 [Pandas 10 分鐘入門](https://pandas.pydata.org/docs/user_guide/10min.html)

---

## Level 2: 深度學習基礎

### 機器學習概念 ✅ 必須

- [ ] **監督學習 vs 非監督學習 vs 強化學習**
- [ ] **分類 vs 迴歸**
- [ ] **訓練集、驗證集、測試集**
- [ ] **過擬合和欠擬合**
- [ ] **偏差-方差權衡**
- [ ] **交叉驗證**
- [ ] **評估指標**
  - 分類：準確率、精確率、召回率、F1
  - 迴歸：MSE、MAE、R²

### 深度學習概念 ✅ 必須

- [ ] **神經網路基礎**
  - 感知器和多層感知器 (MLP)
  - 激活函式（ReLU、Sigmoid、Tanh）
  - 前向傳播和反向傳播

- [ ] **訓練過程**
  - 損失函式（MSE、交叉熵）
  - 優化器（SGD、Adam）
  - 學習率和批次大小
  - Epoch 的概念

- [ ] **正則化**
  - L1/L2 正則化
  - Dropout
  - 早停（Early Stopping）

- [ ] **深度學習框架**
  - PyTorch 或 TensorFlow 基礎
  - 建立簡單模型
  - 訓練迴圈

#### 自測題

```
Q1: 解釋什麼是反向傳播？
A: 計算損失函式對每個參數的梯度，用於更新權重

Q2: 為什麼需要激活函式？
A: 引入非線性，讓網路能學習複雜函數

Q3: 過擬合有哪些常見的解決方法？
A: 更多資料、正則化、Dropout、早停、資料增強

Q4: Adam 優化器比 SGD 有什麼優點？
A: 自適應學習率，結合動量，收斂更快更穩定
```

#### 補充資源

- 📚 本專案：[`1.從AI到LLM基礎/4.DL/`](./1.從AI到LLM基礎/4.DL/)
- 🎬 [李宏毅機器學習課程](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
- 📚 [PyTorch 官方教程](https://pytorch.org/tutorials/)

---

## Level 3: LLM 基礎

### NLP 基礎 ✅ 推薦

- [ ] **文字預處理**
  - 分詞（Tokenization）
  - 詞彙表和 OOV 處理
  - 子詞分詞（BPE、WordPiece）

- [ ] **詞嵌入**
  - One-hot encoding 的問題
  - Word2Vec、GloVe 的概念
  - 為什麼需要嵌入

- [ ] **序列模型**
  - RNN/LSTM 的基本概念
  - 為什麼 Transformer 取代了 RNN

### Transformer 架構 ✅ 必須

- [ ] **注意力機制**
  - 什麼是注意力
  - Self-Attention 如何工作
  - Q、K、V 的直覺理解

- [ ] **Transformer 組件**
  - 多頭注意力
  - 位置編碼
  - 前饋網路
  - 殘差連接和層歸一化

- [ ] **預訓練語言模型**
  - BERT vs GPT 的差異
  - 預訓練任務（MLM、CLM）
  - 微調的概念

#### 自測題

```
Q1: Transformer 相比 RNN 有什麼優勢？
A: 並行計算、長距離依賴、訓練更快

Q2: 為什麼需要位置編碼？
A: Transformer 本身無法區分位置，需要顯式加入位置資訊

Q3: BERT 和 GPT 的主要區別是什麼？
A: BERT 是編碼器（雙向），GPT 是解碼器（單向自迴歸）

Q4: 什麼是 Tokenization？為什麼要用子詞分詞？
A: 將文字切分為模型可處理的單位；子詞分詞平衡了詞彙表大小和 OOV 問題
```

#### 補充資源

- 📚 本專案：[`2.深入LLM模型工程與LLM運維/1.LLM 基礎與架構/`](./2.深入LLM模型工程與LLM運維/1.LLM%20基礎與架構/)
- 📚 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 📚 [Attention Is All You Need 論文](https://arxiv.org/abs/1706.03762)

---

## Level 4: LLM 進階

### 進入進階主題前，確保你了解：

- [ ] **LLM 基礎**
  - 了解 GPT 系列模型的工作原理
  - 知道什麼是 Token、Context Window
  - 會使用 OpenAI/Claude API

- [ ] **微調概念**
  - 全量微調 vs 參數高效微調
  - LoRA、QLoRA 的基本概念
  - 訓練資料格式

- [ ] **推論優化**
  - 量化的概念（INT8、INT4）
  - KV Cache 的作用
  - 批次推理

- [ ] **RAG 基礎**
  - 什麼是 RAG，為什麼需要
  - 向量資料庫的概念
  - 嵌入模型的作用

### 進階主題推薦順序

1. **RAG 系統設計** → [`3.LLM應用工程/4.(RAG) 基礎/`](./3.LLM應用工程/4.(RAG)%20基礎/)
2. **Agent 系統** → [`3.LLM應用工程/3.Agent/`](./3.LLM應用工程/3.Agent/)
3. **模型微調** → [`2.深入LLM模型工程與LLM運維/5.監督微調(SFT)/`](./2.深入LLM模型工程與LLM運維/5.監督微調(SFT)/)
4. **對齊技術** → [`2.深入LLM模型工程與LLM運維/11.現代對齊方法2024-2025/`](./2.深入LLM模型工程與LLM運維/11.現代對齊方法2024-2025/)
5. **推論優化** → [`3.LLM應用工程/6.推論優化/`](./3.LLM應用工程/6.推論優化/)

---

## 📋 完整檢查清單

### 必備知識（所有學習者）

- [ ] Python 基礎程式設計
- [ ] NumPy 和 Pandas 基礎
- [ ] 機器學習基本概念
- [ ] 深度學習基本概念

### 推薦知識（更好的學習體驗）

- [ ] 線性代數基礎
- [ ] 微積分基礎（導數、梯度）
- [ ] PyTorch 或 TensorFlow 基礎
- [ ] Git 版本控制

### 可選知識（進階主題需要）

- [ ] 機率與統計
- [ ] NLP 基礎
- [ ] 雲端服務基礎（AWS/GCP）
- [ ] Docker 容器基礎

---

## 🎯 學習路徑建議

### 🌱 新手路徑（0基礎，約 6 個月）

```
Month 1-2: Python 基礎
├── 學習 Python 語法
├── 練習資料科學套件
└── 完成 10+ 小專案

Month 3-4: ML/DL 基礎
├── 學習數學基礎
├── 了解機器學習概念
├── 動手訓練簡單模型
└── 完成 Kaggle 入門競賽

Month 5-6: LLM 應用
├── 學習 Transformer 架構
├── 使用 LLM API
├── 建立 RAG 應用
└── 完成端到端專案
```

### 🌿 有基礎路徑（會 Python，約 3 個月）

```
Month 1: 快速補齊
├── 複習數學基礎
├── 學習深度學習框架
└── 理解 Transformer

Month 2: LLM 核心
├── LLM API 使用
├── RAG 系統建立
├── Agent 工作流

Month 3: 進階主題
├── 微調技術
├── 推論優化
└── 生產部署
```

### 🌳 進階路徑（有 ML 經驗，約 1 個月）

```
Week 1: LLM 基礎
├── Transformer 深入
├── 預訓練技術
└── 評估方法

Week 2-3: 應用開發
├── RAG 進階
├── Agent 系統
├── 工具整合

Week 4: 生產化
├── 微調和對齊
├── 部署優化
└── 監控運維
```

---

## 常見問題

### Q: 我數學不好，可以學 AI 嗎？

**A:** 可以！入門階段更重視直覺理解而非嚴格推導。建議：
1. 先建立直覺，再補數學
2. 使用視覺化工具輔助理解
3. 從應用角度反推需要的數學知識

### Q: 應該學 PyTorch 還是 TensorFlow？

**A:** 2024-2025 年推薦 **PyTorch**：
- 更直覺的 API
- 研究社區主流選擇
- Hugging Face 生態系統首選
- 本專案主要使用 PyTorch

### Q: 需要 GPU 嗎？

**A:**
- **學習階段**：不需要，Colab 免費 GPU 足夠
- **微調模型**：需要，至少 16GB VRAM
- **推理/應用**：取決於場景，可用 API 替代

### Q: 英文不好怎麼辦？

**A:**
- 本專案完全使用繁體中文
- 但建議逐步提升英文閱讀能力
- 許多最新資源只有英文版

---

## 下一步

完成自我評估後，前往對應的起點開始學習：

- 🌱 [從零開始 → 1.從AI到LLM基礎](./1.從AI到LLM基礎/)
- 🌿 [有基礎 → 2.深入LLM模型工程](./2.深入LLM模型工程與LLM運維/)
- 🌳 [直接應用 → 3.LLM應用工程](./3.LLM應用工程/)

或查看 [學習路徑規劃](./LEARNING_PATHS.md) 獲取更詳細的指引。

---

> 📝 **回饋**：如果你覺得缺少某些先修知識的說明，歡迎透過 Issue 告訴我們！
