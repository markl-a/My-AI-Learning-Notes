# RNN 實戰項目集合

本文件夾包含多個完整的RNN實戰項目，從基礎到進階，幫助你將理論應用到實際問題中。

## 📁 項目列表

### 🟢 初級項目

#### 1. 古詩生成器 (Chinese Poetry Generator)
**難度**: ⭐⭐
**技術**: Character-level LSTM
**資料集**: 唐詩三百首
**學習目標**:
- 字符級語言建模
- 溫度參數調節
- 文字預處理

**項目結構**:
```
poetry_generator/
├── data/
│   └── tang_poetry.txt
├── model.py
├── train.py
├── generate.py
└── README.md
```

**快速開始**:
```bash
cd poetry_generator
python train.py --epochs 100
python generate.py --prefix "春" --length 20
```

---

#### 2. 電影評論情感分析 (Movie Review Sentiment Analysis)
**難度**: ⭐⭐
**技術**: BiLSTM + Attention
**資料集**: IMDb 或自定義中文影評
**學習目標**:
- 文字分類
- 詞嵌入使用
- 類別不平衡處理

**項目結構**:
```
sentiment_analysis/
├── data/
│   ├── train.csv
│   └── test.csv
├── model.py
├── train.py
├── evaluate.py
├── predict.py
└── README.md
```

**評估指標**:
- 準確率 (Accuracy)
- F1-Score
- ROC-AUC

---

#### 3. FAQ問答機器人 (FAQ Chatbot)
**難度**: ⭐⭐⭐
**技術**: Seq2Seq
**資料集**: 客服FAQ對話
**學習目標**:
- 編碼器-解碼器架構
- 問答匹配
- 回覆生成

**項目結構**:
```
faq_chatbot/
├── data/
│   └── qa_pairs.json
├── model.py
├── train.py
├── chat.py
└── README.md
```

---

### 🟡 中級項目

#### 4. 新聞摘要生成 (News Summarization)
**難度**: ⭐⭐⭐⭐
**技術**: Seq2Seq + Attention
**資料集**: CNN/DailyMail 或中文新聞
**學習目標**:
- 抽象式摘要
- 注意力機制
- ROUGE評估

**項目結構**:
```
news_summarizer/
├── data/
│   ├── articles.txt
│   └── summaries.txt
├── model.py
├── train.py
├── summarize.py
├── evaluate.py
└── README.md
```

**挑戰**:
- 處理長文字
- 避免生成重複
- 保持摘要連貫性

---

#### 5. 中文命名實體識別 (Chinese NER)
**難度**: ⭐⭐⭐⭐
**技術**: BiLSTM + CRF
**資料集**: People's Daily NER
**學習目標**:
- 序列標註
- CRF層應用
- 實體級評估

**項目結構**:
```
chinese_ner/
├── data/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
├── model.py
├── train.py
├── evaluate.py
├── predict.py
└── README.md
```

**實體類型**:
- PER (人名)
- LOC (地名)
- ORG (機構名)

---

#### 6. 多類情感分析 (Multi-class Sentiment)
**難度**: ⭐⭐⭐
**技術**: Deep BiLSTM
**資料集**: 商品評論
**學習目標**:
- 多分類問題
- 類別權重平衡
- 混淆矩陣分析

**分類**:
- 非常負面
- 負面
- 中性
- 正面
- 非常正面

---

### 🔴 高級項目

#### 7. 多輪對話系統 (Multi-turn Dialog System)
**難度**: ⭐⭐⭐⭐⭐
**技術**: Hierarchical RNN + Attention
**資料集**: Ubuntu Dialogue Corpus
**學習目標**:
- 上下文管理
- 層次化編碼
- 對話狀態追蹤

**項目結構**:
```
dialog_system/
├── data/
│   └── dialogues.json
├── models/
│   ├── encoder.py
│   ├── context_manager.py
│   └── decoder.py
├── train.py
├── chat.py
└── README.md
```

**特色功能**:
- 多輪上下文記憶
- 意圖識別
- 槽位填充

---

#### 8. 中英機器翻譯 (Chinese-English Translation)
**難度**: ⭐⭐⭐⭐⭐
**技術**: Seq2Seq + Attention + Beam Search
**資料集**: WMT Chinese-English
**學習目標**:
- 機器翻譯全流程
- 注意力機制
- Beam Search優化

**項目結構**:
```
translation/
├── data/
│   ├── train.zh
│   ├── train.en
│   ├── dev.zh
│   └── dev.en
├── model.py
├── train.py
├── translate.py
├── evaluate_bleu.py
└── README.md
```

**評估**:
- BLEU Score
- 人工評估

---

#### 9. 程式碼生成器 (Code Generation)
**難度**: ⭐⭐⭐⭐⭐
**技術**: Attention-based Seq2Seq
**資料集**: CodeSearchNet
**學習目標**:
- 從註釋生成程式碼
- 結構化輸出
- 語法約束

**輸入**: 自然語言描述
**輸出**: Python/Java程式碼

---

## 🛠️ 通用工具和腳本

### data_processor.py
```python
# 通用資料預處理工具
- 文字清洗
- 分詞
- 詞表構建
- 資料增強
```

### trainer.py
```python
# 通用訓練腳本
- 訓練循環
- 驗證
- 早停
- 模型保存
```

### utils.py
```python
# 工具函數
- 指標計算
- 可視化
- 日誌記錄
```

---

## 💡 項目選擇建議

### 如果你是初學者...
從 **古詩生成器** 或 **電影評論情感分析** 開始：
- 資料量小，訓練快
- 概念清晰
- 容易看到效果

### 如果你有一定基礎...
嘗試 **新聞摘要生成** 或 **中文NER**：
- 涉及更複雜的架構
- 需要更多調參經驗
- 更接近實際應用

### 如果你想挑戰自己...
選擇 **多輪對話系統** 或 **機器翻譯**：
- 需要綜合運用多種技術
- 資料處理更複雜
- 有很大優化空間

---

## 📚 開發流程建議

### 1. 需求分析
- 明確任務目標
- 確定輸入輸出格式
- 選擇評估指標

### 2. 資料準備
- 收集資料
- 探索性資料分析(EDA)
- 資料清洗和預處理
- 劃分訓練/驗證/測試集

### 3. 模型設計
- 選擇合適的架構
- 設計網絡結構
- 確定超參數範圍

### 4. 訓練與調優
- 實現訓練腳本
- 監控訓練過程
- 調整超參數
- 處理過擬合/欠擬合

### 5. 評估與分析
- 在測試集上評估
- 錯誤分析
- 可視化結果

### 6. 部署與優化
- 模型壓縮
- 推理加速
- API封裝

---

## 🤖 AI 輔助開發技巧

### 使用AI幫助程式碼開發
```
提示詞示例：
"幫我實現一個BiLSTM情感分析模型的forward函數"
"這段程式碼的時間複雜度是多少？如何優化？"
"為這個NER模型添加CRF層"
```

### 使用AI進行調試
```
"訓練損失不下降，可能的原因和解決方案？"
"為什麼我的LSTM會出現梯度爆炸？"
"如何可視化注意力權重？"
```

### 使用AI優化性能
```
"如何加速RNN訓練？"
"batch_first=True和False有什麼區別？"
"pack_padded_sequence如何使用？"
```

---

## 📊 項目評估標準

### 程式碼品質
- [ ] 程式碼結構清晰
- [ ] 有適當的註釋
- [ ] 遵循PEP 8規範
- [ ] 模塊化設計

### 功能完整性
- [ ] 資料預處理完整
- [ ] 訓練流程穩定
- [ ] 評估方法正確
- [ ] 可視化清晰

### 性能表現
- [ ] 達到基準性能
- [ ] 訓練效率合理
- [ ] 資源佔用可接受

### 文檔質量
- [ ] README完整
- [ ] 使用說明清楚
- [ ] 依賴項明確
- [ ] 有示例和輸出

---

## 🎯 學習路線圖

```
Week 1-2: 古詩生成器
  ↓
Week 3-4: 電影評論情感分析
  ↓
Week 5-6: FAQ問答機器人
  ↓
Week 7-9: 新聞摘要生成
  ↓
Week 10-12: 中文NER
  ↓
Week 13+: 高級項目（選一個深入）
```

**總計**: 3-4個月完整學習

---

## 📦 環境配置

### 依賴安裝
```bash
pip install torch torchvision
pip install numpy pandas matplotlib
pip install scikit-learn
pip install jieba  # 中文分詞
pip install rouge-score  # 摘要評估
pip install seqeval  # NER評估
```

### GPU支持
```bash
# 檢查CUDA
nvidia-smi

# 安裝GPU版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔗 資源鏈接

### 資料集
- [唐詩三百首](https://github.com/chinese-poetry/chinese-poetry)
- [IMDb Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)
- [People's Daily NER](https://github.com/OYE93/Chinese-NLP-Corpus)
- [WMT Translation](https://www.statmt.org/wmt21/)

### 預訓練模型
- [Word2Vec中文](https://github.com/Embedding/Chinese-Word-Vectors)
- [GloVe](https://nlp.stanford.edu/projects/glove/)

### 工具
- [Hugging Face](https://huggingface.co/)
- [Weights & Biases](https://wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

---

## 💬 討論和交流

遇到問題？歡迎：
- 提交Issue討論
- 分享你的實現
- 貢獻新項目

**記住**: 實踐是最好的老師！開始你的第一個項目吧！🚀
