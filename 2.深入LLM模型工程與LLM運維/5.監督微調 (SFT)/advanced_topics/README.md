# SFT 進階主題

本目錄包含監督微調 (SFT) 的進階主題和技術，適合已經掌握基礎知識並希望深入學習的讀者。

## 目錄

### 1. [多任務學習 (Multi-Task Learning)](./multi_task_learning.md)

學習如何在單一模型上訓練多個相關任務，提升泛化能力和資源效率。

**涵蓋內容：**
- 多任務學習的優勢和挑戰
- 任務混合和採樣策略
- 溫度採樣和任務提示
- 數據準備和平衡技術
- 完整實作範例
- 任務衝突和負遷移的解決方案

**適合場景：**
- 需要單一模型處理多種任務
- 希望提升模型泛化能力
- 降低部署和維護成本

---

### 2. [持續學習與災難性遺忘 (Continual Learning)](./continual_learning.md)

深入了解如何在學習新任務時保持舊知識，避免災難性遺忘。

**涵蓋內容：**
- 災難性遺忘的原理和表現
- 經驗重放 (Experience Replay)
- 正則化方法 (EWC, LwF)
- 參數隔離方法 (Adapter Tuning)
- 混合策略和完整實作
- 評估指標和監控方法

**適合場景：**
- 需要持續更新模型
- 無法重新訓練整個模型
- 需要保持模型的通用能力

---

## 學習路徑

### 初學者路徑

1. 先學習主目錄的基礎內容
2. 理解單任務 SFT 的完整流程
3. 然後開始學習進階主題

### 進階路徑

**如果你想要：**

- **提升資源效率** → 學習多任務學習
- **持續更新模型** → 學習持續學習
- **同時處理多個場景** → 兩者都學習

---

## 實踐建議

### 多任務學習

**何時使用：**
- ✓ 有多個相關任務需要處理
- ✓ 希望單一模型服務多個場景
- ✓ 希望任務間互相促進

**注意事項：**
- 選擇相關任務進行組合
- 使用溫度採樣平衡任務
- 監控各任務的性能

### 持續學習

**何時使用：**
- ✓ 需要定期添加新任務
- ✓ 無法從頭重新訓練
- ✓ 需要保持舊任務性能

**注意事項：**
- 選擇合適的防遺忘策略
- 保存少量歷史數據用於重放
- 定期評估所有歷史任務

---

## 工具和資源

### 推薦庫

1. **[Avalanche](https://github.com/ContinualAI/avalanche)**
   - 持續學習的完整框架
   - 提供多種策略和評估工具

2. **[PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)**
   - Adapter、LoRA 等方法
   - 適合參數隔離方法

3. **[Transformers](https://github.com/huggingface/transformers)**
   - 基礎的模型訓練框架

### 評估工具

```python
# 使用我們提供的監控器
from continual_learning import ContinualLearningMonitor

monitor = ContinualLearningMonitor()
# ... 訓練過程中記錄指標
monitor.plot_learning_curve()
```

---

## 進階技巧

### 1. 結合多種策略

不要只使用單一方法，結合多種策略效果更好：

```python
# 推薦組合
strategies = ["replay", "ewc", "lwf"]

# 最小組合（資源受限）
strategies = ["replay"]

# 最佳組合（充足資源）
strategies = ["replay", "ewc", "lwf", "adapter"]
```

### 2. 動態調整權重

根據任務性能動態調整：

```python
# 損失高的任務獲得更多訓練
task_weights = adjust_by_performance(task_losses)
```

### 3. 使用 AI 輔助

利用 AI 生成重放數據：

```python
from data_preparation_tools.ai_assisted_data_generator import AIDataGenerator

# 為舊任務生成偽樣本
generator = AIDataGenerator()
pseudo_samples = generator.generate_examples_from_topic(
    topic=old_task_description,
    num_examples=100
)
```

---

## 常見問題

### Q1: 多任務學習和持續學習的區別？

**多任務學習**：同時訓練所有任務，任務已知且固定

**持續學習**：依次訓練任務，任務可能持續增加

### Q2: 我應該使用哪種防遺忘策略？

**推薦決策樹：**

```
能保存歷史數據？
├─ 是 → Replay + EWC（推薦）
└─ 否 → EWC + LwF

任務差異大？
├─ 是 → Adapter Tuning
└─ 否 → Replay + EWC

計算資源充足？
├─ 是 → Replay + EWC + LwF（最佳）
└─ 否 → 只用 EWC（最輕量）
```

### Q3: 如何知道發生了災難性遺忘？

**監控指標：**
- 舊任務性能下降 > 10%
- 遺忘度 (Forgetting Measure) > 0.1
- 向後遷移 (BWT) < -0.1

### Q4: 多任務學習的任務比例如何設置？

**推薦方法：**
1. 使用溫度採樣（T=0.7）
2. 根據任務重要性手動調整
3. 根據任務性能動態調整

---

## 實踐案例

### 案例 1: 客服機器人的多任務訓練

```python
# 任務：退換貨、物流查詢、產品諮詢
task_datasets = {
    "refund": refund_data,
    "shipping": shipping_data,
    "product": product_data
}

# 使用溫度採樣平衡任務
learner = MultiTaskLearner(temperature=0.7)
learner.train(task_datasets)
```

### 案例 2: 持續更新的代碼助手

```python
# 初始：Python 代碼生成
learner.train_task(python_data, "python")

# 添加：JavaScript 支持
learner.train_task(javascript_data, "javascript")

# 添加：代碼審查
learner.train_task(review_data, "review")

# 評估所有任務
learner.evaluate_all_tasks()
```

---

## 下一步

學完進階主題後，建議：

1. **實踐項目**：在真實場景中應用這些技術
2. **閱讀論文**：深入理解理論基礎
3. **參與社區**：分享經驗，學習他人實踐

---

## 參考文獻

### 多任務學習

- [An Overview of Multi-Task Learning](https://ruder.io/multi-task/)
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
- [FLAN: Finetuned Language Models](https://arxiv.org/abs/2109.01652)

### 持續學習

- [Continual Learning Survey](https://arxiv.org/abs/1909.08383)
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [Experience Replay](https://arxiv.org/abs/1811.11682)

---

## 貢獻

歡迎提交 PR 添加更多進階主題：

- 元學習 (Meta-Learning)
- 課程學習 (Curriculum Learning)
- 主動學習 (Active Learning)
- 聯邦學習 (Federated Learning)
