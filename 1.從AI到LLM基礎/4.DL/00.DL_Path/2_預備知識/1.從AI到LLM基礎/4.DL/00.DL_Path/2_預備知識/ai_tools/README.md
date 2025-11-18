# AI 輔助學習工具集

## 📚 工具概覽

本目錄包含三個強大的 AI 輔助學習工具，幫助你更高效地掌握深度學習的數學基礎。

### 🔧 工具列表

| 工具 | 文件 | 功能 | 使用場景 |
|------|------|------|----------|
| 練習生成器 | `exercise_generator.py` | 自動生成個性化練習題 | 需要額外練習時 |
| 概念可視化器 | `visualizer.py` | 可視化數學概念 | 理解抽象概念時 |
| 進度追蹤器 | `progress_tracker.py` | 追蹤學習進度 | 規劃學習路徑時 |

---

## 🎯 1. 練習生成器 (exercise_generator.py)

### 功能特性

- ✅ 支持多個主題：張量操作、線性代數、微積分、自動微分、概率統計
- ✅ 四個難度等級：easy, medium, hard, expert
- ✅ 自動生成隨機參數
- ✅ 提供詳細的解答和提示
- ✅ 支持批量生成和導出

### 使用方法

#### 基本用法

```bash
# 生成 5 道中等難度的線性代數練習題
python exercise_generator.py --topic linear_algebra --difficulty medium --count 5

# 生成並保存到文件
python exercise_generator.py --topic ndarray --difficulty easy --count 10 --output exercises.json
```

#### 支持的主題

- `ndarray`: 張量操作（創建、變形、索引、廣播）
- `linear_algebra`: 線性代數（向量、矩陣、範數、點積）
- `calculus`: 微積分（導數、梯度、偏導數）
- `autograd`: 自動微分（計算圖、反向傳播）
- `probability`: 概率統計（分佈、期望、方差）

#### 難度等級

- `easy`: 適合初學者，側重基礎概念
- `medium`: 適合有一定基礎的學習者
- `hard`: 適合進階學習者，包含複雜計算
- `expert`: 適合專家級，需要深入理解

### 示例輸出

```
📝 練習 1: linear_algebra_medium_1
難度: medium

問題：
給定矩陣 A (3×4)，計算其轉置並驗證 (A^T)^T = A。

概念: 矩陣轉置, 矩陣性質

提示：
  💡 使用 .T 屬性
  💡 使用 torch.equal() 比較

參考解答：
```python
import torch
A = torch.randn(3, 4)
A_T = A.T
A_T_T = A_T.T
print(torch.equal(A, A_T_T))  # True
```

---

## 📊 2. 概念可視化器 (visualizer.py)

### 功能特性

- ✅ 交互式可視化數學概念
- ✅ 高質量圖表生成
- ✅ 支持多種概念的可視化
- ✅ 自動保存圖片

### 使用方法

#### 可視化梯度下降

```bash
python visualizer.py --concept gradient_descent
```

生成的圖表包括：
- 函數曲線和優化路徑
- 收斂曲線

#### 可視化線性變換

```bash
python visualizer.py --concept linear_transformation
```

展示 6 種線性變換：
- 恆等變換
- 縮放變換
- 旋轉變換
- 剪切變換
- 反射變換
- 投影變換

#### 可視化激活函數

```bash
python visualizer.py --concept activation_functions
```

展示常用激活函數：
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- ELU
- GELU

#### 可視化概率分佈

```bash
python visualizer.py --concept probability_distributions
```

展示常見概率分佈：
- 正態分佈
- 均勻分佈
- 伯努利分佈
- 指數分佈
- 二項分佈
- 泊松分佈

#### 可視化矩陣乘法

```bash
python visualizer.py --concept matrix_multiplication
```

直觀展示矩陣乘法過程。

#### 可視化所有概念

```bash
python visualizer.py --concept all
```

---

## 📈 3. 進度追蹤器 (progress_tracker.py)

### 功能特性

- ✅ 記錄學習進度（分數、時間、練習數）
- ✅ 生成可視化報告
- ✅ 分析薄弱環節
- ✅ 提供個性化學習建議
- ✅ 里程碑追蹤
- ✅ 學習筆記功能

### 使用方法

#### 更新進度

```bash
# 更新某個主題的分數
python progress_tracker.py --update --topic ndarray --score 85

# 記錄學習時間
python progress_tracker.py --update --topic linear_algebra --time 3

# 記錄完成的練習數
python progress_tracker.py --update --topic calculus --exercises 10

# 同時更新多項數據
python progress_tracker.py --update --topic autograd --score 78 --time 2 --exercises 5
```

#### 生成學習報告

```bash
python progress_tracker.py --report
```

報告包括：
- 整體學習時間統計
- 各主題的分數和進度
- 已達成的里程碑
- 可視化圖表（雷達圖和柱狀圖）

#### 獲取學習建議

```bash
python progress_tracker.py --suggest
```

建議包括：
- 需要加強的主題
- 推薦的學習順序
- 時間分配建議
- 學習策略建議

#### 添加學習筆記

```bash
python progress_tracker.py --update --topic probability --note "理解了貝葉斯定理的應用"
```

### 里程碑系統

進度追蹤器會自動識別你達成的里程碑：

- 🥉 **入門** (60分): 掌握基礎概念
- 🥈 **熟練** (75分): 能夠獨立完成練習
- 🥇 **精通** (90分): 深入理解原理
- 🏆 **大師** (100分): 完全掌握並能教授他人

---

## 🚀 快速開始

### 安裝依賴

```bash
# 基礎依賴
pip install torch numpy matplotlib

# 額外依賴（用於可視化器）
pip install scipy
```

### 推薦學習流程

1. **第一週**: 使用進度追蹤器規劃學習路徑
2. **學習過程中**: 使用可視化器理解抽象概念
3. **每完成一個主題**: 用練習生成器生成額外練習
4. **每週末**: 更新進度並查看報告

### 示例工作流

```bash
# 1. 開始學習線性代數
python progress_tracker.py --update --topic linear_algebra --time 2

# 2. 可視化線性變換
python visualizer.py --concept linear_transformation

# 3. 生成練習題鞏固
python exercise_generator.py --topic linear_algebra --difficulty medium --count 5

# 4. 完成練習後更新進度
python progress_tracker.py --update --topic linear_algebra --score 82 --exercises 5

# 5. 週末查看報告
python progress_tracker.py --report
```

---

## 💡 使用技巧

### 練習生成器

1. **循序漸進**: 從 easy 開始，逐步提升到 expert
2. **批量生成**: 使用 `--output` 保存練習題，方便離線學習
3. **針對性練習**: 根據薄弱環節選擇主題

### 可視化器

1. **多次觀察**: 同一概念從不同角度可視化
2. **對比學習**: 使用 `--concept all` 生成所有圖表，進行對比
3. **保存圖片**: 圖片會自動保存，方便製作筆記

### 進度追蹤器

1. **及時更新**: 每次學習後立即更新進度
2. **定期回顧**: 每週生成一次報告
3. **記錄筆記**: 重要的理解和心得及時記錄
4. **設定目標**: 為每個主題設定分數目標

---

## 🔧 高級用法

### 自定義練習模板

編輯 `exercise_generator.py`，在對應的生成函數中添加新的練習模板：

```python
templates = {
    'medium': [
        {
            'question': '你的問題...',
            'solution': '你的解答...',
            'hints': ['提示1', '提示2'],
            'concepts': ['概念1', '概念2']
        }
    ]
}
```

### 自定義可視化

在 `visualizer.py` 中添加新的可視化函數：

```python
def visualize_custom_concept(self):
    """自定義可視化"""
    # 你的可視化代碼
    pass
```

### 數據導出

進度數據保存在 `progress_data.json`，可以導出用於：
- 生成詳細的學習報告
- 與他人分享學習經驗
- 備份學習記錄

---

## 📝 常見問題

### Q1: 練習題的答案都正確嗎？

A: 練習題基於常見的學習場景設計，但建議你：
1. 親自驗證答案
2. 嘗試不同的解法
3. 理解背後的原理

### Q2: 如何重置進度追蹤？

A: 刪除 `progress_data.json` 文件即可重新開始。

### Q3: 可以自定義可視化的樣式嗎？

A: 可以！修改 `visualizer.py` 中的 matplotlib 設置：

```python
plt.style.use('your_style')  # 更改樣式
self.figsize = (width, height)  # 更改圖片大小
```

### Q4: 如何批量生成多個主題的練習？

A: 可以使用 shell 腳本：

```bash
#!/bin/bash
for topic in ndarray linear_algebra calculus autograd probability; do
    python exercise_generator.py --topic $topic --difficulty medium --count 5 --output "${topic}_exercises.json"
done
```

---

## 🤝 貢獻

歡迎改進這些工具！可以：

1. 添加新的練習模板
2. 實現新的可視化
3. 改進進度追蹤算法
4. 修復 bug

---

## 📄 授權

MIT License

---

## 🌟 致謝

這些工具基於深度學習社群的集體智慧開發，感謝所有貢獻者！

---

**最後更新**: 2024-11
**維護者**: AI Learning Community
**版本**: v1.0

開始使用這些工具，讓你的學習之旅更高效！🚀
