# 資料準備工具集

這個工具集提供了一套完整的 SFT 資料準備工具，包括 AI 輔助資料生成、質量檢查和格式轉換。

## 目錄

1. [AI 輔助資料生成器](#ai-輔助資料生成器)
2. [資料品質檢查器](#資料品質檢查器)
3. [資料格式轉換器](#資料格式轉換器)

---

## AI 輔助資料生成器

### 功能

`ai_assisted_data_generator.py` 使用 LLM API (Claude/GPT) 自動生成高品質的訓練資料。

### 主要特性

- **自動生成訓練樣本**：根據主題和任務類型生成多樣化的訓練資料
- **樣本變體生成**：為現有樣本生成變體，增加資料多樣性
- **支持多種 API**：支持 Anthropic Claude 和 OpenAI GPT
- **可自定義**：靈活配置生成的樣本類型和數量

### 使用方法

#### 1. 安裝依賴

```bash
pip install anthropic openai
```

#### 2. 設置 API 密鑰

```bash
# 使用 Claude
export ANTHROPIC_API_KEY="your-api-key"

# 或使用 OpenAI
export OPENAI_API_KEY="your-api-key"
```

#### 3. 基本使用

```python
from ai_assisted_data_generator import AIDataGenerator, save_examples

# 初始化生成器
generator = AIDataGenerator(provider="anthropic")

# 生成客服對話資料
examples = generator.generate_examples_from_topic(
    topic="電商客服對話",
    num_examples=20,
    example_types=["退換貨諮詢", "物流查詢", "產品問題"]
)

# 保存資料
save_examples(examples, "customer_service_data.json")
```

#### 4. 生成樣本變體

```python
# 為現有樣本生成變體
variations = generator.generate_variations(
    original_example=examples[0],
    num_variations=3
)

save_examples(variations, "variations.json")
```

### 應用場景

1. **快速構建原型資料集**：快速生成初始訓練資料
2. **資料增強**：為現有資料集生成變體
3. **多領域資料生成**：生成不同領域的訓練資料
4. **Few-shot 引導**：基於少量示例生成更多類似資料

### 生成資料示例

```json
[
  {
    "instruction": "客戶詢問如何退貨",
    "input": "我買的商品不滿意，想要退貨",
    "output": "您好，我們支持 7 天無理由退貨。請您提供訂單號，我會協助您處理退貨申請。退貨流程如下：1. 提交退貨申請 2. 等待審核 3. 寄回商品 4. 審核通過後退款",
    "metadata": {
      "topic": "電商客服對話",
      "generated": true
    }
  }
]
```

---

## 資料品質檢查器

### 功能

`data_quality_checker.py` 對訓練資料進行全面的質量檢查，確保資料品質。

### 主要特性

- **重複檢測**：識別重複的訓練樣本
- **格式驗證**：檢查必需字段和資料類型
- **長度分析**：檢測異常長度的樣本
- **空值檢查**：識別空字段
- **多樣性分析**：分析資料的多樣性
- **質量評分**：提供 0-100 的質量分數
- **改進建議**：給出具體的改進建議

### 使用方法

#### 命令行使用

```bash
python data_quality_checker.py your_data.json
```

#### Python 程式碼使用

```python
from data_quality_checker import DataQualityChecker, print_report

# 建立檢查器
checker = DataQualityChecker("your_data.json")

# 執行所有檢查
report = checker.check_all()

# 分析多樣性
diversity = checker.analyze_diversity()

# 打印報告
print_report(report)
```

### 輸出示例

```
檢查資料文件: customer_service_data.json
總樣本數: 100
------------------------------------------------------------

檢查重複樣本...
發現 2 對重複樣本

檢查格式...
發現 0 個格式錯誤

檢查長度分佈...
指令長度: 平均=45.2, 標準差=15.3
輸出長度: 平均=156.7, 標準差=48.9
發現 3 個長度異常

檢查空字段...
發現 1 個空字段

分析資料多樣性...

最常見的指令開頭 (前 10):
  請問如何: 15 次 (15.0%)
  我想: 12 次 (12.0%)
  ...

============================================================
資料品質報告
============================================================

總樣本數: 100
質量分數: 88.0/100

問題統計:
  重複樣本: 2 對
  格式錯誤: 0 個
  長度異常: 3 個
  空字段: 1 個

改進建議:
  1. 發現 2 對重複樣本，建議移除以提高資料多樣性
  2. 發現 3 個長度異常，建議檢查是否為資料錯誤
  3. 發現 1 個空字段，建議填充或移除這些樣本

============================================================
```

### 質量評分標準

- **100 分**：完美資料，無任何問題
- **80-99 分**：高品質，有少量可改進之處
- **60-79 分**：中等質量，需要一些清理
- **< 60 分**：低品質，需要大量改進

---

## 資料格式轉換器

### 功能

`data_formatter.py` 提供多種 SFT 資料格式之間的轉換功能。

### 支持的格式

1. **Alpaca 格式**：`{"instruction": "...", "input": "...", "output": "..."}`
2. **ShareGPT 格式**：`{"conversations": [{"from": "human", "value": "..."}, ...]}`
3. **OpenAI 格式**：`{"messages": [{"role": "user", "content": "..."}, ...]}`

### 主要功能

- **格式轉換**：在不同格式間轉換
- **應用模板**：應用 Alpaca/Vicuna/ChatML 等模板
- **資料分割**：分割訓練集和驗證集
- **資料合併**：合併多個資料集
- **長度過濾**：根據長度過濾樣本
- **資料平衡**：平衡類別分佈

### 使用方法

#### 1. 格式轉換

```python
from data_formatter import DataFormatter

formatter = DataFormatter()

# Alpaca -> ShareGPT
sharegpt_data = formatter.alpaca_to_sharegpt(alpaca_data)

# Alpaca -> OpenAI
openai_data = formatter.alpaca_to_openai(alpaca_data)

# ShareGPT -> OpenAI
openai_data = formatter.sharegpt_to_openai(sharegpt_data)
```

#### 2. 應用聊天模板

```python
# 應用 Alpaca 模板
templated_data = formatter.apply_chat_template(
    data=alpaca_data,
    template_name="alpaca"
)

# 支持的模板：alpaca, vicuna, chatml
```

#### 3. 分割資料集

```python
# 分割訓練集和驗證集
train_data, val_data = formatter.split_train_val(
    data=your_data,
    val_ratio=0.1,  # 10% 驗證集
    shuffle=True
)
```

#### 4. 長度過濾

```python
# 過濾長度異常的樣本
filtered_data = formatter.filter_by_length(
    data=your_data,
    min_length=10,
    max_length=2048,
    field="output"
)
```

#### 5. 資料平衡

```python
# 平衡類別分佈
balanced_data = formatter.balance_dataset(
    data=your_data,
    category_field="category",
    max_per_category=100
)
```

#### 6. 命令行使用

```python
from data_formatter import convert_format

convert_format(
    input_file="input.json",
    output_file="output.json",
    input_format="alpaca",
    output_format="openai"
)
```

### 模板示例

#### Alpaca 模板

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
解釋什麼是機器學習

### Response:
機器學習是人工智慧的一個分支...
```

#### Vicuna 模板

```
A chat between a curious user and an artificial intelligence assistant.

USER: 解釋什麼是機器學習
ASSISTANT: 機器學習是人工智慧的一個分支...
```

#### ChatML 模板

```
<|im_start|>user
解釋什麼是機器學習<|im_end|>
<|im_start|>assistant
機器學習是人工智慧的一個分支...<|im_end|>
```

---

## 完整工作流程示例

### 場景：構建客服機器人訓練資料

```python
from ai_assisted_data_generator import AIDataGenerator, save_examples
from data_quality_checker import DataQualityChecker, print_report
from data_formatter import DataFormatter
import json

# 步驟 1: 使用 AI 生成初始資料
print("步驟 1: 生成訓練資料...")
generator = AIDataGenerator(provider="anthropic")

customer_service_data = generator.generate_examples_from_topic(
    topic="電商客服對話",
    num_examples=100,
    example_types=["退換貨", "物流查詢", "產品諮詢", "投訴處理", "優惠活動"]
)

save_examples(customer_service_data, "raw_data.json")

# 步驟 2: 質量檢查
print("\n步驟 2: 檢查資料品質...")
checker = DataQualityChecker("raw_data.json")
report = checker.check_all()
print_report(report)

# 步驟 3: 清理資料（移除重複和異常）
print("\n步驟 3: 清理資料...")
with open("raw_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 移除重複樣本（根據報告中的索引）
# ... 清理邏輯 ...

# 步驟 4: 格式轉換和資料分割
print("\n步驟 4: 格式化和分割資料...")
formatter = DataFormatter()

# 分割訓練集和驗證集
train_data, val_data = formatter.split_train_val(data, val_ratio=0.1)

# 應用模板
train_templated = formatter.apply_chat_template(train_data, template_name="alpaca")
val_templated = formatter.apply_chat_template(val_data, template_name="alpaca")

# 保存最終資料
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_templated, f, ensure_ascii=False, indent=2)

with open("val_data.json", "w", encoding="utf-8") as f:
    json.dump(val_templated, f, ensure_ascii=False, indent=2)

print(f"\n完成！訓練集: {len(train_data)} 樣本, 驗證集: {len(val_data)} 樣本")
```

---

## 最佳實踐

### 資料生成

1. **分批生成**：分批生成資料，而不是一次生成大量資料
2. **人工審核**：AI 生成的資料需要人工審核
3. **混合來源**：結合 AI 生成和人工標註的資料
4. **迭代改進**：根據模型表現持續改進資料品質

### 品質控制

1. **定期檢查**：定期運行質量檢查工具
2. **設置閾值**：質量分數低於 80 分時需要改進
3. **人工抽查**：隨機抽查樣本進行人工評估
4. **版本控制**：使用 Git 管理資料集版本

### 格式選擇

1. **Alpaca 格式**：簡單任務，單輪對話
2. **ShareGPT 格式**：多輪對話，更自然的交互
3. **OpenAI 格式**：需要系統提示詞的場景

---

## 故障排除

### 常見問題

**Q: AI 生成的資料品質不高？**

A:
- 改進提示詞，提供更具體的要求
- 使用更強大的模型（如 GPT-4 或 Claude Opus）
- 提供示例資料作為參考

**Q: 資料格式轉換後出現錯誤？**

A:
- 檢查輸入資料是否符合預期格式
- 使用質量檢查工具驗證格式
- 查看錯誤日志定位問題

**Q: 質量檢查報告分數很低？**

A:
- 根據建議逐項改進
- 移除重複和異常樣本
- 填充空字段或移除不完整的樣本

---

## 依賴項

```txt
anthropic>=0.18.0
openai>=1.0.0
```

安裝：

```bash
pip install -r requirements.txt
```

---

## 許可證

這些工具是 MIT 許可證下的開源工具。

---

## 貢獻

歡迎提交 Issue 和 Pull Request！
