# 模型壓縮與優化練習題

## 📝 練習概覽

本練習集包含從基礎到進階的實作任務，幫助你掌握模型壓縮與優化技術。

**難度等級**：
- 🟢 初級：基礎操作，適合新手
- 🟡 中級：需要理解原理
- 🔴 高級：需要深入知識和實戰經驗

---

## 🟢 初級練習

### 練習 1：基礎量化 (Dynamic Quantization)

**目標**：使用 PyTorch 對簡單模型進行動態量化。

**任務**：
1. 建立一個 3 層的 MLP 模型
2. 訓練模型完成簡單分類任務
3. 應用動態量化
4. 比較量化前後的模型大小和推論速度

**提示程式碼**：
```python
import torch
import torch.nn as nn
import torch.quantization as quant

# TODO: 實現以下功能
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定義 3 層 MLP
        pass

    def forward(self, x):
        # 實現前向傳播
        pass

# 1. 訓練模型
model = SimpleMLP()
# TODO: 訓練邏輯

# 2. 量化
quantized_model = quant.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# 3. 比較
# TODO: 比較大小和速度
```

**檢查點**：
- [ ] 量化後模型大小是否減少約 4 倍？
- [ ] 推論速度是否有提升？
- [ ] 精度損失是否在可接受範圍（< 1%）？

**擴展挑戰**：
- 嘗試靜態量化（需要校準資料）
- 比較不同量化方法的效果

---

### 練習 2：使用 bitsandbytes 進行 8-bit 量化

**目標**：載入 Hugging Face 模型並應用 8-bit 量化。

**任務**：
1. 載入 GPT-2 模型（FP16）
2. 使用 bitsandbytes 載入 8-bit 量化版本
3. 比較顯存使用
4. 測試生成品質

**程式碼模板**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"

# 1. FP16 模型
print("載入 FP16 模型...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
# TODO: 記錄顯存使用

# 2. 8-bit 模型
print("載入 8-bit 模型...")
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
# TODO: 記錄顯存使用

# 3. 測試生成
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Once upon a time"

# TODO: 生成文字並比較
```

**檢查點**：
- [ ] 8-bit 模型顯存是否約為 FP16 的一半？
- [ ] 生成文字質量是否相似？
- [ ] 能否解釋為什麼顯存減少了？

---

### 練習 3：llama.cpp 基礎使用

**目標**：使用 llama.cpp 在 CPU 上運行量化模型。

**任務**：
1. 安裝 llama.cpp
2. 下載並轉換 TinyLlama 模型
3. 量化為 Q4_K_M
4. 測試推論速度

**步驟**：
```bash
# 1. 克隆並編譯
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 2. 轉換模型（假設已下載 TinyLlama）
# TODO: 運行轉換命令

# 3. 量化
# TODO: 運行量化命令

# 4. 推理測試
# TODO: 運行推理並記錄速度
```

**檢查點**：
- [ ] 成功轉換模型為 GGUF 格式？
- [ ] 量化後模型大小約為原始的 1/4？
- [ ] 能夠在 CPU 上運行推理？

---

## 🟡 中級練習

### 練習 4：QLoRA 微調

**目標**：使用 QLoRA 微調 LLaMA-7B 模型。

**任務**：
1. 使用 4-bit 量化載入 LLaMA-7B
2. 配置 LoRA 適配器
3. 在 Alpaca 資料集子集上微調
4. 保存和載入 LoRA 適配器

**程式碼框架**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

# 1. 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 2. 載入模型
model_name = "meta-llama/Llama-2-7b-hf"
# TODO: 載入量化模型

# 3. 準備 LoRA 訓練
# TODO: 配置 LoRA

# 4. 載入資料
# TODO: 準備 Alpaca 資料集

# 5. 訓練
# TODO: 配置 TrainingArguments 和 Trainer

# 6. 保存 LoRA
# TODO: 保存適配器

# 7. 測試載入
# TODO: 載入並測試
```

**檢查點**：
- [ ] 訓練顯存是否在 10GB 以內（單 GPU）？
- [ ] LoRA 適配器大小是否只有幾 MB？
- [ ] 微調後模型是否有改善？
- [ ] 能否成功載入並合併 LoRA？

**進階挑戰**：
- 嘗試不同的 `r` 值（4, 8, 16），比較效果
- 實驗不同的 `target_modules` 組合
- 使用自己的資料集微調

---

### 練習 5：混合精度推理

**目標**：實現混合精度推理系統。

**任務**：
1. 識別模型中的敏感層
2. 為不同層分配不同精度
3. 實現混合精度推理
4. 評估性能和精度

**實作指南**：
```python
import torch
import torch.nn as nn

# 1. 層敏感度分析
def analyze_layer_sensitivity(model, test_data):
    """分析每層對量化的敏感度"""
    sensitivity_map = {}

    # TODO: 實現逐層量化測試
    # 提示：
    # - 遍歷每個線性層
    # - 量化單層
    # - 評估性能變化
    # - 恢復層

    return sensitivity_map

# 2. 配置混合精度
def create_mixed_precision_config(sensitivity_map, target_compression=2.0):
    """根據敏感度建立混合精度配置"""
    precision_config = {}

    # TODO: 實現自動配置生成
    # 提示：
    # - 敏感層使用 FP16
    # - 穩健層使用 INT8
    # - 考慮壓縮目標

    return precision_config

# 3. 應用混合精度
def apply_mixed_precision(model, precision_config):
    """應用混合精度配置"""
    # TODO: 實現模型轉換
    pass

# 測試
model = load_model()
sensitivity = analyze_layer_sensitivity(model, test_data)
config = create_mixed_precision_config(sensitivity)
mixed_model = apply_mixed_precision(model, config)
```

**檢查點**：
- [ ] 能否正確識別敏感層？
- [ ] 混合精度配置是否合理？
- [ ] 性能是否達到目標壓縮比？
- [ ] 精度損失是否小於均勻量化？

---

### 練習 6：GPTQ 量化實踐

**目標**：使用 GPTQ 對模型進行 4-bit 量化。

**任務**：
1. 準備校準資料集
2. 使用 AutoGPTQ 量化模型
3. 保存量化模型
4. 使用 vLLM 部署

**程式碼模板**：
```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

# 1. 準備校準資料
def prepare_calibration_data(dataset_name="c4", n_samples=128):
    """準備校準資料集"""
    # TODO: 載入和預處理資料
    pass

# 2. 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

# 3. 載入並量化
model_name = "facebook/opt-125m"  # 使用小模型測試
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

# TODO: 執行量化
calibration_data = prepare_calibration_data()
model.quantize(calibration_data)

# 4. 保存
# TODO: 保存量化模型

# 5. 測試載入和推理
# TODO: 載入並測試
```

**檢查點**：
- [ ] 量化過程是否成功完成？
- [ ] 量化後模型大小約為原始的 1/4？
- [ ] 困惑度變化是否在 1% 以內？
- [ ] 推論速度是否有提升？

---

## 🔴 高級練習

### 練習 7：自定義量化方法

**目標**：實現自定義的量化演算法。

**任務**：
1. 實現非對稱量化
2. 實現逐通道量化
3. 比較不同量化粒度的效果
4. 實現量化感知訓練 (QAT)

**實作框架**：
```python
import torch
import torch.nn as nn

class CustomQuantizer:
    """自定義量化器"""

    def __init__(self, bits=8, symmetric=True, per_channel=False):
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

    def quantize(self, tensor):
        """量化張量"""
        # TODO: 實現量化邏輯
        # 1. 計算 scale 和 zero_point
        # 2. 量化
        # 3. 返回量化後的張量和參數
        pass

    def dequantize(self, tensor_quant, scale, zero_point):
        """反量化"""
        # TODO: 實現反量化
        pass

class QuantizedLinear(nn.Module):
    """量化線性層"""

    def __init__(self, linear, quantizer):
        super().__init__()
        self.quantizer = quantizer

        # TODO: 量化權重
        # TODO: 存儲量化參數

    def forward(self, x):
        """前向傳播"""
        # TODO: 實現量化推理
        pass

# 測試
quantizer = CustomQuantizer(bits=8, per_channel=True)
# TODO: 測試量化效果
```

**檢查點**：
- [ ] 對稱和非對稱量化是否正確實現？
- [ ] 逐通道量化精度是否優於逐張量？
- [ ] 量化誤差是否符合預期？

---

### 練習 8：生產級 API 部署

**目標**：部署高性能的量化模型 API 服務。

**任務**：
1. 使用 vLLM 部署量化模型
2. 實現負載均衡
3. 添加監控和日誌
4. 實現多租戶支持（LoRA）

**系統架構**：
```python
# 1. vLLM 服務
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import prometheus_client
import logging

app = FastAPI()

# TODO: 初始化 vLLM
# TODO: 配置 Prometheus 監控
# TODO: 配置日誌

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    """生成端點"""
    # TODO: 實現生成邏輯
    # TODO: 添加錯誤處理
    # TODO: 記錄指標
    pass

# 2. 負載均衡配置 (nginx.conf)
# TODO: 編寫 nginx 配置

# 3. Docker 配置
# TODO: 編寫 Dockerfile

# 4. Kubernetes 配置
# TODO: 編寫 k8s deployment.yaml
```

**檢查點**：
- [ ] API 是否能處理高並發？
- [ ] 監控指標是否正常記錄？
- [ ] 錯誤處理是否完善？
- [ ] 是否實現了自動擴展？

---

### 練習 9：端到端優化流程

**目標**：優化一個完整的 LLM 應用。

**任務**：
1. 選擇一個基礎模型（如 LLaMA-13B）
2. 分析目標應用的性能要求
3. 設計優化策略（量化 + LoRA + 混合精度）
4. 實施優化並測試
5. 撰寫性能報告

**流程**：
```
1. 需求分析
   - 延遲要求：？
   - 吞吐量要求：？
   - 硬體限制：？
   - 精度要求：？

2. 基準測試
   - 原始模型性能
   - 瓶頸分析

3. 優化策略
   - 量化方法選擇
   - LoRA 配置
   - 推論優化

4. 實施
   - 量化模型
   - 訓練 LoRA
   - 部署優化

5. 驗證
   - 性能測試
   - 精度評估
   - A/B 測試

6. 報告
   - 優化前後對比
   - 成本分析
   - 經驗總結
```

**提交內容**：
- [ ] 完整程式碼
- [ ] 性能測試報告
- [ ] 優化決策說明
- [ ] 未來改進建議

---

## 🏆 挑戰項目

### 項目 1：多模型推論服務

**描述**：構建支持多個量化模型的推論服務，用戶可以選擇不同的模型和精度。

**要求**：
- 支持至少 3 個不同大小的模型
- 支持 FP16、INT8、INT4 三種精度
- 實現智能路由（根據負載和精度要求）
- 提供統一的 API 接口

---

### 項目 2：移動端 LLM 應用

**描述**：開發一個在手機上運行的 LLM 應用。

**要求**：
- iOS 或 Android 平台
- 使用量化模型（< 2GB）
- 推論延遲 < 200ms
- 提供友好的 UI

---

### 項目 3：自動化壓縮工具

**描述**：開發自動化模型壓縮工具，給定模型和約束，自動選擇最佳壓縮策略。

**要求**：
- 支持多種量化方法
- 自動敏感度分析
- 自動混合精度配置
- 提供性能預測
- CLI 和 Python API

---

## 📊 評分標準

### 初級練習
- 程式碼正確性：40%
- 結果準確性：30%
- 程式碼品質：20%
- 文檔完整性：10%

### 中級練習
- 實現完整性：30%
- 性能優化：30%
- 程式碼品質：20%
- 實驗分析：20%

### 高級練習
- 系統設計：25%
- 實現質量：25%
- 性能表現：25%
- 創新性：15%
- 文檔報告：10%

---

## 💡 學習資源

**推薦閱讀**：
- QLoRA 論文：https://arxiv.org/abs/2305.14314
- GPTQ 論文：https://arxiv.org/abs/2210.17323
- AWQ 論文：https://arxiv.org/abs/2306.00978

**教程**：
- Hugging Face PEFT 文檔
- vLLM 官方指南
- llama.cpp 實踐教程

**社群**：
- Hugging Face Discord
- r/LocalLLaMA Reddit

---

## ✅ 提交指南

1. **程式碼**：上傳到 GitHub repo
2. **報告**：Markdown 格式，包含：
   - 實驗設置
   - 結果分析
   - 遇到的問題和解決方案
   - 性能資料和圖表
3. **演示**：錄製簡短演示影片（可選）

---

祝你學習順利！如有問題，歡迎在社群討論。
