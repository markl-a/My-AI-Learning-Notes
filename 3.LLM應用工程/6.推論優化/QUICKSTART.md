# 快速開始指南

> 5 分鐘快速上手 LLM 推論優化

## 安裝依賴

### 最小化安裝（快速測試）

```bash
pip install torch transformers accelerate
```

### 完整安裝

```bash
cd 3.LLM應用工程/6.推論優化
pip install -r requirements.txt
```

## 運行第一個範例

### 1. 量化對比（推薦新手）

```bash
cd 1.量化技術
python 05_quantization_comparison.py
```

**預期輸出**：
- FP32/FP16/INT8/INT4 量化對比
- 記憶體使用分析
- 推論速度對比
- 視覺化圖表

### 2. KV Cache 演示

```bash
cd 2.KV-Cache
python 01_kv_cache_basic.py
```

**預期輸出**：
- KV Cache 工作原理演示
- 有/無 Cache 性能對比
- 不同生成長度的加速效果

### 3. AI 輔助優化

```bash
cd 6.AI輔助優化工具
python 01_auto_optimizer.py
```

**預期輸出**：
- 系統資源分析
- 智能優化建議
- 自動生成優化程式碼
- 性能預測

## 驗證安裝

運行綜合測試：

```bash
python test_all.py
```

預期看到：
```
✅ 量化技術 - 基礎量化 語法檢查通過
✅ 量化技術 - GPTQ 量化 語法檢查通過
✅ KV Cache - 基礎演示 語法檢查通過
✅ AI 輔助優化工具 - 自動優化器 語法檢查通過
```

## 學習路徑

### 初學者（第 1 週）

1. 閱讀主 [README.md](./README.md)
2. 運行量化對比: `1.量化技術/05_quantization_comparison.py`
3. 理解 KV Cache: `2.KV-Cache/01_kv_cache_basic.py`
4. 使用 AI 優化器獲取建議

### 進階（第 2-3 週）

5. 學習 GPTQ 量化: `1.量化技術/02_gptq_quantization.py`
6. 探索不同量化方法
7. 閱讀各模組的 README 理論部分

### 實戰（第 4 週+）

8. 運行實戰案例: `7.實戰案例/`
9. 應用到自己的項目
10. 參考最佳實踐指南

## 常見問題

### Q: 沒有 GPU 可以運行嗎？

**A**: 可以！大部分範例都支援 CPU。
- 量化對比會自動使用 CPU
- KV Cache 演示可在 CPU 運行
- 只是速度會較慢

### Q: 需要下載模型嗎？

**A**: 是的，首次運行會自動下載。
- 預設使用 GPT-2（較小，~500MB）
- 可以修改為其他模型
- 確保網路暢通或配置 Hugging Face mirror

### Q: 出現 OOM 錯誤怎麼辦？

**A**: 記憶體不足的解決方案：
```python
# 方案 1: 使用更小的模型
model_name = "gpt2"  # 而不是 "gpt2-large"

# 方案 2: 減小批次大小
batch_size = 1

# 方案 3: 啟用更激進的量化
load_in_4bit = True
```

### Q: 如何獲取 API Key（AI 輔助工具）？

**A**: AI 輔助工具的 API Key 是可選的：
- 不提供 API Key 會使用本地規則引擎
- 功能完整，只是建議較為基礎
- 如需 AI 增強，可申請 OpenAI 或 Anthropic API Key

## 最佳實踐

### 1. 從小模型開始

```bash
# ✅ 推薦：先用小模型測試
python 01_basic_quantization.py  # 使用 GPT-2

# ❌ 不推薦：直接用大模型
# model_name = "meta-llama/Llama-2-70b-hf"  # 可能 OOM
```

### 2. 逐步優化

```bash
# 步驟 1: 建立基準
python test_baseline.py

# 步驟 2: 應用量化
python test_quantized.py

# 步驟 3: 對比結果
python compare_results.py
```

### 3. 保存實驗結果

```bash
# 建立實驗目錄
mkdir experiments/exp_001

# 保存結果
python your_script.py > experiments/exp_001/log.txt
```

## 進階配置

### 使用 GPU

```python
import torch

# 檢查 GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("使用 CPU")
```

### 使用多 GPU

```python
# 自動分配到多個 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自動分配
    max_memory={0: "20GB", 1: "20GB"}  # 限制每個 GPU 的記憶體
)
```

### 配置快取目錄

```bash
# 設置 Hugging Face 快取目錄
export HF_HOME=/path/to/cache

# 或在程式碼中
import os
os.environ['HF_HOME'] = '/path/to/cache'
```

## 取得幫助

- 📖 閱讀各模組的 README
- 💬 查看程式碼中的註釋和文檔字串
- 🐛 檢查錯誤資訊並搜尋解決方案
- 📝 參考最佳實踐指南

## 貢獻

發現問題或有改進建議？歡迎提出 Issue 或 Pull Request！

---

**祝你學習順利！** 🚀
