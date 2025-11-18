# KV Cache 機制

> 深入理解 KV Cache 如何加速 LLM 推論

## 目錄

- [什麼是 KV Cache](#什麼是-kv-cache)
- [工作原理](#工作原理)
- [記憶體分析](#記憶體分析)
- [優化策略](#優化策略)
- [實作範例](#實作範例)

## 什麼是 KV Cache

KV Cache 是一種在自迴歸生成過程中快取 Attention 機制的 Key 和 Value 張量的技術，避免重複計算已生成 token 的 KV 值。

### 核心概念

在標準的 Transformer 解碼過程中：
1. 每生成一個新 token
2. 需要對整個序列（包括已生成的所有 token）計算 Attention
3. 重新計算所有 token 的 K 和 V 矩陣

**問題**：已生成 token 的 K 和 V 不會改變，重複計算浪費資源！

**解決方案**：KV Cache
- 快取已生成 token 的 K 和 V
- 新 token 只需計算自己的 K 和 V
- 與快取的 KV 拼接後進行 Attention

### 為什麼重要？

**速度提升**：
- 避免重複計算，顯著減少計算量
- 對長序列生成，加速比可達 **10x-100x**

**權衡**：
- 空間換時間
- 需要額外的 GPU 記憶體存儲 KV Cache

## 工作原理

### Attention 機制回顧

標準 Scaled Dot-Product Attention：

```python
Q = X @ W_q  # Query
K = X @ W_k  # Key
V = X @ W_v  # Value

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

### 自迴歸生成過程

#### 無 KV Cache（低效）

```python
# 時間步 t=0: 生成第一個 token
input_ids = [token_0]
Q_0, K_0, V_0 = compute(input_ids)
output_0 = attention(Q_0, K_0, V_0)

# 時間步 t=1: 生成第二個 token
input_ids = [token_0, token_1]
Q_01, K_01, V_01 = compute(input_ids)  # ❌ 重新計算 token_0 的 KV
output_1 = attention(Q_01, K_01, V_01)

# 時間步 t=2: 生成第三個 token
input_ids = [token_0, token_1, token_2]
Q_012, K_012, V_012 = compute(input_ids)  # ❌ 重新計算 token_0, token_1 的 KV
output_2 = attention(Q_012, K_012, V_012)

# ... 每步都重新計算所有歷史 token 的 KV
```

**時間複雜度**：O(n²)，其中 n 是序列長度

#### 有 KV Cache（高效）

```python
# 時間步 t=0: 生成第一個 token
input_ids = [token_0]
Q_0, K_0, V_0 = compute(token_0)
output_0 = attention(Q_0, K_0, V_0)
cache = {"K": K_0, "V": V_0}  # 快取

# 時間步 t=1: 生成第二個 token
input_ids = [token_1]  # ✅ 只處理新 token
Q_1, K_1, V_1 = compute(token_1)
K_01 = concat(cache["K"], K_1)  # 從快取拼接
V_01 = concat(cache["V"], V_1)
output_1 = attention(Q_1, K_01, V_01)
cache = {"K": K_01, "V": V_01}  # 更新快取

# 時間步 t=2: 生成第三個 token
input_ids = [token_2]  # ✅ 只處理新 token
Q_2, K_2, V_2 = compute(token_2)
K_012 = concat(cache["K"], K_2)  # 從快取拼接
V_012 = concat(cache["V"], V_2)
output_2 = attention(Q_2, K_012, V_012)
cache = {"K": K_012, "V": V_012}  # 更新快取

# ... 每步只計算新 token 的 KV
```

**時間複雜度**：O(n)，線性增長

### 視覺化對比

```
無 KV Cache:
t=0: [T0]           -> 計算 1 個 token 的 KV
t=1: [T0, T1]       -> 計算 2 個 token 的 KV  (T0 重複)
t=2: [T0, T1, T2]   -> 計算 3 個 token 的 KV  (T0, T1 重複)
t=3: [T0, T1, T2, T3] -> 計算 4 個 token 的 KV  (T0, T1, T2 重複)
總計算: 1 + 2 + 3 + 4 = 10 次

有 KV Cache:
t=0: [T0]        -> 計算 T0，快取
t=1: [T1]        -> 計算 T1，與快取拼接
t=2: [T2]        -> 計算 T2，與快取拼接
t=3: [T3]        -> 計算 T3，與快取拼接
總計算: 1 + 1 + 1 + 1 = 4 次
```

## 記憶體分析

### KV Cache 記憶體佔用

**單層 KV Cache 大小**：

```
Memory = 2 × batch_size × seq_length × hidden_dim × precision_bytes
```

- `2`：K 和 V 兩個矩陣
- `batch_size`：批次大小
- `seq_length`：序列長度
- `hidden_dim`：隱藏層維度
- `precision_bytes`：精度（FP32=4, FP16=2, INT8=1）

**多層模型**：

```
Total_Memory = num_layers × 2 × batch_size × seq_length × hidden_dim × precision_bytes
```

### 實際案例

**LLaMA-7B**：
- `num_layers` = 32
- `hidden_dim` = 4096
- `precision` = FP16 (2 bytes)
- `batch_size` = 1
- `seq_length` = 2048

```python
memory = 32 × 2 × 1 × 2048 × 4096 × 2
       = 1,073,741,824 bytes
       = 1 GB
```

對於 batch_size=8, seq_length=4096：

```python
memory = 32 × 2 × 8 × 4096 × 4096 × 2
       = 17.2 GB  # 僅 KV Cache！
```

### 記憶體佔用對比

| 模型 | 層數 | Hidden Dim | Seq Len | Batch | KV Cache (FP16) |
|------|------|------------|---------|-------|-----------------|
| GPT-2 | 12 | 768 | 1024 | 1 | 36 MB |
| GPT-2 Large | 36 | 1280 | 1024 | 1 | 180 MB |
| LLaMA-7B | 32 | 4096 | 2048 | 1 | 1 GB |
| LLaMA-13B | 40 | 5120 | 2048 | 1 | 1.6 GB |
| LLaMA-70B | 80 | 8192 | 2048 | 1 | 5.1 GB |
| LLaMA-7B | 32 | 4096 | 4096 | 8 | 16 GB |

## 優化策略

### 1. 多查詢注意力 (MQA - Multi-Query Attention)

**標準 MHA (Multi-Head Attention)**：
- 每個 head 都有獨立的 K 和 V
- KV Cache 大小：`num_heads × hidden_dim`

**MQA**：
- 所有 head 共享同一組 K 和 V
- KV Cache 大小：`hidden_dim`（減少 num_heads 倍）

```python
# MHA
Q_heads = [Q1, Q2, ..., Qh]  # h 個不同的 Q
K_heads = [K1, K2, ..., Kh]  # h 個不同的 K
V_heads = [V1, V2, ..., Vh]  # h 個不同的 V

# MQA
Q_heads = [Q1, Q2, ..., Qh]  # h 個不同的 Q
K = K_shared  # 1 個共享的 K
V = V_shared  # 1 個共享的 V
```

**優勢**：
- KV Cache 減少到 1/num_heads
- 推論速度更快
- 記憶體大幅節省

**劣勢**：
- 表達能力略微下降
- 需要重新訓練模型

**使用模型**：PaLM, StarCoder

### 2. 分組查詢注意力 (GQA - Grouped-Query Attention)

MHA 和 MQA 的折衷方案。

**GQA**：
- 將 heads 分組
- 每組共享一組 K 和 V

```python
# 假設 32 個 heads，4 個 groups
Group1: Q1-Q8   -> 共享 K1, V1
Group2: Q9-Q16  -> 共享 K2, V2
Group3: Q17-Q24 -> 共享 K3, V3
Group4: Q25-Q32 -> 共享 K4, V4
```

**優勢**：
- 平衡性能和效率
- KV Cache 減少到原來的 num_groups/num_heads
- 精度損失小於 MQA

**使用模型**：LLaMA-2

### 3. PagedAttention

受虛擬記憶體啟發的 KV Cache 管理。

**核心思想**：
- 將 KV Cache 分成固定大小的 blocks（類似分頁）
- 按需分配和釋放 blocks
- 不同請求可以共享相同的 blocks（如共享 prompt）

**優勢**：
- 接近零浪費的記憶體管理
- 支援動態批次調整
- 記憶體利用率提升 2-4x

**實現**：vLLM

### 4. KV Cache 量化

對 KV Cache 進行量化以減少記憶體。

**方法**：
- FP16 → INT8：記憶體減少 50%
- FP16 → INT4：記憶體減少 75%

**挑戰**：
- 激活值量化比權重量化更敏感
- 需要動態量化策略

### 5. 流式 KV Cache

對於長序列，使用滑動窗口或流式處理。

**策略**：
- **滑動窗口**：只保留最近 N 個 token 的 KV
- **選擇性保留**：保留重要 token（如開頭的 system prompt）
- **壓縮**：對舊的 KV 進行壓縮

## 實作範例

### 範例 1：基礎 KV Cache

[01_kv_cache_basic.py](./01_kv_cache_basic.py)

- 實現簡單的 KV Cache 機制
- 對比有/無 Cache 的性能
- 記憶體使用分析

### 範例 2：KV Cache 基準測試

[02_kv_cache_benchmark.py](./02_kv_cache_benchmark.py)

- 不同序列長度的性能測試
- 延遲和吞吐量對比
- 記憶體擴展性分析

### 範例 3：進階優化

[03_advanced_kv_cache.py](./03_advanced_kv_cache.py)

- 實現滑動窗口 KV Cache
- 選擇性保留策略
- 記憶體優化技巧

## 使用 KV Cache 的最佳實踐

### 何時啟用 KV Cache

✅ **應該啟用**：
- 長文本生成（>50 tokens）
- 對話系統（多輪對話）
- 代碼生成
- 文章寫作

❌ **可以禁用**：
- 短文本生成（<20 tokens）
- 極度記憶體受限
- 單次分類任務（無生成）

### 配置建議

```python
# Hugging Face Transformers
model.config.use_cache = True  # 啟用 KV Cache（預設）

# 生成時
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    use_cache=True,  # 確保啟用
    ...
)

# 對話場景：重用 past_key_values
past = None
for turn in conversation:
    outputs = model(input_ids, past_key_values=past)
    past = outputs.past_key_values  # 保留給下一輪
```

### 記憶體管理

```python
import torch

# 1. 清理不需要的 cache
past_key_values = None
torch.cuda.empty_cache()

# 2. 使用 FP16 減少 cache 大小
model = model.half()

# 3. 批次生成時控制 batch_size
# 避免: batch_size 過大導致 OOM
max_batch_size = calculate_max_batch(
    model_size, max_seq_length, gpu_memory
)

# 4. 監控記憶體使用
import torch
print(f"KV Cache 記憶體: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## 常見問題

### Q1: KV Cache 對精度有影響嗎？

**答**：沒有。KV Cache 只是避免重複計算，不改變計算邏輯，輸出完全相同。

### Q2: 為什麼有時候 KV Cache 反而變慢？

**答**：
- 短序列：Cache 的拼接開銷 > 重複計算開銷
- 記憶體帶寬：大量的 Cache 讀寫可能成為瓶頸
- 實現問題：某些庫的 Cache 實現不夠優化

### Q3: 多個請求如何共享 KV Cache？

**答**：使用 **PagedAttention**（vLLM）：
- 相同 prompt 的請求共享 blocks
- 節省記憶體和計算
- 詳見：[../4.vLLM-部署/](../4.vLLM-部署/)

### Q4: KV Cache 可以持久化嗎？

**答**：理論上可以，但實踐中較少：
- 保存到磁碟會很大
- 加載速度慢
- 通常按需計算更合適
- 例外：固定 system prompt 可以預計算並複用

## 參考資源

### 論文

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - MQA
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - GQA

### 工具和庫

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention 實現
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) - use_cache 參數
- [FlexGen](https://github.com/FMInference/FlexGen) - 高吞吐量 LLM 推論

### 博客文章

- [KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)
- [Understanding KV Cache in LLM Inference](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/)

---

**下一步**：實作 [01_kv_cache_basic.py](./01_kv_cache_basic.py) 🚀
