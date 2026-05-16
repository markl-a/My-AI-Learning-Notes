# 實戰案例

> 端到端的 LLM 推論優化實戰指南

## 簡介

本模組提供完整的實戰案例，展示如何在真實場景中應用推論優化技術。每個案例都包含：
- 📋 場景描述和需求分析
- 🔍 性能基準測試
- ⚡ 逐步優化過程
- 📊 效果對比和分析
- 💡 經驗總結和最佳實踐

## 案例列表

### 案例 1: 聊天機器人優化

**場景**：客服聊天機器人，需要支援 100+ 並發用戶

**挑戰**：
- 低延遲要求（<500ms）
- 有限的 GPU 資源（單張 RTX 3090 24GB）
- 需要保持高品質對話

**優化路徑**：
1. 基準測試（FP16 模型）
2. 應用 AWQ 4-bit 量化
3. 啟用 KV Cache 和 PagedAttention
4. 使用 vLLM 的 continuous batching
5. 性能調優和壓力測試

**詳細內容**：[01_chatbot_optimization.py](./01_chatbot_optimization.py)

### 案例 2: 批次文字處理優化

**場景**：離線批次處理大量文檔（100萬條）

**挑戰**：
- 最大化吞吐量
- 有 8 小時時間窗口
- 保證輸出品質

**優化路徑**：
1. 基準測試（計算處理速度）
2. 優化批次大小
3. 實施資料預處理流水線
4. 多 GPU 並行處理
5. 監控和調優

**詳細內容**：[02_batch_processing.py](./02_batch_processing.py)

### 案例 3: 生產環境部署

**場景**：將優化後的模型部署到生產環境

**內容**：
- Docker 容器化
- 負載均衡配置
- 監控和告警
- 自動擴展策略
- 故障恢復

**詳細內容**：[03_production_deployment.py](./03_production_deployment.py)

## 優化流程模板

### 第一步：需求分析

```python
# 定義需求
requirements = {
    "use_case": "chatbot",
    "latency_p99": 500,  # ms
    "throughput_min": 10,  # QPS
    "quality_threshold": 0.95,  # 相對於基準
    "budget": "single_gpu"
}
```

### 第二步：基準測試

```python
# 運行基準測試
baseline = benchmark_model(
    model_name="meta-llama/Llama-2-7b-hf",
    test_data=test_dataset,
    metrics=["latency", "throughput", "memory", "quality"]
)

print(f"基準性能: {baseline}")
```

### 第三步：選擇優化策略

```python
# 使用 AI 輔助工具
from auto_optimizer import LLMOptimizationAdvisor

advisor = LLMOptimizationAdvisor()
strategy = advisor.recommend(
    baseline_metrics=baseline,
    requirements=requirements
)

print(f"推薦策略: {strategy}")
```

### 第四步：逐步應用優化

```python
# 優化 1: 量化
optimized_v1 = apply_quantization(
    model,
    method=strategy.quantization_method,
    bits=strategy.quantization_bits
)
metrics_v1 = benchmark_model(optimized_v1, test_dataset)

# 優化 2: KV Cache
optimized_v2 = configure_kv_cache(
    optimized_v1,
    enabled=strategy.use_kv_cache
)
metrics_v2 = benchmark_model(optimized_v2, test_dataset)

# 優化 3: 批次處理
optimized_v3 = configure_batching(
    optimized_v2,
    batch_size=strategy.batch_size,
    dynamic=strategy.dynamic_batching
)
metrics_v3 = benchmark_model(optimized_v3, test_dataset)
```

### 第五步：驗證和調優

```python
# A/B 測試
ab_test_results = run_ab_test(
    control=baseline_model,
    treatment=optimized_v3,
    test_data=validation_dataset,
    duration_hours=2
)

# 分析結果
if ab_test_results.meets_requirements(requirements):
    print("✅ 優化成功！")
    deploy_to_production(optimized_v3)
else:
    print("❌ 需要進一步調優")
    fine_tune(optimized_v3, ab_test_results)
```

## 效能優化檢查清單

### 記憶體優化 ✓

- [ ] 模型量化（INT8/INT4）
- [ ] 權重共享
- [ ] KV Cache 優化
- [ ] 梯度檢查點（訓練時）
- [ ] 混合精度（FP16/BF16）

### 計算優化 ✓

- [ ] Flash Attention
- [ ] 算子融合
- [ ] JIT 編譯
- [ ] 批次處理
- [ ] 多 GPU 並行

### 延遲優化 ✓

- [ ] 模型預熱
- [ ] 預處理流水線
- [ ] 異步處理
- [ ] 請求優先級
- [ ] Speculative Decoding

### 吞吐量優化 ✓

- [ ] Continuous Batching
- [ ] Dynamic Batching
- [ ] 請求合並
- [ ] 負載均衡
- [ ] 資源池管理

## 最佳實踐總結

### 1. 始終測量

```python
# ❌ 錯誤：盲目優化
apply_all_optimizations()

# ✅ 正確：測量驅動
baseline = measure_performance()
for optimization in optimizations:
    apply(optimization)
    new_metrics = measure_performance()
    if new_metrics.better_than(baseline):
        keep(optimization)
    else:
        rollback(optimization)
```

### 2. 逐步優化

```python
# ❌ 錯誤：一次性應用所有優化
model = apply_all_optimizations_at_once(model)

# ✅ 正確：逐步優化，觀察效果
model = apply_quantization(model)
test_and_validate()

model = enable_kv_cache(model)
test_and_validate()

model = tune_batch_size(model)
test_and_validate()
```

### 3. 保持質量

```python
# ✅ 始終驗證輸出品質
def validate_quality(original_outputs, optimized_outputs):
    similarity = compute_similarity(original_outputs, optimized_outputs)

    if similarity < 0.95:  # 95% 相似度閾值
        print("⚠️  質量下降過多")
        return False

    return True
```

### 4. 監控生產環境

```python
# 設置監控指標
metrics_to_monitor = [
    "latency_p50",
    "latency_p99",
    "throughput",
    "error_rate",
    "gpu_utilization",
    "memory_usage"
]

# 設置告警
alerts = [
    Alert("latency_p99 > 1000ms", severity="high"),
    Alert("error_rate > 1%", severity="critical"),
    Alert("memory_usage > 90%", severity="medium")
]
```

### 5. 準備回滾計劃

```python
# 部署策略
deployment_strategy = {
    "type": "blue_green",  # 藍綠部署
    "rollback_on_error": True,
    "canary_percentage": 10,  # 先給 10% 流量
    "monitoring_period": 3600,  # 監控 1 小時
}

deploy_with_strategy(optimized_model, deployment_strategy)
```

## 故障排查指南

### 問題 1: OOM（記憶體不足）

**症狀**：
```
RuntimeError: CUDA out of memory
```

**解決方案**：
1. 減小批次大小
2. 啟用更激進的量化（4-bit）
3. 禁用 KV Cache（臨時）
4. 使用 CPU 卸載
5. 使用更小的模型

**程式碼**：
```python
# 降低記憶體使用
config = {
    "batch_size": 1,  # 最小批次
    "quantization": "4-bit",
    "kv_cache": False,  # 臨時禁用
    "cpu_offload": True  # 卸載到 CPU
}
```

### 問題 2: 延遲過高

**症狀**：
```
P99 延遲 > 2000ms
```

**解決方案**：
1. 檢查是否啟用 KV Cache
2. 優化批次大小（可能太大）
3. 使用 Flash Attention
4. 檢查是否有 CPU/GPU 瓶頸
5. 考慮使用更小的模型

**診斷程式碼**：
```python
# 延遲分析
latency_breakdown = profile_latency(model)
print(f"Tokenization: {latency_breakdown['tokenization']}ms")
print(f"Model forward: {latency_breakdown['forward']}ms")
print(f"Decoding: {latency_breakdown['decoding']}ms")

# 找出瓶頸
bottleneck = max(latency_breakdown, key=latency_breakdown.get)
print(f"瓶頸: {bottleneck}")
```

### 問題 3: 質量下降

**症狀**：
```
輸出品質明顯變差
```

**解決方案**：
1. 使用更保守的量化（AWQ > GPTQ > INT8）
2. 檢查是否誤用了動態量化
3. 驗證校準資料品質
4. 考慮 QAT（量化感知訓練）

**驗證程式碼**：
```python
# 質量對比
def compare_quality(original_model, optimized_model, test_cases):
    results = []

    for input_text in test_cases:
        original_output = generate(original_model, input_text)
        optimized_output = generate(optimized_model, input_text)

        similarity = compute_similarity(original_output, optimized_output)
        results.append({
            "input": input_text,
            "original": original_output,
            "optimized": optimized_output,
            "similarity": similarity
        })

    avg_similarity = sum(r["similarity"] for r in results) / len(results)
    print(f"平均相似度: {avg_similarity:.2%}")

    return results
```

## 成本優化計算器

```python
def calculate_cost_savings(baseline_metrics, optimized_metrics, hourly_rate=3.0):
    """
    計算成本節省

    Args:
        baseline_metrics: 基準指標
        optimized_metrics: 優化後指標
        hourly_rate: GPU 每小時成本（美元）

    Returns:
        成本節省報告
    """
    # 吞吐量提升
    throughput_improvement = (
        optimized_metrics["throughput"] / baseline_metrics["throughput"]
    )

    # 記憶體節省（可能允許使用更便宜的 GPU）
    memory_reduction = (
        1 - optimized_metrics["memory_gb"] / baseline_metrics["memory_gb"]
    )

    # 處理相同工作量所需時間
    baseline_hours = 1.0  # 基準
    optimized_hours = 1.0 / throughput_improvement

    # 成本計算
    baseline_cost = baseline_hours * hourly_rate
    optimized_cost = optimized_hours * hourly_rate

    savings = baseline_cost - optimized_cost
    savings_percentage = (savings / baseline_cost) * 100

    report = f"""
╔════════════════════════════════════════════════════════════╗
║                  成本優化報告                               ║
╚════════════════════════════════════════════════════════════╝

📊 性能提升:
  • 吞吐量提升: {throughput_improvement:.2f}x
  • 記憶體節省: {memory_reduction:.1%}
  • 延遲改善: {baseline_metrics['latency']/optimized_metrics['latency']:.2f}x

💰 成本分析:
  • 基準成本: ${baseline_cost:.2f}/小時
  • 優化成本: ${optimized_cost:.2f}/小時
  • 節省: ${savings:.2f}/小時 ({savings_percentage:.1f}%)

📈 年度預估:
  • 年度節省: ${savings * 24 * 365:.2f}
  • ROI: {(savings * 24 * 365) / 1000:.1f}x (假設優化成本 $1000)
    """

    return report
```

## 參考架構

### 高可用部署架構

```
                    ┌─────────────┐
                    │  Load       │
                    │  Balancer   │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
       ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
       │ vLLM    │    │ vLLM    │   │ vLLM    │
       │ Server  │    │ Server  │   │ Server  │
       │ (GPU 1) │    │ (GPU 2) │   │ (GPU 3) │
       └────┬────┘    └────┬────┘   └────┬────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────▼──────┐
                    │  Monitoring │
                    │  & Metrics  │
                    └─────────────┘
```

### 推薦配置

**小規模（<100 QPS）**：
- 單 GPU（RTX 3090/4090）
- AWQ 4-bit 量化
- vLLM + continuous batching
- 簡單負載均衡

**中規模（100-1000 QPS）**：
- 多 GPU（2-4 × A100）
- GPTQ/AWQ 量化
- vLLM 集群
- Redis 快取
- 自動擴展

**大規模（>1000 QPS）**：
- GPU 集群（8+ × A100/H100）
- 多模型版本（不同量化）
- 智能路由
- 分布式快取
- 實時監控和自動調優

## 下一步

- 運行實戰案例: [01_chatbot_optimization.py](./01_chatbot_optimization.py)
- 學習生產部署: [03_production_deployment.py](./03_production_deployment.py)
- 返回主目錄: [../README.md](../README.md)

---

**祝你優化成功！** 🚀
