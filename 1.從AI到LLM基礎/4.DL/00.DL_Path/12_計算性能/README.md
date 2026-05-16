# 深度學習計算性能優化完整指南

> **更新日期**: 2024-2025
> **適用框架**: PyTorch 2.0+, TensorFlow 2.x, CUDA 12.x
> **學習目標**: 從入門到精通深度學習計算性能優化

## 📚 目錄

- [深度學習計算性能優化完整指南](#深度學習計算性能優化完整指南)
  - [📚 目錄](#-目錄)
  - [🎯 學習路徑](#-學習路徑)
    - [**入門階段** (1-2週)](#入門階段-1-2週)
    - [**進階階段** (2-3週)](#進階階段-2-3週)
    - [**精通階段** (3-4週)](#精通階段-3-4週)
  - [📖 課程內容](#-課程內容)
    - [**基礎篇**](#基礎篇)
    - [**進階篇** (2024-2025 最新技術)](#進階篇-2024-2025-最新技術)
    - [**實戰篇**](#實戰篇)
  - [🤖 AI 輔助學習建議](#-ai-輔助學習建議)
  - [💡 實用技巧速查](#-實用技巧速查)
    - [**性能優化檢查清單**](#性能優化檢查清單)
    - [**常見性能瓶頸診斷**](#常見性能瓶頸診斷)
  - [🔧 開發環境設置](#-開發環境設置)
  - [📊 性能基準測試](#-性能基準測試)
  - [🌟 最佳實踐](#-最佳實踐)
  - [📚 推薦資源](#-推薦資源)
    - [**官方文檔**](#官方文檔)
    - [**進階閱讀**](#進階閱讀)
    - [**實用工具**](#實用工具)
  - [🚀 進階主題](#-進階主題)
  - [💬 學習建議](#-學習建議)

---

## 🎯 學習路徑

### **入門階段** (1-2週)
1. **理解計算基礎** → `4_hardware.ipynb`
   - CPU、GPU、TPU 架構基礎
   - 記憶體層次結構（L1/L2/L3 Cache, RAM, VRAM）
   - 延遲數字與帶寬概念

2. **掌握基本優化** → `1_hybridize.ipynb`
   - 命令式編程 vs 符號式編程
   - PyTorch JIT 編譯（TorchScript）
   - 模型序列化與部署

3. **異步計算入門** → `2_async-computation.ipynb`
   - 理解前端與後端解耦
   - GPU 異步執行機制
   - 性能基準測試方法

### **進階階段** (2-3週)
4. **混合精度訓練** → `8_mixed_precision_training.ipynb` ⭐ NEW
   - FP32, FP16, BF16 精度選擇
   - PyTorch AMP (Automatic Mixed Precision)
   - 梯度縮放與數值穩定性

5. **性能分析工具** → `9_profiling_tools.ipynb` ⭐ NEW
   - PyTorch Profiler 使用
   - TensorBoard 性能視覺化
   - NVIDIA Nsight Systems

6. **編譯優化技術** → `10_torch_compile.ipynb` ⭐ NEW
   - `torch.compile()` 深入解析
   - TorchDynamo, TorchInductor
   - AOTAutograd 機制

7. **GPU 記憶體優化** → `11_gpu_memory_optimization.ipynb` ⭐ NEW
   - 梯度檢查點（Gradient Checkpointing）
   - 記憶體分析與洩漏診斷
   - 批次大小動態調整

### **精通階段** (3-4週)
8. **模型量化** → `12_quantization.ipynb` ⭐ NEW
   - 訓練後量化（PTQ）
   - 量化感知訓練（QAT）
   - INT8, INT4 推論優化

9. **分佈式訓練** → `13_distributed_training.ipynb` ⭐ NEW
   - DataParallel vs DistributedDataParallel
   - FSDP (Fully Sharded Data Parallel)
   - ZeRO 優化器

10. **多 GPU 訓練**
    - 基礎實現 → `5_multiple-gpus.ipynb`
    - 簡潔實現 → `6_multiple-gpus-concise.ipynb`
    - 自動並行 → `3_auto-parallelism.ipynb`
    - 參數伺服器 → `7_parameterserver.ipynb`

11. **AI 輔助優化** → `14_ai_assisted_optimization.ipynb` ⭐ NEW
    - 使用 AI 自動調整超參數
    - AI 驅動的模型壓縮
    - 智能性能預測與建議

---

## 📖 課程內容

### **基礎篇**

| 編號 | 主題 | 文件 | 難度 | 預計時間 |
|------|------|------|------|----------|
| 0 | 課程索引 | `0_index.ipynb` | ⭐ | 15 分鐘 |
| 1 | 編譯器與解釋器 | `1_hybridize.ipynb` | ⭐⭐ | 2 小時 |
| 2 | 異步計算 | `2_async-computation.ipynb` | ⭐⭐ | 1.5 小時 |
| 3 | 自動並行 | `3_auto-parallelism.ipynb` | ⭐⭐⭐ | 2 小時 |
| 4 | 硬件基礎 | `4_hardware.ipynb` | ⭐⭐ | 3 小時 |
| 5 | 多 GPU 訓練 | `5_multiple-gpus.ipynb` | ⭐⭐⭐ | 3 小時 |
| 6 | 多 GPU 簡潔版 | `6_multiple-gpus-concise.ipynb` | ⭐⭐⭐ | 2 小時 |
| 7 | 參數伺服器 | `7_parameterserver.ipynb` | ⭐⭐⭐⭐ | 3 小時 |

### **進階篇** (2024-2025 最新技術)

| 編號 | 主題 | 文件 | 難度 | 預計時間 | 標籤 |
|------|------|------|------|----------|------|
| 8 | 混合精度訓練 | `8_mixed_precision_training.ipynb` | ⭐⭐⭐ | 2.5 小時 | 🔥 必學 |
| 9 | 性能分析工具 | `9_profiling_tools.ipynb` | ⭐⭐⭐ | 3 小時 | 🛠️ 實用 |
| 10 | Torch 編譯優化 | `10_torch_compile.ipynb` | ⭐⭐⭐⭐ | 3 小時 | 🚀 前沿 |
| 11 | GPU 記憶體優化 | `11_gpu_memory_optimization.ipynb` | ⭐⭐⭐ | 2.5 小時 | 💾 關鍵 |
| 12 | 模型量化 | `12_quantization.ipynb` | ⭐⭐⭐⭐ | 4 小時 | 📦 部署 |
| 13 | 分佈式訓練 | `13_distributed_training.ipynb` | ⭐⭐⭐⭐⭐ | 5 小時 | 🌐 擴展 |
| 14 | AI 輔助優化 | `14_ai_assisted_optimization.ipynb` | ⭐⭐⭐⭐ | 3 小時 | 🤖 創新 |

### **實戰篇**

- **專案 1**: 優化 ResNet-50 訓練速度（目標：2x 加速）
- **專案 2**: 大規模語言模型分佈式訓練（BERT, GPT）
- **專案 3**: 邊緣設備模型部署（量化 + TorchScript）
- **專案 4**: 混合精度訓練實戰（Stable Diffusion）

---

## 🤖 AI 輔助學習建議

在學習過程中，你可以使用 AI 工具來：

1. **程式碼審查與優化建議**
   ```python
   # 將你的訓練程式碼貼給 Claude/ChatGPT，詢問：
   # "請幫我分析這段程式碼的性能瓶頸，並提供優化建議"
   ```

2. **性能問題診斷**
   ```
   提示詞模板：
   "我的模型訓練速度很慢，GPU 利用率只有 30%，
   可能的原因有哪些？如何診斷？"
   ```

3. **超參數自動調整**
   - 使用 Optuna + AI 建議
   - Ray Tune 配合 LLM 推薦

4. **文檔理解輔助**
   ```
   "請用簡單的語言解釋 FSDP 和 DDP 的區別"
   "torch.compile 的 backend 選項有哪些？各自適用場景？"
   ```

---

## 💡 實用技巧速查

### **性能優化檢查清單**

- [ ] 使用混合精度訓練（AMP）
- [ ] 啟用 `torch.compile()` 編譯優化
- [ ] 設置合適的 `num_workers` 和 `pin_memory`
- [ ] 使用梯度累積降低記憶體使用
- [ ] 啟用 cuDNN benchmark: `torch.backends.cudnn.benchmark = True`
- [ ] 優化資料加載管道（避免 CPU 瓶頸）
- [ ] 使用 Profiler 找出性能瓶頸
- [ ] 檢查 GPU 記憶體碎片化
- [ ] 使用異步資料預取
- [ ] 考慮模型量化（部署時）

### **常見性能瓶頸診斷**

| 症狀 | 可能原因 | 解決方案 |
|------|----------|----------|
| GPU 利用率低 (<50%) | 資料加載慢 | 增加 `num_workers`，使用 `prefetch_factor` |
| 顯存不足 (OOM) | 批次過大 | 減小 batch size，使用梯度累積 |
| 訓練速度慢 | 未使用混合精度 | 啟用 AMP，使用 `torch.autocast` |
| 多卡訓練不平衡 | 資料分佈不均 | 使用 `DistributedSampler` |
| 記憶體洩漏 | 梯度未清零 | 正確使用 `optimizer.zero_grad()` |

---

## 🔧 開發環境設置

```bash
# 推薦環境配置（2024-2025）
conda create -n dl-perf python=3.10
conda activate dl-perf

# PyTorch 2.x (支持 torch.compile)
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 性能分析工具
pip install tensorboard torch-tb-profiler

# 分佈式訓練工具
pip install accelerate deepspeed

# 量化工具
pip install pytorch-quantization

# 實驗管理
pip install wandb optuna

# 其他實用工具
pip install nvidia-ml-py3 gpustat py3nvml
```

---

## 📊 性能基準測試

**標準測試腳本**（位於 `benchmarks/` 目錄）：

```python
# 快速性能測試模板
from benchmark_utils import BenchmarkSuite

suite = BenchmarkSuite(model, device='cuda')
results = suite.run_all_tests()
# 自動生成優化建議報告
```

**預期性能指標**（ResNet-50, ImageNet, V100）：

| 配置 | 吞吐量 (images/sec) | 訓練時間 (90 epochs) |
|------|---------------------|----------------------|
| Baseline (FP32) | ~350 | ~30 小時 |
| + AMP | ~850 | ~12 小時 |
| + torch.compile | ~1100 | ~9 小時 |
| + 4-GPU DDP | ~4000 | ~2.5 小時 |

---

## 🌟 最佳實踐

1. **先性能分析，再優化**
   - 不要盲目優化，使用 Profiler 找出真正的瓶頸

2. **漸進式優化**
   - 一次只改一個變量，記錄性能變化

3. **重現性優先**
   ```python
   # 固定隨機種子
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   ```

4. **監控系統資源**
   ```bash
   # 實時監控 GPU
   watch -n 1 nvidia-smi

   # 或使用 gpustat
   gpustat -i 1
   ```

5. **記錄實驗結果**
   - 使用 Weights & Biases 或 TensorBoard
   - 記錄硬件配置、超參數、性能指標

---

## 📚 推薦資源

### **官方文檔**
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)

### **進階閱讀**
- 《深入淺出 PyTorch》- 邱錫鵬
- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

### **實用工具**
- [PyTorch Lightning](https://lightning.ai/) - 簡化訓練流程
- [Accelerate](https://huggingface.co/docs/accelerate/) - Hugging Face 分佈式訓練
- [Composer](https://github.com/mosaicml/composer) - MosaicML 性能優化

---

## 🚀 進階主題

完成本課程後，可以探索：

1. **模型壓縮技術**
   - 知識蒸餾（Knowledge Distillation）
   - 剪枝（Pruning）
   - 神經架構搜索（NAS）

2. **專用硬件優化**
   - TPU 編程（JAX, TensorFlow）
   - 邊緣設備部署（TensorRT, ONNX Runtime）
   - 自定義 CUDA Kernel

3. **大規模訓練**
   - 千億參數模型訓練
   - Pipeline Parallelism
   - 3D Parallelism (資料 + 模型 + 流水線)

4. **效率前沿研究**
   - Flash Attention
   - Memory-Efficient Transformers
   - Mixture of Experts (MoE)

---

## 💬 學習建議

1. **理論與實踐結合**
   - 每學完一個 Notebook，立即在自己的項目中嘗試應用

2. **建立性能基線**
   - 優化前先測量當前性能，建立對比基準

3. **參與社區討論**
   - PyTorch Forums, Reddit r/MachineLearning
   - 閱讀他人的優化經驗分享

4. **持續更新知識**
   - 關注 PyTorch 新版本發布說明
   - 閱讀 NVIDIA、Meta 等公司的技術博客

5. **實驗記錄**
   - 記錄每次優化嘗試的結果
   - 建立自己的性能優化知識庫

---

**版權聲明**: 本教程基於 D2L (Dive into Deep Learning) 並擴展了 2024-2025 最新技術。
**貢獻者**: 歡迎提交 PR 完善內容！
**問題反饋**: 請在 Issues 中提出建議和問題。

---

**開始你的性能優化之旅吧！🚀**
