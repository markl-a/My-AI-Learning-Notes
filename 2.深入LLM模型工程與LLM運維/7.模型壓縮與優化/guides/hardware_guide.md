# 硬體選擇指南

## GPU 選擇矩陣

| 用途 | 推薦 GPU | 顯存 | 價格範圍 | 適用模型 |
|------|---------|------|---------|---------|
| **學習研究** | RTX 4070 Ti | 12GB | $800 | 7B (INT4) |
| **個人開發** | RTX 4090 | 24GB | $1,600 | 13B (INT4), 7B (INT8) |
| **小團隊** | RTX A6000 | 48GB | $4,500 | 30B (INT4), 13B (FP16) |
| **企業推理** | A100 (40GB) | 40GB | $10,000 | 70B (INT4), 30B (INT8) |
| **大規模訓練** | A100 (80GB) | 80GB | $15,000 | 70B (FP16), 180B (INT4) |
| **雲端推理** | H100 | 80GB | $30,000+ | 最大性能 |

## CPU/邊緣設備

| 設備 | RAM | 成本 | 適用模型 | 速度 |
|------|-----|------|---------|------|
| Raspberry Pi 5 | 8GB | $80 | TinyLlama (Q4) | ~8 tok/s |
| Intel NUC 13 | 32GB | $600 | Phi-2 (Q4) | ~15 tok/s |
| Mac Mini M2 Pro | 32GB | $1,500 | LLaMA-7B (Q5) | ~20 tok/s |
| Mac Studio M2 Ultra | 192GB | $6,000 | LLaMA-70B (Q4) | ~30 tok/s |
| NVIDIA Jetson Orin | 32GB | $500 | LLaMA-7B (Q4) | ~25 tok/s |

## 詳細建議

參考 [deployment_cases.md](../advanced/deployment_cases.md) 中的實際部署案例。
