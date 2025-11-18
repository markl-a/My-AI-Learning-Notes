# LLM 部署工具集

实用的 LLM 部署和管理工具。

## 🛠️ 工具列表

### 1. 模型下载管理器 (`model_manager.py`)
自动下载和管理 LLM 模型

功能：
- 从 Hugging Face 下载模型
- 管理 Ollama 模型
- 检查模型完整性
- 清理未使用的模型

使用：
```bash
python model_manager.py --download llama3.1:8b
python model_manager.py --list
python model_manager.py --cleanup
```

### 2. 性能基准测试 (`benchmark.py`)
测试和比较模型性能

功能：
- 推理速度测试
- 内存使用分析
- 多模型对比
- 生成性能报告

使用：
```bash
python benchmark.py --model gpt-4o-mini
python benchmark.py --compare gpt-4o-mini,llama3.1:8b,claude-3-haiku
```

### 3. 健康检查脚本 (`health_check.py`)
检查部署环境和服务状态

功能：
- GPU 可用性检查
- API keys 验证
- 服务连接测试
- 生成诊断报告

使用：
```bash
python health_check.py
python health_check.py --verbose
python health_check.py --fix-issues
```

### 4. 成本计算器 (`cost_calculator.py`)
计算和预估 LLM 使用成本

功能：
- 实时成本计算
- 成本预估
- 多模型对比
- 生成成本报告

使用：
```bash
python cost_calculator.py --tokens 100000 --model gpt-4o-mini
python cost_calculator.py --estimate-monthly 1000000
```

## 📦 安装依赖

```bash
pip install -r ../requirements.txt
```

## 🚀 快速开始

1. 配置环境变量（复制 ../.env.example 到 .env）
2. 运行健康检查确保环境正确
3. 使用对应工具完成任务

## 💡 最佳实践

- 定期运行健康检查
- 使用基准测试选择最佳模型
- 监控成本使用
- 定期清理未使用的模型
