# 测试目录

本目录包含项目的所有测试文件。

## 测试文件说明

- `test_ai_intro_examples.py`: AI 入门示例代码的测试
  - 专家系统测试
  - NumPy 基础操作测试
  - 线性回归测试
  - Sigmoid 函数测试
  - K-Means 聚类测试
  - 路径规划测试

- `test_pruning_utils.py`: GaLore 模型剪枝工具测试
  - 随机剪枝测试
  - 幅度剪枝测试

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定测试文件
```bash
pytest tests/test_ai_intro_examples.py
pytest tests/test_pruning_utils.py
```

### 生成覆盖率报告
```bash
pytest --cov=. --cov-report=html tests/
```

## 测试要求

- 所有测试文件应以 `test_` 开头
- 所有测试函数应以 `test_` 开头
- 使用 pytest 作为测试框架
- 尽可能提高测试覆盖率
