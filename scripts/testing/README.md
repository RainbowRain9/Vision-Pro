# 测试脚本

本目录包含用于验证模型和组件正确性的测试脚本。

## 🧪 测试概述

测试脚本用于确保 Drone-YOLO 模型和相关组件的功能正确性，包括单元测试、集成测试和性能基准测试。

## 🔧 现有测试

### 模型功能测试
- `test_drone_yolo.py` - Drone-YOLO 模型功能测试
  - 测试模型架构的正确性
  - 验证 RepVGGBlock 结构
  - 检查模型输出格式

## 🚀 使用方法

```bash
# 运行 Drone-YOLO 功能测试
python scripts/testing/test_drone_yolo.py

# 查看测试帮助信息
python scripts/testing/test_drone_yolo.py --help
```

## 📋 计划中的测试

### 单元测试
- 数据处理模块测试
- 模型组件单元测试
- 工具函数测试

### 集成测试
- 端到端训练流程测试
- 数据处理管道测试
- 模型推理流程测试

### 性能基准测试
- 模型推理速度测试
- 内存使用量测试
- 精度基准测试

## 🔗 相关工具

- [验证工具](../validation/) - 环境和配置验证
- [演示脚本](../demo/) - 功能演示
- [数据处理](../data_processing/) - 数据处理工具
