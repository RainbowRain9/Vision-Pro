# 🔧 脚本目录

本目录包含 YOLOvision Pro 项目的所有脚本文件，涵盖演示、测试、可视化、训练等各个方面。

## 📁 目录结构

### 🎭 demo/
演示脚本，用于展示模型功能和核心概念
- `drone_yolo_demo.py` - Drone-YOLO 核心概念演示
- `component_demos.py` - 各组件功能演示
- `interactive_demo.py` - 交互式演示脚本

### 🧪 testing/
测试脚本，用于验证模型和组件的正确性
- `test_drone_yolo.py` - Drone-YOLO 模型测试
- `unit_tests.py` - 单元测试
- `integration_tests.py` - 集成测试
- `benchmark_tests.py` - 性能基准测试

### 📊 visualization/
可视化脚本，用于生成图表和可视化结果
- `visualize_drone_yolo.py` - Drone-YOLO 架构可视化
- `plot_results.py` - 结果图表绘制
- `model_analysis.py` - 模型分析可视化
- `training_curves.py` - 训练曲线绘制

### 🚀 training/
训练相关脚本
- `train_drone_yolo.py` - Drone-YOLO 训练脚本
- `evaluate_model.py` - 模型评估脚本
- `hyperparameter_tuning.py` - 超参数调优
- `data_preprocessing.py` - 数据预处理

## 🎯 使用说明

### 快速开始
```bash
# 运行 Drone-YOLO 测试
python scripts/testing/test_drone_yolo.py

# 生成架构可视化图
python scripts/visualization/visualize_drone_yolo.py

# 运行核心概念演示
python scripts/demo/drone_yolo_demo.py
```

### 训练模型
```bash
# 训练 Drone-YOLO 模型
python scripts/training/train_drone_yolo.py --config assets/configs/yolov8s-drone.yaml

# 评估模型性能
python scripts/training/evaluate_model.py --model outputs/models/best.pt
```

## 📋 脚本规范

- 每个脚本都应包含详细的文档字符串
- 使用 argparse 处理命令行参数
- 包含错误处理和日志记录
- 提供使用示例和帮助信息

## 🔗 相关目录

- [文档目录](../docs/README.md) - 查看详细文档
- [资源目录](../assets/README.md) - 获取配置文件和数据
- [输出目录](../outputs/README.md) - 查看运行结果
