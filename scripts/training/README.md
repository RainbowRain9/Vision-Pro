# 训练脚本

本目录包含模型训练相关的脚本，用于 Drone-YOLO 模型训练、评估和超参数调优。

## 🎯 功能概述

- **模型训练**: Drone-YOLO 模型训练脚本
- **模型评估**: 训练结果评估和性能分析
- **超参数调优**: 自动化超参数搜索和优化
- **训练监控**: 训练过程可视化和监控

## 🔧 计划中的脚本

- `train_drone_yolo.py` - Drone-YOLO 模型训练主脚本
- `evaluate_model.py` - 模型评估和性能测试
- `hyperparameter_tuning.py` - 超参数自动调优
- `training_monitor.py` - 训练过程监控和可视化

## 🚀 使用方法

```bash
# 基础训练
python training/train_drone_yolo.py --config assets/configs/yolov8s-drone.yaml

# 使用 VisDrone 数据集训练
python training/train_drone_yolo.py \
    --data data/visdrone_yolo/data.yaml \
    --config assets/configs/yolov8s-drone.yaml \
    --epochs 300

# 模型评估
python training/evaluate_model.py --model runs/train/exp/weights/best.pt
```

## 📊 训练配置

推荐的训练参数：
- **图像尺寸**: 640x640
- **批次大小**: 8-16 (根据GPU内存调整)
- **学习率**: 0.001 (初始)
- **训练轮数**: 300 (VisDrone数据集)
- **优化器**: AdamW

## 🔗 相关资源

- [Drone-YOLO 配置文件](../../assets/configs/)
- [VisDrone 数据集处理](../data_processing/visdrone/)
- [模型验证工具](../validation/)
