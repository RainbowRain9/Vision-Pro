# 可视化脚本

本目录包含用于生成图表、可视化结果和模型分析的脚本。

## 🎨 功能概述

- **模型架构可视化**: 生成 Drone-YOLO 架构图
- **训练结果可视化**: 训练曲线、损失函数图表
- **数据集分析**: 数据分布、类别统计图表
- **检测结果可视化**: 检测结果展示和分析

## 🔧 可视化工具

- `visualize_drone_yolo.py` - Drone-YOLO 架构可视化
- `plot_training_results.py` - 训练结果图表生成
- `analyze_dataset.py` - 数据集分析和可视化
- `visualize_detections.py` - 检测结果可视化

## 🚀 使用方法

```bash
# 生成 Drone-YOLO 架构图
python visualization/visualize_drone_yolo.py

# 绘制训练结果
python visualization/plot_training_results.py --results runs/train/exp/

# 数据集分析
python visualization/analyze_dataset.py --dataset data/visdrone_yolo/

# 可视化检测结果
python visualization/visualize_detections.py --images data/test/ --model best.pt
```

## 📊 输出格式

- **图像格式**: PNG, SVG (高质量)
- **图表类型**: 折线图、柱状图、散点图、热力图
- **保存位置**: `outputs/visualizations/`

## 🎯 可视化内容

### 模型架构
- 网络结构图
- 参数统计
- 计算复杂度分析

### 训练监控
- 损失函数曲线
- 精度变化趋势
- 学习率调度

### 数据分析
- 类别分布统计
- 边界框尺寸分布
- 图像尺寸分析

### 检测结果
- 检测框可视化
- 置信度分布
- 性能指标图表

## 🔗 相关工具

- [训练脚本](../training/)
- [数据处理工具](../data_processing/)
- [验证工具](../validation/)
