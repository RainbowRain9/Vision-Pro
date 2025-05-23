# 🧪 实验目录

本目录用于存放各种实验的配置、脚本和结果，支持系统性的模型研究和性能分析。

## 📁 目录结构

### 📊 baseline_comparison/
基线模型对比实验
- `yolov8_vs_drone_yolo.py` - YOLOv8 与 Drone-YOLO 对比
- `configs/` - 对比实验配置文件
- `results/` - 对比实验结果
- `analysis.md` - 对比分析报告

### 🔬 ablation_studies/
消融实验，研究各组件的贡献
- `repvgg_ablation.py` - RepVGGBlock 消融实验
- `p2_head_ablation.py` - P2 检测头消融实验
- `fusion_ablation.py` - 三明治融合消融实验
- `configs/` - 消融实验配置
- `results/` - 消融实验结果

### 📈 performance_analysis/
性能分析实验
- `speed_benchmark.py` - 速度基准测试
- `memory_analysis.py` - 内存使用分析
- `accuracy_analysis.py` - 精度分析
- `small_object_analysis.py` - 小目标检测分析
- `results/` - 性能分析结果

## 🎯 实验指南

### 运行基线对比
```bash
# 运行 YOLOv8 vs Drone-YOLO 对比实验
cd experiments/baseline_comparison
python yolov8_vs_drone_yolo.py --config configs/comparison_config.yaml
```

### 执行消融实验
```bash
# RepVGGBlock 消融实验
cd experiments/ablation_studies
python repvgg_ablation.py --ablate repvgg

# P2 检测头消融实验
python p2_head_ablation.py --ablate p2_head

# 三明治融合消融实验
python fusion_ablation.py --ablate sandwich_fusion
```

### 性能分析
```bash
# 速度基准测试
cd experiments/performance_analysis
python speed_benchmark.py --model ../../outputs/models/drone_yolo.pt

# 小目标检测分析
python small_object_analysis.py --dataset ../../data/yolo_dataset/
```

## 📊 实验结果

### 基线对比结果
| 模型 | mAP@0.5 | mAP@0.5:0.95 | 速度(FPS) | 参数量(M) |
|------|---------|--------------|-----------|-----------|
| YOLOv8s | - | - | - | 11.2 |
| Drone-YOLO | - | - | - | 11.1 |

### 消融实验结果
| 组件 | mAP@0.5 | 提升 | 说明 |
|------|---------|------|------|
| 基线 | - | - | 原始 YOLOv8s |
| +RepVGG | - | - | 添加 RepVGGBlock |
| +P2 Head | - | - | 添加 P2 检测头 |
| +Sandwich Fusion | - | - | 添加三明治融合 |

## 📋 实验规范

### 实验设计
- 控制变量，确保实验的可重复性
- 使用相同的数据集和评估指标
- 记录详细的实验配置和环境信息

### 结果记录
- 保存完整的训练日志
- 记录模型权重和配置文件
- 生成可视化的结果图表

### 分析报告
- 提供详细的实验分析
- 包含统计显著性检验
- 给出结论和改进建议

## 🔗 相关目录

- [输出目录](../outputs/README.md) - 查看实验输出结果
- [脚本目录](../scripts/README.md) - 实验相关脚本
- [文档目录](../docs/README.md) - 实验方法论文档
