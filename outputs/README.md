# 📊 输出目录

本目录用于存放训练好的模型、日志文件和实验结果，是项目运行产生的所有输出文件的集中存放位置。

## 📁 目录结构

### 🤖 models/
训练好的模型文件
- `drone_yolo_best.pt` - 最佳 Drone-YOLO 模型
- `drone_yolo_last.pt` - 最新 Drone-YOLO 模型
- `yolov8s_baseline.pt` - 基线 YOLOv8s 模型
- `ablation_models/` - 消融实验模型
  - `no_repvgg.pt` - 不使用 RepVGG 的模型
  - `no_p2.pt` - 不使用 P2 检测头的模型
  - `no_fusion.pt` - 不使用三明治融合的模型

### 📝 logs/
训练和实验日志
- `training_logs/` - 训练日志
  - `drone_yolo_train.log` - Drone-YOLO 训练日志
  - `tensorboard/` - TensorBoard 日志文件
- `experiment_logs/` - 实验日志
  - `baseline_comparison.log` - 基线对比日志
  - `ablation_studies.log` - 消融实验日志
- `evaluation_logs/` - 评估日志

### 📈 results/
实验结果和分析报告
- `performance_metrics/` - 性能指标
  - `drone_yolo_metrics.json` - Drone-YOLO 性能指标
  - `comparison_metrics.json` - 对比实验指标
- `visualizations/` - 可视化结果
  - `training_curves.png` - 训练曲线
  - `confusion_matrix.png` - 混淆矩阵
  - `detection_samples.png` - 检测样本
- `reports/` - 分析报告
  - `performance_report.md` - 性能分析报告
  - `ablation_report.md` - 消融实验报告

## 🎯 使用说明

### 模型文件
```bash
# 加载最佳模型进行推理
from ultralytics import YOLO
model = YOLO('outputs/models/drone_yolo_best.pt')
results = model('path/to/image.jpg')

# 继续训练
model = YOLO('outputs/models/drone_yolo_last.pt')
model.train(data='data.yaml', epochs=100)
```

### 日志查看
```bash
# 查看训练日志
tail -f outputs/logs/training_logs/drone_yolo_train.log

# 启动 TensorBoard
tensorboard --logdir outputs/logs/training_logs/tensorboard/
```

### 结果分析
```bash
# 查看性能指标
cat outputs/results/performance_metrics/drone_yolo_metrics.json

# 生成分析报告
python scripts/analysis/generate_report.py --results outputs/results/
```

## 📊 性能指标

### Drone-YOLO 性能
```json
{
  "mAP@0.5": 0.xxx,
  "mAP@0.5:0.95": 0.xxx,
  "precision": 0.xxx,
  "recall": 0.xxx,
  "f1_score": 0.xxx,
  "inference_time": "xx.x ms",
  "model_size": "xx.x MB"
}
```

### 训练统计
- **训练轮数**: xxx epochs
- **最佳轮数**: xxx epoch
- **训练时间**: xx hours
- **GPU 使用**: xxx%
- **内存使用**: xxx GB

## 📋 文件管理

### 自动清理
```bash
# 清理旧的日志文件（保留最近30天）
find outputs/logs/ -name "*.log" -mtime +30 -delete

# 清理临时文件
rm -rf outputs/temp/
```

### 备份策略
- 重要模型文件定期备份到云存储
- 关键实验结果保存多个副本
- 使用版本控制管理配置文件

### 命名规范
- 模型文件: `{model_name}_{version}_{metric}.pt`
- 日志文件: `{experiment_name}_{date}.log`
- 结果文件: `{experiment_type}_{date}_{version}.json`

## 🔗 相关目录

- [实验目录](../experiments/README.md) - 查看实验配置
- [脚本目录](../scripts/README.md) - 生成这些输出的脚本
- [文档目录](../docs/README.md) - 结果分析文档
