# YOLOv8 VisDrone 目标检测项目

本项目实现了基于YOLOv8的VisDrone数据集目标检测，并集成了supervision库进行增强功能开发。

## 项目结构

```
.
├── train_enhanced.py           # 增强版训练脚本，集成supervision
├── supervision_demo.py         # supervision功能演示
├── post_training_validation.py # 训练后验证脚本
├── monitor_training.py         # 训练进度监控
├── simple_analysis.py          # 简化版数据分析
├── visualization_analysis.py   # 可视化分析脚本
├── class_distribution.png      # 类别分布图
├── data/                       # 数据集目录
│   └── visdrone_yolo/         # VisDrone数据集（YOLO格式）
├── runs/                      # 训练输出目录
│   └── train_enhanced/       # 增强版训练输出
└── models/                    # 预训练模型目录
```

## 功能特性

### 1. 环境配置
- PyTorch CPU/GPU支持
- ultralytics YOLOv8框架
- supervision计算机视觉库
- OpenCV图像处理

### 2. 数据处理
- VisDrone数据集转换为YOLO格式
- 数据集分析和可视化
- 类别分布统计

### 3. 模型训练
- YOLOv8模型训练（支持CPU和GPU）
- 集成supervision的数据分析
- 训练进度监控

### 4. 功能演示
- 目标跟踪（ByteTrack）
- 目标计数（PolygonZone）
- 区域分析
- 可视化标注

### 5. 验证评估
- 模型性能验证
- 检测结果可视化
- 准确率评估

## 使用方法

### 1. 环境配置
```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install ultralytics supervision opencv-python

# 验证安装
python -c "import torch, cv2, ultralytics, supervision; print('环境配置完成')"
```

### 2. 数据准备
```bash
# 数据集已预处理为YOLO格式，位于data/visdrone_yolo/
```

### 3. 模型训练
```bash
# 启动训练（已在运行中）
python train_enhanced.py
```

### 4. 功能演示
```bash
# 运行supervision功能演示
python supervision_demo.py

# 运行数据分析
python simple_analysis.py
```

### 5. 训练后验证
```bash
# 训练完成后运行验证
python post_training_validation.py
```

## 输出文件

### 训练输出
- `runs/train_enhanced/YYYYMMDD_HHMMSS/` - 训练结果目录
- `weights/` - 模型权重文件
- `train_batch*.jpg` - 训练批次可视化
- `labels.jpg` - 标签分布图

### 分析输出
- `class_distribution.png` - 类别分布图
- `detection_visualizations/` - 检测结果可视化
- `predictions_visualization/` - 预测结果可视化

## 技术细节

### 模型架构
- 基于YOLOv8s预训练模型
- 支持10类VisDrone目标检测
- 图像输入尺寸: 640x640

### 数据集
- VisDrone2019-DET数据集
- 10个目标类别:
  1. pedestrian (行人)
  2. people (人群)
  3. bicycle (自行车)
  4. car (汽车)
  5. van (面包车)
  6. truck (卡车)
  7. tricycle (三轮车)
  8. awning-tricycle (棚三轮车)
  9. bus (公交车)
  10. motor (摩托车)

### 训练配置
- epochs: 30
- batch_size: 16
- image_size: 640
- optimizer: AdamW (自动调整)

## 项目状态

✅ 环境配置完成
✅ 数据集准备完成
✅ 训练脚本开发完成
⏳ 模型训练进行中 (Epoch 1/30)
✅ 功能演示实现完成
✅ 可视化分析完成
✅ 后续工作准备完成

## 后续步骤

1. 等待训练完成
2. 使用训练好的模型进行验证
3. 分析训练结果
4. 部署模型到实际应用

## 许可证

本项目仅供学习和研究使用。