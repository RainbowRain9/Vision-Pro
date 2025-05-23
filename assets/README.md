# 🎨 资源目录

本目录包含 YOLOvision Pro 项目的所有资源文件，包括图片、配置文件、数据样本等。

## 📁 目录结构

### 🖼️ images/
图片资源，包含架构图、结果图、演示图等
- `architecture/` - 模型架构图和技术图表
  - `repvgg_structure.png` - RepVGGBlock 结构图
  - `detection_layers.png` - 检测层分析图
  - `sandwich_fusion.png` - 三明治融合结构图
- `results/` - 实验结果图和性能图表
- `demos/` - 演示用图片和截图

### ⚙️ configs/
配置文件，包含模型配置、训练配置等
- `yolov8s-drone.yaml` - Drone-YOLO 模型配置文件
- `training_configs/` - 训练配置文件
  - `base_config.yaml` - 基础训练配置
  - `drone_config.yaml` - 无人机数据集配置
  - `ablation_configs/` - 消融实验配置

### 📊 data/
数据样本和标注文件
- `sample_images/` - 示例图片
- `annotations/` - 标注文件样本
- `datasets/` - 数据集链接和说明

## 🎯 使用说明

### 配置文件使用
```bash
# 使用 Drone-YOLO 配置训练
python scripts/training/train_drone_yolo.py --config assets/configs/yolov8s-drone.yaml

# 使用自定义训练配置
python train.py --config assets/configs/training_configs/drone_config.yaml
```

### 图片资源
- `architecture/` 中的图片用于文档说明和技术展示
- `results/` 中的图片展示模型性能和实验结果
- `demos/` 中的图片用于演示和教程

### 数据样本
- `sample_images/` 提供测试用的示例图片
- `annotations/` 包含标注格式的示例文件
- 实际训练数据请参考 `../data/` 目录

## 📋 文件规范

### 配置文件
- 使用 YAML 格式
- 包含详细的注释说明
- 遵循 YOLOv8 配置规范

### 图片文件
- 使用 PNG 格式保存图表
- 文件名使用下划线命名法
- 包含适当的分辨率和质量

### 数据文件
- 遵循 YOLO 格式标准
- 包含数据集说明文档
- 提供数据预处理脚本

## 🔗 相关目录

- [脚本目录](../scripts/README.md) - 使用这些资源的脚本
- [文档目录](../docs/README.md) - 资源使用说明文档
- [实验目录](../experiments/README.md) - 实验中使用的配置
