# 🚁 YOLOvision Pro

## 1. 概述

YOLOvision Pro 是一个完整的YOLO目标检测解决方案，专注于小目标检测优化，特别是无人机场景下的目标检测。项目包含从数据标注、模型训练到模型应用的全流程工具，并实现了先进的 Drone-YOLO 算法。

### 主要特性

1. **🎯 Drone-YOLO 算法** - 基于 YOLOv8 优化的小目标检测模型
   - RepVGGBlock 高效主干网络
   - P2 小目标检测头
   - 三明治融合结构

2. **🖥️ YOLOv8 检测系统** - 基于PyQt5的图形界面应用
   - 多种输入源支持（图片/视频/摄像头）
   - 实时参数调节和结果可视化

3. **🏷️ 数据标注工具链** - 完整的数据准备流程
   - LabelMe 标注工具集成
   - 自动格式转换和数据集划分

4. **🚁 VisDrone 数据集处理** - 专业的 VisDrone2019 数据集处理工具
   - 一键格式转换（VisDrone → YOLO）
   - 智能数据集划分和验证
   - 详细统计分析和可视化

## 2. 项目架构

项目采用模块化的目录结构，便于开发、实验和部署：

```
yolovision_pro/
├── 📄 main.py                   # 主程序入口，实现UI界面
├── 🚀 train.py                  # 模型训练脚本
├── 📚 docs/                     # 文档目录
│   ├── technical_analysis/      # 技术分析文档
│   ├── tutorials/              # 教程指南
│   └── references/             # 参考资料
├── 🔧 scripts/                  # 脚本目录
│   ├── data_processing/        # 数据处理脚本
│   │   ├── visdrone/          # VisDrone 专用工具
│   │   ├── general/           # 通用数据处理工具
│   │   └── demos/             # 数据处理演示
│   ├── validation/             # 验证和检查工具
│   ├── training/               # 训练脚本
│   ├── testing/                # 测试脚本
│   ├── demo/                   # 演示脚本
│   ├── visualization/          # 可视化脚本
│   └── docs/                   # 脚本文档
├── 🎨 assets/                   # 资源目录
│   ├── images/                 # 图片资源（架构图、结果图等）
│   ├── configs/                # 配置文件
│   └── data/                   # 数据样本
├── 🧪 experiments/              # 实验目录
│   ├── baseline_comparison/    # 基线对比实验
│   ├── ablation_studies/       # 消融实验
│   └── performance_analysis/   # 性能分析
├── 📊 outputs/                  # 输出目录
│   ├── models/                 # 训练好的模型
│   ├── logs/                   # 训练日志
│   └── results/                # 实验结果
├── 📁 data/                     # 数据目录
│   ├── raw_images/             # 原始图像
│   ├── annotations/            # 标注文件
│   ├── yolo_dataset/           # YOLO 格式数据集
│   ├── VisDrone2019-DET-train/ # VisDrone 原始数据集
│   ├── visdrone_yolo/          # VisDrone YOLO 格式数据集
│   └── configs/                # 数据集配置文件
├── 🤖 models/                   # 预训练模型
├── 📈 results/                  # 检测结果
│   ├── images/                 # 图片检测结果
│   ├── videos/                 # 视频检测结果
│   └── camera/                 # 摄像头检测结果
└── 🔬 ultralytics/              # YOLOv8 框架（含 Drone-YOLO 实现）
```

### 📖 目录说明

- **📚 docs/**: 包含技术文档、教程和参考资料 → [查看详情](docs/README.md)
- **🔧 scripts/**: 各种功能脚本，包括演示、测试、可视化等 → [查看详情](scripts/README.md)
- **🎨 assets/**: 项目资源文件，包括配置、图片、数据样本 → [查看详情](assets/README.md)
- **🧪 experiments/**: 实验配置和结果，支持系统性研究 → [查看详情](experiments/README.md)
- **📊 outputs/**: 训练输出、日志和结果文件 → [查看详情](outputs/README.md)

## 3. 🎯 Drone-YOLO 算法

### 3.1 核心创新

Drone-YOLO 是基于 YOLOv8 优化的小目标检测算法，专门针对无人机场景进行了以下改进：

1. **🔧 RepVGGBlock 主干网络**
   - 训练时使用多分支结构（3x3卷积 + 1x1卷积 + 恒等映射）
   - 推理时融合为单个3x3卷积，提升速度
   - 平衡了表达能力和计算效率

2. **🎯 P2 小目标检测头**
   - 增加 1/4 下采样的 P2 检测层
   - 160×160 高分辨率特征图，专门检测 4-16 像素小目标
   - 25,600 个检测位置，4倍于传统 P3 层

3. **🥪 三明治融合结构**
   - 融合上层语义信息 + 当前层特征 + 下层细节信息
   - 使用 DWConv 进行高效的跨尺度特征融合
   - 显著提升小目标检测性能

### 3.2 性能指标

| 模型 | 参数量 | 计算量 | 检测层 | 特点 |
|------|--------|--------|--------|------|
| YOLOv8s | 11.2M | 28.6 GFLOPs | 3层 (P3,P4,P5) | 基线模型 |
| Drone-YOLO | 11.1M | 40.3 GFLOPs | 4层 (P2,P3,P4,P5) | 小目标优化 |

### 3.3 快速开始

```bash
# 测试 Drone-YOLO 模型
python scripts/testing/test_drone_yolo.py

# 查看架构可视化
python scripts/visualization/visualize_drone_yolo.py

# 运行技术演示
python scripts/demo/drone_yolo_demo.py
```

## 4. 🚁 VisDrone 数据集处理

### 4.1 功能特性

YOLOvision Pro 提供了完整的 VisDrone2019 数据集处理工具链：

- **🔄 格式转换**: 将 VisDrone 标注格式转换为 YOLO 格式
- **📊 数据划分**: 按 8:1:1 比例智能划分训练/验证/测试集
- **✅ 数据验证**: 检查数据完整性和标注格式正确性
- **📈 统计分析**: 生成详细的数据集统计报告和可视化图表

### 4.2 快速开始

```bash
# 一键处理 VisDrone 数据集（推荐）
python scripts/data_processing/visdrone/process_visdrone_complete.py \
    --input data/VisDrone2019-DET-train \
    --output data/visdrone_yolo \
    --verbose

# 分步处理
python scripts/data_processing/visdrone/convert_visdrone.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo
python scripts/data_processing/visdrone/split_visdrone_dataset.py -i data/visdrone_yolo -o data/visdrone_yolo
python scripts/data_processing/visdrone/validate_visdrone_dataset.py -d data/visdrone_yolo --visualize

# 查看演示和帮助
python scripts/data_processing/demos/demo_visdrone_processing.py
```

### 4.3 类别映射

| VisDrone | YOLO | 类别名称 |
|----------|------|----------|
| 1 | 0 | pedestrian (行人) |
| 2 | 1 | people (人群) |
| 3 | 2 | bicycle (自行车) |
| 4 | 3 | car (汽车) |
| 5 | 4 | van (面包车) |
| 6 | 5 | truck (卡车) |
| 7 | 6 | tricycle (三轮车) |
| 8 | 7 | awning-tricycle (遮阳三轮车) |
| 9 | 8 | bus (公交车) |
| 10 | 9 | motor (摩托车) |

详细文档: [VisDrone 处理工具说明](scripts/README_VisDrone.md)

## 5. YOLOv8目标检测系统

### 5.1 功能概述

YOLOv8目标检测系统是一个基于PyQt5开发的图形界面应用，提供了以下功能：

- **多种输入源支持**：
  - 图片检测：加载并检测单张图片
  - 视频检测：加载并检测视频文件
  - 摄像头检测：实时检测摄像头画面

- **可视化界面**：
  - 双视图对比：同时显示原始图像和检测结果
  - 实时参数调节：调整置信度和IoU阈值
  - 检测结果表格：详细显示检测到的对象信息

- **结果存储**：
  - 保存检测结果图像到results/images目录
  - 自动保存视频检测结果到results/videos目录
  - 自动保存摄像头检测结果到results/camera目录

### 5.2 系统要求

- Python 3.7+
- PyQt5
- OpenCV
- Ultralytics YOLOv8
- NumPy

### 5.3 安装与运行

1. 确保已安装所需依赖：
   ```bash
   pip install PyQt5 opencv-python ultralytics numpy
   ```

2. 运行主程序：
   ```bash
   python main.py
   ```

### 5.4 使用说明

1. **加载模型**：
   - 从下拉列表选择models目录中的模型文件
   - 点击"加载模型"按钮

2. **调整参数**：
   - 使用滑块调整置信度阈值（默认0.25）
   - 使用滑块调整IoU阈值（默认0.45）

3. **选择检测模式**：
   - 图片检测：选择并检测单张图片
   - 视频检测：选择并检测视频文件
   - 摄像头检测：使用摄像头进行实时检测

4. **查看结果**：
   - 上方显示原始图像
   - 下方显示检测结果图像
   - 右侧表格显示检测到的对象详情

5. **保存结果**：
   - 点击"保存结果"按钮保存当前检测结果图像
   - 视频和摄像头检测结果自动保存在results目录下的相应子目录中

## 6. 数据标注与准备流程

### 6.1 准备工作

1. **安装LabelMe**：
   ```bash
   pip install labelme
   ```

2. **准备类别文件**：
   在`data/classes.txt`中列出所有目标类别，每行一个类别名称。

### 6.2 数据标注流程

1. **准备原始图像**：
   将待标注的图像放入`data/raw_images/`目录。

2. **使用LabelMe标注**：
   ```bash
   labelme data/raw_images/ --output data/annotations/ --labels data/classes.txt
   ```

3. **转换为YOLO格式**：
   ```bash
   python scripts/labelme2yolo.py
   ```

4. **划分数据集**：
   ```bash
   python scripts/split_dataset.py
   ```

### 6.3 模型训练

使用准备好的数据集训练模型：

```bash
# 训练标准 YOLOv8 模型
yolo task=detect mode=train model=yolov8s.pt data=data/yolo_dataset/data.yaml epochs=100 imgsz=640

# 训练 Drone-YOLO 模型
python scripts/training/train_drone_yolo.py --config assets/configs/yolov8s-drone.yaml --data data/yolo_dataset/data.yaml
```

训练完成后，将生成的模型文件（如best.pt）放入`models/`目录，即可在UI界面中使用。

## 7. 项目文档

详细文档分布在以下目录：

- **📚 docs/**: 完整的技术文档和教程 → [查看详情](docs/README.md)
- **📖 doc/**: 传统文档目录
  - `ui_guide.md`：UI界面使用指南
  - `development_guide.md`：开发者指南
- **🚁 VisDrone 工具**: VisDrone 数据集处理工具文档
  - [VisDrone 处理工具说明](scripts/docs/VisDrone工具说明.md)
  - [VisDrone 数据集处理工具总结](docs/VisDrone数据集处理工具总结.md)

## 8. 贡献与支持

欢迎提交问题报告和功能建议。如需贡献代码，请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request
