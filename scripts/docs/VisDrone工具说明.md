# VisDrone2019 数据集处理脚本

本目录包含了完整的 VisDrone2019 数据集转换和处理脚本，用于将 VisDrone 格式的数据集转换为 YOLO 格式，并进行数据集划分和验证。

## 📁 脚本文件

### 核心脚本

1. **`convert_visdrone.py`** - VisDrone 格式转换脚本
   - 将 VisDrone 标注格式转换为 YOLO 格式
   - 过滤 ignored regions (class 0)
   - 处理类别映射 (1-10 → 0-9)
   - 支持批量处理和进度显示

2. **`split_visdrone_dataset.py`** - 数据集划分脚本
   - 按 8:1:1 比例划分训练集、验证集和测试集
   - 创建标准 YOLO 数据集目录结构
   - 生成 data.yaml 配置文件

3. **`validate_visdrone_dataset.py`** - 数据集验证脚本
   - 检查数据完整性和标注格式正确性
   - 统计各类别样本数量
   - 生成数据集统计报告和可视化图表

4. **`process_visdrone_complete.py`** - 一键处理脚本
   - 自动执行完整的转换、划分、验证流程
   - 提供详细的处理日志和错误处理

## 🚀 快速开始

### 方法一：一键处理（推荐）

```bash
# 一键完成所有处理步骤
python scripts/process_visdrone_complete.py \
    --input data/VisDrone2019-DET-train \
    --output data/visdrone_yolo \
    --verbose
```

### 方法二：分步处理

```bash
# 步骤1: 转换格式
python scripts/convert_visdrone.py \
    --input data/VisDrone2019-DET-train \
    --output data/visdrone_yolo \
    --verbose

# 步骤2: 划分数据集
python scripts/split_visdrone_dataset.py \
    --input data/visdrone_yolo \
    --output data/visdrone_yolo \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --verbose

# 步骤3: 验证数据集
python scripts/validate_visdrone_dataset.py \
    --dataset data/visdrone_yolo \
    --visualize \
    --output-dir outputs/validation \
    --verbose
```

## 📊 数据集格式转换

### VisDrone 原始格式
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

### YOLO 目标格式
```
<class_id> <x_center> <y_center> <width> <height>
```

### 类别映射

| VisDrone 类别 | YOLO 类别 | 类别名称 |
|---------------|-----------|----------|
| 0 | - | ignored regions (过滤) |
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

## 📂 输出目录结构

处理完成后，输出目录结构如下：

```
data/visdrone_yolo/
├── images/
│   ├── train/          # 训练集图像
│   ├── val/            # 验证集图像
│   └── test/           # 测试集图像
├── labels/
│   ├── train/          # 训练集标签
│   ├── val/            # 验证集标签
│   └── test/           # 测试集标签
├── data.yaml           # YOLO 数据集配置文件
├── classes.txt         # 类别名称文件
└── dataset_statistics.png  # 数据集统计可视化图表
```

## ⚙️ 脚本参数说明

### convert_visdrone.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | VisDrone 数据集根目录 | 必需 |
| `--output, -o` | 输出目录 | 必需 |
| `--verbose, -v` | 显示详细日志 | False |

### split_visdrone_dataset.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | 输入目录 | 必需 |
| `--output, -o` | 输出目录 | 必需 |
| `--train-ratio` | 训练集比例 | 0.8 |
| `--val-ratio` | 验证集比例 | 0.1 |
| `--test-ratio` | 测试集比例 | 0.1 |
| `--seed` | 随机种子 | 42 |
| `--verbose, -v` | 显示详细日志 | False |

### validate_visdrone_dataset.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset, -d` | 数据集根目录 | 必需 |
| `--visualize` | 生成可视化图表 | False |
| `--output-dir, -o` | 输出目录 | None |
| `--verbose, -v` | 显示详细日志 | False |

### process_visdrone_complete.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | VisDrone 数据集根目录 | 必需 |
| `--output, -o` | 输出目录 | 必需 |
| `--train-ratio` | 训练集比例 | 0.8 |
| `--val-ratio` | 验证集比例 | 0.1 |
| `--test-ratio` | 测试集比例 | 0.1 |
| `--no-visualization` | 跳过可视化生成 | False |
| `--verbose, -v` | 显示详细日志 | False |

## 📋 依赖要求

### 必需依赖
```bash
pip install pillow tqdm pyyaml
```

### 可选依赖（用于可视化）
```bash
pip install matplotlib seaborn numpy
```

## 🔧 使用示例

### 基础使用
```bash
# 处理完整的 VisDrone 训练集
python scripts/process_visdrone_complete.py \
    -i data/VisDrone2019-DET-train \
    -o data/visdrone_yolo
```

### 自定义划分比例
```bash
# 使用 7:2:1 的划分比例
python scripts/process_visdrone_complete.py \
    -i data/VisDrone2019-DET-train \
    -o data/visdrone_yolo \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1
```

### 仅转换格式
```bash
python scripts/convert_visdrone.py \
    -i data/VisDrone2019-DET-train \
    -o data/visdrone_converted
```

### 验证现有数据集
```bash
python scripts/validate_visdrone_dataset.py \
    -d data/visdrone_yolo \
    --visualize \
    -o outputs/validation
```

## 📈 训练模型

处理完成后，可以使用生成的数据集训练 YOLO 模型：

### 使用标准 YOLOv8
```bash
yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=100
```

### 使用 Drone-YOLO 配置
```bash
python train.py \
    --data data/visdrone_yolo/data.yaml \
    --cfg assets/configs/yolov8s-drone.yaml \
    --epochs 300 \
    --batch-size 8
```

## 🐛 故障排除

### 常见问题

1. **找不到图像文件**
   - 确保 VisDrone 数据集目录包含 `images` 和 `annotations` 子目录
   - 检查图像文件扩展名是否正确 (.jpg, .jpeg, .png, .bmp)

2. **内存不足**
   - 减少批处理大小
   - 关闭图像缓存 (`cache: false`)

3. **标注格式错误**
   - 检查 VisDrone 标注文件格式是否正确
   - 查看转换日志中的警告信息

4. **可视化失败**
   - 安装可视化依赖: `pip install matplotlib seaborn`
   - 使用 `--no-visualization` 跳过可视化

### 日志文件

脚本运行时会生成详细的日志文件：
- `visdrone_conversion.log` - 格式转换日志
- `visdrone_split.log` - 数据集划分日志
- `visdrone_validation.log` - 数据集验证日志
- `visdrone_complete_process.log` - 完整处理日志

## 📞 技术支持

如果遇到问题，请：
1. 查看相应的日志文件
2. 使用 `--verbose` 参数获取详细信息
3. 检查输入数据格式是否正确
4. 确保所有依赖已正确安装

---

**YOLOvision Pro Team**  
*专业的 YOLO 目标检测解决方案*
