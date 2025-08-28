# 小目标检测使用指南

## 概述

YOLOvision Pro 集成了基于 Supervision 库的 InferenceSlicer 技术，专门用于提高小目标检测的精度。该功能特别适用于无人机图像、监控视频等包含大量小目标的场景。

## 核心技术

### InferenceSlicer 原理

InferenceSlicer 采用 **切片自适应推理 (SAHI)** 技术：

1. **图像切片**: 将大图像分割成多个小切片
2. **独立检测**: 对每个切片单独进行目标检测
3. **结果合并**: 使用 NMS 算法合并重叠检测结果
4. **坐标映射**: 将切片坐标映射回原图坐标

### 优势

- ✅ **提高小目标检测精度**: 小目标在切片中占比更大
- ✅ **处理高分辨率图像**: 避免直接缩放导致的信息丢失
- ✅ **灵活配置**: 支持多种切片策略和参数调整
- ✅ **并行处理**: 支持多线程加速

## 功能特性

### 1. 标准切片检测

使用固定尺寸的切片进行检测：

```python
from scripts.modules.supervision_wrapper import SupervisionWrapper

# 初始化包装器
wrapper = SupervisionWrapper(class_names=['car', 'person', 'bicycle'])

# 执行小目标检测
result = wrapper.detect_small_objects(
    image, model,
    conf=0.25, iou=0.45,
    slice_wh=(640, 640),      # 切片尺寸
    overlap_wh=(128, 128)     # 重叠尺寸
)
```

### 2. 多尺度检测

使用多个不同尺度的切片组合检测：

```python
# 多尺度检测
result = wrapper.detect_with_multiple_scales(
    image, model,
    conf=0.25, iou=0.45
)
```

### 3. 自适应切片

根据图像尺寸自动选择最优切片配置：

```python
# 获取自适应配置
optimal_config = wrapper.get_optimal_slice_config(image.shape[:2])

# 使用自适应配置检测
result = wrapper.detect_small_objects(
    image, model,
    slice_wh=optimal_config['slice_wh'],
    overlap_wh=optimal_config['overlap_wh']
)
```

## 配置参数详解

### 切片参数

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `slice_wh` | 切片尺寸 (宽, 高) | (640, 640) | 越小检测精度越高，处理时间越长 |
| `overlap_wh` | 重叠尺寸 (宽, 高) | (128, 128) | 有助于边界目标检测，增加计算量 |
| `iou_threshold` | NMS IoU 阈值 | 0.5 | 控制重复检测的过滤程度 |
| `thread_workers` | 线程数 | 1 | 并行处理加速，需考虑内存使用 |

### 预设配置

系统提供多种预设配置：

- **ultra_small**: 320×320 切片，适合极小目标
- **small**: 640×640 切片，标准配置
- **medium**: 800×800 切片，适合中等目标
- **large**: 1024×1024 切片，适合大目标

## 使用方法

### 1. GUI 界面使用

1. **启用小目标检测**
   - 勾选 "启用小目标检测 (InferenceSlicer)"

2. **选择检测模式**
   - 标准切片: 使用固定参数
   - 多尺度检测: 组合多个尺度
   - 自适应切片: 自动优化参数

3. **调整参数**
   - 切片尺寸: 选择合适的切片大小
   - 重叠尺寸: 设置重叠区域大小

4. **执行检测**
   - 点击 "图片检测" 开始处理

### 2. 命令行使用

运行演示脚本：

```bash
python scripts/demo/small_object_detection_demo.py
```

### 3. 编程接口

```python
import cv2
from scripts.modules.supervision_wrapper import SupervisionWrapper
from scripts.modules.small_object_config import get_config_manager

# 加载配置
config_manager = get_config_manager()
visdrone_config = config_manager.get_visdrone_optimized_config()

# 初始化包装器
class_names = config_manager.get_visdrone_class_names()
wrapper = SupervisionWrapper(class_names=class_names)

# 读取图像
image = cv2.imread('test_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 执行检测
result = wrapper.detect_small_objects(
    image_rgb, model,
    conf=visdrone_config.confidence_threshold,
    iou=visdrone_config.iou_threshold,
    slice_wh=visdrone_config.slice_wh,
    overlap_wh=visdrone_config.overlap_wh
)

# 获取结果
annotated_image = result['annotated_image']
detection_count = result['detection_count']
statistics = result['statistics']
```

## 性能优化建议

### 1. 参数调优

**切片尺寸选择**:
- 小目标多: 使用较小切片 (320×320)
- 处理速度优先: 使用较大切片 (800×800)
- 平衡选择: 使用标准切片 (640×640)

**重叠尺寸设置**:
- 目标密集: 增加重叠 (256×256)
- 处理速度优先: 减少重叠 (64×64)
- 标准设置: 使用 1/5 切片尺寸

### 2. 硬件优化

- **GPU 加速**: 确保 CUDA 可用
- **内存管理**: 大图像时减少并行线程
- **存储优化**: 使用 SSD 提高 I/O 性能

### 3. 算法优化

- **预处理**: 图像去噪和增强
- **后处理**: 自定义 NMS 参数
- **模型选择**: 使用针对小目标优化的模型

## 应用场景

### 1. 无人机图像分析

```python
# VisDrone 数据集优化配置
config = config_manager.get_visdrone_optimized_config()
result = wrapper.detect_small_objects(
    drone_image, model,
    slice_wh=config.slice_wh,
    overlap_wh=config.overlap_wh,
    conf=0.2  # 降低阈值检测更多小目标
)
```

### 2. 监控视频分析

```python
# 实时视频处理
for frame in video_stream:
    result = wrapper.detect_small_objects(
        frame, model,
        slice_wh=(480, 480),  # 较小切片适合实时处理
        overlap_wh=(96, 96)
    )
    # 处理检测结果...
```

### 3. 卫星图像分析

```python
# 高分辨率卫星图像
adaptive_config = wrapper.get_optimal_slice_config(satellite_image.shape[:2])
result = wrapper.detect_small_objects(
    satellite_image, model,
    slice_wh=adaptive_config['slice_wh'],
    overlap_wh=adaptive_config['overlap_wh']
)
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少切片尺寸
   - 降低并行线程数
   - 使用批处理模式

2. **检测精度低**
   - 减小切片尺寸
   - 增加重叠区域
   - 降低置信度阈值

3. **处理速度慢**
   - 增大切片尺寸
   - 减少重叠区域
   - 启用 GPU 加速

### 调试模式

启用调试模式查看详细信息：

```python
# 在配置文件中设置
debug:
  enabled: true
  show_slice_boundaries: true
  save_intermediate_results: true
```

## 最佳实践

1. **数据预处理**: 确保图像质量和格式正确
2. **参数测试**: 在小样本上测试不同参数组合
3. **性能监控**: 记录处理时间和内存使用
4. **结果验证**: 人工检查关键检测结果
5. **持续优化**: 根据实际效果调整配置

## 更新日志

- **v1.0.0**: 初始版本，支持基础切片检测
- **v1.1.0**: 添加多尺度检测功能
- **v1.2.0**: 集成自适应配置和 VisDrone 优化

---

更多技术细节请参考 [Supervision 官方文档](https://supervision.roboflow.com/latest/how_to/detect_small_objects/) 和项目源码。
