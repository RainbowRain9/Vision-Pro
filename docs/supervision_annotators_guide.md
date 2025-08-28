# Supervision 标注器功能使用指南

## 📖 概述

YOLOvision Pro 集成了 Supervision 库的多种标注器功能，提供丰富的目标检测可视化效果。本指南将详细介绍如何使用这些功能。

## 🎯 支持的标注器

### 1. BoxAnnotator (边界框标注器)
- **功能**: 绘制目标边界框
- **用途**: 基础的目标检测可视化
- **配置参数**: 
  - `thickness`: 边界框线条粗细
  - `color`: 边界框颜色

### 2. LabelAnnotator (标签标注器)
- **功能**: 显示类别标签和置信度
- **用途**: 显示检测结果的详细信息
- **配置参数**:
  - `text_scale`: 文字大小
  - `text_thickness`: 文字粗细
  - `text_padding`: 文字内边距

### 3. MaskAnnotator (分割掩码标注器)
- **功能**: 渲染分割掩码
- **用途**: 显示精确的目标轮廓
- **要求**: 需要分割模型支持 (如 YOLOv8-seg)
- **配置参数**:
  - `opacity`: 掩码透明度

### 4. PolygonAnnotator (多边形标注器)
- **功能**: 绘制多边形轮廓
- **用途**: 显示目标的多边形边界
- **配置参数**:
  - `thickness`: 线条粗细

### 5. HeatMapAnnotator (热力图标注器)
- **功能**: 生成目标密度热力图
- **用途**: 分析目标分布和活动热点
- **特点**: 需要多帧数据累积
- **配置参数**:
  - `opacity`: 热力图透明度
  - `radius`: 热点半径

### 6. BlurAnnotator (模糊标注器)
- **功能**: 对检测区域应用模糊效果
- **用途**: 隐私保护，敏感内容处理
- **配置参数**:
  - `kernel_size`: 模糊核大小

### 7. PixelateAnnotator (像素化标注器)
- **功能**: 对检测区域应用像素化效果
- **用途**: 隐私保护，艺术效果
- **配置参数**:
  - `pixel_size`: 像素块大小

## 🎨 预设配置

### 基础模式 (basic)
- **包含**: BoxAnnotator + LabelAnnotator
- **用途**: 基本的目标检测显示
- **适用场景**: 日常检测任务

### 详细模式 (detailed)
- **包含**: BoxAnnotator + LabelAnnotator + PolygonAnnotator
- **用途**: 详细的检测信息显示
- **适用场景**: 精确分析需求

### 隐私保护模式 (privacy)
- **包含**: BlurAnnotator + LabelAnnotator
- **用途**: 保护隐私的同时显示检测信息
- **适用场景**: 监控视频处理

### 分析模式 (analysis)
- **包含**: BoxAnnotator + LabelAnnotator + HeatMapAnnotator + PolygonAnnotator
- **用途**: 深度分析和统计
- **适用场景**: 行为分析，流量统计

### 分割模式 (segmentation)
- **包含**: MaskAnnotator + LabelAnnotator + PolygonAnnotator
- **用途**: 显示精确的分割结果
- **适用场景**: 实例分割任务

### 演示模式 (presentation)
- **包含**: BoxAnnotator + LabelAnnotator (加粗显示)
- **用途**: 清晰的演示效果
- **适用场景**: 展示和演讲

## 🖥️ 界面使用

### 主界面控制
1. **预设选择**: 在右侧面板的"标注器设置"组中选择预设
2. **应用预设**: 点击"应用预设"按钮
3. **单个控制**: 使用复选框单独启用/禁用标注器
4. **清除热力图**: 点击"清除热力图"按钮重置热力图数据

### 快捷操作
- 选择预设后自动更新复选框状态
- 手动调整复选框会切换到"自定义模式"
- 状态标签实时显示当前配置

## 🔧 配置文件

### 配置文件位置
```
assets/configs/annotator_config.yaml
```

### 配置文件结构
```yaml
annotators:
  box:
    enabled: true
    thickness: 2
  label:
    enabled: true
    text_scale: 0.5
  # ... 其他标注器配置

presets:
  basic:
    annotators: ["box", "label"]
  # ... 其他预设配置
```

### 自定义配置
1. 编辑配置文件
2. 重启应用程序
3. 或使用 API 动态更新配置

## 📝 编程接口

### 基本使用
```python
from scripts.modules.supervision_annotators import AnnotatorManager

# 创建管理器
manager = AnnotatorManager("path/to/config.yaml")

# 设置预设
manager.set_preset('analysis')

# 标注图像
annotated_image = manager.annotate_image(image, detections, labels)
```

### 高级功能
```python
# 启用特定标注器
manager.enable_annotator(AnnotatorType.HEATMAP)

# 更新配置
manager.update_annotator_config(
    AnnotatorType.BOX, 
    thickness=5
)

# 批量标注
annotated_images = manager.batch_annotate(
    images, detections_list, labels_list
)
```

## 🚀 性能优化

### 实时应用建议
1. **限制标注器数量**: 同时启用的标注器越少，性能越好
2. **热力图优化**: 限制历史帧数 (默认100帧)
3. **分辨率控制**: 高分辨率图像会影响性能
4. **GPU加速**: 确保CUDA可用以提升性能

### 推荐配置
- **实时检测**: 使用 basic 或 detailed 预设
- **离线分析**: 可使用 analysis 预设
- **隐私场景**: 使用 privacy 预设

## 🐛 故障排除

### 常见问题

#### 1. 标注器初始化失败
**症状**: 某些标注器不可用
**解决方案**: 
- 检查 Supervision 版本兼容性
- 更新到最新版本: `pip install -U supervision`

#### 2. 分割掩码不显示
**症状**: MaskAnnotator 无效果
**解决方案**:
- 确保使用分割模型 (如 yolov8n-seg.pt)
- 检查检测结果是否包含掩码数据

#### 3. 热力图不显示
**症状**: HeatMapAnnotator 无效果
**解决方案**:
- 需要多帧数据累积 (至少5帧)
- 检查是否有检测结果

#### 4. 性能问题
**症状**: 处理速度慢
**解决方案**:
- 减少同时启用的标注器数量
- 降低图像分辨率
- 使用GPU加速

### 日志调试
启用详细日志查看问题详情:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 性能指标

### 内存使用
- 基础标注器: ~50MB
- 每个额外标注器: ~10MB
- 热力图历史: ~5MB/帧

### 推荐FPS
- 1-2个标注器: 30 FPS
- 3-4个标注器: 20 FPS
- 5+个标注器: 15 FPS

## 🔄 版本兼容性

### Supervision 版本支持
- **推荐版本**: 0.26.1+
- **最低版本**: 0.20.0
- **测试版本**: 0.26.1

### 模型兼容性
- ✅ YOLOv8 检测模型
- ✅ YOLOv8 分割模型
- ✅ Ultralytics 格式
- ✅ 自定义模型

## 📚 更多资源

### 官方文档
- [Supervision 官方文档](https://supervision.roboflow.com/)
- [YOLOv8 文档](https://docs.ultralytics.com/)

### 示例代码
- `scripts/demo/supervision_annotators_demo.py` - 完整演示
- `scripts/testing/test_supervision_annotators.py` - 测试用例

### 配置模板
- `assets/configs/annotator_config.yaml` - 配置文件模板

## 🤝 贡献指南

欢迎贡献新的标注器功能或改进现有功能：

1. Fork 项目
2. 创建功能分支
3. 添加测试用例
4. 提交 Pull Request

## 📄 许可证

本项目遵循 MIT 许可证。详见 LICENSE 文件。
