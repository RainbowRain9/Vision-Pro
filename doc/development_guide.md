# YOLOv8目标检测系统 - 开发指南

## 1. 项目结构

YOLOv8目标检测系统的代码结构如下：

```
yolo_project/
├── main.py                      # 主程序入口
├── output/                      # 检测结果输出目录
├── models/                      # 模型文件目录
│   └── best.pt                  # 默认模型文件
├── doc/                         # 文档目录
│   ├── ui_guide.md              # 用户界面指南
│   └── development_guide.md     # 开发指南
└── README.md                    # 项目说明文档
```

## 2. 代码架构

### 2.1 主要类和函数

- **YOLODetectionUI**：主界面类，继承自QMainWindow
  - `setup_ui()`：设置UI界面
  - `setup_left_panel()`：设置左侧图像显示面板
  - `setup_right_panel()`：设置右侧控制面板
  - `setup_model_group()`：设置模型选择组
  - `setup_param_group()`：设置参数设置组
  - `setup_function_group()`：设置功能按钮组
  - `setup_result_table_group()`：设置结果表格组
  - `set_style()`：设置全局样式
  - `connect_signals()`：连接信号槽

- **功能方法**：
  - `load_model()`：加载YOLO模型
  - `detect_image()`：图片检测功能
  - `detect_video()`：视频检测功能
  - `detect_camera()`：摄像头检测功能
  - `update_camera_frame()`：更新摄像头/视频帧
  - `stop_detection()`：停止检测
  - `save_result()`：保存检测结果
  - `display_image()`：在标签控件中显示图像
  - `update_result_table()`：更新检测结果表格
  - `update_conf_value()`：更新置信度值显示
  - `update_iou_value()`：更新IoU值显示

- **辅助类**：
  - `CenteredDelegate`：表格内容居中显示的代理类

### 2.2 依赖关系

- **PyQt5**：提供GUI界面
- **OpenCV**：处理图像和视频
- **Ultralytics YOLOv8**：提供目标检测功能
- **NumPy**：数据处理

## 3. 开发指南

### 3.1 环境设置

1. 安装Python 3.7+
2. 安装依赖库：
   ```bash
   pip install PyQt5 opencv-python ultralytics numpy
   ```

### 3.2 添加新功能

#### 3.2.1 添加新的检测源

如果要添加新的检测源（如网络摄像头、图片目录等），可以参考以下步骤：

1. 在`YOLODetectionUI`类中添加新的方法，如`detect_network_camera()`
2. 在`setup_function_group()`中添加新的按钮
3. 在`connect_signals()`中连接新按钮的点击事件
4. 实现新的检测逻辑

#### 3.2.2 添加新的结果展示方式

如果要添加新的结果展示方式（如热力图、3D视图等），可以参考以下步骤：

1. 在`setup_left_panel()`中添加新的显示区域
2. 在检测方法中添加新的结果处理逻辑
3. 创建新的显示方法

### 3.3 修改现有功能

#### 3.3.1 修改检测参数

如果要修改或添加检测参数，可以参考以下步骤：

1. 在`setup_param_group()`中添加新的参数控件
2. 在`connect_signals()`中连接新参数的变化事件
3. 在检测方法中使用新参数

#### 3.3.2 修改界面样式

如果要修改界面样式，可以参考以下步骤：

1. 修改`set_style()`方法中的样式表
2. 或者修改各个组件的样式设置

### 3.4 调试技巧

1. 使用`print()`语句或日志记录关键信息
2. 使用`try-except`块捕获和处理异常
3. 使用PyQt的调试工具，如QDebug
4. 使用OpenCV的`imshow()`函数查看中间图像处理结果

## 4. 常见问题解决

### 4.1 模型加载问题

- **问题**：模型加载失败
- **解决方案**：
  - 确保模型文件路径正确
  - 检查模型文件是否损坏
  - 确保已安装正确版本的ultralytics库

### 4.2 视频处理问题

- **问题**：视频处理速度慢
- **解决方案**：
  - 降低视频分辨率
  - 减少处理帧率
  - 使用GPU加速（如果可用）

### 4.3 界面卡顿问题

- **问题**：界面在处理大文件时卡顿
- **解决方案**：
  - 使用多线程处理视频和图像
  - 优化图像显示逻辑
  - 减少UI更新频率

## 5. 扩展开发

### 5.1 添加多模型支持

可以扩展系统以支持多种不同的模型，如YOLOv5、YOLOv7等：

1. 创建模型适配器类
2. 在界面中添加模型类型选择
3. 根据选择的模型类型使用相应的适配器

### 5.2 添加结果分析功能

可以添加检测结果的统计和分析功能：

1. 创建结果分析类
2. 收集和处理检测结果数据
3. 生成统计图表和报告

### 5.3 添加批处理功能

可以添加批量处理图片或视频的功能：

1. 创建批处理管理器类
2. 添加文件队列管理
3. 实现批量处理逻辑
