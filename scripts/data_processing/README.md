# 数据处理脚本

本目录包含所有数据处理相关的脚本，用于数据集转换、预处理和验证。

## 📁 子目录结构

- **visdrone/** - VisDrone 数据集专用处理工具
- **general/** - 通用数据处理工具
- **demos/** - 数据处理演示脚本

## 🚀 快速开始

```bash
# VisDrone 数据集一键处理
python data_processing/visdrone/process_visdrone_complete.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo

# LabelMe 转 YOLO 格式
python data_processing/general/labelme2yolo.py

# 查看 VisDrone 处理演示
python data_processing/demos/demo_visdrone_processing.py
```
