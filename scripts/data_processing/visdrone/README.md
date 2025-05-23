# VisDrone 数据集处理工具

专门用于处理 VisDrone2019 数据集的完整工具链。

## 🔧 核心脚本

- `convert_visdrone.py` - VisDrone 格式转换为 YOLO 格式
- `split_visdrone_dataset.py` - 数据集划分 (8:1:1)
- `validate_visdrone_dataset.py` - 数据集验证和统计
- `process_visdrone_complete.py` - 一键完整处理流程

## 🚀 使用方法

```bash
# 一键处理（推荐）
python process_visdrone_complete.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo

# 分步处理
python convert_visdrone.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo
python split_visdrone_dataset.py -i data/visdrone_yolo -o data/visdrone_yolo
python validate_visdrone_dataset.py -d data/visdrone_yolo --visualize
```

详细说明请参考: [VisDrone工具说明](../docs/VisDrone工具说明.md)
