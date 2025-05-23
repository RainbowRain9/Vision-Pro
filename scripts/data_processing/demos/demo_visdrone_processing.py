#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019 数据集处理演示脚本
演示如何使用 VisDrone 数据集处理工具链

作者: YOLOvision Pro Team
日期: 2024
"""

import os
import sys
from pathlib import Path

def print_banner():
    """打印横幅"""
    print("="*80)
    print("🚁 VisDrone2019 数据集处理工具演示")
    print("="*80)
    print("本演示将展示如何使用 YOLOvision Pro 的 VisDrone 数据集处理工具链")
    print("包括格式转换、数据集划分和验证等功能")
    print("="*80)

def check_requirements():
    """检查依赖要求"""
    print("\n📋 检查依赖要求...")

    required_packages = ['PIL', 'yaml', 'pathlib']
    optional_packages = ['tqdm', 'matplotlib', 'numpy']

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} - 已安装")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package} - 缺失 (必需)")

    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} - 已安装")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠ {package} - 缺失 (可选)")

    if missing_required:
        print(f"\n❌ 缺少必需依赖: {', '.join(missing_required)}")
        print("请运行: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\n⚠️ 缺少可选依赖: {', '.join(missing_optional)}")
        print("建议运行: pip install " + " ".join(missing_optional))
        print("(可选依赖用于进度显示和可视化)")

    print("\n✅ 依赖检查完成")
    return True

def check_scripts():
    """检查脚本文件"""
    print("\n📁 检查脚本文件...")

    # 脚本现在在 visdrone 子目录中
    scripts_dir = Path(__file__).parent.parent / "visdrone"
    required_scripts = [
        'convert_visdrone.py',
        'split_visdrone_dataset.py',
        'validate_visdrone_dataset.py',
        'process_visdrone_complete.py'
    ]

    missing_scripts = []
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✓ {script} - 存在")
        else:
            missing_scripts.append(script)
            print(f"✗ {script} - 缺失")

    if missing_scripts:
        print(f"\n❌ 缺少脚本文件: {', '.join(missing_scripts)}")
        return False

    print("\n✅ 脚本文件检查完成")
    return True

def show_usage_examples():
    """显示使用示例"""
    print("\n📖 使用示例")
    print("-" * 50)

    print("\n1️⃣ 一键处理 (推荐)")
    print("python scripts/data_processing/visdrone/process_visdrone_complete.py \\")
    print("    --input data/VisDrone2019-DET-train \\")
    print("    --output data/visdrone_yolo \\")
    print("    --verbose")

    print("\n2️⃣ 分步处理")
    print("# 步骤1: 格式转换")
    print("python scripts/data_processing/visdrone/convert_visdrone.py \\")
    print("    -i data/VisDrone2019-DET-train \\")
    print("    -o data/visdrone_yolo")

    print("\n# 步骤2: 数据集划分")
    print("python scripts/data_processing/visdrone/split_visdrone_dataset.py \\")
    print("    -i data/visdrone_yolo \\")
    print("    -o data/visdrone_yolo")

    print("\n# 步骤3: 数据集验证")
    print("python scripts/data_processing/visdrone/validate_visdrone_dataset.py \\")
    print("    -d data/visdrone_yolo \\")
    print("    --visualize")

    print("\n3️⃣ 训练模型")
    print("# 使用标准 YOLOv8")
    print("yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=100")

    print("\n# 使用 Drone-YOLO 配置")
    print("python train.py \\")
    print("    --data data/visdrone_yolo/data.yaml \\")
    print("    --cfg assets/configs/yolov8s-drone.yaml")

def show_dataset_info():
    """显示数据集信息"""
    print("\n📊 VisDrone2019 数据集信息")
    print("-" * 50)

    print("🎯 数据集特点:")
    print("  • 无人机航拍图像数据集")
    print("  • 专注于小目标检测")
    print("  • 10个目标类别")
    print("  • 复杂的城市场景")

    print("\n🏷️ 类别映射:")
    categories = [
        ("1→0", "pedestrian", "行人"),
        ("2→1", "people", "人群"),
        ("3→2", "bicycle", "自行车"),
        ("4→3", "car", "汽车"),
        ("5→4", "van", "面包车"),
        ("6→5", "truck", "卡车"),
        ("7→6", "tricycle", "三轮车"),
        ("8→7", "awning-tricycle", "遮阳三轮车"),
        ("9→8", "bus", "公交车"),
        ("10→9", "motor", "摩托车")
    ]

    for mapping, eng_name, chn_name in categories:
        print(f"  {mapping}: {eng_name} ({chn_name})")

    print("\n📁 预期目录结构:")
    print("data/VisDrone2019-DET-train/")
    print("├── images/          # 图像文件")
    print("└── annotations/     # 标注文件")

def show_output_structure():
    """显示输出结构"""
    print("\n📂 输出目录结构")
    print("-" * 50)

    print("data/visdrone_yolo/")
    print("├── images/")
    print("│   ├── train/       # 训练集图像 (80%)")
    print("│   ├── val/         # 验证集图像 (10%)")
    print("│   └── test/        # 测试集图像 (10%)")
    print("├── labels/")
    print("│   ├── train/       # 训练集标签")
    print("│   ├── val/         # 验证集标签")
    print("│   └── test/        # 测试集标签")
    print("├── data.yaml        # YOLO 配置文件")
    print("├── classes.txt      # 类别名称")
    print("└── dataset_statistics.png  # 统计图表")

def main():
    """主函数"""
    print_banner()

    # 检查环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请安装缺失的依赖后重试")
        return

    if not check_scripts():
        print("\n❌ 脚本文件检查失败，请确保所有脚本文件存在")
        return

    # 显示信息
    show_dataset_info()
    show_output_structure()
    show_usage_examples()

    print("\n" + "="*80)
    print("🎉 演示完成!")
    print("="*80)
    print("现在您可以使用上述命令处理 VisDrone 数据集")
    print("如需帮助，请查看 scripts/README_VisDrone.md")
    print("="*80)

if __name__ == "__main__":
    main()
