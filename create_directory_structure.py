#!/usr/bin/env python3
"""
在 yolovision_pro 项目中创建 Drone-YOLO 学习资料目录结构的脚本
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """在当前项目根目录中创建目录结构"""

    # 当前目录作为项目根目录 (yolovision_pro)
    base_dir = Path(".")

    # 定义目录结构 - 直接在项目根目录下创建
    directories = [
        # 文档目录
        base_dir / "docs",
        base_dir / "docs" / "technical_analysis",
        base_dir / "docs" / "tutorials",
        base_dir / "docs" / "references",

        # 脚本目录
        base_dir / "scripts",
        base_dir / "scripts" / "demo",
        base_dir / "scripts" / "testing",
        base_dir / "scripts" / "visualization",
        base_dir / "scripts" / "training",

        # 资源目录
        base_dir / "assets",
        base_dir / "assets" / "images",
        base_dir / "assets" / "images" / "architecture",
        base_dir / "assets" / "images" / "results",
        base_dir / "assets" / "images" / "demos",
        base_dir / "assets" / "configs",
        base_dir / "assets" / "configs" / "training_configs",
        base_dir / "assets" / "data",
        base_dir / "assets" / "data" / "sample_images",
        base_dir / "assets" / "data" / "annotations",

        # 实验目录
        base_dir / "experiments",
        base_dir / "experiments" / "baseline_comparison",
        base_dir / "experiments" / "ablation_studies",
        base_dir / "experiments" / "performance_analysis",

        # 输出目录
        base_dir / "outputs",
        base_dir / "outputs" / "models",
        base_dir / "outputs" / "logs",
        base_dir / "outputs" / "results",
    ]

    # 创建所有目录
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

    return base_dir

def move_existing_files(base_dir):
    """移动现有文件到相应目录"""

    # 定义文件移动映射
    file_moves = [
        # 技术文档
        ("drone_yolo_detailed_explanation.md", "docs/technical_analysis"),

        # 演示脚本
        ("drone_yolo_demo.py", "scripts/demo"),

        # 测试脚本
        ("test_drone_yolo.py", "scripts/testing"),

        # 可视化脚本
        ("visualize_drone_yolo.py", "scripts/visualization"),

        # 配置文件 (复制而不是移动，保持原位置)
        ("ultralytics/ultralytics/cfg/models/v8/yolov8s-drone.yaml", "assets/configs"),
    ]

    # 可能的图片文件
    image_files = [
        "repvgg_structure.png",
        "detection_layers.png",
        "sandwich_fusion.png"
    ]

    # 移动文件
    for src_file, dest_dir in file_moves:
        src_path = Path(src_file)
        dest_path = Path(dest_dir) / src_path.name

        if src_path.exists():
            if "yolov8s-drone.yaml" in src_file:
                # 配置文件复制而不是移动
                shutil.copy2(str(src_path), str(dest_path))
                print(f"📋 复制配置文件: {src_file} → {dest_path}")
            else:
                # 其他文件移动
                shutil.move(str(src_path), str(dest_path))
                print(f"📁 移动文件: {src_file} → {dest_path}")
        else:
            print(f"⚠️ 文件不存在: {src_file}")

    # 移动图片文件
    for img_file in image_files:
        src_path = Path(img_file)
        if src_path.exists():
            dest_path = Path("assets/images/architecture") / img_file
            shutil.move(str(src_path), str(dest_path))
            print(f"🖼️ 移动图片: {img_file} → {dest_path}")

if __name__ == "__main__":
    print("🚁 在 yolovision_pro 项目中创建 Drone-YOLO 学习资料目录结构...")

    # 创建目录结构
    base_dir = create_directory_structure()

    # 移动现有文件
    move_existing_files(base_dir)

    print(f"\n✅ 目录结构创建完成！")
    print(f"📁 项目根目录: {Path('.').absolute()}")
    print(f"📁 新增的主要目录: docs/, scripts/, assets/, experiments/, outputs/")
