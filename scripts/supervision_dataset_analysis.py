#!/usr/bin/env python3
"""
使用 Supervision 进行 VisDrone 数据集分析和可视化
"""

import os
import cv2
import yaml
import numpy as np
import supervision as sv
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import random

def load_visdrone_config():
    """加载 VisDrone 数据集配置"""
    config_path = Path("data/visdrone_yolo/data.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def analyze_dataset_statistics(config):
    """分析数据集统计信息"""
    print("📊 数据集统计分析")
    print("=" * 50)
    
    base_path = Path(config['path'])
    splits = ['train', 'val', 'test']
    
    total_images = 0
    total_annotations = 0
    class_counts = Counter()
    
    for split in splits:
        images_dir = base_path / "images" / split
        labels_dir = base_path / "labels" / split
        
        if not images_dir.exists():
            print(f"⚠️ {split} 图像目录不存在: {images_dir}")
            continue
            
        if not labels_dir.exists():
            print(f"⚠️ {split} 标签目录不存在: {labels_dir}")
            continue
        
        # 统计图像数量
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        split_image_count = len(image_files)
        total_images += split_image_count
        
        # 统计标注数量
        split_annotation_count = 0
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                split_annotation_count += len(lines)
                
                # 统计类别分布
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_name = config['names'][class_id]
                        class_counts[class_name] += 1
        
        total_annotations += split_annotation_count
        
        print(f"✅ {split:5} 集: {split_image_count:5} 图像, {split_annotation_count:6} 标注")
    
    print(f"\n📈 总计: {total_images} 图像, {total_annotations} 标注")
    print(f"📈 平均每张图像: {total_annotations/total_images:.1f} 个目标")
    
    # 显示类别分布
    print(f"\n🏷️ 类别分布:")
    for class_name, count in class_counts.most_common():
        percentage = count / total_annotations * 100
        print(f"   {class_name:15}: {count:6} ({percentage:5.1f}%)")
    
    return {
        'total_images': total_images,
        'total_annotations': total_annotations,
        'class_counts': class_counts
    }

def visualize_sample_images(config, num_samples=6):
    """使用 Supervision 可视化样本图像"""
    print(f"\n🖼️ 可视化 {num_samples} 个样本图像")
    print("=" * 50)
    
    base_path = Path(config['path'])
    train_images_dir = base_path / "images" / "train"
    train_labels_dir = base_path / "labels" / "train"
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        print("❌ 训练集目录不存在")
        return
    
    # 随机选择样本
    image_files = list(train_images_dir.glob("*.jpg"))
    if len(image_files) < num_samples:
        num_samples = len(image_files)
    
    sample_files = random.sample(image_files, num_samples)
    
    # 创建注释器
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_padding=5
    )
    
    # 创建输出目录
    output_dir = Path("outputs/supervision_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, image_file in enumerate(sample_files):
        # 加载图像
        image = cv2.imread(str(image_file))
        if image is None:
            continue
            
        # 加载对应的标签
        label_file = train_labels_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            continue
        
        # 解析 YOLO 格式标签
        detections_data = []
        labels = []
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        h, w = image.shape[:2]
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # 转换为像素坐标
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            # 转换为边界框格式 (x1, y1, x2, y2)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            detections_data.append([x1, y1, x2, y2])
            labels.append(f"{config['names'][class_id]}")
        
        if not detections_data:
            continue
        
        # 创建 Supervision Detections 对象
        detections = sv.Detections(
            xyxy=np.array(detections_data),
            class_id=np.array([i for i in range(len(detections_data))])
        )
        
        # 添加注释
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )
        
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )
        
        # 保存结果
        output_path = output_dir / f"sample_{i+1}_{image_file.name}"
        cv2.imwrite(str(output_path), annotated_image)
        print(f"✅ 保存样本 {i+1}: {output_path}")
    
    print(f"\n📁 可视化结果保存在: {output_dir}")

def create_class_distribution_plot(class_counts, output_dir):
    """创建类别分布图"""
    plt.figure(figsize=(12, 8))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts)
    plt.title('VisDrone Dataset - Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 类别分布图保存在: {plot_path}")

def main():
    """主函数"""
    print("🚀 使用 Supervision 分析 VisDrone 数据集")
    print("=" * 60)
    
    try:
        # 加载配置
        config = load_visdrone_config()
        print(f"✅ 数据集配置加载成功")
        print(f"📁 数据集路径: {config['path']}")
        print(f"🏷️ 类别数量: {config['nc']}")
        
        # 分析统计信息
        stats = analyze_dataset_statistics(config)
        
        # 可视化样本
        visualize_sample_images(config, num_samples=6)
        
        # 创建类别分布图
        output_dir = Path("outputs/supervision_analysis")
        create_class_distribution_plot(stats['class_counts'], output_dir)
        
        print(f"\n🎉 数据集分析完成！")
        print(f"📊 统计信息:")
        print(f"   - 总图像数: {stats['total_images']}")
        print(f"   - 总标注数: {stats['total_annotations']}")
        print(f"   - 平均密度: {stats['total_annotations']/stats['total_images']:.1f} 目标/图像")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
