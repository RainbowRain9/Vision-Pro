#!/usr/bin/env python3
"""
简化版训练结果分析脚本
避免OpenCV问题
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def analyze_class_distribution(label_dir, class_names):
    """分析类别分布"""
    print("📈 分析类别分布...")
    
    # 统计每个类别的实例数
    class_counts = {name: 0 for name in class_names}
    total_boxes = 0
    
    # 遍历所有标签文件
    label_files = list(Path(label_dir).glob("*.txt"))
    for label_file in label_files[:100]:  # 只分析前100个文件以加快速度
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                    class_counts[class_name] += 1
                    total_boxes += 1
    
    # 绘制类别分布图
    plt.figure(figsize=(12, 8))
    
    # 按数量排序
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*sorted_classes)
    
    bars = plt.bar(range(len(names)), counts, color='skyblue')
    plt.title('类别分布')
    plt.xlabel('类别')
    plt.ylabel('实例数')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    
    # 在柱状图上添加数值
    for i, (name, count) in enumerate(sorted_classes):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   类别分布图保存为: class_distribution.png")
    print(f"   总标注框数: {total_boxes}")
    print("\n   各类别统计:")
    for name, count in sorted_classes:
        if count > 0:
            percentage = (count / total_boxes) * 100 if total_boxes > 0 else 0
            print(f"      {name}: {count} ({percentage:.1f}%)")

def main():
    """主函数"""
    print("=" * 50)
    print("简化版训练结果分析")
    print("=" * 50)
    
    # 类别名称
    class_names = [
        'pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # 分析类别分布
    label_dir = "data/visdrone_yolo/labels/train"
    if Path(label_dir).exists():
        analyze_class_distribution(label_dir, class_names)
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()