#!/usr/bin/env python3
"""
训练结果可视化分析脚本
用于分析YOLOv8训练过程和结果
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def analyze_training_logs(log_dir):
    """分析训练日志"""
    print("📊 分析训练日志...")
    
    # 查找训练日志文件
    log_files = list(Path(log_dir).glob("results*.csv"))
    if not log_files:
        print("   ⚠️ 未找到训练日志文件")
        return
    
    log_file = log_files[0]
    print(f"   找到日志文件: {log_file.name}")
    
    # 读取日志数据
    import pandas as pd
    df = pd.read_csv(log_file)
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    plt.title('Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 分类损失
    plt.subplot(2, 3, 2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # DFL损失
    plt.subplot(2, 3, 3)
    plt.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
    plt.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
    plt.title('DFL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # mAP曲线
    plt.subplot(2, 3, 4)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.title('mAP Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    # 精确度和召回率
    plt.subplot(2, 3, 5)
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # F1分数
    plt.subplot(2, 3, 6)
    plt.plot(df['epoch'], df['metrics/F1(B)'], label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   训练分析图表保存为: training_analysis.png")
    
    # 打印最终结果
    final_epoch = df.iloc[-1]
    print(f"\n   最终训练结果 (Epoch {int(final_epoch['epoch'])}):")
    print(f"      Box Loss: {final_epoch['train/box_loss']:.4f}")
    print(f"      Class Loss: {final_epoch['train/cls_loss']:.4f}")
    print(f"      DFL Loss: {final_epoch['train/dfl_loss']:.4f}")
    print(f"      mAP@0.5: {final_epoch['metrics/mAP50(B)']:.4f}")
    print(f"      mAP@0.5:0.95: {final_epoch['metrics/mAP50-95(B)']:.4f}")
    print(f"      Precision: {final_epoch['metrics/precision(B)']:.4f}")
    print(f"      Recall: {final_epoch['metrics/recall(B)']:.4f}")
    print(f"      F1 Score: {final_epoch['metrics/F1(B)']:.4f}")

def visualize_detection_results(image_dir, label_dir, class_names):
    """可视化检测结果"""
    print("\n🎨 可视化检测结果...")
    
    # 获取图像文件
    image_files = list(Path(image_dir).glob("*.jpg"))[:10]  # 只处理前10张
    if not image_files:
        print("   ⚠️ 未找到图像文件")
        return
    
    # 创建可视化目录
    vis_dir = Path("detection_visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # 随机选择几张图像进行可视化
    import random
    selected_files = random.sample(image_files, min(5, len(image_files)))
    
    for img_file in selected_files:
        # 读取图像
        image = cv2.imread(str(img_file))
        h, w = image.shape[:2]
        
        # 读取标签
        label_file = Path(label_dir) / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
            
        # 绘制边界框
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # 转换为像素坐标
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.putText(image, class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存可视化结果
        output_file = vis_dir / f"vis_{img_file.name}"
        cv2.imwrite(str(output_file), image)
    
    print(f"   检测结果可视化保存到: {vis_dir}")

def analyze_class_distribution(label_dir, class_names):
    """分析类别分布"""
    print("\n📈 分析类别分布...")
    
    # 统计每个类别的实例数
    class_counts = {name: 0 for name in class_names}
    total_boxes = 0
    
    # 遍历所有标签文件
    label_files = list(Path(label_dir).glob("*.txt"))
    for label_file in label_files:
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
    print("训练结果可视化分析")
    print("=" * 50)
    
    # 类别名称
    class_names = [
        'pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # 分析训练日志
    log_dir = "runs/train_enhanced/20250902_203931"
    if Path(log_dir).exists():
        analyze_training_logs(log_dir)
    else:
        print("⚠️ 未找到训练日志目录")
    
    # 分析类别分布
    label_dir = "data/visdrone_yolo/labels/train"
    if Path(label_dir).exists():
        analyze_class_distribution(label_dir, class_names)
    
    # 可视化检测结果
    image_dir = "data/visdrone_yolo/images/train"
    if Path(image_dir).exists():
        visualize_detection_results(image_dir, label_dir, class_names)
    
    print("\n✅ 可视化分析完成!")

if __name__ == "__main__":
    main()