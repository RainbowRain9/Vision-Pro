#!/usr/bin/env python3
"""
训练完成后的自动验证脚本
使用supervision进行模型验证和评估
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

def validate_model(model_path, data_yaml):
    """验证训练好的模型"""
    print("🔍 验证训练好的模型...")
    
    # 加载模型
    if not Path(model_path).exists():
        print(f"   ⚠️ 模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"   🤖 加载模型: {model_path}")
    
    # 加载数据集信息
    import yaml
    with open(data_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    class_names = dataset_info['names']
    print(f"   📊 数据集类别: {len(class_names)}")
    
    # 验证集路径
    val_images = Path(data_yaml).parent / "images" / "val"
    val_labels = Path(data_yaml).parent / "labels" / "val"
    
    if not val_images.exists():
        print("   ⚠️ 验证集图像目录不存在")
        return
    
    # 初始化指标
    total_gt = 0
    total_tp = 0
    total_images = 0
    
    # 遍历验证集（前20张图像）
    val_files = list(val_images.glob("*.jpg"))[:20]
    
    print(f"   📷 处理验证图像: {len(val_files)} 张")
    
    for img_file in val_files:
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            continue
            
        h, w = image.shape[:2]
        total_images += 1
        
        # YOLO推理
        results = model(image, verbose=False)[0]
        pred_dets = sv.Detections.from_ultralytics(results)
        
        # 读取真实标注
        label_file = val_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            # 读取YOLO格式标注
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if lines:
                # 解析标注
                class_ids = []
                xyxy_boxes = []
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # 转换为像素坐标
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    x2 = (x_center + width/2) * w
                    y2 = (y_center + height/2) * h
                    
                    class_ids.append(class_id)
                    xyxy_boxes.append([x1, y1, x2, y2])
                
                # 创建Detections对象
                gt_dets = sv.Detections(
                    xyxy=np.array(xyxy_boxes),
                    class_id=np.array(class_ids)
                )
                
                # 计算匹配
                if len(gt_dets) > 0 and len(pred_dets) > 0:
                    # 计算IoU矩阵
                    from supervision.metrics.detection import box_iou_batch
                    iou_matrix = box_iou_batch(gt_dets.xyxy, pred_dets.xyxy)
                    # 计算匹配数（IoU > 0.5）
                    matches = (iou_matrix > 0.5).sum()
                    total_tp += matches
                
                total_gt += len(gt_dets)
    
    # 计算准确率
    if total_gt > 0:
        accuracy = total_tp / total_gt if total_gt > 0 else 0
        print(f"\n📊 验证结果:")
        print(f"   处理图像数: {total_images}")
        print(f"   总真实框数: {total_gt}")
        print(f"   正确检测数: {total_tp}")
        print(f"   准确率 (IoU=0.5): {accuracy:.2%}")
    else:
        print("   ⚠️ 无标注数据用于验证")

def generate_predictions_visualization(model_path, data_yaml):
    """生成预测结果可视化"""
    print("\n🎨 生成预测结果可视化...")
    
    # 加载模型
    if not Path(model_path).exists():
        print(f"   ⚠️ 模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 加载数据集信息
    import yaml
    with open(data_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    class_names = dataset_info['names']
    
    # 验证集路径
    val_images = Path(data_yaml).parent / "images" / "val"
    
    if not val_images.exists():
        print("   ⚠️ 验证集图像目录不存在")
        return
    
    # 创建可视化目录
    vis_dir = Path("predictions_visualization")
    vis_dir.mkdir(exist_ok=True)
    
    # 选择几张图像进行可视化
    image_files = list(val_images.glob("*.jpg"))[:5]
    
    # 创建注释器
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    
    for img_file in image_files:
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # YOLO推理
        results = model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 注释图像
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        
        # 添加标签
        if len(detections) > 0:
            labels = [
                f"{class_names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated_image = label_annotator.annotate(
                scene=annotated_image, 
                detections=detections, 
                labels=labels
            )
        
        # 保存可视化结果
        output_file = vis_dir / f"pred_{img_file.name}"
        cv2.imwrite(str(output_file), annotated_image)
    
    print(f"   预测可视化保存到: {vis_dir}")

def main():
    """主函数"""
    print("=" * 50)
    print("模型验证和评估")
    print("=" * 50)
    
    # 模型路径和数据配置
    model_path = "runs/train_enhanced/20250902_203931/weights/best.pt"
    data_yaml = "data/visdrone_yolo/data.yaml"
    
    # 验证模型
    validate_model(model_path, data_yaml)
    
    # 生成预测可视化
    generate_predictions_visualization(model_path, data_yaml)
    
    print("\n✅ 验证完成!")

if __name__ == "__main__":
    main()