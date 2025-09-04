#!/usr/bin/env python3
"""
Supervision功能演示脚本
展示目标跟踪、计数和可视化功能
"""

import cv2
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def demo_object_tracking():
    """目标跟踪演示"""
    print("🎯 目标跟踪演示")
    
    # 创建视频信息
    video_info = sv.VideoInfo.from_video_path("test_video.mp4") if Path("test_video.mp4").exists() else None
    
    if video_info is None:
        print("   ⚠️ 未找到测试视频，使用图像演示")
        # 使用图像演示
        if not Path("sample_image.jpg").exists():
            # 创建示例图像
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite("sample_image.jpg", img)
        
        # 加载模型
        model = YOLO("yolov8s.pt")
        
        # 读取图像
        image = cv2.imread("sample_image.jpg")
        
        # 检测
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 初始化跟踪器
        byte_tracker = sv.ByteTrack()
        
        # 跟踪
        tracked_detections = byte_tracker.update_with_detections(detections)
        
        print(f"   检测到 {len(detections)} 个目标")
        print(f"   跟踪到 {len(tracked_detections)} 个目标")
        
        # 可视化
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), 
            detections=tracked_detections
        )
        
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"#{tracker_id}"
            for tracker_id in tracked_detections.tracker_id
        ]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=tracked_detections, 
            labels=labels
        )
        
        cv2.imwrite("tracked_image.jpg", annotated_frame)
        print("   跟踪结果保存为 tracked_image.jpg")
    else:
        print("   使用视频进行跟踪演示")
        # 视频跟踪实现...

def demo_object_counting():
    """目标计数演示"""
    print("\n🔢 目标计数演示")
    
    # 创建多边形区域用于计数
    polygon = np.array([
        [0, 0],
        [640, 0],
        [640, 480],
        [0, 480]
    ])
    
    # 创建区域
    polygon_zone = sv.PolygonZone(
        polygon=polygon
    )
    
    # 创建区域注释器
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
    
    # 模拟检测结果
    # 在实际应用中，这些将来自YOLO模型
    detections = sv.Detections(
        xyxy=np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400],
            [500, 100, 600, 200]
        ]),
        class_id=np.array([0, 1, 2]),
        confidence=np.array([0.9, 0.8, 0.7])
    )
    
    # 更新区域触发
    mask = polygon_zone.trigger(detections=detections)
    
    print(f"   区域内目标数: {polygon_zone.current_count}")
    print(f"   总触发数: {sum(mask)}")
    
    # 可视化
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    annotated_frame = zone_annotator.annotate(scene=sample_image)
    
    cv2.imwrite("counted_image.jpg", annotated_frame)
    print("   计数结果保存为 counted_image.jpg")

def demo_visualization():
    """可视化演示"""
    print("\n🎨 可视化演示")
    
    # 创建示例检测结果
    detections = sv.Detections(
        xyxy=np.array([
            [50, 50, 150, 150],
            [200, 100, 300, 200],
            [400, 150, 500, 250]
        ]),
        class_id=np.array([0, 1, 2]),
        confidence=np.array([0.95, 0.85, 0.75])
    )
    
    # 创建不同类型的注释器
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.DEFAULT,
        thickness=2
    )
    
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.DEFAULT
    )
    
    # 类别名称
    class_names = ['行人', '车辆', '自行车']
    
    # 创建标签
    labels = [
        f"{class_names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    # 应用注释
    image = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
    annotated_image = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections, 
        labels=labels
    )
    
    cv2.imwrite("visualized_image.jpg", annotated_image)
    print("   可视化结果保存为 visualized_image.jpg")

def main():
    """主函数"""
    print("=" * 50)
    print("Supervision功能演示")
    print("=" * 50)
    
    # 目标跟踪
    demo_object_tracking()
    
    # 目标计数
    demo_object_counting()
    
    # 可视化
    demo_visualization()
    
    print("\n✅ 所有演示完成!")

if __name__ == "__main__":
    main()