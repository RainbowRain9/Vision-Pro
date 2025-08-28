#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小目标检测演示脚本
展示如何使用 InferenceSlicer 进行小目标检测
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 添加 ultralytics 路径
ultralytics_path = project_root / "ultralytics"
if ultralytics_path.exists():
    sys.path.insert(0, str(ultralytics_path))

from ultralytics import YOLO
from scripts.modules.supervision_wrapper import SupervisionWrapper


def load_model(model_path: str = None):
    """加载 YOLO 模型"""
    if model_path is None:
        # 使用项目中的模型
        models_dir = project_root / "models"
        model_files = list(models_dir.glob("*.pt"))
        if model_files:
            model_path = str(model_files[0])
        else:
            model_path = "yolov8s.pt"  # 默认模型
    
    print(f"🔄 加载模型: {model_path}")
    model = YOLO(model_path)
    print(f"✅ 模型加载成功")
    return model


def demo_basic_detection(image_path: str, model):
    """演示基础检测"""
    print("\n" + "="*50)
    print("📸 基础检测演示")
    print("="*50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📐 图像尺寸: {image_rgb.shape}")
    
    # 基础检测
    start_time = time.time()
    results = model.predict(image_rgb, conf=0.25, iou=0.45, verbose=False)
    processing_time = time.time() - start_time
    
    # 显示结果
    result_img = results[0].plot()
    detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
    
    print(f"🎯 检测结果: {detection_count} 个目标")
    print(f"⏱️  处理时间: {processing_time:.2f}s")
    
    return result_img, detection_count, processing_time


def demo_small_object_detection(image_path: str, model):
    """演示小目标检测"""
    print("\n" + "="*50)
    print("🔍 小目标检测演示 (InferenceSlicer)")
    print("="*50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 初始化 Supervision 包装器
    class_names = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    wrapper = SupervisionWrapper(class_names=class_names)
    
    # 小目标检测
    print("🔄 执行小目标检测...")
    result = wrapper.detect_small_objects(
        image_rgb, model, 
        conf=0.25, iou=0.45,
        slice_wh=(640, 640),
        overlap_wh=(128, 128)
    )
    
    if 'error' in result:
        print(f"❌ 检测失败: {result['error']}")
        return None
    
    # 显示结果
    detection_count = result['detection_count']
    processing_time = result['statistics'].get('processing_time', 0)
    slice_config = result['statistics'].get('slice_config', {})
    total_slices = slice_config.get('total_slices', 0)
    
    print(f"🎯 检测结果: {detection_count} 个目标")
    print(f"📊 处理切片: {total_slices} 个")
    print(f"⏱️  处理时间: {processing_time:.2f}s")
    
    # 显示类别分布
    class_dist = result['statistics'].get('class_distribution', {})
    if class_dist:
        print("📈 类别分布:")
        for class_name, count in class_dist.items():
            print(f"   {class_name}: {count}")
    
    return result['annotated_image'], detection_count, processing_time


def demo_multi_scale_detection(image_path: str, model):
    """演示多尺度检测"""
    print("\n" + "="*50)
    print("🔄 多尺度检测演示")
    print("="*50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 初始化 Supervision 包装器
    class_names = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    wrapper = SupervisionWrapper(class_names=class_names)
    
    # 多尺度检测
    print("🔄 执行多尺度检测...")
    result = wrapper.detect_with_multiple_scales(
        image_rgb, model, conf=0.25, iou=0.45
    )
    
    if 'error' in result:
        print(f"❌ 检测失败: {result['error']}")
        return None
    
    # 显示结果
    detection_count = result['detection_count']
    scale_results = result['statistics'].get('scale_results', {})
    
    print(f"🎯 最终检测结果: {detection_count} 个目标")
    print("📊 各尺度结果:")
    total_time = 0
    for scale_name, scale_info in scale_results.items():
        count = scale_info['detection_count']
        time_cost = scale_info['processing_time']
        config = scale_info['config']
        total_time += time_cost
        print(f"   {scale_name}: {count} 个目标, {time_cost:.2f}s, 切片{config['slice_wh']}")
    
    print(f"⏱️  总处理时间: {total_time:.2f}s")
    
    return result['annotated_image'], detection_count, total_time


def save_results(results: dict, output_dir: Path):
    """保存检测结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name, (result_img, count, time_cost) in results.items():
        if result_img is not None:
            # 保存图像
            output_path = output_dir / f"{method_name}_result.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"💾 保存结果: {output_path}")


def main():
    """主函数"""
    print("🚀 小目标检测演示程序")
    print("="*60)
    
    # 检查测试图像
    test_images_dir = project_root / "assets" / "images"
    if not test_images_dir.exists():
        test_images_dir = project_root / "data" / "raw_images"
    
    if not test_images_dir.exists():
        print("❌ 未找到测试图像目录")
        print("请将测试图像放在以下目录之一:")
        print(f"  - {project_root / 'assets' / 'images'}")
        print(f"  - {project_root / 'data' / 'raw_images'}")
        return
    
    # 查找图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(test_images_dir.glob(ext))
    
    if not image_files:
        print(f"❌ 在 {test_images_dir} 中未找到图像文件")
        return
    
    # 选择第一个图像进行演示
    test_image = str(image_files[0])
    print(f"📸 使用测试图像: {test_image}")
    
    # 加载模型
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 执行各种检测方法
    results = {}
    
    # 1. 基础检测
    basic_result = demo_basic_detection(test_image, model)
    if basic_result:
        results['basic'] = basic_result
    
    # 2. 小目标检测
    small_obj_result = demo_small_object_detection(test_image, model)
    if small_obj_result:
        results['small_object'] = small_obj_result
    
    # 3. 多尺度检测
    multi_scale_result = demo_multi_scale_detection(test_image, model)
    if multi_scale_result:
        results['multi_scale'] = multi_scale_result
    
    # 保存结果
    output_dir = project_root / "results" / "small_object_demo"
    save_results(results, output_dir)
    
    # 性能对比
    print("\n" + "="*50)
    print("📊 性能对比总结")
    print("="*50)
    
    for method_name, (_, count, time_cost) in results.items():
        method_display = {
            'basic': '基础检测',
            'small_object': '小目标检测',
            'multi_scale': '多尺度检测'
        }.get(method_name, method_name)
        
        print(f"{method_display:12s}: {count:3d} 个目标, {time_cost:6.2f}s")
    
    print("\n✅ 演示完成！")
    print(f"📁 结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
