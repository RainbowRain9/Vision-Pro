#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 集成演示脚本
展示如何在 YOLOvision Pro 中集成和使用 Supervision
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import supervision as sv
    from ultralytics import YOLO
    from scripts.modules.supervision_wrapper import SupervisionWrapper, SupervisionAnalyzer
    SUPERVISION_AVAILABLE = True
    print("✅ Supervision 集成模块可用")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请安装: pip install supervision")
    SUPERVISION_AVAILABLE = False


def demo_supervision_wrapper():
    """演示 Supervision 包装器功能"""
    print("\n🎯 Supervision 包装器演示")
    print("=" * 50)
    
    if not SUPERVISION_AVAILABLE:
        print("❌ Supervision 不可用")
        return
    
    # VisDrone 类别
    visdrone_classes = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # 初始化包装器
    wrapper = SupervisionWrapper(class_names=visdrone_classes)
    print("✅ SupervisionWrapper 初始化完成")
    
    # 创建模拟检测结果
    mock_detections = sv.Detections(
        xyxy=np.array([
            [100, 100, 200, 200],  # pedestrian
            [300, 150, 450, 300],  # car
            [50, 250, 150, 350],   # people
        ]),
        confidence=np.array([0.95, 0.87, 0.72]),
        class_id=np.array([0, 3, 1])
    )
    
    # 创建示例图像
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    image.fill(80)  # 深灰色背景
    
    # 模拟 ultralytics 结果对象
    class MockResult:
        def __init__(self, detections):
            self.boxes = MockBoxes(detections)
    
    class MockBoxes:
        def __init__(self, detections):
            self.xyxy = MockTensor(detections.xyxy)
            self.conf = MockTensor(detections.confidence)
            self.cls = MockTensor(detections.class_id)
    
    class MockTensor:
        def __init__(self, data):
            self.data = data
        def cpu(self):
            return self
        def numpy(self):
            return self.data
    
    mock_result = MockResult(mock_detections)
    
    # 使用包装器处理
    processed = wrapper.process_ultralytics_results(mock_result, image)
    
    # 显示结果
    print(f"📊 处理结果:")
    print(f"   检测数量: {processed['detection_count']}")
    
    statistics = processed['statistics']
    print(f"   总检测数: {statistics['total_detections']}")
    print(f"   类别分布: {statistics['class_distribution']}")
    
    if statistics['confidence_stats']:
        conf_stats = statistics['confidence_stats']
        print(f"   平均置信度: {conf_stats['mean']:.3f}")
    
    # 保存结果
    output_dir = project_root / "outputs" / "supervision_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_image = processed['annotated_image']
    output_path = output_dir / "wrapper_demo.jpg"
    cv2.imwrite(str(output_path), enhanced_image)
    print(f"💾 增强图像已保存: {output_path}")
    
    # 生成摘要
    summary = wrapper.generate_detection_summary(statistics)
    print(f"\n📄 检测摘要:\n{summary}")


def demo_real_model_integration():
    """演示真实模型集成"""
    print("\n🤖 真实模型集成演示")
    print("=" * 50)
    
    if not SUPERVISION_AVAILABLE:
        print("❌ Supervision 不可用")
        return
    
    try:
        # 加载模型
        model_path = project_root / "models" / "yolov8s-drone.pt"
        if not model_path.exists():
            print("使用默认 YOLOv8s 模型")
            model = YOLO("yolov8s.pt")
        else:
            print(f"加载 Drone-YOLO 模型: {model_path}")
            model = YOLO(str(model_path))
        
        # 初始化 Supervision 组件
        visdrone_classes = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        
        wrapper = SupervisionWrapper(class_names=visdrone_classes)
        analyzer = SupervisionAnalyzer()
        
        # 创建测试图像（如果没有真实图像）
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 进行检测
        print("🔍 进行目标检测...")
        results = model.predict(test_image, conf=0.25, iou=0.45)
        
        # Supervision 处理
        processed = wrapper.process_ultralytics_results(results[0], test_image)
        
        # 添加到分析器
        analyzer.add_detection_result(processed)
        
        # 显示结果
        print(f"🎯 检测完成:")
        print(f"   检测到 {processed['detection_count']} 个目标")
        
        statistics = processed['statistics']
        if statistics['class_distribution']:
            print("   类别分布:")
            for class_name, count in statistics['class_distribution'].items():
                print(f"     {class_name}: {count}")
        
        # 保存结果
        output_dir = project_root / "outputs" / "supervision_demo"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存增强图像
        enhanced_image = processed['annotated_image']
        enhanced_path = output_dir / "real_model_demo.jpg"
        cv2.imwrite(str(enhanced_path), enhanced_image)
        print(f"💾 增强检测结果: {enhanced_path}")
        
        # 创建对比图
        comparison = wrapper.create_comparison_view(test_image, enhanced_image)
        comparison_path = output_dir / "comparison_demo.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"💾 对比图: {comparison_path}")
        
        # 生成性能报告
        report = analyzer.generate_performance_report()
        print(f"\n📈 性能报告:")
        print(f"   处理图片数: {report['total_processed']}")
        print(f"   总检测数: {report['total_detections']}")
        
    except Exception as e:
        print(f"❌ 真实模型演示失败: {e}")
        import traceback
        traceback.print_exc()


def demo_ui_integration():
    """演示 UI 集成效果"""
    print("\n🖥️ UI 集成演示")
    print("=" * 50)
    
    print("📋 集成步骤说明:")
    print("1. 在 main.py 中添加了 Supervision 初始化")
    print("2. 修改了 detect_image 方法支持 Supervision")
    print("3. 添加了增强的可视化和统计功能")
    print("4. 保持了向后兼容性")
    
    print("\n🎨 新增功能:")
    print("- 增强的边界框和标签标注")
    print("- 实时统计信息显示")
    print("- 检测结果分析和摘要")
    print("- 对比视图生成")
    print("- 性能监控和报告")
    
    print("\n🚀 使用方法:")
    print("1. 确保已安装 supervision: pip install supervision")
    print("2. 运行 main.py 启动 UI")
    print("3. 加载模型后进行图像检测")
    print("4. 系统会自动使用 Supervision 增强功能")


def demo_advanced_features():
    """演示高级功能"""
    print("\n🎨 高级功能演示")
    print("=" * 50)
    
    if not SUPERVISION_AVAILABLE:
        print("❌ Supervision 不可用")
        return
    
    # 演示不同的标注器
    print("🎯 标注器演示:")
    
    # 创建示例数据
    detections = sv.Detections(
        xyxy=np.array([[50, 50, 150, 150], [200, 100, 350, 250]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 3])
    )
    
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image.fill(60)
    
    # 不同标注器
    annotators = {
        'box': sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX),
        'corner': sv.CornerAnnotator(thickness=4),
        'circle': sv.CircleAnnotator(thickness=2),
        'dot': sv.DotAnnotator(radius=8),
    }
    
    output_dir = project_root / "outputs" / "supervision_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, annotator in annotators.items():
        annotated = annotator.annotate(image.copy(), detections)
        
        output_path = output_dir / f"annotator_{name}.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"   {name.capitalize()} 标注器: {output_path}")
    
    print("✅ 高级功能演示完成")


def main():
    """主函数"""
    print("🎉 YOLOvision Pro - Supervision 集成演示")
    print("=" * 60)
    
    if not SUPERVISION_AVAILABLE:
        print("❌ Supervision 不可用，请先安装:")
        print("   pip install supervision")
        return
    
    try:
        # 包装器演示
        demo_supervision_wrapper()
        
        # 真实模型集成演示
        demo_real_model_integration()
        
        # UI 集成说明
        demo_ui_integration()
        
        # 高级功能演示
        demo_advanced_features()
        
        print("\n🎊 所有演示完成!")
        print("📁 结果保存在: outputs/supervision_demo/")
        print("\n💡 下一步:")
        print("1. 运行 python main.py 体验 UI 集成效果")
        print("2. 查看 scripts/modules/supervision_wrapper.py 了解实现细节")
        print("3. 参考 docs/Supervision集成可行性方案.md 了解完整方案")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
