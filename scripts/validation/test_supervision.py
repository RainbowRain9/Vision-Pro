
# -*- coding: utf-8 -*-
"""
YOLOvision Pro - Supervision 集成测试脚本
验证 Supervision.roboflow.com 的安装和基本功能
"""

import sys
import os
from pathlib import Path

def test_supervision_installation():
    """测试 Supervision 安装"""
    print("=== Supervision 安装测试 ===")
    
    try:
        import supervision as sv
        print(f"✓ Supervision 版本: {sv.__version__}")
        
        # 测试核心组件
        components = [
            'BoxAnnotator',
            'LabelAnnotator', 
            'HeatMapAnnotator',
            'DetectionMetrics',
            'Detections'
        ]
        
        for component in components:
            if hasattr(sv, component):
                print(f"✓ {component}: 可用")
            else:
                print(f"✗ {component}: 不可用")
        
        return True
        
    except ImportError as e:
        print(f"✗ Supervision 未安装: {e}")
        print("安装命令: pip install supervision")
        return False

def test_ultralytics_compatibility():
    """测试与 Ultralytics 的兼容性"""
    print("\n=== Ultralytics 兼容性测试 ===")
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics 可用")
        
        # 测试模型加载
        try:
            model = YOLO('yolov8s.pt')
            print("✓ YOLO 模型加载成功")
            return True
        except Exception as e:
            print(f"✗ YOLO 模型加载失败: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Ultralytics 不可用: {e}")
        return False

def test_dependencies():
    """测试依赖项"""
    print("\n=== 依赖项测试 ===")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('pyyaml', 'yaml'),
        ('pillow', 'PIL'),
        ('scipy', 'scipy')
    ]
    
    all_ok = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}: 可用")
        except ImportError:
            print(f"✗ {name}: 不可用")
            all_ok = False
    
    return all_ok

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 基本功能测试 ===")
    
    try:
        import supervision as sv
        import numpy as np
        import cv2
        
        # 创建测试数据
        test_detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200], [150, 150, 250, 250]]),
            confidence=np.array([0.8, 0.9]),
            class_id=np.array([0, 1])
        )
        
        # 测试标注器
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # 创建测试图像
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # 测试标注功能
        annotated = box_annotator.annotate(scene=test_image, detections=test_detections)
        annotated = label_annotator.annotate(scene=annotated, detections=test_detections)
        
        print("✓ 基本标注功能正常")
        
        # 测试指标计算
        metrics = sv.DetectionMetrics()
        print("✓ 检测指标模块可用")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_visdrone_compatibility():
    """测试 VisDrone 兼容性"""
    print("\n=== VisDrone 兼容性测试 ===")
    
    # 检查 VisDrone 数据集
    visdrone_paths = [
        "data/visdrone_yolo/data.yaml",
        "data/visdrone_yolo/images/train",
        "data/visdrone_yolo/labels/train"
    ]
    
    all_exist = True
    for path in visdrone_paths:
        if Path(path).exists():
            print(f"✓ {path}: 存在")
        else:
            print(f"✗ {path}: 不存在")
            all_exist = False
    
    if all_exist:
        print("✓ VisDrone 数据集可用")
        return True
    else:
        print("⚠ VisDrone 数据集部分缺失，可能需要先处理数据")
        return False

def generate_test_report():
    """生成测试报告"""
    print("\n=== 测试报告 ===")
    
    tests = [
        ("Supervision 安装", test_supervision_installation),
        ("Ultralytics 兼容性", test_ultralytics_compatibility),
        ("依赖项", test_dependencies),
        ("基本功能", test_basic_functionality),
        ("VisDrone 兼容性", test_visdrone_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    print("\n=== 测试总结 ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    return passed == total

def main():
    """主函数"""
    print("YOLOvision Pro - Supervision 集成测试")
    print("=" * 50)
    
    # 运行所有测试
    all_passed = generate_test_report()
    
    if all_passed:
        print("\n🎉 所有测试通过！Supervision 集成准备就绪。")
        print("\n下一步:")
        print("1. 运行演示脚本: python scripts/demo/supervision_demo.py")
        print("2. 查看可行性方案: docs/Supervision集成可行性方案.md")
        print("3. 开始集成开发")
    else:
        print("\n⚠ 部分测试失败，请先解决依赖问题。")
        print("\n建议:")
        print("1. 安装缺失的依赖")
        print("2. 检查 VisDrone 数据集")
        print("3. 重新运行测试")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())