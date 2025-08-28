#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 兼容性测试脚本
测试 InferenceSlicer 的版本兼容性
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

def test_supervision_import():
    """测试 Supervision 导入"""
    print("🧪 测试 Supervision 导入")
    print("-" * 40)
    
    try:
        import supervision as sv
        print(f"✅ Supervision 导入成功")
        
        # 检查版本
        if hasattr(sv, '__version__'):
            print(f"   版本: {sv.__version__}")
        else:
            print("   版本: 未知")
        
        # 检查关键组件
        components = [
            'Detections',
            'InferenceSlicer', 
            'BoxAnnotator',
            'LabelAnnotator',
            'OverlapFilter'
        ]
        
        for component in components:
            if hasattr(sv, component):
                print(f"   ✅ {component} 可用")
            else:
                print(f"   ❌ {component} 不可用")
        
        return True
        
    except ImportError as e:
        print(f"❌ Supervision 导入失败: {e}")
        return False

def test_inference_slicer_api():
    """测试 InferenceSlicer API 兼容性"""
    print("\n🧪 测试 InferenceSlicer API")
    print("-" * 40)
    
    try:
        import supervision as sv
        
        # 创建模拟回调函数
        def mock_callback(image_slice: np.ndarray):
            # 返回空的检测结果
            return sv.Detections.empty()
        
        # 测试新版本 API (overlap_wh)
        try:
            slicer_new = sv.InferenceSlicer(
                callback=mock_callback,
                slice_wh=(640, 640),
                overlap_wh=(128, 128),
                iou_threshold=0.5
            )
            print("✅ 新版本 API (overlap_wh) 可用")
            new_api_works = True
        except Exception as e:
            print(f"❌ 新版本 API 失败: {e}")
            new_api_works = False
        
        # 测试旧版本 API (overlap_ratio_wh)
        try:
            slicer_old = sv.InferenceSlicer(
                callback=mock_callback,
                slice_wh=(640, 640),
                overlap_ratio_wh=(0.2, 0.2),
                iou_threshold=0.5
            )
            print("✅ 旧版本 API (overlap_ratio_wh) 可用")
            old_api_works = True
        except Exception as e:
            print(f"❌ 旧版本 API 失败: {e}")
            old_api_works = False
        
        if new_api_works:
            print("💡 建议使用新版本 API")
        elif old_api_works:
            print("💡 使用旧版本 API")
        else:
            print("❌ 两个版本的 API 都不可用")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ InferenceSlicer 测试失败: {e}")
        return False

def test_supervision_wrapper():
    """测试 SupervisionWrapper 兼容性"""
    print("\n🧪 测试 SupervisionWrapper 兼容性")
    print("-" * 40)
    
    try:
        from scripts.modules.supervision_wrapper import SupervisionWrapper
        
        # 初始化包装器
        wrapper = SupervisionWrapper(class_names=['test'])
        print("✅ SupervisionWrapper 初始化成功")
        
        # 测试配置方法
        wrapper.configure_small_object_detection(
            slice_wh=(640, 640),
            overlap_wh=(128, 128)
        )
        print("✅ 配置方法正常")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # 测试自适应配置
        config = wrapper.get_optimal_slice_config(test_image.shape[:2])
        print(f"✅ 自适应配置: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ SupervisionWrapper 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_detection():
    """测试模拟检测（不需要实际模型）"""
    print("\n🧪 测试模拟检测")
    print("-" * 40)
    
    try:
        import supervision as sv
        from scripts.modules.supervision_wrapper import SupervisionWrapper
        
        # 创建模拟模型类
        class MockModel:
            def predict(self, image, conf=0.25, iou=0.45, verbose=True):
                # 返回模拟的检测结果
                class MockResult:
                    def __init__(self):
                        # 创建一些模拟的检测框
                        self.boxes = MockBoxes()
                        self.names = {0: 'person', 1: 'car'}
                        self.masks = None  # 添加 masks 属性

                class MockBoxes:
                    def __init__(self):
                        # 模拟检测框数据 - 使用 MockTensor 来模拟 PyTorch 张量
                        self.xyxy = MockTensor([[100, 100, 200, 200], [300, 300, 400, 400]])
                        self.conf = MockTensor([0.8, 0.9])
                        self.cls = MockTensor([0, 1])

                class MockTensor:
                    def __init__(self, data):
                        self.data = np.array(data)

                    def cpu(self):
                        return self

                    def numpy(self):
                        return self.data

                    def __getitem__(self, key):
                        return self.data[key]

                    def __len__(self):
                        return len(self.data)

                    def item(self):
                        return self.data.item() if self.data.size == 1 else self.data

                    def tolist(self):
                        return self.data.tolist()

                return [MockResult()]
        
        # 初始化包装器和模拟模型
        wrapper = SupervisionWrapper(class_names=['person', 'car'])
        mock_model = MockModel()
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 测试小目标检测
        print("🔄 执行模拟小目标检测...")
        result = wrapper.detect_small_objects(
            test_image, mock_model,
            slice_wh=(320, 320),
            overlap_wh=(64, 64)
        )
        
        if 'error' not in result:
            print(f"✅ 模拟检测成功: {result['detection_count']} 个目标")
            print(f"   处理时间: {result['statistics'].get('processing_time', 0):.3f}s")
            return True
        else:
            print(f"❌ 模拟检测失败: {result['error']}")
            return False
        
    except Exception as e:
        print(f"❌ 模拟检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Supervision 兼容性测试")
    print("=" * 60)
    
    tests = [
        ("Supervision 导入", test_supervision_import),
        ("InferenceSlicer API", test_inference_slicer_api),
        ("SupervisionWrapper", test_supervision_wrapper),
        ("模拟检测", test_mock_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有兼容性测试通过！")
        print("💡 小目标检测功能已准备就绪")
    else:
        print("⚠️  部分测试失败，可能存在兼容性问题")
        print("💡 建议检查 Supervision 版本或重新安装")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
