#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 标注器测试脚本
测试各种标注器的功能和兼容性
"""

import unittest
import cv2
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts/modules"))

try:
    import supervision as sv
    from supervision_annotators import AnnotatorManager, AnnotatorType, AnnotatorPresets
    from supervision_wrapper import SupervisionWrapper
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


class TestSupervisionAnnotators(unittest.TestCase):
    """标注器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 设置日志
        logging.basicConfig(level=logging.WARNING)  # 减少测试时的日志输出
        
        # 创建测试图像
        cls.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 创建测试检测结果
        cls.test_detections = sv.Detections(
            xyxy=np.array([
                [100, 100, 200, 200],
                [300, 150, 400, 250],
                [50, 300, 150, 400]
            ], dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            class_id=np.array([0, 1, 2], dtype=int)
        )
        
        cls.test_labels = ["person: 0.90", "car: 0.80", "bicycle: 0.70"]
        
        # 创建临时配置文件
        cls.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        cls.temp_config.write("""
annotators:
  box:
    enabled: true
    thickness: 2
  label:
    enabled: true
    text_scale: 0.5
  mask:
    enabled: false
  polygon:
    enabled: true
    thickness: 2
  heatmap:
    enabled: false
  blur:
    enabled: true
    kernel_size: 15
  pixelate:
    enabled: true
    pixel_size: 20
""")
        cls.temp_config.close()
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        # 删除临时配置文件
        os.unlink(cls.temp_config.name)
    
    def setUp(self):
        """每个测试方法的初始化"""
        self.annotator_manager = AnnotatorManager(self.temp_config.name)
    
    def test_annotator_manager_initialization(self):
        """测试标注器管理器初始化"""
        self.assertIsInstance(self.annotator_manager, AnnotatorManager)
        self.assertGreater(len(self.annotator_manager.annotators), 0)
        self.assertIn(AnnotatorType.BOX, self.annotator_manager.annotators)
        self.assertIn(AnnotatorType.LABEL, self.annotator_manager.annotators)
    
    def test_annotator_types(self):
        """测试标注器类型枚举"""
        expected_types = ['box', 'label', 'mask', 'polygon', 'heatmap', 'blur', 'pixelate']
        actual_types = [t.value for t in AnnotatorType]
        
        for expected in expected_types:
            self.assertIn(expected, actual_types)
    
    def test_enable_disable_annotators(self):
        """测试启用/禁用标注器"""
        # 测试启用
        self.annotator_manager.enable_annotator(AnnotatorType.MASK)
        self.assertIn(AnnotatorType.MASK, self.annotator_manager.enabled_annotators)
        
        # 测试禁用
        self.annotator_manager.disable_annotator(AnnotatorType.BOX)
        self.assertNotIn(AnnotatorType.BOX, self.annotator_manager.enabled_annotators)
        
        # 测试切换
        initial_state = AnnotatorType.LABEL in self.annotator_manager.enabled_annotators
        self.annotator_manager.toggle_annotator(AnnotatorType.LABEL)
        final_state = AnnotatorType.LABEL in self.annotator_manager.enabled_annotators
        self.assertNotEqual(initial_state, final_state)
    
    def test_preset_configurations(self):
        """测试预设配置"""
        # 测试基础预设
        self.annotator_manager.set_preset('basic')
        enabled = self.annotator_manager.get_enabled_annotators()
        self.assertIn('box', enabled)
        self.assertIn('label', enabled)
        
        # 测试隐私预设
        self.annotator_manager.set_preset('privacy')
        enabled = self.annotator_manager.get_enabled_annotators()
        self.assertIn('blur', enabled)
        
        # 测试无效预设
        self.annotator_manager.set_preset('invalid_preset')
        # 应该不会崩溃，只是记录警告
    
    def test_image_annotation(self):
        """测试图像标注功能"""
        # 测试基本标注
        annotated = self.annotator_manager.annotate_image(
            self.test_image, self.test_detections, self.test_labels
        )
        
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_image.shape)
        self.assertEqual(annotated.dtype, self.test_image.dtype)
    
    def test_empty_detections(self):
        """测试空检测结果"""
        empty_detections = sv.Detections.empty()
        
        annotated = self.annotator_manager.annotate_image(
            self.test_image, empty_detections, []
        )
        
        # 应该返回原图的副本
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_image.shape)
    
    def test_comparison_view(self):
        """测试对比视图"""
        comparison = self.annotator_manager.create_comparison_view(
            self.test_image, self.test_detections, self.test_labels
        )
        
        self.assertIsInstance(comparison, np.ndarray)
        # 对比视图应该是水平拼接，宽度是原图的两倍
        expected_shape = (self.test_image.shape[0], self.test_image.shape[1] * 2, self.test_image.shape[2])
        self.assertEqual(comparison.shape, expected_shape)
    
    def test_batch_annotation(self):
        """测试批量标注"""
        images = [self.test_image, self.test_image.copy()]
        detections_list = [self.test_detections, self.test_detections]
        labels_list = [self.test_labels, self.test_labels]
        
        annotated_images = self.annotator_manager.batch_annotate(
            images, detections_list, labels_list
        )
        
        self.assertEqual(len(annotated_images), 2)
        for annotated in annotated_images:
            self.assertIsInstance(annotated, np.ndarray)
            self.assertEqual(annotated.shape, self.test_image.shape)
    
    def test_annotator_info(self):
        """测试标注器信息获取"""
        info = self.annotator_manager.get_annotator_info()
        
        self.assertIn('available_annotators', info)
        self.assertIn('enabled_annotators', info)
        self.assertIn('total_annotators', info)
        self.assertIn('enabled_count', info)
        self.assertIn('presets', info)
        
        self.assertIsInstance(info['available_annotators'], list)
        self.assertIsInstance(info['enabled_annotators'], list)
        self.assertIsInstance(info['total_annotators'], int)
        self.assertIsInstance(info['enabled_count'], int)
    
    def test_performance_stats(self):
        """测试性能统计"""
        stats = self.annotator_manager.get_performance_stats()
        
        self.assertIn('enabled_annotators_count', stats)
        self.assertIn('memory_usage_estimate', stats)
        self.assertIn('recommended_max_fps', stats)
        
        self.assertIsInstance(stats['enabled_annotators_count'], int)
        self.assertIsInstance(stats['memory_usage_estimate'], str)
        self.assertIsInstance(stats['recommended_max_fps'], int)
    
    def test_heatmap_functionality(self):
        """测试热力图功能"""
        # 启用热力图
        self.annotator_manager.enable_annotator(AnnotatorType.HEATMAP)
        
        # 添加多帧数据
        for _ in range(10):
            self.annotator_manager.annotate_image(
                self.test_image, self.test_detections, self.test_labels
            )
        
        # 检查历史数据
        self.assertGreater(len(self.annotator_manager.heatmap_history), 0)
        
        # 清除历史数据
        self.annotator_manager.clear_heatmap_history()
        self.assertEqual(len(self.annotator_manager.heatmap_history), 0)
    
    def test_config_update(self):
        """测试配置更新"""
        # 更新边界框厚度
        self.annotator_manager.update_annotator_config(
            AnnotatorType.BOX, thickness=5
        )
        
        # 验证配置已更新
        box_config = self.annotator_manager.configs[AnnotatorType.BOX]
        self.assertEqual(box_config.thickness, 5)


class TestSupervisionWrapper(unittest.TestCase):
    """Supervision包装器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.class_names = ["person", "car", "bicycle"]
        self.wrapper = SupervisionWrapper(
            class_names=self.class_names,
            annotator_config_path=TestSupervisionAnnotators.temp_config.name
        )
        
        # 测试数据
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=int)
        )
    
    def test_wrapper_initialization(self):
        """测试包装器初始化"""
        self.assertIsInstance(self.wrapper, SupervisionWrapper)
        self.assertEqual(self.wrapper.class_names, self.class_names)
        self.assertIsNotNone(self.wrapper.annotator_manager)
    
    def test_annotator_management_methods(self):
        """测试标注器管理方法"""
        # 测试启用标注器
        self.wrapper.enable_annotator('mask')
        enabled = self.wrapper.get_enabled_annotators()
        self.assertIn('mask', enabled)
        
        # 测试禁用标注器
        self.wrapper.disable_annotator('mask')
        enabled = self.wrapper.get_enabled_annotators()
        self.assertNotIn('mask', enabled)
        
        # 测试切换标注器
        self.wrapper.toggle_annotator('polygon')
        # 不会抛出异常即可
    
    def test_preset_management(self):
        """测试预设管理"""
        self.wrapper.set_annotator_preset('privacy')
        enabled = self.wrapper.get_enabled_annotators()
        self.assertIn('blur', enabled)
        
        # 测试获取可用预设
        presets = self.wrapper.get_available_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)
    
    def test_annotator_info(self):
        """测试标注器信息"""
        info = self.wrapper.get_annotator_info()
        self.assertIn('available_annotators', info)
        self.assertIn('enabled_annotators', info)
    
    def test_performance_stats(self):
        """测试性能统计"""
        stats = self.wrapper.get_performance_stats()
        self.assertIn('small_object_detection', stats)
        self.assertIn('annotators', stats)


class TestAnnotatorPresets(unittest.TestCase):
    """标注器预设测试类"""
    
    def test_preset_structure(self):
        """测试预设结构"""
        presets = AnnotatorPresets.get_all_presets()
        
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)
        
        for preset in presets:
            self.assertIn('name', preset)
            self.assertIn('description', preset)
            self.assertIn('annotators', preset)
            self.assertIsInstance(preset['annotators'], list)
    
    def test_individual_presets(self):
        """测试单个预设"""
        privacy_preset = AnnotatorPresets.get_privacy_preset()
        self.assertEqual(privacy_preset['name'], 'privacy')
        self.assertIn(AnnotatorType.BLUR, privacy_preset['annotators'])
        
        analysis_preset = AnnotatorPresets.get_analysis_preset()
        self.assertEqual(analysis_preset['name'], 'analysis')
        self.assertIn(AnnotatorType.HEATMAP, analysis_preset['annotators'])


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestSupervisionAnnotators,
        TestSupervisionWrapper,
        TestAnnotatorPresets
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("运行 Supervision 标注器测试...")
    print("="*50)
    
    success = run_tests()
    
    print("="*50)
    if success:
        print("✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败！")
        sys.exit(1)
