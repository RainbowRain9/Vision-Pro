#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 功能包装器
为 YOLOvision Pro 提供增强的可视化和分析功能
支持小目标检测的 InferenceSlicer 功能
集成多种标注器管理功能
"""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from pathlib import Path
import time

# 导入标注器管理模块
try:
    from .supervision_annotators import AnnotatorManager, AnnotatorType, AnnotatorPresets
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append(str(Path(__file__).parent))
    from supervision_annotators import AnnotatorManager, AnnotatorType, AnnotatorPresets


class SupervisionWrapper:
    """Supervision 功能统一包装器，支持小目标检测和多种标注器"""

    def __init__(self, class_names: Optional[List[str]] = None,
                 annotator_config_path: Optional[str] = None):
        """
        初始化 Supervision 包装器

        Args:
            class_names: 类别名称列表
            annotator_config_path: 标注器配置文件路径
        """
        self.class_names = class_names or []

        # 初始化标注器管理器
        config_path = annotator_config_path or str(Path(__file__).parent.parent.parent / "assets/configs/annotator_config.yaml")
        try:
            self.annotator_manager = AnnotatorManager(config_path)
            logging.info("标注器管理器初始化成功")
        except Exception as e:
            logging.warning(f"标注器管理器初始化失败: {e}，使用基础标注器")
            self.annotator_manager = None

        # 保持向后兼容的基础标注器 (适配 Supervision 0.26.1+ API)
        try:
            # 尝试新版本 API
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(
                text_thickness=1,
                text_scale=0.5,
                text_padding=5
            )
        except TypeError:
            # 回退到更简单的初始化
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()

        # 初始化颜色调色板 (适配新版本 API)
        try:
            self.color_palette = sv.ColorPalette.default()
        except AttributeError:
            # 新版本可能使用不同的 API
            try:
                self.color_palette = sv.ColorPalette()
            except:
                self.color_palette = None

        # 性能指标 (使用自定义实现，因为 DetectionMetrics 在新版本中不可用)
        self.detection_metrics = {}

        # 小目标检测配置
        self.small_object_config = {
            'slice_wh': (640, 640),  # 切片尺寸
            'overlap_wh': (128, 128),  # 重叠尺寸
            'iou_threshold': 0.5,  # NMS IoU 阈值
            'overlap_filter': sv.OverlapFilter.NON_MAX_SUPPRESSION,
            'thread_workers': 1  # 线程数
        }

        logging.info("Supervision 包装器初始化完成（支持小目标检测和多种标注器）")
    
    def process_ultralytics_results(self, results, image: np.ndarray) -> Dict[str, Any]:
        """
        处理 Ultralytics 检测结果
        
        Args:
            results: Ultralytics 检测结果
            image: 原始图像
            
        Returns:
            包含增强可视化和统计信息的字典
        """
        try:
            # 转换为 Supervision Detections 格式
            detections = sv.Detections.from_ultralytics(results)
            
            # 生成标签
            labels = self._generate_labels(detections)
            
            # 创建增强可视化
            annotated_image = self._create_enhanced_visualization(
                image.copy(), detections, labels
            )
            
            # 计算统计信息
            statistics = self._calculate_statistics(detections)
            
            # 生成性能指标
            metrics = self._calculate_metrics(detections)
            
            return {
                'annotated_image': annotated_image,
                'detections': detections,
                'labels': labels,
                'statistics': statistics,
                'metrics': metrics,
                'detection_count': len(detections.xyxy)
            }
            
        except Exception as e:
            logging.error(f"处理检测结果时出错: {e}")
            return {
                'annotated_image': image,
                'detections': None,
                'labels': [],
                'statistics': {},
                'metrics': {},
                'detection_count': 0
            }
    
    def _generate_labels(self, detections: sv.Detections) -> List[str]:
        """生成检测标签"""
        labels = []
        
        for i in range(len(detections.xyxy)):
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            
            # 获取类别名称
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            # 格式化标签
            label = f"{class_name}: {confidence:.2f}"
            labels.append(label)
        
        return labels
    
    def _create_enhanced_visualization(self, image: np.ndarray,
                                     detections: sv.Detections,
                                     labels: List[str]) -> np.ndarray:
        """创建增强的可视化效果"""

        # 如果有标注器管理器，使用它进行标注
        if self.annotator_manager:
            return self.annotator_manager.annotate_image(image, detections, labels)

        # 否则使用基础标注器（向后兼容）
        # 添加边界框
        annotated_image = self.box_annotator.annotate(
            scene=image,
            detections=detections
        )

        # 添加标签
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        return annotated_image
    
    def _calculate_statistics(self, detections: sv.Detections) -> Dict[str, Any]:
        """计算检测统计信息"""
        if len(detections.xyxy) == 0:
            return {
                'total_detections': 0,
                'class_distribution': {},
                'confidence_stats': {},
                'bbox_stats': {}
            }
        
        # 类别分布
        class_distribution = {}
        if detections.class_id is not None:
            unique, counts = np.unique(detections.class_id, return_counts=True)
            for class_id, count in zip(unique, counts):
                class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f"Class_{int(class_id)}"
                class_distribution[class_name] = int(count)
        
        # 置信度统计
        confidence_stats = {}
        if detections.confidence is not None:
            confidence_stats = {
                'mean': float(np.mean(detections.confidence)),
                'std': float(np.std(detections.confidence)),
                'min': float(np.min(detections.confidence)),
                'max': float(np.max(detections.confidence))
            }
        
        # 边界框统计
        bbox_stats = {}
        if len(detections.xyxy) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            bbox_stats = {
                'mean_area': float(np.mean(areas)),
                'std_area': float(np.std(areas)),
                'min_area': float(np.min(areas)),
                'max_area': float(np.max(areas))
            }
        
        return {
            'total_detections': len(detections.xyxy),
            'class_distribution': class_distribution,
            'confidence_stats': confidence_stats,
            'bbox_stats': bbox_stats
        }
    
    def _calculate_metrics(self, detections: sv.Detections) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {
            'detection_count': len(detections.xyxy),
            'avg_confidence': 0.0,
            'detection_density': 0.0
        }
        
        if detections.confidence is not None and len(detections.confidence) > 0:
            metrics['avg_confidence'] = float(np.mean(detections.confidence))
        
        return metrics

    def detect_small_objects(self, image: np.ndarray, model,
                           conf: float = 0.25, iou: float = 0.45,
                           slice_wh: Optional[Tuple[int, int]] = None,
                           overlap_wh: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        使用 InferenceSlicer 进行小目标检测

        Args:
            image: 输入图像
            model: YOLO 模型
            conf: 置信度阈值
            iou: IoU 阈值
            slice_wh: 切片尺寸 (width, height)
            overlap_wh: 重叠尺寸 (width, height)

        Returns:
            包含检测结果和统计信息的字典
        """
        try:
            # 使用配置或默认值
            slice_wh = slice_wh or self.small_object_config['slice_wh']
            overlap_wh = overlap_wh or self.small_object_config['overlap_wh']

            # 定义回调函数
            def callback(image_slice: np.ndarray) -> sv.Detections:
                results = model.predict(image_slice, conf=conf, iou=iou, verbose=False)
                return sv.Detections.from_ultralytics(results[0])

            # 创建 InferenceSlicer (兼容不同版本 API)
            # 尝试检测支持的参数
            import inspect
            slicer_signature = inspect.signature(sv.InferenceSlicer.__init__)
            slicer_params = list(slicer_signature.parameters.keys())

            if 'overlap_wh' in slicer_params and 'overlap_ratio_wh' not in slicer_params:
                # 新版本 API (supervision >= 0.27.0) - 只支持 overlap_wh
                slicer = sv.InferenceSlicer(
                    callback=callback,
                    slice_wh=slice_wh,
                    overlap_wh=overlap_wh,
                    iou_threshold=self.small_object_config['iou_threshold'],
                    overlap_filter=self.small_object_config['overlap_filter'],
                    thread_workers=self.small_object_config['thread_workers']
                )
            elif 'overlap_ratio_wh' in slicer_params:
                # 旧版本 API (supervision < 0.27.0) - 使用 overlap_ratio_wh
                # 计算重叠比例
                overlap_ratio_w = overlap_wh[0] / slice_wh[0] if slice_wh[0] > 0 else 0.2
                overlap_ratio_h = overlap_wh[1] / slice_wh[1] if slice_wh[1] > 0 else 0.2

                slicer = sv.InferenceSlicer(
                    callback=callback,
                    slice_wh=slice_wh,
                    overlap_ratio_wh=(overlap_ratio_w, overlap_ratio_h),
                    iou_threshold=self.small_object_config['iou_threshold'],
                    overlap_filter=self.small_object_config['overlap_filter'],
                    thread_workers=self.small_object_config['thread_workers']
                )
            else:
                # 回退到最基本的参数
                slicer = sv.InferenceSlicer(
                    callback=callback,
                    slice_wh=slice_wh,
                    iou_threshold=self.small_object_config['iou_threshold']
                )

            # 记录开始时间
            start_time = time.time()

            # 执行切片检测
            detections = slicer(image)

            # 记录处理时间
            processing_time = time.time() - start_time

            # 生成标签
            labels = self._generate_labels(detections)

            # 创建增强可视化
            annotated_image = self._create_enhanced_visualization(
                image.copy(), detections, labels
            )

            # 计算统计信息
            statistics = self._calculate_statistics(detections)
            statistics['processing_time'] = processing_time
            statistics['slice_config'] = {
                'slice_wh': slice_wh,
                'overlap_wh': overlap_wh,
                'total_slices': self._estimate_slice_count(image.shape[:2], slice_wh, overlap_wh)
            }

            # 生成性能指标
            metrics = self._calculate_metrics(detections)
            metrics['processing_time'] = processing_time

            logging.info(f"小目标检测完成: {len(detections.xyxy)} 个目标, 耗时 {processing_time:.2f}s")

            return {
                'annotated_image': annotated_image,
                'detections': detections,
                'labels': labels,
                'statistics': statistics,
                'metrics': metrics,
                'detection_count': len(detections.xyxy),
                'method': 'InferenceSlicer'
            }

        except Exception as e:
            logging.error(f"小目标检测失败: {e}")
            return {
                'annotated_image': image,
                'detections': None,
                'labels': [],
                'statistics': {},
                'metrics': {},
                'detection_count': 0,
                'method': 'InferenceSlicer',
                'error': str(e)
            }

    def _estimate_slice_count(self, image_shape: Tuple[int, int],
                            slice_wh: Tuple[int, int],
                            overlap_wh: Tuple[int, int]) -> int:
        """估算切片数量"""
        height, width = image_shape
        slice_w, slice_h = slice_wh
        overlap_w, overlap_h = overlap_wh

        # 计算步长
        step_w = slice_w - overlap_w
        step_h = slice_h - overlap_h

        # 计算切片数量
        cols = max(1, (width - overlap_w + step_w - 1) // step_w)
        rows = max(1, (height - overlap_h + step_h - 1) // step_h)

        return rows * cols

    def create_comparison_view(self, original: np.ndarray,
                             annotated: np.ndarray) -> np.ndarray:
        """创建对比视图"""
        # 确保两个图像尺寸相同
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]
        
        if h1 != h2 or w1 != w2:
            annotated = cv2.resize(annotated, (w1, h1))
        
        # 水平拼接
        comparison = np.hstack([original, annotated])
        
        # 添加分割线
        cv2.line(comparison, (w1, 0), (w1, h1), (255, 255, 255), 2)
        
        # 添加标题
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Enhanced", (w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison
    
    def generate_detection_summary(self, statistics: Dict[str, Any]) -> str:
        """生成检测摘要文本"""
        summary_lines = []
        
        # 总检测数
        total = statistics.get('total_detections', 0)
        summary_lines.append(f"总检测数: {total}")
        
        # 类别分布
        class_dist = statistics.get('class_distribution', {})
        if class_dist:
            summary_lines.append("类别分布:")
            for class_name, count in class_dist.items():
                summary_lines.append(f"  {class_name}: {count}")
        
        # 置信度统计
        conf_stats = statistics.get('confidence_stats', {})
        if conf_stats:
            summary_lines.append(f"平均置信度: {conf_stats.get('mean', 0):.3f}")
            summary_lines.append(f"置信度范围: {conf_stats.get('min', 0):.3f} - {conf_stats.get('max', 0):.3f}")
        
        return "\n".join(summary_lines)

    def configure_small_object_detection(self,
                                       slice_wh: Tuple[int, int] = (640, 640),
                                       overlap_wh: Tuple[int, int] = (128, 128),
                                       iou_threshold: float = 0.5,
                                       thread_workers: int = 1):
        """
        配置小目标检测参数

        Args:
            slice_wh: 切片尺寸 (width, height)
            overlap_wh: 重叠尺寸 (width, height)
            iou_threshold: NMS IoU 阈值
            thread_workers: 线程数
        """
        self.small_object_config.update({
            'slice_wh': slice_wh,
            'overlap_wh': overlap_wh,
            'iou_threshold': iou_threshold,
            'thread_workers': thread_workers
        })
        logging.info(f"小目标检测配置已更新: {self.small_object_config}")

    def get_optimal_slice_config(self, image_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        """
        根据图像尺寸推荐最优切片配置

        Args:
            image_shape: 图像尺寸 (height, width)

        Returns:
            推荐的切片配置
        """
        height, width = image_shape

        # 根据图像尺寸选择合适的切片大小
        if width <= 1920 and height <= 1080:  # 1080p 及以下
            slice_wh = (640, 640)
            overlap_wh = (64, 64)
        elif width <= 3840 and height <= 2160:  # 4K
            slice_wh = (800, 800)
            overlap_wh = (128, 128)
        else:  # 更高分辨率
            slice_wh = (1024, 1024)
            overlap_wh = (256, 256)

        return {
            'slice_wh': slice_wh,
            'overlap_wh': overlap_wh,
            'estimated_slices': self._estimate_slice_count(image_shape, slice_wh, overlap_wh)
        }

    def detect_with_multiple_scales(self, image: np.ndarray, model,
                                  conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """
        多尺度小目标检测

        Args:
            image: 输入图像
            model: YOLO 模型
            conf: 置信度阈值
            iou: IoU 阈值

        Returns:
            多尺度检测结果
        """
        try:
            # 定义多个尺度配置
            scale_configs = [
                {'slice_wh': (320, 320), 'overlap_wh': (64, 64)},   # 小切片，适合极小目标
                {'slice_wh': (640, 640), 'overlap_wh': (128, 128)}, # 中等切片
                {'slice_wh': (960, 960), 'overlap_wh': (192, 192)}  # 大切片，适合中等目标
            ]

            all_detections = []
            scale_results = {}

            for i, config in enumerate(scale_configs):
                logging.info(f"执行第 {i+1} 尺度检测: {config}")

                result = self.detect_small_objects(
                    image, model, conf, iou,
                    slice_wh=config['slice_wh'],
                    overlap_wh=config['overlap_wh']
                )

                if result['detections'] is not None:
                    all_detections.append(result['detections'])
                    scale_results[f'scale_{i+1}'] = {
                        'config': config,
                        'detection_count': result['detection_count'],
                        'processing_time': result['statistics'].get('processing_time', 0)
                    }

            # 合并所有尺度的检测结果
            if all_detections:
                # 使用 Supervision 的合并功能
                merged_detections = self._merge_multi_scale_detections(all_detections, iou)

                # 生成最终可视化
                labels = self._generate_labels(merged_detections)
                annotated_image = self._create_enhanced_visualization(
                    image.copy(), merged_detections, labels
                )

                # 计算统计信息
                statistics = self._calculate_statistics(merged_detections)
                statistics['scale_results'] = scale_results
                statistics['total_scales'] = len(scale_configs)

                return {
                    'annotated_image': annotated_image,
                    'detections': merged_detections,
                    'labels': labels,
                    'statistics': statistics,
                    'detection_count': len(merged_detections.xyxy),
                    'method': 'MultiScale'
                }
            else:
                return {
                    'annotated_image': image,
                    'detections': None,
                    'labels': [],
                    'statistics': {'scale_results': scale_results},
                    'detection_count': 0,
                    'method': 'MultiScale'
                }

        except Exception as e:
            logging.error(f"多尺度检测失败: {e}")
            return {
                'annotated_image': image,
                'detections': None,
                'labels': [],
                'statistics': {},
                'detection_count': 0,
                'method': 'MultiScale',
                'error': str(e)
            }

    def _merge_multi_scale_detections(self, detections_list: List[sv.Detections],
                                    iou_threshold: float = 0.5) -> sv.Detections:
        """合并多尺度检测结果"""
        if not detections_list:
            return sv.Detections.empty()

        if len(detections_list) == 1:
            return detections_list[0]

        # 合并所有检测结果
        all_xyxy = []
        all_confidence = []
        all_class_id = []
        all_masks = []

        for detections in detections_list:
            if len(detections.xyxy) > 0:
                all_xyxy.append(detections.xyxy)
                if detections.confidence is not None:
                    all_confidence.append(detections.confidence)
                if detections.class_id is not None:
                    all_class_id.append(detections.class_id)
                if detections.mask is not None:
                    all_masks.append(detections.mask)

        if not all_xyxy:
            return sv.Detections.empty()

        # 拼接所有数据
        merged_xyxy = np.vstack(all_xyxy)
        merged_confidence = np.concatenate(all_confidence) if all_confidence else None
        merged_class_id = np.concatenate(all_class_id) if all_class_id else None
        merged_masks = np.vstack(all_masks) if all_masks else None

        # 创建合并的检测结果
        merged_detections = sv.Detections(
            xyxy=merged_xyxy,
            confidence=merged_confidence,
            class_id=merged_class_id,
            mask=merged_masks
        )

        # 应用 NMS 去除重复检测
        merged_detections = merged_detections.with_nms(threshold=iou_threshold)

        return merged_detections

    # ==================== 新增标注器管理方法 ====================

    def set_annotator_preset(self, preset_name: str):
        """设置标注器预设"""
        if self.annotator_manager:
            self.annotator_manager.set_preset(preset_name)
            logging.info(f"已设置标注器预设: {preset_name}")
        else:
            logging.warning("标注器管理器未初始化，无法设置预设")

    def enable_annotator(self, annotator_type: str):
        """启用指定标注器"""
        if self.annotator_manager:
            try:
                annotator_enum = AnnotatorType(annotator_type)
                self.annotator_manager.enable_annotator(annotator_enum)
            except ValueError:
                logging.error(f"未知的标注器类型: {annotator_type}")
        else:
            logging.warning("标注器管理器未初始化")

    def disable_annotator(self, annotator_type: str):
        """禁用指定标注器"""
        if self.annotator_manager:
            try:
                annotator_enum = AnnotatorType(annotator_type)
                self.annotator_manager.disable_annotator(annotator_enum)
            except ValueError:
                logging.error(f"未知的标注器类型: {annotator_type}")
        else:
            logging.warning("标注器管理器未初始化")

    def toggle_annotator(self, annotator_type: str):
        """切换标注器状态"""
        if self.annotator_manager:
            try:
                annotator_enum = AnnotatorType(annotator_type)
                self.annotator_manager.toggle_annotator(annotator_enum)
            except ValueError:
                logging.error(f"未知的标注器类型: {annotator_type}")
        else:
            logging.warning("标注器管理器未初始化")

    def get_enabled_annotators(self) -> List[str]:
        """获取已启用的标注器列表"""
        if self.annotator_manager:
            return self.annotator_manager.get_enabled_annotators()
        else:
            return ["box", "label"]  # 默认启用的标注器

    def get_annotator_info(self) -> Dict[str, Any]:
        """获取标注器信息"""
        if self.annotator_manager:
            return self.annotator_manager.get_annotator_info()
        else:
            return {
                'available_annotators': ["box", "label"],
                'enabled_annotators': ["box", "label"],
                'total_annotators': 2,
                'enabled_count': 2,
                'presets': ['basic']
            }

    def create_annotated_comparison(self, image: np.ndarray, detections: sv.Detections,
                                  labels: Optional[List[str]] = None) -> np.ndarray:
        """创建标注对比视图"""
        if self.annotator_manager:
            return self.annotator_manager.create_comparison_view(image, detections, labels)
        else:
            # 使用基础方法创建对比视图
            return self.create_comparison_view(image, self._create_enhanced_visualization(image, detections, labels or []))

    def update_annotator_config(self, annotator_type: str, **kwargs):
        """更新标注器配置"""
        if self.annotator_manager:
            try:
                annotator_enum = AnnotatorType(annotator_type)
                self.annotator_manager.update_annotator_config(annotator_enum, **kwargs)
            except ValueError:
                logging.error(f"未知的标注器类型: {annotator_type}")
        else:
            logging.warning("标注器管理器未初始化，无法更新配置")

    def clear_heatmap_history(self):
        """清除热力图历史数据"""
        if self.annotator_manager:
            self.annotator_manager.clear_heatmap_history()
        else:
            logging.warning("标注器管理器未初始化")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        base_stats = {
            'small_object_detection': {
                'slice_config': self.small_object_config,
                'enabled': True
            }
        }

        if self.annotator_manager:
            annotator_stats = self.annotator_manager.get_performance_stats()
            base_stats['annotators'] = annotator_stats

        return base_stats

    def save_annotator_config(self, config_path: Optional[str] = None):
        """保存标注器配置"""
        if self.annotator_manager:
            self.annotator_manager.save_config(config_path)
        else:
            logging.warning("标注器管理器未初始化，无法保存配置")

    def get_available_presets(self) -> List[Dict[str, Any]]:
        """获取可用的预设配置"""
        if self.annotator_manager:
            return AnnotatorPresets.get_all_presets()
        else:
            return [{'name': 'basic', 'description': '基础模式', 'annotators': ['box', 'label']}]


class SupervisionAnalyzer:
    """Supervision 分析工具"""
    
    def __init__(self):
        self.detection_history = []
    
    def add_detection_result(self, result: Dict[str, Any]):
        """添加检测结果到历史记录"""
        self.detection_history.append(result)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.detection_history:
            return {}
        
        # 统计历史数据
        total_detections = sum(r.get('detection_count', 0) for r in self.detection_history)
        avg_detections = total_detections / len(self.detection_history)
        
        return {
            'total_processed': len(self.detection_history),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections,
            'processing_history': self.detection_history[-10:]  # 最近10次结果
        }
