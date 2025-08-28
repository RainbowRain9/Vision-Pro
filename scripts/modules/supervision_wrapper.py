#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 功能包装器
为 YOLOvision Pro 提供增强的可视化和分析功能
"""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path


class SupervisionWrapper:
    """Supervision 功能统一包装器"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        初始化 Supervision 包装器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names or []
        
        # 初始化标注器 (适配 Supervision 0.26.1+ API)
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
        
        logging.info("Supervision 包装器初始化完成")
    
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
