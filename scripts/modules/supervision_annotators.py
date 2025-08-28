#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 标注器管理模块
为 YOLOvision Pro 提供统一的标注器管理和配置功能
支持 7 种标注器：Box, Label, Mask, Polygon, HeatMap, Blur, Pixelate
"""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from enum import Enum
import time


class AnnotatorType(Enum):
    """标注器类型枚举"""
    BOX = "box"
    LABEL = "label"
    MASK = "mask"
    POLYGON = "polygon"
    HEATMAP = "heatmap"
    BLUR = "blur"
    PIXELATE = "pixelate"


@dataclass
class AnnotatorConfig:
    """标注器配置数据类"""
    enabled: bool = True
    thickness: int = 2
    color: Optional[Tuple[int, int, int]] = None
    text_scale: float = 0.5
    text_thickness: int = 1
    text_padding: int = 5
    opacity: float = 1.0
    kernel_size: int = 15
    pixel_size: int = 20
    custom_params: Dict[str, Any] = field(default_factory=dict)


class AnnotatorManager:
    """统一的标注器管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化标注器管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.configs: Dict[AnnotatorType, AnnotatorConfig] = {}
        self.annotators: Dict[AnnotatorType, Any] = {}
        self.enabled_annotators: List[AnnotatorType] = []
        
        # 热力图相关
        self.heatmap_data = None
        self.heatmap_history: List[sv.Detections] = []
        self.max_heatmap_frames = 100
        
        # 初始化
        self._load_config()
        self._initialize_annotators()
        
        logging.info("标注器管理器初始化完成")
    
    def _load_config(self):
        """加载配置文件"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                self._parse_config(config_data)
            except Exception as e:
                logging.warning(f"加载配置文件失败: {e}，使用默认配置")
                self._load_default_config()
        else:
            self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        default_configs = {
            AnnotatorType.BOX: AnnotatorConfig(
                enabled=True,
                thickness=2,
                color=None  # 使用默认颜色
            ),
            AnnotatorType.LABEL: AnnotatorConfig(
                enabled=True,
                text_scale=0.5,
                text_thickness=1,
                text_padding=5
            ),
            AnnotatorType.MASK: AnnotatorConfig(
                enabled=False,
                opacity=0.5
            ),
            AnnotatorType.POLYGON: AnnotatorConfig(
                enabled=False,
                thickness=2
            ),
            AnnotatorType.HEATMAP: AnnotatorConfig(
                enabled=False,
                opacity=0.7
            ),
            AnnotatorType.BLUR: AnnotatorConfig(
                enabled=False,
                kernel_size=15
            ),
            AnnotatorType.PIXELATE: AnnotatorConfig(
                enabled=False,
                pixel_size=20
            )
        }
        self.configs = default_configs
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """解析配置数据"""
        annotators_config = config_data.get('annotators', {})
        
        for annotator_type in AnnotatorType:
            type_config = annotators_config.get(annotator_type.value, {})
            self.configs[annotator_type] = AnnotatorConfig(
                enabled=type_config.get('enabled', False),
                thickness=type_config.get('thickness', 2),
                color=tuple(type_config['color']) if type_config.get('color') else None,
                text_scale=type_config.get('text_scale', 0.5),
                text_thickness=type_config.get('text_thickness', 1),
                text_padding=type_config.get('text_padding', 5),
                opacity=type_config.get('opacity', 1.0),
                kernel_size=type_config.get('kernel_size', 15),
                pixel_size=type_config.get('pixel_size', 20),
                custom_params=type_config.get('custom_params', {})
            )
    
    def _initialize_annotators(self):
        """初始化所有标注器"""
        initialized_count = 0

        # Box Annotator
        try:
            box_config = self.configs[AnnotatorType.BOX]
            self.annotators[AnnotatorType.BOX] = sv.BoxAnnotator(
                thickness=box_config.thickness
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 BoxAnnotator 失败: {e}")
            self.annotators[AnnotatorType.BOX] = sv.BoxAnnotator()

        # Label Annotator
        try:
            label_config = self.configs[AnnotatorType.LABEL]
            self.annotators[AnnotatorType.LABEL] = sv.LabelAnnotator(
                text_thickness=label_config.text_thickness,
                text_scale=label_config.text_scale,
                text_padding=label_config.text_padding
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 LabelAnnotator 失败: {e}")
            self.annotators[AnnotatorType.LABEL] = sv.LabelAnnotator()

        # Mask Annotator
        try:
            mask_config = self.configs[AnnotatorType.MASK]
            self.annotators[AnnotatorType.MASK] = sv.MaskAnnotator(
                opacity=mask_config.opacity
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 MaskAnnotator 失败: {e}")
            # MaskAnnotator 可能在某些版本中不可用
            self.annotators[AnnotatorType.MASK] = None
            self.configs[AnnotatorType.MASK].enabled = False

        # Polygon Annotator
        try:
            polygon_config = self.configs[AnnotatorType.POLYGON]
            self.annotators[AnnotatorType.POLYGON] = sv.PolygonAnnotator(
                thickness=polygon_config.thickness
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 PolygonAnnotator 失败: {e}")
            self.annotators[AnnotatorType.POLYGON] = None
            self.configs[AnnotatorType.POLYGON].enabled = False

        # HeatMap Annotator
        try:
            heatmap_config = self.configs[AnnotatorType.HEATMAP]
            self.annotators[AnnotatorType.HEATMAP] = sv.HeatMapAnnotator(
                opacity=heatmap_config.opacity
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 HeatMapAnnotator 失败: {e}")
            self.annotators[AnnotatorType.HEATMAP] = None
            self.configs[AnnotatorType.HEATMAP].enabled = False

        # Blur Annotator
        try:
            blur_config = self.configs[AnnotatorType.BLUR]
            self.annotators[AnnotatorType.BLUR] = sv.BlurAnnotator(
                kernel_size=blur_config.kernel_size
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 BlurAnnotator 失败: {e}")
            self.annotators[AnnotatorType.BLUR] = None
            self.configs[AnnotatorType.BLUR].enabled = False

        # Pixelate Annotator
        try:
            pixelate_config = self.configs[AnnotatorType.PIXELATE]
            self.annotators[AnnotatorType.PIXELATE] = sv.PixelateAnnotator(
                pixel_size=pixelate_config.pixel_size
            )
            initialized_count += 1
        except Exception as e:
            logging.warning(f"初始化 PixelateAnnotator 失败: {e}")
            self.annotators[AnnotatorType.PIXELATE] = None
            self.configs[AnnotatorType.PIXELATE].enabled = False

        # 更新启用的标注器列表（只包含成功初始化的）
        self.enabled_annotators = [
            annotator_type for annotator_type, config in self.configs.items()
            if config.enabled and self.annotators.get(annotator_type) is not None
        ]

        logging.info(f"成功初始化 {initialized_count}/{len(AnnotatorType)} 个标注器")
        logging.info(f"已启用标注器: {[t.value for t in self.enabled_annotators]}")

        # 确保至少有基本标注器可用
        if not self.enabled_annotators:
            logging.warning("没有可用的标注器，启用基本标注器")
            self.enabled_annotators = [AnnotatorType.BOX, AnnotatorType.LABEL]
            self.configs[AnnotatorType.BOX].enabled = True
            self.configs[AnnotatorType.LABEL].enabled = True
    
    def annotate_image(self, image: np.ndarray, detections: sv.Detections, 
                      labels: Optional[List[str]] = None,
                      custom_annotators: Optional[List[AnnotatorType]] = None) -> np.ndarray:
        """
        使用启用的标注器标注图像
        
        Args:
            image: 输入图像
            detections: 检测结果
            labels: 标签列表
            custom_annotators: 自定义标注器列表（覆盖默认启用列表）
            
        Returns:
            标注后的图像
        """
        if detections is None or len(detections.xyxy) == 0:
            return image.copy()
        
        annotated_image = image.copy()
        annotators_to_use = custom_annotators or self.enabled_annotators
        
        # 按特定顺序应用标注器以获得最佳视觉效果
        annotation_order = [
            AnnotatorType.HEATMAP,  # 背景层
            AnnotatorType.MASK,     # 分割掩码
            AnnotatorType.BLUR,     # 模糊效果
            AnnotatorType.PIXELATE, # 像素化效果
            AnnotatorType.POLYGON,  # 多边形
            AnnotatorType.BOX,      # 边界框
            AnnotatorType.LABEL     # 标签（最上层）
        ]
        
        for annotator_type in annotation_order:
            if annotator_type in annotators_to_use and annotator_type in self.annotators:
                try:
                    annotated_image = self._apply_single_annotator(
                        annotated_image, detections, annotator_type, labels
                    )
                except Exception as e:
                    logging.warning(f"应用标注器 {annotator_type.value} 失败: {e}")
        
        return annotated_image
    
    def _apply_single_annotator(self, image: np.ndarray, detections: sv.Detections,
                               annotator_type: AnnotatorType, labels: Optional[List[str]] = None) -> np.ndarray:
        """应用单个标注器"""
        annotator = self.annotators.get(annotator_type)

        # 检查标注器是否可用
        if annotator is None:
            logging.warning(f"标注器 {annotator_type.value} 不可用，跳过")
            return image

        try:
            if annotator_type == AnnotatorType.LABEL and labels:
                return annotator.annotate(scene=image, detections=detections, labels=labels)
            elif annotator_type == AnnotatorType.HEATMAP:
                # 热力图需要特殊处理
                return self._apply_heatmap_annotator(image, detections)
            elif annotator_type == AnnotatorType.MASK:
                # 检查是否有分割掩码数据
                if detections.mask is not None:
                    return annotator.annotate(scene=image, detections=detections)
                else:
                    logging.debug("没有分割掩码数据，跳过 MaskAnnotator")
                    return image
            else:
                return annotator.annotate(scene=image, detections=detections)
        except Exception as e:
            logging.warning(f"应用标注器 {annotator_type.value} 时出错: {e}")
            return image
    
    def _apply_heatmap_annotator(self, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """应用热力图标注器"""
        # 添加当前检测到历史记录
        self.heatmap_history.append(detections)
        
        # 限制历史记录长度
        if len(self.heatmap_history) > self.max_heatmap_frames:
            self.heatmap_history.pop(0)
        
        # 如果历史记录不足，直接返回原图
        if len(self.heatmap_history) < 5:
            return image
        
        # 使用最新的检测结果应用热力图
        return self.annotators[AnnotatorType.HEATMAP].annotate(scene=image, detections=detections)
    
    def enable_annotator(self, annotator_type: AnnotatorType):
        """启用指定标注器"""
        if annotator_type not in self.enabled_annotators:
            self.enabled_annotators.append(annotator_type)
            self.configs[annotator_type].enabled = True
            logging.info(f"已启用标注器: {annotator_type.value}")
    
    def disable_annotator(self, annotator_type: AnnotatorType):
        """禁用指定标注器"""
        if annotator_type in self.enabled_annotators:
            self.enabled_annotators.remove(annotator_type)
            self.configs[annotator_type].enabled = False
            logging.info(f"已禁用标注器: {annotator_type.value}")
    
    def toggle_annotator(self, annotator_type: AnnotatorType):
        """切换标注器状态"""
        if annotator_type in self.enabled_annotators:
            self.disable_annotator(annotator_type)
        else:
            self.enable_annotator(annotator_type)
    
    def set_preset(self, preset_name: str):
        """设置预设配置"""
        presets = {
            'basic': [AnnotatorType.BOX, AnnotatorType.LABEL],
            'detailed': [AnnotatorType.BOX, AnnotatorType.LABEL, AnnotatorType.POLYGON],
            'privacy': [AnnotatorType.BLUR, AnnotatorType.LABEL],
            'analysis': [AnnotatorType.BOX, AnnotatorType.LABEL, AnnotatorType.HEATMAP],
            'segmentation': [AnnotatorType.MASK, AnnotatorType.LABEL],
            'all': list(AnnotatorType)
        }
        
        if preset_name in presets:
            # 先禁用所有标注器
            self.enabled_annotators.clear()
            for annotator_type in AnnotatorType:
                self.configs[annotator_type].enabled = False
            
            # 启用预设中的标注器
            for annotator_type in presets[preset_name]:
                self.enable_annotator(annotator_type)
            
            logging.info(f"已应用预设配置: {preset_name}")
        else:
            logging.warning(f"未知的预设配置: {preset_name}")
    
    def get_enabled_annotators(self) -> List[str]:
        """获取已启用的标注器列表"""
        return [annotator_type.value for annotator_type in self.enabled_annotators]
    
    def clear_heatmap_history(self):
        """清除热力图历史数据"""
        self.heatmap_history.clear()
        logging.info("已清除热力图历史数据")
    
    def update_annotator_config(self, annotator_type: AnnotatorType, **kwargs):
        """更新标注器配置"""
        if annotator_type in self.configs:
            config = self.configs[annotator_type]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # 重新初始化该标注器
            self._reinitialize_single_annotator(annotator_type)
            logging.info(f"已更新标注器配置: {annotator_type.value}")
    
    def _reinitialize_single_annotator(self, annotator_type: AnnotatorType):
        """重新初始化单个标注器"""
        try:
            config = self.configs[annotator_type]
            
            if annotator_type == AnnotatorType.BOX:
                self.annotators[annotator_type] = sv.BoxAnnotator(thickness=config.thickness)
            elif annotator_type == AnnotatorType.LABEL:
                self.annotators[annotator_type] = sv.LabelAnnotator(
                    text_thickness=config.text_thickness,
                    text_scale=config.text_scale,
                    text_padding=config.text_padding
                )
            elif annotator_type == AnnotatorType.MASK:
                self.annotators[annotator_type] = sv.MaskAnnotator(opacity=config.opacity)
            elif annotator_type == AnnotatorType.POLYGON:
                self.annotators[annotator_type] = sv.PolygonAnnotator(thickness=config.thickness)
            elif annotator_type == AnnotatorType.HEATMAP:
                self.annotators[annotator_type] = sv.HeatMapAnnotator(opacity=config.opacity)
            elif annotator_type == AnnotatorType.BLUR:
                self.annotators[annotator_type] = sv.BlurAnnotator(kernel_size=config.kernel_size)
            elif annotator_type == AnnotatorType.PIXELATE:
                self.annotators[annotator_type] = sv.PixelateAnnotator(pixel_size=config.pixel_size)
                
        except Exception as e:
            logging.error(f"重新初始化标注器 {annotator_type.value} 失败: {e}")

    def save_config(self, config_path: Optional[str] = None):
        """保存当前配置到文件"""
        save_path = config_path or self.config_path
        if not save_path:
            logging.warning("未指定配置文件路径，无法保存配置")
            return

        config_data = {
            'annotators': {}
        }

        for annotator_type, config in self.configs.items():
            config_data['annotators'][annotator_type.value] = {
                'enabled': config.enabled,
                'thickness': config.thickness,
                'color': list(config.color) if config.color else None,
                'text_scale': config.text_scale,
                'text_thickness': config.text_thickness,
                'text_padding': config.text_padding,
                'opacity': config.opacity,
                'kernel_size': config.kernel_size,
                'pixel_size': config.pixel_size,
                'custom_params': config.custom_params
            }

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            logging.info(f"配置已保存到: {save_path}")
        except Exception as e:
            logging.error(f"保存配置失败: {e}")

    def get_annotator_info(self) -> Dict[str, Any]:
        """获取标注器信息"""
        return {
            'available_annotators': [t.value for t in AnnotatorType],
            'enabled_annotators': self.get_enabled_annotators(),
            'total_annotators': len(AnnotatorType),
            'enabled_count': len(self.enabled_annotators),
            'heatmap_frames': len(self.heatmap_history),
            'presets': ['basic', 'detailed', 'privacy', 'analysis', 'segmentation', 'all']
        }

    def create_comparison_view(self, image: np.ndarray, detections: sv.Detections,
                             labels: Optional[List[str]] = None) -> np.ndarray:
        """创建对比视图：原图 vs 标注图"""
        if detections is None or len(detections.xyxy) == 0:
            return np.hstack([image, image])

        # 创建标注图像
        annotated_image = self.annotate_image(image, detections, labels)

        # 确保两个图像尺寸相同
        h, w = image.shape[:2]
        if annotated_image.shape[:2] != (h, w):
            annotated_image = cv2.resize(annotated_image, (w, h))

        # 水平拼接
        comparison = np.hstack([image, annotated_image])

        # 添加分割线
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Annotated", (w + 10, 30), font, 1, (255, 255, 255), 2)

        return comparison

    def batch_annotate(self, images: List[np.ndarray], detections_list: List[sv.Detections],
                      labels_list: Optional[List[List[str]]] = None) -> List[np.ndarray]:
        """批量标注图像"""
        if len(images) != len(detections_list):
            raise ValueError("图像数量与检测结果数量不匹配")

        if labels_list and len(labels_list) != len(images):
            raise ValueError("图像数量与标签数量不匹配")

        annotated_images = []
        for i, (image, detections) in enumerate(zip(images, detections_list)):
            labels = labels_list[i] if labels_list else None
            annotated_image = self.annotate_image(image, detections, labels)
            annotated_images.append(annotated_image)

        return annotated_images

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'enabled_annotators_count': len(self.enabled_annotators),
            'heatmap_history_size': len(self.heatmap_history),
            'memory_usage_estimate': self._estimate_memory_usage(),
            'recommended_max_fps': self._get_recommended_fps()
        }

    def _estimate_memory_usage(self) -> str:
        """估算内存使用量"""
        base_usage = 50  # MB
        annotator_usage = len(self.enabled_annotators) * 10  # MB per annotator
        heatmap_usage = len(self.heatmap_history) * 5  # MB per frame in history

        total_mb = base_usage + annotator_usage + heatmap_usage
        return f"{total_mb}MB"

    def _get_recommended_fps(self) -> int:
        """获取推荐的FPS"""
        if len(self.enabled_annotators) <= 2:
            return 30
        elif len(self.enabled_annotators) <= 4:
            return 20
        else:
            return 15


class AnnotatorPresets:
    """标注器预设配置类"""

    @staticmethod
    def get_privacy_preset() -> Dict[str, Any]:
        """隐私保护预设：模糊敏感区域"""
        return {
            'name': 'privacy',
            'description': '隐私保护模式，模糊检测到的目标',
            'annotators': [AnnotatorType.BLUR, AnnotatorType.LABEL],
            'config_overrides': {
                AnnotatorType.BLUR: {'kernel_size': 25},
                AnnotatorType.LABEL: {'text_scale': 0.4}
            }
        }

    @staticmethod
    def get_analysis_preset() -> Dict[str, Any]:
        """分析预设：详细的检测分析"""
        return {
            'name': 'analysis',
            'description': '分析模式，显示详细的检测信息和热力图',
            'annotators': [AnnotatorType.BOX, AnnotatorType.LABEL, AnnotatorType.HEATMAP, AnnotatorType.POLYGON],
            'config_overrides': {
                AnnotatorType.BOX: {'thickness': 3},
                AnnotatorType.HEATMAP: {'opacity': 0.6}
            }
        }

    @staticmethod
    def get_presentation_preset() -> Dict[str, Any]:
        """演示预设：适合演示的清晰标注"""
        return {
            'name': 'presentation',
            'description': '演示模式，清晰的边界框和标签',
            'annotators': [AnnotatorType.BOX, AnnotatorType.LABEL],
            'config_overrides': {
                AnnotatorType.BOX: {'thickness': 4},
                AnnotatorType.LABEL: {'text_scale': 0.8, 'text_thickness': 2}
            }
        }

    @staticmethod
    def get_segmentation_preset() -> Dict[str, Any]:
        """分割预设：显示分割掩码"""
        return {
            'name': 'segmentation',
            'description': '分割模式，显示分割掩码和标签',
            'annotators': [AnnotatorType.MASK, AnnotatorType.LABEL, AnnotatorType.POLYGON],
            'config_overrides': {
                AnnotatorType.MASK: {'opacity': 0.7},
                AnnotatorType.POLYGON: {'thickness': 2}
            }
        }

    @staticmethod
    def get_all_presets() -> List[Dict[str, Any]]:
        """获取所有预设配置"""
        return [
            AnnotatorPresets.get_privacy_preset(),
            AnnotatorPresets.get_analysis_preset(),
            AnnotatorPresets.get_presentation_preset(),
            AnnotatorPresets.get_segmentation_preset()
        ]


def create_default_config_file(config_path: str):
    """创建默认配置文件"""
    default_config = {
        'annotators': {
            'box': {
                'enabled': True,
                'thickness': 2,
                'color': None
            },
            'label': {
                'enabled': True,
                'text_scale': 0.5,
                'text_thickness': 1,
                'text_padding': 5
            },
            'mask': {
                'enabled': False,
                'opacity': 0.5
            },
            'polygon': {
                'enabled': False,
                'thickness': 2
            },
            'heatmap': {
                'enabled': False,
                'opacity': 0.7
            },
            'blur': {
                'enabled': False,
                'kernel_size': 15
            },
            'pixelate': {
                'enabled': False,
                'pixel_size': 20
            }
        },
        'presets': {
            'basic': ['box', 'label'],
            'detailed': ['box', 'label', 'polygon'],
            'privacy': ['blur', 'label'],
            'analysis': ['box', 'label', 'heatmap'],
            'segmentation': ['mask', 'label'],
            'all': ['box', 'label', 'mask', 'polygon', 'heatmap', 'blur', 'pixelate']
        },
        'performance': {
            'max_heatmap_frames': 100,
            'recommended_fps': {
                'low_load': 30,
                'medium_load': 20,
                'high_load': 15
            }
        }
    }

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        logging.info(f"默认配置文件已创建: {config_path}")
    except Exception as e:
        logging.error(f"创建默认配置文件失败: {e}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建标注器管理器
    manager = AnnotatorManager()

    # 显示信息
    info = manager.get_annotator_info()
    print("标注器信息:", info)

    # 测试预设
    manager.set_preset('analysis')
    print("分析预设已应用，启用的标注器:", manager.get_enabled_annotators())
