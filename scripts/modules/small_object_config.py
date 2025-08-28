#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小目标检测配置管理器
管理小目标检测的各种配置参数
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SliceConfig:
    """切片配置数据类"""
    slice_wh: Tuple[int, int]
    overlap_wh: Tuple[int, int]
    confidence_threshold: float
    iou_threshold: float
    description: str = ""


@dataclass
class MultiScaleConfig:
    """多尺度配置数据类"""
    name: str
    slice_wh: Tuple[int, int]
    overlap_wh: Tuple[int, int]
    weight: float
    description: str = ""


class SmallObjectConfigManager:
    """小目标检测配置管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为 None 则使用默认路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置配置文件路径
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "assets" / "configs" / "small_object_detection_config.yaml"
        
        self.config_path = config_path
        self.config = {}
        self.presets = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                
                # 加载预设配置
                self._load_presets()
                
                self.logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                self.logger.warning(f"配置文件不存在: {self.config_path}")
                self._create_default_config()
                
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            self._create_default_config()
    
    def _load_presets(self):
        """加载预设配置"""
        presets_config = self.config.get('presets', {})
        
        for preset_name, preset_data in presets_config.items():
            self.presets[preset_name] = SliceConfig(
                slice_wh=tuple(preset_data.get('slice_wh', [640, 640])),
                overlap_wh=tuple(preset_data.get('overlap_wh', [128, 128])),
                confidence_threshold=preset_data.get('confidence_threshold', 0.25),
                iou_threshold=preset_data.get('iou_threshold', 0.45),
                description=preset_data.get('description', '')
            )
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config = {
            'basic': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'enable_small_object_detection': False
            },
            'inference_slicer': {
                'default_slice_wh': [640, 640],
                'default_overlap_wh': [128, 128],
                'nms_iou_threshold': 0.5,
                'overlap_filter': 'NON_MAX_SUPPRESSION',
                'thread_workers': 1
            }
        }
        
        # 创建默认预设
        self.presets = {
            'small': SliceConfig(
                slice_wh=(640, 640),
                overlap_wh=(128, 128),
                confidence_threshold=0.25,
                iou_threshold=0.45,
                description="标准小目标检测配置"
            )
        }
        
        self.logger.info("使用默认配置")
    
    def get_basic_config(self) -> Dict[str, Any]:
        """获取基础配置"""
        return self.config.get('basic', {})
    
    def get_inference_slicer_config(self) -> Dict[str, Any]:
        """获取 InferenceSlicer 配置"""
        return self.config.get('inference_slicer', {})
    
    def get_preset_config(self, preset_name: str) -> Optional[SliceConfig]:
        """
        获取预设配置
        
        Args:
            preset_name: 预设名称
            
        Returns:
            预设配置对象，如果不存在则返回 None
        """
        return self.presets.get(preset_name)
    
    def get_available_presets(self) -> List[str]:
        """获取可用的预设名称列表"""
        return list(self.presets.keys())
    
    def get_adaptive_config(self, image_shape: Tuple[int, int]) -> SliceConfig:
        """
        根据图像尺寸获取自适应配置
        
        Args:
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            自适应配置对象
        """
        height, width = image_shape
        adaptive_rules = self.config.get('adaptive_rules', {}).get('resolution_based', {})
        
        # 根据分辨率选择配置
        if width <= adaptive_rules.get('low_res', {}).get('max_width', 1920):
            # 低分辨率配置
            rule = adaptive_rules.get('low_res', {})
            slice_wh = tuple(rule.get('recommended_slice_wh', [640, 640]))
            overlap_wh = tuple(rule.get('recommended_overlap_wh', [64, 64]))
        elif width <= adaptive_rules.get('medium_res', {}).get('max_width', 3840):
            # 中等分辨率配置
            rule = adaptive_rules.get('medium_res', {})
            slice_wh = tuple(rule.get('recommended_slice_wh', [800, 800]))
            overlap_wh = tuple(rule.get('recommended_overlap_wh', [128, 128]))
        else:
            # 高分辨率配置
            rule = adaptive_rules.get('high_res', {})
            slice_wh = tuple(rule.get('recommended_slice_wh', [1024, 1024]))
            overlap_wh = tuple(rule.get('recommended_overlap_wh', [256, 256]))
        
        basic_config = self.get_basic_config()
        
        return SliceConfig(
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            confidence_threshold=basic_config.get('confidence_threshold', 0.25),
            iou_threshold=basic_config.get('iou_threshold', 0.45),
            description=f"自适应配置 (图像尺寸: {width}x{height})"
        )
    
    def get_multi_scale_configs(self) -> List[MultiScaleConfig]:
        """获取多尺度检测配置"""
        multi_scale_config = self.config.get('multi_scale', {})
        scales = multi_scale_config.get('scales', [])
        
        configs = []
        for scale_data in scales:
            config = MultiScaleConfig(
                name=scale_data.get('name', ''),
                slice_wh=tuple(scale_data.get('slice_wh', [640, 640])),
                overlap_wh=tuple(scale_data.get('overlap_wh', [128, 128])),
                weight=scale_data.get('weight', 1.0),
                description=scale_data.get('description', '')
            )
            configs.append(config)
        
        return configs
    
    def get_visdrone_config(self) -> Dict[str, Any]:
        """获取 VisDrone 数据集特定配置"""
        return self.config.get('visdrone', {})
    
    def get_visdrone_class_names(self) -> List[str]:
        """获取 VisDrone 类别名称"""
        visdrone_config = self.get_visdrone_config()
        return visdrone_config.get('class_names', [])
    
    def get_visdrone_optimized_config(self) -> SliceConfig:
        """获取 VisDrone 优化配置"""
        visdrone_config = self.get_visdrone_config()
        optimized = visdrone_config.get('optimized_config', {})
        
        return SliceConfig(
            slice_wh=tuple(optimized.get('slice_wh', [640, 640])),
            overlap_wh=tuple(optimized.get('overlap_wh', [128, 128])),
            confidence_threshold=optimized.get('confidence_threshold', 0.2),
            iou_threshold=optimized.get('iou_threshold', 0.5),
            description=optimized.get('description', 'VisDrone 优化配置')
        )
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.config.get('performance', {})
    
    def is_multi_scale_enabled(self) -> bool:
        """检查是否启用多尺度检测"""
        return self.config.get('multi_scale', {}).get('enabled', False)
    
    def is_debug_enabled(self) -> bool:
        """检查是否启用调试模式"""
        return self.config.get('debug', {}).get('enabled', False)
    
    def save_config(self):
        """保存配置到文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"配置已保存: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"配置保存失败: {e}")
    
    def update_basic_config(self, **kwargs):
        """更新基础配置"""
        if 'basic' not in self.config:
            self.config['basic'] = {}
        
        self.config['basic'].update(kwargs)
        self.logger.info(f"基础配置已更新: {kwargs}")
    
    def create_custom_preset(self, name: str, slice_config: SliceConfig):
        """
        创建自定义预设
        
        Args:
            name: 预设名称
            slice_config: 切片配置
        """
        if 'presets' not in self.config:
            self.config['presets'] = {}
        
        self.config['presets'][name] = {
            'slice_wh': list(slice_config.slice_wh),
            'overlap_wh': list(slice_config.overlap_wh),
            'confidence_threshold': slice_config.confidence_threshold,
            'iou_threshold': slice_config.iou_threshold,
            'description': slice_config.description
        }
        
        # 更新内存中的预设
        self.presets[name] = slice_config
        
        self.logger.info(f"自定义预设已创建: {name}")
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        summary_lines = []
        summary_lines.append("=== 小目标检测配置摘要 ===")
        
        # 基础配置
        basic = self.get_basic_config()
        summary_lines.append(f"基础配置:")
        summary_lines.append(f"  置信度阈值: {basic.get('confidence_threshold', 0.25)}")
        summary_lines.append(f"  IoU 阈值: {basic.get('iou_threshold', 0.45)}")
        summary_lines.append(f"  启用小目标检测: {basic.get('enable_small_object_detection', False)}")
        
        # 预设配置
        summary_lines.append(f"可用预设: {len(self.presets)} 个")
        for name, config in self.presets.items():
            summary_lines.append(f"  {name}: {config.slice_wh}, {config.description}")
        
        # 多尺度配置
        summary_lines.append(f"多尺度检测: {'启用' if self.is_multi_scale_enabled() else '禁用'}")
        
        return "\n".join(summary_lines)


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> SmallObjectConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SmallObjectConfigManager()
    return _config_manager
