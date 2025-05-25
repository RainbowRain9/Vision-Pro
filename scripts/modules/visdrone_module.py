#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone 数据处理模块
提供统一的 VisDrone 数据集处理接口

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class VisDroneModule:
    """VisDrone 数据处理模块"""

    def __init__(self, scripts_dir: Optional[Path] = None):
        if scripts_dir is None:
            scripts_dir = Path(__file__).parent.parent

        self.scripts_dir = scripts_dir
        self.visdrone_dir = scripts_dir / "data_processing" / "visdrone"

        # 脚本路径映射
        self.scripts = {
            'convert': self.visdrone_dir / "convert_visdrone.py",
            'split': self.visdrone_dir / "split_visdrone_dataset.py",
            'validate': self.visdrone_dir / "validate_visdrone_dataset.py",
            'process': self.visdrone_dir / "process_visdrone_complete.py"
        }

    def _run_script(self, script_name: str, args: List[str]) -> Dict[str, Any]:
        """运行指定脚本"""
        script_path = self.scripts.get(script_name)
        if not script_path or not script_path.exists():
            return {
                'success': False,
                'error': f'脚本不存在: {script_name}',
                'returncode': 1
            }

        cmd = [sys.executable, str(script_path)] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                encoding='utf-8',
                errors='ignore'
            )
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'returncode': 1
            }

    def convert_dataset(self, input_dir: str, output_dir: str, verbose: bool = False) -> Dict[str, Any]:
        """转换 VisDrone 数据集格式"""
        args = ['--input', input_dir, '--output', output_dir]
        if verbose:
            args.append('--verbose')

        return self._run_script('convert', args)

    def split_dataset(self, input_dir: str, output_dir: str,
                     train_ratio: float = 0.8, val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, Any]:
        """划分数据集"""
        args = [
            '--input', input_dir,
            '--output', output_dir,
            '--train-ratio', str(train_ratio),
            '--val-ratio', str(val_ratio),
            '--test-ratio', str(test_ratio)
        ]

        return self._run_script('split', args)

    def validate_dataset(self, dataset_dir: str, visualize: bool = False,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """验证数据集"""
        args = ['--dataset', dataset_dir]
        if visualize:
            args.append('--visualize')
        if output_dir:
            args.extend(['--output', output_dir])

        return self._run_script('validate', args)

    def process_complete(self, input_dir: str, output_dir: str,
                        verbose: bool = False, no_visualization: bool = False,
                        train_ratio: float = 0.8, val_ratio: float = 0.1,
                        test_ratio: float = 0.1) -> Dict[str, Any]:
        """完整处理流程"""
        args = [
            '--input', input_dir,
            '--output', output_dir,
            '--train-ratio', str(train_ratio),
            '--val-ratio', str(val_ratio),
            '--test-ratio', str(test_ratio)
        ]

        if verbose:
            args.append('--verbose')
        if no_visualization:
            args.append('--no-visualization')

        return self._run_script('process', args)

    def get_available_operations(self) -> List[str]:
        """获取可用操作列表"""
        return list(self.scripts.keys())

    def check_dependencies(self) -> Dict[str, bool]:
        """检查依赖脚本是否存在"""
        return {name: path.exists() for name, path in self.scripts.items()}

# 便捷函数
def convert_visdrone(input_dir: str, output_dir: str, verbose: bool = False) -> bool:
    """便捷函数：转换 VisDrone 数据集"""
    module = VisDroneModule()
    result = module.convert_dataset(input_dir, output_dir, verbose)
    return result['success']

def process_visdrone_complete(input_dir: str, output_dir: str,
                             verbose: bool = False, no_visualization: bool = False) -> bool:
    """便捷函数：完整处理 VisDrone 数据集"""
    module = VisDroneModule()
    result = module.process_complete(input_dir, output_dir, verbose, no_visualization)
    return result['success']

def validate_visdrone(dataset_dir: str, visualize: bool = False) -> bool:
    """便捷函数：验证 VisDrone 数据集"""
    module = VisDroneModule()
    result = module.validate_dataset(dataset_dir, visualize)
    return result['success']
