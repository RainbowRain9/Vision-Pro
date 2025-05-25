#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证模块
提供统一的环境验证和配置检查接口

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ValidationModule:
    """验证模块"""

    def __init__(self, scripts_dir: Optional[Path] = None):
        if scripts_dir is None:
            scripts_dir = Path(__file__).parent.parent

        self.scripts_dir = scripts_dir
        self.validation_dir = scripts_dir / "validation"

        # 脚本路径映射
        self.scripts = {
            'simple': self.validation_dir / "simple_check.py",
            'quick': self.validation_dir / "quick_check.py",
            'full': self.validation_dir / "verify_local_ultralytics.py",
            'test_visdrone': self.validation_dir / "test_visdrone_conversion.py"
        }

    def _run_script(self, script_name: str, args: List[str] = None) -> Dict[str, Any]:
        """运行指定脚本"""
        if args is None:
            args = []

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

    def simple_check(self) -> Dict[str, Any]:
        """简化版环境检查"""
        return self._run_script('simple')

    def quick_check(self) -> Dict[str, Any]:
        """快速配置检查"""
        return self._run_script('quick')

    def full_verification(self) -> Dict[str, Any]:
        """完整配置验证"""
        return self._run_script('full')

    def test_visdrone_conversion(self) -> Dict[str, Any]:
        """测试 VisDrone 转换功能"""
        return self._run_script('test_visdrone')

    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        results = {}
        checks = ['simple', 'quick', 'full', 'test_visdrone']

        for check in checks:
            print(f"🔍 运行 {check} 检查...")
            results[check] = self._run_script(check)

        # 计算总体状态
        success_count = sum(1 for result in results.values() if result['success'])
        total_count = len(results)

        return {
            'success': success_count >= total_count // 2,  # 至少一半通过
            'results': results,
            'summary': {
                'passed': success_count,
                'total': total_count,
                'pass_rate': success_count / total_count
            }
        }

    def get_available_checks(self) -> List[str]:
        """获取可用检查列表"""
        return list(self.scripts.keys())

    def check_dependencies(self) -> Dict[str, bool]:
        """检查依赖脚本是否存在"""
        return {name: path.exists() for name, path in self.scripts.items()}

# 便捷函数
def simple_environment_check() -> bool:
    """便捷函数：简化版环境检查"""
    module = ValidationModule()
    result = module.simple_check()
    return result['success']

def quick_configuration_check() -> bool:
    """便捷函数：快速配置检查"""
    module = ValidationModule()
    result = module.quick_check()
    return result['success']

def full_system_verification() -> bool:
    """便捷函数：完整系统验证"""
    module = ValidationModule()
    result = module.full_verification()
    return result['success']

def validate_all_systems() -> Dict[str, Any]:
    """便捷函数：验证所有系统"""
    module = ValidationModule()
    return module.run_all_checks()
