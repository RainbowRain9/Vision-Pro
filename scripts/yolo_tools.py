#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOvision Pro 统一工具入口
提供统一的命令行接口来调用各种功能模块

使用方法:
    python scripts/yolo_tools.py <command> <subcommand> [options]

命令分类:
    visdrone    - VisDrone 数据集处理
    validation  - 环境验证和检查
    demo        - 演示和测试
    data        - 通用数据处理
    viz         - 可视化工具

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import importlib.util

# 导入安全运行器
try:
    from safe_runner import SafeRunner
    USE_SAFE_RUNNER = True
except ImportError:
    USE_SAFE_RUNNER = False

class YOLOTools:
    """YOLOvision Pro 统一工具管理器"""

    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.commands = self._init_commands()

    def _init_commands(self) -> Dict[str, Dict[str, Any]]:
        """初始化命令配置"""
        return {
            'visdrone': {
                'description': 'VisDrone 数据集处理工具',
                'subcommands': {
                    'convert': {
                        'script': 'data_processing/visdrone/convert_visdrone.py',
                        'description': '转换 VisDrone 格式到 YOLO 格式',
                        'args': ['--input', '--output', '--verbose']
                    },
                    'split': {
                        'script': 'data_processing/visdrone/split_visdrone_dataset.py',
                        'description': '划分数据集 (8:1:1)',
                        'args': ['--input', '--output', '--train-ratio', '--val-ratio', '--test-ratio']
                    },
                    'validate': {
                        'script': 'data_processing/visdrone/validate_visdrone_dataset.py',
                        'description': '验证数据集完整性',
                        'args': ['--dataset', '--visualize', '--output']
                    },
                    'process': {
                        'script': 'data_processing/visdrone/process_visdrone_complete.py',
                        'description': '一键完整处理流程',
                        'args': ['--input', '--output', '--verbose', '--no-visualization']
                    },
                    'demo': {
                        'script': 'data_processing/demos/demo_visdrone_processing.py',
                        'description': '查看 VisDrone 处理演示',
                        'args': []
                    }
                }
            },
            'validation': {
                'description': '环境验证和配置检查',
                'subcommands': {
                    'check': {
                        'script': 'validation/simple_check.py',
                        'description': '简化版环境检查（推荐）',
                        'args': []
                    },
                    'quick': {
                        'script': 'validation/quick_check.py',
                        'description': '快速配置检查',
                        'args': []
                    },
                    'full': {
                        'script': 'validation/verify_local_ultralytics.py',
                        'description': '完整配置验证',
                        'args': []
                    },
                    'test-visdrone': {
                        'script': 'validation/test_visdrone_conversion.py',
                        'description': '测试 VisDrone 转换功能',
                        'args': []
                    }
                }
            },
            'demo': {
                'description': '演示和测试工具',
                'subcommands': {
                    'drone-yolo': {
                        'script': 'demo/drone_yolo_demo.py',
                        'description': 'Drone-YOLO 核心概念演示',
                        'args': []
                    },
                    'test-model': {
                        'script': 'testing/test_drone_yolo.py',
                        'description': '测试 Drone-YOLO 模型',
                        'args': []
                    }
                }
            },
            'data': {
                'description': '通用数据处理工具',
                'subcommands': {
                    'labelme2yolo': {
                        'script': 'data_processing/general/labelme2yolo.py',
                        'description': 'LabelMe 转 YOLO 格式',
                        'args': []
                    },
                    'split': {
                        'script': 'data_processing/general/split_dataset.py',
                        'description': '通用数据集划分',
                        'args': []
                    }
                }
            },
            'viz': {
                'description': '可视化工具',
                'subcommands': {
                    'architecture': {
                        'script': 'visualization/visualize_drone_yolo.py',
                        'description': 'Drone-YOLO 架构可视化',
                        'args': ['--show', '--arch-only', '--perf-only']
                    }
                }
            }
        }

    def print_help(self):
        """打印帮助信息"""
        print("🔧 YOLOvision Pro 统一工具")
        print("=" * 50)
        print("使用方法: python scripts/yolo_tools.py <command> <subcommand> [options]")
        print()

        for cmd, info in self.commands.items():
            print(f"📋 {cmd} - {info['description']}")
            for subcmd, subinfo in info['subcommands'].items():
                print(f"   {subcmd:<15} - {subinfo['description']}")
            print()

        print("💡 使用示例:")
        print("   python scripts/yolo_tools.py visdrone process --input data/VisDrone2019-DET-train --output data/visdrone_yolo")
        print("   python scripts/yolo_tools.py validation check")
        print("   python scripts/yolo_tools.py demo drone-yolo")
        print("   python scripts/yolo_tools.py viz architecture --show")
        print()
        print("📖 获取子命令帮助:")
        print("   python scripts/yolo_tools.py <command> <subcommand> --help")

    def execute_command(self, command: str, subcommand: str, args: List[str]) -> int:
        """执行指定的命令"""
        if command not in self.commands:
            print(f"❌ 未知命令: {command}")
            self.print_help()
            return 1

        if subcommand not in self.commands[command]['subcommands']:
            print(f"❌ 未知子命令: {command} {subcommand}")
            print(f"可用子命令: {', '.join(self.commands[command]['subcommands'].keys())}")
            return 1

        script_path = self.scripts_dir / self.commands[command]['subcommands'][subcommand]['script']

        if not script_path.exists():
            print(f"❌ 脚本文件不存在: {script_path}")
            return 1

        # 构建执行命令
        cmd = [sys.executable, str(script_path)] + args

        print(f"🚀 执行: {command} {subcommand}")
        print(f"📄 脚本: {script_path}")
        print(f"🔧 参数: {' '.join(args) if args else '无'}")
        print("-" * 50)

        try:
            # 使用安全运行器（如果可用）
            if USE_SAFE_RUNNER:
                result = SafeRunner.run_script(str(script_path), args, capture_output=False)
                return result['returncode']
            else:
                # 回退到标准方法，设置编码以避免 Windows 编码问题
                result = subprocess.run(
                    cmd,
                    check=False,
                    encoding='utf-8',
                    errors='replace',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # 显示输出（如果有）
                if result.stdout:
                    print(result.stdout)
                if result.stderr and result.returncode != 0:
                    print(f"错误信息: {result.stderr}")

                return result.returncode
        except Exception as e:
            print(f"[失败] 执行失败: {e}")
            return 1

    def run(self, argv: List[str]) -> int:
        """运行工具"""
        if len(argv) < 3:
            self.print_help()
            return 0

        command = argv[1]
        subcommand = argv[2]
        args = argv[3:]

        return self.execute_command(command, subcommand, args)

def main():
    """主函数"""
    tools = YOLOTools()
    return tools.run(sys.argv)

if __name__ == "__main__":
    sys.exit(main())
