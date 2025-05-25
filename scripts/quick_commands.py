#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOvision Pro 快捷命令脚本
提供预设的常用操作组合，简化复杂工作流程

使用方法:
    python scripts/quick_commands.py <preset> [options]

预设命令:
    setup           - 环境检查和初始化
    visdrone-full   - VisDrone 完整处理流程
    visdrone-quick  - VisDrone 快速处理（无可视化）
    check-all       - 完整系统检查
    demo-all        - 运行所有演示

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time

class QuickCommands:
    """快捷命令管理器"""

    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.yolo_tools = self.scripts_dir / "yolo_tools.py"

    def run_command(self, cmd: List[str], description: str = "") -> bool:
        """运行命令并显示进度"""
        if description:
            print(f"\n🔧 {description}")
            print("-" * 50)

        try:
            result = subprocess.run(
                cmd,
                check=False,
                encoding='utf-8',
                errors='ignore',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            success = result.returncode == 0

            # 显示输出
            if result.stdout:
                print(result.stdout)
            if result.stderr and not success:
                print(f"错误信息: {result.stderr}")

            if success:
                print(f"✅ 完成: {description}")
            else:
                print(f"❌ 失败: {description}")

            return success
        except Exception as e:
            print(f"❌ 执行错误: {e}")
            return False

    def setup_environment(self) -> bool:
        """环境检查和初始化"""
        print("🚀 YOLOvision Pro 环境初始化")
        print("=" * 50)

        steps = [
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'validation', 'check'],
                'desc': '简化版环境检查'
            },
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'validation', 'quick'],
                'desc': '快速配置检查'
            }
        ]

        success_count = 0
        for step in steps:
            if self.run_command(step['cmd'], step['desc']):
                success_count += 1

        print(f"\n📊 环境检查完成: {success_count}/{len(steps)} 项通过")
        return success_count >= 1  # 至少一项通过

    def visdrone_full_process(self, input_dir: str = None, output_dir: str = None) -> bool:
        """VisDrone 完整处理流程"""
        print("🚁 VisDrone 完整处理流程")
        print("=" * 50)

        # 默认路径
        if not input_dir:
            input_dir = "data/VisDrone2019-DET-train"
        if not output_dir:
            output_dir = "data/visdrone_yolo"

        # 检查输入目录
        if not Path(input_dir).exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            print("💡 请确保 VisDrone2019 数据集已下载到指定位置")
            return False

        cmd = [
            sys.executable, str(self.yolo_tools),
            'visdrone', 'process',
            '--input', input_dir,
            '--output', output_dir,
            '--verbose'
        ]

        return self.run_command(cmd, f"VisDrone 完整处理: {input_dir} → {output_dir}")

    def visdrone_quick_process(self, input_dir: str = None, output_dir: str = None) -> bool:
        """VisDrone 快速处理（无可视化）"""
        print("⚡ VisDrone 快速处理流程")
        print("=" * 50)

        # 默认路径
        if not input_dir:
            input_dir = "data/VisDrone2019-DET-train"
        if not output_dir:
            output_dir = "data/visdrone_yolo"

        # 检查输入目录
        if not Path(input_dir).exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            return False

        cmd = [
            sys.executable, str(self.yolo_tools),
            'visdrone', 'process',
            '--input', input_dir,
            '--output', output_dir,
            '--no-visualization'
        ]

        return self.run_command(cmd, f"VisDrone 快速处理: {input_dir} → {output_dir}")

    def check_all_systems(self) -> bool:
        """完整系统检查"""
        print("🔍 完整系统检查")
        print("=" * 50)

        checks = [
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'validation', 'check'],
                'desc': '基础环境检查'
            },
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'validation', 'full'],
                'desc': '完整配置验证'
            },
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'validation', 'test-visdrone'],
                'desc': 'VisDrone 功能测试'
            }
        ]

        success_count = 0
        for check in checks:
            if self.run_command(check['cmd'], check['desc']):
                success_count += 1
            time.sleep(1)  # 短暂延迟

        print(f"\n📊 系统检查完成: {success_count}/{len(checks)} 项通过")
        return success_count >= 2  # 至少两项通过

    def run_all_demos(self) -> bool:
        """运行所有演示"""
        print("🎭 运行所有演示")
        print("=" * 50)

        demos = [
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'visdrone', 'demo'],
                'desc': 'VisDrone 处理演示'
            },
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'demo', 'drone-yolo'],
                'desc': 'Drone-YOLO 概念演示'
            },
            {
                'cmd': [sys.executable, str(self.yolo_tools), 'viz', 'architecture'],
                'desc': 'Drone-YOLO 架构可视化'
            }
        ]

        success_count = 0
        for demo in demos:
            if self.run_command(demo['cmd'], demo['desc']):
                success_count += 1
            time.sleep(1)

        print(f"\n📊 演示完成: {success_count}/{len(demos)} 项成功")
        return success_count >= 1

    def print_help(self):
        """打印帮助信息"""
        print("⚡ YOLOvision Pro 快捷命令")
        print("=" * 50)
        print("使用方法: python scripts/quick_commands.py <preset> [options]")
        print()

        presets = {
            'setup': '环境检查和初始化',
            'visdrone-full': 'VisDrone 完整处理流程（包含可视化）',
            'visdrone-quick': 'VisDrone 快速处理流程（无可视化）',
            'check-all': '完整系统检查',
            'demo-all': '运行所有演示'
        }

        print("📋 可用预设:")
        for preset, desc in presets.items():
            print(f"   {preset:<15} - {desc}")

        print()
        print("💡 使用示例:")
        print("   python scripts/quick_commands.py setup")
        print("   python scripts/quick_commands.py visdrone-full")
        print("   python scripts/quick_commands.py visdrone-full --input data/custom --output data/output")
        print("   python scripts/quick_commands.py check-all")
        print()
        print("🔗 相关工具:")
        print("   python scripts/yolo_tools.py --help  # 查看详细工具")

    def run(self, argv: List[str]) -> int:
        """运行快捷命令"""
        parser = argparse.ArgumentParser(description="YOLOvision Pro 快捷命令")
        parser.add_argument('preset', nargs='?', help='预设命令名称')
        parser.add_argument('--input', '-i', help='输入目录路径')
        parser.add_argument('--output', '-o', help='输出目录路径')

        if len(argv) < 2:
            self.print_help()
            return 0

        args = parser.parse_args(argv[1:])

        if not args.preset:
            self.print_help()
            return 0

        # 执行对应的预设命令
        if args.preset == 'setup':
            success = self.setup_environment()
        elif args.preset == 'visdrone-full':
            success = self.visdrone_full_process(args.input, args.output)
        elif args.preset == 'visdrone-quick':
            success = self.visdrone_quick_process(args.input, args.output)
        elif args.preset == 'check-all':
            success = self.check_all_systems()
        elif args.preset == 'demo-all':
            success = self.run_all_demos()
        else:
            print(f"❌ 未知预设: {args.preset}")
            self.print_help()
            return 1

        return 0 if success else 1

def main():
    """主函数"""
    quick_commands = QuickCommands()
    return quick_commands.run(sys.argv)

if __name__ == "__main__":
    sys.exit(main())
