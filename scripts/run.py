#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOvision Pro 简化运行脚本
提供最常用功能的快速访问

使用方法:
    python scripts/run.py                    # 显示交互式菜单
    python scripts/run.py check              # 环境检查
    python scripts/run.py visdrone           # VisDrone 处理
    python scripts/run.py demo               # 运行演示

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import os
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.visdrone_module import VisDroneModule
from modules.validation_module import ValidationModule

class YOLORunner:
    """YOLOvision Pro 简化运行器"""

    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.visdrone = VisDroneModule(self.scripts_dir)
        self.validation = ValidationModule(self.scripts_dir)

    def show_menu(self):
        """显示交互式菜单"""
        print("🚁 YOLOvision Pro 快速运行器")
        print("=" * 50)
        print("选择要执行的操作:")
        print()
        print("1. 🔍 环境检查")
        print("2. 🚁 VisDrone 数据处理")
        print("3. 🎭 运行演示")
        print("4. 📊 可视化")
        print("5. ❓ 帮助")
        print("0. 🚪 退出")
        print()

        while True:
            try:
                choice = input("请选择 (0-5): ").strip()

                if choice == '0':
                    print("👋 再见!")
                    break
                elif choice == '1':
                    self.environment_check_menu()
                elif choice == '2':
                    self.visdrone_menu()
                elif choice == '3':
                    self.demo_menu()
                elif choice == '4':
                    self.visualization_menu()
                elif choice == '5':
                    self.show_help()
                else:
                    print("❌ 无效选择，请重新输入")

            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

    def environment_check_menu(self):
        """环境检查菜单"""
        print("\n🔍 环境检查")
        print("-" * 30)
        print("1. 简化检查（推荐）")
        print("2. 快速检查")
        print("3. 完整验证")
        print("4. 全部检查")
        print("0. 返回主菜单")

        choice = input("请选择: ").strip()

        if choice == '1':
            print("\n🔍 执行简化检查...")
            result = self.validation.simple_check()
            self._print_result(result, "简化检查")
        elif choice == '2':
            print("\n🔍 执行快速检查...")
            result = self.validation.quick_check()
            self._print_result(result, "快速检查")
        elif choice == '3':
            print("\n🔍 执行完整验证...")
            result = self.validation.full_verification()
            self._print_result(result, "完整验证")
        elif choice == '4':
            print("\n🔍 执行全部检查...")
            result = self.validation.run_all_checks()
            self._print_comprehensive_result(result)
        elif choice == '0':
            return
        else:
            print("❌ 无效选择")

    def visdrone_menu(self):
        """VisDrone 处理菜单"""
        print("\n🚁 VisDrone 数据处理")
        print("-" * 30)
        print("1. 一键完整处理")
        print("2. 快速处理（无可视化）")
        print("3. 仅格式转换")
        print("4. 仅数据集划分")
        print("5. 仅数据验证")
        print("0. 返回主菜单")

        choice = input("请选择: ").strip()

        if choice in ['1', '2', '3']:
            input_dir = input("输入目录 [data/VisDrone2019-DET-train]: ").strip()
            if not input_dir:
                input_dir = "data/VisDrone2019-DET-train"

            output_dir = input("输出目录 [data/visdrone_yolo]: ").strip()
            if not output_dir:
                output_dir = "data/visdrone_yolo"

            if choice == '1':
                print(f"\n🚁 执行完整处理: {input_dir} → {output_dir}")
                result = self.visdrone.process_complete(input_dir, output_dir, verbose=True)
                self._print_result(result, "完整处理")
            elif choice == '2':
                print(f"\n⚡ 执行快速处理: {input_dir} → {output_dir}")
                result = self.visdrone.process_complete(input_dir, output_dir, verbose=True, no_visualization=True)
                self._print_result(result, "快速处理")
            elif choice == '3':
                print(f"\n🔄 执行格式转换: {input_dir} → {output_dir}")
                result = self.visdrone.convert_dataset(input_dir, output_dir, verbose=True)
                self._print_result(result, "格式转换")

        elif choice == '4':
            input_dir = input("数据集目录 [data/visdrone_yolo]: ").strip()
            if not input_dir:
                input_dir = "data/visdrone_yolo"

            print(f"\n📊 执行数据集划分: {input_dir}")
            result = self.visdrone.split_dataset(input_dir, input_dir)
            self._print_result(result, "数据集划分")

        elif choice == '5':
            dataset_dir = input("数据集目录 [data/visdrone_yolo]: ").strip()
            if not dataset_dir:
                dataset_dir = "data/visdrone_yolo"

            visualize = input("生成可视化? (y/N): ").strip().lower() == 'y'

            print(f"\n✅ 执行数据验证: {dataset_dir}")
            result = self.visdrone.validate_dataset(dataset_dir, visualize)
            self._print_result(result, "数据验证")

        elif choice == '0':
            return
        else:
            print("❌ 无效选择")

    def demo_menu(self):
        """演示菜单"""
        print("\n🎭 运行演示")
        print("-" * 30)
        print("1. VisDrone 处理演示")
        print("2. Drone-YOLO 概念演示")
        print("3. 模型测试")
        print("0. 返回主菜单")

        choice = input("请选择: ").strip()

        if choice == '1':
            self._run_script("data_processing/demos/demo_visdrone_processing.py", "VisDrone 处理演示")
        elif choice == '2':
            self._run_script("demo/drone_yolo_demo.py", "Drone-YOLO 概念演示")
        elif choice == '3':
            self._run_script("testing/test_drone_yolo.py", "模型测试")
        elif choice == '0':
            return
        else:
            print("❌ 无效选择")

    def visualization_menu(self):
        """可视化菜单"""
        print("\n📊 可视化工具")
        print("-" * 30)
        print("1. Drone-YOLO 架构图")
        print("2. 架构图（显示）")
        print("0. 返回主菜单")

        choice = input("请选择: ").strip()

        if choice == '1':
            self._run_script("visualization/visualize_drone_yolo.py", "架构可视化")
        elif choice == '2':
            self._run_script("visualization/visualize_drone_yolo.py", "架构可视化（显示）", ["--show"])
        elif choice == '0':
            return
        else:
            print("❌ 无效选择")

    def show_help(self):
        """显示帮助信息"""
        print("\n❓ 帮助信息")
        print("-" * 30)
        print("YOLOvision Pro 是一个完整的目标检测解决方案")
        print()
        print("🔍 环境检查: 验证系统配置和依赖")
        print("🚁 VisDrone: 处理 VisDrone2019 数据集")
        print("🎭 演示: 查看功能演示和测试")
        print("📊 可视化: 生成架构图和分析图表")
        print()
        print("💡 建议流程:")
        print("1. 先运行环境检查确保配置正确")
        print("2. 使用 VisDrone 工具处理数据集")
        print("3. 运行演示了解功能特性")
        print()
        print("🔗 更多信息:")
        print("- 详细文档: docs/README.md")
        print("- 脚本说明: scripts/README.md")
        print("- 问题排查: scripts/docs/验证工具说明.md")

    def _run_script(self, script_path: str, description: str, args: list = None):
        """运行脚本"""
        if args is None:
            args = []

        full_path = self.scripts_dir / script_path
        if not full_path.exists():
            print(f"❌ 脚本不存在: {script_path}")
            return

        print(f"\n🚀 运行: {description}")
        print("-" * 30)

        import subprocess
        cmd = [sys.executable, str(full_path)] + args

        try:
            result = subprocess.run(
                cmd,
                check=False,
                encoding='utf-8',
                errors='ignore',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # 显示输出
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(f"错误信息: {result.stderr}")

            if result.returncode == 0:
                print(f"✅ {description} 完成")
            else:
                print(f"❌ {description} 失败")
        except Exception as e:
            print(f"❌ 执行错误: {e}")

    def _print_result(self, result: dict, operation: str):
        """打印操作结果"""
        if result['success']:
            print(f"✅ {operation} 成功完成")
        else:
            print(f"❌ {operation} 失败")
            if 'error' in result:
                print(f"错误: {result['error']}")

    def _print_comprehensive_result(self, result: dict):
        """打印综合检查结果"""
        summary = result['summary']
        print(f"\n📊 检查结果: {summary['passed']}/{summary['total']} 通过")
        print(f"通过率: {summary['pass_rate']:.1%}")

        if result['success']:
            print("✅ 系统状态良好")
        else:
            print("⚠️ 系统存在问题，建议查看详细日志")

def main():
    """主函数"""
    runner = YOLORunner()

    if len(sys.argv) == 1:
        # 交互式菜单
        runner.show_menu()
    else:
        # 命令行模式
        command = sys.argv[1].lower()

        if command == 'check':
            result = runner.validation.simple_check()
            runner._print_result(result, "环境检查")
        elif command == 'visdrone':
            input_dir = sys.argv[2] if len(sys.argv) > 2 else "data/VisDrone2019-DET-train"
            output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/visdrone_yolo"
            result = runner.visdrone.process_complete(input_dir, output_dir, verbose=True)
            runner._print_result(result, "VisDrone 处理")
        elif command == 'demo':
            runner._run_script("data_processing/demos/demo_visdrone_processing.py", "VisDrone 演示")
        else:
            print(f"❌ 未知命令: {command}")
            print("可用命令: check, visdrone, demo")

if __name__ == "__main__":
    main()
