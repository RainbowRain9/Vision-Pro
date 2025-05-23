#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019 数据集完整处理脚本
一键完成 VisDrone 数据集的转换、划分和验证

作者: YOLOvision Pro Team
日期: 2024
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visdrone_complete_process.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VisDroneCompleteProcessor:
    """VisDrone 数据集完整处理器"""

    def __init__(self, visdrone_input: str, output_dir: str,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        初始化处理器

        Args:
            visdrone_input: VisDrone 原始数据集目录
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        self.visdrone_input = Path(visdrone_input)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 脚本路径
        self.scripts_dir = Path(__file__).parent
        self.convert_script = self.scripts_dir / "convert_visdrone.py"
        self.split_script = self.scripts_dir / "split_visdrone_dataset.py"
        self.validate_script = self.scripts_dir / "validate_visdrone_dataset.py"

        # 验证脚本存在
        for script in [self.convert_script, self.split_script, self.validate_script]:
            if not script.exists():
                raise FileNotFoundError(f"脚本文件不存在: {script}")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VisDrone 完整处理器初始化完成")
        logger.info(f"输入目录: {self.visdrone_input}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"划分比例: 训练集 {train_ratio}, 验证集 {val_ratio}, 测试集 {test_ratio}")

    def run_command(self, cmd: list, description: str) -> bool:
        """
        运行命令

        Args:
            cmd: 命令列表
            description: 命令描述

        Returns:
            是否成功
        """
        logger.info(f"开始执行: {description}")
        logger.info(f"命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            # 输出标准输出
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"[{description}] {line}")

            logger.info(f"✓ {description} 完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} 失败")
            logger.error(f"返回码: {e.returncode}")

            if e.stdout:
                logger.error("标准输出:")
                for line in e.stdout.strip().split('\n'):
                    logger.error(f"  {line}")

            if e.stderr:
                logger.error("错误输出:")
                for line in e.stderr.strip().split('\n'):
                    logger.error(f"  {line}")

            return False

        except Exception as e:
            logger.error(f"✗ {description} 执行时发生异常: {e}")
            return False

    def step1_convert_dataset(self) -> bool:
        """步骤1: 转换数据集格式"""
        logger.info("\n" + "="*60)
        logger.info("步骤 1/3: 转换 VisDrone 数据集格式")
        logger.info("="*60)

        cmd = [
            sys.executable,
            str(self.convert_script),
            "--input", str(self.visdrone_input),
            "--output", str(self.output_dir),
            "--verbose"
        ]

        return self.run_command(cmd, "VisDrone 格式转换")

    def step2_split_dataset(self) -> bool:
        """步骤2: 划分数据集"""
        logger.info("\n" + "="*60)
        logger.info("步骤 2/3: 划分数据集")
        logger.info("="*60)

        cmd = [
            sys.executable,
            str(self.split_script),
            "--input", str(self.output_dir),
            "--output", str(self.output_dir),
            "--train-ratio", str(self.train_ratio),
            "--val-ratio", str(self.val_ratio),
            "--test-ratio", str(self.test_ratio),
            "--verbose"
        ]

        return self.run_command(cmd, "数据集划分")

    def step3_validate_dataset(self, create_visualization: bool = True) -> bool:
        """步骤3: 验证数据集"""
        logger.info("\n" + "="*60)
        logger.info("步骤 3/3: 验证数据集")
        logger.info("="*60)

        cmd = [
            sys.executable,
            str(self.validate_script),
            "--dataset", str(self.output_dir),
            "--verbose"
        ]

        if create_visualization:
            cmd.extend(["--visualize", "--output-dir", str(self.output_dir)])

        return self.run_command(cmd, "数据集验证")

    def process_complete(self, create_visualization: bool = True) -> bool:
        """执行完整的处理流程"""
        logger.info("开始 VisDrone 数据集完整处理流程...")

        success_count = 0
        total_steps = 3

        # 步骤1: 转换格式
        if self.step1_convert_dataset():
            success_count += 1
        else:
            logger.error("数据集转换失败，停止处理")
            return False

        # 步骤2: 划分数据集
        if self.step2_split_dataset():
            success_count += 1
        else:
            logger.error("数据集划分失败，停止处理")
            return False

        # 步骤3: 验证数据集
        if self.step3_validate_dataset(create_visualization):
            success_count += 1
        else:
            logger.warning("数据集验证失败，但前面步骤已完成")

        # 总结
        logger.info("\n" + "="*60)
        logger.info("处理流程总结")
        logger.info("="*60)
        logger.info(f"完成步骤: {success_count}/{total_steps}")

        if success_count == total_steps:
            logger.info("✓ 所有步骤都已成功完成!")
            logger.info(f"✓ 输出目录: {self.output_dir}")
            logger.info(f"✓ 配置文件: {self.output_dir / 'data.yaml'}")
            logger.info("✓ 现在可以使用生成的数据集进行 YOLO 模型训练")
            return True
        else:
            logger.warning("⚠ 部分步骤未能完成，请检查日志")
            return False

    def print_usage_instructions(self) -> None:
        """打印使用说明"""
        logger.info("\n" + "="*60)
        logger.info("使用说明")
        logger.info("="*60)
        logger.info("数据集处理完成后，您可以:")
        logger.info("")
        logger.info("1. 使用生成的配置文件训练 YOLO 模型:")
        logger.info(f"   yolo train data={self.output_dir / 'data.yaml'} model=yolov8s.pt epochs=100")
        logger.info("")
        logger.info("2. 使用 Drone-YOLO 配置进行训练:")
        logger.info(f"   python train.py --data {self.output_dir / 'data.yaml'} --cfg assets/configs/yolov8s-drone.yaml")
        logger.info("")
        logger.info("3. 查看数据集统计信息:")
        logger.info(f"   python scripts/data_processing/visdrone/validate_visdrone_dataset.py -d {self.output_dir} --visualize")
        logger.info("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="一键完成 VisDrone2019 数据集的转换、划分和验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python process_visdrone_complete.py --input data/VisDrone2019-DET-train --output data/visdrone_yolo
  python process_visdrone_complete.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='VisDrone 原始数据集目录路径 (包含 images 和 annotations 子目录)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例 (默认: 0.1)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例 (默认: 0.1)'
    )

    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='跳过可视化生成'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志信息'
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建处理器并执行完整流程
        processor = VisDroneCompleteProcessor(
            visdrone_input=args.input,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )

        success = processor.process_complete(create_visualization=not args.no_visualization)

        if success:
            processor.print_usage_instructions()
            logger.info("\n🎉 VisDrone 数据集处理完成!")
        else:
            logger.error("\n❌ VisDrone 数据集处理失败!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
