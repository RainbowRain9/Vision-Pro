#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019 数据集划分脚本
将转换后的 VisDrone YOLO 格式数据集按 8:1:1 比例划分为训练集、验证集和测试集

作者: YOLOvision Pro Team
日期: 2024
"""

import os
import sys
import glob
import random
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visdrone_split.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VisDroneDatasetSplitter:
    """VisDrone 数据集划分器"""

    def __init__(self, input_dir: str, output_dir: str, train_ratio: float = 0.8,
                 val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        初始化数据集划分器

        Args:
            input_dir: 输入目录 (包含 images_temp 和 labels_temp)
            output_dir: 输出目录 (将创建标准 YOLO 数据集结构)
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError("训练、验证和测试集的比例之和必须为 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 输入目录
        self.input_images_dir = self.input_dir / "images_temp"
        self.input_labels_dir = self.input_dir / "labels_temp"

        # 验证输入目录
        if not self.input_images_dir.exists():
            raise FileNotFoundError(f"输入图像目录不存在: {self.input_images_dir}")
        if not self.input_labels_dir.exists():
            raise FileNotFoundError(f"输入标签目录不存在: {self.input_labels_dir}")

        # 创建输出目录结构
        self.create_output_structure()

        # 统计信息
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0
        }

    def create_output_structure(self) -> None:
        """创建输出目录结构"""
        sets = ['train', 'val', 'test']
        for set_name in sets:
            (self.output_dir / "images" / set_name).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / set_name).mkdir(parents=True, exist_ok=True)

        logger.info(f"输出目录结构已创建: {self.output_dir}")

    def get_image_files(self) -> List[Path]:
        """获取所有图像文件"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.input_images_dir.glob(ext))
            image_files.extend(self.input_images_dir.glob(ext.upper()))

        return sorted(image_files)

    def split_files(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """
        划分文件列表

        Args:
            image_files: 图像文件列表

        Returns:
            划分后的文件字典
        """
        # 随机打乱
        random.shuffle(image_files)

        num_images = len(image_files)
        num_train = int(num_images * self.train_ratio)
        num_val = int(num_images * self.val_ratio)

        # 划分文件列表
        train_files = image_files[:num_train]
        val_files = image_files[num_train:num_train + num_val]
        test_files = image_files[num_train + num_val:]

        return {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

    def copy_files(self, file_list: List[Path], set_name: str) -> None:
        """
        复制文件到对应的数据集目录

        Args:
            file_list: 文件列表
            set_name: 数据集名称 (train/val/test)
        """
        if not file_list:
            logger.warning(f"{set_name} 集为空")
            return

        for image_path in tqdm(file_list, desc=f"复制 {set_name} 集"):
            try:
                # 复制图像文件
                dst_image_path = self.output_dir / "images" / set_name / image_path.name
                shutil.copy2(image_path, dst_image_path)

                # 复制对应的标签文件
                label_name = image_path.stem + '.txt'
                src_label_path = self.input_labels_dir / label_name
                dst_label_path = self.output_dir / "labels" / set_name / label_name

                if src_label_path.exists():
                    shutil.copy2(src_label_path, dst_label_path)
                else:
                    logger.warning(f"未找到图像 {image_path.name} 对应的标签文件 {label_name}")
                    # 创建空标签文件
                    dst_label_path.touch()

            except Exception as e:
                logger.error(f"复制文件 {image_path} 时出错: {e}")
                continue

        logger.info(f"已复制 {len(file_list)} 个文件到 {set_name} 集")

    def create_yaml_config(self) -> None:
        """创建 YOLO 数据集配置文件"""
        # 读取类别文件
        classes_file = self.input_dir / "classes.txt"
        if not classes_file.exists():
            logger.error(f"类别文件不存在: {classes_file}")
            return

        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

        # 创建配置字典
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }

        # 保存配置文件
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"YOLO 配置文件已创建: {yaml_path}")

    def split_dataset(self) -> None:
        """执行数据集划分"""
        logger.info("开始划分 VisDrone 数据集...")
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"划分比例 - 训练集: {self.train_ratio}, 验证集: {self.val_ratio}, 测试集: {self.test_ratio}")

        # 获取所有图像文件
        image_files = self.get_image_files()
        self.stats['total_images'] = len(image_files)

        if self.stats['total_images'] == 0:
            logger.error("未找到任何图像文件!")
            return

        logger.info(f"找到 {self.stats['total_images']} 个图像文件")

        # 划分文件
        split_files = self.split_files(image_files)

        # 更新统计信息
        self.stats['train_images'] = len(split_files['train'])
        self.stats['val_images'] = len(split_files['val'])
        self.stats['test_images'] = len(split_files['test'])

        # 复制文件到各个数据集
        for set_name, file_list in split_files.items():
            if file_list or set_name != 'test':  # 总是创建 train 和 val，test 可以为空
                self.copy_files(file_list, set_name)

        # 创建 YAML 配置文件
        self.create_yaml_config()

        # 清理临时目录
        self.cleanup_temp_dirs()

        # 打印统计信息
        self.print_statistics()

        logger.info("数据集划分完成!")

    def cleanup_temp_dirs(self) -> None:
        """清理临时目录"""
        try:
            if self.input_images_dir.exists() and not any(self.input_images_dir.iterdir()):
                self.input_images_dir.rmdir()
                logger.info(f"已清理空目录: {self.input_images_dir}")

            if self.input_labels_dir.exists() and not any(self.input_labels_dir.iterdir()):
                self.input_labels_dir.rmdir()
                logger.info(f"已清理空目录: {self.input_labels_dir}")

        except Exception as e:
            logger.warning(f"清理临时目录时出错: {e}")

    def print_statistics(self) -> None:
        """打印划分统计信息"""
        logger.info("\n" + "="*50)
        logger.info("数据集划分统计")
        logger.info("="*50)
        logger.info(f"总图像数量: {self.stats['total_images']}")
        logger.info(f"训练集: {self.stats['train_images']} ({self.stats['train_images']/self.stats['total_images']*100:.1f}%)")
        logger.info(f"验证集: {self.stats['val_images']} ({self.stats['val_images']/self.stats['total_images']*100:.1f}%)")
        logger.info(f"测试集: {self.stats['test_images']} ({self.stats['test_images']/self.stats['total_images']*100:.1f}%)")
        logger.info("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将转换后的 VisDrone YOLO 格式数据集按 8:1:1 比例划分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python split_visdrone_dataset.py --input data/visdrone_yolo --output data/visdrone_yolo
  python split_visdrone_dataset.py -i data/visdrone_yolo -o data/visdrone_yolo --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入目录路径 (包含 images_temp 和 labels_temp 子目录)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径 (将创建标准 YOLO 数据集结构)'
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
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
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

    # 设置随机种子
    random.seed(args.seed)

    try:
        # 创建数据集划分器并执行划分
        splitter = VisDroneDatasetSplitter(
            input_dir=args.input,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )

        splitter.split_dataset()

        logger.info(f"\n数据集划分完成! 输出文件位于: {args.output}")
        logger.info("现在可以使用生成的 data.yaml 文件进行 YOLO 模型训练")

    except Exception as e:
        logger.error(f"数据集划分过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
