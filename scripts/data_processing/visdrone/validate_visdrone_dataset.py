#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019 数据集验证脚本
检查转换后数据的完整性和正确性，统计各类别样本数量，验证标注格式

作者: YOLOvision Pro Team
日期: 2024
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import argparse
import logging
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visdrone_validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VisDroneDatasetValidator:
    """VisDrone 数据集验证器"""

    def __init__(self, dataset_dir: str):
        """
        初始化验证器

        Args:
            dataset_dir: 数据集根目录 (包含 data.yaml)
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_yaml_path = self.dataset_dir / "data.yaml"

        # 验证数据集目录
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")

        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.data_yaml_path}")

        # 加载配置
        self.load_config()

        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'total_annotations': 0,
            'missing_labels': 0,
            'empty_labels': 0,
            'invalid_annotations': 0,
            'class_counts': defaultdict(int),
            'bbox_stats': {
                'width_stats': [],
                'height_stats': [],
                'area_stats': []
            },
            'set_stats': {}
        }

    def load_config(self) -> None:
        """加载数据集配置"""
        try:
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            self.class_names = self.config.get('names', [])
            self.num_classes = self.config.get('nc', len(self.class_names))

            logger.info(f"数据集配置加载成功")
            logger.info(f"类别数量: {self.num_classes}")
            logger.info(f"类别名称: {self.class_names}")

        except Exception as e:
            logger.error(f"加载数据集配置失败: {e}")
            raise

    def validate_annotation_format(self, annotation_line: str) -> Tuple[bool, Optional[Tuple]]:
        """
        验证标注格式

        Args:
            annotation_line: 标注行

        Returns:
            (是否有效, 解析结果)
        """
        try:
            parts = annotation_line.strip().split()
            if len(parts) != 5:
                return False, None

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 验证类别ID
            if class_id < 0 or class_id >= self.num_classes:
                return False, None

            # 验证坐标范围 [0, 1]
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                   0 <= width <= 1 and 0 <= height <= 1):
                return False, None

            # 验证边界框有效性
            if width <= 0 or height <= 0:
                return False, None

            return True, (class_id, x_center, y_center, width, height)

        except (ValueError, IndexError):
            return False, None

    def validate_single_file(self, image_path: Path, label_path: Path) -> Dict:
        """
        验证单个文件对

        Args:
            image_path: 图像文件路径
            label_path: 标签文件路径

        Returns:
            验证结果字典
        """
        result = {
            'image_exists': image_path.exists(),
            'label_exists': label_path.exists(),
            'image_readable': False,
            'label_readable': False,
            'annotations': [],
            'invalid_annotations': 0,
            'image_size': None
        }

        # 验证图像文件
        if result['image_exists']:
            try:
                with Image.open(image_path) as img:
                    result['image_size'] = img.size
                    result['image_readable'] = True
            except Exception as e:
                logger.warning(f"无法读取图像 {image_path}: {e}")

        # 验证标签文件
        if result['label_exists']:
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                result['label_readable'] = True

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue

                    is_valid, parsed = self.validate_annotation_format(line)
                    if is_valid:
                        result['annotations'].append(parsed)
                    else:
                        result['invalid_annotations'] += 1
                        logger.warning(f"无效标注 {label_path}:{line_num}: {line}")

            except Exception as e:
                logger.warning(f"无法读取标签 {label_path}: {e}")

        return result

    def validate_dataset_split(self, split_name: str) -> Dict:
        """
        验证数据集分割

        Args:
            split_name: 分割名称 (train/val/test)

        Returns:
            验证结果
        """
        logger.info(f"验证 {split_name} 集...")

        images_dir = self.dataset_dir / "images" / split_name
        labels_dir = self.dataset_dir / "labels" / split_name

        if not images_dir.exists():
            logger.warning(f"{split_name} 集图像目录不存在: {images_dir}")
            return {}

        if not labels_dir.exists():
            logger.warning(f"{split_name} 集标签目录不存在: {labels_dir}")
            return {}

        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(ext))
            image_files.extend(images_dir.glob(ext.upper()))

        split_stats = {
            'total_images': len(image_files),
            'total_labels': 0,
            'total_annotations': 0,
            'missing_labels': 0,
            'empty_labels': 0,
            'invalid_annotations': 0,
            'class_counts': defaultdict(int),
            'bbox_stats': {
                'width_stats': [],
                'height_stats': [],
                'area_stats': []
            }
        }

        # 验证每个文件
        for image_path in image_files:
            label_path = labels_dir / (image_path.stem + '.txt')

            result = self.validate_single_file(image_path, label_path)

            if not result['label_exists']:
                split_stats['missing_labels'] += 1
                continue

            if result['label_readable']:
                split_stats['total_labels'] += 1

                if not result['annotations']:
                    split_stats['empty_labels'] += 1
                else:
                    split_stats['total_annotations'] += len(result['annotations'])

                    # 统计类别和边界框信息
                    for class_id, x_center, y_center, width, height in result['annotations']:
                        split_stats['class_counts'][class_id] += 1
                        split_stats['bbox_stats']['width_stats'].append(width)
                        split_stats['bbox_stats']['height_stats'].append(height)
                        split_stats['bbox_stats']['area_stats'].append(width * height)

                split_stats['invalid_annotations'] += result['invalid_annotations']

        logger.info(f"{split_name} 集验证完成")
        return split_stats

    def validate_dataset(self) -> None:
        """验证整个数据集"""
        logger.info("开始验证 VisDrone 数据集...")
        logger.info(f"数据集目录: {self.dataset_dir}")

        # 验证各个分割
        splits = ['train', 'val', 'test']
        for split in splits:
            split_stats = self.validate_dataset_split(split)
            if split_stats:
                self.stats['set_stats'][split] = split_stats

                # 累加到总统计
                self.stats['total_images'] += split_stats['total_images']
                self.stats['total_labels'] += split_stats['total_labels']
                self.stats['total_annotations'] += split_stats['total_annotations']
                self.stats['missing_labels'] += split_stats['missing_labels']
                self.stats['empty_labels'] += split_stats['empty_labels']
                self.stats['invalid_annotations'] += split_stats['invalid_annotations']

                # 合并类别统计
                for class_id, count in split_stats['class_counts'].items():
                    self.stats['class_counts'][class_id] += count

                # 合并边界框统计
                for key in ['width_stats', 'height_stats', 'area_stats']:
                    self.stats['bbox_stats'][key].extend(split_stats['bbox_stats'][key])

        self.print_validation_report()
        logger.info("数据集验证完成!")

    def print_validation_report(self) -> None:
        """打印验证报告"""
        logger.info("\n" + "="*60)
        logger.info("VisDrone 数据集验证报告")
        logger.info("="*60)

        # 总体统计
        logger.info(f"总图像数量: {self.stats['total_images']}")
        logger.info(f"总标签数量: {self.stats['total_labels']}")
        logger.info(f"总标注数量: {self.stats['total_annotations']}")
        logger.info(f"缺失标签: {self.stats['missing_labels']}")
        logger.info(f"空标签文件: {self.stats['empty_labels']}")
        logger.info(f"无效标注: {self.stats['invalid_annotations']}")

        # 各分割统计
        logger.info("\n各分割统计:")
        for split, stats in self.stats['set_stats'].items():
            logger.info(f"  {split}:")
            logger.info(f"    图像: {stats['total_images']}")
            logger.info(f"    标注: {stats['total_annotations']}")
            logger.info(f"    缺失标签: {stats['missing_labels']}")

        # 类别统计
        logger.info("\n各类别统计:")
        total_objects = sum(self.stats['class_counts'].values())
        for class_id in range(self.num_classes):
            count = self.stats['class_counts'][class_id]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            logger.info(f"  {class_id}: {class_name} - {count} ({percentage:.1f}%)")

        # 边界框统计
        if self.stats['bbox_stats']['width_stats']:
            import numpy as np
            widths = np.array(self.stats['bbox_stats']['width_stats'])
            heights = np.array(self.stats['bbox_stats']['height_stats'])
            areas = np.array(self.stats['bbox_stats']['area_stats'])

            logger.info("\n边界框统计:")
            logger.info(f"  宽度 - 均值: {widths.mean():.4f}, 中位数: {np.median(widths):.4f}, 标准差: {widths.std():.4f}")
            logger.info(f"  高度 - 均值: {heights.mean():.4f}, 中位数: {np.median(heights):.4f}, 标准差: {heights.std():.4f}")
            logger.info(f"  面积 - 均值: {areas.mean():.4f}, 中位数: {np.median(areas):.4f}, 标准差: {areas.std():.4f}")

        logger.info("="*60)

    def create_visualization(self, output_dir: Optional[str] = None) -> None:
        """
        创建数据集可视化图表

        Args:
            output_dir: 输出目录，如果为 None 则不保存
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('VisDrone 数据集统计可视化', fontsize=16, fontweight='bold')

            # 1. 类别分布
            if self.stats['class_counts']:
                class_ids = list(range(self.num_classes))
                counts = [self.stats['class_counts'][i] for i in class_ids]
                class_labels = [self.class_names[i] if i < len(self.class_names) else f"class_{i}" for i in class_ids]

                axes[0, 0].bar(class_labels, counts, color='skyblue', alpha=0.7)
                axes[0, 0].set_title('类别分布')
                axes[0, 0].set_xlabel('类别')
                axes[0, 0].set_ylabel('数量')
                axes[0, 0].tick_params(axis='x', rotation=45)

            # 2. 各分割数据量
            if self.stats['set_stats']:
                splits = list(self.stats['set_stats'].keys())
                image_counts = [self.stats['set_stats'][split]['total_images'] for split in splits]
                annotation_counts = [self.stats['set_stats'][split]['total_annotations'] for split in splits]

                x = np.arange(len(splits))
                width = 0.35

                axes[0, 1].bar(x - width/2, image_counts, width, label='图像数量', alpha=0.7)
                axes[0, 1].bar(x + width/2, annotation_counts, width, label='标注数量', alpha=0.7)
                axes[0, 1].set_title('各分割数据量')
                axes[0, 1].set_xlabel('数据集分割')
                axes[0, 1].set_ylabel('数量')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(splits)
                axes[0, 1].legend()

            # 3. 边界框宽度分布
            if self.stats['bbox_stats']['width_stats']:
                widths = np.array(self.stats['bbox_stats']['width_stats'])
                axes[1, 0].hist(widths, bins=50, alpha=0.7, color='lightgreen')
                axes[1, 0].set_title('边界框宽度分布')
                axes[1, 0].set_xlabel('宽度 (归一化)')
                axes[1, 0].set_ylabel('频次')
                axes[1, 0].axvline(widths.mean(), color='red', linestyle='--', label=f'均值: {widths.mean():.3f}')
                axes[1, 0].legend()

            # 4. 边界框高度分布
            if self.stats['bbox_stats']['height_stats']:
                heights = np.array(self.stats['bbox_stats']['height_stats'])
                axes[1, 1].hist(heights, bins=50, alpha=0.7, color='lightcoral')
                axes[1, 1].set_title('边界框高度分布')
                axes[1, 1].set_xlabel('高度 (归一化)')
                axes[1, 1].set_ylabel('频次')
                axes[1, 1].axvline(heights.mean(), color='red', linestyle='--', label=f'均值: {heights.mean():.3f}')
                axes[1, 1].legend()

            plt.tight_layout()

            if output_dir:
                output_path = Path(output_dir) / "dataset_statistics.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"可视化图表已保存: {output_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib 或 seaborn 未安装，跳过可视化")
        except Exception as e:
            logger.error(f"创建可视化时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="验证 VisDrone YOLO 格式数据集的完整性和正确性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python validate_visdrone_dataset.py --dataset data/visdrone_yolo
  python validate_visdrone_dataset.py -d data/visdrone_yolo --visualize --output-dir outputs/validation
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='数据集根目录路径 (包含 data.yaml 文件)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='生成可视化图表'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='输出目录 (用于保存可视化图表)'
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
        # 创建验证器并执行验证
        validator = VisDroneDatasetValidator(args.dataset)
        validator.validate_dataset()

        # 生成可视化
        if args.visualize:
            validator.create_visualization(args.output_dir)

        logger.info(f"\n数据集验证完成!")

    except Exception as e:
        logger.error(f"数据集验证过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
