#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019 数据集转换脚本
将 VisDrone 格式标注转换为 YOLO 格式

VisDrone 标注格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
YOLO 标注格式: <class_id> <x_center> <y_center> <width> <height> (归一化坐标)

作者: YOLOvision Pro Team
日期: 2024
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import argparse
import logging
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visdrone_conversion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VisDroneConverter:
    """VisDrone 数据集转换器"""

    # VisDrone 类别映射 (原始类别ID -> YOLO类别ID)
    # 0: ignored regions (过滤掉)
    # 1-10: 有效类别映射到 0-9
    CATEGORY_MAPPING = {
        0: -1,  # ignored regions - 将被过滤
        1: 0,   # pedestrian -> 0
        2: 1,   # people -> 1
        3: 2,   # bicycle -> 2
        4: 3,   # car -> 3
        5: 4,   # van -> 4
        6: 5,   # truck -> 5
        7: 6,   # tricycle -> 6
        8: 7,   # awning-tricycle -> 7
        9: 8,   # bus -> 8
        10: 9   # motor -> 9
    }

    # 类别名称
    CLASS_NAMES = [
        'pedestrian',      # 行人
        'people',          # 人群
        'bicycle',         # 自行车
        'car',             # 汽车
        'van',             # 面包车
        'truck',           # 卡车
        'tricycle',        # 三轮车
        'awning-tricycle', # 遮阳三轮车
        'bus',             # 公交车
        'motor'            # 摩托车
    ]

    def __init__(self, visdrone_root: str, output_root: str):
        """
        初始化转换器

        Args:
            visdrone_root: VisDrone 数据集根目录
            output_root: 输出目录根路径
        """
        self.visdrone_root = Path(visdrone_root)
        self.output_root = Path(output_root)

        # 验证输入目录
        if not self.visdrone_root.exists():
            raise FileNotFoundError(f"VisDrone 数据集目录不存在: {visdrone_root}")

        self.images_dir = self.visdrone_root / "images"
        self.annotations_dir = self.visdrone_root / "annotations"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"标注目录不存在: {self.annotations_dir}")

        # 创建输出目录
        self.output_images_dir = self.output_root / "images_temp"
        self.output_labels_dir = self.output_root / "labels_temp"

        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'filtered_annotations': 0,
            'class_counts': {i: 0 for i in range(10)}
        }

    def parse_visdrone_annotation(self, annotation_line: str) -> Optional[Tuple[int, int, int, int, int]]:
        """
        解析 VisDrone 标注行

        Args:
            annotation_line: 标注行字符串

        Returns:
            (bbox_left, bbox_top, bbox_width, bbox_height, category) 或 None
        """
        try:
            parts = annotation_line.strip().split(',')
            if len(parts) < 6:
                return None

            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            # score = int(parts[4])  # 暂时不使用
            category = int(parts[5])
            # truncation = int(parts[6])  # 暂时不使用
            # occlusion = int(parts[7])   # 暂时不使用

            return bbox_left, bbox_top, bbox_width, bbox_height, category

        except (ValueError, IndexError) as e:
            logger.warning(f"解析标注行失败: {annotation_line.strip()}, 错误: {e}")
            return None

    def convert_bbox_to_yolo(self, bbox_left: int, bbox_top: int, bbox_width: int,
                           bbox_height: int, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        将 VisDrone 边界框格式转换为 YOLO 格式

        Args:
            bbox_left, bbox_top, bbox_width, bbox_height: VisDrone 边界框
            img_width, img_height: 图像尺寸

        Returns:
            (x_center, y_center, width, height) 归一化坐标
        """
        # 计算中心点坐标
        x_center = bbox_left + bbox_width / 2
        y_center = bbox_top + bbox_height / 2

        # 归一化坐标
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = bbox_width / img_width
        height_norm = bbox_height / img_height

        # 确保坐标在 [0, 1] 范围内
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))

        return x_center_norm, y_center_norm, width_norm, height_norm

    def get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """
        获取图像尺寸

        Args:
            image_path: 图像文件路径

        Returns:
            (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.error(f"无法读取图像 {image_path}: {e}")
            raise

    def convert_single_annotation(self, annotation_path: Path, image_path: Path) -> List[str]:
        """
        转换单个标注文件

        Args:
            annotation_path: VisDrone 标注文件路径
            image_path: 对应的图像文件路径

        Returns:
            YOLO 格式的标注行列表
        """
        # 获取图像尺寸
        img_width, img_height = self.get_image_size(image_path)

        yolo_annotations = []

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                self.stats['total_annotations'] += 1

                # 解析 VisDrone 标注
                parsed = self.parse_visdrone_annotation(line)
                if parsed is None:
                    continue

                bbox_left, bbox_top, bbox_width, bbox_height, category = parsed

                # 过滤 ignored regions (category = 0)
                if category == 0:
                    self.stats['filtered_annotations'] += 1
                    continue

                # 检查类别是否有效
                if category not in self.CATEGORY_MAPPING:
                    logger.warning(f"未知类别 {category} 在文件 {annotation_path}")
                    continue

                # 映射到 YOLO 类别
                yolo_class = self.CATEGORY_MAPPING[category]
                if yolo_class == -1:  # 被过滤的类别
                    self.stats['filtered_annotations'] += 1
                    continue

                # 检查边界框有效性
                if bbox_width <= 0 or bbox_height <= 0:
                    logger.warning(f"无效边界框尺寸: w={bbox_width}, h={bbox_height} 在文件 {annotation_path}")
                    continue

                # 转换为 YOLO 格式
                x_center, y_center, width, height = self.convert_bbox_to_yolo(
                    bbox_left, bbox_top, bbox_width, bbox_height, img_width, img_height
                )

                # 创建 YOLO 标注行
                yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_annotations.append(yolo_line)

                # 更新统计
                self.stats['valid_annotations'] += 1
                self.stats['class_counts'][yolo_class] += 1

        except Exception as e:
            logger.error(f"处理标注文件 {annotation_path} 时出错: {e}")
            raise

        return yolo_annotations

    def convert_dataset(self) -> None:
        """转换整个数据集"""
        logger.info("开始转换 VisDrone 数据集...")
        logger.info(f"输入目录: {self.visdrone_root}")
        logger.info(f"输出目录: {self.output_root}")

        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(ext))
            image_files.extend(self.images_dir.glob(ext.upper()))

        self.stats['total_images'] = len(image_files)
        logger.info(f"找到 {self.stats['total_images']} 个图像文件")

        if self.stats['total_images'] == 0:
            logger.warning("未找到任何图像文件!")
            return

        # 处理每个图像及其对应的标注
        for image_path in tqdm(image_files, desc="转换进度"):
            try:
                # 构建对应的标注文件路径
                annotation_name = image_path.stem + '.txt'
                annotation_path = self.annotations_dir / annotation_name

                if not annotation_path.exists():
                    logger.warning(f"未找到图像 {image_path.name} 对应的标注文件 {annotation_name}")
                    continue

                # 转换标注
                yolo_annotations = self.convert_single_annotation(annotation_path, image_path)

                # 复制图像文件
                output_image_path = self.output_images_dir / image_path.name
                shutil.copy2(image_path, output_image_path)

                # 保存 YOLO 标注文件
                output_annotation_path = self.output_labels_dir / annotation_name
                with open(output_annotation_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_annotations))
                    if yolo_annotations:  # 如果有标注，添加最后的换行符
                        f.write('\n')

                self.stats['processed_images'] += 1

            except Exception as e:
                logger.error(f"处理图像 {image_path} 时出错: {e}")
                continue

        self.print_statistics()
        logger.info("数据集转换完成!")

    def print_statistics(self) -> None:
        """打印转换统计信息"""
        logger.info("\n" + "="*50)
        logger.info("转换统计信息")
        logger.info("="*50)
        logger.info(f"总图像数量: {self.stats['total_images']}")
        logger.info(f"成功处理图像: {self.stats['processed_images']}")
        logger.info(f"总标注数量: {self.stats['total_annotations']}")
        logger.info(f"有效标注数量: {self.stats['valid_annotations']}")
        logger.info(f"过滤标注数量: {self.stats['filtered_annotations']}")

        logger.info("\n各类别统计:")
        for class_id, count in self.stats['class_counts'].items():
            class_name = self.CLASS_NAMES[class_id]
            logger.info(f"  {class_id}: {class_name} - {count} 个")

        logger.info("="*50)


def create_classes_file(output_dir: Path, class_names: List[str]) -> None:
    """
    创建类别文件

    Args:
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    classes_file = output_dir / "classes.txt"
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    logger.info(f"类别文件已创建: {classes_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将 VisDrone2019 数据集转换为 YOLO 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python convert_visdrone.py --input data/VisDrone2019-DET-train --output data/visdrone_yolo
  python convert_visdrone.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo --verbose
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='VisDrone 数据集根目录路径 (包含 images 和 annotations 子目录)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径 (将创建 images_temp 和 labels_temp 子目录)'
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
        # 创建转换器并执行转换
        converter = VisDroneConverter(args.input, args.output)
        converter.convert_dataset()

        # 创建类别文件
        create_classes_file(Path(args.output), VisDroneConverter.CLASS_NAMES)

        logger.info(f"\n转换完成! 输出文件位于: {args.output}")
        logger.info("接下来可以运行数据集划分脚本进行训练集/验证集/测试集划分")

    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()