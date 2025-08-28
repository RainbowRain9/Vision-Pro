#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 标注器演示脚本
展示 YOLOvision Pro 中各种标注器的功能和效果
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import logging
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts/modules"))

try:
    from ultralytics import YOLO
    import supervision as sv
    from supervision_annotators import AnnotatorManager, AnnotatorType, AnnotatorPresets
    from supervision_wrapper import SupervisionWrapper
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需依赖: pip install ultralytics supervision")
    sys.exit(1)


class AnnotatorDemo:
    """标注器演示类"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        初始化演示
        
        Args:
            model_path: YOLO模型路径
        """
        self.model_path = model_path
        self.model = None
        self.annotator_manager = None
        self.supervision_wrapper = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize()
    
    def _initialize(self):
        """初始化模型和标注器"""
        try:
            # 加载YOLO模型
            self.logger.info(f"加载YOLO模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 初始化标注器管理器
            config_path = project_root / "assets/configs/annotator_config.yaml"
            self.annotator_manager = AnnotatorManager(str(config_path))
            
            # 初始化Supervision包装器
            self.supervision_wrapper = SupervisionWrapper(
                class_names=list(self.model.names.values()),
                annotator_config_path=str(config_path)
            )
            
            self.logger.info("初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def demo_single_image(self, image_path: str, output_dir: str = "outputs/annotator_demo"):
        """
        单张图像标注演示
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
        """
        if not os.path.exists(image_path):
            self.logger.error(f"图像文件不存在: {image_path}")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"无法读取图像: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 运行检测
        self.logger.info("运行目标检测...")
        results = self.model.predict(image_rgb, conf=0.25, iou=0.45)
        detections = sv.Detections.from_ultralytics(results[0])
        
        if len(detections.xyxy) == 0:
            self.logger.warning("未检测到任何目标")
            return
        
        # 生成标签
        labels = [
            f"{self.model.names[int(class_id)]}: {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        self.logger.info(f"检测到 {len(detections.xyxy)} 个目标")
        
        # 演示不同的预设配置
        presets = ['basic', 'detailed', 'privacy', 'analysis']
        
        for preset in presets:
            self.logger.info(f"演示预设: {preset}")
            
            # 设置预设
            self.annotator_manager.set_preset(preset)
            
            # 应用标注
            annotated_image = self.annotator_manager.annotate_image(
                image_rgb.copy(), detections, labels
            )
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{preset}_annotated.jpg")
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, annotated_bgr)
            self.logger.info(f"保存结果: {output_path}")
        
        # 创建对比视图
        self.logger.info("创建对比视图...")
        self.annotator_manager.set_preset('detailed')
        comparison = self.annotator_manager.create_comparison_view(image_rgb, detections, labels)
        
        comparison_path = os.path.join(output_dir, "comparison_view.jpg")
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(comparison_path, comparison_bgr)
        self.logger.info(f"保存对比视图: {comparison_path}")
    
    def demo_individual_annotators(self, image_path: str, output_dir: str = "outputs/individual_annotators"):
        """
        单个标注器演示
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
        """
        if not os.path.exists(image_path):
            self.logger.error(f"图像文件不存在: {image_path}")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 运行检测
        results = self.model.predict(image_rgb, conf=0.25, iou=0.45)
        detections = sv.Detections.from_ultralytics(results[0])
        
        if len(detections.xyxy) == 0:
            self.logger.warning("未检测到任何目标")
            return
        
        # 生成标签
        labels = [
            f"{self.model.names[int(class_id)]}: {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        # 测试每个标注器
        annotator_types = [
            AnnotatorType.BOX,
            AnnotatorType.LABEL,
            AnnotatorType.POLYGON,
            AnnotatorType.BLUR,
            AnnotatorType.PIXELATE
        ]
        
        for annotator_type in annotator_types:
            self.logger.info(f"测试标注器: {annotator_type.value}")
            
            try:
                # 只启用当前标注器
                self.annotator_manager.enabled_annotators.clear()
                self.annotator_manager.enable_annotator(annotator_type)
                
                # 应用标注
                annotated_image = self.annotator_manager.annotate_image(
                    image_rgb.copy(), detections, labels
                )
                
                # 保存结果
                output_path = os.path.join(output_dir, f"{annotator_type.value}_only.jpg")
                annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, annotated_bgr)
                self.logger.info(f"保存 {annotator_type.value} 结果: {output_path}")
                
            except Exception as e:
                self.logger.error(f"测试标注器 {annotator_type.value} 失败: {e}")
    
    def demo_heatmap(self, video_path: str, output_dir: str = "outputs/heatmap_demo", max_frames: int = 100):
        """
        热力图演示（需要视频输入）
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            max_frames: 最大处理帧数
        """
        if not os.path.exists(video_path):
            self.logger.error(f"视频文件不存在: {video_path}")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频: {video_path}")
            return
        
        # 启用热力图标注器
        self.annotator_manager.set_preset('analysis')
        
        frame_count = 0
        heatmap_frames = []
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 运行检测
            results = self.model.predict(frame_rgb, conf=0.25, iou=0.45, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            
            if len(detections.xyxy) > 0:
                # 应用标注（包含热力图）
                annotated_frame = self.annotator_manager.annotate_image(
                    frame_rgb.copy(), detections
                )
                heatmap_frames.append(annotated_frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                self.logger.info(f"处理帧: {frame_count}/{max_frames}")
        
        cap.release()
        
        # 保存一些关键帧
        if heatmap_frames:
            for i, frame in enumerate(heatmap_frames[::10]):  # 每10帧保存一张
                output_path = os.path.join(output_dir, f"heatmap_frame_{i:03d}.jpg")
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, frame_bgr)
            
            self.logger.info(f"保存了 {len(heatmap_frames[::10])} 张热力图帧")
    
    def print_annotator_info(self):
        """打印标注器信息"""
        info = self.annotator_manager.get_annotator_info()
        
        print("\n" + "="*50)
        print("标注器信息")
        print("="*50)
        print(f"可用标注器: {info['available_annotators']}")
        print(f"已启用标注器: {info['enabled_annotators']}")
        print(f"总数: {info['total_annotators']}")
        print(f"已启用数量: {info['enabled_count']}")
        print(f"可用预设: {info['presets']}")
        
        # 性能统计
        perf_stats = self.annotator_manager.get_performance_stats()
        print(f"\n性能统计:")
        print(f"内存使用估算: {perf_stats['memory_usage_estimate']}")
        print(f"推荐最大FPS: {perf_stats['recommended_max_fps']}")
        print("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Supervision 标注器演示")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--video", type=str, help="输入视频路径（用于热力图演示）")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO模型路径")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--demo-type", type=str, choices=["presets", "individual", "heatmap", "all"], 
                       default="all", help="演示类型")
    
    args = parser.parse_args()
    
    try:
        # 创建演示实例
        demo = AnnotatorDemo(args.model)
        
        # 打印标注器信息
        demo.print_annotator_info()
        
        if args.image:
            if args.demo_type in ["presets", "all"]:
                print("\n运行预设演示...")
                demo.demo_single_image(args.image, os.path.join(args.output, "presets"))
            
            if args.demo_type in ["individual", "all"]:
                print("\n运行单个标注器演示...")
                demo.demo_individual_annotators(args.image, os.path.join(args.output, "individual"))
        
        if args.video and args.demo_type in ["heatmap", "all"]:
            print("\n运行热力图演示...")
            demo.demo_heatmap(args.video, os.path.join(args.output, "heatmap"))
        
        if not args.image and not args.video:
            print("\n请提供 --image 或 --video 参数")
            print("示例:")
            print("  python supervision_annotators_demo.py --image test.jpg")
            print("  python supervision_annotators_demo.py --video test.mp4 --demo-type heatmap")
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
