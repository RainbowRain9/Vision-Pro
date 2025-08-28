#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOvision Pro - Supervision 集成演示脚本
演示如何集成 Supervision.roboflow.com 进行增强可视化
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import supervision as sv
    from ultralytics import YOLO
    SUPERVISION_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 Supervision 或 Ultralytics: {e}")
    SUPERVISION_AVAILABLE = False


class SupervisionDemo:
    """Supervision 集成演示类"""
    
    def __init__(self):
        self.model = None
        self.supervision_available = SUPERVISION_AVAILABLE
        
        if self.supervision_available:
            # 初始化 Supervision 组件
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
            self.heat_map_annotator = sv.HeatMapAnnotator()
            self.detection_metrics = sv.DetectionMetrics()
            
            # VisDrone 类别映射
            self.visdrone_classes = [
                "pedestrian", "people", "bicycle", "car", "van",
                "truck", "tricycle", "awning-tricycle", "bus", "motor"
            ]
    
    def load_model(self, model_path: str = "yolov8s.pt"):
        """加载 YOLO 模型"""
        if not self.supervision_available:
            print("错误: Supervision 不可用")
            return False
            
        try:
            self.model = YOLO(model_path)
            print(f"✓ 模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return False
    
    def process_image_with_supervision(self, image_path: str, 
                                     output_dir: str = "results") -> Dict[str, Any]:
        """使用 Supervision 处理图像"""
        if not self.supervision_available or self.model is None:
            print("错误: Supervision 或模型不可用")
            return {}
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"✗ 无法读取图像: {image_path}")
                return {}
            
            # YOLO 检测
            results = self.model.predict(image, conf=0.25, iou=0.45)
            result = results[0]
            
            # 转换为 Supervision Detections
            detections = self._convert_to_supervision_detections(result)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成多种可视化效果
            visualizations = self._create_visualizations(image, detections, output_dir)
            
            # 生成分析报告
            analysis = self._generate_analysis(detections)
            
            print(f"✓ 处理完成: {os.path.basename(image_path)}")
            print(f"  - 检测目标数: {len(detections.xyxy)}")
            print(f"  - 类别数: {len(set(detections.class_id))}")
            print(f"  - 平均置信度: {np.mean(detections.confidence):.3f}")
            
            return {
                'image_path': image_path,
                'detections': detections,
                'visualizations': visualizations,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            return {}
    
    def _convert_to_supervision_detections(self, result) -> sv.Detections:
        """将 YOLO 结果转换为 Supervision Detections"""
        if result.boxes is None:
            return sv.Detections.empty()
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
    
    def _create_visualizations(self, image: np.ndarray, 
                            detections: sv.Detections, 
                            output_dir: str) -> Dict[str, str]:
        """创建多种可视化效果"""
        base_name = Path(output_dir) / "supervision_demo"
        
        visualizations = {}
        
        # 1. 基础边界框标注
        annotated_image = image.copy()
        annotated_image = self.box_annotator.annotate(
            scene=annotated_image, 
            detections=detections
        )
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, 
            detections=detections,
            labels=self._generate_labels(detections)
        )
        
        bbox_path = f"{base_name}_bbox.jpg"
        cv2.imwrite(bbox_path, annotated_image)
        visualizations['bbox'] = bbox_path
        
        # 2. 热力图
        if len(detections.xyxy) > 0:
            heat_map = self.heat_map_annotator.annotate(
                scene=image.copy(), 
                detections=detections
            )
            heatmap_path = f"{base_name}_heatmap.jpg"
            cv2.imwrite(heatmap_path, heat_map)
            visualizations['heatmap'] = heatmap_path
        
        # 3. 置信度分布图
        if len(detections.confidence) > 0:
            conf_plot_path = self._create_confidence_plot(detections, f"{base_name}_confidence.jpg")
            if conf_plot_path:
                visualizations['confidence'] = conf_plot_path
        
        # 4. 类别分布图
        if len(detections.class_id) > 0:
            class_plot_path = self._create_class_distribution_plot(detections, f"{base_name}_classes.jpg")
            if class_plot_path:
                visualizations['class_dist'] = class_plot_path
        
        return visualizations
    
    def _generate_labels(self, detections: sv.Detections) -> List[str]:
        """生成检测标签"""
        labels = []
        for i in range(len(detections.xyxy)):
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = self.visdrone_classes[class_id] if class_id < len(self.visdrone_classes) else f"class_{class_id}"
            labels.append(f"{class_name} {confidence:.2f}")
        return labels
    
    def _create_confidence_plot(self, detections: sv.Detections, 
                               output_path: str) -> str:
        """创建置信度分布图"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 6))
            plt.hist(detections.confidence, bins=20, alpha=0.7, color='blue')
            plt.xlabel('置信度')
            plt.ylabel('数量')
            plt.title('检测置信度分布')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
        except ImportError:
            print("警告: matplotlib 不可用，跳过置信度图")
            return ""
    
    def _create_class_distribution_plot(self, detections: sv.Detections, 
                                      output_path: str) -> str:
        """创建类别分布图"""
        try:
            import matplotlib.pyplot as plt
            
            # 统计类别分布
            class_counts = {}
            for class_id in detections.class_id:
                class_name = self.visdrone_classes[class_id] if class_id < len(self.visdrone_classes) else f"class_{class_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            plt.bar(classes, counts, color='skyblue')
            plt.xlabel('类别')
            plt.ylabel('数量')
            plt.title('检测类别分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
        except ImportError:
            print("警告: matplotlib 不可用，跳过类别分布图")
            return ""
    
    def _generate_analysis(self, detections: sv.Detections) -> Dict[str, Any]:
        """生成分析报告"""
        analysis = {
            'total_detections': len(detections.xyxy),
            'unique_classes': len(set(detections.class_id)),
            'confidence_stats': {},
            'class_distribution': {}
        }
        
        if len(detections.confidence) > 0:
            analysis['confidence_stats'] = {
                'mean': float(np.mean(detections.confidence)),
                'std': float(np.std(detections.confidence)),
                'min': float(np.min(detections.confidence)),
                'max': float(np.max(detections.confidence))
            }
        
        # 类别分布
        for class_id in detections.class_id:
            class_name = self.visdrone_classes[class_id] if class_id < len(self.visdrone_classes) else f"class_{class_id}"
            analysis['class_distribution'][class_name] = analysis['class_distribution'].get(class_name, 0) + 1
        
        return analysis
    
    def demo_with_visdrone(self, visdrone_dir: str = "data/visdrone_yolo"):
        """使用 VisDrone 数据集进行演示"""
        if not self.supervision_available:
            print("错误: Supervision 不可用")
            return
        
        # 查找 VisDrone 数据集
        images_dir = Path(visdrone_dir) / "images" / "train"
        if not images_dir.exists():
            print(f"✗ VisDrone 图像目录不存在: {images_dir}")
            return
        
        # 获取前 5 张图像
        image_files = list(images_dir.glob("*.jpg"))[:5]
        if not image_files:
            print("✗ 未找到图像文件")
            return
        
        print(f"开始 Supervision 演示，处理 {len(image_files)} 张图像...")
        
        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 处理: {image_path.name}")
            
            output_dir = f"outputs/supervision_demo/image_{i+1}"
            result = self.process_image_with_supervision(str(image_path), output_dir)
            
            if result:
                print("  生成文件:")
                for viz_type, file_path in result['visualizations'].items():
                    print(f"    - {viz_type}: {file_path}")
        
        print("\n✓ Supervision 演示完成！")
        print("结果保存在 outputs/supervision_demo/ 目录中")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOvision Pro - Supervision 集成演示")
    parser.add_argument("--model", type=str, default="yolov8s.pt", 
                       help="YOLO 模型路径")
    parser.add_argument("--image", type=str, 
                       help="单张图像路径")
    parser.add_argument("--visdrone", type=str, default="data/visdrone_yolo",
                       help="VisDrone 数据集路径")
    parser.add_argument("--output", type=str, default="outputs/supervision_demo",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = SupervisionDemo()
    
    if not demo.supervision_available:
        print("请先安装 Supervision:")
        print("pip install supervision")
        return
    
    # 加载模型
    if not demo.load_model(args.model):
        return
    
    if args.image:
        # 处理单张图像
        print(f"处理单张图像: {args.image}")
        result = demo.process_image_with_supervision(args.image, args.output)
        
        if result:
            print("\n分析结果:")
            print(f"  总检测数: {result['analysis']['total_detections']}")
            print(f"  类别数: {result['analysis']['unique_classes']}")
            print(f"  平均置信度: {result['analysis']['confidence_stats'].get('mean', 0):.3f}")
            
            print("\n生成的可视化文件:")
            for viz_type, file_path in result['visualizations'].items():
                print(f"  - {viz_type}: {file_path}")
    
    else:
        # 使用 VisDrone 数据集演示
        demo.demo_with_visdrone(args.visdrone)


if __name__ == "__main__":
    main()