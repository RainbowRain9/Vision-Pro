#!/usr/bin/env python3
"""
增强版YOLOv8训练脚本
集成supervision进行训练监控和增强
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# 导入必要的库
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# 设置项目根目录
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

class EnhancedYOLOTrainer:
    """增强版YOLO训练器，集成supervision功能"""
    
    def __init__(self, model_path="yolov8s.pt", data_yaml="data/visdrone_yolo/data.yaml"):
        """
        初始化训练器
        
        Args:
            model_path: 预训练模型路径或配置文件
            data_yaml: 数据集配置文件路径
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None
        self.dataset_info = None
        
        # 创建输出目录
        self.output_dir = ROOT / "runs" / "train_enhanced" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup(self):
        """设置训练环境"""
        print("=" * 60)
        print("增强版YOLOv8训练 - 集成Supervision")
        print("=" * 60)
        
        # 检查GPU
        if torch.cuda.is_available():
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            self.device = 0
        else:
            print("⚠️ GPU不可用，使用CPU训练")
            self.device = 'cpu'
            
        # 加载数据集信息
        import yaml
        with open(self.data_yaml, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        print(f"\n📊 数据集: {self.data_yaml}")
        print(f"   类别数: {len(self.dataset_info['names'])}")
        print(f"   类别: {', '.join(self.dataset_info['names'])}")
        
        # 初始化模型
        print(f"\n🤖 加载模型: {self.model_path}")
        
        # 判断是配置文件还是权重文件
        if str(self.model_path).endswith('.yaml'):
            # 从配置文件创建新模型
            self.model = YOLO(self.model_path)
            # 加载预训练权重
            if Path("models/yolov8s.pt").exists():
                print("   加载预训练权重: models/yolov8s.pt")
                self.model.load("models/yolov8s.pt")
        else:
            # 直接加载预训练模型
            self.model = YOLO(self.model_path)
            
    def analyze_dataset(self):
        """使用supervision分析数据集"""
        print("\n📈 数据集分析...")
        
        # 读取训练集图像
        train_images = Path(self.data_yaml).parent / "images" / "train"
        train_labels = Path(self.data_yaml).parent / "labels" / "train"
        
        if not train_images.exists():
            print("⚠️ 训练集不存在")
            return
            
        # 统计信息
        image_files = list(train_images.glob("*.jpg"))
        print(f"   训练图像数: {len(image_files)}")
        
        # 分析标注分布
        class_counts = {name: 0 for name in self.dataset_info['names']}
        total_boxes = 0
        
        for img_file in image_files[:100]:  # 分析前100张
            label_file = train_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    total_boxes += len(lines)
                    for line in lines:
                        class_id = int(line.split()[0])
                        class_name = self.dataset_info['names'][class_id]
                        class_counts[class_name] += 1
                        
        print(f"   总标注框数: {total_boxes}")
        print("\n   类别分布 (前100张):")
        for name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"      {name}: {count}")
                
    def train(self, epochs=100, imgsz=640, batch=16):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch: 批次大小
        """
        print(f"\n🚀 开始训练")
        print(f"   轮数: {epochs}")
        print(f"   图像尺寸: {imgsz}")
        print(f"   批次大小: {batch}")
        print(f"   输出目录: {self.output_dir}")
        
        # 训练参数
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            'patience': 20,  # 早停
            'save_period': 10,  # 每10轮保存
            'workers': 4,
            'amp': True if self.device != 'cpu' else False,  # 混合精度训练
            'cache': False,  # 缓存图像到内存
            'verbose': True,
            'plots': True,  # 生成训练图表
        }
        
        # 开始训练
        results = self.model.train(**train_args)
        
        print("\n✅ 训练完成!")
        print(f"   最佳模型: {self.output_dir}/weights/best.pt")
        
        return results
        
    def validate_with_supervision(self):
        """使用supervision进行验证"""
        print("\n🔍 使用Supervision进行验证...")
        
        # 加载最佳模型
        best_model_path = self.output_dir / "weights" / "best.pt"
        if not best_model_path.exists():
            print("⚠️ 未找到训练好的模型")
            return
            
        model = YOLO(best_model_path)
        
        # 验证集路径
        val_images = Path(self.data_yaml).parent / "images" / "val"
        val_labels = Path(self.data_yaml).parent / "labels" / "val"
        
        # 初始化指标
        total_gt = 0
        total_tp = 0
        class_names = self.dataset_info['names']
        
        # 遍历验证集
        val_files = list(val_images.glob("*.jpg"))[:50]  # 验证前50张
        
        for img_file in val_files:
            # 读取图像
            img = cv2.imread(str(img_file))
            h, w = img.shape[:2]
            
            # YOLO推理
            results = model(img, verbose=False)[0]
            pred_dets = sv.Detections.from_ultralytics(results)
            
            # 读取真实标注
            label_file = val_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                # 使用新的supervision API
                import numpy as np
                # 读取YOLO格式标注
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # 解析标注
                    class_ids = []
                    xyxy_boxes = []
                    for line in lines:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # 转换为像素坐标
                        x1 = (x_center - width/2) * w
                        y1 = (y_center - height/2) * h
                        x2 = (x_center + width/2) * w
                        y2 = (y_center + height/2) * h
                        
                        class_ids.append(class_id)
                        xyxy_boxes.append([x1, y1, x2, y2])
                    
                    # 创建Detections对象
                    gt_dets = sv.Detections(
                        xyxy=np.array(xyxy_boxes),
                        class_id=np.array(class_ids)
                    )
                else:
                    # 空标注
                    gt_dets = sv.Detections(
                        xyxy=np.empty((0, 4)),
                        class_id=np.empty(0)
                    )
                
                # 使用新的匹配函数
                try:
                    matches = sv.metrics.detection.match_detections(
                        gt_dets, pred_dets, iou_threshold=0.5
                    )
                    # 确保matches是一个数组或列表
                    if isinstance(matches, (int, np.integer)):
                        matches = [matches] if matches > 0 else []
                except:
                    # 如果没有匹配函数，手动计算IoU匹配
                    matches = []
                    if len(gt_dets) > 0 and len(pred_dets) > 0:
                        from supervision.metrics.detection import box_iou_batch
                        iou_matrix = box_iou_batch(gt_dets.xyxy, pred_dets.xyxy)
                        matched_indices = (iou_matrix > 0.5).sum()
                        matches = [1] * matched_indices  # 创建一个包含匹配数的列表
                
                total_gt += len(gt_dets)
                total_tp += len(matches)
                
        # 计算准确率
        if total_gt > 0:
            accuracy = total_tp / total_gt
            print(f"\n📊 验证结果 (前50张):")
            print(f"   总真实框数: {total_gt}")
            print(f"   正确检测数: {total_tp}")
            print(f"   准确率 (IoU=0.5): {accuracy:.2%}")
        else:
            print("⚠️ 验证集无标注")
            
    def export_model(self, format='onnx'):
        """导出模型"""
        print(f"\n📦 导出模型为 {format.upper()} 格式...")
        
        best_model_path = self.output_dir / "weights" / "best.pt"
        if not best_model_path.exists():
            print("⚠️ 未找到训练好的模型")
            return
            
        model = YOLO(best_model_path)
        export_path = model.export(format=format, dynamic=True)
        print(f"   导出成功: {export_path}")
        
        return export_path


def main():
    """主函数"""
    
    # 创建训练器
    trainer = EnhancedYOLOTrainer(
        model_path="yolov8s.pt",  # 使用YOLOv8s预训练模型
        data_yaml="data/visdrone_yolo/data.yaml"
    )
    
    # 设置环境
    trainer.setup()
    
    # 分析数据集
    trainer.analyze_dataset()
    
    # 使用GPU训练参数
    epochs = 30  # GPU训练30轮
    imgsz = 640
    batch = 16
    
    # 开始训练
    trainer.train(epochs=epochs, imgsz=imgsz, batch=batch)
    
    # 使用supervision验证
    trainer.validate_with_supervision()
    
    # 导出模型
    # trainer.export_model('onnx')
    
    print("\n🎉 所有任务完成!")


if __name__ == "__main__":
    main()