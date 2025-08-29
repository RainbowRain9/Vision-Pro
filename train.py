import os
import sys
from pathlib import Path

# 设置 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加本地 ultralytics 到 Python 路径
current_dir = Path(__file__).parent
ultralytics_path = current_dir / "ultralytics"
if ultralytics_path.exists():
    ultralytics_str = str(ultralytics_path.absolute())
    if ultralytics_str not in sys.path:
        sys.path.insert(0, ultralytics_str)
        print(f"✅ 添加 ultralytics 路径: {ultralytics_str}")

from ultralytics import YOLO

# Load a model
# model = YOLO(model="yolov8s.yaml")  # yolov8s
# model = YOLO(model="yolov8s-p2.yaml")  # yolov8s+head
# model = YOLO(model="yolov8s-p2-repvgg.yaml")  # yolov8s+repvgg
model = YOLO(model="assets/configs/yolov8s-drone.yaml")  # yolov8s+repvgg+sf

model.load('models/yolov8s.pt')  # 加载预训练权重，提升训练效果
# Use the model
# CPU 训练优化参数
model.train(
    data="data/visdrone_yolo/data.yaml", 
    imgsz=416,      # 降低输入尺寸
    epochs=50,      # 减少训练轮次
    workers=4,      # 减少工作进程
    batch=2,        # 大幅减少批次大小
    cache=False,    # 关闭缓存节省内存
    device='cpu',   # 明确指定 CPU
    patience=10,    # 早停机制，10轮无改善自动停止
    save_period=5,  # 每5轮保存一次检查点
    project='runs/train'
)
# path = model.export(format="onnx", dynamic=True)  # export the mode l to ONNX format

