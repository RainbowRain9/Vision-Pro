import os
import sys
from pathlib import Path

# 添加本地 ultralytics 到 Python 路径
current_dir = Path(__file__).parent
ultralytics_path = current_dir / "ultralytics"
if ultralytics_path.exists():
    ultralytics_str = str(ultralytics_path.absolute())
    if ultralytics_str not in sys.path:
        sys.path.insert(0, ultralytics_str)
        print(f"✅ 添加 ultralytics 路径: {ultralytics_str}")

from ultralytics import YOLO
import torch

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ 使用设备: {device}")

# 查找最新的训练检查点
checkpoint_paths = [
    "runs/train/train/weights/last.pt",
    "runs/train/train2/weights/last.pt", 
    "runs/train/train3/weights/last.pt"
]

checkpoint_path = None
for path in checkpoint_paths:
    if Path(path).exists():
        checkpoint_path = path
        break

if checkpoint_path:
    print(f"📁 找到检查点: {checkpoint_path}")
    # 从检查点恢复训练
    model = YOLO(checkpoint_path)
    
    # 根据设备调整参数
    if device == 'cuda':
        # GPU 训练参数
        model.train(
            data="data/visdrone_yolo/data.yaml",
            imgsz=640,      # GPU 可以使用更大尺寸
            epochs=300,     # 完整训练轮次
            workers=8,      # 更多工作进程
            batch=16,       # 更大批次
            cache=True,     # 启用缓存
            device='cuda',
            patience=20,
            save_period=10,
            resume=True,    # 关键：恢复训练
            project='runs/train'
        )
    else:
        # CPU 训练参数
        model.train(
            data="data/visdrone_yolo/data.yaml",
            imgsz=416,
            epochs=50,
            workers=4,
            batch=2,
            cache=False,
            device='cpu',
            patience=10,
            save_period=5,
            resume=True,    # 关键：恢复训练
            project='runs/train'
        )
else:
    print("❌ 未找到训练检查点，请检查 runs/train/ 目录")
    print("💡 如果是首次训练，请运行 train.py")