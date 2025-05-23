#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版配置检查脚本
避免编码问题的简单配置验证工具

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
from pathlib import Path
from datetime import datetime

def print_header():
    """打印标题"""
    print("YOLOvision Pro 简化配置检查")
    print("=" * 50)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {Path.cwd()}")
    print("=" * 50)

def check_ultralytics():
    """检查 ultralytics 安装"""
    print("\n[检查] Ultralytics...")
    
    try:
        import ultralytics
        print(f"[OK] Ultralytics {ultralytics.__version__}")
        
        # 检查是否为本地版本
        ultralytics_path = Path(ultralytics.__file__).parent.parent
        project_path = Path.cwd()
        
        if str(ultralytics_path).startswith(str(project_path)):
            print(f"[OK] 本地开发版本")
            return True
        else:
            print(f"[WARN] 全局版本")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Ultralytics 未安装: {e}")
        return False

def check_yolo():
    """检查 YOLO 功能"""
    print("\n[检查] YOLO 功能...")
    
    try:
        from ultralytics import YOLO
        print("[OK] YOLO 类导入成功")
        
        # 尝试加载模型
        model = YOLO('yolov8s.pt')
        print(f"[OK] 模型加载成功 - {len(model.names)} 类别")
        return True
        
    except Exception as e:
        print(f"[ERROR] YOLO 功能异常: {e}")
        return False

def check_visdrone_dataset():
    """检查 VisDrone 数据集"""
    print("\n[检查] VisDrone 数据集...")
    
    # 检查配置文件
    config_path = Path("data/visdrone_yolo/data.yaml")
    if not config_path.exists():
        print("[ERROR] 数据集配置文件缺失")
        return False
    
    print("[OK] 配置文件存在")
    
    # 检查数据目录
    base_path = Path("data/visdrone_yolo")
    splits = ['train', 'val', 'test']
    
    total_images = 0
    for split in splits:
        images_dir = base_path / "images" / split
        if images_dir.exists():
            img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            total_images += img_count
            print(f"[OK] {split} 集: {img_count} 图像")
        else:
            print(f"[ERROR] {split} 集目录不存在")
    
    if total_images > 0:
        print(f"[OK] 总计: {total_images} 图像")
        return True
    else:
        print("[ERROR] 未找到图像数据")
        return False

def check_environment():
    """检查环境"""
    print("\n[检查] 环境...")
    
    # Python 版本
    python_version = sys.version.split()[0]
    print(f"[OK] Python {python_version}")
    
    # 虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("[OK] 虚拟环境已激活")
        venv_status = True
    else:
        print("[WARN] 未检测到虚拟环境")
        venv_status = False
    
    # 关键依赖
    dependencies = {
        'torch': 'torch',
        'numpy': 'numpy',
        'opencv': 'cv2',
        'pillow': 'PIL',
        'yaml': 'yaml'
    }
    
    dep_status = True
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[ERROR] {name}")
            dep_status = False
    
    return venv_status and dep_status

def main():
    """主函数"""
    print_header()
    
    # 执行检查
    results = {
        'ultralytics': check_ultralytics(),
        'yolo': check_yolo(),
        'dataset': check_visdrone_dataset(),
        'environment': check_environment()
    }
    
    # 统计结果
    passed = sum(results.values())
    total = len(results)
    
    # 显示总结
    print("\n" + "=" * 50)
    print("检查结果总结")
    print("=" * 50)
    
    status_items = [
        ("Ultralytics 本地安装", results['ultralytics']),
        ("YOLO 功能", results['yolo']),
        ("VisDrone 数据集", results['dataset']),
        ("环境配置", results['environment'])
    ]
    
    for item, status in status_items:
        icon = "[OK]" if status else "[ERROR]"
        print(f"{icon} {item}")
    
    print(f"\n总计: {passed}/{total} 项检查通过")
    
    # 给出建议
    if passed == total:
        print("\n[SUCCESS] 所有检查通过！配置正常，可以开始开发！")
    elif passed >= 2:
        print("\n[WARN] 大部分配置正常，有少量问题需要修复")
    else:
        print("\n[ERROR] 配置存在较多问题")
    
    print("=" * 50)
    print("简化检查完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
