#!/usr/bin/env python3
"""
训练进度监控脚本
用于监控YOLOv8训练进度
"""

import time
import os
from pathlib import Path

def monitor_training_progress():
    """监控训练进度"""
    print("🔍 监控训练进度...")
    
    # 训练输出目录
    train_dir = Path("runs/train_enhanced/20250902_203931")
    
    if not train_dir.exists():
        print("   ⚠️ 训练目录不存在")
        return
    
    # 监控权重文件
    weights_dir = train_dir / "weights"
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   监控目录: {train_dir}")
    print("   按 Ctrl+C 停止监控")
    
    try:
        while True:
            # 检查权重文件
            weight_files = list(weights_dir.glob("*.pt"))
            if weight_files:
                print(f"   🔧 已生成权重文件: {len(weight_files)} 个")
                for wf in weight_files:
                    size = wf.stat().st_size / (1024*1024)  # MB
                    print(f"      {wf.name}: {size:.1f} MB")
            
            # 检查日志文件
            log_files = list(train_dir.glob("results*.csv"))
            if log_files:
                log_file = log_files[0]
                lines = 0
                with open(log_file, 'r') as f:
                    lines = len(f.readlines())
                if lines > 1:
                    print(f"   📊 训练日志: {lines-1} 个epoch")
            
            # 检查图表文件
            chart_files = list(train_dir.glob("*.png"))
            if chart_files:
                print(f"   📈 生成图表: {len(chart_files)} 个")
            
            time.sleep(30)  # 每30秒检查一次
            
    except KeyboardInterrupt:
        print("\n   停止监控")

def main():
    """主函数"""
    print("=" * 50)
    print("训练进度监控")
    print("=" * 50)
    
    monitor_training_progress()
    
    print("\n✅ 监控完成!")

if __name__ == "__main__":
    main()