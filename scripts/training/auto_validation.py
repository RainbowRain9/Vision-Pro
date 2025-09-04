#!/usr/bin/env python3
"""
训练完成后的自动验证脚本
在训练完成后自动执行模型验证和评估
"""

import os
import time
import subprocess
from pathlib import Path

def wait_for_training_completion():
    """等待训练完成"""
    print("⏳ 等待训练完成...")
    
    train_dir = Path("runs/train_enhanced/20250902_203931")
    weights_dir = train_dir / "weights"
    
    # 等待最佳权重文件生成
    best_weight_file = weights_dir / "best.pt"
    last_weight_file = weights_dir / "last.pt"
    
    while not best_weight_file.exists():
        print("   🕐 等待最佳权重文件生成...")
        time.sleep(60)  # 每分钟检查一次
        
        # 检查是否有错误
        if not train_dir.exists():
            print("   ⚠️ 训练目录不存在，可能训练已停止")
            return False
    
    print("   ✅ 最佳权重文件已生成")
    return True

def run_validation():
    """运行验证"""
    print("\n🔍 开始模型验证...")
    
    try:
        # 运行验证脚本
        result = subprocess.run([
            "python", "post_training_validation.py"
        ], capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            print("   ✅ 验证完成")
            print(result.stdout)
        else:
            print("   ⚠️ 验证过程中出现错误:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("   ⚠️ 验证超时")
    except Exception as e:
        print(f"   ⚠️ 验证过程中出现异常: {e}")

def generate_final_report():
    """生成最终报告"""
    print("\n📝 生成最终报告...")
    
    # 创建最终报告目录
    report_dir = Path("final_report")
    report_dir.mkdir(exist_ok=True)
    
    # 复制重要文件到报告目录
    files_to_copy = [
        "README.md",
        "project_summary.md",
        "technical_details.md",
        "class_distribution.png",
        "runs/train_enhanced/20250902_203931/labels.jpg"
    ]
    
    for file_path in files_to_copy:
        src = Path(file_path)
        if src.exists():
            dst = report_dir / src.name
            try:
                dst.write_bytes(src.read_bytes())
                print(f"   ✅ 复制 {file_path}")
            except Exception as e:
                print(f"   ⚠️ 复制 {file_path} 失败: {e}")
    
    print(f"   📁 最终报告保存到: {report_dir}")

def main():
    """主函数"""
    print("=" * 50)
    print("训练后自动验证系统")
    print("=" * 50)
    
    # 等待训练完成
    if wait_for_training_completion():
        # 运行验证
        run_validation()
        
        # 生成最终报告
        generate_final_report()
        
        print("\n🎉 所有任务完成!")
    else:
        print("\n❌ 训练未正常完成")

if __name__ == "__main__":
    main()