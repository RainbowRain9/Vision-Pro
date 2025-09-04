#!/usr/bin/env python3
"""
训练完成后的自动化处理脚本
在模型训练完成后自动执行验证、分析和报告生成
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
    
    # 记录等待开始时间
    start_time = time.time()
    
    while not best_weight_file.exists():
        elapsed_time = time.time() - start_time
        print(f"   🕐 等待最佳权重文件生成... (已等待 {int(elapsed_time)} 秒)")
        
        # 检查训练进程是否仍在运行
        try:
            result = subprocess.run(["pgrep", "-f", "train_enhanced.py"], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                print("   ⚠️ 训练进程已停止")
                # 检查是否有last.pt文件
                last_weight_file = weights_dir / "last.pt"
                if last_weight_file.exists():
                    print("   🔄 使用最后保存的权重文件")
                    break
                else:
                    return False
        except Exception as e:
            print(f"   ⚠️ 检查训练进程时出错: {e}")
        
        time.sleep(30)  # 每30秒检查一次
    
    print("   ✅ 训练已完成，权重文件已生成")
    return True

def run_post_training_validation():
    """运行训练后验证"""
    print("\n🔍 运行训练后验证...")
    
    try:
        # 运行验证脚本
        print("   📊 执行模型性能验证...")
        result = subprocess.run([
            "python", "post_training_validation.py"
        ], capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            print("   ✅ 验证完成")
            # 保存验证输出到文件
            with open("validation_output.log", "w") as f:
                f.write(result.stdout)
        else:
            print("   ⚠️ 验证过程中出现错误:")
            print(result.stderr)
            # 保存错误信息到文件
            with open("validation_error.log", "w") as f:
                f.write(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("   ⚠️ 验证超时")
    except Exception as e:
        print(f"   ⚠️ 验证过程中出现异常: {e}")

def run_supervision_demo():
    """运行supervision功能演示"""
    print("\n🎯 运行supervision功能演示...")
    
    try:
        result = subprocess.run([
            "python", "supervision_demo.py"
        ], capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("   ✅ supervision演示完成")
        else:
            print("   ⚠️ supervision演示过程中出现错误:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("   ⚠️ supervision演示超时")
    except Exception as e:
        print(f"   ⚠️ supervision演示过程中出现异常: {e}")

def generate_analysis_report():
    """生成分析报告"""
    print("\n📈 生成分析报告...")
    
    try:
        result = subprocess.run([
            "python", "simple_analysis.py"
        ], capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("   ✅ 分析报告生成完成")
        else:
            print("   ⚠️ 分析报告生成过程中出现错误:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("   ⚠️ 分析报告生成超时")
    except Exception as e:
        print(f"   ⚠️ 分析报告生成过程中出现异常: {e}")

def create_final_package():
    """创建最终成果包"""
    print("\n📦 创建最终成果包...")
    
    # 创建最终成果目录
    final_dir = Path("final_deliverables")
    final_dir.mkdir(exist_ok=True)
    
    # 需要复制的文件列表
    files_to_copy = [
        # 核心脚本
        "train_enhanced.py",
        "supervision_demo.py",
        "post_training_validation.py",
        "monitor_training.py",
        "simple_analysis.py",
        "auto_validation.py",
        
        # 文档
        "README.md",
        "project_summary.md",
        "technical_details.md",
        "final_status_report.md",
        
        # 输出文件
        "class_distribution.png",
        "validation_output.log",
        "validation_error.log",
    ]
    
    # 复制文件
    for file_path in files_to_copy:
        src = Path(file_path)
        if src.exists():
            dst = final_dir / src.name
            try:
                dst.write_bytes(src.read_bytes())
                print(f"   ✅ 复制 {file_path}")
            except Exception as e:
                print(f"   ⚠️ 复制 {file_path} 失败: {e}")
    
    # 复制训练输出
    train_output_dir = Path("runs/train_enhanced/20250902_203931")
    if train_output_dir.exists():
        final_train_dir = final_dir / "training_output"
        final_train_dir.mkdir(exist_ok=True)
        
        # 复制重要文件
        important_files = [
            "args.yaml",
            "labels.jpg",
            "train_batch0.jpg",
            "train_batch1.jpg",
            "train_batch2.jpg",
        ]
        
        for file_name in important_files:
            src = train_output_dir / file_name
            if src.exists():
                dst = final_train_dir / file_name
                try:
                    dst.write_bytes(src.read_bytes())
                    print(f"   ✅ 复制训练输出 {file_name}")
                except Exception as e:
                    print(f"   ⚠️ 复制训练输出 {file_name} 失败: {e}")
    
    # 复制权重文件
    weights_dir = train_output_dir / "weights"
    if weights_dir.exists():
        final_weights_dir = final_train_dir / "weights"
        final_weights_dir.mkdir(exist_ok=True)
        
        weight_files = list(weights_dir.glob("*.pt"))
        for weight_file in weight_files:
            dst = final_weights_dir / weight_file.name
            try:
                dst.write_bytes(weight_file.read_bytes())
                print(f"   ✅ 复制权重文件 {weight_file.name}")
            except Exception as e:
                print(f"   ⚠️ 复制权重文件 {weight_file.name} 失败: {e}")
    
    print(f"   📁 最终成果包保存到: {final_dir}")

def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8 VisDrone 项目自动化处理系统")
    print("=" * 60)
    
    # 等待训练完成
    if wait_for_training_completion():
        print("\n🎉 训练已完成，开始后续处理...")
        
        # 运行训练后验证
        run_post_training_validation()
        
        # 运行supervision演示
        run_supervision_demo()
        
        # 生成分析报告
        generate_analysis_report()
        
        # 创建最终成果包
        create_final_package()
        
        print("\n🎊 所有自动化处理任务完成!")
        print("   请查看 final_deliverables 目录获取所有成果文件")
    else:
        print("\n❌ 训练未正常完成，无法执行后续处理")

if __name__ == "__main__":
    main()