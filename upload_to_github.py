#!/usr/bin/env python3
"""
YOLOvision Pro GitHub 上传脚本
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd or '.',
            encoding='utf-8'
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_git_status():
    """检查 Git 状态"""
    print("🔍 检查 Git 仓库状态...")
    
    # 检查是否是 Git 仓库
    if not Path('.git').exists():
        print("❌ 当前目录不是 Git 仓库")
        return False
    
    # 检查远程仓库
    stdout, stderr, code = run_command("git remote -v")
    if code == 0 and stdout:
        print("✅ 远程仓库配置:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"   {line}")
    else:
        print("⚠️ 未配置远程仓库")
    
    return True

def add_files():
    """添加文件到 Git"""
    print("\n📁 添加文件到 Git...")
    
    stdout, stderr, code = run_command("git add .")
    if code == 0:
        print("✅ 文件添加成功")
        
        # 显示将要提交的文件
        stdout, stderr, code = run_command("git diff --cached --name-only")
        if code == 0 and stdout:
            files = [f for f in stdout.split('\n') if f.strip()]
            print(f"📊 将提交 {len(files)} 个文件:")
            for i, file in enumerate(files[:10]):  # 只显示前10个
                print(f"   {file}")
            if len(files) > 10:
                print(f"   ... 还有 {len(files) - 10} 个文件")
        return True
    else:
        print(f"❌ 添加文件失败: {stderr}")
        return False

def commit_changes():
    """提交更改"""
    print("\n💾 提交更改...")
    
    commit_message = """🚁 重大更新: YOLOvision Pro 项目重组与 Drone-YOLO 集成

✨ 新功能:
- 集成 Drone-YOLO 小目标检测算法
- 添加 RepVGGBlock 高效主干网络
- 实现 P2 小目标检测头
- 集成三明治融合结构

🏗️ 项目重组:
- 创建模块化目录结构 (docs/, scripts/, assets/, experiments/, outputs/)
- 重新组织文档和脚本文件
- 更新 main.py 支持新架构和 Drone-YOLO
- 添加完整的 README 和使用指南

📚 文档完善:
- 详细的 Drone-YOLO 技术解析
- 完整的项目结构说明
- 各目录使用指南和 README
- 代码演示和测试脚本

🔧 技术改进:
- 现代化路径处理 (pathlib)
- 增强错误处理和日志系统
- 清理代码和优化性能
- 添加配置文件支持"""

    stdout, stderr, code = run_command(f'git commit -m "{commit_message}"')
    if code == 0:
        print("✅ 提交成功")
        return True
    else:
        if "nothing to commit" in stderr:
            print("ℹ️ 没有新的更改需要提交")
            return True
        else:
            print(f"❌ 提交失败: {stderr}")
            return False

def push_to_github():
    """推送到 GitHub"""
    print("\n🚀 推送到 GitHub...")
    
    stdout, stderr, code = run_command("git push origin main")
    if code == 0:
        print("✅ 推送成功!")
        print("🎉 项目已成功上传到 GitHub!")
        return True
    else:
        print(f"❌ 推送失败: {stderr}")
        
        # 尝试设置上游分支
        if "no upstream branch" in stderr or "set-upstream" in stderr:
            print("🔧 尝试设置上游分支...")
            stdout, stderr, code = run_command("git push --set-upstream origin main")
            if code == 0:
                print("✅ 推送成功!")
                return True
            else:
                print(f"❌ 设置上游分支失败: {stderr}")
        
        return False

def verify_upload():
    """验证上传结果"""
    print("\n🔍 验证上传结果...")
    
    # 检查远程仓库状态
    stdout, stderr, code = run_command("git ls-remote origin")
    if code == 0:
        print("✅ 远程仓库连接正常")
        
        # 显示最新提交
        stdout, stderr, code = run_command("git log --oneline -1")
        if code == 0:
            print(f"📝 最新提交: {stdout}")
        
        print("\n🌐 GitHub 仓库地址:")
        print("   https://github.com/RainbowRain9/YOLOv8------")
        
        return True
    else:
        print(f"❌ 验证失败: {stderr}")
        return False

def main():
    """主函数"""
    print("🚀 YOLOvision Pro GitHub 上传工具")
    print("=" * 60)
    
    # 检查 Git 状态
    if not check_git_status():
        return 1
    
    # 添加文件
    if not add_files():
        return 1
    
    # 提交更改
    if not commit_changes():
        return 1
    
    # 推送到 GitHub
    if not push_to_github():
        return 1
    
    # 验证上传
    if not verify_upload():
        return 1
    
    print("\n🎯 上传完成!")
    print("📋 后续步骤:")
    print("   1. 访问 GitHub 仓库验证文件完整性")
    print("   2. 检查 README.md 显示是否正确")
    print("   3. 验证目录结构和文档链接")
    print("   4. 测试克隆仓库到新环境")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
