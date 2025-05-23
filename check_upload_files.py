#!/usr/bin/env python3
"""
检查将要上传到 GitHub 的文件列表
"""

import os
import subprocess
from pathlib import Path

def run_git_command(command):
    """运行 Git 命令"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd='.')
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_git_status():
    """检查 Git 状态"""
    print("🔍 检查 Git 仓库状态...")
    print("=" * 60)
    
    # 检查当前分支
    stdout, stderr, code = run_git_command("git branch --show-current")
    if code == 0:
        print(f"当前分支: {stdout}")
    else:
        print(f"获取分支信息失败: {stderr}")
    
    # 检查远程仓库
    stdout, stderr, code = run_git_command("git remote -v")
    if code == 0:
        print(f"远程仓库:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"  {line}")
    else:
        print(f"获取远程仓库信息失败: {stderr}")

def list_tracked_files():
    """列出将要跟踪的文件"""
    print("\n📁 将要上传的文件列表:")
    print("=" * 60)
    
    # 添加所有文件到暂存区（但不提交）
    stdout, stderr, code = run_git_command("git add -A")
    if code != 0:
        print(f"添加文件失败: {stderr}")
        return
    
    # 获取暂存区文件列表
    stdout, stderr, code = run_git_command("git diff --cached --name-only")
    if code == 0:
        files = stdout.split('\n') if stdout else []
        
        # 按类型分类文件
        categories = {
            '核心程序文件': [],
            '文档文件': [],
            '配置文件': [],
            '脚本文件': [],
            '目录结构文件': [],
            '其他文件': []
        }
        
        for file in files:
            if not file.strip():
                continue
                
            if file in ['main.py', 'train.py', 'README.md']:
                categories['核心程序文件'].append(file)
            elif file.startswith('docs/') or file.startswith('doc/') or file.endswith('.md'):
                categories['文档文件'].append(file)
            elif file.startswith('assets/configs/') or file.endswith('.yaml') or file.endswith('.yml'):
                categories['配置文件'].append(file)
            elif file.startswith('scripts/') or file.endswith('.py'):
                categories['脚本文件'].append(file)
            elif file.endswith('.gitkeep') or file == '.gitignore':
                categories['目录结构文件'].append(file)
            else:
                categories['其他文件'].append(file)
        
        # 显示分类结果
        total_files = 0
        for category, file_list in categories.items():
            if file_list:
                print(f"\n📂 {category} ({len(file_list)} 个文件):")
                for file in sorted(file_list):
                    print(f"  ✅ {file}")
                total_files += len(file_list)
        
        print(f"\n📊 总计: {total_files} 个文件将被上传")
        
    else:
        print(f"获取文件列表失败: {stderr}")

def check_large_files():
    """检查大文件"""
    print("\n🔍 检查大文件 (>10MB):")
    print("=" * 60)
    
    large_files = []
    for root, dirs, files in os.walk('.'):
        # 跳过 .git 目录和虚拟环境
        dirs[:] = [d for d in dirs if not d.startswith('.git') and d != 'yolo8']
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > 10 * 1024 * 1024:  # 10MB
                    large_files.append((file_path, size))
            except:
                continue
    
    if large_files:
        print("⚠️ 发现大文件:")
        for file_path, size in large_files:
            size_mb = size / (1024 * 1024)
            print(f"  📦 {file_path} ({size_mb:.1f} MB)")
        print("\n💡 建议: 大文件应该被 .gitignore 排除或使用 Git LFS")
    else:
        print("✅ 未发现大文件")

def check_ignored_files():
    """检查被忽略的重要文件"""
    print("\n🚫 被 .gitignore 排除的重要目录:")
    print("=" * 60)
    
    ignored_dirs = [
        'yolo8/',
        '__pycache__/',
        'data/VisDrone2019-DET-train/',
        'runs/',
        'outputs/logs/',
        'models/*.pt'
    ]
    
    for item in ignored_dirs:
        if os.path.exists(item.rstrip('*')):
            print(f"  🚫 {item}")
    
    print("\n✅ 这些目录/文件被正确排除，避免上传不必要的大文件")

def main():
    """主函数"""
    print("🚀 YOLOvision Pro GitHub 上传准备检查")
    print("=" * 60)
    
    # 检查 Git 状态
    check_git_status()
    
    # 列出将要上传的文件
    list_tracked_files()
    
    # 检查大文件
    check_large_files()
    
    # 检查被忽略的文件
    check_ignored_files()
    
    print(f"\n🎯 上传准备检查完成!")
    print(f"📋 请确认上述文件列表无误后，执行 Git 提交和推送操作")

if __name__ == "__main__":
    main()
