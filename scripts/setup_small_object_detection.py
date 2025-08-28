#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小目标检测功能安装和设置脚本
自动安装依赖、验证功能、提供使用指导
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def print_header(title: str):
    """打印标题"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step: str):
    """打印步骤"""
    print(f"\n📋 {step}")
    print("-" * 40)

def check_python_version():
    """检查 Python 版本"""
    print_step("检查 Python 版本")
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 版本过低，需要 Python 3.8 或更高版本")
        return False
    else:
        print("✅ Python 版本符合要求")
        return True

def check_package(package_name: str, import_name: str = None) -> bool:
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False

def install_package(package_name: str) -> bool:
    """安装包"""
    try:
        print(f"🔄 正在安装 {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def check_and_install_dependencies():
    """检查并安装依赖"""
    print_step("检查和安装依赖包")
    
    # 必需的包列表
    required_packages = [
        ("pyyaml", "yaml"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("supervision", "supervision"),
    ]
    
    # 可选的包（深度学习相关）
    optional_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
    ]
    
    # 检查必需包
    missing_required = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_required.append(package_name)
    
    # 安装缺失的必需包
    if missing_required:
        print(f"\n需要安装的必需包: {', '.join(missing_required)}")
        for package in missing_required:
            if not install_package(package):
                print(f"❌ 关键依赖 {package} 安装失败，程序可能无法正常运行")
                return False
    
    # 检查可选包
    print(f"\n检查可选包（深度学习功能）:")
    missing_optional = []
    for package_name, import_name in optional_packages:
        if not check_package(package_name, import_name):
            missing_optional.append(package_name)
    
    if missing_optional:
        print(f"\n⚠️  以下可选包未安装: {', '.join(missing_optional)}")
        print("这些包是深度学习功能所必需的，建议安装：")
        
        # 提供安装建议
        if "torch" in missing_optional:
            print("💡 PyTorch 安装建议:")
            print("   CPU 版本: pip install torch torchvision")
            print("   GPU 版本: 访问 https://pytorch.org 获取适合您系统的安装命令")
        
        if "ultralytics" in missing_optional:
            print("💡 Ultralytics 安装: pip install ultralytics")
        
        # 询问是否安装
        try:
            install_optional = input("\n是否现在安装可选包？(y/N): ").lower().strip()
            if install_optional in ['y', 'yes']:
                for package in missing_optional:
                    install_package(package)
        except KeyboardInterrupt:
            print("\n用户取消安装")
    
    return True

def verify_small_object_detection():
    """验证小目标检测功能"""
    print_step("验证小目标检测功能")
    
    try:
        # 运行测试脚本
        test_script = PROJECT_ROOT / "scripts" / "testing" / "test_small_object_config.py"
        if test_script.exists():
            print("🧪 运行功能测试...")
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 小目标检测功能验证通过")
                # 显示测试结果的关键信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if '总计:' in line or '🎉' in line:
                        print(f"   {line}")
                return True
            else:
                print("❌ 功能验证失败")
                print("错误信息:")
                print(result.stderr)
                return False
        else:
            print("❌ 测试脚本不存在")
            return False
            
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False

def show_usage_guide():
    """显示使用指南"""
    print_step("使用指南")
    
    print("🎯 小目标检测功能已就绪！")
    print("\n📖 使用方法:")
    
    print("\n1. GUI 界面使用:")
    print("   python main.py")
    print("   - 勾选 '启用小目标检测 (InferenceSlicer)'")
    print("   - 选择检测模式和参数")
    print("   - 执行图片检测")
    
    print("\n2. 命令行演示:")
    print("   python scripts/demo/small_object_detection_demo.py")
    
    print("\n3. 编程接口:")
    print("   from scripts.modules.supervision_wrapper import SupervisionWrapper")
    print("   wrapper = SupervisionWrapper(class_names=['car', 'person'])")
    print("   result = wrapper.detect_small_objects(image, model)")
    
    print("\n📚 文档位置:")
    print(f"   - 使用指南: {PROJECT_ROOT / 'docs' / 'tutorials' / 'small_object_detection_guide.md'}")
    print(f"   - 配置文件: {PROJECT_ROOT / 'assets' / 'configs' / 'small_object_detection_config.yaml'}")
    print(f"   - 功能总结: {PROJECT_ROOT / 'docs' / '小目标检测功能实现总结.md'}")
    
    print("\n🔧 配置选项:")
    print("   - ultra_small: 320×320 切片，适合极小目标")
    print("   - small: 640×640 切片，标准配置")
    print("   - medium: 800×800 切片，适合中等目标")
    print("   - large: 1024×1024 切片，适合大目标")
    
    print("\n💡 性能提示:")
    print("   - 切片越小，检测精度越高，但处理时间更长")
    print("   - 重叠区域有助于边界目标检测")
    print("   - 使用 GPU 可显著提升处理速度")

def create_desktop_shortcut():
    """创建桌面快捷方式（Windows）"""
    if sys.platform == "win32":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "YOLOvision Pro 小目标检测.lnk")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = str(PROJECT_ROOT / "main.py")
            shortcut.WorkingDirectory = str(PROJECT_ROOT)
            shortcut.IconLocation = str(PROJECT_ROOT / "icon.ico") if (PROJECT_ROOT / "icon.ico").exists() else ""
            shortcut.save()
            
            print(f"✅ 桌面快捷方式已创建: {shortcut_path}")
            
        except ImportError:
            print("⚠️  无法创建桌面快捷方式（需要 pywin32 包）")
        except Exception as e:
            print(f"⚠️  创建桌面快捷方式失败: {e}")

def main():
    """主函数"""
    print_header("YOLOvision Pro 小目标检测功能安装向导")
    
    # 1. 检查 Python 版本
    if not check_python_version():
        print("\n❌ 安装失败：Python 版本不符合要求")
        return False
    
    # 2. 检查和安装依赖
    if not check_and_install_dependencies():
        print("\n❌ 安装失败：依赖包安装出错")
        return False
    
    # 3. 验证功能
    if not verify_small_object_detection():
        print("\n⚠️  功能验证失败，但基础功能可能仍可使用")
    
    # 4. 显示使用指南
    show_usage_guide()
    
    # 5. 询问是否创建快捷方式
    if sys.platform == "win32":
        try:
            create_shortcut = input("\n是否创建桌面快捷方式？(y/N): ").lower().strip()
            if create_shortcut in ['y', 'yes']:
                create_desktop_shortcut()
        except KeyboardInterrupt:
            print("\n用户取消")
    
    print_header("安装完成")
    print("🎉 小目标检测功能安装完成！")
    print("现在您可以使用 YOLOvision Pro 的小目标检测功能了。")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断安装")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 安装过程出现未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
