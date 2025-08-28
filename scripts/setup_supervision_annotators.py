#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervision 标注器功能安装和验证脚本
自动检查依赖、配置环境并验证功能
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import logging
import yaml

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class SupervisionAnnotatorsSetup:
    """Supervision 标注器安装和验证类"""
    
    def __init__(self):
        self.project_root = project_root
        self.logger = self._setup_logging()
        self.required_packages = [
            'supervision>=0.20.0',
            'ultralytics',
            'opencv-python',
            'numpy',
            'PyQt5'
        ]
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def check_python_version(self):
        """检查Python版本"""
        self.logger.info("检查Python版本...")
        
        if sys.version_info < (3, 8):
            self.logger.error("需要Python 3.8或更高版本")
            return False
        
        self.logger.info(f"Python版本: {sys.version}")
        return True
    
    def check_dependencies(self):
        """检查依赖包"""
        self.logger.info("检查依赖包...")
        
        missing_packages = []
        
        for package in self.required_packages:
            package_name = package.split('>=')[0].split('==')[0]
            try:
                importlib.import_module(package_name.replace('-', '_'))
                self.logger.info(f"✅ {package_name} 已安装")
            except ImportError:
                missing_packages.append(package)
                self.logger.warning(f"❌ {package_name} 未安装")
        
        return missing_packages
    
    def install_dependencies(self, packages):
        """安装缺失的依赖包"""
        if not packages:
            self.logger.info("所有依赖包已安装")
            return True
        
        self.logger.info(f"安装缺失的依赖包: {packages}")
        
        for package in packages:
            try:
                self.logger.info(f"安装 {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
                self.logger.info(f"✅ {package} 安装成功")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ {package} 安装失败: {e}")
                return False
        
        return True
    
    def verify_supervision_version(self):
        """验证Supervision版本"""
        try:
            import supervision as sv
            version = sv.__version__
            self.logger.info(f"Supervision版本: {version}")
            
            # 检查版本兼容性
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse("0.20.0"):
                self.logger.warning("Supervision版本过低，建议升级到0.26.1+")
                return False
            
            return True
        except ImportError:
            self.logger.error("Supervision未安装")
            return False
        except Exception as e:
            self.logger.error(f"检查Supervision版本失败: {e}")
            return False
    
    def create_directories(self):
        """创建必要的目录"""
        self.logger.info("创建项目目录...")
        
        directories = [
            self.project_root / "outputs" / "annotator_demo",
            self.project_root / "outputs" / "individual_annotators",
            self.project_root / "outputs" / "heatmap_demo",
            self.project_root / "assets" / "configs",
            self.project_root / "docs",
            self.project_root / "scripts" / "modules",
            self.project_root / "scripts" / "demo",
            self.project_root / "scripts" / "testing"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ 创建目录: {directory}")
    
    def verify_config_files(self):
        """验证配置文件"""
        self.logger.info("验证配置文件...")
        
        config_file = self.project_root / "assets" / "configs" / "annotator_config.yaml"
        
        if not config_file.exists():
            self.logger.warning("配置文件不存在，创建默认配置...")
            self.create_default_config(config_file)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证配置结构
            required_sections = ['annotators', 'presets']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"配置文件缺少 {section} 节")
                    return False
            
            self.logger.info("✅ 配置文件验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"配置文件验证失败: {e}")
            return False
    
    def create_default_config(self, config_path):
        """创建默认配置文件"""
        default_config = {
            'annotators': {
                'box': {'enabled': True, 'thickness': 2},
                'label': {'enabled': True, 'text_scale': 0.5},
                'mask': {'enabled': False, 'opacity': 0.5},
                'polygon': {'enabled': False, 'thickness': 2},
                'heatmap': {'enabled': False, 'opacity': 0.7},
                'blur': {'enabled': False, 'kernel_size': 15},
                'pixelate': {'enabled': False, 'pixel_size': 20}
            },
            'presets': {
                'basic': ['box', 'label'],
                'detailed': ['box', 'label', 'polygon'],
                'privacy': ['blur', 'label'],
                'analysis': ['box', 'label', 'heatmap'],
                'segmentation': ['mask', 'label']
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"✅ 创建默认配置文件: {config_path}")
    
    def test_annotator_functionality(self):
        """测试标注器功能"""
        self.logger.info("测试标注器功能...")
        
        try:
            # 导入模块
            sys.path.append(str(self.project_root / "scripts" / "modules"))
            from supervision_annotators import AnnotatorManager, AnnotatorType
            
            # 创建管理器
            config_path = self.project_root / "assets" / "configs" / "annotator_config.yaml"
            manager = AnnotatorManager(str(config_path))
            
            # 测试基本功能
            info = manager.get_annotator_info()
            self.logger.info(f"可用标注器: {info['available_annotators']}")
            self.logger.info(f"已启用标注器: {info['enabled_annotators']}")
            
            # 测试预设
            manager.set_preset('basic')
            self.logger.info("✅ 预设功能正常")
            
            # 测试标注器切换
            manager.enable_annotator(AnnotatorType.POLYGON)
            manager.disable_annotator(AnnotatorType.POLYGON)
            self.logger.info("✅ 标注器切换功能正常")
            
            return True
            
        except Exception as e:
            self.logger.error(f"标注器功能测试失败: {e}")
            return False
    
    def run_demo_test(self):
        """运行演示测试"""
        self.logger.info("运行演示测试...")
        
        try:
            # 检查演示脚本
            demo_script = self.project_root / "scripts" / "demo" / "supervision_annotators_demo.py"
            if not demo_script.exists():
                self.logger.warning("演示脚本不存在")
                return False
            
            # 运行基本测试
            import subprocess
            result = subprocess.run([
                sys.executable, str(demo_script), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.logger.info("✅ 演示脚本可正常运行")
                return True
            else:
                self.logger.error(f"演示脚本运行失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"演示测试失败: {e}")
            return False
    
    def print_summary(self, success: bool):
        """打印安装总结"""
        print("\n" + "="*60)
        print("Supervision 标注器功能安装总结")
        print("="*60)
        
        if success:
            print("✅ 安装成功！")
            print("\n📋 功能清单:")
            print("  • 7种标注器支持 (Box, Label, Mask, Polygon, HeatMap, Blur, Pixelate)")
            print("  • 6种预设配置 (basic, detailed, privacy, analysis, segmentation, presentation)")
            print("  • 主界面集成控制")
            print("  • 配置文件管理")
            print("  • 演示和测试脚本")
            
            print("\n🚀 快速开始:")
            print("  1. 运行主程序: python main.py")
            print("  2. 在右侧面板找到'标注器设置'组")
            print("  3. 选择预设或手动配置标注器")
            print("  4. 开始检测并查看效果")
            
            print("\n📖 更多信息:")
            print("  • 使用指南: docs/supervision_annotators_guide.md")
            print("  • 演示脚本: scripts/demo/supervision_annotators_demo.py")
            print("  • 测试脚本: scripts/testing/test_supervision_annotators.py")
            
        else:
            print("❌ 安装失败！")
            print("\n🔧 故障排除:")
            print("  1. 检查Python版本 (需要3.8+)")
            print("  2. 手动安装依赖: pip install supervision ultralytics")
            print("  3. 查看日志了解详细错误信息")
            print("  4. 参考文档: docs/supervision_annotators_guide.md")
        
        print("="*60)
    
    def run_setup(self):
        """运行完整安装流程"""
        self.logger.info("开始 Supervision 标注器功能安装...")
        
        success = True
        
        # 1. 检查Python版本
        if not self.check_python_version():
            success = False
        
        # 2. 检查和安装依赖
        if success:
            missing_packages = self.check_dependencies()
            if missing_packages:
                if not self.install_dependencies(missing_packages):
                    success = False
        
        # 3. 验证Supervision版本
        if success:
            if not self.verify_supervision_version():
                success = False
        
        # 4. 创建目录
        if success:
            self.create_directories()
        
        # 5. 验证配置文件
        if success:
            if not self.verify_config_files():
                success = False
        
        # 6. 测试功能
        if success:
            if not self.test_annotator_functionality():
                success = False
        
        # 7. 运行演示测试
        if success:
            self.run_demo_test()  # 非关键，失败不影响整体
        
        # 8. 打印总结
        self.print_summary(success)
        
        return success


def main():
    """主函数"""
    setup = SupervisionAnnotatorsSetup()
    success = setup.run_setup()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
