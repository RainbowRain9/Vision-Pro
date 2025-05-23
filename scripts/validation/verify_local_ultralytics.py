#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOvision Pro 本地 ultralytics 配置验证脚本
验证本地 ultralytics 源代码安装和 VisDrone 数据集配置

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import os
from pathlib import Path
import yaml
from datetime import datetime

def print_banner():
    """打印横幅"""
    print("="*60)
    print("🔍 YOLOvision Pro 本地 ultralytics 配置验证")
    print("="*60)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {Path.cwd()}")
    print("="*60)

def check_ultralytics_installation():
    """检查 ultralytics 安装状态"""
    print("\n📦 检查 ultralytics 安装...")

    try:
        import ultralytics
        print(f"✅ Ultralytics 版本: {ultralytics.__version__}")
        print(f"✅ Ultralytics 路径: {ultralytics.__file__}")

        # 检查是否为本地开发版本
        ultralytics_path = Path(ultralytics.__file__).parent.parent
        project_path = Path.cwd()

        if str(ultralytics_path).startswith(str(project_path)):
            print("✅ 使用本地开发版本 ultralytics")
            print(f"   本地路径: {ultralytics_path}")
            return True, ultralytics.__version__, str(ultralytics_path)
        else:
            print("⚠️ 可能使用的是全局安装版本")
            print(f"   安装路径: {ultralytics_path}")
            return False, ultralytics.__version__, str(ultralytics_path)

    except ImportError as e:
        print(f"❌ Ultralytics 导入失败: {e}")
        return False, "未安装", "N/A"

def check_yolo_class():
    """检查 YOLO 类导入"""
    print("\n🎯 检查 YOLO 类...")

    try:
        from ultralytics import YOLO
        print("✅ YOLO 类导入成功")

        # 检查 YOLO 类的关键方法
        yolo_methods = ['train', 'predict', 'val', 'export']
        for method in yolo_methods:
            if hasattr(YOLO, method):
                print(f"✅ YOLO.{method} 方法可用")
            else:
                print(f"⚠️ YOLO.{method} 方法不可用")

        return True
    except ImportError as e:
        print(f"❌ YOLO 类导入失败: {e}")
        return False

def check_yolo_model():
    """检查 YOLO 模型加载"""
    print("\n🤖 检查 YOLO 模型加载...")

    try:
        from ultralytics import YOLO

        # 检查是否已有模型文件
        model_path = Path("yolov8s.pt")
        if model_path.exists():
            print(f"✅ 找到模型文件: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print("ℹ️ 模型文件不存在，将在首次使用时下载")

        # 尝试加载模型（不执行预测）
        print("正在加载 YOLOv8s 模型...")
        model = YOLO('yolov8s.pt')
        print("✅ YOLOv8s 模型加载成功")
        print(f"✅ 模型类别数: {len(model.names)}")
        print(f"✅ 模型设备: {model.device}")
        print(f"✅ 模型类型: {model.task}")

        # 显示部分类别名称
        class_names = list(model.names.values())[:5]
        print(f"✅ 前5个类别: {', '.join(class_names)}")

        return True, len(model.names), str(model.device)

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False, 0, "N/A"

def check_visdrone_dataset():
    """检查 VisDrone 数据集配置"""
    print("\n📊 检查 VisDrone 数据集...")

    # 检查配置文件
    config_path = Path("data/visdrone_yolo/data.yaml")
    if not config_path.exists():
        print("❌ VisDrone 数据集配置文件不存在")
        print("💡 请先运行: python scripts/process_visdrone_complete.py")
        return False, {}

    print(f"✅ 配置文件存在: {config_path}")

    # 读取配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✅ 数据集路径: {config.get('path', 'N/A')}")
        print(f"✅ 类别数量: {config.get('nc', 'N/A')}")
        print(f"✅ 类别名称: {len(config.get('names', []))} 个类别")

        # 显示类别名称
        if 'names' in config:
            class_names = config['names']
            if isinstance(class_names, dict):
                print("✅ VisDrone 类别:")
                for i, name in class_names.items():
                    print(f"   {i}: {name}")
            elif isinstance(class_names, list):
                print("✅ VisDrone 类别:")
                for i, name in enumerate(class_names):
                    print(f"   {i}: {name}")

    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False, {}

    # 检查数据集目录结构
    base_path = Path("data/visdrone_yolo")
    splits = ['train', 'val', 'test']

    dataset_stats = {
        'total_images': 0,
        'total_labels': 0,
        'splits': {}
    }

    for split in splits:
        images_dir = base_path / "images" / split
        labels_dir = base_path / "labels" / split

        if images_dir.exists() and labels_dir.exists():
            img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))

            img_count = len(img_files)
            label_count = len(label_files)

            print(f"✅ {split} 集 - 图像: {img_count}, 标签: {label_count}")

            dataset_stats['total_images'] += img_count
            dataset_stats['total_labels'] += label_count
            dataset_stats['splits'][split] = {
                'images': img_count,
                'labels': label_count
            }

            if img_count != label_count:
                print(f"⚠️ {split} 集图像和标签数量不匹配")
        else:
            print(f"❌ {split} 集目录不存在")
            dataset_stats['splits'][split] = {
                'images': 0,
                'labels': 0
            }

    print(f"📈 总计 - 图像: {dataset_stats['total_images']}, 标签: {dataset_stats['total_labels']}")

    # 检查类别文件
    classes_file = base_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ 类别文件存在: {len(class_list)} 个类别")
    else:
        print("⚠️ 类别文件不存在")

    return dataset_stats['total_images'] > 0 and dataset_stats['total_labels'] > 0, dataset_stats

def check_environment():
    """检查环境信息"""
    print("\n🌍 检查环境信息...")

    env_info = {
        'python_version': sys.version,
        'working_directory': str(Path.cwd()),
        'python_executable': sys.executable,
        'virtual_env': False,
        'dependencies': {}
    }

    print(f"✅ Python 版本: {sys.version.split()[0]}")
    print(f"✅ 当前工作目录: {Path.cwd()}")
    print(f"✅ Python 路径: {sys.executable}")

    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 运行在虚拟环境中")
        env_info['virtual_env'] = True
    else:
        print("⚠️ 未检测到虚拟环境")

    # 检查重要依赖
    dependencies = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'yaml': 'yaml',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm'
    }

    for dep_name, import_name in dependencies.items():
        try:
            if import_name == 'cv2':
                import cv2
                version = cv2.__version__
                print(f"✅ OpenCV 版本: {version}")
            elif import_name == 'PIL':
                from PIL import Image
                version = getattr(Image, 'VERSION', 'Unknown')
                print(f"✅ Pillow 版本: {version}")
            elif import_name == 'yaml':
                import yaml
                print(f"✅ PyYAML 可用")
                version = "Available"
            else:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {dep_name} 版本: {version}")

            env_info['dependencies'][dep_name] = version

        except ImportError:
            print(f"❌ {dep_name} 未安装")
            env_info['dependencies'][dep_name] = "Not installed"

    return env_info

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")

    test_results = {
        'model_init': False,
        'config_access': False,
        'import_test': False
    }

    try:
        # 测试模型初始化
        from ultralytics import YOLO
        model = YOLO('yolov8s.pt')
        print("✅ 模型初始化成功")
        test_results['model_init'] = True

        # 测试配置加载
        config_path = Path("data/visdrone_yolo/data.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ 数据集配置可访问")
            test_results['config_access'] = True
        else:
            print("⚠️ 数据集配置不可用")

        # 测试关键模块导入
        from ultralytics.models import YOLO as YOLOModel
        from ultralytics.utils import LOGGER
        print("✅ 关键模块导入成功")
        test_results['import_test'] = True

    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")

    return test_results

def generate_report(results):
    """生成验证报告"""
    print("\n📋 生成验证报告...")

    # 确保输出目录存在
    report_path = Path("outputs/verification_report.txt")
    report_path.parent.mkdir(exist_ok=True)

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLOvision Pro 本地 ultralytics 配置验证报告\n")
            f.write("="*60 + "\n")
            f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"工作目录: {Path.cwd()}\n")
            f.write(f"Python 版本: {sys.version.split()[0]}\n")
            f.write(f"Python 路径: {sys.executable}\n\n")

            # Ultralytics 信息
            f.write("Ultralytics 配置:\n")
            f.write("-" * 30 + "\n")
            ultralytics_info = results.get('ultralytics', {})
            f.write(f"版本: {ultralytics_info.get('version', 'N/A')}\n")
            f.write(f"路径: {ultralytics_info.get('path', 'N/A')}\n")
            f.write(f"本地开发版本: {'是' if ultralytics_info.get('is_local', False) else '否'}\n\n")

            # 模型信息
            f.write("YOLO 模型信息:\n")
            f.write("-" * 30 + "\n")
            model_info = results.get('model', {})
            f.write(f"加载状态: {'成功' if model_info.get('loaded', False) else '失败'}\n")
            f.write(f"类别数量: {model_info.get('num_classes', 'N/A')}\n")
            f.write(f"设备: {model_info.get('device', 'N/A')}\n\n")

            # 数据集信息
            f.write("VisDrone 数据集信息:\n")
            f.write("-" * 30 + "\n")
            dataset_info = results.get('dataset', {})
            f.write(f"配置状态: {'正常' if dataset_info.get('configured', False) else '异常'}\n")
            if 'stats' in dataset_info:
                stats = dataset_info['stats']
                f.write(f"总图像数: {stats.get('total_images', 0)}\n")
                f.write(f"总标签数: {stats.get('total_labels', 0)}\n")

                if 'splits' in stats:
                    for split, data in stats['splits'].items():
                        f.write(f"{split} 集: {data.get('images', 0)} 图像, {data.get('labels', 0)} 标签\n")
            f.write("\n")

            # 环境信息
            f.write("环境信息:\n")
            f.write("-" * 30 + "\n")
            env_info = results.get('environment', {})
            f.write(f"虚拟环境: {'是' if env_info.get('virtual_env', False) else '否'}\n")

            if 'dependencies' in env_info:
                f.write("依赖包状态:\n")
                for dep, version in env_info['dependencies'].items():
                    f.write(f"  {dep}: {version}\n")

            # 测试结果
            f.write("\n功能测试结果:\n")
            f.write("-" * 30 + "\n")
            test_info = results.get('tests', {})
            for test_name, result in test_info.items():
                f.write(f"{test_name}: {'通过' if result else '失败'}\n")

            # 总结
            f.write(f"\n总体状态: {'配置正常' if results.get('overall_status', False) else '需要修复'}\n")

        print(f"✅ 验证报告已保存: {report_path}")
        return True

    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        return False

def provide_troubleshooting_tips(results):
    """提供故障排除建议"""
    print("\n💡 故障排除建议:")
    print("-" * 30)

    # 检查 ultralytics 安装
    if not results.get('ultralytics', {}).get('is_local', False):
        print("🔧 Ultralytics 配置问题:")
        print("   1. 重新安装本地版本: pip install -e ./ultralytics")
        print("   2. 检查虚拟环境是否正确激活")
        print("   3. 确认项目目录结构正确")

    # 检查模型加载
    if not results.get('model', {}).get('loaded', False):
        print("🔧 模型加载问题:")
        print("   1. 检查网络连接（首次下载模型需要）")
        print("   2. 清理缓存: rm -rf ~/.cache/torch")
        print("   3. 手动下载模型文件")

    # 检查数据集配置
    if not results.get('dataset', {}).get('configured', False):
        print("🔧 数据集配置问题:")
        print("   1. 运行数据集处理: python scripts/process_visdrone_complete.py")
        print("   2. 检查数据集路径是否正确")
        print("   3. 验证数据集完整性: python scripts/validate_visdrone_dataset.py")

    # 检查环境
    if not results.get('environment', {}).get('virtual_env', False):
        print("🔧 环境配置问题:")
        print("   1. 激活虚拟环境: yolo8\\Scripts\\Activate.ps1")
        print("   2. 检查 Python 版本兼容性")
        print("   3. 重新安装依赖包")

def main():
    """主函数"""
    print_banner()

    # 存储所有检查结果
    results = {
        'ultralytics': {},
        'model': {},
        'dataset': {},
        'environment': {},
        'tests': {},
        'overall_status': False
    }

    # 执行各项检查
    print("开始执行配置验证...")

    # 1. 检查 ultralytics 安装
    is_local, version, path = check_ultralytics_installation()
    results['ultralytics'] = {
        'is_local': is_local,
        'version': version,
        'path': path
    }

    # 2. 检查 YOLO 类
    yolo_class_ok = check_yolo_class()

    # 3. 检查模型加载
    model_loaded, num_classes, device = check_yolo_model()
    results['model'] = {
        'loaded': model_loaded,
        'num_classes': num_classes,
        'device': device
    }

    # 4. 检查数据集
    dataset_ok, dataset_stats = check_visdrone_dataset()
    results['dataset'] = {
        'configured': dataset_ok,
        'stats': dataset_stats
    }

    # 5. 检查环境
    env_info = check_environment()
    results['environment'] = env_info

    # 6. 测试基本功能
    test_results = test_basic_functionality()
    results['tests'] = test_results

    # 计算总体状态
    critical_checks = [
        is_local,
        yolo_class_ok,
        model_loaded,
        dataset_ok,
        test_results.get('model_init', False)
    ]

    passed_checks = sum(critical_checks)
    total_checks = len(critical_checks)
    results['overall_status'] = passed_checks >= 4  # 至少4项通过

    # 生成报告
    generate_report(results)

    # 显示总结
    print("\n" + "="*60)
    print("📊 验证结果总结")
    print("="*60)

    check_items = [
        ("Ultralytics 本地安装", is_local),
        ("YOLO 类导入", yolo_class_ok),
        ("YOLO 模型加载", model_loaded),
        ("VisDrone 数据集", dataset_ok),
        ("基本功能测试", test_results.get('model_init', False))
    ]

    for item_name, status in check_items:
        status_text = "✅ 通过" if status else "❌ 失败"
        print(f"{item_name}: {status_text}")

    print(f"\n总计: {passed_checks}/{total_checks} 项检查通过")

    if results['overall_status']:
        print("\n🎉 恭喜！配置验证通过，可以开始使用 YOLOvision Pro 进行开发！")
        print("\n💡 下一步建议:")
        print("   1. 运行训练测试: yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=1")
        print("   2. 查看项目文档: docs/README.md")
        print("   3. 开始 Drone-YOLO 开发: docs/drone_yolo/README.md")
    else:
        print(f"\n⚠️ 配置验证未完全通过，有 {total_checks - passed_checks} 项需要修复")
        provide_troubleshooting_tips(results)

    print("="*60)
    print(f"📋 详细报告已保存至: outputs/verification_report.txt")
    print("="*60)

if __name__ == "__main__":
    main()
