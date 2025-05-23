#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone 数据集转换测试脚本
使用现有的 VisDrone 数据样本测试转换功能

作者: YOLOvision Pro Team
日期: 2024
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

def create_test_data():
    """创建测试数据"""
    print("📁 创建测试数据...")

    # 创建临时目录
    test_dir = Path("test_visdrone_temp")
    test_dir.mkdir(exist_ok=True)

    # 创建输入目录结构
    input_dir = test_dir / "input"
    images_dir = input_dir / "images"
    annotations_dir = input_dir / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否有现有的 VisDrone 数据
    visdrone_dir = Path("data/VisDrone2019-DET-train")
    if visdrone_dir.exists():
        print(f"✓ 找到现有 VisDrone 数据: {visdrone_dir}")

        # 复制几个样本文件进行测试
        visdrone_images = list((visdrone_dir / "images").glob("*.jpg"))[:3]
        visdrone_annotations = list((visdrone_dir / "annotations").glob("*.txt"))[:3]

        for img_file in visdrone_images:
            shutil.copy2(img_file, images_dir)
            print(f"  复制图像: {img_file.name}")

        for ann_file in visdrone_annotations:
            shutil.copy2(ann_file, annotations_dir)
            print(f"  复制标注: {ann_file.name}")
    else:
        print("⚠️ 未找到现有 VisDrone 数据，创建模拟数据...")

        # 创建模拟图像文件（空文件）
        test_images = ["test_001.jpg", "test_002.jpg", "test_003.jpg"]
        for img_name in test_images:
            (images_dir / img_name).touch()
            print(f"  创建模拟图像: {img_name}")

        # 创建模拟标注文件
        test_annotations = [
            ("test_001.txt", "100,50,80,120,1,4,0,0\n200,150,60,90,1,1,0,1\n"),
            ("test_002.txt", "50,30,40,60,1,3,0,0\n300,200,100,150,1,5,0,2\n"),
            ("test_003.txt", "150,100,70,110,1,2,0,1\n")
        ]

        for ann_name, content in test_annotations:
            with open(annotations_dir / ann_name, 'w') as f:
                f.write(content)
            print(f"  创建模拟标注: {ann_name}")

    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)

    return input_dir, output_dir, test_dir

def test_conversion(input_dir, output_dir):
    """测试转换功能"""
    print("\n🔄 测试格式转换...")

    # 导入转换模块
    visdrone_scripts_dir = Path(__file__).parent.parent / "data_processing" / "visdrone"
    sys.path.append(str(visdrone_scripts_dir))

    try:
        from convert_visdrone import VisDroneConverter

        # 创建转换器
        converter = VisDroneConverter(str(input_dir), str(output_dir))

        # 执行转换
        converter.convert_dataset()

        print("✅ 格式转换测试通过")
        return True

    except Exception as e:
        print(f"❌ 格式转换测试失败: {e}")
        return False

def test_splitting(output_dir):
    """测试数据集划分"""
    print("\n📊 测试数据集划分...")

    try:
        from split_visdrone_dataset import VisDroneDatasetSplitter

        # 创建划分器
        splitter = VisDroneDatasetSplitter(
            input_dir=str(output_dir),
            output_dir=str(output_dir),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # 执行划分
        splitter.split_dataset()

        print("✅ 数据集划分测试通过")
        return True

    except Exception as e:
        print(f"❌ 数据集划分测试失败: {e}")
        return False

def test_validation(output_dir):
    """测试数据集验证"""
    print("\n✅ 测试数据集验证...")

    try:
        from validate_visdrone_dataset import VisDroneDatasetValidator

        # 创建验证器
        validator = VisDroneDatasetValidator(str(output_dir))

        # 执行验证
        validator.validate_dataset()

        print("✅ 数据集验证测试通过")
        return True

    except Exception as e:
        print(f"❌ 数据集验证测试失败: {e}")
        return False

def check_output(output_dir):
    """检查输出结果"""
    print("\n📋 检查输出结果...")

    expected_files = [
        "data.yaml",
        "classes.txt",
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]

    all_exist = True
    for file_path in expected_files:
        full_path = output_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_exist = False

    return all_exist

def cleanup(test_dir):
    """清理测试数据"""
    print(f"\n🧹 清理测试数据: {test_dir}")
    try:
        shutil.rmtree(test_dir)
        print("✅ 清理完成")
    except Exception as e:
        print(f"⚠️ 清理失败: {e}")

def main():
    """主函数"""
    print("="*60)
    print("🧪 VisDrone 数据集转换工具测试")
    print("="*60)

    test_results = []
    test_dir = None

    try:
        # 创建测试数据
        input_dir, output_dir, test_dir = create_test_data()

        # 测试转换
        result1 = test_conversion(input_dir, output_dir)
        test_results.append(("格式转换", result1))

        if result1:
            # 测试划分
            result2 = test_splitting(output_dir)
            test_results.append(("数据集划分", result2))

            if result2:
                # 测试验证
                result3 = test_validation(output_dir)
                test_results.append(("数据集验证", result3))

                # 检查输出
                result4 = check_output(output_dir)
                test_results.append(("输出检查", result4))

        # 打印测试结果
        print("\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)

        passed = 0
        total = len(test_results)

        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
            if result:
                passed += 1

        print(f"\n总计: {passed}/{total} 测试通过")

        if passed == total:
            print("🎉 所有测试都通过了！VisDrone 转换工具工作正常。")
        else:
            print("⚠️ 部分测试失败，请检查错误信息。")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")

    finally:
        # 清理测试数据
        if test_dir and test_dir.exists():
            cleanup(test_dir)

    print("="*60)

if __name__ == "__main__":
    main()
