#!/usr/bin/env python3
"""
测试 Drone-YOLO 模型的脚本
"""

import sys
import os
import torch

# 添加 ultralytics 到路径
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))

try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.block import RepVGGBlock
    print("✅ 成功导入 ultralytics 模块")
except ImportError as e:
    print(f"❌ 导入 ultralytics 模块失败: {e}")
    sys.exit(1)

def test_repvgg_block():
    """测试 RepVGGBlock 模块"""
    print("\n🔧 测试 RepVGGBlock 模块...")
    try:
        # 创建一个简单的 RepVGGBlock
        block = RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)

        # 创建测试输入
        x = torch.randn(1, 64, 32, 32)

        # 前向传播
        output = block(x)
        print(f"✅ RepVGGBlock 测试成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")

        return True
    except Exception as e:
        print(f"❌ RepVGGBlock 测试失败: {e}")
        return False

def test_drone_yolo_model():
    """测试 Drone-YOLO 模型"""
    print("\n🚀 测试 Drone-YOLO 模型...")
    try:
        # 模型配置文件路径
        model_config = "ultralytics/ultralytics/cfg/models/v8/yolov8s-drone.yaml"

        if not os.path.exists(model_config):
            print(f"❌ 模型配置文件不存在: {model_config}")
            return False

        # 创建模型
        model = DetectionModel(model_config, ch=3, nc=80, verbose=True)
        print(f"✅ Drone-YOLO 模型创建成功")

        # 创建测试输入
        x = torch.randn(1, 3, 640, 640)

        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"✅ Drone-YOLO 模型前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出数量: {len(output)}")
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                print(f"   输出 {i} 形状: {out.shape}")
            else:
                print(f"   输出 {i} 类型: {type(out)}, 长度: {len(out) if hasattr(out, '__len__') else 'N/A'}")

        return True
    except Exception as e:
        print(f"❌ Drone-YOLO 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🎯 开始测试 Drone-YOLO 实现...")

    # 测试 RepVGGBlock
    repvgg_success = test_repvgg_block()

    # 测试 Drone-YOLO 模型
    model_success = test_drone_yolo_model()

    # 总结
    print("\n📊 测试结果总结:")
    print(f"   RepVGGBlock: {'✅ 通过' if repvgg_success else '❌ 失败'}")
    print(f"   Drone-YOLO 模型: {'✅ 通过' if model_success else '❌ 失败'}")

    if repvgg_success and model_success:
        print("\n🎉 所有测试通过！Drone-YOLO 实现成功！")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
