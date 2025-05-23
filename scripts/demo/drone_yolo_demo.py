#!/usr/bin/env python3
"""
Drone-YOLO 核心概念演示脚本
通过实际代码展示各个组件的工作原理
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# 添加 ultralytics 到路径
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))

from ultralytics.nn.modules.block import RepVGGBlock, conv_bn
from ultralytics.nn.modules.conv import DWConv, Conv, Concat

def demo_repvgg_fusion():
    """演示 RepVGGBlock 的训练和推理模式"""
    print("🔧 RepVGGBlock 融合演示")
    print("=" * 50)
    
    # 创建 RepVGGBlock
    block = RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, deploy=False)
    
    # 创建测试输入
    x = torch.randn(1, 64, 32, 32)
    print(f"输入形状: {x.shape}")
    
    # 训练模式前向传播
    block.train()
    with torch.no_grad():
        output_train = block(x)
    print(f"训练模式输出形状: {output_train.shape}")
    
    # 检查训练时的分支
    print(f"\n训练时的分支结构:")
    print(f"  - rbr_dense (3x3): {block.rbr_dense}")
    print(f"  - rbr_1x1 (1x1): {block.rbr_1x1}")
    print(f"  - rbr_identity: {block.rbr_identity}")
    
    # 切换到推理模式
    print(f"\n🔄 切换到推理模式...")
    block.switch_to_deploy()
    
    # 推理模式前向传播
    block.eval()
    with torch.no_grad():
        output_deploy = block(x)
    print(f"推理模式输出形状: {output_deploy.shape}")
    
    # 检查推理时的结构
    print(f"\n推理时的结构:")
    print(f"  - rbr_reparam (融合后): {hasattr(block, 'rbr_reparam')}")
    print(f"  - 原分支已删除: {not hasattr(block, 'rbr_dense')}")
    
    # 验证输出一致性
    diff = torch.abs(output_train - output_deploy).max().item()
    print(f"\n✅ 输出差异: {diff:.8f} (应该非常小)")
    
    return block

def demo_detection_layers():
    """演示不同检测层的输出计算"""
    print("\n🎯 检测层输出计算演示")
    print("=" * 50)
    
    # 模拟不同尺度的特征图
    feature_maps = {
        'P2': torch.randn(1, 64, 160, 160),   # 1/4 下采样
        'P3': torch.randn(1, 128, 80, 80),    # 1/8 下采样  
        'P4': torch.randn(1, 256, 40, 40),    # 1/16 下采样
        'P5': torch.randn(1, 512, 20, 20),    # 1/32 下采样
    }
    
    total_locations = 0
    
    for name, feature_map in feature_maps.items():
        b, c, h, w = feature_map.shape
        locations = h * w
        total_locations += locations
        
        print(f"{name}: {h}×{w} = {locations:,} 个检测位置")
        print(f"     特征维度: {c} 通道")
        
        # 计算适合的目标尺寸范围
        if name == 'P2':
            target_range = "4-16 像素"
        elif name == 'P3':
            target_range = "8-32 像素"
        elif name == 'P4':
            target_range = "16-64 像素"
        else:  # P5
            target_range = "32+ 像素"
        
        print(f"     适合目标: {target_range}")
        print()
    
    print(f"📊 总检测位置: {total_locations:,}")
    print(f"📊 每个位置输出: 84 (4坐标 + 80类别)")
    print(f"📊 最终输出形状: [1, 84, {total_locations}]")

def demo_sandwich_fusion():
    """演示三明治融合的具体过程"""
    print("\n🥪 三明治融合演示")
    print("=" * 50)
    
    # 模拟P4层融合过程
    print("以P4层融合为例:")
    
    # 三个输入源
    P5_features = torch.randn(1, 512, 20, 20)    # 来自更深层
    P4_backbone = torch.randn(1, 256, 40, 40)    # 当前层主干特征
    P3_features = torch.randn(1, 128, 80, 80)    # 来自更浅层
    
    print(f"P5 特征: {P5_features.shape}")
    print(f"P4 主干: {P4_backbone.shape}")
    print(f"P3 特征: {P3_features.shape}")
    
    # 1. P5上采样
    upsample = nn.Upsample(scale_factor=2, mode='nearest')
    P5_upsampled = upsample(P5_features)
    print(f"\n1. P5 上采样后: {P5_upsampled.shape}")
    
    # 2. P3下采样 (使用DWConv)
    dwconv = DWConv(128, 128, k=3, s=2)  # 深度可分离卷积
    P3_downsampled = dwconv(P3_features)
    print(f"2. P3 DWConv下采样后: {P3_downsampled.shape}")
    
    # 3. 三明治融合 (Concat)
    concat = Concat(dim=1)
    fused_features = concat([P5_upsampled, P4_backbone, P3_downsampled])
    print(f"3. 融合后特征: {fused_features.shape}")
    
    # 计算通道数
    total_channels = P5_upsampled.shape[1] + P4_backbone.shape[1] + P3_downsampled.shape[1]
    print(f"   通道计算: {P5_upsampled.shape[1]} + {P4_backbone.shape[1]} + {P3_downsampled.shape[1]} = {total_channels}")
    
    # 4. 最终处理 (模拟C2f)
    final_conv = Conv(total_channels, 256, k=1)  # 1x1卷积降维
    final_output = final_conv(fused_features)
    print(f"4. 最终输出: {final_output.shape}")
    
    print(f"\n✅ 三明治融合完成！")
    print(f"💡 优势: 结合了深层语义 + 当前层信息 + 浅层细节")

def demo_parameter_analysis():
    """分析模型参数分布"""
    print("\n📊 参数分布分析")
    print("=" * 50)
    
    # 创建各个组件并计算参数量
    components = {}
    
    # RepVGGBlock (4个)
    repvgg_params = 0
    for i, (in_ch, out_ch) in enumerate([(32, 64), (64, 128), (128, 256), (256, 512)]):
        block = RepVGGBlock(in_ch, out_ch, stride=2)
        params = sum(p.numel() for p in block.parameters())
        repvgg_params += params
        print(f"RepVGGBlock {i+1} ({in_ch}→{out_ch}): {params:,} 参数")
    
    components['RepVGGBlock'] = repvgg_params
    
    # 模拟其他组件的参数量 (基于实际模型输出)
    components['C2f模块'] = 6_500_000  # 约6.5M
    components['检测头'] = 1_700_000   # 约1.7M  
    components['其他'] = 1_100_000     # 约1.1M
    
    total_params = sum(components.values())
    
    print(f"\n📈 参数分布:")
    for name, params in components.items():
        percentage = (params / total_params) * 100
        print(f"  {name}: {params:,} ({percentage:.1f}%)")
    
    print(f"\n🎯 总参数量: {total_params:,}")
    print(f"🎯 实际模型: 11,084,080 参数")

def main():
    """主演示函数"""
    print("🚁 Drone-YOLO 核心概念演示")
    print("=" * 60)
    
    try:
        # 1. RepVGGBlock 融合演示
        demo_repvgg_fusion()
        
        # 2. 检测层演示
        demo_detection_layers()
        
        # 3. 三明治融合演示
        demo_sandwich_fusion()
        
        # 4. 参数分析
        demo_parameter_analysis()
        
        print(f"\n🎉 所有演示完成！")
        print(f"💡 这些演示展示了 Drone-YOLO 的核心创新点:")
        print(f"   1. RepVGG: 训练时复杂，推理时简单")
        print(f"   2. P2检测头: 专门处理小目标")
        print(f"   3. 三明治融合: 多尺度信息整合")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
