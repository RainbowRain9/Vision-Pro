#!/usr/bin/env python3
"""
Drone-YOLO 模型结构可视化脚本
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 添加 ultralytics 到路径
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))

def visualize_repvgg_structure():
    """可视化 RepVGGBlock 的训练和推理结构"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 训练时的多分支结构
    ax1.set_title("RepVGGBlock - 训练时 (多分支)", fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    
    # 输入
    input_box = FancyBboxPatch((1, 6), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black')
    ax1.add_patch(input_box)
    ax1.text(2, 6.5, 'Input\nFeature', ha='center', va='center', fontweight='bold')
    
    # 三个分支
    # 3x3 分支
    conv3x3_box = FancyBboxPatch((0.5, 4), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black')
    ax1.add_patch(conv3x3_box)
    ax1.text(1.5, 4.5, '3x3 Conv\n+ BN', ha='center', va='center')
    
    # 1x1 分支  
    conv1x1_box = FancyBboxPatch((3, 4), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightcoral', edgecolor='black')
    ax1.add_patch(conv1x1_box)
    ax1.text(4, 4.5, '1x1 Conv\n+ BN', ha='center', va='center')
    
    # Identity 分支
    identity_box = FancyBboxPatch((5.5, 4), 2, 1, boxstyle="round,pad=0.1",
                                  facecolor='lightyellow', edgecolor='black')
    ax1.add_patch(identity_box)
    ax1.text(6.5, 4.5, 'Identity\n+ BN', ha='center', va='center')
    
    # 加法操作
    add_box = FancyBboxPatch((3, 2), 2, 1, boxstyle="round,pad=0.1",
                             facecolor='lightgray', edgecolor='black')
    ax1.add_patch(add_box)
    ax1.text(4, 2.5, 'Add', ha='center', va='center', fontweight='bold')
    
    # 输出
    output_box = FancyBboxPatch((3, 0.5), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor='black')
    ax1.add_patch(output_box)
    ax1.text(4, 1, 'Output\nFeature', ha='center', va='center', fontweight='bold')
    
    # 连接线
    ax1.arrow(2, 6, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax1.arrow(1.5, 5.8, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax1.arrow(2, 5.8, 1.8, -1.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax1.arrow(2, 5.8, 4.3, -1.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax1.arrow(1.5, 4, 1.3, -1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax1.arrow(4, 4, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax1.arrow(6.5, 4, -1.3, -1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax1.arrow(4, 2, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    
    # 推理时的单分支结构
    ax2.set_title("RepVGGBlock - 推理时 (融合后)", fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    
    # 输入
    input_box2 = FancyBboxPatch((4, 6), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor='black')
    ax2.add_patch(input_box2)
    ax2.text(5, 6.5, 'Input\nFeature', ha='center', va='center', fontweight='bold')
    
    # 融合的3x3卷积
    fused_conv_box = FancyBboxPatch((4, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                                    facecolor='orange', edgecolor='black')
    ax2.add_patch(fused_conv_box)
    ax2.text(5, 4.25, 'Fused\n3x3 Conv\n(with bias)', ha='center', va='center', fontweight='bold')
    
    # 输出
    output_box2 = FancyBboxPatch((4, 1), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightblue', edgecolor='black')
    ax2.add_patch(output_box2)
    ax2.text(5, 1.5, 'Output\nFeature', ha='center', va='center', fontweight='bold')
    
    # 连接线
    ax2.arrow(5, 6, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(5, 3.5, 0, -1.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('repvgg_structure.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_detection_layers():
    """可视化不同检测层的特征图尺寸和感受野"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 输入图像
    input_rect = patches.Rectangle((1, 7), 8, 1, linewidth=2, edgecolor='black', facecolor='lightblue')
    ax.add_patch(input_rect)
    ax.text(5, 7.5, 'Input Image: 640×640×3', ha='center', va='center', fontweight='bold', fontsize=12)
    
    # 不同检测层
    layers = [
        {'name': 'P2', 'size': '160×160', 'targets': '4-16px', 'color': 'lightgreen', 'y': 5.5},
        {'name': 'P3', 'size': '80×80', 'targets': '8-32px', 'color': 'lightcoral', 'y': 4.5},
        {'name': 'P4', 'size': '40×40', 'targets': '16-64px', 'color': 'lightyellow', 'y': 3.5},
        {'name': 'P5', 'size': '20×20', 'targets': '32px+', 'color': 'lightgray', 'y': 2.5}
    ]
    
    for layer in layers:
        # 特征图框
        rect = patches.Rectangle((1, layer['y']), 2, 0.8, linewidth=2, 
                                edgecolor='black', facecolor=layer['color'])
        ax.add_patch(rect)
        ax.text(2, layer['y']+0.4, layer['name'], ha='center', va='center', fontweight='bold')
        
        # 特征图尺寸
        ax.text(4, layer['y']+0.4, f"特征图: {layer['size']}", ha='left', va='center')
        
        # 适合的目标尺寸
        ax.text(6.5, layer['y']+0.4, f"目标尺寸: {layer['targets']}", ha='left', va='center')
        
        # 连接线
        ax.arrow(3, layer['y']+0.4, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 输出统计
    output_rect = patches.Rectangle((1, 1), 8, 1, linewidth=2, edgecolor='black', facecolor='lightsteelblue')
    ax.add_patch(output_rect)
    ax.text(5, 1.5, 'Total Detection Points: 34,000\n(25,600 + 6,400 + 1,600 + 400)', 
            ha='center', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_title('Drone-YOLO 多尺度检测层分析', fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_layers.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sandwich_fusion():
    """可视化三明治融合结构"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 以P4层融合为例
    ax.set_title('三明治融合结构 (以P4层为例)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # P5层 (深层特征)
    p5_box = FancyBboxPatch((1, 8), 2, 1, boxstyle="round,pad=0.1",
                            facecolor='lightcoral', edgecolor='black')
    ax.add_patch(p5_box)
    ax.text(2, 8.5, 'P5 Features\n20×20×512', ha='center', va='center', fontweight='bold')
    
    # 上采样
    upsample_box = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1",
                                  facecolor='lightblue', edgecolor='black')
    ax.add_patch(upsample_box)
    ax.text(5, 8.5, 'Upsample\n40×40×512', ha='center', va='center')
    
    # P4层 (当前层特征)
    p4_box = FancyBboxPatch((4, 6), 2, 1, boxstyle="round,pad=0.1",
                            facecolor='lightgreen', edgecolor='black')
    ax.add_patch(p4_box)
    ax.text(5, 6.5, 'P4 Backbone\n40×40×256', ha='center', va='center', fontweight='bold')
    
    # P3层 (浅层特征)
    p3_box = FancyBboxPatch((1, 4), 2, 1, boxstyle="round,pad=0.1",
                            facecolor='lightyellow', edgecolor='black')
    ax.add_patch(p3_box)
    ax.text(2, 4.5, 'P3 Features\n80×80×128', ha='center', va='center', fontweight='bold')
    
    # DWConv下采样
    dwconv_box = FancyBboxPatch((4, 4), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='lightsteelblue', edgecolor='black')
    ax.add_patch(dwconv_box)
    ax.text(5, 4.5, 'DWConv\n40×40×128', ha='center', va='center')
    
    # Concat融合
    concat_box = FancyBboxPatch((8, 6), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='orange', edgecolor='black')
    ax.add_patch(concat_box)
    ax.text(9, 6.5, 'Concat\n40×40×896', ha='center', va='center', fontweight='bold')
    
    # C2f处理
    c2f_box = FancyBboxPatch((8, 4), 2, 1, boxstyle="round,pad=0.1",
                             facecolor='lightpink', edgecolor='black')
    ax.add_patch(c2f_box)
    ax.text(9, 4.5, 'C2f\n40×40×256', ha='center', va='center', fontweight='bold')
    
    # 最终输出
    output_box = FancyBboxPatch((8, 2), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='lightcyan', edgecolor='black')
    ax.add_patch(output_box)
    ax.text(9, 2.5, 'P4 Neck\nOutput', ha='center', va='center', fontweight='bold')
    
    # 连接线
    # P5 -> Upsample
    ax.arrow(3, 8.5, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    # P3 -> DWConv  
    ax.arrow(3, 4.5, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    # 三个输入到Concat
    ax.arrow(6, 8.5, 1.8, -1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 6.5, 1.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 4.5, 1.8, 1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    # Concat -> C2f -> Output
    ax.arrow(9, 6, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(9, 4, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 添加说明文字
    ax.text(0.5, 9, '深层特征 (语义丰富)', ha='left', va='center', fontsize=12, style='italic')
    ax.text(0.5, 7, '当前层特征 (主要信息)', ha='left', va='center', fontsize=12, style='italic')
    ax.text(0.5, 3, '浅层特征 (细节丰富)', ha='left', va='center', fontsize=12, style='italic')
    
    ax.text(10.5, 7, '三明治融合:\n上层 + 中层 + 下层', ha='center', va='center', 
            fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sandwich_fusion.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🎨 生成 Drone-YOLO 可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        print("1. 生成 RepVGGBlock 结构图...")
        visualize_repvgg_structure()
        
        print("2. 生成检测层分析图...")
        visualize_detection_layers()
        
        print("3. 生成三明治融合结构图...")
        visualize_sandwich_fusion()
        
        print("✅ 所有可视化图表生成完成！")
        print("📁 生成的文件:")
        print("   - repvgg_structure.png")
        print("   - detection_layers.png") 
        print("   - sandwich_fusion.png")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        print("💡 提示: 请确保安装了 matplotlib: pip install matplotlib")

if __name__ == "__main__":
    main()
