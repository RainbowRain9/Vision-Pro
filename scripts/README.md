# 🔧 YOLOvision Pro 脚本目录

本目录包含 YOLOvision Pro 项目的所有脚本文件，按功能分类组织，涵盖数据处理、模型训练、验证测试、可视化演示等各个方面。经过重新组织后，目录结构更加清晰，便于维护和使用。

## 📢 重要更新说明

**目录结构已重新组织！** 如果您之前使用过本项目，请注意以下变化：

### 🔄 主要变化
- **VisDrone 脚本** 从根目录移动到 `data_processing/visdrone/`
- **验证工具** 集中到 `validation/` 目录
- **通用工具** 移动到 `data_processing/general/`
- **文档文件** 移动到 `docs/` 目录
- **新增** 简化版检查脚本 `validation/simple_check.py`

### 📝 路径更新对照表
| 旧路径 | 新路径 |
|--------|--------|
| `scripts/convert_visdrone.py` | `scripts/data_processing/visdrone/convert_visdrone.py` |
| `scripts/process_visdrone_complete.py` | `scripts/data_processing/visdrone/process_visdrone_complete.py` |
| `scripts/verify_local_ultralytics.py` | `scripts/validation/verify_local_ultralytics.py` |
| `scripts/quick_check.py` | `scripts/validation/quick_check.py` |
| `scripts/labelme2yolo.py` | `scripts/data_processing/general/labelme2yolo.py` |
| `scripts/README_VisDrone.md` | `scripts/docs/VisDrone工具说明.md` |

### ✅ 兼容性说明
- 所有脚本功能保持不变
- 命令行参数和使用方法不变
- 只需要更新脚本路径即可
- **新增统一工具系统，提供更简洁的使用方式**

### 🔧 统一工具系统

为了解决脚本过多且分散的问题，我们新增了统一的工具管理系统：

#### 📋 三种使用方式

1. **🔧 统一工具入口** (`yolo_tools.py`)
   - 支持所有功能的子命令模式
   - 清晰的命令分类和参数传递
   - 完整的帮助系统

2. **📱 简化运行器** (`run.py`)
   - 交互式菜单界面
   - 快速命令行模式
   - 适合新用户和日常使用

3. **⚡ 快捷命令** (`quick_commands.py`)
   - 预设的常用操作组合
   - 一键执行复杂工作流程
   - 减少重复的参数输入

#### 🎯 推荐使用流程

```bash
# 1. 环境检查
python scripts/run.py check
# 或
python scripts/yolo_tools.py validation check

# 2. VisDrone 数据处理
python scripts/quick_commands.py visdrone-full
# 或
python scripts/run.py visdrone

# 3. 模型开发和测试
python scripts/quick_commands.py demo-all
# 或
python scripts/run.py demo
```

## 📁 目录结构

```
scripts/
├── README.md                          # 主说明文档
├── 🔧 yolo_tools.py                   # 统一工具入口（新增）
├── 📱 run.py                          # 简化运行器（新增）
├── ⚡ quick_commands.py               # 快捷命令（新增）
├── modules/                           # 功能模块（新增）
│   ├── __init__.py
│   ├── visdrone_module.py            # VisDrone 模块化接口
│   └── validation_module.py          # 验证模块化接口
├── data_processing/                   # 数据处理脚本
│   ├── README.md                      # 数据处理总览
│   ├── visdrone/                      # VisDrone2019 数据集专用工具
│   │   ├── README.md                  # VisDrone 工具详细说明
│   │   ├── convert_visdrone.py        # VisDrone 格式转换
│   │   ├── split_visdrone_dataset.py  # VisDrone 数据集划分
│   │   ├── validate_visdrone_dataset.py # VisDrone 数据集验证
│   │   └── process_visdrone_complete.py # VisDrone 一键处理
│   ├── general/                       # 通用数据处理工具
│   │   ├── README.md                  # 通用工具说明
│   │   ├── labelme2yolo.py           # LabelMe 转 YOLO 格式
│   │   └── split_dataset.py          # 通用数据集划分
│   └── demos/                         # 数据处理演示脚本
│       ├── README.md                  # 演示脚本说明
│       └── demo_visdrone_processing.py # VisDrone 处理演示
├── validation/                        # 验证和检查工具
│   ├── README.md                      # 验证工具总览
│   ├── verify_local_ultralytics.py   # 完整配置验证
│   ├── quick_check.py                 # 快速配置检查
│   ├── simple_check.py               # 简化版检查（避免编码问题）
│   ├── test_visdrone_conversion.py   # VisDrone 转换测试
│   └── run_verification.ps1          # PowerShell 验证脚本
├── training/                          # 训练相关脚本
│   └── README.md                      # 训练脚本说明
├── testing/                           # 测试脚本
│   ├── README.md                      # 测试脚本说明
│   └── test_drone_yolo.py            # Drone-YOLO 功能测试
├── demo/                              # 演示脚本
│   ├── README.md                      # 演示脚本说明
│   └── drone_yolo_demo.py            # Drone-YOLO 核心演示
├── visualization/                     # 可视化脚本
│   ├── README.md                      # 可视化工具说明
│   └── visualize_drone_yolo.py       # Drone-YOLO 架构可视化
└── docs/                              # 脚本相关文档
    ├── README.md                      # 文档总览
    ├── VisDrone工具说明.md           # VisDrone 工具详细文档
    └── 验证工具说明.md               # 验证工具详细文档
```

### 📊 data_processing/ - 数据处理脚本
数据处理脚本，用于数据集转换、预处理和验证

#### 子目录说明：
- **visdrone/** - VisDrone2019 数据集专用处理工具
  - `convert_visdrone.py` - VisDrone 格式转换为 YOLO 格式
  - `split_visdrone_dataset.py` - 数据集按 8:1:1 比例划分
  - `validate_visdrone_dataset.py` - 数据集验证和统计分析
  - `process_visdrone_complete.py` - 一键完整处理流程
- **general/** - 通用数据处理工具
  - `labelme2yolo.py` - LabelMe 标注格式转换为 YOLO 格式
  - `split_dataset.py` - 通用数据集划分工具
- **demos/** - 数据处理功能演示脚本
  - `demo_visdrone_processing.py` - VisDrone 处理工具演示

### ✅ validation/ - 验证和检查工具
验证和检查工具，确保环境配置和数据处理结果正确

#### 主要工具：
- `verify_local_ultralytics.py` - 完整的本地 ultralytics 配置验证
- `quick_check.py` - 快速配置检查（包含 emoji，可能有编码问题）
- `simple_check.py` - 简化版检查（避免编码问题，推荐使用）
- `test_visdrone_conversion.py` - VisDrone 转换功能测试
- `run_verification.ps1` - PowerShell 自动化验证脚本（Windows）

### 🚀 training/ - 训练相关脚本
训练相关脚本（规划中，部分功能待实现）

#### 计划功能：
- Drone-YOLO 模型训练脚本
- 模型评估和性能测试
- 超参数自动调优
- 训练过程监控和可视化

### 🧪 testing/ - 测试脚本
测试脚本，用于验证模型和组件的正确性

#### 现有测试：
- `test_drone_yolo.py` - Drone-YOLO 模型功能测试
- 单元测试和集成测试（规划中）
- 性能基准测试（规划中）

### 🎭 demo/ - 演示脚本
演示脚本，用于展示模型功能和核心概念

#### 演示内容：
- `drone_yolo_demo.py` - Drone-YOLO 核心概念演示
- 交互式功能演示（规划中）

### 📊 visualization/ - 可视化脚本
可视化脚本，用于生成图表和可视化结果

#### 可视化功能：
- `visualize_drone_yolo.py` - Drone-YOLO 架构可视化
- 训练结果图表绘制（规划中）
- 模型分析可视化（规划中）

### 📚 docs/ - 脚本相关文档
脚本相关文档和使用说明

#### 文档内容：
- `VisDrone工具说明.md` - VisDrone 数据集处理工具详细说明
- `验证工具说明.md` - 配置验证工具使用指南
- 技术文档和最佳实践

## 🎯 快速开始

### ⚡ 统一工具入口（推荐）

**新增统一工具系统！** 现在可以通过统一入口访问所有功能：

```bash
# 🔧 统一工具入口 - 支持所有功能的子命令模式
python scripts/yolo_tools.py <command> <subcommand> [options]

# 📱 简化运行器 - 交互式菜单或快速命令
python scripts/run.py                    # 交互式菜单
python scripts/run.py check              # 快速环境检查
python scripts/run.py visdrone           # 快速 VisDrone 处理

# ⚡ 快捷命令 - 预设的常用操作组合
python scripts/quick_commands.py setup           # 环境初始化
python scripts/quick_commands.py visdrone-full   # VisDrone 完整处理
```

### 🔧 环境检查（推荐首先执行）

**新方式（推荐）：**
```bash
# 统一工具方式
python scripts/yolo_tools.py validation check    # 简化检查
python scripts/yolo_tools.py validation quick    # 快速检查
python scripts/yolo_tools.py validation full     # 完整验证

# 快捷命令方式
python scripts/quick_commands.py setup           # 环境初始化
python scripts/quick_commands.py check-all       # 完整系统检查

# 简化运行器方式
python scripts/run.py check                      # 快速检查
```

**传统方式（仍然支持）：**
```bash
# 直接调用脚本（向后兼容）
python scripts/validation/simple_check.py
python scripts/validation/quick_check.py
python scripts/validation/verify_local_ultralytics.py

# PowerShell 自动化验证（Windows）
.\scripts\validation\run_verification.ps1 -Mode full
```

### 📊 VisDrone 数据处理工作流程

**新方式（推荐）：**
```bash
# 🔧 统一工具方式
python scripts/yolo_tools.py visdrone process \
    --input data/VisDrone2019-DET-train \
    --output data/visdrone_yolo \
    --verbose

# ⚡ 快捷命令方式
python scripts/quick_commands.py visdrone-full    # 完整处理（含可视化）
python scripts/quick_commands.py visdrone-quick   # 快速处理（无可视化）

# 📱 简化运行器方式
python scripts/run.py visdrone                    # 交互式处理
python scripts/run.py visdrone data/input data/output  # 命令行处理

# 🔧 分步处理（统一工具）
python scripts/yolo_tools.py visdrone convert --input data/VisDrone2019-DET-train --output data/visdrone_yolo
python scripts/yolo_tools.py visdrone split --input data/visdrone_yolo --output data/visdrone_yolo
python scripts/yolo_tools.py visdrone validate --dataset data/visdrone_yolo --visualize

# 🎭 查看演示
python scripts/yolo_tools.py visdrone demo
```

**传统方式（仍然支持）：**
```bash
# 方法1: 一键处理
python scripts/data_processing/visdrone/process_visdrone_complete.py \
    --input data/VisDrone2019-DET-train \
    --output data/visdrone_yolo \
    --verbose

# 方法2: 分步处理
python scripts/data_processing/visdrone/convert_visdrone.py \
    -i data/VisDrone2019-DET-train -o data/visdrone_yolo
python scripts/data_processing/visdrone/split_visdrone_dataset.py \
    -i data/visdrone_yolo -o data/visdrone_yolo
python scripts/data_processing/visdrone/validate_visdrone_dataset.py \
    -d data/visdrone_yolo --visualize

# 查看处理演示
python scripts/data_processing/demos/demo_visdrone_processing.py
```

### 🚀 模型开发工作流程

**新方式（推荐）：**
```bash
# 🔧 统一工具方式
python scripts/yolo_tools.py demo test-model        # Drone-YOLO 测试
python scripts/yolo_tools.py viz architecture       # 架构可视化
python scripts/yolo_tools.py demo drone-yolo        # 核心概念演示

# ⚡ 快捷命令方式
python scripts/quick_commands.py demo-all           # 运行所有演示

# 📱 简化运行器方式
python scripts/run.py demo                          # 交互式演示菜单
```

**传统方式（仍然支持）：**
```bash
# 1. 运行 Drone-YOLO 测试
python scripts/testing/test_drone_yolo.py

# 2. 生成架构可视化
python scripts/visualization/visualize_drone_yolo.py

# 3. 运行核心概念演示
python scripts/demo/drone_yolo_demo.py

# 4. 开始模型训练（使用 VisDrone 数据集）
yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=100
```

### 🔧 通用数据处理

**新方式（推荐）：**
```bash
# 🔧 统一工具方式
python scripts/yolo_tools.py data labelme2yolo      # LabelMe 转 YOLO
python scripts/yolo_tools.py data split             # 通用数据集划分
```

**传统方式（仍然支持）：**
```bash
# LabelMe 转 YOLO 格式
python scripts/data_processing/general/labelme2yolo.py

# 通用数据集划分
python scripts/data_processing/general/split_dataset.py
```

## 📋 脚本开发规范

- 每个脚本都应包含详细的文档字符串和中文注释
- 使用 argparse 处理命令行参数，提供 --help 选项
- 包含完善的错误处理和日志记录
- 提供使用示例和故障排除指南
- 遵循项目的代码风格和命名规范

## 🔗 相关目录

- [📚 项目文档](../docs/README.md) - 查看完整技术文档
- [🎯 Drone-YOLO 文档](../docs/drone_yolo/README.md) - Drone-YOLO 专项文档
- [⚙️ 配置文件](../assets/configs/) - 模型和训练配置
- [📊 输出结果](../outputs/README.md) - 查看运行结果和报告

## 💡 使用建议

### 🆕 新用户入门
1. **快速开始**: 运行 `python scripts/run.py` 进入交互式菜单
2. **环境检查**: 使用 `python scripts/run.py check` 或 `python scripts/yolo_tools.py validation check`
3. **了解功能**: 运行 `python scripts/quick_commands.py demo-all` 查看所有演示
4. **阅读文档**: 查看 `scripts/docs/` 目录下的详细文档
5. **实践操作**: 使用 `python scripts/quick_commands.py visdrone-full` 体验完整流程

### 👨‍💻 日常开发
1. **统一入口**: 优先使用 `python scripts/yolo_tools.py` 统一工具
2. **快捷操作**: 使用 `python scripts/quick_commands.py` 执行常用组合操作
3. **交互模式**: 使用 `python scripts/run.py` 进行交互式操作
4. **传统方式**: 需要时仍可直接调用具体脚本文件

### 🔄 迁移指南
**从传统方式迁移到统一工具系统：**

| 传统方式 | 统一工具方式 | 快捷命令方式 |
|----------|-------------|-------------|
| `python scripts/validation/simple_check.py` | `python scripts/yolo_tools.py validation check` | `python scripts/run.py check` |
| `python scripts/data_processing/visdrone/process_visdrone_complete.py` | `python scripts/yolo_tools.py visdrone process` | `python scripts/quick_commands.py visdrone-full` |
| `python scripts/demo/drone_yolo_demo.py` | `python scripts/yolo_tools.py demo drone-yolo` | `python scripts/quick_commands.py demo-all` |

### 🔧 问题排查
1. **查看文档**: 检查 `scripts/docs/` 目录下的详细文档
2. **运行验证**: 使用验证脚本诊断问题
   ```bash
   # 简化版检查
   python scripts/validation/simple_check.py

   # 完整验证
   python scripts/validation/verify_local_ultralytics.py
   ```
3. **检查日志**: 查看 `outputs/` 目录下的日志和报告
4. **测试功能**: 运行相应的测试脚本确认功能状态

### 📚 推荐学习路径
1. **基础配置** → 运行环境检查脚本
2. **数据处理** → 学习 VisDrone 数据集处理流程
3. **模型开发** → 了解 Drone-YOLO 架构和训练
4. **高级功能** → 探索可视化和自定义脚本开发

## ❓ 常见问题解答

### Q1: 为什么要重新组织目录结构？
**A:** 随着项目发展，脚本数量增加，原来的平铺结构不便于管理。新的分类结构使得：
- 相关功能的脚本集中在一起
- 更容易找到需要的工具
- 便于添加新功能和维护
- 提高团队协作效率

### Q2: 重组后我的旧脚本路径失效了怎么办？
**A:** 请参考上面的"路径更新对照表"，将旧路径替换为新路径即可。所有脚本的功能和参数都没有变化。

### Q3: `quick_check.py` 和 `simple_check.py` 有什么区别？
**A:**
- `scripts/validation/quick_check.py`: 功能完整但包含 emoji 字符，在某些环境下可能有编码问题
- `scripts/validation/simple_check.py`: 简化版本，避免编码问题，推荐在有编码问题的环境中使用

### Q4: 如何选择合适的验证脚本？
**A:** 建议按以下顺序尝试：
1. `scripts/validation/simple_check.py` - 快速检查，避免编码问题
2. `scripts/validation/quick_check.py` - 功能更完整的快速检查
3. `scripts/validation/verify_local_ultralytics.py` - 最详细的完整验证

### Q5: VisDrone 数据集处理失败怎么办？
**A:**
1. 首先运行环境检查脚本确认配置正确
2. 查看 `scripts/docs/VisDrone工具说明.md` 详细文档
3. 运行演示脚本了解正确的使用方法
4. 检查输入数据格式和路径是否正确

### Q6: 如何贡献新的脚本？
**A:**
1. 根据功能选择合适的子目录
2. 遵循项目的代码规范和命名约定
3. 添加详细的文档字符串和中文注释
4. 在对应的 README.md 中添加说明
5. 提供使用示例和测试用例

### Q7: 训练脚本目录为什么是空的？
**A:** `training/` 目录是为未来的训练脚本预留的。目前可以使用标准的 YOLO 训练命令：
```bash
yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=100
```

## 📞 技术支持

如果遇到问题，请：
1. 查看相应目录下的 README.md 文档
2. 运行验证脚本诊断问题
3. 查看 `outputs/` 目录下的日志文件
4. 参考 `scripts/docs/` 目录下的详细文档

---

**YOLOvision Pro Team**
*专业的目标检测开发工具链*
