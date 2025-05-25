# 功能模块

本目录包含 YOLOvision Pro 脚本系统的模块化接口，提供统一的 Python API 来调用各种功能。

## 📋 模块概述

### 🚁 VisDrone 模块 (`visdrone_module.py`)
提供 VisDrone 数据集处理的统一接口：

```python
from modules.visdrone_module import VisDroneModule

# 创建模块实例
visdrone = VisDroneModule()

# 转换数据集
result = visdrone.convert_dataset(
    input_dir="data/VisDrone2019-DET-train",
    output_dir="data/visdrone_yolo",
    verbose=True
)

# 完整处理流程
result = visdrone.process_complete(
    input_dir="data/VisDrone2019-DET-train",
    output_dir="data/visdrone_yolo",
    verbose=True,
    no_visualization=False
)
```

### ✅ 验证模块 (`validation_module.py`)
提供环境验证和配置检查的统一接口：

```python
from modules.validation_module import ValidationModule

# 创建模块实例
validation = ValidationModule()

# 简化检查
result = validation.simple_check()

# 运行所有检查
results = validation.run_all_checks()
print(f"通过率: {results['summary']['pass_rate']:.1%}")
```

## 🔧 便捷函数

每个模块都提供了便捷函数，可以直接导入使用：

```python
# VisDrone 便捷函数
from modules.visdrone_module import convert_visdrone, process_visdrone_complete

success = convert_visdrone("input_dir", "output_dir", verbose=True)
success = process_visdrone_complete("input_dir", "output_dir")

# 验证便捷函数
from modules.validation_module import simple_environment_check, validate_all_systems

success = simple_environment_check()
results = validate_all_systems()
```

## 🎯 设计原则

1. **统一接口**: 所有模块提供一致的 API 设计
2. **向后兼容**: 不影响现有脚本的独立使用
3. **错误处理**: 统一的错误处理和结果返回格式
4. **易于扩展**: 便于添加新的功能模块

## 📊 返回格式

所有模块方法都返回统一的结果格式：

```python
{
    'success': bool,           # 操作是否成功
    'returncode': int,         # 返回码
    'stdout': str,             # 标准输出（如果有）
    'stderr': str,             # 错误输出（如果有）
    'error': str               # 错误信息（如果有）
}
```

## 🔗 相关工具

- [统一工具入口](../yolo_tools.py) - 命令行统一接口
- [简化运行器](../run.py) - 交互式界面
- [快捷命令](../quick_commands.py) - 预设操作组合

## 💡 使用建议

1. **Python 脚本开发**: 使用模块化接口进行二次开发
2. **自动化流程**: 集成到自动化脚本中
3. **批量处理**: 处理多个数据集或执行批量操作
4. **状态监控**: 获取详细的执行状态和结果信息
