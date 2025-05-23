# 验证和检查工具

用于验证项目配置、环境设置和数据处理结果的工具集。

## 🔧 验证工具

- `verify_local_ultralytics.py` - 本地 ultralytics 配置完整验证
- `quick_check.py` - 快速配置检查
- `test_visdrone_conversion.py` - VisDrone 转换功能测试
- `run_verification.ps1` - PowerShell 验证运行脚本

## 🚀 使用方法

```bash
# 快速检查
python quick_check.py

# 完整验证
python verify_local_ultralytics.py

# PowerShell 脚本（Windows）
.un_verification.ps1 -Mode full
```

详细说明请参考: [验证工具说明](../docs/验证工具说明.md)
