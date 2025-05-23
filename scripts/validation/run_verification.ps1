# YOLOvision Pro 配置验证运行脚本
# PowerShell 脚本，用于方便地运行各种验证检查

param(
    [string]$Mode = "quick",  # quick, full, help
    [switch]$Verbose = $false
)

# 设置控制台编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Show-Help {
    Write-Host "🔍 YOLOvision Pro 配置验证工具" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host ""
    Write-Host "用法:" -ForegroundColor Yellow
    Write-Host "  .\scripts\run_verification.ps1 [选项]" -ForegroundColor White
    Write-Host ""
    Write-Host "选项:" -ForegroundColor Yellow
    Write-Host "  -Mode <模式>     验证模式 (quick|full|help)" -ForegroundColor White
    Write-Host "  -Verbose         显示详细输出" -ForegroundColor White
    Write-Host ""
    Write-Host "模式说明:" -ForegroundColor Yellow
    Write-Host "  quick           快速检查 (默认)" -ForegroundColor Green
    Write-Host "  full            完整验证" -ForegroundColor Green
    Write-Host "  help            显示帮助" -ForegroundColor Green
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Yellow
    Write-Host "  .\scripts\run_verification.ps1" -ForegroundColor White
    Write-Host "  .\scripts\run_verification.ps1 -Mode full" -ForegroundColor White
    Write-Host "  .\scripts\run_verification.ps1 -Mode quick -Verbose" -ForegroundColor White
    Write-Host ""
}

function Test-Environment {
    Write-Host "🌍 检查运行环境..." -ForegroundColor Cyan
    
    # 检查当前目录
    $currentDir = Get-Location
    if (-not (Test-Path "ultralytics" -PathType Container)) {
        Write-Host "❌ 错误: 请在项目根目录运行此脚本" -ForegroundColor Red
        Write-Host "   当前目录: $currentDir" -ForegroundColor Red
        Write-Host "   应该包含 ultralytics 目录" -ForegroundColor Red
        return $false
    }
    
    # 检查 Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Python 未找到或无法执行" -ForegroundColor Red
        return $false
    }
    
    # 检查虚拟环境
    $pythonPath = python -c "import sys; print(sys.executable)" 2>&1
    if ($pythonPath -like "*yolo8*") {
        Write-Host "✅ 虚拟环境: yolo8" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ 虚拟环境可能未激活" -ForegroundColor Yellow
        Write-Host "   Python 路径: $pythonPath" -ForegroundColor Yellow
    }
    
    return $true
}

function Run-QuickCheck {
    Write-Host "🚀 运行快速检查..." -ForegroundColor Cyan
    Write-Host ""
    
    if ($Verbose) {
        python scripts/quick_check.py
    }
    else {
        python scripts/quick_check.py 2>$null
    }
    
    $exitCode = $LASTEXITCODE
    Write-Host ""
    
    if ($exitCode -eq 0) {
        Write-Host "✅ 快速检查完成" -ForegroundColor Green
    }
    else {
        Write-Host "❌ 快速检查发现问题 (退出码: $exitCode)" -ForegroundColor Red
    }
    
    return $exitCode
}

function Run-FullVerification {
    Write-Host "🔍 运行完整验证..." -ForegroundColor Cyan
    Write-Host ""
    
    if ($Verbose) {
        python scripts/verify_local_ultralytics.py
    }
    else {
        python scripts/verify_local_ultralytics.py 2>$null
    }
    
    $exitCode = $LASTEXITCODE
    Write-Host ""
    
    if ($exitCode -eq 0) {
        Write-Host "✅ 完整验证完成" -ForegroundColor Green
        
        # 检查报告文件
        if (Test-Path "outputs/verification_report.txt") {
            Write-Host "📋 验证报告已生成: outputs/verification_report.txt" -ForegroundColor Cyan
        }
    }
    else {
        Write-Host "❌ 完整验证发现问题 (退出码: $exitCode)" -ForegroundColor Red
    }
    
    return $exitCode
}

function Show-PostActions {
    param([int]$ExitCode)
    
    Write-Host ""
    Write-Host "📋 后续操作建议:" -ForegroundColor Cyan
    Write-Host "-" * 30 -ForegroundColor Cyan
    
    if ($ExitCode -eq 0) {
        Write-Host "🎉 配置验证通过！可以进行以下操作:" -ForegroundColor Green
        Write-Host ""
        Write-Host "1. 测试训练功能:" -ForegroundColor Yellow
        Write-Host "   yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=1" -ForegroundColor White
        Write-Host ""
        Write-Host "2. 查看项目文档:" -ForegroundColor Yellow
        Write-Host "   Get-Content docs/README.md" -ForegroundColor White
        Write-Host ""
        Write-Host "3. 开始 Drone-YOLO 开发:" -ForegroundColor Yellow
        Write-Host "   Get-Content docs/drone_yolo/README.md" -ForegroundColor White
    }
    else {
        Write-Host "⚠️ 发现配置问题，建议进行以下操作:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "1. 运行完整验证查看详细信息:" -ForegroundColor Yellow
        Write-Host "   .\scripts\run_verification.ps1 -Mode full -Verbose" -ForegroundColor White
        Write-Host ""
        Write-Host "2. 重新安装 ultralytics:" -ForegroundColor Yellow
        Write-Host "   pip install -e ./ultralytics" -ForegroundColor White
        Write-Host ""
        Write-Host "3. 处理 VisDrone 数据集:" -ForegroundColor Yellow
        Write-Host "   python scripts/process_visdrone_complete.py -i data/VisDrone2019-DET-train -o data/visdrone_yolo" -ForegroundColor White
    }
    
    Write-Host ""
}

# 主执行逻辑
function Main {
    # 显示标题
    Write-Host ""
    Write-Host "🔍 YOLOvision Pro 配置验证工具" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "模式: $Mode" -ForegroundColor Gray
    Write-Host "=" * 50 -ForegroundColor Cyan
    
    # 处理帮助模式
    if ($Mode -eq "help") {
        Show-Help
        return 0
    }
    
    # 检查环境
    if (-not (Test-Environment)) {
        Write-Host ""
        Write-Host "❌ 环境检查失败，无法继续" -ForegroundColor Red
        return 1
    }
    
    # 执行相应的验证
    $exitCode = 0
    
    switch ($Mode.ToLower()) {
        "quick" {
            $exitCode = Run-QuickCheck
        }
        "full" {
            $exitCode = Run-FullVerification
        }
        default {
            Write-Host "❌ 未知模式: $Mode" -ForegroundColor Red
            Write-Host "使用 -Mode help 查看帮助" -ForegroundColor Yellow
            return 1
        }
    }
    
    # 显示后续操作建议
    Show-PostActions -ExitCode $exitCode
    
    Write-Host ""
    Write-Host "🔍 验证完成！" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    
    return $exitCode
}

# 执行主函数
$result = Main
exit $result
