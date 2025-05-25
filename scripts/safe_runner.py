#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码安全的脚本运行器
解决 Windows 系统下的编码问题

作者: YOLOvision Pro Team
日期: 2024
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class SafeRunner:
    """编码安全的脚本运行器"""
    
    @staticmethod
    def run_script(script_path: str, args: List[str] = None, 
                   capture_output: bool = True, show_output: bool = True) -> Dict[str, Any]:
        """
        安全运行脚本，处理编码问题
        
        Args:
            script_path: 脚本路径
            args: 命令行参数
            capture_output: 是否捕获输出
            show_output: 是否显示输出
        
        Returns:
            包含执行结果的字典
        """
        if args is None:
            args = []
        
        # 构建命令
        cmd = [sys.executable, script_path] + args
        
        try:
            if capture_output:
                # 捕获输出模式
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # 替换无法解码的字符
                    check=False
                )
                
                # 显示输出
                if show_output:
                    if result.stdout:
                        # 清理输出中的特殊字符
                        clean_stdout = SafeRunner._clean_output(result.stdout)
                        print(clean_stdout)
                    
                    if result.stderr and result.returncode != 0:
                        clean_stderr = SafeRunner._clean_output(result.stderr)
                        print(f"错误信息: {clean_stderr}")
                
                return {
                    'success': result.returncode == 0,
                    'returncode': result.returncode,
                    'stdout': SafeRunner._clean_output(result.stdout) if result.stdout else '',
                    'stderr': SafeRunner._clean_output(result.stderr) if result.stderr else ''
                }
            else:
                # 直接执行模式（不捕获输出）
                # 设置环境变量以处理编码
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                result = subprocess.run(cmd, check=False, env=env)
                
                return {
                    'success': result.returncode == 0,
                    'returncode': result.returncode,
                    'stdout': '',
                    'stderr': ''
                }
                
        except Exception as e:
            error_msg = f"执行脚本时发生错误: {str(e)}"
            if show_output:
                print(f"❌ {error_msg}")
            
            return {
                'success': False,
                'returncode': 1,
                'stdout': '',
                'stderr': error_msg,
                'error': str(e)
            }
    
    @staticmethod
    def _clean_output(text: str) -> str:
        """清理输出文本，移除或替换问题字符"""
        if not text:
            return ''
        
        # 替换常见的问题字符
        replacements = {
            '🔧': '[工具]',
            '📊': '[数据]',
            '✅': '[成功]',
            '❌': '[失败]',
            '⚠️': '[警告]',
            '🚀': '[启动]',
            '📁': '[目录]',
            '📄': '[文件]',
            '🎯': '[目标]',
            '💡': '[提示]',
            '🔍': '[检查]',
            '📋': '[列表]',
            '🎭': '[演示]',
            '📈': '[分析]',
            '🔗': '[链接]',
            '⚡': '[快速]',
            '📱': '[应用]'
        }
        
        # 执行替换
        cleaned_text = text
        for emoji, replacement in replacements.items():
            cleaned_text = cleaned_text.replace(emoji, replacement)
        
        # 移除其他可能的问题字符
        try:
            # 尝试编码到 GBK 再解码，移除无法处理的字符
            cleaned_text = cleaned_text.encode('gbk', errors='ignore').decode('gbk')
        except:
            # 如果还有问题，使用 ASCII 安全模式
            cleaned_text = cleaned_text.encode('ascii', errors='ignore').decode('ascii')
        
        return cleaned_text
    
    @staticmethod
    def run_command_safe(command: str, description: str = "") -> bool:
        """
        安全运行命令
        
        Args:
            command: 要执行的命令字符串
            description: 操作描述
        
        Returns:
            是否执行成功
        """
        if description:
            print(f"\n[工具] {description}")
            print("-" * 50)
        
        # 解析命令
        cmd_parts = command.split()
        if not cmd_parts:
            print("[失败] 空命令")
            return False
        
        # 如果是 Python 脚本，使用安全运行器
        if cmd_parts[0] == 'python' and len(cmd_parts) > 1:
            script_path = cmd_parts[1]
            args = cmd_parts[2:] if len(cmd_parts) > 2 else []
            
            result = SafeRunner.run_script(script_path, args, capture_output=False)
            
            if result['success']:
                print(f"[成功] 完成: {description}")
            else:
                print(f"[失败] 失败: {description}")
            
            return result['success']
        else:
            # 非 Python 命令，直接执行
            try:
                result = subprocess.run(cmd_parts, check=False)
                success = result.returncode == 0
                
                if success:
                    print(f"[成功] 完成: {description}")
                else:
                    print(f"[失败] 失败: {description}")
                
                return success
            except Exception as e:
                print(f"[失败] 执行错误: {e}")
                return False

def main():
    """测试函数"""
    runner = SafeRunner()
    
    # 测试运行一个简单脚本
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        
        print(f"安全运行脚本: {script_path}")
        result = runner.run_script(script_path, args)
        
        print(f"执行结果: {'成功' if result['success'] else '失败'}")
        print(f"返回码: {result['returncode']}")
        
        return result['returncode']
    else:
        print("用法: python safe_runner.py <script_path> [args...]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
