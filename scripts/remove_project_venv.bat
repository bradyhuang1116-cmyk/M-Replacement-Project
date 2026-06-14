@echo off
REM 删除项目 .venv，避免 PyCharm 自动激活虚拟环境
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist ".venv" (
    echo .venv 不存在，无需删除
    pause
    exit /b 0
)

echo 将删除: %ROOT%\.venv
echo 删除后请在 PyCharm 中:
echo   1. Settings - Python - Interpreter
echo   2. 删除 Python 3.10 带 V 的那条 ^(.venv^)
echo   3. 添加: C:\Users\Administrator\miniconda3\envs\mitsubishi\python.exe
echo   4. Settings - Tools - Terminal - 取消勾选 Activate virtual environment
echo.
pause

rmdir /s /q ".venv"
if exist ".venv" (
    echo [错误] 删除失败，请关闭 PyCharm 后重试
    pause
    exit /b 1
)

echo 已删除 .venv
pause
