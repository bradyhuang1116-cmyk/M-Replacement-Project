@echo off
REM ============================================================
REM 一键配置打包用 Python 环境（conda mitsubishi 3.10，不用 .venv）
REM 用法：scripts\setup_mitsubishi_env.bat
REM ============================================================
setlocal EnableExtensions
chcp 65001 >nul

set "PY=C:\Users\Administrator\miniconda3\envs\mitsubishi\python.exe"
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist "%PY%" (
    echo [错误] 未找到 %PY%
    echo 请先执行: conda create -n mitsubishi python=3.10 -y
    pause
    exit /b 1
)

echo ==================================================
echo  配置打包环境: mitsubishi (Python 3.10)
echo  %PY%
echo ==================================================
"%PY%" --version
echo.

echo [1/3] 安装打包工具 ...
"%PY%" -m pip install -U pip
"%PY%" -m pip install nuitka ordered-set zstandard pyinstaller
"%PY%" -m pip install -r manual_editor\requirements.txt
if errorlevel 1 goto :fail

echo.
echo [2/3] 验证 ...
"%PY%" -m nuitka --version
"%PY%" -m PyInstaller --version
echo.

echo [3/3] 完成
echo.
echo  PyCharm 请配置解释器为（不要选带 V 的 .venv）:
echo    %PY%
echo.
echo  若项目仍有 .venv，可运行: scripts\remove_project_venv.bat
echo ==================================================
pause
exit /b 0

:fail
echo [错误] 安装失败
pause
exit /b 1
