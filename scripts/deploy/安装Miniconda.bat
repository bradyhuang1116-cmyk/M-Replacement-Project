@echo off
REM ============================================================
REM 静默安装 Miniconda Python 3.10 到 C:\Miniconda3，并配置 Path
REM 用法：管理员运行；安装包需在 offline\ 目录
REM ============================================================
setlocal
chcp 65001 >nul

net session >nul 2>&1
if errorlevel 1 (
    echo [错误] 请右键以管理员身份运行
    pause & exit /b 1
)

set "DELIVERY=%~dp0"
set "INSTALLER=%DELIVERY%offline\Miniconda3-py310_26.3.2-2-Windows-x86_64.exe"
set "CONDA_ROOT=C:\Miniconda3"

if not exist "%INSTALLER%" (
    echo [错误] 未找到安装包：
    echo   %INSTALLER%
    pause & exit /b 1
)

echo ==================================================
echo  安装 Miniconda Python 3.10
echo ==================================================

if exist "%CONDA_ROOT%\python.exe" (
    echo 已存在 %CONDA_ROOT%\python.exe，跳过安装，仅配置 Path ...
    goto :path
)

echo 正在静默安装到 %CONDA_ROOT% ...
"%INSTALLER%" /InstallationType=AllUsers /RegisterPython=0 /S /D=%CONDA_ROOT%
if errorlevel 1 (
    echo [错误] Miniconda 安装失败
    pause & exit /b 1
)

:path
call "%DELIVERY%配置Python环境变量.bat"
exit /b %ERRORLEVEL%
