@echo off
REM ============================================================
REM Nuitka 编译 modules + config → .pyd（Windows cmd，PowerShell 也可调用）
REM 用法：scripts\compile_core.bat
REM       或在 PowerShell：cmd /c scripts\compile_core.bat
REM ============================================================
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0.."
set "PY=C:\Users\Administrator\miniconda3\envs\mitsubishi\python.exe"
set "OUT=%ROOT%\build\nuitka"

cd /d "%ROOT%"

if not exist "%PY%" (
    echo [错误] 未找到 Python: %PY%
    echo 请修改本脚本 PY 变量，或先创建 conda 环境 mitsubishi ^(python=3.10^)
    pause
    exit /b 1
)

echo ==================================================
echo  Nuitka 编译（Python 3.10）
echo  %PY%
echo ==================================================
"%PY%" --version
echo.

if not exist "%OUT%" mkdir "%OUT%"

echo [1/14] 编译 config.py ...
"%PY%" -m nuitka --module config.py --output-dir="%OUT%" --assume-yes-for-downloads
if errorlevel 1 goto :fail

set /a N=1
for %%f in (modules\*.py) do (
    if /I not "%%~nf"=="__init__" (
        set /a N+=1
        echo [!N!/14] 编译 modules\%%~nxf ...
        "%PY%" -m nuitka --module "modules\%%~nxf" --output-dir="%OUT%" --assume-yes-for-downloads
        if errorlevel 1 goto :fail
    )
)

echo.
echo ==================================================
echo  编译完成，产物目录: %OUT%
dir /b "%OUT%\*.pyd"
echo ==================================================
pause
exit /b 0

:fail
echo.
echo [错误] 编译失败，请查看上方报错
pause
exit /b 1
