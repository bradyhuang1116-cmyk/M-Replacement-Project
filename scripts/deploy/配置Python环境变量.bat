@echo off
REM ============================================================
REM 将 C:\Miniconda3 写入系统 Path（管理员运行）
REM 可重复执行；已存在则跳过
REM ============================================================
setlocal
chcp 65001 >nul

net session >nul 2>&1
if errorlevel 1 (
    echo [错误] 请右键本文件，选择「以管理员身份运行」
    pause & exit /b 1
)

set "CONDA_ROOT=C:\Miniconda3"
if not exist "%CONDA_ROOT%\python.exe" (
    echo [错误] 未找到 %CONDA_ROOT%\python.exe
    echo 请先运行「安装Miniconda.bat」，或修改本脚本 CONDA_ROOT
    pause & exit /b 1
)

echo ==================================================
echo  配置系统环境变量 Path
echo  Miniconda: %CONDA_ROOT%
echo ==================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root='%CONDA_ROOT%';" ^
  "$add=@($root, \"$root\Scripts\", \"$root\Library\bin\");" ^
  "$cur=[Environment]::GetEnvironmentVariable('Path','Machine');" ^
  "if (-not $cur) { $cur='' };" ^
  "$parts=$cur -split ';' | Where-Object { $_ -ne '' };" ^
  "foreach ($p in $add) { if ($parts -notcontains $p) { $parts = ,$p + $parts } };" ^
  "$new=($parts -join ';').TrimEnd(';');" ^
  "[Environment]::SetEnvironmentVariable('Path', $new, 'Machine');" ^
  "Write-Host '已写入系统 Path。请关闭并重新打开 PowerShell 后执行 python --version 验证。'"

if errorlevel 1 (
    echo [错误] PowerShell 写入 Path 失败
    pause & exit /b 1
)

echo.
echo 验证（需新开 PowerShell 窗口）：
echo   python --version
echo   where.exe python
echo.
pause
