@echo off
REM ============================================================
REM 云端 API 模式 — 仅 pip 离线安装（不执行 docker load）
REM 用法：管理员运行；已配置好 Python Path
REM ============================================================
setlocal
chcp 65001 >nul

set "DELIVERY=%~dp0"
cd /d "%DELIVERY%"

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 找不到 python，请先运行「安装Miniconda.bat」和「配置Python环境变量.bat」
    pause & exit /b 1
)

echo ==================================================
echo  云端 API 模式 — 离线安装 Python 依赖
echo ==================================================
python --version
echo.

pip install --no-index --find-links=offline\offline_wheels -r offline\requirements_local.txt
if errorlevel 1 (
    echo [错误] pip 安装失败
    pause & exit /b 1
)
pip install --no-index --find-links=offline\offline_wheels oracledb

echo.
echo 完成。下一步：运行「初始化配置.bat」→「注册服务_云端API.bat」
pause
