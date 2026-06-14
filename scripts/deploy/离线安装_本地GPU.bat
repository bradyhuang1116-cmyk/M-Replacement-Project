@echo off
REM ============================================================
REM 本地 GPU 模式 — docker load + pip 离线安装
REM 用法：管理员运行；Docker Desktop 已启动
REM ============================================================
setlocal
chcp 65001 >nul

set "DELIVERY=%~dp0"
cd /d "%DELIVERY%"

if not exist "NodexelOCR.tar.gz" (
    echo [错误] 未找到 NodexelOCR.tar.gz
    pause & exit /b 1
)

echo ==================================================
echo  本地 GPU 模式 — docker load + pip
echo ==================================================

echo [1/2] docker load ...
docker load -i "NodexelOCR.tar.gz"
if errorlevel 1 (
    echo [错误] 镜像载入失败，请确认 Docker Desktop 已启动
    pause & exit /b 1
)

echo [2/2] pip install ...
pip install --no-index --find-links=offline\offline_wheels -r offline\requirements_local.txt
if errorlevel 1 pause & exit /b 1
pip install --no-index --find-links=offline\offline_wheels oracledb

echo.
echo 完成。下一步：
echo   docker run -d --gpus all -p 8080:8080 --name nodexel --restart unless-stopped nodexelocr:v1
echo   初始化配置.bat → 注册服务_本地GPU.bat
pause
