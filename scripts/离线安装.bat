@echo off
REM ============================================================
REM NodexelOCR 客户机离线安装 —— 在客户内网机器运行（管理员）
REM 前提：NodexelOCR_Delivery\ 已拷到本机；基础环境已装
REM       (Docker Desktop / NVIDIA Toolkit / Python 3.10)
REM 用法：在 NodexelOCR_Delivery\ 目录下，管理员权限运行本脚本
REM ============================================================
setlocal
chcp 65001 >nul

echo ==================================================
echo  NodexelOCR 离线安装
echo ==================================================

REM ── 1. 载入 NodexelOCR 镜像（模型已封装在内，零挂载）──
echo.
echo [1/3] 载入 NodexelOCR 镜像（需 Docker 已运行，~10GB 较慢）...
docker load -i "NodexelOCR.tar.gz"
if errorlevel 1 (
    echo   [错误] 镜像载入失败，确认 Docker Desktop 已启动
    pause & exit /b 1
)
echo   完成。启动推理服务：docker run -d --gpus all -p 8080:8080 --name nodexel nodexelocr:v1

REM ── 2. 离线安装 Python 依赖（含 oracledb）──
echo.
echo [2/3] 离线安装 Python 依赖...
pip install --no-index --find-links="offline\offline_wheels" -r "offline\requirements_local.txt"
if errorlevel 1 (
    echo   [错误] pip 离线安装失败，确认 Python 版本与打包机一致(3.10)
    pause & exit /b 1
)
pip install --no-index --find-links="offline\offline_wheels" oracledb

REM ── 3. 提示注册服务 ──
echo.
echo [3/3] 后续手动步骤：
echo   1. 复制 config\.env.example 为 app\.env，填 Oracle 连接 / D:\SMEC 归档路径
echo   2. 复制 config\server.properties.example 为 app\config\server.properties，填本机 IP
echo   3. 起推理容器：docker run -d --gpus all -p 8080:8080 --name nodexel nodexelocr:v1
echo   4. 注册应用服务：管理员运行 install_service_本地GPU.bat
echo   5. 健康检查：http://localhost:8000/api/v1/health

echo.
echo ==================================================
echo  离线安装完成
echo ==================================================
pause
