@echo off
REM 内部脚本 — 注册后端 + 前端 Windows 服务，勿直接运行
setlocal
chcp 65001 >nul

if "%~1"=="" (
    echo [错误] 缺少部署模式参数 cloud 或 gpu
    exit /b 1
)
set "DEPLOY_MODE=%~1"
set "DELIVERY=%~dp0"

set "BACKEND_SVC=NodexelOCRService"
set "FRONTEND_SVC=NodexelOCRDashboard"
set "NSSM=%DELIVERY%offline\nssm.exe"
set "PYTHON=C:\Miniconda3\python.exe"
set "APP_DIR=%DELIVERY%app"
set "FRONTEND_DIR=%DELIVERY%frontend"
set "NODE_EXE=%DELIVERY%runtime\node\node.exe"

if /I "%DEPLOY_MODE%"=="cloud" (
    set "MODE_LABEL=云端 API 模式"
) else (
    set "MODE_LABEL=本地 GPU 模式"
)

echo ==================================================
echo  注册 Windows 服务
echo  模式: %MODE_LABEL%
echo ==================================================

if not exist "%NSSM%" (
    echo [错误] 未找到 %NSSM%
    pause & exit /b 1
)
if not exist "%PYTHON%" (
    echo [错误] 未找到 %PYTHON%
    echo 请修改本文件 PYTHON= 或先运行「安装Miniconda.bat」
    pause & exit /b 1
)
if not exist "%APP_DIR%\start_v2.py" (
    echo [错误] 未找到后端入口 %APP_DIR%\start_v2.py
    pause & exit /b 1
)
if not exist "%FRONTEND_DIR%\server.js" (
    echo [错误] 未找到前端 %FRONTEND_DIR%\server.js
    echo 打包机需先运行 scripts\build_dashboard.bat
    pause & exit /b 1
)

if not exist "%NODE_EXE%" (
    where node >nul 2>&1
    if errorlevel 1 (
        echo [错误] 未找到 Node：请安装 Node.js 或将便携版解压到 runtime\node\
        pause & exit /b 1
    )
    set "NODE_EXE=node"
)

if not exist "%APP_DIR%\logs" mkdir "%APP_DIR%\logs"
if not exist "%FRONTEND_DIR%\logs" mkdir "%FRONTEND_DIR%\logs"

REM ── 后端服务 ──
"%NSSM%" stop %BACKEND_SVC% >nul 2>&1
"%NSSM%" remove %BACKEND_SVC% confirm >nul 2>&1
"%NSSM%" install %BACKEND_SVC% "%PYTHON%" "%APP_DIR%\start_v2.py"
"%NSSM%" set %BACKEND_SVC% AppDirectory "%APP_DIR%"
"%NSSM%" set %BACKEND_SVC% DisplayName "NodexelOCR Backend API"
"%NSSM%" set %BACKEND_SVC% Description "NodexelOCR 图纸替换后端 API :8000"
"%NSSM%" set %BACKEND_SVC% Start SERVICE_AUTO_START
"%NSSM%" set %BACKEND_SVC% AppStdout "%APP_DIR%\logs\service_out.log"
"%NSSM%" set %BACKEND_SVC% AppStderr "%APP_DIR%\logs\service_err.log"
"%NSSM%" set %BACKEND_SVC% AppExit Default Restart

REM ── 前端 Dashboard ──
"%NSSM%" stop %FRONTEND_SVC% >nul 2>&1
"%NSSM%" remove %FRONTEND_SVC% confirm >nul 2>&1
"%NSSM%" install %FRONTEND_SVC% "%NODE_EXE%" "server.js"
"%NSSM%" set %FRONTEND_SVC% AppDirectory "%FRONTEND_DIR%"
"%NSSM%" set %FRONTEND_SVC% DisplayName "NodexelOCR Dashboard"
"%NSSM%" set %FRONTEND_SVC% Description "NodexelOCR 管理面板 :3000"
"%NSSM%" set %FRONTEND_SVC% Start SERVICE_AUTO_START
"%NSSM%" set %FRONTEND_SVC% AppStdout "%FRONTEND_DIR%\logs\service_out.log"
"%NSSM%" set %FRONTEND_SVC% AppStderr "%FRONTEND_DIR%\logs\service_err.log"
"%NSSM%" set %FRONTEND_SVC% AppExit Default Restart
"%NSSM%" set %FRONTEND_SVC% AppEnvironmentExtra PORT=3000
"%NSSM%" set %FRONTEND_SVC% AppEnvironmentExtra HOSTNAME=0.0.0.0

echo.
echo 已注册：
echo   %BACKEND_SVC%  → http://localhost:8000/api/v1/health
echo   %FRONTEND_SVC% → http://localhost:3000/settings
echo.

if /I "%DEPLOY_MODE%"=="cloud" (
    echo [云端 API 模式] 无需 Docker。请运行「启动服务_云端API.bat」
) else (
    echo [本地 GPU 模式] 请先启动 nodexel 容器，再运行「启动服务_本地GPU.bat」
)

pause
