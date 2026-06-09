@echo off
REM ============================================================
REM 在客户机注册 NodexelOCR 应用服务为 Windows 服务（NSSM）
REM 前置：Python 环境就绪、依赖已离线安装、app/ 已就位
REM 用法：管理员权限运行 install_service.bat
REM ============================================================
setlocal
chcp 65001 >nul

set SERVICE_NAME=NodexelOCRService
set NSSM=%~dp0offline\nssm.exe
set PYTHON=C:\Miniconda3\envs\mitsubishi\python.exe
set APP_DIR=%~dp0app
set START_SCRIPT=%APP_DIR%\start_v2.py

echo ==================================================
echo  注册 Windows 服务: %SERVICE_NAME%
echo ==================================================

if not exist "%NSSM%" (
    echo [错误] 未找到 %NSSM%
    pause & exit /b 1
)
if not exist "%PYTHON%" (
    echo [错误] 未找到 Python: %PYTHON%
    echo 请修改本脚本 PYTHON 变量为客户机实际路径
    pause & exit /b 1
)

REM 删除同名旧服务（如有）
"%NSSM%" stop %SERVICE_NAME% >nul 2>&1
"%NSSM%" remove %SERVICE_NAME% confirm >nul 2>&1

REM 注册服务
"%NSSM%" install %SERVICE_NAME% "%PYTHON%" "%START_SCRIPT%"
"%NSSM%" set %SERVICE_NAME% AppDirectory "%APP_DIR%"
"%NSSM%" set %SERVICE_NAME% DisplayName "NodexelOCR Service"
"%NSSM%" set %SERVICE_NAME% Description "图纸 OCR 替换处理服务"
"%NSSM%" set %SERVICE_NAME% Start SERVICE_AUTO_START
REM 日志重定向（可选，便于排障）
"%NSSM%" set %SERVICE_NAME% AppStdout "%APP_DIR%\logs\service_out.log"
"%NSSM%" set %SERVICE_NAME% AppStderr "%APP_DIR%\logs\service_err.log"
REM 崩溃自动重启
"%NSSM%" set %SERVICE_NAME% AppExit Default Restart

echo.
echo 服务已注册。启动命令：
echo   "%NSSM%" start %SERVICE_NAME%
echo 健康检查：浏览器访问 http://localhost:8000/api/v1/health
echo.
echo 提示：先确保 NodexelOCR 容器已 docker load 并运行，再启动本服务。
pause
