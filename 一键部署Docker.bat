@echo off
chcp 65001 >nul
echo ========================================
echo   三菱图纸替换系统 - Docker 一键部署
echo ========================================
echo.

cd /d %~dp0

echo [1/3] 清理旧环境...
docker compose down --rmi all --volumes --remove-orphans 2>nul
echo.

echo [2/3] 构建镜像并启动容器（首次约5-10分钟）...
docker compose up -d --build
if %errorlevel% neq 0 (
    echo.
    echo [错误] 构建失败，请检查 Docker Desktop 是否已启动
    pause
    exit /b 1
)
echo.

echo [3/3] 等待服务就绪...
timeout /t 10 /nobreak >nul
docker compose ps
echo.

echo ========================================
echo   部署完成!
echo   Gradio界面 - http://localhost:7860
echo   API服务    - http://localhost:8000
echo   查看日志   - docker compose logs gradio -f
echo ========================================
echo.
pause
