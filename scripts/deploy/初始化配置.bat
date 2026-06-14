@echo off
REM ============================================================
REM 复制配置模板到 app\（可重复执行）
REM ============================================================
setlocal
chcp 65001 >nul

set "DELIVERY=%~dp0"
cd /d "%DELIVERY%"

echo 复制配置模板 ...

if not exist "config\.env.example" (
    echo [错误] 未找到 config\.env.example
    pause & exit /b 1
)

copy /Y "config\.env.example" "app\.env" >nul
if not exist "app\config" mkdir "app\config"
copy /Y "config\server.properties.example" "app\config\server.properties" >nul
if not exist "app\logs" mkdir "app\logs"
if not exist "app\data" mkdir "app\data"
if not exist "frontend\logs" mkdir "frontend\logs" 2>nul

echo.
echo 已生成：
echo   app\.env
echo   app\config\server.properties
echo.
echo 请编辑 app\.env（推理模式/Token）和 server.properties（本机 IP）
echo 或在服务启动后打开 http://localhost:3000/settings 网页配置
echo.
notepad "app\.env"
pause
