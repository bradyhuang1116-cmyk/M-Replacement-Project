@echo off
REM ============================================================
REM 构建 Dashboard 生产包（Next.js standalone）→ NodexelOCR_Delivery\frontend\
REM 前置：本机已安装 Node.js 18+，且能访问 npm  registry
REM 用法：在项目根目录运行 scripts\build_dashboard.bat
REM ============================================================
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0.."
set "DASH=%ROOT%\dashboard"
set "OUT=%ROOT%\NodexelOCR_Delivery\frontend"

cd /d "%DASH%"
if not exist package.json (
    echo [错误] 未找到 %DASH%\package.json
    pause & exit /b 1
)

echo ==================================================
echo  构建 Dashboard standalone
echo ==================================================

where node >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 node，请先安装 Node.js 18+
    pause & exit /b 1
)

echo [1/3] npm ci ...
call npm ci
if errorlevel 1 pause & exit /b 1

echo [2/3] npm run build ...
call npm run build
if errorlevel 1 pause & exit /b 1

if not exist ".next\standalone" (
    echo [错误] 未生成 .next\standalone，请确认 next.config.ts 中 output=standalone
    pause & exit /b 1
)

echo [3/3] 复制到 %OUT% ...
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%"
xcopy /E /I /Y /Q ".next\standalone\*" "%OUT%\"
if exist ".next\static" (
    if not exist "%OUT%\.next\static" mkdir "%OUT%\.next\static"
    xcopy /E /I /Y /Q ".next\static\*" "%OUT%\.next\static\"
)
if exist "public" (
    if not exist "%OUT%\public" mkdir "%OUT%\public"
    xcopy /E /I /Y /Q "public\*" "%OUT%\public\"
)

if not exist "%OUT%\server.js" (
    echo [错误] 未找到 %OUT%\server.js
    pause & exit /b 1
)

echo.
echo 完成。验证：
echo   Test-Path NodexelOCR_Delivery\frontend\server.js
echo 交付机启动：
echo   cd frontend
echo   node server.js
echo ==================================================
pause
