@echo off
setlocal
chcp 65001 >nul
set "DELIVERY=%~dp0"
set "NSSM=%DELIVERY%offline\nssm.exe"

echo 启动云端模式服务（后端 + 前端）...
"%NSSM%" start NodexelOCRService
"%NSSM%" start NodexelOCRDashboard

echo.
echo 后端: http://localhost:8000/api/v1/health
echo 前端: http://localhost:3000/settings
pause
