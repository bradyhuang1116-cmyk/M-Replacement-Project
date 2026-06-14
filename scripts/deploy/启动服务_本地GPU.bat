@echo off
setlocal
chcp 65001 >nul
set "DELIVERY=%~dp0"
set "NSSM=%DELIVERY%offline\nssm.exe"

echo 启动推理容器 nodexel ...
docker start nodexel >nul 2>&1
if errorlevel 1 (
    echo 容器不存在或未创建，尝试 docker run ...
    docker run -d --gpus all -p 8080:8080 --name nodexel --restart unless-stopped nodexelocr:v1
)

echo 启动后端 + 前端服务 ...
"%NSSM%" start NodexelOCRService
"%NSSM%" start NodexelOCRDashboard

echo.
echo 推理: http://localhost:8080/v1/models
echo 后端: http://localhost:8000/api/v1/health
echo 前端: http://localhost:3000/settings
pause
