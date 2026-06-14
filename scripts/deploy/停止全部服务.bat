@echo off
setlocal
chcp 65001 >nul
set "NSSM=%~dp0offline\nssm.exe"

echo 停止 NodexelOCR 服务 ...
"%NSSM%" stop NodexelOCRDashboard >nul 2>&1
"%NSSM%" stop NodexelOCRService >nul 2>&1
echo 完成。
pause
