@echo off
chcp 65001 >nul
echo ========================================
echo   图纸 Y→HY 替换系统 Web 界面
echo ========================================
echo.
echo 正在启动 Web 服务...
echo 服务地址: http://localhost:7860
echo.
echo 按 Ctrl+C 可停止服务
echo ========================================
echo.

start http://localhost:7860

python web_app.py

pause
