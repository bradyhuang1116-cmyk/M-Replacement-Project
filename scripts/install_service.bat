@echo off
setlocal
chcp 65001 >nul
set "DELIVERY=%~dp0"
if exist "%DELIVERY%app\.env" (
    findstr /I /C:"VLM_PROVIDER=paddleocr_api" "%DELIVERY%app\.env" >nul 2>&1
    if not errorlevel 1 (
        call "%DELIVERY%注册服务_云端API.bat"
        exit /b %ERRORLEVEL%
    )
)
call "%DELIVERY%注册服务_本地GPU.bat"
