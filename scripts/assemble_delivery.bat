@echo off
REM ============================================================
REM 组装 NodexelOCR_Delivery（后端 + 前端 + 部署脚本）
REM 用法：scripts\assemble_delivery.bat
REM 前置：build\nuitka\*.pyd、dist\ManualEditor.exe、frontend\（build_dashboard.bat）
REM ============================================================
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0.."
set "OUT=%ROOT%\NodexelOCR_Delivery"
cd /d "%ROOT%"

if not exist "build\nuitka\config.cp310-win_amd64.pyd" (
    echo [错误] 未找到 build\nuitka\*.pyd，请先 Nuitka 编译
    pause & exit /b 1
)

echo ==================================================
echo  组装交付目录: %OUT%
echo ==================================================

mkdir "%OUT%\app\modules" 2>nul
mkdir "%OUT%\ManualEditor" 2>nul
mkdir "%OUT%\config" 2>nul
mkdir "%OUT%\docs" 2>nul
mkdir "%OUT%\offline" 2>nul
mkdir "%OUT%\runtime\node" 2>nul

echo [1/6] 复制 .pyd（build\nuitka\*.cp310-win_amd64.pyd → app\modules\*.pyd）...
set "PYD_ERR=0"
copy /Y "build\nuitka\config.cp310-win_amd64.pyd" "%OUT%\app\config.pyd" >nul
if errorlevel 1 set "PYD_ERR=1"
for %%f in (modules\*.py) do (
    if /I not "%%~nf"=="__init__" (
        if not exist "build\nuitka\%%~nf.cp310-win_amd64.pyd" (
            echo   [错误] 缺少编译产物: build\nuitka\%%~nf.cp310-win_amd64.pyd
            set "PYD_ERR=1"
        ) else (
            copy /Y "build\nuitka\%%~nf.cp310-win_amd64.pyd" "%OUT%\app\modules\%%~nf.pyd" >nul
            if errorlevel 1 (
                echo   [错误] 复制失败: %%~nf.pyd
                set "PYD_ERR=1"
            )
        )
    )
)
copy /Y "modules\__init__.py" "%OUT%\app\modules\" >nul
if "%PYD_ERR%"=="1" (
    echo.
    echo [错误] .pyd 复制不完整。请先运行 scripts\compile_core.bat ^(Python 3.10^)
    pause & exit /b 1
)
echo   已复制 app\config.pyd 与 app\modules\ 下 16 个 .pyd
echo   关键文件（请核对大小，job_queue 新版约 474112 字节）:
for %%p in (job_queue worker) do (
    if exist "%OUT%\app\modules\%%p.pyd" (
        for %%s in ("%OUT%\app\modules\%%p.pyd") do echo     %%p.pyd  %%~zs 字节
    ) else (
        echo     [缺失] %%p.pyd
        set "PYD_ERR=1"
    )
)
if "%PYD_ERR%"=="1" pause & exit /b 1

echo [2/6] 复制后端应用 ...
if exist "%OUT%\app\api\api" rmdir /S /Q "%OUT%\app\api\api" 2>nul
xcopy "api" "%OUT%\app\api\" /E /I /Y /Q >nul
xcopy "fonts" "%OUT%\app\fonts\" /E /I /Y /Q >nul
if exist "app_config" xcopy "app_config" "%OUT%\app\app_config\" /E /I /Y /Q >nul
copy /Y "start_v2.py" "%OUT%\app\" >nul
if exist "dist\ManualEditor.exe" (
    copy /Y "dist\ManualEditor.exe" "%OUT%\ManualEditor\" >nul
) else (
    echo   警告: dist\ManualEditor.exe 不存在
)

echo [3/6] 复制前端 standalone ...
if exist "%OUT%\frontend\server.js" (
    echo   已存在 frontend\，跳过（请先运行 scripts\build_dashboard.bat 更新）
) else if exist "%ROOT%\NodexelOCR_Delivery\frontend\server.js" (
    xcopy "%ROOT%\NodexelOCR_Delivery\frontend" "%OUT%\frontend\" /E /I /Y /Q >nul
) else (
    echo   警告: 未找到 frontend\，请运行 scripts\build_dashboard.bat
)

echo [4/6] 复制配置模板与部署脚本 ...
copy /Y ".env.example" "%OUT%\config\.env.example" >nul 2>nul
> "%OUT%\config\server.properties.example" (
    echo # 本机 IP（举例）：开发 192.168.10.101 / 生产 192.168.20.101
    echo server.ip=192.168.10.101
)
for %%f in (deploy\*.bat) do copy /Y "scripts\%%f" "%OUT%\" >nul
copy /Y "scripts\离线安装.bat" "%OUT%\" >nul
if exist "scripts\nssm.exe" copy /Y "scripts\nssm.exe" "%OUT%\offline\" >nul

echo [5/6] 复制文档 ...
if exist "docs\部署手册.md" copy /Y "docs\部署手册.md" "%OUT%\docs\" >nul
if exist "docs\部署手册_本地GPU模式.md" copy /Y "docs\部署手册_本地GPU模式.md" "%OUT%\docs\" >nul
if exist "docs\部署手册_云端API模式.md" copy /Y "docs\部署手册_云端API模式.md" "%OUT%\docs\" >nul
if exist "docs\打包加密操作手册.md" copy /Y "docs\打包加密操作手册.md" "%OUT%\docs\" >nul

echo [6/6] 完成
echo.
echo 交付包内 .pyd 路径（不在 Git 仓库内，PyCharm 默认搜不到）:
echo   %OUT%\app\config.pyd
echo   %OUT%\app\modules\job_queue.pyd
echo   %OUT%\app\modules\worker.pyd
echo   ^(编译中间产物名: build\nuitka\job_queue.cp310-win_amd64.pyd^)
echo.
echo 还需手动:
echo   - scripts\build_dashboard.bat（若 frontend 缺失）
echo   - 放入 NodexelOCR.tar.gz（本地 GPU 模式）
echo   - offline\ 放入 Miniconda / Docker / WinSCP / Node 便携包
echo   - runtime\node\ 解压 node-v20-win-x64.zip（可选，或用系统 Node）
echo ==================================================
pause
