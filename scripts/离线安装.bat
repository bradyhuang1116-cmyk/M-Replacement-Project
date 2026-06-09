@echo off
REM ============================================================
REM 离线安装脚本 —— 在「客户内网机器」上运行（双击或命令行）
REM 前提：offline_bundle\ 已拷到本机，且代码仓库已拉取/拷贝到位。
REM      基础环境（Docker Desktop / NVIDIA Toolkit / Python / Node）已装好。
REM 用法：把本 .bat 放在 offline_bundle\ 旁，在项目根目录下执行
REM ============================================================
setlocal
chcp 65001 >nul

set BUNDLE=offline_bundle

echo ==================================================
echo  离线安装开始
echo ==================================================

REM ── 1. 载入 vLLM 镜像 ──────────────────────────────
echo.
echo [1/4] 载入 vLLM Docker 镜像（需 Docker Desktop 已运行）...
docker load -i "%BUNDLE%\vllm_image.tar"
if errorlevel 1 (
    echo   [错误] 镜像载入失败，确认 Docker Desktop 已启动
    pause & exit /b 1
)

REM ── 2. 放置模型权重 ────────────────────────────────
echo.
echo [2/4] 拷贝模型权重到 models\...
if not exist "models" mkdir "models"
xcopy /E /I /Y "%BUNDLE%\models\PaddleOCR-VL-1.5" "models\PaddleOCR-VL-1.5" >nul
echo   完成

REM ── 3. 离线安装 Python 依赖 ────────────────────────
echo.
echo [3/4] 离线安装 Python 依赖...
pip install --no-index --find-links="%BUNDLE%\offline_wheels" -r "%BUNDLE%\requirements_local.txt"
if errorlevel 1 (
    echo   [错误] pip 离线安装失败，确认客户机 Python 版本与打包机一致
    pause & exit /b 1
)

REM ── 4. 放置前端依赖 ────────────────────────────────
echo.
echo [4/4] 拷贝前端依赖 node_modules\...
if exist "%BUNDLE%\dashboard\node_modules" (
    if not exist "dashboard" mkdir "dashboard"
    xcopy /E /I /Y "%BUNDLE%\dashboard\node_modules" "dashboard\node_modules" >nul
    echo   完成
) else (
    echo   [跳过] 包内无 node_modules
)

echo.
echo ==================================================
echo  离线安装完成。接下来：
echo   1. 复制 .env.example 为 .env，按客户共享盘路径修改
echo   2. 双击 start_vllm_server.bat 起推理服务
echo   3. 双击 start_v2.bat 起后端+前端
echo ==================================================
pause
