@echo off
REM ── 启动 PaddleOCR-VL-1.5 vLLM 推理服务器（Docker）──
REM 使用百度官方 PaddleOCR Docker 镜像
REM 默认端口 8080，与 config.py 中的 VLLM_BASE_URL 对应
REM 需要：Docker Desktop + NVIDIA GPU 支持

set CONTAINER_NAME=paddleocr-vl-vllm
set PORT=8080
set MODEL_PATH=%~dp0models\PaddleOCR-VL-1.5
set CONFIG_PATH=%~dp0vllm_config.yaml
set IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu

echo ====================================
echo  PaddleOCR-VL-1.5 vLLM Server
echo  Port:  %PORT%
echo  Model: %MODEL_PATH%
echo ====================================

REM 如果容器已存在则先删除
docker rm -f %CONTAINER_NAME% 2>nul

docker run -d ^
    --name %CONTAINER_NAME% ^
    --gpus all ^
    -p %PORT%:8080 ^
    -v "%MODEL_PATH%:/models/PaddleOCR-VL-1.5:ro" ^
    -v "%CONFIG_PATH%:/config/vllm_config.yaml:ro" ^
    %IMAGE% ^
    bash -c "mkdir -p /home/paddleocr/.paddlex/official_models && cp -r /models/PaddleOCR-VL-1.5 /home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.5 && paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8080 --backend vllm --backend_config /config/vllm_config.yaml"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo 容器已启动: %CONTAINER_NAME%
    echo 等待模型加载...
    echo 可通过 docker logs -f %CONTAINER_NAME% 查看日志
    echo 服务就绪后访问: http://localhost:%PORT%/v1/models
) else (
    echo 启动失败，请检查 Docker Desktop 和 GPU 驱动
)

pause
