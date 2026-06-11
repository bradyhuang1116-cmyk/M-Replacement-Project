#!/bin/bash
# NodexelOCR 推理服务启动脚本
set -e

echo "[NodexelOCR] Starting inference service..."

export VLLM_LOGGING_LEVEL=WARNING

exec paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --model_dir /home/paddleocr/models/nodexel-ocr \
    --host 0.0.0.0 \
    --port 8080 \
    --backend vllm \
    --backend_config /home/paddleocr/vllm_config.yaml
